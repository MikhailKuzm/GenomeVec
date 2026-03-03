"""
Эмбеддинги DNABERT2: скользящее окно по контигам, сумма → среднее, один вектор на набор контигов.
Функция make_embed: токены (от tokenize_genome) → эмбеддинг + meta_info (число окон, время).
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import torch
from tqdm import tqdm

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # В интерактивной сессии (REPL) __file__ нет — задаём от cwd или укажите вручную
    _SCRIPT_DIR = Path.cwd()
    if (_SCRIPT_DIR / "inference" / "DNABERT2").exists():
        _SCRIPT_DIR = _SCRIPT_DIR / "inference" / "DNABERT2"
    elif not (_SCRIPT_DIR / "embeddings.py").exists():
        for p in Path.cwd().iterdir():
            if p.is_dir() and (p / "embeddings.py").exists():
                _SCRIPT_DIR = p
                break
_DEFAULT_MODEL_DIR = _SCRIPT_DIR / "model"


def _ensure_triton_stub():
    """Triton нет под Windows — при обращении к атрибутам поднимаем ImportError,
    чтобы импорт flash_attn_triton упал и bert_layers использовал fallback (flash_attn_qkvpacked_func = None)."""
    if "triton" in sys.modules:
        return
    stub = types.ModuleType("triton")

    def __getattr__(name):
        raise ImportError("triton is not available on this platform (e.g. Windows)")

    stub.__getattr__ = __getattr__
    sys.modules["triton"] = stub


def _patch_bert_layers_alibi_device():
    """Патчит кэшированный bert_layers.py DNABERT-2: вызов rebuild_alibi_tensor с device='cpu',
    чтобы избежать RuntimeError «Tensor on device meta is not on the expected device cpu!»."""
    import os
    candidates = [
        Path(os.environ.get("HF_HUB_CACHE", "")).parent if os.environ.get("HF_HUB_CACHE") else None,
        Path.home() / ".cache" / "huggingface",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
    ]
    modules_root = None
    for c in candidates:
        if c is None or not c.is_dir():
            continue
        # модули лежат в <cache>/modules/transformers_modules или <cache>/hub/../modules/...
        for base in [c / "modules" / "transformers_modules", c.parent / "modules" / "transformers_modules"]:
            dnabert = base / "zhihan1996" / "DNABERT_hyphen_2_hyphen_117M"
            if dnabert.is_dir():
                modules_root = dnabert
                break
        if modules_root is not None:
            break
    if modules_root is None:
        return
    for rev_dir in modules_root.iterdir():
        if not rev_dir.is_dir():
            continue
        bert_file = rev_dir / "bert_layers.py"
        if not bert_file.is_file():
            continue
        try:
            text = bert_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        old_line = "self.rebuild_alibi_tensor(size=config.alibi_starting_size)"
        new_line = "self.rebuild_alibi_tensor(size=config.alibi_starting_size, device='cpu')"
        if old_line in text and new_line not in text:
            text = text.replace(old_line, new_line)
            try:
                bert_file.write_text(text, encoding="utf-8")
            except Exception:
                pass


def _load_model_and_tokenizer(model_path: str | Path | None = None, device: str | None = None):
    """Загружает модель и токенизатор из model_path или HF."""
    _ensure_triton_stub()
    # Устройство по умолчанию CPU — иначе ALiBi в bert_layers создаётся с device=None и уходит на meta
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    path = model_path or _DEFAULT_MODEL_DIR
    path = Path(path)
    hf_id = "zhihan1996/DNABERT-2-117M"
    if (path / "config.json").exists():
        config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # Конфиг DNABERT-2 может не содержать pad_token_id — подставляем для совместимости с bert_layers
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    # Отключаем dynamo/compile при загрузке — иначе ALiBi-буферы создаются на meta и падают с "meta vs cpu"
    _dynamo = getattr(torch, "_dynamo", None)
    if _dynamo is not None and hasattr(_dynamo, "disable"):
        _dynamo.disable()
    # Патчим кэшированный bert_layers.py (device='cpu' в rebuild_alibi_tensor), чтобы не было meta/cpu
    _patch_bert_layers_alibi_device()
    load_from_hf = not (path / "config.json").exists()
    last_error = None
    for attempt in range(2):
        try:
            if (path / "config.json").exists():
                model = AutoModel.from_pretrained(
                    str(path), config=config, trust_remote_code=True, low_cpu_mem_usage=False
                )
            else:
                model = AutoModel.from_pretrained(
                    hf_id, config=config, trust_remote_code=True, low_cpu_mem_usage=False
                )
            last_error = None
            break
        except RuntimeError as e:
            last_error = e
            if "meta" in str(e) and "expected device" in str(e) and load_from_hf and attempt == 0:
                _patch_bert_layers_alibi_device()
                continue
            raise
    if last_error is not None:
        raise last_error
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def load_fasta_dir(path: str | Path) -> list[tuple[Path, list[str], dict]]:
    """
    Обходит каталог, для каждого *.fna / *.fasta вызывает read_fasta.
    Returns: список (path, contigs, meta) по файлам.
    """
    import sys
    p = Path(__file__).resolve().parent
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    from read_fasta import read_fasta
    path = Path(path)
    if path.is_file():
        contigs, meta = read_fasta(path)
        return [(path, contigs, meta)]
    out = []
    for ext in ("*.fna", "*.fasta", "*.fa"):
        for f in path.glob(ext):
            contigs, meta = read_fasta(f)
            out.append((f, contigs, meta))
    return out


def make_embed(
    tokens_per_contig: list[list[int]],
    model=None,
    tokenizer=None,
    model_path: str | Path | None = None,
    window_size: int = 512,
    overlap_tokens: int = 0,
    device: str | None = None,
    batch_size: int = 1,
) -> tuple[torch.Tensor, dict]:
    """
    По списку контигов (каждый — список token id) строит один усреднённый эмбеддинг:
    скользящее окно window_size с перекрытием overlap_tokens, для каждого окна mean-pool → сумма → деление на число окон.

    Args:
        tokens_per_contig: список списков int (результат tokenize_genome).
        model, tokenizer: если None — загружаются по model_path или HF.
        model_path: путь к папке модели.
        window_size: размер окна в токенах (для DNABERT2 обычно 512).
        overlap_tokens: перекрытие в токенах; шаг окна = window_size - overlap_tokens.
        device: cuda/cpu.
        batch_size: сколько окон обрабатывать за один forward (1 = по одному).

    Returns:
        (embedding, meta_info): embedding — тензор формы (hidden_size,); meta_info — num_windows, time_seconds, num_contigs.
    """
    if not tokens_per_contig:
        raise ValueError("tokens_per_contig пустой")
    if model is None or tokenizer is None:
        model, tokenizer, device = _load_model_and_tokenizer(model_path, device)
    else:
        if device is None:
            device = next(model.parameters()).device
    stride = max(1, window_size - overlap_tokens)
    hidden_size = model.config.hidden_size
    total_embed = torch.zeros(hidden_size, dtype=torch.float32, device=device)
    num_windows = 0
    start_time = time.perf_counter()

    # Собираем все окна для прогресс-бара (короткие контиги паддим до window_size)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    windows_to_process = []
    for contig_tokens in tokens_per_contig:
        if len(contig_tokens) < window_size:
            win = list(contig_tokens) + [pad_id] * (window_size - len(contig_tokens))
            windows_to_process.append(win)
            continue
        start = 0
        while start + window_size <= len(contig_tokens):
            win = contig_tokens[start : start + window_size]
            windows_to_process.append(win)
            start += stride
        # Последнее неполное окно (если остаток >= 1 токена)
        if start < len(contig_tokens) and len(contig_tokens) - start >= 1:
            win = contig_tokens[-window_size:]  # последние window_size токенов
            windows_to_process.append(win)

    for i in tqdm(range(0, len(windows_to_process), batch_size), desc="Windows", unit="batch"):
        batch = windows_to_process[i : i + batch_size]
        # Каждое окно — список из window_size id; паддинг до одной длины уже есть
        input_ids = torch.tensor([w for w in batch], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # DNABERT-2 BertModel возвращает (encoder_outputs, pooled_output); у других — объект с .last_hidden_state
            if isinstance(outputs, tuple):
                hidden = outputs[0]
                if isinstance(hidden, list):
                    hidden = hidden[-1]
            else:
                hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)  # (batch, hidden_size)
        for j in range(pooled.size(0)):
            total_embed += pooled[j]
            num_windows += 1

    elapsed = time.perf_counter() - start_time
    if num_windows == 0:
        raise ValueError("Не получилось ни одного окна")
    embedding = total_embed / num_windows
    meta_info = {
        "num_windows": num_windows,
        "time_seconds": round(elapsed, 4),
        "num_contigs": len(tokens_per_contig),
    }
    return embedding, meta_info
