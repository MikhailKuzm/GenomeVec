"""
Токенизация контигов для DNABERT2.
Получает контиги (строки нуклеотидов) от read_fasta, возвращает список последовательностей токенов по контигам.
Токенизатор по умолчанию: embedders/DNABERT2/model или zhihan1996/DNABERT-2-117M с Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

# Корень репозитория (родитель embedders)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_MODEL_DIR = _SCRIPT_DIR / "model"


def _get_tokenizer(tokenizer=None, model_path: str | Path | None = None):
    """Загружает токенизатор из model_path или HF; если передан tokenizer — возвращает его."""
    if tokenizer is not None:
        return tokenizer
    path = model_path or _DEFAULT_MODEL_DIR
    path = Path(path)
    # Проверяем, есть ли локальная модель (достаточно config.json или tokenizer_config.json)
    if (path / "config.json").exists() or (path / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


def tokenize_contigs(
    contigs: list[str],
    tokenizer=None,
    model_path: str | Path | None = None,
    add_special_tokens: bool = False,
) -> list[list[int]]:
    """
    Токенизирует каждый контиг (строка нуклеотидов) в список id токенов.
    Нарезка по окнам и ограничение длины — в embeddings.py.

    Args:
        contigs: список строк нуклеотидов (результат read_fasta).
        tokenizer: уже загруженный токенизатор (если None — загружается по model_path или HF).
        model_path: путь к папке модели (по умолчанию embedders/DNABERT2/model).
        add_special_tokens: добавлять ли [CLS]/[SEP]; для скользящих окон обычно False.

    Returns:
        Список списков int: по одному списку token id на контиг (длина может быть любой).
    """
    tok = _get_tokenizer(tokenizer=tokenizer, model_path=model_path)
    result = []
    for seq in contigs:
        # Без обрезки: полная последовательность токенов по контигу
        enc = tok.encode(
            seq,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_tensors=None,
        )
        result.append(enc)
    return result
