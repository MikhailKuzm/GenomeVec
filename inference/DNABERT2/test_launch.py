"""
Интерактивный прогон одного FASTA через пайплайн: read_fasta → tokenize → make_embed.
Запуск: python -i inference/DNABERT2/test_launch.py (из корня репозитория)
       или выполнить блок в REPL.
"""

import os
from glob import glob
from collections import defaultdict

# Сделаем текущую папку = inference/DNABERT2, чтобы импорты работали без sys.path
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # В REPL __file__ нет: стартуем от cwd или укажите HERE вручную
    HERE = os.getcwd()
    if os.path.isdir(os.path.join(HERE, "inference", "DNABERT2")):
        HERE = os.path.join(HERE, "inference", "DNABERT2")
os.chdir(HERE)

# Путь к FASTA-датасету
FASTA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "genoms", "fasta"))
FASTA_FILES = sorted(glob(os.path.join(FASTA_DIR, "*.fna")))
if not FASTA_FILES:
    raise FileNotFoundError("В genoms/fasta не найден ни один .fna — укажите FASTA_FILE вручную в скрипте.")

# Куда сохранять результат
VECTORS_DIR = os.path.join(HERE, "vectors")
os.makedirs(VECTORS_DIR, exist_ok=True)
OUT_FILE = os.path.join(VECTORS_DIR, "dnabert2_vectors.pt")

import torch
from tqdm import tqdm

# 1) Чтение FASTA: контиги и мета
from read_fasta import read_fasta

# 2) Токенизация контигов
from tokenize_genome import tokenize_contigs

# 3) Эмбеддинг: скользящее окно, усреднение
from embeddings import make_embed, _load_model_and_tokenizer
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
# Гиперпараметры
WINDOW_SIZE = 512
OVERLAP_TOKENS = 0

# Загружаем модель 1 раз
model, tokenizer, device = _load_model_and_tokenizer()
print("Device:", device, "FASTA files:", len(FASTA_FILES))

vectors_by_organism = defaultdict(list)  # organism_name -> [embedding, ...]
meta_by_organism = defaultdict(list)     # organism_name -> [meta_entry, ...]

for fasta_path in tqdm(FASTA_FILES, desc="FASTA", unit="file"):
    contigs, meta = read_fasta(fasta_path)
    organism = meta.get("organism_name", "unknown") or "unknown"

    # Токенизация контигов тем же токенизатором, что и у модели
    tokens_per_contig = tokenize_contigs(contigs, tokenizer=tokenizer)

    embedding, meta_info = make_embed(
        tokens_per_contig,
        model=model,
        tokenizer=tokenizer,
        window_size=WINDOW_SIZE,
        overlap_tokens=OVERLAP_TOKENS,
        device=device,
    )

    vectors_by_organism[organism].append(embedding.detach().to("cpu"))
    meta_by_organism[organism].append(
        {
            "fasta_path": fasta_path,
            "num_contigs": meta.get("num_contigs"),
            "total_nucleotides": meta.get("total_nucleotides"),
            "num_windows": meta_info.get("num_windows"),
            "time_seconds": meta_info.get("time_seconds"),
        }
    )

# Складываем в один объект и сохраняем одним файлом
payload = {
    "vectors_by_organism": {k: torch.stack(v, dim=0) for k, v in vectors_by_organism.items()},
    "meta_by_organism": dict(meta_by_organism),
    "model_id": "zhihan1996/DNABERT-2-117M",
    "window_size": WINDOW_SIZE,
    "overlap_tokens": OVERLAP_TOKENS,
}
torch.save(payload, OUT_FILE)
print("Saved:", OUT_FILE)

# Результат: embedding — один вектор на геном; meta_info — число окон и время
