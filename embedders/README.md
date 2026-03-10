# embedders

Пакет для получения эмбеддингов по FASTA-файлам (векторизация геномов).

## Назначение

- Эмбеддеры возвращают **векторы по каждому окну** (матрица `num_windows × hidden_size`), без усреднения.
- Усреднение по окнам (один вектор на геном) выполняется в вызывающем коде (например в скрипте обучения в `train/`).

## DNABERT2

- Класс `FastaToEmbeddings`: скользящее окно по FASTA, возвращает тензор `(num_windows, hidden_size)`.
- Готовые векторы можно сохранять и загружать из каталога `embedders/DNABERT2/vectors/`.

## Использование

```python
from embedders.DNABERT2 import FastaToEmbeddings

embedder = FastaToEmbeddings(window_size=512, overlap_tokens=0)
# matrix: (num_windows, hidden_size) — по одному вектору на окно
matrix = embedder("path/to/genome.fna")
# Усреднение при необходимости — в скрипте обучения:
# vec = matrix.mean(dim=0)
```
