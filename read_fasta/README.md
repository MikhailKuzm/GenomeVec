# read_fasta

Пакет для чтения FASTA-файлов: извлечение контигов (последовательностей нуклеотидов) и метаданных.

## Назначение

- Парсинг FASTA: разбиение на контиги по заголовкам `>...`
- Метаданные: `num_contigs`, `total_nucleotides`, `organism_name`, `source_file`
- Название организма извлекается из первого заголовка (до "strain" или запятой) для использования как фенотип/класс бактерии

## Использование

```python
from read_fasta import read_fasta

contigs, meta = read_fasta("path/to/genome.fna")
# contigs — список строк нуклеотидов
# meta["organism_name"] — название организма
```

Класс-обёртка:

```python
from read_fasta import FastaLoader
loader = FastaLoader()
contigs, meta = loader.load("path/to/genome.fna")
```
