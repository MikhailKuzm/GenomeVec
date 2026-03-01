# Парсер референсных геномов прокариот (NCBI)

Скрипты для загрузки референсных геномов бактерий с [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/genome/?taxon=2&reference_only=true) и подготовки каталога FASTA и лога статистики.

## Требования

1. **Python 3.8+** с пакетами из `requirements.txt`.
2. **NCBI Datasets CLI** — официальная утилита для скачивания данных.

### Установка NCBI Datasets CLI

**Windows (через conda):**
```bash
conda install -c conda-forge ncbi-datasets-cli
```

**Или скачать бинарник (без добавления в PATH):**
- [Releases](https://github.com/ncbi/datasets/releases) — скачайте архив для Windows, распакуйте `datasets.exe` в папку проекта (например, `parser/tools/`) и в `config.yaml` укажите: `datasets_cli_path: "parser/tools/datasets.exe"`.

Проверка:
```bash
datasets --version
```

## Структура выходных данных

- **genoms/fasta/** — по одному FASTA-файлу на геном. Имена файлов маркированы по фенотипу: `Организм_Accession_genomic.fna` (например, `Escherichia_coli_GCF_000005845_genomic.fna`).
- **genoms/statistic.log** — сводка: число геномов по фенотипам (видам/организмам), общее число FASTA, число уникальных фенотипов.

Под «фенотипом» здесь понимается **организм (вид)** из метаданных NCBI Assembly (`organism` в отчёте). При необходимости фенотипы можно дообогатить из других источников (например, базы фенотипов).

## Конфигурация

Редактируйте `parser/config.yaml`: таксон, пути, лимиты. Если CLI установлен не в PATH — задайте `datasets_cli_path` (путь к `datasets.exe`).

## Запуск

Из корня проекта:

```bash
# Установка зависимостей Python
pip install -r parser/requirements.txt

# Скачивание и разбор геномов (запуск из корня проекта)
python parser/run_parser.py
```

Скрипт:

1. Скачивает пакет референсных геномов прокариот. Если в конфиге задано `max_genomes_to_download: N`, сначала загружаются только метаданные, затем скачиваются только N геномов по списку accession (малый объём). Если не задано — полная загрузка по таксону (`datasets download genome taxon 2 --reference`).
2. Распаковывает архив и копирует `*_genomic.fna` в `genoms/fasta/` с именами вида `Фенотип_Accession_genomic.fna`.
3. Читает `assembly_data_report.jsonl`, группирует по полю организма (фенотип).
4. Пишет в `genoms/statistic.log`: количество геномов по фенотипам, общее число FASTA, число фенотипов.

## Ограничения

- Большие загрузки (>1000 геномов или >15 GB) лучше делать через [large download](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/genomes/large-download/) (dehydrated + rehydrate).
- Чтобы скачать только N геномов (без полного архива), задайте в `config.yaml`: `max_genomes_to_download: 10`. Тогда сначала загружаются метаданные, затем только указанное число геномов по списку accession.
- При полной загрузке таксона можно ограничить число копируемых FASTA: `max_genomes: 10`.

## Ссылки

- [NCBI Datasets — Download a genome](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/genomes/download-genome/)
- [NCBI Datasets CLI — taxon](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/download/genome/datasets_download_genome_taxon/)
