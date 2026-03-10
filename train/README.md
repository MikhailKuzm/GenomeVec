# train

Скрипты обучения классификаторов по эмбеддингам геномов.

## Назначение

- **fcn_train.py** — обучение FCN-классификатора на эмбеддингах.
- Поддержка двух режимов: загрузка готовых векторов (`--vectors-path`) или вычисление по FASTA (`--fasta-dir`) с помощью эмбеддера (усреднение по окнам выполняется в скрипте).
- Таргет по умолчанию — фенотип бактерии (organism из метаданных FASTA). Таргет можно переопределить через `--target-csv` (колонки: path или genome_id, label).
- Результат сохраняется в каталог **models/**: файл весов (`.pt`) и YAML-конфиг с описанием архитектуры, путём к весам, метриками на тесте и количеством параметров.

## Использование

```bash
# Из корня репозитория
python -m train.fcn_train --fasta-dir genoms/fasta --model-name fcn_organism
python -m train.fcn_train --vectors-path embedders/DNABERT2/vectors/dnabert2_vectors.pt --model-name fcn_organism
```

Программно:

```python
from train import run_training
run_training(fasta_dir=Path("genoms/fasta"), model_name="fcn_organism")
```
