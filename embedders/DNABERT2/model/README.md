# DNABERT2 model

Здесь должны лежать файлы модели DNABERT2 (конфиг, токенизатор, веса).

## Вариант 1: Скачать в эту папку

```bash
# из корня репозитория
huggingface-cli download zhihan1996/DNABERT-2-117M --local-dir embedders/DNABERT2/model
```

После этого в `embedders/DNABERT2/model` появятся `config.json`, файлы токенизатора, веса и т.д.

## Вариант 2: Без скачивания

Если папка пустая или без `config.json`, код будет загружать модель с Hugging Face при первом запуске: `from_pretrained("zhihan1996/DNABERT-2-117M")`.
