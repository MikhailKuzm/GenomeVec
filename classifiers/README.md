# classifiers

Пакет с классификаторами по эмбеддингам геномов.

## Назначение

- **FCNClassifier** — полносвязная сеть (Linear → ReLU → Dropout → Linear) для классификации по заданному набору классов (например фенотип бактерии по organism_name).
- Используется в пайплайне обучения в `train/fcn_train.py`.

## Использование

```python
from classifiers import FCNClassifier

model = FCNClassifier(
    input_dim=768,   # размерность эмбеддинга DNABERT2
    num_classes=10,
    hidden=256,
    dropout=0.2,
)
# forward(embeddings, labels=None) → {"logits", "loss"?}
```
