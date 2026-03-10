"""
Подсчёт важности окон для предсказания класса: градиент логита выигравшего класса по E,
затем вклад каждого окна = g · e_i (в том же порядке, что и окна на входе).

Использование:
  from compute_importance import compute_window_importance
  importance = compute_window_importance(tokens_per_contig, classifier, ...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Корень репозитория для импорта embedders
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embedders.DNABERT2.embeddings import (
    _load_model_and_tokenizer,
    embeddings_matrix_for_windows,
)


def compute_window_importance(
    tokens_per_contig: list[list[int]],
    classifier: torch.nn.Module,
    *,
    model=None,
    tokenizer=None,
    model_path: str | Path | None = None,
    window_size: int = 512,
    overlap_tokens: int = 0,
    device: str | torch.device | None = None,
    batch_size: int = 1,
) -> tuple[list[float], int, torch.Tensor]:
    """
    Считает важность каждого окна для предсказанного класса (класса с наибольшим логитом).

    Логика:
      1. По токенам контигов строится матрица эмбеддингов DNABERT2: по одному вектору на окно.
      2. Усредняем по окнам: E = mean(окна). E подаётся в классификатор → logits.
      3. Предсказанный класс = argmax(logits). Берём логит этого класса и делаем backward.
      4. g = ∂(logit_победителя)/∂E. Важность окна i = g · e_i (вклад окна в текущий логит).
      5. Порядок важностей совпадает с порядком окон (сверху вниз по контигам).

    Args:
        tokens_per_contig: список списков token id по контигам (как от tokenize_contigs).
        classifier: обученный FCN-классификатор (или другой nn.Module с forward(embeddings) -> dict с "logits").
        model, tokenizer: модель и токенизатор DNABERT2 (если None — загружаются по model_path или HF).
        model_path: путь к папке DNABERT2 (если model/tokenizer не переданы).
        window_size, overlap_tokens: параметры скользящего окна для эмбеддингов.
        device: устройство для вычислений (по умолчанию cpu или устройство классификатора).
        batch_size: батч при расчёте эмбеддингов окон.

    Returns:
        (importance_list, predicted_class_idx, logits):
        - importance_list: список float, важность по каждому окну в порядке окон.
        - predicted_class_idx: индекс предсказанного класса (argmax logits).
        - logits: тензор логитов (1, num_classes) для возможной дальнейшей интерпретации.
    """
    if not tokens_per_contig:
        raise ValueError("tokens_per_contig не должен быть пустым")

    # Устройство классификатора (на нём считаем E и backward)
    classifier_device = next(classifier.parameters()).device
    if device is not None:
        classifier_device = torch.device(device) if isinstance(device, str) else device

    # DNABERT2 использует Flash Attention — он требует CUDA. Запускаем эмбеддер на GPU, если доступен.
    embedder_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if embedder_device.type == "cpu":
        raise RuntimeError(
            "DNABERT2 в этой конфигурации требует CUDA (Flash Attention). "
            "Запустите на машине с GPU или используйте вариант модели без Flash Attention (например quietflamingo/dnabert2-no-flashattention)."
        )

    # Матрица эмбеддингов по окнам (num_windows, hidden_size); градиент через BERT не нужен
    matrix, meta = embeddings_matrix_for_windows(
        tokens_per_contig,
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
        window_size=window_size,
        overlap_tokens=overlap_tokens,
        device=embedder_device,
        batch_size=batch_size,
    )
    # Переносим на устройство классификатора для дальнейших вычислений
    matrix = matrix.to(classifier_device)

    num_windows = matrix.size(0)
    # Копия с графом: градиент будет течь в E = mean(matrix_grad), затем в классификатор
    matrix_grad = matrix.detach().clone().requires_grad_(True)
    E = matrix_grad.mean(dim=0)
    E.retain_grad()  # чтобы после backward() сохранить E.grad (E не листовой тензор)

    # Классификатор в режиме eval; один вектор E -> logits
    classifier.eval()
    out = classifier(E.unsqueeze(0))  # (1, num_classes)
    logits = out["logits"]

    predicted_class_idx = logits[0].argmax().item()
    logit_win = logits[0, predicted_class_idx]

    logit_win.backward()

    # g = ∂(logit_победителя)/∂E — градиент логита выигравшего класса по усреднённому эмбеддингу
    g = E.grad
    if g is None:
        raise RuntimeError("E.grad пуст: убедитесь, что E.retain_grad() вызван до backward()")

    # Важность окна i = g · e_i (скаляр на окно); порядок как у окон
    # (g * matrix_grad).sum(dim=1) даёт тензор (num_windows,); matrix_grad в графе, для суммы используем .detach() для ясности
    importance_tensor = (g * matrix_grad.detach()).sum(dim=1)
    importance_list = importance_tensor.cpu().tolist()

    return importance_list, predicted_class_idx, logits.detach()


def load_fcn_classifier(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, list[str]]:
    """
    Загружает FCN-классификатор из чекпоинта (.pt с state_dict и config).

    Args:
        checkpoint_path: путь к .pt (например models/fcn_organism.pt).
        device: устройство для модели.

    Returns:
        (model, unique_classes): модель и список имён классов в порядке индексов.
    """
    from classifiers import FCNClassifier

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config")
    if not config:
        raise KeyError("В чекпоинте отсутствует ключ 'config'")

    model = FCNClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        hidden=config.get("hidden", 256),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()

    unique_classes = payload.get("unique_classes", [str(i) for i in range(config["num_classes"])])
    return model, list(unique_classes)
