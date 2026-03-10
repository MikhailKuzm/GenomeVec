"""
FCN-классификатор: линейные слои по эмбеддингам, выход — logits по классам.
Используется в пайплайне обучения в train/.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FCNClassifier(nn.Module):
    """
    Многослойный классификатор: вход — вектор эмбеддинга, выход — logits по классам.
    Архитектура: Linear → ReLU → Dropout → Linear.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._hidden = hidden
        self._dropout = dropout

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        """
        Args:
            embeddings: (batch, input_dim)
            labels: (batch,) — опционально, для расчёта loss

        Returns:
            dict с "logits" и при переданных labels — "loss".
        """
        x = self.dropout(self.relu(self.fc1(embeddings)))
        logits = self.fc2(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out

    def get_config(self) -> dict:
        """Параметры архитектуры для сохранения в YAML."""
        return {
            "input_dim": self._input_dim,
            "num_classes": self._num_classes,
            "hidden": self._hidden,
            "dropout": self._dropout,
        }
