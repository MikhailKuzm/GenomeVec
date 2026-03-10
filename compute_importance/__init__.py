"""
Пакет compute_importance: подсчёт важности окон (участков генома) для предсказания класса.

Использование:
  from compute_importance import compute_window_importance, load_fcn_classifier
  importance_list, pred_class_idx, logits = compute_window_importance(tokens_per_contig, classifier, ...)
"""

from .fcn_importance import compute_window_importance, load_fcn_classifier

__all__ = ["compute_window_importance", "load_fcn_classifier"]
