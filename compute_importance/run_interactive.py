"""
Интерактивный пайплайн: загрузка FASTA → контиги → токенизация → эмбеддинги → классификатор
→ предсказание класса → градиент по логиту победителя → важность по каждому окну.
Запуск из корня репозитория: python -i compute_importance/run_interactive.py
или в REPL после выполнения скрипта — смотреть переменные fasta_path, importance, predicted_class_name и т.д.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Корень репозитория. В REPL без запуска скрипта __file__ нет — используем cwd (запускайте из корня репозитория)
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from read_fasta import read_fasta
from embedders.DNABERT2.tokenize_genome import tokenize_contigs
from compute_importance.fcn_importance import (
    compute_window_importance,
    load_fcn_classifier,
)

# --- Пути по умолчанию ---
FASTA_DIR = REPO_ROOT / "genoms" / "fasta"
CLASSIFIER_CHECKPOINT = REPO_ROOT / "models" / "fcn_organism.pt"


def run_pipeline(
    fasta_path: str | Path,
    classifier_path: str | Path = CLASSIFIER_CHECKPOINT,
    dnabert_model_path: str | Path | None = None,
) -> dict:
    """
    Полный пайплайн: FASTA → важность по окнам.

    Шаги:
      1. Чтение FASTA (read_fasta) → контиги и метаданные.
      2. Токенизация контигов (tokenize_contigs) → tokens_per_contig.
      3. Загрузка FCN-классификатора из чекпоинта.
      4. compute_window_importance(tokens_per_contig, classifier, ...) → важность по окнам.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA не найден: {fasta_path}")

    # 1. Загрузка и разбор FASTA
    contigs, meta = read_fasta(fasta_path)
    if not contigs:
        raise ValueError("В FASTA нет последовательностей")

    # 2. Токенизация (как при инференсе в embedders/DNABERT2)
    tokens_per_contig = tokenize_contigs(contigs, model_path=dnabert_model_path)

    # 3. Загрузка классификатора
    classifier, unique_classes = load_fcn_classifier(classifier_path)

    # 4. Подсчёт важности окон (внутри: эмбеддинги DNABERT2, усреднение, классификатор, backward)
    importance_list, predicted_class_idx, logits = compute_window_importance(
        tokens_per_contig,
        classifier,
        model_path=dnabert_model_path,
    )

    predicted_class_name = unique_classes[predicted_class_idx] if predicted_class_idx < len(unique_classes) else str(predicted_class_idx)

    return {
        "fasta_path": str(fasta_path),
        "contigs": contigs,
        "meta": meta,
        "tokens_per_contig": tokens_per_contig,
        "num_windows": len(importance_list),
        "importance": importance_list,
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_name": predicted_class_name,
        "unique_classes": unique_classes,
        "logits": logits,
    }


def main() -> None:
    """Точка входа: один FASTA из genoms/fasta по умолчанию, вывод в консоль."""
    fasta_files = sorted((FASTA_DIR).glob("*.fna"))
    if not fasta_files:
        fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    if not fasta_files:
        print(f"В {FASTA_DIR} не найдено .fna/.fasta. Укажите fasta_path вручную и вызовите run_pipeline(fasta_path).")
        return

    fasta_path = fasta_files[0]
    print(f"Загрузка FASTA: {fasta_path}")
    result = run_pipeline(fasta_path)

    importance = result["importance"]
    pred_name = result["predicted_class_name"]
    num_windows = result["num_windows"]

    print(f"\nПредсказанный класс: {pred_name} (индекс {result['predicted_class_idx']})")
    print(f"Окон: {num_windows}")
    print("\nВажность по окнам (первые 20 и последние 5):")
    for i in range(min(20, num_windows)):
        print(f"  окно {i}: {importance[i]:.6f}")
    if num_windows > 25:
        print("  ...")
        for i in range(max(20, num_windows - 5), num_windows):
            print(f"  окно {i}: {importance[i]:.6f}")

    # Топ-5 окон по важности (по убыванию)
    indexed = sorted(enumerate(importance), key=lambda x: -abs(x[1]))
    print("\nТоп-5 окон по |важность|:")
    for rank, (win_idx, val) in enumerate(indexed[:5], 1):
        print(f"  {rank}. окно {win_idx}: {val:.6f}")

    # Сохраняем в глобальные переменные для интерактивного просмотра
    globals().update(
        fasta_path=result["fasta_path"],
        importance=importance,
        predicted_class_name=pred_name,
        predicted_class_idx=result["predicted_class_idx"],
        unique_classes=result["unique_classes"],
        num_windows=num_windows,
        result=result,
    )
    print("\nПеременные в REPL: fasta_path, importance, predicted_class_name, result, num_windows, unique_classes")


if __name__ == "__main__":
    main()
