#!/usr/bin/env python3
"""
Обучение FCN-классификатора на эмбеддингах: поддержка FASTA или готовых векторов,
таргет по умолчанию — фенотип бактерии (organism), сохранение в models/ (чекпоинт + YAML).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Корень репозитория и каталог train для логов
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
LOG_FILE = TRAIN_DIR / "training_log.txt"
METRICS_FILE = TRAIN_DIR / "final_metrics.txt"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from read_fasta import read_fasta
from embedders.DNABERT2 import FastaToEmbeddings
from classifiers import FCNClassifier


class EmbeddingDataset(Dataset):
    """Датасет пар (эмбеддинг, метка класса)."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"embeddings": self.embeddings[i], "labels": self.labels[i]}


def load_vectors_from_pt(vectors_path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Загружает .pt с ключом vectors_by_organism (dict: organism_name -> tensor (N, hidden)).
    Возвращает X, y (индексы классов), unique_classes (порядок классов).
    """
    payload = torch.load(vectors_path, map_location="cpu", weights_only=False)
    vo = payload["vectors_by_organism"]
    unique_classes = sorted(vo.keys())
    name_to_idx = {name: i for i, name in enumerate(unique_classes)}
    X_list = []
    y_list = []
    for name, vecs in vo.items():
        if isinstance(vecs, torch.Tensor):
            n = vecs.size(0)
            X_list.append(vecs)
        else:
            vecs = torch.stack([torch.tensor(v) for v in vecs])
            n = vecs.size(0)
            X_list.append(vecs)
        y_list.extend([name_to_idx[name]] * n)
    X = torch.cat(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y, unique_classes


def compute_embeddings_from_fasta(
    fasta_dir: Path,
    target_csv: Path | None,
    embedder: FastaToEmbeddings,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Обходит fasta_dir (*.fna), для каждого файла: read_fasta (метка из meta или CSV),
    embedder(path) -> matrix (num_windows, hidden), усредняем matrix.mean(dim=0).
    Возвращает X, y, unique_classes.
    """
    from tqdm import tqdm

    path_to_label: dict[str, str] = {}
    if target_csv is not None and target_csv.is_file():
        import csv
        with open(target_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("path", row.get("genome_id", ""))
                path_to_label[Path(p).resolve().as_posix()] = row.get("label", row.get("target", ""))

    fasta_files = sorted(fasta_dir.glob("*.fna"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"В {fasta_dir} не найдено .fna / .fasta")

    all_vectors = []
    all_labels_names = []
    for fasta_path in tqdm(fasta_files, desc="FASTA", unit="file"):
        _, meta = read_fasta(fasta_path)
        label = path_to_label.get(Path(fasta_path).resolve().as_posix()) or meta.get("organism_name", "unknown") or "unknown"
        matrix = embedder(fasta_path)
        vec = matrix.mean(dim=0).detach().cpu()
        all_vectors.append(vec)
        all_labels_names.append(label)

    X = torch.stack(all_vectors)
    unique_classes = sorted(set(all_labels_names))
    name_to_idx = {name: i for i, name in enumerate(unique_classes)}
    y = torch.tensor([name_to_idx[n] for n in all_labels_names], dtype=torch.long)
    return X, y, unique_classes


def run_training(
    *,
    fasta_dir: Path | None = None,
    vectors_path: Path | None = None,
    target_csv: Path | None = None,
    model_name: str = "fcn_classifier",
    output_dir: Path | None = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """
    Запуск обучения: загрузка/вычисление эмбеддингов, разбиение train/val/test,
    обучение FCNClassifier, сохранение в output_dir (по умолчанию models/).
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        print(msg)

    if vectors_path is not None:
        vectors_path = Path(vectors_path)
        X, y, unique_classes = load_vectors_from_pt(vectors_path)
    elif fasta_dir is not None:
        fasta_dir = Path(fasta_dir)
        embedder = FastaToEmbeddings(window_size=512, overlap_tokens=0)
        X, y, unique_classes = compute_embeddings_from_fasta(fasta_dir, target_csv, embedder)
    else:
        default_vectors = REPO_ROOT / "embedders" / "DNABERT2" / "vectors" / "dnabert2_vectors.pt"
        if default_vectors.is_file():
            X, y, unique_classes = load_vectors_from_pt(default_vectors)
        else:
            default_fasta = REPO_ROOT / "genoms" / "fasta"
            if default_fasta.is_dir():
                embedder = FastaToEmbeddings(window_size=512, overlap_tokens=0)
                X, y, unique_classes = compute_embeddings_from_fasta(default_fasta, target_csv, embedder)
            else:
                raise FileNotFoundError("Укажите --vectors-path или --fasta-dir, либо поместите векторы в embedders/DNABERT2/vectors/dnabert2_vectors.pt или FASTA в genoms/fasta")

    num_classes = len(unique_classes)
    hidden_size = X.shape[1]
    name_to_idx = {name: i for i, name in enumerate(unique_classes)}

    n = len(X)
    n_test = max(1, int((1 - train_ratio - val_ratio) * n))
    n_val = max(1, int(val_ratio * n))
    n_train = n - n_test - n_val
    gen = torch.Generator().manual_seed(seed)
    full_ds = EmbeddingDataset(X, y)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    LOG_FILE.write_text("", encoding="utf-8")
    log(f"Train: {n_train}, Val: {n_val}, Test: {n_test}, Classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNClassifier(hidden_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            lab = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(emb, labels=lab)
            out["loss"].backward()
            optimizer.step()
            train_loss += out["loss"].item() * emb.size(0)
            train_total += emb.size(0)
            train_correct += (out["logits"].argmax(dim=1) == lab).sum().item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embeddings"].to(device)
                lab = batch["labels"].to(device)
                out = model(emb, labels=lab)
                val_total += emb.size(0)
                val_correct += (out["logits"].argmax(dim=1) == lab).sum().item()

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        train_loss_avg = train_loss / train_total if train_total else 0
        log(f"Epoch {epoch+1}/{num_epochs}  train_loss={train_loss_avg:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            emb = batch["embeddings"].to(device)
            lab = batch["labels"].to(device)
            out = model(emb)
            test_total += emb.size(0)
            test_correct += (out["logits"].argmax(dim=1) == lab).sum().item()
    test_acc = test_correct / test_total if test_total else 0
    final_metrics_text = (
        f"Test accuracy: {test_acc:.4f}\n"
        f"Test samples: {test_total}\n"
        f"Num classes: {num_classes}\n"
    )
    METRICS_FILE.write_text(final_metrics_text, encoding="utf-8")
    log("--- Final ---")
    log(final_metrics_text.strip())

    # Сохранение: чекпоинт + YAML
    base_name = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    ckpt_path = output_dir / base_name
    save_payload = {
        "state_dict": model.state_dict(),
        "unique_classes": unique_classes,
        "name_to_idx": name_to_idx,
        "config": model.get_config(),
    }
    torch.save(save_payload, ckpt_path)

    num_params = sum(p.numel() for p in model.parameters())
    try:
        import yaml
        yaml_path = output_dir / (Path(base_name).stem + "_config.yaml")
        yaml_config = {
            "architecture": model.get_config(),
            "weights": str(ckpt_path),
            "num_parameters": num_params,
            "train_samples": n_train,
            "val_samples": n_val,
            "test_metrics": {
                "accuracy": float(test_acc),
                "test_samples": int(test_total),
                "num_classes": num_classes,
            },
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        pass

    log(f"Model saved to {ckpt_path}")
    return {"test_accuracy": test_acc, "ckpt_path": str(ckpt_path), "unique_classes": unique_classes}


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение FCN-классификатора на эмбеддингах")
    p.add_argument("--fasta-dir", type=Path, default=None, help="Каталог с FASTA (.fna). Если не задан и нет --vectors-path, используется genoms/fasta или embedders/.../vectors/")
    p.add_argument("--vectors-path", type=Path, default=None, help="Путь к .pt с vectors_by_organism")
    p.add_argument("--target-csv", type=Path, default=None, help="CSV с колонками path (или genome_id) и label для переопределения таргета")
    p.add_argument("--model-name", type=str, default="fcn_classifier", help="Имя модели для сохранения в models/")
    p.add_argument("--output-dir", type=Path, default=None, help="Каталог для чекпоинта и YAML (по умолчанию models/)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_training(
        fasta_dir=args.fasta_dir,
        vectors_path=args.vectors_path,
        target_csv=args.target_csv,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
