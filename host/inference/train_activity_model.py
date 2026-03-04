"""Train a simple softmax activity model from CSV features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from host.inference.training import save_softmax_model, train_softmax_classifier


def load_dataset(csv_path: Path, label_col: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feature_names = [name for name in fieldnames if name != label_col]

        rows: list[list[float]] = []
        labels: list[str] = []
        for row in reader:
            rows.append([float(row[name]) for name in feature_names])
            labels.append(row[label_col])

    classes = sorted(set(labels))
    class_to_idx = {name: i for i, name in enumerate(classes)}
    y = np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)
    x = np.asarray(rows, dtype=np.float32)
    return x, y, feature_names, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phase-1 softmax activity model")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args()

    x, y, feature_names, class_names = load_dataset(Path(args.dataset_csv), args.label_col)
    model = train_softmax_classifier(
        x=x,
        y_idx=y,
        feature_names=feature_names,
        class_names=class_names,
        epochs=args.epochs,
        learning_rate=args.lr,
        l2=args.l2,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_softmax_model(output, model)
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
