"""Train/evaluate phase-2 activity model with a held-out split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from host.inference.calibration import MotionThresholdCalibrator, save_thresholds
from host.inference.training import (
    predict_with_softmax,
    save_softmax_model,
    train_softmax_classifier,
)


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels_train: np.ndarray
    labels_test: np.ndarray
    train_sessions: list[str]
    test_sessions: list[str]


def load_dataset(
    csv_path: Path,
    label_col: str,
    session_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        drop_cols = {label_col, session_col, "timestamp_us"}
        feature_names = [name for name in fieldnames if name not in drop_cols]

        rows: list[list[float]] = []
        labels: list[str] = []
        sessions: list[str] = []
        for row in reader:
            rows.append([float(row[name]) for name in feature_names])
            labels.append(row[label_col])
            sessions.append(row.get(session_col, "unknown"))

    if not rows:
        raise ValueError("dataset CSV has no rows")

    x = np.asarray(rows, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=object)
    sessions_arr = np.asarray(sessions, dtype=object)
    return x, labels_arr, sessions_arr, feature_names


def _split_by_session(
    x: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    class_to_idx: dict[str, int],
    test_ratio: float,
    seed: int,
) -> SplitData:
    if x.shape[0] < 4:
        raise ValueError("need at least 4 rows for split/evaluation")
    if not (0.05 <= test_ratio <= 0.9):
        raise ValueError("test_ratio must be in [0.05, 0.9]")

    rng = np.random.default_rng(seed)
    unique_sessions = sorted({str(s) for s in sessions.tolist()})
    if len(unique_sessions) >= 2:
        shuffled = unique_sessions.copy()
        rng.shuffle(shuffled)
        test_count = int(round(len(shuffled) * test_ratio))
        test_count = max(1, min(len(shuffled) - 1, test_count))
        test_set = set(shuffled[:test_count])
        train_mask = np.asarray([str(s) not in test_set for s in sessions], dtype=bool)
    else:
        idx = np.arange(x.shape[0])
        rng.shuffle(idx)
        test_count = max(1, int(round(x.shape[0] * test_ratio)))
        test_count = min(x.shape[0] - 1, test_count)
        train_mask = np.ones((x.shape[0],), dtype=bool)
        train_mask[idx[:test_count]] = False

    if np.sum(train_mask) == 0 or np.sum(~train_mask) == 0:
        raise ValueError("invalid split produced an empty train or test set")

    x_train = x[train_mask]
    labels_train = labels[train_mask]
    x_test = x[~train_mask]
    labels_test = labels[~train_mask]

    y_train = np.asarray([class_to_idx[str(label)] for label in labels_train], dtype=np.int64)
    y_test = np.asarray([class_to_idx[str(label)] for label in labels_test], dtype=np.int64)
    train_sessions = sorted({str(s) for s in sessions[train_mask].tolist()})
    test_sessions = sorted({str(s) for s in sessions[~train_mask].tolist()})
    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        labels_train=labels_train,
        labels_test=labels_test,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, object]:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(truth), int(pred)] += 1

    rows: list[dict[str, object]] = []
    f1_values: list[float] = []
    for idx, class_name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(np.sum(cm[:, idx]) - tp)
        fn = int(np.sum(cm[idx, :]) - tp)
        support = int(np.sum(cm[idx, :]))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if support > 0:
            f1_values.append(f1)
        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": rows,
        "confusion_matrix": cm.tolist(),
    }


def _presence_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    true_occ = np.asarray([label != "empty" for label in true_labels.tolist()], dtype=np.int8)
    pred_occ = np.asarray([label != "empty" for label in pred_labels.tolist()], dtype=np.int8)

    tp = int(np.sum((true_occ == 1) & (pred_occ == 1)))
    fp = int(np.sum((true_occ == 0) & (pred_occ == 1)))
    fn = int(np.sum((true_occ == 1) & (pred_occ == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _render_report(
    dataset_csv: Path,
    output_model: Path,
    split: SplitData,
    metrics: dict[str, object],
    presence: dict[str, float],
    class_names: list[str],
    threshold_path: Path | None,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Dataset CSV: {dataset_csv}",
        f"- Output model: {output_model}",
        f"- Train rows: {split.x_train.shape[0]}",
        f"- Test rows: {split.x_test.shape[0]}",
        f"- Train sessions: {', '.join(split.train_sessions)}",
        f"- Test sessions: {', '.join(split.test_sessions)}",
        f"- Activity accuracy: {metrics['accuracy']:.4f}",
        f"- Activity macro-F1: {metrics['macro_f1']:.4f}",
        f"- Presence F1: {presence['f1']:.4f}",
    ]
    if threshold_path:
        lines.append(f"- Threshold file: {threshold_path}")
    lines.extend(
        [
            "",
            "## Activity Metrics",
            "",
            "| class | precision | recall | f1 | support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["per_class"]:
        lines.append(
            f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            f"Classes: {class_names}",
            "",
        ]
    )
    for matrix_row in metrics["confusion_matrix"]:
        lines.append("- " + ", ".join(str(v) for v in matrix_row))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a held-out phase-2 activity model")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold-output", default="")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    output_model = Path(args.output_model)
    output_report = Path(args.output_report)
    threshold_path = Path(args.threshold_output) if args.threshold_output else None

    x, labels, sessions, feature_names = load_dataset(
        csv_path=dataset_csv,
        label_col=args.label_col,
        session_col=args.session_col,
    )
    class_names = sorted({str(label) for label in labels.tolist()})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    split = _split_by_session(
        x=x,
        labels=labels,
        sessions=sessions,
        class_to_idx=class_to_idx,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = train_softmax_classifier(
        x=split.x_train,
        y_idx=split.y_train,
        feature_names=feature_names,
        class_names=class_names,
        epochs=args.epochs,
        learning_rate=args.lr,
        l2=args.l2,
    )
    probs = predict_with_softmax(model, split.x_test)
    y_pred = np.argmax(probs, axis=1)
    pred_labels = np.asarray([class_names[int(idx)] for idx in y_pred.tolist()], dtype=object)

    metrics = _compute_metrics(split.y_test, y_pred, class_names)
    presence = _presence_f1(split.labels_test, pred_labels)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    save_softmax_model(output_model, model)

    if threshold_path:
        if "motion_index" not in feature_names:
            raise ValueError("cannot calibrate thresholds: motion_index feature not found")
        calibrator = MotionThresholdCalibrator()
        motion_idx = feature_names.index("motion_index")
        for row_idx in range(split.x_train.shape[0]):
            calibrator.add(str(split.labels_train[row_idx]), float(split.x_train[row_idx, motion_idx]))
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        save_thresholds(threshold_path, calibrator.compute())

    output_report.parent.mkdir(parents=True, exist_ok=True)
    report = _render_report(
        dataset_csv=dataset_csv,
        output_model=output_model,
        split=split,
        metrics=metrics,
        presence=presence,
        class_names=class_names,
        threshold_path=threshold_path,
    )
    output_report.write_text(report, encoding="utf-8")

    print(f"Saved model: {output_model}")
    print(f"Saved report: {output_report}")
    if threshold_path:
        print(f"Saved thresholds: {threshold_path}")
    print(f"Activity accuracy: {metrics['accuracy']:.4f}")
    print(f"Activity macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Presence F1: {presence['f1']:.4f}")


if __name__ == "__main__":
    main()
