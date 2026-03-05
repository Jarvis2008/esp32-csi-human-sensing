"""Dataset utilities for phase-2 labeled session exports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAINING_LABELS = ("empty", "stationary", "walking", "gesture", "fall_like")


@dataclass
class DatasetExportResult:
    output_csv: Path
    feature_names: list[str]
    total_rows: int
    kept_rows: int
    skipped_unlabeled: int
    skipped_unknown_label: int
    label_counts: dict[str, int]


def _load_window_npz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    bundle = np.load(path, allow_pickle=True)
    feature_names = [str(name) for name in bundle["feature_names"].tolist()]
    values = np.asarray(bundle["values"], dtype=np.float32)

    if "timestamp_us" in bundle:
        timestamp_us = np.asarray(bundle["timestamp_us"], dtype=np.uint64)
    else:
        timestamp_us = np.arange(values.shape[0], dtype=np.uint64)

    if "label" in bundle:
        labels = np.asarray(bundle["label"], dtype=object)
    else:
        labels = np.asarray(["unlabeled"] * values.shape[0], dtype=object)

    if values.shape[0] != timestamp_us.shape[0] or values.shape[0] != labels.shape[0]:
        raise ValueError(f"row alignment mismatch in {path}")

    return feature_names, values, timestamp_us, labels


def write_labeled_dataset_csv(
    session_root: Path,
    output_csv: Path,
    include_unlabeled: bool = False,
    allowed_labels: tuple[str, ...] = TRAINING_LABELS,
) -> DatasetExportResult:
    session_root = Path(session_root)
    output_csv = Path(output_csv)
    allowed = set(allowed_labels)

    sources: list[tuple[str, Path]] = []
    for session_dir in sorted(
        [path for path in session_root.glob("session_*") if path.is_dir()],
        key=lambda p: p.name,
    ):
        window_path = session_dir / "window_features.npz"
        if window_path.exists():
            sources.append((session_dir.name, window_path))

    # Backward compatibility for phase-1 flat artifact paths.
    for legacy_path in sorted(session_root.glob("window_features_*.npz"), key=lambda p: p.name):
        suffix = legacy_path.stem.replace("window_features_", "", 1)
        sources.append((f"legacy_{suffix}", legacy_path))

    if not sources:
        raise FileNotFoundError(f"no window feature artifacts found under {session_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    baseline_feature_names: list[str] | None = None
    total_rows = 0
    kept_rows = 0
    skipped_unlabeled = 0
    skipped_unknown = 0
    label_counts: dict[str, int] = {}

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer: csv.DictWriter | None = None

        for session_id, window_path in sources:
            feature_names, values, timestamps, labels = _load_window_npz(window_path)
            if baseline_feature_names is None:
                baseline_feature_names = feature_names
            elif feature_names != baseline_feature_names:
                raise ValueError(
                    f"feature schema mismatch for {window_path}: "
                    f"expected {baseline_feature_names}, got {feature_names}"
                )

            if writer is None:
                header = ["session_id", "timestamp_us", "label", *baseline_feature_names]
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

            assert writer is not None
            for row_idx in range(values.shape[0]):
                total_rows += 1
                label = str(labels[row_idx])
                if label == "unlabeled" and not include_unlabeled:
                    skipped_unlabeled += 1
                    continue
                if label != "unlabeled" and label not in allowed:
                    skipped_unknown += 1
                    continue

                row: dict[str, object] = {
                    "session_id": session_id,
                    "timestamp_us": int(timestamps[row_idx]),
                    "label": label,
                }
                for feature_idx, feature_name in enumerate(feature_names):
                    row[feature_name] = float(values[row_idx, feature_idx])
                writer.writerow(row)
                kept_rows += 1
                label_counts[label] = label_counts.get(label, 0) + 1

    if baseline_feature_names is None:
        raise FileNotFoundError(f"no window_features.npz files found under {session_root}")

    return DatasetExportResult(
        output_csv=output_csv,
        feature_names=baseline_feature_names,
        total_rows=total_rows,
        kept_rows=kept_rows,
        skipped_unlabeled=skipped_unlabeled,
        skipped_unknown_label=skipped_unknown,
        label_counts=label_counts,
    )
