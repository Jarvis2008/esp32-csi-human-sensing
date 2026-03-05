import csv
from pathlib import Path

import numpy as np
import pytest

from host.inference.dataset import write_labeled_dataset_csv


def _write_window_file(
    path: Path,
    feature_names: list[str],
    values: list[list[float]],
    labels: list[str],
    timestamps: list[int],
) -> None:
    np.savez(
        path,
        feature_names=np.asarray(feature_names, dtype=object),
        values=np.asarray(values, dtype=np.float32),
        label=np.asarray(labels, dtype=object),
        timestamp_us=np.asarray(timestamps, dtype=np.uint64),
    )


def test_export_labeled_dataset_csv_filters_unlabeled_and_unknown(tmp_path: Path):
    s1 = tmp_path / "session_1"
    s1.mkdir()
    _write_window_file(
        s1 / "window_features.npz",
        feature_names=["amp_mean", "motion_index"],
        values=[[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]],
        labels=["empty", "unlabeled", "alien"],
        timestamps=[100, 200, 300],
    )

    output_csv = tmp_path / "features.csv"
    result = write_labeled_dataset_csv(
        session_root=tmp_path,
        output_csv=output_csv,
        include_unlabeled=False,
        allowed_labels=("empty", "walking"),
    )

    assert result.total_rows == 3
    assert result.kept_rows == 1
    assert result.skipped_unlabeled == 1
    assert result.skipped_unknown_label == 1
    assert result.label_counts == {"empty": 1}

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "session_1"
    assert rows[0]["label"] == "empty"
    assert rows[0]["timestamp_us"] == "100"


def test_export_labeled_dataset_csv_rejects_feature_mismatch(tmp_path: Path):
    s1 = tmp_path / "session_1"
    s2 = tmp_path / "session_2"
    s1.mkdir()
    s2.mkdir()
    _write_window_file(
        s1 / "window_features.npz",
        feature_names=["amp_mean", "motion_index"],
        values=[[1.0, 0.1]],
        labels=["empty"],
        timestamps=[100],
    )
    _write_window_file(
        s2 / "window_features.npz",
        feature_names=["amp_mean", "amp_std"],
        values=[[2.0, 0.2]],
        labels=["walking"],
        timestamps=[200],
    )

    with pytest.raises(ValueError):
        write_labeled_dataset_csv(
            session_root=tmp_path,
            output_csv=tmp_path / "features.csv",
        )
