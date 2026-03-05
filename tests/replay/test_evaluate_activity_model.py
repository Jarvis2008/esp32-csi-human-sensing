import csv
from pathlib import Path

import numpy as np

from host.inference.evaluate_activity_model import (
    _compute_metrics,
    _presence_f1,
    _split_by_session,
    load_dataset,
)


def test_split_by_session_keeps_sessions_disjoint():
    x = np.asarray([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]], dtype=np.float32)
    labels = np.asarray(["empty", "empty", "walking", "walking", "gesture", "gesture"], dtype=object)
    sessions = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"], dtype=object)
    class_to_idx = {"empty": 0, "walking": 1, "gesture": 2}

    split = _split_by_session(
        x=x,
        labels=labels,
        sessions=sessions,
        class_to_idx=class_to_idx,
        test_ratio=0.34,
        seed=7,
    )

    assert split.x_train.shape[0] > 0
    assert split.x_test.shape[0] > 0
    assert set(split.train_sessions).isdisjoint(set(split.test_sessions))


def test_metrics_compute_expected_values():
    y_true = np.asarray([0, 1, 1, 0], dtype=np.int64)
    y_pred = np.asarray([0, 1, 0, 0], dtype=np.int64)
    metrics = _compute_metrics(y_true, y_pred, ["empty", "walking"])

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] > 0.7
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_presence_f1_maps_empty_vs_occupied():
    presence = _presence_f1(
        true_labels=np.asarray(["empty", "walking", "gesture"], dtype=object),
        pred_labels=np.asarray(["empty", "empty", "gesture"], dtype=object),
    )
    assert 0.0 <= presence["f1"] <= 1.0
    assert presence["f1"] < 1.0


def test_load_dataset_parses_expected_columns(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session_id", "timestamp_us", "label", "amp_mean", "motion_index"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "session_1",
                "timestamp_us": "123",
                "label": "walking",
                "amp_mean": "1.2",
                "motion_index": "0.3",
            }
        )

    x, labels, sessions, feature_names = load_dataset(
        csv_path=csv_path,
        label_col="label",
        session_col="session_id",
    )
    assert x.shape == (1, 2)
    assert labels.tolist() == ["walking"]
    assert sessions.tolist() == ["session_1"]
    assert feature_names == ["amp_mean", "motion_index"]
