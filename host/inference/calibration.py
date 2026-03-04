"""Threshold calibration helpers for baseline model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass
class ThresholdSet:
    presence_threshold: float
    walk_threshold: float
    gesture_threshold: float


class MotionThresholdCalibrator:
    """Calibrate motion thresholds from labeled windows."""

    def __init__(self) -> None:
        self._by_label: dict[str, list[float]] = {
            "empty": [],
            "stationary": [],
            "walking": [],
            "gesture": [],
        }

    def add(self, label: str, motion_index: float) -> None:
        if label in self._by_label:
            self._by_label[label].append(float(motion_index))

    def compute(self) -> ThresholdSet:
        empty = np.asarray(self._by_label["empty"] or [2.0], dtype=np.float32)
        stationary = np.asarray(self._by_label["stationary"] or [10.0], dtype=np.float32)
        walking = np.asarray(self._by_label["walking"] or [20.0], dtype=np.float32)
        gesture = np.asarray(self._by_label["gesture"] or [30.0], dtype=np.float32)

        presence = float(np.percentile(empty, 95))
        walk = float(np.percentile(stationary, 95))
        gesture_thr = float(np.percentile(walking, 95))
        # Ensure ordered thresholds even with sparse labels.
        walk = max(walk, presence + 1.0)
        gesture_thr = max(gesture_thr, walk + 1.0)

        # Allow gesture class to stretch upper threshold when available.
        if gesture.size > 0:
            gesture_thr = min(gesture_thr, float(np.percentile(gesture, 50)))

        return ThresholdSet(
            presence_threshold=presence,
            walk_threshold=walk,
            gesture_threshold=gesture_thr,
        )


def save_thresholds(path: Path, thresholds: ThresholdSet) -> None:
    path.write_text(json.dumps(asdict(thresholds), indent=2), encoding="utf-8")


def load_thresholds(path: Path) -> ThresholdSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThresholdSet(
        presence_threshold=float(payload["presence_threshold"]),
        walk_threshold=float(payload["walk_threshold"]),
        gesture_threshold=float(payload["gesture_threshold"]),
    )
