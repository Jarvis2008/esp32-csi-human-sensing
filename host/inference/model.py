"""Activity model implementations for phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from host.inference.calibration import ThresholdSet, load_thresholds
from host.inference.training import TrainedSoftmaxModel, load_softmax_model, predict_with_softmax


@dataclass
class ActivityState:
    presence: str
    activity: str
    confidence: float


class ThresholdActivityModel:
    """Rule-based baseline until trained classifier is available."""

    def __init__(
        self,
        presence_threshold: float = 8.0,
        walk_threshold: float = 18.0,
        gesture_threshold: float = 28.0,
    ):
        self.presence_threshold = presence_threshold
        self.walk_threshold = walk_threshold
        self.gesture_threshold = gesture_threshold

    @classmethod
    def from_thresholds(cls, thresholds: ThresholdSet) -> "ThresholdActivityModel":
        return cls(
            presence_threshold=thresholds.presence_threshold,
            walk_threshold=thresholds.walk_threshold,
            gesture_threshold=thresholds.gesture_threshold,
        )

    def predict(self, features: dict[str, float]) -> ActivityState:
        motion = features.get("motion_index", 0.0)
        if motion < self.presence_threshold:
            return ActivityState("empty", "empty", 0.90)
        if motion < self.walk_threshold:
            return ActivityState("occupied", "stationary", 0.75)
        if motion < self.gesture_threshold:
            return ActivityState("occupied", "walking", 0.72)
        return ActivityState("occupied", "gesture", 0.68)


class SoftmaxActivityModel:
    def __init__(self, model: TrainedSoftmaxModel) -> None:
        self.model = model

    def predict(self, features: dict[str, float]) -> ActivityState:
        x = np.asarray([[features.get(name, 0.0) for name in self.model.feature_names]], dtype=np.float32)
        probs = predict_with_softmax(self.model, x)[0]
        idx = int(np.argmax(probs))
        activity = self.model.class_names[idx]
        presence = "empty" if activity == "empty" else "occupied"
        confidence = float(probs[idx])
        return ActivityState(presence=presence, activity=activity, confidence=confidence)


def build_model(
    model_path: Path | None = None,
    threshold_path: Path | None = None,
) -> ThresholdActivityModel | SoftmaxActivityModel:
    if model_path and model_path.exists():
        return SoftmaxActivityModel(load_softmax_model(model_path))

    if threshold_path and threshold_path.exists():
        return ThresholdActivityModel.from_thresholds(load_thresholds(threshold_path))

    return ThresholdActivityModel()
