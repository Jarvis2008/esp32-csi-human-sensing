"""Simple deterministic baseline activity model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActivityState:
    presence: str
    activity: str
    confidence: float


class ThresholdActivityModel:
    """Rule-based baseline until trained classifier is available."""

    def __init__(self, presence_threshold: float = 8.0, walk_threshold: float = 18.0, gesture_threshold: float = 28.0):
        self.presence_threshold = presence_threshold
        self.walk_threshold = walk_threshold
        self.gesture_threshold = gesture_threshold

    def predict(self, features: dict[str, float]) -> ActivityState:
        motion = features.get("motion_index", 0.0)
        if motion < self.presence_threshold:
            return ActivityState("empty", "empty", 0.90)
        if motion < self.walk_threshold:
            return ActivityState("occupied", "stationary", 0.75)
        if motion < self.gesture_threshold:
            return ActivityState("occupied", "walking", 0.72)
        return ActivityState("occupied", "gesture", 0.68)
