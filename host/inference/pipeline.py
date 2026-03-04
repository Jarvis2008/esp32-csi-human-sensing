"""Windowing + baseline inference pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import numpy as np

from host.inference.features import iq_to_amplitude_phase, summarize_window
from host.inference.model import ThresholdActivityModel


class InferencePipeline:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.amp_window: deque[np.ndarray] = deque(maxlen=window_size)
        self.model = ThresholdActivityModel()
        self.last_features: dict[str, float] = {}
        self.last_state: dict[str, object] = {
            "presence": "unknown",
            "activity": "unknown",
            "confidence": 0.0,
        }

    def add_frame(self, csi_iq: tuple[int, ...]) -> dict[str, object] | None:
        if not csi_iq:
            return None
        arr = np.asarray(csi_iq, dtype=np.int8)
        if arr.size % 2 != 0:
            return None
        iq = arr.reshape(-1, 2)
        amp, _phase = iq_to_amplitude_phase(iq)
        self.amp_window.append(amp)
        if len(self.amp_window) < self.window_size:
            return None

        window = np.stack(self.amp_window, axis=0)
        self.last_features = summarize_window(window)
        state = self.model.predict(self.last_features)
        self.last_state = asdict(state)
        return self.last_state
