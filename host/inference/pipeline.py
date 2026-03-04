"""Windowing + baseline inference pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from host.inference.features import iq_to_amplitude_phase, summarize_window
from host.inference.model import build_model


class InferencePipeline:
    def __init__(
        self,
        window_size: int = 50,
        sample_rate_hz: float = 50.0,
        model_path: Path | None = None,
        threshold_path: Path | None = None,
    ):
        self.window_size = window_size
        self.sample_rate_hz = sample_rate_hz
        self.amp_window: deque[np.ndarray] = deque(maxlen=window_size)
        self.model = build_model(model_path=model_path, threshold_path=threshold_path)
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
        self.last_features = summarize_window(window, sample_rate_hz=self.sample_rate_hz)
        state = self.model.predict(self.last_features)
        self.last_state = asdict(state)
        return self.last_state
