"""Feature extraction for presence/activity baseline."""

from __future__ import annotations

import numpy as np


def iq_to_amplitude_phase(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    complex_vals = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)
    return np.abs(complex_vals), np.angle(complex_vals)


def motion_index(window_amplitudes: np.ndarray) -> float:
    if window_amplitudes.shape[0] < 2:
        return 0.0
    diffs = np.diff(window_amplitudes, axis=0)
    return float(np.mean(np.abs(diffs)))


def summarize_window(window_amplitudes: np.ndarray) -> dict[str, float]:
    return {
        "amp_mean": float(np.mean(window_amplitudes)),
        "amp_std": float(np.std(window_amplitudes)),
        "motion_index": motion_index(window_amplitudes),
    }
