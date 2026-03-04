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


def _signal_entropy(values: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log(hist + 1e-9)))


def _spectral_band_energy(signal: np.ndarray, sample_rate_hz: float, band_hz: tuple[float, float]) -> float:
    if signal.size < 4:
        return 0.0
    centered = signal - np.mean(signal)
    fft = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate_hz)
    low, high = band_hz
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    power = np.abs(fft) ** 2
    return float(np.mean(power[mask]))


def summarize_window(window_amplitudes: np.ndarray, sample_rate_hz: float = 50.0) -> dict[str, float]:
    mean_series = np.mean(window_amplitudes, axis=1)
    std_series = np.std(window_amplitudes, axis=1)
    return {
        "amp_mean": float(np.mean(window_amplitudes)),
        "amp_std": float(np.std(window_amplitudes)),
        "amp_entropy": _signal_entropy(window_amplitudes.ravel()),
        "motion_index": motion_index(window_amplitudes),
        "temporal_std_mean": float(np.mean(std_series)),
        "energy_low": _spectral_band_energy(mean_series, sample_rate_hz, (0.1, 2.0)),
        "energy_mid": _spectral_band_energy(mean_series, sample_rate_hz, (2.0, 7.0)),
        "energy_high": _spectral_band_energy(mean_series, sample_rate_hz, (7.0, 15.0)),
    }
