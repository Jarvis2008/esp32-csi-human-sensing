"""Numpy-only training utilities for phase-1 activity model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class TrainedSoftmaxModel:
    feature_names: list[str]
    class_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: np.ndarray


def _sanitize_features(x: np.ndarray) -> np.ndarray:
    clean = np.asarray(x, dtype=np.float32)
    clean = np.nan_to_num(clean, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(clean, -1e6, 1e6)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def train_softmax_classifier(
    x: np.ndarray,
    y_idx: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-4,
) -> TrainedSoftmaxModel:
    x = _sanitize_features(x)
    n_samples, n_features = x.shape
    n_classes = len(class_names)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x_norm = np.clip((x - mean) / std, -50.0, 50.0)

    rng = np.random.default_rng(7)
    weights = rng.normal(scale=0.05, size=(n_features, n_classes)).astype(np.float32)
    bias = np.zeros((n_classes,), dtype=np.float32)

    y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    y_onehot[np.arange(n_samples), y_idx] = 1.0

    for _ in range(epochs):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = x_norm @ weights + bias
        logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
        logits = np.clip(logits, -60.0, 60.0)
        probs = _softmax(logits)
        err = probs - y_onehot

        grad_w = (x_norm.T @ err) / n_samples + l2 * weights
        grad_b = np.mean(err, axis=0)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        weights = np.clip(weights, -10.0, 10.0)
        bias = np.clip(bias, -10.0, 10.0)

    return TrainedSoftmaxModel(
        feature_names=feature_names,
        class_names=class_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
    )


def predict_with_softmax(model: TrainedSoftmaxModel, x: np.ndarray) -> np.ndarray:
    x = _sanitize_features(x)
    x_norm = np.clip((x - model.mean) / model.std, -50.0, 50.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = x_norm @ model.weights + model.bias
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    logits = np.clip(logits, -60.0, 60.0)
    return _softmax(logits)


def save_softmax_model(path: Path, model: TrainedSoftmaxModel) -> None:
    payload = {
        "type": "softmax_linear",
        "feature_names": model.feature_names,
        "class_names": model.class_names,
        "mean": model.mean.tolist(),
        "std": model.std.tolist(),
        "weights": model.weights.tolist(),
        "bias": model.bias.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_softmax_model(path: Path) -> TrainedSoftmaxModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "softmax_linear":
        raise ValueError("unsupported model type")

    return TrainedSoftmaxModel(
        feature_names=list(payload["feature_names"]),
        class_names=list(payload["class_names"]),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        weights=np.asarray(payload["weights"], dtype=np.float32),
        bias=np.asarray(payload["bias"], dtype=np.float32),
    )
