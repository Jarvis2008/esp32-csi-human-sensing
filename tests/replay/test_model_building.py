import json
from pathlib import Path

from host.inference.model import SoftmaxActivityModel, ThresholdActivityModel, build_model
from host.inference.training import TrainedSoftmaxModel, save_softmax_model
import numpy as np


def test_build_model_defaults_to_threshold(tmp_path: Path):
    model = build_model(model_path=tmp_path / "missing.json")
    assert isinstance(model, ThresholdActivityModel)


def test_build_model_loads_softmax(tmp_path: Path):
    model_path = tmp_path / "softmax.json"
    trained = TrainedSoftmaxModel(
        feature_names=["motion_index"],
        class_names=["empty", "walking"],
        mean=np.asarray([0.0], dtype=np.float32),
        std=np.asarray([1.0], dtype=np.float32),
        weights=np.asarray([[-1.0, 1.0]], dtype=np.float32),
        bias=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    save_softmax_model(model_path, trained)

    loaded = build_model(model_path=model_path)
    assert isinstance(loaded, SoftmaxActivityModel)

    state = loaded.predict({"motion_index": 2.0})
    assert state.activity == "walking"
    assert state.presence == "occupied"


def test_build_model_loads_thresholds(tmp_path: Path):
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "presence_threshold": 5.0,
                "walk_threshold": 10.0,
                "gesture_threshold": 20.0,
            }
        ),
        encoding="utf-8",
    )
    model = build_model(model_path=None, threshold_path=threshold_path)
    assert isinstance(model, ThresholdActivityModel)
    state = model.predict({"motion_index": 6.0})
    assert state.activity == "stationary"
