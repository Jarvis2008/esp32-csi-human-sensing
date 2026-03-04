# Inference V1

Phase-1 inference supports two runtime model modes:

1. Threshold model (default):
- Uses calibrated `motion_index` thresholds.

2. Softmax model:
- Trained with `host/inference/train_activity_model.py` from feature CSV.
- Loaded by passing `model_path` to `InferencePipeline`.

## Training Example
```bash
uv run python -m host.inference.train_activity_model \
  --dataset-csv data/sessions/features_labeled.csv \
  --label-col label \
  --output data/sessions/model_softmax.json
```

## Threshold Calibration
Use `host/inference/calibration.py` helpers to build `thresholds.json` from labeled motion windows.
