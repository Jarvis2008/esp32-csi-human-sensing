# Inference V1

Phase-1 inference supports two runtime model modes:

1. Threshold model (default):
- Uses calibrated `motion_index` thresholds.

2. Softmax model:
- Trained with `host/inference/train_activity_model.py` from feature CSV.
- Loaded by passing `model_path` to `InferencePipeline`.

## Phase-2 Data/Model Loop
1. Run ingest service and capture labeled windows using dashboard label buttons (`/api/labels/current`).
2. Stop service to persist `data/sessions/session_*/window_features.npz`.
3. Export dataset CSV:
```bash
uv run python -m host.inference.export_labeled_windows \
  --session-root data/sessions \
  --output-csv data/sessions/features_labeled.csv
```
4. Train + evaluate with held-out split:
```bash
uv run python -m host.inference.evaluate_activity_model \
  --dataset-csv data/sessions/features_labeled.csv \
  --output-model data/sessions/model_softmax.json \
  --output-report data/sessions/eval_report_phase2.md \
  --threshold-output data/sessions/thresholds.json
```

## Training Example
```bash
uv run python -m host.inference.train_activity_model \
  --dataset-csv data/sessions/features_labeled.csv \
  --label-col label \
  --output data/sessions/model_softmax.json
```

## Threshold Calibration
Use `host/inference/calibration.py` helpers to build `thresholds.json` from labeled motion windows.
