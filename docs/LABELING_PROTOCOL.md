# Local Labeling Protocol

## Classes
- `empty`
- `stationary`
- `walking`
- `gesture`
- `fall_like`

## Collection Recommendation
- 10-15 min per class
- 2-3 sessions/day
- 3-5 days

## Notes
- Keep AP placement and environment metadata in `labels.json`.
- Re-run calibration whenever AP channel/layout changes.
- During capture, switch labels from dashboard controls (or `POST /api/labels/current`).
- Each run persists as `data/sessions/session_<epoch>/` with:
  - `window_features.npz` (features + `timestamp_us` + `label`)
  - `labels.json` (label event timeline + counts)
