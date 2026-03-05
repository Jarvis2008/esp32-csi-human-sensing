# Dashboard

Live browser dashboard for phase-1 CSI sensing.

## Entry Points
- `/` -> dashboard index
- `/dashboard/*` -> static assets
- `/api/status`, `/api/state`, `/api/metrics`, `/api/nodes`, `/api/labels`
- `POST /api/labels/current` -> set active capture label
- `/ws/live`

## Run
```bash
uv run python -m host.ingest.service --bind 0.0.0.0 --port 8000 --udp-port 3334
```

Open: `http://localhost:8000`

## UI Sections
- Current inferred state
- Packet/parser health
- Node-level counters
- Motion/confidence trend charts
- Labeling control for phase-2 dataset capture
