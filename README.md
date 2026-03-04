# esp32-csi-human-sensing

Fresh implementation for ESP32 CSI-based human presence and activity sensing.

## Scope (Phase 1)
- Single-node ESP32 CSI pipeline
- Presence + activity (`empty`, `stationary`, `walking`, `gesture`, `fall_like`)
- REST + WebSocket live serving
- Replay/regression-first quality gates

## Repository Layout
- `firmware/esp32_node/`: ESP-IDF firmware for CSI capture and `CSI_FRAME_V2` UDP streaming
- `host/ingest/`: UDP ingest, parser, recorder, telemetry service
- `host/inference/`: feature extraction + model pipeline + calibration
- `host/dashboard/`: UI integration contract and static assets
- `data/contracts/`: binary and artifact contracts
- `tests/replay/`: packet parser and replay stability tests
- `docs/`: runbooks and protocols

## Quick Start (Host)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m host.ingest.service --bind 0.0.0.0 --port 8000 --udp-port 3334
```

## ESP-IDF Flow
```bash
source /Users/jarvis/Work/esp-idf-v5.5.2/export.sh
idf.py -p /dev/cu.usbserial-0001 flash monitor
```

## Quality Gates
See `docs/QUALITY_GATES.md`.
