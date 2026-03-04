# ESP32 Node Firmware (Phase 1)

This folder contains the fresh firmware implementation plan for `CSI_FRAME_V2`.

## Targets
- Capture CSI callback frames
- Push frames to ring buffer
- Stream `CSI_FRAME_V2` over UDP
- Runtime tuning via NVS

## Expected build flow
```bash
source /Users/jarvis/Work/esp-idf-v5.5.2/export.sh
idf.py -p /dev/cu.usbserial-0001 build flash monitor
```
