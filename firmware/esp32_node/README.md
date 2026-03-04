# ESP32 Node Firmware (Phase 1)

Fresh firmware for CSI capture and `CSI_FRAME_V2` UDP streaming.

## Features
- WiFi STA connect (credentials from Kconfig or NVS)
- CSI callback capture + ring-buffer decoupling
- UDP stream of `CSI_FRAME_V2`
- Runtime config through NVS (`csi_cfg` namespace)

## NVS Keys (`csi_cfg`)
- `ssid` (string)
- `password` (string)
- `target_ip` (string, e.g. `192.168.1.20` or `255.255.255.255`)
- `target_port` (u16)
- `node_id` (u16)
- `capture_ms` (u16)
- `traffic_ms` (u16)

## Build + Flash
```bash
source /Users/jarvis/Work/esp-idf-v5.5.2/export.sh
cd firmware/esp32_node
idf.py set-target esp32
idf.py -p /dev/cu.usbserial-0001 build flash monitor
```

## Stable Flash (if serial drops during flash)
Use the bundled helper to flash at a conservative baud:
```bash
cd firmware/esp32_node
./flash_safe.sh /dev/cu.usbserial-0001
```

## Packet Contract
See `data/contracts/csi_frame_v2.md` in the repo root.
