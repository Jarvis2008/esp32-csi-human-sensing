#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-/dev/cu.usbserial-0001}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /Users/jarvis/Work/esp-idf-v5.5.2/export.sh >/tmp/esp_idf_export_flash_safe.log 2>&1
cd "$ROOT_DIR"

idf.py -p "$PORT" -b 115200 flash monitor
