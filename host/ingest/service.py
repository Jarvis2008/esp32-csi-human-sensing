"""UDP ingest + REST/WS service for CSI phase 1."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from host.ingest.packet import FrameParseError, parse_csi_frame_v2
from host.ingest.state import RuntimeMetrics
from host.inference.pipeline import InferencePipeline


class CSIIngestService:
    def __init__(self, udp_port: int, session_dir: Path) -> None:
        self.udp_port = udp_port
        self.session_dir = session_dir
        self.metrics = RuntimeMetrics()
        self.pipeline = InferencePipeline(window_size=40)
        self._running = False
        self._clients: set[WebSocket] = set()
        self._state_lock = threading.Lock()
        self._last_state = {"presence": "unknown", "activity": "unknown", "confidence": 0.0}
        self._frame_buffer: list[tuple[int, ...]] = []

    def run_udp_listener(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.settimeout(1.0)
        self._running = True

        while self._running:
            try:
                packet, _addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            self.metrics.packets_received += 1
            self.metrics.last_packet_time = time.time()
            try:
                frame = parse_csi_frame_v2(packet)
                self.metrics.packets_parsed += 1
                if self.metrics.last_seq >= 0 and frame.seq > self.metrics.last_seq + 1:
                    self.metrics.dropped_estimate += frame.seq - self.metrics.last_seq - 1
                self.metrics.last_seq = frame.seq
                self._frame_buffer.append(frame.csi_iq)
                state = self.pipeline.add_frame(frame.csi_iq)
                if state:
                    with self._state_lock:
                        self._last_state = state
            except FrameParseError:
                self.metrics.packets_invalid += 1

        sock.close()

    async def broadcast(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            with self._state_lock:
                payload = {
                    "ts": time.time(),
                    "state": self._last_state,
                    "metrics": {
                        "packets_received": self.metrics.packets_received,
                        "packets_parsed": self.metrics.packets_parsed,
                        "packets_invalid": self.metrics.packets_invalid,
                        "dropped_estimate": self.metrics.dropped_estimate,
                    },
                    "features": self.pipeline.last_features,
                }
            stale: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_text(json.dumps(payload))
                except Exception:
                    stale.append(client)
            for client in stale:
                self._clients.discard(client)

    def stop(self) -> None:
        self._running = False
        self.persist_session()

    def persist_session(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        session_id = int(time.time())
        raw_path = self.session_dir / f"raw_frames_{session_id}.npz"
        if self._frame_buffer:
            np.savez(raw_path, frames=np.asarray(self._frame_buffer, dtype=object))


service: CSIIngestService | None = None
app = FastAPI(title="ESP32 CSI Human Sensing", version="0.1.0")


@app.get("/api/status")
def get_status() -> dict[str, object]:
    assert service is not None
    return {
        "uptime_sec": service.metrics.uptime_sec(),
        "packets_received": service.metrics.packets_received,
        "packets_parsed": service.metrics.packets_parsed,
        "packets_invalid": service.metrics.packets_invalid,
        "dropped_estimate": service.metrics.dropped_estimate,
        "last_packet_age_sec": max(0.0, time.time() - service.metrics.last_packet_time) if service.metrics.last_packet_time else None,
    }


@app.get("/api/state")
def get_state() -> dict[str, object]:
    assert service is not None
    return service.pipeline.last_state


@app.get("/api/metrics")
def get_metrics() -> dict[str, object]:
    assert service is not None
    return {
        "features": service.pipeline.last_features,
        "packets_parsed": service.metrics.packets_parsed,
        "packets_invalid": service.metrics.packets_invalid,
    }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    assert service is not None
    await ws.accept()
    service._clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        service._clients.discard(ws)


@app.on_event("startup")
async def on_startup() -> None:
    assert service is not None
    asyncio.create_task(service.broadcast())


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI ingest service")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--udp-port", type=int, default=3334)
    parser.add_argument("--session-dir", default="data/sessions")
    args = parser.parse_args()

    global service
    service = CSIIngestService(args.udp_port, Path(args.session_dir))
    th = threading.Thread(target=service.run_udp_listener, daemon=True)
    th.start()

    try:
        uvicorn.run(app, host=args.bind, port=args.port)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
