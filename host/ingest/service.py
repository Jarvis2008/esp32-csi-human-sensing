"""UDP ingest + REST/WS service for CSI phase 1/2."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from host.ingest.packet import FrameParseError, parse_csi_frame_v2
from host.ingest.session import FrameSnapshot, SessionRecorder
from host.ingest.state import RuntimeMetrics
from host.inference.pipeline import InferencePipeline


LABELS = ("unlabeled", "empty", "stationary", "walking", "gesture", "fall_like")


class LabelUpdateRequest(BaseModel):
    label: str
    note: Optional[str] = None


class CSIIngestService:
    def __init__(
        self,
        udp_port: int,
        session_dir: Path,
        window_size: int,
        max_frames: int,
    ) -> None:
        self.udp_port = udp_port
        self.session_dir = session_dir
        self.metrics = RuntimeMetrics()
        self.pipeline = InferencePipeline(window_size=window_size)
        self.recorder = SessionRecorder(max_frames=max_frames)
        self._running = False
        self._clients: set[WebSocket] = set()
        self._state_lock = threading.Lock()
        self._last_state = {"presence": "unknown", "activity": "unknown", "confidence": 0.0}
        self._active_label = "unlabeled"
        self._label_events: list[dict[str, object]] = [
            {
                "label": "unlabeled",
                "event_time_s": time.time(),
                "note": "startup",
            }
        ]
        self._udp_thread: threading.Thread | None = None

    def set_label(self, label: str, note: str | None = None) -> dict[str, object]:
        if label not in LABELS:
            raise ValueError(f"unsupported label '{label}'")

        event = {
            "label": label,
            "event_time_s": time.time(),
        }
        cleaned_note = (note or "").strip()
        if cleaned_note:
            event["note"] = cleaned_note

        with self._state_lock:
            self._active_label = label
            self._label_events.append(event)
            return {
                "active_label": self._active_label,
                "label_events_count": len(self._label_events),
                "last_event": dict(event),
            }

    def get_label_state(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "active_label": self._active_label,
                "labels": list(LABELS),
                "label_events_count": len(self._label_events),
                "recent_events": self._label_events[-20:],
            }

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

            with self._state_lock:
                self.metrics.packets_received += 1
                self.metrics.last_packet_time = time.time()

            try:
                frame = parse_csi_frame_v2(packet)
            except FrameParseError:
                with self._state_lock:
                    self.metrics.register_invalid()
                continue

            with self._state_lock:
                self.metrics.register_frame(frame.seq, frame.node_id)
                self.recorder.add_frame(
                    FrameSnapshot(
                        seq=frame.seq,
                        timestamp_us=frame.timestamp_us,
                        node_id=frame.node_id,
                        rssi=frame.rssi,
                        channel=frame.channel,
                        csi_iq=frame.csi_iq,
                    )
                )

                state = self.pipeline.add_frame(frame.csi_iq)
                if self.pipeline.last_features:
                    self.recorder.add_feature_window(
                        self.pipeline.last_features,
                        timestamp_us=frame.timestamp_us,
                        label=self._active_label,
                    )
                if state:
                    self._last_state = state

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
                        "error_rate": self.metrics.parser_error_rate(),
                    },
                    "features": self.pipeline.last_features,
                    "labeling": {
                        "active_label": self._active_label,
                        "label_events_count": len(self._label_events),
                    },
                    "nodes": {
                        node_id: {
                            "packets": stats.packets,
                            "last_seq": stats.last_seq,
                            "dropped_estimate": stats.dropped_estimate,
                            "last_seen_time": stats.last_seen_time,
                        }
                        for node_id, stats in self.metrics.node_stats.items()
                    },
                }
            stale: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_text(json.dumps(payload))
                except Exception:
                    stale.append(client)
            for client in stale:
                self._clients.discard(client)

    def start(self) -> None:
        self._udp_thread = threading.Thread(target=self.run_udp_listener, daemon=True)
        self._udp_thread.start()

    def stop(self) -> dict[str, str]:
        self._running = False
        if self._udp_thread:
            self._udp_thread.join(timeout=2.0)

        with self._state_lock:
            metrics_snapshot = {
                "packets_received": self.metrics.packets_received,
                "packets_parsed": self.metrics.packets_parsed,
                "packets_invalid": self.metrics.packets_invalid,
                "dropped_estimate": self.metrics.dropped_estimate,
                "error_rate": self.metrics.parser_error_rate(),
            }
            last_state = dict(self._last_state)
            label_events = list(self._label_events)
            metadata = {
                "window_size": self.pipeline.window_size,
                "sample_rate_hz": self.pipeline.sample_rate_hz,
                "udp_port": self.udp_port,
            }

        return self.recorder.persist(
            self.session_dir,
            metrics_snapshot,
            last_state,
            label_events=label_events,
            metadata=metadata,
        )


service: CSIIngestService | None = None
app = FastAPI(title="ESP32 CSI Human Sensing", version="0.1.0")
DASHBOARD_STATIC_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "static"

if DASHBOARD_STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_STATIC_DIR), html=True), name="dashboard")


@app.get("/")
def dashboard_home() -> FileResponse:
    if DASHBOARD_STATIC_DIR.exists():
        return FileResponse(DASHBOARD_STATIC_DIR / "index.html")
    raise HTTPException(status_code=404, detail="dashboard static directory not found")


@app.get("/api/status")
def get_status() -> dict[str, object]:
    assert service is not None
    with service._state_lock:
        last_packet_age = max(0.0, time.time() - service.metrics.last_packet_time) if service.metrics.last_packet_time else None
        return {
            "uptime_sec": service.metrics.uptime_sec(),
            "packets_received": service.metrics.packets_received,
            "packets_parsed": service.metrics.packets_parsed,
            "packets_invalid": service.metrics.packets_invalid,
            "dropped_estimate": service.metrics.dropped_estimate,
            "parser_error_rate": service.metrics.parser_error_rate(),
            "last_packet_age_sec": last_packet_age,
            "frames_buffered": service.recorder.frame_count(),
        }


@app.get("/api/state")
def get_state() -> dict[str, object]:
    assert service is not None
    with service._state_lock:
        return dict(service._last_state)


@app.get("/api/metrics")
def get_metrics() -> dict[str, object]:
    assert service is not None
    with service._state_lock:
        return {
            "features": dict(service.pipeline.last_features),
            "packets_parsed": service.metrics.packets_parsed,
            "packets_invalid": service.metrics.packets_invalid,
            "node_distribution": service.recorder.node_distribution(),
            "active_label": service._active_label,
            "label_events_count": len(service._label_events),
        }


@app.get("/api/nodes")
def get_nodes() -> dict[int, dict[str, object]]:
    assert service is not None
    with service._state_lock:
        return {
            node_id: {
                "packets": stats.packets,
                "last_seq": stats.last_seq,
                "dropped_estimate": stats.dropped_estimate,
                "last_seen_time": stats.last_seen_time,
            }
            for node_id, stats in service.metrics.node_stats.items()
        }


@app.get("/api/labels")
def get_labels() -> dict[str, object]:
    assert service is not None
    return service.get_label_state()


@app.post("/api/labels/current")
def set_label(req: LabelUpdateRequest) -> dict[str, object]:
    assert service is not None
    try:
        return service.set_label(req.label, req.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=12000)
    args = parser.parse_args()

    global service
    service = CSIIngestService(
        udp_port=args.udp_port,
        session_dir=Path(args.session_dir),
        window_size=args.window_size,
        max_frames=args.max_frames,
    )
    service.start()

    artifacts: dict[str, str] = {}
    try:
        uvicorn.run(app, host=args.bind, port=args.port)
    finally:
        artifacts = service.stop()

    if artifacts:
        print("Saved session artifacts:")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
