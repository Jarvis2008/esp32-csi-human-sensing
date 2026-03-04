"""Session artifact writer for CSI ingest runs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np


@dataclass
class FrameSnapshot:
    seq: int
    timestamp_us: int
    node_id: int
    rssi: int
    channel: int
    csi_iq: tuple[int, ...]


class SessionRecorder:
    def __init__(self, max_frames: int = 12000, max_feature_windows: int = 3000) -> None:
        self._frames: deque[FrameSnapshot] = deque(maxlen=max_frames)
        self._feature_windows: deque[dict[str, float]] = deque(maxlen=max_feature_windows)

    def add_frame(self, frame: FrameSnapshot) -> None:
        self._frames.append(frame)

    def add_feature_window(self, features: dict[str, float]) -> None:
        self._feature_windows.append(dict(features))

    def frame_count(self) -> int:
        return len(self._frames)

    def persist(
        self,
        session_dir: Path,
        metrics: dict[str, object],
        last_state: dict[str, object],
    ) -> dict[str, str]:
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = int(time.time())

        raw_path = session_dir / f"raw_frames_{session_id}.npz"
        features_path = session_dir / f"window_features_{session_id}.npz"
        labels_path = session_dir / "labels.json"
        eval_report_path = session_dir / "eval_report.md"

        if self._frames:
            seq = np.asarray([f.seq for f in self._frames], dtype=np.uint32)
            ts = np.asarray([f.timestamp_us for f in self._frames], dtype=np.uint64)
            node = np.asarray([f.node_id for f in self._frames], dtype=np.int16)
            rssi = np.asarray([f.rssi for f in self._frames], dtype=np.int8)
            channel = np.asarray([f.channel for f in self._frames], dtype=np.uint8)
            csi_iq = np.asarray([np.asarray(f.csi_iq, dtype=np.int8) for f in self._frames], dtype=object)
            np.savez(raw_path, seq=seq, timestamp_us=ts, node_id=node, rssi=rssi, channel=channel, csi_iq=csi_iq)

        if self._feature_windows:
            names = sorted(self._feature_windows[0].keys())
            matrix = np.asarray([[window.get(name, 0.0) for name in names] for window in self._feature_windows], dtype=np.float32)
            np.savez(features_path, feature_names=np.asarray(names, dtype=object), values=matrix)

        labels_payload = {
            "schema": "phase1-labels-v1",
            "classes": ["empty", "stationary", "walking", "gesture", "fall_like"],
            "entries": [],
            "notes": "Fill entries with timestamp ranges and class labels during manual annotation.",
        }
        labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")

        report_lines = [
            "# Eval Report",
            "",
            f"- Session ID: {session_id}",
            f"- Recorded frames: {len(self._frames)}",
            f"- Feature windows: {len(self._feature_windows)}",
            f"- Last state: {last_state}",
            f"- Metrics snapshot: {metrics}",
            "",
            "This report is a runtime snapshot. Model quality metrics are added by the inference evaluation pipeline.",
        ]
        eval_report_path.write_text("\n".join(report_lines), encoding="utf-8")

        return {
            "raw_frames": str(raw_path),
            "window_features": str(features_path),
            "labels": str(labels_path),
            "eval_report": str(eval_report_path),
        }

    def node_distribution(self) -> dict[int, int]:
        distribution: dict[int, int] = {}
        for frame in self._frames:
            distribution[frame.node_id] = distribution.get(frame.node_id, 0) + 1
        return distribution

    def peek_last(self) -> FrameSnapshot | None:
        return self._frames[-1] if self._frames else None

    def snapshot_frames(self) -> list[dict[str, object]]:
        return [asdict(f) for f in self._frames]
