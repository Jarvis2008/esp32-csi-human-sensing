"""Runtime shared state for ingest/service."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class RuntimeMetrics:
    packets_received: int = 0
    packets_parsed: int = 0
    packets_invalid: int = 0
    last_seq: int = -1
    dropped_estimate: int = 0
    started_at: float = field(default_factory=time.time)
    last_packet_time: float = 0.0

    def uptime_sec(self) -> float:
        return time.time() - self.started_at
