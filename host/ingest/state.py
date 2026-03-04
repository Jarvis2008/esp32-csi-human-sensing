"""Runtime shared state for ingest/service."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class NodeStats:
    packets: int = 0
    last_seq: int = -1
    dropped_estimate: int = 0
    last_seen_time: float = 0.0


@dataclass
class RuntimeMetrics:
    packets_received: int = 0
    packets_parsed: int = 0
    packets_invalid: int = 0
    last_seq: int = -1
    dropped_estimate: int = 0
    started_at: float = field(default_factory=time.time)
    last_packet_time: float = 0.0
    node_stats: dict[int, NodeStats] = field(default_factory=dict)

    def uptime_sec(self) -> float:
        return time.time() - self.started_at

    def register_frame(self, seq: int, node_id: int) -> None:
        self.packets_parsed += 1
        if self.last_seq >= 0 and seq > self.last_seq + 1:
            self.dropped_estimate += seq - self.last_seq - 1
        self.last_seq = seq

        node = self.node_stats.setdefault(node_id, NodeStats())
        node.packets += 1
        node.last_seen_time = time.time()
        if node.last_seq >= 0 and seq > node.last_seq + 1:
            node.dropped_estimate += seq - node.last_seq - 1
        node.last_seq = seq

    def register_invalid(self) -> None:
        self.packets_invalid += 1

    def parser_error_rate(self) -> float:
        total = self.packets_received if self.packets_received else 1
        return self.packets_invalid / total
