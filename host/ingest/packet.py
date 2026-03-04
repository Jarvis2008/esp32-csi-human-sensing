"""CSI_FRAME_V2 parser and validator."""

from __future__ import annotations

from dataclasses import dataclass
import struct

MAGIC = 0xC511
VERSION = 2
PKT_TYPE_CSI = 0x04
HEADER_FMT = "!HBBIQH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PAYLOAD_PREFIX_FMT = "!hbbH"
PAYLOAD_PREFIX_SIZE = struct.calcsize(PAYLOAD_PREFIX_FMT)
MAX_CSI_LEN = 512


@dataclass(frozen=True)
class CSIFrameV2:
    seq: int
    timestamp_us: int
    node_id: int
    rssi: int
    channel: int
    csi_iq: tuple[int, ...]


class FrameParseError(ValueError):
    pass


def parse_csi_frame_v2(packet: bytes) -> CSIFrameV2:
    if len(packet) < HEADER_SIZE:
        raise FrameParseError("packet shorter than header")

    magic, version, pkt_type, seq, timestamp_us, payload_len = struct.unpack(
        HEADER_FMT, packet[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise FrameParseError("invalid magic")
    if version != VERSION:
        raise FrameParseError("unsupported version")
    if pkt_type != PKT_TYPE_CSI:
        raise FrameParseError("unsupported packet type")

    payload = packet[HEADER_SIZE:]
    if len(payload) != payload_len:
        raise FrameParseError("payload length mismatch")
    if len(payload) < PAYLOAD_PREFIX_SIZE:
        raise FrameParseError("payload too short")

    node_id, rssi, channel, csi_len = struct.unpack(
        PAYLOAD_PREFIX_FMT, payload[:PAYLOAD_PREFIX_SIZE]
    )
    csi_bytes = payload[PAYLOAD_PREFIX_SIZE:]

    if csi_len != len(csi_bytes):
        raise FrameParseError("csi_len mismatch")
    if csi_len > MAX_CSI_LEN:
        raise FrameParseError("csi_len too large")
    if csi_len % 2 != 0:
        raise FrameParseError("csi_len must be even")

    csi_iq = struct.unpack(f"!{csi_len}b", csi_bytes) if csi_len else ()
    return CSIFrameV2(
        seq=seq,
        timestamp_us=timestamp_us,
        node_id=node_id,
        rssi=rssi,
        channel=channel,
        csi_iq=csi_iq,
    )
