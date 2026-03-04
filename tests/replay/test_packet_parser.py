import struct

import pytest

from host.ingest.packet import (
    HEADER_FMT,
    MAGIC,
    PKT_TYPE_CSI,
    VERSION,
    FrameParseError,
    parse_csi_frame_v2,
)


def build_packet(csi_iq_vals, seq: int = 5, node_id: int = 1, rssi: int = -45, channel: int = 6):
    csi_iq = struct.pack(f"!{len(csi_iq_vals)}b", *csi_iq_vals)
    payload = struct.pack("!hbbH", node_id, rssi, channel, len(csi_iq)) + csi_iq
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, PKT_TYPE_CSI, seq, 123456789, len(payload))
    return header + payload


def test_parse_valid_packet():
    pkt = build_packet([1, 2, -3, 4])
    frame = parse_csi_frame_v2(pkt)
    assert frame.seq == 5
    assert frame.node_id == 1
    assert frame.channel == 6
    assert len(frame.csi_iq) == 4


@pytest.mark.parametrize(
    "mutator",
    [
        lambda pkt: b"\x00" + pkt[1:],
        lambda pkt: pkt[:2] + bytes([3]) + pkt[3:],
        lambda pkt: pkt[:-1],
    ],
)
def test_parse_invalid_packet(mutator):
    pkt = build_packet([1, 2, 3, 4])
    with pytest.raises(FrameParseError):
        parse_csi_frame_v2(mutator(pkt))
