from pathlib import Path

import numpy as np

from host.ingest.session import FrameSnapshot, SessionRecorder


def test_session_recorder_persists_artifacts(tmp_path: Path):
    rec = SessionRecorder(max_frames=10, max_feature_windows=10)
    rec.add_frame(
        FrameSnapshot(
            seq=1,
            timestamp_us=123,
            node_id=7,
            rssi=-40,
            channel=6,
            csi_iq=(1, 2, 3, 4),
        )
    )
    rec.add_feature_window(
        {"amp_mean": 1.2, "motion_index": 0.4},
        timestamp_us=123,
        label="walking",
    )

    out = rec.persist(
        session_dir=tmp_path,
        metrics={"packets_parsed": 1},
        last_state={"presence": "occupied", "activity": "walking", "confidence": 0.7},
        label_events=[{"label": "walking", "event_time_s": 1.0}],
    )

    assert Path(out["session_dir"]).exists()
    assert Path(out["raw_frames"]).exists()
    assert Path(out["window_features"]).exists()
    assert Path(out["labels"]).exists()
    assert Path(out["eval_report"]).exists()

    features = np.load(out["window_features"], allow_pickle=True)
    assert features["label"].tolist() == ["walking"]
    assert features["timestamp_us"].tolist() == [123]


def test_node_distribution_counts():
    rec = SessionRecorder(max_frames=10, max_feature_windows=10)
    rec.add_frame(FrameSnapshot(seq=1, timestamp_us=1, node_id=1, rssi=-40, channel=1, csi_iq=(1, 2)))
    rec.add_frame(FrameSnapshot(seq=2, timestamp_us=2, node_id=2, rssi=-41, channel=1, csi_iq=(1, 2)))
    rec.add_frame(FrameSnapshot(seq=3, timestamp_us=3, node_id=1, rssi=-42, channel=1, csi_iq=(1, 2)))

    assert rec.node_distribution() == {1: 2, 2: 1}
