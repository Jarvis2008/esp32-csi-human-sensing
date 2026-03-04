from pathlib import Path

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
    rec.add_feature_window({"amp_mean": 1.2, "motion_index": 0.4})

    out = rec.persist(
        session_dir=tmp_path,
        metrics={"packets_parsed": 1},
        last_state={"presence": "occupied", "activity": "walking", "confidence": 0.7},
    )

    assert Path(out["raw_frames"]).exists()
    assert Path(out["window_features"]).exists()
    assert Path(out["labels"]).exists()
    assert Path(out["eval_report"]).exists()


def test_node_distribution_counts():
    rec = SessionRecorder(max_frames=10, max_feature_windows=10)
    rec.add_frame(FrameSnapshot(seq=1, timestamp_us=1, node_id=1, rssi=-40, channel=1, csi_iq=(1, 2)))
    rec.add_frame(FrameSnapshot(seq=2, timestamp_us=2, node_id=2, rssi=-41, channel=1, csi_iq=(1, 2)))
    rec.add_frame(FrameSnapshot(seq=3, timestamp_us=3, node_id=1, rssi=-42, channel=1, csi_iq=(1, 2)))

    assert rec.node_distribution() == {1: 2, 2: 1}
