from host.inference.pipeline import InferencePipeline


def test_replay_stability_constant_signal():
    pipeline = InferencePipeline(window_size=5)
    frame = tuple([4, 1] * 32)

    out = None
    for _ in range(10):
        out = pipeline.add_frame(frame)

    assert out is not None
    assert out["presence"] in {"empty", "occupied"}
    assert 0.0 <= out["confidence"] <= 1.0


def test_replay_deterministic_output():
    p1 = InferencePipeline(window_size=5)
    p2 = InferencePipeline(window_size=5)
    frames = [tuple([x % 8, (x + 2) % 8] * 32) for x in range(10)]

    out1 = []
    out2 = []
    for frame in frames:
        s1 = p1.add_frame(frame)
        s2 = p2.add_frame(frame)
        if s1 is not None:
            out1.append(s1)
        if s2 is not None:
            out2.append(s2)

    assert out1 == out2
