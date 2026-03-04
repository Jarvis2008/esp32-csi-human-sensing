import numpy as np

from host.inference.training import predict_with_softmax, train_softmax_classifier


def test_softmax_training_learns_separable_data():
    rng = np.random.default_rng(42)

    c0 = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(50, 2))
    c1 = rng.normal(loc=[2.0, 2.0], scale=0.2, size=(50, 2))
    c2 = rng.normal(loc=[-2.0, 2.0], scale=0.2, size=(50, 2))

    x = np.vstack([c0, c1, c2]).astype(np.float32)
    y = np.asarray([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int64)

    model = train_softmax_classifier(
        x=x,
        y_idx=y,
        feature_names=["f1", "f2"],
        class_names=["empty", "walking", "gesture"],
        epochs=500,
        learning_rate=0.2,
        l2=1e-4,
    )

    probs = predict_with_softmax(model, x)
    pred = np.argmax(probs, axis=1)
    accuracy = np.mean(pred == y)

    assert accuracy > 0.95
