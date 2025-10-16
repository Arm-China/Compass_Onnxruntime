import onnx
import numpy as np
from expect import *


def test_prelu_example():
    node = onnx.helper.make_node(
        "PRelu",
        inputs=["x", "slope"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
    expect(
        node,
        inputs=[x, slope],
        outputs=[y],
        name="test_prelu_example",
        initializer=["slope"],
    )


def test_prelu_broadcast():
    node = onnx.helper.make_node(
        "PRelu",
        inputs=["x", "slope"],
        outputs=["y"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

    expect(
        node,
        inputs=[x, slope],
        outputs=[y],
        name="test_prelu_broadcast",
        initializer=["slope"],
    )


if __name__ == "__main__":
    test_prelu_example()
    test_prelu_broadcast()
