import onnx
import numpy as np
from expect import *


def test_squeeze():
    node = onnx.helper.make_node(
        "Squeeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    y = np.squeeze(x, axis=0)

    expect(
        node, inputs=[x, axes], outputs=[y], name="test_squeeze", initializer=["axes"]
    )


def test_squeeze_negative_axes():
    node = onnx.helper.make_node(
        "Squeeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2], dtype=np.int64)
    y = np.squeeze(x, axis=-2)
    expect(
        node,
        inputs=[x, axes],
        outputs=[y],
        name="test_squeeze_negative_axes",
        initializer=["axes"],
    )


if __name__ == "__main__":
    test_squeeze()
    test_squeeze_negative_axes()
