import onnx
import numpy as np
from expect import *


def test_sub_example():
    node = onnx.helper.make_node(
        "Sub",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([3, 2, 1]).astype(np.float32)
    z = x - y  # expected output [-2., 0., 2.]
    expect(node, inputs=[x, y], outputs=[z], name="test_sub_example")

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = x - y
    expect(node, inputs=[x, y], outputs=[z], name="test_sub")

    x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
    z = x - y
    expect(node, inputs=[x, y], outputs=[z], name="test_sub_uint8")


def test_sub_broadcast():
    node = onnx.helper.make_node(
        "Sub",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x - y
    expect(node, inputs=[x, y], outputs=[z], name="test_sub_bcast")


if __name__ == "__main__":
    test_sub_example()
    test_sub_broadcast()
