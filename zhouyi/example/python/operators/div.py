import onnx
import numpy as np
from expect import *


def test_div_example():
    node = onnx.helper.make_node(
        "Div",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.array([3, 4]).astype(np.float32)
    y = np.array([1, 2]).astype(np.float32)
    z = x / y  # expected output [3., 2.]
    expect(node, inputs=[x, y], outputs=[z], name="test_div_example")

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
    z = x / y
    expect(node, inputs=[x, y], outputs=[z], name="test_div")

    x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8) + 1
    z = x // y
    expect(node, inputs=[x, y], outputs=[z], name="test_div_uint8", tol=1.0)


def test_div_broadcast():
    node = onnx.helper.make_node(
        "Div",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32) + 1.0
    z = x / y
    expect(node, inputs=[x, y], outputs=[z], name="test_div_bcast")


if __name__ == "__main__":
    test_div_example()
    test_div_broadcast()
