import onnx
import numpy as np
from expect import *


def pow(x, y):  # type: ignore  # noqa: A001
    z = np.power(x, y).astype(x.dtype)
    return z


def test_pow_example():
    node = onnx.helper.make_node(
        "Pow",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = pow(x, y)  # expected output [1., 32., 729.]
    expect(node, inputs=[x, y], outputs=[z], name="test_pow_example")

    x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = pow(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_pow")


def test_pow_broadcast():
    node = onnx.helper.make_node(
        "Pow",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array(2).astype(np.float32)
    z = pow(x, y)  # expected output [1., 4., 9.]
    expect(node, inputs=[x, y], outputs=[z], name="test_pow_bcast_scalar")

    node = onnx.helper.make_node(
        "Pow",
        inputs=["x", "y"],
        outputs=["z"],
    )
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([1, 2, 3]).astype(np.float32)
    # expected output [[1, 4, 27], [4, 25, 216]]
    z = pow(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_pow_bcast_array")


if __name__ == "__main__":
    test_pow_example()
    test_pow_broadcast()
