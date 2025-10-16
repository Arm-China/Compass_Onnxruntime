import onnx
import numpy as np
from expect import *


def test_greater_equal():
    node = onnx.helper.make_node(
        "GreaterOrEqual",
        inputs=["x", "y"],
        outputs=["greater_equal"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = np.greater_equal(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_greater_equal")


if __name__ == "__main__":
    test_greater_equal()
