import onnx
import numpy as np
from expect import *


def test_greater():
    node = onnx.helper.make_node(
        "Greater",
        inputs=["x", "y"],
        outputs=["greater"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = np.greater(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_greater")


if __name__ == "__main__":
    test_greater()
