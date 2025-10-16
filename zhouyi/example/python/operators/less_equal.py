import onnx
import numpy as np
from expect import *


def test_less_equal():
    node = onnx.helper.make_node(
        "LessOrEqual",
        inputs=["x", "y"],
        outputs=["less_equal"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = np.less_equal(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_less_equal")


if __name__ == "__main__":
    test_less_equal()
