import onnx
import numpy as np
from expect import *


def test_equal():
    node = onnx.helper.make_node(
        "Equal",
        inputs=["x", "y"],
        outputs=["z"],
    )

    x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
    z = np.equal(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_equal")


if __name__ == "__main__":
    test_equal()
