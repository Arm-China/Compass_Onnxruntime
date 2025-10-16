import onnx
import numpy as np
from expect import *


def test_less():
    node = onnx.helper.make_node(
        "Less",
        inputs=["x", "y"],
        outputs=["less"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = np.less(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_less")


if __name__ == "__main__":
    test_less()
