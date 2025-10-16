import onnx
import numpy as np
from expect import *


def test_not():
    node = onnx.helper.make_node(
        "Not",
        inputs=["x"],
        outputs=["not"],
    )

    # 2d
    x = (np.random.randn(3, 4) > 0).astype(bool)
    expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_2d")

    # 3d
    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_3d")

    # 4d
    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    expect(node, inputs=[x], outputs=[np.logical_not(x)], name="test_not_4d")


if __name__ == "__main__":
    test_not()
