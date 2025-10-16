import onnx
import numpy as np
from expect import *


def test_relu():
    node = onnx.helper.make_node(
        "Relu",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf)

    expect(node, inputs=[x], outputs=[y], name="test_relu")


if __name__ == "__main__":
    test_relu()
