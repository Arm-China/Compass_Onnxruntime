import onnx
import numpy as np
from expect import *


def test_tanh():
    node = onnx.helper.make_node(
        "Tanh",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.tanh(x)
    expect_qdq(node, inputs=[x], outputs=[y], name="test_tanh", tol=1e-01)


if __name__ == "__main__":
    test_tanh()
