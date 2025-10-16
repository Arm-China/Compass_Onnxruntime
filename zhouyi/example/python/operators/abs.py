import onnx
import numpy as np
from expect import *
from onnx.backend.sample.ops.abs import abs


def test_abs():
    node = onnx.helper.make_node(
        "Abs",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = abs(x)

    expect(node, inputs=[x], outputs=[y], name="test_abs")


if __name__ == "__main__":
    test_abs()
