import onnx
import numpy as np
from expect import *


def hardswish(x: np.ndarray) -> np.ndarray:
    alfa = float(1 / 6)
    beta = 0.5
    return x * np.maximum(0, np.minimum(1, alfa * x + beta))


def test_hardswish():
    node = onnx.helper.make_node(
        "HardSwish",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = hardswish(x)

    expect(node, inputs=[x], outputs=[y], name="test_hardswish")


if __name__ == "__main__":
    test_hardswish()
