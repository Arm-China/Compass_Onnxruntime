import onnx
import numpy as np
from expect import *
import math


def test_erf():
    node = onnx.helper.make_node(
        "Erf",
        inputs=["x"],
        outputs=["y"],
    )

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = np.vectorize(math.erf)(x).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name="test_erf", tol=1e-01)


if __name__ == "__main__":
    test_erf()
