import onnx
import numpy as np
from expect import *


def test_matmul():
    node = onnx.helper.make_node(
        "MatMul",
        inputs=["a", "b"],
        outputs=["c"],
    )

    # 2d
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_2d")

    # 3d
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_3d")

    # 4d
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    b = np.random.randn(1, 2, 4, 3).astype(np.float32)
    c = np.matmul(a, b)
    expect(node, inputs=[a, b], outputs=[c], name="test_matmul_4d")


if __name__ == "__main__":
    test_matmul()
