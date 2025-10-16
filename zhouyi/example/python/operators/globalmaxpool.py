import onnx
import numpy as np
from expect import *


def test_globalmaxpool():
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    y = np.max(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool")


def test_globalmaxpool_precomputed():
    node = onnx.helper.make_node(
        "GlobalMaxPool",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[9]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name="test_globalmaxpool_precomputed")


if __name__ == "__main__":
    test_globalmaxpool()
    test_globalmaxpool_precomputed()
