import onnx
import numpy as np
from expect import *


def test_globalaveragepool():
    node = onnx.helper.make_node(
        "GlobalAveragePool",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    expect(node, inputs=[x], outputs=[y], name="test_globalaveragepool")


def test_globalaveragepool_precomputed():
    node = onnx.helper.make_node(
        "GlobalAveragePool",
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
    y = np.array([[[[5]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name="test_globalaveragepool_precomputed")


if __name__ == "__main__":
    test_globalaveragepool()
    test_globalaveragepool_precomputed()
