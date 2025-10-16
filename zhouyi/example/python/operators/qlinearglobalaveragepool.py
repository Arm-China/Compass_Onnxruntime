import onnx
import numpy as np
from expect import *


def test_qlinearglobalaverage():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QLinearGlobalAveragePool",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        domain=domain,
    )

    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ],
            ],
        ],
        dtype=np.uint8,
    ).reshape(1, 1, 5, 5)

    y = np.array(
        [22],
        dtype=np.uint8,
    ).reshape(1, 1, 1, 1)
    x_scale = np.float32(0.15942417085170746)
    x_zero_point = np.uint8(10)
    y_scale = np.float32(0.038893215358257294)
    y_zero_point = np.uint8(10)
    expect(
        node,
        inputs=[
            x,
            x_scale,
            x_zero_point,
            y_scale,
            y_zero_point,
        ],
        outputs=[y],
        name="test_qlinearglobalaverage",
        initializer=[
            "x_scale",
            "x_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        opset_imports=[
            onnx.helper.make_opsetid("", 19),
            onnx.helper.make_opsetid(domain, 1),
        ],
    )


if __name__ == "__main__":
    test_qlinearglobalaverage()
