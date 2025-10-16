import onnx
import numpy as np
from expect import *


def test_qlinearleakyrelu():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QLinearLeakyRelu",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        alpha=0.2,
        domain=domain,
    )
    x = np.array(
        [
            [1, -2],
            [3, -4],
        ],
        dtype=np.int8,
    )

    y = np.array(
        [
            [1, 0],
            [3, -1],
        ],
        dtype=np.int8,
    )
    x_scale = np.float32(1.0)
    x_zero_point = np.int8(0)
    y_scale = np.float32(1.0)
    y_zero_point = np.int8(0)
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
        name="test_qlinearleakyrelu",
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
    test_qlinearleakyrelu()
