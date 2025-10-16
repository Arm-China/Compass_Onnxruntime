import onnx
import numpy as np
from expect import *


def test_qlinearsigmoid():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QLinearSigmoid",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["z"],
        domain=domain,
    )
    x = np.array(
        [
            [126, 127],
            [128, 129],
        ],
        dtype=np.uint8,
    )

    y = np.array(
        [
            [127, 127],
            [128, 128],
        ],
        dtype=np.uint8,
    )
    x_scale = np.float32(0.5)
    x_zero_point = np.uint8(127)
    y_scale = np.float32(1.0)
    y_zero_point = np.uint8(127)
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
        name="test_qlinearsigmoid",
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
    test_qlinearsigmoid()
