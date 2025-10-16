import onnx
import numpy as np
from expect import *


def test_qlinearaconcat():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QLinearConcat",
        inputs=[
            "z_scale",
            "z_zero_point",
            "x",
            "x_scale",
            "x_zero_point",
            "y",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["z"],
        axis=-1,
        domain=domain,
    )
    x = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.uint8,
    )
    y = np.array(
        [
            [5, 6],
            [7, 8],
        ],
        dtype=np.uint8,
    )
    z = np.array(
        [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
        ],
        dtype=np.uint8,
    )
    x_scale = np.float32(1.0)
    x_zero_point = np.uint8(0)
    y_scale = np.float32(1.0)
    y_zero_point = np.uint8(0)
    z_scale = np.float32(1.0)
    z_zero_point = np.uint8(0)
    expect(
        node,
        inputs=[
            z_scale,
            z_zero_point,
            x,
            x_scale,
            x_zero_point,
            y,
            y_scale,
            y_zero_point,
        ],
        outputs=[z],
        name="test_qlinearaconcat",
        initializer=[
            "z_scale",
            "z_zero_point",
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
    test_qlinearaconcat()
