import onnx
import numpy as np
from expect import *


def test_qlinearaverage():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QLinearAveragePool",
        inputs=[
            "x",
            "x_scale",
            "x_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        strides=[1, 1],
        kernel_shape=[5, 5],
        pads=[2, 2, 2, 2],
        auto_pad="NOTSET",
        ceil_mode=False,
        channels_last=False,
        count_include_pad=False,
        domain=domain,
    )

    x = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=np.uint8,
    ).reshape(1, 1, 5, 5)

    y = np.array(
        [
            [
                [
                    [7, 8, 8, 8, 9],
                    [10, 10, 10, 11, 12],
                    [12, 12, 13, 14, 14],
                    [14, 15, 16, 16, 16],
                    [17, 18, 18, 18, 19],
                ]
            ]
        ],
        dtype=np.uint8,
    )
    x_scale = np.float32(0.0369204697)
    x_zero_point = np.uint8(10)
    y_scale = np.float32(0.0369204697)
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
        name="test_qlinearaverage",
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
        tol=1,  #  +-1
    )


if __name__ == "__main__":
    test_qlinearaverage()
