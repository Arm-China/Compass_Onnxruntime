import onnx
import numpy as np
from expect import *


def test_qlinearqgemm():
    domain = "com.microsoft"
    node = onnx.helper.make_node(
        "QGemm",
        inputs=[
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "c",
            "y_scale",
            "y_zero_point",
        ],
        outputs=["y"],
        alpha=1.0,
        transA=0,
        transB=1,
        domain=domain,
    )
    a = np.array([[2, 2, 2, 1]], dtype=np.int8).reshape(1, 4)
    b = np.array([[11, 4, 0, 8], [15, 2, 8, 9], [7, 11, 10, 2]], dtype=np.int8).reshape(
        3, 4
    )
    c = np.array([5, 8, 3], dtype=np.int32).reshape(3)
    y = np.array([[43, 67, 61]], dtype=np.int8).reshape(1, 3)

    # zhouyi FullyConnected unsupport  input scale != output scale
    # CPU EP  output_scale = a_scale * b_scale
    # So only support scale=1.0
    a_scale = np.float32(1.0)
    a_zero_point = np.int8(0)
    b_scale = np.float32(1.0)
    b_zero_point = np.int8(0)
    y_scale = np.float32(1.0)
    y_zero_point = np.int8(0)
    expect(
        node,
        inputs=[
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            c,
            y_scale,
            y_zero_point,
        ],
        outputs=[y],
        name="test_qlinearqgemm",
        initializer=[
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "c",
            "y_scale",
            "y_zero_point",
        ],
        opset_imports=[
            onnx.helper.make_opsetid(domain, 1),
        ],
    )


if __name__ == "__main__":
    test_qlinearqgemm()
