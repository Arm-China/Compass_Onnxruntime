import onnx
import numpy as np
from expect import *
from onnx.numpy_helper import create_random_int


def test_bitwise_and():
    node = onnx.helper.make_node(
        "BitwiseAnd",
        inputs=["x", "y"],
        outputs=["bitwiseand"],
    )

    # 2d
    x = create_random_int((3, 4), np.uint16)
    y = create_random_int((3, 4), np.uint16)
    z = np.bitwise_and(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_and_u16_2d")

    # 3d
    x = create_random_int((3, 4, 5), np.int16)
    y = create_random_int((3, 4, 5), np.int16)
    z = np.bitwise_and(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_and_i16_3d")


if __name__ == "__main__":
    test_bitwise_and()
