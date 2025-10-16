import onnx
import numpy as np
from expect import *

from onnx.numpy_helper import create_random_int


def test_bitwise_xor():
    node = onnx.helper.make_node(
        "BitwiseXor",
        inputs=["x", "y"],
        outputs=["bitwisexor"],
    )

    # 2d
    x = create_random_int((3, 4), np.int16)
    y = create_random_int((3, 4), np.int16)
    z = np.bitwise_xor(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i16_2d")

    # 3d
    x = create_random_int((3, 4, 5), np.int16)
    y = create_random_int((3, 4, 5), np.int16)
    z = np.bitwise_xor(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_xor_i16_3d")


if __name__ == "__main__":
    test_bitwise_xor()
