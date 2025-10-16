import onnx
import numpy as np
from expect import *

from onnx.numpy_helper import create_random_int


def test_bitwise_or():
    node = onnx.helper.make_node(
        "BitwiseOr",
        inputs=["x", "y"],
        outputs=["bitwiseor"],
    )
    # 2d
    x = create_random_int((3, 4), np.int16)
    y = create_random_int((3, 4), np.int16)
    z = np.bitwise_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_i16_2d")

    # 4d
    x = create_random_int((3, 4, 5, 6), np.int8)
    y = create_random_int((3, 4, 5, 6), np.int8)
    z = np.bitwise_or(x, y)
    expect(node, inputs=[x, y], outputs=[z], name="test_bitwise_or_i16_4d")


if __name__ == "__main__":
    test_bitwise_or()
