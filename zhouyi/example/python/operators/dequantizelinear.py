import onnx
import numpy as np
from expect import *


def test_dequantizelinear():
    node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
    )

    # scalar zero point and scale
    x = np.array([0, 3, 128, 255]).astype(np.uint8)
    x_scale = np.float32(2)
    x_zero_point = np.uint8(128)
    y = np.array([-256, -250, 0, 254], dtype=np.float32)

    expect(
        node,
        inputs=[x, x_scale, x_zero_point],
        outputs=[y],
        name="test_dequantizelinear",
        initializer=["x_scale", "x_zero_point"],
    )


def test_dequantizelinear_uint16():
    node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
    )

    x = np.array([30000, 31000, 32768, 33000]).astype(np.uint16)
    x_scale = np.float32(2)
    x_zero_point = np.uint16(32767)
    y = np.array([-5534.0, -3534.0, 2.0, 466.0], dtype=np.float32)

    expect(
        node,
        inputs=[x, x_scale, x_zero_point],
        outputs=[y],
        name="test_dequantizelinear_uint16",
        initializer=["x_scale", "x_zero_point"],
    )


def test_dequantizelinear_int16():
    node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
    )

    x = np.array([-300, -30, -1025, 1270]).astype(np.int16)
    x_scale = np.float32(2)
    x_zero_point = np.int16(-1024)
    y = np.array([1448.0, 1988.0, -2.0, 4588.0], dtype=np.float32)

    expect(
        node,
        inputs=[x, x_scale, x_zero_point],
        outputs=[y],
        name="test_dequantizelinear_int16",
        initializer=["x_scale", "x_zero_point"],
    )


if __name__ == "__main__":
    test_dequantizelinear()
    test_dequantizelinear_uint16()
    test_dequantizelinear_int16()
