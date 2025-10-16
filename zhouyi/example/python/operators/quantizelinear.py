import onnx
import numpy as np
from expect import *
from onnx import TensorProto
from onnx.helper import make_tensor


def test_quantizelinear():
    node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero_point"],
        outputs=["y"],
    )

    x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
    y_scale = np.float32(2)
    y_zero_point = np.uint8(128)
    y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

    expect(
        node,
        inputs=[x, y_scale, y_zero_point],
        outputs=[y],
        name="test_quantizelinear",
        initializer=["y_scale", "y_zero_point"],
    )


def test_uint16():
    node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero_point"],
        outputs=["y"],
    )

    x = np.array(
        [
            0.0,
            -128.0,
            3.0,
            -3.0,
            2.9,
            -2.9,
            3.1,
            -3.1,
            65536.0,
            -65534.0,
            70000.0,
            -70000.0,
        ]
    ).astype(np.float32)
    y_scale = np.float32(2.0)
    y_zero_point = np.uint16(32767)
    y = np.array(
        [
            32767,
            32703,
            32769,
            32765,
            32768,
            32766,
            32769,
            32765,
            65535,
            0,
            65535,
            0,
        ]
    ).astype(np.uint16)

    expect(
        node,
        inputs=[x, y_scale, y_zero_point],
        outputs=[y],
        name="test_quantizelinear_uint16",
        initializer=["y_scale", "y_zero_point"],
    )


def test_int16():
    node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero_point"],
        outputs=["y"],
    )

    x = np.array(
        [
            0.0,
            -514.0,
            3.0,
            -3.0,
            2.9,
            -2.9,
            3.1,
            -3.1,
            65022.0,
            -66046.0,
            65023.0,
            -66047.0,
            65024.0,
            -66048.0,
            70000.0,
            -70000.0,
        ]
    ).astype(np.float32)
    y_scale = np.float32(2.0)
    y_zero_point = np.int16(256)
    y = np.array(
        [
            256,
            -1,
            258,
            254,
            257,
            255,
            258,
            254,
            32767,
            -32767,
            32767,
            -32768,
            32767,
            -32768,
            32767,
            -32768,
        ]
    ).astype(np.int16)

    expect(
        node,
        inputs=[x, y_scale, y_zero_point],
        outputs=[y],
        name="test_quantizelinear_int16",
        initializer=["y_scale", "y_zero_point"],
    )


if __name__ == "__main__":
    test_quantizelinear()
    test_uint16()
    test_int16()
