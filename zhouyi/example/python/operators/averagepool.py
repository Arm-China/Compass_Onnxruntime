import onnx
import numpy as np
from expect import *

from onnx.reference.ops.op_pool_common import (
    get_output_shape_auto_pad,
    get_output_shape_explicit_padding,
    get_pad_shape,
    pool,
)


def test_averagepool_2d_default():
    """input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 31, 31]
    """
    node = onnx.helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2, 2],
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    pads = None
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape, _ = get_output_shape_explicit_padding(
        pads, x_shape[2:], kernel_shape, strides
    )
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, "AVG")
    expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_default")


def test_averagepool_2d_pads():
    """input_shape: [1, 3, 28, 28]
    output_shape: [1, 3, 30, 30]
    pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
    """
    node = onnx.helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2],
    )
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = 2
    pad_top = 2
    pad_right = 2
    pad_left = 2
    pads = [pad_top, pad_left, pad_bottom, pad_right]
    out_shape, pads = get_output_shape_explicit_padding(
        pads, x_shape[2:], kernel_shape, strides, ceil_mode=False
    )
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
        mode="constant",
        constant_values=np.nan,
    )
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, "AVG", pads)

    expect(node, inputs=[x], outputs=[y], name="test_averagepool_2d_pads")


if __name__ == "__main__":
    test_averagepool_2d_default()
    test_averagepool_2d_pads()
