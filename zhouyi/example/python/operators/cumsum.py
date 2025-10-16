import onnx
import numpy as np
from expect import *


def test_cumsum_1d():
    node = onnx.helper.make_node("CumSum", inputs=["x", "axis"], outputs=["y"])
    x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    axis = np.int32(0)
    y = np.array([1, 3, 6, 10, 15]).astype(np.int32)
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_1d",
        initializer="axis",
    )


def test_cumsum_1d_exclusive():
    node = onnx.helper.make_node(
        "CumSum", inputs=["x", "axis"], outputs=["y"], exclusive=1
    )
    x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    axis = np.int32(0)
    y = np.array([0, 1, 3, 6, 10]).astype(np.int32)
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_1d_exclusive",
        initializer="axis",
    )


def test_cumsum_1d_reverse():
    node = onnx.helper.make_node(
        "CumSum", inputs=["x", "axis"], outputs=["y"], reverse=1
    )
    x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    axis = np.int32(0)
    y = np.array([15, 14, 12, 9, 5]).astype(np.int32)
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_1d_reverse",
        initializer="axis",
    )


def test_cumsum_1d_reverse_exclusive():
    node = onnx.helper.make_node(
        "CumSum", inputs=["x", "axis"], outputs=["y"], reverse=1, exclusive=1
    )
    x = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    axis = np.int32(0)
    y = np.array([14, 12, 9, 5, 0]).astype(np.int32)
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_1d_reverse_exclusive",
        initializer="axis",
    )


def test_cumsum_2d_axis_0():
    node = onnx.helper.make_node(
        "CumSum",
        inputs=["x", "axis"],
        outputs=["y"],
    )
    x = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32).reshape((2, 3))
    axis = np.int32(0)
    y = np.array([1, 2, 3, 5, 7, 9]).astype(np.int32).reshape((2, 3))
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_2d_axis_0",
        initializer="axis",
    )


def test_cumsum_2d_axis_1():
    node = onnx.helper.make_node(
        "CumSum",
        inputs=["x", "axis"],
        outputs=["y"],
    )
    x = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32).reshape((2, 3))
    axis = np.int32(1)
    y = np.array([1, 3, 6, 4, 9, 15]).astype(np.int32).reshape((2, 3))
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_2d_axis_1",
        initializer="axis",
    )


def test_cumsum_2d_negative_axis():
    node = onnx.helper.make_node(
        "CumSum",
        inputs=["x", "axis"],
        outputs=["y"],
    )
    x = np.array([1, 2, 3, 4, 5, 6]).astype(np.int32).reshape((2, 3))
    axis = np.int32(-1)
    y = np.array([1, 3, 6, 4, 9, 15]).astype(np.int32).reshape((2, 3))
    expect(
        node,
        inputs=[x, axis],
        outputs=[y],
        name="test_cumsum_2d_negative_axis",
        initializer="axis",
    )


if __name__ == "__main__":
    test_cumsum_1d()
    test_cumsum_1d_exclusive()
    test_cumsum_1d_reverse()
    test_cumsum_1d_reverse_exclusive()
    test_cumsum_2d_axis_0()
    test_cumsum_2d_axis_1()
    test_cumsum_2d_negative_axis()
