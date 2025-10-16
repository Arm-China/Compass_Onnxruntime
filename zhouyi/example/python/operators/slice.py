import onnx
import numpy as np
from expect import *


def test_slice():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes", "steps"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    y = x[0:3, 0:10]
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)

    expect(
        node,
        inputs=[x, starts, ends, axes, steps],
        outputs=[y],
        name="test_slice",
        initializer=["starts", "ends", "axes", "steps"],
    )


def test_slice_neg():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes", "steps"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 0:-1]

    expect(
        node,
        inputs=[x, starts, ends, axes, steps],
        outputs=[y],
        name="test_slice_neg",
        initializer=["starts", "ends", "axes", "steps"],
    )


def test_slice_default_axes():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(
        node,
        inputs=[x, starts, ends],
        outputs=[y],
        name="test_slice_default_axes",
        initializer=["starts", "ends"],
    )


def test_slice_default_steps():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(
        node,
        inputs=[x, starts, ends, axes],
        outputs=[y],
        name="test_slice_default_steps",
        initializer=["starts", "ends", "axes"],
    )


def test_slice_neg_steps():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes", "steps"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2]).astype(np.int64)
    y = x[20:0:-1, 10:0:-3, 4:1:-2]

    expect(
        node,
        inputs=[x, starts, ends, axes, steps],
        outputs=[y],
        name="test_slice_neg_steps",
        initializer=["starts", "ends", "axes", "steps"],
    )


def test_slice_negative_axes():
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends", "axes"],
        outputs=["y"],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(
        node,
        inputs=[x, starts, ends, axes],
        outputs=[y],
        name="test_slice_negative_axes",
        initializer=[
            "starts",
            "ends",
            "axes",
        ],
    )


if __name__ == "__main__":
    test_slice()
    test_slice_neg()
    test_slice_default_axes()
    test_slice_default_steps()
    test_slice_neg_steps()
    test_slice_negative_axes()
