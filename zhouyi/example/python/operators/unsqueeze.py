import onnx
import numpy as np
from expect import *


def test_unsqueeze_one_axis():
    x = np.random.randn(3, 4, 5).astype(np.float32)

    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=["x", "axes"],
            outputs=["y"],
        )
        y = np.expand_dims(x, axis=i)

        expect(
            node,
            inputs=[x, axes],
            outputs=[y],
            name="test_unsqueeze_axis_" + str(i),
            initializer=["axes"],
        )


def test_unsqueeze_two_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)

    node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=4)

    expect(
        node,
        inputs=[x, axes],
        outputs=[y],
        name="test_unsqueeze_two_axes",
        initializer=["axes"],
    )


def test_unsqueeze_three_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([2, 4, 5]).astype(np.int64)

    node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    expect(
        node,
        inputs=[x, axes],
        outputs=[y],
        name="test_unsqueeze_three_axes",
        initializer=["axes"],
    )


def test_unsqueeze_unsorted_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)

    node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    expect(
        node,
        inputs=[x, axes],
        outputs=[y],
        name="test_unsqueeze_unsorted_axes",
        initializer=["axes"],
    )


def test_unsqueeze_negative_axes():
    node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["x", "axes"],
        outputs=["y"],
    )
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    y = np.expand_dims(x, axis=-2)
    expect(
        node,
        inputs=[x, axes],
        outputs=[y],
        name="test_unsqueeze_negative_axes",
        initializer=["axes"],
    )


if __name__ == "__main__":
    test_unsqueeze_one_axis()
    test_unsqueeze_two_axes()
    test_unsqueeze_three_axes()
    test_unsqueeze_unsorted_axes()
    test_unsqueeze_negative_axes()
