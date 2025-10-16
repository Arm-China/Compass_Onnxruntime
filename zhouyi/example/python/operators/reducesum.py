import onnx
import numpy as np
from expect import *


def test_do_not_keepdims():
    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 0

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    # print(reduced)
    # [[4., 6.]
    # [12., 14.]
    # [20., 22.]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_do_not_keepdims_example",
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_do_not_keepdims_random",
        initializer=["axes"],
    )


def test_keepdims():
    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    # print(reduced)
    # [[[4., 6.]]
    # [[12., 14.]]
    # [[20., 22.]]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_keepdims_example",
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_keepdims_random",
        initializer=["axes"],
    )


def test_default_axes_keepdims():
    shape = [3, 2, 2]
    axes = np.array([], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.sum(data, axis=None, keepdims=keepdims == 1)
    # print(reduced)
    # [[[78.]]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_default_axes_keepdims_example",
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=None, keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_default_axes_keepdims_random",
        initializer=["axes"],
    )


def test_negative_axes_keepdims():
    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceSum", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32
    )
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    # print(reduced)
    # [[[4., 6.]]
    # [[12., 14.]]
    # [[20., 22.]]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_negative_axes_keepdims_example",
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_sum_negative_axes_keepdims_random",
        initializer=["axes"],
    )


if __name__ == "__main__":
    test_do_not_keepdims()
    test_keepdims()
    test_default_axes_keepdims()
    test_negative_axes_keepdims()
