import onnx
import numpy as np
from expect import *


def test_do_not_keepdims():
    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 0

    node = onnx.helper.make_node(
        "ReduceMin",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
    # print(reduced)
    # [[5., 1.]
    # [30., 1.]
    # [55., 1.]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_do_not_keepdims_example",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_do_not_keepdims_random",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )


def test_keepdims():
    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceMin",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
    # print(reduced)
    # [[[5., 1.]]
    # [[30., 1.]]
    # [[55., 1.]]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_keepdims_example",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_keepdims_random",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )


def test_default_axes_keepdims():
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceMin", inputs=["data"], outputs=["reduced"], keepdims=keepdims
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
    # print(reduced)
    # [[[1.]]]

    expect(
        node,
        inputs=[data],
        outputs=[reduced],
        name="test_reduce_min_default_axes_keepdims_example",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data],
        outputs=[reduced],
        name="test_reduce_min_default_axes_keepdims_random",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )


def test_negative_axes_keepdims():
    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceMin",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
        dtype=np.float32,
    )
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
    # print(reduced)
    # [[[5., 1.]]
    # [[30., 1.]]
    # [[55., 1.]]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_negative_axes_keepdims_example",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_negative_axes_keepdims_random",
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        initializer=["axes"],
    )


def test_bool_inputs():
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceMin",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )

    data = np.array(
        [[True, True], [True, False], [False, True], [False, False]],
    )
    reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=bool(keepdims))
    # print(reduced)
    # [[ True],
    #  [False],
    #  [False],
    #  [False]]

    expect(
        node,
        inputs=[data, axes],
        outputs=[reduced],
        name="test_reduce_min_bool_inputs",
        initializer=["axes"],
    )


if __name__ == "__main__":
    test_do_not_keepdims()
    test_keepdims()
    test_default_axes_keepdims()
    test_negative_axes_keepdims()
    test_bool_inputs()
