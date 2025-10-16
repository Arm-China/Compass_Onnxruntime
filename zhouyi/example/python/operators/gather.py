import onnx
import numpy as np
from expect import *


def test_gather_0():
    node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=0,
    )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=0)

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_0",
        initializer=["indices"],
    )


def test_gather_1():
    node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=1,
    )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=1)

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_1",
        initializer=["indices"],
    )


def test_gather_2d_indices():
    node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=1,
    )
    data = np.random.randn(3, 3).astype(np.float32)
    indices = np.array([[0, 2]])
    y = np.take(data, indices, axis=1)

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_2d_indices",
        initializer=["indices"],
    )


if __name__ == "__main__":
    test_gather_0()
    test_gather_1()
    test_gather_2d_indices()
