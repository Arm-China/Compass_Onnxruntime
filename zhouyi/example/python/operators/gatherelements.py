import onnx
import numpy as np
from expect import *


def gather_elements(data, indices, axis=0):  # type: ignore
    data_swaped = np.swapaxes(data, 0, axis)
    index_swaped = np.swapaxes(indices, 0, axis)
    gathered = np.choose(index_swaped, data_swaped, mode="wrap")
    y = np.swapaxes(gathered, 0, axis)
    return y


def test_gather_elements_0():
    axis = 1
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
    )
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int32)

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[1, 1],
    #  [4, 3]]

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_elements_0",
        initializer=["indices"],
    )


def test_gather_elements_1():
    axis = 0
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
    )
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int32)

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[4, 8, 3],
    #  [7, 2, 3]]

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_elements_1",
        initializer=["indices"],
    )


def test_gather_elements_negative_indices():
    axis = 0
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
    )
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[-1, -2, 0], [-2, 0, 0]], dtype=np.int32)

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[7, 5, 3],
    #  [4, 2, 3]]

    expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        name="test_gather_elements_negative_indices",
        initializer=["indices"],
    )


if __name__ == "__main__":
    test_gather_elements_0()
    test_gather_elements_1()
    test_gather_elements_negative_indices()
