import onnx
import numpy as np
from expect import *
import itertools


def test_default():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)

    node = onnx.helper.make_node("Transpose", inputs=["data"], outputs=["transposed"])

    transposed = np.transpose(data)
    expect(node, inputs=[data], outputs=[transposed], name="test_transpose_default")


def test_all_permutations():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    permutations = list(itertools.permutations(np.arange(len(shape))))

    for i, permutation in enumerate(permutations):
        node = onnx.helper.make_node(
            "Transpose",
            inputs=["data"],
            outputs=["transposed"],
            perm=permutation,
        )
        transposed = np.transpose(data, permutation)
        expect(
            node,
            inputs=[data],
            outputs=[transposed],
            name=f"test_transpose_all_permutations_{i}",
        )


if __name__ == "__main__":
    test_default()
    test_all_permutations()
