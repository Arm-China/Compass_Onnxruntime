import onnx
import numpy as np
from expect import *


def test_max_two_inputs():
    data_0 = np.array([3, 2, 1]).astype(np.float32)
    data_1 = np.array([1, 4, 4]).astype(np.float32)
    result = np.array([3, 4, 4]).astype(np.float32)

    result = np.maximum(data_0, data_1)
    node = onnx.helper.make_node(
        "Max",
        inputs=["data_0", "data_1"],
        outputs=["result"],
    )
    expect(node, inputs=[data_0, data_1], outputs=[result], name="test_max_two_inputs")


def test_max_all_numeric_types() -> None:
    test_numeric_dtypes = [
        np.int8,
        np.int16,
        np.uint8,
        np.uint16,
        np.float16,
        np.float32,
    ]
    for op_dtype in test_numeric_dtypes:
        data_0 = np.array([3, 2, 1]).astype(op_dtype)
        data_1 = np.array([1, 4, 4]).astype(op_dtype)
        result = np.array([3, 4, 4]).astype(op_dtype)
        node = onnx.helper.make_node(
            "Max",
            inputs=["data_0", "data_1"],
            outputs=["result"],
        )
        expect(
            node,
            inputs=[data_0, data_1],
            outputs=[result],
            name=f"test_max_{np.dtype(op_dtype).name}",
        )


if __name__ == "__main__":
    test_max_two_inputs()
    test_max_all_numeric_types()
