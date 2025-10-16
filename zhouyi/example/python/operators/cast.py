import onnx
import numpy as np
from expect import *
from onnx import TensorProto, helper, subbyte


def test_cast():
    shape = (3, 4)
    support_input_dtypes = [
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "FLOAT",
        "FLOAT16",
    ]

    support_output_dtypes = [
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "FLOAT",
        "FLOAT16",
    ]
    for from_type in support_input_dtypes:
        for to_type in support_output_dtypes:
            if from_type == to_type:
                continue
            input = np.random.random_sample(shape).astype(
                onnx.helper.tensor_dtype_to_np_dtype(getattr(TensorProto, from_type))
            )
            output = input.astype(
                helper.tensor_dtype_to_np_dtype(getattr(TensorProto, to_type))
            )

            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=getattr(TensorProto, to_type),
            )
            print("expect test_cast_" + from_type + "_to_" + to_type)
            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_" + from_type + "_to_" + to_type,
            )


if __name__ == "__main__":
    test_cast()
