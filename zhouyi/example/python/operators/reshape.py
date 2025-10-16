import onnx
import numpy as np
from expect import *


def reshape_reference_implementation(
    data: np.ndarray, shape: np.ndarray, allowzero: int = 0
) -> np.ndarray:
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped


def test_reshape():
    original_shape = [2, 3, 4]
    test_cases = {
        "reordered_all_dims": np.array([4, 2, 3], dtype=np.int64),
        "reordered_last_dims": np.array([2, 4, 3], dtype=np.int64),
        "reduced_dims": np.array([2, 12], dtype=np.int64),
        "extended_dims": np.array([2, 3, 2, 2], dtype=np.int64),
        "one_dim": np.array([24], dtype=np.int64),
        "negative_dim": np.array([2, -1, 2], dtype=np.int64),
        "negative_extended_dims": np.array([-1, 2, 3, 4], dtype=np.int64),
    }
    data = np.random.random_sample(original_shape).astype(np.float32)

    for test_name, shape in test_cases.items():
        node = onnx.helper.make_node(
            "Reshape",
            inputs=["data", "shape"],
            outputs=["reshaped"],
        )

        reshaped = reshape_reference_implementation(data, shape)

        expect(
            node,
            inputs=[data, shape],
            outputs=[reshaped],
            name="test_reshape_" + test_name,
            initializer=["shape"],
        )


if __name__ == "__main__":
    test_reshape()
