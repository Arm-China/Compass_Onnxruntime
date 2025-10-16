import onnx
import numpy as np
from expect import *


def test_tile():
    node = onnx.helper.make_node("Tile", inputs=["x", "y"], outputs=["z"])

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)

    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)

    z = np.tile(x, repeats)

    expect(node, inputs=[x, repeats], outputs=[z], name="test_tile", initializer=["y"])


def test_tile_precomputed():
    node = onnx.helper.make_node("Tile", inputs=["x", "y"], outputs=["z"])

    x = np.array([[0, 1], [2, 3]], dtype=np.float32)

    repeats = np.array([2, 2], dtype=np.int64)

    z = np.array(
        [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=np.float32
    )

    expect(
        node,
        inputs=[x, repeats],
        outputs=[z],
        name="test_tile_precomputed",
        initializer=["y"],
    )


if __name__ == "__main__":
    test_tile()
    test_tile_precomputed()
