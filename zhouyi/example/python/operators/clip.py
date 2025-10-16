import onnx
import numpy as np
from expect import *


def test_clip_splitbounds():
    node = onnx.helper.make_node(
        "Clip",
        inputs=["x", "min", "max"],
        outputs=["y"],
    )

    x = np.array([-2, 0, 2]).astype(np.float16)
    min_val = np.float16(-1)
    max_val = np.float16(1)
    y = np.clip(x, min_val, max_val).astype(np.float16)  # expected output [-1., 0., 1.]
    expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        name="test_clip_example",
        initializer=["min", "max"],
    )

    x = np.random.randn(3, 4, 5).astype(np.float16)
    y = np.clip(x, min_val, max_val)
    expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        name="test_clip",
        initializer=["min", "max"],
    )

    node = onnx.helper.make_node(
        "Clip",
        inputs=["x", "min", "max"],
        outputs=["y"],
    )

    min_val = np.float16(-5)
    max_val = np.float16(5)

    x = np.array([-1, 0, 1]).astype(np.float16)
    y = np.array([-1, 0, 1]).astype(np.float16)
    expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        name="test_clip_inbounds",
        initializer=["min", "max"],
    )

    x = np.array([-6, 0, 6]).astype(np.float16)
    y = np.array([-5, 0, 5]).astype(np.float16)
    expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        name="test_clip_outbounds",
        initializer=["min", "max"],
    )

    x = np.array([-1, 0, 6]).astype(np.float16)
    y = np.array([-1, 0, 5]).astype(np.float16)
    expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        name="test_clip_splitbounds",
        initializer=["min", "max"],
    )


if __name__ == "__main__":
    test_clip_splitbounds()
