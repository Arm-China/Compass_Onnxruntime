import onnx
import numpy as np
from expect import *


def test_add_int8():
    node = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["sum"],
    )
    x = np.random.randint(24, size=(3, 4, 5), dtype=np.int8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.int8)
    expect(node, inputs=[x, y], outputs=[x + y], name="test_add_int8")


def test_add_uint8():
    node = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["sum"],
    )
    x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint8")


def test_add_int16():
    node = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["sum"],
    )
    x = np.random.randint(24, size=(3, 4, 5), dtype=np.int16)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.int16)
    expect(node, inputs=[x, y], outputs=[x + y], name="test_add_int16")


def test_add_uint16():
    node = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["sum"],
    )
    x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint16)
    expect(node, inputs=[x, y], outputs=[x + y], name="test_add_uint16")


def test_add_float32():
    node = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["sum"],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    expect(node, inputs=[x, y], outputs=[x + y], name="test_add_float32")


if __name__ == "__main__":
    test_add_int8()
    test_add_uint8()
    test_add_int16()
    test_add_uint16()
    test_add_float32()
