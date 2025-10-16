import onnx
import numpy as np
from expect import *


def gemm_reference_implementation(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    transA: int = 0,
    transB: int = 0,
) -> np.ndarray:
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y.astype(A.dtype)


def test_gemm_default_zero_bias0():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([4, 5]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, 1.0, 1.0, 0, 1)
    expect(
        node,
        inputs=[a, b, c],
        outputs=[y],
        name="test_gemm_default_zero_bias0",
        initializer=["b", "c"],
    )


def test_gemm_default_no_bias():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b"],
        outputs=["y"],
    )
    a = np.random.ranf([2, 10]).astype(np.float32)
    b = np.random.ranf([10, 3]).astype(np.float32)
    y = gemm_reference_implementation(a, b)
    expect(node, inputs=[a, b], outputs=[y], name="test_gemm_default_no_bias")


def test_gemm_default_scalar_bias():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )
    a = np.random.ranf([2, 3]).astype(np.float32)
    b = np.random.ranf([4, 3]).astype(np.float32)
    c = np.array([3.14, 3.14, 3.14, 3.14]).astype(np.float32)
    print(c)
    y = gemm_reference_implementation(a, b, c, 1.0, 1.0, 0, 1)
    expect(
        node,
        inputs=[a, b, c],
        outputs=[y],
        name="test_gemm_default_scalar_bias",
        initializer=["b", "c"],
    )


def test_gemm_default_single_elem_vector_bias():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )
    a = np.random.ranf([3, 7]).astype(np.float32)
    b = np.random.ranf([3, 7]).astype(np.float32)
    c = np.random.ranf([3]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, 1.0, 1.0, 0, 1)
    expect(
        node,
        inputs=[a, b, c],
        outputs=[y],
        name="test_gemm_default_single_elem_vector_bias",
        initializer=["b", "c"],
    )


def test_gemm_default_vector_bias():
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([4, 7]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, 1.0, 1.0, 0, 1)
    expect(
        node,
        inputs=[a, b, c],
        outputs=[y],
        name="test_gemm_default_vector_bias",
        initializer=["b", "c"],
    )


def test_gemm_transposeA():
    node = onnx.helper.make_node("Gemm", inputs=["a", "b"], outputs=["y"], transA=1)
    a = np.random.ranf([6, 3]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, transA=1)
    expect(node, inputs=[a, b], outputs=[y], name="test_gemm_transposeA")


def test_gemm_transposeB():
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], transB=1
    )
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([4, 6]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transB=1)
    expect(
        node,
        inputs=[a, b, c],
        outputs=[y],
        name="test_gemm_transposeB",
        initializer=["b", "c"],
    )


if __name__ == "__main__":
    test_gemm_default_zero_bias0()
    test_gemm_default_no_bias()
    test_gemm_default_scalar_bias()
    test_gemm_default_single_elem_vector_bias()
    test_gemm_default_vector_bias()
    test_gemm_transposeA()
    test_gemm_transposeB()
