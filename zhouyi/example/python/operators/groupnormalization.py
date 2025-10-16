import onnx
import numpy as np
from expect import *


# Group normalization's reference implementation
def _group_normalization(x, num_groups, scale, bias, epsilon=1e-5):
    # Assume channel is first dim
    assert x.shape[1] % num_groups == 0
    group_size = x.shape[1] // num_groups
    # Reshape to [N, group_size, C/group_size, H, W, ...]
    new_shape = [x.shape[0], num_groups, group_size, *list(x.shape[2:])]
    x_reshaped = x.reshape(new_shape)
    axes = tuple(range(2, len(new_shape)))
    mean = np.mean(x_reshaped, axis=axes, keepdims=True)
    var = np.var(x_reshaped, axis=axes, keepdims=True)
    x_normalized = ((x_reshaped - mean) / np.sqrt(var + epsilon)).reshape(x.shape)
    dim_ones = (1,) * (len(x.shape) - 2)
    scale = scale.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return scale * x_normalized + bias


def test_group_normalization_example():
    c = 4
    num_groups = 2
    x = np.random.randn(3, c, 2, 2).astype(np.float32)
    scale = np.random.randn(c).astype(np.float32)
    bias = np.random.randn(c).astype(np.float32)
    y = _group_normalization(x, num_groups, scale, bias).astype(np.float32)

    node = onnx.helper.make_node(
        "GroupNormalization",
        inputs=["x", "scale", "bias"],
        outputs=["y"],
        num_groups=num_groups,
    )

    expect(
        node,
        inputs=[x, scale, bias],
        outputs=[y],
        name="test_group_normalization_example",
        initializer=["scale", "bias"],
    )


# def test_group_normalization_epsilon():
#     c = 4
#     num_groups = 2
#     x = np.random.randn(3, c, 2, 2).astype(np.float32)
#     scale = np.random.randn(c).astype(np.float32)
#     bias = np.random.randn(c).astype(np.float32)
#     epsilon = 1e-2
#     y = _group_normalization(x, num_groups, scale, bias, epsilon).astype(np.float32)

#     node = onnx.helper.make_node(
#         "GroupNormalization",
#         inputs=["x", "scale", "bias"],
#         outputs=["y"],
#         epsilon=epsilon,
#         num_groups=num_groups,
#     )

#     expect(
#         node,
#         inputs=[x, scale, bias],
#         outputs=[y],
#         name="test_group_normalization_epsilon",
#         initializer=["scale", "bias"],
#     )


if __name__ == "__main__":
    test_group_normalization_example()
    # test_group_normalization_epsilon()
