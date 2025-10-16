import onnx
import numpy as np
from expect import *

from onnx.reference.ops.op_resize import _interpolate_nd as interpolate_nd
from onnx.reference.ops.op_resize import _linear_coeffs as linear_coeffs
from onnx.reference.ops.op_resize import (
    _linear_coeffs_antialias as linear_coeffs_antialias,
)
from onnx.reference.ops.op_resize import _nearest_coeffs as nearest_coeffs


def test_resize_upsample_scales_nearest():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="nearest",
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # [[[[1. 1. 1. 2. 2. 2.]
    #    [1. 1. 1. 2. 2. 2.]
    #    [3. 3. 3. 4. 4. 4.]
    #    [3. 3. 3. 4. 4. 4.]]]]
    output = interpolate_nd(
        data, lambda x, _: nearest_coeffs(x), scale_factors=scales
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, scales],
        outputs=[output],
        name="test_resize_upsample_scales_nearest",
        initializer=["scales"],
    )


def test_resize_downsample_scales_nearest():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="nearest",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # [[[[1. 3.]]]]
    output = interpolate_nd(
        data, lambda x, _: nearest_coeffs(x), scale_factors=scales
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, scales],
        outputs=[output],
        name="test_resize_downsample_scales_nearest",
        initializer=["scales"],
    )


def test_resize_upsample_sizes_nearest():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="nearest",
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 7, 8], dtype=np.int64)

    # [[[[1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]]]]
    output = interpolate_nd(
        data, lambda x, _: nearest_coeffs(x), output_size=sizes
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_upsample_sizes_nearest",
        initializer=["sizes"],
    )


def test_resize_downsample_sizes_nearest():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="nearest",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 1, 3], dtype=np.int64)

    # [[[[1. 2. 4.]]]]
    output = interpolate_nd(
        data, lambda x, _: nearest_coeffs(x), output_size=sizes
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_downsample_sizes_nearest",
        initializer=["sizes"],
    )


def test_resize_upsample_scales_linear():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="linear",
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[1.   1.25 1.75 2.  ]
    #    [1.5  1.75 2.25 2.5 ]
    #    [2.5  2.75 3.25 3.5 ]
    #    [3.   3.25 3.75 4.  ]]]]
    output = interpolate_nd(
        data, lambda x, _: linear_coeffs(x), scale_factors=scales
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, scales],
        outputs=[output],
        name="test_resize_upsample_scales_linear",
        initializer=["scales"],
    )


def test_resize_upsample_scales_linear_align_corners():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "scales"],
        outputs=["Y"],
        mode="linear",
        coordinate_transformation_mode="align_corners",
    )

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[1.         1.33333333 1.66666667 2.        ]
    #    [1.66666667 2.         2.33333333 2.66666667]
    #    [2.33333333 2.66666667 3.         3.33333333]
    #    [3.         3.33333333 3.66666667 4.        ]]]]
    output = interpolate_nd(
        data,
        lambda x, _: linear_coeffs(x),
        scale_factors=scales,
        coordinate_transformation_mode="align_corners",
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, scales],
        outputs=[output],
        name="test_resize_upsample_scales_linear_align_corners",
        initializer=["scales"],
    )


def test_resize_downsample_sizes_linear_pytorch_half_pixel():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="linear",
        coordinate_transformation_mode="pytorch_half_pixel",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 3, 1], dtype=np.int64)

    # [[[[ 1.6666666]
    #    [ 7.       ]
    #    [12.333333 ]]]]
    output = interpolate_nd(
        data,
        lambda x, _: linear_coeffs(x),
        output_size=sizes,
        coordinate_transformation_mode="pytorch_half_pixel",
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_downsample_sizes_linear_pytorch_half_pixel",
        initializer=["sizes"],
    )


def test_resize_upsample_sizes_nearest_floor_align_corners():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="nearest",
        coordinate_transformation_mode="align_corners",
        nearest_mode="floor",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
    #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
    #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
    #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
    #    [13. 13. 13. 14. 14. 15. 15. 16.]]]]
    output = interpolate_nd(
        data,
        lambda x, _: nearest_coeffs(x, mode="floor"),
        output_size=sizes,
        coordinate_transformation_mode="align_corners",
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_upsample_sizes_nearest_floor_align_corners",
        initializer=["sizes"],
    )


def test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="round_prefer_ceil",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
    output = interpolate_nd(
        data,
        lambda x, _: nearest_coeffs(x, mode="round_prefer_ceil"),
        output_size=sizes,
        coordinate_transformation_mode="asymmetric",
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric",
        initializer=["sizes"],
    )


def test_resize_upsample_sizes_nearest_ceil_half_pixel():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="nearest",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="ceil",
    )

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=np.float32,
    )

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
    output = interpolate_nd(
        data, lambda x, _: nearest_coeffs(x, mode="ceil"), output_size=sizes
    ).astype(np.float32)

    expect(
        node,
        inputs=[data, sizes],
        outputs=[output],
        name="test_resize_upsample_sizes_nearest_ceil_half_pixel",
        initializer=["sizes"],
    )


# def test_resize_upsample_scales_nearest_axes_2_3():
#     axes = [2, 3]
#     node = onnx.helper.make_node(
#         "Resize",
#         inputs=["X", "", "scales"],
#         outputs=["Y"],
#         mode="nearest",
#         axes=axes,
#     )

#     data = np.array(
#         [
#             [
#                 [
#                     [1, 2],
#                     [3, 4],
#                 ]
#             ]
#         ],
#         dtype=np.float32,
#     )

#     scales = np.array([2.0, 3.0], dtype=np.float32)

#     # [[[[1. 1. 1. 2. 2. 2.]
#     #    [1. 1. 1. 2. 2. 2.]
#     #    [3. 3. 3. 4. 4. 4.]
#     #    [3. 3. 3. 4. 4. 4.]]]]
#     output = interpolate_nd(
#         data, lambda x, _: nearest_coeffs(x), scale_factors=scales, axes=axes
#     ).astype(np.float32)

#     expect(
#         node,
#         inputs=[data, scales],
#         outputs=[output],
#         name="test_resize_upsample_scales_nearest_axes_2_3",
#         initializer=["scales"],
#     )


if __name__ == "__main__":
    test_resize_upsample_scales_nearest()
    test_resize_downsample_scales_nearest()
    test_resize_upsample_sizes_nearest()
    test_resize_downsample_sizes_nearest()
    test_resize_upsample_scales_linear()
    test_resize_upsample_scales_linear_align_corners()
    test_resize_downsample_sizes_linear_pytorch_half_pixel()
    test_resize_upsample_sizes_nearest_floor_align_corners()
    test_resize_upsample_sizes_nearest_ceil_half_pixel()
    test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric()
    # test_resize_upsample_scales_nearest_axes_2_3()  # crashed TODO: dump graph
