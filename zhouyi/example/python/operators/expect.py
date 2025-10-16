import onnx
import onnxruntime
import numpy as np
from typing import Sequence
from ZhouyiOperators import operators
from onnx.backend.test.case.node import _extract_value_info
from numpy.testing import assert_allclose
from onnx.helper import tensor_dtype_to_np_dtype, np_dtype_to_tensor_dtype
from onnx import TensorProto


def expect(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    initializer: Sequence[str] = [],
    tol: float = 1e-02,
    opset_imports=[],
) -> None:
    """
    Modified from the following site:
    https://onnx.ai/onnx/expect_onnxruntime.html

    Use to run onnx example:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add

    env: export PYTHONPATH=$PYTHONPATH:~/project/tools/

    param tol: tolenrance
    Usage:
    from expect import expect
    import onnx
    import numpy as np
    node = onnx.helper.make_node("Add",inputs=["x", "y"],outputs=["sum"], )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    expect(node, inputs=[x, y], outputs=[x+y], name="test_add")
    """

    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]

    feeds = {name: value for name, value in zip(present_inputs, inputs)}
    initializers = {}
    real_inputs = []
    real_inputs_name = []
    init_inputs_name = []
    for inp in present_inputs:
        if inp in initializer:
            initializers[inp] = feeds[inp]
            init_inputs_name.append(inp)
            feeds.pop(inp)
        else:
            real_inputs_name.append(inp)
            real_inputs.append(feeds[inp])
    input_type_protos = [None] * len(feeds)

    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(
            real_inputs, real_inputs_name, input_type_protos
        )
    ]

    initializers_vi = []
    for init_input in init_inputs_name:
        init_tensor = onnx.helper.make_tensor(
            init_input,
            onnx.helper.np_dtype_to_tensor_dtype(initializers[init_input].dtype),
            (
                initializers[init_input].shape
                if type(initializers[init_input]) is np.ndarray
                else []
            ),
            (
                initializers[init_input]
                if type(initializers[init_input]) is np.ndarray
                else [initializers[init_input]]
            ),
        )
        initializers_vi.append(init_tensor)

    output_type_protos = [None] * len(outputs)
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(
            outputs, present_outputs, output_type_protos
        )
    ]

    graph = onnx.helper.make_graph(
        nodes=[node],
        name=name,
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers_vi,
    )

    kwargs = {}
    kwargs["producer_name"] = "backend-test"
    if opset_imports:
        kwargs["opset_imports"] = opset_imports
    else:
        kwargs["opset_imports"] = [onnx.helper.make_operatorsetid(node.domain, 21)]

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    # model = onnx.helper.make_model_gen_version(graph, **kwargs)
    model = onnx.helper.make_model(graph, **kwargs)
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0
    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options,
        providers=["ZhouyiExecutionProvider"],
        # providers=["CPUExecutionProvider"],
    )
    results = sess.run(None, feeds)
    for expected, output in zip(outputs, results):
        assert_allclose(expected, output, rtol=tol, atol=tol)


def quantization_params(input, quant_type=np.int8):
    """
    get scale/zero_point
    """
    q_min = np.iinfo(quant_type).min
    q_max = np.iinfo(quant_type).max
    f_min = np.min(input)
    f_max = np.max(input)
    scale = 1.0
    if f_min == f_max:
        scale = 1 / (q_max - q_min)
        return (scale, 0)
    scale = (f_max - f_min) / (q_max - q_min)
    zero_point_from_min = q_min - f_min / scale
    zero_point_from_max = q_max - f_max / scale
    zero_point_from_min_error = abs(q_min) + abs(f_min / scale)
    zero_point_from_max_error = abs(q_max) + abs(f_max / scale)
    if zero_point_from_min_error < zero_point_from_max_error:
        zero_point_double = zero_point_from_min
    else:
        zero_point_double = zero_point_from_max

    nudged_zero_point = 0
    if zero_point_double < q_min:
        nudged_zero_point = q_min
    elif zero_point_double > q_max:
        nudged_zero_point = q_max
    else:
        nudged_zero_point = round(zero_point_double)
    nudged_zero_point = min(q_max, max(q_min, nudged_zero_point))
    return (scale, int(nudged_zero_point))


def simplify_quantize(data, scale, zero_point, quant_type=np.int8):
    """
    quantize data
    """
    q_min = np.iinfo(quant_type).min
    q_max = np.iinfo(quant_type).max
    data_quant = [
        min(q_max, max(q_min, np.round(zero_point + f / scale).astype(quant_type)))
        for f in data.flatten()
    ]
    return np.array(data_quant, dtype=quant_type).reshape(data.shape)


def simplify_dequantize(data, quant_param):
    """
    dequantize data
    """
    scale = quant_param[0]
    zero_point = quant_param[1]
    data_float = [(value - zero_point) * scale for value in data.flatten()]
    return np.array(data_float, dtype=np.float32).reshape(data.shape)


def expect_qdq(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: str,
    initializer: Sequence[str] = [],
    tol: float = 1e-02,
    opset_imports=[],
) -> None:
    """
    1. qdq model need quantize input data
    2. dequantize output data
    3. model is
        input(int8/int16...) ...
            |
            v
        Dequantize (float)
            |
            v
           OP (float)
            |
            v
         output(float)
            |
            v
         Quantize(int8/int16...)
    Usage:
    from expect import expect_qdq
    node = onnx.helper.make_node(
        "Tanh",
        inputs=["x"],
        outputs=["y"],
    )
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
    expect_qdq(node, inputs=[x], outputs=[y], name="test_tanh_example")
    """

    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    output_type_protos = [None] * len(outputs)

    quant_type = np.int8
    quant_param = quantization_params(inputs[0], quant_type)
    quant_input = simplify_quantize(
        inputs[0], quant_param[0], quant_param[1], quant_type
    )

    quant_out_param = quantization_params(outputs[0], quant_type)
    quant_output = simplify_quantize(
        outputs[0], quant_out_param[0], quant_out_param[1], quant_type
    )

    # 1. Dequantize
    dq = onnx.helper.make_tensor_value_info(
        "dq", np_dtype_to_tensor_dtype(quant_input.dtype), quant_input.shape
    )
    dq_scale = onnx.helper.make_tensor(
        "dq_scale", TensorProto.FLOAT, [], [quant_param[0]]
    )
    dq_zero_point = onnx.helper.make_tensor(
        "dq_zero_point",
        np_dtype_to_tensor_dtype(quant_input.dtype),
        [],
        [quant_param[1]],
    )
    dq_output = onnx.helper.make_tensor_value_info(
        "dq_output", TensorProto.FLOAT, quant_input.shape
    )
    node_dq = onnx.helper.make_node(
        op_type="DequantizeLinear",  # node name
        inputs=[
            "dq",
            "dq_scale",
            "dq_zero_point",
        ],
        outputs=["dq_output"],  # outputs
        name=f"node_dq",
    )
    initializer = [dq_scale, dq_zero_point]
    nodes = [node_dq]

    # 2. Create new float node
    node_out = onnx.helper.make_tensor_value_info(
        f"{node.name}_out", TensorProto.FLOAT, quant_input.shape
    )
    node_op = onnx.helper.make_node(
        node.op_type,
        inputs=[
            "dq_output",
        ],
        outputs=[f"{node.op_type}_out"],  # outputs
        name=f"node_{node.op_type}",
    )
    nodes.append(node_op)

    # 3. Quantize
    q_scale = onnx.helper.make_tensor(
        "q_scale", TensorProto.FLOAT, [], [quant_out_param[0]]
    )
    q_zero_point = onnx.helper.make_tensor(
        "q_zero_point",
        np_dtype_to_tensor_dtype(quant_output.dtype),
        [],
        [quant_out_param[1]],
    )
    output_quantize = onnx.helper.make_tensor_value_info(
        "output_quantize",
        np_dtype_to_tensor_dtype(quant_output.dtype),
        quant_output.shape,
    )
    node_quantize = onnx.helper.make_node(
        op_type="QuantizeLinear",  # op_type
        inputs=[
            f"{node.op_type}_out",
            "q_scale",
            "q_zero_point",
        ],
        outputs=["output_quantize"],  # outputs
        name=f"node_{node.op_type}_q",
    )
    initializer.append(q_scale)
    initializer.append(q_zero_point)
    nodes.append(node_quantize)

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=[dq],
        outputs=[output_quantize],
        initializer=initializer,
    )

    kwargs = {}
    kwargs["producer_name"] = "backend-test"
    if opset_imports:
        kwargs["opset_imports"] = opset_imports
    else:
        kwargs["opset_imports"] = [onnx.helper.make_operatorsetid(node.domain, 21)]

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]
    model = onnx.helper.make_model_gen_version(graph, **kwargs)
    # onnx.save(model, name + "_quant_qdq.onnx")
    # Checking the produces are the expected ones.

    sess_zhouyi = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["ZhouyiExecutionProvider"]
    )
    quant_inputs = [quant_input]
    feeds = {}
    for i, input_ele in enumerate(sess_zhouyi.get_inputs()):
        feeds[input_ele.name] = quant_inputs[i]
    results_zhouyi = sess_zhouyi.run(None, feeds)
    # sess = onnxruntime.InferenceSession(
    #     model.SerializeToString(), providers=["CPUExecutionProvider"]
    # )
    # results = sess.run(None, feeds)

    for expected, output in zip(outputs, results_zhouyi):
        output_float = simplify_dequantize(output, quant_out_param)
        # assert_allclose(expected, output_float)
        assert_allclose(expected, output_float, rtol=tol, atol=tol)
