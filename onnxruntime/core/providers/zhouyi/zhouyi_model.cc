// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#include "zhouyi_model.h"

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace zhouyi {
namespace {
std::unordered_map<std::string, aipubt::ops::ResizeMethod> supported_modes = {
    {std::string("linear"), aipubt::ops::ResizeMethod::Bilinear},
    {std::string("nearest"), aipubt::ops::ResizeMethod::NearestNeighbor},
};
std::unordered_map<std::string, aipubt::ops::ResizeNearestMode> supported_nearest_modes = {
    {"round_prefer_floor", aipubt::ops::ResizeNearestMode::RoundPreferFloor},
    {"round_prefer_ceil", aipubt::ops::ResizeNearestMode::RoundPreferCeil},
    {"floor", aipubt::ops::ResizeNearestMode::Floor},
    {"ceil", aipubt::ops::ResizeNearestMode::Ceil},
};

std::unordered_map<std::string, aipubt::ops::ResizeMode> supported_ct_modes = {
    {"half_pixel", aipubt::ops::ResizeMode::HalfPixel},
    {"align_corners", aipubt::ops::ResizeMode::AlignCorners},
    {"asymmetric", aipubt::ops::ResizeMode::Asymmetric},
    {"pytorch_half_pixel", aipubt::ops::ResizeMode::PytorchHalfPixel},
};
}  // namespace

ZhouyiModel::ZhouyiModel(const onnxruntime::GraphViewer* graph_viewer, const logging::Logger& logger)
    : graph_viewer_(graph_viewer), logger_(logger) {
  std::tie(node_unit_holder_, node_unit_map_) = QDQ::GetAllNodeUnits(*graph_viewer_, logger_);
  aipubt_graph_ = aipubt::Graph::create(graph_viewer_->Name(), aipubt::DataLayout::DataLayout_NHWC);
  for (auto initialized : graph_viewer_->GetAllInitializedTensors()) {
    init_tensors_.insert(initialized.first);
  }
}

bool ZhouyiModel::IsInitTensors(const std::string& name) const {
  return init_tensors_.find(name) != init_tensors_.end();
}

void ZhouyiModel::EmplaceInputs() {
  TensorPtrList inputs_;
  for (auto input : graph_viewer_->GetInputs()) {
    const auto& input_name = input->Name();
    // exclude initializer inputs,initializer inputs as constont tensor
    if (IsInitTensors(input_name)) {
      continue;
    }
    LOGS(logger_, VERBOSE) << "input " << input_name;

    auto aipu_dtype = ParseDType(input->TypeAsProto());
    auto quant = GetQuant(input_name);
    auto shape = ParseShape(*input);

    auto input_t = aipubt::ops::input(shape, aipu_dtype, quant, aipubt_graph_);
    tensor_maps_.emplace(std::piecewise_construct, std::forward_as_tuple(input_name),
                         std::forward_as_tuple(input_t));

    inputs_.push_back(input_t);
  }

  aipubt_graph_->record_input_tensors(inputs_);
  return;
}

Status ZhouyiModel::Build() {
  if (prepared_)
    return Status::OK();
  EmplaceInputs();
  // Op builer
  const auto& node_indices = graph_viewer_->GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_->GetNode(node_indices[i]));

    // Check whether it's part of NodeUnit
    auto* node_unit = node_unit_map_.at(node);
    const std::string& op_type = node_unit->OpType();
    LOGS(logger_, VERBOSE) << " node name: [" << node->Name()
                           << "] node optype: [" << op_type
                           << "] as part of the NodeUnit type: [" << op_type
                           << "] name: [" << node_unit->Name()
                           << "] Index: [" << node_unit->Index()
                           << "]";
    if (node != &node_unit->GetNode()) {
      continue;
    }
    ORT_RETURN_IF_ERROR(AddNode(node_unit));
  }

  // record input/output
  TensorPtrList outputs_;
  for (auto t : graph_viewer_->GetOutputs()) {
    outputs_.push_back(AipuTensorMatch(t));
  }
  aipubt_graph_->record_output_tensors(outputs_);

  prepared_ = true;
  return Status::OK();
}

Status ZhouyiModel::AddNode(const NodeUnit* node_unit) {
  TensorPtrList aipu_outputs;
  Status ret = Status::OK();
  Status failed_ret = Status_FAIL("Add node '", node_unit->Name(), "' failed.");
  auto optype = node_unit->OpType();
  LOGS(logger_, INFO) << " optype: " << optype;

  if (optype == "Add") {
    aipu_outputs = OpAdd(node_unit);
  } else if (optype == "Sub") {
    aipu_outputs = OpSub(node_unit);
  } else if (optype == "Mul") {
    aipu_outputs = OpMul(node_unit);
  } else if (optype == "Max") {
    aipu_outputs = OpMax(node_unit);
  } else if (optype == "Min") {
    aipu_outputs = OpMin(node_unit);
  } else if (optype == "QLinearAdd") {
    aipu_outputs = OpAdd(node_unit);
  } else if (optype == "QLinearMul") {
    aipu_outputs = OpMul(node_unit);
  } else if (optype == "MatMul") {
    aipu_outputs = OpMatMul(node_unit);
  } else if (optype == "QLinearMatMul") {
    aipu_outputs = OpMatMul(node_unit);
  } else if (optype == "Gemm") {
    aipu_outputs = OpGemm(node_unit);
  } else if (optype == "QGemm") {
    aipu_outputs = OpQGemm(node_unit);
  } else if (optype == "Softmax") {
    aipu_outputs = OpSoftmax(node_unit);
  } else if (optype == "LogSoftmax") {
    aipu_outputs = OpLogSoftmax(node_unit);
  } else if (optype == "Cast") {
    aipu_outputs = OpCast(node_unit);
  } else if (optype == "And") {
    aipu_outputs = OpAnd(node_unit);
  } else if (optype == "Or") {
    aipu_outputs = OpOr(node_unit);
  } else if (optype == "Not") {
    aipu_outputs = OpNot(node_unit);
  } else if (optype == "Greater") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::greater);
  } else if (optype == "GreaterOrEqual") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::greater_equal);
  } else if (optype == "Less") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::less);
  } else if (optype == "LessOrEqual") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::less_equal);
  } else if (optype == "Equal") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::equal);
  } else if (optype == "BitwiseAnd") {
    aipu_outputs = OpBitwise(node_unit, aipubt::ops::BitwiseMethod::AND);
  } else if (optype == "BitwiseOr") {
    aipu_outputs = OpBitwise(node_unit, aipubt::ops::BitwiseMethod::OR);
  } else if (optype == "BitwiseXor") {
    aipu_outputs = OpBitwise(node_unit, aipubt::ops::BitwiseMethod::XOR);
  } else if (optype == "BitwiseNot") {
    aipu_outputs = OpBitwise(node_unit, aipubt::ops::BitwiseMethod::NOT);
  } else if (optype == "Pad") {
    aipu_outputs = OpPad(node_unit);
  } else if (optype == "QuantizeLinear") {
    aipu_outputs = OpQuantizeLinear(node_unit);
  } else if (optype == "DequantizeLinear") {
    aipu_outputs = OpDequantizeLinear(node_unit);
  } else if (optype == "Conv") {
    aipu_outputs = OpConv(node_unit);
  } else if (optype == "QLinearConv") {
    aipu_outputs = OpConv(node_unit);
  } else if (optype == "AveragePool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "GlobalAveragePool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "GlobalMaxPool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "MaxPool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "QLinearAveragePool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "QLinearGlobalAveragePool") {
    aipu_outputs = OpPool(node_unit);
  } else if (optype == "Relu") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::relu);
  } else if (optype == "Sigmoid") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::sigmoid);
  } else if (optype == "QLinearSigmoid") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::sigmoid);
  } else if (optype == "HardSwish") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::hard_swish);
  } else if (optype == "Tanh") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::tanh);
  } else if (optype == "Pow") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::pow);
  } else if (optype == "Div") {
    aipu_outputs = OpBinary(node_unit, aipubt::ops::div);
  } else if (optype == "Sqrt") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::sqrt);
  } else if (optype == "Abs") {
    aipu_outputs = OpUnary(node_unit, aipubt::ops::abs);
  } else if (optype == "Erf") {
    aipu_outputs = OpErf(node_unit);
  } else if (optype == "Concat") {
    aipu_outputs = OpConcat(node_unit);
  } else if (optype == "QLinearConcat") {
    aipu_outputs = OpConcat(node_unit);
  } else if (optype == "PRelu") {
    aipu_outputs = OpPrelu(node_unit);
  } else if (optype == "LeakyRelu") {
    aipu_outputs = OpLeakyRelu(node_unit);
  } else if (optype == "QLinearLeakyRelu") {
    aipu_outputs = OpLeakyRelu(node_unit);
  } else if (optype == "Clip") {
    aipu_outputs = OpClip(node_unit);
  } else if (optype == "Reshape") {
    aipu_outputs = OpReshape(node_unit);
  } else if (optype == "Flatten") {
    aipu_outputs = OpFlatten(node_unit);
  } else if (optype == "Squeeze") {
    aipu_outputs = OpSqueeze(node_unit);
  } else if (optype == "Unsqueeze") {
    aipu_outputs = OpUnSqueeze(node_unit);
  } else if (optype == "SpaceToDepth") {
    aipu_outputs = OpSpaceToDepth(node_unit);
  } else if (optype == "DepthToSpace") {
    aipu_outputs = OpDepthToSpace(node_unit);
  } else if (optype == "Transpose") {
    aipu_outputs = OpTranspose(node_unit);
  } else if (optype == "Split") {
    aipu_outputs = OpSplit(node_unit);
  } else if (optype == "Slice") {
    aipu_outputs = OpSlice(node_unit);
  } else if (optype == "ReduceMax") {
    aipu_outputs = OpReduce(node_unit);
  } else if (optype == "ReduceMin") {
    aipu_outputs = OpReduce(node_unit);
  } else if (optype == "ReduceMean") {
    aipu_outputs = OpReduce(node_unit);
  } else if (optype == "ReduceSum") {
    aipu_outputs = OpReduce(node_unit);
  } else if (optype == "ReduceProd") {
    aipu_outputs = OpReduce(node_unit);
  } else if (optype == "LRN") {
    aipu_outputs = OpLRN(node_unit);
  } else if (optype == "Gather") {
    aipu_outputs = OpGather(node_unit);
  } else if (optype == "GatherND") {
    aipu_outputs = OpGatherND(node_unit);
  } else if (optype == "GatherElements") {
    aipu_outputs = OpGatherElements(node_unit);
  } else if (optype == "Resize") {
    aipu_outputs = OpResize(node_unit);
  } else if (optype == "GroupNormalization") {
    aipu_outputs = OpGroupNormalization(node_unit);
  } else if (optype == "BatchNormalization") {
    aipu_outputs = OpBatchNormalization(node_unit);
  } else if (optype == "Tile") {
    aipu_outputs = OpTile(node_unit);
  } else if (optype == "Expand") {
    aipu_outputs = OpExpand(node_unit);
  } else if (optype == "CumSum") {
    aipu_outputs = OpCumSum(node_unit);
  } else {
    ret = failed_ret;
  }

  if (aipu_outputs.size() == 0) {
    ret = failed_ret;
  }
  if (ret.IsOK()) {
    for (size_t i = 0; i < node_unit->Outputs().size(); ++i) {
      if (aipu_outputs[i] == nullptr) {
        ret = failed_ret;
        break;
      }
      tensor_maps_.emplace(std::piecewise_construct,
                           std::forward_as_tuple(node_unit->Outputs()[i].node_arg.Name()),
                           std::forward_as_tuple(aipu_outputs[i]));
    }
  }
  return ret;
}

TensorPtrList ZhouyiModel::OpAdd(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto output = aipubt::ops::add(input0, input1, aipubt::ops::Activation("NONE"), quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpSub(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::ops::Activation act("NONE");
  auto aipu_dtype = ParseDType(node_unit->Outputs()[0].node_arg.TypeAsProto());
  if (aipu_dtype == aipubt::Tensor::UINT8 ||
      aipu_dtype == aipubt::Tensor::UINT16 ||
      aipu_dtype == aipubt::Tensor::UINT32) {
    act = aipubt::ops::Activation("RELU");
  }
  auto output = aipubt::ops::sub(input0, input1, act, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpMul(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::ops::Activation act("NONE");
  auto output = aipubt::ops::mul(input0, input1, act, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpMax(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::ops::Activation act("NONE");
  auto output = aipubt::ops::max(input0, input1, act, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpMin(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::ops::Activation act("NONE");
  auto output = aipubt::ops::min(input0, input1, act, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpMatMul(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto output = aipubt::ops::matmul(input0, input1, false, false, 1.0, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpGemm(const NodeUnit* node_unit) {
  NodeAttrHelper node_helper(*node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  auto transA = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::TensorPtr output = nullptr;
  if (node_unit->Inputs().size() == 3) {
    auto input1 = AipuTensorMatch(node_unit->Inputs()[1], true);
    auto input2 = AipuTensorMatch(node_unit->Inputs()[2], true);
    aipubt::ops::Activation act("NONE");
    output = aipubt::ops::fully_connected(input0, input1, input2, act, quant);
  } else {
    auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
    quant.set_scale(quant.scale() / alpha);
    output = aipubt::ops::matmul(input0, input1, transA, transB, 1.0, quant);
  }
  return {output};
}

TensorPtrList ZhouyiModel::OpQGemm(const NodeUnit* node_unit) {
  // com.microsoft.QGemm
  // input size ==6  A,A_scale,A_zp, B,B_scale,B_zp,
  // input size > 6  A,A_scale,A_zp, B,B_scale,B_zp, C,y_scale,y_zp
  NodeAttrHelper node_helper(*node_unit);
  auto transA = node_helper.Get("transA", static_cast<int64_t>(0));
  auto transB = node_helper.Get("transB", static_cast<int64_t>(0));
  auto alpha = node_helper.Get("alpha", static_cast<float>(1.0f));
  float beta = 1.0;

  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  float a_scale = GetInitValue<float>(node_unit->Inputs()[1].node_arg);
  int32_t a_zp = GetInitValue<int32_t>(node_unit->Inputs()[2].node_arg);
  input0->quantization().set_scale(a_scale);
  input0->quantization().set_offset(a_zp);

  auto input1 = AipuTensorMatch(node_unit->Inputs()[3], true);
  float b_scale = GetInitValue<float>(node_unit->Inputs()[4].node_arg);
  int32_t b_zp = GetInitValue<int32_t>(node_unit->Inputs()[5].node_arg);
  input1->quantization().set_scale(b_scale);
  input1->quantization().set_offset(b_zp);
  aipubt::TensorPtr input2 = nullptr;
  // Optional input tensor C. If not specified, the computation is done as if C is a scalar 0.
  //  The shape of C should be unidirectional broadcastable to (M, N).
  // Its type is int32_t and must be quantized with zero_point = 0
  //  and scale = alpha / beta * a_scale * b_scale.
  if (node_unit->Inputs().size() > 6 &&
      !node_unit->Inputs()[6].node_arg.Name().empty()) {
    input2 = AipuTensorMatch(node_unit->Inputs()[6], true);
    // zero_point = 0 and scale = alpha / beta * a_scale * b_scale.
    input2->quantization().set_scale(alpha / beta * a_scale * b_scale);
    input2->quantization().set_offset(0);
  }

  auto quant = ParseQuantization(node_unit->Outputs()[0]);
  // has y_scale,y_zp
  if (node_unit->Inputs().size() > 7) {
    float y_scale = GetInitValue<float>(node_unit->Inputs()[7].node_arg);
    int32_t y_zp = GetInitValue<uint8_t>(node_unit->Inputs()[8].node_arg);

    quant.set_scale(y_scale);
    quant.set_offset(y_zp);
  }

  aipubt::TensorPtr output;
  if (input0->type() == aipubt::Tensor::FP16 || input0->type() == aipubt::Tensor::FP32) {
    output = aipubt::ops::gemm(input0, input1, input2, alpha, transA, transB, beta, quant);
    return {output};
  } else {
    if (node_unit->Inputs().size() == 9) {
      aipubt::ops::Activation act("NONE");
      output = aipubt::ops::fully_connected(input0, input1, input2, act, quant);
    } else {
      output = aipubt::ops::matmul(input0, input1, transA, transB, 1.0, quant);
    }
  }
  return {output};
}

TensorPtrList ZhouyiModel::OpSoftmax(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  NodeAttrHelper node_helper(*node_unit);
  auto axis = node_helper.Get("axis", -1);
  auto output = aipubt::ops::softmax(input0, axis, 1.0, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpLogSoftmax(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  NodeAttrHelper node_helper(*node_unit);
  auto axis = node_helper.Get("axis", -1);
  auto output = aipubt::ops::log_softmax(input0, axis, quant);
  output->node()->attrs()["min_compatible_zhouyi_target"] = aipubt::NodeParamValue("X3_1304");
  return {output};
}

TensorPtrList ZhouyiModel::OpCast(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto aipu_dtype = ParseDType(node_unit->Outputs()[0].node_arg.TypeAsProto());
  auto output = aipubt::ops::cast(input0, aipu_dtype, true, aipubt::ops::ClipMode::Truncation, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpAnd(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto output = aipubt::ops::logical_and(input0, input1);
  return {output};
}

TensorPtrList ZhouyiModel::OpOr(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto output = aipubt::ops::logical_or(input0, input1);
  return {output};
}

TensorPtrList ZhouyiModel::OpNot(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto output = aipubt::ops::logical_not(input0);
  return {output};
}

TensorPtrList ZhouyiModel::OpBinary(const NodeUnit* node_unit,
                                    OpBinaryFunc func) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto output = func(input0, input1, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpBitwise(const NodeUnit* node_unit,
                                     aipubt::ops::BitwiseMethod method) {
  TensorPtrList inputs;
  for (auto inp : node_unit->Inputs()) {
    auto input = AipuTensorMatch(inp);
    inputs.push_back(input);
  }
  auto quant = ParseQuantization(node_unit->Outputs()[0], inputs[0]->quantization());
  auto output = aipubt::ops::bitwise(inputs, method, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpPad(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  aipubt::ops::PadMode pad_mode = aipubt::ops::PadMode::Constant;
  NodeAttrHelper node_helper(*node_unit);
  std::string mode = node_helper.Get("mode", "constant");
  if ("constant" == mode) {
    pad_mode = aipubt::ops::PadMode::Constant;
  } else if ("reflect" == mode) {
    pad_mode = aipubt::ops::PadMode::Reflect;
  } else if ("edge" == mode) {
    pad_mode = aipubt::ops::PadMode::Edge;
  } else if ("wrap" == mode) {
    pad_mode = aipubt::ops::PadMode::Wrap;
  } else {
    LOGS(logger_, WARNING) << "Unsupport mode " << mode.data();
  }

  std::vector<std::pair<uint32_t, uint32_t>> pads;
  std::vector<uint32_t> pads_value;
  GetInitValueList<uint32_t>(node_unit->Inputs()[1].node_arg, pads_value);
  for (uint32_t i = 0; i < pads_value.size() / 2; ++i) {
    pads.push_back(std::make_pair(pads_value[i], pads_value[pads_value.size() / 2 + i]));
  }
  aipubt::TensorPtr output = nullptr;
  // Process optional input constant_value
  if (node_unit->Inputs().size() > 2) {
    int32_t value_type = node_unit->Inputs()[2].node_arg.TypeAsProto()->tensor_type().elem_type();
    if (value_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      const auto value_tensor = GetInitTensorProto(node_unit->Inputs()[2].node_arg.Name());
      Initializer unpacked_tensor(*value_tensor);
      float value_fp16 = unpacked_tensor.DataAsSpan<MLFloat16>()[0].ToFloat();
      output = aipubt::ops::pad(input0, pads, pad_mode, value_fp16);
    } else if (value_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      float constant_value = GetInitValue<float>(node_unit->Inputs()[2].node_arg);
      output = aipubt::ops::pad(input0, pads, pad_mode, constant_value);
    } else if (value_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      int32_t constant_value = GetInitValue<int32_t>(node_unit->Inputs()[2].node_arg);
      output = aipubt::ops::pad(input0, pads, pad_mode, constant_value);
    } else {
      int32_t constant_value = GetInitValue<int32_t>(node_unit->Inputs()[2].node_arg);
      output = aipubt::ops::pad(input0, pads, pad_mode, constant_value);
    }
  } else {
    output = aipubt::ops::pad(input0, pads, pad_mode);
  }

  return {output};
}

TensorPtrList ZhouyiModel::OpQuantizeLinear(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto aipu_dtype = ParseDType(node_unit->Outputs()[0].node_arg.TypeAsProto());
  auto output = aipubt::ops::quantize(
      input0, aipu_dtype, aipubt::ops::QuantizeRoundMode::RoundToEven, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpDequantizeLinear(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto input_dtype = ParseDType(node_unit->Inputs()[0].node_arg.TypeAsProto());
  auto aipu_dtype = ParseDType(node_unit->Outputs()[0].node_arg.TypeAsProto());
  aipubt::TensorPtr output = nullptr;
  if (input_dtype == aipubt::Tensor::FP16 && input_dtype == aipubt::Tensor::FP32) {
    output = aipubt::ops::cast(input0, aipu_dtype, true);
  } else {
    output = aipubt::ops::dequantize(input0, aipu_dtype, quant);
    output->node()->params()["is_perf_mode"] = false;
  }
  return {output};
}

TensorPtrList ZhouyiModel::OpConv(const NodeUnit* node_unit) {
  auto GetAutoPadding = [](std::vector<uint32_t>& pads,
                           const std::string& auto_pad,
                           std::string conv_type,
                           const std::array<uint32_t, 2>& strides,
                           const std::array<uint32_t, 2>& dilations,
                           const std::array<uint32_t, 2>& input_dims,
                           const std::array<uint32_t, 2>& filter_dims,
                           const std::array<uint32_t, 2>& output_dims) {
    constexpr size_t HEIGHT_IDX = 0;
    constexpr size_t WIDTH_IDX = 1;

    std::array<uint32_t, 2> total_padding = {};
    // dilated_filter_height = (shape(in[1])[height] - 1) * dilation[0] + 1
    // height_out = floor((pad_amount[0,0] + shape(in[0])[height] + pad_amount[0,1] - dilated_filter_height) / stride[0] + 1)
    //
    // Set total_height_padding equal to pad_amount[0,0] + pad_amount[0,1] and solve for it.
    uint32_t dilated_filter_height = (filter_dims[HEIGHT_IDX] - 1) * dilations[HEIGHT_IDX] + 1;
    total_padding[HEIGHT_IDX] = (output_dims[HEIGHT_IDX] - 1) * strides[HEIGHT_IDX] + dilated_filter_height - input_dims[HEIGHT_IDX];  // Total height padding

    // dilated_filter_width = (shape(in[1])[width] - 1) * dilation[1] + 1
    // width_out = floor((pad_amount[1,0] + shape(in[0])[width] + pad_amount[1,1] - dilated_filter_width) / stride[1] + 1)
    //
    // Set total_width_padding equal to pad_amount[1,0] + pad_amount[1,1] and solve for it.
    uint32_t dilated_filter_width = (filter_dims[WIDTH_IDX] - 1) * dilations[WIDTH_IDX] + 1;
    total_padding[WIDTH_IDX] = (output_dims[WIDTH_IDX] - 1) * strides[WIDTH_IDX] + dilated_filter_width - input_dims[WIDTH_IDX];  // Total width padding

    pads.resize(4);  // Make room.

    if (auto_pad == "SAME_UPPER") {
      pads[0] = total_padding[0] / 2;
      pads[1] = total_padding[1] / 2;
      pads[2] = total_padding[0] - pads[0];
      pads[3] = total_padding[1] - pads[1];
    } else if (auto_pad == "SAME_LOWER") {
      pads[2] = total_padding[0] / 2;
      pads[3] = total_padding[1] / 2;
      pads[0] = total_padding[0] - pads[2];
      pads[1] = total_padding[1] - pads[3];
    } else {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, FAIL,
          "ZHouyi: Cannot calculate auto-padding for unsupported auto_pad setting: ",
          auto_pad.c_str());
    }

    return Status::OK();
  };
  auto conv_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit) {
    NodeAttrHelper node_helper(*node_unit);
    const uint32_t group = node_helper.Get("group", static_cast<uint32_t>(1));

    // Strides parameter.
    auto strides = node_helper.Get("strides", std::vector<uint32_t>{1, 1});
    std::vector<int32_t> a_strides(strides.begin(), strides.end());
    // Dilations parameter
    std::vector<uint32_t> dilations = {1, 1};
    dilations = node_helper.Get("dilations", std::vector<uint32_t>{1, 1});
    std::vector<int32_t> a_dilations(dilations.begin(), dilations.end());
    // Pads attribute
    std::vector<uint32_t> pads =
        node_helper.Get("pads", std::vector<uint32_t>({0, 0, 0, 0}));
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    if (auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" &&
        auto_pad != "SAME_UPPER" && auto_pad != "VALID") {
      LOGS(logger_, WARNING) << "Unsupport pad: " << auto_pad << ", Use default: NOTSET";
      auto_pad = "NOTSET";
    }

    std::vector<int32_t> input_1_shape;  // NCHW
    if (!GetTensorDims(node_unit->Inputs()[1].node_arg, input_1_shape)) {
      LOGS(logger_, WARNING) << "Cannot get weight shape ";
    }

    if (auto_pad != "NOTSET" && auto_pad != "VALID") {
      std::vector<int32_t> input_0_shape;  // NCHW
      std::vector<int32_t> output_shape;
      if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_0_shape)) {
        LOGS(logger_, WARNING) << "Cannot get input shape ";
      }

      if (!GetTensorDims(node_unit->Outputs()[0].node_arg, output_shape)) {
        LOGS(logger_, WARNING) << "Cannot get input shape ";
      }
      const bool is_1d_conv = output_shape.size() == 3;
      std::array<uint32_t, 2> input_dims = {};
      std::array<uint32_t, 2> filter_dims = {};
      std::array<uint32_t, 2> output_dims = {};

      if (is_1d_conv) {
        input_dims[0] = input_0_shape[1];
        input_dims[1] = 1;

        filter_dims[0] = input_1_shape[2];
        filter_dims[1] = 1;

        output_dims[0] = output_shape[1];
        output_dims[1] = 1;
      } else {
        input_dims[0] = input_0_shape[2];
        input_dims[1] = input_0_shape[3];

        filter_dims[0] = input_1_shape[2];
        filter_dims[1] = input_1_shape[3];

        output_dims[0] = output_shape[2];
        output_dims[1] = output_shape[3];
      }
      auto ret = GetAutoPadding(
          pads, auto_pad, node_unit->OpType(), {strides[0], strides[1]},
          {dilations[0], dilations[1]}, input_dims, filter_dims, output_dims);
      if (!ret.IsOK()) {
        LOGS(logger_, WARNING) << "GetAutoPadding failed!";
      }
    }

    std::vector<int32_t> a_pads;
    for (size_t i = 0; i < pads.size() / 2; ++i) {
      a_pads.push_back(pads[i]);
      a_pads.push_back(pads[i + pads.size() / 2]);
    }

    // auto filter = AipuTensorMatch(node_unit->Inputs()[1], true, true);
    auto filter_quant = ParseQuantization(node_unit->Inputs()[1]);
    auto filter = CreateWeight(node_unit->Inputs()[1].node_arg);
    aipubt::TensorPtr bias = nullptr;
    if (node_unit->Inputs().size() > 2) {
      bias = AipuTensorMatch(node_unit->Inputs()[2], true);
    } else {
      // make vitrual bias
      int32_t bias_elements = filter->shape()[0];
      int32_t enforce_channels = filter_quant.scales().size();
      int32_t bias_size = bias_elements * 4;
      uint8_t* data = new uint8_t[bias_size];
      float bias_scale = 1.0;
      int32_t bias_zp = 0;
      memset(data, 0, bias_size);
      aipubt::Tensor::Dtype dtype = aipubt::Tensor::INT32;
      if (input0->type() == aipubt::Tensor::FP32 || input0->type() == aipubt::Tensor::FP16) {
        dtype = aipubt::Tensor::FP32;
      }
      aipubt::Quantization quant(bias_scale, bias_zp);
      std::vector<float> scales;
      std::vector<int32_t> offsets;
      scales.assign(enforce_channels, bias_scale);
      if (scales.size() > 0) {
        offsets.assign(scales.size(), 0);
        quant.set_scales(scales);
        quant.set_offsets(offsets);
        quant.set_group(scales.size());
      }
      quant.set_aipu_dtype(dtype);
      quant.set_data_unchanged(true);
      quant.set_unquantifiable(true);
      bias = aipubt::ops::make_bias(aipubt_graph_, data, bias_size, dtype, quant);
      delete[] data;
    }
    {
      std::vector<float> scales(filter_quant.scales());
      std::vector<int32_t> offsets(scales.size(), 0);

      std::for_each(scales.begin(), scales.end(),
                    [input0](float& scale) { return scale *= input0->quantization().scale(); });
      bias->quantization().set_scales(scales);
      bias->quantization().set_offsets(offsets);
    }

    auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
    aipubt::ops::Activation act("NONE");
    aipubt::TensorPtr output;
    if (group == 1) {
      output = aipubt::ops::conv2d(
          input0, filter, bias, a_strides, a_pads, act, a_dilations, quant);
    } else if (input_1_shape[1] == 1) {
      output = aipubt::ops::depthwise_conv2d(
          input0, filter, bias, a_strides, a_pads, act, a_dilations,
          quant, filter->quantization().quantization_dim());
    } else {
      output = aipubt::ops::conv_group(
          input0, filter, bias, a_strides, a_pads, act, a_dilations, quant);
    }
    return output;
  };
  auto output = NchwOpCreate(node_unit, conv_func);
  return {output};
}

TensorPtrList
ZhouyiModel::OpPool(const NodeUnit* node_unit) {
  auto HandleAutoPad = [](const std::vector<int32_t>& input_dimen,
                          const uint32_t weight_size_y,
                          const uint32_t weight_size_x,
                          const std::vector<uint32_t>& onnx_strides,
                          const std::vector<uint32_t>& onnx_dilations,
                          std::vector<uint32_t>& onnx_pads,
                          AutoPadType auto_pad_type) {
    if (auto_pad_type != AutoPadType::NOTSET) {
      const int32_t input_size_y = input_dimen[2];
      const int32_t input_size_x = input_dimen[3];
      const int32_t stride_y = onnx_strides[0];
      const int32_t stride_x = onnx_strides[1];
      const int32_t dilation_y = onnx_dilations[0];
      const int32_t dilation_x = onnx_dilations[1];

      int64_t padding_top = onnx_pads[0];
      int64_t padding_bottom = onnx_pads[2];
      int64_t padding_left = onnx_pads[1];
      int64_t padding_right = onnx_pads[3];

      ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                     stride_y, weight_size_y, dilation_y,
                                     auto_pad_type,
                                     padding_top, padding_bottom));
      ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                     stride_x, weight_size_x, dilation_x,
                                     auto_pad_type,
                                     padding_left, padding_right));

      onnx_pads = {static_cast<uint32_t>(padding_top), static_cast<uint32_t>(padding_left),
                   static_cast<uint32_t>(padding_bottom), static_cast<uint32_t>(padding_right)};
    }
    return Status::OK();
  };

  auto pool_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit_);
    const auto& op_type = node_unit_->OpType();
    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit_->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, WARNING) << "Cannot get input shape ";
    }

    std::vector<int32_t> stride = {1, 1};
    std::vector<int32_t> dilation = {1, 1};
    std::vector<int32_t> kernel = {(int32_t)input_shape[2], (int32_t)input_shape[3]};
    std::vector<int32_t> a_pads = {0, 0, 0, 0};
    bool ceil_mode = false;
    bool count_include_pad = false;
    // GlobalAveragePool and GlobalMaxPool, QLinearGlobalAveragePool no attribute
    bool is_attr_pool = op_type == "AveragePool" ||
                        op_type == "QLinearAveragePool" ||
                        op_type == "MaxPool";
    if (is_attr_pool) {
      // Strides parameter.
      auto strides = node_helper.Get("strides", std::vector<uint32_t>{1, 1});
      stride = {(int32_t)strides[0], (int32_t)strides[1]};
      auto kernel_shape = node_helper.Get("kernel_shape", std::vector<int32_t>{1, 1});
      kernel = {kernel_shape[0], kernel_shape[1]};
      ceil_mode = node_helper.Get("ceil_mode", 0);

      auto dilations = std::vector<uint32_t>{1, 1};
      // Unique attribute
      if (op_type == "MaxPool") {
        dilations = node_helper.Get("dilations", std::vector<uint32_t>{1, 1});
        dilation = {(int32_t)dilations[0], (int32_t)dilations[1]};
      }
      if ((op_type == "QLinearAveragePool") || (op_type == "AveragePool")) {
        // Whether include pad pixels when calculating values for the edges.
        //  Default is 0, doesn't count include pad.
        count_include_pad = node_helper.Get("count_include_pad", 0);
      }
      // Pads attribute
      auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
      const auto auto_pad_type = StringToAutoPadType(auto_pad);
      std::vector<uint32_t> pads = node_helper.Get("pads", std::vector<uint32_t>({0, 0, 0, 0}));
      auto ret = HandleAutoPad(input_shape, kernel_shape[0], kernel_shape[1],
                               strides, dilations, pads,
                               auto_pad_type);
      if (!ret.IsOK()) {
        LOGS(logger_, WARNING) << "HandleAutoPad failed.";
      }
      a_pads[0] = pads[0];  // padding_top
      a_pads[1] = pads[2];  // padding_bottom
      a_pads[2] = pads[1];  // padding_left
      a_pads[3] = pads[3];  // padding_right
    }

    auto quant = ParseQuantization(node_unit_->Outputs()[0], input0->quantization());
    aipubt::TensorPtr output = nullptr;

    if ((op_type == "MaxPool") || (op_type == "GlobalMaxPool")) {
      output = aipubt::ops::max_pool(input0, kernel, stride, dilation, a_pads, ceil_mode);
    } else {
      output = aipubt::ops::avg_pool(input0, kernel, stride, dilation, a_pads, ceil_mode,
                                     count_include_pad, quant);
    }
    return output;
  };

  if (node_unit->OpType() == "QLinearAveragePool") {
    // Works on NHWC layout or not? Default not.
    NodeAttrHelper node_helper(*node_unit);
    bool channels_last = node_helper.Get("channels_last", 0);
    if (channels_last) {
      auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
      auto output = pool_func(input0, node_unit);
      return {output};
    }
  }

  auto output = NchwOpCreate(node_unit, pool_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpUnary(const NodeUnit* node_unit, OpUnaryFunc func) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto output = func(input0, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpConcat(const NodeUnit* node_unit) {
  std::vector<aipubt::TensorPtr> inputs;
  for (auto inp : node_unit->Inputs()) {
    inputs.push_back(AipuTensorMatch(inp));
  }
  NodeAttrHelper node_helper(*node_unit);
  int32_t axis = node_helper.Get("axis", -1);
  auto quant = ParseQuantization(node_unit->Outputs()[0], inputs[0]->quantization());
  auto output = aipubt::ops::concat(inputs, axis, quant);

  return {output};
}

TensorPtrList ZhouyiModel::OpPrelu(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto input1 = AipuTensorMatch(node_unit->Inputs()[1], true);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto output = aipubt::ops::prelu(input0, input1, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpLeakyRelu(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  NodeAttrHelper node_helper(*node_unit);
  float alpha = node_helper.Get("alpha", 0.1f);
  auto output = aipubt::ops::leaky_relu(input0, alpha, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpClip(const NodeUnit* node_unit) {
  auto GetDataTypeLimites = [](int32_t onnx_data_type) {
    float min = std::numeric_limits<float>::lowest();
    float max = std::numeric_limits<float>::max();
    switch (onnx_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      min = static_cast<float>(std::numeric_limits<bool>::min());
      max = static_cast<float>(std::numeric_limits<bool>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      min = static_cast<float>(std::numeric_limits<int8_t>::min());
      max = static_cast<float>(std::numeric_limits<int8_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      min = static_cast<float>(std::numeric_limits<uint8_t>::min());
      max = static_cast<float>(std::numeric_limits<uint8_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      min = static_cast<float>(std::numeric_limits<int16_t>::min());
      max = static_cast<float>(std::numeric_limits<int16_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      min = static_cast<float>(std::numeric_limits<uint16_t>::min());
      max = static_cast<float>(std::numeric_limits<uint16_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      min = static_cast<float>(std::numeric_limits<int32_t>::min());
      max = static_cast<float>(std::numeric_limits<int32_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      min = static_cast<float>(std::numeric_limits<uint32_t>::min());
      max = static_cast<float>(std::numeric_limits<uint32_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      min = static_cast<float>(std::numeric_limits<int64_t>::min());
      max = static_cast<float>(std::numeric_limits<int64_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      min = static_cast<float>(std::numeric_limits<uint64_t>::min());
      max = static_cast<float>(std::numeric_limits<uint64_t>::max());
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      min = static_cast<float>(-65504);
      max = static_cast<float>(65504);
      break;
    default:;
    }
    return std::make_pair(min, max);
  };
  auto GetInitData = [&](float& value, int id) {
    if (IsInitTensors(node_unit->Inputs()[id].node_arg.Name())) {
      const auto value_tensor = GetInitTensorProto(node_unit->Inputs()[id].node_arg.Name());
      Initializer unpacked_tensor_min(*value_tensor);
      if (unpacked_tensor_min.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        float value_fp16 = unpacked_tensor_min.DataAsSpan<MLFloat16>()[0].ToFloat();
        value = value_fp16;
      } else {
        value = GetInitValue<float>(node_unit->Inputs()[id].node_arg);
      }
    }
  };

  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto elem_type = node_unit->Inputs()[0].node_arg.TypeAsProto()->tensor_type().elem_type();
  auto clip_min_max = GetDataTypeLimites(elem_type);
  float clip_min = clip_min_max.first;
  float clip_max = clip_min_max.second;
  if (node_unit->Inputs().size() > 1) {
    GetInitData(clip_min, 1);
  }
  if (node_unit->Inputs().size() > 2) {
    GetInitData(clip_max, 2);
  }
  auto output = aipubt::ops::clip(input0, clip_min, clip_max, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpReshape(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);

  std::vector<int32_t> new_shape;
  if (node_unit->Inputs().size() > 1) {
    GetInitValueList<int32_t>(node_unit->Inputs()[1].node_arg, new_shape);
  }
  std::vector<int32_t> dims;
  if (new_shape.size() != 0) {
    int num_negative_1 = 0;
    uint32_t new_shape_positive = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (new_shape[i] == -1) {
        ++num_negative_1;
      } else {
        new_shape_positive *= new_shape[i];
      }
    }
    if (num_negative_1 == 0) {
      dims.insert(dims.end(), new_shape.begin(), new_shape.end());
    } else if (num_negative_1 == 1) {
      for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
          dims.push_back(input0->shape().size() / new_shape_positive);
        } else {
          dims.push_back(new_shape[i]);
        }
      }
    }
  }

  if (dims.size() == 0) {
    if (!GetTensorDims(node_unit->Outputs()[0].node_arg, dims)) {
      LOGS(logger_, WARNING) << "Cannot get output shape ";
      return {};
    }
  }
  auto output = aipubt::ops::reshape(input0, dims);
  return {output};
}

TensorPtrList ZhouyiModel::OpFlatten(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  std::vector<int32_t> new_shape;
  std::vector<int32_t> output_shape;
  if (!GetTensorDims(node_unit->Outputs()[0].node_arg, output_shape)) {
    LOGS(logger_, WARNING) << "Cannot get output shape ";
  }

  std::vector<int32_t> input_shape;
  if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
    LOGS(logger_, WARNING) << "Cannot get input shape ";
  }
  if (output_shape.size() == 2) {
    for (int i = 0; i < 2; ++i) {
      new_shape.push_back(output_shape[i]);
    }
  } else {
    NodeAttrHelper node_helper(*node_unit);
    int32_t axis = node_helper.Get("axis", static_cast<int32_t>(1));
    if (axis < 0)
      axis += output_shape.size();
    int32_t o_shape_0 = 1;
    int32_t o_shape_1 = 1;
    if (axis == 0) {
      for (auto s : input_shape) {
        o_shape_1 *= s;
      }
    } else {
      for (int32_t i = 0; i < axis; ++i)
        o_shape_0 *= input_shape[i];
      for (size_t i = axis; i < input_shape.size(); ++i)
        o_shape_1 *= input_shape[i];
    }
    new_shape.push_back(o_shape_0);
    new_shape.push_back(o_shape_1);
  }
  auto output = aipubt::ops::reshape(input0, new_shape);
  return {output};
}

TensorPtrList ZhouyiModel::OpSqueeze(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  std::vector<int32_t> axis;
  if (node_unit->Inputs().size() > 1) {
    GetInitValueList<int32_t>(node_unit->Inputs()[1].node_arg, axis);
  }
  auto output = aipubt::ops::squeeze(input0, axis);
  return {output};
}

TensorPtrList ZhouyiModel::OpUnSqueeze(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  std::vector<int32_t> axis;
  if (node_unit->Inputs().size() > 1) {
    GetInitValueList<int32_t>(node_unit->Inputs()[1].node_arg, axis);
  }
  std::vector<int32_t> new_shape(input0->shape().begin(), input0->shape().end());
  for (auto ax : axis) {
    if (ax < static_cast<int32_t>(new_shape.size())) {
      new_shape.insert(new_shape.begin() + ax, 1);
    } else {
      new_shape.insert(new_shape.end(), 1);
    }
  }
  auto output = aipubt::ops::reshape(input0, new_shape);
  return {output};
}

TensorPtrList ZhouyiModel::OpSpaceToDepth(const NodeUnit* node_unit) {
  auto s2d_func = [](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit_);
    auto block_size = node_helper.Get("blocksize", 1);
    auto output = aipubt::ops::space_to_depth(input0, block_size);
    return output;
  };
  auto output = NchwOpCreate(node_unit, s2d_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpDepthToSpace(const NodeUnit* node_unit) {
  auto d2s_func = [](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit_);
    auto block_size = node_helper.Get("blocksize", 1);
    auto mode = node_helper.Get("mode", std::string("DCR"));
    aipubt::ops::DepthToSpaceMode a_mode = aipubt::ops::DepthToSpaceMode::DCR;
    if (mode == "DCR") {
      a_mode = aipubt::ops::DepthToSpaceMode::DCR;
    } else {
      a_mode = aipubt::ops::DepthToSpaceMode::CRD;
    }
    auto output = aipubt::ops::depth_to_space(input0, block_size, a_mode);
    return output;
  };
  auto output = NchwOpCreate(node_unit, d2s_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpTranspose(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  NodeAttrHelper node_helper(*node_unit);
  std::vector<uint32_t> perm = node_helper.Get("perm", std::vector<uint32_t>({}));
  if (perm.size() == 0) {
    LOGS(logger_, VERBOSE) << "The perm size is 0, by default, reverse the dimensions.";
    const auto& input = node_unit->Inputs()[0];
    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, WARNING) << "Cannot get input shape ";
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
      perm.push_back(input_shape.size() - i - 1);
    }
  }
  auto output = aipubt::ops::transpose(input0, perm);
  return {output};
}

TensorPtrList ZhouyiModel::OpSplit(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);

  NodeAttrHelper node_helper(*node_unit);
  auto axis = node_helper.Get("axis", 0);
  std::vector<uint32_t> splits_size;
  if (node_unit->Inputs().size() > 1) {
    GetInitValueList(node_unit->Inputs()[1].node_arg, splits_size);
  } else {
    uint32_t splits_num_default = node_unit->Outputs().size();
    uint32_t splits_num = node_helper.Get("num_outputs", splits_num_default);

    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, WARNING) << "Cannot get input shape ";
    }
    if (axis < 0) {
      axis = axis + input_shape.size();
    }
    uint32_t dim_axis = std::ceil(float(input_shape[axis]) / splits_num);
    uint32_t rest_dim = input_shape[axis];
    splits_size.resize(splits_num);
    for (uint32_t i = 0; i < splits_num; ++i) {
      if (rest_dim < dim_axis) {
        splits_size[i] = rest_dim;
      } else {
        splits_size[i] = dim_axis;
      }
      rest_dim = rest_dim - dim_axis;
    }
  }

  auto output = aipubt::ops::split(input0, splits_size, axis);
  return output;
}

TensorPtrList ZhouyiModel::OpSlice(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);

  std::vector<int32_t> input_shape;
  if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
    LOGS(logger_, WARNING) << "Cannot get input shape ";
  }
  std::vector<int32_t> begin(input_shape.size());
  std::vector<int32_t> end(input_shape.size());
  std::vector<int32_t> strides(input_shape.size());
  for (size_t i = 0; i < input_shape.size(); ++i) {
    begin[i] = 0;
    end[i] = input_shape[i];
    strides[i] = 1;
  }

  if (node_unit->SinceVersion() < 10) {
    std::vector<int64_t> axes(input_shape.size());
    NodeAttrHelper node_helper(*node_unit);
    auto starts = node_helper.Get("starts", std::vector<int64_t>{0});
    auto ends = node_helper.Get("ends", std::vector<int64_t>{0});

    uint32_t valid_num = starts.size();
    if (node_helper.HasAttr("axes")) {
      axes = node_helper.Get("axes", std::vector<int64_t>{0});
    } else {
      for (uint32_t i = 0; i < valid_num; ++i) {
        axes[i] = i;
      }
    }
    for (uint32_t i = 0; i < valid_num; ++i) {
      int64_t index = axes[i];
      begin[index] = starts[index];
      end[index] = ends[index];
    }
  } else {
    constexpr size_t starts_index = 1;
    constexpr size_t ends_index = 2;
    constexpr size_t axes_index = 3;
    constexpr size_t steps_index = 4;
    // Starts input (required).
    std::vector<int64_t> starts_value;
    std::vector<int64_t> ends_value;
    std::vector<int64_t> axes_value;
    std::vector<int64_t> steps_value;

    // Starts input (required).
    GetInitValueList(node_unit->Inputs()[starts_index].node_arg, starts_value);
    uint32_t valid_num = starts_value.size();
    // Ends input (required).
    GetInitValueList(node_unit->Inputs()[ends_index].node_arg, ends_value);

    // Axes input (optional).
    if (node_unit->Inputs().size() > axes_index &&
        !node_unit->Inputs()[axes_index].node_arg.Name().empty()) {
      GetInitValueList(node_unit->Inputs()[axes_index].node_arg, axes_value);
    }
    for (auto& axis : axes_value) {
      if (axis < 0) {
        axis += input_shape.size();
      }
    }

    if (axes_value.size() == 0) {
      for (uint32_t i = 0; i < valid_num; ++i) {
        axes_value.push_back(i);
      }
    }

    // Steps input (optional).
    if (node_unit->Inputs().size() > steps_index &&
        !node_unit->Inputs()[steps_index].node_arg.Name().empty()) {
      GetInitValueList(node_unit->Inputs()[steps_index].node_arg, steps_value);
    }
    if (steps_value.size() == 0) {
      for (uint32_t i = 0; i < valid_num; ++i) {
        steps_value.push_back(1);
      }
    }

    for (uint32_t i = 0; i < valid_num; ++i) {
      if (starts_value[i] < 0) {
        starts_value[i] = starts_value[i] + input_shape[axes_value[i]];
      }
      if (starts_value[i] > input_shape[axes_value[i]]) {
        starts_value[i] = input_shape[axes_value[i]];
      }
    }
    for (uint32_t i = 0; i < valid_num; ++i) {
      if (ends_value[i] < 0) {
        ends_value[i] = ends_value[i] + input_shape[axes_value[i]];
      }
      if (ends_value[i] > input_shape[axes_value[i]]) {
        ends_value[i] = input_shape[axes_value[i]];
      }
    }

    for (uint32_t i = 0; i < valid_num; ++i) {
      int64_t index = axes_value[i];
      begin[index] = starts_value[i];
      end[index] = ends_value[i];
      strides[index] = steps_value[i];
    }
  }
  auto output = aipubt::ops::slice(input0, begin, end, strides);
  return {output};
}

TensorPtrList ZhouyiModel::OpReduce(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  auto optype = node_unit->OpType();
  aipubt::ops::ReduceMethod method = aipubt::ops::ReduceMethod::All;
  if (optype == "ReduceMax") {
    method = aipubt::ops::ReduceMethod::Max;
  } else if (optype == "ReduceMin") {
    method = aipubt::ops::ReduceMethod::Min;
  } else if (optype == "ReduceMean") {
    method = aipubt::ops::ReduceMethod::Mean;
  } else if (optype == "ReduceProd") {
    method = aipubt::ops::ReduceMethod::Prod;
  } else if (optype == "ReduceSum") {
    method = aipubt::ops::ReduceMethod::Sum;
  }
  NodeAttrHelper node_helper(*node_unit);
  auto keepdims = node_helper.Get("keepdims", 1);
  std::vector<int32_t> axis;
  bool axes_in_attr = false;
  if (optype == "ReduceMax" || optype == "ReduceMin" ||
      optype == "ReduceMean" || optype == "ReduceProd") {
    if (node_unit->SinceVersion() < 18) {
      axes_in_attr = true;
    }
  } else if (optype == "ReduceSum") {
    if (node_unit->SinceVersion() < 13) {
      axes_in_attr = true;
    }
  }
  if (axes_in_attr) {
    axis = node_helper.Get("axes", std::vector<int32_t>({}));
  } else {
    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, WARNING) << "Cannot get input shape ";
    }
    if (node_unit->Inputs().size() > 1) {
      std::vector<int64_t> axes_value;
      GetInitValueList(node_unit->Inputs()[1].node_arg, axes_value);
      if (axes_value.size() > 0) {
        for (size_t i = 0; i < axes_value.size(); ++i) {
          if (axes_value[i] < 0) {
            axis.push_back(axes_value[i] + input_shape.size());
          } else {
            axis.push_back(axes_value[i]);
          }
        }
      } else {
        for (size_t i = 0; i < input_shape.size(); ++i) {
          axis.push_back(i);
        }
      }
    } else {
      for (size_t i = 0; i < input_shape.size(); ++i) {
        axis.push_back(i);
      }
    }
  }
  auto output = aipubt::ops::reduce(input0, method, axis, keepdims, quant);
  return {output};
}

TensorPtrList ZhouyiModel::OpLRN(const NodeUnit* node_unit) {
  auto lrn_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit_);

    auto alpha = node_helper.Get("alpha", 0.0001f);
    auto beta = node_helper.Get("beta", 0.75f);
    auto bias = node_helper.Get("bias", 1.0f);
    auto size = node_helper.Get("size", 1);
    auto quant = ParseQuantization(node_unit_->Outputs()[0], input0->quantization());
    quant.set_lut_items_bits(12);
    auto output = aipubt::ops::lrn(
        input0, size, aipubt::ops::LrnMethod::AcrossChannels, alpha, beta, bias, quant);
    return output;
  };
  auto output = NchwOpCreate(node_unit, lrn_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpGather(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto indices = AipuTensorMatch(node_unit->Inputs()[1]);
  NodeAttrHelper node_helper(*node_unit);
  int32_t batch_dims = 0;
  int32_t axis = 0;
  if (node_helper.HasAttr("batch_dims")) {
    batch_dims = node_helper.Get("batch_dims", 0);
  }
  if (node_helper.HasAttr("axis")) {
    axis = node_helper.Get("axis", 0);
  }
  auto output = aipubt::ops::gather(input0, indices, axis, batch_dims);
  return {output};
}

TensorPtrList ZhouyiModel::OpGatherND(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto indices = AipuTensorMatch(node_unit->Inputs()[1]);
  NodeAttrHelper node_helper(*node_unit);
  int32_t batch_dims = 0;
  if (node_helper.HasAttr("batch_dims")) {
    batch_dims = node_helper.Get("batch_dims", 0);
  }
  auto output = aipubt::ops::gather_nd(input0, indices, batch_dims);
  return {output};
}

TensorPtrList ZhouyiModel::OpGatherElements(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto indices = AipuTensorMatch(node_unit->Inputs()[1]);
  NodeAttrHelper node_helper(*node_unit);
  int32_t axis = 0;
  if (node_helper.HasAttr("axis")) {
    axis = node_helper.Get("axis", 0);
  }
  auto output = aipubt::ops::gather_elements(input0, indices, axis);
  return {output};
}

TensorPtrList ZhouyiModel::OpResize(const NodeUnit* node_unit) {
  auto resize_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit_);
    std::string mode = node_helper.Get("mode", std::string("nearest"));
    std::string ct_mode =
        node_helper.Get("coordinate_transformation_mode", std::string("half_pixel"));
    std::string nearest_mode = node_helper.Get("nearest_mode", std::string("round_prefer_floor"));

    aipubt::ops::ResizeMethod method = aipubt::ops::ResizeMethod::NearestNeighbor;
    if (supported_modes.count(mode)) {
      method = supported_modes[mode];
    }

    aipubt::ops::ResizeNearestMode n_mode = aipubt::ops::ResizeNearestMode::RoundPreferFloor;
    if (supported_nearest_modes.count(nearest_mode)) {
      n_mode = supported_nearest_modes[nearest_mode];
    }

    aipubt::ops::ResizeMode resize_mode = aipubt::ops::ResizeMode::HalfPixel;
    if (supported_ct_modes.count(ct_mode)) {
      resize_mode = supported_ct_modes[ct_mode];
    }

    aipubt::TensorPtr output = nullptr;
    if (node_unit_->Inputs().size() == 4) {
      std::vector<int64_t> sizes_value;
      GetInitValueList(node_unit_->Inputs()[3].node_arg, sizes_value);
      std::vector<uint32_t> pooled_shape =
          {static_cast<uint32_t>(sizes_value[2]), static_cast<uint32_t>(sizes_value[3])};
      output = aipubt::ops::resize(input0, pooled_shape, {}, method, resize_mode, n_mode);
    } else if (node_unit_->Inputs().size() == 3) {
      std::vector<float> scales_value;
      GetInitValueList(node_unit_->Inputs()[2].node_arg, scales_value);
      std::vector<float> pooled_scale = {scales_value[2], scales_value[3]};
      output = aipubt::ops::resize(input0, {}, pooled_scale, method, resize_mode, n_mode);
    }
    return output;
  };
  auto output = NchwOpCreate(node_unit, resize_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpGroupNormalization(const NodeUnit* node_unit) {
  auto gn_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    auto gamma = AipuTensorMatch(node_unit_->Inputs()[1], true);
    auto beta = AipuTensorMatch(node_unit_->Inputs()[2], true);
    auto quant = ParseQuantization(node_unit_->Outputs()[0], input0->quantization());
    NodeAttrHelper node_helper(*node_unit_);
    const auto epsilon = node_helper.Get("epsilon", 1e-05f);
    const auto num_groups = node_helper.Get("num_groups", 1);
    auto output = aipubt::ops::group_normalization(
        input0, gamma, beta, -1, num_groups, epsilon, quant);
    return output;
  };
  auto output = NchwOpCreate(node_unit, gn_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpBatchNormalization(const NodeUnit* node_unit) {
  auto bn_func = [&](aipubt::TensorPtr input0, const NodeUnit* node_unit_) {
    NodeAttrHelper node_helper(*node_unit);
    const auto epsilon = node_helper.Get("epsilon", 1e-05f);

    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, WARNING) << "Cannot get input shape ";
    }
    int32_t axis = input_shape.size() - 1;
    if (input_shape.size() == 3) {
      axis = 1;
    }

    auto gamma_beta_to_weight_bias = [&]() {
      auto gamma = AipuTensorMatch(node_unit->Inputs()[1], true);
      auto beta = AipuTensorMatch(node_unit->Inputs()[2], true);
      auto mean = AipuTensorMatch(node_unit->Inputs()[3], true);
      auto variance = AipuTensorMatch(node_unit->Inputs()[4], true);
      auto input_quant = input0->quantization();
      auto gamma_quant = gamma->quantization();
      auto beta_quant = beta->quantization();
      auto gamma_ct = aipubt::ComputeTensor(gamma).astype(aipubt::Tensor::FP32);
      auto beta_ct = aipubt::ComputeTensor(beta).astype(aipubt::Tensor::FP32);

      gamma_ct = (gamma_ct - gamma_quant.offset()) * gamma_quant.scale();
      beta_ct = (beta_ct - beta_quant.offset()) * beta_quant.scale();

      int32_t input_axis_dim = input_shape[axis];
      std::vector<float> mean_data(input_axis_dim, 0);
      auto mean_ct = aipubt::ComputeTensor(mean_data).astype(aipubt::Tensor::FP32);
      auto mean_quant = mean->quantization();
      mean_ct = aipubt::ComputeTensor(mean).astype(aipubt::Tensor::FP32);
      mean_ct = (mean_ct - mean_quant.offset()) * mean_quant.scale();
      std::vector<float> var_data(input_axis_dim, 1.0);
      auto variance_ct = aipubt::ComputeTensor(var_data).astype(aipubt::Tensor::FP32);
      auto variance_quant = variance->quantization();
      variance_ct = aipubt::ComputeTensor(variance).astype(aipubt::Tensor::FP32);
      variance_ct = (variance_ct - variance_quant.offset()) * variance_quant.scale();

      auto weight_data_float = variance_ct + epsilon;
      weight_data_float = weight_data_float.sqrt();
      weight_data_float = gamma_ct / weight_data_float;
      auto bias_data_float = mean_ct * weight_data_float;
      bias_data_float = beta_ct - bias_data_float;
      aipubt::TensorPtr weights, biases;
      auto aipu_dtype = ParseDType(node_unit->Inputs()[0].node_arg.TypeAsProto());
      if (aipu_dtype == aipubt::Tensor::UINT16 ||
          aipu_dtype == aipubt::Tensor::UINT8 ||
          aipu_dtype == aipubt::Tensor::INT16 ||
          aipu_dtype == aipubt::Tensor::INT8) {
        float weight_data_f_min = weight_data_float->min<float>();
        float weight_data_f_max = weight_data_float->max<float>();
        float scale = std::max(std::abs(weight_data_f_min), std::abs(weight_data_f_max));
        scale = scale / std::numeric_limits<int16_t>::max();
        auto weight_data_quant = (weight_data_float / scale);
        weight_data_quant.round();

        auto bias_data_quant = (bias_data_float / (scale * input_quant.scale()));
        bias_data_quant.round();
        weights = weight_data_quant.astype(aipubt::Tensor::INT16);
        weights->quantization() = aipubt::Quantization(scale, 0);
        biases = bias_data_quant.astype(aipubt::Tensor::INT32);
        biases->quantization() = aipubt::Quantization(scale * input_quant.scale(), 0);
      } else {
        weights = weight_data_float.astype(aipubt::Tensor::FP32);
        weights->quantization() = aipubt::Quantization(1.0, 0);
        biases = bias_data_float.astype(aipubt::Tensor::FP32);
        biases->quantization() = aipubt::Quantization(1.0, 0);
      }
      return std::make_pair(weights, biases);
    };

    aipubt::TensorPtr weight = nullptr;
    aipubt::TensorPtr bias = nullptr;
    auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
    if (node_unit->Inputs().size() == 3) {
      auto weight = AipuTensorMatch(node_unit->Inputs()[1], true);
      auto bias = AipuTensorMatch(node_unit->Inputs()[2], true);

    } else if (node_unit->Inputs().size() == 5) {
      auto weight_bias = gamma_beta_to_weight_bias();
      weight = weight_bias.first;
      bias = weight_bias.second;
    }
    auto output = aipubt::ops::batch_normalization(
        input0, weight, bias, axis, epsilon, quant);
    return output;
  };
  auto output = NchwOpCreate(node_unit, bn_func);
  return {output};
}

TensorPtrList ZhouyiModel::OpTile(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  std::vector<int64_t> values;
  GetInitValueList(node_unit->Inputs()[1].node_arg, values);

  std::vector<uint32_t> repeats(values.begin(), values.end());
  auto output = aipubt::ops::tile(input0, repeats);
  return {output};
}

TensorPtrList ZhouyiModel::OpExpand(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);

  std::vector<int32_t> input_shape;
  if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
    LOGS(logger_, WARNING) << "Cannot get input shape ";
  }
  std::vector<int64_t> values;
  GetInitValueList(node_unit->Inputs()[1].node_arg, values);
  std::vector<uint32_t> repeats(values.begin(), values.end());

  for (size_t i = 0; i < repeats.size(); ++i) {
    if (i >= input_shape.size())
      break;
    uint32_t last_repeat_id = repeats.size() - i - 1;
    uint32_t last_input_id = input_shape.size() - i - 1;
    if (repeats[last_repeat_id] != 1) {
      if (input_shape[last_input_id] == 1) {
        continue;
      }
      if (static_cast<int32_t>(repeats[last_repeat_id]) == input_shape[last_input_id]) {
        repeats[last_repeat_id] = 1;
      } else {
        LOGS(logger_, ERROR)
            << "Two corresponding dimensions must have the same value,"
            << "or one of them is equal to 1.";
        return {};
      }
    }
  }
  uint32_t input_dim_size = input_shape.size();
  if (repeats.size() < input_dim_size) {
    repeats.insert(repeats.begin(), input_dim_size - repeats.size(), 1);
  } else if (repeats.size() > input_dim_size) {
    std::vector<int32_t> shape;
    shape.insert(shape.begin(), input0->shape().begin(),
                 input0->shape().end());
    shape.insert(shape.begin(), repeats.size() - input_dim_size, 1);
    input0 = aipubt::ops::reshape(input0, shape);
  }
  auto output = aipubt::ops::tile(input0, repeats);
  return {output};
}

TensorPtrList ZhouyiModel::OpCumSum(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  NodeAttrHelper node_helper(*node_unit);
  auto exclusive = node_helper.Get("exclusive", 0);
  auto reverse = node_helper.Get("reverse", 0);
  auto axis_type = node_unit->Inputs()[1].node_arg.TypeAsProto()->tensor_type().elem_type();
  int32_t axis = GetInitValue<int32_t>(node_unit->Inputs()[1].node_arg);
  auto output = aipubt::ops::cumulate(
      input0, aipubt::ops::CumulateMethod::SUM, axis, exclusive, reverse);
  return {output};
}

TensorPtrList ZhouyiModel::OpErf(const NodeUnit* node_unit) {
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  auto quant = ParseQuantization(node_unit->Outputs()[0], input0->quantization());
  aipubt::TensorPtr output = nullptr;
  float float_input_max = 3.0;
  if (input0->type() == aipubt::Tensor::UINT16) {
    int i8_max = 256;
    if (i8_max * input0->quantization().scale() > float_input_max) {
      output = aipubt::ops::cast(input0, aipubt::Tensor::INT8, false, aipubt::ops::ClipMode::Saturation, quant);
      output = aipubt::ops::erf(output, quant);
      output = aipubt::ops::cast(output, input0->type(), false, aipubt::ops::ClipMode::Saturation, quant);
    }
  } else if (input0->type() == aipubt::Tensor::INT16) {
    int i8_max = 128;
    if (i8_max * input0->quantization().scale() > float_input_max) {
      output = aipubt::ops::cast(input0, aipubt::Tensor::INT8, false, aipubt::ops::ClipMode::Saturation, quant);
      output = aipubt::ops::erf(output, quant);
      output = aipubt::ops::cast(output, input0->type(), false, aipubt::ops::ClipMode::Saturation, quant);
    }
  }
  if (output == nullptr) {
    output = aipubt::ops::erf(input0, quant);
  }
  return {output};
}

aipubt::TensorPtr ZhouyiModel::AipuTensorMatch(const NodeArg* node_arg) {
  auto name = node_arg->Name();
  if (tensor_maps_.count(name)) {
    return tensor_maps_[name];
  } else {
    LOGS(logger_, ERROR) << name << " not Find! ";
    return nullptr;
  }
}

aipubt::TensorPtr ZhouyiModel::AipuTensorMatch(const NodeUnitIODef& io_def,
                                               bool constant_tensor) {
  auto name = io_def.node_arg.Name();
  if (tensor_maps_.count(name)) {
    return tensor_maps_[name];
  } else if (IsInitTensors(name)) {
    if (constant_tensor) {
      return ConstantTensor(io_def.node_arg);
    } else {
      return OpConstant(io_def.node_arg);
    }
  } else {
    LOGS(logger_, WARNING) << name << " not create ";
    return nullptr;
  }
}

aipubt::TensorPtr ZhouyiModel::ConstantTensor(const NodeArg& node_arg) {
  auto aipu_dtype = ParseDType(node_arg.TypeAsProto());
  std::vector<int32_t> dims;
  if (!GetTensorDims(node_arg, dims)) {
    LOGS(logger_, WARNING) << "Cannot get shape ";
  }
  if (dims.size() == 0) {
    dims.push_back(1);
  }
  aipubt::TensorShape shape(dims);
  auto quant = GetQuant(node_arg.Name());
  quant.set_data_unchanged(true);
  aipubt::DataLayout layout = (dims.size() == 4)
                                  ? aipubt::DataLayout::DataLayout_NHWC
                                  : aipubt::DataLayout::DataLayout_Unkown;
  shape.set_layout(layout);
  std::vector<uint8_t> unpacked_tensor;
  const auto value_tensor = GetInitTensorProto(node_arg.Name());
  if (!UnpackInitializerData(*value_tensor, unpacked_tensor).IsOK()) {
    LOGS(logger_, ERROR) << "Cannot get Constant data.";
  }

  auto output = aipubt::ops::make_tensor(
      aipubt_graph_, unpacked_tensor.data(), shape, aipu_dtype, quant,
      aipubt::DataLayout::DataLayout_NHWC, node_arg.Name());
  return output;
}

namespace {
template <typename T1>
aipubt::TensorPtr layout_transpose(aipubt::TensorPtr tensor, std::vector<int>& perm,
                                   aipubt::Tensor::Dtype dtype) {
  aipubt::ComputeTensor cp_output = aipubt::ComputeTensor::create(tensor);
  cp_output = (cp_output - tensor->quantization().offset()) * tensor->quantization().scale();
  float weight_data_f_min = cp_output->min<float>();
  float weight_data_f_max = cp_output->max<float>();
  float scale = std::max(std::abs(weight_data_f_min), std::abs(weight_data_f_max));
  scale = scale / std::numeric_limits<T1>::max();
  auto weight_data_quant = (cp_output / scale);
  weight_data_quant = weight_data_quant.round();
  weight_data_quant = weight_data_quant.permute(perm);
  aipubt::TensorPtr output = weight_data_quant.astype(dtype);
  output->quantization() = aipubt::Quantization(scale, 0);
  output->quantization().set_aipu_dtype(dtype);
  return output;
};
}  // namespace

aipubt::TensorPtr ZhouyiModel::CreateWeight(const NodeArg& node_arg) {
  auto aipu_dtype = ParseDType(node_arg.TypeAsProto());
  std::vector<int32_t> dims;
  if (!GetTensorDims(node_arg, dims)) {
    LOGS(logger_, WARNING) << "Cannot get shape ";
  }
  if (dims.size() == 0) {
    dims.push_back(1);
  }
  aipubt::TensorShape shape(dims);
  auto quant = GetQuant(node_arg.Name());
  std::vector<uint8_t> unpacked_tensor;
  const auto value_tensor = GetInitTensorProto(node_arg.Name());
  if (!UnpackInitializerData(*value_tensor, unpacked_tensor).IsOK()) {
    LOGS(logger_, ERROR) << "Cannot get Constant data.";
  }
  auto output = aipubt::ops::make_tensor(
      aipubt_graph_, unpacked_tensor.data(), shape, aipu_dtype, quant,
      aipubt::DataLayout::DataLayout_NHWC, node_arg.Name());
  std::vector<int> perm = {0, 2, 3, 1};

  // NCHW to NHWC
  if (aipu_dtype == aipubt::Tensor::UINT8) {
    output = layout_transpose<int8_t>(output, perm, aipubt::Tensor::INT8);
  } else if (aipu_dtype == aipubt::Tensor::UINT16) {
    output = layout_transpose<int16_t>(output, perm, aipubt::Tensor::INT16);
  } else if (aipu_dtype == aipubt::Tensor::UINT32) {
    output = layout_transpose<int32_t>(output, perm, aipubt::Tensor::INT32);
  } else {
    aipubt::ComputeTensor cp_output = aipubt::ComputeTensor::create(output);
    output = cp_output.permute(perm);
    output->quantization() = quant;
  }
  output->quantization().set_data_unchanged(true);
  return output;
}

aipubt::TensorPtr ZhouyiModel::OpConstant(const NodeArg& node_arg) {
  auto aipu_dtype = ParseDType(node_arg.TypeAsProto());
  auto shape = ParseShape(node_arg);
  auto quant = GetQuant(node_arg.Name());
  quant.set_data_unchanged(true);
  std::vector<uint8_t> unpacked_tensor;
  const auto value_tensor = GetInitTensorProto(node_arg.Name());
  if (!UnpackInitializerData(*value_tensor, unpacked_tensor).IsOK()) {
    LOGS(logger_, ERROR) << "Cannot get Constant data.";
  }
  void* data_p = unpacked_tensor.data();
  std::vector<uint8_t> unpacked_tensor_int32;

  // int64 to int32
  if (node_arg.TypeAsProto()->tensor_type().elem_type() ==
      ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    size_t data_size = unpacked_tensor.size() / 8;
    unpacked_tensor_int32.resize(data_size * 4);
    for (size_t i = 0; i < data_size; ++i) {
      int64_t* indices_value_i64 = reinterpret_cast<int64_t*>(&unpacked_tensor[i * 8]);
      int32_t* indices_value_i32 = reinterpret_cast<int32_t*>(&unpacked_tensor_int32[i * 4]);
      *indices_value_i32 = *indices_value_i64;
    }
    aipu_dtype = aipubt::Tensor::INT32;
    data_p = unpacked_tensor_int32.data();
  }
  auto output = aipubt::ops::constant(aipubt_graph_, data_p, shape, aipu_dtype, quant);
  tensor_maps_.emplace(std::piecewise_construct, std::forward_as_tuple(node_arg.Name()),
                       std::forward_as_tuple(output));
  return output;
}

aipubt::Tensor::Dtype
ZhouyiModel::ParseDType(const ONNX_NAMESPACE::TypeProto* proto) const {
  int32_t data_type = proto->tensor_type().elem_type();
  std::unordered_map<int32_t, aipubt::Tensor::Dtype> dtype_maps = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, aipubt::Tensor::INT8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, aipubt::Tensor::INT16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, aipubt::Tensor::INT32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, aipubt::Tensor::INT64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, aipubt::Tensor::UINT8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, aipubt::Tensor::UINT16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, aipubt::Tensor::UINT32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, aipubt::Tensor::UINT64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, aipubt::Tensor::FP16},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, aipubt::Tensor::FP32},
      {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, aipubt::Tensor::BFP16},
      {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, aipubt::Tensor::FP64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, aipubt::Tensor::FP8E5M2},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, aipubt::Tensor::FP8E4M3FN},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, aipubt::Tensor::BOOL},
  };

  if (dtype_maps.count(data_type)) {
    return dtype_maps[data_type];
  } else {
    LOGS(logger_, WARNING) << "Unsupport TensorProto DataType: " << data_type
                           << ". Set default value: aipubt::Tensor::UINT8";
    return aipubt::Tensor::UINT8;
  }
}

aipubt::TensorShape ZhouyiModel::ParseShape(const NodeArg& node_arg) {
  std::vector<int32_t> dims;
  if (!GetTensorDims(node_arg, dims)) {
    LOGS(logger_, WARNING) << "Cannot get shape ";
  }
  // For Scalar data, we need to set shape to 1 for AIPU
  if (dims.size() < 1) {
    dims.push_back(1);
  }

  aipubt::TensorShape shape(dims);
  aipubt::DataLayout layout = (dims.size() == 4)
                                  ? aipubt::DataLayout::DataLayout_NHWC
                                  : aipubt::DataLayout::DataLayout_Unkown;
  shape.set_layout(layout);
  return shape;
}

const ONNX_NAMESPACE::TensorProto* ZhouyiModel::GetInitTensorProto(std::string name) const {
  const auto& graph_initializers = graph_viewer_->GetAllInitializedTensors();
  auto it = graph_initializers.find(name);
  if (it == graph_initializers.end()) {
    LOGS(logger_, ERROR) << "Not able to find initializer: " << name;
    return nullptr;
  }
  return it->second;
}

int32_t ZhouyiModel::ParseOffset(const NodeArg* node_arg) const {
  if (node_arg == nullptr)
    return 0;
  std::vector<int32_t> offset_value;
  GetInitValueList(*node_arg, offset_value);
  if (offset_value.size() > 0) {
    return offset_value[0];
  } else {
    return 0;
  }
}

std::vector<float> ZhouyiModel::ParseScale(const std::string& scale_name) const {
  std::vector<float> scales;
  const auto& graph_initializers = graph_viewer_->GetAllInitializedTensors();
  auto offset_it = graph_initializers.find(scale_name);
  if (offset_it == graph_initializers.end()) {
    LOGS(logger_, ERROR) << "Not able to find initializer: " << scale_name;
    return scales;
  }
  const auto scale_tensor = offset_it->second;
  Initializer unpacked_scale(*scale_tensor);
  if (unpacked_scale.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    for (size_t i = 0; i < unpacked_scale.size(); ++i) {
      float scale_fp16 = unpacked_scale.DataAsSpan<MLFloat16>()[i].ToFloat();
      scales.push_back(scale_fp16);
    }
  } else {
    for (size_t i = 0; i < unpacked_scale.size(); ++i) {
      float scale = unpacked_scale.DataAsSpan<float>()[i];
      scales.push_back(scale);
    }
  }

  return scales;
}

aipubt::Quantization ZhouyiModel::ParseQuantization(
    const NodeUnitIODef& io_def, const aipubt::Quantization pre_quant) const {
  int32_t zp = 0;
  std::vector<float> scales;
  auto quant_param = io_def.quant_param;
  auto aipu_dtype = ParseDType(io_def.node_arg.TypeAsProto());
  // Parse scale & zero_point
  if (quant_param.has_value()) {
    scales = ParseScale(quant_param->scale.Name());
    if (quant_param->zero_point) {
      zp = ParseOffset(quant_param->zero_point);
    }
    if (scales.size() == 0) {
      scales.push_back(1.0);
    }

    auto quant = aipubt::Quantization(scales, zp);
    if (scales.size() > 0) {
      std::vector<int32_t> offsets;
      offsets.assign(scales.size(), zp);
      quant.set_offsets(offsets);
      quant.set_group(scales.size());
    }
    quant.set_unquantifiable(true);
    quant.set_aipu_dtype(aipu_dtype);
    return quant;
  } else if (aipu_dtype != aipubt::Tensor::FP16 &&
             aipu_dtype != aipubt::Tensor::FP32 &&
             !pre_quant.empty()) {
    return pre_quant;
  } else {
    aipubt::Quantization default_quant = aipubt::Quantization(1.0, 0);
    default_quant.set_unquantifiable(true);
    return default_quant;
  }
}

aipubt::Quantization ZhouyiModel::GetQuant(std::string name) {
  aipubt::Quantization quant(1.0, 0);
  quant.set_unquantifiable(true);
  const auto& node_indices = graph_viewer_->GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_->GetNode(node_indices[i]));
    auto* node_unit = node_unit_map_.at(node);
    bool b_find = false;
    for (auto t : node_unit->Inputs()) {
      if (name == t.node_arg.Name()) {
        b_find = true;
        quant = ParseQuantization(t);
        break;
      }
    }
    if (b_find)
      break;
  }
  return quant;
}

bool ZhouyiModel::GetTensorDims(const NodeArg& node_arg, std::vector<int32_t>& shape) const {
  return zhouyi::utils::GetTensorDims(node_arg, shape);
}

Status ZhouyiModel::UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                          std::vector<uint8_t>& unpacked_tensor) const {
  if (initializer.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    return onnxruntime::utils::UnpackInitializerData(initializer, graph_viewer_->ModelPath(), unpacked_tensor);
  }

  return onnxruntime::utils::UnpackInitializerData(initializer, unpacked_tensor);
}

aipubt::GraphPtr ZhouyiModel::Graph() {
  return aipubt_graph_;
}

bool ZhouyiModel::IsNodeSupported(const NodeUnit* node_unit) const {
  const std::vector<std::string> supported_types{
      "Add", "Sub", "Mul", "QLinearAdd", "QLinearMul", "Max", "Min",
      "MatMul", "QLinearMatMul", "Gemm", "QGemm",
      "Softmax", "LogSoftmax", "Cast", "And", "Or", "Not",
      "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal",
      "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot", "Pad",
      "QuantizeLinear", "DequantizeLinear",
      "Conv", "QLinearConv",
      "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
      "MaxPool", "QLinearAveragePool", "QLinearGlobalAveragePool",
      "Clip", "Relu", "QLinearLeakyRelu",
      "QLinearSigmoid", "Sigmoid", "Tanh", "HardSwish", "PRelu",
      "Pow", "Div", "Sqrt", "Abs", "Erf",
      "Concat", "QLinearConcat",
      "Reshape", "Flatten", "Squeeze", "Unsqueeze",
      "SpaceToDepth", "DepthToSpace", "Transpose",
      "Split", "Slice",
      "ReduceMax", "ReduceMin", "ReduceMean", "ReduceSum", "ReduceProd",
      "LRN",
      "Gather", "GatherND", "GatherElements",
      "Resize", "GroupNormalization", "BatchNormalization",
      "Tile", "Expand", "CumSum",
      "EPContext"};
  const std::string& op_type = node_unit->OpType();
  if (std::find(supported_types.begin(), supported_types.end(), op_type) ==
      supported_types.end()) {
    return false;
  }
  auto InitTensorCheck = [&](std::string node_name, std::string tensor_name) {
    if (!IsInitTensors(tensor_name)) {
      LOGS(logger_, INFO) << "The node " << node_name
                          << " input " << tensor_name
                          << " is not initializer tensor.\n"
                          << "Zhouyi only support initializer tensor.";
      return false;
    }
    return true;
  };

  NodeAttrHelper node_helper(*node_unit);
  if (op_type == "BatchNormalization") {
    if (node_helper.HasAttr("training_mode")) {
      const bool training_mode = node_helper.Get("training_mode", 0) != 0;
      if (training_mode) {
        LOGS(logger_, INFO) << "BatchNormalization doesn't support training_mode.";
        return false;
      }
    }
    for (size_t i = 1; i < node_unit->Inputs().size(); ++i) {
      if (!InitTensorCheck(node_unit->Name(), node_unit->Inputs()[i].node_arg.Name())) {
        return false;
        break;
      }
    }
  } else if (op_type == "CumSum") {
    return InitTensorCheck(node_unit->Name(), node_unit->Inputs()[1].node_arg.Name());
  } else if (op_type == "Expand") {
    return InitTensorCheck(node_unit->Name(), node_unit->Inputs()[1].node_arg.Name());
  } else if (op_type == "GroupNormalization") {
    if (node_unit->SinceVersion() >= 21) {
      // Zhouyi doesn't support stash_type  (added in opset 21)
      const bool be_stash_type = node_helper.Get("stash_type", 0) != 0;
      if (be_stash_type) {
        LOGS(logger_, INFO) << "GroupNormalization doesn't support stash_type=0.";
        return false;
      }
    }
    for (size_t i = 1; i < 3; ++i) {
      if (!InitTensorCheck(node_unit->Name(), node_unit->Inputs()[i].node_arg.Name())) {
        return false;
        break;
      }
    }
  } else if (op_type == "LRN") {
    const auto alpha = node_helper.Get("alpha", 0.0001f);
    const auto beta = node_helper.Get("beta", 0.75f);
    const auto size = node_helper.Get("size", 1);
    // 'size' attribute must be odd and > 0.
    if (size % 2 == 0) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator's size attribute must be odd.";
      return false;
    }

    // 'alpha' attribute must be > 0.0f.
    if (alpha <= 0.0f) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator's alpha attribute must be greater than zero.";
      return false;
    }
    // 'alpha' attribute must be > 0.0f.
    if (beta <= 0.0f) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator's beta attribute must be greater than zero.";
      return false;
    }

    if (node_unit->Inputs().size() != 1) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator must have 1 input.";
      return false;
    }
    if (node_unit->Outputs().size() != 1) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator must have 1 output.";
      return false;
    }
    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, INFO) << "Cannot get input shape ";
      return false;
    }
    if (input_shape.size() != 4) {
      LOGS(logger_, INFO) << "Zhouyi EP: LRN operator only supports input rank 4.";
      return false;
    }
  } else if (op_type == "MaxPool") {
    const auto storage_order = node_helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS(logger_, INFO) << "storage_order == 1 is not supported";
      return false;
    }
    const auto dilations = node_helper.Get("dilations", std::vector<int32_t>{1, 1});
    if (dilations != std::vector<int32_t>{1, 1}) {
      LOGS(logger_, INFO) << "Dilations of pooling is not supported";
      return false;
    }

    if (node_helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS(logger_, INFO) << "Only pooling 2d is supported";
      return false;
    }

    if (node_unit->Outputs().size() != 1) {
      LOGS(logger_, INFO) << "Argmax in MaxPool is not supported";
      return false;
    }
  } else if (op_type == "AveragePool") {
    if (node_helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS(logger_, INFO) << "Only pooling 2d is supported";
      return false;
    }
  } else if (op_type == "QLinearAveragePool") {
    if (node_helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS(logger_, INFO) << "Only pooling 2d is supported";
      return false;
    }
  } else if (op_type == "PRelu") {
    return InitTensorCheck(node_unit->Name(), node_unit->Inputs()[1].node_arg.Name());
  } else if (op_type == "Gemm") {
    auto alpha = node_helper.Get("alpha", (float)1.0);
    auto beta = node_helper.Get("beta", (float)1.0);
    auto transA = node_helper.Get("transA", (int64_t)0);
    auto transB = node_helper.Get("transB", (int64_t)0);
    if (node_unit->Inputs().size() == 3) {
      if ((alpha != 1.0) || (beta != 1.0) || (transA != 0) || (transB != 1)) {
        return false;
      }
      std::string weight_name = node_unit->Inputs()[1].node_arg.Name();
      std::string bias_name = node_unit->Inputs()[2].node_arg.Name();
      if (!IsInitTensors(weight_name) || !IsInitTensors(bias_name)) {
        LOGS(logger_, INFO) << "weight/bias not is initializer tensor.";
        return false;
      }
    }
  } else if (op_type == "ReduceMax") {
    std::set<int> supported_versions = {1, 11, 12, 13, 18, 20};
    return supported_versions.count(node_unit->SinceVersion()) != 0;
  } else if (op_type == "ReduceMin") {
    std::set<int> supported_versions = {1, 11, 12, 13, 18, 20};
    return supported_versions.count(node_unit->SinceVersion()) != 0;
  } else if (op_type == "ReduceMean") {
    std::set<int> supported_versions = {1, 11, 13, 18};
    return supported_versions.count(node_unit->SinceVersion()) != 0;
  } else if (op_type == "ReduceProd") {
    std::set<int> supported_versions = {1, 11, 13, 18};
    return supported_versions.count(node_unit->SinceVersion()) != 0;
  } else if (op_type == "ReduceSum") {
    std::set<int> supported_versions = {1, 11, 13};
    return supported_versions.count(node_unit->SinceVersion()) != 0;
  } else if (op_type == "Reshape") {
    auto allowzero = node_helper.Get("allowzero", static_cast<int64_t>(0));
    if (allowzero != 0) {
      LOGS(logger_, INFO) << "Zhouyi reshape doesn't support dynamic shape!";
      return false;
    }
  } else if (op_type == "QGemm") {
    auto alpha = node_helper.Get("alpha", (float)1.0);
    auto beta = node_helper.Get("beta", (float)1.0);
    auto transA = node_helper.Get("transA", (int64_t)0);
    auto transB = node_helper.Get("transB", (int64_t)0);
    if (node_unit->Inputs().size() == 9) {
      auto input0_type = node_unit->Inputs()[0].node_arg.TypeAsProto()->tensor_type().elem_type();
      if (input0_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
          input0_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        if ((alpha != 1.0) || (beta != 1.0) || (transA != 0) || (transB != 1)) {
          return false;
        }
        std::string weight_name = node_unit->Inputs()[3].node_arg.Name();
        std::string bias_name = node_unit->Inputs()[6].node_arg.Name();
        if (!IsInitTensors(weight_name) || !IsInitTensors(bias_name)) {
          LOGS(logger_, INFO) << "weight/bias not is initializer tensor.";
          return false;
        }
      }
    }
  } else if (op_type == "Split") {
    auto axis = node_helper.Get("axis", 0);
    std::vector<uint32_t> splits_size;
    if (node_unit->Inputs().size() == 1) {
      uint32_t splits_num_default = node_unit->Outputs().size();
      uint32_t splits_num = node_helper.Get("num_outputs", splits_num_default);

      std::vector<int32_t> input_shape;
      if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
        LOGS(logger_, INFO) << "Cannot get input shape ";
      }
      if (axis < 0) {
        axis = axis + input_shape.size();
      }
      if (std::ceil(float(input_shape[axis]) / splits_num) * (splits_num - 1) >= input_shape[axis]) {
        LOGS(logger_, INFO) << "The given parameters cannot be split!";
        return false;
      };
    }
  } else if (op_type == "Tile") {
    return InitTensorCheck(node_unit->Name(), node_unit->Inputs()[1].node_arg.Name());
  } else if (op_type == "Resize") {
    const std::string ct_mode =
        node_helper.Get("coordinate_transformation_mode", std::string("half_pixel"));
    if (supported_ct_modes.count(ct_mode) != 1) {
      LOGS(logger_, INFO) << "Resize does not support coordinate_transformation_mode "
                          << ct_mode;
      return false;
    }

    const std::string mode = node_helper.Get("mode", std::string("nearest"));
    if (supported_modes.count(mode) != 1) {
      LOGS(logger_, INFO) << "Resize does not support mode " << mode;
      return false;
    }
    const std::string nearest_mode =
        node_helper.Get("nearest_mode", std::string("round_prefer_floor"));
    if (supported_nearest_modes.count(nearest_mode) != 1) {
      LOGS(logger_, INFO) << "Resize does not support mode " << nearest_mode;
      return false;
    }

    // Zhouyi doesn't support anti-aliasing (added in opset 18)
    if (node_unit->SinceVersion() >= 18) {
      const bool antialias = node_helper.Get("antialias", 0) != 0;
      if (antialias) {
        LOGS(logger_, INFO) << "Resize doesn't support anti-aliasing.";
        return false;
      }
    }

    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, INFO) << "Cannot get input shape ";
      return false;
    }
    if (input_shape.size() != 4) {
      LOGS(logger_, INFO) << "Only support 4-D input.";
      return false;
    }
    if (node_unit->Inputs().size() == 2) {
      LOGS(logger_, INFO) << "Zhouyi doesn't support roi value input.";
      return false;
    }

    if (node_unit->Inputs().size() == 3) {
      if (!InitTensorCheck(node_unit->Name(), node_unit->Inputs()[2].node_arg.Name())) {
        return false;
      }
      std::vector<float> scales_value;
      GetInitValueList(node_unit->Inputs()[2].node_arg, scales_value);
      if ((scales_value.size() == 4) &&
          !(scales_value[0] == 1.0 && scales_value[1] == 1.0)) {
        LOGS(logger_, INFO) << "Only support height and weight resize.";
        return false;
      }
    } else if (node_unit->Inputs().size() == 4) {
      std::vector<int64_t> sizes_value;
      if (!InitTensorCheck(node_unit->Name(), node_unit->Inputs()[3].node_arg.Name())) {
        return false;
      }
      GetInitValueList(node_unit->Inputs()[3].node_arg, sizes_value);
      if ((sizes_value.size() == 4) &&
          !(sizes_value[0] == input_shape[0] &&
            sizes_value[1] == input_shape[1])) {
        LOGS(logger_, INFO) << "Only support height and weight resize.";
        return false;
      }
    }
  } else if (op_type == "MatMul") {
    std::vector<int32_t> input_shape;
    if (!GetTensorDims(node_unit->Inputs()[0].node_arg, input_shape)) {
      LOGS(logger_, INFO) << "Cannot get input shape ";
      return false;
    }
    if (aipuruntime::get_target() == "X2_1204" || aipuruntime::get_target() == "X2_1204MP3") {
      if (input_shape.size() == 4) {
        auto elem_type = node_unit->Inputs()[0].node_arg.TypeAsProto()->tensor_type().elem_type();
        if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
            elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          LOGS(logger_, INFO) << "Zhouyi EP: Float MatMul operator not supports input rank 4.";
          return false;
        }
      }
    }
  }
  return true;
}

std::unordered_set<const Node*> ZhouyiModel::GetSupportedNodes() const {
  std::unordered_set<const Node*> supported_nodes{};
  std::unordered_map<const NodeUnit*, bool> supports;
  supports.reserve(node_unit_holder_.size());
  const auto& node_indices = graph_viewer_->GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    gsl::not_null<const onnxruntime::Node*> node(graph_viewer_->GetNode(node_indices[i]));

    // Get the node_unit associated with the node. Note that the node may not be the node_unit's target node.

    const NodeUnit* node_unit = node_unit_map_.at(node);
    if (node_unit == nullptr) {
      LOGS(logger_, ERROR) << "node_unit == nullptr ";
    }
    // Visiting 'nodes' in topological order does not guarantee that 'node_units' are
    // also visited in topological order. Skip this node if it is not the node_unit's target node
    // to ensure 'node_units' are visited in topological order.
    if (node != &node_unit->GetNode()) {
      LOGS(logger_, VERBOSE) << "----Not in unit node "
                             << " index: [" << node->Index()
                             << "] name: [" << node->Name()
                             << "] Operator type: [" << node->OpType()
                             << "] as part of the NodeUnit type: [" << node_unit->OpType()
                             << "] index: [" << node_unit->Index()
                             << "] name: [" << node_unit->Name()
                             << "]";
      continue;
    }
    bool supported = true;
    // If we have visited one of the nodes in the node_unit, use the result directly
    if (supports.find(node_unit) != supports.cend()) {
      supported = supports[node_unit];
    } else {
      supported = IsNodeSupported(node_unit);
      supports[node_unit] = supported;
    }
    LOGS(logger_, VERBOSE) << "Node supported: [" << supported
                           << "] index: [" << node->Index()
                           << "] name: [" << node->Name()
                           << "] Operator type: [" << node->OpType()
                           << "] as part of the NodeUnit type: [" << node_unit->OpType()
                           << "] index: [" << node_unit->Index()
                           << "] name: [" << node_unit->Name()
                           << "]";
    if (supported) {
      // If the node_unit is supported, add all of its nodes to the supported list.
      for (const auto* node_in_unit : node_unit->GetAllNodesInGroup()) {
        if (std::find(supported_nodes.begin(), supported_nodes.end(), node_in_unit) == supported_nodes.end())
          supported_nodes.insert(node_in_unit);
      }
    }
  }

  return supported_nodes;
}

aipubt::TensorPtr ZhouyiModel::NchwOpCreate(const NodeUnit* node_unit, NCHWOpFunc op_func) {
  // 1. to NHWC
  auto input0 = AipuTensorMatch(node_unit->Inputs()[0]);
  if (input0->shape().dim() != 4) {
    return op_func(input0, node_unit);
  }
  std::vector<uint32_t> perm = {0, 2, 3, 1};
  auto transpose_output = aipubt::ops::transpose(input0, perm);

  // 2. func
  auto op_output = op_func(transpose_output, node_unit);

  // 3. to NCHW
  perm = {0, 3, 1, 2};
  auto output = aipubt::ops::transpose(op_output, perm);

  return output;
}

}  // namespace zhouyi
}  // namespace onnxruntime
