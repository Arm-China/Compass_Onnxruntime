// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "aipuruntime_cxx_api.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/node_unit.h"
#include "core/graph/basic_types.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/zhouyi/zhouyi_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
using NodeUnitMap = std::unordered_map<const Node*, const NodeUnit*>;
namespace zhouyi {
using TensorPtrList = std::vector<aipubt::TensorPtr>;
using OpBinaryFunc = std::function<aipubt::TensorPtr(
    aipubt::TensorPtr, aipubt::TensorPtr, const aipubt::Quantization&)>;
using OpUnaryFunc = std::function<aipubt::TensorPtr(
    aipubt::TensorPtr, const aipubt::Quantization&)>;
using NCHWOpFunc = std::function<aipubt::TensorPtr(
    aipubt::TensorPtr, const NodeUnit* node_unit)>;

class ZhouyiModel {
 public:
  ZhouyiModel(const onnxruntime::GraphViewer* graph_viewer,
              const logging::Logger& logger);
  ~ZhouyiModel() = default;

  //
  aipubt::GraphPtr Graph();

  //
  Status Build();

  //
  std::unordered_set<const Node*> GetSupportedNodes() const;

 private:
  template <typename T>
  void GetInitValueList(const NodeArg& node_arg, std::vector<T>& values) const {
    if (IsInitTensors(node_arg.Name())) {
      std::vector<uint8_t> unpacked_tensor;
      const auto value_tensor = GetInitTensorProto(node_arg.Name());
      if (UnpackInitializerData(*value_tensor, unpacked_tensor).IsOK()) {
        values.clear();
        zhouyi::utils::InitDataToVector<T>(
            values,
            node_arg.TypeAsProto()->tensor_type().elem_type(),
            unpacked_tensor.data(),
            unpacked_tensor.size());
      }
    }
  }

  template <typename T>
  T GetInitValue(const NodeArg& node_arg) {
    std::vector<T> values;
    GetInitValueList<T>(node_arg, values);
    if (values.size() > 0)
      return values[0];
    else
      return 0;
  }

  // Parse the zero point
  int32_t ParseOffset(const NodeArg* node_arg) const;

  // Parse the scale
  std::vector<float> ParseScale(const std::string& scale_name) const;

  // Get Quantization info by name
  aipubt::Quantization GetQuant(std::string name);

  // NodeUnitIODef::QuantParam  -> aipubt::Quantization
  aipubt::Quantization ParseQuantization(
      const NodeUnitIODef& io_def, const aipubt::Quantization pre_quant = {}) const;

  //
  void EmplaceInputs();

  /**
   * @brief Returns an aiput tensor dtype based on the given onnxruntime TypeProto type.
   * @returns aipubt::Tensor::Dtype
              If no match is found, return the default value aipubt::Tensor::UINT8.
   */
  aipubt::Tensor::Dtype ParseDType(const ONNX_NAMESPACE::TypeProto* proto) const;

  //
  aipubt::TensorShape ParseShape(const NodeArg& node_arg);

  bool IsInitTensors(const std::string& name) const;

  const ONNX_NAMESPACE::TensorProto* GetInitTensorProto(std::string name) const;

  Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                               std::vector<uint8_t>& unpacked_tensor) const;

  // Add a new node to the Graph
  Status AddNode(const NodeUnit* node_unit);

  //
  TensorPtrList OpAdd(const NodeUnit* node_unit);
  TensorPtrList OpSub(const NodeUnit* node_unit);
  TensorPtrList OpMul(const NodeUnit* node_unit);
  TensorPtrList OpMax(const NodeUnit* node_unit);
  TensorPtrList OpMin(const NodeUnit* node_unit);
  TensorPtrList OpMatMul(const NodeUnit* node_unit);
  TensorPtrList OpGemm(const NodeUnit* node_unit);
  TensorPtrList OpQGemm(const NodeUnit* node_unit);
  TensorPtrList OpSoftmax(const NodeUnit* node_unit);
  TensorPtrList OpLogSoftmax(const NodeUnit* node_unit);
  TensorPtrList OpCast(const NodeUnit* node_unit);
  TensorPtrList OpAnd(const NodeUnit* node_unit);
  TensorPtrList OpOr(const NodeUnit* node_unit);
  TensorPtrList OpNot(const NodeUnit* node_unit);
  TensorPtrList OpBinary(const NodeUnit* node_unit, OpBinaryFunc func);
  TensorPtrList OpBitwise(const NodeUnit* node_unit,
                          aipubt::ops::BitwiseMethod method);
  TensorPtrList OpPad(const NodeUnit* node_unit);
  TensorPtrList OpQuantizeLinear(const NodeUnit* node_unit);
  TensorPtrList OpDequantizeLinear(const NodeUnit* node_unit);
  TensorPtrList OpConv(const NodeUnit* node_unit);
  TensorPtrList OpPool(const NodeUnit* node_unit);
  TensorPtrList OpUnary(const NodeUnit* node_unit, OpUnaryFunc func);
  TensorPtrList OpConcat(const NodeUnit* node_unit);
  TensorPtrList OpPrelu(const NodeUnit* node_unit);
  TensorPtrList OpLeakyRelu(const NodeUnit* node_unit);
  TensorPtrList OpClip(const NodeUnit* node_unit);
  TensorPtrList OpReshape(const NodeUnit* node_unit);
  TensorPtrList OpFlatten(const NodeUnit* node_unit);
  TensorPtrList OpSqueeze(const NodeUnit* node_unit);
  TensorPtrList OpUnSqueeze(const NodeUnit* node_unit);
  TensorPtrList OpSpaceToDepth(const NodeUnit* node_unit);
  TensorPtrList OpDepthToSpace(const NodeUnit* node_unit);
  TensorPtrList OpTranspose(const NodeUnit* node_unit);
  TensorPtrList OpSplit(const NodeUnit* node_unit);
  TensorPtrList OpSlice(const NodeUnit* node_unit);
  TensorPtrList OpReduce(const NodeUnit* node_unit);
  TensorPtrList OpLRN(const NodeUnit* node_unit);
  TensorPtrList OpGather(const NodeUnit* node_unit);
  TensorPtrList OpGatherND(const NodeUnit* node_unit);
  TensorPtrList OpGatherElements(const NodeUnit* node_unit);
  TensorPtrList OpResize(const NodeUnit* node_unit);
  TensorPtrList OpGroupNormalization(const NodeUnit* node_unit);
  TensorPtrList OpBatchNormalization(const NodeUnit* node_unit);
  TensorPtrList OpTile(const NodeUnit* node_unit);
  TensorPtrList OpExpand(const NodeUnit* node_unit);
  TensorPtrList OpCumSum(const NodeUnit* node_unit);

  aipubt::TensorPtr NchwOpCreate(const NodeUnit* node_unit, NCHWOpFunc op_func);

  bool GetTensorDims(const NodeArg& node_arg, std::vector<int32_t>& shape) const;
  //
  aipubt::TensorPtr ConstantTensor(const NodeArg& node_arg);

  aipubt::TensorPtr OpConstant(const NodeArg& node_arg);

  aipubt::TensorPtr CreateWeight(const NodeArg& node_arg);

  aipubt::TensorPtr AipuTensorMatch(const NodeArg* node_arg);

  aipubt::TensorPtr AipuTensorMatch(const NodeUnitIODef& io_def,
                                    bool constant_tensor = false);

  bool IsNodeSupported(const NodeUnit* node_unit) const;

  const onnxruntime::GraphViewer* graph_viewer_;
  const logging::Logger& logger_;
  std::unordered_set<std::string> init_tensors_;
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder_;
  NodeUnitMap node_unit_map_;
  aipubt::GraphPtr aipubt_graph_ = nullptr;
  std::map<std::string, aipubt::TensorPtr> tensor_maps_;
  bool prepared_ = false;
};

}  // namespace zhouyi
}  // namespace onnxruntime
