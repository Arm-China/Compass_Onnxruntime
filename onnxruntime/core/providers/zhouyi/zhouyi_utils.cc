// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#include "zhouyi_utils.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "constants.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/data_types.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace zhouyi {
namespace utils {

void DataToParamsFP16(void* params, uint32_t& params_size, const void* data, uint32_t data_len) {
  params_size = data_len / 2 > ZHOUYI_MAX_RANK ? ZHOUYI_MAX_RANK : data_len / 2;
  memcpy(params, data, params_size * 2);

  return;
}

Status GetZhouyiDataType(const ONNX_NAMESPACE::TypeProto* type_proto,
                         Zhouyi_DataType_t& tensor_data_type) {
  auto OnnxDataTypeToZhouyiDataType = [](const int32_t onnx_data_type,
                                         Zhouyi_DataType_t& zhouyi_data_type) {
    const std::unordered_map<int32_t, Zhouyi_DataType_t> onnx_to_zhouyi_data_type = {
        {ONNX_NAMESPACE::TensorProto_DataType_INT8, ZHOUYI_DT_INT8},
        {ONNX_NAMESPACE::TensorProto_DataType_INT16, ZHOUYI_DT_INT16},
        {ONNX_NAMESPACE::TensorProto_DataType_INT32, ZHOUYI_DT_INT32},
        {ONNX_NAMESPACE::TensorProto_DataType_INT64, ZHOUYI_DT_INT32},
        {ONNX_NAMESPACE::TensorProto_DataType_UINT8, ZHOUYI_DT_UINT8},
        {ONNX_NAMESPACE::TensorProto_DataType_UINT16, ZHOUYI_DT_UINT16},
        {ONNX_NAMESPACE::TensorProto_DataType_UINT32, ZHOUYI_DT_UINT32},
        {ONNX_NAMESPACE::TensorProto_DataType_UINT64, ZHOUYI_DT_UINT32},
        {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, ZHOUYI_DT_FLOAT16},
        {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ZHOUYI_DT_FLOAT32},
        {ONNX_NAMESPACE::TensorProto_DataType_BOOL, ZHOUYI_DT_BOOL},
    };

    const auto do_type_mapping = [](const std::unordered_map<int32_t, Zhouyi_DataType_t>& mapping_table,
                                    const int32_t onnx_data_type,
                                    Zhouyi_DataType_t& zhouyi_data_type) -> bool {
      auto pos = mapping_table.find(onnx_data_type);
      if (pos == mapping_table.end()) {
        LOGS_DEFAULT(WARNING) << "Unsupport TensorProto DataType: " << onnx_data_type
                              << ". Set default value: ZHOUYI_DT_UINT8";
        zhouyi_data_type = ZHOUYI_DT_UINT8;
        return false;
      }
      zhouyi_data_type = pos->second;
      return true;
    };
    return do_type_mapping(onnx_to_zhouyi_data_type, onnx_data_type, zhouyi_data_type);
  };

  if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The tensor doesn't have elem_type.");
  }

  int32_t onnx_data_type = type_proto->tensor_type().elem_type();
  ORT_RETURN_IF_NOT(OnnxDataTypeToZhouyiDataType(onnx_data_type, tensor_data_type),
                    "Failed to map Onnx data type to Zhouyi data type!");

  return Status::OK();
}

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type) {
  std::unordered_map<ONNXTensorElementDataType, size_t> elem_type_to_size = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)},
  };
  if (elem_type_to_size.count(elem_type)) {
    return elem_type_to_size[elem_type];
  } else {
    LOGS_DEFAULT(WARNING) << "Unsupport TensorProto DataType: " << elem_type
                          << ". use default value: 1";
    return 1;
  }
}

size_t GetElementSizeByType(const ONNX_NAMESPACE::TypeProto* proto) {
  int32_t data_type = proto->tensor_type().elem_type();
  std::unordered_map<int32_t, size_t> elem_type_to_size = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, sizeof(int8_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, sizeof(int16_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, sizeof(int32_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, sizeof(int64_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, sizeof(uint8_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, sizeof(uint16_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, sizeof(uint32_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, sizeof(uint64_t)},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, 2},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, sizeof(float)},
      {ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, 2},
      {ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, sizeof(double)},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2, 2},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, 2},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, sizeof(bool)},
  };
  if (elem_type_to_size.count(data_type)) {
    return elem_type_to_size[data_type];
  } else {
    LOGS_DEFAULT(WARNING) << "Unsupport TensorProto DataType: " << data_type
                          << ". use default value: 1";
    return 1;
  }
}

bool GetTensorDims(const NodeArg& node_arg, std::vector<int32_t>& shape) {
  const auto* shape_proto = node_arg.Shape();
  if (shape_proto == nullptr) {
    return false;
  }

  // For Scalar data, we need to set shape to 1 for ZHOUYI
  if (shape_proto->dim_size() < 1) {
    shape.push_back(1);
    return true;
  }

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    if (onnxruntime::utils::HasDimValue(dim)) {
      shape.push_back(SafeInt<int32_t>(dim.dim_value()));
    } else {
      LOGS_DEFAULT(ERROR) << "Shape Must has no dynamic dimension.";
      return false;
    }
  }

  return true;
}

int32_t GetEnv(const char* env_name, int32_t default_val) {
  const char* env = std::getenv(env_name);
  if (env != nullptr) {
    return std::stoi(env);
  }
  return default_val;
}

bool GetEnv(const char* env_name, bool default_val) {
  const char* env = std::getenv(env_name);
  if (env != nullptr) {
    const std::map<std::string, bool> table = {
        {"1", true},
        {"on", true},
        {"true", true},
        {"0", false},
        {"off", false},
        {"false", true},
    };
    std::string env_str(env);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
    if (table.count(env_str)) {
      return table.at(env_str);
    }
  }
  return default_val;
}

const char* GetEnv(const char* env_name, const char* default_val) {
  const char* env = std::getenv(env_name);
  if (env != nullptr) {
    return env;
  }
  return default_val;
}

}  // namespace utils
}  // namespace zhouyi
}  // namespace onnxruntime