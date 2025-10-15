// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "aipuruntime_cxx_api.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/node_arg.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace zhouyi {
#define Status_FAIL(...)                             \
  common::Status(::onnxruntime::common::ONNXRUNTIME, \
                 ::onnxruntime::common::FAIL,        \
                 ::onnxruntime::MakeString(ORT_WHERE.ToString(), " ", __VA_ARGS__));

using Zhouyi_DataType_t = ZhouyiOperandType;
using Zhouyi_Operation_t = ZhouyiOperationType;

namespace utils {
Status GetZhouyiDataType(const ONNX_NAMESPACE::TypeProto* type_proto,
                         Zhouyi_DataType_t& tensor_data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

size_t GetElementSizeByType(const ONNX_NAMESPACE::TypeProto* proto);

template <typename T1, typename T2>
void DataToVector(std::vector<T1>& params, const void* data, uint32_t data_len) {
  uint32_t params_size = data_len / sizeof(T2);
  const T2* data_t2_ptr = static_cast<const T2*>(data);
  for (uint32_t i = 0; i < params_size; ++i) {
    params.push_back(static_cast<T1>(data_t2_ptr[i]));
  }
  return;
}

template <typename T1>
void FP16DataToVector(std::vector<T1>& params, const void* data, uint32_t data_len) {
  params.resize(data_len / 2);
  memcpy(params.data(), data, data_len);
  return;
}

// Some Initialized Tensors are param in zhouyi
template <typename T1>
void InitDataToVector(std::vector<T1>& params,
                      int32_t onnx_data_type,
                      const void* data,
                      uint32_t data_len) {
  switch (onnx_data_type) {
  case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    DataToVector<T1, bool>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    DataToVector<T1, int8_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    DataToVector<T1, uint8_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    DataToVector<T1, int16_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
    DataToVector<T1, uint16_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    DataToVector<T1, int32_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
    DataToVector<T1, uint32_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_INT64:
    DataToVector<T1, int64_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
    DataToVector<T1, uint64_t>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    FP16DataToVector<T1>(params, data, data_len);
    break;
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    DataToVector<T1, float>(params, data, data_len);
    break;
  default:
    break;
  }
  return;
}

template <typename T1, typename T2>
void DataTypeTranspose(T1* dst, const T2* src, uint32_t data_len) {
  for (uint32_t i = 0; i < data_len; ++i) {
    dst[i] = T2(src[i]);
  }
  return;
}
bool GetTensorDims(const NodeArg& node_arg, std::vector<int32_t>& shape);

int32_t GetEnv(const char* env_name, int32_t default_val);

bool GetEnv(const char* env_name, bool default_val);

const char* GetEnv(const char* env_name, const char* default_val);

}  // namespace utils
}  // namespace zhouyi
}  // namespace onnxruntime
