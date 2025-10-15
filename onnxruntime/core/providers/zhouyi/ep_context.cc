// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#include "ep_context.h"

#include <filesystem>
#include <fstream>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/data_types.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/zhouyi/constants.h"
#include "core/providers/zhouyi/zhouyi_utils.h"

namespace onnxruntime {
namespace zhouyi {

namespace {
std::filesystem::path GetFullFilepathOfCtxBinary(const GraphViewer& graph_viewer,
                                                 const std::string& ctx_cache_path) {
  NodeAttrHelper node_helper(*(graph_viewer.Nodes().begin()));
  std::string cache_file = node_helper.Get(constants::EP_CACHE_CONTEXT, "");
  return std::filesystem::path(ctx_cache_path).parent_path().append(cache_file);
}

uint64_t GetCtxBinaryFileSize(const GraphViewer& graph_viewer,
                              const std::string& ctx_cache_path) {
  std::filesystem::path ctx_bin_fullpath = GetFullFilepathOfCtxBinary(graph_viewer, ctx_cache_path);
  if (!std::filesystem::is_regular_file(ctx_bin_fullpath) || !std::filesystem::exists(ctx_bin_fullpath)) {
    LOGS_DEFAULT(ERROR) << "Cannot get valid binary file: " << ctx_bin_fullpath;
    return 0;
  }
  return std::filesystem::file_size(ctx_bin_fullpath);
}

bool CheckCtxIO(const GraphViewer& graph_viewer, aipuruntime::SessionPtr session) {
  auto dataSize = [](const NodeArg* node_arg) {
    int32_t data_size = zhouyi::utils::GetElementSizeByType(node_arg->TypeAsProto());
    std::vector<int32_t> dims;
    if (!zhouyi::utils::GetTensorDims(*node_arg, dims)) {
      LOGS_DEFAULT(ERROR) << "Cannot get shape ";
    }
    for (auto d : dims) {
      data_size *= d;
    }
    return data_size;
  };
  auto bin_io_info = session->get_binary_io_info();
  if (bin_io_info.inputs.size() != graph_viewer.GetInputs().size() ||
      bin_io_info.outputs.size() != graph_viewer.GetOutputs().size()) {
    LOGS_DEFAULT(ERROR) << "Binary graph inputs/outputs number cannot match current model.onnx's number!";
    return false;
  }
  for (size_t i = 0; i < graph_viewer.GetInputs().size(); ++i) {
    Zhouyi_DataType_t tensor_data_type;
    if (!zhouyi::utils::GetZhouyiDataType(graph_viewer.GetInputs()[i]->TypeAsProto(), tensor_data_type).IsOK())
      return false;
    if (tensor_data_type != bin_io_info.inputs[i].type) {
      LOGS_DEFAULT(ERROR) << "Binary graph " << i << "th input data type cannot match!";
      return false;
    }
    u_int32_t data_size = dataSize(graph_viewer.GetInputs()[i]);
    if (data_size != bin_io_info.inputs[i].data_len) {
      LOGS_DEFAULT(ERROR) << "Binary graph " << i << "th input data length cannot match!";

      return false;
    }
  }

  for (size_t i = 0; i < graph_viewer.GetOutputs().size(); ++i) {
    Zhouyi_DataType_t tensor_data_type;
    if (!zhouyi::utils::GetZhouyiDataType(graph_viewer.GetOutputs()[i]->TypeAsProto(), tensor_data_type).IsOK())
      return false;
    if (tensor_data_type != bin_io_info.outputs[i].type) {
      LOGS_DEFAULT(ERROR) << "Binary graph " << i << "th input data type cannot match!";
      return false;
    }

    u_int32_t data_size = dataSize(graph_viewer.GetOutputs()[i]);
    if (data_size != bin_io_info.outputs[i].data_len) {
      LOGS_DEFAULT(ERROR) << "Binary graph " << i << "th output data length cannot match!";
      return false;
    }
  }

  return true;
}

Status ComposeCtxInternal(const GraphViewer& graph_viewer,
                          aipuruntime::SessionPtr session,
                          const void* cache_buffer,
                          uint64_t cache_size) {
  ORT_RETURN_IF(cache_buffer == nullptr || cache_size == 0, "Invalid cache buffer or size!");
  int32_t ret = session->set_binary(cache_buffer, cache_size);
  for (size_t i = 0; i < graph_viewer.GetOutputs().size(); ++i) {
    std::vector<int32_t> dims;
    if (!zhouyi::utils::GetTensorDims(*(graph_viewer.GetOutputs()[i]), dims)) {
      LOGS_DEFAULT(WARNING) << "Cannot get shape ";
    }
    session->set_output_dims(i, dims);
  }

  ORT_RETURN_IF(ret != 0, "ZHOUYI set cache error. Error code: ", ret);

  ORT_RETURN_IF(graph_viewer.NumberOfNodes() != 1,
                "Context node number must be 1!");
  ORT_RETURN_IF(graph_viewer.Nodes().begin()->OpType() != constants::EPCONTEXT_OP,
                "Context node type must be EPContext!");
  ORT_RETURN_IF_NOT(CheckCtxIO(graph_viewer, session),
                    "Check with binary graph info failed.");
  return Status::OK();
}

// X2_1204
bool ValidateZhouyiArch(const std::string& arch) {
  auto split = [](const std::string& s, char delim) {
    std::vector<std::string> ret;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
      ret.push_back(item);
    }

    return ret;
  };
  std::vector<std::string> arch_strs = split(arch, '_');
  if (arch_strs.size() < 2)
    return false;

  transform(arch_strs[0].begin(), arch_strs[0].end(), arch_strs[0].begin(), tolower);

  if (arch_strs[0][0] != 'z' && arch_strs[0][0] != 'x') {
    return false;
  }

  auto number_error = [](char ch) {
    return ch < '0' || ch > '9';
  };
  if (number_error(arch_strs[1][0]) || number_error(arch_strs[1][1]) ||
      number_error(arch_strs[1][2]) || number_error(arch_strs[1][3])) {
    return false;
  }
  return true;
}

bool ValidateCtxNode(const GraphViewer& graph_viewer) {
  if (graph_viewer.NumberOfNodes() != 1) {
    LOGS_DEFAULT(ERROR) << "Context graph can only have one EPContext node!";
    return false;
  }

  auto index = graph_viewer.Nodes().begin()->Index();
  if (graph_viewer.GetNode(index)->OpType() != constants::EPCONTEXT_OP) {
    LOGS_DEFAULT(ERROR) << "Context model first node must be EPContext!";
    return false;
  }

  NodeAttrHelper node_helper(*(graph_viewer.GetNode(index)));
  const std::string& zhouyi_arch = node_helper.Get(constants::HARDWARE_ARCHITECTURE, "");
  LOGS_DEFAULT(VERBOSE) << "Zhouyi EPContext hardware arch: " << zhouyi_arch;
  if (zhouyi_arch.empty()) {
    LOGS_DEFAULT(ERROR) << "Cannot find Zhouyi architecture information in EPContext node!";
    return false;
  }

  if (!ValidateZhouyiArch(zhouyi_arch)) {
    LOGS_DEFAULT(ERROR) << "Zhouyi architecture information is wrong!";
    return false;
  }

  if (!node_helper.HasAttr(constants::EMBED_MODE) || !node_helper.HasAttr(constants::EP_CACHE_CONTEXT)) {
    LOGS_DEFAULT(ERROR) << "Cannot find Zhouyi embed_mode or ep_cache_context attribute in EPContext node!";
    return false;
  }
  return true;
}

Status LoadCtxBufferFromBinary(const GraphViewer& graph_viewer,
                               const std::string& ctx_cache_path,
                               char* buffer,
                               uint64_t size) {
  std::filesystem::path ctx_bin_fullpath =
      GetFullFilepathOfCtxBinary(graph_viewer, ctx_cache_path);
  ORT_RETURN_IF(!std::filesystem::is_regular_file(ctx_bin_fullpath) ||
                    !std::filesystem::exists(ctx_bin_fullpath),
                "Invalid context file: " + ctx_bin_fullpath.string());

  uint64_t file_size = std::filesystem::file_size(ctx_bin_fullpath);
  ORT_RETURN_IF(size > file_size,
                "Load size " + std::to_string(size) +
                    " is bigger than file size " + std::to_string(file_size));

  std::ifstream if_cache(ctx_bin_fullpath.string(), std::ios::binary | std::ios::in);
  if_cache.read(buffer, size);
  return Status::OK();
}

}  // namespace

bool HasCtxNode(const GraphViewer& graph_viewer) {
  for (const auto& node : graph_viewer.Nodes()) {
    if (constants::EPCONTEXT_OP == node.OpType()) {
      NodeAttrHelper node_helper(node);
      std::string cache_source = node_helper.Get(constants::SOURCE, "");
      if (cache_source == kZhouyiExecutionProvider) {
        return true;
      }
    }
  }
  return false;
}

std::string GetFullpathOfCtxModel(const std::string& user_ctx_cache_path,
                                  const std::string& model_path,
                                  bool is_ctx_model) {
  std::string cache_path;
  if (!user_ctx_cache_path.empty()) {
    cache_path = user_ctx_cache_path;
  } else if (is_ctx_model) {
    cache_path = model_path;
  } else if (!model_path.empty()) {
    cache_path = model_path + "_ctx.onnx";
  }
  return cache_path;
}

Status CreateCtx(const GraphViewer& graph_viewer,
                 aipuruntime::SessionPtr session,
                 const std::string& name,
                 Model* ctx_model,
                 bool embed_mode,
                 const std::string& ctx_cache_path) {
  ORT_RETURN_IF(ctx_model == nullptr, "Context model is nullptr!");
  auto& graph = ctx_model->MainGraph();
  std::vector<NodeArg*> inputs;
  for (const auto input : graph_viewer.GetInputs()) {
    auto& arg = graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
    inputs.push_back(&arg);
  }
  std::vector<NodeArg*> outputs;
  for (const auto output : graph_viewer.GetOutputs()) {
    auto& arg = graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
    outputs.push_back(&arg);
  }

  std::string target = aipuruntime::get_target();
  ORT_RETURN_IF(!ValidateZhouyiArch(target), "Get AIPU target failed!");

  uint64_t ctx_size = session->get_binary_size();
  ORT_RETURN_IF(ctx_size == 0, "Get binary buffer size failed!");

  // get aipu.bin to ctx_buffer
  std::unique_ptr<char[]> ctx_buffer = std::make_unique<char[]>(ctx_size);
  auto ret = session->get_binary(ctx_buffer.get(), ctx_size);
  ORT_RETURN_IF(ret != 0, "Get binary buffer failed!");

  // creat xxx_ctx.onnx, and save to ctx_cache_path
  auto& ctx_node = graph.AddNode(name, constants::EPCONTEXT_OP, "Zhouyi EPContext node: " + name,
                                 inputs, outputs, nullptr, kMSDomain);
  ctx_node.AddAttribute(constants::EMBED_MODE,
                        (embed_mode ? static_cast<int64_t>(1) : static_cast<int64_t>(0)));
  ctx_node.AddAttribute(constants::HARDWARE_ARCHITECTURE, target);
  ctx_node.AddAttribute(constants::PARTITION_NAME, name);
  ctx_node.AddAttribute(constants::SOURCE, kZhouyiExecutionProvider);
  if (embed_mode) {
    std::string content(ctx_buffer.get(), ctx_buffer.get() + ctx_size);
    ctx_node.AddAttribute(constants::EP_CACHE_CONTEXT, content);
  } else {
    onnxruntime::PathString context_file_path = ctx_cache_path + ToPathString("_" + name + ".bin");
    ORT_RETURN_IF(std::filesystem::exists(context_file_path),
                  "If you try to re-generate binary files, please remove old files: " + context_file_path);
    std::ofstream oss(context_file_path.c_str(), std::ofstream::binary);
    if (!oss) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to open context cache file!");
    }
    oss.write(ctx_buffer.get(), ctx_size);
    std::string context_file_name(std::filesystem::path(context_file_path).filename().string());
    ctx_node.AddAttribute(constants::EP_CACHE_CONTEXT, context_file_name);
  }

  return Status::OK();
}

Status ComposeCtx(const GraphViewer& graph_viewer,
                  aipuruntime::SessionPtr session,
                  const std::string& ctx_cache_path) {
  LOGS_DEFAULT(VERBOSE) << "ComposeCtx Graph name: " << graph_viewer.Name();

  ORT_RETURN_IF_NOT(ValidateCtxNode(graph_viewer), "Validate EPContext node fail!");

  NodeAttrHelper node_helper(*(graph_viewer.Nodes().begin()));
  int64_t embed_mode = node_helper.Get(constants::EMBED_MODE, static_cast<int64_t>(1));
  if (embed_mode) {
    const std::string& context_binary = node_helper.Get(constants::EP_CACHE_CONTEXT, "");
    ORT_RETURN_IF_ERROR(ComposeCtxInternal(
        graph_viewer, session, context_binary.c_str(), context_binary.size()));
  } else {
    uint64_t file_size = GetCtxBinaryFileSize(graph_viewer, ctx_cache_path);
    ORT_RETURN_IF(file_size == 0, "Cannot get valid context binary size: " + std::to_string(file_size));
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(file_size);
    ORT_RETURN_IF_ERROR(LoadCtxBufferFromBinary(graph_viewer, ctx_cache_path, buffer.get(), file_size));
    ORT_RETURN_IF_ERROR(ComposeCtxInternal(graph_viewer, session, buffer.get(), file_size));
  }

  LOGS_DEFAULT(VERBOSE) << "Compose Context completed.";
  return Status::OK();
}

std::vector<std::unique_ptr<ComputeCapability>>
PartitionCtxModel(const GraphViewer& graph_viewer,
                  const size_t num_nodes_in_graph,
                  const std::string& provider_name,
                  const onnxruntime::utils::GenerateMetadefNameFn& gen_metadef_name,
                  const logging::Logger& logger) {
  std::unordered_set<const Node*> supported_nodes;
  std::vector<std::vector<const Node*>> supported_groups;

  for (const auto& node : graph_viewer.Nodes()) {
    NodeAttrHelper node_helper(node);
    std::string ctx_source = node_helper.Get(constants::SOURCE, "");
    if (constants::EPCONTEXT_OP == node.OpType() && ctx_source == kZhouyiExecutionProvider) {
      supported_nodes.insert(&node);
      std::vector<const Node*> supported_group = {&node};
      supported_groups.emplace_back(std::move(supported_group));
    }
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  result.reserve(supported_groups.size());
  std::transform(
      supported_groups.begin(), supported_groups.end(),
      std::back_inserter(result),
      [&](const auto& supported_partition) {
        return onnxruntime::utils::MakeComputeCapability(graph_viewer, supported_partition, gen_metadef_name, provider_name, false);
      });
  return result;
}

}  // namespace zhouyi
}  // namespace onnxruntime
