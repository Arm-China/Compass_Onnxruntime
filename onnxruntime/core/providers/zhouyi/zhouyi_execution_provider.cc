// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#include "zhouyi_execution_provider.h"

#include <unistd.h>

#include <functional>
#include <limits>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/memcpy.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/zhouyi/constants.h"
#include "core/providers/zhouyi/ep_context.h"
#include "core/providers/zhouyi/zhouyi_model.h"
#include "core/providers/zhouyi/zhouyi_utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
constexpr const char* ZHOUYI = "Zhouyi";
namespace {
ZhouyiProfileReportType ProfileSuffixToEnum(std::string suf) {
  std::map<std::string, ZhouyiProfileReportType> suffixtab = {
      {"csv", ZHOUYI_PROFILE_REPORT_TYPE_CSV},
      {"html", ZHOUYI_PROFILE_REPORT_TYPE_HTML},
      {"json", ZHOUYI_PROFILE_REPORT_TYPE_JSON}};

  std::transform(suf.begin(), suf.end(), suf.begin(), ::tolower);
  if (suffixtab.count(suf))
    return suffixtab[suf];
  else {
    return ZHOUYI_PROFILE_REPORT_TYPE_CSV;
  }
}
}  // namespace

void AipuInterfaceInit() {
  static bool b_init = false;
  if (b_init)
    return;
  aipuruntime::GlobalConfig config = {
      .log_level = static_cast<uint32_t>(zhouyi::utils::GetEnv(zhouyi::constants::TOOLKIT_LOG_LEVEL, 0)),
      .intermidiate_path = zhouyi::utils::GetEnv(zhouyi::constants::INTERMIDIATE_PATH, "./"),
      .operator_path = zhouyi::utils::GetEnv(zhouyi::constants::OPERATOR_PATH, "./operator"),
      .simulator_path = zhouyi::utils::GetEnv(zhouyi::constants::SIMULATOR_PATH, "simulator/"),
      .profile_report_type = ProfileSuffixToEnum(
          zhouyi::utils::GetEnv(zhouyi::constants::PROFILE_REPORT_SUFFIX, "csv")),
  };
  aipuruntime::init(config);
}

ZhouyiExecutionProvider::ZhouyiExecutionProvider(const SessionOptions* session_options)
    : IExecutionProvider{kZhouyiExecutionProvider} {
  AipuInterfaceInit();
  if (session_options) {
    ctx_cache_enable_ = session_options->config_options.GetConfigOrDefault(
                            kOrtSessionOptionEpContextEnable, "0") == "1";
    ctx_embed_mode_ = session_options->config_options.GetConfigOrDefault(
                          kOrtSessionOptionEpContextEmbedMode, "1") == "1";
    ctx_cache_path_ = session_options->config_options.GetConfigOrDefault(
        kOrtSessionOptionEpContextFilePath, "");
    LOGS_DEFAULT(INFO) << "Context cache enable: " << (ctx_cache_enable_ ? "true" : "false")
                       << ", context embed mode: " << (ctx_embed_mode_ ? "1" : "0")
                       << ", context cache path: " << ctx_cache_path_;
  }
}

ZhouyiExecutionProvider::~ZhouyiExecutionProvider() {
}

std::vector<std::unique_ptr<ComputeCapability>>
ZhouyiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/,
                                       const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                                       IResourceAccountant* /* resource_accountant */) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto gen_metadef_name = [&]() {
    HashValue model_hash;
    int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    return MakeString(ZHOUYI, "_", model_hash, "_", metadef_id);
  };

  const size_t num_nodes_in_graph = static_cast<size_t>(graph_viewer.NumberOfNodes());
  const auto& logger = *GetLogger();
  bool is_ctx_model = zhouyi::HasCtxNode(graph_viewer);
  if (is_ctx_model) {
    return zhouyi::PartitionCtxModel(graph_viewer, num_nodes_in_graph, ZHOUYI, gen_metadef_name, logger);
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer, logger);
  zhouyi::ZhouyiModel zhouyi_model(&graph_viewer, logger);
  const auto supported_nodes = zhouyi_model.GetSupportedNodes();
  //(node_unit_map,node_unit_holder.size());

  // Helper function that returns a string that lists all unsupported nodes.
  // Ex: { name: mul_123, type: Mul }, {}, ...
  auto get_unsupported_node_names = [&node_unit_holder, &supported_nodes]() -> std::string {
    std::stringstream ss;
    const size_t num_node_units = node_unit_holder.size();

    for (size_t i = 0; i < num_node_units; ++i) {
      const auto& node_unit = node_unit_holder[i];

      if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
        ss << "{ name: " << node_unit->Name() << ", type: " << node_unit->OpType() << " }\n";
        if (i == num_node_units - 1) {
          ss << ", ";
        }
      }
    }

    return ss.str();
  };

  if (supported_nodes.empty()) {
    LOGS(logger, INFO) << "No partition supported by ZHOUYI";
    return result;
  }

  size_t num_of_supported_nodes = 0;

  // Create partitions from supported nodes.
  {
    std::vector<std::unique_ptr<ComputeCapability>> partitions =
        utils::CreateSupportedPartitions(graph_viewer, supported_nodes, {},
                                         gen_metadef_name, ZHOUYI, kZhouyiExecutionProvider, &node_unit_map, true);

    // Filter out partitions that consist of a single QuantizeLinear or DequantizeLinear node.
    // We also count the number of supported nodes in all valid partitions.
    for (auto& partition : partitions) {
      bool is_valid_partition = true;
      size_t nodes_in_partition = 0;

      if (partition && partition->sub_graph) {
        nodes_in_partition = partition->sub_graph->nodes.size();

        if (nodes_in_partition == 1) {
          const Node* node = graph_viewer.GetNode(partition->sub_graph->nodes[0]);

          if (!node) {
            LOGS(logger, ERROR) << "ZHOUYI: Invalid node in partition of one node.";
            is_valid_partition = false;
          }
        }
      } else {
        LOGS(logger, ERROR) << "ZHOUYI: Invalid partition.";
        is_valid_partition = false;
      }

      if (is_valid_partition) {
        result.push_back(std::move(partition));
        num_of_supported_nodes += nodes_in_partition;
      }
    }
  }

  const size_t num_of_partitions = result.size();
  const auto summary_msg = MakeString("Number of partitions supported by ZHOUYI: ", num_of_partitions,
                                      ", number of nodes in the graph: ", num_nodes_in_graph,
                                      ", number of nodes supported by ZHOUYI: ", num_of_supported_nodes);
  LOGS(logger, INFO) << summary_msg;
  if (num_nodes_in_graph != num_of_supported_nodes) {
    LOGS(logger, WARNING) << "Unsupported nodes in ZHOUYI:\n"
                          << get_unsupported_node_names();
  }

  std::set<std::string> supported_node_types;
  std::set<std::string> unsupported_node_types;
  for (size_t i = 0; i < node_unit_holder.size(); ++i) {
    const auto& node_unit = node_unit_holder[i];
    if (supported_nodes.find(&node_unit->GetNode()) == supported_nodes.end()) {
      unsupported_node_types.insert(node_unit->OpType());
    } else {
      supported_node_types.insert(node_unit->OpType());
    }
  }

  LOGS(logger, VERBOSE) << "Supported node type in ZHOUYI:";
  for (auto t : supported_node_types) {
    LOGS(logger, VERBOSE) << t;
  }
  if (unsupported_node_types.size() > 1) {
    LOGS(logger, VERBOSE) << "Unsupported node type in ZHOUYI:";
    for (auto t : unsupported_node_types) {
      LOGS(logger, VERBOSE) << t;
    }
  }

  return result;
}

Status CreateComputeFunc(std::vector<NodeComputeInfo>& node_compute_funcs,
                         AipuSessionMap& zhouyi_sessions_,
                         const logging::Logger& logger) {
  auto TensorDataSize = [&](auto ort_tensor) -> uint32_t {
    auto tensor_type_and_shape = ort_tensor.GetTensorTypeAndShapeInfo();
    size_t count = tensor_type_and_shape.GetElementCount();
    ONNXTensorElementDataType element_type = tensor_type_and_shape.GetElementType();
    uint32_t element_size = zhouyi::utils::GetElementSizeByType(element_type);
    return element_size * count;
  };
  NodeComputeInfo compute_info;
  compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
    LOGS(logger, VERBOSE) << "compute_info.create_state_func context->node_name: " << context->node_name;
    *state = zhouyi_sessions_[context->node_name].get();
    return 0;
  };

  compute_info.release_state_func = [](FunctionState state) {
    aipuruntime::Session* session = reinterpret_cast<aipuruntime::Session*>(state);
    return Status::OK();
  };

  compute_info.compute_func = [&](FunctionState state, const OrtApi*, OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    aipuruntime::Session* session = reinterpret_cast<aipuruntime::Session*>(state);
    const size_t input_count = ctx.GetInputCount();
    std::vector<aipuruntime::TensorData> inputs;
    for (size_t i = 0; i < input_count; ++i) {
      auto ort_tensor = ctx.GetInput(i);
      inputs.push_back({const_cast<void*>(ort_tensor.GetTensorData<void>()),
                        TensorDataSize(ort_tensor)});
    }

    const size_t output_count = ctx.GetOutputCount();
    std::vector<aipuruntime::TensorData> outputs;
    for (size_t i = 0; i < output_count; ++i) {
      auto dims = session->get_output_dims(i);
      std::vector<int64_t> vec(dims.begin(), dims.end());
      auto ort_tensor = ctx.GetOutput(i, vec);
      outputs.push_back({const_cast<void*>(ort_tensor.GetTensorData<void>()),
                         TensorDataSize(ort_tensor)});
    }
    int32_t result = session->execute(inputs, outputs);
    return (result == 0) ? Status::OK() : Status_FAIL("Execute failed.");
  };

  node_compute_funcs.push_back(compute_info);

  return Status::OK();
}

common::Status ZhouyiExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                std::vector<NodeComputeInfo>& node_compute_funcs) {
  LOGS_DEFAULT(VERBOSE) << "ZhouyiExecutionProvider::Compile";

  const auto& logger = *GetLogger();

  // TODO: support multiple partitions?

  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
    bool is_ctx_model = zhouyi::HasCtxNode(graph_viewer);

    zhouyi::ZhouyiModel zhouyi_model(&graph_viewer, logger);
    aipuruntime::SessionConfig config = {
        .enable_profile = zhouyi::utils::GetEnv(zhouyi::constants::ENABLE_PROFILE, false),
        .tiling_method = zhouyi::utils::GetEnv(zhouyi::constants::TILING_METHOD, "fps"),
        .dump_file = zhouyi::utils::GetEnv(zhouyi::constants::DUMP_FILE, false),
    };
    // seesion for compile and execute
    aipuruntime::SessionPtr session = std::make_shared<aipuruntime::Session>(config);
    std::string ctx_cache_path;
    if (is_ctx_model || ctx_cache_enable_) {
      ctx_cache_path = zhouyi::GetFullpathOfCtxModel(ctx_cache_path_, graph_viewer.ModelPath().string(), is_ctx_model);
      ORT_RETURN_IF(ctx_cache_path.empty() || std::filesystem::is_directory(ctx_cache_path),
                    "Cannot find usable ctx fullpath(include filename): " + ctx_cache_path);
    }
    if (is_ctx_model) {
      ORT_RETURN_IF_ERROR(zhouyi::ComposeCtx(graph_viewer, session, ctx_cache_path));
    } else {
      ORT_RETURN_IF_ERROR(zhouyi_model.Build());
      // compile
      int32_t result = session->compile(zhouyi_model.Graph());
      ORT_RETURN_IF(result != 0, "Session compile failed!");

      if (ctx_cache_enable_) {
        zhouyi_ctx_model_ = std::make_unique<onnxruntime::Model>("zhouyi_ctx_model", false, logger);
        std::filesystem::path p(ctx_cache_path);
        ORT_RETURN_IF(std::filesystem::is_regular_file(p) && std::filesystem::exists(p),
                      "If you try to re-generate ctx files, please remove old files: " + ctx_cache_path);
        ORT_RETURN_IF_ERROR(zhouyi::CreateCtx(
            graph_viewer, session, fused_node.Name(),
            zhouyi_ctx_model_.get(), ctx_embed_mode_, ctx_cache_path));
      }
    }

    int32_t result = session->release_binary();
    ORT_RETURN_IF(result != 0, "%s: Session release binary failed!");
    zhouyi_sessions_.emplace(fused_node.Name(), session);

    LOGS(logger, VERBOSE) << "fused node name: " << fused_node.Name();
    ORT_RETURN_IF_ERROR(CreateComputeFunc(node_compute_funcs, zhouyi_sessions_, logger));
  }
  return Status::OK();
}  // namespace onnxruntime

const InlinedVector<const Node*> ZhouyiExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_nodes;
  if (zhouyi_ctx_model_) {
    const auto& graph = zhouyi_ctx_model_->MainGraph();
    for (const auto& node : graph.Nodes()) {
      ep_context_nodes.push_back(graph.GetNode(node.Index()));
    }
  }

  return ep_context_nodes;
}

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kZhouyiExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kZhouyiExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kZhouyiExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(
    kZhouyiExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static void RegisterZhouyiKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kZhouyiExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kZhouyiExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    auto status = kernel_registry.Register(function_table_entry());
    if (Status::OK() != status) {
      LOGS_DEFAULT(WARNING) << status.ErrorMessage();
    }
  }
}

std::shared_ptr<KernelRegistry> GetZhouyiKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry =
      std::make_shared<KernelRegistry>();
  RegisterZhouyiKernels(*kernel_registry);

  return kernel_registry;
}

std::shared_ptr<KernelRegistry>
ZhouyiExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry =
      onnxruntime::GetZhouyiKernelRegistry();
  return kernel_registry;
}

}  // namespace onnxruntime
