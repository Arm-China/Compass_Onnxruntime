// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#include "aipuruntime_cxx_api.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/framework/node_unit.h"
#include "core/graph/basic_types.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
namespace onnxruntime {
namespace zhouyi {

// return path+filename of ctx model
std::string GetFullpathOfCtxModel(const std::string& ctx_cache_path,
                                  const std::string& model_path,
                                  bool is_ctx_model);

bool HasCtxNode(const GraphViewer& graph_viewer);

Status CreateCtx(const GraphViewer& graph_viewer,
                 aipuruntime::SessionPtr session,
                 const std::string& name,
                 onnxruntime::Model* ctx_model,
                 bool ctx_embed_mode,
                 const std::string& ctx_cache_path);

Status ComposeCtx(const GraphViewer& graph_viewer,
                  aipuruntime::SessionPtr session,
                  const std::string& ctx_cache_path);

std::vector<std::unique_ptr<ComputeCapability>>
PartitionCtxModel(const GraphViewer& graph_viewer,
                  const size_t num_nodes_in_graph,
                  const std::string& provider_name,
                  const onnxruntime::utils::GenerateMetadefNameFn& gen_metadef_name,
                  const logging::Logger& logger);

}  // namespace zhouyi
}  // namespace onnxruntime
