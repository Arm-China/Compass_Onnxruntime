// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "aipuruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/session_options.h"
#include "core/graph/model.h"

namespace onnxruntime {
using AipuSessionMap = std::unordered_map<std::string, aipuruntime::SessionPtr>;

class ZhouyiExecutionProvider : public IExecutionProvider {
 public:
  ZhouyiExecutionProvider(const SessionOptions* session_options);
  virtual ~ZhouyiExecutionProvider();

  /**
      Get execution provider's capability for the specified <graph>.
      Return a bunch of IndexedSubGraphs <*this> execution provider can run if
      the sub-graph contains only one node or can fuse to run if the sub-graph
      contains more than one node. The node indexes contained in sub-graphs may
      have overlap, and it's ONNXRuntime's responsibility to do the partition
      and decide whether a node will be assigned to <*this> execution provider.
      For kernels registered in a kernel registry, `kernel_lookup` must be used
      to find a matching kernel for this EP.
   */
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& /*kernel_lookup*/,
                const GraphOptimizerRegistry& /* graph_optimizer_registry */,
                IResourceAccountant* /* resource_accountant */) const override;

  /**
   Given a collection of fused Nodes and the respective GraphViewer instance for the nodes that were fused,
   return create_state/compute/release_state func for each node.
   @remarks This is now the default interface when execution provider wants to compile nodes
            for both minimal build and complete ort build.

            Do NOT cache the GraphViewer in FusedNodeAndGraph.filtered_graph in any of the NodeComputeInfo functions
            as it is only valid for the duration of the call to Compile.
    */
  virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  /**
   Get kernel registry per execution provider type.
   The KernelRegistry share pointer returned is shared across sessions.
   */
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  /**
   * Get the array of pointers for EPContext nodes
   * EP needs to implement this if has the requirement to generate the context cache model. Otherwise leave it.
   * Default return an empty vector if not provided by the Execution Provider
   */
  const InlinedVector<const Node*> GetEpContextNodes() const override;

 private:
  /**
   *
   */
  bool ctx_cache_enable_ = false;

  /**
   *
   */
  bool ctx_embed_mode_ = true;

  /**
   *
   */
  std::string ctx_cache_path_ = "";

  /**
   *
   */
  std::unique_ptr<onnxruntime::Model> zhouyi_ctx_model_;

  /**
   *
   */
  AipuSessionMap zhouyi_sessions_;

  /**
   *
   */
  ModelMetadefIdGenerator metadef_id_generator_;
};
}  // namespace onnxruntime
