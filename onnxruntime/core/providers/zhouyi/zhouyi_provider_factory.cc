
// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0


#include "core/providers/zhouyi/zhouyi_provider_factory.h"
#include "zhouyi_execution_provider.h"
#include "zhouyi_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct ZhouyiProviderFactory : IExecutionProviderFactory {
  ZhouyiProviderFactory(const SessionOptions* session_options) : session_options_(session_options) {
  }
  ~ZhouyiProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<ZhouyiExecutionProvider>(session_options_);
  }

 private:
  const SessionOptions* session_options_;
};

std::shared_ptr<IExecutionProviderFactory>
ZhouyiProviderFactoryCreator::Create(const SessionOptions* session_options) {
  return std::make_shared<onnxruntime::ZhouyiProviderFactory>(session_options);
}
} // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Zhouyi,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(
      onnxruntime::ZhouyiProviderFactoryCreator::Create(&options->value));
  return nullptr;
}
