// Copyright (C) 2024-2025 Arm Technology (China) Co. Ltd.
//
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <memory>

#include "core/framework/provider_options.h"
#include "core/framework/session_options.h"
#include "core/providers/providers.h"

namespace onnxruntime {
struct ZhouyiProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const SessionOptions* session_options);
};
}  // namespace onnxruntime
