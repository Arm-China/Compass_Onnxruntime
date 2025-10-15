// Copyright 2024 armchina Inc.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
namespace onnxruntime {
namespace test {

void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<uint8_t>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<uint8_t> found(rtensor.Data<uint8_t>(), rtensor.Data<uint8_t>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

TEST(ZhouyiExecutionProviderTest, FunctionTest) {
  DefaultLoggingManager().SetDefaultLoggerSeverity(logging::Severity::kVERBOSE);
  onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // UINT8 tensor.
  ONNX_NAMESPACE::TypeProto uint8_tensor;
  uint8_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  uint8_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &uint8_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &uint8_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &uint8_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &uint8_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &uint8_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "zhouyi_execution_provider_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<uint8_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  OrtValue ml_value_x;
  CreateMLValue<uint8_t>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<uint8_t>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<uint8_t>(cpu_allocator, dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {1, 1, 3, 2};
  std::vector<uint8_t> expected_values_mul_m = {3, 6, 9, 12, 15, 18};

  SessionOptions so;
  so.session_logid = "ZhouyiExecutionProviderTest.FunctionTest";
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSession session_object{so, GetEnvironment()};
  auto zhouyi_provider = DefaultZhouyiExecutionProvider();
  status = session_object.RegisterExecutionProvider(std::move(zhouyi_provider));
  ASSERT_TRUE(status.IsOK());
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

}  // namespace test
}  // namespace onnxruntime
