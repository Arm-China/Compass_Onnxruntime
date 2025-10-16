// Copyright(c) Arm china.All rights reserved.
// Licensed under the MIT License.

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <getopt.h>
#include <filesystem>
#include <numeric>
#include <string>

#include "providers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace {

struct TestOptions {
  std::string model;
  std::string inputs;
  bool verbose = false;
  bool enable_cache = false;
  int embed_mode = 1;
  std::string cache_path;
};

TestOptions ParseOptions(int argc, char* argv[]) {
  auto help = []() {
    std::cout <<
      "Usage: ./zhouyi_example -m <model> -i <input> [-v] [-f] [-b <embed_mode>] [-p <case_path>]\n"
      "  -m <model>: specify model\n"
      "  -i <input>: specify inputs, split by ',', such as input1.bin,input2.bin...\n"
      "  -v print verbose log\n"
      "  -f enable cached, Only valid when it is normal model\n"
      "  -b <embed_mode>: specify embed mode(default:1): Only valid when cache is enable\n"
      "     0: cache content into *.bin\n"
      "     1: cache content into node\n"
      "  -p <cache_path>: specify the file name and path for ctx generating, such as '-p ./xx.onnx', Only valid when cache is enable\n"
      "  -h: if you want to generate new ctx model, delete the old one first.\n";
    exit(0);
  };

  const char* const short_opts = "m:i:b:p:fvh";
  const option long_opts[] = {
    {"model", required_argument, nullptr, 'm'},
    {"input", required_argument, nullptr, 'i'},
    {"verbose", no_argument, nullptr, 'v'},
    {"enable_cache", no_argument, nullptr, 'f'},
    {"embed_mode", required_argument, nullptr, 'b'},
    {"cache_path", required_argument, nullptr, 'p'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, no_argument, nullptr, 0}
  };
  TestOptions options;
  while(1) {
    int32_t opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
    if (opt == -1) {
      break;
    }
    switch(opt) {
      case 'm':
        options.model = optarg;
      break;
      case 'i':
        options.inputs = optarg;
      break;
      case 'v':
        options.verbose = true;
      break;
      case 'f':
        options.enable_cache = true;
      break;
      case 'b':
        options.embed_mode = atoi(optarg);
      break;
      case 'p':
        options.cache_path = optarg;
      break;
      case 'h':
      case '?':
        default:
          help();
        break;
    }
  }
  return options;
}

static inline void ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
}

static inline void rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); })
              .base(),
          s.end());
}

std::string &trim(std::string &s)
{
  ltrim(s);
  rtrim(s);
  return s;
}

std::vector<std::string> SplitString(const std::string &s, const char &splitter)
{
  std::vector<std::string> ret;
  if (s.find(splitter) == std::string::npos)
  {
    if (s.size())
      ret.push_back(s);
    return ret;
  }

  std::istringstream iss(s);
  while (iss.good())
  {
    std::string substr;
    std::getline(iss, substr, splitter);
    substr = trim(substr);
    ret.push_back(substr);
  }

  return ret;
}

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type) {
  const static std::unordered_map<ONNXTensorElementDataType, size_t> elem_type_to_size = {
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
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)}};

  auto pos = elem_type_to_size.find(elem_type);
  if (pos == elem_type_to_size.end()) {
    std::cout << "Cannot find onnx tensor data type: " << int(elem_type) << std::endl;
    exit(-1);
  }
  return pos->second;
}

std::vector<Ort::Value> CreateTensors(const Ort::Session& session, const std::vector<std::string>& inputs, std::vector<std::vector<uint8_t>>& datas) {
  std::vector<Ort::Value> input_tensors;
  for (uint32_t i = 0; i < session.GetInputCount(); ++i) {
    uint8_t ele_size = GetElementSizeByType(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    int64_t size = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount() * ele_size;
    datas[i].resize(size);

    std::ifstream file(inputs[i], std::ios::ate | std::ios::binary);
    size_t file_size = file.tellg();
    if (file_size != size_t(size)) {
      std::cout << i << "th size not match, provider file: " << file_size << ", onnx: " << size << std::endl;
      exit(-1);
    }
    file.seekg(0);
    file.read(reinterpret_cast<char*>(datas[i].data()), file_size);
    file.close();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    auto type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<int8_t>(memory_info, reinterpret_cast<int8_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<int16_t>(memory_info, reinterpret_cast<int16_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, reinterpret_cast<int32_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type ==ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, reinterpret_cast<int64_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<uint8_t>(memory_info, reinterpret_cast<uint8_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<uint16_t>(memory_info, reinterpret_cast<uint16_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<uint32_t>(memory_info, reinterpret_cast<uint32_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<uint64_t>(memory_info, reinterpret_cast<uint64_t*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::cout << "We haven't supported fp16 yet!" << std::endl;
      exit(-1);
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<double>(memory_info, reinterpret_cast<double*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      input_tensors.emplace_back(Ort::Value::CreateTensor<bool>(memory_info, reinterpret_cast<bool*>(datas[i].data()), datas[i].size(), shape.data(), shape.size()));
    }
  }
  return input_tensors;
}

}  // namespace

int main(int argc, char* argv[]) {
  TestOptions options = ParseOptions(argc, argv);

  if (!std::filesystem::exists(options.model)) {
    std::cout << "Cannot find model: " << options.model << std::endl;
    return -1;
  }

  auto inputs = SplitString(options.inputs, ',');
  for (auto input : inputs) {
    if (!std::filesystem::exists(input)) {
      std::cout << "Cannot find input: " << input << std::endl;
      return -1;
    }
  }

  OrtLoggingLevel log_level = options.verbose ? ORT_LOGGING_LEVEL_VERBOSE : ORT_LOGGING_LEVEL_WARNING;
  Ort::Env env(log_level, "zhouyi_example");
  Ort::SessionOptions sf;
#ifdef USE_ZHOUYI
  sf.AddConfigEntry(kOrtSessionOptionEpContextEnable, options.enable_cache ? "1" : "0");
  sf.AddConfigEntry(kOrtSessionOptionEpContextFilePath, options.cache_path.c_str());
  sf.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, options.embed_mode ? "1" : "0");
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Zhouyi(sf));
#endif

  Ort::Session session(env, options.model.c_str(), sf);

  // Get input/output node names
  using AllocatedStringPtr = std::unique_ptr<char, Ort::detail::AllocatedFree>;
  std::vector<const char*> input_names;
  std::vector<AllocatedStringPtr> inputNodeNameAllocatedStrings;
  std::vector<const char*> output_names;
  std::vector<AllocatedStringPtr> outputNodeNameAllocatedStrings;
  Ort::AllocatorWithDefaultOptions allocator;
  size_t numInputNodes = session.GetInputCount();
  for (size_t i = 0; i < numInputNodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    input_names.emplace_back(inputNodeNameAllocatedStrings.back().get());
  }
  size_t numOutputNodes = session.GetOutputCount();
  for (size_t i = 0; i < numOutputNodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    output_names.emplace_back(outputNodeNameAllocatedStrings.back().get());
  }

  if (inputs.size() != numInputNodes) {
    std::cout << "Inputs number doesn't match, provider:" << inputs.size() << ", onnx: " << numInputNodes << std::endl;
    return -1;
  }

  std::vector<std::vector<uint8_t>> datas(inputs.size());
  std::vector<Ort::Value> input_tensors = CreateTensors(session, inputs, datas);
  std::vector<Ort::Value> output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

  if (output_tensors.size() != numOutputNodes) {
    std::cout << "Output number doesn't match, onnx model:" << numOutputNodes << ", inference: " << output_tensors.size() << std::endl;
    return -1;
  }
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    auto tts = output_tensors[i].GetTensorTypeAndShapeInfo();
    size_t size = tts.GetElementCount() * GetElementSizeByType(tts.GetElementType());
    std::string file_name = std::string("output_") + std::to_string(i) + ".bin";
    auto output_file = std::filesystem::path(inputs[0]).parent_path().append(file_name);
    std::ofstream oss(output_file.string().c_str(), std::ofstream::binary);
    const uint8_t* data = output_tensors[i].GetTensorData<uint8_t>();
    oss.write(reinterpret_cast<const char*>(data), size);
    oss.close();
  }
  return 0;
}
