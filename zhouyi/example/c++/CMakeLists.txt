# Copyright (c) Arm china. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.13)

project(onnxruntime_zhouyi_samples  C CXX)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

#onnxruntime providers
option(onnxruntime_USE_ZHOUYI "Build with Zhouyi support" OFF)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
option(ONNXRUNTIME_LIB_DIR "libonnxruntime.so dir")

set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime")
endif()

if(onnxruntime_USE_ZHOUYI)
  add_definitions(-DUSE_ZHOUYI)
endif()

message("ONNXRUNTIME_ROOTDIR:"  ${ONNXRUNTIME_ROOTDIR})
message("onnxruntime_USE_ZHOUYI:"  ${onnxruntime_USE_ZHOUYI})
message("C Compiler Version:" ${CMAKE_C_COMPILER_VERSION})
message("CXX Compiler Version:" ${CMAKE_CXX_COMPILER_VERSION})
# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"  # Windows local install
)

# directories under include/onnxruntime/core/providers/zhouyi
include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/zhouyi"
)
include_directories(
  {PROJECT_SOURCE_DIR}
)

add_executable(zhouyi_example  main.cc)

# Library path generated after compilation
target_link_directories(zhouyi_example PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(zhouyi_example onnxruntime)


