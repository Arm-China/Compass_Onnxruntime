# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter -Wno-error=ignored-qualifiers")
  add_definitions(-DUSE_ZHOUYI=1)
  option(ZHOUYI_RUNTIME_SIMULATOR "" OFF)

  if (NOT AIPU_TOOLKIT_DIR)
    message(FATAL_ERROR "AIPU_TOOLKIT_DIR required for onnxruntime_USE_ZHOUYI")
  endif()
  set(AIPURUNTIME_INCLUDE_DIR ${AIPU_TOOLKIT_DIR}/include)
  set(ZHOUYI_RUNTIME_LIB_DIR ${AIPU_TOOLKIT_DIR}/lib/${ZHOUYI_RUNTIME_LIB_ARCH})

  file(GLOB_RECURSE
    onnxruntime_providers_zhouyi_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/zhouyi/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/zhouyi/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
    "${ONNXRUNTIME_ROOT}/core/framework/node_unit.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_zhouyi_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_zhouyi ${onnxruntime_providers_zhouyi_cc_srcs})
  
  onnxruntime_add_include_to_target(onnxruntime_providers_zhouyi
    onnxruntime_common
    onnxruntime_framework
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
    flatbuffers::flatbuffers
    Boost::mp11
    safeint_interface
    ${ONNXRUNTIME_ROOT}
  )

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_runtime.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_runtime.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_layerlib.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_layerlib.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_dsl.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_dsl.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_toolchain_core.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_toolchain_core.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaiputoolchain.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaiputoolchain.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_buildtool.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_buildtool.so COPYONLY)

  configure_file(${ZHOUYI_RUNTIME_LIB_DIR}/libaipu_driver.so
   ${CMAKE_CURRENT_BINARY_DIR}/libaipu_driver.so COPYONLY)

  set(ZHOUYI_RUNTIME_LIBS
      aipu_runtime
      aipu_layerlib
      aipu_dsl
      aipu_toolchain_core
      aiputoolchain
      aipu_buildtool
      aipu_driver
  )

  target_link_libraries(onnxruntime_providers_zhouyi PRIVATE ${ZHOUYI_RUNTIME_LIBS})
  add_dependencies(onnxruntime_providers_zhouyi onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_zhouyi PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_zhouyi PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(onnxruntime_providers_zhouyi PRIVATE ${ONNXRUNTIME_ROOT})
  target_include_directories(onnxruntime_providers_zhouyi PRIVATE ${AIPURUNTIME_INCLUDE_DIR})
  target_include_directories(onnxruntime_providers_zhouyi PRIVATE ${AIPURUNTIME_INCLUDE_DIR}/gbuilder)
  link_directories(${ZHOUYI_RUNTIME_LIB_DIR})

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_zhouyi
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
