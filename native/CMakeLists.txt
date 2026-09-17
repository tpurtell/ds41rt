cmake_minimum_required(VERSION 3.24)
project(ds41rt_native LANGUAGES C CXX)

# Single-config serving builds should optimize the native host path too.
# Respect an explicitly selected Debug/RelWithDebInfo or multi-config generator.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Native build configuration" FORCE)
endif()

option(DS41RT_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT "Build full RTX backbone expert kernels alongside dSpark" OFF)
option(DS41RT_ENABLE_V41_TP2_EXPERT_AOT "Build dual RTX TP2 encoder expert kernels" OFF)
option(DS41RT_V41_TP2_COMPACT_EXPERIMENT "Use experimental native compact TP2 experts at capacities 1/16" OFF)
option(DS41RT_V41_SPARK_COMPACT_EXPERIMENT "Use experimental compact Spark experts through two live rows" OFF)
option(DS41RT_V41_VOCAB_ROW_EXPERIMENT "Use experimental single-row FP32 vocabulary projection" OFF)
option(DS41RT_ENABLE_V41_EXL3_AOT "Build loadable mixed EXL3 expert modules" OFF)
option(DS41RT_ENABLE_V41_EXPERT_AOT "Build native V4.1 expert AOT kernels" OFF)
option(DS41RT_ENABLE_V41_NARROW_AOT "Use candidate V4.1 narrow projection scheduling" OFF)
option(DS41RT_ENABLE_V41_HC_LAGGED_AOT "Build candidate native lagged mHC begin" OFF)
option(DS41RT_ENABLE_V41_FP8_AOT "Build native V4.1 coordinator FP8 AOT kernels" OFF)
if((DS41RT_ENABLE_V41_HC_LAGGED_AOT OR DS41RT_ENABLE_V41_NARROW_AOT) AND NOT DS41RT_ENABLE_V41_FP8_AOT)
  message(FATAL_ERROR "Candidate mHC/narrow AOT requires V4.1 FP8 AOT")
endif()
option(DS41RT_ENABLE_V41_ATTENTION_AOT "Build candidate direct V4.1 attention AOT kernels" OFF)
option(DS41RT_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(DS41RT_ENABLE_SPARKINFER_AOT "Build Spark-side SparkInfer AOT kernels" OFF)
option(DS41RT_ENABLE_DS4_FLASH_AOT
  "Build DeepSeek-V4-Flash Spark expert-TP4 SparkInfer AOT kernels" OFF)
option(DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT
  "Build coordinator SparkInfer AOT kernels" OFF)
option(DS41RT_ENABLE_B12X_AOT
  "Deprecated alias for DS41RT_ENABLE_SPARKINFER_AOT" OFF)
option(DS41RT_ENABLE_B12X_COORDINATOR_AOT
  "Deprecated alias for DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT" OFF)
option(DS41RT_ENABLE_W8A16_AOT "Build coordinator row-major W8A16 Triton AOT kernels" OFF)
option(DS41RT_ENABLE_FLASHINFER_PACKED_MLA
  "Build the exact grouped-split SM120 packed-FP8 MLA kernel" OFF)
option(DS41RT_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
option(DS41RT_ENABLE_XGRAMMAR "Build constrained-decoding grammar support" ON)
set(DS41RT_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(DS41RT_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer"
  CACHE PATH "Pinned SparkInfer source tree used by every SparkInfer-backed AOT export")
set(DS41RT_SPARKINFER_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer.lock.json"
  CACHE FILEPATH "Provenance lock for DS41RT_SPARKINFER_SOURCE_DIR")
set(DS41RT_XGRAMMAR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar"
  CACHE PATH "Pinned XGrammar source tree")
set(DS41RT_XGRAMMAR_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar.lock.json"
  CACHE FILEPATH "Provenance lock for DS41RT_XGRAMMAR_SOURCE_DIR")

if(DS41RT_ENABLE_B12X_AOT)
  message(DEPRECATION
    "DS41RT_ENABLE_B12X_AOT is deprecated; use DS41RT_ENABLE_SPARKINFER_AOT")
  set(DS41RT_ENABLE_SPARKINFER_AOT ON)
endif()
if(DS41RT_ENABLE_B12X_COORDINATOR_AOT)
  message(DEPRECATION
    "DS41RT_ENABLE_B12X_COORDINATOR_AOT is deprecated; "
    "use DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT")
  set(DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DS41RT_NATIVE_SOURCES src/ds41rt_native.cc)

if(DS41RT_ENABLE_XGRAMMAR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${DS41RT_XGRAMMAR_SOURCE_DIR}/include/xgrammar/compiler.h" OR
     NOT EXISTS "${DS41RT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" OR
     NOT EXISTS "${DS41RT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson/picojson.h")
    message(FATAL_ERROR
      "Pinned XGrammar source is incomplete under ${DS41RT_XGRAMMAR_SOURCE_DIR}; "
      "initialize third_party/xgrammar and its nested submodules")
  endif()
  set(DS41RT_XGRAMMAR_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-xgrammar-source.py")
  if(NOT EXISTS "${DS41RT_XGRAMMAR_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "XGrammar source verifier was not found: ${DS41RT_XGRAMMAR_VERIFY_SCRIPT}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${DS41RT_XGRAMMAR_VERIFY_SCRIPT}"
      --source "${DS41RT_XGRAMMAR_SOURCE_DIR}"
      --lock "${DS41RT_XGRAMMAR_LOCK_FILE}"
    RESULT_VARIABLE DS41RT_XGRAMMAR_VERIFY_RESULT
    OUTPUT_VARIABLE DS41RT_XGRAMMAR_VERIFY_OUTPUT
    ERROR_VARIABLE DS41RT_XGRAMMAR_VERIFY_ERROR
  )
  if(NOT DS41RT_XGRAMMAR_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "XGrammar source verification failed:\n${DS41RT_XGRAMMAR_VERIFY_ERROR}")
  endif()
  string(STRIP "${DS41RT_XGRAMMAR_VERIFY_OUTPUT}" DS41RT_XGRAMMAR_VERIFY_OUTPUT)
  message(STATUS "${DS41RT_XGRAMMAR_VERIFY_OUTPUT}")
  file(GLOB_RECURSE DS41RT_XGRAMMAR_SOURCES
    CONFIGURE_DEPENDS
    "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
  )
  list(FILTER DS41RT_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
  # Extend the pinned schema factory in a generated translation unit. The
  # anonymous upstream parser is reused without changing the vendored tree.
  file(READ "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp/json_schema_converter.cc" DS41RT_SCHEMA_SOURCE)
  set(DS41RT_SCHEMA_FACTORY "switch (json_format) {\n    case JSONFormat::kJSON:")
  string(FIND "${DS41RT_SCHEMA_SOURCE}" "${DS41RT_SCHEMA_FACTORY}" DS41RT_SCHEMA_FACTORY_AT)
  if(DS41RT_SCHEMA_FACTORY_AT EQUAL -1)
    message(FATAL_ERROR "Pinned XGrammar schema factory no longer matches the V4.1 extension")
  endif()
  string(REPLACE "#include \"json_schema_converter.h\""
    "#include \"json_schema_converter.h\"\n#include \"xgrammar_v41_tools.h\""
    DS41RT_SCHEMA_SOURCE "${DS41RT_SCHEMA_SOURCE}")
  string(REPLACE "${DS41RT_SCHEMA_FACTORY}"
    "switch (json_format) {\n    case static_cast<JSONFormat>(5): {\n      V41ToolCallingConverter converter(ref_resolver);\n      return converter.Convert(spec);\n    }\n    case JSONFormat::kJSON:"
    DS41RT_SCHEMA_SOURCE "${DS41RT_SCHEMA_SOURCE}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc" "${DS41RT_SCHEMA_SOURCE}")
  list(FILTER DS41RT_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/json_schema_converter\\.cc$")
  # Correct UTF-8 range tails and Unicode negation in a generated copy. Source
  # provenance is checked above; the patcher additionally verifies its anchors.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_unicode_fsm.inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/prepare-xgrammar-unicode.py")
  execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/prepare-xgrammar-unicode.py"
    --source "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp/grammar_functor.cc"
    --replacement "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_unicode_fsm.inc"
    --output "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc"
    COMMAND_ERROR_IS_FATAL ANY)
  list(FILTER DS41RT_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/grammar_functor\\.cc$")
  list(APPEND DS41RT_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_unicode.cc")
  list(APPEND DS41RT_XGRAMMAR_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/v41_xgrammar_schema.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_regular.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/xgrammar_v41_tools.cc")
  add_library(ds41rt_xgrammar STATIC ${DS41RT_XGRAMMAR_SOURCES})
  target_include_directories(ds41rt_xgrammar PRIVATE
    "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_include_directories(ds41rt_xgrammar
    PUBLIC "${DS41RT_XGRAMMAR_SOURCE_DIR}/include"
  )
  target_include_directories(ds41rt_xgrammar SYSTEM PUBLIC
    "${DS41RT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include"
    "${DS41RT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  )
  target_compile_definitions(ds41rt_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  )
  find_package(Threads REQUIRED)
  target_link_libraries(ds41rt_xgrammar PUBLIC Threads::Threads)
  list(APPEND DS41RT_NATIVE_SOURCES src/xgrammar_adapter.cc)
endif()

set(DS41RT_BUILD_SPARKINFER_AOT OFF)
if(DS41RT_ENABLE_SPARKINFER_AOT OR DS41RT_ENABLE_DS4_FLASH_AOT OR
   DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS41RT_ENABLE_W8A16_AOT OR
   DS41RT_ENABLE_V41_EXPERT_AOT OR DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT OR
   DS41RT_ENABLE_V41_TP2_EXPERT_AOT OR DS41RT_ENABLE_V41_EXL3_AOT OR DS41RT_ENABLE_V41_FP8_AOT OR DS41RT_ENABLE_V41_ATTENTION_AOT)
  set(DS41RT_BUILD_SPARKINFER_AOT ON)
endif()
if(DS41RT_BUILD_SPARKINFER_AOT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/__init__.py")
    message(FATAL_ERROR
      "SparkInfer was not found under DS41RT_SPARKINFER_SOURCE_DIR="
      "${DS41RT_SPARKINFER_SOURCE_DIR}. Initialize third_party/sparkinfer or "
      "select an explicit pinned source tree.")
  endif()
  set(DS41RT_SPARKINFER_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-sparkinfer-source.py")
  set(DS41RT_SPARKINFER_PYTHON_BOOTSTRAP
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_pinned_sparkinfer.py")
  if(NOT EXISTS "${DS41RT_SPARKINFER_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "SparkInfer source verifier was not found: ${DS41RT_SPARKINFER_VERIFY_SCRIPT}")
  endif()
  if(NOT EXISTS "${DS41RT_SPARKINFER_PYTHON_BOOTSTRAP}")
    message(FATAL_ERROR
      "SparkInfer Python bootstrap was not found: "
      "${DS41RT_SPARKINFER_PYTHON_BOOTSTRAP}")
  endif()
  set(DS41RT_SPARKINFER_VERIFY_COMMAND
    "${Python3_EXECUTABLE}" "${DS41RT_SPARKINFER_VERIFY_SCRIPT}"
    --source "${DS41RT_SPARKINFER_SOURCE_DIR}"
    --lock "${DS41RT_SPARKINFER_LOCK_FILE}"
  )
  set(DS41RT_SPARKINFER_PYTHON_ENV
    "DS41RT_SPARKINFER_SOURCE_DIR=${DS41RT_SPARKINFER_SOURCE_DIR}"
    "DS41RT_SPARKINFER_LOCK_FILE=${DS41RT_SPARKINFER_LOCK_FILE}"
    # Persisted objects are executable-only ExternalBinaryModules. AOT export
    # needs the live compiler IR consumed by export_to_c(), so never substitute
    # a runtime disk-cache hit in these build-time Python processes.
    "B12X_COMPILE_DISK_CACHE=0"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${DS41RT_SPARKINFER_SOURCE_DIR}"
  )
  execute_process(
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    RESULT_VARIABLE DS41RT_SPARKINFER_VERIFY_RESULT
    OUTPUT_VARIABLE DS41RT_SPARKINFER_VERIFY_OUTPUT
    ERROR_VARIABLE DS41RT_SPARKINFER_VERIFY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT DS41RT_SPARKINFER_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Pinned SparkInfer verification failed:\n${DS41RT_SPARKINFER_VERIFY_ERROR}")
  endif()
  message(STATUS "${DS41RT_SPARKINFER_VERIFY_OUTPUT}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${DS41RT_SPARKINFER_LOCK_FILE}"
    "${DS41RT_SPARKINFER_VERIFY_SCRIPT}"
  )
  set(DS41RT_SPARKINFER_PROVENANCE_INPUTS
    "${DS41RT_SPARKINFER_LOCK_FILE}"
    "${DS41RT_SPARKINFER_VERIFY_SCRIPT}"
    "${DS41RT_SPARKINFER_PYTHON_BOOTSTRAP}"
  )
  file(GLOB_RECURSE DS41RT_SPARKINFER_EXPORT_INPUTS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/*.py"
    "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/*.cu"
    "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/*.cuh"
    "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/*.h"
    "${DS41RT_SPARKINFER_SOURCE_DIR}/b12x/*.hpp"
  )
  list(APPEND DS41RT_SPARKINFER_EXPORT_INPUTS
    "${DS41RT_SPARKINFER_SOURCE_DIR}/pyproject.toml"
  )
  add_custom_target(ds41rt_verify_sparkinfer_source
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMENT "Verifying pinned SparkInfer source"
    VERBATIM
  )
endif()

if(DS41RT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND DS41RT_NATIVE_SOURCES
    cuda/kernels/v41_expert_pack.cu
    cuda/kernels/v41_exl3_wire.cu
    cuda/kernels/v41_route_reduce.cu
    cuda/kernels/engram.cu
    cuda/kernels/v41_dspark.cu
    cuda/kernels/v41_hc.cu
    cuda/kernels/v41_router.cu
    cuda/kernels/v41_dspark_cache.cu
    cuda/kernels/v41_dspark_attention.cu
    cuda/kernels/v41_compressor.cu
    cuda/kernels/v41_vision.cu
    cuda/kernels/v41_kv.cu
    cuda/kernels/v41_sparse_attention.cu
    cuda/kernels/v41_peer_copy.cu
    cuda/kernels/v41_index_scores.cu
    cuda/kernels/v41_index_topk.cu
    cuda/kernels/v41_candidate_blocks.cu
    cuda/kernels/v41_attention_ops.cu
    cuda/kernels/norm.cu
    cuda/kernels/mlp.cu
    cuda/kernels/residual.cu
    cuda/kernels/kv_cache.cu
    cuda/kernels/mla_indexing.cu
    cuda/kernels/router.cu
    cuda/kernels/linear.cu
    cuda/kernels/attention.cu
    cuda/kernels/embedding.cu
    cuda/kernels/sampling.cu
    cuda/kernels/nibbles.cu
    cuda/kernels/nvfp4_payload.cu
    cuda/kernels/ds4_flash_direct.cu
  )
endif()

set(DS41RT_BUILD_FLASHINFER_PACKED_MLA OFF)
if(DS41RT_ENABLE_FLASHINFER_PACKED_MLA OR DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS41RT_ENABLE_W8A16_AOT)
  set(DS41RT_BUILD_FLASHINFER_PACKED_MLA ON)
endif()

if(DS41RT_BUILD_FLASHINFER_PACKED_MLA)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS41RT_ENABLE_FLASHINFER_PACKED_MLA=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent / 'data' / 'include')"
    OUTPUT_VARIABLE DS41RT_FLASHINFER_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT EXISTS "${DS41RT_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh")
    message(FATAL_ERROR
      "FlashInfer SM120 sparse-MLA headers were not found under ${DS41RT_FLASHINFER_INCLUDE_DIR}")
  endif()
  list(APPEND DS41RT_NATIVE_SOURCES
    cuda/kernels/packed_fp8_mla_exact.cu
  )
  set_source_files_properties(cuda/kernels/packed_fp8_mla_exact.cu PROPERTIES
    COMPILE_OPTIONS "-use_fast_math;-O3"
  )
  if(DS41RT_CUDA_ARCHITECTURES STREQUAL "120")
    set(DS41RT_CUDA_ARCHITECTURES "120f")
  endif()
endif()

if(DS41RT_ENABLE_SPARKINFER_AOT)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS41RT_ENABLE_SPARKINFER_AOT=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS41RT_B12X_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/b12x_aot")
  set(DS41RT_B12X_AOT_REGIMES 1 2 4 8 16 32 64 128 256)
  set(DS41RT_B12X_AOT_TOPK8_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(DS41RT_B12X_AOT_OBJECTS)
  set(DS41RT_B12X_AOT_HEADERS)
  list(APPEND DS41RT_B12X_AOT_OBJECTS
    "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.o"
    "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.o"
  )
  foreach(regime IN LISTS DS41RT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND DS41RT_B12X_AOT_OBJECTS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS DS41RT_B12X_AOT_REGIMES)
    list(APPEND DS41RT_B12X_AOT_OBJECTS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.o"
    )
  endforeach()
  list(APPEND DS41RT_B12X_AOT_HEADERS
    "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.h"
    "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.h"
    "${DS41RT_B12X_AOT_DIR}/b12x_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS DS41RT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND DS41RT_B12X_AOT_HEADERS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS DS41RT_B12X_AOT_REGIMES)
    list(APPEND DS41RT_B12X_AOT_HEADERS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${DS41RT_B12X_AOT_OBJECTS}
      ${DS41RT_B12X_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      --output-dir "${DS41RT_B12X_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting Spark-side SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_b12x_aot_export
    DEPENDS
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${DS41RT_B12X_AOT_OBJECTS}
      ${DS41RT_B12X_AOT_HEADERS}
  )
  add_dependencies(ds41rt_b12x_aot_export ds41rt_verify_sparkinfer_source)
  set_source_files_properties(${DS41RT_B12X_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS41RT_NATIVE_SOURCES
    cuda/kernels/b12x_direct.cu
    ${DS41RT_B12X_AOT_OBJECTS}
  )

  set(DS41RT_B12X_M1_PARITY_AOT_REGIMES 2 3 4 5 6 7 8)
  set(DS41RT_B12X_M1_PARITY_AOT_OBJECTS)
  set(DS41RT_B12X_M1_PARITY_AOT_HEADERS
    "${DS41RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot_config.h"
  )
  foreach(regime IN LISTS DS41RT_B12X_M1_PARITY_AOT_REGIMES)
    list(APPEND DS41RT_B12X_M1_PARITY_AOT_OBJECTS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.o"
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.o"
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.o"
    )
    list(APPEND DS41RT_B12X_M1_PARITY_AOT_HEADERS
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.h"
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.h"
      "${DS41RT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${DS41RT_B12X_M1_PARITY_AOT_OBJECTS}
      ${DS41RT_B12X_M1_PARITY_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      --output-dir "${DS41RT_B12X_AOT_DIR}"
      --target-sms 48
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ordered Spark W4A16 M=2..8 parity candidates"
    VERBATIM
  )
  add_custom_target(ds41rt_b12x_m1_parity_aot_export
    DEPENDS
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${DS41RT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${DS41RT_B12X_M1_PARITY_AOT_OBJECTS}
      ${DS41RT_B12X_M1_PARITY_AOT_HEADERS}
  )
  add_dependencies(
    ds41rt_b12x_m1_parity_aot_export
    ds41rt_verify_sparkinfer_source
  )
  set_source_files_properties(${DS41RT_B12X_M1_PARITY_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS41RT_NATIVE_SOURCES ${DS41RT_B12X_M1_PARITY_AOT_OBJECTS})

endif()

if(DS41RT_ENABLE_DS4_FLASH_AOT)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS41RT_ENABLE_DS4_FLASH_AOT=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS41RT_DS4_FLASH_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_flash_aot")
  set(DS41RT_DS4_FLASH_AOT_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(DS41RT_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES 3 5 6 7 9 10 11 12)
  set(DS41RT_DS4_FLASH_AOT_OBJECTS
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.o"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.o")
  set(DS41RT_DS4_FLASH_AOT_HEADERS
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_decode_m1_fused_sum.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m1.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m1.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_topk6_sum.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m1.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m1.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_up_m8.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m1.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_shared_down_m8.h"
    "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS DS41RT_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND DS41RT_DS4_FLASH_AOT_OBJECTS
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.o"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.o"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.o"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND DS41RT_DS4_FLASH_AOT_HEADERS
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_w4a16_prefill_m${regime}_topk6.h"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k2_prefill_m${regime}_topk6.h"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_k3_prefill_m${regime}_topk6.h"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  foreach(regime IN LISTS DS41RT_DS4_FLASH_AOT_MIXED_EXACT_DECODE_REGIMES)
    list(APPEND DS41RT_DS4_FLASH_AOT_OBJECTS
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.o")
    list(APPEND DS41RT_DS4_FLASH_AOT_HEADERS
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_tp4_exl3_mixed_k2_k3_m${regime}.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${DS41RT_DS4_FLASH_AOT_OBJECTS}
      ${DS41RT_DS4_FLASH_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${DS41RT_DS4_FLASH_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Flash Spark expert-TP4 AOT kernel"
    VERBATIM
  )
  add_custom_target(ds41rt_ds4_flash_aot_export
    DEPENDS
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_aot.stamp"
      "${DS41RT_DS4_FLASH_AOT_DIR}/ds4_flash_spark_moe_aot.meta"
      ${DS41RT_DS4_FLASH_AOT_OBJECTS}
      ${DS41RT_DS4_FLASH_AOT_HEADERS}
  )
  add_dependencies(ds41rt_ds4_flash_aot_export ds41rt_verify_sparkinfer_source)

  # Pro experts are trellis-only. Keep their generated symbols, resident
  # geometry, and native bridge separate from the Flash FP4-compatible path.
  set(DS41RT_DS4_PRO_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ds4_pro_aot")
  set(DS41RT_DS4_PRO_AOT_OBJECTS
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.o"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.o")
  set(DS41RT_DS4_PRO_AOT_HEADERS
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_decode_m1.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_topk6_sum.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m1.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m8.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_up_m2048.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m1.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m8.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_down_m2048.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_input_quant_m2048.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_shared_activated_quant_m2048.h"
    "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot_config.h")
  foreach(regime IN LISTS DS41RT_DS4_FLASH_AOT_PREFILL_REGIMES)
    list(APPEND DS41RT_DS4_PRO_AOT_OBJECTS
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.o")
    list(APPEND DS41RT_DS4_PRO_AOT_HEADERS
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_tp4_exl3_k2_prefill_m${regime}_topk6.h")
  endforeach()
  add_custom_command(
    OUTPUT
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${DS41RT_DS4_PRO_AOT_OBJECTS}
      ${DS41RT_DS4_PRO_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      --output-dir "${DS41RT_DS4_PRO_AOT_DIR}"
      --target-sms 48
      --model-variant pro
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_ds4_flash_spark_moe_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting DeepSeek-V4-Pro EXL3-only Spark expert-TP4 AOT kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_ds4_pro_aot_export
    DEPENDS
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_aot.stamp"
      "${DS41RT_DS4_PRO_AOT_DIR}/ds4_pro_spark_moe_aot.meta"
      ${DS41RT_DS4_PRO_AOT_OBJECTS}
      ${DS41RT_DS4_PRO_AOT_HEADERS}
  )
  add_dependencies(ds41rt_ds4_pro_aot_export ds41rt_verify_sparkinfer_source)

  set_source_files_properties(${DS41RT_DS4_FLASH_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  set_source_files_properties(${DS41RT_DS4_PRO_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS41RT_NATIVE_SOURCES
    cuda/kernels/ds4_flash_aot.cu
    cuda/kernels/ds4_pro_aot.cu
    ${DS41RT_DS4_FLASH_AOT_OBJECTS}
    ${DS41RT_DS4_PRO_AOT_OBJECTS}
  )
endif()

if(DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR
      "DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS41RT_B12X_COORDINATOR_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/b12x_coordinator_aot"
  )
  set(DS41RT_B12X_COORDINATOR_AOT_OBJECTS
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.o"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.o"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.o"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.o"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.o"
  )
  set(DS41RT_B12X_COORDINATOR_AOT_HEADERS
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.h"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.h"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.h"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.h"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.h"
    "${DS41RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${DS41RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${DS41RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${DS41RT_B12X_COORDINATOR_AOT_OBJECTS}
      ${DS41RT_B12X_COORDINATOR_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      --output-dir "${DS41RT_B12X_COORDINATOR_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_b12x_coordinator_aot_export
    DEPENDS
      "${DS41RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${DS41RT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${DS41RT_B12X_COORDINATOR_AOT_OBJECTS}
      ${DS41RT_B12X_COORDINATOR_AOT_HEADERS}
  )
  add_dependencies(
    ds41rt_b12x_coordinator_aot_export
    ds41rt_verify_sparkinfer_source
  )
  set_source_files_properties(${DS41RT_B12X_COORDINATOR_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS41RT_NATIVE_SOURCES
    cuda/kernels/b12x_coordinator.cu
    ${DS41RT_B12X_COORDINATOR_AOT_OBJECTS}
  )
endif()

if(DS41RT_ENABLE_W8A16_AOT)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS41RT_ENABLE_W8A16_AOT=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(DS41RT_W8A16_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/w8a16_aot")
  set(DS41RT_W8A16_AOT_HEADER
    "${DS41RT_W8A16_AOT_DIR}/w8a16_row_major_aot.h"
  )
  set(DS41RT_W8A16_AOT_SOURCE
    "${DS41RT_W8A16_AOT_DIR}/w8a16_row_major_aot.cc"
  )
  set(DS41RT_W8A16_PACKED_O_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/w8a16_packed_o_aot"
  )
  set(DS41RT_W8A16_PACKED_O_AOT_OBJECTS
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.o"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.o"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.o"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.o"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.o"
  )
  set(DS41RT_W8A16_PACKED_O_AOT_HEADERS
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.h"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.h"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.h"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.h"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.h"
    "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${DS41RT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${DS41RT_W8A16_AOT_HEADER}"
      "${DS41RT_W8A16_AOT_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      --output-dir "${DS41RT_W8A16_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_triton_prefill.py"
    COMMENT "Exporting coordinator row-major W8A16 AOT kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_w8a16_aot_export
    DEPENDS
      "${DS41RT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${DS41RT_W8A16_AOT_HEADER}"
      "${DS41RT_W8A16_AOT_SOURCE}"
  )
  add_custom_command(
    OUTPUT
      "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${DS41RT_W8A16_PACKED_O_AOT_OBJECTS}
      ${DS41RT_W8A16_PACKED_O_AOT_HEADERS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      --output-dir "${DS41RT_W8A16_PACKED_O_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_cute_packed_prefill.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS}
      ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator packed W8A16 O-projection AOT kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_w8a16_packed_o_aot_export
    DEPENDS
      "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${DS41RT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${DS41RT_W8A16_PACKED_O_AOT_OBJECTS}
      ${DS41RT_W8A16_PACKED_O_AOT_HEADERS}
  )
  add_dependencies(
    ds41rt_w8a16_packed_o_aot_export
    ds41rt_verify_sparkinfer_source
  )
  set_source_files_properties("${DS41RT_W8A16_AOT_SOURCE}" PROPERTIES
    GENERATED TRUE
  )
  set_source_files_properties(${DS41RT_W8A16_PACKED_O_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND DS41RT_NATIVE_SOURCES
    "${DS41RT_W8A16_AOT_SOURCE}"
    cuda/kernels/w8a16_packed.cu
    ${DS41RT_W8A16_PACKED_O_AOT_OBJECTS}
  )
endif()

if(DS41RT_ENABLE_SPARKINFER_AOT OR DS41RT_ENABLE_DS4_FLASH_AOT OR
   DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS41RT_ENABLE_W8A16_AOT OR
   DS41RT_ENABLE_V41_EXPERT_AOT OR DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT OR
   DS41RT_ENABLE_V41_TP2_EXPERT_AOT OR DS41RT_ENABLE_V41_EXL3_AOT OR DS41RT_ENABLE_V41_FP8_AOT OR DS41RT_ENABLE_V41_ATTENTION_AOT)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE DS41RT_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(DS41RT_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${DS41RT_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

if(DS41RT_ENABLE_V41_EXPERT_AOT)
  include(cmake/v41_experts.cmake)
endif()
if(DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT)
  include(cmake/v41_local_experts.cmake)
endif()
if(DS41RT_ENABLE_V41_TP2_EXPERT_AOT)
  include(cmake/v41_tp2_experts.cmake)
  include(cmake/v41_dspark_tp2_experts.cmake)
endif()
if(DS41RT_ENABLE_V41_FP8_AOT)
  include(cmake/v41_fp8.cmake)
  include(cmake/v41_router.cmake)
endif()
if(DS41RT_ENABLE_V41_ATTENTION_AOT)
  include(cmake/v41_attention.cmake)
endif()
if(DS41RT_ENABLE_V41_EXL3_AOT)
  include(cmake/v41_exl3.cmake)
endif()
add_library(ds41rt_native SHARED ${DS41RT_NATIVE_SOURCES})
if(DS41RT_ENABLE_V41_EXL3_AOT)
  add_dependencies(ds41rt_native ds41rt_v41_exl3_export)
endif()
if(DS41RT_ENABLE_V41_ATTENTION_AOT)
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_HAVE_V41_ATTENTION_AOT=1)
  add_dependencies(ds41rt_native ds41rt_v41_attention_export ds41rt_v41_attention_heads32_export)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_V41_ATTENTION_DIR}")
endif()
if(DS41RT_ENABLE_V41_FP8_AOT)
  # Packed FP8->BF16 conversion requires the SM120 architecture-specific ISA.
  # Keep the normal target's portable image as a fallback in the same fatbin.
  set_property(SOURCE cuda/kernels/v41_sparse_attention.cu APPEND PROPERTY
    COMPILE_OPTIONS "--generate-code=arch=compute_120a,code=sm_120a")
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_HAVE_V41_HC_AOT=1 DS41RT_HAVE_V41_ROUTER_AOT=1)
  add_dependencies(ds41rt_native ds41rt_v41_fp8_export ds41rt_v41_index_export ds41rt_v41_router_export)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_V41_FP8_DIR}" "${DS41RT_V41_INDEX_DIR}" "${DS41RT_V41_ROUTER_DIR}")
endif()
if(DS41RT_ENABLE_V41_EXPERT_AOT)
  add_dependencies(ds41rt_native ds41rt_v41_experts_export)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_V41_EXPERT_DIR}")
endif()
if(DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT)
  add_dependencies(ds41rt_native ds41rt_v41_local_experts_export)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_V41_LOCAL_EXPERT_DIR}")
endif()
if(DS41RT_ENABLE_V41_TP2_EXPERT_AOT)
  add_dependencies(ds41rt_native ds41rt_v41_tp2_experts_export ds41rt_v41_dspark_tp2_experts_export)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_V41_TP2_EXPERT_DIR}" "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}")
endif()
target_include_directories(ds41rt_native PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(DS41RT_ENABLE_XGRAMMAR)
  target_link_libraries(ds41rt_native PRIVATE ds41rt_xgrammar)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp")
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_XGRAMMAR=1)
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_XGRAMMAR=0)
endif()
if(DS41RT_ENABLE_CUDA)
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_CUDA=1)
  if(DS41RT_V41_VOCAB_ROW_EXPERIMENT)
    target_compile_definitions(ds41rt_native PRIVATE DS41RT_V41_VOCAB_ROW_EXPERIMENT=1)
  endif()
  target_link_libraries(ds41rt_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(ds41rt_native PROPERTIES
    CUDA_ARCHITECTURES "${DS41RT_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_CUDA=0)
endif()
if(DS41RT_BUILD_FLASHINFER_PACKED_MLA)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_FLASHINFER_INCLUDE_DIR}")
endif()
if(DS41RT_ENABLE_SPARKINFER_AOT)
  add_dependencies(
    ds41rt_native
    ds41rt_b12x_aot_export
    ds41rt_b12x_m1_parity_aot_export
  )
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_B12X_AOT=1)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_B12X_AOT_DIR}")
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_B12X_AOT=0)
endif()
if(DS41RT_ENABLE_DS4_FLASH_AOT)
  add_dependencies(
    ds41rt_native
    ds41rt_ds4_flash_aot_export
    ds41rt_ds4_pro_aot_export
  )
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_DS4_FLASH_AOT=1)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_DS4_FLASH_AOT_DIR}")
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_DS4_PRO_AOT_DIR}")
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_DS4_FLASH_AOT=0)
endif()
if(DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  add_dependencies(ds41rt_native ds41rt_b12x_coordinator_aot_export)
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=1)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_B12X_COORDINATOR_AOT_DIR}")
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=0)
endif()
if(DS41RT_ENABLE_W8A16_AOT)
  add_dependencies(ds41rt_native ds41rt_w8a16_aot_export)
  add_dependencies(ds41rt_native ds41rt_w8a16_packed_o_aot_export)
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_W8A16_AOT=1)
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_W8A16_AOT_DIR}")
  target_include_directories(ds41rt_native PRIVATE "${DS41RT_W8A16_PACKED_O_AOT_DIR}")
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_W8A16_AOT=0)
endif()
if(DS41RT_ENABLE_SPARKINFER_AOT OR DS41RT_ENABLE_DS4_FLASH_AOT OR
   DS41RT_ENABLE_SPARKINFER_COORDINATOR_AOT OR DS41RT_ENABLE_W8A16_AOT OR
   DS41RT_ENABLE_V41_EXPERT_AOT OR DS41RT_ENABLE_V41_LOCAL_EXPERT_AOT OR
   DS41RT_ENABLE_V41_TP2_EXPERT_AOT OR DS41RT_ENABLE_V41_EXL3_AOT OR DS41RT_ENABLE_V41_FP8_AOT OR DS41RT_ENABLE_V41_ATTENTION_AOT)
  target_link_libraries(ds41rt_native PUBLIC "${DS41RT_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET ds41rt_native APPEND PROPERTY BUILD_RPATH "${DS41RT_B12X_AOT_RUNTIME_DIR}")
endif()
if(DS41RT_ENABLE_RDMA)
  find_path(DS41RT_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(DS41RT_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT DS41RT_IBVERBS_INCLUDE_DIR OR NOT DS41RT_IBVERBS_LIBRARY)
    message(FATAL_ERROR "DS41RT_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_RDMA=1)
  target_include_directories(ds41rt_native PRIVATE ${DS41RT_IBVERBS_INCLUDE_DIR})
  target_link_libraries(ds41rt_native PUBLIC ${DS41RT_IBVERBS_LIBRARY})
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_RDMA=0)
endif()
if(DS41RT_ENABLE_NCCL)
  if(NOT DS41RT_ENABLE_CUDA)
    message(FATAL_ERROR "DS41RT_ENABLE_NCCL=ON requires DS41RT_ENABLE_CUDA=ON")
  endif()
  find_path(DS41RT_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(DS41RT_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_NCCL=1)
  target_include_directories(ds41rt_native PRIVATE ${DS41RT_NCCL_INCLUDE_DIR})
  target_link_libraries(ds41rt_native PUBLIC ${DS41RT_NCCL_LIBRARY})
else()
  target_compile_definitions(ds41rt_native PRIVATE DS41RT_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(ds41rt_native_selftest tests/native_selftest.cc)
target_link_libraries(ds41rt_native_selftest PRIVATE ds41rt_native)
# The self-test intentionally uses assertions for both the ABI call and its
# result check. Keep those expressions live in Release builds; defining
# NDEBUG would otherwise skip allocations and make the following copy probe
# dereference an uninitialized buffer.
if(MSVC)
  target_compile_options(ds41rt_native_selftest PRIVATE /UNDEBUG)
else()
  target_compile_options(ds41rt_native_selftest PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME ds41rt_native_selftest COMMAND ds41rt_native_selftest)

if(DS41RT_ENABLE_XGRAMMAR)
  add_executable(ds41rt_xgrammar_regular_selftest tests/xgrammar_regular_selftest.cc)
  target_link_libraries(ds41rt_xgrammar_regular_selftest PRIVATE ds41rt_xgrammar)
  target_include_directories(ds41rt_xgrammar_regular_selftest PRIVATE
    "${DS41RT_XGRAMMAR_SOURCE_DIR}/cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  add_test(NAME ds41rt_xgrammar_regular_selftest COMMAND ds41rt_xgrammar_regular_selftest)
  add_executable(ds41rt_xgrammar_selftest tests/xgrammar_selftest.cc)
  target_link_libraries(ds41rt_xgrammar_selftest PRIVATE ds41rt_native)
  target_compile_definitions(ds41rt_xgrammar_selftest PRIVATE
    "DS41RT_XGRAMMAR_TEST_TOKENIZER=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/xgrammar_tiny_tokenizer.json\""
  )
  if(MSVC)
    target_compile_options(ds41rt_xgrammar_selftest PRIVATE /UNDEBUG)
  else()
    target_compile_options(ds41rt_xgrammar_selftest PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ds41rt_xgrammar_selftest COMMAND ds41rt_xgrammar_selftest)
endif()

if(DS41RT_ENABLE_CUDA)
  add_executable(ds41rt_v41_peer_copy_selftest tests/v41_peer_copy_selftest.cc)
  target_link_libraries(ds41rt_v41_peer_copy_selftest PRIVATE ds41rt_native CUDA::cudart)
  add_test(NAME ds41rt_v41_peer_copy_selftest COMMAND ds41rt_v41_peer_copy_selftest)
  set_tests_properties(ds41rt_v41_peer_copy_selftest PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(ds41rt_v41_fp4_attention_selftest tests/v41_fp4_attention_selftest.cc)
  target_link_libraries(ds41rt_v41_fp4_attention_selftest PRIVATE ds41rt_native CUDA::cudart)
  if(DS41RT_ENABLE_V41_ATTENTION_AOT)
    target_compile_definitions(ds41rt_v41_fp4_attention_selftest PRIVATE DS41RT_TEST_ATTENTION_AOT=1)
  endif()
  add_test(NAME ds41rt_v41_fp4_attention_selftest COMMAND ds41rt_v41_fp4_attention_selftest)
  add_executable(ds41rt_cuda_inplace_rope_selftest tests/cuda_inplace_rope_selftest.cc)
  target_link_libraries(ds41rt_cuda_inplace_rope_selftest PRIVATE ds41rt_native)
  add_test(NAME ds41rt_cuda_inplace_rope_selftest COMMAND ds41rt_cuda_inplace_rope_selftest)
  set_tests_properties(ds41rt_cuda_inplace_rope_selftest PROPERTIES TIMEOUT 15)
  add_executable(ds41rt_cuda_checked_argmax_selftest tests/cuda_checked_argmax_selftest.cc)
  target_link_libraries(ds41rt_cuda_checked_argmax_selftest PRIVATE ds41rt_native)
  add_test(NAME ds41rt_cuda_checked_argmax_selftest COMMAND ds41rt_cuda_checked_argmax_selftest)
  set_tests_properties(ds41rt_cuda_checked_argmax_selftest PROPERTIES TIMEOUT 15)
  add_executable(ds41rt_cuda_stream_query_selftest tests/cuda_stream_query_selftest.cc)
  target_link_libraries(ds41rt_cuda_stream_query_selftest PRIVATE ds41rt_native CUDA::cudart)
  add_test(NAME ds41rt_cuda_stream_query_selftest COMMAND ds41rt_cuda_stream_query_selftest)
  set_tests_properties(ds41rt_cuda_stream_query_selftest PROPERTIES TIMEOUT 15)
  add_executable(ds41rt_cuda_peer_selftest tests/cuda_peer_selftest.cc)
  target_link_libraries(ds41rt_cuda_peer_selftest PRIVATE ds41rt_native CUDA::cudart)
  add_test(NAME ds41rt_cuda_peer_selftest COMMAND ds41rt_cuda_peer_selftest)
  set_tests_properties(ds41rt_cuda_peer_selftest PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
  add_executable(ds41rt_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(ds41rt_cuda_selftest PRIVATE ds41rt_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME ds41rt_cuda_selftest COMMAND ds41rt_cuda_selftest)
endif()
