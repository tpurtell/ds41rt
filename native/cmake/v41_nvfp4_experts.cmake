# NVFP4 (W4A4) expert kernels. A separate symbol family from the W4A8 and EXL3
# variants: the daemon selects it from the checkpoint's expert format, so all
# families coexist in one library.
if(NOT DS41RT_ENABLE_CUDA)
  message(FATAL_ERROR "NVFP4 expert AOT requires CUDA")
endif()
if(DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f")
  set(DS41RT_V41_NVFP4_ROLES rtx_tp2 rtx_backbone)
elseif(DS41RT_CUDA_ARCHITECTURES STREQUAL "121")
  set(DS41RT_V41_NVFP4_ROLES spark)
else()
  message(FATAL_ERROR "NVFP4 expert AOT requires a single native SM120 or SM121 target")
endif()
set(DS41RT_V41_NVFP4_CAPACITIES "1;16;80;256;1024;4096" CACHE STRING "NVFP4 expert capacities to package")
# Keep the conservative tile-16 default until the tile ladder is GPU qualified.
set(DS41RT_V41_NVFP4_TILE_M "16" CACHE STRING "NVFP4 expert tile M (auto opts into the planner ladder; GPU qualification required)")
set_property(CACHE DS41RT_V41_NVFP4_TILE_M PROPERTY STRINGS auto 16 32 64 128)
if(NOT "${DS41RT_V41_NVFP4_TILE_M}" MATCHES "^(auto|16|32|64|128)$")
  message(FATAL_ERROR "DS41RT_V41_NVFP4_TILE_M must be auto, 16, 32, 64, or 128")
endif()
set(DS41RT_V41_NVFP4_OUTPUT_SHARDS "0" CACHE STRING "NVFP4 output splitting: 0 adaptive, 1 disabled, positive divisor of 40 direct-only")
if(NOT "${DS41RT_V41_NVFP4_OUTPUT_SHARDS}" MATCHES "^(0|1|2|4|5|8|10|20|40)$")
  message(FATAL_ERROR "DS41RT_V41_NVFP4_OUTPUT_SHARDS must be 0 or a positive divisor of 40")
endif()
set(DS41RT_V41_NVFP4_INCLUDE_DIRS)
# Quantize each token's activation once with a shared scale and fan it out to
# every routed expert instead of re-quantizing the identical BF16 row per route.
# The host publishes a uniform FC1 activation scale to match (v41_experts/nvfp4.rs).
option(DS41RT_V41_NVFP4_SHARE_INPUT "Export NVFP4 experts with shared-input activation quantization" OFF)
if(DS41RT_V41_NVFP4_SHARE_INPUT)
  set(DS41RT_V41_NVFP4_SHARE_INPUT_ARG "--share-input")
else()
  set(DS41RT_V41_NVFP4_SHARE_INPUT_ARG "")
endif()
list(JOIN DS41RT_V41_NVFP4_CAPACITIES "," DS41RT_V41_NVFP4_CAPACITY_ARG)
option(DS41RT_V41_NVFP4_PAD_INTERMEDIATE "Zero-pad NVFP4 shards to avoid transposed FC1" ON)
set(DS41RT_V41_NVFP4_PAD_ARG "")
if(DS41RT_V41_NVFP4_PAD_INTERMEDIATE)
  set(DS41RT_V41_NVFP4_PAD_ARG "--pad-intermediate")
endif()
foreach(role IN LISTS DS41RT_V41_NVFP4_ROLES)
  set(nvfp4_dir "${CMAKE_CURRENT_BINARY_DIR}/v41_nvfp4_${role}")
  set(nvfp4_objects)
  set(nvfp4_headers)
  foreach(rows IN LISTS DS41RT_V41_NVFP4_CAPACITIES)
    list(APPEND nvfp4_objects "${nvfp4_dir}/v41_nvfp4_${role}_m${rows}.o")
    list(APPEND nvfp4_headers "${nvfp4_dir}/v41_nvfp4_${role}_m${rows}.h")
  endforeach()
  add_custom_command(
    OUTPUT "${nvfp4_dir}/v41_nvfp4_experts.json"
      "${nvfp4_dir}/v41_nvfp4_${role}_variants.h"
      ${nvfp4_objects} ${nvfp4_headers}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_nvfp4_aot.py"
      --output-dir "${nvfp4_dir}" --role "${role}"
      --rows "${DS41RT_V41_NVFP4_CAPACITY_ARG}"
      --tile-m "${DS41RT_V41_NVFP4_TILE_M}"
      --output-shards "${DS41RT_V41_NVFP4_OUTPUT_SHARDS}"
      ${DS41RT_V41_NVFP4_SHARE_INPUT_ARG}
      ${DS41RT_V41_NVFP4_PAD_ARG}
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${nvfp4_dir}/v41_expert_variants.h"
      "${nvfp4_dir}/v41_nvfp4_${role}_variants.h"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_nvfp4_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting NVFP4 ${role} expert kernels"
    VERBATIM
  )
  set_source_files_properties(${nvfp4_objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND DS41RT_NATIVE_SOURCES ${nvfp4_objects})
  if(role STREQUAL "rtx_tp2")
    list(APPEND DS41RT_NATIVE_SOURCES src/v41_nvfp4_rtx_tp2_experts.cc)
  elseif(role STREQUAL "rtx_backbone")
    list(APPEND DS41RT_NATIVE_SOURCES src/v41_nvfp4_rtx_backbone_experts.cc)
  elseif(role STREQUAL "spark")
    list(APPEND DS41RT_NATIVE_SOURCES src/v41_nvfp4_spark_experts.cc)
  else()
    message(FATAL_ERROR "NVFP4 role ${role} has no native translation unit")
  endif()
  list(APPEND DS41RT_V41_NVFP4_INCLUDE_DIRS "${nvfp4_dir}")
  add_custom_target(ds41rt_v41_nvfp4_${role}_export DEPENDS
    "${nvfp4_dir}/v41_nvfp4_${role}_variants.h"
    "${nvfp4_dir}/v41_nvfp4_experts.json"
    ${nvfp4_objects} ${nvfp4_headers})
endforeach()
