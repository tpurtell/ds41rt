# NVFP4 (W4A4) expert kernels. A separate symbol family from the W4A8 and EXL3
# variants: the daemon selects it from the checkpoint's expert format, so all
# families coexist in one library.
if(NOT DS41RT_ENABLE_CUDA)
  message(FATAL_ERROR "NVFP4 expert AOT requires CUDA")
endif()
if(DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f")
  set(DS41RT_V41_NVFP4_ROLES rtx_tp2)
elseif(DS41RT_CUDA_ARCHITECTURES STREQUAL "121")
  set(DS41RT_V41_NVFP4_ROLES spark)
else()
  message(FATAL_ERROR "NVFP4 expert AOT requires a single native SM120 or SM121 target")
endif()
set(DS41RT_V41_NVFP4_CAPACITIES "1;16;80;256;1024;4096" CACHE STRING "NVFP4 expert capacities to package")
list(JOIN DS41RT_V41_NVFP4_CAPACITIES "," DS41RT_V41_NVFP4_CAPACITY_ARG)
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
  elseif(role STREQUAL "spark")
    list(APPEND DS41RT_NATIVE_SOURCES src/v41_nvfp4_spark_experts.cc)
  else()
    message(FATAL_ERROR "NVFP4 role ${role} has no native translation unit")
  endif()
  add_custom_target(ds41rt_v41_nvfp4_${role}_export DEPENDS
    "${nvfp4_dir}/v41_nvfp4_${role}_variants.h"
    "${nvfp4_dir}/v41_nvfp4_experts.json"
    ${nvfp4_objects} ${nvfp4_headers})
endforeach()
