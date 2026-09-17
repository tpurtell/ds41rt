# Coordinator-only native K32 activation and block-FP8 projection artifacts.
if(NOT DS41RT_ENABLE_CUDA OR NOT
   (DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f"))
  message(FATAL_ERROR "V4.1 coordinator FP8 AOT requires a native SM120 CUDA target")
endif()
set(DS41RT_V41_FP8_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_fp8")
set(DS41RT_V41_FP8_OBJECTS)
set(DS41RT_V41_FP8_HEADERS)
foreach(projection IN ITEMS engram ffn_up ffn_down main q_a q_b kv o_b o_a index_q q_b_tp2 o_b_tp2 ffn_tp2_up ffn_tp2_down)
foreach(rows IN ITEMS 1 16 80 256 1024 4096)
  set(kinds quant gemm)
  if(projection STREQUAL "o_a")
    list(APPEND kinds quant_rope)
  endif()
  foreach(kind IN LISTS kinds)
    set(stem "${DS41RT_V41_FP8_DIR}/v41_${projection}_fp8_m${rows}_${kind}")
    list(APPEND DS41RT_V41_FP8_OBJECTS "${stem}.o")
    list(APPEND DS41RT_V41_FP8_HEADERS "${stem}.h")
  endforeach()
endforeach()
endforeach()
list(APPEND DS41RT_V41_FP8_OBJECTS "${DS41RT_V41_FP8_DIR}/v41_hc_project.o")
list(APPEND DS41RT_V41_FP8_HEADERS "${DS41RT_V41_FP8_DIR}/v41_hc_project.h")
set(DS41RT_NARROW_ENV "DS41RT_EXPORT_NARROW_AOT=0")
if(DS41RT_ENABLE_V41_NARROW_AOT)
  set(DS41RT_NARROW_ENV "DS41RT_EXPORT_NARROW_AOT=1")
endif()
set(DS41RT_HC_LAGGED_ENV "DS41RT_EXPORT_HC_LAGGED=0")
if(DS41RT_ENABLE_V41_HC_LAGGED_AOT)
  set(DS41RT_HC_LAGGED_ENV "DS41RT_EXPORT_HC_LAGGED=1")
  list(APPEND DS41RT_V41_FP8_OBJECTS "${DS41RT_V41_FP8_DIR}/v41_hc_lagged.o")
  list(APPEND DS41RT_V41_FP8_HEADERS "${DS41RT_V41_FP8_DIR}/v41_hc_lagged.h")
endif()
add_custom_command(
  OUTPUT "${DS41RT_V41_FP8_DIR}/v41_fp8.json" "${DS41RT_V41_FP8_DIR}/v41_fp8_variants.h"
    ${DS41RT_V41_FP8_OBJECTS} ${DS41RT_V41_FP8_HEADERS}
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV} ${DS41RT_HC_LAGGED_ENV} ${DS41RT_NARROW_ENV}
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_fp8_aot.py"
    --output-dir "${DS41RT_V41_FP8_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_fp8_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting native V4.1 K32 quantization and FP8 projections"
  VERBATIM
)
add_custom_target(ds41rt_v41_fp8_export DEPENDS "${DS41RT_V41_FP8_DIR}/v41_fp8.json" "${DS41RT_V41_FP8_DIR}/v41_fp8_variants.h"
  ${DS41RT_V41_FP8_OBJECTS} ${DS41RT_V41_FP8_HEADERS})
add_dependencies(ds41rt_v41_fp8_export ds41rt_verify_sparkinfer_source)
set_source_files_properties(${DS41RT_V41_FP8_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES ${DS41RT_V41_FP8_OBJECTS} src/v41_fp8.cc cuda/kernels/v41_fp8.cu)

# Independent scorer export keeps its pointer ABI auditable.
set(DS41RT_V41_INDEX_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_index")
add_custom_command(
  OUTPUT "${DS41RT_V41_INDEX_DIR}/v41_index_score.json" "${DS41RT_V41_INDEX_DIR}/v41_index_score.h" "${DS41RT_V41_INDEX_DIR}/v41_index_score.o"
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_index_aot.py"
    --output-dir "${DS41RT_V41_INDEX_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_index_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  VERBATIM)
add_custom_target(ds41rt_v41_index_export DEPENDS "${DS41RT_V41_INDEX_DIR}/v41_index_score.json")
add_dependencies(ds41rt_v41_index_export ds41rt_verify_sparkinfer_source)
set_source_files_properties("${DS41RT_V41_INDEX_DIR}/v41_index_score.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES "${DS41RT_V41_INDEX_DIR}/v41_index_score.o" src/v41_index_score.cc)
