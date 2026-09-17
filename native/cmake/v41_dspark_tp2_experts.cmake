# TP2 draft experts coexist with full draft and backbone variants.
if(NOT DS41RT_ENABLE_V41_EXPERT_AOT OR NOT DS41RT_V41_EXPERT_ROLE STREQUAL "coordinator")
  message(FATAL_ERROR "TP2 V4.1 experts require the SM120 coordinator expert build")
endif()
set(DS41RT_V41_DSPARK_TP2_EXPERT_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_dspark_tp2_experts")
set(DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS)
set(DS41RT_V41_DSPARK_TP2_EXPERT_HEADERS)
foreach(rows IN ITEMS 1 16 80 256 1024 4096)
  set(stem "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_dspark_tp2_m${rows}")
  list(APPEND DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS "${stem}.o")
  list(APPEND DS41RT_V41_DSPARK_TP2_EXPERT_HEADERS "${stem}.h")
endforeach()
add_custom_command(
  OUTPUT "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_experts.json"
    "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_dspark_tp2_expert_variants.h"
    ${DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS} ${DS41RT_V41_DSPARK_TP2_EXPERT_HEADERS}
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_slices_aot.py"
    --output-dir "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}" --role dspark_tp2
    --rows 1,16,80,256,1024,4096 --width 192 --standard-names
  COMMAND "${CMAKE_COMMAND}" -E copy
    "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_expert_variants.h"
    "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_dspark_tp2_expert_variants.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_slices_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_experts_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting TP2 RTX draft expert kernels"
  VERBATIM
)
add_custom_target(ds41rt_v41_dspark_tp2_experts_export DEPENDS
  "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_dspark_tp2_expert_variants.h"
  "${DS41RT_V41_DSPARK_TP2_EXPERT_DIR}/v41_experts.json"
  ${DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS} ${DS41RT_V41_DSPARK_TP2_EXPERT_HEADERS})
add_dependencies(ds41rt_v41_dspark_tp2_experts_export ds41rt_verify_sparkinfer_source)
set_source_files_properties(${DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES ${DS41RT_V41_DSPARK_TP2_EXPERT_OBJECTS} src/v41_dspark_tp2_experts.cc)
