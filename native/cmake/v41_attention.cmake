if(NOT DS41RT_ENABLE_CUDA OR NOT
   (DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f"))
  message(FATAL_ERROR "V4.1 native attention AOT requires SM120")
endif()
set(DS41RT_V41_ATTENTION_DIR "${CMAKE_CURRENT_BINARY_DIR}/v41_attention")
add_custom_command(
  OUTPUT "${DS41RT_V41_ATTENTION_DIR}/v41_attention.json"
    "${DS41RT_V41_ATTENTION_DIR}/v41_attention.h" "${DS41RT_V41_ATTENTION_DIR}/v41_attention.o"
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_attention_aot.py"
    --output-dir "${DS41RT_V41_ATTENTION_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_attention_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
  COMMENT "Exporting direct V4.1 FP4 attention and sink merge"
  VERBATIM)
add_custom_target(ds41rt_v41_attention_export DEPENDS "${DS41RT_V41_ATTENTION_DIR}/v41_attention.json"
  "${DS41RT_V41_ATTENTION_DIR}/v41_attention.h" "${DS41RT_V41_ATTENTION_DIR}/v41_attention.o")
add_dependencies(ds41rt_v41_attention_export ds41rt_verify_sparkinfer_source)
set_source_files_properties("${DS41RT_V41_ATTENTION_DIR}/v41_attention.o"
  PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
list(APPEND DS41RT_NATIVE_SOURCES "${DS41RT_V41_ATTENTION_DIR}/v41_attention.o" src/v41_attention_aot.cc)
