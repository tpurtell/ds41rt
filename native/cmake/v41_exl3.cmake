# Each native architecture builds a self-contained set of loadable modules.
# Kernels remain unloaded unless an EXL3 checkpoint is selected at runtime.
if(NOT DS41RT_ENABLE_CUDA)
  message(FATAL_ERROR "EXL3 AOT requires CUDA")
endif()
if(DS41RT_CUDA_ARCHITECTURES STREQUAL "120" OR DS41RT_CUDA_ARCHITECTURES STREQUAL "120f")
  set(DS41RT_EXL3_ROLE coordinator)
  set(DS41RT_EXL3_LAYOUTS rtx-tp1 rtx-tp2 dspark)
elseif(DS41RT_CUDA_ARCHITECTURES STREQUAL "121")
  set(DS41RT_EXL3_ROLE spark)
  set(DS41RT_EXL3_LAYOUTS tp4-rank0 tp4-rank1 tp4-rank2 tp4-rank3)
else()
  message(FATAL_ERROR "EXL3 AOT requires a single native SM120 or SM121 target")
endif()
set(DS41RT_V41_EXL3_CAPACITIES "1;16;80;256;1024;4096" CACHE STRING "EXL3 batch capacities to package")
# V7 targets the uniform K=2 raw publication family (resident tiers [2,3]).
# The staged K3.25 family needs -DDS41RT_V41_EXL3_BITS="3;4".
set(DS41RT_V41_EXL3_BITS "2;3" CACHE STRING "EXL3 decoder tiers for the packaged checkpoint family")
option(DS41RT_V41_EXL3_PAIRED_TP4 "Build paired H128 ownership modules for Spark TP4" OFF)
set(DS41RT_V41_EXL3_RESIDENCY "" CACHE STRING "Explicit paired EXL3 capacity=blocks/SM overrides (for example 80=2)")
set(DS41RT_EXL3_LAYOUT_ARGS)
if(DS41RT_V41_EXL3_PAIRED_TP4)
  list(LENGTH DS41RT_V41_EXL3_BITS DS41RT_EXL3_TIER_COUNT)
  if(NOT DS41RT_EXL3_ROLE STREQUAL "spark" OR NOT DS41RT_EXL3_TIER_COUNT EQUAL 2)
    message(FATAL_ERROR "Paired EXL3 TP4 requires SM121 and exactly two decoder tiers")
  endif()
  list(APPEND DS41RT_EXL3_LAYOUT_ARGS --paired-tp4)
elseif(DS41RT_V41_EXL3_RESIDENCY)
  message(FATAL_ERROR "EXL3 residency overrides require paired TP4")
endif()
set(DS41RT_EXL3_OVERRIDE_CAPACITIES)
foreach(override IN LISTS DS41RT_V41_EXL3_RESIDENCY)
  if(NOT override MATCHES "^([1-9][0-9]*)=([12])$")
    message(FATAL_ERROR "EXL3 residency must be capacity=1 or capacity=2")
  endif()
  set(capacity "${CMAKE_MATCH_1}")
  if(NOT capacity IN_LIST DS41RT_V41_EXL3_CAPACITIES OR capacity IN_LIST DS41RT_EXL3_OVERRIDE_CAPACITIES)
    message(FATAL_ERROR "EXL3 residency requires a selected, nonduplicate capacity")
  endif()
  list(APPEND DS41RT_EXL3_OVERRIDE_CAPACITIES "${capacity}")
  list(APPEND DS41RT_EXL3_LAYOUT_ARGS --residency "${override}")
endforeach()
set(DS41RT_EXL3_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/exl3")
set(DS41RT_EXL3_BYPRODUCTS)
foreach(layout IN LISTS DS41RT_EXL3_LAYOUTS)
  foreach(rows IN LISTS DS41RT_V41_EXL3_CAPACITIES)
    foreach(name v41_exl3.json trellis_lut.bin libds41rt_exl3.so)
      list(APPEND DS41RT_EXL3_BYPRODUCTS "${DS41RT_EXL3_PACKAGE}/${layout}/m${rows}/${name}")
    endforeach()
  endforeach()
endforeach()
list(JOIN DS41RT_V41_EXL3_CAPACITIES "," DS41RT_EXL3_CAPACITIES_ARG)
list(GET CUDAToolkit_INCLUDE_DIRS 0 DS41RT_EXL3_CUDA_INCLUDE)
set(DS41RT_EXL3_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/package_v41_exl3_aot.py")
# Large prefill export arenas must not overlap other GPU compiler jobs.
get_property(DS41RT_EXL3_PREDECESSORS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER DS41RT_EXL3_PREDECESSORS INCLUDE REGEX "_export$")
add_custom_command(
  OUTPUT "${DS41RT_EXL3_PACKAGE}/manifest.json"
  BYPRODUCTS ${DS41RT_EXL3_BYPRODUCTS}
  COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
  COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
    "${Python3_EXECUTABLE}" "${DS41RT_EXL3_TOOL}" build
    --role "${DS41RT_EXL3_ROLE}" --capacities "${DS41RT_EXL3_CAPACITIES_ARG}"
    --bits ${DS41RT_V41_EXL3_BITS}
    ${DS41RT_EXL3_LAYOUT_ARGS}
    --build-dir "${CMAKE_CURRENT_BINARY_DIR}/v41_exl3_exports"
    --output "${DS41RT_EXL3_PACKAGE}"
    --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${DS41RT_EXL3_CUDA_INCLUDE}"
    --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>"
    --cuda-driver "$<TARGET_FILE:CUDA::cuda_driver>"
    --runtime "${DS41RT_B12X_AOT_RUNTIME_LIBRARY}"
  DEPENDS "${DS41RT_EXL3_TOOL}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_exl3_aot.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_exl3_routes_aot.py"
    ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    ${DS41RT_EXL3_PREDECESSORS}
  COMMENT "Building native EXL3 modules and verified runtime package"
  VERBATIM
)
add_custom_target(ds41rt_v41_exl3_export DEPENDS "${DS41RT_EXL3_PACKAGE}/manifest.json")
add_dependencies(ds41rt_v41_exl3_export ds41rt_verify_sparkinfer_source)
