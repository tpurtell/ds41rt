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
# V7 ships two decoder families side by side: the uniform K=2 raw
# publication family (resident tiers [2,3]) and the staged K3.25 family
# ([3,4]). Each family lands in its own sibling package directory
# (exl3-k23, exl3-k34) and the daemon selects by checkpoint tiers.
set(DS41RT_V41_EXL3_BIT_FAMILIES "2,3;3,4" CACHE STRING "EXL3 decoder tier families to package; one comma-joined tier list per family")
set(DS41RT_V41_EXL3_BITS "" CACHE STRING "Single EXL3 decoder tier family override (replaces DS41RT_V41_EXL3_BIT_FAMILIES)")
option(DS41RT_V41_EXL3_PAIRED_TP4 "Build paired H128 ownership modules for Spark TP4" OFF)
set(DS41RT_V41_EXL3_RESIDENCY "" CACHE STRING "Explicit paired EXL3 capacity=blocks/SM overrides (for example 80=2)")
if(DS41RT_V41_EXL3_BITS)
  list(JOIN DS41RT_V41_EXL3_BITS "," DS41RT_V41_EXL3_BITS_JOINED)
  set(DS41RT_V41_EXL3_BIT_FAMILIES "${DS41RT_V41_EXL3_BITS_JOINED}")
endif()
set(DS41RT_EXL3_LAYOUT_ARGS)
if(DS41RT_V41_EXL3_PAIRED_TP4)
  list(LENGTH DS41RT_V41_EXL3_BIT_FAMILIES DS41RT_EXL3_FAMILY_COUNT)
  if(NOT DS41RT_EXL3_FAMILY_COUNT EQUAL 1)
    message(FATAL_ERROR "Paired EXL3 TP4 requires exactly one decoder tier family")
  endif()
  list(GET DS41RT_V41_EXL3_BIT_FAMILIES 0 DS41RT_EXL3_PAIRED_FAMILY)
  string(REPLACE "," ";" DS41RT_EXL3_PAIRED_TIERS "${DS41RT_EXL3_PAIRED_FAMILY}")
  list(LENGTH DS41RT_EXL3_PAIRED_TIERS DS41RT_EXL3_TIER_COUNT)
  if(NOT DS41RT_EXL3_ROLE STREQUAL "spark" OR NOT DS41RT_EXL3_TIER_COUNT EQUAL 2)
    message(FATAL_ERROR "Paired EXL3 TP4 requires SM121 and exactly two decoder tiers")
  endif()
  list(APPEND DS41RT_EXL3_LAYOUT_ARGS --paired-tp4)
elseif(DS41RT_V41_EXL3_RESIDENCY)
  message(FATAL_ERROR "EXL3 residency overrides require paired TP4")
endif()
# Disjoint Spark packages serve both TP4 and equal-width TP2 ownership.
# Paired H128 packages remain TP4-only and must not declare TP2 byproducts.
if(DS41RT_EXL3_ROLE STREQUAL "spark" AND NOT DS41RT_V41_EXL3_PAIRED_TP4)
  list(APPEND DS41RT_EXL3_LAYOUTS tp2-rank0 tp2-rank1)
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
list(JOIN DS41RT_V41_EXL3_CAPACITIES "," DS41RT_EXL3_CAPACITIES_ARG)
list(GET CUDAToolkit_INCLUDE_DIRS 0 DS41RT_EXL3_CUDA_INCLUDE)
set(DS41RT_EXL3_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/package_v41_exl3_aot.py")
# Large prefill export arenas must not overlap other GPU compiler jobs.
get_property(DS41RT_EXL3_PREDECESSORS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
list(FILTER DS41RT_EXL3_PREDECESSORS INCLUDE REGEX "_export$")
# Family exports must also serialize against each other: chain each family
# on the previous family's package manifest.
set(DS41RT_EXL3_FAMILY_MANIFESTS)
set(DS41RT_EXL3_FAMILY_CHAIN ${DS41RT_EXL3_PREDECESSORS})
foreach(family IN LISTS DS41RT_V41_EXL3_BIT_FAMILIES)
  string(REPLACE "," "" DS41RT_EXL3_FAMILY_TAG "${family}")
  string(REPLACE "," ";" DS41RT_EXL3_FAMILY_TIERS "${family}")
  if(NOT DS41RT_EXL3_FAMILY_TAG MATCHES "^[0-9]+$")
    message(FATAL_ERROR "EXL3 tier family '${family}' must be comma-joined integers")
  endif()
  set(DS41RT_EXL3_PACKAGE "${CMAKE_CURRENT_BINARY_DIR}/exl3-k${DS41RT_EXL3_FAMILY_TAG}")
  set(DS41RT_EXL3_BYPRODUCTS)
  foreach(layout IN LISTS DS41RT_EXL3_LAYOUTS)
    foreach(rows IN LISTS DS41RT_V41_EXL3_CAPACITIES)
      foreach(name v41_exl3.json trellis_lut.bin libds41rt_exl3.so)
        list(APPEND DS41RT_EXL3_BYPRODUCTS "${DS41RT_EXL3_PACKAGE}/${layout}/m${rows}/${name}")
      endforeach()
    endforeach()
  endforeach()
  add_custom_command(
    OUTPUT "${DS41RT_EXL3_PACKAGE}/manifest.json"
    BYPRODUCTS ${DS41RT_EXL3_BYPRODUCTS}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}" "${DS41RT_EXL3_TOOL}" build
      --role "${DS41RT_EXL3_ROLE}" --capacities "${DS41RT_EXL3_CAPACITIES_ARG}"
      --bits ${DS41RT_EXL3_FAMILY_TIERS}
      ${DS41RT_EXL3_LAYOUT_ARGS}
      --build-dir "${CMAKE_CURRENT_BINARY_DIR}/v41_exl3_exports/k${DS41RT_EXL3_FAMILY_TAG}"
      --output "${DS41RT_EXL3_PACKAGE}"
      --cxx "${CMAKE_CXX_COMPILER}" --cuda-include "${DS41RT_EXL3_CUDA_INCLUDE}"
      --cuda-libdir "$<TARGET_FILE_DIR:CUDA::cudart>"
      --cuda-driver "$<TARGET_FILE:CUDA::cuda_driver>"
      --runtime "${DS41RT_B12X_AOT_RUNTIME_LIBRARY}"
    DEPENDS "${DS41RT_EXL3_TOOL}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_exl3_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_exl3_routes_aot.py"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
      ${DS41RT_EXL3_FAMILY_CHAIN}
    COMMENT "Building native EXL3 modules and verified runtime package (tiers ${family})"
    VERBATIM
  )
  list(APPEND DS41RT_EXL3_FAMILY_MANIFESTS "${DS41RT_EXL3_PACKAGE}/manifest.json")
  set(DS41RT_EXL3_FAMILY_CHAIN "${DS41RT_EXL3_PACKAGE}/manifest.json")
endforeach()
add_custom_target(ds41rt_v41_exl3_export DEPENDS ${DS41RT_EXL3_FAMILY_MANIFESTS})
add_dependencies(ds41rt_v41_exl3_export ds41rt_verify_sparkinfer_source)
