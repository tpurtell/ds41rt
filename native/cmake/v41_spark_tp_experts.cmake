# Replicated-group Spark TP2/TP3/TP6 expert shards (native FP8 K32 family, SM121).
#
# Opt-in only: `DS41RT_V41_SPARK_TP_ROLES` is empty by default so no Spark
# TP2/TP3/TP6 object or symbol is compiled or linked and every release default
# behaves exactly as the historical Spark TP4 shard. When the option lists
# `tp2`, `tp3` and/or `tp6`, this exports those roles and compiles one distinct
# symbol family per degree into the same libds41rt_native.so.
#
# The exporter runs on the target SM121 device and pre-compiles capacities
# 1/16/80/256/1024/4096 ahead of runtime. The TP degree is a plan-time role
# property; live row counts are never part of the AOT compile key. TP6 is pure
# tensor parallelism over the official intermediate 2304: six 384-wide ranks
# with no storage padding (kernel_intermediate == intermediate == 384).
if(NOT DS41RT_ENABLE_V41_EXPERT_AOT OR NOT DS41RT_V41_EXPERT_ROLE STREQUAL "spark")
  message(FATAL_ERROR "Spark TP2/TP3/TP6 experts require the native SM121 Spark expert build")
endif()

set(DS41RT_V41_SPARK_TP_SELECTED)
foreach(tp IN LISTS DS41RT_V41_SPARK_TP_ROLES)
  if(tp STREQUAL "tp2" OR tp STREQUAL "tp3" OR tp STREQUAL "tp6")
    if(tp IN_LIST DS41RT_V41_SPARK_TP_SELECTED)
      message(FATAL_ERROR "DS41RT_V41_SPARK_TP_ROLES lists ${tp} more than once")
    endif()
    list(APPEND DS41RT_V41_SPARK_TP_SELECTED "${tp}")
  else()
    message(FATAL_ERROR "DS41RT_V41_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got ${tp}")
  endif()
endforeach()

set(DS41RT_V41_SPARK_TP_EXPERT_TARGETS)
set(DS41RT_V41_SPARK_TP_EXPERT_INCLUDE_DIRS)
set(DS41RT_V41_SPARK_TP_EXPERT_ROWS 1 16 80 256 1024 4096)
set(DS41RT_V41_SPARK_TP_EXPERT_ROWS_ARG "1,16,80,256,1024,4096")
# Capacity 1 is the narrow decode tile; every wider capacity uses width 192.
# Per-role cache strings so a build can select a width map for TP2, TP3 and TP6
# independently (for example a capacity-80 candidate) without patching this file.
# All defaults are byte-identical to the previous single shared map, so an
# unconfigured build exports exactly the same objects. Each value is either a
# scalar 64/128/192 or a full `capacity:width` map; the exporter validates that a
# map covers every precompiled capacity exactly once and that widths are
# 64/128/192, so an invalid override fails the export instead of being silently
# accepted here. For TP6 the extra `slices * width <= kernel_intermediate`
# relation (384 % 32 == 0, so 384/64, 384/128 and 384/192 all tile exactly) is
# checked by the exporter's slice kernel too; an untileable width fails there.
set(DS41RT_V41_SPARK_TP2_SLICE_WIDTH "1:64,16:192,80:192,256:192,1024:192,4096:192" CACHE STRING
  "Spark TP2 expert slice width: scalar 64/128/192 or capacity:width map")
set(DS41RT_V41_SPARK_TP3_SLICE_WIDTH "1:64,16:192,80:192,256:192,1024:192,4096:192" CACHE STRING
  "Spark TP3 expert slice width: scalar 64/128/192 or capacity:width map")
set(DS41RT_V41_SPARK_TP6_SLICE_WIDTH "1:64,16:192,80:192,256:192,1024:192,4096:192" CACHE STRING
  "Spark TP6 expert slice width: scalar 64/128/192 or capacity:width map")

foreach(tp IN LISTS DS41RT_V41_SPARK_TP_SELECTED)
  if(tp STREQUAL "tp2")
    set(role spark_tp2)
    set(width_map "${DS41RT_V41_SPARK_TP2_SLICE_WIDTH}")
    set(wrapper_src src/v41_spark_tp2_experts.cc)
    set(variant_header v41_spark_tp2_expert_variants.h)
  elseif(tp STREQUAL "tp3")
    set(role spark_tp3)
    set(width_map "${DS41RT_V41_SPARK_TP3_SLICE_WIDTH}")
    set(wrapper_src src/v41_spark_tp3_experts.cc)
    set(variant_header v41_spark_tp3_expert_variants.h)
  else()
    set(role spark_tp6)
    set(width_map "${DS41RT_V41_SPARK_TP6_SLICE_WIDTH}")
    set(wrapper_src src/v41_spark_tp6_experts.cc)
    set(variant_header v41_spark_tp6_expert_variants.h)
  endif()
  # Content-stable per-role stamp. The output stems are constant
  # `v41_{role}_m{capacity}`, so a width change must still invalidate an existing
  # export. `file(GENERATE)` rewrites the stamp only when its content changes,
  # giving Make generators (which track dependency timestamps rather than command
  # strings) the same re-export trigger Ninja gets from the changed command line,
  # while an unchanged configure leaves the timestamp alone and does not churn the
  # AOT objects. The atomic threshold is part of the stamped identity so it cannot
  # be lost when a width map is overridden.
  set(width_stamp "${CMAKE_CURRENT_BINARY_DIR}/v41_${role}_width.stamp")
  file(GENERATE OUTPUT "${width_stamp}"
    CONTENT "role=${role}\nwidth=${width_map}\natomic_min_capacity=256\n")
  set(dir "${CMAKE_CURRENT_BINARY_DIR}/v41_${role}_experts")
  set(objects)
  set(headers)
  foreach(rows IN LISTS DS41RT_V41_SPARK_TP_EXPERT_ROWS)
    set(stem "${dir}/v41_${role}_m${rows}")
    list(APPEND objects "${stem}.o")
    list(APPEND headers "${stem}.h")
  endforeach()
  set(json "${dir}/v41_experts.json")
  add_custom_command(
    OUTPUT "${json}" "${dir}/${variant_header}" ${objects} ${headers}
    COMMAND ${DS41RT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env ${DS41RT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_slices_aot.py"
      --output-dir "${dir}" --role "${role}"
      --rows "${DS41RT_V41_SPARK_TP_EXPERT_ROWS_ARG}"
      --width "${width_map}"
      --atomic-min-capacity 256 --standard-names
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${dir}/v41_expert_variants.h" "${dir}/${variant_header}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_slices_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_v41_experts_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_src}"
      "${width_stamp}"
      ${DS41RT_SPARKINFER_PROVENANCE_INPUTS} ${DS41RT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ${role} replicated-group Spark expert kernels"
    VERBATIM
  )
  add_custom_target(ds41rt_v41_${role}_experts_export DEPENDS
    "${dir}/${variant_header}" "${json}" ${objects} ${headers})
  add_dependencies(ds41rt_v41_${role}_experts_export ds41rt_verify_sparkinfer_source)
  set_source_files_properties(${objects} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND DS41RT_V41_SPARK_TP_EXPERT_INCLUDE_DIRS "${dir}")
  list(APPEND DS41RT_V41_SPARK_TP_EXPERT_TARGETS "ds41rt_v41_${role}_experts_export")
  list(APPEND DS41RT_NATIVE_SOURCES ${objects} "${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_src}")
endforeach()
