#!/usr/bin/env python3
"""Export official V4.1 expert kernels and planner-owned native scratch layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import _pinned_sparkinfer


POINTER_SLOTS = (
    "a_ptr",
    "topk_ids_ptr",
    "topk_weights_ptr",
    "packed_a_ptr",
    "sfa_ptr",
    "packed_a_storage_ptr",
    "scale_storage_ptr",
    "intermediate_ptr",
    "barrier_count",
    "barrier_epoch",
    "pair_head",
    "producers_done_count",
    "all_work_published",
    "task_head",
    "task_tail",
    "task_ready_ptr",
    "task_expert_ptr",
    "task_m_tile_ptr",
    "task_slice_begin_ptr",
    "task_slice_count_ptr",
    "task_valid_rows_ptr",
    "tile_write_count_ptr",
    "b_w13",
    "sfb_w13_ptr",
    "b_down",
    "sfb_down_ptr",
    "sfb_w13_mx_ptr",
    "sfb_down_mx_ptr",
    "w13_residual_ptr",
    "down_residual_ptr",
    "w13_rp_ptr",
    "w13_sfb_rp_ptr",
    "down_rp_ptr",
    "down_sfb_rp_ptr",
    "row_counts",
    "expert_write_rows",
    "expert_tile_base",
    "input_global_scale",
    "alpha",
    "down_alpha",
    "global_scale",
    "scatter_ptr",
    "token_map_ptr",
    "token_weights_ptr",
)
# Aliased views have the same base address in b12x's core workspace.
SCRATCH_SLOTS = {
    "packed_a_ptr": "packed_input", "packed_a_storage_ptr": "packed_input",
    "sfa_ptr": "packed_input_scale", "scale_storage_ptr": "packed_input_scale",
    "intermediate_ptr": "materialized_intermediate",
    **{name: name for name in ("barrier_count", "barrier_epoch", "pair_head",
       "producers_done_count", "all_work_published", "task_head", "task_tail",
       "row_counts", "expert_write_rows", "expert_tile_base")},
    **{name + "_ptr": name for name in ("task_ready", "task_expert", "task_m_tile",
       "task_slice_begin", "task_slice_count", "task_valid_rows", "tile_write_count",
       "token_map", "token_weights")},
    "input_global_scale": "input_gs", "global_scale": "down_input_scale",
    "scatter_ptr": "route_output",
}

SCALAR_SLOTS = (
    "num_tokens",
    "max_rows",
    "scatter_rows",
    "rows_padded",
    "max_tasks",
    "max_phys_tiles",
    "max_active_clusters",
    "stream",
)


def write_native_bridge(output_dir: Path, manifest: dict) -> None:
    """Reject exporter ABI drift before generating a native argument bridge."""
    entries, includes = [], []
    geometry = manifest["geometry"]
    for variant in manifest["variants"]:
        name = variant["name"]
        header = (output_dir / (name + ".h")).read_text()
        wrapper = re.search(
            r"static inline int32_t cute_dsl_\w+_wrapper\((.*?)\) \{", header, re.S
        )
        if wrapper is None:
            raise ValueError(f"missing C wrapper for {name}")
        parameters = tuple(
            re.search(r"(\w+)$", arg.strip())[1] for arg in wrapper[1].split(",")[1:]
        )
        if parameters != POINTER_SLOTS + SCALAR_SLOTS:
            raise ValueError(
                f"unsupported V4.1 native launch ABI for {name}: {parameters}"
            )
        tensor_slots = (
            set(POINTER_SLOTS[8:15]) | {"b_w13", "b_down"} | set(POINTER_SLOTS[34:41])
        )
        prefix = "ds41rt_" + name
        expected_declarations = [f"{prefix}_Kernel_Module_t*module"]
        expected_declarations += [
            f"{prefix}_Tensor_{slot}_t*{slot}"
            if slot in tensor_slots
            else f"void*{slot}"
            for slot in POINTER_SLOTS
        ]
        expected_declarations += [f"int32_t{slot}" for slot in SCALAR_SLOTS[:-1]]
        expected_declarations.append("cudaStream_tstream")
        if [
            re.sub(r"\s+", "", arg) for arg in wrapper[1].split(",")
        ] != expected_declarations:
            raise ValueError(f"unsupported exported parameter types for {name}")
        tensors = re.findall(
            r"typedef struct\s*\{([^}]+)\}\s*\w+_Tensor_\w+_t;", header
        )
        if len(tensors) != 16 or any(
            re.sub(r"\s+", "", body) != "void*data;" for body in tensors
        ):
            raise ValueError(f"unsupported exported tensor ABI for {name}")
        entry = re.search(
            r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\)", header
        )
        if entry is None:
            raise ValueError(f"missing native launch entry for {name}")
        variant["native_entry"] = entry[1]
        packed = next(
            t for t in variant["scratch_tensors"] if t["name"] == "packed_input"
        )
        variant["rows_padded"] = packed["shape"][1]
        info = [
            2,
            int(manifest["role"] == "spark"),
            geometry["experts"],
            geometry["hidden"],
            geometry["intermediate"],
            geometry["kernel_intermediate"],
            geometry["topk"],
            variant["capacity_rows"],
            variant["core_scratch_nbytes"],
            variant["max_rows"],
            variant["rows_padded"],
            variant["task_capacity"],
            variant["physical_tiles"],
            variant["max_active_clusters"],
            7 if manifest["input_format"] == "fp8_k32" else 1,
        ]
        scratch = {tensor["name"]: tensor for tensor in variant["scratch_tensors"]}
        if len(scratch) != len(variant["scratch_tensors"]) or set(scratch) != set(SCRATCH_SLOTS.values()):
            raise ValueError("unsupported V4.1 core scratch inventory")
        dtypes = {name: ("int32", 4) for name in scratch}
        dtypes.update({name: ("float32", 4) for name in
                       ("token_weights", "route_output", "input_gs", "down_input_scale")})
        dtypes.update({"materialized_intermediate": ("bfloat16", 2),
                       "packed_input": ("uint8", 1), "packed_input_scale": ("uint8", 1)})
        end = 0
        for tensor in variant["scratch_tensors"]:
            dtype, itemsize = dtypes[tensor["name"]]
            if (tensor["dtype"] != dtype or not tensor["shape"]
                    or any(not isinstance(dim, int) or dim <= 0 for dim in tensor["shape"])
                    or math.prod(tensor["shape"]) * itemsize != tensor["nbytes"]):
                raise ValueError("unsupported V4.1 scratch dtype or extent")
            if (tensor["init"] not in ("zeros", "empty") or tensor["offset"] % 16
                    or tensor["offset"] < end or tensor["nbytes"] <= 0):
                raise ValueError("unsupported V4.1 scratch layout or initialization")
            end = tensor["offset"] + tensor["nbytes"]
        if end != variant["core_scratch_nbytes"]:
            raise ValueError("inconsistent V4.1 scratch capacity")
        for scale in ("input_gs", "down_input_scale"):
            if scratch[scale]["dtype"] != "float32" or scratch[scale]["shape"] != [geometry["experts"]]:
                raise ValueError("unsupported native expert global scale layout")
        offsets = [str(scratch[SCRATCH_SLOTS[slot]]["offset"]) if slot in SCRATCH_SLOTS
                   else "UINT64_MAX" for slot in POINTER_SLOTS]
        variant["scratch_pointer_offsets"] = [int(x) if x != "UINT64_MAX" else None for x in offsets]
        entries.append(
            "{{"
            + ", ".join(map(str, info))
            + "}, "
            + f"_mlir_ds41rt_{name}_cuda_init, _mlir_ds41rt_{name}_cuda_load_to_device, "
            + entry[1]
            + ", {" + ", ".join(offsets) + "}}"
        )
        includes.append(f'#include "{name}.h"')
    manifest["native_abi_version"] = 2
    manifest["pointer_slots"] = list(POINTER_SLOTS)
    lines = [
        "#pragma once",
        *includes,
        f"#define DS41RT_V41_CC_MINOR {manifest['capability'][1]}",
        f"#define DS41RT_V41_SMS {manifest['physical_sms']}",
        "#define DS41RT_V41_VARIANTS " + ", ".join(entries),
    ]
    (output_dir / "v41_expert_variants.h").write_text("\n".join(lines) + "\n")


def export_input_quantizer(output_dir: Path, manifest: dict) -> None:
    from b12x._lib.quant.mxfp8_rows import (
        compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid,
    )
    from export_b12x_v41_fp8_aot import validate_abi
    label = "v41_expert_input_quant"
    compiled = compile_mxfp8_rows_quant_aot(
        size_k=5120, expected_m=80, amax_floor=1e-4, wire_rows=True,
    )
    compiled.export_to_c(str(output_dir), label, "ds41rt_" + label)
    abi = validate_abi(output_dir / (label + ".h"), label, "quant")
    grids = [mxfp8_rows_quant_aot_grid(size_k=5120, rows=m, expected_m=80,
             sm_count=manifest["physical_sms"]) for m in range(1, 4097)]
    header = ["#pragma once", f'#include "{label}.h"',
              f"#define DS41RT_V41_INPUT_QUANT_ENTRY {abi['symbol']}",
              "static const uint32_t ds41rt_v41_input_quant_grids[] = {" +
              ",".join(map(str, grids)) + "};"]
    (output_dir / "v41_input_quant_dispatch.h").write_text("\n".join(header) + "\n")
    manifest["input_quantizer"] = {"format": "row E4M3 payload then UE8M0 K32 scales",
        "row_bytes": 5280, "max_rows": 4096, "expected_m": 80, "amax_floor": 1e-4,
        "abi": abi, "object_sha256": hashlib.sha256((output_dir / (label + ".o")).read_bytes()).hexdigest()}


# Roles served by the SM121 Spark expert family. `spark` is the historical TP4
# shard; `spark_tp2`/`spark_tp3` are the replicated-group TP shards. The degree
# is a plan-time role property, never derived from live rows.
SPARK_ROLES = ("spark", "spark_tp2", "spark_tp3")
SPARK_TP_DEGREES = {"spark": 4, "spark_tp2": 2, "spark_tp3": 3}


def export(output_dir: Path, role: str, rows: tuple[int, ...], input_format: str = "bf16", compact_live_rows: int | None = None) -> None:
    if input_format not in ("bf16", "fp8_k32") or (role not in SPARK_ROLES and input_format != "bf16"):
        raise ValueError("FP8 K32 input is supported only for Spark backbone experts")
    if role in ("spark_tp2", "spark_tp3") and input_format != "fp8_k32":
        raise ValueError("Spark TP2/TP3 use the native FP8 K32 slice export only")
    if compact_live_rows is not None and (role not in SPARK_ROLES or input_format != "fp8_k32"):
        raise ValueError("compact dispatch requires native FP8 Spark experts")
    if role == "coordinator":
        from b12x.moe._shared.kernels.v41_slice_pipeline import V41DraftSlicePipeline
        from export_b12x_v41_slices_aot import export as export_slices

        export_slices(output_dir, rows, V41DraftSlicePipeline.DEFAULT_WIDTH,
                      role=role, standard_names=True)
        return
    if role in SPARK_ROLES and input_format == "fp8_k32":
        from export_b12x_v41_slices_aot import export as export_slices

        # Match the qualified backbone worker: narrow single-row decode,
        # wider grouped execution, and direct token output for prefill. The
        # compact specialization stays limited to the historical TP4 role.
        widths = {capacity: 64 if capacity == 1 else 192 for capacity in rows}
        compact = role == "spark" and compact_live_rows is not None
        export_slices(output_dir, rows, widths, atomic_min_capacity=256,
                      role=role, standard_names=True,
                      compact_max_capacity=16 if compact else None,
                      compact_live_rows=compact_live_rows if compact else None)
        return
    # Export requires compiler IR, which executable-only cache entries omit.
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    import torch
    from b12x.moe.fused_moe import _impl as moe

    torch.empty(1, dtype=torch.uint8, device="cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    capability = (properties.major, properties.minor)
    expected = (12, 1) if role in SPARK_ROLES else (12, 0)
    if capability != expected:
        raise ValueError(
            f"{role} exports require native SM{expected[0]}{expected[1]}, got {capability}"
        )
    experts, intermediate, topk = (384, 576, 6) if role in SPARK_ROLES else (128, 2304, 3)
    weight_plan = moe.plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu_v41",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=5120,
        intermediate_size=intermediate,
        w13_layout="w13",
    )
    kernel_intermediate = moe._dynamic_kernel_intermediate_size(intermediate, "w4a8_mx")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "role": role,
        "spark_tp_degree": SPARK_TP_DEGREES.get(role),
        "input_format": input_format,
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "device": properties.name,
        "capability": list(capability),
        "physical_sms": properties.multi_processor_count,
        "geometry": {
            "experts": experts,
            "hidden": 5120,
            "intermediate": intermediate,
            "kernel_intermediate": kernel_intermediate,
            "topk": topk,
        },
        "activation_quantization": {
            "format": "FP8 E4M3 with UE8M0 K32 scales",
            "input_amax_floor": 1e-4,
            "intermediate_amax_floor": 1e-4,
            "intermediate_boundary": "routing-weighted FP32 SwiGLU rounded to BF16",
        },
        "output": "FP32 token-major route planes; reduction is a separate launch",
        "variants": [],
    }
    for requested_rows in rows:
        scratch_plan = moe.plan_tp_moe_scratch(
            moe.TPMoEScratchCaps(
                max_tokens=requested_rows,
                core_token_counts=(requested_rows,),
                num_topk=topk,
                device=device,
                weight_plan=weight_plan,
                quant_mode="w4a8_mx",
                deterministic_output=True,
                swiglu_limit=10,
            ),
            prewarm_launches=False,
        )
        plan = scratch_plan.launch_plan
        capacity = plan.routed_rows // topk
        if plan.policy_resolution is None:
            raise RuntimeError("V4.1 export requires a resolved b12x launch policy")
        config = plan.policy_resolution.config
        if config.backend != "dynamic" or config.route_planner != "internal":
            raise ValueError("V4.1 native export requires internal dynamic routing")
        direct_routing = config.dynamic_route_mode == "direct"
        core = scratch_plan._core_workspace_plan
        compiled, clusters = moe._get_dynamic_kernel(
            experts,
            capacity,
            5120,
            kernel_intermediate,
            topk,
            plan.max_rows,
            topk_ids_dtype=torch.int32,
            fast_math=True,
            activation="silu_v41",
            quant_mode="w4a8_mx",
            w4a8_repacked=True,
            prequantized_input=input_format == "fp8_k32",
            direct_routing=direct_routing,
            planned_tile_m=config.dynamic_tile_m,
            deterministic_output=True,
            swiglu_limit=10,
        )
        name = f"v41_{role}_m{requested_rows}"
        compiled.export_to_c(str(output_dir), name, f"ds41rt_{name}")
        tensors = []
        offset = 0
        for spec in core.tensor_specs:
            alignment = max(16, spec.dtype.itemsize)
            offset = (offset + alignment - 1) // alignment * alignment
            nbytes = math.prod(spec.shape) * spec.dtype.itemsize
            tensors.append(
                {
                    "name": spec.name,
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype).removeprefix("torch."),
                    "offset": offset,
                    "nbytes": nbytes,
                    "init": spec.init,
                }
            )
            offset += nbytes
        assert offset == moe._core_workspace_nbytes(core)
        manifest["variants"].append(
            {
                "name": name,
                "requested_rows": requested_rows,
                "capacity_rows": capacity,
                "max_rows": plan.max_rows,
                "route_mode": config.dynamic_route_mode,
                "output_splits": 5 if direct_routing else 1,
                "max_active_clusters": clusters,
                "physical_tiles": core.dynamic_physical_tiles,
                "task_capacity": core.dynamic_task_capacity,
                "core_scratch_nbytes": offset,
                "scratch_tensors": tensors,
                "files": {
                    ext: {
                        "name": name + ext,
                        "sha256": hashlib.sha256(
                            (output_dir / (name + ext)).read_bytes()
                        ).hexdigest(),
                    }
                    for ext in (".h", ".o")
                },
            }
        )
        print(f"exported {name}: capacity={capacity}, scratch={offset}", flush=True)
    export_input_quantizer(output_dir, manifest)
    write_native_bridge(output_dir, manifest)
    (output_dir / "v41_experts.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("spark", "spark_tp2", "spark_tp3", "coordinator"), required=True)
    parser.add_argument("--rows", default="1,16,80,256,1024,4096")
    parser.add_argument("--input-format", choices=("bf16", "fp8_k32"), default="bf16")
    parser.add_argument("--compact-live-rows", type=int, help="Experimental Spark live-row compact cutoff")
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(value < 1 or value > 4096 for value in rows)
    ):
        parser.error("--rows must contain distinct positive capacities up to 4096")
    export(args.output_dir, args.role, rows, args.input_format, args.compact_live_rows)


if __name__ == "__main__":
    main()
