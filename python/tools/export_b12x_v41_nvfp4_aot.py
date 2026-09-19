#!/usr/bin/env python3
"""Export V4.1 W4A4 (ModelOpt NVFP4) routed-expert kernels as native AOT.

The b12x dynamic nvfp4 kernel takes 37 pointer operands plus seven int32
scheduling scalars and a stream. The engine's 44-slot V4.1 expert ABI is a
superset of that signature, so each variant is wrapped by a small generated
bridge that reorders the engine's slots into the kernel's parameter order:

  slot  0..21 -> kernel  0..21   (input, routing scratch, task queue)
  slot    22 -> kernel 22        (fused FC1 payload)
  slot    23 -> kernel 23/24     (FC1 scales and aliased gate view;
                                  separate_w13_halves=false at every n)
  slot 24/25 -> kernel 25/26     (FC2 payload/scales)
  slot 34..43 -> kernel 27..36   (row counts, write rows, alpha vectors,
                                  token map/weights)
  args 44..51 -> kernel scalars and stream

Kernel-owned scratch comes from the b12x core workspace plan; alpha and the
activation-scale vectors are supplied by the weight binder (slots 38/39) and
the per-expert scale upload (slots 37/40).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import _pinned_sparkinfer

# role -> (experts, intermediate, topk, capability)
ROLES = {
    "spark": (384, 576, 6, (12, 1)),
    "rtx_backbone": (384, 2304, 6, (12, 0)),
    "rtx_tp2": (384, 1152, 6, (12, 0)),
    "dspark_tp2": (128, 1152, 3, (12, 0)),
}
DEFAULT_ROWS = (1, 16, 80, 256, 1024, 4096)
TILE_M_CHOICES = (16, 32, 64, 128)


def parse_tile_m(value: str) -> int | None:
    if value == "auto":
        return None
    if value not in tuple(str(tile) for tile in TILE_M_CHOICES):
        raise argparse.ArgumentTypeError("tile-m must be auto, 16, 32, 64, or 128")
    return int(value)


# Small-M decode uses direct routing (no route-packing pass); larger
# prefill capacities use grouped routing.
DIRECT_ROUTE_MAX_ROWS = 4
# b12x core workspace tensor name -> engine pointer slots.
SCRATCH_SLOTS = {
    "packed_input": (3, 5),
    "packed_input_scale": (4, 6),
    "materialized_intermediate": (7,),
    "barrier_count": (8,),
    "barrier_epoch": (9,),
    "pair_head": (10,),
    "producers_done_count": (11,),
    "all_work_published": (12,),
    "task_head": (13,),
    "task_tail": (14,),
    "task_ready": (15,),
    "task_expert": (16,),
    "task_m_tile": (17,),
    "task_slice_begin": (18,),
    "task_slice_count": (19,),
    "task_valid_rows": (20,),
    "tile_write_count": (21,),
    "row_counts": (34,),
    "expert_write_rows": (35,),
    "expert_tile_base": (36,),
    "input_gs": (37,),
    "down_input_scale": (40,),
    "route_output": (41,),
    "token_map": (42,),
    "token_weights": (43,),
}
# Engine slots whose pointers are supplied by the weight binder, not scratch.
WEIGHT_BOUND_SLOTS = (22, 23, 24, 25, 38, 39)
# Bridge mapping in the kernel's parameter order. `None` marks an engine slot
# that is not consumed by this kernel (the W4A8-only scale/residual/repack
# planes). The gate scale view aliases the fused FC1 plane.
BRIDGE_SLOTS = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 23, 24, 25,
    34, 35, 36, 37, 38, 39, 40,
    41, 42, 43,
]
SCALAR_SLOT_COUNT = 7
SCALAR_BASE = 44
STREAM_SLOT = SCALAR_BASE + SCALAR_SLOT_COUNT
STATUS_SLOT = STREAM_SLOT + 1
LAUNCH_ARGUMENT_COUNT = len(BRIDGE_SLOTS) + SCALAR_SLOT_COUNT + 2


def export(
    output: Path,
    role: str,
    rows: list[int],
    tile_m: int | None,
    max_active_clusters: int | None = None,
    output_shards: int = 1,
    share_input: bool = False,
) -> None:
    if type(share_input) is not bool:
        raise TypeError("share_input must be boolean")
    if type(output_shards) is not int or output_shards < 1 or 40 % output_shards:
        raise ValueError("output_shards must be a positive divisor of 40")
    if tile_m is not None and (type(tile_m) is not int or tile_m not in TILE_M_CHOICES):
        raise ValueError("tile_m must be None (auto), 16, 32, 64, or 128")

    import torch
    from b12x.moe.fused_moe import _impl as moe
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    if role not in ROLES:
        raise ValueError(f"unsupported NVFP4 expert role {role!r}")
    experts, intermediate, topk, capability = ROLES[role]
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != capability:
        raise ValueError(
            f"{role} NVFP4 exports require SM{capability[0]}{capability[1]}, "
            f"got {properties.major}{properties.minor}"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    weight_plan = moe.plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=experts,
        hidden_size=5120,
        intermediate_size=intermediate,
        w13_layout="w13",
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "role": role,
        "quant_mode": "nvfp4",
        "source_format": "modelopt_nvfp4",
        "input_format": "bf16",
        "activation": "silu",
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "device": properties.name,
        "capability": [properties.major, properties.minor],
        "physical_sms": properties.multi_processor_count,
        "tile_m": tile_m,
        "share_input": share_input,
        "geometry": {
            "experts": experts,
            "hidden": 5120,
            "intermediate": intermediate,
            "kernel_intermediate": intermediate,
            "topk": topk,
        },
        "activation_quantization": {
            "format": "FP4 E2M1 in 16-wide groups with E4M3 scales (W4A4)",
            "weight_scale_layout": "F8_128x4 swizzled E4M3 K16 plane",
            "payload_policy": "keeper-source E2M1 with [up(w3); gate(w1)] rows",
            "launch_scales": (
                "input_global_scale=1/input_scale, alpha=weight_scale_2*input_scale, "
                "down_alpha=weight_scale_2(down)*input_scale(down), "
                "global_scale=1/input_scale(down)"
            ),
        },
        "variants": [],
    }
    includes: list[str] = []
    entries: list[str] = []
    for requested_rows in rows:
        route_mode = "direct" if requested_rows <= DIRECT_ROUTE_MAX_ROWS else "grouped"
        config = MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=tile_m,
            dynamic_route_mode=route_mode,
            w4a16_route_mode=None,
            nvfp4_share_input=share_input,
        )
        scratch_plan = moe.plan_tp_moe_scratch(
            moe.TPMoEScratchCaps(
                max_tokens=requested_rows,
                core_token_counts=(requested_rows,),
                num_topk=topk,
                device=device,
                weight_plan=weight_plan,
                quant_mode="nvfp4",
                decode_config=config,
                deterministic_output=True,
                swiglu_limit=10,
            ),
            prewarm_launches=False,
        )
        plan = scratch_plan.launch_plan
        capacity = plan.routed_rows // topk
        core = scratch_plan._core_workspace_plan
        compiled, clusters = moe._get_dynamic_kernel(
            experts,
            capacity,
            5120,
            intermediate,
            topk,
            plan.max_rows,
            topk_ids_dtype=torch.int32,
            fast_math=True,
            activation="silu",
            quant_mode="nvfp4",
            w4a8_repacked=False,
            nvfp4_materialize_intermediate=False,
            # The front end quantizes each token once with a single shared
            # activation scale and fans the row out to every routed expert,
            # instead of re-quantizing the identical BF16 row per route.
            share_input_across_experts=share_input,
            direct_routing=route_mode == "direct",
            # Compile with the same resolved tile that sized the scratch arena.
            planned_tile_m=plan.execution.tile_m,
            deterministic_output=True,
            swiglu_limit=10,
            mac_override=max_active_clusters,
            nvfp4_output_shards=output_shards if route_mode == "direct" else 1,
        )
        if not 0 < clusters <= 2 * properties.multi_processor_count:
            raise ValueError(f"invalid NVFP4 cooperative launch grid: {clusters}")
        label = f"v41_nvfp4_{role}_m{requested_rows}"
        symbol = "ds41rt_" + label
        compiled.export_to_c(str(output), label, symbol)
        header = (output / f"{label}.h").read_text()
        entry = re.findall(r"void (_mlir_\w+)\(void \*\*args, int32_t num_args\);", header)
        if len(entry) != 1:
            raise ValueError(f"unexpected NVFP4 export entry for {label}")
        expected_args = re.search(r"void \*args\[(\d+)\] = \{", header)
        if expected_args is None or int(expected_args.group(1)) != LAUNCH_ARGUMENT_COUNT:
            raise ValueError(f"unexpected NVFP4 launch arity for {label}")
        bodies = re.findall(r"typedef struct\s*\{([^}]+)\}\s*\w+_Tensor_\w+_t;", header)
        if any(re.sub(r"\s+", "", body) != "void*data;" for body in bodies):
            raise ValueError(f"unexpected NVFP4 tensor ABI for {label}")
        # Scratch layout: one arena, 16-byte aligned regions, slot aliases share
        # a region. Weight-bound and request-bound slots stay unbound here.
        offset = 0
        scratch = []
        offsets: list[int | None] = [None] * 44
        for spec in core.tensor_specs:
            slots = SCRATCH_SLOTS.get(spec.name)
            if slots is None:
                raise ValueError(f"unmapped NVFP4 scratch tensor {spec.name}")
            alignment = max(16, spec.dtype.itemsize)
            offset = (offset + alignment - 1) // alignment * alignment
            nbytes = math.prod(spec.shape) * spec.dtype.itemsize
            for slot in slots:
                offsets[slot] = offset
            scratch.append(
                {
                    "name": spec.name,
                    "slots": list(slots),
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype).removeprefix("torch."),
                    "offset": offset,
                    "nbytes": nbytes,
                    "init": spec.init,
                }
            )
            offset += nbytes
        scratch_bytes = moe._core_workspace_nbytes(core)
        if offset > scratch_bytes:
            raise ValueError(f"NVFP4 scratch accounting overruns the arena for {label}")
        for slot in WEIGHT_BOUND_SLOTS:
            offsets[slot] = None
        # Unused W4A8-only slots still need non-null placeholders for the
        # shared engine guard. Preserve request/weight-bound slots: callers
        # may bind a smaller scratch arena after installing their I/O.
        for slot in range(26, 34):
            offsets[slot] = 0
        includes.append(f'#include "{label}.h"')
        mapped = ",".join(
            f"args[{slot}]" if slot is not None else "nullptr" for slot in BRIDGE_SLOTS
        )
        scalar_args = ",".join(
            f"args[{SCALAR_BASE + index}]" for index in range(SCALAR_SLOT_COUNT)
        )
        includes.extend(
            [
                f"static void {label}_bridge(void** args, int32_t count) {{",
                f"  if (count != {STATUS_SLOT + 1}) {{ "
                f"*static_cast<int32_t*>(args[{STATUS_SLOT}]) = 1; return; }}",
                f"  void* mapped[] = {{{mapped},{scalar_args},"
                f"args[{STREAM_SLOT}],args[{STATUS_SLOT}]}};",
                f"  {entry[0]}(mapped, {LAUNCH_ARGUMENT_COUNT});",
                "}",
            ]
        )
        packed = next(
            tensor for tensor in core.tensor_specs if tensor.name == "packed_input"
        )
        route_output = next(
            tensor for tensor in core.tensor_specs if tensor.name == "route_output"
        )
        if (route_output.dtype != torch.bfloat16 or
                tuple(route_output.shape) != (capacity * topk, 5120)):
            raise ValueError(f"unexpected deterministic NVFP4 route output for {label}")
        info = [
            2,  # abi_version: bridge layout only, no compact variant
            {"spark": 1, "rtx_backbone": 2, "rtx_tp2": 3, "dspark_tp2": 4}[role],
            experts,
            5120,
            intermediate,
            intermediate,
            topk,
            capacity,
            scratch_bytes,
            plan.max_rows,
            packed.shape[1],
            core.dynamic_task_capacity,
            core.dynamic_physical_tiles,
            # This is the compiled kernel's positive cooperative grid size,
            # NOT policy_max_active_clusters (-1 means unset only in the
            # Python policy wrapper). _launch_dynamic_impl resolves that
            # policy before passing the mac returned by _get_dynamic_kernel.
            clusters,
            1,  # input_dtype: BF16 hidden rows
        ]
        prefix = "_mlir_" + symbol
        entries.append(
            "{"
            + ",".join(map(str, info))
            + ","
            + prefix
            + "_cuda_init,"
            + prefix
            + "_cuda_load_to_device,"
            + label
            + "_bridge,{"
            + ",".join("UINT64_MAX" if value is None else str(value) for value in offsets)
            + "}}"
        )
        manifest["variants"].append(
            {
                "name": label,
                "native_entry": entry[0],
                "route_mode": route_mode,
                "output_shards": output_shards if route_mode == "direct" else 1,
                "tile_m": plan.execution.tile_m,
                "output_kind": 2,
                "output_format": "bf16_routes",
                "requested_rows": requested_rows,
                "capacity_rows": capacity,
                "max_rows": plan.max_rows,
                "rows_padded": packed.shape[1],
                "task_capacity": core.dynamic_task_capacity,
                "physical_tiles": core.dynamic_physical_tiles,
                "max_active_clusters": clusters,
                "compile_time_clusters": clusters,
                "core_scratch_nbytes": scratch_bytes,
                "scratch": scratch,
                "scratch_pointer_offsets": offsets,
            }
        )
    (output / "v41_expert_variants.h").write_text(
        "\n".join(
            [
                "#pragma once",
                *includes,
                f"#define DS41RT_V41_CC_MINOR {properties.minor}",
                f"#define DS41RT_V41_SMS {properties.multi_processor_count}",
                "// Deterministic dynamic NVFP4 publishes BF16 per-route rows.",
                "#define DS41RT_V41_OUTPUT_KIND(capacity) 2",
                "#define DS41RT_V41_VARIANTS " + ",".join(entries),
                "",
            ]
        )
    )
    manifest["artifact_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.suffix in (".h", ".o")
    }
    (output / "v41_nvfp4_experts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(ROLES), required=True)
    parser.add_argument("--rows", default=",".join(str(row) for row in DEFAULT_ROWS))
    parser.add_argument(
        "--tile-m",
        type=parse_tile_m,
        default=16,
        metavar="{auto,16,32,64,128}",
        help="default 16 until GPU qualified; auto opts into the planner's tile ladder",
    )
    parser.add_argument(
        "--max-active-clusters",
        type=int,
        default=None,
        help="positive cooperative grid override; unset uses measured kernel occupancy",
    )
    parser.add_argument("--output-shards", type=int, default=1,
                        help="experimental direct-route output shards; positive divisor of 40")
    parser.add_argument(
        "--share-input",
        action="store_true",
        help="quantize each token's activation once with a shared scale and fan it "
             "out to every routed expert instead of per route",
    )
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",") if value]
    export(args.output_dir, args.role, rows, args.tile_m, args.max_active_clusters,
           args.output_shards, args.share_input)


if __name__ == "__main__":
    main()
