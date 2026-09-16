#!/usr/bin/env python3
"""Export mixed EXL3 compute/epilogue objects and exact native buffer metadata.

Packed exports include native route preparation. This export does not claim
that an end-to-end native serving backend is available.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re

import _pinned_sparkinfer


def write_bridge(output: Path, manifest: dict) -> None:
    """Generate a small owned-device C bridge; reject unfamiliar exported types."""
    # CuTe AOT stores kernel handles in globals inside each loaded DSO. Every
    # context must retain the same CUDA libraries; recreating them per lane or
    # device replaces those globals and loses another device's launch attributes.
    lines = ['#include <new>', '#include <mutex>', '#include "v41_exl3_core.h"', '#include "v41_exl3_sum.h"',
        'struct Context { int device; };',
        'struct Modules { std::mutex mutex; unsigned users = 0; ds41rt_v41_exl3_core_Kernel_Module_t core{}; ds41rt_v41_exl3_sum_Kernel_Module_t sum{}; };',
        'static Modules modules;',
        'static void unload_modules() { if (modules.sum.module) cudaLibraryUnload(modules.sum.module); if (modules.core.module) cudaLibraryUnload(modules.core.module); modules.sum.module = nullptr; modules.core.module = nullptr; }']
    for entry in manifest['objects']:
        label = entry['label']; role = label.rsplit('_', 1)[1]
        lines += [f'static int load_{role}(int device) {{',
            f'cudaLibrary_t* library = &modules.{role}.module; cudaError_t status = cudaSuccess;',
            'if (!*library) {',
            'struct { cudaLibrary_t** library; cudaError_t* status; } init{&library, &status};',
            f'_mlir_ds41rt_{label}_cuda_init(reinterpret_cast<void**>(&init));',
            'if (status != cudaSuccess) return int(status); }',
            'struct { cudaLibrary_t** library; int32_t* device; cudaError_t* status; } load{&library, &device, &status};',
            f'_mlir_ds41rt_{label}_cuda_load_to_device(reinterpret_cast<void**>(&load));',
            'return int(status); }']
    lines += ['extern "C" int ds41rt_exl3_create(void** out) {',
        'if (!out) return int(cudaErrorInvalidValue); *out = nullptr;',
        'Context* ctx = new(std::nothrow) Context; if (!ctx) return int(cudaErrorMemoryAllocation);',
        'cudaError_t status = cudaGetDevice(&ctx->device); cudaDeviceProp props{};',
        'if (status == cudaSuccess) status = cudaGetDeviceProperties(&props, ctx->device);',
        f'if (status != cudaSuccess || props.major != {manifest["compute"][0]} || props.minor != {manifest["compute"][1]} || props.multiProcessorCount != {manifest["sms"]}) {{ delete ctx; return int(cudaErrorInvalidDevice); }}',
        'std::lock_guard<std::mutex> lock(modules.mutex);',
        'int error = load_core(ctx->device); if (!error) error = load_sum(ctx->device);',
        'if (error) { if (!modules.users) unload_modules(); delete ctx; return error; }',
        '++modules.users; *out = ctx; return 0; }',
        'extern "C" void ds41rt_exl3_destroy(void* opaque) { auto* ctx = static_cast<Context*>(opaque); if (!ctx) return; std::lock_guard<std::mutex> lock(modules.mutex); if (--modules.users == 0) unload_modules(); delete ctx; }']
    for entry in manifest['objects']:
        role = entry['label'].rsplit('_', 1)[1]
        args = [f'&modules.{role}']; declarations = []; checks = []; pointers = []; scalars = []
        for parameter in entry['parameters'][1:]:
            match = re.fullmatch(r'(.+?)(\w+)', parameter)
            if match is None: raise ValueError(f'unrecognized parameter: {parameter}')
            type_name, name = match[1].strip(), match[2]
            if type_name == 'cudaStream_t':
                args.append('static_cast<cudaStream_t>(stream)')
            elif type_name == 'int32_t':
                index = len(scalars); scalars.append(name); args.append(f's[{index}]')
                if name == 'active_m': checks.append(f'if (s[{index}] < 1 || s[{index}] > {manifest["capacity"]}) return int(cudaErrorInvalidValue);')
            elif type_name == 'void *' or re.fullmatch(r'ds41rt_v41_exl3_\w+_Tensor_\w+_t \*', type_name):
                index = len(pointers); pointers.append(name)
                checks.append(f'if (!p[{index}]) return int(cudaErrorInvalidValue);')
                if type_name == 'void *': args.append(f'p[{index}]')
                else:
                    tensor_type = type_name[:-1].strip()
                    header = (output / (entry['label'] + '.h')).read_text()
                    if not re.search(r'typedef struct\s*\{\s*void\s*\*data;\s*\}\s*' + re.escape(tensor_type) + r';', header):
                        raise ValueError(f'unsupported tensor descriptor {tensor_type}')
                    declarations.append(f'{tensor_type} {name}{{p[{index}]}};'); args.append('&' + name)
            else: raise ValueError(f'unsupported native parameter {parameter}')
        entry['pointer_slots'] = pointers; entry['scalar_slots'] = scalars
        lines += [f'extern "C" int ds41rt_exl3_{role}(void* opaque, void* const* p, const int32_t* s, void* stream) {{',
            'if (!opaque || !p || !s) return int(cudaErrorInvalidValue); auto* ctx = static_cast<Context*>(opaque);',
            'int device = -1; if (cudaGetDevice(&device) != cudaSuccess || device != ctx->device) return int(cudaErrorInvalidDevice);',
            *checks, *declarations, f'return {entry["wrapper"]}({", ".join(args)});', '}']
    core, epilogue = manifest['objects']
    info = [2, manifest['hidden'], manifest['intermediate'], manifest['experts'],
        manifest['capacity'], manifest['top_k'], len(manifest['bits']),
        len(core['pointer_slots']), len(core['scalar_slots']),
        len(epilogue['pointer_slots']), len(epilogue['scalar_slots']),
        *manifest['bits'], *([0] * (4 - len(manifest['bits']))),
        2 if manifest['output_dtype'] == 'bf16' else 4]
    lines += ['extern "C" int ds41rt_exl3_info(uint32_t* out, uint32_t words) {',
        'if (!out || words != 16) return int(cudaErrorInvalidValue);',
        'const uint32_t info[16] = {' + ','.join(map(str, info)) + '};',
        'for (int i = 0; i < 16; ++i) out[i] = info[i]; return 0; }']
    (output / 'v41_exl3_bridge.cc').write_text('\n'.join(lines) + '\n')


def export(output: Path, intermediate: int, experts: int, capacity: int,
           bits: tuple[int, ...], routing: str, topk: int = 6, output_dtype: str = "bf16") -> dict:
    if output_dtype not in ("bf16", "fp32"):
        raise ValueError("EXL3 output must be bf16 or fp32")
    # Disk-loaded B12x executors omit the compiler IR required by export_to_c.
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    import torch
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, compile_mixed_trellis3, make_mixed_trellis_buffers,
    )
    from b12x.moe._shared.kernels.w4a16.mixed_trellis4 import compile_mixed_trellis4
    from b12x.moe.fused_moe._impl import (
        _projection_mixed_direct_topk_routes, _projection_mixed_tile_config,
    )

    if len(bits) not in (2, 3, 4) or len(set(bits)) != len(bits) or any(b not in range(2, 6) for b in bits):
        raise ValueError("export requires two to four distinct K2..K5 decoder tiers")
    if not 1 <= capacity <= 4096 or topk not in (3, 6) or experts < topk or experts > 384:
        raise ValueError("invalid V4.1 capacity or expert count")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        raise ValueError("V4.1 export requires native SM120 or SM121")
    direct = _projection_mixed_direct_topk_routes(capacity, topk, direct_exl3=len(bits) == 2)
    if routing != "auto":
        direct = routing == "direct"
    if direct and len(bits) != 2:
        raise ValueError("three/four-tier export requires packed routing")
    block_m = 8
    route_slots = capacity * topk if direct else route_pack_capacity(capacity * topk, block_m, experts, topk=topk)[1]
    route_blocks = route_slots if direct else (route_slots + block_m - 1) // block_m
    options = dict(size_m=capacity, hidden_size=5120, intermediate_size=intermediate,
        tier0_num_experts=experts, tier1_num_experts=experts, top_k=topk,
        route_num_experts=experts, max_m_blocks=route_blocks,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=_projection_mixed_tile_config(None, hidden_size=5120,
            intermediate_size=intermediate, token_count=capacity, direct_topk_routes=direct),
        tier0_bits=bits[0], tier1_bits=bits[1], trellis_codebook="mcg", swiglu_limit=10.0,
        moe_block_size=block_m, rotation_input_dtype="bf16", full_rotation_output_dtype=output_dtype,
        route_ids_dtype=torch.int32)
    if len(bits) == 2:
        launch = compile_mixed_trellis(**options, direct_topk_routes=direct)
    elif len(bits) == 3:
        launch = compile_mixed_trellis3(**options, tier2_num_experts=experts, tier2_bits=bits[2])
    else:
        launch = compile_mixed_trellis4(**options, tier2_num_experts=experts, tier2_bits=bits[2],
            tier3_num_experts=experts, tier3_bits=bits[3])
    output.mkdir(parents=True, exist_ok=True)
    objects = []
    for label, compiled in [("v41_exl3_core", launch.compiled), ("v41_exl3_sum", launch.topk_sum.compiled)]:
        compiled.export_to_c(str(output), label, "ds41rt_" + label)
        header = (output / (label + ".h")).read_text()
        wrapper = re.search(r"static inline int32_t (cute_dsl_\w+_wrapper)\((.*?)\) \{", header, re.S)
        if wrapper is None:
            raise ValueError(f"missing native wrapper for {label}")
        objects.append({"label": label, "wrapper": wrapper[1],
            "parameters": [p.strip() for p in wrapper[2].split(',')],
            "object_sha256": hashlib.sha256((output / (label + '.o')).read_bytes()).hexdigest(),
            "header_sha256": hashlib.sha256(header.encode()).hexdigest()})
    buffers = make_mixed_trellis_buffers(launch, device=torch.device("cuda", 0), sms=props.multi_processor_count)
    lut_bytes = launch.trellis_lut.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    (output / 'trellis_lut.bin').write_bytes(lut_bytes)
    layouts = {}
    owners = {}
    for field in fields(buffers):
        value = getattr(buffers, field.name)
        address = value.untyped_storage().data_ptr()
        owner = owners.setdefault(address, field.name)
        layouts[field.name] = {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(), "allocation": owner,
            "zero_on_create": field.name == "workspace"}
    # The mixed executor's Python buffers cover exact live-capacity routes,
    # while the precompiled route packer rounds token capacity to its bucket.
    # Its initialization kernels write that entire bucket, including padding.
    # Publish the canonical packer's larger metadata allocations; compute data
    # buffers still cover only the actual token capacity.
    if not direct:
        for name, count in (("packed_route_indices", route_slots), ("block_expert_ids", route_blocks)):
            spec = layouts[name]
            if spec['allocation'] != name or spec['dtype'] != 'torch.int32' or len(spec['shape']) != 1:
                raise ValueError(f'unexpected route metadata layout: {name}')
            if spec['bytes'] > count * 4:
                raise ValueError(f'route metadata exceeds canonical capacity: {name}')
            spec.update(shape=[count], bytes=count * 4)
    manifest = {"schema": "ds41rt.v41-exl3-aot.v1", "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "gpu": props.name, "compute": [props.major, props.minor], "sms": props.multi_processor_count,
        "hidden": 5120, "intermediate": intermediate, "experts": experts, "top_k": topk,
        "capacity": capacity, "output_dtype": output_dtype, "bits": list(bits), "swiglu_limit": 10.0,
        "direct": direct, "route_slots": route_slots, "route_blocks": route_blocks,
        "tile": list(options['force_tile_config']), "blocks_per_sm": launch.blocks_per_sm,
        "shared_memory_bytes": launch.shared_memory_bytes, "buffers": layouts, "objects": objects,
        "unique_execution_buffer_bytes": sum(v['bytes'] for k,v in layouts.items() if v['allocation'] == k),
        "requires_route_preparation": not direct,
        "required_link_libraries": ["cudart", "cute_dsl_runtime"],
        "trellis_lut": {"file": "trellis_lut.bin", "bytes": len(lut_bytes),
            "sha256": hashlib.sha256(lut_bytes).hexdigest()},
        "native_execution_verified": False}
    write_bridge(output, manifest)
    if not direct:
        from export_b12x_v41_exl3_routes_aot import export as export_routes
        route_manifest = export_routes(output / 'routes', capacity, experts, topk)
        for name in ('packed_route_indices', 'block_expert_ids', 'packed_route_count', 'expert_offsets', 'expert_counts'):
            if route_manifest['buffers'][name]['bytes'] > layouts[name]['bytes']:
                raise ValueError(f'mixed execution buffer {name} cannot hold route preparation')
        manifest['route_preparation'] = {
            'manifest': 'routes/v41_exl3_routes.json',
            'sha256': hashlib.sha256((output / 'routes/v41_exl3_routes.json').read_bytes()).hexdigest(),
        }
    (output / "v41_exl3.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ['capacity','intermediate','bits','direct','unique_execution_buffer_bytes']}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intermediate", type=int, choices=(512, 640, 1152, 2304), required=True)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--routing", choices=("auto", "direct", "packed"), default="auto")
    parser.add_argument("--topk", type=int, choices=(3, 6), default=6)
    parser.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    export(args.output, args.intermediate, args.experts, args.capacity, tuple(args.bits), args.routing, args.topk, args.output_dtype)


if __name__ == "__main__":
    main()
