#!/usr/bin/env python3
"""Export B12x route packing as a native, stream-ordered CUDA driver bridge.

Uses the pinned B12x metadata kernels and capacity policy. Callers own seven
distinct, aligned int32 device buffers in manifest order through completion.
Create/destroy happen outside graph capture, on the owning CUDA context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import _pinned_sparkinfer


def export(output: Path, capacity: int, experts: int, topk: int) -> dict:
    import torch
    from b12x.moe._shared.kernels.w4a16.route_pack import compile_w4a16_route_pack_launches

    if not 1 <= capacity <= 4096 or not topk <= experts <= 384:
        raise ValueError('invalid V4.1 route capacity or expert count')
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((12, 0), (12, 1)):
        raise ValueError('requires SM120 or SM121')
    plan = compile_w4a16_route_pack_launches(tokens=capacity, topk=topk,
        block_size=8, num_experts=experts, ordinal=0)
    output.mkdir(parents=True, exist_ok=True)
    # The seven pointers follow the small-prefix ABI; larger plans reuse them.
    slots = dict(topk_ids=capacity*topk, expert_map=experts,
        packed_route_indices=plan.max_packed_routes,
        block_expert_ids=plan.max_route_blocks, packed_route_count=1,
        expert_offsets=experts+1, expert_counts=experts)
    # stage, pointer indices, whether live_numel follows pointers, grid expression
    stages = ([('small_prefix', [0,1,2,3,4,5,6], True, '1')]
        if plan.use_small_prefix else [
            ('count', [0,1,6], True, '(live + 1023) / 1024'),
            ('prefix', [6,4,5], False, '1'),
            ('post_prefix', [2,3,5], True,
             str((max(plan.max_packed_routes, plan.max_route_blocks)+255)//256))])
    stages += [('sort', [0,1,2,5], True, '(live + 255) / 256')]
    source = ['#include <cuda.h>', '#include <cstdint>', '#include <new>']
    objects = []
    for name, indices, live, grid in stages:
        mapped = name not in ('prefix', 'post_prefix')
        kernel = plan.programs[name, torch.int32, mapped]
        if getattr(kernel.metadata, 'num_ctas', 1) != 1:
            raise ValueError('cluster launch ABI is not supported')
        for key in ('global_scratch_size', 'profile_scratch_size'):
            if getattr(kernel.metadata, key, 0):
                raise ValueError(f'unexpected {key}')
        cubin = kernel.asm['cubin']
        # Triton appends global/profile scratch pointers even when their sizes
        # are zero. They are real CUDA parameters, not Python launcher metadata.
        ptx = kernel.asm['ptx']
        entry = re.search(r'\.entry\s+' + re.escape(kernel.name) + r'\s*\((.*?)\)', ptx, re.S)
        if entry is None:
            raise ValueError('missing PTX entry parameter declaration')
        parameters = re.findall(r'\.param\s+\.(u\d+)\b', entry[1])
        expected = ['u64']*len(indices) + (['u32'] if live else []) + ['u64','u64']
        if parameters != expected:
            raise ValueError(f'unexpected route kernel ABI: {parameters}, expected {expected}')
        (output / (name+'.cubin')).write_bytes(cubin)
        (output / (name+'.ptx')).write_text(ptx)
        source += [f'alignas(16) static const unsigned char code_{name}[] = {{',
            ',\n'.join(','.join(str(v) for v in cubin[i:i+24]) for i in range(0,len(cubin),24)), '};']
        objects.append(dict(stage=name, symbol=kernel.name,
            sha256=hashlib.sha256(cubin).hexdigest(), threads=kernel.metadata.num_warps*32,
            shared_bytes=kernel.metadata.shared, pointer_slots=indices,
            live_numel=live, grid_x=grid, trailing_null_scratch_pointers=2))
    source += [f'struct Context {{ CUcontext owner; CUmodule modules[{len(objects)}]{{}}; CUfunction functions[{len(objects)}]{{}}; }};',
        'extern "C" int ds41rt_exl3_routes_destroy(void* opaque) {',
        'auto* ctx = static_cast<Context*>(opaque); if (!ctx) return CUDA_SUCCESS;',
        'CUcontext current; CUresult error = cuCtxGetCurrent(&current); if (error) return error;',
        'if (current != ctx->owner) return CUDA_ERROR_INVALID_CONTEXT;',
        'for (auto module : ctx->modules) if (module) cuModuleUnload(module); delete ctx; return CUDA_SUCCESS; }',
        'extern "C" int ds41rt_exl3_routes_create(void** output) {',
        'if (!output) return CUDA_ERROR_INVALID_VALUE; *output = nullptr;',
        'auto* ctx = new(std::nothrow) Context; if (!ctx) return CUDA_ERROR_OUT_OF_MEMORY;',
        'CUresult error = cuCtxGetCurrent(&ctx->owner);',
        'if (error || !ctx->owner) { delete ctx; return CUDA_ERROR_INVALID_CONTEXT; }',
        'CUdevice device; int major = 0, minor = 0;',
        'error = cuCtxGetDevice(&device);',
        'if (!error) error = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);',
        'if (!error) error = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);',
        f'if (!error && (major != {props.major} || minor != {props.minor})) error = CUDA_ERROR_INVALID_DEVICE;']
    for i, obj in enumerate(objects):
        source += [f'if (!error) error = cuModuleLoadData(&ctx->modules[{i}], code_{obj["stage"]});',
            f'if (!error) error = cuModuleGetFunction(&ctx->functions[{i}], ctx->modules[{i}], "{obj["symbol"]}");']
        if obj['shared_bytes'] > 49152:
            source += [f'if (!error) error = cuFuncSetAttribute(ctx->functions[{i}], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {obj["shared_bytes"]});']
    source += ['if (error) { ds41rt_exl3_routes_destroy(ctx); return error; } *output = ctx; return CUDA_SUCCESS; }',
        'extern "C" int ds41rt_exl3_routes_launch(void* opaque, void* const* pointers, const uint64_t* bytes, int32_t rows, void* stream) {',
        f'if (!opaque || !pointers || !bytes || rows < 1 || rows > {capacity}) return CUDA_ERROR_INVALID_VALUE;',
        'auto* ctx = static_cast<Context*>(opaque); CUcontext current;',
        'CUresult error = cuCtxGetCurrent(&current); if (error) return error;',
        'if (current != ctx->owner) return CUDA_ERROR_INVALID_CONTEXT;',
        'CUdeviceptr p[7];']
    for i, count in enumerate(slots.values()):
        source += [f'p[{i}] = reinterpret_cast<CUdeviceptr>(pointers[{i}]);',
            f'if (!p[{i}] || p[{i}] % 16 || bytes[{i}] < {count*4} || p[{i}] > UINT64_MAX - {count*4}) return CUDA_ERROR_INVALID_VALUE;']
    source += ['for (int i = 0; i < 7; ++i) for (int j = i+1; j < 7; ++j) {',
        'if (bytes[i] > UINT64_MAX-p[i] || bytes[j] > UINT64_MAX-p[j]) return CUDA_ERROR_INVALID_VALUE;',
        'if (p[i] < p[j]+bytes[j] && p[j] < p[i]+bytes[i]) return CUDA_ERROR_INVALID_VALUE; }',
        f'int32_t live = rows * {topk}; CUdeviceptr null_scratch = 0; auto cuda_stream = static_cast<CUstream>(stream);']
    if not plan.use_small_prefix:
        source += [f'error = cuMemsetD32Async(p[6], 0, {experts}, cuda_stream); if (error) return error;']
    for i, obj in enumerate(objects):
        args = [f'&p[{j}]' for j in obj['pointer_slots']]
        if obj['live_numel']:
            args += ['&live']
        args += ['&null_scratch', '&null_scratch']
        source += ['{ void* args[] = {' + ','.join(args) + '};',
            f'error = cuLaunchKernel(ctx->functions[{i}], {obj["grid_x"]}, 1, 1, {obj["threads"]}, 1, 1, {obj["shared_bytes"]}, cuda_stream, args, nullptr);',
            'if (error) return error; }']
    source += ['return CUDA_SUCCESS; }']
    (output/'v41_exl3_routes.cc').write_text('\n'.join(source)+'\n')
    manifest = dict(schema='ds41rt.v41-exl3-routes-aot.v1',
        sparkinfer_revision=_pinned_sparkinfer.REVISION,
        compute=[props.major,props.minor], capacity=capacity, topk=topk,
        experts=experts, block_size=8, small_prefix=plan.use_small_prefix,
        buffers={k:dict(elements=v,bytes=v*4,dtype='int32') for k,v in slots.items()},
        objects=objects)
    (output/'v41_exl3_routes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('capacity','experts','small_prefix')}),flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--experts', type=int, default=384)
    parser.add_argument('--topk', type=int, choices=(3,6), default=6)
    args = parser.parse_args()
    export(args.output,args.capacity,args.experts,args.topk)
