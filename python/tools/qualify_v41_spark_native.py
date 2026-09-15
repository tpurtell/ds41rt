#!/usr/bin/env python3
"""Qualify the Spark native packed-FP8 consumer against quantized expert math."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import torch
import _pinned_sparkinfer
from _v41_expert_native import Native, library, P, check
from tests.moe.test_v41_expert_numerics import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(410916)
    h, n, active = 5120, 576, 8
    lib = library(str(args.native_lib))
    sizes_fn = lib.ds41rt_v41_expert_packed_sizes
    sizes_fn.argtypes = [C.c_uint32, C.POINTER(C.c_uint64)]
    sizes_fn.restype = C.c_int32
    sizes = (C.c_uint64 * 4)()
    check(sizes_fn(n, sizes))
    pools = [torch.empty((384, size), device='cuda', dtype=torch.uint8) for size in sizes]
    weights, scales = {}, {}
    for name, shape in [('w1', (active, n, h//2)), ('w3', (active, n, h//2)),
                        ('w2', (active, h, n//2))]:
        weights[name] = torch.randint(0, 256, shape, device='cuda', dtype=torch.uint8)
        scales[name] = torch.randint(121, 126, (*shape[:-1], shape[-1]//16),
                                     device='cuda', dtype=torch.uint8)
    for expert in range(active):
        source = [bank[name][expert] for bank in (weights, scales) for name in ('w1','w3','w2')]
        check(lib.ds41rt_v41_pack_expert_async(
            (P*6)(*[v.data_ptr() for v in source]),
            (P*4)(*[v[expert].data_ptr() for v in pools]), n,
            torch.cuda.current_stream().cuda_stream))
    cases = []
    for capacity in (1, 16, 80, 256):
        x = torch.randn(capacity, h, device='cuda').mul_(.2).bfloat16()
        wire = torch.empty(capacity, h+h//32, device='cuda', dtype=torch.uint8)
        ids = torch.rand(capacity, active, device='cuda').topk(6, -1).indices.int()
        routing = torch.rand(capacity, 6, device='cuda')
        routing.div_(routing.sum(-1, keepdim=True))
        consumer = Native(lib, capacity, pools, wire, ids, routing)

        def encode():
            blocks = x.float().reshape(capacity, h//32, 32)
            exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448))
            wire[:, :h].copy_((blocks/torch.exp2(exponent)[..., None])
                              .to(torch.float8_e4m3fn).view(torch.uint8).reshape(capacity, h))
            wire[:, h:].copy_((exponent+127).to(torch.uint8))

        for rows in sorted({1, min(7, capacity), capacity}):
            encode()
            consumer.run(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                consumer.run(rows)
            x.mul_(-.5)
            ids.add_(1).remainder_(active)
            encode()
            expected = reference(x[:rows], ids[:rows], routing[:rows], weights, scales)
            consumer.output.fill_(float('nan'))
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            output_rows = rows if consumer.token_accumulation else rows*6
            actual = (consumer.output[:rows] if consumer.token_accumulation else
                      consumer.output[:output_rows].reshape(rows, 6, h).sum(1))
            assert torch.isfinite(actual).all() and actual.norm() > 0
            relative = ((actual-expected).norm()/expected.norm()).item()
            cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
            assert relative < .01 and cosine > .9999, (capacity, rows, relative, cosine)
            wire[:, :h].zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(consumer.output[:output_rows]) == 0
            graph.reset()
            cases.append(dict(capacity=capacity, rows=rows,
                              token_accumulation=consumer.token_accumulation,
                              relative_l2=relative, cosine=cosine))
            print('passed', cases[-1], flush=True)
    args.output.write_text(json.dumps(dict(
        revision=_pinned_sparkinfer.REVISION, cases=cases,
        library_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        gpu=str(torch.cuda.get_device_properties(0).uuid),
        scope=__doc__), indent=2)+'\n')


if __name__ == '__main__':
    main()
