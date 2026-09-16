#!/usr/bin/env python3
"""Compare native selection with upstream row-topk (selection-only lower bound)."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b12x-root', required=True)
    p.add_argument('--native', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, a.b12x_root)
    import torch
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
    torch.cuda.set_device(a.device)
    lib = C.CDLL(a.native)
    native = lib.ds41rt_v41_index_top512
    native.argtypes = [C.c_void_p]*4 + [C.c_uint64, C.c_void_p] + [C.c_int32]*3 + [C.c_void_p]
    native.restype = C.c_int32
    results = []
    for rows, width in [(1, 512), (1, 4096), (16, 4096), (16, 16384)]:
        torch.manual_seed(rows+width)
        scores = torch.randn(rows, width, device='cuda').bfloat16().float()
        positions = torch.arange(width, device='cuda', dtype=torch.int64).expand(rows, width).contiguous()
        carry = torch.empty(rows, 512, device='cuda', dtype=torch.int64)
        scratch = torch.empty(rows*((width+1023)//1024)*512*16, device='cuda', dtype=torch.uint8)
        out = torch.empty(rows, 512, device='cuda', dtype=torch.int32)
        indices = torch.empty_like(out)
        values = torch.empty(rows, 512, device='cuda')
        lengths = torch.full((rows,), width, device='cuda', dtype=torch.int32)
        def old():
            status = native(scores.data_ptr(), positions.data_ptr(), carry.data_ptr(), scratch.data_ptr(), scratch.numel(), out.data_ptr(), rows, width, 1, torch.cuda.current_stream().cuda_stream)
            if status: raise RuntimeError(f'native status {status}')
        def new():
            run_row_topk(row_logits=scores, lengths=lengths, topk=512, output_values=values, output_indices=indices)
        old(); new(); torch.cuda.synchronize()
        graphs = []
        for fn in (old, new):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): fn()
            graphs.append(graph)
        checks = []
        for cycle in range(3):
            if cycle == 1: scores.neg_()
            if cycle == 2: scores.zero_()  # adversarial complete tie
            out.fill_(-9); indices.fill_(-9)
            allocated = torch.cuda.memory_allocated()
            for graph in graphs: graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :512].sort(dim=1).values
            native_ok = torch.equal(out.long(), expected)
            upstream_ok = torch.equal(indices.long().sort(dim=1).values, expected)
            valid = bool(((indices >= 0) & (indices < width)).all())
            score_equal = valid and torch.equal(scores.gather(1, indices.long()).sort(dim=1).values, scores.gather(1, expected).sort(dim=1).values)
            unique = valid and bool((indices.sort(dim=1).values[:, 1:] != indices.sort(dim=1).values[:, :-1]).all())
            checks.append({'cycle': cycle, 'native_exact': native_ok, 'upstream_set_exact': upstream_ok, 'upstream_score_multiset_exact': score_equal, 'upstream_unique_valid_indices': unique})
        samples = [[], []]
        if all(c['native_exact'] and c['upstream_set_exact'] for c in checks):
            scores.normal_(); scores.copy_(scores.bfloat16().float())
            for repeat in range(6):
                for arm in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100): graphs[arm].replay()
                    end.record(); end.synchronize()
                    samples[arm].append(start.elapsed_time(end)*10)
        results.append({'rows': rows, 'width': width, 'checks': checks, 'native_us': samples[0], 'upstream_selection_only_us': samples[1]})
        print(results[-1], flush=True)
    report = {'scope': 'Native full top512 versus upstream selection-only lower bound; upstream excludes position sorting and carry merge, so not an equivalent serving replacement.', 'native_sha256': hashlib.sha256(Path(a.native).read_bytes()).hexdigest(), 'device': str(torch.cuda.get_device_properties(a.device)), 'results': results}
    Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    return 0 if all(c['native_exact'] and c['upstream_set_exact'] for r in results for c in r['checks']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
