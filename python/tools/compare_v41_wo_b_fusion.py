#!/usr/bin/env python3
"""Bounded upstream WO-B fusion probe; not a native serving benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b12x-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.b12x_root.resolve()))
    import torch
    from b12x.gemm._shared.wo_mxfp8 import (
        empty_dense_gemm_mnl_view, quantize_mxfp8_rows_torch,
        quantize_wo_b_input_mxfp8, wo_b_dense_gemm_mxfp8,
        wo_b_dense_gemm_fused_quant_mxfp8,
    )
    torch.manual_seed(31009)
    weights = quantize_mxfp8_rows_torch(
        torch.randn((5120, 8192), device='cuda', dtype=torch.bfloat16) / 8192**0.5)
    report = {
        'scope': 'Upstream Python GPU component probe; not native ABI or serving acceptance',
        'command': sys.argv, 'revision': subprocess.check_output(
            ['git', '-C', str(args.b12x_root), 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,power.limit,clocks.mem',
                                        '--format=csv'], text=True),
        'shape': {'groups': 8, 'rank': 1024, 'hidden': 5120}, 'cases': [],
    }
    for rows in (1, 4, 8):
        source = empty_dense_gemm_mnl_view(rows, 1024, 8, device='cuda', dtype=torch.bfloat16)
        source.normal_(std=0.25)
        quantized = quantize_wo_b_input_mxfp8(source)
        plain_out = empty_dense_gemm_mnl_view(rows, 5120, 1, device='cuda', dtype=torch.bfloat16)
        fused_out = torch.empty_like(plain_out)

        def plain():
            quantize_wo_b_input_mxfp8(source, out=quantized)
            wo_b_dense_gemm_mxfp8(quantized, weights, out=plain_out, expected_m=rows)

        def fused():
            wo_b_dense_gemm_fused_quant_mxfp8(source, weights, out=fused_out, expected_m=rows)

        for _ in range(3):
            plain()
            fused()
        torch.cuda.synchronize()
        torch.testing.assert_close(fused_out, plain_out, rtol=0, atol=0)
        graphs = {}
        for name, fn in [('plain', plain), ('fused', fused)]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    fn()
            graphs[name] = graph
        before = plain_out.clone()
        source.normal_(std=0.5)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fused_out, plain_out, rtol=0, atol=0)
        assert torch.isfinite(fused_out).all() and torch.count_nonzero(fused_out) > 0
        assert not torch.equal(before, plain_out), 'Replay did not consume changed input'
        samples = {'plain': [], 'fused': []}
        allocated = torch.cuda.memory_allocated()
        for iteration in range(6):
            for name in (('plain', 'fused') if iteration % 2 == 0 else ('fused', 'plain')):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))  # 1000 launches: ms == us/launch
        assert torch.cuda.memory_allocated() == allocated
        case = {'rows': rows, 'exact_match_and_graph_mutation': True,
                'stable_replay_allocation': True, 'warm_us': samples,
                'median_us': {key: statistics.median(value) for key, value in samples.items()}}
        report['cases'].append(case)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
