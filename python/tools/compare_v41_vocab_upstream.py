#!/usr/bin/env python3
"""Bounded native/upstream vocabulary projection screen; not serving acceptance."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b12x-root', type=Path, required=True)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--candidate-native-lib', type=Path)
    parser.add_argument('--snapshot', type=Path, help='Use checkpoint head.weight with synthetic hidden states')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.b12x_root.resolve()))
    import torch
    from b12x.gemm.bf16_vocab_projection._kernel import _row_kernel

    torch.manual_seed(91641)
    lib = C.CDLL(str(args.native_lib.resolve()))
    create = lib.ds41rt_v41_vocabulary_shard_create
    create.argtypes = [C.c_void_p, C.c_uint64, C.c_int32, C.POINTER(C.c_void_p)]
    launch = lib.ds41rt_v41_vocabulary_head_launch
    launch.argtypes = [C.c_void_p] * 4 + [C.c_int32, C.c_void_p]
    destroy = lib.ds41rt_v41_markov_destroy
    destroy.argtypes = [C.c_void_p]
    candidate_lib = None
    if args.candidate_native_lib:
        candidate_lib = C.CDLL(str(args.candidate_native_lib.resolve()))
        candidate_lib.ds41rt_v41_vocabulary_shard_create.argtypes = create.argtypes
        candidate_lib.ds41rt_v41_vocabulary_head_launch.argtypes = launch.argtypes
        candidate_lib.ds41rt_v41_markov_destroy.argtypes = destroy.argtypes
    report = dict(scope=__doc__, command=sys.argv,
        revision=subprocess.check_output(['git', '-C', str(args.b12x_root), 'rev-parse', 'HEAD'], text=True).strip(),
        native_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,power.limit,clocks.mem', '--format=csv'], text=True),
        protocol='Synthetic BF16 weights; native FP32 logits versus upstream row kernel instantiated with FP32 output. Upstream public BF16 rounding assessed separately. Balanced warm CUDA graphs.',
        cases=[])
    if candidate_lib:
        report['candidate_sha256'] = hashlib.sha256(args.candidate_native_lib.read_bytes()).hexdigest()
    checkpoint_weight = None
    if args.snapshot:
        from safetensors import safe_open
        index_path = args.snapshot / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        weight_path = args.snapshot / index['weight_map']['head.weight']
        with safe_open(weight_path, framework='pt', device='cpu') as checkpoint:
            checkpoint_weight = checkpoint.get_tensor('head.weight')
        assert checkpoint_weight.shape == (129280, 5120) and checkpoint_weight.dtype == torch.bfloat16
        report['checkpoint'] = dict(snapshot=str(args.snapshot), tensor='head.weight',
            index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            tensor_sha256=hashlib.sha256(checkpoint_weight.view(torch.uint16).numpy()).hexdigest(),
            inputs='Synthetic hidden states, not captured serving activations')
    for vocab in (64640, 129280):
        weight = (checkpoint_weight[:vocab].to('cuda') if checkpoint_weight is not None else
                  torch.empty((vocab, 5120), device='cuda', dtype=torch.bfloat16).normal_(std=.02))
        workspace = torch.empty(4 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        handle = C.c_void_p()
        assert create(workspace.data_ptr(), workspace.numel(), vocab, C.byref(handle)) == 0
        candidate_handle = C.c_void_p()
        if candidate_lib:
            assert candidate_lib.ds41rt_v41_vocabulary_shard_create(
                workspace.data_ptr(), workspace.numel(), vocab, C.byref(candidate_handle)) == 0
        try:
            for rows in (1, 4, 16):
                x = torch.empty((rows, 5120), device='cuda', dtype=torch.bfloat16).normal_(std=.2)
                native = torch.empty((rows, vocab), device='cuda', dtype=torch.float32)
                upstream = torch.empty_like(native)
                candidate = torch.empty_like(native) if candidate_lib else None

                def native_run():
                    assert launch(handle, x.data_ptr(), weight.data_ptr(), native.data_ptr(), rows,
                                  torch.cuda.current_stream().cuda_stream) == 0

                def upstream_run():
                    _row_kernel[(vocab, rows)](x, weight, upstream, K=5120, BLOCK_K=8192, N=vocab, num_warps=8)

                functions = {'native': native_run, 'upstream_fp32': upstream_run}
                if candidate_lib:
                    def candidate_run():
                        assert candidate_lib.ds41rt_v41_vocabulary_head_launch(
                            candidate_handle, x.data_ptr(), weight.data_ptr(), candidate.data_ptr(),
                            rows, torch.cuda.current_stream().cuda_stream) == 0
                    functions['candidate_native'] = candidate_run
                graphs = {}
                for name, fn in functions.items():
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(10):
                            fn()
                    graphs[name] = graph
                checks = []
                for scale in (.2, .01, 0.):
                    x.normal_(std=scale)
                    for graph in graphs.values():
                        graph.replay()
                    torch.cuda.synchronize()
                    columns = [0, 1, vocab // 2, vocab - 1]
                    reference = x.float().cpu() @ weight[columns].float().cpu().T
                    for output in (native, upstream):
                        torch.testing.assert_close(output[:, columns].cpu(), reference, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(upstream, native, atol=2e-5, rtol=2e-5)
                    rounded = upstream.bfloat16().float()
                    checks.append(dict(input_std=scale, max_error=(native-upstream).abs().max().item(),
                        greedy_exact=torch.equal(native.argmax(1), upstream.argmax(1)),
                        bf16_logit_max_error=(native-rounded).abs().max().item(),
                        bf16_greedy_exact=torch.equal(native.argmax(1), rounded.argmax(1))))
                    if candidate_lib:
                        torch.testing.assert_close(candidate, native, atol=2e-5, rtol=2e-5)
                        torch.testing.assert_close(candidate[:, columns].cpu(), reference, atol=2e-5, rtol=2e-5)
                        checks[-1].update(candidate_max_error=(candidate-native).abs().max().item(),
                            candidate_greedy_exact=torch.equal(native.argmax(1), candidate.argmax(1)))
                        assert checks[-1]['candidate_greedy_exact']
                x.normal_(std=.2)
                samples = {name: [] for name in graphs}
                allocated = torch.cuda.memory_allocated()
                for iteration in range(6):
                    for name in (list(graphs) if iteration % 2 == 0 else list(reversed(graphs))):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 10)
                assert torch.cuda.memory_allocated() == allocated
                case = dict(vocab=vocab, rows=rows, checks=checks, stable_replay_allocation=True,
                            warm_us=samples, median_us={k: statistics.median(v) for k, v in samples.items()})
                report['cases'].append(case)
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(case), flush=True)
                for graph in graphs.values():
                    graph.reset()
        finally:
            torch.cuda.synchronize()
            assert destroy(handle) == 0
            if candidate_lib:
                assert candidate_lib.ds41rt_v41_markov_destroy(candidate_handle) == 0


if __name__ == '__main__':
    main()
