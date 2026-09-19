"""Probe real-geometry NVFP4 kernels with synthetic weights and graph replay.

This is a component diagnostic, not serving throughput or an independent math
oracle. Compare --intermediate 576 against --intermediate 576 --pad-to 640,
using --save-output/--reference to enforce exact route-output equivalence.
Default kernels use the checked-in source; --source selects an isolated b12x
experiment (e.g. grouped output sharding). Coordinate GPU ownership before use.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=str(Path(__file__).resolve().parents[2] / 'third_party/sparkinfer'))
    p.add_argument('--intermediate', type=int, default=1152)
    p.add_argument('--pad-to', type=int, default=0)
    p.add_argument('--save-output')
    p.add_argument('--reference')
    p.add_argument('--tile-m', type=int, choices=(16, 32, 64, 128), default=16)
    p.add_argument('--capacity', type=int, default=16)
    p.add_argument('--live', type=int, default=8)
    p.add_argument('--cases', default='grouped:1')
    p.add_argument('--check-live-counts', action='store_true',
                   help='check sparse/full live counts with the already resolved kernels')
    p.add_argument('--routing', choices=['shared', 'distinct'], default='shared')
    a = p.parse_args()
    sys.path.insert(0, a.source)
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from b12x._lib.utils import current_cuda_stream
    from b12x.moe.fused_moe import _impl as moe
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    E, H, N, K, C, M = (384, 5120, a.intermediate, 6, a.capacity, a.live)
    assert 0 < M <= C
    assert not a.pad_to or (a.pad_to >= N and a.pad_to % 128 == 0)
    source_hashes = {name: hashlib.sha256((Path(a.source) / name).read_bytes()).hexdigest()
                     for name in ('b12x/moe/fused_moe/_impl.py', 'b12x/moe/_shared/kernels/dynamic.py')}
    print(json.dumps({'source': str(Path(a.source).resolve()), 'source_sha256': source_hashes}), flush=True)
    torch.manual_seed(43)
    dev = torch.device('cuda')
    print(json.dumps({'device': torch.cuda.get_device_name(), 'geometry': [E, H, N, K], 'capacity': C, 'live': M, 'routing': a.routing, 'tile_m': a.tile_m, 'padded_intermediate': a.pad_to or N}), flush=True)
    w13 = torch.randint(0, 256, (E, 2 * N, H // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, H, N // 2), dtype=torch.uint8, device=dev)
    if a.pad_to:
        padded13 = torch.zeros((E, 2 * a.pad_to, H // 2), dtype=torch.uint8, device=dev)
        padded13[:, :N] = w13[:, :N]
        padded13[:, a.pad_to:a.pad_to + N] = w13[:, N:]
        padded2 = torch.zeros((E, H, a.pad_to // 2), dtype=torch.uint8, device=dev)
        padded2[:, :, :N // 2] = w2
        w13, w2 = (padded13, padded2)
        N = a.pad_to
    s13 = torch.full((E, 2 * N, H // 16), 0.0625, device=dev).to(torch.float8_e4m3fn)
    s2 = torch.full((E, H, N // 16), 0.0625, device=dev).to(torch.float8_e4m3fn)
    idx = torch.arange(E, dtype=torch.float32, device=dev)
    a1 = 1.0 + idx.remainder(7) / 8
    a2 = 1.0 + idx.remainder(3) / 4
    alpha = (1 + idx.remainder(5) / 4) / a1
    down = (1 + idx.remainder(2) / 4) / a2
    x = torch.randn(C, H, dtype=torch.bfloat16, device=dev) * 0.125
    ids = (torch.arange(C, device=dev)[:, None] * (0 if a.routing == 'shared' else K) + torch.arange(K, device=dev)[None, :]).remainder(E).to(torch.int32)
    routing = torch.softmax(torch.randn(C, K, device=dev), -1)
    weight_plan = moe.plan_b12x_fp4_moe_weights(quant_modes='nvfp4', source_format='modelopt_nvfp4', activation='silu', params_dtype=torch.bfloat16, num_experts=E, hidden_size=H, intermediate_size=N, w13_layout='w13')

    def ptr(t, dtype, align=16):
        return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)
    I, F, B, U, S, Q = (cutlass.Int32, cutlass.Float32, cutlass.BFloat16, cutlass.Uint8, cutlass.Float8E4M3FN, cutlass.Float4E2M1FN)
    arms = []
    live_invocations = []
    reference = None
    for case in a.cases.split(','):
        route, shards = case.split(':')
        shards = int(shards)
        cfg = MoeDecodeConfig(backend='dynamic', route_planner='internal', max_active_clusters=None, dynamic_tile_m=a.tile_m, dynamic_route_mode=route, nvfp4_share_input=False)
        scratch = moe.plan_tp_moe_scratch(moe.TPMoEScratchCaps(max_tokens=C, core_token_counts=(C,), num_topk=K, device=dev, weight_plan=weight_plan, quant_mode='nvfp4', decode_config=cfg, deterministic_output=True, swiglu_limit=10), prewarm_launches=False)
        plan = scratch.launch_plan
        ws = {s.name: torch.zeros(s.shape, dtype=s.dtype, device=dev) for s in scratch._core_workspace_plan.tensor_specs}
        print('compile', case, flush=True)
        kernel, mac = moe._get_dynamic_kernel(E, C, H, N, K, plan.max_rows, topk_ids_dtype=torch.int32, fast_math=True, activation='silu', quant_mode='nvfp4', w4a8_repacked=False, nvfp4_materialize_intermediate=False, share_input_across_experts=False, direct_routing=route == 'direct', planned_tile_m=a.tile_m, deterministic_output=True, swiglu_limit=10, nvfp4_output_shards=shards)
        args = [ptr(x, B), ptr(ids, I, 4), ptr(routing, F, 4), ptr(ws['packed_input'], Q), ptr(ws['packed_input_scale'], S), ptr(ws['packed_input'], U), ptr(ws['packed_input_scale'], U), ptr(ws['materialized_intermediate'], cutlass.Uint32)]
        args += [ws[n] for n in ['barrier_count', 'barrier_epoch', 'pair_head', 'producers_done_count', 'all_work_published', 'task_head', 'task_tail']]
        args += [ptr(ws[n], I, 4) for n in ['task_ready', 'task_expert', 'task_m_tile', 'task_slice_begin', 'task_slice_count', 'task_valid_rows', 'tile_write_count']]
        args += [w13, ptr(s13, S), ptr(s13, S), w2, ptr(s2, S), ws['row_counts'], ws['expert_write_rows'], ws['expert_tile_base'], a1, alpha, down, a2, ptr(ws['route_output'], B), ptr(ws['token_map'], I, 4), ptr(ws['token_weights'], F, 4)]
        args += [M, plan.max_rows, M * K, plan.dynamic_physical_tiles * a.tile_m, plan.dynamic_task_capacity, plan.dynamic_physical_tiles, mac, current_cuda_stream()]

        def launch(kernel=kernel, args=args):
            args[-1] = current_cuda_stream()
            kernel(*args)
        ws['route_output'].fill_(float('nan'))
        launch()
        torch.cuda.synchronize()
        out = ws['route_output'][:M * K].clone()
        assert torch.isfinite(out).all() and out.abs().sum() > 0
        if reference is None:
            reference = out
            if a.save_output:
                torch.save(out.cpu(), a.save_output)
            if a.reference:
                expected = torch.load(a.reference, weights_only=True).to(dev)
                print('padding compare', 'equal', torch.equal(out, expected), 'maxabs', (out.float() - expected.float()).abs().max().item(), flush=True)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                launch()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(ws['route_output'][:M * K], out, rtol=0, atol=0)
        arms.append((case, graph, ws, []))
        live_invocations.append((case, launch, args, ws))
        print('correct', case, 'mac', mac, flush=True)
    # Reuse the captured kernels and addresses while route occupancy changes.
    # This also catches a policy accidentally frozen to its warmup work count.
    original_ids, original_x = ids.clone(), x.clone()
    for stride in (0, K, 2):
        ids.copy_((torch.arange(C, device=dev)[:, None] * stride
                   + torch.arange(K, device=dev)[None, :]).remainder(E).to(torch.int32))
        x.copy_(original_x * (1.0 + (stride + 1) / 16))
        mutated_reference = None
        for case, graph, ws, _ in arms:
            ws['route_output'].fill_(float('nan'))
            graph.replay()
            torch.cuda.synchronize()
            actual = ws['route_output'][:M * K].clone()
            assert torch.isfinite(actual).all() and actual.abs().sum() > 0
            if mutated_reference is None:
                mutated_reference = actual
            else:
                torch.testing.assert_close(actual, mutated_reference, rtol=0, atol=0)
        print('mutated graph routes exact', stride, flush=True)
    ids.copy_(original_ids)
    x.copy_(original_x)
    for _, graph, _, _ in arms:
        graph.replay()
    torch.cuda.synchronize()
    for rep in range(9):
        for case, graph, ws, samples in arms if rep % 2 == 0 else list(reversed(arms)):
            start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 200)
    for case, graph, ws, samples in arms:
        print(json.dumps({'case': case, 'us': samples, 'median_us': statistics.median(samples), 'bit_exact': True}), flush=True)

    if a.check_live_counts:
        for live in sorted({1, M, C}):
            live_reference = None
            for case, launch, args, ws in live_invocations:
                # Runtime scalar arguments only: do not resolve another kernel.
                args[-8], args[-6] = live, live * K
                ws['route_output'].fill_(float('nan'))
                launch()
                torch.cuda.synchronize()
                actual = ws['route_output'][:live * K].clone()
                assert torch.isfinite(actual).all() and actual.abs().sum() > 0
                if live_reference is None:
                    live_reference = actual
                else:
                    torch.testing.assert_close(actual, live_reference, rtol=0, atol=0)
                live_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(live_graph):
                    launch()
                for _ in range(3):
                    live_graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(ws['route_output'][:live * K], actual,
                                           rtol=0, atol=0)
            print('frozen kernels live count exact', live, flush=True)


if __name__ == "__main__":
    main()
