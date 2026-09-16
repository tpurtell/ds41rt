#!/usr/bin/env python3
"""Offline Spark tile/residency experiment; never changes the serving policy.

Uses all 384 experts of a checkpoint layer and compares against the production
tile. Graph timings include routing, mixed core and reduction, not transport.
The explicit residency option changes a static compile key; it is not a
proposed production policy or a live-count specialization.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import _pinned_sparkinfer


def load_weights(args, torch, safe_open, ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights, replace):
    experts, hidden, width, start = 384, 5120, args.intermediate, args.slice_start
    layer_prefix = f'layers.{args.layer}'
    index = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}; bitmaps = {}
    for expert in range(experts):
        for projection in ['w1', 'w3', 'w2']:
            prefix = f'{layer_prefix}.ffn.experts.{expert}.{projection}'
            for suffix in ['trellis','suh','svh','mcg']:
                name = prefix+'.'+suffix
                with safe_open(args.snapshot/index[name], framework='pt', device='cpu') as f:
                    value = f.get_tensor(name)
                    if suffix == 'mcg':
                        assert int(value.item()) & 0xffffffff == 0xCBAC1FED
                        continue
                    if suffix == 'trellis':
                        bitmaps[expert, projection] = value.shape[-1] // 16
                        value = value[start//16:(start+width)//16] if projection == 'w2' else value[:, start//16:(start+width)//16]
                    elif (projection == 'w2' and suffix == 'suh') or (projection != 'w2' and suffix == 'svh'):
                        value = value[start:start+width]
                    tensors[expert, projection, suffix] = value.contiguous().to('cuda')
    def rotations(proj, suffix):
        return torch.stack([tensors[e,proj,suffix] for e in range(experts)])
    tiers=[]
    for bits in (3,4):
        members = {p: tuple(e for e in range(experts) if bitmaps[e,p] == bits) for p in ['w1','w3','w2']}
        def packed(proj):
            shape = (width//16, hidden//16, 16*bits) if proj=='w2' else (hidden//16, width//16, 16*bits)
            return torch.stack([tensors[e,proj,'trellis'] for e in members[proj]]) if members[proj] else torch.empty((0,*shape),dtype=torch.int16,device='cuda')
        tiers.append(ProjectionTrellisTierWeights(bits, torch.cat((packed('w1'),packed('w3'))),
            packed('w2'), members['w1'],members['w3'],members['w2']))
    prepared = prepare_projection_native_trellis_weights(tuple(tiers),
        gate_suh=rotations('w1','suh'), up_suh=rotations('w3','suh'),
        intermediate_rotations=torch.cat((rotations('w1','svh'),rotations('w3','svh'),rotations('w2','suh')),dim=1),
        down_svh=rotations('w2','svh'), activation='silu',params_dtype=torch.bfloat16,
        num_experts=experts, hidden_size=hidden, intermediate_size=width)
    # Empty projection membership is legal. Keep one unreachable physical
    # plane for the native binding's non-null storage ABI; descriptor membership
    # and gate/up counts remain unchanged, so this adds no routed expert.
    padded=[]
    for bits,tier in zip((3,4),prepared.tiers):
        stride=(width//16)*(hidden//16)*(8*bits)
        updates={}
        if tier.w13.numel()==0:
            updates['w13']=torch.zeros(stride,dtype=torch.int32,device='cuda')
        if tier.w2.numel()==0:
            updates.update(w2=torch.zeros(stride,dtype=torch.int32,device='cuda'),
                w2_global_scale=torch.ones(1,dtype=torch.float32,device='cuda'))
        padded.append(replace(tier,**updates))
    prepared=replace(prepared,tiers=tuple(padded))
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--intermediate', type=int, choices=(512, 640), required=True)
    parser.add_argument('--slice-start', type=int, default=0)
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--capacity', type=int, choices=(16, 80), default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import (
        ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights,
    )
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe.fused_moe._impl import _projection_mixed_tile_config
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers,
        bind_mixed_trellis, run_bound_mixed_trellis,
    )
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (12, 1)
    assert args.slice_start % 128 == 0 and args.slice_start + args.intermediate <= 2304
    prepared = load_weights(args, torch, safe_open, ProjectionTrellisTierWeights,
                            prepare_projection_native_trellis_weights, replace)
    baseline_tile = _projection_mixed_tile_config(None, hidden_size=5120,
        intermediate_size=args.intermediate, token_count=args.capacity, direct_topk_routes=False)
    variants = [('baseline', baseline_tile, 1),
                ('k64-one-block', (64, 128, 64, 128), 1),
                ('k64-two-blocks', (64, 128, 64, 128), 2)]
    plans = []
    record = dict(scope=__doc__, source_revision=_pinned_sparkinfer.REVISION,
                  gpu=props.name, sm_count=props.multi_processor_count,
                  torch=torch.__version__, configuration=vars(args).copy(),
                  variants=[], results=[], passed=False)
    record['configuration'] = {k: str(v) if isinstance(v, Path) else v for k, v in record['configuration'].items()}

    def save():
        args.output.write_text(json.dumps(record, indent=2) + '\n')

    for name, tile, blocks in variants:
        started = time.monotonic()
        launch = compile_mixed_trellis(
                size_m=args.capacity, hidden_size=5120, intermediate_size=args.intermediate,
                tier0_num_experts=384, tier1_num_experts=384, route_num_experts=384,
                top_k=6, max_m_blocks=route_pack_capacity(args.capacity * 6, 8, 384, topk=6)[1] // 8,
                sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                force_tile_config=tile, swiglu_limit=10., full_rotation_output_dtype='bf16',
                direct_topk_routes=False, force_blocks_per_sm=blocks)
        buffers = make_mixed_trellis_buffers(launch, device=torch.device('cuda', 0), sms=props.multi_processor_count)
        binding = bind_mixed_trellis(*prepared.tiers, prepared.global_to_combined,
            prepared.descriptor_map, prepared.rotations, launch,
            gate_experts=prepared.gate_counts, up_experts=prepared.up_counts)
        plans.append((name, binding, buffers))
        record['variants'].append(dict(name=name, tile=tile, blocks_per_sm=launch.blocks_per_sm,
                                       compile_seconds=time.monotonic() - started))
        print('compiled', record['variants'][-1], flush=True)
        save()

    cases = [(1, 6), (8, 16), (8, 30), (16, 30)]
    if args.capacity == 80:
        cases += [(24, 30), (64, 30)]
    for rows, unique in cases:
        torch.manual_seed(410917 + rows + unique)
        x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
        members = torch.randperm(384, device='cuda')[:unique]
        ids = members[torch.arange(rows * 6, device='cuda').reshape(rows, 6) % unique].to(torch.int32)
        weights = torch.softmax(torch.randn(rows, 6, device='cuda'), dim=1)
        graphs, outputs = [], []
        for name, binding, buffers in plans:
            for _ in range(3):
                run_bound_mixed_trellis(x, weights, ids, binding, buffers)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run_bound_mixed_trellis(x, weights, ids, binding, buffers)
            graphs.append(graph)
            outputs.append(out)
        checks = []
        eligible = True
        for mutation in range(2):
            if mutation:
                x.mul_(-.75)
                ids.copy_(ids.roll(1, dims=0).roll(1, dims=1))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            reference = outputs[0].clone()
            assert torch.isfinite(reference).all() and reference.abs().max() > 0
            # A residency-only change must preserve the whole-tile arithmetic.
            assert torch.equal(outputs[1], outputs[2]), 'changing residency changed same-tile output'
            for (name, binding, buffers), output in zip(plans, outputs):
                assert torch.isfinite(output).all() and output.abs().max() > 0
                error = float((output.float() - reference.float()).abs().max() / reference.float().abs().max())
                eligible &= error <= .004
                copied = output.clone()
                eager = run_bound_mixed_trellis(x, weights, ids, binding, buffers)
                torch.cuda.synchronize()
                assert torch.equal(copied, eager), (name, 'graph differs from eager')
                check = dict(variant=name, mutation=mutation, relative_max_error=error,
                             passed=error <= .004,
                             relative_l2_error=float((copied.float() - reference.float()).norm() / reference.float().norm()))
                if error > .004:
                    indices = (copied.float() - reference.float()).abs().flatten().topk(8).indices
                    check['largest_differences'] = dict(indices=indices.tolist(),
                        reference=reference.flatten()[indices].tolist(), candidate=copied.flatten()[indices].tolist())
                checks.append(check)
        if not eligible:
            record['results'].append(dict(rows=rows, distinct_experts=unique, checks=checks,
                                          same_tile_residency_bitwise=True, timings=None))
            print('cross-tile correctness gate failed', rows, unique, flush=True)
            save()
            del graphs, outputs
            continue
        # Balanced forward/reverse order; raw samples are retained. Stable graphs
        # replay repeatedly without resolving kernels or allocating workspaces.
        samples = [[] for _ in plans]
        for order in [range(len(plans)), range(len(plans)-1, -1, -1)] * 3:
            for index in order:
                for _ in range(5):
                    graphs[index].replay()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(100):
                    graphs[index].replay()
                end.record()
                end.synchronize()
                samples[index].append(begin.elapsed_time(end) * 10)
        result = dict(rows=rows, distinct_experts=unique, checks=checks,
                      timings=[dict(variant=p[0], samples_us=s, median_us=statistics.median(s))
                               for p, s in zip(plans, samples)])
        record['results'].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)
        save()
        del graphs, outputs
    record['passed'] = all(check['passed'] for result in record['results'] for check in result['checks'])
    save()


if __name__ == '__main__':
    main()
