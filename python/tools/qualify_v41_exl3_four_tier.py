#!/usr/bin/env python3
"""Qualify native four-tier dispatch, dynamic rows and changed-input graphs.

Homogeneous experts use separate single-tier B12x grids as the reference. With
--mixed-projections, each expert uses three distinct bitrates and is evaluated
separately by the existing three-tier grid before summing reference outputs.
"""
import argparse
import ctypes as ct
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import _pinned_sparkinfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument("--mixed-projections", action="store_true")
    args = parser.parse_args()
    import torch
    meta = json.loads((args.aot / 'v41_exl3.json').read_text())
    assert meta['bits'] == [2, 3, 4, 5] and meta['experts'] == 6
    assert meta['sparkinfer_revision'] == _pinned_sparkinfer.REVISION
    props = torch.cuda.get_device_properties(0)
    assert [props.major, props.minor] == meta['compute']
    path = _pinned_sparkinfer.SOURCE / 'tests/moe/test_w4a16_mixed_trellis.py'
    spec = importlib.util.spec_from_file_location('mixed_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    h, w, cap, topk = (meta[k] for k in ('hidden', 'intermediate', 'capacity', 'top_k'))
    device = torch.device('cuda', 0)
    tiers = [reference._prepared(experts=6, hidden=h, intermediate=w, bits=b,
        seed=9400+b, device=device, tile_config=tuple(meta['tile'])) for b in meta['bits']]
    torch.manual_seed(4104)
    x = (torch.randn(cap, h, device=device) * .001).to(torch.bfloat16)
    ids = torch.arange(cap*topk, dtype=torch.int32, device=device).reshape(cap, topk) % 6
    weights = torch.softmax(torch.randn(cap, topk, device=device), dim=-1)
    maps = [torch.tensor([e if e % 4 == tier else -1 for e in range(6)],
        dtype=torch.int32, device=device) for tier in range(4)]
    identity = torch.arange(6, dtype=torch.int32, device=device)
    descriptor = torch.full((3, 24), -1, dtype=torch.int32, device=device)
    # Existing three-tier preparation accepts consecutive bitrates. Alternate
    # K2/K3/K4 and K3/K4/K5, rotating projection order to cover every decoder.
    choices = [[(e % 2 + (e+p) % 3) if args.mixed_projections else e % 4
        for p in range(3)] for e in range(6)]
    for e in range(6):
        for projection in range(3):
            descriptor[projection, e] = (choices[e][projection] << 9) | e
    def rotations(name):
        if name == 'intermediate_rotations':
            rows = torch.stack([torch.cat([tiers[choices[e][p]].intermediate_rotations[e, p*w:(p+1)*w] for p in range(3)]) for e in range(6)])
        else:
            projection = {'gate_suh':0, 'up_suh':1, 'down_svh':2}[name]
            rows = torch.stack([getattr(tiers[choices[e][projection]], name)[e] for e in range(6)])
        return torch.cat((rows, torch.zeros((18, rows.shape[1]), dtype=rows.dtype, device=device)))
    allocations = {}
    buffers = {}
    for name, entry in meta['buffers'].items():
        owner = entry['allocation']
        if owner not in allocations:
            dtype = getattr(torch, entry['dtype'].split('.')[-1])
            allocations[owner] = torch.empty(entry['shape'], dtype=dtype, device=device)
        buffers[name] = allocations[owner].view(entry['shape'])
        if entry['zero_on_create']:
            buffers[name].zero_()
    lut = torch.tensor(list((args.aot/'trellis_lut.bin').read_bytes()), dtype=torch.uint8, device=device)
    pointers = dict(buffers, rotation_input_ptr=x, raw_topk_ids=ids, topk_weights_ptr=weights,
        descriptor_map_ptr=descriptor, global_to_combined_ptr=identity,
        intermediate_rotations_ptr=rotations('intermediate_rotations'),
        gate_suh_ptr=rotations('gate_suh'), up_suh_ptr=rotations('up_suh'),
        trellis_lut_ptr=lut, fc2_ptr=buffers['fc2'], output_ptr=buffers['output'],
        route_expert_ids_ptr=ids, expert_map_ptr=identity, svh_ptr=rotations('down_svh'))
    scalars = dict(grid_x=meta['blocks_per_sm']*meta['sms'], route_num_experts=6, weight_num_experts=24)
    for i, tier in enumerate(tiers):
        for key, field in [('w13','w13'),('w2','w2'),('w13_scales','w13_scale'),('w2_scales','w2_scale'),('w13_global','w13_global_scale'),('w2_global','w2_global_scale')]:
            pointers[f't{i}_{key}_ptr'] = getattr(tier, field)
        for key in ('num_experts', 'fc2_experts', 'gate_experts', 'up_experts'):
            scalars[f'tier{i}_{key}'] = 6
    lib = ct.CDLL(str(args.aot/'libds41rt_exl3.so'))
    lib.ds41rt_exl3_create.argtypes = [ct.POINTER(ct.c_void_p)]
    lib.ds41rt_exl3_destroy.argtypes = [ct.c_void_p]
    context = ct.c_void_p()
    assert lib.ds41rt_exl3_create(ct.byref(context)) == 0
    route = ct.CDLL(str(args.aot/'routes/libv41_exl3_routes.so'))
    route.ds41rt_exl3_routes_create.argtypes = [ct.POINTER(ct.c_void_p)]
    route.ds41rt_exl3_routes_destroy.argtypes = [ct.c_void_p]
    route.ds41rt_exl3_routes_launch.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint64), ct.c_int32, ct.c_void_p]
    route_context = ct.c_void_p()
    assert route.ds41rt_exl3_routes_create(ct.byref(route_context)) == 0
    route_meta = json.loads((args.aot/'routes/v41_exl3_routes.json').read_text())
    route_tensors = dict(topk_ids=ids, expert_map=identity,
        **{name:buffers[name] for name in list(route_meta['buffers'])[2:]})
    rp = (ct.c_void_p*7)(*[t.data_ptr() for t in route_tensors.values()])
    rb = (ct.c_uint64*7)(*[t.numel()*t.element_size() for t in route_tensors.values()])
    calls = []
    for entry in meta['objects']:
        fn = getattr(lib, 'ds41rt_exl3_'+entry['label'].rsplit('_',1)[1])
        fn.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int32), ct.c_void_p]
        p = (ct.c_void_p*len(entry['pointer_slots']))(*[pointers[n].data_ptr() for n in entry['pointer_slots']])
        s = (ct.c_int32*len(entry['scalar_slots']))(*[scalars.get(n, 1) for n in entry['scalar_slots']])
        calls.append((fn, p, s, entry['scalar_slots'].index('active_m')))
    def native(rows):
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        assert route.ds41rt_exl3_routes_launch(route_context, rp, rb, rows, stream) == 0
        for fn, p, s, index in calls:
            s[index] = rows
            assert fn(context, p, s, stream) == 0
    mixed_references = []
    if args.mixed_projections:
        from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights
        from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
            compile_mixed_trellis3, make_mixed_trellis3_buffers, bind_mixed_trellis3, run_bound_mixed_trellis3,
        )
        for expert, selected in enumerate(choices):
            selected_tiers = sorted(set(selected))
            native_tiers = []
            for tier_id in selected_tiers:
                bits = meta['bits'][tier_id]
                tier = tiers[tier_id]
                fc1 = tier.w13.view(torch.int16).reshape(12, h//16, w//16, 16*bits)
                fc2 = tier.w2.view(torch.int16).reshape(6, w//16, h//16, 16*bits)
                gate = fc1[expert:expert+1] if selected[0] == tier_id else fc1[:0]
                up = fc1[6+expert:7+expert] if selected[1] == tier_id else fc1[:0]
                down = fc2[expert:expert+1] if selected[2] == tier_id else fc2[:0]
                native_tiers.append(ProjectionTrellisTierWeights(bits, torch.cat((gate,up)), down,
                    (0,) if selected[0] == tier_id else (), (0,) if selected[1] == tier_id else (),
                    (0,) if selected[2] == tier_id else ()))
            prepared = prepare_projection_native_trellis_weights(tuple(native_tiers),
                gate_suh=pointers['gate_suh_ptr'][expert:expert+1],
                up_suh=pointers['up_suh_ptr'][expert:expert+1],
                intermediate_rotations=pointers['intermediate_rotations_ptr'][expert:expert+1],
                down_svh=pointers['svh_ptr'][expert:expert+1], activation='silu', params_dtype=torch.bfloat16,
                num_experts=1, hidden_size=h, intermediate_size=w)
            padded = []
            for tier, tier_id in zip(prepared.tiers, selected_tiers):
                stride = (w//16)*(h//16)*8*meta['bits'][tier_id]
                updates = {}
                if tier.w13.numel() == 0:
                    updates['w13'] = torch.zeros(stride, dtype=torch.int32, device=device)
                if tier.w2.numel() == 0:
                    updates.update(w2=torch.zeros(stride, dtype=torch.int32, device=device),
                        w2_global_scale=torch.ones(1,device=device))
                padded.append(replace(tier, **updates))
            launch = compile_mixed_trellis3(size_m=cap, hidden_size=h, intermediate_size=w,
                tier0_num_experts=1, tier1_num_experts=1, tier2_num_experts=1,
                tier0_bits=meta['bits'][selected_tiers[0]], tier1_bits=meta['bits'][selected_tiers[1]],
                tier2_bits=meta['bits'][selected_tiers[2]], top_k=topk, route_num_experts=1,
                max_m_blocks=meta['route_blocks'], sms=meta['sms'],
                max_shared_mem=props.shared_memory_per_block_optin, force_tile_config=tuple(meta['tile']),
                swiglu_limit=10.0, broadcast_suh=True, broadcast_svh=True)
            binding = bind_mixed_trellis3(*padded, prepared.global_to_combined, prepared.descriptor_map,
                prepared.rotations, launch, gate_experts=prepared.gate_counts, up_experts=prepared.up_counts)
            ref_buffers = make_mixed_trellis3_buffers(launch, device=device, sms=meta['sms'])
            mixed_references.append((binding, ref_buffers))
    def expected(rows):
        if args.mixed_projections:
            total = torch.zeros((rows,h), device=device)
            for expert, (binding, ref_buffers) in enumerate(mixed_references):
                ref_ids = torch.where(ids[:rows] == expert, 0, -1).to(torch.int32)
                live_buffers = replace(ref_buffers, output=ref_buffers.output[:rows])
                total += run_bound_mixed_trellis3(x[:rows], weights[:rows], ref_ids, binding, live_buffers)
            return total.to(buffers['output'].dtype)
        return sum((reference._serial_tier(x[:rows], tier, weights[:rows], ids[:rows], mapping,
            swiglu_limit=10.0) for tier, mapping in zip(tiers, maps)),
            torch.zeros((rows,h),device=device)).to(buffers['output'].dtype)
    results = []
    try:
        for rows in (cap, 1, min(3,cap), cap):
            want = expected(rows)
            native(rows)
            torch.cuda.synchronize()
            actual = buffers['output'][:rows].clone()
            assert torch.isfinite(actual).all() and actual.abs().max() > 0
            relative = float((actual.float()-want.float()).norm()/want.float().norm())
            assert relative < .004, relative
            results.append(dict(rows=rows, relative_l2=relative))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            native(cap)
        for step in range(2):
            x.mul_(-.75)
            ids.copy_(ids.roll(1, dims=1))
            want = expected(cap)
            graph.replay()
            torch.cuda.synchronize()
            actual = buffers['output'].clone()
            relative = float((actual.float()-want.float()).norm()/want.float().norm())
            assert torch.isfinite(actual).all() and relative < .004, relative
            native(cap)
            torch.cuda.synchronize()
            assert torch.equal(actual, buffers['output'])
            results.append(dict(graph_step=step, relative_l2=relative))
        report = dict(passed=True, scope=__doc__, cases=results, source_revision=_pinned_sparkinfer.REVISION,
            mixed_projections=args.mixed_projections, projection_tiers=choices, compute=meta['compute'], bits=meta['bits'], capacity=cap, hidden=h, intermediate=w)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    finally:
        torch.cuda.synchronize()
        route.ds41rt_exl3_routes_destroy(route_context)
        lib.ds41rt_exl3_destroy(context)


if __name__ == '__main__':
    main()
