#!/usr/bin/env python3
"""Compare a direct native EXL3 AOT bridge with B12x on real checkpoint tiles."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import _pinned_sparkinfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aot', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers, bind_mixed_trellis, run_bound_mixed_trellis,
    )
    meta = json.loads((args.aot/'v41_exl3.json').read_text())
    assert meta['direct'] and meta['bits'] == [3, 4] and meta['experts'] == 6
    assert meta['sparkinfer_revision'] == _pinned_sparkinfer.REVISION
    hidden, width, experts, capacity = meta['hidden'], meta['intermediate'], meta['experts'], meta['capacity']
    index = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}; bitmaps = {}
    for expert in range(experts):
        for projection in ['w1', 'w3', 'w2']:
            prefix = f'layers.0.ffn.experts.{expert}.{projection}'
            for suffix in ['trellis','suh','svh','mcg']:
                name = prefix+'.'+suffix
                with safe_open(args.snapshot/index[name], framework='pt', device='cpu') as f:
                    value = f.get_tensor(name)
                    if suffix == 'mcg':
                        assert int(value.item()) & 0xffffffff == 0xCBAC1FED
                        continue
                    if suffix == 'trellis':
                        bitmaps[expert, projection] = value.shape[-1] // 16
                        value = value[:width//16] if projection == 'w2' else value[:, :width//16]
                    elif (projection == 'w2' and suffix == 'suh') or (projection != 'w2' and suffix == 'svh'):
                        value = value[:width]
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
    props=torch.cuda.get_device_properties(0)
    launch=compile_mixed_trellis(size_m=capacity,hidden_size=hidden,intermediate_size=width,
        tier0_num_experts=experts,tier1_num_experts=experts,route_num_experts=experts,
        top_k=6,max_m_blocks=meta['route_blocks'],sms=props.multi_processor_count,
        max_shared_mem=props.shared_memory_per_block_optin,force_tile_config=tuple(meta['tile']),
        swiglu_limit=10.0, direct_topk_routes=True,full_rotation_output_dtype='bf16')
    buffers=make_mixed_trellis_buffers(launch,device=torch.device('cuda',0),sms=props.multi_processor_count)
    binding=bind_mixed_trellis(*prepared.tiers,prepared.global_to_combined,prepared.descriptor_map,prepared.rotations,launch,
        gate_experts=prepared.gate_counts,up_experts=prepared.up_counts)
    lib=ct.CDLL(str(args.aot/'libv41_exl3_probe.so'))
    lib.ds41rt_exl3_create.argtypes=[ct.POINTER(ct.c_void_p)];lib.ds41rt_exl3_create.restype=ct.c_int
    lib.ds41rt_exl3_destroy.argtypes=[ct.c_void_p]
    for role in ['core','sum']:
        fn=getattr(lib,'ds41rt_exl3_'+role)
        fn.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int32),ct.c_void_p];fn.restype=ct.c_int
    context=ct.c_void_p();assert lib.ds41rt_exl3_create(ct.byref(context))==0
    torch.manual_seed(4105)
    x=torch.randn(capacity,hidden,device='cuda',dtype=torch.bfloat16)
    ids=torch.arange(6,device='cuda',dtype=torch.int32).repeat(capacity,1)
    weights=torch.softmax(torch.randn(capacity,6,device='cuda'),dim=1)
    pointers={name:getattr(buffers,name) for name in meta['buffers']}
    pointers.update(rotation_input_ptr=x,raw_topk_ids=ids,topk_weights_ptr=weights,
        descriptor_map_ptr=binding.descriptor_map,global_to_combined_ptr=binding.global_to_combined,
        intermediate_rotations_ptr=binding.rotations.intermediate,gate_suh_ptr=binding.rotations.gate_suh,
        up_suh_ptr=binding.rotations.up_suh,trellis_lut_ptr=launch.trellis_lut,
        fc2_ptr=buffers.fc2,output_ptr=buffers.output,route_expert_ids_ptr=ids,
        expert_map_ptr=binding.global_to_combined,svh_ptr=binding.rotations.down_svh)
    scalars=dict(grid_x=meta['blocks_per_sm']*meta['sms'],route_num_experts=experts,
        weight_num_experts=launch.topk_sum.num_experts)
    for i,tier in enumerate(prepared.tiers):
        for key,field in [('w13','w13'),('w2','w2'),('w13_scales','w13_scale'),('w2_scales','w2_scale'),('w13_global','w13_global_scale'),('w2_global','w2_global_scale')]:
            pointers[f't{i}_{key}_ptr']=getattr(tier,field)
        scalars.update({f'tier{i}_num_experts':experts,f'tier{i}_fc2_experts':binding.fc2_counts[i],
            f'tier{i}_gate_experts':binding.gate_counts[i],f'tier{i}_up_experts':binding.up_counts[i]})
    def native(rows):
        scalars['active_m']=rows
        for entry in meta['objects']:
            role=entry['label'].rsplit('_',1)[1]
            p=(ct.c_void_p*len(entry['pointer_slots']))(*(pointers[n].data_ptr() for n in entry['pointer_slots']))
            s=(ct.c_int32*len(entry['scalar_slots']))(*(scalars[n] for n in entry['scalar_slots']))
            status=getattr(lib,'ds41rt_exl3_'+role)(context,p,s,ct.c_void_p(torch.cuda.current_stream().cuda_stream))
            assert status==0,(role,status)
    assert any(len({bitmaps[e,p] for p in ['w1','w3','w2']}) > 1 for e in range(experts))
    results=[]; graph=None
    try:
        for rows in [1,3,capacity]:
            expected=run_bound_mixed_trellis(x[:rows],weights[:rows],ids[:rows],binding,buffers).clone()
            buffers.output.fill_(float('nan'));native(rows);torch.cuda.synchronize()
            assert torch.isfinite(expected).all() and bool(expected.abs().sum()>0)
            assert torch.equal(buffers.output[:rows],expected)
            results.append({'rows':rows,'bitwise_equal':True})
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): native(3)
        x.mul_(1.1);ids.copy_(ids.roll(1,dims=1))
        expected=run_bound_mixed_trellis(x[:3],weights[:3],ids[:3],binding,buffers).clone()
        buffers.output.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        assert torch.equal(buffers.output[:3],expected)
        args.output.write_text(json.dumps({'passed':True,'scope':'native AOT versus B12x, real first six layer-0 experts; not full-model qualification',
            'sparkinfer_revision':_pinned_sparkinfer.REVISION,'compute':meta['compute'],'intermediate':width,
            'snapshot_revision':args.snapshot.name,
            'projection_tiers':[[bitmaps[e,p] for p in ['w1','w3','w2']] for e in range(experts)],
            'aot_manifest_sha256':hashlib.sha256((args.aot/'v41_exl3.json').read_bytes()).hexdigest(),
            'bridge_sha256':hashlib.sha256((args.aot/'libv41_exl3_probe.so').read_bytes()).hexdigest(),
            'checks':results,'graph_changed_inputs_and_routes':True},indent=2)+'\n')
        print(args.output.read_text(),flush=True)
    finally:
        torch.cuda.synchronize()
        if graph is not None: del graph
        lib.ds41rt_exl3_destroy(context)


if __name__=='__main__': main()
