#!/usr/bin/env python3
"""Compare four paired TP4 partials with four disjoint partials on one Spark.

Uses real checkpoint weights and production hidden width. This isolates tensor
partitioning and graph replay; it does not qualify distributed transport,
performance, or the independent dequantized quantization oracle.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import _pinned_sparkinfer
from bench_v41_exl3_tiles import load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--capacity', type=int, choices=(16, 80), default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        compile_mixed_trellis, make_mixed_trellis_buffers, bind_mixed_trellis, run_bound_mixed_trellis,
    )
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) == (12, 1)
    record = dict(scope=__doc__, source_revision=_pinned_sparkinfer.REVISION,
        source_tree_sha256=_pinned_sparkinfer.LOCK_DATA['source_tree_sha256'],
        snapshot=args.snapshot.name, layer=args.layer, capacity=args.capacity,
        gpu=props.name, sm_count=props.multi_processor_count, torch=torch.__version__,
        tile=[64,128,64,128], ranks=[], checks=[], passed=False)
    def save(): args.output.write_text(json.dumps(record, indent=2)+'\n')
    save()
    plans=[]
    for paired, ranges in [(False, [(0,640),(640,640),(1280,512),(1792,512)]),
                           (True, [(0,640),(512,640),(1152,640),(1664,640)])]:
        for rank, (start,width) in enumerate(ranges):
            prepared = load_weights(SimpleNamespace(snapshot=args.snapshot, layer=args.layer,
                intermediate=width, slice_start=start), torch, safe_open,
                ProjectionTrellisTierWeights, prepare_projection_native_trellis_weights, replace)
            launch = compile_mixed_trellis(size_m=args.capacity, hidden_size=5120,
                intermediate_size=width, tier0_num_experts=384, tier1_num_experts=384,
                route_num_experts=384, top_k=6,
                max_m_blocks=route_pack_capacity(args.capacity*6,8,384,topk=6)[1]//8,
                sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
                force_tile_config=(64,128,64,128), swiglu_limit=10.,
                full_rotation_output_dtype='bf16',
                paired_boundary=('first' if rank%2 else 'last') if paired else None)
            descriptor=prepared.descriptor_map
            ownership=None
            if paired:
                descriptor=torch.cat((descriptor, torch.ones(768,device='cuda',dtype=torch.int32)))
                descriptor._mt_projection_counts=prepared.descriptor_map._mt_projection_counts
                ownership=descriptor[-768:]
            binding=bind_mixed_trellis(*prepared.tiers,prepared.global_to_combined,
                descriptor,prepared.rotations,launch,gate_experts=prepared.gate_counts,up_experts=prepared.up_counts)
            buffers=make_mixed_trellis_buffers(launch,device=torch.device('cuda',0),sms=props.multi_processor_count)
            plans.append((binding,buffers,ownership))
            record['ranks'].append(dict(paired=paired,rank=rank,start=start,width=width,
                gate_counts=prepared.gate_counts,up_counts=prepared.up_counts))
            print('loaded', record['ranks'][-1], flush=True); save()
    cases=[(1,6),(8,6),(8,30),(16,60)]
    if args.capacity==80: cases += [(24,60),(64,384)]
    for rows,unique in cases:
        torch.manual_seed(410917+rows+unique)
        x=torch.randn(rows,5120,device='cuda',dtype=torch.bfloat16)
        members=torch.randperm(384,device='cuda')[:unique]
        ids=members[torch.arange(rows*6,device='cuda').reshape(rows,6)%unique].to(torch.int32)
        weights=torch.softmax(torch.randn(rows,6,device='cuda'),dim=1)
        if rows>1: ids[1,1]=-1; weights[1,1]=0
        graphs=[];outputs=[]
        for binding,buffers,_ in plans:
            for _ in range(2): run_bound_mixed_trellis(x,weights,ids,binding,buffers)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out=run_bound_mixed_trellis(x,weights,ids,binding,buffers)
            graphs.append(graph);outputs.append(out)
        # Per-expert bits select a partner in each pair, never per-row ownership.
        expert=torch.arange(384,device='cuda',dtype=torch.int32)
        patterns=[torch.zeros_like(expert),torch.full_like(expert,3),expert%4,(expert*7+1)%4]
        for mutation,owners in enumerate(patterns):
            for rank,(_,buffers,metadata) in enumerate(plans[4:]):
                metadata.fill_(1)
                metadata[:384].copy_(((owners>>(rank//2))&1)==rank%2)
                buffers.fc1.fill_(float('nan'))
            x.mul_(-.75)
            if mutation==3: ids.copy_(ids.roll(1,dims=1));weights.copy_(weights.roll(1,dims=1))
            for graph in graphs: graph.replay()
            torch.cuda.synchronize()
            reference=sum(out[:rows].float() for out in outputs[:4])
            candidate=sum(out[:rows].float() for out in outputs[4:])
            assert torch.isfinite(reference).all() and reference.abs().max()>0
            assert torch.isfinite(candidate).all()
            delta=(candidate-reference).abs()
            error=float(delta.max()/reference.abs().max())
            relative_l2=float((candidate-reference).norm()/reference.norm())
            check=dict(rows=rows,unique=unique,mutation=mutation,
                relative_max_error=error,relative_l2=relative_l2,passed=error<=.006 and relative_l2<=.003)
            record['checks'].append(check);save();print(check,flush=True)
            assert check['passed'],check
            for (binding,buffers,_),out in zip(plans[4:],outputs[4:]):
                before=out[:rows].clone()
                run_bound_mixed_trellis(x,weights,ids,binding,buffers)
                torch.cuda.synchronize()
                assert torch.equal(before,out[:rows]),'paired graph/eager mismatch'
    record['passed']=True;save()

if __name__=='__main__': main()
