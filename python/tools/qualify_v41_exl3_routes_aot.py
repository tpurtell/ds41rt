#!/usr/bin/env python3
"""Check native route packing against a CPU ownership/coverage oracle."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    meta=json.loads((args.aot/'v41_exl3_routes.json').read_text())
    capacity,experts,topk=meta['capacity'],meta['experts'],meta['topk']
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    guard=0x12345678
    owners={name:torch.full((spec['elements']+32,),guard,device=device,dtype=torch.int32)
        for name,spec in meta['buffers'].items()}
    buffers={name:value[16:-16] for name,value in owners.items()}
    pointers=(ct.c_void_p*7)(*[value.data_ptr() for value in buffers.values()])
    sizes=(ct.c_uint64*7)(*[value.numel()*4 for value in buffers.values()])
    library=args.aot/'libv41_exl3_routes.so'
    lib=ct.CDLL(str(library))
    lib.ds41rt_exl3_routes_create.argtypes=[ct.POINTER(ct.c_void_p)]
    lib.ds41rt_exl3_routes_create.restype=ct.c_int
    lib.ds41rt_exl3_routes_destroy.argtypes=[ct.c_void_p]
    lib.ds41rt_exl3_routes_destroy.restype=ct.c_int
    lib.ds41rt_exl3_routes_launch.argtypes=[ct.c_void_p,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_uint64),ct.c_int32,ct.c_void_p]
    lib.ds41rt_exl3_routes_launch.restype=ct.c_int
    context=ct.c_void_p()
    assert lib.ds41rt_exl3_routes_create(ct.byref(context))==0
    graph=None

    def launch(rows,byte_sizes=sizes):
        return lib.ds41rt_exl3_routes_launch(context,pointers,byte_sizes,rows,
            ct.c_void_p(torch.cuda.current_stream().cuda_stream))

    def prepare(seed,invalid=False):
        generator=torch.Generator().manual_seed(seed)
        ids=torch.randint(-2,experts+2,(capacity*topk,),generator=generator,dtype=torch.int32)
        mapping=torch.randperm(experts,generator=generator,dtype=torch.int32)
        mapping[::11]=-1
        mapping[1]=experts # Both invalid raw IDs and invalid mapped IDs are ignored.
        if invalid: ids.fill_(-1)
        buffers['topk_ids'].copy_(ids)
        buffers['expert_map'].copy_(mapping)
        return ids.tolist(),mapping.tolist()

    def check(rows,ids,mapping):
        live=rows*topk
        expected=[[] for _ in range(experts)]
        for slot,raw in enumerate(ids[:live]):
            if 0<=raw<experts and 0<=mapping[raw]<experts:
                expected[mapping[raw]].append(slot)
        expected_blocks=[e for e,values in enumerate(expected) for _ in range((len(values)+7)//8)]
        count=int(buffers['packed_route_count'].item())
        assert count==len(expected_blocks)*8,(count,len(expected_blocks))
        blocks=buffers['block_expert_ids'].cpu().tolist()
        packed=buffers['packed_route_indices'].cpu().tolist()
        assert blocks[:len(expected_blocks)]==expected_blocks
        assert all(v==-1 for v in blocks[len(expected_blocks):])
        found=[[] for _ in range(experts)]
        for block,e in enumerate(expected_blocks):
            for slot in packed[block*8:block*8+8]:
                assert 0<=slot<=live
                if slot!=live: found[e].append(slot)
        assert [sorted(v) for v in found]==expected
        assert all(v==live for v in packed[count:])
        for owner in owners.values():
            assert bool((owner[:16]==guard).all()) and bool((owner[-16:]==guard).all())

    checks=[]
    try:
        assert launch(0)==1 and launch(capacity+1)==1
        too_small=(ct.c_uint64*7)(*sizes);too_small[2]-=4
        assert launch(1,too_small)==1
        for rows in sorted(set([1,min(3,capacity),max(1,capacity-1),capacity])):
            for seed,invalid in [(41,False),(42,False),(43,True)]:
                ids,mapping=prepare(seed,invalid)
                assert launch(rows)==0
                check(rows,ids,mapping)
            checks.append(dict(rows=rows,random_and_empty_routes=True,guard_regions=True))
        rows=min(3,capacity)
        ids,mapping=prepare(44)
        assert launch(rows)==0
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert launch(rows)==0
        for seed in (45,46):
            ids,mapping=prepare(seed)
            graph.replay()
            check(rows,ids,mapping)
        report=dict(passed=True,scope='Native metadata route packing; CPU coverage, ownership, padding and guard oracle. Not expert compute or serving qualification.',
            manifest_sha256=hashlib.sha256((args.aot/'v41_exl3_routes.json').read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            compute=meta['compute'],capacity=capacity,experts=experts,topk=topk,
            small_prefix=meta['small_prefix'],checks=checks,
            changed_graph_inputs_and_mapping=True,invalid_host_bounds_rejected=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        torch.cuda.synchronize()
        del graph
        assert lib.ds41rt_exl3_routes_destroy(context)==0


if __name__=='__main__':
    main()
