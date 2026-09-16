#!/usr/bin/env python3
"""Qualify and time complete native index selection, including carried winners."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', type=int, default=0)
    a = p.parse_args()
    torch.cuda.set_device(a.device)
    libraries = [C.CDLL(x) for x in (a.baseline, a.candidate)]
    results = []
    for k, limit, name in [(512, 1048576, 'ds41rt_v41_index_top512'), (2048, 131072, 'ds41rt_v41_index_top2048_blocks')]:
        fns = [getattr(lib, name) for lib in libraries]
        for fn in fns:
            fn.argtypes = [C.c_void_p]*4+[C.c_uint64, C.c_void_p]+[C.c_int32]*3+[C.c_void_p]
            fn.restype = C.c_int32
        for rows, width in [(1, 8), (1, 512), (1, 4096), (16, 4096), (16, 16384)]:
            torch.manual_seed(rows+width+k)
            scores = [torch.randn(rows, width, device='cuda').bfloat16().float() for _ in range(2)]
            # Disjoint high logical positions, in descending input order.
            positions = [torch.arange(limit-(i+1)*width, limit-i*width, device='cuda', dtype=torch.int64).flip(0).expand(rows,width).contiguous() for i in range(2)]
            buffers = []
            for _ in range(2):
                buffers.append((torch.empty(rows,k,device='cuda',dtype=torch.int64), torch.empty(rows*((width+2*k-1)//(2*k))*k*16,device='cuda',dtype=torch.uint8), torch.empty(rows,k,device='cuda',dtype=torch.int32)))
            def launch(arm):
                carry,scratch,out = buffers[arm]
                for chunk in range(2):
                    status=fns[arm](scores[chunk].data_ptr(), positions[chunk].data_ptr(), carry.data_ptr(), scratch.data_ptr(), scratch.numel(), out.data_ptr(), rows, width, 1-chunk, torch.cuda.current_stream().cuda_stream)
                    if status: raise RuntimeError(status)
            for arm in range(2): launch(arm)
            graphs=[]
            for arm in range(2):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): launch(arm)
                graphs.append(graph)
            checks=[]
            for cycle in range(4):
                if cycle==1:
                    for score in scores: score.zero_()
                if cycle==2:
                    for score in scores:
                        score.normal_(); score[:,::3]=float('-inf'); score[:,1::11]=float('nan')
                if cycle==3:
                    for pos in positions: pos[:,::5]=-1
                for carry,_,out in buffers: carry.fill_(-9);out.fill_(-9)
                allocated=torch.cuda.memory_allocated()
                for graph in graphs: graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated()==allocated
                assert torch.equal(buffers[0][0],buffers[1][0])
                assert torch.equal(buffers[0][2],buffers[1][2])
                # Independent oracle: position order first, then stable descending score.
                sc=torch.cat(scores,dim=1);pos=torch.cat(positions,dim=1)
                valid=(pos>=0)&(pos<limit)&(~torch.isnan(sc))&(sc!=float('-inf'))
                order=pos.masked_fill(~valid,limit).argsort(dim=1,stable=True)
                sc=sc.masked_fill(~valid,float('-inf')).gather(1,order)
                pos=pos.masked_fill(~valid,limit).gather(1,order)
                selected=sc.argsort(dim=1,descending=True,stable=True)[:,:k]
                expected=pos.gather(1,selected)
                if expected.shape[1]<k:
                    expected=torch.cat([expected,torch.full((rows,k-expected.shape[1]),limit,device='cuda')],dim=1)
                expected=expected.sort(dim=1).values
                expected[expected==limit]=-1
                assert torch.equal(buffers[1][2].long(),expected), (k,rows,width,cycle)
                checks.append({'cycle':cycle,'carry_exact':True,'output_exact':True,'oracle_exact':True})
            for score in scores: score.normal_();score.copy_(score.bfloat16().float())
            timings=[[],[]]
            for repeat in range(6):
                for arm in ([0,1] if repeat%2==0 else [1,0]):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100): graphs[arm].replay()
                    end.record();end.synchronize()
                    timings[arm].append(start.elapsed_time(end)*10)
            result={'k':k,'rows':rows,'width':width,'chunks':2,'checks':checks,'baseline_us':timings[0],'candidate_us':timings[1]}
            results.append(result);print(json.dumps(result),flush=True)
    report={'command':sys.argv,'device':str(torch.cuda.get_device_properties(a.device)),'libraries':{key:{'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for key,path in [('baseline',a.baseline),('candidate',a.candidate)]},'results':results}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
