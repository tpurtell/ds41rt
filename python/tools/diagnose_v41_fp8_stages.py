#!/usr/bin/env python3
"""Time exact exported one-row quantization and GEMM stages for narrow FP8 plans."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from safetensors import safe_open
    torch.manual_seed(916)
    manifest=json.loads(args.manifest.read_text())
    for name,digest in manifest['artifacts'].items():
        assert hashlib.sha256((args.manifest.parent/name).read_bytes()).hexdigest()==digest,name
    header=(args.manifest.parent/'v41_fp8_variants.h').read_text()
    lib=C.CDLL(str(args.native.resolve()));P,I,U=C.c_void_p,C.c_int32,C.c_uint64
    def bind(name,signature):
        fn=getattr(lib,name);fn.argtypes=signature;fn.restype=I
        def run(*values):
            result=fn(*values);assert result==0,(name,result)
        return run
    initialize=bind('ds41rt_v41_fp8_matrix_initialize',[I,I,I,C.POINTER(P)])
    pack=bind('ds41rt_v41_fp8_matrix_pack_scales',[P,P,I,I,P])
    init_scratch=bind('ds41rt_v41_fp8_initialize_scratch',[P,P,U,P,P])
    full=bind('ds41rt_v41_fp8_launch',[P,P,P,P,P,U,P,P,I,P])
    reduce=bind('ds41rt_v41_fp8_reduce_splits',[P,P,I,I,I,P])
    weight_map=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    def load(name):
        with safe_open(args.snapshot/weight_map[name],framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
    def raw(symbol,values,stream_index):
        function=getattr(lib,symbol);function.argtypes=[C.POINTER(P),I];function.restype=None
        pointers=(P*len(values))(*(C.cast(C.pointer(value),P) for value in values))
        def run():
            values[stream_index].value=torch.cuda.current_stream().cuda_stream
            values[-1].value=0
            function(pointers,len(values));assert values[-1].value==0
        return run
    report={'scope':__doc__,'native_sha256':hashlib.sha256(args.native.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'cases':[]}
    for projection,prefix in [('q_a','layers.0.attn.wq_a'),('kv','layers.0.attn.wkv')]:
        weight,scale=load(prefix+'.weight'),load(prefix+'.scale');n,k=weight.shape
        source=torch.randn((1,k),device='cuda').bfloat16()
        alpha=torch.empty(1,device='cuda');packed=torch.empty(n*k//32,device='cuda',dtype=torch.uint8)
        pack(scale.data_ptr(),packed.data_ptr(),k,n,torch.cuda.current_stream().cuda_stream)
        owners=[];graphs={}
        for capacity in (1,16):
            variant=next(v for v in manifest['variants'] if v['label']==f'v41_{projection}_fp8_m{capacity}')
            handle=P();initialize(capacity,k,n,C.byref(handle))
            scratch=torch.empty(variant['scratch_bytes'],device='cuda',dtype=torch.uint8)
            out=torch.empty((1,n),device='cuda',dtype=torch.bfloat16)
            init_scratch(handle,scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),torch.cuda.current_stream().cuda_stream)
            base=scratch.data_ptr();a=base+variant['activation_values_offset'];sr=base+variant['activation_row_scales_offset'];sm=base+variant['activation_mma_scales_offset'];c=base+variant['split_k_offset']
            grid=int(re.search(variant['label']+r'_grids\[\] = \{(\d+)',header)[1])
            quant=raw(variant['quant_abi']['symbol'],[P(source.data_ptr()),P(a),P(sr),P(sm),I(1),I(grid),P(),I()],6)
            gemm=raw(variant['gemm_abi']['symbol'],[P(a),P(weight.data_ptr()),P(sm),P(packed.data_ptr()),P(c),P(c),P(c),P(c),P(alpha.data_ptr()),I(1),P(),I()],10)
            def gemm_reduce(gemm=gemm,c=c,out=out,variant=variant):
                gemm();reduce(c,out.data_ptr(),1,n,variant['split_k_slices'],torch.cuda.current_stream().cuda_stream)
            full(handle,source.data_ptr(),weight.data_ptr(),packed.data_ptr(),base,scratch.numel(),alpha.data_ptr(),out.data_ptr(),1,torch.cuda.current_stream().cuda_stream)
            reference=out.clone();out.fill_(float('nan'))
            quant();gemm_reduce();torch.cuda.synchronize();torch.testing.assert_close(out,reference,rtol=0,atol=0)
            for stage,fn in [('quant',quant),('gemm_reduce',gemm_reduce)]:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):fn()
                graphs[f'{capacity}_{stage}']=graph
            owners.append((scratch,out,quant,gemm,gemm_reduce))
        samples={key:[] for key in graphs}
        for iteration in range(8):
            order=list(graphs)
            for key in (order if iteration%2 else list(reversed(order))):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):graphs[key].replay()
                end.record();end.synchronize();samples[key].append(start.elapsed_time(end))
        case={'projection':projection,'native_equivalence':True,'warm_us':samples,'median_us':{key:statistics.median(value) for key,value in samples.items()}}
        report['cases'].append(case);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(case),flush=True)
        for graph in graphs.values():graph.reset()


if __name__=='__main__':main()
