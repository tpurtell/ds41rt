#!/usr/bin/env python3
"""Compare native FP8 with upstream BF16 narrow projections on checkpoint weights.

This is a warm component probe. BF16 requires expanded resident weights and
changes activation quantization; neither performance nor numerical acceptance
of this probe constitutes serving acceptance.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b12x-root', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0,str(args.b12x_root.resolve()))
    import torch
    from safetensors import safe_open
    from b12x.gemm import bf16_gemv
    from b12x.preparation import PreparationSession, PreparedCall
    from tests.gemm.test_gemm_block_fp8_linear import _assert_v41_accumulation_matches_reference
    from tests.gemm.test_bf16_gemv import _assert_matches_f32_ref
    torch.manual_seed(41410)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(args.native.resolve()))
    ptr,i32,u64 = C.c_void_p,C.c_int32,C.c_uint64
    class Info(C.Structure):
        _fields_ = [(name,C.c_uint32) for name in ('abi','capacity','k','n')] + [(name,u64) for name in ('scratch','values','row_scales','mma_scales','weight_scales')]
    def bind(name,types):
        fn = getattr(lib,'ds41rt_v41_fp8_'+name); fn.argtypes=types; fn.restype=i32
        def run(*values):
            result = fn(*values)
            assert result == 0,(name,result)
        return run
    info_fn=bind('matrix_info',[i32,i32,i32,C.POINTER(Info)])
    init=bind('matrix_initialize',[i32,i32,i32,C.POINTER(ptr)])
    pack=bind('matrix_pack_scales',[ptr,ptr,i32,i32,ptr])
    scratch_init=bind('initialize_scratch',[ptr,ptr,u64,ptr,ptr])
    launch=bind('launch',[ptr,ptr,ptr,ptr,ptr,u64,ptr,ptr,i32,ptr])
    weight_map=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    def load(name):
        with safe_open(args.snapshot/weight_map[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name).cuda().contiguous()
    report={'scope':__doc__,'command':sys.argv,'revision':subprocess.check_output(['git','-C',str(args.b12x_root),'rev-parse','HEAD'],text=True).strip(),
            'native_sha256':hashlib.sha256(args.native.read_bytes()).hexdigest(),'cases':[]}
    for prefix in ('layers.0.attn.wkv','layers.0.attn.wq_a','layers.8.attn.indexer.wq_b'):
        weight,scales=load(prefix+'.weight'),load(prefix+'.scale')
        n,k=weight.shape
        expanded=(weight.float()*scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
        for capacity in (1,16,256):
            info,handle=Info(),ptr()
            info_fn(capacity,k,n,C.byref(info)); init(capacity,k,n,C.byref(handle))
            packed=torch.empty(info.weight_scales,device='cuda',dtype=torch.uint8)
            scratch=torch.empty(info.scratch,device='cuda',dtype=torch.uint8)
            alpha=torch.empty(1,device='cuda')
            stream=torch.cuda.current_stream().cuda_stream
            pack(scales.data_ptr(),packed.data_ptr(),k,n,stream)
            scratch_init(handle,scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),stream)
            source=torch.randn((capacity,k),device='cuda').bfloat16()
            fp8_out=torch.empty((capacity,n),device='cuda',dtype=torch.bfloat16)
            bf16_out=torch.empty_like(fp8_out)
            alternative = capacity == 1
            if alternative:
                alt_info,alt_handle=Info(),ptr()
                info_fn(16,k,n,C.byref(alt_info));init(16,k,n,C.byref(alt_handle))
                alt_scratch=torch.empty(alt_info.scratch,device='cuda',dtype=torch.uint8)
                alt_out=torch.empty_like(fp8_out)
                scratch_init(alt_handle,alt_scratch.data_ptr(),alt_scratch.numel(),alpha.data_ptr(),stream)
            with PreparationSession(device=source.device,autotune=False,compile_workers=2) as session:
                plan=bf16_gemv.plan(bf16_gemv.query_from_call(source,expanded,out=bf16_out))
                session.prepare((plan.request(name='narrow',prepare_call=lambda state:PreparedCall(run=lambda:state.run(source,expanded,out=bf16_out))),))
                session.freeze()
                for rows in sorted({1,capacity}):
                    def fp8():
                        launch(handle,source.data_ptr(),weight.data_ptr(),packed.data_ptr(),scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),fp8_out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
                    def bf16():
                        bf16_gemv.mm(source[:rows],expanded,out=bf16_out[:rows],plan=plan)
                    def alt():
                        launch(alt_handle,source.data_ptr(),weight.data_ptr(),packed.data_ptr(),alt_scratch.data_ptr(),alt_scratch.numel(),alpha.data_ptr(),alt_out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
                    graphs={}
                    calls=[('fp8',fp8),('bf16',bf16)]
                    if alternative:calls.append(('fp8_capacity16',alt))
                    for name,fn in calls:
                        fn();torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            for _ in range(20):fn()
                        graphs[name]=graph
                    source.neg_()
                    fp8_out.fill_(float('nan'));bf16_out.fill_(float('nan'))
                    if alternative:alt_out.fill_(float('nan'))
                    for graph in graphs.values():graph.replay()
                    torch.cuda.synchronize()
                    _assert_v41_accumulation_matches_reference(source[:rows],weight,scales.view(torch.float8_e8m0fnu),fp8_out[:rows])
                    _assert_matches_f32_ref(bf16_out[:rows],source[:rows],expanded)
                    assert torch.isfinite(fp8_out[:rows]).all() and torch.isfinite(bf16_out[:rows]).all()
                    assert torch.isnan(fp8_out[rows:]).all() and torch.isnan(bf16_out[rows:]).all()
                    if alternative:torch.testing.assert_close(alt_out,fp8_out,rtol=0,atol=0)
                    samples={key:[] for key in graphs}
                    before=torch.cuda.memory_allocated()
                    for iteration in range(6):
                        order=list(graphs)
                        for name in (order if iteration%2 else list(reversed(order))):
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(50):graphs[name].replay()
                            end.record();end.synchronize()
                            samples[name].append(start.elapsed_time(end))
                    assert torch.cuda.memory_allocated()==before
                    case={'prefix':prefix,'capacity':capacity,'rows':rows,'separate_oracles_pass':True,'graph_mutation_and_tail_pass':True,
                          'extra_resident_weight_bytes':expanded.numel()*expanded.element_size()-weight.numel()*weight.element_size()-scales.numel()*scales.element_size(),
                          'alternate_capacity_exact':alternative,'warm_us':samples,'median_us':{key:statistics.median(value) for key,value in samples.items()}}
                    report['cases'].append(case);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(case),flush=True)
                    for graph in graphs.values():graph.reset()


if __name__ == '__main__':
    main()
