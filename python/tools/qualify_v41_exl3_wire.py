#!/usr/bin/env python3
"""Qualify native FP8 wire reconstruction, including the actual B12x quantizer."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import _pinned_sparkinfer


def quantize_wire(source,wire):
    import torch
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.utils import make_ptr,current_cuda_stream
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot,mxfp8_rows_quant_aot_grid
    quant=compile_mxfp8_rows_quant_aot(size_k=5120,expected_m=80,amax_floor=1e-4,wire_rows=True)
    rows=source.shape[0]
    ptr=lambda dtype,address:make_ptr(dtype,address,cute.AddressSpace.gmem,assumed_align=16)
    quant(ptr(cutlass.BFloat16,source.data_ptr()),ptr(cutlass.Uint32,wire.data_ptr()),
        ptr(cutlass.Uint8,wire.data_ptr()+5120),ptr(cutlass.Uint8,wire.data_ptr()),rows,
        mxfp8_rows_quant_aot_grid(size_k=5120,rows=rows,expected_m=80,
            sm_count=torch.cuda.get_device_properties(source.device).multi_processor_count),current_cuda_stream())


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    capacity=4096
    wire_owner=torch.full((capacity*5280+64,),0x5a,dtype=torch.uint8,device='cuda')
    out_owner=torch.full((capacity*5120*2+64,),0x5a,dtype=torch.uint8,device='cuda')
    wire=wire_owner[32:-32].view(capacity,5280)
    output=out_owner[32:-32].view(torch.bfloat16).view(capacity,5120)
    lib=ct.CDLL(str(args.library))
    lib.ds41rt_v41_exl3_wire_initialize.argtypes=[ct.POINTER(ct.c_void_p)]
    lib.ds41rt_v41_exl3_wire_initialize.restype=ct.c_int
    lib.ds41rt_v41_exl3_wire_destroy.argtypes=[ct.c_void_p]
    lib.ds41rt_v41_exl3_wire_decode.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,ct.c_void_p,ct.c_uint64,ct.c_uint32,ct.c_void_p]
    lib.ds41rt_v41_exl3_wire_decode.restype=ct.c_int
    context=ct.c_void_p();assert lib.ds41rt_v41_exl3_wire_initialize(ct.byref(context))==0
    def launch(rows,input_bytes=wire.numel(),out_ptr=output.data_ptr()):
        return lib.ds41rt_v41_exl3_wire_decode(context,wire.data_ptr(),input_bytes,out_ptr,output.numel()*2,rows,torch.cuda.current_stream().cuda_stream)
    def reference(rows):
        values=wire[:rows,:5120].contiguous().view(torch.float8_e4m3fn).float()
        scales=wire[:rows,5120:].int()
        result=torch.ldexp(values,scales.repeat_interleave(32,dim=1)-127)
        result.masked_fill_(scales.repeat_interleave(32,dim=1)==255,float('nan'))
        return result.to(torch.bfloat16)
    def check(rows):
        expected=reference(rows)
        actual=output[:rows]
        assert torch.equal(torch.isnan(actual),torch.isnan(expected))
        valid=~torch.isnan(expected)
        assert torch.equal(actual.view(torch.int16)[valid],expected.view(torch.int16)[valid])
        for owner in [wire_owner,out_owner]:
            assert bool((owner[:32]==0x5a).all()) and bool((owner[-32:]==0x5a).all())
    graph=None
    try:
        assert launch(0)==1 and launch(capacity+1)==1 and launch(1,5279)==1
        assert launch(1,out_ptr=wire.data_ptr())==1
        rows_index=torch.arange(capacity,device='cuda')[:,None]
        wire[:,:5120]=((torch.arange(5120,device='cuda')[None,:]+rows_index*7)%256).to(torch.uint8)
        wire[:,5120:]=((torch.arange(160,device='cuda')[None,:]+rows_index*13)%256).to(torch.uint8)
        checks=[]
        for rows in [1,3,16,1023,4096]:
            out_owner[32:-32].fill_(0x5a)
            assert launch(rows)==0
            check(rows)
            assert bool((out_owner[32+rows*5120*2:-32]==0x5a).all())
            checks.append(rows)
        # Actual serving quantizer: BF16 -> E4M3 + UE8M0 K32 with the V4.1 floor.
        torch.manual_seed(4128)
        source=torch.randn(16,5120,device='cuda',dtype=torch.bfloat16)
        source[0].zero_();source[1].mul_(0.00001)
        quantize_wire(source,wire)
        assert launch(16)==0;check(16)
        assert torch.isfinite(output[:16]).all() and output[2:16].abs().sum()>0
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): assert launch(3)==0
        wire[2,:5120].zero_()
        output[:3].fill_(float('nan'))
        graph.replay();check(3)
        assert bool((output[2]==0).all())
        report=dict(passed=True,scope='FP8 wire to BF16 boundary only; no expert/serving performance claim.',
            gpu=torch.cuda.get_device_name(0),compute=list(torch.cuda.get_device_capability(0)),
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
            rows=checks,all_fp8_codes_and_scale_bytes=True,finite_bits_and_nan_masks_equal=True,
            actual_b12x_wire_quantizer=True,zero_and_small_groups=True,
            changed_input_graph_replay=True,allocation_guards_and_tail=True,invalid_bounds_and_alias_rejected=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    finally:
        torch.cuda.synchronize()
        del graph
        lib.ds41rt_v41_exl3_wire_destroy(context)


if __name__=='__main__':main()
