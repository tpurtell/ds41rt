#!/usr/bin/env python3
"""Screen upstream expert numerics at DS41RT geometry before native adaptation.

Uses prepared GPU paths, not the serving native ABI. No serving speedup claim.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def compact_reference(x, ids, routing, weights, scales):
    """Compact FP8 contract: BF16 FC1/activation, routing after FC2.

    Independent unpack + Torch matmuls; deliberately distinct from the native
    reference's routing-before-intermediate-quantization contract.
    """
    import torch
    from tests.moe.test_v41_expert_numerics import quantized_rows

    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        0, -.5, -1, -1.5, -2, -3, -4, -6], device=x.device)

    def unpack(name, expert):
        codes = weights[name][expert]
        values = torch.stack((lut[(codes & 15).long()],
                              lut[(codes >> 4).long()]), -1).flatten(-2)
        return values * scales[name][expert].view(
            torch.float8_e8m0fnu).float().repeat_interleave(32, -1)

    qx = quantized_rows(x)
    partials = torch.zeros((*ids.shape, x.shape[1]), device=x.device)
    for expert in ids.unique().tolist():
        rows, slots = torch.where(ids == expert)
        gate = (qx[rows] @ unpack('w1', expert).T).bfloat16().float().clamp_max(10)
        up = (qx[rows] @ unpack('w3', expert).T).bfloat16().float().clamp(-10, 10)
        mid = (torch.nn.functional.silu(gate) * up).bfloat16()
        down = quantized_rows(mid) @ unpack('w2', expert).T
        partials[rows, slots] = (down * routing[rows, slots, None]).bfloat16().float()
    return partials.sum(1).bfloat16().float()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b12x-root',type=Path,required=True)
    p.add_argument('--intermediate',type=int,choices=(576,1152),required=True)
    p.add_argument('--rows',type=int,nargs='+',default=[1,16])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--activation',choices=('silu_v41','silu'),action='append')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--direct-compact', action='store_true',
                   help='Exercise compact FP8 kernels directly, outside public backend selection')
    args=p.parse_args()
    if args.direct_compact and args.activation != ['silu']:
        p.error('--direct-compact requires --activation silu')
    sys.path.insert(0,str(args.b12x_root))
    import torch
    from tests._reference.helpers import prepare_tp_moe_fp4_experts, make_tp_moe_fp4_binding
    from tests.moe.test_v41_expert_numerics import reference
    from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    h,e,n=5120,384,args.intermediate
    torch.manual_seed(41900+n)
    weights,scales={},{}
    for name,shape in [('w1',(e,n,h//2)),('w3',(e,n,h//2)),('w2',(e,h,n//2))]:
        weights[name]=torch.randint(256,shape,device='cuda',dtype=torch.uint8)
        scales[name]=torch.randint(121,124,(*shape[:-1],shape[-1]//16),device='cuda',dtype=torch.uint8)
    ones=torch.ones(e,device='cuda')
    prepared={}
    for activation in (args.activation or ['silu_v41','silu']):
        prepared[activation]=prepare_tp_moe_fp4_experts(
            a=torch.empty(1,h,device='cuda',dtype=torch.bfloat16),a1_gscale=ones,
            w1_fp4=torch.cat([weights['w3'],weights['w1']],dim=1),
            w1_blockscale=torch.cat([scales['w3'],scales['w1']],dim=1),w1_alphas=ones,
            a2_gscale=ones,w2_fp4=weights['w2'].clone(),w2_blockscale=scales['w2'].clone(),
            w2_alphas=ones,activation=activation,quant_mode='w4a8_mx',
            source_format='fp4_e8m0_k32',swiglu_limit=10)
    report={'scope':__doc__,'command':sys.argv,'fork_revision':subprocess.check_output(['git','-C',str(args.b12x_root),'rev-parse','HEAD'],text=True).strip(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'device':str(torch.cuda.get_device_properties(args.device)),
            'geometry':{'experts':e,'hidden':h,'intermediate':n,'topk':6},'cases':[]}
    report['fork_dirty']=subprocess.check_output(['git','-C',str(args.b12x_root),'status','--porcelain'],text=True)
    report['fork_source_sha256']={name:hashlib.sha256((args.b12x_root/name).read_bytes()).hexdigest() for name in ['b12x/moe/_shared/kernels/tiny_decode.py','b12x/moe/fused_moe/_impl.py','b12x/moe/fused_moe/_preparation.py','b12x/moe/_shared/kernels/reference.py','b12x/moe/_shared/kernels/w4a8_compact_micro.py','b12x/moe/_shared/kernels/w4a8_compact_projection.py','b12x/moe/_shared/kernels/w4a8_compact_activation.py','b12x/moe/_shared/kernels/w4a8_phase2.py']}
    report['reference_semantics']={'native':'V4.1 BF16 projection boundaries and routing before intermediate FP8 quantization','generic':'Declared W4A8-MX FP8 activation reference; tiny-decode actually consumes BF16 and accumulates BF16 atomically, so this is not its exact arithmetic oracle','compact':'FP8 input, BF16 FC1 and activation boundaries, FP8 intermediate, routing after FC2 before BF16 route output; FP32 top-k sum rounded to BF16'}
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for rows in args.rows:
        torch.manual_seed(42000+rows+n)
        original_x=(torch.randn(rows,h,device='cuda')*.5).bfloat16()
        original_ids=torch.rand(rows,e,device='cuda').topk(6,dim=1).indices.int()
        routing=torch.rand(rows,6,device='cuda');routing=(routing/routing.sum(-1,keepdim=True)*1.5).float()
        expected=[reference(original_x,original_ids,routing,weights,scales),reference(original_x*-.5,(original_ids+1)%e,routing,weights,scales)]
        generic_expected=[moe_reference_w4a8_mx(xx,torch.cat([weights['w3'],weights['w1']],dim=1),torch.cat([scales['w3'],scales['w1']],dim=1),None,ones,weights['w2'],scales['w2'],None,ones,ii,routing,e,h,n,activation='silu',swiglu_limit=10) for xx,ii in [(original_x,original_ids),(original_x*-.5,(original_ids+1)%e)]]
        compact_expected = [compact_reference(xx, ii, routing, weights, scales)
                            for xx, ii in [(original_x, original_ids),
                                           (original_x * -.5, (original_ids + 1) % e)]]
        for activation in (args.activation or ['silu_v41','silu']):
            x=original_x.clone();ids=original_ids.clone();output=torch.empty_like(x)
            if args.direct_compact:
                from b12x.moe._shared.kernels.w4a8_compact_micro import launch_w4a8_compact_micro, micro_scratch_nbytes
                from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat
                runtime = prepared[activation]._impl.representation_for('w4a8_mx')
                scratch = torch.empty(micro_scratch_nbytes(rows, h, n, 6), device='cuda', dtype=torch.uint8)
                def run_compact():
                    routes = launch_w4a8_compact_micro(
                        scratch=scratch, a=x, topk_ids=ids, topk_weights=routing,
                        w13=runtime.w13_rp, w13_scales=runtime.w13_sfb,
                        w2=runtime.w2_rp, w2_scales=runtime.w2_sfb,
                        alpha1=ones, alpha2=ones, input_scale=ones, down_scale=ones,
                        max_tokens=rows, num_topk=6, swiglu_limit=10, fast_math=False)
                    _w4a16_topk_sum_launch_flat(routes, output, rows, 6, h, 'bf16', torch.cuda.current_stream().cuda_stream)
                    return output
                context = nullcontext(SimpleNamespace(run=run_compact, implementation='direct_compact'))
            else:
                context = make_tp_moe_fp4_binding(a=x,experts=prepared[activation],topk_weights=routing,topk_ids=ids,quant_mode='w4a8_mx',swiglu_limit=10,output=output,fast_math=False)
            with context as binding:
                binding.run();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):actual=binding.run()
                checks=[]
                for cycle in range(3):
                    if cycle==1:x.copy_(original_x*-.5);ids.copy_((original_ids+1)%e)
                    output.fill_(float('nan'))
                    allocated=torch.cuda.memory_allocated()
                    allocations=torch.cuda.memory_stats()['allocation.all.allocated']
                    graph.replay();torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated()==allocated
                    assert torch.cuda.memory_stats()['allocation.all.allocated']==allocations
                    assert torch.isfinite(actual).all() and actual.norm()>0
                    wanted=expected[min(cycle,1)];value=actual.float()
                    rel=float((value-wanted).norm()/wanted.norm())
                    cosine=float(torch.nn.functional.cosine_similarity(value.flatten(),wanted.flatten(),dim=0))
                    own_wanted=expected[min(cycle,1)] if activation=='silu_v41' else generic_expected[min(cycle,1)]
                    own_rel=float((value-own_wanted).norm()/own_wanted.norm())
                    own_cosine=float(torch.nn.functional.cosine_similarity(value.flatten(),own_wanted.flatten(),dim=0))
                    repeat_equal=None if cycle!=2 else torch.equal(actual,previous)
                    previous=actual.clone()
                    checks.append({'identical_input_replay_exact':repeat_equal,'cycle':cycle,'relative_l2':rel,'cosine':cosine,'native_reference_gate':rel<.01 and cosine>.9999,'own_reference_relative_l2':own_rel,'own_reference_cosine':own_cosine})
                    if args.direct_compact or (activation == 'silu' and n % 128 == 64 and binding.implementation == 'micro'):
                        compact_wanted = compact_expected[min(cycle, 1)]
                        compact_rel = float((value - compact_wanted).norm() / compact_wanted.norm())
                        compact_cos = float(torch.nn.functional.cosine_similarity(value.flatten(), compact_wanted.flatten(), dim=0))
                        checks[-1].update(compact_reference_relative_l2=compact_rel,
                                          compact_reference_cosine=compact_cos,
                                          compact_reference_gate=compact_rel < .01 and compact_cos > .9999)
                result={'rows':rows,'activation':activation,'implementation':binding.implementation,'checks':checks}
                report['cases'].append(result);save();print(json.dumps(result),flush=True)
                graph.reset()
    # A failed compatibility gate is evidence requiring adaptation, not a relaxed oracle.
    return 0 if all(c['native_reference_gate'] for r in report['cases'] for c in r['checks']) else 1


if __name__=='__main__':raise SystemExit(main())
