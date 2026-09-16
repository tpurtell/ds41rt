#!/usr/bin/env python3
"""Screen upstream expert numerics at DS41RT geometry before native adaptation.

Uses prepared GPU paths, not the serving native ABI. No serving speedup claim.
"""
import argparse
import ctypes as C
from contextlib import ExitStack, nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
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
    p.add_argument('--compact-native', action='store_true', help='Adapted compact native V4.1 routing and FP32 route output')
    p.add_argument('--cold-timing', action='store_true', help='Also measure after a 256 MiB device-buffer write')
    p.add_argument('--native-lib', type=Path, help='Actual TP2 serving library baseline (N1152 only)')
    p.add_argument('--compare-timing', action='store_true',
                   help='Balanced prepared-path graph comparison; not native ABI or serving timing')
    args=p.parse_args()
    if args.compare_timing:
        args.direct_compact = True
        args.activation = ['silu_v41', 'silu']
    if args.direct_compact and not args.compare_timing and args.activation != ['silu']:
        p.error('--direct-compact requires --activation silu')
    if args.native_lib and (not args.compare_timing or args.intermediate != 1152):
        p.error('--native-lib requires --compare-timing and --intermediate 1152')
    if args.compact_native and not args.native_lib:
        p.error('--compact-native requires --native-lib')
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
        if args.native_lib and activation == 'silu_v41':
            continue
        prepared[activation]=prepare_tp_moe_fp4_experts(
            a=torch.empty(1,h,device='cuda',dtype=torch.bfloat16),a1_gscale=ones,
            w1_fp4=torch.cat([weights['w3'],weights['w1']],dim=1),
            w1_blockscale=torch.cat([scales['w3'],scales['w1']],dim=1),w1_alphas=ones,
            a2_gscale=ones,w2_fp4=weights['w2'].clone(),w2_blockscale=scales['w2'].clone(),
            w2_alphas=ones,activation=activation,quant_mode='w4a8_mx',
            source_format='fp4_e8m0_k32',swiglu_limit=10)
    if args.native_lib:
        from _v41_expert_native import Native, library, P, U, check
        lib = library(str(args.native_lib), tp2=True)
        for name, types in {
            'ds41rt_v41_expert_input_quant_initialize': [C.POINTER(P)],
            'ds41rt_v41_expert_input_quantize_async': [P, P, P, U, P],
            'ds41rt_v41_finish_local_experts_async': [P, P, P, U, U, P],
        }.items():
            fn = getattr(lib, name); fn.argtypes = types; fn.restype = C.c_int32
        quant = P(); check(lib.ds41rt_v41_expert_input_quant_initialize(C.byref(quant)))
        native_weights = [torch.empty((e, size), device='cuda', dtype=torch.uint8)
                          for size in (n*h, n*h//16, n*h//2, n*h//32)]
        for expert in range(e):
            sources = [weights[name][expert] for name in ('w1', 'w3', 'w2')]
            sources += [scales[name][expert] for name in ('w1', 'w3', 'w2')]
            check(lib.ds41rt_v41_pack_expert_async(
                (P*6)(*[v.data_ptr() for v in sources]),
                (P*4)(*[v[expert].data_ptr() for v in native_weights]),
                n, torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
    report={'scope':__doc__,'command':sys.argv,'fork_revision':subprocess.check_output(['git','-C',str(args.b12x_root),'rev-parse','HEAD'],text=True).strip(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'device':str(torch.cuda.get_device_properties(args.device)),
            'geometry':{'experts':e,'hidden':h,'intermediate':n,'topk':6},'cases':[]}
    if args.native_lib:
        report['native_library'] = {'path': str(args.native_lib), 'sha256': hashlib.sha256(args.native_lib.read_bytes()).hexdigest(), 'output': 'Native local FP32 routes summed and rounded once to BF16; excludes cross-device TP2 reduction'}
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
        owners = ExitStack()
        timed_graphs = []
        native_route_references = []
        for activation in (args.activation or ['silu_v41','silu']):
            x=original_x.clone();ids=original_ids.clone();output=torch.empty_like(x)
            use_compact = args.direct_compact and activation == 'silu'
            if args.native_lib and activation == 'silu_v41':
                wire = torch.empty((rows, 5280), device='cuda', dtype=torch.uint8)
                native = Native(lib, rows, native_weights, wire, ids, routing, tp2=True)
                def run_native():
                    check(lib.ds41rt_v41_expert_input_quantize_async(
                        quant, x.data_ptr(), wire.data_ptr(), rows, torch.cuda.current_stream().cuda_stream))
                    native.run(rows)
                    check(lib.ds41rt_v41_finish_local_experts_async(native.output.data_ptr(), None, output.data_ptr(), rows, int(native.token_accumulation), torch.cuda.current_stream().cuda_stream))
                    return output
                context = nullcontext(SimpleNamespace(run=run_native, implementation='native_tp2'))
            elif use_compact:
                from b12x.moe._shared.kernels.w4a8_compact_micro import launch_w4a8_compact_micro, micro_scratch_nbytes
                from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_topk_sum_launch_flat
                runtime = prepared[activation]._impl.representation_for('w4a8_mx')
                scratch = torch.empty(micro_scratch_nbytes(rows, h, n, 6, native_v41=args.compact_native), device='cuda', dtype=torch.uint8)
                route_holder = []
                def run_compact():
                    routes = launch_w4a8_compact_micro(
                        scratch=scratch, a=x, topk_ids=ids, topk_weights=routing,
                        w13=runtime.w13_rp, w13_scales=runtime.w13_sfb,
                        w2=runtime.w2_rp, w2_scales=runtime.w2_sfb,
                        alpha1=ones, alpha2=ones, input_scale=ones, down_scale=ones,
                        max_tokens=rows, num_topk=6, swiglu_limit=10, fast_math=False, native_v41=args.compact_native)
                    route_holder[:] = [routes]
                    if args.compact_native:
                        check(lib.ds41rt_v41_finish_local_experts_async(routes.data_ptr(), None, output.data_ptr(), rows, 0, torch.cuda.current_stream().cuda_stream))
                    else:
                        _w4a16_topk_sum_launch_flat(routes, output, rows, 6, h, 'bf16', torch.cuda.current_stream().cuda_stream)
                    return output
                context = nullcontext(SimpleNamespace(run=run_compact, implementation='direct_compact'))
            else:
                context = make_tp_moe_fp4_binding(a=x,experts=prepared[activation],topk_weights=routing,topk_ids=ids,quant_mode='w4a8_mx',swiglu_limit=10,output=output,fast_math=False)
            binding = owners.enter_context(context)
            with nullcontext(binding):
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
                    if (use_compact and not args.compact_native) or (not args.compact_native and activation == 'silu' and n % 128 == 64 and binding.implementation == 'micro'):
                        compact_wanted = compact_expected[min(cycle, 1)]
                        compact_rel = float((value - compact_wanted).norm() / compact_wanted.norm())
                        compact_cos = float(torch.nn.functional.cosine_similarity(value.flatten(), compact_wanted.flatten(), dim=0))
                        checks[-1].update(compact_reference_relative_l2=compact_rel,
                                          compact_reference_cosine=compact_cos,
                                          compact_reference_gate=compact_rel < .01 and compact_cos > .9999)
                    if args.native_lib and not use_compact:
                        native_route_references.append(native.output.clone())
                    elif args.compact_native:
                        reference_routes = native_route_references[cycle]
                        route_rel = float((route_holder[0] - reference_routes).norm() / reference_routes.norm())
                        checks[-1]['native_route_relative_l2'] = route_rel
                        assert route_rel < .01, route_rel
                result={'rows':rows,'activation':activation,'implementation':binding.implementation,'checks':checks}
                report['cases'].append(result);save();print(json.dumps(result),flush=True)
                if args.compare_timing:
                    gate = 'compact_reference_gate' if use_compact and not args.compact_native else 'native_reference_gate'
                    assert all(check[gate] for check in checks), result
                    timed_graphs.append((activation, graph, result, (binding, x, ids, output, scratch if use_compact else None, (native, wire) if args.native_lib and not use_compact else None)))
                else:
                    graph.reset()
        if args.compare_timing:
            def gpu_state():
                return subprocess.check_output([
                    'nvidia-smi', '--query-gpu=uuid,pstate,clocks.sm,clocks.mem,power.limit,clocks_event_reasons.active',
                    '--format=csv,noheader'], text=True).strip()
            before = gpu_state()
            samples = {name: [] for name, *_ in timed_graphs}
            for _, graph, *_ in timed_graphs:
                for _ in range(20): graph.replay()
            torch.cuda.synchronize()
            for repetition in range(8):
                order = timed_graphs if repetition % 2 == 0 else timed_graphs[::-1]
                for name, graph, *_ in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100): graph.replay()
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 10)
            cold_samples = {name: [] for name, *_ in timed_graphs}
            if args.cold_timing:
                eviction = torch.empty(256 * 1024 * 1024, device='cuda', dtype=torch.uint8)
                for repetition in range(8):
                    order = timed_graphs if repetition % 2 == 0 else timed_graphs[::-1]
                    for name, graph, *_ in order:
                        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(32)]
                        for start, end in events:
                            eviction.zero_()
                            start.record(); graph.replay(); end.record()
                        torch.cuda.synchronize()
                        cold_samples[name].append(statistics.median(start.elapsed_time(end)*1000 for start, end in events))
            after = gpu_state()
            for name, graph, result, _ in timed_graphs:
                result['prepared_graph_timing'] = {'scope': 'Includes BF16 input quantization and route reduction; prepared graph, not native serving ABI',
                    'samples_us': samples[name], 'median_us': statistics.median(samples[name]),
                    'order': 'AB/BA, eight samples of 100 replays', 'gpu_before': before, 'gpu_after': after,
                    'cold_samples_us': cold_samples[name], 'cold_protocol': 'Eight balanced samples; median of 32 individual graph replays each following a 256 MiB write, write excluded' if args.cold_timing else None}
                graph.reset()
            save()
            print(json.dumps({'rows': rows, 'timing_us': samples}), flush=True)
        owners.close()
    # A failed compatibility gate is evidence requiring adaptation, not a relaxed oracle.
    return 0 if all(c['native_reference_gate'] for r in report['cases'] for c in r['checks']) else 1


if __name__=='__main__':raise SystemExit(main())
