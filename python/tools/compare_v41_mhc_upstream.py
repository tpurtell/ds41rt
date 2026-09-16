#!/usr/bin/env python3
"""Compare native mHC begin against prepared upstream lagged pre on identical inputs."""
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rounding-diagnostics', action='store_true')
    p.add_argument('--snapshot', type=Path, help='Also compare every real mHC weight set against upstream')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--native-bridge', action='store_true', help='Use serving begin ABI from --aot library')
    p.add_argument('--aot', type=Path, help='Optional native lagged-pre probe shared library')
    p.add_argument('--rows', type=int, nargs='+', default=[1,8,16,80,512])
    args = p.parse_args()
    sys.path.insert(0, str(args.b12x_root.resolve()))
    import torch
    from b12x.norm import mhc
    from b12x.preparation import PreparationSession
    from tests.norm._mhc import prepare, bind
    from tests.norm.test_mhc_lagged import _lagged_reference
    torch.cuda.set_device(args.device)
    torch.manual_seed(731)
    lib = C.CDLL(str(args.native.resolve()))
    lib.ds41rt_v41_hc_mixes_workspace.argtypes = [C.c_void_p]*8 + [C.c_uint64, C.c_int32, C.c_void_p]
    lib.ds41rt_v41_hc_pre.argtypes = [C.c_void_p]*3 + [C.c_int32, C.c_void_p]
    lib.ds41rt_cuda_ds4_rmsnorm_bf16_rne_async.argtypes = [C.c_void_p]*3 + [C.c_int, C.c_int, C.c_float, C.c_void_p]
    assert lib.ds41rt_v41_hc_project_initialize() == 0
    aot = C.CDLL(str(args.aot.resolve())) if args.aot else None
    if aot:
        aot_launch = aot.ds41rt_v41_hc_begin if args.native_bridge else aot.launch
        aot_launch.argtypes = [C.c_void_p]*11 + ([C.c_uint64] if args.native_bridge else []) + [C.c_int32,C.c_void_p]
        initialize = aot.ds41rt_v41_hc_project_initialize if args.native_bridge else aot.initialize
        assert initialize() == 0
        assert initialize() == 0
    report = {'scope': 'Native begin versus upstream prepared lagged pre; warm component diagnostic, not serving',
              'command': sys.argv, 'native_sha256': hashlib.sha256(args.native.read_bytes()).hexdigest(),
              'revision': subprocess.check_output(['git','-C',str(args.b12x_root),'rev-parse','HEAD'], text=True).strip(),
              'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,power.limit,clocks.mem','--format=csv'],text=True), 'cases': []}
    if args.aot:
        report['aot_sha256'] = hashlib.sha256(args.aot.read_bytes()).hexdigest()
    weight_map = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())['weight_map'] if args.snapshot else {}
    real_names = sorted(name for name in weight_map if name.endswith(('hc_attn_fn','hc_ffn_fn')))
    if args.snapshot:
        assert aot and real_names
        from safetensors import safe_open
    for rows in args.rows:
        residual = torch.randn((rows,4,5120), device='cuda', dtype=torch.bfloat16)
        fn = torch.randn((24,20480), device='cuda') * 0.01
        scale = torch.tensor([0.1]*3, device='cuda')
        bias = torch.randn(24, device='cuda')
        incoming = torch.rand((rows,4), device='cuda')
        weight = torch.linspace(.5,1.5,5120,device='cuda').bfloat16()
        predicted = torch.empty_like(incoming)
        native_pre = torch.empty_like(incoming)
        native_post = torch.empty_like(incoming)
        native_comb = torch.empty((rows,4,4), device='cuda')
        collapsed = torch.empty((rows,5120),device='cuda',dtype=torch.bfloat16)
        normalized = torch.empty_like(collapsed)
        if aot:
            aot_outputs = [torch.empty_like(t) for t in (native_post,native_comb,normalized,native_pre)]
            aot_scratch = torch.empty((rows*2000+16,),device='cuda')
            def native_aot():
                post,comb,y,pre = aot_outputs
                assert aot_launch(*[t.data_ptr() for t in (residual,fn,scale,bias,incoming,weight,pre,post,comb,y,aot_scratch)],*([aot_scratch.numel()*4] if args.native_bridge else []),rows,torch.cuda.current_stream().cuda_stream) == 0
            if args.native_bridge:
                post,comb,y,pre = aot_outputs
                pointers = [t.data_ptr() for t in (residual,fn,scale,bias,incoming,weight,pre,post,comb,y,aot_scratch)]
                for count, size in ((0,rows*8000),(81,rows*8000),(rows,rows*8000-1)):
                    assert aot_launch(*pointers,size,count,torch.cuda.current_stream().cuda_stream) == 1
                for index, replacement in ((0,0),(6,pointers[4]),(10,pointers[9]),(9,pointers[0]),(10,pointers[10]+4)):
                    invalid = pointers.copy(); invalid[index] = replacement
                    assert aot_launch(*invalid,rows*8000,rows,torch.cuda.current_stream().cuda_stream) == 1
        opts = dict(pre_mix=incoming,norm_weight=weight,norm_eps=1e-20,rms_eps=1e-20,hc_eps=1e-6,sinkhorn_iters=20)
        with PreparationSession(device=residual.device,autotune=False,compile_workers=2) as session:
            plan = prepare(session,'pre',(residual,fn,scale,bias),opts)
            binding = bind(plan,pre_out=predicted)
            session.freeze()
            def upstream():
                return mhc.run_pre(residual,fn,scale,bias,binding=binding,**opts)
            def native():
                stream = torch.cuda.current_stream().cuda_stream
                assert lib.ds41rt_v41_hc_mixes_workspace(
                    *[t.data_ptr() for t in (residual,fn,scale,bias,native_pre,native_post,native_comb,collapsed)],
                    collapsed.numel()*2, rows, stream) == 0
                assert lib.ds41rt_v41_hc_pre(residual.data_ptr(),incoming.data_ptr(),collapsed.data_ptr(),rows,stream) == 0
                assert lib.ds41rt_cuda_ds4_rmsnorm_bf16_rne_async(collapsed.data_ptr(),weight.data_ptr(),normalized.data_ptr(),rows,5120,1e-20,stream) == 0
            actual = upstream()
            torch.cuda.synchronize()
            graphs = {}
            calls = [('native',native),('upstream',upstream)]
            if aot:
                calls.append(('aot',native_aot))
            for name, call in calls:
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(20):
                        call()
                graphs[name] = graph
            errors = []
            for amplitude in (1.0,0.0,0.01):
                residual.normal_().mul_(amplitude)
                incoming.uniform_()
                saved_residual = residual.clone()
                if aot:
                    aot_scratch.fill_(12345)
                    for out in aot_outputs:
                        out.fill_(float('nan'))
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(residual,saved_residual), 'Input residual was modified'
                if aot:
                    assert (aot_scratch[rows*2000:] == 12345).all()
                    for out, ref in zip(aot_outputs,(*actual[1:],predicted),strict=True):
                        torch.testing.assert_close(out,ref,rtol=0,atol=0)
                expected = _lagged_reference(residual,fn,scale,bias,incoming,weight)
                for name, outputs in [('native',(native_post,native_comb,normalized,native_pre)),
                                      ('upstream',(*actual[1:],predicted))]:
                    failures = []
                    for component, (output, reference) in enumerate(zip(outputs,expected,strict=True)):
                        assert torch.isfinite(output).all()
                        try:
                            torch.testing.assert_close(output,reference,rtol=2e-5,atol=.008 if output.dtype == torch.bfloat16 else 4e-5)
                        except AssertionError as error:
                            failure = {'component':component,'error':str(error)}
                            if args.rounding_diagnostics and component == 2:
                                products = incoming.unsqueeze(-1) * residual.float()
                                reference_collapse = products.sum(1).bfloat16()
                                sequential = products[:,0]
                                for index in range(1,4):
                                    sequential = sequential + products[:,index]
                                sequential = sequential.bfloat16()
                                exact_collapse = (incoming.double().unsqueeze(-1)*residual.double()).sum(1).bfloat16()
                                exact = exact_collapse.double()
                                gold = (exact * torch.rsqrt(exact.square().mean(-1,keepdim=True)+1e-20) * weight.double()).bfloat16()
                                def stats(value, target):
                                    delta = value.float()-target.float()
                                    return {'differing_elements':int(torch.count_nonzero(delta)),
                                            'elements':delta.numel(),
                                            'max_abs':float(delta.abs().max()),
                                            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(target.float()).clamp_min(1e-30))}
                                failure['rounding'] = {
                                    'native_collapse_vs_torch_sum':stats(collapsed,reference_collapse),
                                    'native_collapse_vs_sequential':stats(collapsed,sequential),
                                    'native_collapse_vs_fp64':stats(collapsed,exact_collapse),
                                    'torch_sum_collapse_vs_fp64':stats(reference_collapse,exact_collapse),
                                    'output_vs_fp64_pipeline':stats(output,gold),
                                    'reference_vs_fp64_pipeline':stats(reference,gold),
                                }
                                mask = (output.float()-reference.float()).abs() > (.008+2e-5*reference.float().abs())
                                examples = []
                                for row,col in mask.nonzero()[:8].tolist():
                                    examples.append({'row':row,'column':col,'output':float(output[row,col]),
                                        'reference':float(reference[row,col]),'fp64_pipeline':float(gold[row,col]),
                                        'native_collapse':float(collapsed[row,col]),
                                        'torch_sum_collapse':float(reference_collapse[row,col]),
                                        'fp64_collapse':float(exact_collapse[row,col])})
                                failure['examples'] = examples
                            failures.append(failure)
                    errors.append({'amplitude':amplitude,'path':name,'oracle_pass':not failures,'failures':failures})
            real_checks = []
            if args.snapshot:
                saved_weights = [t.clone() for t in (fn,scale,bias,weight)]
                residual.normal_()
                for name in real_names:
                    prefix, kind = name.rsplit('hc_',1)
                    kind = kind.removesuffix('_fn')
                    keys = (name,name.removesuffix('_fn')+'_scale',name.removesuffix('_fn')+'_base',prefix+kind+'_norm.weight')
                    for target, key in zip((fn,scale,bias,weight),keys,strict=True):
                        with safe_open(args.snapshot / weight_map[key],framework='pt',device='cpu') as file:
                            target.copy_(file.get_tensor(key).to(dtype=target.dtype,device=target.device))
                    graphs['upstream'].replay()
                    graphs['aot'].replay()
                    torch.cuda.synchronize()
                    for output, reference in zip(aot_outputs,(*actual[1:],predicted),strict=True):
                        assert torch.isfinite(output).all()
                        torch.testing.assert_close(output,reference,rtol=0,atol=0)
                    real_checks.append(name)
                for target, original in zip((fn,scale,bias,weight),saved_weights,strict=True):
                    target.copy_(original)
            if any(not check['oracle_pass'] for check in errors):
                case = {'rows':rows,'aot_exact_replay_and_scratch_guard':bool(aot),'real_weight_exact_checks':real_checks,'checks':errors,'timing_skipped':'Numerical gate failed'}
                report['cases'].append(case)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(case),flush=True)
                continue
            # Restore representative nonzero data before timing.
            residual.normal_()
            samples = {name:[] for name in graphs}
            allocated = torch.cuda.memory_allocated()
            for i in range(6):
                order = list(graphs)
                for name in (order if i%2 else list(reversed(order))):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):
                        graphs[name].replay()
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end))
            assert torch.cuda.memory_allocated() == allocated
            case = {'rows':rows,'aot_exact_replay_and_scratch_guard':bool(aot),'real_weight_exact_checks':real_checks,'checks':errors,'warm_us':samples,
                    'median_us':{k:statistics.median(v) for k,v in samples.items()}}
            report['cases'].append(case)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case),flush=True)

    if any('timing_skipped' in case for case in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
