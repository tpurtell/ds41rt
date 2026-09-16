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
    args = p.parse_args()
    sys.path.insert(0, str(args.b12x_root.resolve()))
    import torch
    from b12x.norm import mhc
    from b12x.preparation import PreparationSession
    from tests.norm._mhc import prepare, bind
    from tests.norm.test_mhc_lagged import _lagged_reference
    torch.manual_seed(731)
    lib = C.CDLL(str(args.native.resolve()))
    lib.ds41rt_v41_hc_mixes_workspace.argtypes = [C.c_void_p]*8 + [C.c_uint64, C.c_int32, C.c_void_p]
    lib.ds41rt_v41_hc_pre.argtypes = [C.c_void_p]*3 + [C.c_int32, C.c_void_p]
    lib.ds41rt_cuda_ds4_rmsnorm_bf16_rne_async.argtypes = [C.c_void_p]*3 + [C.c_int, C.c_int, C.c_float, C.c_void_p]
    assert lib.ds41rt_v41_hc_project_initialize() == 0
    report = {'scope': 'Native begin versus upstream prepared lagged pre; warm component diagnostic, not serving',
              'command': sys.argv, 'native_sha256': hashlib.sha256(args.native.read_bytes()).hexdigest(),
              'revision': subprocess.check_output(['git','-C',str(args.b12x_root),'rev-parse','HEAD'], text=True).strip(),
              'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,power.limit,clocks.mem','--format=csv'],text=True), 'cases': []}
    for rows in (1, 8, 16, 80, 512):
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
            for name, call in [('native',native),('upstream',upstream)]:
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
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                expected = _lagged_reference(residual,fn,scale,bias,incoming,weight)
                for name, outputs in [('native',(native_post,native_comb,normalized,native_pre)),
                                      ('upstream',(*actual[1:],predicted))]:
                    failures = []
                    for component, (output, reference) in enumerate(zip(outputs,expected,strict=True)):
                        assert torch.isfinite(output).all()
                        try:
                            torch.testing.assert_close(output,reference,rtol=2e-5,atol=.008 if output.dtype == torch.bfloat16 else 4e-5)
                        except AssertionError as error:
                            failures.append({'component':component,'error':str(error)})
                    errors.append({'amplitude':amplitude,'path':name,'oracle_pass':not failures,'failures':failures})
            if any(not check['oracle_pass'] for check in errors):
                case = {'rows':rows,'checks':errors,'timing_skipped':'Numerical gate failed'}
                report['cases'].append(case)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(case),flush=True)
                continue
            # Restore representative nonzero data before timing.
            residual.normal_()
            samples = {name:[] for name in graphs}
            allocated = torch.cuda.memory_allocated()
            for i in range(6):
                for name in (('native','upstream') if i%2 else ('upstream','native')):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):
                        graphs[name].replay()
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end))
            assert torch.cuda.memory_allocated() == allocated
            case = {'rows':rows,'checks':errors,'warm_us':samples,
                    'median_us':{k:statistics.median(v) for k,v in samples.items()}}
            report['cases'].append(case)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(case),flush=True)

    if any('timing_skipped' in case for case in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
