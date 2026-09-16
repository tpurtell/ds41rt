#!/usr/bin/env python3
"""Run official shared-expert TP2 projections on both GPUs against a full-width oracle."""
import argparse
import ctypes as C
import json
import hashlib
import statistics
from pathlib import Path
import torch
from safetensors import safe_open


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("abi", "capacity", "k", "n")] + [
        (n, C.c_uint64) for n in ("scratch", "values", "row_scales", "mma_scales", "weight_scales")]


def quantize(x):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
    return ((blocks / scale[..., None]).to(torch.float8_e4m3fn).float() * scale[..., None]).reshape(x.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--rows", default="1,16", help="Comma-separated exported capacities")
    parser.add_argument("--timing-output", type=Path, help="Record warm full-chain GPU timings after correctness gates")
    args = parser.parse_args()
    capacities = [int(value) for value in args.rows.split(",")]
    assert capacities and all(value in (1, 16, 80, 256, 1024, 4096) for value in capacities)
    assert torch.cuda.device_count() >= 2
    torch.manual_seed(41152)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(args.native_lib)
    P, I = C.c_void_p, C.c_int32
    def bind(name, types):
        fn = getattr(lib, name)
        fn.argtypes = types
        def call(*values):
            code = fn(*values)
            assert code == 0, (name, code)
        return call
    info_fn = bind("ds41rt_v41_fp8_matrix_info", [I, I, I, C.POINTER(Info)])
    initialize = bind("ds41rt_v41_fp8_matrix_initialize", [I, I, I, C.POINTER(P)])
    scratch_init = bind("ds41rt_v41_fp8_initialize_scratch", [P, P, C.c_uint64, P, P])
    pack = bind("ds41rt_v41_fp8_matrix_pack_scales", [P, P, I, I, P])
    linear = bind("ds41rt_v41_fp8_launch", [P, P, P, P, P, C.c_uint64, P, P, I, P])
    activation = bind("ds41rt_v41_shared_tp2_swiglu", [P, P, P, I, P])
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    raw = {}
    for name in ("w1", "w3", "w2"):
        for suffix in ("weight", "scale"):
            key = f"layers.0.ffn.shared_experts.{name}.{suffix}"
            with safe_open(args.snapshot / index[key], framework="pt", device="cpu") as file:
                raw[name, suffix] = file.get_tensor(key)
    results, timings = [], []
    for rows in capacities:
        x = torch.randn(rows, 5120).mul_(0.5).bfloat16()
        rank_results, handles = [], []
        for rank in (0, 1):
            with torch.cuda.device(rank):
                stream = torch.cuda.current_stream().cuda_stream
                source = x.cuda()
                states = {}
                for name in ("w1", "w3", "w2"):
                    axis = 1 if name == "w2" else 0
                    w = raw[name, "weight"].view(torch.uint8).chunk(2, axis)[rank].contiguous().cuda()
                    s = raw[name, "scale"].view(torch.uint8).chunk(2, axis)[rank].contiguous().cuda()
                    n, k = w.shape
                    info, handle = Info(), P()
                    info_fn(rows, k, n, C.byref(info))
                    initialize(rows, k, n, C.byref(handle))
                    handles.append((rank, name, handle.value))
                    scratch = torch.empty(info.scratch, dtype=torch.uint8, device="cuda")
                    alpha = torch.empty(4, dtype=torch.float32, device="cuda")
                    packed = torch.empty(info.weight_scales, dtype=torch.uint8, device="cuda")
                    output = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
                    scratch_init(handle, scratch.data_ptr(), scratch.numel(), alpha.data_ptr(), stream)
                    pack(s.data_ptr(), packed.data_ptr(), k, n, stream)
                    states[name] = (handle, w, packed, scratch, alpha, output)
                intermediate = torch.empty((rows, 1152), dtype=torch.bfloat16, device="cuda")
                def run():
                    for name in ("w1", "w3"):
                        h, w, s, scratch, alpha, out = states[name]
                        linear(h, source.data_ptr(), w.data_ptr(), s.data_ptr(), scratch.data_ptr(), scratch.numel(), alpha.data_ptr(), out.data_ptr(), rows, torch.cuda.current_stream().cuda_stream)
                    activation(states["w1"][-1].data_ptr(), states["w3"][-1].data_ptr(), intermediate.data_ptr(), rows, torch.cuda.current_stream().cuda_stream)
                    h, w, s, scratch, alpha, out = states["w2"]
                    linear(h, intermediate.data_ptr(), w.data_ptr(), s.data_ptr(), scratch.data_ptr(), scratch.numel(), alpha.data_ptr(), out.data_ptr(), rows, torch.cuda.current_stream().cuda_stream)
                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                capture = torch.cuda.Stream(device=rank)
                with torch.cuda.graph(graph, stream=capture): run()
                outputs = []
                for changed in (False, True):
                    if changed: source.mul_(-0.5)
                    graph.replay()
                    outputs.append(states["w2"][-1].float().cpu())
                rank_results.append(outputs)
                if args.timing_output:
                    samples = []
                    allocated = torch.cuda.memory_allocated(rank)
                    for _ in range(6):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 10)
                    assert torch.cuda.memory_allocated(rank) == allocated
                    timings.append(dict(rows=rows, rank=rank, samples_us=samples,
                                        median_us=statistics.median(samples)))
        assert handles[0][2] != handles[3][2] and handles[2][2] != handles[5][2]
        with torch.cuda.device(0):
            full = {name: raw[name, "weight"].cuda().float() * raw[name, "scale"].cuda().float().repeat_interleave(32, 0).repeat_interleave(32, 1) for name in ("w1", "w3", "w2")}
            for changed in (False, True):
                qx = quantize((x * (-0.5 if changed else 1)).cuda())
                gate = (qx @ full["w1"].T).bfloat16().float().clamp_max(10)
                up = (qx @ full["w3"].T).bfloat16().float().clamp(-10, 10)
                mid = (torch.nn.functional.silu(gate) * up).bfloat16()
                expected = (quantize(mid) @ full["w2"].T).bfloat16().float().cpu()
                actual = rank_results[0][changed] + rank_results[1][changed]
                measured, oracle = actual.double(), expected.double()
                relative = ((measured-oracle).norm()/oracle.norm()).item()
                cosine = torch.nn.functional.cosine_similarity(measured.flatten(), oracle.flatten(), dim=0).item()
                assert torch.isfinite(actual).all() and relative < 0.01 and cosine > 0.9999, (rows, changed, relative, cosine)
                results.append(dict(rows=rows, changed=changed, relative_l2=relative, cosine=cosine))
    print(json.dumps(results, indent=2))
    if args.timing_output:
        args.timing_output.write_text(json.dumps(dict(
            scope="Real layer-0 shared experts: per-rank up/gate, clamp-SwiGLU, quantization and down projection. Warm graphs; excludes inter-rank reduction and serving scheduler.",
            native_lib=args.native_lib, native_sha256=hashlib.sha256(Path(args.native_lib).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            snapshot=str(args.snapshot), checks=results, timings=timings), indent=2) + '\n')


if __name__ == "__main__":
    main()
