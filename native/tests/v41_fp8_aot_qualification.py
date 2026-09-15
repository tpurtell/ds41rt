"""Exercise native AOT pointer ABI and graph replay against a quantized oracle.

Run with PYTHONPATH pointing to the SparkInfer source matching the supplied
manifest. This checks correctness, not throughput.
"""
import argparse
import ctypes as C
import hashlib
import json
import subprocess
from pathlib import Path

import b12x
import torch

from tests.gemm.test_gemm_block_fp8_linear import (
    _assert_v41_accumulation_matches_reference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source_root = Path(b12x.__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != manifest["sparkinfer_revision"]:
        raise RuntimeError("oracle source revision differs from the export manifest")
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((args.manifest.parent / name).read_bytes()).hexdigest() == digest, name
    lib = C.CDLL(str(args.library.resolve()))
    ptr, i32, u64 = C.c_void_p, C.c_int32, C.c_uint64

    class Info(C.Structure):
        _fields_ = [(n, C.c_uint32) for n in ("abi", "capacity", "k", "n")] + [
            (n, u64) for n in ("scratch", "values", "row_scales", "mma_scales", "weight_scales")
        ]

    def bind(name, signature):
        fn = getattr(lib, "ds41rt_v41_fp8_" + name)
        fn.argtypes, fn.restype = signature, i32

        def checked(*values):
            status = fn(*values)
            if status:
                raise RuntimeError(f"{name}: CUDA status {status}")
        return checked

    info_fn = bind("matrix_info", [i32, i32, i32, C.POINTER(Info)])
    initialize = bind("matrix_initialize", [i32, i32, i32, C.POINTER(ptr)])
    pack = bind("matrix_pack_scales", [ptr, ptr, i32, i32, ptr])
    initialize_scratch = bind("initialize_scratch", [ptr, ptr, u64, ptr, ptr])
    launch = bind("launch", [ptr, ptr, ptr, ptr, ptr, u64, ptr, ptr, i32, ptr])
    results = []
    for variant in manifest["variants"]:
        if not any(variant["label"].startswith(f"v41_{name}_fp8_m")
                   for name in ("q_a", "kv", "ffn_tp2_up", "ffn_tp2_down")):
            continue
        capacity, k, n = variant["capacity"], variant["input_dim"], variant["output_dim"]
        torch.manual_seed(41000 + k + n + capacity)
        info, handle = Info(), ptr()
        info_fn(capacity, k, n, C.byref(info))
        initialize(capacity, k, n, C.byref(handle))
        source = torch.randn(capacity, k, device="cuda").mul_(.2).bfloat16()
        weight = torch.randn(n, k, device="cuda").mul_(.1).to(torch.float8_e4m3fn)
        scales = torch.randint(123, 129, (n // 32, k // 32), device="cuda", dtype=torch.uint8)
        packed = torch.empty(info.weight_scales, device="cuda", dtype=torch.uint8)
        scratch = torch.empty(info.scratch, device="cuda", dtype=torch.uint8)
        alpha = torch.empty(1, device="cuda", dtype=torch.float32)
        output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.current_stream().cuda_stream
        pack(scales.data_ptr(), packed.data_ptr(), k, n, stream)
        initialize_scratch(handle, scratch.data_ptr(), info.scratch, alpha.data_ptr(), stream)
        for rows in sorted({1, min(7, capacity), capacity}):
            def run():
                launch(handle, source.data_ptr(), weight.data_ptr(), packed.data_ptr(),
                       scratch.data_ptr(), info.scratch, alpha.data_ptr(), output.data_ptr(), rows, stream)
            run()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    # CUDA graph capture switches the current stream.
                    stream = torch.cuda.current_stream().cuda_stream
                    run()
                stream = torch.cuda.current_stream().cuda_stream
                source.mul_(-.5)
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                _assert_v41_accumulation_matches_reference(
                    source[:rows], weight, scales.view(torch.float8_e8m0fnu), output[:rows],
                )
            finally:
                graph.reset()
        results.append(dict(label=variant["label"], live_rows=sorted({1, min(7, capacity), capacity})))
        print("passed", variant["label"], flush=True)
    if not results:
        raise RuntimeError("manifest contains no supported qualification shapes")
    args.output.write_text(json.dumps(dict(
        revision=manifest["sparkinfer_revision"], cases=results,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        gpu=str(torch.cuda.get_device_properties(0).uuid),
        torch_version=torch.__version__,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
