#!/usr/bin/env python3
"""Native NVFP4/W4A8 launch-contract regression (GPU; coordinate idle GPUs first).

Pass LIB MANIFEST PREFIX [--rows 1] [--device 0]. Run each family separately.
No compilation/export occurs. Requires exported positive resident-cluster counts.
This checks zero-data output and ABI validation, not nonzero numerical accuracy.
Example:
  python native/tests/v41_nvfp4_launch_selftest.py libds41rt_native.so \
    v41_nvfp4_experts.json ds41rt_v41_nvfp4_tp2_expert
"""
import argparse
import ctypes as C
import json
from pathlib import Path


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in (
        "abi_version", "role", "experts", "hidden_size", "logical_intermediate",
        "kernel_intermediate", "topk", "capacity_rows")]
    _fields_ += [("scratch_bytes", C.c_uint64)]
    _fields_ += [(n, C.c_int32) for n in (
        "max_rows", "rows_padded", "max_tasks", "max_phys_tiles", "max_active_clusters")]
    _fields_ += [("input_dtype", C.c_uint32)]


class Launch(C.Structure):
    _fields_ = [("tensors", C.c_void_p * 44)]
    _fields_ += [(n, C.c_int32) for n in (
        "num_tokens", "max_rows", "scatter_rows", "rows_padded", "max_tasks",
        "max_phys_tiles", "max_active_clusters")]
    _fields_ += [("stream", C.c_void_p)]


assert C.sizeof(Info) == 64
assert C.sizeof(Launch) == 392 and Launch.stream.offset == 384


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("manifest")
    parser.add_argument("prefix")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    opt = parser.parse_args()
    import torch

    torch.cuda.set_device(opt.device)
    manifest = json.loads(Path(opt.manifest).read_text())
    matches = [v for v in manifest["variants"]
               if v.get("requested_rows", v.get("capacity_rows")) == opt.rows]
    check(len(matches) == 1, f"expected one variant for requested rows {opt.rows}")
    variant = matches[0]
    capacity = variant["capacity_rows"]
    nvfp4 = manifest.get("quant_mode") == "nvfp4" or "nvfp4" in opt.prefix
    lib = C.CDLL(opt.library)

    def function(suffix, arguments):
        fn = getattr(lib, opt.prefix + suffix)
        fn.argtypes, fn.restype = arguments, C.c_int32
        return fn

    query = function("_info", [C.c_int32, C.POINTER(Info)])
    initialize = function("_initialize", [C.c_int32, C.POINTER(C.c_void_p)])
    bind = function("_bind_scratch", [C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p)])
    reset = function("_initialize_scratch_async", [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p])
    launch = function("_launch", [C.c_void_p, C.POINTER(Launch)])
    info, handle = Info(), C.c_void_p()
    check(query(capacity, C.byref(info)) == 0, "info failed")
    check(info.capacity_rows == capacity and 0 < opt.rows <= capacity, "invalid capacity")
    check(info.scratch_bytes == variant["core_scratch_nbytes"], "manifest/library scratch mismatch")
    check(info.input_dtype == (1 if nvfp4 else 7), "unexpected family input dtype")
    check(initialize(capacity, C.byref(handle)) == 0, "initialize failed")
    stream = torch.cuda.current_stream().cuda_stream
    arena = torch.empty(info.scratch_bytes, dtype=torch.uint8, device="cuda")
    slots = (C.c_void_p * 44)()
    check(bind(handle, arena.data_ptr(), arena.numel(), slots) == 0, "bind failed")
    for slot, offset in enumerate(variant["scratch_pointer_offsets"]):
        expected = None if offset is None else arena.data_ptr() + offset
        check(slots[slot] == expected, f"manifest/library scratch slot {slot} mismatch")
    # bind_scratch must preserve caller-owned slots marked unbound by the manifest.
    # Use a real live allocation (not a fabricated pointer), but never launch it.
    sentinel = torch.empty(16, dtype=torch.uint8, device="cuda")
    preserved = [i for i, offset in enumerate(variant["scratch_pointer_offsets"]) if offset is None]
    for slot in preserved:
        slots[slot] = sentinel.data_ptr()
    check(bind(handle, arena.data_ptr(), arena.numel(), slots) == 0, "repeat bind failed")
    for slot in preserved:
        check(slots[slot] == sentinel.data_ptr(), f"bind overwrote external slot {slot}")
    check(bind(handle, arena.data_ptr(), arena.numel() - 1, slots) == 1,
          "undersized scratch must be rejected")

    n, h = info.kernel_intermediate, info.hidden_size
    if nvfp4:
        ceil_to = lambda x, m: (x + m - 1) // m * m
        sizes = [n * h, ceil_to(2 * n, 128) * ceil_to(h // 16, 4),
                 h * (n // 2), ceil_to(h, 128) * ceil_to(n // 16, 4)]
        scale_byte = 0x38  # E4M3 1.0
    else:
        sizes_fn = lib.ds41rt_v41_expert_packed_sizes
        sizes_fn.argtypes = [C.c_uint32, C.POINTER(C.c_uint64)]
        sizes_fn.restype = C.c_int32
        packed = (C.c_uint64 * 4)()
        check(sizes_fn(info.logical_intermediate, packed) == 0, "packed sizes failed")
        sizes, scale_byte = list(packed), 127  # UE8M0 1.0
    # Own all expert extents even though only topk distinct experts are routed.
    weights = [torch.full((info.experts * size,), scale_byte if i % 2 else 0,
                         dtype=torch.uint8, device="cuda") for i, size in enumerate(sizes)]
    scales = [torch.ones(info.experts, dtype=torch.float32, device="cuda") for _ in range(4)]
    for slot, tensor in zip((22, 23, 24, 25), weights):
        slots[slot] = tensor.data_ptr()
    if nvfp4:
        for slot, tensor in zip((37, 38, 39, 40), scales):
            slots[slot] = tensor.data_ptr()
    else:
        for slot, source in ((26, 23), (27, 25), (28, 34), (29, 34),
                             (30, 22), (31, 23), (32, 24), (33, 25),
                             (38, 37), (39, 37)):
            slots[slot] = slots[source]
    hidden = torch.zeros((opt.rows, h if nvfp4 else h + h // 32),
                         dtype=torch.bfloat16 if nvfp4 else torch.uint8, device="cuda")
    if not nvfp4:
        hidden[:, h:] = 127  # zero E4M3 payload, unit K32 scale
    ids = torch.arange(info.topk, dtype=torch.int32, device="cuda").repeat(opt.rows)
    routing = torch.full((opt.rows * info.topk,), 1.0 / info.topk,
                         dtype=torch.float32, device="cuda")
    for slot, tensor in enumerate((hidden, ids, routing)):
        slots[slot] = tensor.data_ptr()
    check(all(slots), "unpopulated pointer slot")
    tensors = variant.get("scratch", variant.get("scratch_tensors"))
    output_spec = next(t for t in tensors if t.get("name") == "route_output" or t.get("slot") == 41)
    output = arena[output_spec["offset"]:output_spec["offset"] + output_spec["nbytes"]]
    if nvfp4:
        output_kind = function("_output_kind", [C.c_int32, C.POINTER(C.c_uint32)])
        kind = C.c_uint32(99)
        check(output_kind(capacity, C.byref(kind)) == 0 and kind.value == 2,
              "NVFP4 output kind must be 2 (BF16 routes)")
        check(output_spec["dtype"] == "bfloat16", "NVFP4 output must be BF16 routes")
        check(output_spec["shape"] == [capacity * info.topk, h], "unexpected BF16 route shape")
        check(output_spec["nbytes"] == capacity * info.topk * h * 2, "route output extent mismatch")
    # Check live rows only; padded-capacity routes are not necessarily written.
    itemsize = 2 if nvfp4 else 4
    output_rows = opt.rows if variant.get("output_kind") == "fp32_tokens" else opt.rows * info.topk
    live_output = output[:output_rows * h * itemsize]
    defaults = dict(num_tokens=opt.rows, max_rows=info.max_rows,
                    scatter_rows=opt.rows * info.topk, rows_padded=info.rows_padded,
                    max_tasks=info.max_tasks, max_phys_tiles=info.max_phys_tiles,
                    max_active_clusters=info.max_active_clusters, stream=stream)

    def attempt(label, expected, null_slot=None, **overrides):
        check(reset(handle, arena.data_ptr(), arena.numel(), stream) == 0, "scratch reset failed")
        # A launcher that returns success without writing output must not pass.
        live_output.fill_(0x5A)
        args = Launch(tensors=slots, **(defaults | overrides))
        if null_slot is not None:
            args.tensors[null_slot] = None
        code = launch(handle, C.byref(args))
        torch.cuda.synchronize()
        print(f"{label}: status={code} expected={expected}", flush=True)
        check(code == expected, f"{label}: native status {code}, expected {expected}")
        if code == 0:
            check(torch.count_nonzero(live_output).item() == 0, "zero weights produced nonzero output")

    # Policy None/-1 must be resolved to a positive resident grid before export.
    # The compiled ABI uses this scalar as grid.z; it is NOT a policy sentinel.
    check(info.max_active_clusters > 0, "exported unresolved cluster-policy sentinel")
    attempt("as-exported", 0)
    attempt("repeat-as-exported", 0)
    attempt("reject-policy-sentinel", 1, max_active_clusters=-1)
    for clusters in (-2, 0, 2 * manifest["physical_sms"] + 1):
        attempt(f"reject-clusters-{clusters}", 1, max_active_clusters=clusters)
    attempt("positive-clusters", 0, max_active_clusters=1)
    # Capture the same scratch reset/launch stream sequence and replay repeatedly.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_args = Launch(tensors=slots, **(defaults | {"stream": capture_stream.cuda_stream}))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        check(reset(handle, arena.data_ptr(), arena.numel(), capture_stream.cuda_stream) == 0,
              "captured reset failed")
        live_output.fill_(0x5A)
        check(launch(handle, C.byref(capture_args)) == 0, "captured launch failed")
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        check(torch.count_nonzero(live_output).item() == 0, "graph zero output mismatch")
    print("graph replay: PASS (3 replays)", flush=True)
    # Every pointer is required by the common ABI, even slots NVFP4 does not read.
    for slot in range(44):
        attempt(f"null-slot-{slot}", 1, null_slot=slot)
    # Contractually rejected values: never intentionally dispatch invalid extents.
    attempt("reject-zero-tokens", 1, num_tokens=0)
    attempt("reject-overcapacity", 1, num_tokens=capacity + 1,
            scatter_rows=(capacity + 1) * info.topk)
    attempt("reject-scatter-mismatch", 1, scatter_rows=opt.rows * info.topk + 1)
    for field in ("max_rows", "rows_padded", "max_tasks", "max_phys_tiles"):
        attempt(f"reject-{field}-mismatch", 1, **{field: getattr(info, field) + 1})
    print(f"PASS {opt.prefix} requested_rows={opt.rows} capacity={capacity} device={opt.device}")


if __name__ == "__main__":
    main()
