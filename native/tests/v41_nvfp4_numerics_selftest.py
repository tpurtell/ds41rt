#!/usr/bin/env python3
"""Compare nonzero native NVFP4 BF16 routes with normal public b12x output.

GPU AND COMPILATION WORK: coordinate with the engine/build owner before running.
Public b12x preparation can compile kernels; this is not a build-free probe.
Pass LIB MANIFEST [--rows 1] [--device 0] [--sparkinfer PATH]. Requires corrected
positive-cluster export and output_kind=2. Test direct and grouped capacities
separately (e.g. --rows 1 and --rows 16), on each target device.

Both arms receive identical packed ModelOpt-NVFP4 tensors. The oracle is the
normal public prepare/bind/run path, NOT return_route_partials (which supports
only the silu_v41/FP32 family). This detects native ABI/bridge/scales/routing and
route reduction mistakes, not bugs shared by b12x's underlying CUDA kernel.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys

from v41_nvfp4_launch_selftest import Info, Launch, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("manifest")
    parser.add_argument("--prefix", default="ds41rt_v41_nvfp4_tp2_expert")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sparkinfer", type=Path, default=None)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    opt = parser.parse_args()
    if opt.sparkinfer is None:
        candidates = [p / "third_party/sparkinfer" for p in Path(__file__).resolve().parents]
        opt.sparkinfer = next((p for p in candidates if p.is_dir()), None)
        if opt.sparkinfer is None:
            parser.error("pass --sparkinfer PATH when running outside the source checkout")
    sys.path.insert(0, str(opt.sparkinfer.resolve()))
    import torch
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    torch.cuda.set_device(opt.device)
    manifest = json.loads(Path(opt.manifest).read_text())
    variant = next(v for v in manifest["variants"]
                   if v.get("requested_rows", v["capacity_rows"]) == opt.rows)
    lib = C.CDLL(opt.library)

    def function(suffix, arguments):
        fn = getattr(lib, opt.prefix + suffix)
        fn.argtypes, fn.restype = arguments, C.c_int32
        return fn

    query = function("_info", [C.c_int32, C.POINTER(Info)])
    init = function("_initialize", [C.c_int32, C.POINTER(C.c_void_p)])
    kind_fn = function("_output_kind", [C.c_int32, C.POINTER(C.c_uint32)])
    bind = function("_bind_scratch", [C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p)])
    reset = function("_initialize_scratch_async", [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p])
    launch = function("_launch", [C.c_void_p, C.POINTER(Launch)])
    info, handle, kind = Info(), C.c_void_p(), C.c_uint32(99)
    capacity = variant["capacity_rows"]
    check(query(capacity, C.byref(info)) == 0, "native info failed")
    check(info.input_dtype == 1 and info.max_active_clusters > 0, "invalid native NVFP4 info")
    check(kind_fn(capacity, C.byref(kind)) == 0 and kind.value == 2, "expected BF16 routes")
    check(info.scratch_bytes == variant["core_scratch_nbytes"], "manifest/library mismatch")
    check(init(capacity, C.byref(handle)) == 0, "native initialize failed")
    e, h, n, k = info.experts, info.hidden_size, info.kernel_intermediate, info.topk
    check(n % 64 == 0 and h % 128 == 0, "test requires V4.1 scale-plane alignment")
    m = opt.rows
    device = torch.device("cuda", opt.device)
    # Uniform block scales are invariant under the 128x4 scale swizzle. This
    # deliberately avoids depending on a second packer or disposable run files.
    # Payload nibble 2 is E2M1 1.0, nibble 4 is 2.0; full extents are owned.
    w13 = torch.full((e, 2 * n, h // 2), 0x22, dtype=torch.uint8, device=device)
    w2 = torch.full((e, h, n // 2), 0x22, dtype=torch.uint8, device=device)
    sf13 = torch.full((e, 2 * n, h // 16), 1 / 16, device=device).to(torch.float8_e4m3fn)
    sf2 = torch.full((e, h, n // 16), 1 / 16, device=device).to(torch.float8_e4m3fn)
    # Expert-specific gate and down payloads make route/weight slot mistakes visible.
    w13[1::2, n:] = 0x44
    w2[2::3] = 0x44
    ws13 = (torch.arange(e, device=device, dtype=torch.float32).remainder(3) + 1) / 64
    ws2 = (torch.arange(e, device=device, dtype=torch.float32).remainder(2) + 1) / 32
    a1 = torch.full((e,), 16.0, device=device)
    a2 = torch.full((e,), 16.0, device=device)
    alpha, down_alpha = ws13 / a1, ws2 / a2
    x = torch.full((m, h), 1 / 16, dtype=torch.bfloat16, device=device)
    x[:, 1::2] = 1 / 32
    ids = (torch.arange(m, device=device)[:, None] * k
           + torch.arange(k, device=device)[None, :]).remainder(e).to(torch.int32)
    routing = torch.arange(1, k + 1, dtype=torch.float32, device=device).repeat(m, 1)
    routing /= routing.sum(dim=1, keepdim=True)

    arena = torch.empty(info.scratch_bytes, dtype=torch.uint8, device=device)
    slots = (C.c_void_p * 44)()
    check(bind(handle, arena.data_ptr(), arena.numel(), slots) == 0, "native scratch bind failed")
    for slot, tensor in ((0, x), (1, ids), (2, routing), (22, w13), (23, sf13),
                         (24, w2), (25, sf2), (37, a1), (38, alpha),
                         (39, down_alpha), (40, a2)):
        slots[slot] = tensor.data_ptr()
    check(all(slots), "missing native pointer")
    spec = next(t for t in variant["scratch"] if t["name"] == "route_output")
    check(spec["dtype"] == "bfloat16" and spec["shape"] == [capacity * k, h],
          "native route extent mismatch")
    routes = arena[spec["offset"]:spec["offset"] + spec["nbytes"]].view(torch.bfloat16)
    routes = routes.reshape(capacity, k, h)[:m]
    stream = torch.cuda.current_stream().cuda_stream
    check(k == 6 and h == 5120, "native BF16 reducer requires six 5120-wide routes")
    reduce_routes = lib.ds41rt_v41_compact_bf16_routes_async
    reduce_routes.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p]
    reduce_routes.restype = C.c_int32
    actual = torch.empty_like(x)
    args = Launch(tensors=slots, num_tokens=m, max_rows=info.max_rows,
                  scatter_rows=m * k, rows_padded=info.rows_padded,
                  max_tasks=info.max_tasks, max_phys_tiles=info.max_phys_tiles,
                  max_active_clusters=info.max_active_clusters, stream=stream)

    def native():
        check(reset(handle, arena.data_ptr(), arena.numel(), stream) == 0, "native reset failed")
        routes.fill_(float("nan"))
        check(launch(handle, C.byref(args)) == 0, "native launch failed")
        actual.fill_(float("nan"))
        check(reduce_routes(routes.data_ptr(), actual.data_ptr(), m, stream) == 0,
              "native BF16 route reduction failed")

    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
                                     w13_layout=fused_moe.W13Layout("w13")),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A4,
                                           nonlinearity="silu", io_dtype=torch.bfloat16,
                                           swiglu_limit=10),
        geometry=fused_moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n))
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=sf13, w2_block_scales=sf2,
        w13_global_scales=ws13, w2_global_scales=ws2,
        input_scale=a1, intermediate_scale=a2))
    declaration = fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(max_tokens=m, top_k=k),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False,
                                     deterministic_output=True),
        invocation=FrozenMapping({"fast_math": True}),
        override=MoeDecodeConfig(backend="dynamic", route_planner="internal",
                                 max_active_clusters=None,
                                 dynamic_tile_m=manifest["tile_m"],
                                 # Public planner forbids direct+deterministic NVFP4;
                                 # grouped is the supported independent route oracle.
                                 dynamic_route_mode="grouped",
                                 w4a16_route_mode=None, nvfp4_share_input=False,
                                 nvfp4_materialize_intermediate=False))

    def prepare_call(state):
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                        for s in state.scratch.scratch_specs())
        output = torch.empty_like(x)
        binding = state.bind(scratch=scratch, a=x, experts=experts, topk_weights=routing,
                             topk_ids=ids, output=output, input_scales_static=True, fast_math=True)
        return PreparedCall(run=lambda: state.run(binding), owners=(*scratch, output))

    request = declaration.request(name="native-nvfp4-numerics", prepare_call=prepare_call)
    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    prepared = None
    try:
        prepared = session.prepare((request,))
        state = require_prepared(request.plan, "moe.decode")
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                        for s in state.scratch.scratch_specs())
        output = torch.empty_like(x)
        binding = fused_moe.bind(request.plan, scratch=scratch, a=x, experts=experts,
                                topk_weights=routing, topk_ids=ids, output=output,
                                input_scales_static=True, fast_math=True)

        def compare(label):
            expected = fused_moe.run(binding=binding).clone()
            native()
            torch.cuda.synchronize()
            check(torch.isfinite(routes).all().item(), "native routes not fully written/finite")
            # b12x deterministic top-k reduction accumulates route order in FP32.
            total = torch.zeros((m, h), dtype=torch.float32, device=device)
            for route in range(k):
                total.add_(routes[:, route].float())
            check(torch.equal(actual.view(torch.uint16), total.bfloat16().view(torch.uint16)),
                  "native BF16 reducer differs bitwise from ordered FP32 route sum")
            check(torch.count_nonzero(expected).item() > 0, "public oracle unexpectedly zero")
            check(torch.count_nonzero(actual).item() > 0, "native output unexpectedly zero")
            error = (actual.float() - expected.float()).abs().max().item()
            print(f"{label}: max_abs={error:.8g} exact={torch.equal(actual, expected)} "
                  f"native_absmax={actual.float().abs().max().item():.8g}", flush=True)
            torch.testing.assert_close(actual, expected, rtol=opt.rtol, atol=opt.atol)
            return expected

        first = compare("initial")
        # Stable addresses, changed contents: catches stale input/routing or captured pointers.
        x.mul_(2)
        routing.copy_(routing.flip(1))
        ids.add_(k).remainder_(e)
        second = compare("mutated-input-routing")
        check(not torch.equal(first, second), "mutation did not change public oracle")
        # Native graph capture/replay uses the same already-warmed callable/allocations.
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        capture_stream = torch.cuda.Stream()
        args.stream = capture_stream.cuda_stream
        with torch.cuda.graph(graph, stream=capture_stream):
            check(reset(handle, arena.data_ptr(), arena.numel(), capture_stream.cuda_stream) == 0,
                  "capture reset failed")
            routes.fill_(float("nan"))
            check(launch(handle, C.byref(args)) == 0, "capture launch failed")
            actual.fill_(float("nan"))
            check(reduce_routes(routes.data_ptr(), actual.data_ptr(), m,
                                capture_stream.cuda_stream) == 0,
                  "captured native BF16 route reduction failed")
        for replay in range(3):
            graph.replay()
            torch.cuda.synchronize()
            total = torch.zeros((m, h), dtype=torch.float32, device=device)
            for route in range(k):
                total.add_(routes[:, route].float())
            check(torch.equal(actual.view(torch.uint16), total.bfloat16().view(torch.uint16)),
                  "graph native BF16 reducer differs bitwise from ordered FP32 route sum")
            torch.testing.assert_close(actual, second, rtol=opt.rtol, atol=opt.atol)
        print(f"PASS nonzero public parity and 3 graph replays: rows={m} device={opt.device}")
    finally:
        torch.cuda.synchronize()
        if prepared is not None:
            prepared.close()


if __name__ == "__main__":
    main()
