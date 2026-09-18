#!/usr/bin/env python3
"""Prove the launch cluster cap is a ceiling, not a requirement.

The AOT artifact is identical whatever cap the export host chose, so a card
with fewer SMs (a 5090 against a 188-SM RTX PRO 6000) can run the same cubin
if the launch clamps the cooperative grid. That only holds if a reduced cap
leaves the result unchanged; this runs the same inputs at two caps and
compares outputs.
"""
import os
import sys

import torch

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

from b12x.moe import fused_moe
from b12x.moe.fused_moe._tuning import MoeDecodeConfig
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared


def run(cap: int, x, ids, weights, pack):
    override = MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=cap,
        dynamic_tile_m=16, dynamic_route_mode="grouped", w4a16_route_mode=None,
        nvfp4_share_input=False, nvfp4_materialize_intermediate=False,
    )
    experts = pack["experts"]
    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=int(x.shape[0]), top_k=int(ids.shape[1])),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
        invocation=FrozenMapping({"fast_math": True}),
        override=override,
    )

    def prepare_call(state):
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                        for s in state.scratch.scratch_specs())
        output = torch.empty_like(x)
        binding = state.bind(scratch=scratch, a=x, experts=experts, topk_weights=weights,
                             topk_ids=ids, output=output, input_scales_static=True, fast_math=True)
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    request = declaration.request(name=f"cap-{cap}", prepare_call=prepare_call)
    session = PreparationSession(device=x.device, autotune=False, compile_workers=2)
    session.prepare((request,))
    plan = require_prepared(request.plan, "moe.decode")
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device)
                    for s in plan.scratch.scratch_specs())
    output = torch.empty_like(x)
    binding = fused_moe.bind(plan, scratch=scratch, a=x, experts=experts, topk_weights=weights,
                             topk_ids=ids, output=output, input_scales_static=True, fast_math=True)
    fused_moe.run(plan, binding)
    torch.cuda.synchronize()
    return output.clone()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device SMs: {sms}")
    E, K, n, topk, rows = 8, 5120, 1152, 6, 4
    source = fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
        w13_layout=fused_moe.W13Layout("w13"),
    )
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A4, nonlinearity="silu",
                                            io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=n),
    )
    # Random payloads in the kernel's own layout: the comparison is between two
    # launches of the same weights, so the values need only be representative.
    def payload(*shape):
        return torch.randint(-128, 127, shape, dtype=torch.int8, device=device).view(torch.uint8)
    # Rank-3, one leading expert axis, in the kernel-native layout.
    w13 = payload(E, 2 * n, K // 2)
    down = payload(E, K, n // 2)
    s13 = payload(E, 2 * n, K // 16)
    s2 = payload(E, K, n // 16)
    experts = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=w13, w2=down,
            w13_block_scales=s13.view(torch.float8_e4m3fn),
            w2_block_scales=s2.view(torch.float8_e4m3fn),
            w13_global_scales=torch.ones(E, dtype=torch.float32, device=device),
            w2_global_scales=torch.ones(E, dtype=torch.float32, device=device),
            input_scale=torch.ones(E, dtype=torch.float32, device=device),
            intermediate_scale=torch.ones(E, dtype=torch.float32, device=device),
        ),
    )
    pack = {"experts": experts}
    x = torch.randn(rows, K, dtype=torch.bfloat16, device=device) * 0.1
    ids = (torch.arange(rows * topk, dtype=torch.int32, device=device).reshape(rows, topk) % E)
    wts = torch.full((rows, topk), 1.0 / topk, dtype=torch.float32, device=device)

    reference = run(sms, x, ids, wts, pack)
    for cap in (max(1, sms - 18), max(1, sms // 2), 1):
        other = run(cap, x, ids, wts, pack)
        same = bool(torch.equal(reference, other))
        delta = (reference.float() - other.float()).abs().max().item()
        print(f"cap {cap:4d}: identical={same} max_abs_diff={delta:.3e}")
        if not same:
            print("  -> the cap is not a pure ceiling at this setting")
            return 1
    print("PASS: every reduced cap reproduced the reference output exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
