#!/usr/bin/env python3
"""Qualify emitted Spark TP2/TP3 AOT expert kernels through the production C ABI.

Reuses the stable, already-validated grouped-slice fixture
(``tests.moe.test_v41_grouped_slices._check_grouped_slices``) for both the
SAME-WIDTH Python kernel execution and the independent scalar oracle, then
drives the exported native ABI for the requested Spark TP degree (role 5 for
TP2, role 6 for TP3) on the DODO GB10/SM121 lease.

Deliberately does not use the EP benchmark helper: the fixture is the frozen,
known-valid preparation. Weight arenas are pinned to the full 384-expert layout
even though the fixture populates 32 experts, so expert indices 352..383 are
exercised and the remaining arena must be untouched.

Correctness gates, in order: native launch succeeds, output equals the
independent oracle within rel_l2 < 0.01 / cosine > 0.9999, graph replay
allocates nothing, and sentinel-masked routes (id 384, weight 0) produce an
exact zero row.

Run inside the DODO lease container with the pinned SparkInfer tree on
PYTHONPATH (or via the project runner). No daemon/service is touched.
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import statistics
import time
from pathlib import Path

import torch

import _pinned_sparkinfer  # noqa: F401
from _v41_expert_native import Info, L, Native, P, check, library
from tests.moe.test_v41_grouped_slices import _check_grouped_slices

# Role 5 = Spark TP2 (I=1152), role 6 = Spark TP3 (I=768); both unpadded.
SPARK_TP_INTERMEDIATE = {2: 1152, 3: 768}
ARENA_EXPERTS = 384
SENTINEL = 384
REL_TOL, COS_TOL = 0.01, 0.9999
FULL_INTERMEDIATE = 2304
OFFICIAL_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
DEFAULT_SNAPSHOT = (
    "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/"
    f"snapshots/{OFFICIAL_REVISION}"
)


def _quantize_wire(x, capacity, torch_module):
    """BF16 [capacity,5120] -> 5280-byte FP8 E4M3 + UE8M0 K32 rows."""
    wire = torch_module.empty((capacity, 5280), device="cuda", dtype=torch_module.uint8)
    qa = wire[:, :5120].view(torch_module.uint32)
    qs = wire[:, 5120:]
    blocks = x.float().reshape(capacity, 5120 // 32, 32)
    exponent = torch_module.ceil(
        torch_module.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    qa.copy_((blocks / torch_module.exp2(exponent)[..., None])
             .to(torch_module.float8_e4m3fn).view(torch_module.uint32)
             .reshape(capacity, 5120 // 4))
    qs.copy_((exponent + 127).to(torch_module.uint8))
    return wire


def _time_graph(torch_module, graph, intervals, launches_per_replay):
    """Device microseconds per launch and host enqueue microseconds per replay.

    `t0`/`t1` bracket only `graph.replay()` so the host number is the launch
    cost, not the device wait; the event window covers device execution.
    """
    device, host = [], []
    for _ in range(intervals):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        host_start = time.perf_counter()
        graph.replay()
        host.append((time.perf_counter() - host_start) * 1e6)
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end) * 1000.0 / launches_per_replay)
    return device, host


def _run_timing(options, torch_module, lib, arena, intermediate, role, gids, base):
    """M1 device timing crosscheck: actual cap1 vs cap80 AOT, warm + cold.

    Timing method: one CUDA graph is captured OUTSIDE the timed region. The
    amortized graph contains `--repeats` back-to-back launches, so the single
    graph launch amortizes host enqueue cost; a 1-launch graph is timed the same
    way to expose the per-launch device floor. Events bracket the replay on the
    stream (device time); wall time around `replay()` gives host enqueue time.
    Cold uses a >24 MiB device flush before one timed replay.
    """
    capacities = sorted({int(x) for x in options.capacities.split(",")})
    active_counts = [int(x) for x in options.active_counts.split(",")]
    flush_bytes = int(options.cold_bytes)
    results = []
    for capacity in capacities:
        meta = Info()
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(meta)))
        assert meta.role == role, (capacity, meta.role)
        x = torch_module.randn(capacity, 5120, device="cuda").mul_(0.5).bfloat16()
        wire = _quantize_wire(x, capacity, torch_module)
        ids = torch_module.empty(capacity, 6, dtype=torch_module.int32, device="cuda")
        rw = torch_module.empty(capacity, 6, device="cuda")
        native = Native(lib, capacity, arena, wire, ids, rw,
                        spark_tp=options.spark_tp)
        flush = torch_module.empty(flush_bytes // 4, device="cuda",
                                   dtype=torch_module.float32)
        for active in active_counts:
            ids.fill_(-1)
            rw.zero_()
            for slot in range(min(active, 6)):
                ids[0, slot] = base + (slot % len(gids))
                rw[0, slot] = 1.0 / active
            single = torch_module.cuda.CUDAGraph()
            with torch_module.cuda.graph(single):
                native.run(1)
            amortized = torch_module.cuda.CUDAGraph()
            with torch_module.cuda.graph(amortized):
                for _ in range(options.repeats):
                    native.run(1)
            # Preflight outside capture: poison the output, replay once, and
            # require the compiled graph to write finite values in the region it
            # owns (rows=1; the rest of an ABI 2 capacity buffer is untouched).
            native.output.fill_(float("nan"))
            single.replay()
            torch_module.cuda.synchronize()
            written = (native.output[:1] if native.token_accumulation
                       else native.output[:native.info.topk])
            assert bool(torch_module.isfinite(written).all()), (
                "timing graph did not write the output")
            for _ in range(10):
                single.replay()
                amortized.replay()
            torch_module.cuda.synchronize()
            warm_single_device, warm_single_host = _time_graph(
                torch_module, single, options.single_replays, 1)
            warm_amortized_device, warm_amortized_host = _time_graph(
                torch_module, amortized, options.warm_replays, options.repeats)
            cold_device = []
            for _ in range(options.cold_replays):
                flush.fill_(1.0)
                cold_device.extend(
                    _time_graph(torch_module, single, 1, 1)[0])
            record = dict(
                kind="native_m1_timing", role=role, capacity=capacity,
                active=active, rows=1, abi_version=meta.abi_version,
                intermediate=meta.logical_intermediate,
                kernel_intermediate=meta.kernel_intermediate,
                repeats=options.repeats, cold_bytes=flush_bytes,
                warm_single_device_us=statistics.median(warm_single_device),
                warm_amortized_device_us=statistics.median(warm_amortized_device),
                warm_single_host_us=statistics.median(warm_single_host),
                warm_amortized_host_us=statistics.median(warm_amortized_host),
                cold_device_us=statistics.median(cold_device),
                warm_single_samples_us=warm_single_device,
                warm_amortized_samples_us=warm_amortized_device,
                cold_samples_us=cold_device,
                scratch_bytes=native.storage.numel())
            results.append(record)
            print("TIMING " + json.dumps(record, sort_keys=True), flush=True)
        del native, wire, ids, rw, flush
        torch_module.cuda.empty_cache()
    return results


def run_checkpoint(options, torch_module, lib):
    """Real official weights through the native packer, plus ABI3 token output."""
    import safetensors

    from b12x.moe.fused_moe._impl import (
        _e8m0_scale_to_w4a8_sfb_inplace, _logical_weight_to_w4a8_rp_inplace)
    from tests.moe.test_v41_expert_numerics import reference

    snapshot = options.snapshot
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    role = 1 if options.tp4_legacy else 3 + options.spark_tp
    intermediate = (576 if options.tp4_legacy
                    else SPARK_TP_INTERMEDIATE[options.spark_tp])
    lo = options.tp_rank * intermediate
    hi = lo + intermediate
    gids = [int(x) for x in options.expert_ids.split(",")]
    assert gids and len(gids) <= ARENA_EXPERTS
    base = ARENA_EXPERTS - len(gids)
    files = {}

    def tensor(key):
        shard = index[key]
        if shard not in files:
            files[shard] = safetensors.safe_open(
                snapshot / shard, framework="pt", device="cpu")
        return files[shard].get_tensor(key)

    weights, scales, sources = {}, {}, []
    for name in ("w1", "w3", "w2"):
        weight_rows, scale_rows = [], []
        for gid in gids:
            weight = tensor(f"layers.{options.layer}.ffn.experts.{gid}.{name}.weight")
            scale = tensor(f"layers.{options.layer}.ffn.experts.{gid}.{name}.scale")
            w = weight.view(torch_module.uint8)
            s = scale.view(torch_module.uint8)
            if name == "w2":
                w = w[:, lo // 2: lo // 2 + intermediate // 2]
                s = s[:, lo // 32: lo // 32 + intermediate // 32]
            else:
                w = w[lo:hi, :]
                s = s[lo:hi, :]
            weight_rows.append(w.contiguous().cuda())
            scale_rows.append(s.contiguous().cuda())
        weights[name] = torch_module.stack(weight_rows)
        scales[name] = torch_module.stack(scale_rows)

    sizes = (L * 4)()
    check(lib.ds41rt_v41_expert_packed_sizes(intermediate, sizes))
    per = [int(sizes[i]) for i in range(4)]
    arena = [torch_module.zeros(ARENA_EXPERTS * per[i], dtype=torch_module.uint8,
                                device="cuda") for i in range(4)]
    stream = torch_module.cuda.current_stream().cuda_stream
    for k in range(len(gids)):
        slot = base + k
        src = (P * 6)(*[
            weights["w1"][k].data_ptr(), weights["w3"][k].data_ptr(),
            weights["w2"][k].data_ptr(), scales["w1"][k].data_ptr(),
            scales["w3"][k].data_ptr(), scales["w2"][k].data_ptr()])
        dst = (P * 4)(*[arena[i].data_ptr() + slot * per[i] for i in range(4)])
        check(lib.ds41rt_v41_pack_expert_async(src, dst, intermediate, stream))
    torch_module.cuda.synchronize()

    # Cross-check the native packer against the b12x repack the Python pipeline
    # uses. Reported, not asserted: a mismatch is evidence, not a reason to hide
    # the numeric result.
    repack_mismatch = None
    if options.compare_repack:
        rp13 = _logical_weight_to_w4a8_rp_inplace(
            torch_module.cat([weights["w3"], weights["w1"]], 1).clone(),
            size_k=5120, size_n=2 * intermediate, gated_half_rows=intermediate)
        rs13 = _e8m0_scale_to_w4a8_sfb_inplace(
            torch_module.cat([scales["w3"], scales["w1"]], 1).clone(),
            weight_E=len(gids), rows=2 * intermediate, k_dim=5120,
            gated_half_rows=intermediate)
        rw2 = _logical_weight_to_w4a8_rp_inplace(
            weights["w2"].clone(), size_k=intermediate, size_n=5120)
        rs2 = _e8m0_scale_to_w4a8_sfb_inplace(
            scales["w2"].clone(), weight_E=len(gids), rows=5120, k_dim=intermediate)
        repacked = [t.view(torch_module.uint32).flatten()
                    for t in (rp13, rs13, rw2, rs2)]
        repack_mismatch = 0
        for k in range(len(gids)):
            for i in range(4):
                words = per[i] // 4
                native_words = arena[i][(base + k) * per[i]:(base + k + 1) * per[i]] \
                    .view(torch_module.uint32)
                repack_mismatch += int(
                    (native_words != repacked[i][k * words:(k + 1) * words]).sum().item())
        del repacked, rp13, rs13, rw2, rs2

    if options.timing:
        return _run_timing(options, torch_module, lib, arena, intermediate, role,
                           gids, base)

    capacities = sorted({int(x) for x in options.capacities.split(",")})
    live_rows = [int(x) for x in options.live_rows.split(",")]
    results = []
    for capacity in capacities:
        # Confirm the library actually exposes this AOT variant and its ABI.
        meta = Info()
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(meta)))
        assert meta.role == role
        expect_abi = 2 if capacity <= 80 else 3
        assert meta.abi_version == expect_abi, (capacity, meta.abi_version)

        x = torch_module.randn(capacity, 5120, device="cuda").mul_(0.5).bfloat16()
        wire = _quantize_wire(x, capacity, torch_module)
        ids = torch_module.empty(capacity, 6, dtype=torch_module.int32, device="cuda")
        rw = torch_module.empty(capacity, 6, device="cuda")
        native = Native(lib, capacity, arena, wire, ids, rw,
                        spark_tp=options.spark_tp)
        assert native.token_accumulation == (capacity > 80)
        if native.token_accumulation:
            assert native.info.abi_version == 3
        candidates = [m for m in live_rows if m <= capacity]
        if capacity <= 80 and capacity not in candidates:
            candidates.append(capacity)
        graphs = []
        for rows in sorted(set(candidates)):
            # Restore the wire to the current base activation: the changed-
            # activation gate below rewrites it, and each row case must start
            # from `x` again.
            wire.copy_(_quantize_wire(x, capacity, torch_module))
            local_ids = (torch_module.arange(rows * 6, device="cuda",
                                             dtype=torch_module.int32).reshape(rows, 6)
                         % len(gids))
            routing = torch_module.full((rows, 6), 1.0 / 6, device="cuda")
            ids.fill_(-1)
            rw.zero_()
            ids[:rows].copy_(local_ids + base)
            rw[:rows].copy_(routing)
            native.output.fill_(float("nan"))
            graph = torch_module.cuda.CUDAGraph()
            with torch_module.cuda.graph(graph):
                native.run(rows)
            graphs.append(graph)
            allocated = torch_module.cuda.memory_allocated()
            graph.replay()
            torch_module.cuda.synchronize()
            assert torch_module.cuda.memory_allocated() == allocated, "replay allocated"
            topk = native.info.topk
            if native.token_accumulation:
                actual = native.output[:rows]
            else:
                actual = native.output[:rows * topk].view(rows, topk, 5120).sum(1)
            assert bool(torch_module.isfinite(actual).all())
            expected = reference(x[:rows], local_ids, routing,
                                 weights, scales)
            rel = ((actual - expected).norm() / expected.norm()).item()
            cosine = torch_module.nn.functional.cosine_similarity(
                actual.flatten(), expected.flatten(), dim=0).item()
            assert rel < REL_TOL and cosine > COS_TOL, (capacity, rows, rel, cosine)

            # Runtime BF16 compaction on the native FP32 output. ABI 3 uses the
            # token compactor; ABI 2 uses the local route compactor. Compared
            # against the Python BF16 oracle (tolerance, not bit identity).
            compact_bf16_rel = None
            compact_bf16_cos = None
            bf16_expected = expected.bfloat16().float()
            compact = torch_module.empty((rows, 5120), device="cuda",
                                         dtype=torch_module.bfloat16)
            if native.token_accumulation:
                check(lib.ds41rt_v41_compact_tokens_bf16_async(
                    native.output[:rows].data_ptr(), compact.data_ptr(), rows,
                    torch_module.cuda.current_stream().cuda_stream))
                compact_actual = compact.float()
            else:
                routes = native.output[:rows * topk].contiguous()
                check(lib.ds41rt_v41_compact_routes_bf16_async(
                    routes.data_ptr(), compact.data_ptr(), rows,
                    torch_module.cuda.current_stream().cuda_stream))
                compact_actual = compact.float()
            torch_module.cuda.synchronize()
            compact_bf16_rel = (
                (compact_actual - bf16_expected).norm() / bf16_expected.norm()).item()
            compact_bf16_cos = torch_module.nn.functional.cosine_similarity(
                compact_actual.flatten(), bf16_expected.flatten(), dim=0).item()
            assert compact_bf16_rel < REL_TOL and compact_bf16_cos > COS_TOL, (
                "compaction", capacity, rows, compact_bf16_rel, compact_bf16_cos)

            # Changed-mask replay: one sentinel row, then every row inactive.
            ids[rows - 1].fill_(SENTINEL)
            rw[rows - 1].zero_()
            native.output.fill_(float("nan"))
            graph.replay()
            torch_module.cuda.synchronize()
            masked = (native.output[rows - 1] if native.token_accumulation
                      else native.output[(rows - 1) * topk: rows * topk].sum(0))
            assert bool((masked == 0).all()), "sentinel row not exact zero"
            ids[:rows].fill_(SENTINEL)
            rw[:rows].zero_()
            native.output.fill_(float("nan"))
            graph.replay()
            torch_module.cuda.synchronize()
            inactive = (native.output[:rows] if native.token_accumulation
                        else native.output[:rows * topk].view(rows, topk, 5120).sum(1))
            assert bool((inactive == 0).all()), "all-inactive rows not exact zero"

            # Changed-activation replay: rewrite the wire the captured graph
            # reads, restore the canonical mask, replay, and require the result
            # to match the NEW oracle. Then mask, restore the mask, and require
            # the restored run to match the new oracle again.
            gen = torch_module.Generator(device="cuda").manual_seed(
                0x9E3779B9 ^ capacity ^ rows)
            x2 = torch_module.randn(capacity, 5120, generator=gen, device="cuda")
            x2 = x2.mul_(0.5).bfloat16()
            wire.copy_(_quantize_wire(x2, capacity, torch_module))
            ids.fill_(-1)
            rw.zero_()
            ids[:rows].copy_(local_ids + base)
            rw[:rows].copy_(routing)
            native.output.fill_(float("nan"))
            graph.replay()
            torch_module.cuda.synchronize()
            changed = (native.output[:rows] if native.token_accumulation
                       else native.output[:rows * topk].view(rows, topk, 5120).sum(1))
            expected2 = reference(x2[:rows], local_ids, routing, weights, scales)
            changed_rel = ((changed - expected2).norm() / expected2.norm()).item()
            changed_cos = torch_module.nn.functional.cosine_similarity(
                changed.flatten(), expected2.flatten(), dim=0).item()
            assert changed_rel < REL_TOL and changed_cos > COS_TOL, (
                "changed activation", capacity, rows, changed_rel, changed_cos)
            ids[:rows].fill_(SENTINEL)
            rw[:rows].zero_()
            native.output.fill_(float("nan"))
            graph.replay()
            torch_module.cuda.synchronize()
            masked_changed = (native.output[:rows] if native.token_accumulation
                              else native.output[:rows * topk].view(rows, topk, 5120).sum(1))
            assert bool((masked_changed == 0).all()), "changed-activation mask not zero"
            ids.fill_(-1)
            rw.zero_()
            ids[:rows].copy_(local_ids + base)
            rw[:rows].copy_(routing)
            native.output.fill_(float("nan"))
            graph.replay()
            torch_module.cuda.synchronize()
            restored = (native.output[:rows] if native.token_accumulation
                        else native.output[:rows * topk].view(rows, topk, 5120).sum(1))
            restored_rel = ((restored - expected2).norm() / expected2.norm()).item()
            restored_cos = torch_module.nn.functional.cosine_similarity(
                restored.flatten(), expected2.flatten(), dim=0).item()
            assert restored_rel < REL_TOL and restored_cos > COS_TOL, (
                "mask restore", capacity, rows, restored_rel, restored_cos)

            results.append(dict(
                spark_tp=options.spark_tp, role=meta.role, layer=options.layer,
                tp_rank=options.tp_rank,
                experts=[gids[0], gids[-1]], resident=len(gids),
                arena_experts=ARENA_EXPERTS, capacity=capacity,
                abi_version=native.info.abi_version,
                token_output=native.token_accumulation, rows=rows,
                intermediate=native.info.logical_intermediate,
                kernel_intermediate=native.info.kernel_intermediate,
                scratch_bytes=native.storage.numel(), rel_l2=rel, cosine=cosine,
                mask_zero=True, all_inactive_zero=True,
                compact_bf16_rel_l2=compact_bf16_rel,
                compact_bf16_cosine=compact_bf16_cos,
                activation_changed_rel_l2=changed_rel,
                activation_changed_cosine=changed_cos,
                masked_changed_zero=True,
                mask_restored_rel_l2=restored_rel,
                mask_restored_cosine=restored_cos,
                repack_mismatch_bytes=repack_mismatch))
            print("NATIVE " + json.dumps(results[-1], sort_keys=True), flush=True)
        for graph in graphs:
            graph.reset()
        del native, wire, ids, rw
        torch_module.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--spark-tp", type=int, choices=(2, 3))
    parser.add_argument("--tp4-legacy", action="store_true",
                        help="Use role 1 TP4 (I=576, pad 640) for the timing baseline")
    parser.add_argument("--width", type=int, choices=(64, 128, 192), default=192)
    parser.add_argument("--capacities", default="1,16,80",
                        help="Native capacity variants to check")
    parser.add_argument("--rows", default="",
                        help="Optional comma list to restrict fixture row cases")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Official checkpoint weights via the native packer + ABI3")
    parser.add_argument("--timing", action="store_true",
                        help="M1 cap1 vs cap80 device timing crosscheck")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Launches captured in the amortized timing graph")
    parser.add_argument("--warm-replays", type=int, default=30)
    parser.add_argument("--single-replays", type=int, default=30)
    parser.add_argument("--cold-bytes", type=int, default=32 << 20)
    parser.add_argument("--cold-replays", type=int, default=5)
    parser.add_argument("--active-counts", default="6,3,2")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert-ids", default="0,1,2,3,4,5,383")
    parser.add_argument("--live-rows", default="1,16,80,256")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--no-repack-compare", dest="compare_repack",
                        action="store_false")
    options = parser.parse_args()
    options.snapshot = Path(options.snapshot)
    if (options.spark_tp is None) == (not options.tp4_legacy):
        parser.error("select exactly one of --spark-tp 2|3 or --tp4-legacy")
    if options.timing and options.checkpoint:
        parser.error("--timing and --checkpoint are mutually exclusive")
    if not (options.timing or options.checkpoint):
        parser.error("one of --checkpoint or --timing is required")

    lib = library(options.native_lib, spark_tp=options.spark_tp)
    if options.checkpoint or options.timing:
        run_checkpoint(options, torch, lib)
        return

    capacities = {int(x) for x in options.capacities.split(",")}
    if not capacities or not capacities <= {1, 16, 80}:
        parser.error("synthetic capacities must be a nonempty subset of 1,16,80; "
                     "use --checkpoint for 256/1024/4096")
    row_filter = {int(x) for x in options.rows.split(",")} if options.rows else None
    intermediate = SPARK_TP_INTERMEDIATE[options.spark_tp]

    owners = {}
    state = {"arena": None, "base_experts": None}
    checked = set()

    def arena_weights(weights, base_experts):
        """Pad the fixture's resident-expert operands into a full 384-expert arena."""
        if state["arena"] is not None:
            assert state["base_experts"] == base_experts
            return state["arena"]
        padded = []
        for weight in weights:
            per_expert = weight.numel() // base_experts
            arena = torch.zeros(ARENA_EXPERTS * per_expert, dtype=weight.dtype,
                                device=weight.device)
            arena[(ARENA_EXPERTS - base_experts) * per_expert:].copy_(
                weight.reshape(-1))
            padded.append(arena)
        state["arena"], state["base_experts"] = padded, base_experts
        return padded

    # The fixture populates a fixed resident expert count (32 for these widths);
    # derive it from the packed tensor extent, never from the routed ids.
    w13_words_per_expert = intermediate * 1280  # kernel_intermediate == intermediate

    def check(**case):
        rows = case["rows"]
        if row_filter is not None and rows not in row_filter:
            return
        capacity = 1 if rows == 1 else 16 if rows <= 16 else 80
        if capacity not in capacities:
            return
        weights, wire, ids_cpu, routing = case["native_inputs"]
        base_experts, remainder = divmod(int(weights[0].numel()), w13_words_per_expert)
        assert remainder == 0 and base_experts > 0, (
            "packed w13 extent is not a whole number of resident experts",
            weights[0].numel(), w13_words_per_expert)
        assert int(ids_cpu.max().item()) < base_experts, "routed id exceeds resident experts"
        arena = arena_weights(weights, base_experts)
        offset = ARENA_EXPERTS - base_experts

        if capacity not in owners:
            ids = torch.empty(capacity, 6, dtype=torch.int32, device="cuda")
            rw = torch.empty(capacity, 6, device="cuda")
            native = Native(lib, capacity, arena, wire, ids, rw,
                            spark_tp=options.spark_tp)
            assert native.info.role == 3 + options.spark_tp, native.info.role
            ids.fill_(-1)
            rw.zero_()
            native.run(1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                native.run(capacity)
            owners[capacity] = (native, ids, rw, graph)
        native, ids, rw, graph = owners[capacity]

        ids.fill_(-1)
        ids[:rows].copy_(ids_cpu + offset)
        rw.zero_()
        rw[:rows].copy_(routing)

        native.output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated, "replay allocated"

        expected = case["expected"]
        topk = native.info.topk
        if native.token_accumulation:
            actual = native.output[:rows]
        else:
            actual = native.output[: rows * topk].view(rows, topk, 5120).sum(1)
        assert bool(torch.isfinite(actual).all()), "non-finite native output"
        rel = ((actual - expected).norm() / expected.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0).item()
        assert rel < REL_TOL and cosine > COS_TOL, (case["case"], rel, cosine)

        # Sentinel masking (id 384, weight 0) must yield an exact zero row.
        masked_zero = None
        if rows >= 1:
            ids[rows - 1].fill_(SENTINEL)
            rw[rows - 1].zero_()
            native.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            if native.token_accumulation:
                masked_zero = native.output[rows - 1]
            else:
                masked_zero = native.output[(rows - 1) * topk: rows * topk].sum(0)
            assert bool((masked_zero == 0).all()), "masked row is not exact zero"
            masked_zero = float(masked_zero.abs().max().item())

        checked.add(capacity)
        print("NATIVE " + json.dumps(dict(
            case=case["case"], spark_tp=options.spark_tp,
            role=native.info.role, width=options.width,
            intermediate=native.info.logical_intermediate,
            kernel_intermediate=native.info.kernel_intermediate,
            abi_version=native.info.abi_version, topk=topk, capacity=capacity,
            rows=rows, experts=ARENA_EXPERTS, rel_l2=rel, cosine=cosine,
            masked_zero=masked_zero, scratch_bytes=native.storage.numel()),
            sort_keys=True), flush=True)

    _check_grouped_slices(options.width, after_case=check, n=intermediate, topk=6)
    assert checked == capacities, (checked, capacities)
    for native, _, _, graph in owners.values():
        graph.reset()


if __name__ == "__main__":
    main()
