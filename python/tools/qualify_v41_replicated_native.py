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

``--timing`` runs those same gates FIRST for every ``(capacity, rows, active)``
case it is about to time, then records two device timings per case: the expert
launch alone (``kernel_only``) and the expert launch plus the production BF16
compaction (``kernel_plus_compact``). Every ``TIMING`` record carries the seeded
input and route hashes, the manifest-verified per-capacity compiled width, and
the library/manifest identity, so a raw matrix proves which shape, routing,
shard geometry and artifact produced each number. ``--manifest`` is required
with ``--timing``; row counts must be covered by a requested capacity variant
(they are never clamped).

Run inside the DODO lease container with the pinned SparkInfer tree on
PYTHONPATH (or via the project runner). No daemon/service is touched.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

import _pinned_sparkinfer  # noqa: F401
from _v41_expert_native import Info, L, Native, P, check, library
from tests.moe.test_v41_grouped_slices import _check_grouped_slices

# Role 5 = Spark TP2 (I=1152), role 6 = Spark TP3 (I=768), role 7 = Spark TP6
# (I=384). TP2/TP3/TP6 are unpadded; the legacy TP4 shard stores 640 for a 576
# logical intermediate.
SPARK_TP_INTERMEDIATE = {2: 1152, 3: 768, 6: 384}
# Native `ds41rt_v41_expert_info_t.role` per TP degree, and the per-rank storage
# extent that role's exported variants must declare.
SPARK_TP_ROLE = {2: 5, 3: 6, 6: 7}
SPARK_TP_KERNEL_INTERMEDIATE = {2: 1152, 3: 768, 6: 384}
LEGACY_TP4_ROLE = 1
LEGACY_TP4_LOGICAL, LEGACY_TP4_STORAGE = 576, 640
ARENA_EXPERTS = 384
SENTINEL = 384
REL_TOL, COS_TOL = 0.01, 0.9999
FULL_INTERMEDIATE = 2304
OFFICIAL_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
DEFAULT_SNAPSHOT = (
    "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/"
    f"snapshots/{OFFICIAL_REVISION}"
)


def assert_native_role_geometry(lib, capacity, *, tp_degree=None, tp4_legacy=False):
    """Fail closed unless the loaded library really is the requested role family.

    `--width` selects which exported variant the fixture exercises; it does not
    prove the shared library was built with that role. Read the role id and the
    per-rank storage extent straight out of the native info struct for one
    precompiled capacity, so a TP6 run against a TP2/TP3/TP4-only library (or a
    library whose padded extent disagrees) stops here instead of silently
    measuring another topology's kernel.
    """
    meta = Info()
    check(lib.ds41rt_v41_expert_info(capacity, C.byref(meta)))
    if tp4_legacy:
        expected = (LEGACY_TP4_ROLE, LEGACY_TP4_LOGICAL, LEGACY_TP4_STORAGE)
    else:
        expected = (
            SPARK_TP_ROLE[tp_degree],
            SPARK_TP_INTERMEDIATE[tp_degree],
            SPARK_TP_KERNEL_INTERMEDIATE[tp_degree],
        )
    role, logical, storage = expected
    actual = (meta.role, meta.logical_intermediate, meta.kernel_intermediate)
    assert actual == expected, (
        f"native library does not expose the requested role at capacity "
        f"{capacity}: got role={meta.role} logical={meta.logical_intermediate} "
        f"storage={meta.kernel_intermediate}, expected role={role} "
        f"logical={logical} storage={storage}")
    assert meta.experts == ARENA_EXPERTS and meta.hidden_size == 5120, (
        meta.experts, meta.hidden_size)
    assert meta.topk == 6, meta.topk
    return meta


def report_compiled_widths(manifest_path, expected_intermediate, capacities,
                           requested_width=None):
    """Print and verify the width actually compiled into each AOT variant.

    `--width` is a REQUEST, not a property of the shared library. The exporter's
    qualified map emits capacity 1 at width 64 and every wider capacity at 192
    (or 128 for the width-B strategy), so a run that says "--width 192" for all
    capacities is misreporting the capacity-1 arm. When the role's export
    manifest is supplied, every requested capacity must be present and its
    compiled width is reported as requested-vs-actual.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    variants = manifest.get("variants", [])
    compiled = {}
    for variant in variants:
        capacity = variant.get("capacity_rows")
        if capacity is not None:
            compiled[int(capacity)] = variant.get("width")
    assert compiled, f"{manifest_path}: manifest declares no capacity variants"
    missing = [c for c in capacities if int(c) not in compiled]
    assert not missing, (
        f"{manifest_path}: no compiled variant for capacities {missing}; "
        f"have {sorted(compiled)}")
    per_capacity = {str(c): compiled[int(c)] for c in capacities}
    if requested_width is not None:
        mismatched = {c: (requested_width, compiled[int(c)]) for c in capacities
                      if compiled[int(c)] != requested_width}
        assert not mismatched, (
            f"--width {requested_width} does not match the compiled width for "
            f"capacities {sorted(mismatched)} as (requested, compiled)="
            f"{mismatched}; the qualified map emits capacity 1 at 64 and wider "
            "capacities at 192/128, so read the real widths with --manifest")
    print("COMPILED_WIDTHS " + json.dumps(
        {"manifest": str(manifest_path), "role": manifest.get("role"),
         "spark_tp_degree": manifest.get("spark_tp_degree"),
         "intermediate": expected_intermediate,
         "requested_width": requested_width,
         "per_capacity": per_capacity},
        sort_keys=True), flush=True)
    return compiled


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor):
    """Hash the exact bytes of a CUDA/CPU tensor, on the CPU, deterministically.

    bfloat16 has no numpy dtype, so it is widened to float32 (exact) before
    hashing; the dtype name is part of the digest, so tensors of different dtypes
    never collide.
    """
    cpu = tensor.detach().to("cpu").contiguous()
    dtype = str(cpu.dtype)
    if cpu.dtype == torch.bfloat16:
        payload = cpu.to(torch.float32).numpy().tobytes()
    else:
        payload = cpu.numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(dtype.encode())
    digest.update(payload)
    return digest.hexdigest()


def _routing_sha256(ids, routing):
    """Content hash of a routing pair without mixing dtypes in one cat()."""
    return hashlib.sha256(
        (_tensor_sha256(ids) + _tensor_sha256(routing)).encode()).hexdigest()


def parse_width(raw):
    """`--width` is either an explicit 64/128/192 or `auto` (None).

    `auto` means: trust and record the per-capacity widths compiled into the
    manifest, without asserting one scalar for a map that legitimately differs by
    capacity (capacity 1 is compiled at 64; wider capacities at 192/128).
    """
    value = str(raw).strip().lower()
    if value in ("auto", "none", ""):
        return None
    if value not in ("64", "128", "192"):
        raise SystemExit(f"--width must be auto, 64, 128 or 192, got {raw!r}")
    return int(value)


def parse_timing_rows(raw):
    """Comma list of positive row counts; ordered, de-duplicated, validated."""
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit() or int(item) < 1:
            raise SystemExit(f"--timing-rows entries must be positive integers, got {item!r}")
        values.append(int(item))
    if not values:
        raise SystemExit("--timing-rows must name at least one positive row count")
    return sorted(set(values))


def assert_rows_covered(rows_list, capacities):
    """Every requested row count must be covered by a compiled capacity variant.

    A silent `min(rows, capacity)` is forbidden: it would measure a different
    shape and still label it with the requested rows.
    """
    uncovered = [rows for rows in rows_list if rows > max(capacities)]
    if uncovered:
        raise SystemExit(
            f"--timing-rows {uncovered} exceed every requested capacity "
            f"{sorted(capacities)}; include the smallest AOT variant >= those rows "
            "(256/1024/4096 are precompiled)")
    return [(capacity, rows) for capacity in sorted(capacities)
            for rows in rows_list if rows <= capacity]


def verify_manifest_identity(manifest, expected_intermediate, expected_degree):
    """The supplied manifest must describe the arm being measured, not another."""
    geometry = manifest.get("geometry") or {}
    degree = manifest.get("spark_tp_degree")
    if degree != expected_degree:
        raise SystemExit(
            f"manifest spark_tp_degree={degree!r} does not match the requested "
            f"TP degree {expected_degree}")
    if geometry.get("intermediate") != expected_intermediate:
        raise SystemExit(
            f"manifest intermediate={geometry.get('intermediate')!r} does not "
            f"match the requested logical intermediate {expected_intermediate}")
    if geometry.get("experts") != ARENA_EXPERTS:
        raise SystemExit(f"manifest experts={geometry.get('experts')!r} != {ARENA_EXPERTS}")
    return manifest


def _native_forward(torch_module, native, rows):
    """The compiled output region for `rows`, shaped [rows, 5120]."""
    topk = native.info.topk
    if native.token_accumulation:
        return native.output[:rows]
    return native.output[:rows * topk].view(rows, topk, 5120).sum(1)


def _fill_routing(torch_module, ids, rw, rows, base, gids, active, device="cuda"):
    """Populate `rows` with `active` resident experts; inactive slots are SKIPPED.

    The first `active` slots carry valid resident ids and weights `1/active`.
    Slots at or beyond `active` carry the documented unassigned id `SENTINEL`
    (384) and weight 0, so the kernel does not dispatch a route for them:
    `active` is a real work axis, not a weights-only decoration. `SENTINEL` is
    the pinned contract's unassigned id (`V41_NATIVE_UNASSIGNED_EXPERT_ID`) and
    is already proven to contribute an exact zero row by the mask gate on every
    run; `-1` is the kernel's internal inverse, not a validated input sentinel,
    so it is deliberately not used here. `active == 6` fills every slot and is
    identical to the primary configuration. Returns the oracle-space ids (always
    valid indices, because the scalar oracle indexes its operand list by id and
    only the weight decides participation) and the weights.
    """
    active = max(1, min(int(active), 6))
    local_ids = (torch_module.arange(rows * 6, device=device, dtype=torch_module.int32)
                 .reshape(rows, 6) % len(gids))
    dispatch = local_ids + base
    dispatch[:, active:] = SENTINEL  # unassigned: no route for this slot
    routing = torch_module.zeros((rows, 6), device=device)
    routing[:, :active] = 1.0 / active
    ids.fill_(SENTINEL)
    rw.zero_()
    ids[:rows].copy_(dispatch)
    rw[:rows].copy_(routing)
    return local_ids, routing


def _route_counts(ids, rw):
    """`(valid_route_count, weighted_route_count)` for one routing pair.

    `valid_route_count` counts dispatched routes (any id other than the
    unassigned `SENTINEL`); the weighted count counts nonzero weights. They
    differ only if a slot carries an assigned id with a zero weight, which is
    exactly the ambiguity this record rules out.
    """
    valid = int((ids != SENTINEL).sum().item())
    weighted = int((rw > 0).sum().item())
    return valid, weighted


def _oracle_compact_mask_checks(torch_module, lib, native, wire, x, weights, scales,
                                ids, rw, capacity, rows, base, gids, active=6, *,
                                reference):
    """The shared native correctness gates: oracle, compaction, sentinel masks.

    This is the checkpoint path's gate set, factored out so the timing path runs
    exactly the same checks for every (capacity, rows, active) it times. The
    scalar oracle is passed in explicitly because `run_checkpoint` imports it as
    a local (it lives in the pinned SparkInfer test tree), so it is not a module
    global. Returns `(metrics, local_ids, routing, graph)`; `wire` is left holding
    the base activation and `ids`/`rw` are left all-inactive.
    """
    topk = native.info.topk
    wire.copy_(_quantize_wire(x, capacity, torch_module))
    local_ids, routing = _fill_routing(torch_module, ids, rw, rows, base, gids, active)
    native.output.fill_(float("nan"))
    graph = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(graph):
        native.run(rows)
    allocated = torch_module.cuda.memory_allocated()
    graph.replay()
    torch_module.cuda.synchronize()
    assert torch_module.cuda.memory_allocated() == allocated, "replay allocated"
    actual = _native_forward(torch_module, native, rows)
    assert bool(torch_module.isfinite(actual).all())
    expected = reference(x[:rows], local_ids, routing, weights, scales)
    rel = ((actual - expected).norm() / expected.norm()).item()
    cosine = torch_module.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0).item()
    assert rel < REL_TOL and cosine > COS_TOL, (capacity, rows, active, rel, cosine)

    # Runtime BF16 compaction on the native FP32 output. ABI 3 uses the token
    # compactor; ABI 2 uses the route compactor. Compared against the Python BF16
    # oracle (tolerance, not bit identity).
    bf16_expected = expected.bfloat16().float()
    compact = torch_module.empty((rows, 5120), device="cuda", dtype=torch_module.bfloat16)
    if native.token_accumulation:
        check(lib.ds41rt_v41_compact_tokens_bf16_async(
            native.output[:rows].data_ptr(), compact.data_ptr(), rows,
            torch_module.cuda.current_stream().cuda_stream))
    else:
        routes = native.output[:rows * topk].contiguous()
        check(lib.ds41rt_v41_compact_routes_bf16_async(
            routes.data_ptr(), compact.data_ptr(), rows,
            torch_module.cuda.current_stream().cuda_stream))
    torch_module.cuda.synchronize()
    compact_actual = compact.float()
    compact_rel = ((compact_actual - bf16_expected).norm() / bf16_expected.norm()).item()
    compact_cos = torch_module.nn.functional.cosine_similarity(
        compact_actual.flatten(), bf16_expected.flatten(), dim=0).item()
    assert compact_rel < REL_TOL and compact_cos > COS_TOL, (
        "compaction", capacity, rows, active, compact_rel, compact_cos)

    # Sentinel masking (id 384, weight 0) must yield an exact zero row, and every
    # inactive row must be an exact zero row.
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
    inactive = _native_forward(torch_module, native, rows)
    assert bool((inactive == 0).all()), "all-inactive rows not exact zero"
    metrics = dict(
        rel_l2=rel, cosine=cosine,
        compact_bf16_rel_l2=compact_rel, compact_bf16_cosine=compact_cos,
        mask_zero=True, all_inactive_zero=True)
    return metrics, local_ids, routing, graph


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


def _run_timing(options, torch_module, lib, arena, intermediate, role, gids, base,
                weights=None, scales=None, *, reference):
    """Device timing crosscheck with correctness and identity bound per case.

    For every (capacity, rows, active) the shared correctness gates run FIRST
    (numeric oracle, BF16 compaction, sentinel masks), so no timed number can
    come from a shape that was never checked. Two graphs are then timed per case:
    the expert launch alone (`kernel_only`) and the expert launch plus the
    production BF16 compaction (`kernel_plus_compact`). Every record carries the
    seeded input and route hashes, the manifest-verified per-capacity compiled
    width, and the library/manifest identity, so a raw matrix proves what it
    measured and which shard geometry produced it.

    Timing method: graphs are captured OUTSIDE the timed region. The amortized
    graph contains `--repeats` back-to-back launches, so one graph launch
    amortizes host enqueue cost; a 1-launch graph is timed the same way to expose
    the per-launch device floor. Events bracket the replay on the stream (device
    time); wall time around `replay()` gives host enqueue time. Cold uses a large
    device flush before each timed replay.
    """
    capacities = sorted({int(x) for x in options.capacities.split(",")})
    manifest = json.loads(Path(options.manifest).read_text())
    degree = 4 if options.tp4_legacy else options.spark_tp
    verify_manifest_identity(manifest, intermediate, degree)
    compiled = report_compiled_widths(options.manifest, intermediate, capacities,
                                      requested_width=options.width)
    rows_list = parse_timing_rows(options.timing_rows)
    grid = assert_rows_covered(rows_list, capacities)
    active_counts = [int(x) for x in options.active_counts.split(",")]
    assert active_counts and all(1 <= a <= 6 for a in active_counts), active_counts
    flush_bytes = int(options.cold_bytes)
    lib_sha = _sha256_file(options.native_lib)
    manifest_sha = _sha256_file(options.manifest)
    seed = int(getattr(options, "timing_seed", 0x51))
    results = []
    for capacity in capacities:
        # Confirm the library actually exposes this role family and extent
        # before any AOT variant is exercised or timed.
        assert_native_role_geometry(
            lib, capacity, tp_degree=options.spark_tp,
            tp4_legacy=options.tp4_legacy)
        meta = Info()
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(meta)))
        assert meta.role == role, (capacity, meta.role)
        generator = torch_module.Generator(device="cuda").manual_seed(seed)
        x = torch_module.randn(capacity, 5120, generator=generator,
                               device="cuda").mul_(0.5).bfloat16()
        input_sha = _tensor_sha256(x)
        wire = _quantize_wire(x, capacity, torch_module)
        ids = torch_module.empty(capacity, 6, dtype=torch_module.int32, device="cuda")
        rw = torch_module.empty(capacity, 6, device="cuda")
        native = Native(lib, capacity, arena, wire, ids, rw,
                        spark_tp=options.spark_tp)
        flush = torch_module.empty(flush_bytes // 4, device="cuda",
                                   dtype=torch_module.float32)
        compact = torch_module.empty((capacity, 5120), device="cuda",
                                     dtype=torch_module.bfloat16)
        topk = native.info.topk

        def compact_launch(rows):
            """The worker's production BF16 compaction on the native FP32 output."""
            if native.token_accumulation:
                check(lib.ds41rt_v41_compact_tokens_bf16_async(
                    native.output[:rows].data_ptr(), compact.data_ptr(), rows,
                    torch_module.cuda.current_stream().cuda_stream))
            else:
                check(lib.ds41rt_v41_compact_routes_bf16_async(
                    native.output[:rows * topk].data_ptr(), compact.data_ptr(), rows,
                    torch_module.cuda.current_stream().cuda_stream))
        # Every (capacity, rows) pair this variant covers; rows are never clamped.
        for case_capacity, rows in grid:
            if case_capacity != capacity:
                continue
            for active in active_counts:
                # B2: the checkpoint path's own gates for this exact
                # (capacity, rows, active), before any timed capture.
                metrics, _oracle_ids, _oracle_routing, check_graph = (
                    _oracle_compact_mask_checks(
                        torch_module, lib, native, wire, x, weights, scales,
                        ids, rw, capacity, rows, base, gids, active=active,
                        reference=reference))
                check_graph.reset()
                # Restore the base activation and the exact routing the timed
                # graphs read (the gates above left both mutated), then hash the
                # DISPATCHED device routing: inactive slots carry id -1, so this
                # binds the real work, not just the weights.
                wire.copy_(_quantize_wire(x, capacity, torch_module))
                _fill_routing(torch_module, ids, rw, rows, base, gids, active)
                route_sha = _routing_sha256(ids[:rows], rw[:rows])
                valid_routes, weighted_routes = _route_counts(ids[:rows], rw[:rows])

                single = torch_module.cuda.CUDAGraph()
                with torch_module.cuda.graph(single):
                    native.run(rows)
                single_compact = torch_module.cuda.CUDAGraph()
                with torch_module.cuda.graph(single_compact):
                    native.run(rows)
                    compact_launch(rows)
                amortized = torch_module.cuda.CUDAGraph()
                with torch_module.cuda.graph(amortized):
                    for _ in range(options.repeats):
                        native.run(rows)
                amortized_compact = torch_module.cuda.CUDAGraph()
                with torch_module.cuda.graph(amortized_compact):
                    for _ in range(options.repeats):
                        native.run(rows)
                        compact_launch(rows)
                # Both variants must write the region they own.
                for graph in (single, single_compact):
                    native.output.fill_(float("nan"))
                    graph.replay()
                    torch_module.cuda.synchronize()
                    written = (native.output[:rows] if native.token_accumulation
                               else native.output[:rows * native.info.topk])
                    assert bool(torch_module.isfinite(written).all()), (
                        "timing graph did not write the output")
                for _ in range(10):
                    single.replay()
                    amortized.replay()
                    single_compact.replay()
                    amortized_compact.replay()
                torch_module.cuda.synchronize()
                # kernel_only and kernel_plus_compact are measured in SEPARATE
                # windows, so a small negative delta (compact faster) is
                # unresolved latency variance between windows -- NOT proof of a
                # clock difference and not a valid per-case compact gain.
                warm_single_device, warm_single_host = _time_graph(
                    torch_module, single, options.single_replays, 1)
                warm_amortized_device, warm_amortized_host = _time_graph(
                    torch_module, amortized, options.warm_replays, options.repeats)
                warm_single_compact, warm_single_compact_host = _time_graph(
                    torch_module, single_compact, options.single_replays, 1)
                # The compact graph runs two kernels per repeat, so per-repeat
                # division keeps the two amortized numbers directly comparable.
                warm_amortized_compact, warm_amortized_compact_host = _time_graph(
                    torch_module, amortized_compact, options.warm_replays,
                    options.repeats)
                cold_device, cold_compact = [], []
                for _ in range(options.cold_replays):
                    flush.fill_(1.0)
                    cold_device.extend(_time_graph(torch_module, single, 1, 1)[0])
                    flush.fill_(1.0)
                    cold_compact.extend(
                        _time_graph(torch_module, single_compact, 1, 1)[0])
                record = dict(
                    kind="native_timing", role=role, spark_tp=options.spark_tp,
                    tp4_legacy=bool(options.tp4_legacy), capacity=capacity,
                    rows=rows, active=active, abi_version=meta.abi_version,
                    valid_route_count=valid_routes,
                    weighted_route_count=weighted_routes,
                    intermediate=meta.logical_intermediate,
                    kernel_intermediate=meta.kernel_intermediate,
                    width=compiled[int(capacity)], repeats=options.repeats,
                    cold_bytes=flush_bytes, timing_seed=seed,
                    input_sha256=input_sha, route_sha256=route_sha,
                    lib_sha256=lib_sha, manifest_path=str(options.manifest),
                    manifest_sha256=manifest_sha,
                    manifest_role=manifest.get("role"),
                    manifest_spark_tp_degree=manifest.get("spark_tp_degree"),
                    scratch_bytes=native.storage.numel(),
                    compact_output_bytes=compact.numel() * compact.element_size(),
                    kernel_only=dict(
                        warm_single_device_us=statistics.median(warm_single_device),
                        warm_amortized_device_us=statistics.median(warm_amortized_device),
                        warm_single_host_us=statistics.median(warm_single_host),
                        warm_amortized_host_us=statistics.median(warm_amortized_host),
                        cold_device_us=statistics.median(cold_device),
                        warm_single_samples_us=warm_single_device,
                        warm_amortized_samples_us=warm_amortized_device,
                        cold_samples_us=cold_device),
                    kernel_plus_compact=dict(
                        warm_single_device_us=statistics.median(warm_single_compact),
                        warm_amortized_device_us=statistics.median(warm_amortized_compact),
                        warm_single_host_us=statistics.median(warm_single_compact_host),
                        warm_amortized_host_us=statistics.median(warm_amortized_compact_host),
                        cold_device_us=statistics.median(cold_compact),
                        warm_single_samples_us=warm_single_compact,
                        warm_amortized_samples_us=warm_amortized_compact,
                        cold_samples_us=cold_compact),
                    **metrics)
                results.append(record)
                print("TIMING " + json.dumps(record, sort_keys=True), flush=True)
                for graph in (single, single_compact, amortized, amortized_compact):
                    graph.reset()
        del native, wire, ids, rw, flush, compact
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
    role = (LEGACY_TP4_ROLE if options.tp4_legacy
            else SPARK_TP_ROLE[options.spark_tp])
    intermediate = (576 if options.tp4_legacy
                    else SPARK_TP_INTERMEDIATE[options.spark_tp])
    lo = options.tp_rank * intermediate
    hi = lo + intermediate
    gids = [int(x) for x in options.expert_ids.split(",")]
    assert gids and len(gids) <= ARENA_EXPERTS
    base = ARENA_EXPERTS - len(gids)
    # The library must really be the requested role family. A TP6 run against a
    # library built without the tp6 role would otherwise fail later with a
    # confusing symbol/geometry error, or worse, measure another topology.
    probe_capacity = min(int(x) for x in options.capacities.split(","))
    assert_native_role_geometry(lib, probe_capacity, tp_degree=options.spark_tp,
                                tp4_legacy=options.tp4_legacy)
    if getattr(options, "manifest", None) is not None:
        report_compiled_widths(
            options.manifest, intermediate,
            sorted({int(x) for x in options.capacities.split(",")}),
            requested_width=options.width)
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
                           gids, base, weights=weights, scales=scales,
                           reference=reference)

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
            # The shared correctness gates (oracle, compaction, sentinel masks)
            # are the same code the timing path runs for every case it times.
            # The changed-activation gate below rewrites the wire, and each row
            # case restores it from `x` inside the helper.
            metrics, local_ids, routing, graph = _oracle_compact_mask_checks(
                torch_module, lib, native, wire, x, weights, scales,
                ids, rw, capacity, rows, base, gids, active=6,
                reference=reference)
            graphs.append(graph)
            rel = metrics["rel_l2"]
            cosine = metrics["cosine"]
            compact_bf16_rel = metrics["compact_bf16_rel_l2"]
            compact_bf16_cos = metrics["compact_bf16_cosine"]
            topk = native.info.topk

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
    parser.add_argument("--spark-tp", type=int, choices=(2, 3, 6),
                        help="Spark TP degree: 2 (I=1152), 3 (I=768) or 6 (I=384)")
    parser.add_argument("--tp4-legacy", action="store_true",
                        help="Use role 1 TP4 (I=576, pad 640) for the timing baseline")
    parser.add_argument("--width", type=parse_width, default="auto",
                        help="fixture slice width, or 'auto' to trust and record the "
                             "per-capacity widths compiled into --manifest (capacity 1 "
                             "is compiled at 64; wider capacities at 192/128)")
    parser.add_argument("--capacities", default="1,16,80",
                        help="Native capacity variants to check")
    parser.add_argument("--rows", default="",
                        help="Optional comma list to restrict fixture row cases")
    parser.add_argument("--checkpoint", action="store_true",
                        help="Official checkpoint weights via the native packer + ABI3")
    parser.add_argument("--timing", action="store_true",
                        help="Device timing crosscheck: correctness gates + kernel-only "
                             "and kernel+compaction timings per (capacity, rows, active)")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Launches captured in the amortized timing graph")
    parser.add_argument("--warm-replays", type=int, default=30)
    parser.add_argument("--single-replays", type=int, default=30)
    parser.add_argument("--cold-bytes", type=int, default=32 << 20)
    parser.add_argument("--cold-replays", type=int, default=5)
    parser.add_argument("--active-counts", default="6,3,2")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--timing-rows", default="1",
                        help="comma list of rows driven per timed replay (for example "
                             "1,8,16,64,256,1024,4096); each must be covered by a "
                             "requested capacity or the run fails instead of clamping")
    parser.add_argument("--timing-seed", type=int, default=0x51,
                        help="seed for the timed activation, recorded with its hash so "
                             "every arm provably runs the same input")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="role export v41_experts.json; REQUIRED for --timing: the "
                             "per-capacity compiled width and the manifest identity are "
                             "verified and recorded in every timing record")
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
    if options.timing:
        if options.manifest is None:
            parser.error("--timing requires --manifest: the compiled per-capacity "
                         "width and the artifact identity must be verified, not assumed")
        assert_rows_covered(parse_timing_rows(options.timing_rows),
                            sorted({int(x) for x in options.capacities.split(",")}))

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
    # The synthetic grouped-slices path needs a concrete slice width; `auto`
    # (the manifest-driven mode used by --timing) resolves to the historical
    # default here because this path has no manifest to read.
    fixture_width = options.width if options.width is not None else 192

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
            assert native.info.role == SPARK_TP_ROLE[options.spark_tp], native.info.role
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
            role=native.info.role, width=fixture_width,
            intermediate=native.info.logical_intermediate,
            kernel_intermediate=native.info.kernel_intermediate,
            abi_version=native.info.abi_version, topk=topk, capacity=capacity,
            rows=rows, experts=ARENA_EXPERTS, rel_l2=rel, cosine=cosine,
            masked_zero=masked_zero, scratch_bytes=native.storage.numel()),
            sort_keys=True), flush=True)

    _check_grouped_slices(fixture_width, after_case=check, n=intermediate, topk=6)
    assert checked == capacities, (checked, capacities)
    for native, _, _, graph in owners.values():
        graph.reset()


if __name__ == "__main__":
    main()
