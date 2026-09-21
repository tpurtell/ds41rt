#!/usr/bin/env python3
"""Rank-local kernel micro-timing for replicated expert groups (phase 0).

Scope and honesty rules, all enforced here rather than asserted in prose:

* **Rank-local critical path only.** One physical rank is timed. Parallel TP
  ranks are never summed and no cross-Spark or transport latency is claimed.
* **Warm and cold are separate conditions.** Warm times ``replays`` consecutive
  graph replays. Cold flushes a buffer larger than the device L2, then times
  exactly ONE replay. The flush is issued through stream-ordered CUDA events and
  is excluded from the timed region. Several post-flush replays are never
  averaged and called "cold".
* **The output is never poisoned inside a timing interval.** Poisoning is a
  correctness preflight only; once warmup starts the timed path is exactly the
  serving path. A correctness check runs once before any sample is taken, and a
  post-timing check confirms the timed path still produces the same finite
  result (a no-op graph cannot pass).
* **Statistics**: median, p10 and p90 over at least ``--samples`` intervals,
  after warmup, with clocks, temperature and memory recorded per arm.
* Ownership is the diagnostic modulo mask unless ``--lpt`` is given, in which
  case groups are assigned by greedy longest-processing-time over the measured
  route histogram. Both are reported as a projected cost model, not as a
  production scheduler.

The native AOT leg is NOT implemented and is disabled fail-closed; this harness
only exercises the true Python ``V41SlicePipeline`` path.

Run through ``scripts/run-tp-ep-kernel-checks.sh bench-kernel``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

# Canonical DS41RT resolver: verifies the pinned SparkInfer tree and its lock
# (honouring DS41RT_SPARKINFER_SOURCE_DIR), puts that tree first on sys.path, and
# fails closed if `b12x` does not import from it. Import it before anything from
# SparkInfer so an unverified copy can never be used.
import _pinned_sparkinfer  # noqa: F401  (import side effect is the point)
from b12x._lib.utils import current_cuda_stream  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# One shard geometry per topology: the TP degree fixes how wide this rank's
# slice is, so the per-rank intermediate comes with it. `tp6` is the pure
# six-way split of the official 2304 (384 per rank, no padding); TP2/TP3 are the
# replicated-group degrees and TP4 is the historical padded shard.
TP_GEOMETRIES = ((2, 1152, "tp2"), (4, 576, "tp4"), (3, 768, "tp3"),
                 (6, 384, "tp6"))
# The slice kernel asserts width in (64, 128, 192); anything else cannot be
# exported, so the harness must not silently accept it.
SUPPORTED_SLICE_WIDTHS = (64, 128, 192)


def _load_bench():
    path = ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py"
    spec = importlib.util.spec_from_file_location("ds41rt_ep_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()
HIDDEN = bench.HIDDEN


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        raise SystemExit("Blackwell SM12x device required")


# --------------------------------------------------------------------------- #
# Clock / thermal sampling
# --------------------------------------------------------------------------- #

CLOCK_FIELDS = ("clocks.sm,clocks.mem,temperature.gpu,power.draw,"
                "utilization.gpu,memory.used")


def gpu_state():
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={CLOCK_FIELDS}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    values = [v.strip() for v in proc.stdout.strip().splitlines()[0].split(",")]
    keys = ["sm_clock", "mem_clock", "temperature_c", "power_w", "util_pct",
            "memory_used"]
    return dict(zip(keys, values))


def host_state():
    try:
        load = Path("/proc/loadavg").read_text().split()[:3]
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = rest.strip()
    except OSError as error:
        return {"error": str(error)}
    return {"loadavg": load, "memory": mem}


# --------------------------------------------------------------------------- #
# Timing primitives
# --------------------------------------------------------------------------- #

def warm_samples(replay, samples, replays, torch_module):
    """Per-interval means from CUDA events around host-submitted replays.

    WARNING: the event window contains host submission. On a slow-submission
    host the GPU idles between replays inside the window, so this is an upper
    bound that can be dominated by Python/launch overhead rather than kernel
    time. Compare with `amortised_samples` before attributing anything to the
    GPU.
    """
    out = []
    for _ in range(samples):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / replays)
    return out


def _assert_finite(case, rows, label, where):
    assert bool(torch.isfinite(case.out[: rows * 6]).all()), (
        f"{label}: {where} produced non-finite output (no-op capture?)")


def cold_samples(replay, flush, samples, torch_module):
    """Flush, then time exactly ONE replay, stream-ordered, repeated.

    The flush is issued on the same stream before the start event, so it is
    excluded from the timed region and the GPU is cold when the single timed
    replay runs. Several post-flush replays are never averaged into one sample.
    """
    out = []
    for _ in range(samples):
        flush()
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0)
    return out


def amortised_samples(launch, samples, replays, torch_module, verify=None):
    """GPU-resident repeat: capture N DIRECT launches into ONE outer graph.

    Repeating host-submitted graph replays does NOT amortise anything --
    `T = N * (gpu + host_gap) / N` is constant -- so that approach proves nothing
    and was wrong. The valid construction captures N direct compiled launches as
    N nodes of a single outer graph. Replaying that outer graph once executes all
    N back-to-back on the GPU with no host gap between them, so the timed window
    measures GPU-resident work.

    Nested *graph replay* inside a capture is illegal; nested direct *kernel*
    launches are exactly what capture is for. `launch(stream)` must therefore
    perform one direct compiled launch on the given stream, not a graph replay.

    `verify` (optional) runs after capture and before timing: a no-op outer graph
    must not be able to pass, so the first replay is checked against the oracle.
    """
    outer = torch_module.cuda.CUDAGraph()
    with torch_module.cuda.graph(outer):
        for _ in range(replays):
            launch(current_cuda_stream())
    # Prove the captured outer graph actually contains work.
    outer.replay()
    torch_module.cuda.synchronize()
    if verify is not None:
        verify()

    out = []
    for _ in range(samples):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        outer.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / replays)
    del outer
    return out


def stats(samples):
    ordered = sorted(samples)
    def pct(q):
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]
    return {"n": len(ordered), "median_us": statistics.median(ordered),
            "p10_us": pct(0.10), "p90_us": pct(0.90),
            "min_us": ordered[0], "max_us": ordered[-1],
            "mean_us": statistics.fmean(ordered), "samples_us": list(samples)}


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #

def load_route_fixture(path, experts, topk, rows):
    """Load a validated route fixture and return an [rows, topk] int32 tensor.

    A fixture exists so a comparison can run on a distribution-representative or
    reuse-bearing route set instead of the all-distinct synthetic table. It is
    validated here rather than trusted: shape, expert-id range, distinct within a
    row, non-empty, and that it actually contains reuse when it claims to.
    """
    import json as _json
    payload = _json.loads(Path(path).read_text())
    if payload.get("schema") != "ds41rt.tp-ep-reuse-fixture.v1":
        raise ValueError(f"unsupported fixture schema {payload.get('schema')!r}")
    routes = payload["routes"]
    if len(routes) < rows:
        raise ValueError(f"fixture has {len(routes)} rows, need {rows}")
    out = torch.zeros((rows, topk), dtype=torch.int32)
    for r in range(rows):
        row = routes[r]
        if len(row) != topk:
            raise ValueError(f"row {r} has {len(row)} routes, expected {topk}")
        if len(set(row)) != topk:
            raise ValueError(f"row {r} has duplicate expert ids: {row}")
        for c, expert in enumerate(row):
            if not 0 <= int(expert) < experts:
                raise ValueError(f"row {r} expert {expert} out of range 0..{experts - 1}")
            out[r, c] = int(expert)
    return out, payload


def synthetic_route_table(capacity, experts):
    """The all-distinct synthetic route table, on CPU.

    `((r + 1) * 6 + slot) % experts`: distinct within a row, and at M8/M16 it
    names distinct experts (48 at M8, 96 at M16) with no reuse.
    """
    return (torch.arange(1, capacity + 1, dtype=torch.int32).reshape(capacity, 1) * 6
            + torch.arange(6, dtype=torch.int32).reshape(1, 6)) % experts


def iter_live_rows(options, experts):
    """Yield (rows, full_ids_cpu, fixture_payload) for each live row count.

    This is the caller-level orchestration `measure_group` runs. Each live row count
    is bound explicitly to its own route table, and a multi-row request fails closed
    instead of silently reusing a leaked `rows` binding. It is separated out so the
    caller path can be regression-tested without a GPU.
    """
    if not options.rows:
        raise SystemExit("no live row counts requested")
    if len(options.rows) > 1:
        raise SystemExit(
            "one live row count per invocation is required; the route table is bound "
            "per row count and a multi-row request is not supported")
    for rows in options.rows:
        full_ids_cpu, fixture_payload = build_route_table(
            options, experts, rows, options.capacity)
        yield rows, full_ids_cpu, fixture_payload


def build_route_table(options, experts, rows, capacity):
    """Return (full_ids_on_cpu, fixture_payload) with an explicit live-row contract.

    Contract: exactly `rows` live routes are selected, in order, from the fixture
    when one is configured, otherwise from the synthetic table. The buffer is
    capacity-sized because the kernel upload is padded to capacity, so the unused
    tail is zero-filled; those rows are never routed (`live = rows`) and cannot be
    read.

    This is deliberately a single helper: before it existed the selection logic was
    inline and a synthetic-plus-fixture call could establish `rows` in one branch
    while a later branch selected rows for another, which is the defect this
    replaces. It fails closed if the fixture cannot supply the live rows.
    """
    if options.route_fixture is None:
        return synthetic_route_table(capacity, experts), None
    if rows > capacity:
        raise ValueError(f"live rows {rows} exceed capacity {capacity}")
    live, payload = load_route_fixture(options.route_fixture, experts, 6, rows)
    if live.shape != (rows, 6):
        raise ValueError(f"fixture selection gave {tuple(live.shape)}, expected {(rows, 6)}")
    if capacity > rows:
        padding = torch.zeros((capacity - rows, 6), dtype=torch.int32)
        live = torch.cat([live, padding], dim=0)
    return live, payload


def _padded(masked, masked_weights, capacity, rows):
    """Pad the live rows to capacity with sentinel ids and zero weights."""
    if rows >= capacity:
        return masked, masked_weights
    pad_ids = torch.full((capacity - rows, 6), bench.SENTINEL, device="cuda",
                         dtype=torch.int32)
    pad_weights = torch.zeros((capacity - rows, 6), device="cuda",
                              dtype=torch.float32)
    return (torch.cat([masked, pad_ids], 0), torch.cat([masked_weights, pad_weights], 0))


def modulo_owner(ids, groups, rank):
    return bench.owners_for(ids, groups, rank)


def expert_route_counts(ids, experts):
    """Route count per expert as a single host-side Python list.

    Diagnostic cost-model input, NOT a measurement of scheduler CPU overhead.
    The Rust scheduler builds its host histogram from the already-available
    canonical router ids and performs no new device-to-host transfer; this
    harness's bincount is a harness convenience, not a model of that path.
    """
    return torch.bincount(ids.flatten().to(torch.int64),
                          minlength=experts).tolist()


def expert_cost(rows, *, weight_cost=1.0, tile_cost=0.0, tile_rows=16):
    """Cost of serving one expert with ``rows`` routed rows.

    Authoritative form, matching ``ds41rt-core``'s ``ReplicatedExpertCostModel``
    (``rust/crates/ds41rt-core/src/replicated_expert_schedule.rs``):

        active   -> expert_weight_cost + ceil(rows / tile_rows) * tile_cost
        inactive -> 0

    ``weight_cost`` is a UNIFORM per-active-expert term, charged once, NOT a
    per-row term. Inactive experts are free and are never assigned.
    """
    rows = int(rows)
    if rows <= 0:
        return 0.0
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    return float(weight_cost) + float(tile_cost) * (-(-rows // int(tile_rows)))


def lpt_owner(ids, groups, rank, experts, *, weight_cost=1.0, tile_cost=0.0,
              tile_rows=16):
    """Greedy LPT assignment of WHOLE experts to groups.

    Cost model per expert, the same shape the coordinator's boundary model uses:

        cost(expert) = expert_weight_cost
                     + ceil(count(expert) / tile_rows) * tile_cost     (active)
        cost(expert) = 0                                               (inactive)

    ``expert_weight_cost`` is a UNIFORM cost for serving the expert at all — it is
    charged ONCE per active expert, NOT per routed row. ``tile_cost`` prices each
    partially filled tile of ``tile_rows`` rows. This mirrors
    ``ds41rt-core``'s ``ReplicatedExpertCostModel``. Inactive experts cost exactly
    0 and are skipped, so a positive tile_cost never charges a group for an expert
    it does not serve.

    Ties are broken by lowest group index after a stable descending sort, so the
    result is deterministic. This is a DIAGNOSTIC projected cost model; the Rust
    scheduler's tie-breaking may differ and is not reproduced here.
    """
    counts = expert_route_counts(ids, experts)
    rows_per_expert = [int(c) for c in counts]
    cost = [expert_cost(rows, weight_cost=weight_cost, tile_cost=tile_cost,
                        tile_rows=tile_rows)
            for rows in rows_per_expert]
    # Stable descending by cost, then by expert id, so ties are deterministic.
    order = sorted(range(experts), key=lambda e: (-cost[e], e))
    load = [0.0] * groups
    assignment = [0] * experts
    for expert in order:
        if cost[expert] == 0.0:
            continue                      # inactive: never assigned, never charged
        target = min(range(groups), key=lambda g: (load[g], g))
        assignment[expert] = target
        load[target] += cost[expert]
    owner = torch.zeros_like(ids, dtype=torch.bool)
    for expert, group in enumerate(assignment):
        if group == rank:
            owner |= (ids == expert)
    schedule = {
        "group_loads": [round(v, 6) for v in load],
        "assignment": assignment,
        "rows_per_expert": rows_per_expert,
        "expert_cost": [round(v, 6) for v in cost],
        "cost_model": {"weight_cost": weight_cost, "tile_cost": tile_cost,
                       "tile_rows": tile_rows},
        "tie_break": "stable descending cost then ascending expert id; "
                     "the Rust scheduler may break ties differently",
        "histogram_transfer": "one diagnostic device-to-host bincount; not a "
                             "measurement of scheduler CPU overhead",
    }
    return owner, schedule


# --------------------------------------------------------------------------- #
# One measured configuration
# --------------------------------------------------------------------------- #

def _stage_breakdown(case, rows, options, label):
    """Time plan / compute / reduce separately, each under its own graph.

    Isolated stage timings; they do not reconstruct the full pipeline numerics.
    The launch geometry differs from the real pipeline (each stage is launched on
    its own), so these attribute fixed overhead and must not be summed into a
    claim about the full path.
    """
    import cutlass

    out = {}
    launches = {
        "plan": lambda stream: case.plan(
            case._args[6], case._args[7], case._args[8], case._args[9],
            case._args[10], case._args[11], case._args[12], case._args[13],
            case._args[14], stream),
        "reduce": lambda stream: case.reduce(
            case._args[15], case._args[16], case._args[14], case._args[8], stream,
            cutlass.Int32(rows)),
    }
    if case.compute is None:
        out["compute"] = {"unavailable": case.compute_error}
        print(json.dumps(dict(label=label, stage="compute",
                              unavailable=case.compute_error)), flush=True)
    else:
        launches["compute"] = lambda stream: case.compute(
            case._args[0], case._args[1], *case._weight_buffers, case._args[9],
            case._args[11], cutlass.Int32(rows), stream, case._args[8],
            max(1, min(rows * 6, case.experts)))
    for name, launch in launches.items():
        for _ in range(3):
            launch(current_cuda_stream())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(current_cuda_stream())
        for _ in range(options.warmup):
            graph.replay()
        torch.cuda.synchronize()
        stats_out = stats(warm_samples(graph.replay, options.samples,
                                       options.replays, torch))
        out[name] = {"median_us": stats_out["median_us"],
                     "p10_us": stats_out["p10_us"], "p90_us": stats_out["p90_us"],
                     "n": stats_out["n"]}
        print(json.dumps(dict(label=label, stage=name,
                              median_us=stats_out["median_us"],
                              p10_us=stats_out["p10_us"],
                              p90_us=stats_out["p90_us"],
                              n=stats_out["n"])), flush=True)
        del graph
        torch.cuda.synchronize()
    return out


def _post_timing_check(graph, case, rows, expected, label):
    """Replay and compare against THIS arm's oracle while its state is intact."""
    case.out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    tail = case.out[: rows * 6].view(rows, 6, HIDDEN)
    assert bool(torch.isfinite(tail).all()), f"{label}: post-timing non-finite"
    tail_rel = ((tail.sum(1) - expected).norm() / expected.norm()).item()
    assert tail_rel < 0.01, f"{label}: post-timing drift {tail_rel}"
    return tail_rel


def _snapshot_arm_state(ids_buffer, routing_buffer, upload_ids, upload_weights,
                        rows, capacity):
    """Restore the arm's own ids/routing after a mutating optional block.

    The buffers are capacity-sized (`capacity * topk`), so the live rows must be
    padded back to the padded upload form. Copying the unpadded `rows * topk`
    view raises a size mismatch for any arm with rows < capacity.
    """
    assert ids_buffer.numel() == capacity * 6, (
        f"ids buffer is {ids_buffer.numel()}, expected {capacity * 6}")
    ids_buffer.copy_(upload_ids.flatten())
    routing_buffer.copy_(upload_weights.flatten())
    return {"restored_rows": rows, "restored_capacity": capacity}


def _per_group_latency(case, rows, options, local, experts, label,
                      live_ids, live_weights):
    """Measure EVERY EP group's own mask under the SAME conditions.

    Each group gets its own capture and its own warm / GPU-resident / cold
    samples through `_timed_condition`, so a group's maximum is comparable across
    groups instead of mixing one group's cold number with another's warm one.
    Groups are measured one at a time on the same shard, so any maximum is a
    PROJECTED critical group, not a measured concurrent distributed path.
    """
    from tests.moe.test_v41_expert_numerics import reference
    import cutlass

    # The per-group block MUST use the same routes and weights as the timed arm.
    # Regenerating a synthetic table here silently measured a different workload
    # from the arm's own routes (and was incompatible with --route-fixture).
    assert live_ids.shape == (rows, 6), (live_ids.shape, rows)
    assert live_weights.shape == (rows, 6), live_weights.shape
    measured = {}
    for group in range(options.ep_degree):
        if options.lpt:
            owner, _ = lpt_owner(live_ids, options.ep_degree, group, experts,
                                 weight_cost=options.weight_cost,
                                 tile_cost=options.tile_cost,
                                 tile_rows=options.tile_rows)
        else:
            owner = modulo_owner(live_ids, options.ep_degree, group)
        masked = bench.mask_ids(live_ids, owner)
        masked_weights = torch.where(owner, live_weights,
                                     torch.zeros_like(live_weights)).contiguous()
        upload_ids, upload_weights = _padded(masked, masked_weights,
                                             options.capacity, rows)
        active = int(masked[owner].unique().numel())
        expected = None
        if active:
            expected = reference(case.x[:rows], live_ids, masked_weights,
                                 local.weights, local.scales)

        # Preflight: oracle agreement AND an exact zero on unowned routes.
        case.out.fill_(float("nan"))
        case.ids.copy_(upload_ids.flatten())
        case.routing.copy_(upload_weights.flatten())
        case.live.fill_(rows)
        case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        plane = case.out[: rows * 6].view(rows, 6, HIDDEN)
        assert bool(torch.isfinite(plane).all()), f"{label} group {group}: non-finite"
        owned = owner[:rows].flatten()
        if not bool(owned.all()):
            assert bool((plane.view(-1, HIDDEN)[~owned] == 0).all()), (
                f"{label} group {group}: unowned route not an exact zero")
        if active:
            got = plane.sum(1)
            rel = ((got - expected).norm() / expected.norm()).item()
            cosine = torch.nn.functional.cosine_similarity(
                got.flatten(), expected.flatten(), dim=0).item()
            assert rel < 0.01 and cosine > 0.9999, (label, group, rel, cosine)
        else:
            # A group owning nothing must produce an exact zero everywhere.
            assert bool((plane == 0).all()), (
                f"{label} group {group}: inactive group produced output")

        for _ in range(3):
            case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(case.out[: rows * 6]).all()), (
            f"{label} group {group}: empty capture")

        warm, amortised, cold = _timed_condition(graph, expected, rows, case,
                                                 options, label)
        # Post-timing oracle: the timed path still computes the right thing.
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        tail = case.out[: rows * 6].view(rows, 6, HIDDEN)
        assert bool(torch.isfinite(tail).all()), (
            f"{label} group {group}: post-timing non-finite")
        if active:
            tail_got = tail.sum(1)
            tail_cos = torch.nn.functional.cosine_similarity(
                tail_got.flatten(), expected.flatten(), dim=0).item()
            assert tail_cos > 0.9999, (label, group, tail_cos)
        else:
            assert bool((tail == 0).all()), (
                f"{label} group {group}: inactive group wrote on replay")

        entry = {"active_experts": active,
                 "owned_routes": int(owner[:rows].sum().item()),
                 "warm": warm, "amortised": amortised, "cold": cold}
        measured[f"group{group}"] = entry
        print(json.dumps(dict(
            label=label, per_group=f"group{group}", active_experts=active,
            owned_routes=entry["owned_routes"],
            warm_median_us=warm["median_us"],
            amortised_median_us=(amortised or {}).get("median_us"),
            cold_median_us=(cold or {}).get("median_us"))), flush=True)
        del graph
        torch.cuda.synchronize()

    # Only the per-group entries are dicts; the summary keys added below are
    # scalars, so iterate the group entries explicitly rather than every key.
    group_entries = {k: v for k, v in measured.items()
                     if k.startswith("group") and isinstance(v, dict)}
    for metric in ("warm", "amortised", "cold"):
        entries = [(g, v[metric]["median_us"]) for g, v in group_entries.items()
                   if v.get(metric)]
        if entries:
            worst_group, worst_value = max(entries, key=lambda kv: kv[1])
            measured[f"max_{metric}_group"] = worst_group
            measured[f"max_{metric}_median_us"] = worst_value
    measured["max_group_measured_sequentially"] = True
    measured["max_group_is_prediction"] = (
        "groups measured one at a time on one shard; this is a projected "
        "critical group, not a concurrent distributed path, and the Rust "
        "scheduler's own assignment may differ")
    print(json.dumps(dict(
        label=label,
        max_warm_group=measured.get("max_warm_group"),
        max_warm_us=measured.get("max_warm_median_us"),
        max_amortised_group=measured.get("max_amortised_group"),
        max_amortised_us=measured.get("max_amortised_median_us"),
        max_cold_group=measured.get("max_cold_group"),
        max_cold_us=measured.get("max_cold_median_us"),
        note="sequential measurement; projected critical group")), flush=True)
    return measured


# --------------------------------------------------------------------------- #
# Ownership
# --------------------------------------------------------------------------- #
def _timed_condition(graph, expected, rows, case, options, label):
    """Warmup, then warm and (optionally) cold sampling. No poison in intervals."""
    import cutlass
    for _ in range(options.warmup):
        graph.replay()
    torch.cuda.synchronize()
    warm = stats(warm_samples(graph.replay, options.samples, options.replays, torch))
    amortised = None
    if options.amortised:
        def _direct_launch(stream):
            case.full(*case._args, cutlass.Int32(rows), stream)

        case.out.fill_(float("nan"))
        amortised = stats(amortised_samples(
            _direct_launch, options.samples, options.replays, torch,
            verify=lambda: _assert_finite(case, rows, label, "amortised")))

    cold = None
    if options.cold:
        flush = torch.empty(options.flush_bytes // 4, device="cuda",
                            dtype=torch.float32)

        def do_flush():
            flush.fill_(1.0)          # stream-ordered; excluded from the timed region

        for _ in range(options.warmup):
            do_flush()
        torch.cuda.synchronize()
        cold = stats(cold_samples(graph.replay, do_flush, options.samples, torch))
    return warm, amortised, cold


def measure_group(options, tp_degree, intermediate, width, experts):
    """Measure ONE live row count for one (topology, width) on one compiled case.

    Exactly one positive row count is accepted per invocation (enforced in
    `parse_args`): the route table is bound per live row count, so a multi-row
    request is rejected rather than measured. The case is built once per geometry
    and the single live row count is uploaded into it.
    """
    from b12x._lib.utils import current_cuda_stream
    from tests.moe.test_v41_expert_numerics import reference
    import cutlass

    capacity = options.capacity
    # Operand source. `checkpoint` reads REAL weights, but only for the experts
    # the requested live histograms actually route to; unused dense slots are
    # zeroed. The dense bank is allocated at the FULL width the caller requests
    # (`intermediate * tp_degree`), so its size follows that width and the expert
    # count; the identity records the exact byte count and every record carries
    # the measured peak. These are packed 8-bit bytes, never an FP32 dequantised
    # bank. Do not restate a figure here -- read it from the record.
    if options.operands == "checkpoint":
        union = set()
        for rows in options.rows:
            ids_for_rows = (torch.arange(1, rows + 1, device="cuda",
                                         dtype=torch.int32).reshape(rows, 1) * 6
                            + torch.arange(6, device="cuda",
                                           dtype=torch.int32).reshape(1, 6)
                            ) % experts
            union.update(bench.active_expert_ids(ids_for_rows, experts))
        active = sorted(union)
        # The label must follow the actual active count: when the active union
        # covers every expert the bank IS the full real expert bank, and saying
        # otherwise is untruthful (it is what a fixture M64 run produces).
        print(json.dumps(dict(starting_operands="checkpoint",
                              layer=options.layer, active_count=len(active),
                              active_experts=active,
                              geometry_experts=experts,
                              is_full_real_bank=len(active) == experts,
                              label=bench.bank_label(len(active), experts))),
              flush=True)
        logical = bench.checkpoint_selected_operands(
            options.snapshot, options.layer, experts, intermediate * tp_degree,
            active, torch, expect_revision=options.expect_revision)
    else:
        logical = bench.LogicalWeights.synthetic(experts, intermediate * tp_degree,
                                                 torch)
    case = bench.build_case(width, intermediate, experts, capacity, 6, logical,
                            torch, tp_degree)
    case.install_rank(0, tp_degree)
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    callable_id = id(case.full)

    full_weights = torch.full((capacity, 6), 1.0 / 6, device="cuda",
                              dtype=torch.float32)

    # Ownership is a function of the LIVE route histogram, so it is recomputed
    # per row count. Scheduling once at capacity and slicing would schedule a
    # histogram the arm never runs.
    ownership = ("greedy LPT, diagnostic projected cost model "
                 "(not production, not the Rust scheduler)" if options.lpt
                 else "diagnostic modulo; not a production scheduler")

    records = []
    # The route table is bound per live row count by `iter_live_rows`, which is the
    # caller-level orchestration. Building it from a `rows` variable leaked out of
    # the operand-union loop above raised NameError for non-checkpoint operands and
    # silently used the last row count for a multi-row request.
    for rows, full_ids_cpu, _fixture_payload in iter_live_rows(options, experts):
        full_ids = full_ids_cpu.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        label = (f"tp{tp_degree}ep{options.ep_degree}-i{intermediate}-w{width}"
                 f"-m{rows}{'-lpt' if options.lpt else ''}")
        live_ids = full_ids[:rows]
        live_weights = full_weights[:rows]
        if options.lpt:
            live_owner, schedule = lpt_owner(
                live_ids, options.ep_degree, 0, experts,
                weight_cost=options.weight_cost, tile_cost=options.tile_cost,
                tile_rows=options.tile_rows)
        else:
            live_owner, schedule = modulo_owner(live_ids, options.ep_degree, 0), {}
        sub_owner = live_owner
        masked = bench.mask_ids(live_ids, sub_owner)
        masked_weights = torch.where(sub_owner, live_weights,
                                     torch.zeros_like(live_weights)).contiguous()
        upload_ids, upload_weights = _padded(masked, masked_weights, capacity, rows)
        # Oracle built from the SAME owner mapping this arm actually runs.
        expected = reference(case.x[:rows], live_ids, masked_weights,
                             local.weights, local.scales)

        # ---- correctness preflight, ONCE, before any timing -------------
        case.out.fill_(float("nan"))
        case.ids.copy_(upload_ids.flatten())
        case.routing.copy_(upload_weights.flatten())
        case.live.fill_(rows)
        case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        assert id(case.full) == callable_id, f"{label}: callable changed"
        plane = case.out[: rows * 6].view(rows, 6, HIDDEN)
        assert bool(torch.isfinite(plane).all()), f"{label}: preflight non-finite"
        actual = plane.sum(1)
        rel = ((actual - expected).norm() / expected.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0).item()
        assert rel < 0.01 and cosine > 0.9999, (label, rel, cosine)
        owned = sub_owner.flatten()
        if not bool(owned.all()):
            assert bool((plane.view(-1, HIDDEN)[~owned] == 0).all()), (
                f"{label}: unowned route not exact zero")
        preflight = {"rel_l2": rel, "cosine": cosine,
                     "owned_routes": int(owned.sum().item()),
                     "active_experts": int(masked[sub_owner].unique().numel())}

        # ---- capture the timed graph ------------------------------------
        for _ in range(3):
            case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())

        # Prove the graph is not a no-op BEFORE trusting a sample. This is the
        # only poison, and it is outside every timing interval.
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        probe = case.out[: rows * 6].view(rows, 6, HIDDEN)
        assert bool(torch.isfinite(probe).all()), f"{label}: replay was a no-op"
        probe_rel = ((probe.sum(1) - expected).norm() / expected.norm()).item()
        assert probe_rel < 0.01, f"{label}: graph replay mismatch {probe_rel}"

        gpu_before = gpu_state()
        warm, amortised, cold = _timed_condition(graph, expected, rows, case,
                                                 options, label)
        tail_rel = _post_timing_check(graph, case, rows, expected, label)
        # Optional blocks below mutate case.ids/routing (per-group masks, stage
        # launches). They run AFTER the arm's own state has been validated, and
        # each restores the arm's live state before returning.
        per_group = {}
        if options.max_group:
            per_group = _per_group_latency(case, rows, options, local, experts,
                                           label, live_ids, live_weights)
        stages = {}
        if options.stages:
            stages = _stage_breakdown(case, rows, options, label)
        arm_state = _snapshot_arm_state(case.ids, case.routing, upload_ids,
                                        upload_weights, rows, capacity)
        gpu_after = gpu_state()

        # Post-timing correctness, evaluated while the state still belongs to
        # THIS arm. `--max-group` and `--stages` mutate case.ids/routing, which is
        # what previously made the tail check compare another group's output
        # against this arm's oracle. The check now runs here, and `tail_post`
        # below re-validates the same graph after the optional blocks restore the
        # arm's own state.
        case.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        tail = case.out[: rows * 6].view(rows, 6, HIDDEN)
        assert bool(torch.isfinite(tail).all()), f"{label}: post-timing non-finite"
        tail_rel = ((tail.sum(1) - expected).norm() / expected.norm()).item()
        assert tail_rel < 0.01, f"{label}: post-timing drift {tail_rel}"

        record = dict(
            label=label, tp_degree=tp_degree, ep_degree=options.ep_degree,
            intermediate=intermediate, kernel_intermediate=case.kernel_intermediate,
            width=width, slices_per_expert=case.slices, experts=experts, rows=rows,
            capacity=capacity, ownership=ownership,
            oracle_dtype=dict(
                oracle="float32",
                compared_actual="float32",
                note="test_v41_expert_numerics.reference is an FP32 torch oracle; "
                     "the arm's FP32 route planes are compared against it",
            ),
            operand_source_mode=logical.source_mode,
            operand_identity=logical.identity,
            device_memory=dict(
                baseline_allocated=torch.cuda.memory_allocated(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                baseline_reserved=torch.cuda.memory_reserved(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                weight_buffers_bytes=sum(
                    b.numel() * b.element_size()
                    for b in case._weight_buffers),
                operand_hashed_active_bytes=logical.identity.get(
                    "hashed_active_bytes"),
                operand_full_dense_bank_bytes=logical.identity.get(
                    "full_dense_bank_bytes"),
                note="torch allocator counters only; this is not process VRAM "
                     "or the CUDA context"),
            group_loads=schedule.get("group_loads"),
            group_assignment=schedule.get("assignment"),
            schedule_cost_model=schedule.get("cost_model"),
            rows_per_expert=schedule.get("rows_per_expert"),
            schedule_note=schedule.get("histogram_transfer"),
            schedule_tie_break=schedule.get("tie_break"),
            preflight=preflight, warm=warm, amortised=amortised, cold=cold,
            per_group=per_group, stages=stages,
            post_timing_rel_l2=tail_rel, graph_verified_not_noop=True,
            poisoned_only_in_preflight=True,
            compiled_callable_unchanged=True,
            gpu_before=gpu_before, gpu_after=gpu_after,
            note=("rank-local critical path; no cross-rank sum and no transport "
                  "latency is claimed"),
        )
        records.append(record)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("warm", "cold")}), flush=True)
        print(json.dumps(dict(label=label, condition="warm",
                              median_us=warm["median_us"], p10_us=warm["p10_us"],
                              p90_us=warm["p90_us"], n=warm["n"])), flush=True)
        if amortised:
            print(json.dumps(dict(label=label, condition="amortised",
                                  median_us=amortised["median_us"],
                                  p10_us=amortised["p10_us"],
                                  p90_us=amortised["p90_us"], n=amortised["n"],
                                  note="N direct launches captured in one outer "
                                       "graph, replayed once; no host gap "
                                       "between the inner launches")),
                  flush=True)
        if cold:
            print(json.dumps(dict(label=label, condition="cold",
                                  median_us=cold["median_us"],
                                  p10_us=cold["p10_us"],
                                  p90_us=cold["p90_us"], n=cold["n"])), flush=True)
        del graph, probe, tail
        torch.cuda.synchronize()

    del case, logical
    torch.cuda.synchronize()
    return records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topologies", default="tp2,tp3,tp4",
                        help="comma list of tp2,tp3,tp4,tp6 (each implies its "
                             "own per-rank intermediate: 1152 / 768 / 576 / 384; "
                             "tp6 is the pure six-way split of the official 2304)")
    parser.add_argument("--ep-degree", type=int, choices=(1, 2, 3), default=1,
                        help="ownership groups; 1 = this rank owns everything")
    parser.add_argument("--widths", default="192",
                        help="comma list of intermediate slice widths; each must "
                             "be 64/128/192 and divide the topology's per-rank "
                             "intermediate exactly (tp6 384 = 2*192 = 3*128 = 6*64)")
    parser.add_argument("--rows", default="1,2,8,16,80")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--operands", choices=("synthetic", "checkpoint"),
                        default="synthetic",
                        help="synthetic pseudo-random operands, or REAL weights "
                             "for the active experts read from a checkpoint")
    parser.add_argument("--snapshot", type=Path,
                        help="official snapshot directory (required for "
                             "--operands checkpoint)")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--route-fixture", type=Path, default=None,
                        help="validated route fixture JSON; replaces the "
                             "all-distinct synthetic route table")
    parser.add_argument("--expect-revision", default=None,
                        help="fail if the snapshot revision differs")
    parser.add_argument("--capacity", type=int, default=80)
    parser.add_argument("--samples", type=int, default=30,
                        help="timing intervals per condition (>=30 for a claim)")
    parser.add_argument("--replays", type=int, default=10,
                        help="replays per warm interval")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cold", action="store_true",
                        help="also measure the cold condition")
    parser.add_argument("--flush-bytes", type=int, default=256 << 20,
                        help="must exceed the device L2 size; the harness reads "
                             "L2_cache_size from the device at runtime and "
                             "refuses a flush that does not exceed it. Do not "
                             "assume a particular L2 size.")
    parser.add_argument("--max-group", action="store_true",
                        help="measure every EP group's own mask and report the "
                             "predicted maximum, sequentially")
    parser.add_argument("--amortised", action="store_true",
                        help="also time N inner replays inside one outer graph, "
                             "removing host submission gaps")
    parser.add_argument("--stages", action="store_true",
                        help="also time plan/compute/reduce in isolation")
    parser.add_argument("--lpt", action="store_true",
                        help="use greedy LPT ownership instead of modulo")
    parser.add_argument("--weight-cost", type=float, default=1.0,
                        help="uniform cost per ACTIVE expert (charged once, not "
                             "per row); matches ds41rt-core expert_weight_cost")
    parser.add_argument("--tile-cost", type=float, default=0.0,
                        help="cost per partially filled tile_rows work tile")
    parser.add_argument("--tile-rows", type=int, default=16,
                        help="rows per work tile for the LPT tile term")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.lpt and options.weight_cost == 0 and options.tile_cost == 0:
        parser.error("the LPT cost model needs a positive weight_cost or "
                     "tile_cost; both zero is degenerate")
    if options.tile_cost < 0 or options.weight_cost < 0:
        parser.error("cost model terms must be non-negative")
    options.widths = [int(x) for x in options.widths.split(",") if x]
    options.rows = [int(x) for x in options.rows.split(",") if x]
    if not options.rows:
        parser.error("--rows must name at least one positive row count")
    if any(r <= 0 for r in options.rows):
        parser.error(f"--rows must be positive, got {options.rows}")
    if len(options.rows) != 1:
        parser.error(
            f"exactly one row count is required per invocation, got "
            f"{options.rows}; the route table is bound per live row count")
    if max(options.rows) > options.capacity:
        parser.error("rows must not exceed capacity")
    if options.operands == "checkpoint" and options.snapshot is None:
        parser.error("--operands checkpoint requires --snapshot")
    # Width must be a real export tile (the slice kernel asserts 64/128/192) and
    # must tile the topology's per-rank intermediate exactly, so a request that
    # would need storage padding or an extra partial slice fails before any GPU
    # work. TP6's 384 is 2*192, 3*128 and 6*64 -- all three tile exactly.
    wanted = {t for t in options.topologies.split(",") if t}
    if not wanted or not any(g[2] in wanted for g in TP_GEOMETRIES):
        parser.error(f"no topology selected by --topologies {options.topologies!r}")
    for width in options.widths:
        if width not in SUPPORTED_SLICE_WIDTHS:
            parser.error(
                f"--widths must be {SUPPORTED_SLICE_WIDTHS}, got {width}")
    for tp_degree, intermediate, tag in TP_GEOMETRIES:
        if tag not in wanted:
            continue
        for width in options.widths:
            if intermediate % width:
                parser.error(
                    f"--widths {width} does not tile {tag} per-rank intermediate "
                    f"{intermediate} exactly (slices would leave a partial tile)")
    return options


def main(argv=None):
    options = parse_args(argv)
    _require_blackwell()
    props = torch.cuda.get_device_properties(0)
    l2 = props.L2_cache_size
    if options.cold and options.flush_bytes <= l2:
        raise SystemExit(f"flush {options.flush_bytes} must exceed L2 {l2}")
    def _sha256_file(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    provenance = dict(
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        harness_path=str(Path(__file__).resolve()),
        harness_sha256=_sha256_file(__file__),
        benchmark_sha256=_sha256_file(
            Path(__file__).resolve().parent / "benchmark_v41_ep_groups.py"),
        route_fixture=(str(options.route_fixture)
                       if options.route_fixture else None),
        route_fixture_sha256=(_sha256_file(options.route_fixture)
                              if options.route_fixture else None),
        input_dtype=dict(
            source_activation="bfloat16",
            kernel_wire_values="uint32 (row-packed FP8 E4M3, 5280-byte rows)",
            kernel_wire_scales="uint8 (UE8M0 K/32)",
            note="the timed kernel consumes the wire form; source_activation is "
                 "pre-quantization",
        ),
        output_dtype=dict(
            route_planes="float32",
            reduced_output="bfloat16",
            note="FP32 route planes are produced by the expert stage; the "
                 "reduction rounds once to BF16",
        ),
        host=platform.node(), python=sys.version.split()[0],
        torch=torch.__version__, device=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        sm_count=props.multi_processor_count, l2_bytes=l2,
        host_state=host_state(), gpu_state=gpu_state(),
        samples=options.samples, replays_per_warm_interval=options.replays,
        warmup=options.warmup, cold=options.cold,
        flush_bytes=options.flush_bytes,
        operands=options.operands,
        snapshot=(str(options.snapshot) if options.snapshot else None),
        layer=(options.layer if options.operands == "checkpoint" else None),
        lpt=options.lpt, weight_cost=options.weight_cost,
        tile_cost=options.tile_cost, tile_rows=options.tile_rows,
        scope=["rank-local critical path only",
               "parallel TP ranks are never summed",
               "no cross-Spark or transport latency claim",
               "poison only in correctness preflight, never in a timing interval",
               "cold = flush > L2 then exactly one timed replay",
               "LPT group loads are a projected cost model, not a measured "
               "scheduler and not the Rust implementation",
               "per-group maxima are measured sequentially and are predictions",
               "checkpoint operands are an ACTIVE-FILLED dense layout: real "
               "weights only for the active experts, unused slots zeroed; this "
               "is not a full real expert bank"],
    )
    # One shard geometry per topology: TP degree fixes how wide this rank's
    # slice is, so intermediate comes with it. TP6 is the pure six-way split of
    # the official 2304 and is the only geometry whose per-rank intermediate is
    # 384 (no 128-alignment padding).
    wanted = {t for t in options.topologies.split(",") if t}
    geometries = [g for g in TP_GEOMETRIES if g[2] in wanted]
    records = []
    for tp_degree, intermediate, tag in geometries:
        for width in options.widths:
            print(json.dumps(dict(starting=tag, tp_degree=tp_degree,
                                  intermediate=intermediate, width=width)),
                  flush=True)
            records.extend(measure_group(options, tp_degree, intermediate,
                                         width, options.experts))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    # One width and therefore exactly one record per file: the consolidation
    # pairs arms by a filename that encodes a single width, and it refuses a file
    # whose record count is not one. Fail closed here rather than emit a file the
    # consolidator would reject (or, under -O, silently reduce to records[0]).
    if len(options.widths) != 1:
        raise SystemExit(
            f"exactly one width per invocation is required (got {options.widths}); "
            "the consolidation pairs arms by a per-width filename")
    if len(records) != 1:
        raise SystemExit(
            f"expected exactly one record, got {len(records)}; one arm per file")
    options.output.write_text(json.dumps(
        {"provenance": provenance, "records": records}, indent=1) + "\n")
    print(json.dumps({"written": str(options.output), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
