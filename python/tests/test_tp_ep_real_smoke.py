"""Targeted real-checkpoint smoke at a chosen layer, per TP rank.

Correctness gate proving REAL official-checkpoint bytes survive the per-rank pack
+ kernel path. The harness's own driver only exercises TP rank 0, while packing is
per-shard, so each shard is compared against the oracle built from that same
shard's sliced real weights.

Two gates, deliberately separate:

* **Unmasked**: every route must name a LOADED expert, so the result is real-weight
  evidence. This is asserted on CPU before any device work; a route to an unloaded
  slot would compare zeros against zeros and prove nothing.
* **Masked**: a rank owns a subset; unowned routes must be exactly zero and the
  owned routes must still match the oracle. The oracle keeps the ORIGINAL ids (the
  fixture oracle cannot index the 384 sentinel) with masked weights.

NaN is diagnosed, never just reported: a zero denominator and a non-finite tensor
are distinguished, and both the actual and the oracle side are characterised.

Run:
    python -m pytest python/tests/test_tp_ep_real_smoke.py -q -s
env: DS41RT_TPEP_SMOKE_LAYER, DS41RT_TPEP_SNAPSHOT, DS41RT_TPEP_TARGETS,
     DS41RT_TPEP_ACTIVE (comma ids)
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py"

REL_TOL, COS_TOL = 0.01, 0.9999

LAYER = int(os.environ.get("DS41RT_TPEP_SMOKE_LAYER", "20"))
SNAPSHOT = os.environ.get("DS41RT_TPEP_SNAPSHOT")
TARGETS = os.environ.get("DS41RT_TPEP_TARGETS", "2:0,2:1,4:0")
ACTIVE = [int(x) for x in os.environ.get(
    "DS41RT_TPEP_ACTIVE", "1,4,7,11,19,23").split(",") if x]


def _load_bench():
    spec = importlib.util.spec_from_file_location("ds41rt_ep_bench_smoke", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()
HIDDEN = bench.HIDDEN

PER_RANK_INTERMEDIATE = {2: 1152, 3: 768, 4: 576}


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x required")


def _snapshot():
    if SNAPSHOT:
        return Path(SNAPSHOT)
    hub = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    return (hub / "hub" / "models--deepseek-ai--DeepSeek-V4.1-Flash" / "snapshots"
            / "dba1be0a40aa45a94ad051997016db3960a90277")


def _describe(name, tensor):
    """Characterise a tensor so a NaN names its own source."""
    finite = bool(torch.isfinite(tensor).all())
    return {
        f"{name}_finite": finite,
        f"{name}_has_nan": bool(torch.isnan(tensor).any()),
        f"{name}_has_inf": bool(torch.isinf(tensor).any()),
        f"{name}_norm": float(tensor.norm()) if finite else None,
        f"{name}_absmax": float(tensor.abs().max()) if finite else None,
    }


def _routes_to_loaded(capacity, topk, active, experts):
    """Canonical distinct top-k that only ever names LOADED experts.

    Every route must hit a real weight, otherwise the comparison degrades into
    zeros against zeros. This is the fix for the earlier smoke, which routed
    `% experts` over all 384 slots while only 6 were loaded.
    """
    assert len(active) >= topk, f"need >= {topk} loaded experts, have {len(active)}"
    ids = torch.zeros((capacity, topk), dtype=torch.int32)
    for row in range(capacity):
        for slot in range(topk):
            ids[row, slot] = active[(row + slot) % len(active)]
    return ids


def test_active_ids_are_a_valid_route_universe():
    """CPU-only preflight: ids in range, distinct within a token, no duplicates."""
    assert ACTIVE, "no active ids configured"
    assert len(set(ACTIVE)) == len(ACTIVE), f"duplicate active ids: {ACTIVE}"
    assert all(0 <= e < 384 for e in ACTIVE), f"active id out of range: {ACTIVE}"
    ids = _routes_to_loaded(2, 6, ACTIVE, 384)
    for row in ids.tolist():
        assert len(set(row)) == 6, f"row is not distinct top-6: {row}"
        assert set(row) <= set(ACTIVE), f"row routes to an unloaded expert: {row}"


@pytest.mark.parametrize("target", TARGETS.split(","), ids=TARGETS.split(","))
def test_real_checkpoint_rank_smoke(target):
    """One physical rank of one topology, real layer-L weights, M1."""
    _require_gpu()
    from tests.moe.test_v41_expert_numerics import reference

    tp_degree, tp_rank = (int(x) for x in target.split(":"))
    intermediate = PER_RANK_INTERMEDIATE[tp_degree]
    full_width = intermediate * tp_degree
    snapshot = _snapshot()
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip(f"official snapshot not present at {snapshot}")

    experts, capacity, topk, width = 384, 1, 6, 192
    # CPU gate before any device work: all routes must name loaded experts.
    base = _routes_to_loaded(capacity, topk, ACTIVE, experts)
    assert set(base.flatten().tolist()) <= set(ACTIVE)
    assert len(set(base.flatten().tolist())) >= 1
    # Every tested owned arm must contain at least one real (loaded) route, so a
    # pass cannot come from comparing zeros with zeros.
    assert len(set(base.flatten().tolist()) & set(ACTIVE)) >= 1

    logical = bench.checkpoint_selected_operands(
        snapshot, LAYER, experts, full_width, ACTIVE, torch,
        expect_revision=snapshot.name)
    local = bench.rank_logical_slice(logical, tp_rank, tp_degree, intermediate,
                                     torch)
    # The fixture oracle runs on the device, so the rank-local logical operands it
    # reads must be on the device too (rank_logical_slice returns the bank's
    # slices, which live where the bank lives).
    device = torch.device("cuda", torch.cuda.current_device())
    local.weights = {k: v.to(device) for k, v in local.weights.items()}
    local.scales = {k: v.to(device) for k, v in local.scales.items()}
    case = bench.build_case(width, intermediate, experts, capacity, topk, logical,
                            torch, tp_degree)
    hashes = case.install_rank(tp_rank, tp_degree)

    weights = torch.full((capacity, topk), 1.0 / topk, device="cuda",
                         dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()

    expected = reference(case.x, base, weights, local.weights, local.scales)
    assert bool(torch.isfinite(expected).all()), (
        "oracle itself is non-finite; fix the operand/oracle path before timing")
    assert float(expected.norm()) > 0, "oracle norm is zero"

    case.out.fill_(float("nan"))
    plane = case.run(capacity, base, weights)
    peak_bytes = torch.cuda.max_memory_allocated()
    got = plane.sum(1)

    report = dict(
        case="real_checkpoint_rank_smoke", layer=LAYER, tp_degree=tp_degree,
        tp_rank=tp_rank, per_rank_intermediate=intermediate,
        full_width=full_width, width=width, active_experts=ACTIVE,
        routes_to_loaded_only=True,
        snapshot_revision=snapshot.name,
        operands_sha256=logical.identity["operands_sha256"],
        hashed_active_bytes=logical.identity["hashed_active_bytes"],
        full_dense_bank_bytes=logical.identity["full_dense_bank_bytes"],
        unused_slots_zeroed=logical.identity["unused_slots_zeroed"],
        weight_hashes=hashes,
        baseline_allocated=baseline_bytes, peak_allocated=peak_bytes,
        memory_note="torch allocator counters only, not process VRAM",
    )
    report.update(_describe("actual", got))
    report.update(_describe("oracle", expected))
    rel = ((got - expected).norm() / expected.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        got.flatten(), expected.flatten(), dim=0).item()
    report["rel_l2"] = rel
    report["cosine"] = cosine
    print(json.dumps(report), flush=True)
    assert bool(report["actual_finite"]), f"{target}: actual non-finite"
    assert rel < REL_TOL and cosine > COS_TOL, (target, rel, cosine)

    # Masked: this rank owns the experts where expert % tp_degree == tp_rank.
    # `base` is built on CPU (it is a route table, not device data), so the owner
    # mask must be moved to the weights' device before any torch.where against
    # them.
    owner = ((base % tp_degree) == tp_rank).to(weights.device)
    owned_ids = sorted(set(base[owner.cpu()].tolist()))
    unowned = ~owner
    print(json.dumps(dict(
        case="masked_gate", target=target, tp_degree=tp_degree, tp_rank=tp_rank,
        owned_ids=owned_ids, owned_routes=int(owner.sum().item()),
        unowned_routes=int(unowned.sum().item()),
        owned_nonzero_active=len(owned_ids) >= 1)), flush=True)
    if not owned_ids:
        pytest.skip(f"{target}: rank owns no loaded expert, masked arm vacuous")

    masked = bench.mask_ids(base, owner.cpu())
    masked_weights = torch.where(owner, weights, torch.zeros_like(weights)).contiguous()
    masked_expected = reference(case.x, base, masked_weights, local.weights,
                                local.scales)
    diag = _describe("masked_oracle", masked_expected)
    print(json.dumps(dict(case="masked_oracle_diag", target=target, **diag)),
          flush=True)
    assert bool(diag["masked_oracle_finite"]), (
        f"{target}: oracle non-finite on the masked arm -> "
        f"nan={diag['masked_oracle_has_nan']} inf={diag['masked_oracle_has_inf']}")
    oracle_norm = float(masked_expected.norm())
    assert oracle_norm > 0, f"{target}: masked oracle norm is zero (no real work)"

    case.out.fill_(float("nan"))
    masked_plane = case.run(capacity, masked, masked_weights)
    got_m = masked_plane.sum(1)
    mdiag = _describe("masked_actual", got_m)
    rel_m = ((got_m - masked_expected).norm() / oracle_norm).item()
    cos_m = torch.nn.functional.cosine_similarity(
        got_m.flatten(), masked_expected.flatten(), dim=0).item()
    print(json.dumps(dict(case="masked_result", target=target, **mdiag,
                          rel_l2=rel_m, cosine=cos_m,
                          oracle_norm=oracle_norm)), flush=True)
    assert bool(mdiag["masked_actual_finite"]), (
        f"{target}: actual non-finite on the masked arm -> "
        f"nan={mdiag['masked_actual_has_nan']} inf={mdiag['masked_actual_has_inf']}")
    assert rel_m < REL_TOL and cos_m > COS_TOL, (target, rel_m, cos_m)
    flat = masked_plane.view(-1, HIDDEN)
    assert bool((flat[unowned.flatten()] == 0).all()), (
        f"{target}: unowned route is not an exact zero")

    # Explicit zero-denominator gate: a rank that owns NOTHING must produce an
    # exact zero, never a 0/0 relative error that could be mistaken for a pass.
    # mask_ids indexes the CPU route table, so the mask stays on CPU; the zeroed
    # weights are what must live on the device.
    none_owner = torch.zeros_like(base, dtype=torch.bool)
    none_weights = torch.zeros_like(weights, device=weights.device)
    none_expected = reference(case.x, base, none_weights, local.weights,
                              local.scales)
    none_norm = float(none_expected.norm())
    case.out.fill_(float("nan"))
    none_plane = case.run(capacity, bench.mask_ids(base, none_owner),
                          none_weights)
    none_flat = none_plane.view(-1, HIDDEN)
    none_zero = bool((none_plane == 0).all())
    print(json.dumps(dict(
        case="all_unowned_zero_gate", target=target,
        oracle_norm=none_norm, oracle_finite=bool(torch.isfinite(none_expected).all()),
        actual_all_exact_zero=none_zero,
        note="all-unowned must be an exact zero; a 0/0 rel is never a pass")),
        flush=True)
    assert none_norm == 0.0, (
        f"{target}: all-unowned oracle norm {none_norm} should be exactly zero")
    assert none_zero, (
        f"{target}: all-unowned arm is not an exact zero; "
        f"nan={bool(torch.isnan(none_flat).any())} "
        f"inf={bool(torch.isinf(none_flat).any())}")
