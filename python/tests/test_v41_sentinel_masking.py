"""Minimal TP x EP sentinel smoke for the official V4.1 grouped slice pipeline.

Priority-one smoke: ONE physical rank of a TP layout, per-rank intermediate
576/768/1152, canonical distinct top-6 over at least six experts, M1. It answers
the only question that gates everything else: does a masked, rank-sliced forward
execute and match the oracle?

Correctness rules applied here and in the wider matrix to follow:

* A single physical rank is compared against the oracle built from ITS OWN
  sliced logical weights. Comparing one rank against the full-width oracle would
  be wrong: the full output only appears after summing every rank.
* Cross-topology comparison (TP4 vs TP2xEP2 vs TP3xEP2) is tolerance-based and
  never bit-exact, because the per-rank intermediate width changes the FP8
  intermediate quantization granularity.
* Masking uses sentinel id 384 with weight exactly 0, so an unowned route
  produces no work group and its output must be an exact zero.

No path manipulation happens here: the verified runner
(``scripts/run-tp-ep-kernel-checks.sh``) sets and verifies PYTHONPATH, including
the pinned SparkInfer tree. If a test is run outside that runner the import
provenance check is bypassed, so run it through the runner.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("ds41rt_ep_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()
HIDDEN = bench.HIDDEN
REL_TOL, COS_TOL = 0.01, 0.9999


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")


def _assemble(plane, rows, topk):
    return plane.view(-1, HIDDEN)[: rows * topk].view(rows, topk, HIDDEN).sum(1)


def _compare(actual, expected, label):
    denom = float(expected.norm())
    if denom == 0.0:
        assert bool((actual == 0).all()), f"{label}: expected exact zero"
        return 0.0, 1.0
    rel = ((actual - expected).norm() / expected.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0
    ).item()
    assert rel < REL_TOL and cosine > COS_TOL, (label, rel, cosine)
    return rel, cosine


@pytest.mark.parametrize(
    "tp_degree,intermediate",
    [(2, 1152), (4, 576), (3, 768)],
    ids=["tp2-i1152", "tp4-i576", "tp3-i768"],
)
def test_rank_local_forward_matches_local_oracle(tp_degree, intermediate):
    """One physical rank compared against its own sliced-weight oracle."""
    _require_blackwell()
    from tests.moe.test_v41_expert_numerics import reference

    width, experts, capacity, topk = 192, 6, 1, 6
    x = torch.randn(capacity, HIDDEN, device="cuda").mul_(0.5).bfloat16()
    logical = bench.LogicalWeights.synthetic(experts, intermediate * tp_degree, torch)
    base = (torch.arange(capacity * topk, device="cuda", dtype=torch.int32)
            .reshape(capacity, topk) % experts)
    weights = torch.full((capacity, topk), 1.0 / topk, device="cuda",
                         dtype=torch.float32)

    case = bench.build_case(width, intermediate, experts, capacity, topk,
                            logical, torch, tp_degree)
    case.x = x
    case._quantize_input()
    hashes = case.install_rank(0, tp_degree)

    # Real memory numbers, not estimates. Baseline is measured after every
    # allocation the case needs; peak covers the forward.
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    # Rank-local oracle: the same slice this rank actually computes.
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    expected = reference(x, base, weights, local.weights, local.scales)
    plane = case.run(capacity, base, weights)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    rel, cosine = _compare(_assemble(plane, capacity, topk), expected,
                           ("rank_local", tp_degree, intermediate))
    print(json.dumps(dict(
        case="rank_local_forward", tp_degree=tp_degree, tp_rank=0,
        intermediate=intermediate, kernel_intermediate=case.kernel_intermediate,
        slices_per_expert=case.slices, experts=experts, topk=topk, rows=capacity,
        rel_l2=rel, cosine=cosine,
        device_memory=dict(
            baseline_allocated=baseline_allocated,
            baseline_reserved=baseline_reserved,
            peak_allocated=peak_allocated,
            peak_reserved=peak_reserved,
            weight_buffers_bytes=sum(
                b.numel() * b.element_size() for b in case._weight_buffers),
        ),
        allocation=case.allocation,
        weight_hashes=hashes)), flush=True)

    # Masking: this rank owns only its group's experts. The oracle is always fed
    # the ORIGINAL ids -- the fixture oracle cannot index the 384 sentinel -- with
    # the masked weights; unowned routes carry weight exactly 0 so they
    # contribute nothing. The kernel keeps the sentinel ids, because that is what
    # makes it emit no work group for an unowned expert.
    owner = bench.owners_for(base, tp_degree, 0)
    masked = bench.mask_ids(base, owner)
    weights_arm = torch.where(owner, weights, torch.zeros_like(weights)).contiguous()
    expected_masked = reference(x, base, weights_arm, local.weights, local.scales)
    case.out.fill_(12345)
    masked_plane = case.run(capacity, masked, weights_arm)
    rel_m, cos_m = _compare(_assemble(masked_plane, capacity, topk),
                            expected_masked, ("masked_local", tp_degree))
    flat = masked_plane.view(-1, HIDDEN)
    owned = owner.flatten()
    if bool(owned.any()):
        assert bool((flat[owned] != 0).any()), "owned routes produced nothing"
    if not owner.all():
        assert bool((flat[~owned] == 0).all()), "unowned route is not an exact zero"
    print(json.dumps(dict(
        case="rank_local_masked", tp_degree=tp_degree,
        owned_routes=int(owner.sum().item()),
        active_experts=int(masked[owner].unique().numel()),
        rel_l2=rel_m, cosine=cos_m)), flush=True)
