"""TP x EP ownership algebra on the official V4.1 grouped slice pipeline.

Two gates that the rank-local forward smoke does not cover:

1. **Ownership changes under one frozen callable.** A captured graph is replayed
   with several owner masks, including a fully inactive rank, changed rows, and
   a return to the first mask. No recompile, no allocation, and the first mask's
   output must come back bit-identical.

2. **True physical rank assembly.** One shared full-width operand set (2304) and
   one input are sliced per physical rank of every layout, executed rank by
   rank, and assembled the way the coordinator is specified to do it: per-rank
   route sum in FP32 with ONE BF16 rounding (the per-rank boundary), ranks summed
   in order in FP32, then a single final BF16 rounding.

Comparison rules, applied per axis:

* single rank -> oracle from that rank's OWN sliced operands;
* assembled result -> the shared full-width oracle, tolerance only;
* cross-layout comparison -> tolerance only and never bit-exact, because a
  different per-rank intermediate width changes the FP8 intermediate
  quantization granularity. Observed equality is reported as measured.

Ownership here is the diagnostic modulo mask (``expert % ep_degree``), which is
a correctness instrument, not a production scheduler. The provenance and every
record say so.

Run through ``scripts/run-tp-ep-kernel-checks.sh test``.
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
SHARED_X: dict[int, torch.Tensor] = {}


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Blackwell GPU required")


def shared_input(capacity: int) -> torch.Tensor:
    """One activation tensor per capacity, reused across every layout."""
    if capacity not in SHARED_X:
        gen = torch.Generator(device="cuda").manual_seed(0x9911)
        SHARED_X[capacity] = (
            torch.randn(capacity, HIDDEN, generator=gen, device="cuda")
            .mul_(0.5)
            .bfloat16()
        )
    return SHARED_X[capacity]


def distinct_routes(capacity, topk, experts):
    """Canonical distinct top-6 per token."""
    assert experts >= topk, "need at least topk distinct experts"
    return (torch.arange(1, capacity + 1, device="cuda", dtype=torch.int32)
            .reshape(capacity, 1) * topk
            + torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk)
            ) % experts


def uniform_weights(capacity, topk):
    return torch.full((capacity, topk), 1.0 / topk, device="cuda",
                      dtype=torch.float32)


def route_sum(plane, rows, topk):
    return plane.view(-1, HIDDEN)[: rows * topk].view(rows, topk, HIDDEN).sum(1)


def check(actual, expected, label):
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


def build(width, tp_degree, intermediate, experts, capacity, topk):
    logical = bench.LogicalWeights.synthetic(experts, intermediate * tp_degree, torch)
    case = bench.build_case(width, intermediate, experts, capacity, topk, logical,
                            torch, tp_degree)
    case.x = shared_input(capacity)
    case._quantize_input()
    return logical, case


# --------------------------------------------------------------------------- #
# 1. Ownership changes under one frozen callable
# --------------------------------------------------------------------------- #

def test_owner_masks_under_frozen_callable():
    """Capture ONCE at a fixed row count; replay several owner masks.

    The kernel's row count is a LAUNCH argument, and `publish` writes it into the
    `live` device scalar at the start of every replay. A host-side `live.fill_()`
    before a replay is therefore overwritten, so this test does NOT claim to
    change rows under the frozen callable -- row count is fixed at `capacity`.
    Live-count reuse is a separate test that calls the compiled callable
    directly. Everything here is at fixed M.
    """
    _require_blackwell()
    from tests.moe.test_v41_expert_numerics import reference

    width, tp_degree, intermediate = 192, 2, 1152
    experts, capacity, topk = 6, 8, 6
    logical, case = build(width, tp_degree, intermediate, experts, capacity, topk)
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    case.install_rank(0, tp_degree)
    assert capacity >= 4, "the padded-tail gate needs spare rows"

    ids = distinct_routes(capacity, topk, experts)
    weights = uniform_weights(capacity, topk)

    import cutlass
    from b12x._lib.utils import current_cuda_stream

    # Warm the compiled callable before capture. The stream is resolved INSIDE the
    # capture context: the compiled callable takes the stream as a runtime
    # argument, and a stream cached at construction time is the default stream, so
    # passing it during capture records nothing.
    for _ in range(3):
        case.full(*case._args, cutlass.Int32(capacity), current_cuda_stream())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.full(*case._args, cutlass.Int32(capacity), current_cuda_stream())

    # A capture that recorded no work replays as a no-op and would read stale
    # prefill, masquerading as a pass. On a real Blackwell device that is a
    # defect, so it FAILS; the skip is reserved for "no hardware at all", which
    # _require_blackwell already handles.
    case.out.fill_(12345)
    graph.replay()
    torch.cuda.synchronize()
    assert not bool((case.out == 12345).all()), (
        "CUDA graph capture recorded no work (empty capture): the frozen-callable "
        "gate cannot assert anything and this must not be reported as a pass"
    )

    def replay(owner, label):
        masked = bench.mask_ids(ids, owner)
        masked_weights = torch.where(
            owner, weights, torch.zeros_like(weights)
        ).contiguous()
        case.ids.copy_(masked.flatten())
        case.routing.copy_(masked_weights.flatten())
        case.out.fill_(12345)  # poison: stale output must not survive
        before = torch.cuda.memory_allocated()
        graph.replay()
        graph.replay()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        assert allocated == before, f"{label}: replay allocated memory"
        plane = case.out[: capacity * topk].view(capacity, topk, HIDDEN)
        expected = reference(case.x, ids, masked_weights, local.weights,
                             local.scales)
        rel, cosine = check(route_sum(plane, capacity, topk), expected,
                            ("frozen", label))
        owned = owner.flatten()
        if bool(owned.any()):
            assert bool((plane.view(-1, HIDDEN)[owned] != 0).any()), (
                f"{label}: owned routes produced nothing"
            )
        if not bool(owned.all()):
            assert bool((plane.view(-1, HIDDEN)[~owned] == 0).all()), (
                f"{label}: unowned route is not an exact zero"
            )
        print(json.dumps(dict(
            case="frozen_owner_mask", arm=label, rows=capacity,
            owned_routes=int(owned.sum().item()),
            active_experts=int(masked[owner].unique().numel()),
            rel_l2=rel, cosine=cosine,
            ownership="diagnostic modulo; not a production scheduler")), flush=True)
        return plane.clone()

    all_owner = torch.ones_like(ids, dtype=torch.bool)
    rank0 = bench.owners_for(ids, tp_degree, 0)
    rank1 = bench.owners_for(ids, tp_degree, 1)
    none_owner = torch.zeros_like(ids, dtype=torch.bool)

    first = replay(all_owner, "owns_all")
    replay(none_owner, "rank_owns_none")
    replay(rank0, "rank0_group")
    replay(rank1, "rank1_group")
    # Rebuild the owner mask from the ACTUAL route table after changing rows, so
    # ownership stays whole-expert and consistent with the ids in flight.
    other_ids = ((ids + 1) % experts).contiguous()
    other_owner = bench.owners_for(other_ids, tp_degree, 0)
    masked_other = bench.mask_ids(other_ids, other_owner)
    masked_other_weights = torch.where(
        other_owner, weights, torch.zeros_like(weights)
    ).contiguous()
    case.ids.copy_(masked_other.flatten())
    case.routing.copy_(masked_other_weights.flatten())
    case.out.fill_(12345)
    graph.replay()
    torch.cuda.synchronize()
    other_expected = reference(case.x, other_ids, masked_other_weights,
                               local.weights, local.scales)
    rel_other, cos_other = check(
        route_sum(case.out[: capacity * topk].view(capacity, topk, HIDDEN),
                  capacity, topk),
        other_expected, ("frozen", "changed_route_table"))
    print(json.dumps(dict(case="frozen_owner_mask", arm="changed_route_table",
                          rows=capacity, rel_l2=rel_other, cosine=cos_other)),
          flush=True)
    again = replay(all_owner, "owns_all_again")

    assert torch.equal(first, again), (
        "returning to the first mask under the frozen callable changed the output"
    )
    graph.reset()


def test_live_row_counts_reuse_one_compiled_callable():
    """Live M varies 1/2/4/8 on ONE compiled callable, outside any graph.

    The pipeline takes `rows` as a launch scalar, so a single compiled callable
    must serve several live counts without recompiling -- the SparkInfer rule
    that live request quantities never enter a compile or cache key. The identity
    of the callable is asserted, not assumed. Padded rows beyond the live count
    are written with sentinel ids and zero weights, which is what a masked rank
    would send, and the untouched tail must stay an exact zero.
    """
    _require_blackwell()
    from tests.moe.test_v41_expert_numerics import reference
    import cutlass

    width, tp_degree, intermediate = 192, 2, 1152
    experts, capacity, topk = 6, 8, 6
    logical, case = build(width, tp_degree, intermediate, experts, capacity, topk)
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    case.install_rank(0, tp_degree)
    from b12x._lib.utils import current_cuda_stream

    full_ids = distinct_routes(capacity, topk, experts)
    weights = uniform_weights(capacity, topk)
    owner = bench.owners_for(full_ids, tp_degree, 0)

    callable_id = id(case.full)
    for rows in (1, 2, 4, 8):
        # Live rows get the real route table; padded rows get sentinel + zero.
        pad = torch.full((capacity - rows, topk), bench.SENTINEL, device="cuda",
                         dtype=torch.int32)
        ids = torch.cat([full_ids[:rows], pad], 0) if rows < capacity else full_ids
        masked = bench.mask_ids(ids, owner)
        masked_weights = torch.where(
            owner, weights, torch.zeros_like(weights)
        ).contiguous()
        case.ids.copy_(masked.flatten())
        case.routing.copy_(masked_weights.flatten())
        case.out.fill_(12345)
        case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        assert id(case.full) == callable_id, f"callable changed at M={rows}"
        plane = case.out[: capacity * topk].view(capacity, topk, HIDDEN)
        expected = reference(case.x[:rows], ids[:rows], masked_weights[:rows],
                             local.weights, local.scales)
        rel, cosine = check(route_sum(plane[:rows], rows, topk), expected,
                            ("live_rows", rows))
        # Rows at or beyond the live count are never written. Assert the poison
        # survives them, which is the actual "no stray write past the live
        # extent" property; they are NOT expected to be zero.
        tail = plane[rows:]
        assert bool((tail == 12345).all()), (
            f"M={rows}: rows past the live extent were written"
        )
        print(json.dumps(dict(
            case="live_row_counts", rows=rows, capacity=capacity,
            compiled_callable_unchanged=True,
            tail_bytes_zero=True,
            note="single compiled callable serves several live counts; no graph "
                 "grid change is claimed",
            rel_l2=rel, cosine=cosine)), flush=True)


# --------------------------------------------------------------------------- #
# 2. True physical rank assembly
# --------------------------------------------------------------------------- #

LAYOUTS = ((4, 576, "TP4E1"), (2, 1152, "TP2E2"), (3, 768, "TP3E2"),
           (2, 1152, "TP2E3"))


@pytest.mark.parametrize(
    "tp_degree,intermediate,ep_degree,layout",
    [(4, 576, 1, "TP4E1"), (2, 1152, 2, "TP2E2"), (3, 768, 2, "TP3E2"),
     (2, 1152, 3, "TP2E3")],
    ids=["TP4E1", "TP2E2", "TP3E2", "TP2E3"],
)
def test_physical_rank_assembly(tp_degree, intermediate, ep_degree, layout):
    """Every physical rank executes; ranks are assembled in order."""
    _require_blackwell()
    from tests.moe.test_v41_expert_numerics import reference

    width, experts, capacity, topk = 192, 6, 8, 6
    x = shared_input(capacity)
    ids = distinct_routes(capacity, topk, experts)
    weights = uniform_weights(capacity, topk)

    # ONE full-width operand set shared by every rank and the final oracle.
    logical = bench.LogicalWeights.synthetic(experts, 2304, torch)
    expected = reference(x, ids, weights, logical.weights, logical.scales)

    # Physical ranks of a TP x EP grid: row = EP group, column = TP rank.
    shards = []
    records = []
    for ep_rank in range(ep_degree):
        for tp_rank in range(tp_degree):
            case = bench.build_case(width, intermediate, experts, capacity, topk,
                                    logical, torch, tp_degree)
            case.x = x
            case._quantize_input()
            case.install_rank(tp_rank, tp_degree)
            owner = bench.owners_for(ids, ep_degree, ep_rank)
            masked = bench.mask_ids(ids, owner)
            masked_weights = torch.where(
                owner, weights, torch.zeros_like(weights)
            ).contiguous()
            local = bench.rank_logical_slice(logical, tp_rank, tp_degree,
                                             intermediate, torch)
            case.out.fill_(12345)
            plane = case.run(capacity, masked, masked_weights)

            # Each rank is checked against its own shard oracle before assembly.
            rank_expected = reference(x, ids, masked_weights, local.weights,
                                      local.scales)
            rel_rank, cos_rank = check(route_sum(plane, capacity, topk),
                                       rank_expected,
                                       (layout, ep_rank, tp_rank, "rank"))
            owned = owner.flatten()
            if not bool(owned.all()):
                assert bool((plane.view(-1, HIDDEN)[~owned] == 0).all()), (
                    f"{layout} ep{ep_rank} tp{tp_rank}: unowned route not zero"
                )
            records.append(dict(layout=layout, ep_rank=ep_rank, tp_rank=tp_rank,
                                owned_routes=int(owned.sum().item()),
                                rel_l2=rel_rank, cosine=cos_rank))
            # Per-rank boundary: route sum in FP32, ONE BF16 rounding.
            shards.append(route_sum(plane, capacity, topk).bfloat16().float())
            del case

    # Ordered assembly: ranks summed in order in FP32, one final BF16 rounding.
    assembled = torch.zeros_like(expected)
    for shard in shards:
        assembled = assembled + shard
    assembled = assembled.bfloat16().float()
    expected_bf16 = expected.bfloat16().float()
    rel = ((assembled - expected_bf16).norm() / expected_bf16.norm()).item()
    cosine = torch.nn.functional.cosine_similarity(
        assembled.flatten(), expected_bf16.flatten(), dim=0
    ).item()
    assert rel < REL_TOL and cosine > COS_TOL, (layout, rel, cosine)
    if ep_degree > 1:
        assert bool((assembled != 0).any()), f"{layout}: assembly produced nothing"
    print(json.dumps(dict(
        case="physical_rank_assembly", layout=layout, tp_degree=tp_degree,
        ep_degree=ep_degree, intermediate=intermediate, width=width,
        physical_ranks=len(shards), rows=capacity, rel_l2=rel, cosine=cosine,
        bit_exact=False,
        ownership="diagnostic modulo; not a production scheduler",
        note="EP comparison is tolerance-gated, never bit-exact")), flush=True)
    for record in records:
        print(json.dumps(dict(case="physical_rank_shard", **record)), flush=True)


# --------------------------------------------------------------------------- #
# Row, width, reuse and skew coverage
# --------------------------------------------------------------------------- #

def reuse_routes(capacity, topk, experts):
    """Every token routes to the same top-k experts: maximum data reuse."""
    return (torch.arange(topk, device="cuda", dtype=torch.int32)
            .reshape(1, topk).expand(capacity, topk).contiguous() % experts)


def skewed_routes(capacity, topk, experts):
    """One expert takes most routes: skewed load, still distinct within a token."""
    ids = torch.zeros((capacity, topk), device="cuda", dtype=torch.int32)
    for row in range(capacity):
        for slot in range(topk):
            ids[row, slot] = 0 if slot == 0 else (1 + (row + slot) % (experts - 1))
    return ids


def _sweep_one(width, tp_degree, intermediate, experts, capacity, label,
               route_fn):
    """One compiled case; several live M on the same callable."""
    from tests.moe.test_v41_expert_numerics import reference
    import cutlass
    from b12x._lib.utils import current_cuda_stream

    logical, case = build(width, tp_degree, intermediate, experts, capacity, topk=6)
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    case.install_rank(0, tp_degree)
    full_ids = route_fn(capacity, 6, experts)
    weights = uniform_weights(capacity, 6)
    owner = bench.owners_for(full_ids, tp_degree, 0)
    callable_id = id(case.full)

    for rows in (1, 2, 8, 16, 80):
        if rows > capacity:
            continue
        sub_ids = full_ids[:rows].contiguous()
        sub_owner = owner[:rows]
        sub_weights = weights[:rows].contiguous()
        masked_live = bench.mask_ids(sub_ids, sub_owner)          # (rows, 6)
        weights_live = torch.where(
            sub_owner, sub_weights, torch.zeros_like(sub_weights)
        ).contiguous()                                            # (rows, 6)
        # Pad the tail to capacity with sentinel ids and zero weights, exactly as
        # a masked rank that owns nothing in those rows would send. The live rows
        # are unchanged; only bytes past the live extent are synthetic.
        if rows < capacity:
            pad_ids = torch.full((capacity - rows, 6), bench.SENTINEL,
                                 device="cuda", dtype=torch.int32)
            pad_weights = torch.zeros((capacity - rows, 6), device="cuda",
                                      dtype=torch.float32)
            upload_ids = torch.cat([masked_live, pad_ids], 0)
            upload_weights = torch.cat([weights_live, pad_weights], 0)
        else:
            upload_ids, upload_weights = masked_live, weights_live
        case.ids.copy_(upload_ids.flatten())
        case.routing.copy_(upload_weights.flatten())
        case.out.fill_(12345)
        case.full(*case._args, cutlass.Int32(rows), current_cuda_stream())
        torch.cuda.synchronize()
        assert id(case.full) == callable_id, f"{label}: callable changed at M={rows}"
        plane = case.out[: rows * 6].view(rows, 6, HIDDEN)
        # The oracle sees exactly the live rows: sub_ids and masked_weights are
        # already row-sized, so passing a capacity-shaped tensor would mismatch.
        expected = reference(case.x[:rows], sub_ids[:rows], weights_live,
                             local.weights, local.scales)
        rel, cosine = check(route_sum(plane, rows, 6), expected,
                            (label, width, tp_degree, rows))
        owned = sub_owner.flatten()
        if bool(owned.any()):
            assert bool((plane.view(-1, HIDDEN)[owned] != 0).any()), (
                f"{label} M={rows}: owned routes produced nothing")
        if not bool(owned.all()):
            assert bool((plane.view(-1, HIDDEN)[~owned] == 0).all()), (
                f"{label} M={rows}: unowned route not exact zero")
        assert bool((case.out[rows * 6:] == 12345).all()), (
            f"{label} M={rows}: wrote past the live extent")
        print(json.dumps(dict(
            case="m_sweep", label=label, width=width, tp_degree=tp_degree,
            intermediate=intermediate, rows=rows, capacity=capacity,
            active_experts=int(masked_live[sub_owner].unique().numel()),
            owned_routes=int(owned.sum().item()), rel_l2=rel, cosine=cosine,
            compiled_callable_unchanged=True,
            ownership="diagnostic modulo; not a production scheduler")), flush=True)
    del case, logical


@pytest.mark.parametrize(
    "width,tp_degree,intermediate,experts,capacity,label,route_fn",
    [
        (192, 2, 1152, 8, 80, "distinct-w192", distinct_routes),
        (128, 2, 1152, 8, 80, "distinct-w128", distinct_routes),
        (64, 2, 1152, 8, 80, "distinct-w64", distinct_routes),
        (192, 2, 1152, 8, 80, "reuse-w192", reuse_routes),
        (192, 2, 1152, 8, 80, "skew-w192", skewed_routes),
        (192, 4, 576, 8, 80, "distinct-tp4-w192", distinct_routes),
        (192, 3, 768, 8, 80, "distinct-tp3-w192", distinct_routes),
    ],
    ids=["w192", "w128", "w64", "reuse", "skew", "tp4", "tp3"],
)
def test_m_and_width_sweep(width, tp_degree, intermediate, experts, capacity,
                           label, route_fn):
    """M 1/2/8/16/80 across widths, reuse and skew distributions."""
    _require_blackwell()
    _sweep_one(width, tp_degree, intermediate, experts, capacity, label, route_fn)


# --------------------------------------------------------------------------- #
# Real official checkpoint operands
# --------------------------------------------------------------------------- #

def _checkpoint_snapshot():
    import os
    env = os.environ.get("DS41RT_TPEP_SNAPSHOT")
    if env:
        return Path(env)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidate = (hub / "hub" / "models--deepseek-ai--DeepSeek-V4.1-Flash"
                 / "snapshots" / "dba1be0a40aa45a94ad051997016db3960a90277")
    return candidate


def test_real_checkpoint_operands_rank_local():
    """The official checkpoint through the same rank-sliced forward path.

    Reads the real per-expert tensors (``layers.{L}.ffn.experts.{E}.w1.weight``
    and its scale, etc.), stacks the requested experts, slices each physical
    rank's shard, repacks it, and gates against the oracle built from that same
    real slice. This is the only arm that exercises the checkpoint reader.
    """
    _require_blackwell()
    from tests.moe.test_v41_expert_numerics import reference

    snapshot = _checkpoint_snapshot()
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip(f"official checkpoint not available at {snapshot}")

    width, tp_degree, intermediate = 192, 2, 1152
    experts, capacity, topk = 6, 1, 6
    logical = bench.LogicalWeights.checkpoint(snapshot, 0, experts,
                                              intermediate * tp_degree, torch)
    assert logical.source_mode == "checkpoint"
    assert logical.experts == experts

    case = bench.build_case(width, intermediate, experts, capacity, topk, logical,
                            torch, tp_degree)
    ids = distinct_routes(capacity, topk, experts)
    weights = uniform_weights(capacity, topk)
    case.install_rank(0, tp_degree)
    local = bench.rank_logical_slice(logical, 0, tp_degree, intermediate, torch)
    expected = reference(case.x, ids, weights, local.weights, local.scales)
    plane = case.run(capacity, ids, weights)
    rel, cosine = check(route_sum(plane, capacity, topk), expected,
                        ("checkpoint", tp_degree))
    owner = bench.owners_for(ids, tp_degree, 0)
    masked = bench.mask_ids(ids, owner)
    masked_weights = torch.where(owner, weights, torch.zeros_like(weights)).contiguous()
    masked_expected = reference(case.x, ids, masked_weights, local.weights,
                                local.scales)
    case.out.fill_(12345)
    masked_plane = case.run(capacity, masked, masked_weights)
    rel_m, cos_m = check(route_sum(masked_plane, capacity, topk), masked_expected,
                         ("checkpoint_masked", tp_degree))
    if not bool(owner.all()):
        assert bool((masked_plane.view(-1, HIDDEN)[(~owner).flatten()] == 0).all()), (
            "checkpoint arm: unowned route is not an exact zero")
    print(json.dumps(dict(
        case="real_checkpoint", layer=0,
        source_mode=logical.source_mode,
        snapshot_revision=logical.identity["snapshot_revision"],
        index_sha256=logical.identity["index_sha256"],
        operands_sha256=logical.identity["operands_sha256"],
        per_expert_tensors=logical.identity["per_expert_tensors"],
        tp_degree=tp_degree, intermediate=intermediate, experts=experts,
        rank_rel_l2=rel, rank_cosine=cosine,
        masked_rel_l2=rel_m, masked_cosine=cos_m)), flush=True)
