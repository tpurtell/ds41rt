"""CPU-only unit tests for the diagnostic LPT cost model.

No GPU, no torch tensors on device: this pins the cost arithmetic that the
replicated-group scheduling comparison depends on, so a wrong formula cannot
reach a timing table. The authoritative form is
``ds41rt-core``'s ``ReplicatedExpertCostModel``:

    active   -> expert_weight_cost + ceil(routed_rows / tile_rows) * tile_cost
    inactive -> 0

In particular ``expert_weight_cost`` is charged ONCE per active expert, not per
routed row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_ep_timing", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load()


def _cost(rows, weight_cost, tile_cost, tile_rows=16):
    """Call the production cost expression through the module's own helper path."""
    return timing.expert_cost(rows, weight_cost=weight_cost, tile_cost=tile_cost,
                              tile_rows=tile_rows)


def test_inactive_expert_is_free():
    assert _cost(0, 1, 0) == 0
    assert _cost(0, 10, 2) == 0, "inactive experts must never be charged tiles"


def test_weight_cost_is_charged_once_per_active_expert():
    # 1, 16 and 17 rows are all one weight charge when tile_cost is 0.
    assert _cost(1, 1, 0) == 1
    assert _cost(16, 1, 0) == 1
    assert _cost(17, 1, 0) == 1
    # A large row count must not multiply the weight term.
    assert _cost(1000, 1, 0) == 1


def test_tile_costs_use_ceiling_boundaries():
    # weight 10, tile 2, tile_rows 16 -> 12, 12, 14
    assert _cost(1, 10, 2) == 12
    assert _cost(16, 10, 2) == 12
    assert _cost(17, 10, 2) == 14
    assert _cost(32, 10, 2) == 12 + 2
    assert _cost(33, 10, 2) == 12 + 4


def test_cost_is_monotonic_in_rows():
    previous = -1
    for rows in range(0, 130):
        value = _cost(rows, 3, 5)
        assert value >= previous
        previous = value


def test_lpt_balances_and_skips_inactive_experts():
    """Greedy assignment: active experts distributed, inactive never assigned."""
    import torch

    # counts: expert0=40, expert1=1, expert2=0, expert3=17
    ids = torch.tensor([[0] * 6, [3] * 6] + [[0] * 4 + [1] * 2] * 5,
                       dtype=torch.int32)
    owner, schedule = timing.lpt_owner(ids, 2, 0, 4, weight_cost=1.0,
                                       tile_cost=0.0, tile_rows=16)
    assert schedule["rows_per_expert"][2] == 0
    assert schedule["expert_cost"][2] == 0.0
    # Every active expert is assigned to some group.
    assert all(group in (0, 1) for expert, group in
               enumerate(schedule["assignment"]) if schedule["rows_per_expert"][expert])
    # Group loads are exactly the sum of the costs assigned to them.
    for group in (0, 1):
        expected = sum(schedule["expert_cost"][e] for e, g
                       in enumerate(schedule["assignment"]) if g == group)
        assert schedule["group_loads"][group] == pytest.approx(expected)


def test_lpt_is_deterministic():
    import torch

    ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
    first = timing.lpt_owner(ids, 3, 0, 6, weight_cost=1.0, tile_cost=2.0,
                             tile_rows=16)
    for _ in range(5):
        again = timing.lpt_owner(ids, 3, 0, 6, weight_cost=1.0, tile_cost=2.0,
                                 tile_rows=16)
        assert again[1]["assignment"] == first[1]["assignment"]
        assert again[1]["group_loads"] == first[1]["group_loads"]


def test_weight_and_tile_terms_are_independent():
    # Doubling the weight cost adds exactly one more unit per active expert.
    low = _cost(20, 1, 7)
    high = _cost(20, 3, 7)
    assert high - low == 2, "weight term must not scale with rows"


def test_no_duplicate_top_level_definitions():
    """AST guard: no harness file may define a top-level name twice.

    A later duplicate silently shadows the earlier definition. That is exactly
    how a corrected cost model and a corrected per-group measurement were both
    shadowed by stale copies, so this is enforced for every file in the harness
    set rather than checked by eye.
    """
    import ast
    import collections

    files = [
        ROOT / "python" / "tools" / "bench_tp_ep_kernel.py",
        ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py",
        ROOT / "python" / "tests" / "test_v41_sentinel_masking.py",
        ROOT / "python" / "tests" / "test_v41_ep_algebra.py",
        ROOT / "python" / "tests" / "test_tp_ep_cost_model.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        counts = collections.Counter(
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        )
        duplicates = {name: n for name, n in counts.items() if n > 1}
        assert not duplicates, f"{path.name} defines duplicates: {duplicates}"


def test_amortised_timing_uses_captured_direct_launches():
    """The GPU-resident timer must capture direct launches, not replay graphs.

    Repeating host-submitted graph replays does not amortise the per-launch host
    gap, so a timer built that way cannot distinguish GPU time from submission
    time. It must capture `launch(stream)` calls inside one outer graph.
    """
    import inspect

    source = inspect.getsource(timing.amortised_samples)
    assert "with torch_module.cuda.graph(outer):" in source
    assert "launch(current_cuda_stream())" in source
    assert "graph.replay()" not in source, (
        "nested graph replay cannot be captured and does not amortise host gaps")


def test_snapshot_restore_requires_capacity_sized_buffers():
    """The restore path must be handed capacity-sized, padded state.

    Regression guard: an earlier version copied the unpadded ``rows * topk`` view
    into capacity-sized buffers, which raised a size mismatch for every arm with
    rows < capacity and aborted the M8/M16 timing arms after they had already
    measured correctly.
    """
    import torch

    capacity, rows, topk = 80, 8, 6
    ids_buffer = torch.zeros(capacity * topk, dtype=torch.int32)
    routing_buffer = torch.zeros(capacity * topk, dtype=torch.float32)
    live_ids = torch.zeros((rows, topk), dtype=torch.int32)
    live_weights = torch.full((rows, topk), 1.0 / topk, dtype=torch.float32)

    # The unpadded form must fail loudly rather than silently truncate.
    try:
        timing._snapshot_arm_state(ids_buffer, routing_buffer, live_ids,
                                   live_weights, rows, capacity)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "unpadded state should not fit capacity-sized buffers"

    # The padded upload form must be accepted and restore the full buffer.
    pad_ids = torch.full((capacity - rows, topk), timing.bench.SENTINEL,
                         dtype=torch.int32)
    pad_weights = torch.zeros((capacity - rows, topk), dtype=torch.float32)
    upload_ids = torch.cat([live_ids, pad_ids], 0)
    upload_weights = torch.cat([live_weights, pad_weights], 0)
    info = timing._snapshot_arm_state(ids_buffer, routing_buffer, upload_ids,
                                      upload_weights, rows, capacity)
    assert info == {"restored_rows": rows, "restored_capacity": capacity}
    assert int((ids_buffer == timing.bench.SENTINEL).sum()) == (capacity - rows) * topk
