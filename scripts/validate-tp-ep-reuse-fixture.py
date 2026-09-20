#!/usr/bin/env python3
"""Validate the deterministic TP×EP reuse fixture (CPU-only, no benchmark edits).

The fixture is a distribution-representative M=64 route table (48 experts, 384
routes) with intentional cross-row reuse and a 17+ tail. This validator proves:

- declared per-expert counts are exact and fully realized (no remainder);
- every row has six distinct in-range official expert ids (E=384);
- the degree sequence is realizable by the deterministic greedy
  highest-remaining construction with stable ties (not round-robin);
- whole-expert LPT ownership for EP2/EP3 assigns each expert to exactly one
  group, and equal expert counts do NOT imply balanced routed rows.

The LPT tie-break here mirrors the **benchmark's** stable `(cost, expert id)`
ordering (`bench_tp_ep_kernel.lpt_owner`); it is a diagnostic, not the Rust
coordinator's seeded tie-break, and makes no production-owner-exactness claim.

Run: `python scripts/validate-tp-ep-reuse-fixture.py [--fixture PATH]`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tp-ep-reuse-m64-e384.json"


def load_fixture(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def counts_from_routes(routes: list[list[int]], experts: int) -> list[int]:
    counts = [0] * experts
    for row in routes:
        for expert in row:
            counts[expert] += 1
    return counts


def realize_counts(counts: list[int], rows: int, topk: int) -> list[list[int]]:
    """Deterministic greedy highest-remaining construction with stable ties.

    Repeatedly selects the `topk` experts with the largest remaining count,
    breaking ties by lowest expert id. Raises if a row cannot be filled, so the
    caller can prove the degree sequence is realizable with no remainder.
    """
    remaining = {expert: count for expert, count in enumerate(counts) if count > 0}
    table: list[list[int]] = []
    for index in range(rows):
        chosen = sorted(remaining, key=lambda expert: (-remaining[expert], expert))[:topk]
        if len(chosen) < topk or any(remaining[expert] <= 0 for expert in chosen):
            raise ValueError(f"route construction stranded at row {index}")
        table.append(chosen)
        for expert in chosen:
            remaining[expert] -= 1
    remainder = sum(remaining.values())
    if remainder:
        raise ValueError(f"route construction left {remainder} unrouted slots")
    return table


def expert_cost(rows: int, *, weight_cost: float = 1.0, tile_cost: float = 0.0,
                tile_rows: int = 16) -> float:
    if rows <= 0:
        return 0.0
    if tile_rows <= 0:
        raise ValueError("tile_rows must be positive")
    return float(weight_cost) + float(tile_cost) * (-(-rows // int(tile_rows)))


def lpt_ownership(counts: list[int], groups: int, *, weight_cost: float = 1.0,
                  tile_cost: float = 0.0, tile_rows: int = 16) -> dict:
    """Whole-expert greedy LPT with the BENCHMARK's stable tie ordering.

    Stable descending by (cost, expert id); each active expert goes to the group
    with the smallest current load (ties by lowest group index). Every expert has
    exactly one owner; rows are never split across groups. This is a diagnostic
    and does not reproduce the Rust coordinator's seeded tie-break.
    """
    cost = [expert_cost(rows, weight_cost=weight_cost, tile_cost=tile_cost,
                        tile_rows=tile_rows) for rows in counts]
    order = sorted(range(len(counts)), key=lambda expert: (-cost[expert], expert))
    load = [0.0] * groups
    assignment = [0] * len(counts)
    assigned = [False] * len(counts)
    for expert in order:
        if cost[expert] == 0.0:
            continue
        target = min(range(groups), key=lambda group: (load[group], group))
        assignment[expert] = target
        assigned[expert] = True
        load[target] += cost[expert]
    group_experts = [0] * groups
    group_routes = [0] * groups
    for expert, count in enumerate(counts):
        if count > 0:
            group_experts[assignment[expert]] += 1
            group_routes[assignment[expert]] += count
    return {"groups": groups, "assignment": assignment, "assigned": assigned,
            "cost": cost, "group_experts": group_experts,
            "group_routes": group_routes}


def _strict_int(value, name: str) -> int:
    """Reject bools, floats and non-integral values instead of truncating."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def balanced_rows_observed(group_routes: list[int]) -> bool:
    """Whether the realized group route loads are exactly equal.

    This observes all groups (including middle ones); it is never a guarantee:
    weight-only LPT balances active-expert counts, and no general balanced-rows
    guarantee exists.
    """
    return len(set(group_routes)) == 1


def validate(fixture: dict) -> dict:
    experts = _strict_int(fixture["experts"], "experts")
    rows = _strict_int(fixture["rows"], "rows")
    topk = _strict_int(fixture["topk"], "topk")
    if fixture.get("schema") != "ds41rt.tp-ep-reuse-fixture.v1":
        raise ValueError("unsupported fixture schema")
    if experts != 384 or topk != 6:
        raise ValueError(f"fixture must use the official E=384/top-6 geometry, got E={experts} topk={topk}")

    declared = [0] * experts
    seen_ids = set()
    for entry in fixture["expert_counts"]:
        expert = _strict_int(entry["id"], "expert id")
        count = _strict_int(entry["rows"], "expert rows")
        if not 0 <= expert < experts:
            raise ValueError(f"expert id {expert} outside 0..{experts - 1}")
        if expert in seen_ids:
            raise ValueError(f"duplicate expert id {expert}")
        if not 1 <= count <= rows:
            raise ValueError(f"expert {expert} count {count} outside 1..{rows}")
        seen_ids.add(expert)
        declared[expert] = count
    if sum(declared) != rows * topk:
        raise ValueError(f"declared counts sum to {sum(declared)}, expected {rows * topk}")

    routes = [[_strict_int(expert, "route expert id") for expert in row]
              for row in fixture["routes"]]
    if len(routes) != rows or any(len(row) != topk for row in routes):
        raise ValueError(f"routes must be {rows} rows of {topk}")
    for index, row in enumerate(routes):
        if len(set(row)) != topk:
            raise ValueError(f"row {index} repeats an expert")
        if any(not 0 <= expert < experts for expert in row):
            raise ValueError(f"row {index} has an out-of-range expert id")

    realized = counts_from_routes(routes, experts)
    if realized != declared:
        mismatch = [i for i in range(experts) if realized[i] != declared[i]][:5]
        raise ValueError(f"route counts differ from declared at experts {mismatch}")

    # Independent realizability proof: the declared sequence must be constructible
    # by the deterministic greedy with no remainder.
    regenerated = realize_counts(declared, rows, topk)
    regenerated_counts = counts_from_routes(regenerated, experts)
    if regenerated_counts != declared:
        raise ValueError("greedy realization did not reproduce the declared counts")

    tail = fixture.get("tail_bin") or {}
    tail_experts = _strict_int(tail.get("experts", 0), "tail experts")
    tail_routes = _strict_int(tail.get("routes", 0), "tail routes")
    tail_counts = [_strict_int(v, "tail per-expert count")
                   for v in tail.get("per_expert_counts", [])]
    actual_tail = sorted((count for count in declared if count >= 17), reverse=True)
    if len(actual_tail) != tail_experts or sum(actual_tail) != tail_routes:
        raise ValueError("tail_bin aggregate count/routes disagree with the declared counts")
    if tail_counts and sorted(tail_counts, reverse=True) != actual_tail:
        raise ValueError("tail_bin per-expert counts disagree with the declared counts")

    ownership = {}
    for groups in (2, 3):
        plan = lpt_ownership(declared, groups)
        if sum(plan["assigned"]) != len(seen_ids):
            raise ValueError(f"EP{groups}: every active expert must have exactly one owner")
        if sum(plan["group_experts"]) != len(seen_ids):
            raise ValueError(f"EP{groups}: group expert counts do not sum to active experts")
        if sum(plan["group_routes"]) != sum(declared):
            raise ValueError(f"EP{groups}: group route counts do not sum to total routes")
        ownership[f"ep{groups}"] = {
            "group_experts": plan["group_experts"],
            "group_routes": plan["group_routes"],
            # Observed over ALL groups (middle included) and never a guarantee:
            # weight-only LPT has no general balanced-rows property.
            "balanced_rows_observed": balanced_rows_observed(plan["group_routes"]),
            "balanced_rows_guaranteed": False,
            "tie_break": "benchmark stable ties (cost, expert id); not the Rust coordinator's seeded tie-break",
        }

    return {
        "fixture": fixture.get("label"),
        "experts_active": len(seen_ids),
        "routes": sum(declared),
        "rows": rows,
        "topk": topk,
        "mean_rows_per_expert": sum(declared) / len(seen_ids),
        "tail_experts": tail_experts,
        "tail_routes": tail_routes,
        "tail_per_expert_synthetic": bool(tail.get("synthetic_per_expert", True)),
        "realized_by_greedy_highest_remaining": True,
        "no_remainder": True,
        "ownership": ownership,
        "note": ("whole-expert LPT only; equal expert counts per group must not be "
                 "read as balanced routed rows (see ownership group_routes)"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    report = validate(load_fixture(args.fixture))
    print(json.dumps(report, indent=2))
