"""Regression for the benchmark's route-setup control flow (post-run V2 fix).

The earlier `test_tp_ep_route_fixture.py` exercises `load_route_fixture` in
isolation only. The defect being pinned here lived in the CALLER's control flow:
one branch could establish the live-row count while another selected rows, so a
synthetic-plus-fixture call or a multi-row request could leak or select the wrong
rows. These tests therefore drive the actual `build_route_table` helper that the
benchmark arm now uses, and they assert on behaviour, not on source text.
"""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"
FIXTURE = ROOT / "scripts" / "fixtures" / "tp-ep-reuse-m64-e384.json"


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_route_setup", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load()


def opts(fixture=None):
    return types.SimpleNamespace(route_fixture=fixture)


def test_synthetic_path_has_no_fixture_and_is_all_distinct_at_m8():
    table, payload = timing.build_route_table(opts(), 384, rows=8, capacity=80)
    assert payload is None
    assert tuple(table.shape) == (80, 6)
    live = table[:8]
    assert len(set(live.flatten().tolist())) == 48, "M8 synthetic must be all-distinct"
    # The synthetic table is capacity-sized by construction; only the live prefix is
    # ever routed (`live = rows`), so the tail is not required to be zero. What must
    # hold is that the live prefix is exactly the synthetic formula.
    expected = (torch.arange(1, 9, dtype=torch.int32).reshape(8, 1) * 6
                + torch.arange(6, dtype=torch.int32).reshape(1, 6)) % 384
    assert torch.equal(live, expected)


def test_synthetic_path_matches_the_documented_formula():
    table, _ = timing.build_route_table(opts(), 384, rows=4, capacity=4)
    expected = (torch.arange(1, 5, dtype=torch.int32).reshape(4, 1) * 6
                + torch.arange(6, dtype=torch.int32).reshape(1, 6)) % 384
    assert torch.equal(table, expected)


def test_fixture_path_live64_cap80_selects_exactly_64_live_rows():
    """The live64/cap80 case: 64 fixture rows live, 16 padded, all 384 routes."""
    if not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    table, payload = timing.build_route_table(opts(FIXTURE), 384, rows=64, capacity=80)
    assert payload is not None
    assert tuple(table.shape) == (80, 6)
    live = table[:64]
    assert len(set(live.flatten().tolist())) == 48, "48 distinct experts over 384 routes"
    assert live.numel() == 384
    # Everything beyond the live rows is padding, so it cannot leak fixture routes.
    assert int(table[64:].abs().sum()) == 0


def test_fixture_live_rows_are_exactly_the_first_rows_of_the_fixture():
    """No leak: live selection is the first `rows` fixture rows, in order."""
    if not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    payload = json.loads(FIXTURE.read_text())
    table, _ = timing.build_route_table(opts(FIXTURE), 384, rows=64, capacity=80)
    for r in range(64):
        assert table[r].tolist() == payload["routes"][r], f"row {r} leaked or reordered"


def test_fixture_partial_rows_select_a_prefix_not_a_suffix():
    if not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    full, _ = timing.build_route_table(opts(FIXTURE), 384, rows=64, capacity=64)
    part, _ = timing.build_route_table(opts(FIXTURE), 384, rows=16, capacity=64)
    assert torch.equal(part[:16], full[:16])
    assert not torch.equal(part[:16], full[48:64]), "must not select the tail"


def test_multiple_rows_request_fails_closed_before_loading(tmp_path):
    """A malformed multi-row fixture must be rejected, not silently mis-selected."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "ds41rt.tp-ep-reuse-fixture.v1",
                               "routes": [[0, 1, 2, 3, 4, 5]]}))
    with pytest.raises(ValueError):
        timing.build_route_table(opts(bad), 384, rows=8, capacity=80)


def test_rows_exceeding_capacity_fails_closed():
    with pytest.raises(ValueError):
        timing.build_route_table(opts(FIXTURE), 384, rows=96, capacity=80)


def test_synthetic_and_fixture_are_independent_paths():
    """Switching the fixture must not perturb the synthetic table."""
    a, _ = timing.build_route_table(opts(), 384, rows=64, capacity=80)
    b, payload = timing.build_route_table(opts(FIXTURE), 384, rows=64, capacity=80)
    assert payload is not None
    assert not torch.equal(a[:64], b[:64]), "fixture must actually change the routes"
    # ...and the synthetic table is reproducible regardless of prior fixture use.
    c, _ = timing.build_route_table(opts(), 384, rows=64, capacity=80)
    assert torch.equal(a, c), "synthetic path leaked state from the fixture path"


# ---------------------------------------------------------------------------
# Caller-level regressions: these drive the orchestration `measure_group` runs,
# not the helper in isolation. The reviewed defect lived at the CALL SITE, where
# `rows` was bound only by the checkpoint operand-union loop, so a synthetic run
# raised NameError and a multi-row checkpoint run silently used the last row
# count. A helper-only test cannot catch either.
# ---------------------------------------------------------------------------

def _options(rows, fixture=None, capacity=80, operands="synthetic"):
    return types.SimpleNamespace(rows=rows, route_fixture=fixture,
                                 capacity=capacity, operands=operands)


def test_caller_synthetic_single_row_binds_without_a_checkpoint_loop():
    """Synthetic operands have NO checkpoint loop; the caller must still bind rows."""
    opts = _options([8], fixture=None, operands="synthetic")
    got = list(timing.iter_live_rows(opts, 384))
    assert len(got) == 1
    rows, table, payload = got[0]
    assert rows == 8 and payload is None
    assert tuple(table.shape) == (80, 6)
    assert len(set(table[:8].flatten().tolist())) == 48


def test_caller_fixture_live64_cap80_through_the_orchestrator():
    if not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    opts = _options([64], fixture=FIXTURE, capacity=80, operands="checkpoint")
    rows, table, payload = next(iter(timing.iter_live_rows(opts, 384)))
    assert rows == 64 and payload is not None
    assert len(set(table[:64].flatten().tolist())) == 48
    assert int(table[64:].abs().sum()) == 0


def test_caller_multi_row_request_fails_closed():
    """A multi-row request must fail before binding, not reuse the last row count."""
    opts = _options([8, 16], fixture=None, operands="synthetic")
    with pytest.raises(SystemExit, match="one live row count per invocation"):
        list(timing.iter_live_rows(opts, 384))


def test_caller_empty_rows_fails_closed():
    with pytest.raises(SystemExit, match="no live row counts"):
        list(timing.iter_live_rows(_options([], operands="synthetic"), 384))


def test_caller_binds_each_row_count_to_its_own_table():
    """No shared mutable binding: each yielded table matches its own rows."""
    opts = _options([8], fixture=None, operands="synthetic")
    rows, table, _ = next(iter(timing.iter_live_rows(opts, 384)))
    expected = (torch.arange(1, rows + 1, dtype=torch.int32).reshape(rows, 1) * 6
                + torch.arange(6, dtype=torch.int32).reshape(1, 6)) % 384
    assert torch.equal(table[:rows], expected)


def test_measure_group_no_longer_references_a_preloop_rows_binding():
    """AST check: the route table must be built inside the live-rows loop.

    This is the specific structural defect the reviewer found (call outside the
    loop, `rows` bound only under `--operands checkpoint`). It is asserted on the
    AST, and the behavioural tests above cover the same path by execution.
    """
    import ast

    tree = ast.parse(HARNESS.read_text())
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "measure_group":
            target = node
            break
    assert target is not None, "measure_group not found"

    # Any call to build_route_table directly inside measure_group is the bug shape;
    # it must go through iter_live_rows instead.
    direct = [n for n in ast.walk(target)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)
              and n.func.id == "build_route_table"]
    assert not direct, (
        "measure_group must not call build_route_table directly; bind routes through "
        "iter_live_rows so each live row count gets its own table")

    uses = [n for n in ast.walk(target)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "iter_live_rows"]
    assert uses, "measure_group must obtain routes from iter_live_rows"


# ---------------------------------------------------------------------------
# Bank-label truthfulness and early row-count validation.
#
# A bank whose active union covers every expert IS the full real expert bank, so
# the previous unconditional "not a full real bank" label was false for a fixture
# M64 run. These tests drive the label function and the argument guard directly.
# ---------------------------------------------------------------------------

def _bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ds41rt_bank_label_bench", ROOT / "python" / "tools" /
        "benchmark_v41_ep_groups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_bank_label_is_truthful():
    label = _bench().bank_label(384, 384)
    assert "full real expert bank" in label
    assert "NOT a full real expert bank" not in label
    assert "zero-filled" in label


def test_subset_bank_label_says_not_full():
    label = _bench().bank_label(48, 384)
    assert "48 active experts" in label
    assert "NOT a full real expert bank" in label
    assert "full real expert bank:" not in label


def test_bank_label_boundary_is_exact():
    b = _bench()
    assert "full real expert bank" in b.bank_label(384, 384)
    # One short of the whole bank must NOT claim to be full.
    assert "NOT a full real expert bank" in b.bank_label(383, 384)


def test_operand_identity_uses_the_shared_label():
    """The identity must not carry a hand-rolled label that can drift."""
    src = (ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py").read_text()
    assert "bank_label(len(ids), experts)" in src
    assert "is_full_real_bank" in src


def _parse(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ds41rt_parse_args", ROOT / "python" / "tools" / "bench_tp_ep_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_args(args)


BASE = ["--topologies", "tp4", "--ep-degree", "1", "--widths", "192",
        "--capacity", "80", "--operands", "synthetic",
        "--output", "/tmp/ds41rt-parse-args-test.json"]


def test_parse_args_rejects_empty_rows_gracefully():
    with pytest.raises(SystemExit) as exc:
        _parse(BASE + ["--rows", ""])
    assert exc.value.code != 0


def test_parse_args_rejects_multi_row_before_any_gpu_work():
    with pytest.raises(SystemExit):
        _parse(BASE + ["--rows", "8,16"])


def test_parse_args_rejects_non_positive_rows():
    with pytest.raises(SystemExit):
        _parse(BASE + ["--rows", "0"])
    with pytest.raises(SystemExit):
        _parse(BASE + ["--rows", "-4"])


def test_parse_args_accepts_exactly_one_positive_row():
    opts = _parse(BASE + ["--rows", "64"])
    assert opts.rows == [64]


def test_row_guard_precedes_blackwell_and_operand_load():
    """The guard must be in parse_args, i.e. before _require_blackwell."""
    import ast
    tree = ast.parse((ROOT / "python" / "tools" /
                      "bench_tp_ep_kernel.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args":
            body = ast.dump(node)
            assert "exactly one row count is required" in body
            assert "must name at least one positive row count" in body
            return
    raise AssertionError("parse_args not found")
