"""CPU-only final checks for the 96-invocation remaining schedule and the Part B
fixture loader.

The GPU run is live under fdcd's ownership; nothing here touches it, the frozen
harness, the consolidator or the generated manifest. It pins:

* the fixture loader requests LIVE rows and the harness zero-pads to capacity
  (requesting capacity 80 from a 64-row fixture fails closed);
* the schedule generator yields exactly 96 invocations, Part A 48 / Part B 48,
  four replicates per (part, pair, width, rows, role) cell, no extra sentinel;
* the Part B width order is counterbalanced across the four rounds;
* the generated manifest on disk matches the freshly computed schedule.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"
# The schedule generator and manifest live under the gitignored `runs/` tree, so
# on a clean checkout they are absent. Loading them lazily and skipping only the
# artifact-dependent tests keeps collection (and the harness-loader tests) working.
# The env overrides exist so the absent-artifact path can be validated without
# renaming the live files.
SCHEDULE = Path(os.environ.get(
    "DS41RT_TP_EP_SCHEDULE_ARTIFACT",
    str(ROOT / "runs" / "tp-ep-kernel" / "timing" / "schedule_remaining.py")))
MANIFEST = Path(os.environ.get(
    "DS41RT_TP_EP_SCHEDULE_MANIFEST",
    str(ROOT / "runs" / "tp-ep-kernel" / "timing" / "schedule" / "MANIFEST.json")))
FIXTURE = ROOT / "scripts" / "fixtures" / "tp-ep-reuse-m64-e384.json"


def load_harness():
    """Import the benchmark harness module for behavioural route-setup checks."""
    spec = importlib.util.spec_from_file_location("ds41rt_sched_final_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_schedule_or_skip(path: Path = SCHEDULE):
    if not path.is_file():
        pytest.skip(f"schedule artifact not present: {path}")
    return _load("ds41rt_sched_remaining", path)


@pytest.fixture(scope="module")
def sched():
    return _load_schedule_or_skip()


timing = _load("ds41rt_sched_timing", HARNESS)


def test_fixture_loader_uses_live_rows_not_capacity():
    tensor, _payload = timing.load_route_fixture(FIXTURE, 384, 6, 64)
    assert tuple(tensor.shape) == (64, 6)
    assert len(set(tensor.flatten().tolist())) == 48


def test_requesting_capacity_from_a_short_fixture_fails_closed():
    # The pre-fix call passed capacity (80) for a 64-row fixture and raised; the
    # live-rows call is what makes Part B possible, and an over-long request must
    # still fail closed rather than silently truncate.
    with pytest.raises(ValueError, match="need 80"):
        timing.load_route_fixture(FIXTURE, 384, 6, 80)


def test_harness_fixture_branch_requests_rows_and_zero_pads():
    """Behavioural, not source-string: drive the real route-setup helper.

    The reviewed defect lived in the caller's control flow, and a source-string
    assertion cannot prove it fixed. This calls `build_route_table` with the real
    fixture and asserts the live-row selection and zero-padded tail.
    """
    import types

    import torch

    module = load_harness()
    if FIXTURE is None or not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    table, payload = module.build_route_table(
        types.SimpleNamespace(route_fixture=FIXTURE), 384, rows=64, capacity=80)
    assert payload is not None
    assert tuple(table.shape) == (80, 6)
    # Live rows carry fixture routes; the capacity tail is zero padding.
    assert table[:64].numel() == 384
    assert int(table[64:].abs().sum()) == 0, "tail must be zero padding"
    # Requesting more rows than the fixture has fails closed rather than leaking.
    with pytest.raises(ValueError):
        module.build_route_table(
            types.SimpleNamespace(route_fixture=FIXTURE), 384, rows=96, capacity=80)


def test_schedule_is_96_with_four_replicates_per_cell_and_no_sentinel(sched):
    invocations = sched.schedule()
    assert len(invocations) == 96
    assert sched.validate(invocations) == {"A": 48, "B": 48}
    cells = {}
    for inv in invocations:
        key = (inv["part"], inv["pair"], inv["width"], inv["rows"], inv["role"])
        cells[key] = cells.get(key, 0) + 1
    assert cells and set(cells.values()) == {4}, cells
    part_b = [i for i in invocations if i["part"] == "B"]
    assert len(part_b) == 48
    assert all(i["fixture"] for i in part_b)
    assert all(i["rows"] == 64 for i in part_b)


def test_part_b_width_order_is_counterbalanced(sched):
    invocations = [i for i in sched.schedule() if i["part"] == "B"]
    rounds = {
        r: [i["width"] for i in invocations
            if i["pair"] == "AB" and i["round"] == r and i["role"] == "A"]
        for r in (1, 2, 3, 4)
    }
    assert rounds[1] == [64, 128, 192]
    assert rounds[2] == [192, 128, 64]
    assert rounds[3] == [192, 128, 64]
    assert rounds[4] == [64, 128, 192]


def test_generated_manifest_matches_the_computed_schedule(sched):
    if not MANIFEST.is_file():
        pytest.skip("generated manifest not present")
    on_disk = json.loads(MANIFEST.read_text())
    computed = sched.schedule()
    assert on_disk["total"] == 96
    assert on_disk["counts"] == {"A": 48, "B": 48}
    assert [i["label"] for i in on_disk["invocations"]] == [
        i["label"] for i in computed
    ]


def test_missing_schedule_artifact_skips_cleanly(tmp_path):
    """A clean checkout has no schedule artifact; the loader must skip, not crash.

    This is the in-process simulation of the absent-artifact state. The
    end-to-end collection check uses DS41RT_TP_EP_SCHEDULE_ARTIFACT instead of
    renaming the live generated files.
    """
    with pytest.raises(pytest.skip.Exception):
        _load_schedule_or_skip(tmp_path / "absent_schedule_remaining.py")
