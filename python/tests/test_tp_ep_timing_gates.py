"""CPU-only negative tests for the timing-consolidation admissibility gates.

The consolidation script decides which timing arms may contribute a number. Every
gate must actually reject, so each has a negative test here: a fixed record is
mutated one field at a time and the corresponding rejection reason is required.
A gate that silently stops enforcing would otherwise let a bad arm into the
reported table.

No GPU, no timing, no network. Runs against synthetic records only.
"""

from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Every test in this module drives the consolidator's gates, so the whole module
# is skipped when the gitignored `runs/` artifact is absent (a clean checkout)
# rather than crashing during collection. The env override validates the absent
# state without renaming the live generated file.
CONSOLIDATE = Path(os.environ.get(
    "DS41RT_TP_EP_CONSOLIDATE_ARTIFACT",
    str(ROOT / "runs" / "tp-ep-kernel" / "timing" / "consolidate.py")))
if not CONSOLIDATE.is_file():
    pytest.skip(f"timing consolidator artifact not present: {CONSOLIDATE}",
                allow_module_level=True)


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_consolidate", CONSOLIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


con = _load()


def good_prov():
    return {"operands": "checkpoint", "samples": 30,
            "replays_per_warm_interval": 10, "flush_bytes": 268435456,
            "layer": 20, "lpt": True, "weight_cost": 1.0, "tile_cost": 0.0,
            "tile_rows": 16}


def good_record(rows=1, capacity=None):
    """A minimal record that must pass every gate."""
    if capacity is None:
        capacity = 1 if rows == 1 else 80
    groups = {
        "group0": {c: {"median_us": 100.0} for c in con.CONDITIONS},
        "max_group_measured_sequentially": True,
    }
    for c in con.CONDITIONS:
        groups[f"max_{c}_median_us"] = 100.0
    return {
        "capacity": capacity,
        "preflight": {"rel_l2": 0.0016, "cosine": 0.999998},
        "graph_verified_not_noop": True,
        "poisoned_only_in_preflight": True,
        "post_timing_rel_l2": 0.0016,
        "compiled_callable_unchanged": True,
        "per_group": groups,
    }


def reasons(prov=None, rec=None, rows=1):
    return con.admission_reasons(prov or good_prov(), rec or good_record(rows), rows)


def test_baseline_record_passes_every_gate():
    assert reasons() == [], "the baseline fixture must be admissible"


@pytest.mark.parametrize("rows,capacity", [(1, 1), (8, 80), (16, 80)])
def test_row_bucket_capacity_is_accepted(rows, capacity):
    assert reasons(rec=good_record(rows, capacity), rows=rows) == []


def test_wrong_capacity_for_row_bucket_is_rejected():
    # capacity 80 with rows 1 is the exact provenance defect found in audit.
    assert "capacity 80 != row bucket 1" in reasons(rows=1,
                                                    rec=good_record(1, 80))
    assert "capacity 1 != row bucket 80" in reasons(rows=8,
                                                    rec=good_record(8, 1))


def test_missing_capacity_is_rejected():
    rec = good_record(1)
    del rec["capacity"]
    assert any("row bucket 1" in r for r in reasons(rec=rec, rows=1))


def test_low_rel_l2_and_cosine_gate():
    rec = good_record()
    rec["preflight"]["rel_l2"] = 0.5
    assert "numerics gate" in reasons(rec=rec)
    rec = good_record()
    rec["preflight"]["cosine"] = 0.5
    assert "numerics gate" in reasons(rec=rec)


def test_graph_and_preflight_flags_gate():
    for field in ("graph_verified_not_noop", "poisoned_only_in_preflight"):
        rec = good_record()
        rec[field] = False
        assert "graph/preflight gate" in reasons(rec=rec), field


def test_post_timing_drift_gate():
    rec = good_record()
    rec["post_timing_rel_l2"] = 0.0020
    assert "post-timing drift" in reasons(rec=rec)


def test_compile_key_gate():
    rec = good_record()
    rec["compiled_callable_unchanged"] = False
    assert "compile-key gate" in reasons(rec=rec)


@pytest.mark.parametrize("condition", ["warm", "amortised", "cold"])
def test_missing_group_max_projection_is_rejected(condition):
    rec = good_record()
    del rec["per_group"][f"max_{condition}_median_us"]
    assert f"missing max_{condition} group projection" in reasons(rec=rec)


def test_sequential_projection_flag_must_be_true():
    rec = good_record()
    rec["per_group"]["max_group_measured_sequentially"] = False
    assert "sequential-projection flag unset" in reasons(rec=rec)
    rec = good_record()
    del rec["per_group"]["max_group_measured_sequentially"]
    assert "sequential-projection flag unset" in reasons(rec=rec)


def test_stated_max_must_equal_the_worst_group():
    rec = good_record()
    # two groups; the stated max understates the worst group
    rec["per_group"]["group1"] = {c: {"median_us": 250.0} for c in con.CONDITIONS}
    assert any("does not equal worst group" in r for r in reasons(rec=rec))
    # correcting the stated max admits it again
    rec["per_group"]["group1"] = {c: {"median_us": 250.0} for c in con.CONDITIONS}
    for c in con.CONDITIONS:
        rec["per_group"][f"max_{c}_median_us"] = 250.0
    assert reasons(rec=rec) == []


def test_flush_size_gate():
    prov = good_prov()
    prov["flush_bytes"] = 33554432
    assert "flush 33554432 != 268435456" in reasons(prov=prov)
    prov = good_prov()
    del prov["flush_bytes"]
    assert any("flush None" in r for r in reasons(prov=prov))


def test_sample_count_gate():
    prov = good_prov()
    prov["samples"] = 5
    assert "samples 5 != 30" in reasons(prov=prov)
    prov = good_prov()
    del prov["samples"]
    assert any("samples None" in r for r in reasons(prov=prov))


def test_gates_accumulate_rather_than_short_circuit():
    """Every failing gate is reported, so one bad field cannot hide another."""
    rec = good_record()
    rec["capacity"] = 80
    rec["compiled_callable_unchanged"] = False
    prov = good_prov()
    prov["samples"] = 1
    got = reasons(prov=prov, rec=rec, rows=1)
    assert "compile-key gate" in got
    assert "capacity 80 != row bucket 1" in got
    assert "samples 1 != 30" in got
    assert len(got) >= 3


def test_signature_excludes_capacity_but_includes_the_real_fields():
    """The provenance signature must not carry the null capacity key."""
    assert "capacity" not in con.SIGNATURE_FIELDS
    for field in ("flush_bytes", "samples", "operands", "layer", "lpt"):
        assert field in con.SIGNATURE_FIELDS


def test_real_arm_directory_still_fully_admissible():
    """The preserved 96-arm set must remain 96/96 admissible under the gates."""
    arm_dir = ROOT / "runs" / "tp-ep-kernel" / "timing" / "arms"
    arms = sorted(arm_dir.glob("*.json"))
    if not arms:
        pytest.skip("archived arm set not present")
    accepted = 0
    for path in arms:
        m = con.ARM_RE.match(path.name)
        if not m:
            continue
        prov, rec = con.load_arm(path)
        rows = int(m.group("m"))
        assert con.admission_reasons(prov, rec, rows) == [], path.name
        accepted += 1
    assert accepted == len(arms) == 96
