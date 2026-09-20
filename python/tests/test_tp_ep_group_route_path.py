"""CPU-only regressions for the per-group route path and single-record output.

Two defects found in review are pinned here because neither needs a GPU to catch
at the source level, and both would silently corrupt a comparison:

* the per-group block used to rebuild an ALL-DISTINCT synthetic route table
  instead of using the arm's own routes, so under ``--route-fixture`` the group
  numbers measured a different workload from the timed arm; and
* one invocation could emit several records or several widths, which the
  consolidator cannot pair (it encodes one width per filename) and, under
  ``python -O``, would have silently reduced to ``records[0]``.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"
# The consolidator lives under the gitignored `runs/` tree: absent on a clean
# checkout. Load it lazily so the harness route checks still run and only the
# consolidator-dependent tests skip. The env override validates absence without
# touching the live generated file.
CONSOLIDATE = Path(os.environ.get(
    "DS41RT_TP_EP_CONSOLIDATE_ARTIFACT",
    str(ROOT / "runs" / "tp-ep-kernel" / "timing" / "consolidate.py")))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_consolidator_or_skip(path: Path = CONSOLIDATE):
    if not path.is_file():
        pytest.skip(f"timing consolidator artifact not present: {path}")
    return _load("ds41rt_group_route_con", path)


@pytest.fixture(scope="module")
def con():
    return _load_consolidator_or_skip()


timing = _load("ds41rt_group_route_path", HARNESS)


def _group_body():
    src = HARNESS.read_text()
    m = re.search(r"def _per_group_latency\((.*?)\n(?=\ndef )", src, re.S)
    assert m, "_per_group_latency not found"
    return m.group(0)


def test_per_group_accepts_the_arm_routes():
    """The group block must be parameterised by the arm's routes and weights."""
    body = _group_body()
    head = body[: body.index(")")]
    assert "live_ids" in head and "live_weights" in head


def test_per_group_does_not_regenerate_a_synthetic_route_table():
    """No `arange(1, rows+1)*6 ... % experts` regeneration inside the group block."""
    body = _group_body()
    assert "arange(1, rows + 1" not in body, (
        "the per-group block rebuilds a synthetic route table instead of using "
        "the arm's routes; fixture runs would measure a different workload")
    assert "% experts" not in body, "synthetic modulo table found in group block"


def test_per_group_is_called_with_the_arm_routes():
    """The call site must forward the arm's live routes, not omit them."""
    src = HARNESS.read_text()
    m = re.search(r"per_group = _per_group_latency\((.*?)\)", src, re.S)
    assert m, "per-group call site not found"
    args = m.group(1)
    assert "live_ids" in args and "live_weights" in args


def test_arm_routes_come_from_the_fixture_when_given():
    """Behavioural: the fixture must actually change the live routes.

    Superseded a source-string check, which the V2 review showed cannot prove the
    caller selects the right rows. This drives the real route-setup helper.
    """
    import types

    import torch as _torch

    fixture = ROOT / "scripts" / "fixtures" / "tp-ep-reuse-m64-e384.json"
    if not fixture.is_file():
        pytest.skip("reuse fixture not present")
    synthetic, payload_a = timing.build_route_table(
        types.SimpleNamespace(route_fixture=None), 384, 64, 80)
    fixture_table, payload_b = timing.build_route_table(
        types.SimpleNamespace(route_fixture=fixture), 384, 64, 80)
    assert payload_a is None and payload_b is not None
    assert not _torch.equal(synthetic[:64], fixture_table[:64]), (
        "fixture must actually change the live routes")
    assert len(set(fixture_table[:64].flatten().tolist())) == 48


def test_multi_width_output_is_rejected_by_the_harness():
    src = HARNESS.read_text()
    assert "exactly one width per invocation is required" in src
    assert "expected exactly one record" in src


def test_consolidator_raises_rather_than_asserts_on_multi_record(tmp_path, con):
    """A multi-record file must raise explicitly, not assert (stripped by -O)."""
    bad = tmp_path / "A_w64_m1_1_1.json"
    bad.write_text('{"provenance": {}, "records": [{}, {}]}')
    with pytest.raises(ValueError, match="one arm per file"):
        con.load_arm(bad)


def test_consolidator_rejects_a_two_width_file(tmp_path, con):
    bad = tmp_path / "A_w64_m1_1_1.json"
    bad.write_text('{"provenance": {}, "records": [{}, {}]}')
    try:
        con.load_arm(bad)
        raise AssertionError("multi-record file was accepted")
    except ValueError:
        pass


def test_signature_fields_include_revision_hashes(con):
    """Mixed harness/benchmark/fixture revisions must not report one set."""
    for field in ("harness_sha256", "benchmark_sha256", "route_fixture_sha256"):
        assert field in con.SIGNATURE_FIELDS, field


def test_timing_functions_are_unchanged_against_archived_historical_source():
    """The four timed functions must match the archived historical source.

    Limit: only `c741d34a` is retained on disk. The `452d6eb6` revision that
    produced the archived 96 arms was **not** archived, so equivalence for those
    arms cannot be verified from source and is not claimed here.
    """
    archived = ROOT / "runs" / "tp-ep-kernel" / "source" / "bench_tp_ep_kernel.py"
    if not archived.is_file():
        pytest.skip("archived historical source not present")
    active_src = HARNESS.read_text()
    old_src = archived.read_text()
    for fn in ("warm_samples", "cold_samples", "amortised_samples",
               "_timed_condition"):
        pat = re.compile(rf"def {fn}\(.*?\n(?=\ndef |\nclass )", re.S)
        a, b = pat.search(active_src), pat.search(old_src)
        assert a and b, fn
        assert a.group(0) == b.group(0), (
            f"timed function {fn} differs from the archived historical source")


def test_missing_consolidator_artifact_skips_cleanly(tmp_path):
    """A clean checkout has no consolidator; the loader must skip, not crash.

    In-process simulation of the absent-artifact state. The end-to-end
    collection check uses DS41RT_TP_EP_CONSOLIDATE_ARTIFACT instead of renaming
    the live generated file.
    """
    with pytest.raises(pytest.skip.Exception):
        _load_consolidator_or_skip(tmp_path / "absent_consolidate.py")
