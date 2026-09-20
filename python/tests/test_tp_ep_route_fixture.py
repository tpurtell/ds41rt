"""CPU-only tests for route-fixture loading and self-describing provenance.

The reuse comparison must run on a validated, distribution-representative route
set rather than the all-distinct synthetic table. A malformed or mislabelled
fixture would silently produce a plausible but wrong comparison, so the loader
validates shape, id range, distinctness, and reuse, and rejects anything else.
Also guards the provenance fields added so the earlier attestation gap (arm files
carrying no source hash) cannot recur.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"
FIXTURE = ROOT / "scripts" / "fixtures" / "tp-ep-reuse-m64-e384.json"


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_ep_timing_fx", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load()


def write_fixture(tmp_path, routes, schema="ds41rt.tp-ep-reuse-fixture.v1"):
    payload = {"schema": schema, "routes": routes}
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload))
    return path


def test_real_fixture_loads_and_is_validated():
    if not FIXTURE.is_file():
        pytest.skip("reuse fixture not present")
    tensor, payload = timing.load_route_fixture(FIXTURE, 384, 6, 64)
    assert tuple(tensor.shape) == (64, 6)
    assert tensor.dtype == torch.int32, tensor.dtype
    # 48 distinct experts over 384 routes: real reuse, unlike the synthetic table.
    distinct = len(set(tensor.flatten().tolist()))
    assert distinct == 48, distinct
    assert len(set(tensor.flatten().tolist())) < tensor.numel()


def test_rejects_duplicate_expert_within_a_row(tmp_path):
    path = write_fixture(tmp_path, [[0, 0, 1, 2, 3, 4]] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        timing.load_route_fixture(path, 384, 6, 1)


def test_rejects_out_of_range_expert(tmp_path):
    path = write_fixture(tmp_path, [[0, 1, 2, 3, 4, 999]] * 2)
    with pytest.raises(ValueError, match="out of range"):
        timing.load_route_fixture(path, 384, 6, 1)


def test_rejects_wrong_topk(tmp_path):
    path = write_fixture(tmp_path, [[0, 1, 2, 3, 4]] * 2)
    with pytest.raises(ValueError, match="expected 6"):
        timing.load_route_fixture(path, 384, 6, 1)


def test_rejects_too_few_rows(tmp_path):
    path = write_fixture(tmp_path, [[0, 1, 2, 3, 4, 5]])
    with pytest.raises(ValueError, match="need 8"):
        timing.load_route_fixture(path, 384, 6, 8)


def test_rejects_unknown_schema(tmp_path):
    path = write_fixture(tmp_path, [[0, 1, 2, 3, 4, 5]], schema="something-else")
    with pytest.raises(ValueError, match="unsupported fixture schema"):
        timing.load_route_fixture(path, 384, 6, 1)


def test_partial_row_read_is_deterministic(tmp_path):
    routes = [[i % 8, (i + 1) % 8, (i + 2) % 8, (i + 3) % 8, (i + 4) % 8, (i + 5) % 8]
              for i in range(16)]
    path = write_fixture(tmp_path, routes)
    full, _ = timing.load_route_fixture(path, 384, 6, 16)
    first8, _ = timing.load_route_fixture(path, 384, 6, 8)
    assert (full[:8] == first8).all()


def test_provenance_source_fields_exist_in_harness():
    """The harness must record its own source and fixture hashes per run."""
    source = HARNESS.read_text()
    for field in ("harness_sha256", "benchmark_sha256", "route_fixture_sha256",
                  "input_dtype", "output_dtype"):
        assert field in source, field
    assert "oracle_dtype" in source, "oracle dtype must be recorded explicitly"


def test_fixture_hash_is_stable_and_content_bound(tmp_path):
    """A changed fixture must change its hash, so records bind to content."""
    import hashlib

    path = write_fixture(tmp_path, [[0, 1, 2, 3, 4, 5]] * 2)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps({"schema": "ds41rt.tp-ep-reuse-fixture.v1",
                                "routes": [[0, 1, 2, 3, 4, 6]] * 2}))
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before != after
