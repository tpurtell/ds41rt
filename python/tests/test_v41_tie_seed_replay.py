"""CPU-only tests for the tie-seed replay diagnostic.

Covers the production-equivalent ordered reduction (including non-finite
rejection), the owner-export validation, the pure rank-input mapping and the
net result. No GPU, no native library, no checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python" / "tools" / "qualify_v41_tie_seed_replay.py"


def _load():
    spec = importlib.util.spec_from_file_location("_v41_tie_seed_replay_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owners(tp, ep, experts, *, differing):
    owners = [255] * 384
    for index, expert in enumerate(experts):
        owners[expert] = index % ep
    if differing and ep > 1:
        # Move the first expert to a different group.
        owners[experts[0]] = (owners[experts[0]] + 1) % ep
    return owners


def _rank_mask(tp, ep, experts, owners):
    mask = []
    for rank in range(tp * ep):
        group, tp_rank = rank // tp, rank % tp
        ids, weights = [], []
        for expert in experts:
            if owners[expert] == group:
                ids.append(expert)
                weights.append(1.0 / len(experts))
            else:
                ids.append(384)
                weights.append(0.0)
        mask.append({"rank": rank, "group": group, "tp_rank": tp_rank,
                     "executor_id": rank + 1, "ids": ids, "weights": weights})
    return mask


def _export(tp, ep, *, differing, cross_identical=False):
    experts = [0, 383, 1, 2, 3, 4]
    same_owners = _owners(tp, ep, experts, differing=False)
    cross_owners = same_owners if (not differing or cross_identical) else _owners(tp, ep, experts, differing=True)
    same_hash = "a" * 16
    cross_hash = same_hash if cross_owners == same_owners else "b" * 16
    base = {
        "schema": "v41-tie-seed-owner-export-v1",
        "topology": f"tp{tp}ep{ep}",
        "tp": tp, "ep": ep, "layer": 20, "fixed_experts": experts,
        "ids_scanned": 2, "same_id": 1, "cross_id": 2,
        "owners_differ": cross_owners != same_owners,
        "case_same_id": {
            "request_id": 1, "header_request_id": 1, "seed": 7,
            "owners": same_owners, "owners_hash": same_hash,
            "encoded_words": [0] * 6, "decoded_owners": [0] * 6,
            "rank_mask": _rank_mask(tp, ep, experts, same_owners),
        },
        "case_cross_id": {
            "request_id": 2, "header_request_id": 2, "seed": 9,
            "owners": cross_owners, "owners_hash": cross_hash,
            "encoded_words": [0] * 6, "decoded_owners": [0] * 6,
            "rank_mask": _rank_mask(tp, ep, experts, cross_owners),
        },
    }
    return base


def test_reducer_selfcheck_matches_ordered_semantics() -> None:
    module = _load()
    result = module.reducer_selfcheck()
    assert result["cancellation_ordered_bf16"] == 1.0
    assert result["six_slot_compact"] == 260.0
    assert result["torch_bf16_agrees"] is True


def test_bfloat16_rejects_non_finite() -> None:
    module = _load()
    import numpy as np

    with pytest.raises(ValueError):
        module.float2bfloat16_rn(np.array([float("nan")], dtype=np.float32))
    with pytest.raises(ValueError):
        module.float2bfloat16_rn(np.array([float("inf")], dtype=np.float32))


def test_net_result_is_ordered_rank_add_with_one_rounding() -> None:
    module = _load()
    import numpy as np

    planes = [
        np.array([[16777216.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[-16777216.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
    ]
    assert module.net_result(planes)[0, 0] == 1.0


def test_valid_export_loads_and_analyses() -> None:
    module = _load()
    export = _export(2, 2, differing=True)
    module.load_export_json(export)
    analysis = module.analyse_export(export)
    assert analysis["owner_delta_experts"] == 1
    assert analysis["rank_mask_delta_ranks"] >= 1
    assert analysis["ep1_negative"] is False


def test_ep1_export_must_be_seed_insensitive() -> None:
    module = _load()
    export = _export(4, 1, differing=False)
    module.load_export_json(export)
    analysis = module.analyse_export(export)
    assert analysis["ep1_negative"] is True
    assert analysis["owner_delta_experts"] == 0
    assert analysis["rank_mask_delta_ranks"] == 0


def test_invalid_exports_are_rejected() -> None:
    module = _load()
    # Cross-id pair that does not differ at EP>1.
    bad = _export(2, 2, differing=True, cross_identical=True)
    with pytest.raises(AssertionError):
        module.load_export_json(bad)
    # Owned route delivered as sentinel.
    bad = _export(2, 2, differing=True)
    bad["case_same_id"]["rank_mask"][0]["ids"][0] = 384
    with pytest.raises(AssertionError):
        module.load_export_json(bad)
    # Unowned route delivered as a real expert.
    bad = _export(2, 2, differing=True)
    bad["case_same_id"]["rank_mask"][0]["ids"][1] = 383
    with pytest.raises(AssertionError):
        module.load_export_json(bad)
    # Non group-major rank mapping (rank 1 must be tp_rank 1, not 0).
    bad = _export(2, 2, differing=True)
    bad["case_same_id"]["rank_mask"][1]["tp_rank"] = 0
    with pytest.raises(AssertionError):
        module.load_export_json(bad)


def test_rank_inputs_maps_group_major() -> None:
    module = _load()
    export = _export(2, 2, differing=True)
    case = export["case_same_id"]
    group, tp_rank, ids, weights = module.rank_inputs(case, 3, 2, 2)
    assert (group, tp_rank) == (1, 1)
    assert len(ids) == 6 and len(weights) == 6


def test_bf16_bits_distinguishes_signed_zero_and_hashes_planes() -> None:
    module = _load()
    import numpy as np

    pos = module.bf16_bits(np.array([0.0], dtype=np.float32))
    neg = module.bf16_bits(np.array([-0.0], dtype=np.float32))
    assert int(pos[0]) == 0x0000 and int(neg[0]) == 0x8000
    # Float equality would equate the two; the auditable bit hash must not.
    assert module.plane_sha256_bits(pos) != module.plane_sha256_bits(neg)
    values = np.array([1.0, -2.0, 261.0, 0.00390625], dtype=np.float32)
    assert np.array_equal(module.bf16_bits(values),
                          module.bf16_bits(module.float2bfloat16_rn(values)))
    with pytest.raises(ValueError):
        module.bf16_bits(np.array([float("nan")], dtype=np.float32))


def test_arena_slot_is_identity_for_nonconsecutive_experts() -> None:
    module = _load()
    experts = [0, 383, 1, 2, 3, 4]
    slots = [module.arena_slot(expert) for expert in experts]
    # Identity mapping: no base compaction, so slot == the transport expert id.
    assert slots == experts
    assert len(set(slots)) == 6
    # A compacted base would have produced 378..383, which the native ABI would
    # read as empty slots for ids 0..4.
    assert slots != list(range(378, 384))


def test_abi_layout_branches_for_v2_routes_and_v3_tokens() -> None:
    module = _load()
    kind, symbol, shape = module.abi_layout(False, 2)
    assert kind == "fp32_routes"
    assert symbol == "ds41rt_v41_compact_routes_bf16_async"
    assert shape == (1, 6, 5120)
    kind, symbol, shape = module.abi_layout(True, 3)
    assert kind == "fp32_tokens"
    assert symbol == "ds41rt_v41_compact_tokens_bf16_async"
    assert shape == (1, 5120)
    with pytest.raises(AssertionError):
        module.abi_layout(True, 2)
    with pytest.raises(AssertionError):
        module.abi_layout(False, 3)


class _FakeOutput:
    def __getitem__(self, item):
        return self

    def reshape(self, shape):
        return self

    def data_ptr(self):
        return 0


def _fake_native_module(log, abi_version):
    import types

    class FakeNative:
        def __init__(self, *args, **kwargs):
            log.append("construct")
            self.token_accumulation = abi_version == 3
            self.info = types.SimpleNamespace(abi_version=abi_version)
            self.output = _FakeOutput()

        def run(self, rows):
            log.append("run")

    module = types.ModuleType("_v41_expert_native")
    module.Native = FakeNative
    module.check = lambda code: None
    return module


def _fake_torch(log):
    import sys
    import types

    class FakeTensor:
        def data_ptr(self):
            return 0

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            import numpy as np

            return np.ones((1, 5120), dtype=np.float32)

    class FakeFinite:
        def all(self):
            log.append("read_output")
            return True

    class FakeCuda:
        def current_stream(self):
            return types.SimpleNamespace(cuda_stream=0)

        def synchronize(self):
            log.append("sync")

        def empty_cache(self):
            pass

    class FakeTorch:
        int32 = 1
        float32 = 2
        bfloat16 = 3

        def tensor(self, *args, **kwargs):
            return FakeTensor()

        def empty(self, *args, **kwargs):
            return FakeTensor()

        def isfinite(self, value):
            return FakeFinite()

        @property
        def cuda(self):
            return FakeCuda()

    module = types.ModuleType("_v41_expert_native_fake_torch")
    return FakeTorch()


def test_run_precedes_output_read_and_compaction(monkeypatch) -> None:
    module = _load()
    import sys
    import types

    log: list[str] = []
    fake_native = _fake_native_module(log, abi_version=2)
    monkeypatch.setitem(sys.modules, "_v41_expert_native", fake_native)
    monkeypatch.setattr(module, "quantize_wire", lambda x, capacity, torch: None)

    class FakeLib:
        def __getattr__(self, name):
            def call(*args):
                log.append("compact")
                return 0

            return call

    case = _export(2, 2, differing=True)["case_same_id"]
    plane, info = module._run_rank_plane(
        FakeLib(), 2, None, case, 0, 2, 2, None, _fake_torch(log))
    assert log.index("run") < log.index("read_output"), log
    assert log.index("run") < log.index("compact"), log
    assert info["output_kind"] == "fp32_routes"
    assert plane.shape == (1, 5120)


def test_artifact_slug_is_filesystem_safe() -> None:
    module = _load()
    assert module.artifact_slug(2, 2) == "tp2ep2"
    assert module.artifact_slug(4, 1) == "tp4ep1"
    assert module.artifact_slug(3, 2) == "tp3ep2"
    for slug in ("tp2ep2", "tp4ep1"):
        assert " " not in slug and "{" not in slug and "/" not in slug
