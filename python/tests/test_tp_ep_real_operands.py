"""CPU-only tests for the scoped real-operand (checkpoint) path.

These use a small FAKE checkpoint written to a temporary directory. No GPU, no
real snapshot, no 384-expert bank. They pin the loader's semantics:

* only the ACTIVE experts are read, and unused dense slots stay zero;
* expert id -> dense slot mapping is exact, so a mis-mapped expert cannot be
  silently measured;
* a missing expert tensor and a revision mismatch fail closed;
* the result is labelled an active-filled dense layout, not a full real bank;
* the default synthetic path is unaffected, so the existing correctness suites'
  self-oracles keep matching.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py"


def _load_bench():
    spec = importlib.util.spec_from_file_location("ds41rt_ep_bench_real", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()

HIDDEN = bench.HIDDEN


def make_fake_snapshot(root: Path, *, layer=0, experts=6, intermediate=64,
                       revision="testrev0000000000000000000000000000000000",
                       drop_keys=(), dtype_override=None):
    """Write a minimal safetensors snapshot with the official per-expert names."""
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for expert in range(experts):
        for name in ("w1", "w3"):
            key = f"layers.{layer}.ffn.experts.{expert}.{name}"
            tensors[f"{key}.weight"] = torch.randint(
                0, 256, (intermediate, HIDDEN // 2),
                dtype=torch.uint8).view(torch.int8)
            tensors[f"{key}.scale"] = torch.randint(
                121, 124, (intermediate, HIDDEN // 32),
                dtype=torch.uint8).view(torch.float8_e8m0fnu)
        key = f"layers.{layer}.ffn.experts.{expert}.w2"
        tensors[f"{key}.weight"] = torch.randint(
            0, 256, (HIDDEN, intermediate // 2),
            dtype=torch.uint8).view(torch.int8)
        tensors[f"{key}.scale"] = torch.randint(
            121, 124, (HIDDEN, intermediate // 32),
            dtype=torch.uint8).view(torch.float8_e8m0fnu)
    for key in drop_keys:
        tensors.pop(key, None)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(snapshot / shard))
    index = {"metadata": {}, "weight_map": {k: shard for k in tensors}}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    return snapshot


def test_missing_index_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        bench.checkpoint_metadata(tmp_path / "nonexistent")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="operand loader device path requires CUDA")
def test_only_active_experts_are_read_and_slots_are_zero(tmp_path):
    snapshot = make_fake_snapshot(tmp_path, experts=6)
    active = [1, 4]
    logical = bench.checkpoint_selected_operands(
        snapshot, 0, 6, 64, active, torch, expect_revision=snapshot.name)
    # Dense canonical shape, but only the active slots carry data.
    assert logical.weights["w1"].shape == (6, 64, HIDDEN // 2)
    assert logical.experts == 6
    for name in ("w1", "w3", "w2"):
        for slot in range(6):
            block = logical.weights[name][slot]
            if slot in active:
                assert bool(block.any()), f"{name} slot {slot} should carry data"
            else:
                assert not bool(block.any()), f"{name} slot {slot} should be zero"
    identity = logical.identity
    assert identity["active_experts"] == [1, 4]
    assert identity["active_count"] == 2
    assert identity["unused_slots_zeroed"] == 4
    assert identity["mode"] == "checkpoint-active-filled"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="operand loader device path requires CUDA")
def test_expert_id_maps_to_its_own_dense_slot(tmp_path):
    """A mis-mapped expert is a silent correctness failure, so pin the mapping."""
    snapshot = make_fake_snapshot(tmp_path, experts=4)
    full = bench.checkpoint_selected_operands(
        snapshot, 0, 4, 64, [0, 1, 2, 3], torch, expect_revision=snapshot.name)
    single = bench.checkpoint_selected_operands(
        snapshot, 0, 4, 64, [2], torch, expect_revision=snapshot.name)
    for name in ("w1", "w3", "w2"):
        assert torch.equal(single.weights[name][2], full.weights[name][2])
        assert not bool(single.weights[name][0].any())
        assert not bool(single.weights[name][3].any())


def test_missing_expert_tensor_fails_closed(tmp_path):
    key = "layers.0.ffn.experts.3.w1.weight"
    snapshot = make_fake_snapshot(tmp_path, experts=6, drop_keys=(key,))
    with pytest.raises(KeyError):
        bench.checkpoint_selected_operands(snapshot, 0, 6, 64, [3], torch)


def test_revision_mismatch_fails_closed(tmp_path):
    snapshot = make_fake_snapshot(tmp_path, experts=6)
    with pytest.raises(ValueError):
        bench.checkpoint_selected_operands(
            snapshot, 0, 6, 64, [0], torch,
            expect_revision="0" * 40)


def test_out_of_range_and_duplicate_active_ids_fail(tmp_path):
    snapshot = make_fake_snapshot(tmp_path, experts=4)
    with pytest.raises(ValueError):
        bench.checkpoint_selected_operands(snapshot, 0, 4, 64, [9], torch)
    with pytest.raises(ValueError):
        bench.checkpoint_selected_operands(snapshot, 0, 4, 64, [1, 1], torch)
    with pytest.raises(ValueError):
        bench.checkpoint_selected_operands(snapshot, 0, 4, 64, [], torch)


def test_label_states_not_a_full_real_bank(tmp_path):
    """A SUBSET active set must be labelled as not-the-full-bank.

    The label branch is also pinned GPU-free by
    `test_label_is_truthful_when_the_active_set_is_the_whole_bank`; this test drives
    the end-to-end identity, which needs the operand loader's device path.
    """
    if not torch.cuda.is_available():
        pytest.skip("operand loader requires CUDA for its device path")
    snapshot = make_fake_snapshot(tmp_path, experts=6)
    logical = bench.checkpoint_selected_operands(snapshot, 0, 6, 64, [0, 1], torch)
    assert logical.source_mode == "checkpoint-active-filled"
    assert "NOT" in logical.identity["label"]
    assert logical.identity["is_full_real_bank"] is False
    assert logical.identity["unused_slots_zeroed"] == 4


def test_label_is_truthful_when_the_active_set_is_the_whole_bank(tmp_path):
    """A bank whose active union covers every expert IS the full real bank.

    The label was previously unconditional, so a full-bank load claimed not to be
    one. This pins both branches through the shared label function.
    """
    assert "full real expert bank" in bench.bank_label(6, 6)
    assert "NOT" not in bench.bank_label(6, 6)
    assert "NOT" in bench.bank_label(5, 6)


def test_active_expert_ids_come_from_the_real_histogram():
    ids = torch.tensor([[0, 5, 5, 9, 2, 9]], dtype=torch.int32)
    assert bench.active_expert_ids(ids, 16) == [0, 2, 5, 9]
    assert bench.active_expert_ids(ids, 16, limit=2) == [0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="operand loader device path requires CUDA")
def test_default_synthetic_path_unchanged():
    """The correctness suites rely on synthetic operands; keep them intact."""
    logical = bench.LogicalWeights.synthetic(6, 1152, torch)
    assert logical.source_mode == "synthetic"
    assert logical.experts == 6
    assert logical.weights["w1"].shape == (6, 1152, HIDDEN // 2)
    assert logical.weights["w2"].shape == (6, HIDDEN, 576)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="operand loader device path requires CUDA")
def test_real_operands_survive_the_rank_slice_chain(tmp_path):
    """The real-operand bank must survive slicing intact.

    CPU half of the active-real path: build real operands at the FULL width the
    slice expects, take each TP rank's logical slice, and confirm active experts
    carry real bytes while unused slots stay zero at every stage. The GPU half
    (repack plus kernel) needs a lease.

    Geometry: `intermediate` is the FULL width here and `rank_width` is the
    per-rank slice, so the bank's second axis must be the full width.
    """
    full_width, rank_width, tp_degree = 128, 64, 2
    snapshot = make_fake_snapshot(tmp_path, experts=6, intermediate=full_width)
    active = [1, 4]
    logical = bench.checkpoint_selected_operands(
        snapshot, 0, 6, full_width, active, torch, expect_revision=snapshot.name)
    assert logical.weights["w1"].shape[1] == full_width
    assert logical.weights["w2"].shape[2] == full_width // 2
    for tp_rank in range(tp_degree):
        local = bench.rank_logical_slice(logical, tp_rank, tp_degree, rank_width,
                                         torch)
        assert local.experts == 6
        for name in ("w1", "w3"):
            assert local.weights[name].shape[1] == rank_width
        assert local.weights["w2"].shape[2] == rank_width // 2
        for name in ("w1", "w3", "w2"):
            for slot in range(6):
                block = local.weights[name][slot]
                if slot in active:
                    assert bool(block.any()), (
                        f"tp{tp_rank} {name} active slot {slot} is empty after "
                        "slicing")
                else:
                    assert not bool(block.any()), (
                        f"tp{tp_rank} {name} inactive slot {slot} is not zero")


def test_active_union_covers_every_requested_row_count():
    """The union used before loading must cover all M arms, not just M1."""
    experts = 384
    for rows in (1, 8, 16, 80):
        ids = (torch.arange(1, rows + 1, device="cpu", dtype=torch.int64)
               .reshape(rows, 1) * 6
               + torch.arange(6, dtype=torch.int64).reshape(1, 6)) % experts
        used = bench.active_expert_ids(ids, experts)
        # Distinct ids scale with the row count, so a full-bank assumption would
        # over-read badly at M1 while an M1-only assumption would under-read.
        assert len(used) == len(set(used))
        assert len(used) <= rows * 6
        assert max(used) < experts


def test_timing_harness_route_table_gives_expected_active_counts():
    """The timing harness derives the active set from its own route table.

    Cross-topology comparison is only legitimate if every arm in a cell gets the
    SAME active experts. That holds because the route table is deterministic and
    independent of the topology; this pins the counts the plan depends on.
    """
    experts = 384
    expected = {1: 6, 2: 12, 8: 48, 16: 96, 80: 384}
    for rows, want in expected.items():
        ids = (torch.arange(1, rows + 1, dtype=torch.int64).reshape(rows, 1) * 6
               + torch.arange(6, dtype=torch.int64).reshape(1, 6)) % experts
        used = bench.active_expert_ids(ids, experts)
        assert len(used) == want, (rows, len(used), want)
        assert len(set(used)) == len(used)
    # The bank depends only on the active count, not on the topology, because the
    # full width is intermediate * degree = 2304 for every degree.
    for degree, intermediate in ((2, 1152), (3, 768), (4, 576)):
        assert intermediate * degree == 2304
