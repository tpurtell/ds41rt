"""CPU-only validation tests for the TP6 topology in `bench_tp_ep_kernel.py`.

These exercise the real `parse_args` guard, not source text: the CLI must accept
`--topologies tp6` and the tile widths the slice kernel can actually compile,
and must reject a width that is not a real export tile or that would leave a
partial slice on the per-rank intermediate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "python" / "tools" / "bench_tp_ep_kernel.py"


def _load():
    torch = pytest.importorskip("torch")
    assert torch is not None
    import sys

    tools = str(ROOT / "python" / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        spec = importlib.util.spec_from_file_location("ds41rt_tp6_bench", HARNESS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer/b12x unavailable: {error}")
    return module


def _args(topology: str, widths: str, rows: str = "8"):
    return [
        "--topologies", topology, "--ep-degree", "1", "--widths", widths,
        "--capacity", "80", "--operands", "synthetic",
        "--output", "/tmp/ds41rt-tp6-parse-test.json", "--rows", rows,
    ]


def test_geometry_table_carries_the_pure_tp6_split() -> None:
    module = _load()
    geometry = {tag: (degree, intermediate) for degree, intermediate, tag
                in module.TP_GEOMETRIES}
    assert geometry["tp6"] == (6, 384)
    assert geometry["tp4"] == (4, 576)
    assert geometry["tp3"] == (3, 768)
    assert geometry["tp2"] == (2, 1152)
    # Every geometry is the official intermediate 2304 split by its degree.
    for degree, intermediate in geometry.values():
        assert degree * intermediate == 2304


@pytest.mark.parametrize("widths", ["192", "128", "64", "64,128,192"])
def test_tp6_accepts_every_supported_slice_width(widths: str) -> None:
    module = _load()
    options = module.parse_args(_args("tp6", widths))
    assert options.widths == [int(x) for x in widths.split(",")]
    # tp6 is a distinct, selectable topology tag.
    assert "tp6" in options.topologies


@pytest.mark.parametrize("widths", ["96", "256", "0", "-64"])
def test_rejects_a_width_that_is_not_an_export_tile(widths: str) -> None:
    module = _load()
    with pytest.raises(SystemExit) as excinfo:
        module.parse_args(_args("tp6", widths))
    assert excinfo.value.code == 2


def test_rejects_a_width_that_does_not_tile_the_intermediate() -> None:
    module = _load()
    # TP4's 576 is not divisible by 128, so a 128-wide slice would leave a
    # partial tile; TP2/TP3/TP6 are all divisible by every supported width.
    with pytest.raises(SystemExit):
        module.parse_args(_args("tp4", "128"))
    for topology in ("tp2", "tp3", "tp6"):
        options = module.parse_args(_args(topology, "128"))
        assert options.widths == [128]


def test_rejects_an_unknown_topology() -> None:
    module = _load()
    with pytest.raises(SystemExit):
        module.parse_args(_args("tp5", "192"))


def test_tp6_topk_and_hidden_match_the_official_geometry() -> None:
    module = _load()
    # The harness builds every case with topk 6 and hidden 5120; tp6 must not be
    # special-cased to a different value, or route-plane sizing would drift.
    assert module.HIDDEN == 5120
    source = HARNESS.read_text(encoding="utf-8")
    assert "capacity, 6, logical" in source


# --------------------------------------------------------------------------- #
# The standalone per-topology benchmark (`benchmark_v41_ep_groups.py`)
# --------------------------------------------------------------------------- #

STANDALONE = ROOT / "python" / "tools" / "benchmark_v41_ep_groups.py"


def _load_standalone():
    torch = pytest.importorskip("torch")
    assert torch is not None
    import sys

    tools = str(ROOT / "python" / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        spec = importlib.util.spec_from_file_location("ds41rt_tp6_standalone", STANDALONE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer/b12x unavailable: {error}")
    return module


def test_standalone_benchmark_accepts_tp6_with_its_own_intermediate() -> None:
    module = _load_standalone()
    assert 6 in module.TP_DEGREES
    assert module.OFFICIAL_INTERMEDIATE == 2304
    options = module.parse_args([
        "--phase", "full", "--tp-degree", "6", "--intermediate", "384",
        "--output", "/tmp/ds41rt-tp6-standalone.json",
    ])
    assert options.tp_degree == 6 and options.intermediate == 384


def test_standalone_benchmark_rejects_a_degree_intermediate_mismatch() -> None:
    module = _load_standalone()
    # A degree/intermediate mismatch would slice a different width than the
    # topology claims, so it must fail closed rather than report a wrong shard.
    for degree, intermediate in ((6, 1152), (4, 384), (2, 576)):
        with pytest.raises(SystemExit):
            module.parse_args([
                "--phase", "full", "--tp-degree", str(degree),
                "--intermediate", str(intermediate),
                "--output", "/tmp/ds41rt-tp6-standalone.json",
            ])
    # An unsupported degree is refused by argparse itself.
    with pytest.raises(SystemExit):
        module.parse_args([
            "--phase", "full", "--tp-degree", "5", "--intermediate", "384",
            "--output", "/tmp/ds41rt-tp6-standalone.json",
        ])


def test_standalone_slice_helper_handles_the_tp6_intermediate() -> None:
    """`_slice_logical`/`rank_logical_slice` must tile a 384 rank exactly."""
    torch = pytest.importorskip("torch")
    module = _load_standalone()
    # Build a full-width logical set for the 2304 intermediate as CPU tensors and
    # slice all six ranks; the concatenation must reconstruct the original.
    full = 2304
    per_rank = 384
    gen = torch.Generator().manual_seed(0x61)
    weights = {
        "w1": torch.randint(0, 256, (1, full, 2560), generator=gen, dtype=torch.uint8),
        "w3": torch.randint(0, 256, (1, full, 2560), generator=gen, dtype=torch.uint8),
        "w2": torch.randint(0, 256, (1, 5120, full // 2), generator=gen,
                            dtype=torch.uint8),
    }
    scales = {
        name: torch.randint(121, 124, (*t.shape[:-1], t.shape[-1] // 16),
                            generator=gen, dtype=torch.uint8)
        for name, t in weights.items()
    }
    logical = module.LogicalWeights(weights, scales, "synthetic", {}, 1)
    pieces_w1, pieces_w2 = [], []
    for rank in range(6):
        sliced = module.rank_logical_slice(logical, rank, 6, per_rank, torch)
        assert sliced.weights["w1"].shape == (1, per_rank, 2560)
        assert sliced.weights["w2"].shape == (1, 5120, per_rank // 2)
        assert sliced.scales["w2"].shape == (1, 5120, per_rank // 32)
        pieces_w1.append(sliced.weights["w1"])
        pieces_w2.append(sliced.weights["w2"])
    assert torch.equal(torch.cat(pieces_w1, dim=1), weights["w1"])
    assert torch.equal(torch.cat(pieces_w2, dim=2), weights["w2"])
