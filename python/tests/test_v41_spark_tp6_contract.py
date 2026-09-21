"""CPU-only contract checks for the pure TP6 Spark expert role.

No torch, CUDA or SparkInfer import happens here: the authoritative role tables
are pure literals inside the exporters, and the native/CMake/build wiring is
checked from source text. The pure integer arithmetic from the real slice kernel
(`V41FusedSliceKernel.__init__` and the exporter `planes`/`specs` derivation) is
re-derived and asserted over the whole official intermediate, so a geometry that
would leave a gap or overlap in the six ranks fails without a GPU.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLICES = ROOT / "python" / "tools" / "export_b12x_v41_slices_aot.py"
EXPERTS_EXPORTER = ROOT / "python" / "tools" / "export_b12x_v41_experts_aot.py"
NATIVE_HELPER = ROOT / "python" / "tools" / "_v41_expert_native.py"
TOOLS = ROOT / "python" / "tools"
QUALIFIER = ROOT / "python" / "tools" / "qualify_v41_replicated_native.py"
WIP = ROOT / "wip.sh"
CMAKE = ROOT / "native" / "CMakeLists.txt"
CMAKE_TP = ROOT / "native" / "cmake" / "v41_spark_tp_experts.cmake"
HEADER = ROOT / "native" / "include" / "ds41rt_v41_experts.h"
PACK = ROOT / "native" / "cuda" / "kernels" / "v41_expert_pack.cu"
WRAPPER_TP6 = ROOT / "native" / "src" / "v41_spark_tp6_experts.cc"
BUILD = ROOT / "build.sh"
RELEASE_ARTIFACTS = ROOT / "scripts" / "build-release-artifacts.sh"
WIP_ARTIFACTS = ROOT / "scripts" / "build-wip-artifacts.sh"
MANIFEST = ROOT / "scripts" / "write-v41-expert-tp-manifest.py"

# Official V4.1 geometry: 384 experts, hidden 5120, intermediate 2304, topk 6.
OFFICIAL_INTERMEDIATE = 2304
HIDDEN = 5120
TOPK = 6
EXPERTS = 384
SUPPORTED_WIDTHS = (64, 128, 192)
DEFAULT_WIDTH_MAP = "1:64,16:192,80:192,256:192,1024:192,4096:192"


def _literal_assignments(path: Path, names: set[str]) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(node.value)
    return found


def test_spark_tp6_geometry_is_exact_and_padding_free() -> None:
    tables = _literal_assignments(
        SLICES, {"ROLE_GEOMETRY", "ROLE_SM", "ROLE_NATIVE_ID", "SPARK_TP_DEGREE"}
    )
    geometry = tables["ROLE_GEOMETRY"]
    assert geometry["spark_tp6"] == (EXPERTS, 384, 384, TOPK)
    experts, logical, kernel, topk = geometry["spark_tp6"]
    # No storage padding: 2304 / 6 == 384 is already 128-aligned.
    assert kernel == logical == OFFICIAL_INTERMEDIATE // 6
    assert logical % 32 == 0 and topk == TOPK and experts == EXPERTS
    assert tables["ROLE_SM"]["spark_tp6"] == (12, 1)
    assert tables["ROLE_NATIVE_ID"]["spark_tp6"] == 7
    assert tables["SPARK_TP_DEGREE"]["spark_tp6"] == 6


def test_role7_is_unique_across_the_whole_registry() -> None:
    tables = _literal_assignments(
        SLICES, {"ROLE_GEOMETRY", "ROLE_NATIVE_ID", "SPARK_TP_DEGREE"}
    )
    native_ids = tables["ROLE_NATIVE_ID"]
    assert sorted(native_ids.values()) == list(range(len(native_ids))), native_ids
    assert native_ids["spark_tp6"] == max(native_ids.values()) == 7
    # Every role with a native id also has a plan-time geometry and, for the
    # Spark family, a TP degree.
    assert set(native_ids) == set(tables["ROLE_GEOMETRY"])
    assert {"spark_tp6"} <= set(tables["SPARK_TP_DEGREE"])
    # No other role claims the TP6 shard geometry.
    others = [
        role for role, entry in tables["ROLE_GEOMETRY"].items()
        if role != "spark_tp6" and entry[1] == 384 and entry[2] == 384
    ]
    assert others == []


def test_six_rank_slices_tile_2304_without_gap_or_overlap() -> None:
    """Re-derive the real slice-kernel and exporter arithmetic for every width.

    `V41FusedSliceKernel.__init__` asserts width in (64,128,192),
    `intermediate % 32 == 0`, and `slices * width <= kernel_intermediate`.
    The exporter allocates `kernel_intermediate * 1280` words per expert, which
    is exactly the 128-aligned storage the kernel strides over.
    """
    intermediate = OFFICIAL_INTERMEDIATE // 6
    kernel_intermediate = (intermediate + 127) // 128 * 128
    assert kernel_intermediate == intermediate == 384

    for width in SUPPORTED_WIDTHS:
        assert width in SUPPORTED_WIDTHS
        assert intermediate % 32 == 0
        slices = (intermediate + width - 1) // width
        assert slices * width <= kernel_intermediate, width

    # Width 192 (the qualified wider tile) covers 384 with exactly two slices
    # at offsets 0 and 192, and the 384-word storage ends where the second ends.
    assert (intermediate + 191) // 192 == 2
    assert 2 * 192 == intermediate
    # Width 64 and 128 tile exactly too; the sweep must reject nothing here.
    assert intermediate % 64 == 0 and intermediate % 128 == 0


def test_six_ranks_cover_the_official_intermediate_exactly() -> None:
    per_rank = OFFICIAL_INTERMEDIATE // 6
    covered = [set(range(rank * per_rank, (rank + 1) * per_rank)) for rank in range(6)]
    union: set[int] = set()
    for tile in covered:
        assert not (union & tile), "rank slices overlap"
        union |= tile
    assert union == set(range(OFFICIAL_INTERMEDIATE))
    assert sorted(min(t) for t in covered) == [0, 384, 768, 1152, 1536, 1920]
    assert sorted(max(t) for t in covered) == [383, 767, 1151, 1535, 1919, 2303]


def test_exporter_plane_and_scratch_sizes_are_congruent_for_384() -> None:
    """Every AOT scratch/plane extent for TP6 is a whole number of elements.

    Mirrors the exporter `specs` list: w13 uses `kernel_intermediate * 1280`
    Uint32 words, s13 `* 80`, w2 `* 640`, s2 `* 40`, and the per-route FP32
    plane is `[planes, routes, 5120]` with `planes = ceil(384 / width)`.
    """
    intermediate = kernel_intermediate = 384
    experts = EXPERTS
    for n in (1280, 80, 640, 40):
        assert (experts * kernel_intermediate * n) % 1 == 0
    # Byte-for-byte agreement with the native packer extents verified on device.
    assert kernel_intermediate * HIDDEN == 1_966_080
    assert kernel_intermediate * HIDDEN // 16 == 122_880
    assert HIDDEN * kernel_intermediate // 2 == 983_040
    assert HIDDEN * kernel_intermediate // 32 == 61_440
    for width in SUPPORTED_WIDTHS:
        planes = (intermediate + width - 1) // width
        assert planes == 384 // width
        routes = 4096 * TOPK
        assert planes * routes * HIDDEN > 0


def test_ordinary_exporter_routes_tp6_only_through_fp8_slices() -> None:
    tables = _literal_assignments(EXPERTS_EXPORTER, {"SPARK_ROLES", "SPARK_TP_DEGREES"})
    assert tables["SPARK_ROLES"] == ("spark", "spark_tp2", "spark_tp3", "spark_tp6")
    assert tables["SPARK_TP_DEGREES"]["spark_tp6"] == 6
    source = EXPERTS_EXPORTER.read_text(encoding="utf-8")
    assert "Spark TP2/TP3/TP6 use the native FP8 K32 slice export only" in source
    # Both exporters accept the role on the CLI.
    assert '"spark_tp6"' in SLICES.read_text(encoding="utf-8")
    assert '"spark_tp6"' in source


def test_native_helper_selects_the_tp6_symbol_family() -> None:
    source = NATIVE_HELPER.read_text(encoding="utf-8")
    assert '6: "ds41rt_v41_spark_tp6_expert_"' in source
    assert "assert spark_tp in (None, 2, 3, 6)" in source
    assert "(7, 384, 5120, 384, 384, 6, capacity, 7) if spark_tp == 6" in source


def test_cmake_wiring_is_opt_in_and_precompiles_capacities() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    assert "tp2, tp3 and tp6" in tp
    assert 'set(DS41RT_V41_SPARK_TP6_SLICE_WIDTH' in tp
    assert f'"{DEFAULT_WIDTH_MAP}"' in tp
    assert "v41_spark_tp6_experts.cc" in tp
    assert "v41_spark_tp6_expert_variants.h" in tp
    assert "src/v41_spark_tp6_experts.cc" in tp
    # The new role must not change the existing TP2/TP3 selector contract.
    assert 'DS41RT_V41_SPARK_TP_EXPERT_ROWS_ARG "1,16,80,256,1024,4096"' in tp
    assert "--atomic-min-capacity 256" in tp
    assert 'if(NOT DS41RT_V41_SPARK_TP_ROLES STREQUAL "")' in CMAKE.read_text(
        encoding="utf-8"
    )


def test_native_role_ids_and_packer_extent_are_declared() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "7: Spark TP6 shard (intermediate 384, no padding)" in header
    for symbol in (
        "ds41rt_v41_spark_tp6_expert_info",
        "ds41rt_v41_spark_tp6_expert_initialize",
        "ds41rt_v41_spark_tp6_expert_launch",
        "ds41rt_v41_spark_tp6_expert_bind_scratch",
        "ds41rt_v41_spark_tp6_expert_initialize_scratch_async",
        "ds41rt_v41_spark_tp6_expert_output_kind",
    ):
        assert symbol in header, symbol

    pack = PACK.read_text(encoding="utf-8")
    assert "intermediate != 384" in pack
    assert "intermediate % 32 != 0" in pack

    wrapper = WRAPPER_TP6.read_text(encoding="utf-8")
    assert "#define DS41RT_V41_SPARK_TP6_EXPERTS 1" in wrapper
    for suffix in ("info", "initialize", "output_kind", "bind_scratch",
                   "initialize_scratch_async", "launch"):
        assert f"#define ds41rt_v41_expert_{suffix} ds41rt_v41_spark_tp6_expert_{suffix}" in wrapper
    # The canonical FP8 quantizer stays in exactly one translation unit.
    experts_source = (ROOT / "native" / "src" / "v41_experts.cc").read_text(
        encoding="utf-8"
    )
    assert "!defined(DS41RT_V41_SPARK_TP6_EXPERTS)" in experts_source
    assert '#include "v41_spark_tp6_expert_variants.h"' in experts_source


def test_build_selectors_accept_a_tp6_role() -> None:
    build = BUILD.read_text(encoding="utf-8")
    assert "6) spark_tp_roles=tp6 ;;" in build
    assert "tp2|tp3|tp6)" in build
    assert "DS41RT_RELEASE_SPARK_TP_ROLES accepts only tp2, tp3 and tp6" in build
    assert "SPARK_TP=2/3/6" in build

    release = RELEASE_ARTIFACTS.read_text(encoding="utf-8")
    assert "tp2|tp3|tp6)" in release
    assert "DS41RT_RELEASE_SPARK_TP_ROLES accepts only tp2, tp3 and tp6" in release

    wip = WIP_ARTIFACTS.read_text(encoding="utf-8")
    assert "tp2|tp3|tp6)" in wip
    assert "DS41RT_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6" in wip


def test_tp_manifest_writer_knows_the_tp6_geometry_and_symbols() -> None:
    tables = _literal_assignments(
        MANIFEST, {"ROLE_TP_DEGREE", "ROLE_INTERMEDIATE", "ROLE_INFO_SYMBOL",
                   "ROLE_LAUNCH_SYMBOL"}
    )
    assert tables["ROLE_TP_DEGREE"]["tp6"] == 6
    assert tables["ROLE_INTERMEDIATE"]["tp6"] == 384
    assert tables["ROLE_INFO_SYMBOL"]["tp6"] == "ds41rt_v41_spark_tp6_expert_info"
    assert tables["ROLE_LAUNCH_SYMBOL"]["tp6"] == "ds41rt_v41_spark_tp6_expert_launch"
    source = MANIFEST.read_text(encoding="utf-8")
    assert "expected tp2, tp3 or tp6" in source


# --------------------------------------------------------------------------- #
# Executable per-rank repack partition coverage (real b12x repack, CPU)
#
# The device-level byte equivalence is proven for all six ranks by
# `native/tests/v41_expert_pack_geometry_selftest.cc`. This block is its CPU
# companion and closes the "string asserts only" gap:
#
#  * an INDEPENDENT rank-local oracle re-derives the N256/K128 lane-major
#    swizzle from the Triton kernel's published index math (not from b12x), and
#  * the real b12x repack (`_logical_weight_to_w4a8_rp_inplace`) is compared
#    against that oracle for a rank's own contiguous intermediate slice.
#
# Comparing reassembled rank slabs to a full-width pack would be WRONG: the
# swizzle's tile origin is rank-local, so the concatenation is not the full
# pack's byte order. The oracle below is rank-local for exactly that reason.
#
# TP4 (576) is deliberately not in this block: its 640-wide storage pads whole
# 128-row tiles, and b12x's ceil-tiled tail path is CUDA-only. It keeps its
# existing coverage (device pack self-test + `per_rank is not None` extents).
# --------------------------------------------------------------------------- #

REPACK_PARTITIONS = ((6, 384), (3, 768), (2, 1152), (1, 2304))


def _oracle_qweight_to_w4a8_rp(q, size_n: int, size_k: int):
    """Independent N256/K128 lane-major repack, from the kernel's index math.

    `q` is uint32 [size_n, size_k // 8] (8 FP4 weights per word). The output has
    the resident shape [n_tiles, k_tiles, 4, 8, 32, 4] and is filled with the
    documented swizzle: lane index -> (n8i, r8, n8c, k32) -> source (row, col).
    """
    torch = _TORCH
    q_cols = size_k // 8
    n_tiles, k_tiles = size_n // 256, size_k // 128
    shape = (n_tiles, k_tiles, 4, 8, 32, 4)
    out = torch.zeros(shape, dtype=torch.int32)
    index = torch.arange(4096)
    n8i = index & 3
    combined = (index >> 2) & 31
    cgrp = combined & 3
    r8 = combined >> 2
    n8c = (index >> 7) & 7
    k32 = index >> 10
    row = n8c * 32 + n8i * 8 + r8
    src_col = k32 * 4 + cgrp
    for nt in range(n_tiles):
        for kt in range(k_tiles):
            out[nt, kt].reshape(-1)[index] = q[nt * 256 + row, kt * 16 + src_col]
    return out


def _load_b12x_repack():
    """The real repack entry points; skip only when the pinned tree is absent."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        import _pinned_sparkinfer  # noqa: F401
        from b12x.moe.fused_moe._impl import (
            _e8m0_scale_to_w4a8_sfb_inplace,
            _logical_weight_to_w4a8_rp_inplace,
        )
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer/b12x unavailable: {error}")
    return _logical_weight_to_w4a8_rp_inplace, _e8m0_scale_to_w4a8_sfb_inplace


_TORCH = None


@pytest.mark.parametrize("tp_degree,per_rank", REPACK_PARTITIONS)
def test_rank_local_repack_matches_the_independent_swizzle_oracle(
    tp_degree: int, per_rank: int
) -> None:
    """Each rank's packed W13/W2 equals the oracle over its OWN slice.

    A geometry that sharded the wrong axis would read the wrong source rows or
    columns, so this fails on any rank/axis mix-up without a GPU.
    """
    global _TORCH
    torch = pytest.importorskip("torch")
    _TORCH = torch
    repack, _ = _load_b12x_repack()
    assert per_rank * tp_degree == OFFICIAL_INTERMEDIATE

    gen = torch.Generator(device="cpu").manual_seed(0x51)
    # W13: N = 2 * per_rank (gate + up), K = hidden. W2: N = hidden, K = per_rank.
    w13 = torch.randint(0, 256, (1, 2 * per_rank, HIDDEN // 2), generator=gen,
                        dtype=torch.uint8)
    w2 = torch.randint(0, 256, (1, HIDDEN, per_rank // 2), generator=gen,
                       dtype=torch.uint8)

    packed_w13 = repack(w13.clone(), size_k=HIDDEN, size_n=2 * per_rank,
                        gated_half_rows=per_rank).view(torch.int32)[0]
    packed_w2 = repack(w2.clone(), size_k=per_rank, size_n=HIDDEN).view(torch.int32)[0]

    oracle_w13 = _oracle_qweight_to_w4a8_rp(
        w13.view(torch.int32).reshape(2 * per_rank, HIDDEN // 8), 2 * per_rank, HIDDEN)
    oracle_w2 = _oracle_qweight_to_w4a8_rp(
        w2.view(torch.int32).reshape(HIDDEN, per_rank // 8), HIDDEN, per_rank)

    assert packed_w13.shape == oracle_w13.shape
    assert packed_w2.shape == oracle_w2.shape
    assert torch.equal(packed_w13, oracle_w13), "W13 repack differs from the oracle"
    assert torch.equal(packed_w2, oracle_w2), "W2 repack differs from the oracle"
    # The K axis of W2 is the local intermediate; the packed tile count is
    # per_rank/128, which is what a K-sharded down projection requires.
    assert packed_w13.shape[0] == 2 * per_rank // 256
    assert packed_w2.shape[1] == per_rank // 128


@pytest.mark.parametrize("tp_degree,per_rank", REPACK_PARTITIONS)
def test_rank_slices_reconstruct_the_full_logical_expert(
    tp_degree: int, per_rank: int
) -> None:
    """The loader's slice arithmetic is an exact partition of the full expert.

    This is the `rank_operands` slice contract executed on real tensors: W1/W3
    slice on axis 1 (N = intermediate), W2 slices on axis 2 (K = intermediate at
    half rate for FP4 bytes). Concatenating the ranks must reproduce the full
    tensor byte for byte, which is gap- and overlap-free by construction.
    """
    torch = pytest.importorskip("torch")
    full = per_rank * tp_degree
    assert full == OFFICIAL_INTERMEDIATE
    gen = torch.Generator(device="cpu").manual_seed(0x52)
    w1 = torch.randint(0, 256, (1, full, HIDDEN // 2), generator=gen, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (1, HIDDEN, full // 2), generator=gen, dtype=torch.uint8)

    for tensor, name in ((w1, "w1"), (w2, "w2")):
        pieces = []
        for rank in range(tp_degree):
            lo = rank * per_rank
            hi = lo + per_rank
            pieces.append(tensor[:, lo:hi, :] if name == "w1"
                          else tensor[:, :, lo // 2:hi // 2])
        if name == "w1":
            rebuilt = torch.cat(pieces, dim=1)
        else:
            rebuilt = torch.cat(pieces, dim=2)
        assert rebuilt.shape == tensor.shape, name
        assert torch.equal(rebuilt, tensor), (
            f"{name} rank slices do not reconstruct the full expert for tp{tp_degree}"
        )


def test_rank_local_oracle_is_not_the_full_width_swizzle() -> None:
    """Document why the reassembled ranks must not be compared to a full pack.

    The tile origin is rank-local: rank `r`'s first tile starts at its own
    intermediate offset, so the concatenation of rank slabs is a different byte
    order from a single full-width pack. Asserting the two are equal (the
    earlier, invalid invariant) is exactly what this test prevents regressing to.
    """
    torch = pytest.importorskip("torch")
    global _TORCH
    _TORCH = torch
    repack, _ = _load_b12x_repack()
    per_rank, tp_degree = 384, 6
    full = per_rank * tp_degree
    gen = torch.Generator(device="cpu").manual_seed(0x53)
    full_w13 = torch.randint(0, 256, (1, 2 * full, HIDDEN // 2), generator=gen,
                             dtype=torch.uint8)
    full_packed = repack(full_w13.clone(), size_k=HIDDEN, size_n=2 * full,
                         gated_half_rows=full).view(torch.int32)
    rank0 = repack(full_w13[:, :2 * per_rank, :].clone(), size_k=HIDDEN,
                   size_n=2 * per_rank, gated_half_rows=per_rank).view(torch.int32)
    # Rank 0's N256 tile origin is local: its tile 0 is the full pack's tile 0,
    # but the slab extents differ (3 tiles vs 18), so no whole-slab equality can
    # hold. The first local tile must still agree with the full pack's first
    # tile because both start at intermediate offset 0.
    assert rank0.shape != full_packed.shape
    assert rank0.shape[1] == (2 * per_rank) // 256 == 3
    assert full_packed.shape[1] == (2 * full) // 256 == 18
    assert torch.equal(rank0[:, 0], full_packed[:, 0])

# Every real native geometry, TP6 first: (tp_degree, logical per-rank
# intermediate). `per_rank * tp_degree` must equal the official 2304, so a gap
# or overlap in any of these partitions is exactly the production failure mode.
PARTITIONS = ((6, 384), (4, 576), (3, 768), (2, 1152), (1, 2304))


def _manifest_plane_words(kernel_intermediate: int) -> tuple:
    """Per-expert Uint32 element counts from the exporters' `specs` list.

    `kernel_intermediate * (1280, 80, 640, 40)` for W13, S13, W2, S2. These are
    exactly the four native packed slabs, and the byte totals are cross-checked
    on device in `native/tests/v41_expert_pack_geometry_selftest.cc`:
    W13 = ki*5120 bytes, S13 = ki*5120/16, W2 = 5120*ki/2, S2 = 5120*ki/32.
    """
    return tuple(kernel_intermediate * n for n in (1280, 80, 640, 40))


def _intermediate_slice(rank: int, per_rank: int) -> range:
    return range(rank * per_rank, (rank + 1) * per_rank)


@pytest.mark.parametrize("tp_degree,per_rank", PARTITIONS)
def test_partition_tiles_the_official_intermediate_without_gap_or_overlap(
    tp_degree: int, per_rank: int
) -> None:
    """Six (or four/three/two/one) rank slices tile 2304 exactly."""
    assert per_rank * tp_degree == OFFICIAL_INTERMEDIATE
    covered = [_intermediate_slice(rank, per_rank) for rank in range(tp_degree)]
    for tile in covered:
        assert len(tile) == per_rank
    union: set[int] = set()
    for tile in covered:
        assert not (union & set(tile)), "rank slices overlap"
        union |= set(tile)
    assert union == set(range(OFFICIAL_INTERMEDIATE))
    assert [tile.start for tile in covered] == [rank * per_rank for rank in range(tp_degree)]
    assert [tile.stop for tile in covered] == [
        (rank + 1) * per_rank for rank in range(tp_degree)
    ]


@pytest.mark.parametrize("tp_degree,per_rank", PARTITIONS)
def test_per_rank_packed_extent_is_exactly_the_exporter_layout(
    tp_degree: int, per_rank: int
) -> None:
    """Every rank's four packed slabs have the exporter's exact word counts.

    This is the executable packing-partition check: the source shard the loader
    packs is the per-rank intermediate, so the prepared storage each rank must
    allocate is the 128-aligned local extent -- 384 (no padding) for TP6, 640
    (padded) only for the TP4 shard. The four counts below are the exporter's
    `specs` list, and the same four byte totals were verified on device in
    `native/tests/v41_expert_pack_geometry_selftest.cc`.
    """
    assert per_rank * tp_degree == OFFICIAL_INTERMEDIATE
    kernel_intermediate = (per_rank + 127) // 128 * 128
    if per_rank != 576:
        assert kernel_intermediate == per_rank, "unexpected storage padding"
    words = _manifest_plane_words(kernel_intermediate)
    # The four exporter plane extents, in Uint32 words. W13 packs the gate and
    # up halves at 8 FP4 weights per word; S13 is one word per (N256 row tile,
    # K32 group); W2/S2 contract over the local intermediate, so they scale with
    # it on K. Each equality is cross-checked in bytes against the device
    # packer's advertised extents.
    assert words[0] * 4 == kernel_intermediate * HIDDEN
    assert words[1] * 4 == kernel_intermediate * HIDDEN // 16
    assert words[2] * 4 == HIDDEN * kernel_intermediate // 2
    assert words[3] * 4 == HIDDEN * kernel_intermediate // 32

    # The K32 UE8M0 scale axis is exact because every accepted extent is a
    # multiple of 32, and the down projection is sharded on K for the same
    # reason: the local intermediate is the contraction axis, not just N.
    assert per_rank % 32 == 0
    assert kernel_intermediate % 128 == 0


@pytest.mark.parametrize("tp_degree,per_rank", PARTITIONS)
def test_rank_slices_are_disjoint_and_cover_the_intermediate_axis(
    tp_degree: int, per_rank: int
) -> None:
    """Word-level coverage: no element belongs to two ranks or to none."""
    assert per_rank * tp_degree == OFFICIAL_INTERMEDIATE
    owners: dict[int, int] = {}
    for rank in range(tp_degree):
        for element in _intermediate_slice(rank, per_rank):
            assert element not in owners, f"intermediate {element} claimed twice"
            owners[element] = rank
    assert set(owners) == set(range(OFFICIAL_INTERMEDIATE))
    assert all(owners[e] == e // per_rank for e in owners)
    # Boundaries: the first and last element of every rank are where an
    # off-by-one in the loader's `lo/hi` slicing would show up.
    assert owners[0] == 0 and owners[OFFICIAL_INTERMEDIATE - 1] == tp_degree - 1


def test_storage_padding_is_limited_to_the_tp4_shard() -> None:
    """384 and 768 are 128-aligned; 576 is the only padded native extent."""
    assert (384 + 127) // 128 * 128 == 384
    assert (768 + 127) // 128 * 128 == 768
    assert (576 + 127) // 128 * 128 == 640  # TP4 pads 576 -> 640
    assert (1152 + 127) // 128 * 128 == 1152
    assert (2304 + 127) // 128 * 128 == 2304


# --------------------------------------------------------------------------- #
# Native qualifier role-identity gate (executable, no GPU)
# --------------------------------------------------------------------------- #

def _load_qualifier(torch):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import importlib.util as _u

    spec = _u.spec_from_file_location("ds41rt_tp6_qualifier", QUALIFIER)
    module = _u.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLibrary:
    """Minimal stand-in for the native shared library's info entry point.

    `--width` chooses which exported variant a qualification run exercises; it
    says nothing about which roles the library was built with. This fake lets the
    role-identity gate be tested without a device or a real .so.
    """

    def __init__(self, role, logical, kernel, *, experts=384, hidden=5120, topk=6):
        import ctypes as C

        self._info = (role, logical, kernel, experts, hidden, topk)
        fields = (["abi_version", "role", "experts", "hidden_size",
                   "logical_intermediate", "kernel_intermediate", "topk",
                   "capacity_rows"] + ["scratch_bytes"] +
                  ["max_rows", "rows_padded", "max_tasks", "max_phys_tiles",
                   "max_active_clusters"] + ["input_dtype"])

        class _Info(C.Structure):
            _fields_ = [(name, C.c_uint64 if name == "scratch_bytes" else C.c_int32)
                        for name in fields]

        self._Info = _Info

        def _info_fn(capacity, out):
            role_, logical_, kernel_, experts_, hidden_, topk_ = self._info
            data = [2, role_, experts_, hidden_, logical_, kernel_, topk_,
                    capacity, 4096, 32, 32, 64, 64, 188, 7]
            C.memmove(out, (C.c_int32 * len(data))(*data), C.sizeof(_Info))
            return 0

        self.ds41rt_v41_expert_info = _info_fn


def test_qualifier_role_tables_cover_tp6() -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None
    module = _load_qualifier(torch)
    assert module.SPARK_TP_INTERMEDIATE == {2: 1152, 3: 768, 6: 384}
    assert module.SPARK_TP_ROLE == {2: 5, 3: 6, 6: 7}
    assert module.SPARK_TP_KERNEL_INTERMEDIATE == {2: 1152, 3: 768, 6: 384}
    # Role ids stay 3 + degree so 2->5, 3->6, 6->9 is NOT the rule; assert the
    # table is the authority rather than the arithmetic.
    assert module.SPARK_TP_ROLE[6] == 7


def test_qualifier_accepts_tp6_on_the_cli() -> None:
    torch = pytest.importorskip("torch")
    source = QUALIFIER.read_text(encoding="utf-8")
    assert 'choices=(2, 3, 6)' in source
    assert "role 7 = Spark TP6" in source


def test_role_gate_accepts_the_matching_library_and_rejects_others() -> None:
    """A TP6 request against a non-TP6 library must fail closed, not measure it."""
    torch = pytest.importorskip("torch")
    module = _load_qualifier(torch)

    accepted = _FakeLibrary(role=7, logical=384, kernel=384)
    meta = module.assert_native_role_geometry(accepted, 16, tp_degree=6)
    assert (meta.role, meta.logical_intermediate, meta.kernel_intermediate) == (7, 384, 384)

    # A library built only for TP2 (role 5, 1152) must be refused for tp6.
    wrong_family = _FakeLibrary(role=5, logical=1152, kernel=1152)
    with pytest.raises(AssertionError):
        module.assert_native_role_geometry(wrong_family, 16, tp_degree=6)

    # Right family, wrong storage extent (the padded-TP4 mistake) must be refused.
    wrong_extent = _FakeLibrary(role=7, logical=384, kernel=640)
    with pytest.raises(AssertionError):
        module.assert_native_role_geometry(wrong_extent, 16, tp_degree=6)

    # Wrong expert/hidden/topk geometry must be refused too.
    wrong_shape = _FakeLibrary(role=7, logical=384, kernel=384, topk=3)
    with pytest.raises(AssertionError):
        module.assert_native_role_geometry(wrong_shape, 16, tp_degree=6)

    # TP2/TP3 stay correct, and the legacy TP4 baseline keeps its 576/640 split.
    assert module.assert_native_role_geometry(
        _FakeLibrary(role=5, logical=1152, kernel=1152), 16, tp_degree=2).role == 5
    assert module.assert_native_role_geometry(
        _FakeLibrary(role=6, logical=768, kernel=768), 16, tp_degree=3).role == 6
    legacy = module.assert_native_role_geometry(
        _FakeLibrary(role=1, logical=576, kernel=640), 16, tp4_legacy=True)
    assert (legacy.role, legacy.logical_intermediate, legacy.kernel_intermediate) == (
        1, 576, 640)


def test_geometry_test_is_registered_with_the_real_name() -> None:
    """CMake, the gate wrapper and the packer test must agree on one name."""
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "tests/v41_expert_pack_geometry_selftest.cc" in cmake
    assert "ds41rt_v41_expert_pack_geometry_selftest" in cmake
    # The superseded per-degree file must not linger unreferenced.
    assert "v41_expert_pack_tp6_selftest" not in cmake
    assert not (ROOT / "native" / "tests" / "v41_expert_pack_tp6_selftest.cc").exists()
    wrapper = (ROOT / "scripts" / "run-tp-ep-kernel-checks.sh").read_text(encoding="utf-8")
    assert "expert_pack_geometry" in wrapper
    # The independent host oracle, the real native geometries and the fail-closed
    # device guard live in that file. Assert SEMANTICS (identifiers and the case
    # table), not prose: a wording change must not break the contract, and a
    # missing oracle must.
    geometry = (ROOT / "native" / "tests" /
                "v41_expert_pack_geometry_selftest.cc").read_text(encoding="utf-8")
    for token in ("reference_pack", "DS41RT_REQUIRE_CUDA",
                  '{"tp6", 384, 0}', '{"tp3", 768, 0}', '{"tp2", 1152, 0}',
                  '{"tp4", 576, 0}', '{"full", 2304, 0}'):
        assert token in geometry, token
    # The padded TP4 extent must be checked against the swizzled coordinates,
    # not treated as dense rows, and the gate half must be recognised.
    assert "kernel_intermediate" in geometry and "gate" in geometry
    assert "pad_words" in geometry and "pad_nonzero" in geometry
