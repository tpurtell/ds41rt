"""CPU-only contract tests for the native Spark TP3 launch-geometry identity.

Nothing here needs a GPU. The geometry identity is a pure function; the qualifier
extensions that consume it (manifest geometry, route-weight modes, identity
tuples, throttle masks, repeat aggregation) are exercised through their real
production helpers with CUDA hidden. Pinned-source tripwires live here, not in
production: a pin bump that changes the kernel geometry must fail this file.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
GEOMETRY = TOOLS / "v41_spark_tp3_launch_geometry.py"
QUALIFIER = TOOLS / "qualify_v41_replicated_native.py"
SLICES = TOOLS / "export_b12x_v41_slices_aot.py"
CMAKE_TP = ROOT / "native" / "cmake" / "v41_spark_tp_experts.cmake"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v41_spark_tp3_launch_geometry as geometry  # noqa: E402

CAPACITIES = (1, 16, 80, 256, 1024, 4096)
# The reviewed default width map is the CMake cache default, not a production
# constant: the module deliberately carries no second copy of it.
CMAKE_DEFAULT_WIDTHS = {1: 64, 16: 192, 80: 192, 256: 192, 1024: 192, 4096: 192}

# Pinned-source tripwires for the constants this module asserts. Kept in the test
# so production has one authority (the pinned kernel) and no duplicated prose.
PINNED_ANCHORS = {
    "fc1.mma_atom_mnk": (
        "sparkinfer", "b12x/moe/_shared/kernels/w4a8_v41_slice.py",
        "mxfp8_mma_m16n8k32_f32_e2m1("),
    "fc1.m_rows": (
        "sparkinfer", "b12x/moe/_shared/kernels/v41_route_plan.py",
        "min(Int32(16), count - j)"),
    "fc1.k_mma_atom": (
        "sparkinfer", "b12x/moe/_shared/kernels/w4a8_v41_slice.py",
        "kt * 32 + kb * 8 + c * 2"),
    "fc1.k_stages": (
        "sparkinfer", "b12x/moe/_shared/kernels/w4a8_v41_slice.py",
        "for kt in range(40):"),
    "grid.block_threads": (
        "sparkinfer", "b12x/moe/_shared/kernels/w4a8_v41_slice.py",
        "block=(128, 1, 1),"),
    "supported_slice_widths": (
        "sparkinfer", "b12x/moe/_shared/kernels/w4a8_v41_slice.py",
        "assert width in (64, 128, 192)"),
    "spark_tp3.geometry": (
        "ds41rt", "python/tools/export_b12x_v41_slices_aot.py",
        '"spark_tp3": (384, 768, 768, 6)'),
}

LIVE_QUANTITY_KEYS = frozenset(
    {"num_tokens", "token_count", "tokens", "rows", "live_rows", "live",
     "routed_rows", "scatter_rows", "routes", "groups", "group_count",
     "expert_counts", "batch", "sequence", "seq_len", "occupancy", "pages",
     "blocks", "requests"})


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


# --------------------------------------------------------------------------- #
# Geometry identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("width", geometry.SUPPORTED_SLICE_WIDTHS)
def test_geometry_matches_the_pinned_slice_arithmetic(capacity, width) -> None:
    record = geometry.launch_geometry(
        capacity, width, intermediate=768, kernel_intermediate=768,
        output_kind="fp32_routes", revision="abc123")
    assert record["schema_version"] == geometry.GEOMETRY_SCHEMA_VERSION == 2
    assert record["slices"] == (768 + width - 1) // width
    assert record["fc1"]["logical_n_per_projection"] == width
    assert record["fc1"]["fused_w13_n"] == 2 * width
    assert record["fc1"]["m_rows"] == 16
    assert record["fc1"]["k_elements"] == 5120
    assert record["fc1"]["k_mma_atom"] == 32
    assert record["fc1"]["k_stage_elements"] == 128
    assert record["fc1"]["k_stages"] == 40
    assert record["fc1"]["mma_atom_mnk"] == [16, 8, 32]
    assert record["fc2"]["k_elements"] == width
    assert record["fc2"]["k_mma_atoms"] == width // 32
    assert record["fc2"]["n_elements"] == 5120
    assert record["fc2"]["n_stages"] == 40
    assert record["metadata"]["columns"] == 3 + 16
    assert record["grid"]["block_threads"] == 128
    assert record["grid"]["warps"] == 4
    assert record["grid"]["y"] == "runtime_route_group_count"
    assert record["output_kind"] == "fp32_routes"
    # Per-stage control authority, not one ambiguous global tile control.
    assert record["controls"]["fc1"] == {
        "m": "pinned_kernel_constant", "n": "artifact_manifest_slice_width",
        "k": "pinned_kernel_constant"}
    assert record["controls"]["fc2"] == {
        "m": "pinned_kernel_constant", "n": "pinned_kernel_constant",
        "k": "artifact_manifest_slice_width"}
    assert record["controls"]["grid"]["x_slices"] == "derived_from_manifest_slice_width"
    assert record["controls"]["grid"]["y"] == "runtime_route_group_count"
    assert record["controls"]["grid"]["block"] == "pinned_kernel_constant"
    assert record["source"].endswith("@abc123")


def test_geometry_is_static_only() -> None:
    record = geometry.launch_geometry(
        80, 192, intermediate=768, kernel_intermediate=768,
        output_kind="fp32_tokens")
    leaked = sorted(set(_walk_keys(record)) & LIVE_QUANTITY_KEYS)
    assert not leaked, leaked


@pytest.mark.parametrize(
    "capacity,width,kwargs,message",
    [
        (80, 96, {}, "not one of"),
        (80, 0, {}, "width"),
        (0, 192, {}, "capacity"),
        (80, 192, {"kernel_intermediate": 640}, "128-roundup"),
        (80, 64, {"intermediate": 160, "kernel_intermediate": 160}, "128-roundup"),
        (80, 192, {"hidden": 600}, "hidden"),
        (80, 192, {"output_kind": "bf16"}, "output_kind"),
    ],
)
def test_geometry_fails_closed(capacity, width, kwargs, message) -> None:
    arguments = {"intermediate": 768, "kernel_intermediate": 768,
                 "output_kind": "fp32_routes"}
    arguments.update(kwargs)
    with pytest.raises(geometry.LaunchGeometryError) as error:
        geometry.launch_geometry(capacity, width, **arguments)
    assert message in str(error.value)


def test_pinned_source_anchors_still_hold() -> None:
    try:
        import _pinned_sparkinfer
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer is unavailable: {error}")
    source_root = Path(_pinned_sparkinfer.SOURCE)
    for name, (root_name, relative, text) in PINNED_ANCHORS.items():
        root = source_root if root_name == "sparkinfer" else ROOT
        path = root / relative
        assert path.is_file(), f"{name}: missing {path}"
        assert text in path.read_text(encoding="utf-8"), (
            f"{name}: {relative} no longer contains {text!r}; re-derive the "
            "launch-geometry identity")
    atom = re.search(r"m(\d+)n(\d+)k(\d+)", PINNED_ANCHORS["fc1.mma_atom_mnk"][2])
    assert tuple(int(part) for part in atom.groups()) == geometry.FC1_MMA_ATOM_MNK
    assert geometry.HIDDEN_ELEMENTS // geometry.K_STAGE_ELEMENTS == 40
    assert f"block=({geometry.BLOCK_THREADS}," in PINNED_ANCHORS["grid.block_threads"][2]


def test_cmake_default_width_map_matches_the_reviewed_widths() -> None:
    match = re.search(
        r'set\(DS41RT_V41_SPARK_TP3_SLICE_WIDTH "([^"]+)"',
        CMAKE_TP.read_text(encoding="utf-8"))
    assert match is not None
    parsed = {}
    for pair in match[1].split(","):
        capacity, width = pair.split(":")
        parsed[int(capacity)] = int(width)
    assert parsed == CMAKE_DEFAULT_WIDTHS
    # Production must not carry a second copy of the policy.
    assert "DEFAULT_TP3_WIDTH_MAP" not in GEOMETRY.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Manifest geometry verification
# --------------------------------------------------------------------------- #


def _variant(capacity, width, output_kind, record=True):
    variant = {"capacity_rows": capacity, "width": width,
               "output_kind": output_kind, "core_scratch_nbytes": capacity * 64}
    if record:
        variant["launch_geometry"] = geometry.launch_geometry(
            capacity, width, intermediate=768, kernel_intermediate=768,
            output_kind=output_kind, revision="rev1")
    return variant


def _manifest(variants=None, *, revision="rev1", capability=(12, 1), sms=48):
    return {
        "schema": 1,
        "role": "spark_tp3",
        "spark_tp_degree": 3,
        "input_format": "fp8_k32",
        "sparkinfer_revision": revision,
        "capability": list(capability),
        "physical_sms": sms,
        "geometry": {"experts": 384, "hidden": 5120, "intermediate": 768,
                     "kernel_intermediate": 768, "topk": 6},
        "variants": variants or [_variant(1, 64, "fp32_routes"),
                                 _variant(80, 192, "fp32_tokens")],
    }


def test_manifest_geometry_resolves_and_verifies_recorded_identity() -> None:
    resolved = geometry.manifest_geometry(_manifest(), degree=3, intermediate=768)
    assert sorted(resolved) == [1, 80]
    assert resolved[80]["fc1"]["fused_w13_n"] == 384


def test_manifest_geometry_rejects_recorded_drift_and_wrong_arm() -> None:
    payload = _manifest()
    payload["variants"][0]["launch_geometry"]["fc1"]["m_rows"] = 32
    with pytest.raises(geometry.LaunchGeometryError):
        geometry.manifest_geometry(payload, degree=3, intermediate=768)
    with pytest.raises(geometry.LaunchGeometryError):
        geometry.manifest_geometry(_manifest(), degree=6, intermediate=768)
    with pytest.raises(geometry.LaunchGeometryError):
        geometry.manifest_geometry(_manifest(), degree=3, intermediate=384)
    with pytest.raises(geometry.LaunchGeometryError):
        geometry.manifest_geometry({"spark_tp_degree": 3, "geometry": {}, "variants": []},
                                   degree=3, intermediate=768)


def test_manifest_geometry_is_backward_compatible_without_recorded_blocks() -> None:
    payload = _manifest(variants=[_variant(1, 64, "fp32_routes", record=False)])
    resolved = geometry.manifest_geometry(payload, degree=3, intermediate=768)
    assert resolved[1]["fc1"]["logical_n_per_projection"] == 64


def test_manifest_geometry_requires_capacity_width_and_hidden() -> None:
    missing_width = _manifest(variants=[{"capacity_rows": 80,
                                         "output_kind": "fp32_routes"}])
    with pytest.raises(geometry.LaunchGeometryError) as error:
        geometry.manifest_geometry(missing_width, degree=3, intermediate=768)
    assert "missing its compiled width" in str(error.value)
    missing_capacity = _manifest(variants=[{"width": 192,
                                            "output_kind": "fp32_routes"}])
    with pytest.raises(geometry.LaunchGeometryError) as error:
        geometry.manifest_geometry(missing_capacity, degree=3, intermediate=768)
    assert "missing capacity_rows" in str(error.value)
    hiddenless = _manifest()
    hiddenless["geometry"] = {"experts": 384, "intermediate": 768}
    with pytest.raises(geometry.LaunchGeometryError) as error:
        geometry.manifest_geometry(hiddenless, degree=3, intermediate=768)
    assert "hidden and kernel_intermediate" in str(error.value)


# --------------------------------------------------------------------------- #
# Qualifier extensions
# --------------------------------------------------------------------------- #


def _load_qualifier():
    pytest.importorskip("torch")
    try:
        spec = importlib.util.spec_from_file_location("ds41rt_tp3_qualifier", QUALIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"qualifier import unavailable: {error}")
    return module


def test_manifest_geometry_gate_is_tp3_only() -> None:
    q = _load_qualifier()
    assert q.verify_manifest_geometry(_manifest(), 768, 3)[1]["slices"] == 12
    # Non-TP3 degrees resolve to an empty map rather than an invented one.
    assert q.verify_manifest_geometry(_manifest(), 384, 6) == {}
    bad = _manifest()
    bad["variants"][0]["launch_geometry"]["fc2"]["k_mma_atoms"] = 99
    with pytest.raises(SystemExit):
        q.verify_manifest_geometry(bad, 768, 3)


def test_manifest_device_identity_capability_and_build_ceiling() -> None:
    q = _load_qualifier()
    device = {"name": "g", "compute_capability": [12, 1], "sm_count": 48}
    assert q.verify_manifest_device(_manifest(), device)["role"] == "spark_tp3"
    # physical_sms is the export host ceiling, not the live device: a smaller
    # live part (the loader clamps min(export, live)) is valid.
    assert q.verify_manifest_device(_manifest(sms=72), device)["role"] == "spark_tp3"
    for bad in (_manifest(capability=(12, 0)), _manifest(sms=0), _manifest(sms=-1)):
        with pytest.raises(SystemExit):
            q.verify_manifest_device(bad, device)


def test_loaded_meta_crosscheck_matches_the_manifest_arm() -> None:
    q = _load_qualifier()

    class Meta:
        role, experts, hidden_size = 6, 384, 5120
        logical_intermediate, kernel_intermediate, topk = 768, 768, 6
        capacity_rows = 80

    assert q.verify_loaded_meta(Meta(), _manifest(), 80, 3) is not None
    Meta.capacity_rows = 1
    with pytest.raises(SystemExit):
        q.verify_loaded_meta(Meta(), _manifest(), 80, 3)
    Meta.capacity_rows = 80
    Meta.kernel_intermediate = 640
    with pytest.raises(SystemExit):
        q.verify_loaded_meta(Meta(), _manifest(), 80, 3)


def test_route_weight_modes_are_named_and_nonuniform_when_requested() -> None:
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    uniform = q._route_weights(torch, 2, 4, "uniform", device="cpu")
    assert bool(torch.allclose(uniform, torch.full((2, 4), 0.25)))
    geometric = q._route_weights(torch, 2, 4, "geometric", device="cpu")
    assert bool(torch.allclose(geometric.sum(-1), torch.ones(2)))
    row = geometric[0].tolist()
    assert row[0] > row[1] > row[2] > row[3] > 0
    with pytest.raises(SystemExit):
        q._route_weights(torch, 2, 4, "ramp", device="cpu")


def test_fill_routing_keeps_sentinels_and_applies_the_selected_weights() -> None:
    q = _load_qualifier()
    torch = pytest.importorskip("torch")
    gids = [0, 1, 2, 3, 4, 5, 383]
    base = q.ARENA_EXPERTS - len(gids)
    for mode in ("uniform", "geometric"):
        ids = torch.full((2, 6), 7, dtype=torch.int32)
        rw = torch.full((2, 6), 3.0)
        _, routing = q._fill_routing(torch, ids, rw, 2, base, gids, 3,
                                     device="cpu", weight_mode=mode)
        assert bool((ids[:, 3:] == q.SENTINEL).all())
        assert bool((rw[:, 3:] == 0).all())
        assert bool(torch.allclose(routing[:, 3:], torch.zeros(2, 3)))
        assert q._route_counts(ids, rw) == (6, 6)


def test_record_identity_excludes_timing_and_includes_the_geometry_contract() -> None:
    q = _load_qualifier()
    geometry_record = geometry.launch_geometry(
        80, 192, intermediate=768, kernel_intermediate=768,
        output_kind="fp32_routes")
    base = {"spark_tp": 3, "capacity": 80, "rows": 16, "active": 6, "width": 192,
            "timing_seed": 81, "route_weights": "geometric",
            "rel_tolerance": 0.01, "cosine_tolerance": 0.9999,
            "sparkinfer_revision": "rev", "lib_sha256": "l", "manifest_sha256": "m",
            "launch_geometry": geometry_record,
            "device": {"uuid": "GPU-1", "name": "g"},
            "kernel_only": {"warm_amortized_device_us": 1.0}}
    other = dict(base, kernel_only={"warm_amortized_device_us": 999.0})
    assert q._record_identity(base) == q._record_identity(other)
    changed = dict(base, route_weights="uniform")
    assert q._record_identity(base) != q._record_identity(changed)
    changed_device = dict(base, device={"uuid": "GPU-2", "name": "g"})
    assert q._record_identity(base) != q._record_identity(changed_device)
    # The cross-width context ignores build-specific fields but keeps the seed,
    # routing, tolerances, revision, model geometry and device.
    other_build = dict(base, lib_sha256="other", manifest_sha256="other",
                       width=64, launch_geometry=None)
    assert q._context_identity(base) == q._context_identity(other_build)
    assert q._context_identity(base) != q._context_identity(dict(base, timing_seed=99))
    assert q._record_identity(base) != q._record_identity(dict(base, wire_sha256="x"))
    assert q._geometry_sha256(geometry_record) == q._geometry_sha256(
        json.loads(json.dumps(geometry_record)))
    # Prose is not part of the hashed contract; a structured field change is.
    assert q._geometry_sha256(geometry_record) == q._geometry_sha256(
        dict(geometry_record, source="rewritten prose"))
    changed_field = dict(geometry_record)
    changed_field["fc1"] = dict(geometry_record["fc1"], m_rows=32)
    assert q._geometry_sha256(geometry_record) != q._geometry_sha256(changed_field)


def _timing_record(width, *, median, lib=None, status="ok", uniform=False,
                   capacity=80, rows=16):
    return {
        "kind": "native_timing", "status": status, "spark_tp": 3,
        "capacity": capacity, "rows": rows, "active": 6, "width": width,
        "timing_seed": 81,
        "route_weights": "uniform" if uniform else "geometric",
        "rel_tolerance": 0.01, "cosine_tolerance": 0.9999,
        "sparkinfer_revision": "rev",
        "manifest_geometry": {"experts": 384, "hidden": 5120,
                              "intermediate": 768, "kernel_intermediate": 768,
                              "topk": 6},
        "lib_sha256": lib or f"lib-{width}", "manifest_sha256": "manifest",
        "wire_sha256": "wire", "input_sha256": "input", "route_sha256": "route",
        "launch_geometry": geometry.launch_geometry(
            capacity, width, intermediate=768, kernel_intermediate=768,
            output_kind="fp32_routes"),
        "device": {"uuid": "GPU-1", "name": "g"},
        "kernel_only": {"warm_amortized_device_us": median},
    }


def _aggregate_options(tmp_path, paths, repeats=3, allow_uniform=False):
    from types import SimpleNamespace
    return SimpleNamespace(aggregate=paths, aggregate_repeats=repeats,
                           allow_uniform_diagnostic=allow_uniform,
                           output=tmp_path / "comparison.json")


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return path


def test_aggregate_reports_widths_without_any_promotion(tmp_path) -> None:
    q = _load_qualifier()
    first = _write(tmp_path, "w192.json",
                   [_timing_record(192, median=100.0) for _ in range(3)])
    second = _write(tmp_path, "w64.json",
                    [_timing_record(64, median=90.0) for _ in range(3)])
    summary = q.aggregate_results(_aggregate_options(tmp_path, [first, second]))
    assert summary["promotes_default"] is False
    assert summary["report_only"] is True
    assert summary["repeats_per_width"] == 3
    assert {entry["width"] for entry in summary["widths"]} == {64, 192}
    assert "fastest_width_by_median" not in summary
    assert "groups" not in summary
    assert json.loads((tmp_path / "comparison.json").read_text())["widths"]


def test_aggregate_requires_three_ok_repeats_with_one_identity(tmp_path) -> None:
    q = _load_qualifier()
    path = tmp_path / "cell.json"
    # Two records where three are required (also a single width).
    path.write_text(json.dumps([_timing_record(192, median=100.0) for _ in range(2)]))
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, [path]))
    # Three records with a mixed repeat identity is not a valid repeat set.
    path.write_text(json.dumps(
        [_timing_record(192, median=100.0),
         _timing_record(192, median=101.0, lib="other-lib"),
         _timing_record(192, median=102.0)]))
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, [path]))
    # A failed record is rejected outright.
    path.write_text(json.dumps(
        [_timing_record(192, median=100.0, status="failed") for _ in range(3)]))
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, [path]))


def test_aggregate_rejects_uniform_same_library_and_multi_context(tmp_path) -> None:
    q = _load_qualifier()
    uniform = [
        _write(tmp_path, "u192.json",
               [_timing_record(192, median=100.0, uniform=True) for _ in range(3)]),
        _write(tmp_path, "u64.json",
               [_timing_record(64, median=90.0, uniform=True) for _ in range(3)]),
    ]
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, uniform))
    # Explicit diagnostic escape hatch is the only way to compare uniform runs.
    assert q.aggregate_results(
        _aggregate_options(tmp_path, uniform, allow_uniform=True))["report_only"]
    # The same library cannot back two width arms.
    same_lib = [
        _write(tmp_path, "s1.json",
               [_timing_record(192, median=100.0, lib="shared") for _ in range(3)]),
        _write(tmp_path, "s2.json",
               [_timing_record(64, median=90.0, lib="shared") for _ in range(3)]),
    ]
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, same_lib))
    # Mixed measurement contexts are refused before any comparison.
    mixed = [
        _write(tmp_path, "m1.json",
               [_timing_record(192, median=100.0) for _ in range(3)]),
        _write(tmp_path, "m2.json",
               [_timing_record(64, median=90.0, capacity=1) for _ in range(3)]),
    ]
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, mixed))
    # One width is not a comparison.
    single = _write(tmp_path, "only.json",
                    [_timing_record(192, median=100.0) for _ in range(3)])
    with pytest.raises(SystemExit):
        q.aggregate_results(_aggregate_options(tmp_path, [single]))


def test_aggregate_reads_jsonl_and_timing_prefixed_lines(tmp_path) -> None:
    q = _load_qualifier()
    path = tmp_path / "stdout.log"
    lines = ["TIMING " + json.dumps(_timing_record(192, median=100.0 + index))
             for index in range(3)]
    lines += ["TIMING " + json.dumps(_timing_record(64, median=90.0 + index))
              for index in range(3)]
    path.write_text("\n".join(lines))
    assert len(q._load_timing_records(path)) == 6
    summary = q.aggregate_results(_aggregate_options(tmp_path, [path]))
    assert {entry["width"] for entry in summary["widths"]} == {64, 192}


def test_throttle_mask_is_exact_zero_and_state_is_device_scoped() -> None:
    q = _load_qualifier()
    assert q.parse_throttle_mask("0x0") == [0]
    for bad in ("", "   ", "0x0,", "0x4", "0x0,0x4", "nope", "4"):
        with pytest.raises(SystemExit):
            q.parse_throttle_mask(bad)
    clean = {"rows": ["0, GPU-1, P1, 2100, 9501, 0x0000000000000000"],
             "found": True, "error": None, "selected_by": "uuid"}
    hot = {"rows": ["0, GPU-1, P1, 2100, 9501, 0x0000000000000004"],
           "found": True, "error": None, "selected_by": "uuid"}
    assert q.verify_throttle(clean, clean, [0], {"uuid": "GPU-1"})["ok"] is True
    with pytest.raises(SystemExit):
        q.verify_throttle(hot, hot, [0], {"uuid": "GPU-1"})
    # Unknown or missing device state fails closed; it cannot certify 0x0.
    for bad_state in ({"rows": [], "found": False, "error": None},
                      {"rows": [], "found": False, "error": "nvidia-smi missing"}):
        with pytest.raises(SystemExit):
            q.verify_throttle(bad_state, clean, [0], {"uuid": "GPU-1"})
    # An unparseable / [N/A] / truncated reason field is a hard failure, never a
    # silent zero, and the message must carry the device row context.
    for row in ("0, GPU-1, P1, 2100, 9501, [N/A]",
                "0, GPU-1, P1, 2100, 9501",
                "0, GPU-1, P1, 2100, 9501, not-hex"):
        unparseable = {"rows": [row], "found": True, "error": None,
                       "selected_by": "uuid"}
        with pytest.raises(SystemExit) as error:
            q.verify_throttle(unparseable, clean, [0], {"uuid": "GPU-1"})
        assert "GPU-1" in str(error.value)
        assert row in str(error.value)


def test_immutability_gate_fails_closed_on_any_change() -> None:
    q = _load_qualifier()
    assert q._immutability("quantized wire", "a" * 64, "a" * 64) is True
    assert q._immutability("expert operand arenas", ["a", "b"], ["a", "b"]) is True
    with pytest.raises(SystemExit) as error:
        q._immutability("quantized wire", "a" * 64, "b" * 64)
    assert "quantized wire" in str(error.value)


# --------------------------------------------------------------------------- #
# Exporter plumbing
# --------------------------------------------------------------------------- #


def test_exporter_records_geometry_from_the_shared_identity() -> None:
    source = SLICES.read_text(encoding="utf-8")
    assert "from v41_spark_tp3_launch_geometry import launch_geometry" in source
    assert "launch_geometry(" in source
    assert 'output_kind="fp32_tokens" if atomic else "fp32_routes"' in source
    assert "revision=_pinned_sparkinfer.REVISION" in source
    assert 'manifest["launch_geometry_contract"]' not in source


def test_exporter_module_resolves_tp3_geometry() -> None:
    spec = importlib.util.spec_from_file_location("ds41rt_slices_export_probe", SLICES)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - pinned tree dependent
        pytest.skip(f"pinned SparkInfer verification unavailable: {error}")
    assert module.ROLE_GEOMETRY["spark_tp3"] == (384, 768, 768, 6)
    record = module.launch_geometry(80, 192, intermediate=768,
                                    kernel_intermediate=768,
                                    output_kind="fp32_routes")
    assert record["slices"] == 4


def test_duplicate_second_driver_is_gone() -> None:
    # The TP3 tile qualification extends the canonical replicated-native
    # qualifier instead of shipping a second 700-line driver.
    assert not (TOOLS / "qualify_v41_spark_tp3_native_tiles.py").exists()
