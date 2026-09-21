"""CPU-only static checks for the replicated-group Spark TP2/TP3 native roles.

No torch, CUDA or SparkInfer import happens here: the authoritative role tables
are pure literals inside the exporters, and the native/CMake wiring is checked
from source text. These tests prove the plan-time role geometry and guards
without a GPU so they can run before any export/build lease.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLICES = ROOT / "python" / "tools" / "export_b12x_v41_slices_aot.py"
EXPERTS = ROOT / "python" / "tools" / "export_b12x_v41_experts_aot.py"
CMAKE = ROOT / "native" / "CMakeLists.txt"
CMAKE_TP = ROOT / "native" / "cmake" / "v41_spark_tp_experts.cmake"
HEADER = ROOT / "native" / "include" / "ds41rt_v41_experts.h"
REDUCE = ROOT / "native" / "cuda" / "kernels" / "v41_route_reduce.cu"
PACK = ROOT / "native" / "cuda" / "kernels" / "v41_expert_pack.cu"


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


def test_spark_tp2_tp3_geometry_is_exact_and_padding_free() -> None:
    tables = _literal_assignments(
        SLICES, {"ROLE_GEOMETRY", "ROLE_SM", "ROLE_NATIVE_ID", "SPARK_TP_DEGREE"}
    )
    geometry = tables["ROLE_GEOMETRY"]
    sm = tables["ROLE_SM"]
    native_id = tables["ROLE_NATIVE_ID"]
    degree = tables["SPARK_TP_DEGREE"]

    assert geometry["spark_tp2"] == (384, 1152, 1152, 6)
    assert geometry["spark_tp3"] == (384, 768, 768, 6)
    # Kernel extent equals logical extent for both new roles: no storage padding.
    for role in ("spark_tp2", "spark_tp3"):
        experts, logical, kernel, topk = geometry[role]
        assert experts == 384 and topk == 6
        assert kernel == logical and logical % 32 == 0

    assert sm["spark_tp2"] == (12, 1)
    assert sm["spark_tp3"] == (12, 1)
    assert native_id["spark_tp2"] == 5
    assert native_id["spark_tp3"] == 6
    # TP6 (role 7) is covered by its own contract test; the historical
    # replicated-group degrees must not be disturbed by its addition.
    assert {k: degree[k] for k in ("spark", "spark_tp2", "spark_tp3")} == {
        "spark": 4, "spark_tp2": 2, "spark_tp3": 3
    }
    assert "spark_tp6" not in degree or degree["spark_tp6"] == 6


def test_existing_tp4_and_rtx_roles_are_unchanged() -> None:
    tables = _literal_assignments(
        SLICES, {"ROLE_GEOMETRY", "ROLE_SM", "ROLE_NATIVE_ID"}
    )
    geometry = tables["ROLE_GEOMETRY"]
    sm = tables["ROLE_SM"]
    native_id = tables["ROLE_NATIVE_ID"]

    # Spark TP4 keeps 576 logical -> 640 kernel storage padding and SM121.
    assert geometry["spark"] == (384, 576, 640, 6)
    assert sm["spark"] == (12, 1)
    assert native_id["spark"] == 1
    # RTX TP2 keeps its SM120 semantics and 1152 extent.
    assert geometry["rtx_tp2"] == (384, 1152, 1152, 6)
    assert sm["rtx_tp2"] == (12, 0)
    assert native_id["rtx_tp2"] == 3
    # Full RTX backbone and coordinator dSpark unchanged.
    assert geometry["rtx_backbone"] == (384, 2304, 2304, 6)
    assert geometry["coordinator"] == (128, 2304, 2304, 3)
    assert geometry["dspark_tp2"] == (128, 1152, 1152, 3)


def test_ordinary_exporter_routes_new_roles_only_through_fp8_slices() -> None:
    tables = _literal_assignments(EXPERTS, {"SPARK_ROLES", "SPARK_TP_DEGREES"})
    roles = tables["SPARK_ROLES"]
    degrees = tables["SPARK_TP_DEGREES"]
    # The replicated-group roles are present with their plan-time degrees; TP6
    # (degree 6) has its own contract test and must not renumber TP2/TP3.
    for role in ("spark", "spark_tp2", "spark_tp3", "spark_tp6"):
        assert role in roles
    assert {k: degrees[k] for k in ("spark", "spark_tp2", "spark_tp3")} == {
        "spark": 4, "spark_tp2": 2, "spark_tp3": 3
    }
    assert degrees.get("spark_tp6") == 6
    source = EXPERTS.read_text(encoding="utf-8")
    assert "Spark TP2/TP3/TP6 use the native FP8 K32 slice export only" in source


def test_cmake_wiring_is_opt_in_and_precompiles_capacities() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    assert 'set(DS41RT_V41_SPARK_TP_ROLES "" CACHE STRING' in cmake
    assert "include(cmake/v41_spark_tp_experts.cmake)" in cmake
    assert "DS41RT_V41_SPARK_TP_EXPERT_TARGETS" in cmake

    tp = CMAKE_TP.read_text(encoding="utf-8")
    assert "SM121 Spark expert build" in tp
    assert 'DS41RT_V41_SPARK_TP_EXPERT_ROWS_ARG "1,16,80,256,1024,4096"' in tp
    assert "--atomic-min-capacity 256" in tp
    assert "--role \"${role}\"" in tp
    assert "v41_spark_tp2_experts.cc" in tp
    assert "v41_spark_tp3_experts.cc" in tp
    # The extra artifacts are opt-in: the include only runs for a non-empty list.
    assert 'if(NOT DS41RT_V41_SPARK_TP_ROLES STREQUAL "")' in cmake


def test_native_role_ids_and_reducer_abi_are_declared() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "5: Spark TP2 shard (intermediate 1152)" in header
    assert "6: Spark TP3 shard (intermediate 768)" in header
    assert "ds41rt_v41_reduce_compact_bf16_planes_async" in header
    assert "const uint16_t* const planes[6]" in header
    assert "uint32_t ranks" in header

    reduce_source = REDUCE.read_text(encoding="utf-8")
    assert "if constexpr (Ranks >= 6)" in reduce_source
    assert "valid_compact_planes" in reduce_source
    # The two historical fixed entry points must remain.
    assert "ds41rt_v41_reduce_compact_bf16_async" in reduce_source
    assert "ds41rt_v41_reduce_tp2_compact_bf16_async" in reduce_source


def test_packer_accepts_tp3_extent_and_requires_scale_alignment() -> None:
    pack = PACK.read_text(encoding="utf-8")
    assert "intermediate != 768" in pack
    assert "intermediate % 32 != 0" in pack


SPARK_WIDTH_DEFAULT = "1:64,16:192,80:192,256:192,1024:192,4096:192"


def test_spark_tp_width_maps_are_per_role_cache_knobs_with_identical_defaults() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    assert (
        f'set(DS41RT_V41_SPARK_TP2_SLICE_WIDTH "{SPARK_WIDTH_DEFAULT}" CACHE STRING'
        in tp
    )
    assert (
        f'set(DS41RT_V41_SPARK_TP3_SLICE_WIDTH "{SPARK_WIDTH_DEFAULT}" CACHE STRING'
        in tp
    )
    assert (
        f'set(DS41RT_V41_SPARK_TP6_SLICE_WIDTH "{SPARK_WIDTH_DEFAULT}" CACHE STRING'
        in tp
    )
    # The previous shared normal variable is gone: a normal set would shadow -D
    # and re-couple the roles. Exactly one copy of the default remains per role.
    assert "set(DS41RT_V41_SPARK_TP_SLICE_WIDTH" not in tp
    assert "DS41RT_V41_SPARK_TP_SLICE_WIDTH" not in tp
    assert tp.count(SPARK_WIDTH_DEFAULT) == 3


def test_each_role_forwards_its_own_width_map_to_the_exporter() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    assert 'set(width_map "${DS41RT_V41_SPARK_TP2_SLICE_WIDTH}")' in tp
    assert 'set(width_map "${DS41RT_V41_SPARK_TP3_SLICE_WIDTH}")' in tp
    # One shared exporter invocation consumes the role-selected map, so a TP2
    # override cannot leak into the TP3 export or vice versa.
    assert tp.count('--width "${width_map}"') == 1
    assert '--role "${role}"' in tp


def test_spark_tp_width_knobs_leave_the_generic_tp4_knob_untouched() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    assert "DS41RT_V41_EXPERT_SLICE_WIDTH" not in tp
    experts_cmake = (ROOT / "native" / "cmake" / "v41_experts.cmake").read_text(
        encoding="utf-8"
    )
    assert 'set(DS41RT_V41_EXPERT_SLICE_WIDTH "" CACHE STRING' in experts_cmake


def test_invalid_width_maps_fail_in_the_existing_exporter_validator() -> None:
    # CMake forwards the map untouched; the exporter owns the parser. A map that
    # does not cover every precompiled capacity, or a width outside 64/128/192,
    # is a hard parser error before any export runs.
    slices = SLICES.read_text(encoding="utf-8")
    assert "width map must cover every capacity exactly once" in slices
    assert "width must be 64, 128 or 192" in slices
    assert "parser.error(str(error))" in slices
    # No duplicated domain parser in the Spark TP2/TP3 CMake.
    assert "MATCHES" not in CMAKE_TP.read_text(encoding="utf-8")


def test_width_override_invalidates_stale_exports_without_reconfigure_churn() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    # Output stems are constant v41_{role}_m{capacity}, so a width change must
    # still invalidate an existing export. A content-stable per-role stamp is a
    # dependency, giving Make (timestamp-driven) the same re-export trigger Ninja
    # gets from the changed command line.
    assert (
        'set(width_stamp "${CMAKE_CURRENT_BINARY_DIR}/v41_${role}_width.stamp")'
        in tp
    )
    assert 'file(GENERATE OUTPUT "${width_stamp}"' in tp
    assert "width=${width_map}" in tp
    assert "atomic_min_capacity=256" in tp
    assert '"${width_stamp}"' in tp
    # Content-stable generation only: no unconditional write that would churn the
    # AOT objects on every unchanged configure.
    assert "file(WRITE" not in tp
    assert "configure_file" not in tp


def test_atomic_threshold_survives_a_width_override() -> None:
    tp = CMAKE_TP.read_text(encoding="utf-8")
    # The width map and the atomic threshold travel in the same exporter command,
    # so overriding a width cannot silently drop direct token accumulation.
    assert '--width "${width_map}"\n      --atomic-min-capacity 256' in tp
    assert "--atomic-min-capacity 256 --standard-names" in tp


# --------------------------------------------------------------------------- #
# Real bounded CPU behavior tests (toy CMake/Make fixture + exporter CLI AST)
# --------------------------------------------------------------------------- #

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402


def _load_width_fixture():
    path = Path(__file__).with_name("tp_ep_width_cmake_fixture.py")
    spec = importlib.util.spec_from_file_location("tp_ep_width_cmake_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cmake_make_width_override_invalidation() -> None:
    if shutil.which("cmake") is None or shutil.which("make") is None:
        pytest.skip("cmake and make are required for the toy fixture")
    fixture = _load_width_fixture()
    steps = fixture.run_width_override_scenario()

    # Initial defaults: both roles export the unchanged map with atomic 256.
    assert len(steps["after_default"]["spark_tp2"]) == 1
    assert len(steps["after_default"]["spark_tp3"]) == 1
    for role in ("spark_tp2", "spark_tp3"):
        assert steps["default_manifests"][role]["width"] == fixture.DEFAULT_MAP
        assert steps["default_manifests"][role]["atomic_min_capacity"] == "256"

    # TP2 cap80 -> 128 alone: TP2 re-exports, TP3 does not, atomic stays.
    assert len(steps["after_override"]["spark_tp2"]) == 2
    assert steps["override_manifests"]["spark_tp2"]["width"] == fixture.TP2_CAP80_128
    assert steps["override_manifests"]["spark_tp2"]["atomic_min_capacity"] == "256"
    assert len(steps["after_override"]["spark_tp3"]) == 1
    assert steps["override_manifests"]["spark_tp3"]["width"] == fixture.DEFAULT_MAP

    # Unchanged override reconfigure: no re-export at all (content-stable stamp).
    assert len(steps["after_unchanged"]["spark_tp2"]) == 2
    assert len(steps["after_unchanged"]["spark_tp3"]) == 1

    # Restoring the default invalidates TP2 again (both directions).
    assert len(steps["after_restore"]["spark_tp2"]) == 3
    assert steps["after_restore"]["spark_tp2"][-1]["width"] == fixture.DEFAULT_MAP
    assert len(steps["after_restore"]["spark_tp3"]) == 1


def _exporter_main_block():
    tree = ast.parse(SLICES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                return node.body
    raise AssertionError("exporter has no __main__ block")


def _run_exporter_cli(argv: list[str]) -> list:
    calls: list = []
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "export": lambda *args, **kwargs: calls.append((args, kwargs)),
        "__doc__": "cli test",
    }
    module = ast.Module(body=_exporter_main_block(), type_ignores=[])
    code = compile(ast.fix_missing_locations(module), str(SLICES), "exec")
    previous = sys.argv
    sys.argv = ["export_b12x_v41_slices_aot.py"] + argv
    try:
        exec(code, namespace)  # noqa: S102 - executing the exporter's own CLI block
    finally:
        sys.argv = previous
    return calls


def _exporter_argv(width: str, rows: str = "1,16,80") -> list[str]:
    return [
        "--output-dir", "cli-test-output",
        "--role", "spark_tp2",
        "--rows", rows,
        "--width", width,
        "--atomic-min-capacity", "256",
        "--standard-names",
    ]


def test_exporter_cli_accepts_valid_width_maps_and_scalars() -> None:
    calls = _run_exporter_cli(_exporter_argv("1:64,16:192,80:192"))
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert str(args[0]) == "cli-test-output"
    assert args[1] == (1, 16, 80)
    assert args[2] == {1: 64, 16: 192, 80: 192}
    assert args[3] == 256 and args[4] == "spark_tp2"
    assert kwargs["standard_names"] is True

    scalar = _run_exporter_cli(_exporter_argv("192"))
    assert scalar[0][0][2] == 192


def test_exporter_cli_rejects_invalid_width_maps(
    capsys: pytest.CaptureFixture
) -> None:
    duplicate_key = ["1:64,1:128,80:192"]
    missing_capacity = ["1:64,80:192", "1:64,16:192"]
    bad_width = ["1:64,16:193,80:192", "1:64,16:0,80:192"]
    for width in duplicate_key + missing_capacity:
        with pytest.raises(SystemExit) as excinfo:
            _run_exporter_cli(_exporter_argv(width))
        assert excinfo.value.code == 2
        assert "width map must cover every capacity exactly once" in capsys.readouterr().err
    for width in bad_width:
        with pytest.raises(SystemExit) as excinfo:
            _run_exporter_cli(_exporter_argv(width))
        assert excinfo.value.code == 2
        assert "width must be 64, 128 or 192" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        _run_exporter_cli(_exporter_argv("1:64,16:192,80:192", rows="1,1,80"))
    assert excinfo.value.code == 2
    assert "rows must be unique capacities in 1..4096" in capsys.readouterr().err
