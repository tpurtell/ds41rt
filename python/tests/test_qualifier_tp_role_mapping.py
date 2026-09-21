"""CPU regression tests for the replicated-native qualifier's TP-degree -> role id.

Two independent code paths compute the native role family:

* `run_checkpoint` (the real-checkpoint oracle) and
* the synthetic/timing path.

Both previously used the legacy `3 + spark_tp` arithmetic, which is correct for
TP2 (5) and TP3 (6) but maps TP6 to 9 instead of the registered role 7, so a TP6
run died on `assert meta.role == role`. These tests assert the mapping is driven
by the role TABLE, not by arithmetic, and that no `3 + ...` formula survives.

They also assert the failure is loud rather than silent: the role assertion must
actually compare against the table entry.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
QUALIFIER = TOOLS / "qualify_v41_replicated_native.py"


def _load_qualifier():
    torch = pytest.importorskip("torch")
    assert torch is not None
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("ds41rt_qualifier_roles", QUALIFIER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned SparkInfer/b12x unavailable: {error}")
    return module


def test_role_table_is_the_authority_for_every_degree() -> None:
    module = _load_qualifier()
    assert module.SPARK_TP_ROLE == {2: 5, 3: 6, 6: 7}
    assert module.SPARK_TP_INTERMEDIATE == {2: 1152, 3: 768, 6: 384}
    assert module.SPARK_TP_KERNEL_INTERMEDIATE == {2: 1152, 3: 768, 6: 384}
    # The legacy arithmetic must NOT reproduce the TP6 entry; if it did, this
    # test could not distinguish table-driven selection from 3 + degree.
    assert module.SPARK_TP_ROLE[6] != 3 + 6
    assert module.SPARK_TP_ROLE[2] == 3 + 2 and module.SPARK_TP_ROLE[3] == 3 + 3


def test_no_legacy_role_arithmetic_survives_in_the_source() -> None:
    source = QUALIFIER.read_text(encoding="utf-8")
    # Any `3 + options.spark_tp` (in either path) is the bug this prevents.
    assert not re.search(r"3\s*\+\s*options\.spark_tp", source), (
        "legacy 3 + spark_tp role arithmetic is back; TP6 would map to role 9")
    # Both role determinations must read the table.
    assert source.count("SPARK_TP_ROLE[options.spark_tp]") >= 2, (
        "expected the checkpoint and timing paths to both use SPARK_TP_ROLE")
    assert "LEGACY_TP4_ROLE" in source


def test_both_role_paths_are_covered_by_executable_checks() -> None:
    """The two paths must contain an executed role assertion, not a comment."""
    tree = ast.parse(QUALIFIER.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    assert "run_checkpoint" in functions
    checkpoint_asserts = [
        node for node in ast.walk(functions["run_checkpoint"])
        if isinstance(node, ast.Assert)
    ]
    assert checkpoint_asserts, "run_checkpoint has no role assertion"

    # The timing path calls assert_native_role_geometry, which is the shared gate.
    timing_asserts = [
        node for node in ast.walk(functions.get("_run_timing", ast.Pass()))
        if isinstance(node, ast.Call)
    ]
    names = {getattr(node.func, "id", None) for node in timing_asserts}
    assert "assert_native_role_geometry" in names, (
        "_run_timing must gate on the shared geometry/role helper")


def test_role_gate_rejects_the_legacy_tp6_mapping() -> None:
    """A library publishing role 9 for a TP6 request must be refused."""
    module = _load_qualifier()
    import ctypes as C
    import ctypes.util  # noqa: F401

    fields = (["abi_version", "role", "experts", "hidden_size",
               "logical_intermediate", "kernel_intermediate", "topk",
               "capacity_rows"] + ["scratch_bytes"] +
              ["max_rows", "rows_padded", "max_tasks", "max_phys_tiles",
               "max_active_clusters"] + ["input_dtype"])

    class _Info(C.Structure):
        _fields_ = [(name, C.c_uint64 if name == "scratch_bytes" else C.c_int32)
                    for name in fields]

    def make(role):
        def fn(capacity, out):
            data = [2, role, 384, 5120, 384, 384, 6, capacity, 4096, 32, 32, 64, 64, 188, 7]
            C.memmove(out, (C.c_int32 * len(data))(*data), C.sizeof(_Info))
            return 0
        return type("L", (), {"ds41rt_v41_expert_info": staticmethod(fn)})()

    # role 9 is what the legacy formula would have expected: must be rejected.
    with pytest.raises(AssertionError):
        module.assert_native_role_geometry(make(9), 16, tp_degree=6)
    # role 7 is the registered TP6 family: accepted.
    assert module.assert_native_role_geometry(make(7), 16, tp_degree=6).role == 7
