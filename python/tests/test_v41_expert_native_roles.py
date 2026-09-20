"""CPU-only contract checks for the replicated-group native expert wrapper.

Covers the pure symbol/geometry selection added for Spark TP2/TP3 (roles 5/6)
without loading a library or touching a GPU: the ABI structs, the family prefix
mapping, and mutual exclusion between families.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python" / "tools" / "_v41_expert_native.py"


def _load():
    torch = pytest.importorskip("torch")
    assert torch is not None
    spec = importlib.util.spec_from_file_location("_v41_expert_native_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abi_struct_layout_is_unchanged() -> None:
    module = _load()
    assert C.sizeof(module.Info) == 64
    assert C.sizeof(module.Launch) == 392
    assert module.Launch.stream.offset == 384


def test_symbol_prefix_selects_new_spark_families() -> None:
    module = _load()
    assert module.expert_symbol_prefix() == "ds41rt_v41_expert_"
    assert module.expert_symbol_prefix(local=True) == "ds41rt_v41_local_expert_"
    assert module.expert_symbol_prefix(tp2=True) == "ds41rt_v41_tp2_expert_"
    assert module.expert_symbol_prefix(spark_tp=2) == "ds41rt_v41_spark_tp2_expert_"
    assert module.expert_symbol_prefix(spark_tp=3) == "ds41rt_v41_spark_tp3_expert_"
    assert module.SPARK_TP_PREFIX == {
        2: "ds41rt_v41_spark_tp2_expert_",
        3: "ds41rt_v41_spark_tp3_expert_",
    }


def test_only_launch_abi_symbols_are_namespaced() -> None:
    module = _load()
    prefix = "ds41rt_v41_spark_tp2_expert_"
    assert module.namespaced_symbol("ds41rt_v41_expert_launch", prefix) == \
        "ds41rt_v41_spark_tp2_expert_launch"
    assert module.namespaced_symbol("ds41rt_v41_expert_info", prefix) == \
        "ds41rt_v41_spark_tp2_expert_info"
    # The packer size query and the packer are canonical for every family.
    assert module.namespaced_symbol("ds41rt_v41_expert_packed_sizes", prefix) == \
        "ds41rt_v41_expert_packed_sizes"
    assert module.namespaced_symbol("ds41rt_v41_pack_expert_async", prefix) == \
        "ds41rt_v41_pack_expert_async"
    # The canonical prefix is an identity mapping.
    assert module.namespaced_symbol("ds41rt_v41_expert_launch",
                                    "ds41rt_v41_expert_") == "ds41rt_v41_expert_launch"


def test_symbol_families_are_mutually_exclusive() -> None:
    module = _load()
    for kwargs in (
        {"local": True, "tp2": True},
        {"spark_tp": 2, "tp2": True},
        {"spark_tp": 3, "local": True},
        {"spark_tp": 2, "spark_tp": 3, "tp2": True},
    ):
        with pytest.raises(AssertionError):
            module.expert_symbol_prefix(**kwargs)
    with pytest.raises(AssertionError):
        module.expert_symbol_prefix(spark_tp=4)
    assert module.expert_symbol_prefix(spark_tp=2) == "ds41rt_v41_spark_tp2_expert_"
