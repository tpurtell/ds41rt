from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python/tools/export_b12x_v41_nvfp4_aot.py"
CMAKE = ROOT / "native/cmake/v41_nvfp4_experts.cmake"


@pytest.fixture
def exporter(monkeypatch):
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", NS(REVISION="test"))
    spec = importlib.util.spec_from_file_location("nvfp4_export_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,expected", [(None, 16), ("auto", None), ("16", 16),
                                           ("32", 32), ("64", 64), ("128", 128)])
def test_cli_tile_policy(exporter, monkeypatch, tmp_path, value, expected):
    arguments = [str(EXPORTER), "--output-dir", str(tmp_path), "--role", "rtx_tp2"]
    if value is not None:
        arguments += ["--tile-m", value]
    monkeypatch.setattr(sys, "argv", arguments)
    exporter.export = Mock()
    exporter.main()
    assert exporter.export.call_args.args[3] == expected


@pytest.mark.parametrize("value", ["0", "8", "256", "AUTO", "16.0"])
def test_cli_rejects_invalid_tiles(exporter, monkeypatch, tmp_path, value):
    monkeypatch.setattr(sys, "argv", [str(EXPORTER), "--output-dir", str(tmp_path),
                                    "--role", "rtx_tp2", "--tile-m", value])
    exporter.export = Mock()
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 2
    exporter.export.assert_not_called()


@pytest.mark.parametrize("value", [0, 8, 256, "auto", 16.0, True])
def test_export_rejects_invalid_tiles_before_gpu_import(exporter, monkeypatch, tmp_path, value):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="tile_m"):
        exporter.export(tmp_path / "output", "rtx_tp2", [1], value)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("value", [-1, 3, 6, True, 5.0, "5", None])
def test_export_rejects_invalid_shards_before_gpu_import(exporter, monkeypatch, tmp_path, value):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="output_shards"):
        exporter.export(tmp_path / "output", "rtx_tp2", [1], 16, output_shards=value)


@pytest.mark.parametrize("requested", [None, 16, 32, 64, 128])
@pytest.mark.parametrize("shards", [0, 1, 5])
def test_export_uses_resolved_scratch_tile(exporter, monkeypatch, tmp_path, requested, shards):
    dtype = NS(itemsize=2)
    torch = NS(bfloat16=dtype, int32="int32", device=lambda *args: "cuda",
               cuda=NS(init=Mock(), current_device=lambda: 0,
                       get_device_properties=lambda _: NS(major=12, minor=0,
                           name="mock GPU", multi_processor_count=100)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    configs, resolved, compiled_tiles, compiled_shards = [], [], [], []

    def plan_scratch(caps, **kwargs):
        configs.append(caps.decode_config)
        # Deliberately resolve a different tile for explicit requests too: the
        # exporter must trust the plan, never its input or a second policy call.
        tile = {1: 16, 80: 32, 256: 64, 4096: 128}[caps.max_tokens]
        resolved.append(tile)
        core = NS(tensor_specs=[
            NS(name="packed_input", shape=(1, tile), dtype=dtype, init=None),
            NS(name="route_output", shape=(caps.max_tokens * 6, 5120), dtype=dtype, init=None),
        ], dynamic_task_capacity=7, dynamic_physical_tiles=9)
        return NS(launch_plan=NS(routed_rows=caps.max_tokens * 6, max_rows=tile,
                                execution=NS(tile_m=tile)), _core_workspace_plan=core)

    def get_kernel(*args, **kwargs):
        compiled_tiles.append(kwargs["planned_tile_m"])
        compiled_shards.append(kwargs["nvfp4_output_shards"])
        def export_to_c(output, label, symbol):
            (Path(output) / f"{label}.h").write_text(
                f"void _mlir_{symbol}(void **args, int32_t num_args);\n"
                f"void *args[{exporter.LAUNCH_ARGUMENT_COUNT}] = {{}};\n")
        return NS(export_to_c=export_to_c), 10

    moe = NS(plan_b12x_fp4_moe_weights=lambda **kwargs: object(),
             TPMoEScratchCaps=lambda **kwargs: NS(**kwargs),
             plan_tp_moe_scratch=plan_scratch, _get_dynamic_kernel=get_kernel,
             _core_workspace_nbytes=lambda core: sum(
                 spec.dtype.itemsize * math.prod(spec.shape)
                 for spec in core.tensor_specs))
    monkeypatch.setitem(sys.modules, "b12x", NS())
    monkeypatch.setitem(sys.modules, "b12x.moe", NS())
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe", NS(_impl=moe))
    monkeypatch.setitem(sys.modules, "b12x.moe.fused_moe._tuning",
                        NS(MoeDecodeConfig=lambda **kwargs: NS(**kwargs)))
    exporter.export(tmp_path, "rtx_tp2", [1, 80, 256, 4096], requested, output_shards=shards)
    manifest = json.loads((tmp_path / "v41_nvfp4_experts.json").read_text())
    assert [config.dynamic_tile_m for config in configs] == [requested] * 4
    assert compiled_tiles == resolved == [16, 32, 64, 128]
    expected_shards = [0] * 4 if shards == 0 else [shards, 1, 1, 1]
    assert compiled_shards == expected_shards
    assert [v["output_shards"] for v in manifest["variants"]] == expected_shards
    assert manifest["tile_m"] == requested
    assert [variant["tile_m"] for variant in manifest["variants"]] == resolved
    assert [variant["route_mode"] for variant in manifest["variants"]] == [
        "direct", "grouped", "grouped", "grouped"]


@pytest.mark.parametrize("value", [None, "auto", "16", "32", "64", "128", "8", "AUTO", ""])
def test_cmake_tile_validation(tmp_path, value):
    if shutil.which("cmake") is None:
        pytest.skip("cmake unavailable")
    # Evaluate the real cache/default/validation declarations without needing
    # CUDA or invoking the custom export commands in this CPU-only test.
    source = CMAKE.read_text()
    declarations = source.split("set(DS41RT_V41_NVFP4_INCLUDE_DIRS)")[0]
    script = tmp_path / "policy.cmake"
    script.write_text('set(DS41RT_ENABLE_CUDA ON)\nset(DS41RT_CUDA_ARCHITECTURES 120)\n'
                      + declarations + '\nmessage(STATUS "TILE=${DS41RT_V41_NVFP4_TILE_M}")\n')
    args = ["cmake"]
    if value is not None:
        args += [f"-DDS41RT_V41_NVFP4_TILE_M={value}"]
    result = subprocess.run([*args, "-P", str(script)], capture_output=True, text=True)
    if value in ("8", "AUTO", ""):
        assert result.returncode != 0
        assert "must be auto, 16, 32, 64, or 128" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f"TILE={value or '16'}" in result.stdout
    assert '--tile-m "${DS41RT_V41_NVFP4_TILE_M}"' in source
    assert "PROPERTY STRINGS auto 16 32 64 128" in source
