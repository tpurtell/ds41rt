"""Isolated toy CMake fixture for the Spark TP2/TP3 width knobs.

Runs a real CMake configure/build with the UnixMakefiles generator in a guarded
root-NVMe temporary directory. The actual
``native/cmake/v41_spark_tp_experts.cmake`` is included with a stub exporter that
records the argv it receives, so re-export invalidation is proven by CMake/Make
behavior rather than by source-string assertions.

No native project, CUDA, Cargo, real export, benchmark or GPU work happens here:
the toy project has no language, the verify command is ``cmake -E true``, and the
"exporter" only writes stub objects, a per-role manifest and an append-only call
log. Nothing outside the temporary directory is written.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMAKE_FILE = ROOT / "native" / "cmake" / "v41_spark_tp_experts.cmake"

DEFAULT_MAP = "1:64,16:192,80:192,256:192,1024:192,4096:192"
TP2_CAP80_128 = "1:64,16:128,80:128,256:192,1024:192,4096:192"

_STUB_EXPORTER = '''\
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]

    def value(flag):
        return argv[argv.index(flag) + 1]

    role = value("--role")
    out = Path(value("--output-dir"))
    rows = [int(x) for x in value("--rows").split(",")]
    width = value("--width")
    atomic = value("--atomic-min-capacity") if "--atomic-min-capacity" in argv else None
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (out / ("v41_%s_m%d.o" % (role, row))).write_bytes(b"stub\\n")
        (out / ("v41_%s_m%d.h" % (role, row))).write_text("// stub\\n", encoding="utf-8")
    (out / "v41_expert_variants.h").write_text("// stub\\n", encoding="utf-8")
    (out / "v41_experts.json").write_text(
        json.dumps(dict(role=role, rows=rows, width=width,
                        atomic_min_capacity=atomic)) + "\\n", encoding="utf-8")
    log = os.environ.get("DS41RT_WIDTH_EXPORT_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(role=role, width=width,
                                         atomic_min_capacity=atomic)) + "\\n")


main()
'''

_TOY_CMAKE = '''\
cmake_minimum_required(VERSION 3.20)
project(ds41rt_width_fixture NONE)
set(DS41RT_ENABLE_V41_EXPERT_AOT ON)
set(DS41RT_V41_EXPERT_ROLE spark)
set(DS41RT_V41_SPARK_TP_ROLES "tp2;tp3" CACHE STRING "roles")
set(DS41RT_SPARKINFER_VERIFY_COMMAND "${CMAKE_COMMAND}" -E true)
set(DS41RT_SPARKINFER_PYTHON_ENV "")
set(DS41RT_SPARKINFER_PROVENANCE_INPUTS "")
set(DS41RT_SPARKINFER_EXPORT_INPUTS "")
set(DS41RT_NATIVE_SOURCES "")
set(Python3_EXECUTABLE "__PYTHON__")
add_custom_target(ds41rt_verify_sparkinfer_source)
include("__CMAKE_FILE__")
'''


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600, check=False
    )


def run_width_override_scenario() -> dict:
    """Configure/build a toy project four times and return the observed behavior.

    Steps: default -> TP2 cap80=128 -> same override again -> default again.
    Raises RuntimeError with the failing command output on any non-zero exit.
    """
    base = Path(
        os.environ.get(
            "DS41RT_WIDTH_TEST_TMPDIR", str(Path.home() / ".cache" / "ds41rt" / "tests")
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="width-cmake-", dir=base))
    try:
        src = tmp / "native"
        (src / "src").mkdir(parents=True)
        tools = tmp / "python" / "tools"
        tools.mkdir(parents=True)
        (src / "CMakeLists.txt").write_text(
            _TOY_CMAKE.replace("__PYTHON__", str(sys.executable)).replace(
                "__CMAKE_FILE__", str(CMAKE_FILE)
            ),
            encoding="utf-8",
        )
        for wrapper in ("v41_spark_tp2_experts.cc", "v41_spark_tp3_experts.cc"):
            (src / "src" / wrapper).write_text("// stub\n", encoding="utf-8")
        (tools / "export_b12x_v41_slices_aot.py").write_text(
            _STUB_EXPORTER, encoding="utf-8"
        )
        (tools / "export_b12x_v41_experts_aot.py").write_text(
            "# stub\n", encoding="utf-8"
        )
        build = tmp / "build"
        build.mkdir()
        log = tmp / "exports.log"
        log.touch()
        env = dict(os.environ, DS41RT_WIDTH_EXPORT_LOG=str(log))

        def configure(extra: list[str]) -> dict:
            proc = _run(
                ["cmake", "-S", str(src), "-B", str(build), "-G", "Unix Makefiles"]
                + extra,
                tmp,
                env,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"configure failed: {proc.stdout}\n{proc.stderr}")
            return {"returncode": proc.returncode, "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:]}

        def build_targets() -> dict:
            proc = _run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "ds41rt_v41_spark_tp2_experts_export",
                    "ds41rt_v41_spark_tp3_experts_export",
                ],
                tmp,
                env,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"build failed: {proc.stdout}\n{proc.stderr}")
            return {"returncode": proc.returncode, "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:]}

        def exports() -> dict[str, list[dict]]:
            observed: dict[str, list[dict]] = {"spark_tp2": [], "spark_tp3": []}
            for line in log.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                observed.setdefault(item["role"], []).append(item)
            return observed

        def manifest(role: str) -> dict:
            path = build / f"v41_{role}_experts" / "v41_experts.json"
            return json.loads(path.read_text(encoding="utf-8"))

        steps = {}
        steps["default_configure"] = configure([])
        steps["default_build"] = build_targets()
        steps["after_default"] = exports()
        steps["default_manifests"] = {
            role: manifest(role) for role in ("spark_tp2", "spark_tp3")
        }

        override = [f"-DDS41RT_V41_SPARK_TP2_SLICE_WIDTH={TP2_CAP80_128}"]
        steps["override_configure"] = configure(override)
        steps["override_build"] = build_targets()
        steps["after_override"] = exports()
        steps["override_manifests"] = {
            role: manifest(role) for role in ("spark_tp2", "spark_tp3")
        }

        # Reconfigure with an unchanged override: the content-stable stamp must
        # leave both timestamps alone, so Make re-exports nothing.
        steps["unchanged_configure"] = configure(override)
        steps["unchanged_build"] = build_targets()
        steps["after_unchanged"] = exports()

        # Restoring the default must re-export TP2 again (both directions). A -D
        # override persists in the CMake cache, so the default is passed
        # explicitly rather than by omitting the flag.
        steps["restore_configure"] = configure(
            [f"-DDS41RT_V41_SPARK_TP2_SLICE_WIDTH={DEFAULT_MAP}"]
        )
        steps["restore_build"] = build_targets()
        steps["after_restore"] = exports()
        return steps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
