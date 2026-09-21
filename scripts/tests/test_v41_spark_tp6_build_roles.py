#!/usr/bin/env python3
"""Executable coverage for `build.sh` spark-TP role selection and the
`write-v41-expert-tp-manifest.py` role table.

No hardware, Docker, SSH, cmake or cargo is touched. The test extracts the real
role-selection block from `build.sh` by its boundary comments, sources the real
`scripts/release-common.sh` for `release_die`/`release_spark_topology_explicit`,
and runs the block in bash under a matrix of configurations. A change to the
allowlist, the topology map or the multi-role syntax fails here rather than at
release time.

The manifest half writes a synthetic role export for a requested role and checks
that the writer accepts the geometry it was told to expect (tp6: intermediate
384, no padding) and rejects a mismatched one, so a future geometry disagreement
cannot pass silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
WIP = REPO / "wip.sh"
RELEASE_COMMON = REPO / "scripts" / "release-common.sh"
MANIFEST = REPO / "scripts" / "write-v41-expert-tp-manifest.py"

# Both scripts resolve the opt-in Spark TP roles from the same inputs with their
# own variable names (`spark_tp_roles` for build.sh, `wip_spark_tp_roles` for
# wip.sh) before their dry-run block. wip.sh names the WIP role variable in its
# usage text earlier in the file, so its block is located from the right.
BLOCK_START = "# Opt-in replicated-group Spark expert roles."
BLOCK_END = "if ((dry_run)); then"


def _role_block(script: Path = BUILD, *, variable: str = "spark_tp_roles",
                last: bool = False, marker: str = BLOCK_START) -> str:
    text = script.read_text(encoding="utf-8")
    finder = text.rindex if last else text.index
    start = finder(marker)
    end = text.index(BLOCK_END, start)
    block = text[start:end]
    assert f"{variable}=" in block, f"{script.name} role block moved"
    # wip.sh names its loop variable `wip_spark_tp_role`, build.sh uses
    # `spark_tp_role`; both must still carry the validation case.
    loop_var = "wip_spark_tp_role" if variable.startswith("wip_") else "spark_tp_role"
    assert f'case "${loop_var}" in' in block, (
        f"{script.name} role selector moved; update this extractor"
    )
    return block


def _run_roles(spark_tp: str, override: str | None) -> subprocess.CompletedProcess:
    harness = f"""
set -uo pipefail
source "{RELEASE_COMMON}"
{_role_block()}
printf '%s\\n' "${{spark_tp_roles}}"
"""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "SPARK_TP": spark_tp,
        # `release_spark_topology_explicit` reads SPARK_EP under `set -u`; an
        # explicit topology always sets it, so mirror that here.
        "SPARK_EP": "1" if spark_tp else "",
    }
    # An unset override means "derive from the topology"; a non-empty override
    # is the documented multi-role escape hatch.
    if override:
        env["DS41RT_RELEASE_SPARK_TP_ROLES"] = override
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, env=env,
        timeout=60, check=False,
    )


def _run_wip_roles(spark_tp: str, override: str | None) -> subprocess.CompletedProcess:
    harness = f"""
set -uo pipefail
source "{RELEASE_COMMON}"
{_role_block(WIP, variable="wip_spark_tp_roles", last=True,
             marker="# Opt-in replicated-group Spark expert roles")}
printf '%s\\n' "${{wip_spark_tp_roles}}"
"""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "SPARK_TP": spark_tp,
        "SPARK_EP": "1" if spark_tp else "",
    }
    if override:
        env["DS41RT_WIP_SPARK_TP_ROLES"] = override
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, env=env,
        timeout=60, check=False,
    )


@pytest.mark.parametrize(
    "spark_tp,override,expected",
    [
        # Default: no explicit topology, no override -> only the implicit TP4.
        ("", None, ""),
        # Explicit TP4xEP1 also selects no extra role.
        ("4", None, ""),
        # The new explicit TP6 topology selects the tp6 role.
        ("6", None, "tp6"),
        # SPARK_TP6 with no override maps to tp6, not to a generic role.
        ("2", None, "tp2"),
        ("3", None, "tp3"),
        # Multi-role escape hatch, including the release combination.
        (None, "tp2;tp3;tp6", "tp2;tp3;tp6"),
        (None, "tp6", "tp6"),
        # The escape hatch wins over the topology mapping.
        ("4", "tp2;tp3;tp6", "tp2;tp3;tp6"),
    ],
)
def test_build_role_selection_matrix(spark_tp, override, expected) -> None:
    result = _run_roles(spark_tp if spark_tp else "", override)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_build_rejects_an_unknown_role_with_a_clear_error() -> None:
    result = _run_roles("", "tp5")
    assert result.returncode == 2, result.stdout
    assert "accepts only tp2, tp3 and tp6" in result.stderr
    assert "tp5" in result.stderr


@pytest.mark.parametrize(
    "spark_tp,override,expected",
    [
        ("", None, ""),
        ("4", None, ""),
        ("6", None, "tp6"),
        ("2", None, "tp2"),
        ("3", None, "tp3"),
        (None, "tp2;tp3;tp6", "tp2;tp3;tp6"),
        (None, "tp6", "tp6"),
        ("4", "tp2;tp3;tp6", "tp2;tp3;tp6"),
    ],
)
def test_wip_role_selection_matrix(spark_tp, override, expected) -> None:
    """wip.sh must resolve the same role plan as build.sh.

    Before this, wip.sh accepted only tp2|tp3, so the candidate six-role WIP
    build aborted at argument validation even though the artifact script and
    CMake already understood tp6.
    """
    result = _run_wip_roles(spark_tp if spark_tp else "", override)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def test_wip_rejects_an_unknown_role_and_names_the_wip_variable() -> None:
    result = _run_wip_roles("", "tp5")
    assert result.returncode == 2, result.stdout
    assert "DS41RT_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6" in result.stderr


def test_wip_and_build_agree_on_every_supported_topology() -> None:
    for spark_tp in ("2", "3", "4", "6"):
        build = _run_roles(spark_tp, None)
        wip = _run_wip_roles(spark_tp, None)
        assert build.returncode == wip.returncode == 0, (spark_tp, build.stderr, wip.stderr)
        assert build.stdout.strip() == wip.stdout.strip(), spark_tp


def test_release_and_wip_artifact_scripts_share_the_allowlist() -> None:
    for path, variable in (
        (REPO / "scripts" / "build-release-artifacts.sh", "DS41RT_RELEASE_SPARK_TP_ROLES"),
        (REPO / "scripts" / "build-wip-artifacts.sh", "DS41RT_WIP_SPARK_TP_ROLES"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "tp2|tp3|tp6)" in text, path
        assert f"{variable} accepts only tp2, tp3 and tp6" in text, path


def _role_geometry(role: str) -> tuple:
    sys.path.insert(0, str(REPO / "python" / "tools"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_tp6_export_tables", REPO / "python" / "tools" / "export_b12x_v41_slices_aot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ROLE_GEOMETRY[role]


CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def _write_export(directory: Path, role: str) -> Path:
    import hashlib

    experts, intermediate, kernel_intermediate, topk = _role_geometry(role)
    directory.mkdir(parents=True, exist_ok=True)
    variants = []
    artifact_sha256 = {}
    for capacity in CAPACITIES:
        header = directory / f"v41_{role}_m{capacity}.h"
        obj = directory / f"v41_{role}_m{capacity}.o"
        header.write_text("// stub\n", encoding="utf-8")
        obj.write_bytes(f"stub-object-{capacity}\n".encode())
        variants.append({"name": f"v41_{role}_m{capacity}", "capacity_rows": capacity})
        artifact_sha256[header.name] = hashlib.sha256(header.read_bytes()).hexdigest()
        artifact_sha256[obj.name] = hashlib.sha256(obj.read_bytes()).hexdigest()
    manifest = {
        "schema": 1,
        "role": role,
        "spark_tp_degree": {"spark": 4, "spark_tp2": 2, "spark_tp3": 3, "spark_tp6": 6}[role],
        "capability": [12, 1],
        "sparkinfer_revision": "0" * 40,
        "geometry": {
            "experts": experts,
            "hidden": 5120,
            "intermediate": intermediate,
            "kernel_intermediate": kernel_intermediate,
            "topk": topk,
        },
        "variants": variants,
        "artifact_sha256": artifact_sha256,
    }
    path = directory / "v41_experts.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_writer_accepts_tp6_and_rejects_a_wrong_extent(tmp_path: Path) -> None:
    role = "tp6"
    export_dir = tmp_path / f"v41_spark_{role}_experts"
    _write_export(export_dir, "spark_tp6")
    output = tmp_path / "V41_EXPERT_TP_AOT.json"
    # The writer requires the built library whenever a role is requested; a stub
    # file exercises the hash/validation path without any native build.
    library = tmp_path / "libds41rt_native.so"
    library.write_bytes(b"\x7fELF-stub\n")

    def invoke() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MANIFEST), "--role", "expert", "--requested", role,
             "--native-build-dir", str(tmp_path), "--native-library", str(library),
             "--output", str(output)],
            capture_output=True, text=True, timeout=120, check=False,
        )

    result = invoke()
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["spark_tp_roles"] == ["tp6"]
    assert document["manifests"]["tp6"]["spark_tp_degree"] == 6
    assert document["manifests"]["tp6"]["geometry"]["intermediate"] == 384
    assert document["manifests"]["tp6"]["geometry"]["kernel_intermediate"] == 384

    # A geometry that disagrees with the official 384 must fail closed.
    manifest_path = export_dir / "v41_experts.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["geometry"]["kernel_intermediate"] = 640
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    failed = invoke()
    assert failed.returncode == 2
    assert "kernel_intermediate" in failed.stderr


def test_manifest_writer_rejects_an_unknown_role_before_touching_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "V41_EXPERT_TP_AOT.json"
    result = subprocess.run(
        [sys.executable, str(MANIFEST), "--role", "expert", "--requested", "tp5",
         "--native-build-dir", str(tmp_path), "--output", str(output)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 2
    assert "expected tp2, tp3 or tp6" in result.stderr
    assert not output.exists()
