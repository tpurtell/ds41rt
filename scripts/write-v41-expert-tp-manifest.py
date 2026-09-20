#!/usr/bin/env python3
"""Write the built-role manifest for a DS41RT expert image.

The manifest is derived from the AOT export manifests that CMake actually
produced, never from a caller-supplied claim, and it is only written after the
role export has been validated against the build that produced it:

* the export manifest declares ``schema == 1`` and ``capability == [12, 1]``;
* its geometry matches the official checkpoint (experts 384, hidden 5120,
  intermediate 1152 for tp2 / 768 for tp3, topk 6) with no kernel padding
  (``kernel_intermediate == intermediate``);
* its non-empty ``variants`` cover capacities 1/16/80/256/1024/4096;
* every file named in ``artifact_sha256`` exists in the role export directory
  and hashes to the declared value, so a stale or partial JSON cannot be
  advertised; and
* when ``--native-library`` is given (required whenever a role is requested)
  the built shared library is hashed into the manifest and, when a symbol
  reader (``nm -D`` or ``readelf``) is available, its dynamic symbols for the
  role are verified.

Trust boundary: the manifest binds the *role export directory contents* and the
built library hash. Symbol verification additionally proves the library was
linked with the role module. It does not execute the kernels and does not prove
numerical correctness or device compatibility; that remains a hardware
qualification gate.

Exit status 2 means the build did not produce a validated role it was asked
for; the build must fail rather than ship an image that advertises an unbuilt
topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROLE_TP_DEGREE = {"tp2": 2, "tp3": 3}
ROLE_INTERMEDIATE = {"tp2": 1152, "tp3": 768}
ROLE_INFO_SYMBOL = {
    "tp2": "ds41rt_v41_spark_tp2_expert_info",
    "tp3": "ds41rt_v41_spark_tp3_expert_info",
}
ROLE_LAUNCH_SYMBOL = {
    "tp2": "ds41rt_v41_spark_tp2_expert_launch",
    "tp3": "ds41rt_v41_spark_tp3_expert_launch",
}
EXPECTED_EXPERTS = 384
EXPECTED_HIDDEN = 5120
EXPECTED_TOPK = 6
EXPECTED_CAPABILITY = [12, 1]
EXPECTED_CAPACITIES = (1, 16, 80, 256, 1024, 4096)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> "None":
    print(f"write-v41-expert-tp-manifest: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_roles(value: str) -> list[str]:
    roles = [entry for entry in value.split(";") if entry]
    if len(set(roles)) != len(roles):
        fail(f"duplicate Spark TP role in {value!r}")
    for role in roles:
        if role not in ROLE_TP_DEGREE:
            fail(f"unsupported Spark TP role {role!r}; expected tp2 or tp3")
    return roles


def dynamic_symbols(library: Path) -> tuple[set[str], str]:
    """Return (symbols, tool). An empty tool means no reader was available."""
    for tool, command in (
        ("nm", ["nm", "-D", "--defined-only", str(library)]),
        ("readelf", ["readelf", "--dyn-syms", "--wide", str(library)]),
    ):
        if shutil.which(tool) is None:
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        symbols = set()
        for line in result.stdout.splitlines():
            fields = line.split()
            if not fields:
                continue
            name = fields[-1].split("@", 1)[0]
            symbols.add(name)
        return symbols, tool
    return set(), ""


def validate_role_export(role: str, manifest_path: Path, export_dir: Path) -> dict:
    if not manifest_path.is_file():
        fail(f"requested role {role} produced no manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != 1:
        fail(f"{manifest_path}: schema {payload.get('schema')!r}, expected 1")
    if payload.get("capability") != EXPECTED_CAPABILITY:
        fail(f"{manifest_path}: capability {payload.get('capability')!r}, expected {EXPECTED_CAPABILITY}")
    if payload.get("role") != f"spark_{role}":
        fail(f"{manifest_path}: role {payload.get('role')!r}, expected spark_{role}")
    expected_degree = ROLE_TP_DEGREE[role]
    if payload.get("spark_tp_degree") != expected_degree:
        fail(f"{manifest_path}: spark_tp_degree {payload.get('spark_tp_degree')!r}, expected {expected_degree}")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        fail(f"{manifest_path}: missing geometry")
    expected_intermediate = ROLE_INTERMEDIATE[role]
    expected_geometry = {
        "experts": EXPECTED_EXPERTS,
        "hidden": EXPECTED_HIDDEN,
        "intermediate": expected_intermediate,
        "kernel_intermediate": expected_intermediate,
        "topk": EXPECTED_TOPK,
    }
    for key, expected in expected_geometry.items():
        if geometry.get(key) != expected:
            fail(f"{manifest_path}: geometry.{key} {geometry.get(key)!r}, expected {expected}")

    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        fail(f"{manifest_path}: variants must be a non-empty list")
    capacities = sorted({variant.get("capacity_rows") for variant in variants if isinstance(variant, dict)})
    missing = [capacity for capacity in EXPECTED_CAPACITIES if capacity not in capacities]
    if missing:
        fail(f"{manifest_path}: variants do not cover capacities {missing}; have {capacities}")

    artifact_hashes = payload.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        fail(f"{manifest_path}: artifact_sha256 must be a non-empty mapping")
    verified_objects = 0
    for name, expected_digest in artifact_hashes.items():
        artifact = export_dir / name
        if not artifact.is_file():
            fail(f"{manifest_path}: artifact {name} is missing from {export_dir}")
        actual = sha256_file(artifact)
        if actual != expected_digest:
            fail(f"{manifest_path}: artifact {name} hashes to {actual}, expected {expected_digest}")
        if artifact.suffix == ".o":
            verified_objects += 1
    if verified_objects == 0:
        fail(f"{manifest_path}: artifact_sha256 lists no compiled object")

    return {
        "file": str(manifest_path.relative_to(export_dir.parent)),
        "sha256": sha256_file(manifest_path),
        "spark_tp_degree": expected_degree,
        "geometry": geometry,
        "capability": payload.get("capability"),
        "sparkinfer_revision": payload.get("sparkinfer_revision"),
        "capacities": capacities,
        "verified_artifacts": len(artifact_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("coordinator", "expert"), required=True)
    parser.add_argument("--requested", default="", help="semicolon-separated tp2/tp3 list")
    parser.add_argument("--native-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-library",
        type=Path,
        default=None,
        help="built libds41rt_native.so; required when a role is requested",
    )
    args = parser.parse_args()

    roles = parse_roles(args.requested)
    if args.role != "expert" and roles:
        fail("only the expert role can build extra Spark TP roles")

    native_library_sha256 = None
    symbols: set[str] = set()
    symbol_tool = ""
    if args.native_library is not None:
        if not args.native_library.is_file():
            fail(f"native library not found: {args.native_library}")
        native_library_sha256 = sha256_file(args.native_library)
        if roles:
            symbols, symbol_tool = dynamic_symbols(args.native_library)
    elif roles:
        fail("--native-library is required when requesting Spark TP roles")

    manifests: dict[str, dict] = {}
    for role in roles:
        export_dir = args.native_build_dir / f"v41_spark_{role}_experts"
        manifests[role] = validate_role_export(
            role, export_dir / "v41_experts.json", export_dir
        )
        if args.native_library is not None and symbol_tool:
            missing_symbols = [
                symbol
                for symbol in (ROLE_INFO_SYMBOL[role], ROLE_LAUNCH_SYMBOL[role])
                if symbol not in symbols
            ]
            if missing_symbols:
                fail(
                    f"{args.native_library} is missing {role} symbols {missing_symbols}; "
                    "the role module was not linked into the built library"
                )

    document = {
        "schema": 1,
        "role": args.role,
        "requested": args.requested,
        "spark_tp_roles": roles,
        "native_library_sha256": native_library_sha256,
        "symbols_verified": bool(roles) and bool(symbol_tool),
        "symbol_verification_tool": symbol_tool or None,
        "manifests": manifests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
