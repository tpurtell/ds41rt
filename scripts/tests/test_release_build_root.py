#!/usr/bin/env python3
"""CPU-only behavioural proof that the release build root really moves.

`build.sh` binds one unique per-task path into both release container legs and
exports it as `DS41RT_RELEASE_BUILD_ROOT`; `build-release-artifacts.sh` creates its
writable staging copy beneath that parent and builds the daemon and the native
library there. Relocating it cannot be done with `TMPDIR`, because mktemp is handed
an absolute template, so this test drives the real script twice with stub
`cargo`/`cmake`/`python3` on PATH and reads back every directory those stages were
asked to write:

* unset -> the historical `/tmp/ds41rt-release-build.XXXXXX` scratch, removed after;
* set   -> only paths beneath the requested parent, and that parent is what the
  filesystem guard was asked about.

The stub `python3` logs the guard invocation instead of resolving it: this
checkout's temporary directories are not always probeable by findmnt, and the
guard's own behaviour is covered by `test_build_filesystem.py` and
`test_release_artifact_build_paths.py`. No GPU, Docker, SSH, real Cargo or real
CMake is involved. The fake source tree and stub writers are imported from
`test_release_artifact_build_paths.py` so the two files share one harness.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIBLING = REPO / "scripts" / "tests" / "test_release_artifact_build_paths.py"
# Root NVMe, workspace-local and excluded from the release source inventory
# (.ds41rt-cache is in the manifest tool's ignored names).
SCRATCH = REPO / ".ds41rt-cache" / "test-release-build-root"


def _load_harness():
    spec = importlib.util.spec_from_file_location("release_artifact_build_paths", SIBLING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


@pytest.fixture(autouse=True)
def _clean_scratch():
    yield
    shutil.rmtree(SCRATCH, ignore_errors=True)


def _recording_shims(log: Path, guard_log: Path) -> dict[str, str]:
    """Stubs that log the directories they were told to write into, then succeed."""
    return {
        "cargo": f"""
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$CARGO_TARGET_DIR" >>{log}
            mkdir -p "$CARGO_TARGET_DIR/release"
            printf 'daemon\\n' >"$CARGO_TARGET_DIR/release/ds41rt"
            chmod +x "$CARGO_TARGET_DIR/release/ds41rt"
            """,
        "cmake": f"""
            #!/usr/bin/env bash
            set -euo pipefail
            build_dir=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -B) build_dir="$2"; shift 2 ;;
                *) shift ;;
              esac
            done
            [[ -n "$build_dir" ]] || exit 0
            printf '%s\\n' "$build_dir" >>{log}
            mkdir -p "$build_dir/v41_experts" "$build_dir/v41_fp8"
            printf 'native\\n' >"$build_dir/libds41rt_native.so"
            printf '{{}}\\n' >"$build_dir/v41_experts/v41_experts.json"
            printf '{{}}\\n' >"$build_dir/v41_fp8/v41_fp8.json"
            """,
        "python3": f"""
            #!/usr/bin/env bash
            set -euo pipefail
            for arg in "$@"; do
              if [[ "$arg" == *assert-build-filesystem.py ]]; then
                # Log the probe as one token per line, then accept it: this sandbox
                # may not resolve every temporary path, and the guard's own logic is
                # covered elsewhere. What matters here is which path was probed.
                for token in "$@"; do printf 'A\\t%s\\n' "$token" >>{guard_log}; done
                printf 'E\\n' >>{guard_log}
                exit 0
              fi
            done
            previous=""
            for arg in "$@"; do
              case "$previous" in
                --output|--write)
                  mkdir -p "$(dirname "$arg")"
                  printf '{{}}\\n' >"$arg"
                  ;;
              esac
              previous="$arg"
            done
            """,
    }


def _parse_records(text: str) -> list[list[str]]:
    records: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line == "E":
            records.append(current)
            current = []
        elif line.startswith("A\t"):
            current.append(line[2:])
        else:
            raise AssertionError(f"unexpected guard log line: {line!r}")
    assert not current, f"truncated guard log: {current}"
    return records


def _recorded_dirs(
    tmp_path: Path, build_root: Path | None, create: bool = True
) -> tuple[list[Path], list[list[str]], subprocess.CompletedProcess[str]]:
    src = HARNESS._fake_source(tmp_path)
    output = tmp_path / "output"
    shims = tmp_path / "shims"
    shims.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "written-dirs.log"
    guard_log = tmp_path / "guard-invocations.jsonl"

    for name, body in _recording_shims(log, guard_log).items():
        HARNESS._write_stub(shims / name, body)

    env = dict(os.environ)
    env["PATH"] = f"{shims}:{env['PATH']}"
    env.pop("CARGO_TARGET_DIR", None)
    env.pop("DS41RT_RELEASE_SPARK_TP_ROLES", None)
    env.pop("DS41RT_RELEASE_BUILD_ROOT", None)
    if build_root is not None:
        if create:
            build_root.mkdir(parents=True, exist_ok=True)
        env["DS41RT_RELEASE_BUILD_ROOT"] = str(build_root)

    result = subprocess.run(
        ["bash", str(HARNESS.SCRIPT), str(src), "coordinator", "120", str(output)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    written = [Path(line) for line in log.read_text().splitlines() if line.strip()]
    assert written, "neither cargo nor cmake reported a write directory"
    guarded = _parse_records(guard_log.read_text()) if guard_log.exists() else []
    return written, guarded, result


def test_default_build_root_is_the_container_tmp_scratch_and_is_removed(tmp_path):
    written, guarded, _ = _recorded_dirs(tmp_path, None)
    for path in written:
        assert str(path).startswith("/tmp/ds41rt-release-build."), path
    roots = {part for path in written for part in path.parts if part.startswith("ds41rt-release-build.")}
    assert len(roots) == 1, f"one build root per invocation, saw {roots}"
    survivor = Path("/tmp") / roots.pop()
    assert not survivor.exists(), f"{survivor} survived the build"
    # Unset must keep guarding /tmp, exactly as before the hook existed, once only.
    tmp_probes = [arguments for arguments in guarded if "/tmp" in arguments]
    assert len(tmp_probes) == 1, f"/tmp must be probed exactly once: {guarded}"


def test_relocated_root_covers_every_write_and_replaces_the_tmp_probe(tmp_path):
    build_root = SCRATCH / "task-build-root"
    written, guarded, _ = _recorded_dirs(tmp_path, build_root)
    for path in written:
        assert path.is_relative_to(build_root), path
    assert all("/tmp/ds41rt-release-build." not in str(path) for path in written), written
    probes = [arguments for arguments in guarded if str(build_root) in arguments]
    assert probes, f"the relocated parent must be what the guard probes: {guarded}"
    assert not any("/tmp" in arguments for arguments in guarded), (
        f"a relocated build must not be judged on a directory it never writes: {guarded}"
    )
    assert build_root.is_dir(), "the caller-owned parent outlives the per-build root"
    assert list(build_root.iterdir()) == [], f"per-build scratch leaked: {list(build_root.iterdir())}"


def test_a_requested_root_the_container_cannot_write_reports_the_cause(tmp_path):
    """The classic mistake is a bind mount that was never made."""
    missing = SCRATCH / "never-mounted"
    src = HARNESS._fake_source(tmp_path)
    shims = tmp_path / "shims"
    shims.mkdir()
    log = tmp_path / "written-dirs.log"
    for name, body in _recording_shims(log, tmp_path / "guard.jsonl").items():
        HARNESS._write_stub(shims / name, body)
    env = dict(os.environ)
    env["PATH"] = f"{shims}:{env['PATH']}"
    env["DS41RT_RELEASE_BUILD_ROOT"] = str(missing)
    result = subprocess.run(
        ["bash", str(HARNESS.SCRIPT), str(src), "coordinator", "120", str(tmp_path / "output")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode != 0
    assert "not writable inside this container" in result.stderr, result.stderr
    assert "bind-mounted at the identical path" in result.stderr, result.stderr
    assert not log.exists(), "the build must stop before it writes anywhere"
