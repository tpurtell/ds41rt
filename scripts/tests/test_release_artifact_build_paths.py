#!/usr/bin/env python3
"""CPU-only regressions for the release artifact build path handling.

The release artifact compiler runs inside a container that mounts the source
tree read-only (`build.sh`: `-v "$repo_root:/source:ro"`). Two failures are
guarded here; neither needs a GPU, CMake or Cargo:

1. `assert-build-filesystem.py` fails closed on read-only paths, so probing the
   read-only SOURCE_DIR aborted every release build before it started:
   `unsafe build filesystem: build path /source is on a read-only filesystem`.
2. The daemon was built into one cargo target directory and installed from a
   different one, so relocating the target broke the install.

Mount options cannot be faked with chmod, so the read-only cases drive the real
guard with a stubbed `findmnt` (the same technique as
`test_build_filesystem.py`). The last test is behavioural: a stub `cargo` writes
only into the selected target directory and the daemon must still reach the
output directory.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build-release-artifacts.sh"
GUARD = REPO / "scripts" / "assert-build-filesystem.py"


def script_text() -> str:
    return SCRIPT.read_text()


def guard_arguments() -> str:
    """The `assert-build-filesystem.py` argument text, whitespace-normalised."""
    text = script_text()
    match = re.search(
        r'python3 "\$\(dirname "\$0"\)/assert-build-filesystem\.py"'
        r"(?P<args>(?:[^\n]*\\\n)*[^\n]*)",
        text,
    )
    assert match, "guard invocation not found in build-release-artifacts.sh"
    return " ".join(match.group("args").replace("\\\n", " ").split())


def load_guard():
    spec = importlib.util.spec_from_file_location("release_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def findmnt_stub(read_only: set[str]):
    """Report `ro` for the given targets and `rw` for everything else."""

    def run(command, **_kwargs):
        target = command[command.index("--target") + 1]
        options = "ro,relatime" if str(target) in read_only else "rw,relatime"
        payload = json.dumps(
            {"filesystems": [{"fstype": "ext4", "options": options}]}
        )
        return subprocess.CompletedProcess(command, 0, payload, "")

    return run


# --------------------------------------------------------------------------
# Static wiring: the guard must probe the written paths, never the input.
# --------------------------------------------------------------------------


def test_guard_does_not_probe_the_read_only_source():
    assert "$source_dir" not in guard_arguments(), (
        "the guard must not probe SOURCE_DIR: the release container mounts it "
        "read-only and the guard fails closed on read-only paths"
    )


def test_guard_still_probes_the_writable_output_and_target():
    arguments = guard_arguments()
    assert "$output_dir" in arguments
    assert "CARGO_TARGET_DIR" in arguments


def test_cargo_build_and_daemon_install_share_one_target_dir():
    text = script_text()
    assert 'CARGO_TARGET_DIR="$cargo_target_dir" cargo build' in text, (
        "cargo must build into the selected target directory"
    )
    assert 'install -m 0755 "$cargo_target_dir/release/ds41rt"' in text, (
        "the daemon install must read from the same selected target directory"
    )
    assert "$build_root/source/rust/target/release/ds41rt" not in text, (
        "no install may hardcode the default target path"
    )


# --------------------------------------------------------------------------
# Guard behaviour under a read-only SOURCE mount (stubbed findmnt).
# --------------------------------------------------------------------------


def test_fixed_guard_arguments_accept_a_read_only_source(tmp_path, monkeypatch):
    guard = load_guard()
    source = tmp_path / "source"
    output = tmp_path / "output"
    target = output / "cargo-target"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(guard.subprocess, "run", findmnt_stub({str(source)}))

    # The pre-fix argument set aborted the real build.
    with pytest.raises(ValueError, match="read-only filesystem"):
        for value in (str(source), str(output), str(source / "rust" / "target")):
            guard.check_path(value)

    # The fixed argument set never probes the read-only input.
    for value in (str(output), str(target), str(tmp_path / "cargo-home"), str(tmp_path)):
        guard.check_path(value)


def test_guard_still_rejects_a_read_only_writable_target(tmp_path, monkeypatch):
    """Fail-closed is preserved for the paths that really are written."""
    guard = load_guard()
    target = tmp_path / "ro-target"
    target.mkdir()
    monkeypatch.setattr(guard.subprocess, "run", findmnt_stub({str(target)}))
    with pytest.raises(ValueError, match="read-only filesystem"):
        guard.check_path(str(target))


# --------------------------------------------------------------------------
# Behavioural: a relocated cargo target still yields an installed daemon.
# --------------------------------------------------------------------------


def _write_stub(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body))
    path.chmod(0o755)


def _fake_source(root: Path) -> Path:
    """Minimal tree satisfying the script's structural checks."""
    src = root / "source"
    for relative in (
        "rust/Cargo.toml",
        "native/CMakeLists.txt",
        "THIRD_PARTY_NOTICES.md",
        "third_party/sparkinfer/LICENSE",
        "third_party/sparkinfer.lock.json",
        "third_party/xgrammar/LICENSE",
        "third_party/xgrammar.lock.json",
    ):
        target = src / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub\n")
    (src / "scripts").mkdir(parents=True, exist_ok=True)
    (src / "scripts" / "assert-build-filesystem.py").write_bytes(GUARD.read_bytes())
    return src


def test_relocated_cargo_target_is_where_the_daemon_is_installed(tmp_path):
    """Behavioural proof that build and install follow one selected target."""
    src = _fake_source(tmp_path)
    output = tmp_path / "output"
    relocated = tmp_path / "relocated-target"
    shims = tmp_path / "shims"
    shims.mkdir()

    _write_stub(
        shims / "cargo",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        mkdir -p "$CARGO_TARGET_DIR/release"
        printf 'daemon\\n' >"$CARGO_TARGET_DIR/release/ds41rt"
        chmod +x "$CARGO_TARGET_DIR/release/ds41rt"
        """,
    )
    _write_stub(
        shims / "cmake",
        """
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
        mkdir -p "$build_dir/v41_experts" "$build_dir/v41_fp8"
        printf 'native\\n' >"$build_dir/libds41rt_native.so"
        printf '{}\\n' >"$build_dir/v41_experts/v41_experts.json"
        printf '{}\\n' >"$build_dir/v41_fp8/v41_fp8.json"
        exit 0
        """,
    )
    _write_stub(
        shims / "python3",
        f"""
        #!/usr/bin/env bash
        set -euo pipefail
        for arg in "$@"; do
          if [[ "$arg" == *assert-build-filesystem.py ]]; then
            exec {sys.executable} "$@"
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
        exit 0
        """,
    )

    env = dict(os.environ)
    env["PATH"] = f"{shims}:{env['PATH']}"
    env["CARGO_TARGET_DIR"] = str(relocated)
    env.pop("DS41RT_RELEASE_SPARK_TP_ROLES", None)

    result = subprocess.run(
        ["bash", str(SCRIPT), str(src), "coordinator", "120", str(output)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The stub only ever wrote into the relocated target...
    assert (relocated / "release" / "ds41rt").exists()
    # ...and the daemon still reached the output, because build and install
    # read the same selected target rather than a hardcoded default.
    assert (output / "ds41rt").read_text() == "daemon\n"
    assert not (output / "cargo-target").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
