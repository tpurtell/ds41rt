#!/usr/bin/env python3
"""CPU-only regressions for the release build's SSH transport and build root.

Two host-environment failures used to make a release build impossible to run
without editing the script or hand-wrapping every call:

1. A build host's system ssh_config can be unusable (a wrong-owner or
   world-writable drop-in under /etc/ssh/ssh_config.d/ makes OpenSSH abort with
   "Bad owner or permissions" before any host is contacted). `build.sh` now derives
   one option set for every remote step, so an operator cannot fix it by wrapping
   only their own interactive ssh. Stock resolution stays the default, because a
   build host's `~/.ssh/config` legitimately carries the aliases and identity files
   that reach the Sparks; discarding a broken *system* include with an explicit
   `/dev/null` is the opt-in.
2. The release containers wrote their whole build root into a generic
   in-container /tmp, which no environment variable could move: the artifact
   compiler hands mktemp an absolute template, so TMPDIR is ignored by design.
   `DS41RT_RELEASE_BUILD_ROOT` now names one unique per-task path, bound at the
   identical path inside the container, because the filesystem guard must resolve
   the device that is actually written.

Everything here is textual or executed in bash against stub binaries. No Docker,
SSH, GPU, Cargo or CMake is touched: the transport block is extracted between its
markers and run with stub `ssh`/`rsync`/`rdmasync`/`python3` on PATH, the same way
the universal-role tests execute the role canonicalizer.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "build.sh"
COMMON = REPO / "scripts" / "release-common.sh"
ARTIFACTS = REPO / "scripts" / "build-release-artifacts.sh"

TRANSPORT_BLOCK = ("# release-build-transport:start", "# release-build-transport:end")
# The block resolves paths relative to the checkout, so the harness supplies one.
# release-common.sh is a library of declarations: sourcing it is side-effect
# free, and it is where release_ssh and the path validator really live.
PREAMBLE = "source scripts/release-common.sh\nrepo_root=/source\n"
DIE = 2  # release_die's status in the shared library


def build_text() -> str:
    return BUILD.read_text(encoding="utf-8")


COMMON_TEXT = COMMON.read_text(encoding="utf-8")


def transport_block() -> str:
    text = build_text()
    start, end = TRANSPORT_BLOCK
    for marker in TRANSPORT_BLOCK:
        assert marker in text, f"build.sh lost its {marker!r} marker"
    return PREAMBLE + text.split(start, 1)[1].split(end, 1)[0]


def sync_definition() -> str:
    """`release_sync()` lives outside the block, so extraction needs it too."""
    text = build_text()
    assert "release_sync() {" in text, "build.sh lost release_sync()"
    body = text.split("release_sync() {", 1)[1].split("\n}", 1)[0]
    return f"release_sync() {{{body}\n}}"


def _shim(bin_dir: Path, name: str, log: Path) -> None:
    """Log every invocation of `name`: one `A\\t<token>` line per argv element.

    The `E` sentinel ends an invocation. One token per line keeps the shim free of
    nested quoting and still proves a value travelled as a single argv element
    rather than being word-split by an intermediate shell.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / name).write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"for token in \"$@\"; do printf 'A\\t%s\\n' \"$token\" >>{log}; done\n"
        f"printf 'E\\n' >>{log}\n"
        'exit "${SHIM_EXIT:-0}"\n'
    )
    (bin_dir / name).chmod(0o755)


def _parse_invocations(text: str) -> list[list[str]]:
    invocations: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line == "E":
            invocations.append(current)
            current = []
        elif line.startswith("A\t"):
            current.append(line[2:])
        else:
            raise AssertionError(f"unexpected shim log line: {line!r}")
    assert not current, f"truncated shim log: {current}"
    return invocations


def _run(body: str, tmp: Path, env: dict[str, str] | None = None, shims: tuple[str, ...] = ()):
    """Execute bash against the extracted wiring; return result plus logged calls."""
    log = tmp / "invocations.jsonl"
    bin_dir = tmp / "bin"
    for name in shims:
        _shim(bin_dir, name, log)
    environment = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin" if shims else "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp),
    }
    environment.update(env or {})
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{body}"],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        check=False,
        cwd=REPO,
    )
    invocations = _parse_invocations(log.read_text()) if log.exists() else []
    return result, invocations


class TestSshTransport:
    def test_default_keeps_stock_config_and_forces_batch_mode(self, tmp_path):
        result, invocations = _run(
            f"{transport_block()}\n"
            "printf '%s\\n' \"$release_rsh\"\nrelease_ssh host true",
            tmp_path,
            shims=("ssh",),
        )
        assert result.returncode == 0, result.stderr
        # A working ~/.ssh/config carrying the Spark aliases must keep working by
        # default; only the non-interactive guarantee is added.
        assert result.stdout.strip() == "ssh -o BatchMode=yes"
        assert invocations == [["-o", "BatchMode=yes", "host", "true"]]

    def test_explicit_config_reaches_ssh_and_rsh(self, tmp_path):
        result, invocations = _run(
            f"{transport_block()}\nprintf '%s\\n' \"$release_rsh\"\nrelease_ssh host true",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/home/build/.ssh/release_config"},
            shims=("ssh",),
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ssh -o BatchMode=yes -F /home/build/.ssh/release_config"
        assert invocations == [
            ["-o", "BatchMode=yes", "-F", "/home/build/.ssh/release_config", "host", "true"]
        ]

    def test_dev_null_discards_a_broken_system_include(self, tmp_path):
        result, _ = _run(
            f"{transport_block()}\nprintf '%s\\n' \"$release_rsh\"",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/dev/null"},
        )
        assert result.stdout.strip() == "ssh -o BatchMode=yes -F /dev/null"

    @pytest.mark.parametrize(
        "value",
        [
            "relative/config",
            "~/config",
            "/tmp/ssh config",
            "/tmp/config; rm -rf /",
            "-oProxyCommand=touch /tmp/pwned",
            "/tmp/$(touch /tmp/pwned)",
            "/tmp/`touch /tmp/pwned`",
            "/tmp/./config",
            "/tmp/../etc/config",
            "/tmp/config/",
            "/",
            "//tmp/config",
        ],
    )
    def test_unsafe_config_values_are_refused_without_side_effects(self, tmp_path, value):
        result, invocations = _run(
            transport_block(), tmp_path, env={"DS41RT_RELEASE_SSH_CONFIG": value}
        )
        assert result.returncode == DIE, result.stdout + result.stderr
        assert "canonical absolute path" in result.stderr
        assert invocations == []
        assert not Path("/tmp/pwned").exists()

    def test_ssh_receives_the_config_as_its_own_token(self, tmp_path):
        result, invocations = _run(
            f"{transport_block()}\n"
            "release_ssh seed echo ready\n"
            "release_ssh -o ConnectTimeout=10 seed true",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/dev/null"},
            shims=("ssh",),
        )
        assert result.returncode == 0, result.stderr
        assert invocations == [
            ["-o", "BatchMode=yes", "-F", "/dev/null", "seed", "echo", "ready"],
            ["-o", "BatchMode=yes", "-F", "/dev/null", "-o", "ConnectTimeout=10", "seed", "true"],
        ]

    @pytest.mark.parametrize("program", ["rsync", "rdmasync"])
    def test_sync_programs_get_one_rsh_value_and_never_a_bare_F(self, tmp_path, program):
        result, invocations = _run(
            f"{transport_block()}\n{sync_definition()}\n"
            f"release_sync_program={program}\nrelease_sync src dst",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/dev/null"},
            shims=(program,),
        )
        assert result.returncode == 0, result.stderr
        tokens = invocations[0]
        assert "--rsh=ssh -o BatchMode=yes -F /dev/null" in tokens, tokens
        # A bare -F would be rsync/rdmasync's own --filter=dir-merge option.
        assert "-F" not in tokens, tokens
        assert tokens[-2:] == ["src", "dst"], tokens

    def test_rsync_children_inherit_the_same_transport(self):
        text = build_text()
        assert 'export RSYNC_RSH="$release_rsh"' in COMMON_TEXT
        fallback = text.split("phase0-spark-tcp-bench.sh", 1)[0]
        fallback = fallback.rsplit("using serial netcat image distribution", 1)[1]
        assert 'RSYNC_RSH="$release_rsh"' in fallback
        assert 'DS41RT_RELEASE_SSH_CONFIG="$release_ssh_config"' in fallback


class TestBuildRootWiring:
    def test_unset_build_root_adds_no_mount_and_no_env(self, tmp_path):
        result, _ = _run(
            f"{transport_block()}\n"
            "printf '%s|%s\\n' \"${#release_build_root_args[@]}\" \"$release_build_root\"",
            tmp_path,
        )
        assert result.stdout.strip() == "0|"

    @pytest.mark.parametrize(
        "value",
        [
            "/",
            "/a/",
            "//a",
            "relative/root",
            "/tmp/build root",
            "/tmp/build;rm",
            "/tmp/$(id)",
            "/tmp/`id`",
            "/tmp/a/../b",
            "/tmp/./b",
            "/source",
            "/source/build-root",
        ],
    )
    def test_unsafe_or_colliding_build_root_values_are_refused(self, tmp_path, value):
        result, _ = _run(transport_block(), tmp_path, env={"DS41RT_RELEASE_BUILD_ROOT": value})
        assert result.returncode == DIE, result.stdout + result.stderr
        assert "DS41RT_RELEASE_BUILD_ROOT" in result.stderr

    def test_sibling_of_the_source_tree_is_allowed(self, tmp_path):
        root = "/scratch/ds41rt-build.task-1"
        result, _ = _run(
            f"{transport_block()}\nprintf '%s\\n' \"$release_build_root\"",
            tmp_path,
            env={"DS41RT_RELEASE_BUILD_ROOT": root},
        )
        assert result.stdout.strip() == root

    def test_mount_and_env_share_the_identical_path(self, tmp_path):
        root = "/scratch/ds41rt-build.task-1"
        result, _ = _run(
            f"{transport_block()}\n"
            'for arg in "${release_build_root_args[@]}"; do printf \'%s\\n\' "$arg"; done',
            tmp_path,
            env={"DS41RT_RELEASE_BUILD_ROOT": root},
        )
        assert result.stdout.splitlines() == [
            "-v", f"{root}:{root}", "-e", f"DS41RT_RELEASE_BUILD_ROOT={root}",
        ]

    def test_local_leg_creates_the_root_then_filesystem_guards_it(self, tmp_path):
        root = tmp_path / "build-root"
        result, invocations = _run(
            f"{transport_block()}\nrelease_prepare_build_root '' /source",
            tmp_path,
            env={"DS41RT_RELEASE_BUILD_ROOT": str(root)},
            shims=("python3",),
        )
        assert result.returncode == 0, result.stderr
        assert root.is_dir(), "the root must exist before a container binds it"
        assert invocations == [["/source/scripts/assert-build-filesystem.py", str(root)]]

    def test_guard_rejection_is_reported_for_the_build_root(self, tmp_path):
        result, invocations = _run(
            f"{transport_block()}\nrelease_prepare_build_root '' /source",
            tmp_path,
            env={"DS41RT_RELEASE_BUILD_ROOT": str(tmp_path / "br"), "SHIM_EXIT": "9"},
            shims=("python3",),
        )
        assert result.returncode == DIE
        assert "not a safe writable filesystem" in result.stderr
        assert len(invocations) == 1

    def test_remote_leg_guards_on_the_seed_host_through_release_ssh(self, tmp_path):
        root = "/home/spark/.cache/ds41rt-builds/task-1/build-root"
        result, invocations = _run(
            f"{transport_block()}\nrelease_prepare_build_root seed /home/spark/src",
            tmp_path,
            env={"DS41RT_RELEASE_BUILD_ROOT": root},
            shims=("ssh",),
        )
        assert result.returncode == 0, result.stderr
        assert invocations == [
            [
                "-o", "BatchMode=yes", "seed",
                f"mkdir -p {root} && python3 /home/spark/src/scripts/assert-build-filesystem.py {root}",
            ]
        ]


class TestPipelineWiring:
    """The helpers only help if every real call site actually uses them."""

    def test_build_sh_delegates_the_transport_instead_of_copying_it(self):
        text = build_text()
        # Two copies of one validator drift; the release build must use the shared
        # definitions, so nothing here may re-declare them.
        for declaration in (
            "release_canonical_path=",
            "release_validate_path_setting() {",
            "release_path_within() {",
            "release_path_has_no_dot_segment() {",
            "release_configure_ssh_transport() {",
            "release_ssh() {",
        ):
            assert declaration not in text, f"build.sh duplicates {declaration!r}"
        assert "release_configure_ssh_transport" in transport_block()
        helper = COMMON_TEXT.split("release_ssh() {", 1)[1].split("\n}", 1)[0]
        assert "release_configure_ssh_transport" in helper
        assert 'ssh ${release_ssh_opts[@]+"${release_ssh_opts[@]}"}' in helper, (
            "the expansion stays guarded: an empty ${array[@]} aborts under set -u"
            " on bash 4.2, and this helper is reached from EXIT traps"
        )

    def test_only_the_shared_helper_spells_out_batch_mode(self):
        for label, text in (("build.sh", build_text()), ("release-common.sh", COMMON_TEXT)):
            offenders = [
                line.strip()
                for line in text.splitlines()
                if re.search(r"(?<![\w-])ssh -o BatchMode=yes", line)
                and not line.lstrip().startswith("release_rsh=")
            ]
            assert offenders == [], f"{label}: {offenders}"
        assert "release_ssh_opts=(-o BatchMode=yes)" in COMMON_TEXT

    def test_the_remote_staging_path_is_validated_and_quoted(self):
        text = build_text()
        assert 'release_validate_path_setting DS41RT_RELEASE_REMOTE_BUILD_DIR "$remote_dir"' in text
        assert "printf -v remote_dir_quoted '%q'" in text
        # Every remote use must be quoted, never wrapped in shell string literals.
        assert "\"mkdir -p '$remote_dir'\"" not in text

    def test_the_build_root_stays_disjoint_from_the_staging_tree(self):
        text = build_text()
        assert 'release_path_within "$release_build_root" "$remote_dir"' in text
        assert 'release_path_within "$remote_dir" "$release_build_root"' in text
        assert 'release_path_within "$release_build_root" "$repo_root"' in text

    @pytest.mark.parametrize("value", ["/tmp/host:/abs/path", "/tmp/a:b"])
    def test_a_colon_is_refused_so_a_remote_spec_cannot_be_forged(self, tmp_path, value):
        # rsync and rdmasync split `host:path` on the first colon, so a colon in one
        # of these settings could otherwise redirect the transfer to another host.
        result, _ = _run(transport_block(), tmp_path, env={"DS41RT_RELEASE_BUILD_ROOT": value})
        assert result.returncode == DIE
        assert "DS41RT_RELEASE_BUILD_ROOT" in result.stderr

    def test_both_container_legs_bind_the_build_root(self):
        text = build_text()
        legs = re.findall(
            r"docker run --rm(?:[^\n]*\\\n)*[^\n]*build-release-artifacts\.sh[^\n]*", text
        )
        assert len(legs) == 2, f"expected a coordinator and an expert leg, found {len(legs)}"
        for leg in legs:
            assert "release_build_root_args[@]" in leg, leg
        assert 'mkdir -p "$release_build_root"' in text
        assert 'release_prepare_build_root "" "$repo_root"' in text
        assert 'release_prepare_build_root "$seed_host" "$remote_dir"' in text

    def test_the_remote_leg_region_stays_argument_transport_only(self):
        # scripts/tests/test_native_release_launcher.py extracts everything between
        # the leg's summary echo and its quoted heredoc and runs it against a stub,
        # so no helper other than the ssh transport may be called inside that region.
        text = build_text()
        leg = text.split(
            'echo "== building Spark development and inference images natively on $seed_host =="',
            1,
        )[1].split("<<'REMOTE'", 1)[0]
        assert "release_prepare_build_root" not in leg, leg
        assert "release_ssh" in leg

    def test_the_expert_leg_receives_the_root_as_a_sentinel_positional(self):
        text = build_text()
        assert '"${release_build_root:-__legacy__}"' in text
        assert 'release_build_root="${10-__legacy__}"' in text
        assert '[[ "$release_build_root" != "__legacy__" ]] || release_build_root=' in text
        remote_block = text.split("building Spark development and inference images natively", 1)[1]
        remote_block = remote_block.split("<<'REMOTE'", 1)[1].split("\nREMOTE", 1)[0]
        assert "release_build_root_args=(" in remote_block, "the remote leg must build its own array"

    def test_no_generic_tmp_bind_or_tmpfs_was_introduced(self):
        text = build_text()
        assert "--tmpfs" not in text
        assert not re.search(r'-v "[^"]*:/tmp"', text), "the container /tmp must not be shadowed"

    def test_capacity_is_checked_for_the_relocated_root(self):
        text = build_text()
        assert 'df -Pk "$release_build_root"' in text
        assert "df -Pk $release_build_root_quoted" in text

    def test_new_settings_are_documented_and_reported(self):
        usage = build_text().split("usage() {", 1)[1].split("\nEOF", 1)[0]
        assert "DS41RT_RELEASE_SSH_CONFIG" in usage
        assert "DS41RT_RELEASE_BUILD_ROOT" in usage
        assert "stock" in usage
        # /dev/null drops the user's own host aliases too, so the docs must name the
        # alternative that keeps them: verified with `ssh -F <file> -G`, where an
        # Include of ~/.ssh/config still resolves a Spark alias to its rail address
        # and bare /dev/null resolves to the literal name instead.
        assert "Include ~/.ssh/config" in usage
        assert "DS41RT_RELEASE_REMOTE_BUILD_DIR" in usage
        assert 'echo "  ssh config: ${release_ssh_config:-<stock>}"' in build_text()
        assert 'echo "  release build root: ${release_build_root:-<container /tmp>}"' in build_text()


class TestArtifactCompilerBuildRoot:
    def test_mktemp_follows_the_relocated_parent(self):
        text = ARTIFACTS.read_text(encoding="utf-8")
        assert 'build_root_parent="${DS41RT_RELEASE_BUILD_ROOT:-/tmp}"' in text
        assert 'mktemp -d "$build_root_parent/ds41rt-release-build.XXXXXX"' in text
        assert "mktemp -d /tmp/ds41rt-release-build.XXXXXX" not in text, (
            "an absolute /tmp template silently ignores TMPDIR and defeats relocation"
        )

    def test_each_written_filesystem_is_probed_once(self):
        text = ARTIFACTS.read_text(encoding="utf-8")
        assert (
            'if [[ -n "${CARGO_TARGET_DIR:-}" && "$CARGO_TARGET_DIR" != "$build_root_parent" ]]; then'
            in text
        )
        match = re.search(
            r'python3 "\$\(dirname "\$0"\)/assert-build-filesystem\.py"'
            r'(?P<args>(?:[^\n]*\\\n)*[^\n]*)',
            text,
        )
        assert match
        arguments = " ".join(match.group("args").replace("\\\n", " ").split())
        assert '"$output_dir"' in arguments
        assert '"${CARGO_TARGET_DIR:-$build_root_parent}"' in arguments
        assert '"${CARGO_HOME:-$HOME/.cargo}"' in arguments
        assert '${build_root_probe[@]+"${build_root_probe[@]}"}' in arguments
        assert '"$source_dir"' not in arguments, "the read-only input must stay unprobed"

    def test_unwritable_requested_root_reports_the_real_cause(self):
        text = ARTIFACTS.read_text(encoding="utf-8")
        assert 'test -w "$build_root_parent"' in text
        probe = text.split('test -w "$build_root_parent"', 1)[1].split("exit 2", 1)[0]
        assert "not writable inside this container" in probe
        assert "bind-mounted at the identical path" in probe
        assert text.index('test -w "$build_root_parent"') < text.index('build_root="$(mktemp')
