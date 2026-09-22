#!/usr/bin/env python3
"""CPU-only tests for the release helper scripts in `scripts/release/`.

These cover the release executor's own tooling, with no GPU, container, registry,
SSH or host access:

- the v11 release build config resolves to the intended pair and differs from the
  runtime default only in the release-pair lines (the tag is derived from that
  pair, so this is what selects `v11`);
- `verify-release-artifacts.sh` REQUIRES the four build hosts that the v10
  pipeline asserted (`runs/v10-release/build/10-verify.sh`, HOSTS=(ostrich dodo
  emu kiwi)): a missing required host or a divergent Spark image id must FAIL,
  never SKIP. Its offline `--fleet-file` replay makes that check testable here.
- `run-release-build.sh` refuses a missing config, a non-DS41RT source tree, a
  dirty tree without `--allow-dirty`, and an evidence directory that already
  holds a completed build, and it passes the canonicalized config path through
  to the build so a relative `--config` resolves against the caller's cwd.
- the helper scripts carry the repository's executable mode.

No `build.sh`, Docker daemon, SSH or network is used: `docker` is a recording
stub on PATH, and the build runner is exercised against a throwaway source tree
whose `build.sh` is a stub.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RELEASE_DIR = REPO / "scripts" / "release"
PROBE = RELEASE_DIR / "image-identity-probe.sh"
VERIFY = RELEASE_DIR / "verify-release-artifacts.sh"
BUILD_RUNNER = RELEASE_DIR / "run-release-build.sh"
EVIDENCE_SUMS = RELEASE_DIR / "write-evidence-sums.sh"
V10_CONFIG = REPO / "ds41rt.build-v10.config"
RUNTIME_CONFIG = REPO / "ds41rt.config"

# The published v10 Spark identity: what `docker image inspect` reported on the
# four workers (runs/v10-release/build/10-fleet-spark-images.txt).
SPARK_IMAGE = "ghcr.io/tpurtell/ds41rt-spark-expert"
SPARK_ID = "sha256:d1b668cd7e87079b5b57e381bdd45dfb4858646533611f18f22061f58ae18dec"
SOURCE_REVISION = "3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0"
SPARKINFER_REVISION = "4b0954148523b5a2e93813f963d483ffd350b9c9"
FOUR_HOSTS = ("ostrich", "dodo", "emu", "kiwi")

BUILD_STUB = textwrap.dedent(
    """#!/usr/bin/env bash
    echo "BUILD-STUB-CALLED"
    printf 'args=%s\\n' "$*"
    exit "${DS41RT_TEST_BUILD_RC:-0}"
    """
)

DOCKER_STUB = textwrap.dedent(
    """#!/usr/bin/env bash
    # Read-only stub: reports a fully labelled image derived from the requested
    # reference, so the verifier's required-host and identity checks are
    # exercised offline. Never touches a daemon.
    ref=""
    for token in "$@"; do
      case "$token" in ghcr.io/*) ref="$token" ;; esac
    done
    case "${1:-}" in
      info) exit 0 ;;
      image)
        [[ "${2:-}" == inspect ]] || exit 1
        shift 2
        fmt=""
        if [[ "${1:-}" == "-f" ]]; then fmt="$2"; fi
        [[ -n "$ref" ]] || exit 1
        tag="${ref##*:}"
        if [[ -z "$fmt" ]]; then exit 0; fi
        value=""
        case "$fmt" in
          *org.opencontainers.image.version*) value="$tag" ;;
          *org.opencontainers.image.revision*) value="3dd9a4ac2be9fd17ecf4cb8b7746efdc900d38f0" ;;
          *io.ds41rt.sparkinfer.revision*) value="4b0954148523b5a2e93813f963d483ffd350b9c9" ;;
          *io.ds41rt.role*) case "$ref" in *spark-expert*) value="expert" ;; *) value="coordinator" ;; esac ;;
          *io.ds41rt.cuda_arch*) case "$ref" in *spark-expert*) value="121" ;; *) value="120" ;; esac ;;
          *io.ds41rt.v41.spark_tp_roles*) case "$ref" in *spark-expert*) value="tp2;tp3;tp6" ;; *) value="" ;; esac ;;
          *io.ds41rt.source-manifest.sha256*) value="" ;;
          *Architecture*) value="amd64" ;;
          *) value="" ;;
        esac
        printf '%s\\n' "$value"
        exit 0
        ;;
      *) exit 1 ;;
    esac
    """
)


def host_record(host: str, *, image_id: str = SPARK_ID, status: str = "present") -> str:
    lines = [f"===== {host} =====", f"image={SPARK_IMAGE}:v10", f"status={status}"]
    if status == "present":
        lines += [
            f"id={image_id}",
            "architecture=arm64",
            "os=linux",
            f"label.org.opencontainers.image.revision={SOURCE_REVISION}",
            "label.org.opencontainers.image.version=v10",
            f"label.io.ds41rt.sparkinfer.revision={SPARKINFER_REVISION}",
            "label.io.ds41rt.cuda_arch=121",
            "label.io.ds41rt.role=expert",
            "label.io.ds41rt.v41.spark_tp_roles=tp2;tp3;tp6",
            "label.io.ds41rt.source-manifest.sha256=",
        ]
    return "\n".join(lines) + "\n"


def fleet_file(tmp_path: Path, hosts: dict[str, str]) -> Path:
    """hosts maps host name to rendered record text."""
    path = tmp_path / "fleet.txt"
    path.write_text("".join(hosts[h] for h in sorted(hosts)), encoding="utf-8")
    return path


def stub_bin(tmp_path: Path, docker: bool = True) -> Path:
    """PATH shims for the tools release-common.sh demands, plus docker.

    Real `git`, `python3` and coreutils stay in place: the build runner and the
    tests need them, and none of them reaches hardware.
    """
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    for name in ("docker", "ssh", "rsync", "sha256sum", "install", "curl"):
        if name == "docker" and not docker:
            continue
        target = directory / name
        target.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        target.chmod(0o755)
    if docker:
        (directory / "docker").write_text(DOCKER_STUB, encoding="utf-8")
        (directory / "docker").chmod(0o755)
    return directory


def run_verify(tmp_path: Path, *args: str, fleet: Path | None = None) -> subprocess.CompletedProcess:
    evidence = tmp_path / "evidence"
    command = [
        "bash", str(VERIFY),
        "--config", str(V10_CONFIG),
        "--evidence", str(evidence),
        "--require-hosts", ",".join(FOUR_HOSTS),
        "--scan-hosts", ",".join(FOUR_HOSTS),
    ]
    if fleet is not None:
        command += ["--fleet-file", str(fleet)]
    command += list(args)
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin(tmp_path)}:{env['PATH']}"
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)


def summary_value(evidence: Path, key: str) -> list[str]:
    text = (evidence / "10-verify-summary.txt").read_text(encoding="utf-8")
    return [line.split("=", 1)[1] for line in text.splitlines() if line.startswith(f"{key}=")]


########################################################################
# v11 release build config
########################################################################


def assignments(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.lstrip().startswith("#")]


def test_v11_build_config_exists_and_names_the_v11_pair() -> None:
    config = REPO / "ds41rt.build-v11.config"
    assert config.is_file()
    values = dict(line.split("=", 1) for line in assignments(config))
    assert values["COORDINATOR_DOCKER_INFERENCE"] == "ghcr.io/tpurtell/ds41rt-coordinator:v11"
    assert values["SPARK_EXPERT_DOCKER_INFERENCE"] == "ghcr.io/tpurtell/ds41rt-spark-expert:v11"


def test_v11_build_config_differs_from_runtime_only_in_the_pair() -> None:
    """build.sh derives the tag from the pair, so this difference is the release."""
    base = assignments(RUNTIME_CONFIG)
    target = assignments(REPO / "ds41rt.build-v11.config")
    assert len(base) == len(target)
    differing = [(a, b) for a, b in zip(base, target) if a != b]
    assert differing == [
        ("COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v10",
         "COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v11"),
        ("SPARK_EXPERT_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-spark-expert:v10",
         "SPARK_EXPERT_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-spark-expert:v11"),
    ]


@pytest.mark.parametrize("config,expected", [
    ("ds41rt.config", "v10"),
    ("ds41rt.build-v11.config", "v11"),
])
def test_build_dry_run_reports_the_config_tag(config: str, expected: str) -> None:
    result = subprocess.run(
        ["bash", "build.sh", "--config", config, "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert f"release tag: {expected}" in result.stdout


########################################################################
# verify-release-artifacts.sh
########################################################################


def test_verify_usage_errors_exit_two(tmp_path: Path) -> None:
    for args in ([], ["--config"], ["--config", str(V10_CONFIG)], ["--bogus"]):
        result = subprocess.run(
            ["bash", str(VERIFY), *args], capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 2, (args, result.stdout, result.stderr)


def test_help_lists_required_hosts_flag(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(VERIFY), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--require-hosts" in result.stdout
    assert "--fleet-file" not in result.stdout  # offline replay is test-only


def test_verify_default_required_hosts_are_the_first_four_configured(tmp_path: Path) -> None:
    """No --require-hosts: the config's first four SPARK_*_HOST values are required."""
    fleet = fleet_file(tmp_path, {h: host_record(h) for h in FOUR_HOSTS})
    evidence = tmp_path / "evidence"
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin(tmp_path)}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(VERIFY), "--config", str(V10_CONFIG), "--evidence", str(evidence),
         "--scan-hosts", ",".join(FOUR_HOSTS), "--fleet-file", str(fleet)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert summary_value(evidence, "required_hosts") == ["ostrich,dodo,emu,kiwi"]
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_all_four_required_hosts_present_and_identical_pass(tmp_path: Path) -> None:
    fleet = fleet_file(tmp_path, {h: host_record(h) for h in FOUR_HOSTS})
    result = run_verify(tmp_path, fleet=fleet)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "PASS spark.required_hosts_identical_ids" in result.stdout
    assert "SKIP spark.required_hosts_identical_ids" not in result.stdout


def test_a_missing_required_host_fails_and_is_not_skipped(tmp_path: Path) -> None:
    records = {h: host_record(h) for h in FOUR_HOSTS}
    records["emu"] = host_record("emu", status="unreachable")
    fleet = fleet_file(tmp_path, records)
    result = run_verify(tmp_path, fleet=fleet)
    assert result.returncode == 1
    assert "FAIL spark.required_host.emu host unreachable over SSH" in result.stdout
    assert "FAIL spark.required_hosts_identical_ids" in result.stdout


def test_a_host_absent_from_the_record_fails_explicitly(tmp_path: Path) -> None:
    fleet = fleet_file(tmp_path, {h: host_record(h) for h in ("ostrich", "dodo", "kiwi")})
    result = run_verify(tmp_path, fleet=fleet)
    assert result.returncode == 1
    assert "FAIL spark.required_host.emu host not present in the fleet record" in result.stdout


def test_a_divergent_spark_image_id_fails_the_required_check(tmp_path: Path) -> None:
    records = {h: host_record(h) for h in FOUR_HOSTS}
    records["kiwi"] = host_record("kiwi", image_id="sha256:" + "9" * 64)
    fleet = fleet_file(tmp_path, records)
    result = run_verify(tmp_path, fleet=fleet)
    assert result.returncode == 1
    assert "FAIL spark.required_hosts_identical_ids" in result.stdout
    assert "FAIL spark.fleet_identical_ids" in result.stdout


def test_skips_never_hide_a_required_check(tmp_path: Path) -> None:
    """The only SKIPs in a four-host run are the dist extras, which need --dist."""
    fleet = fleet_file(tmp_path, {h: host_record(h) for h in FOUR_HOSTS})
    result = run_verify(tmp_path, fleet=fleet)
    assert result.returncode == 0
    skipped = [line for line in result.stdout.splitlines() if line.startswith("SKIP")]
    assert skipped, "expected the optional dist checks to report SKIP"
    assert all("dist." in line for line in skipped), skipped


def test_missing_dist_directory_is_a_failure_when_requested(tmp_path: Path) -> None:
    fleet = fleet_file(tmp_path, {h: host_record(h) for h in FOUR_HOSTS})
    result = run_verify(tmp_path, "--dist", str(tmp_path / "no-such-dist"), fleet=fleet)
    assert result.returncode == 1
    assert "FAIL dist.present" in result.stdout


########################################################################
# run-release-build.sh
########################################################################


def make_source(tmp_path: Path, *, with_git: bool = False, dirty: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "build.sh").write_text(BUILD_STUB, encoding="utf-8")
    (source / "build.sh").chmod(0o755)
    if with_git:
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=source, check=True)
        (source / "tracked.txt").write_text("one\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt", "build.sh"], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=source, check=True)
        if dirty:
            (source / "tracked.txt").write_text("two\n", encoding="utf-8")
    return source


def run_build_runner(tmp_path: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin(tmp_path)}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(BUILD_RUNNER), *args],
        capture_output=True, text=True, env=env, cwd=str(cwd or tmp_path), timeout=180,
    )


def test_build_runner_usage_error_without_config(tmp_path: Path) -> None:
    result = run_build_runner(tmp_path, "--evidence", str(tmp_path / "ev"))
    assert result.returncode == 2
    assert "--config" in result.stderr or "Usage" in result.stderr


def test_build_runner_refuses_a_missing_config(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    result = run_build_runner(
        tmp_path, "--config", "nope.config", "--source", str(source),
        "--evidence", str(tmp_path / "ev"),
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_build_runner_refuses_a_non_ds41rt_source_tree(tmp_path: Path) -> None:
    not_a_tree = tmp_path / "empty"
    not_a_tree.mkdir()
    result = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(not_a_tree),
        "--evidence", str(tmp_path / "ev"),
    )
    assert result.returncode == 2
    assert "no build.sh" in result.stderr


def test_build_runner_resolves_a_relative_config_against_the_caller_cwd(tmp_path: Path) -> None:
    """The v10 flow ran from the clone; a relative config must still resolve."""
    source = make_source(tmp_path)
    evidence = tmp_path / "ev"
    result = run_build_runner(
        tmp_path, "--config", "ds41rt.build-v10.config", "--source", str(source),
        "--evidence", str(evidence), "--label", "t", cwd=REPO,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    log = (evidence / "t-build.log").read_text(encoding="utf-8")
    assert "BUILD-STUB-CALLED" in log
    # The canonical absolute path is what reaches the build.
    assert f"--config {V10_CONFIG}" in log
    assert (evidence / "t-build.rc").read_text(encoding="utf-8").strip() == "0"


def test_build_runner_refuses_a_dirty_tree_without_allow_dirty(tmp_path: Path) -> None:
    source = make_source(tmp_path, with_git=True, dirty=True)
    result = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(source),
        "--evidence", str(tmp_path / "ev"),
    )
    assert result.returncode == 2
    assert "dirty source tree" in result.stderr
    assert not (tmp_path / "ev" / "v10-build.log").exists()


def test_build_runner_records_a_clean_revision(tmp_path: Path) -> None:
    source = make_source(tmp_path, with_git=True)
    evidence = tmp_path / "ev"
    result = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(source),
        "--evidence", str(evidence), "--label", "clean",
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    log = (evidence / "clean-build.log").read_text(encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                          capture_output=True, text=True).stdout.strip()
    assert f"source_revision={head}" in log


def test_build_runner_refuses_a_completed_evidence_directory(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    evidence = tmp_path / "ev"
    first = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(source),
        "--evidence", str(evidence), "--label", "t",
    )
    assert first.returncode == 0
    second = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(source),
        "--evidence", str(evidence), "--label", "t",
    )
    assert second.returncode == 2
    assert "already holds a completed build" in second.stderr


def test_build_runner_does_not_run_the_build_on_a_refusal(tmp_path: Path) -> None:
    """A refusal must happen before any build work, not after it starts."""
    source = make_source(tmp_path)
    marker = tmp_path / "ran"
    (source / "build.sh").write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit 0\n", encoding="utf-8")
    (source / "build.sh").chmod(0o755)
    evidence = tmp_path / "ev"
    evidence.mkdir()
    (evidence / "t-build.rc").write_text("0\n", encoding="utf-8")
    result = run_build_runner(
        tmp_path, "--config", str(V10_CONFIG), "--source", str(source),
        "--evidence", str(evidence), "--label", "t",
    )
    assert result.returncode == 2
    assert not marker.exists()


########################################################################
# image-identity-probe.sh and write-evidence-sums.sh
########################################################################


def test_identity_probe_reports_a_missing_image_without_failing(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PATH"] = str(stub_bin(tmp_path, docker=False)) + ":" + env["PATH"]
    result = subprocess.run(
        ["bash", str(PROBE), "ghcr.io/example/absent:v1"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert result.returncode == 0
    assert "status=missing" in result.stdout


def test_evidence_sums_write_and_check(tmp_path: Path) -> None:
    evidence = tmp_path / "ev"
    evidence.mkdir()
    (evidence / "a.txt").write_text("a\n", encoding="utf-8")
    (evidence / "b.txt").write_text("b\n", encoding="utf-8")
    write = subprocess.run(
        ["bash", str(EVIDENCE_SUMS), "--evidence", str(evidence), "--quiet"],
        capture_output=True, text=True, timeout=60,
    )
    assert write.returncode == 0, write.stderr
    assert (evidence / "SHA256SUMS").is_file()
    check = subprocess.run(
        ["bash", str(EVIDENCE_SUMS), "--evidence", str(evidence), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    assert check.returncode == 0, check.stderr
    (evidence / "a.txt").write_text("tampered\n", encoding="utf-8")
    tampered = subprocess.run(
        ["bash", str(EVIDENCE_SUMS), "--evidence", str(evidence), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    assert tampered.returncode != 0


def test_helper_scripts_are_executable_and_text_artifacts_are_readable() -> None:
    for name in ("image-identity-probe.sh", "verify-release-artifacts.sh",
                 "run-release-build.sh", "write-evidence-sums.sh"):
        assert (RELEASE_DIR / name).stat().st_mode & 0o777 == 0o755, name
    for path in (REPO / "ds41rt.build-v11.config",
                 REPO / "docs" / "release-v11-notes.md",
                 REPO / "docs" / "release-v11-checklist.md"):
        assert path.stat().st_mode & 0o777 == 0o644, path
