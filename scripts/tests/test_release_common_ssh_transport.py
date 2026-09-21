#!/usr/bin/env python3
"""CPU-only tests for the one SSH option set shared by every scripted Spark call.

`scripts/release-common.sh` declares the transport so `build.sh` and
`scripts/phase0-spark-tcp-bench.sh` cannot drift into resolving it two different
ways:

* unset or empty `DS41RT_RELEASE_SSH_CONFIG` -> stock OpenSSH resolution, because a
  host's `~/.ssh/config` legitimately carries the aliases that reach the Sparks
  (`ostrich` -> 10.55.0.1). Discarding the whole chain with `/dev/null` is a
  deliberate, visible opt-in and is documented as also dropping those aliases.
* a value that is not a canonical absolute path is refused outright. The accepted
  alphabet is a whitelist, because these strings reach remote shell command lines,
  Docker bind sources and rsync `host:path` specs - a colon in particular would let
  a "path" name another host.
* `BatchMode=yes` always, so an unattended build or readiness poll can never stall
  on a password or host-key prompt.

The library is declarations only: sourcing it changes nothing until a caller invokes
`release_configure_ssh_transport` or `release_ssh`, which is why a script that keeps
its own ssh calls is unaffected by sourcing it.

Stubs on PATH replace `ssh`; no SSH, Docker, GPU or host is contacted. The phase0
checks are the script's own startup behavior, which fails before any call.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "scripts" / "release-common.sh"
PHASE0 = REPO / "scripts" / "phase0-spark-tcp-bench.sh"
RUN = REPO / "run.sh"

DIE = 2  # release_die's status in the library
SOURCE = 'source scripts/release-common.sh\n'


def _sh(path: Path, name: str, log: Path, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(body)
    (path / name).chmod(0o755)


def _run(body: str, tmp: Path, env: dict[str, str] | None = None):
    """Run bash with the library sourced and a recording `ssh` stub on PATH."""
    log = tmp / "ssh.log"
    _sh(tmp / "bin", "ssh", log,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"for token in \"$@\"; do printf 'A\\t%s\\n' \"$token\" >>{log}; done\n"
        f"printf 'E\\n' >>{log}\n"
        'exit "${STUB_SSH_EXIT:-0}"\n')
    environment = {
        "PATH": f"{tmp / 'bin'}:/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp),
    }
    # A caller-provided value must win over anything inherited from the test runner.
    environment.update({k: v for k, v in (env or {}).items()})
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{SOURCE}{body}"],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        cwd=REPO,
        check=False,
    )
    invocations: list[list[str]] = []
    if log.exists():
        current: list[str] = []
        for line in log.read_text().splitlines():
            if line == "E":
                invocations.append(current)
                current = []
            elif line.startswith("A\t"):
                current.append(line[2:])
            else:
                raise AssertionError(f"unexpected stub line: {line!r}")
        assert not current, f"truncated stub log: {current}"
    return result, invocations


class TestSharedTransportSemantics:
    def test_stock_is_the_default_and_batch_mode_is_always_forced(self, tmp_path):
        result, invocations = _run("release_ssh seed true", tmp_path)
        assert result.returncode == 0, result.stderr
        assert invocations == [["-o", "BatchMode=yes", "seed", "true"]]

    def test_explicit_include_config_reaches_ssh_as_one_token(self, tmp_path):
        """The safe opt-out for a broken system include is a file that Includes the
        user's own config, not /dev/null, and its path must never be re-parsed."""
        config = tmp_path / ".ssh" / "ds41rt-release.config"
        config.parent.mkdir(parents=True)
        config.write_text("Include ~/.ssh/config\n")
        result, invocations = _run(
            "release_configure_ssh_transport\n"
            "printf '%s\\n' \"$release_rsh\"\nrelease_ssh seed true",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": str(config)},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [f"ssh -o BatchMode=yes -F {config}"]
        assert invocations == [
            ["-o", "BatchMode=yes", "-F", str(config), "seed", "true"]
        ], "the path must be its own argv element, not text inside another argument"

    @pytest.mark.parametrize(
        "value",
        [
            "relative/config",
            "~/config",
            "/tmp/ssh config",
            "/tmp/config; rm -rf /",
            "-oProxyCommand=touch /tmp/ssh-transport-pwned",
            "/tmp/$(touch /tmp/ssh-transport-pwned)",
            "/tmp/`touch /tmp/ssh-transport-pwned`",
            "/tmp/config/",
            "/tmp/./config",
            "/tmp/a/../b/config",
            "/",
            "//tmp/config",
            "/tmp/host:/etc/ssh/ssh_config",
            "/tmp/a:b",
        ],
    )
    def test_unsafe_values_are_refused_without_contacting_a_host(self, tmp_path, value):
        result, invocations = _run(
            "release_configure_ssh_transport\nrelease_ssh seed true",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": value},
        )
        assert result.returncode == DIE, result.stdout + result.stderr
        assert "canonical absolute path" in result.stderr
        assert invocations == [], "a refused value must not reach any ssh call"
        assert not Path("/tmp/ssh-transport-pwned").exists()

    def test_rsync_children_inherit_the_same_options(self, tmp_path):
        """rsync and rdmasync inherit the option set through their own standard
        variable, which is what lets a child script use the transport unmodified."""
        printf = "printf '%s\\n' \"$RSYNC_RSH\"\n"
        result, _ = _run("release_configure_ssh_transport\n" + printf, tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ssh -o BatchMode=yes"
        result, _ = _run(
            "release_configure_ssh_transport\n" + printf,
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/home/spark/.ssh/release.config"},
        )
        assert result.stdout.strip() == (
            "ssh -o BatchMode=yes -F /home/spark/.ssh/release.config"
        )

    def test_configuration_is_resolved_once_per_process(self, tmp_path):
        # The first resolution wins: a script that configures at start-up and later
        # calls release_ssh must not silently switch hosts mid-build.
        result, invocations = _run(
            "export DS41RT_RELEASE_SSH_CONFIG=/dev/null\n"
            "release_configure_ssh_transport\n"
            "export DS41RT_RELEASE_SSH_CONFIG=/home/spark/.ssh/other.config\n"
            "release_ssh seed true",
            tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert invocations == [
            ["-o", "BatchMode=yes", "-F", "/dev/null", "seed", "true"]
        ]

    def test_release_ssh_configures_lazily_when_a_caller_forgets(self, tmp_path):
        result, invocations = _run(
            "release_ssh seed true", tmp_path, env={"DS41RT_RELEASE_SSH_CONFIG": "/dev/null"}
        )
        assert result.returncode == 0, result.stderr
        assert invocations[0][:4] == ["-o", "BatchMode=yes", "-F", "/dev/null"]

    def test_a_set_u_caller_can_read_the_names_before_configuring(self, tmp_path):
        """A consumer builds its own docker and rsync arguments from these names, so
        reading one before configuring must yield an empty value, not an abort."""
        result, _ = _run(
            "printf '%s|%s|%s\\n' \"$release_rsh\" \"$release_ssh_config\" "
            "\"${#release_ssh_opts[@]}\"",
            tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "||0"

    def test_sourcing_the_library_alone_changes_nothing(self, tmp_path):
        result, invocations = _run("true", tmp_path, env={"DS41RT_RELEASE_SSH_CONFIG": "/nope"})
        assert result.returncode == 0, result.stderr
        assert invocations == []


# --- run.sh: the serving lifecycle reaches the Sparks the same way ------------ #

# Each slice below is executed verbatim against a recording `ssh` stub, so the
# assertions are about the real statement text, not about a paraphrase of it.
# Each entry is (the first characters of the real statement, the text that closes
# it). The slice is taken verbatim out of run.sh and executed against a recording
# `ssh` stub, so these tests assert on shipped text rather than a paraphrase of it.
RUN_STATEMENTS = {
    "spark-preflight": (
        'spark_manifest="$(release_ssh',
        'REMOTE\n)"',
    ),
    "role-label-preflight": (
        'release_ssh -o ConnectTimeout=10 "$host" \\\n        "docker image inspect',
        '$SPARK_EXPERT_DOCKER_INFERENCE\'"\n',
    ),
    "container-conflict-check": (
        'release_ssh "${hosts[$i]}" "! docker inspect',
        'already exists; use --restart"\n',
    ),
    "exit-cleanup-loop": (
        'for i in "${!hosts[@]}"; do release_ssh',
        '|| true; done\n',
    ),
    "expert-launch": (
        'release_ssh "$host" bash -s -- "$SPARK_EXPERT_DOCKER_INFERENCE" "$remote" "$i"',
        '\nREMOTE\n',
    ),
    "readiness-poll": (
        "until release_ssh",
        "\n  done\n",
    ),
    "failure-log-fetch": (
        'release_ssh "$host" "docker logs --tail 100',
        "|| true;",
    ),
}

RUN_VARS = """
hosts=(seed0 seed1)
i=0
host=seed0
SPARK_EXPERT_DOCKER_INFERENCE=ds41rt-spark-expert:v10
engine_commit=engine-revision
sparkinfer_commit=sparkinfer-revision
snapshot_rel=hub/models--x--M/snapshots/rev
model_is_exl3=true
exl3_family_tag=k34
EXPERT_PORT=19441
spark_prefix=ds41rt-spark-expert
remote=ds41rt-spark-expert-seed0-19441
expert_capacity=4096
SPARK_DEVICE_BUDGET_BYTES=107374182400
fingerprint=fp
spark_first_layer=0
SPARK_COUNT=2
topology_explicit=explicit
spark_tp=2
spark_ep=2
"""

BATCH = ["-o", "BatchMode=yes"]
BATCH_TIMEOUT = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def run_text() -> str:
    return RUN.read_text(encoding="utf-8")


def _slice(name: str) -> str:
    """One real run.sh statement, verbatim."""
    start, end = RUN_STATEMENTS[name]
    text = run_text()
    first = text.index(start)
    stop = text.index(end, first) + len(end)
    body = text[first:stop]
    if name == "expert-launch":
        body += "wait\n"  # the real statement backgrounds the launch
    return body + "\n"


class TestRunShLifecycle:
    """Serving must not keep its own idea of how to reach a Spark.

    run.sh reads the checkpoint and role-label preflights, launches each expert,
    polls readiness and cleans up on EXIT, all over ssh. Those calls have to honor
    the same DS41RT_RELEASE_SSH_CONFIG that placed the image on the host, or a
    build-time workaround turns into a runtime failure that looks like an
    unreachable Spark. The one intentional behavior change is the failure-path log
    fetch, which carried no options at all and now also runs in BatchMode.
    """

    def _run(self, tmp_path, body, env=None):
        return _run(RUN_VARS + body, tmp_path, env)

    def test_no_direct_ssh_call_survives_in_the_lifecycle(self):
        # Only the code is scanned: the usage heredoc talks *about* ssh options for
        # the operator, which is prose rather than a call site.
        body = run_text().split("EOF\n}\n", 1)[1]
        offenders = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(body.splitlines(), start=1)
            if re.search(r"(?<![\w-])ssh\s", line)
            and not line.lstrip().startswith("#")
            and "release_need" not in line  # the tool-presence list names programs
        ]
        assert offenders == [], f"route these through release_ssh: {offenders}"
        assert 'for tool in docker ssh curl jq ss nvidia-smi' in run_text()
        assert run_text().count("release_ssh ") >= 8

    def test_transport_is_resolved_after_validation_and_before_any_host(self):
        text = run_text()
        configure = text.index("release_configure_ssh_transport")
        assert configure > text.index("release_validate_compact_spark"), (
            "resolving the transport before configuration validation would report a"
            " transport error in place of a bad config value"
        )
        assert configure < text.index("release_ssh "), (
            "the option set must exist before the first remote call"
        )
        assert text.count("release_configure_ssh_transport") == 1
        preamble = text[max(0, configure - 700):configure]
        assert "DS41RT_RELEASE_SSH_CONFIG" in preamble
        assert "EXIT cleanup" in preamble

    @pytest.mark.parametrize(
        "name,expected_options",
        [
            ("spark-preflight", BATCH_TIMEOUT),
            ("role-label-preflight", BATCH_TIMEOUT),
            ("container-conflict-check", BATCH),
            ("exit-cleanup-loop", BATCH),
            ("expert-launch", BATCH),
            ("readiness-poll", BATCH),
            ("failure-log-fetch", BATCH),
        ],
    )
    def test_statement_options_are_the_shared_set_plus_what_it_already_passed(
        self, tmp_path, name, expected_options
    ):
        result, invocations = self._run(tmp_path, _slice(name))
        assert result.returncode == 0, f"{_slice(name)[:70]}...\n{result.stderr}"
        assert invocations, "the statement made no remote call at all"
        first = invocations[0]
        assert first[: len(expected_options)] == expected_options, first
        assert first[len(expected_options)] == "seed0", (
            "the host must stay the first argument after the options: " + " ".join(first)
        )

    @pytest.mark.parametrize(
        "name",
        ["spark-preflight", "expert-launch", "exit-cleanup-loop", "failure-log-fetch"],
    )
    def test_statement_inherits_an_explicit_config_as_its_own_token(self, tmp_path, name):
        config = "/home/spark/.ssh/release.config"
        result, invocations = self._run(
            tmp_path, _slice(name), env={"DS41RT_RELEASE_SSH_CONFIG": config}
        )
        assert result.returncode == 0, result.stderr
        call = invocations[0]
        assert call[call.index("-F") + 1] == config, call
        # The config is an ssh option; it must never reach the remote command line,
        # where OpenSSH's flattened string would hand it to a second shell.
        after_host = call[call.index("seed0") + 1:]
        assert not [token for token in after_host if "release.config" in token], after_host

    def test_usage_documents_the_shared_transport(self):
        # An operator reading `./run.sh --help` must learn the same thing the build
        # documents, including the alias warning that makes /dev/null the wrong
        # answer for most hosts.
        usage = run_text().split("usage() {", 1)[1].split("\nEOF", 1)[0]
        assert "DS41RT_RELEASE_SSH_CONFIG" in usage
        assert "BatchMode" in usage
        assert "Include ~/.ssh/config" in usage

    def test_cleanup_loop_reaches_every_configured_host(self, tmp_path):
        result, invocations = self._run(tmp_path, _slice("exit-cleanup-loop"))
        assert result.returncode == 0, result.stderr
        assert [call[2] for call in invocations] == ["seed0", "seed1"], invocations
        for call in invocations:
            assert call[:3] == [*BATCH, call[2]], call

    def test_preflight_positional_vector_arrives_intact(self, tmp_path):
        result, invocations = self._run(tmp_path, _slice("spark-preflight"))
        assert result.returncode == 0, result.stderr
        call = invocations[0]
        assert call[call.index("--") + 1:] == [
            "ds41rt-spark-expert:v10",
            "engine-revision",
            "sparkinfer-revision",
            "hub/models--x--M/snapshots/rev",
            "seed0",
            "true",
            "k34",
        ], call

    def test_launch_passes_its_contract_positionally(self, tmp_path):
        """The worker argument contract is positional: an option set change must
        not shift a single one of the sixteen values the remote script decodes."""
        result, invocations = self._run(tmp_path, _slice("expert-launch"))
        assert result.returncode == 0, result.stderr
        call = invocations[0]
        payload = call[call.index("--") + 1:]
        assert payload[:5] == [
            "ds41rt-spark-expert:v10", "ds41rt-spark-expert-seed0-19441", "0",
            "4096", "107374182400",
        ], payload
        assert len(payload) == 16, f"the launch contract has 16 positionals: {payload}"

    def test_empty_optionals_do_not_shift_the_launch_vector(self, tmp_path):
        result, invocations = self._run(
            tmp_path,
            _slice("expert-launch"),
            env={"DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP": "/dev/infiniband"},
        )
        assert result.returncode == 0, result.stderr
        payload = invocations[0]
        payload = payload[payload.index("--") + 1:]
        assert payload[-3:] == ["/dev/infiniband", "", ""], payload
        assert len(payload) == 16, payload



# --- push-containers.sh: the publish step reaches the same host ---------------- #

PUSH = REPO / "push-containers.sh"

PUSH_STATEMENTS = {
    "spark-health-probe": (
        'release_ssh -o ConnectTimeout=10 "$spark_host" bash -s --',
        "\nREMOTE\n",
    ),
    "revision-label-fetch": (
        'spark_revision="$(',
        '\n)"',
    ),
    "expert-image-push": (
        'release_ssh "$spark_host" docker push "$spark_repository:$tag"',
        "\n",
    ),
}

PUSH_VARS = """
spark_host=seed
SPARK_EXPERT_DOCKER_INFERENCE=ds41rt-spark-expert:v10
COORDINATOR_DOCKER_INFERENCE=ds41rt-coordinator:v10
spark_repository=ghcr.io/tpurtell/ds41rt-spark-expert
coordinator_repository=ghcr.io/tpurtell/ds41rt-coordinator
tag=v10
"""


def push_text() -> str:
    return PUSH.read_text(encoding="utf-8")


def _push_slice(name: str) -> str:
    start, end = PUSH_STATEMENTS[name]
    text = push_text()
    first = text.index(start)
    return text[first:text.index(end, first) + len(end)] + "\n"


class TestPushContainersConsumer:
    """Publishing the Spark image is the last step that touches the Spark.

    The image was placed on SPARK_0_HOST by build.sh over the shared transport, and
    run.sh serves it the same way. If the publish step kept its own literal `ssh`,
    an operator's DS41RT_RELEASE_SSH_CONFIG would be honored while building and
    serving and then fail only at the moment of release.
    """

    def _run(self, tmp_path, body, env=None):
        return _run(PUSH_VARS + body, tmp_path, env)

    def test_no_direct_ssh_call_survives_in_the_publisher(self):
        offenders = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(push_text().splitlines(), start=1)
            if re.search(r"(?<![\w-])ssh\s+(?:-[A-Za-z]|[\"$])", line)
        ]
        assert offenders == [], f"route these through release_ssh: {offenders}"
        assert push_text().count("release_ssh ") == 8

    def test_delegates_the_transport_instead_of_copying_it(self):
        text = push_text()
        for declaration in (
            "release_canonical_path=",
            "release_validate_path_setting() {",
            "release_configure_ssh_transport() {",
            "release_ssh() {",
        ):
            assert declaration not in text, f"push-containers.sh duplicates {declaration!r}"

    def test_transport_is_resolved_after_validation_and_before_any_host(self):
        text = push_text()
        configure = text.index("release_configure_ssh_transport\n")
        assert configure > text.index('release_load_config "$repo_root/ds41rt.config"')
        assert configure > text.index("release_need ssh")
        assert configure < text.index("docker info"), (
            "the daemon probe is the first thing a bad transport setting should not"
            " have to wait behind"
        )
        assert configure < text.index("release_ssh ")
        assert text.count("release_configure_ssh_transport") == 1

    @pytest.mark.parametrize(
        "name,expected_options",
        [
            ("spark-health-probe", BATCH_TIMEOUT),
            ("revision-label-fetch", BATCH),
            ("expert-image-push", BATCH),
        ],
    )
    def test_statement_options_are_the_shared_set_plus_what_it_already_passed(
        self, tmp_path, name, expected_options
    ):
        result, invocations = self._run(tmp_path, _push_slice(name))
        assert result.returncode == 0, f"{_push_slice(name)[:60]}...\n{result.stderr}"
        assert invocations, "the statement made no remote call at all"
        call = invocations[0]
        assert call[: len(expected_options)] == expected_options, call
        assert call[len(expected_options)] == "seed", call

    @pytest.mark.parametrize("name", ["spark-health-probe", "revision-label-fetch"])
    def test_statement_inherits_an_explicit_config_as_its_own_token(self, tmp_path, name):
        config = "/home/spark/.ssh/release.config"
        result, invocations = self._run(
            tmp_path, _push_slice(name), env={"DS41RT_RELEASE_SSH_CONFIG": config}
        )
        assert result.returncode == 0, result.stderr
        call = invocations[0]
        assert call[call.index("-F") + 1] == config, call
        after_host = call[call.index("seed") + 1:]
        assert not [token for token in after_host if "release.config" in token], after_host

    def test_probe_still_delivers_its_positional_contract(self, tmp_path):
        result, invocations = self._run(tmp_path, _push_slice("spark-health-probe"))
        assert result.returncode == 0, result.stderr
        call = invocations[0]
        assert call[call.index("--") + 1:] == ["ds41rt-spark-expert:v10"], call

    def test_usage_documents_the_shared_transport(self):
        usage = push_text().split("usage() {", 1)[1].split("\nEOF", 1)[0]
        assert "DS41RT_RELEASE_SSH_CONFIG" in usage
        assert "BatchMode" in usage
        assert "Include ~/.ssh/config" in usage


class TestRuntimeStopConsumer:
    """The library's own stop helpers are scripted Spark calls too.

    Leaving them on a literal `ssh` would mean an operator's config choice was
    honored while building images and silently ignored while tearing containers
    down. With the setting unset these produce byte-identical argv to the literal
    they replaced, which is what this pins.
    """

    def _stop(self, tmp_path: Path, config: str | None):
        log = tmp_path / "stop.ssh"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "ssh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'opts=(); rest=()\n'
            'while [[ $# -gt 0 ]]; do\n'
            '  case "$1" in\n'
            '    -o|-F|-i|-l|-p) opts+=("$1" "$2"); shift 2 ;;\n'
            '    -*) opts+=("$1"); shift ;;\n'
            '    *) shift; rest+=("$@"); break ;;\n'
            '  esac\n'
            'done\n'
            '{ for token in "${opts[@]}"; do printf "O\\t%s\\n" "$token"; done\n'
            '  for token in "${rest[@]}"; do printf "C\\t%s\\n" "$token"; done; } '
            f'>>"{log}"\n'
            'exec bash -c "${rest[*]}"\n'
        )
        (bin_dir / "ssh").chmod(0o755)
        # The remote body stops at `docker container inspect`: absent container means
        # nothing to do, so no other stub is needed to exercise the transport.
        (bin_dir / "docker").write_text("#!/usr/bin/env bash\nexit 1\n")
        (bin_dir / "docker").chmod(0o755)
        environment = {
            "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/local/bin",
            "HOME": str(tmp_path),
        }
        if config is not None:
            environment["DS41RT_RELEASE_SSH_CONFIG"] = config
        result = subprocess.run(
            ["bash", "-c", SOURCE + "release_stop_persistent_remote_container seed ds41rt-x"],
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
            cwd=REPO,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        options, command = [], []
        for line in log.read_text().splitlines():
            kind, _, token = line.partition("\t")
            (options if kind == "O" else command).append(token)
        return options, command

    def test_stop_uses_the_same_options_as_the_build(self, tmp_path):
        options, command = self._stop(tmp_path, None)
        assert options == ["-o", "BatchMode=yes"]
        assert command[:3] == ["bash", "-s", "--"], command

    def test_stop_follows_an_explicit_config(self, tmp_path):
        options, command = self._stop(tmp_path, "/home/spark/.ssh/release.config")
        assert options == ["-o", "BatchMode=yes", "-F", "/home/spark/.ssh/release.config"]
        # The config path is an ssh option, never part of the remote command line.
        assert not [token for token in command if "release.config" in token], command

    def test_no_library_call_site_bypasses_the_helper(self):
        offenders = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(
                COMMON.read_text(encoding="utf-8").splitlines(), start=1
            )
            if re.search(r"(?<![\w-])ssh -o BatchMode=yes", line)
            and not line.lstrip().startswith("release_rsh=")
        ]
        assert offenders == [], f"route these through release_ssh: {offenders}"


class TestPhase0Consumer:
    """The image-distribution fallback reaches the Sparks the same way."""

    def text(self) -> str:
        return PHASE0.read_text(encoding="utf-8")

    def test_it_uses_the_shared_resolver_and_keeps_none_of_its_own(self):
        text = self.text()
        assert 'source "$repo_root/scripts/release-common.sh"' in text
        assert "release_configure_ssh_transport" in text
        assert "release_canonical_path=" not in text
        assert "release_ssh() {" not in text, "the helper must not be re-declared"
        assert "release_ssh_opts=" not in text

    def test_no_direct_ssh_call_escapes_the_helper(self):
        offenders = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(self.text().splitlines(), start=1)
            if re.search(r"(?<![\w-])ssh (?:-o |-G\b)", line)
        ]
        assert offenders == [], f"route these through release_ssh: {offenders}"
        assert self.text().count("release_ssh") > 20, "the fallback has many remote steps"

    def test_rsync_is_left_to_the_inherited_rsh_without_a_bare_F(self):
        for line in self.text().splitlines():
            if re.search(r"(?<![\w-])rsync ", line):
                assert "-F " not in line, f"bare -F is rsync's own --filter: {line}"

    def test_usage_documents_the_setting_and_the_alias_warning(self):
        usage = self.text().split("cat <<'EOF'", 1)[1].split("\nEOF", 1)[0]
        assert "DS41RT_RELEASE_SSH_CONFIG" in usage
        assert "Include ~/.ssh/config" in usage

    def test_a_mistyped_value_fails_before_any_host_is_contacted(self, tmp_path):
        result, invocations = _run(
            "bash scripts/phase0-spark-tcp-bench.sh",
            tmp_path,
            env={
                "DS41RT_RELEASE_SSH_CONFIG": "/tmp/bad config",
                "DS41RT_SPARK_HOSTS": "nonexistent-host",
            },
        )
        assert result.returncode == DIE, result.stdout + result.stderr
        assert "canonical absolute path" in result.stderr
        assert invocations == []

    def test_help_still_works_regardless_of_the_environment(self, tmp_path):
        result, invocations = _run(
            "bash scripts/phase0-spark-tcp-bench.sh --help",
            tmp_path,
            env={"DS41RT_RELEASE_SSH_CONFIG": "/tmp/bad config"},
        )
        assert result.returncode == 0, result.stderr
        assert "DS41RT_RELEASE_SSH_CONFIG" in result.stdout
        assert invocations == []
