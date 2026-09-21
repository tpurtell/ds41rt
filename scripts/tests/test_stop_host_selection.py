"""Mocked stop.sh lifecycle: cleanup host scope and failure aggregation.

CPU-only and non-disruptive. Every external command stop.sh touches (docker,
ssh, ss, ps) is replaced by a logging stub on PATH, so no container, process,
host or GPU is ever contacted. These tests pin the stop-side contract:

* cleanup covers every Spark host the configuration names, independent of
  SPARK_COUNT, so stale containers on the fifth/sixth hosts are not missed;
* launch/restart helpers keep cleaning only the active ranks;
* a missing or malformed configuration fails before any side effect;
* one failing host or phase never prevents the remaining hosts or phases.
"""
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOP = ROOT / "stop.sh"
COMMON = ROOT / "scripts" / "release-common.sh"
EXAMPLES = ROOT / "examples" / "configs"
SIX_HOSTS = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
FOUR_HOSTS = SIX_HOSTS[:4]

# Log the host operand and the whole argument vector. The full vector lets a
# test tell a release-container stop from a WIP-container stop.
SSH_STUB = """#!/usr/bin/env bash
args=("$@"); host=""; i=0
while (( i < ${#args[@]} )); do
  a="${args[$i]}"
  case "$a" in
    -o|-i|-p|-l|-F|-E|-c|-m|-Q|-S|-b|-w) i=$((i+2)); continue ;;
    -*) i=$((i+1)); continue ;;
    *) host="$a"; break ;;
  esac
done
printf 'ssh host=%s args=%s\\n' "$host" "$*" >> "${MOCK_LOG:?}"
if [[ -n "${MOCK_FAIL_HOST:-}" && "$host" == "$MOCK_FAIL_HOST" ]]; then
  exit 255
fi
exit 0
"""

DOCKER_STUB = """#!/usr/bin/env bash
[[ -n "${MOCK_DOCKER_LOG:-}" ]] && printf '%s\\n' "$*" >> "$MOCK_DOCKER_LOG"
case "${1:-}" in
  info) exit 0 ;;
  container) exit 1 ;;                 # no local container present
  inspect) printf 'false\\n'; exit 0 ;;
  *) exit 0 ;;
esac
"""

NOOP_STUB = """#!/usr/bin/env bash
exit 0
"""


def write_config(path, hosts, *, spark_count=None, topology=None, extra=""):
    lines = [
        "MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash",
        "MODEL_VARIANT=flash",
        "EXPERT_FORMAT=native",
    ]
    if spark_count is not None:
        lines.append(f"SPARK_COUNT={spark_count}")
    if topology is not None:
        tp, ep = topology
        lines.append(f"SPARK_TP={tp}")
        lines.append(f"SPARK_EP={ep}")
    for index, host in enumerate(hosts):
        lines.append(f"SPARK_{index}_HOST={host}")
        lines.append(f"SPARK_{index}_LANE_A=10.55.0.{index + 1}")
    if extra:
        lines.append(extra)
    path.write_text("\n".join(lines) + "\n")
    return path


class StopHarness(unittest.TestCase):
    """Run the real stop.sh against logging stubs on PATH."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, body in (("ssh", SSH_STUB), ("docker", DOCKER_STUB),
                           ("ss", NOOP_STUB), ("ps", NOOP_STUB)):
            stub = self.bin / name
            stub.write_text(body)
            stub.chmod(0o755)
        self.ssh_log = self.root / "ssh.log"
        self.docker_log = self.root / "docker.log"

    def run_stop(self, *args, fail_host=None):
        ssh_log = self.ssh_log
        ssh_log.write_text("")
        docker_log = self.docker_log
        docker_log.write_text("")
        env = dict(os.environ)
        env["PATH"] = str(self.bin) + os.pathsep + env["PATH"]
        env["MOCK_LOG"] = str(ssh_log)
        env["MOCK_DOCKER_LOG"] = str(docker_log)
        if fail_host:
            env["MOCK_FAIL_HOST"] = fail_host
        result = subprocess.run(
            ["bash", str(STOP), *args], cwd=ROOT, env=env,
            text=True, capture_output=True,
        )
        return result, self.ssh_lines(), docker_log.read_text()

    def ssh_lines(self):
        return [line for line in self.ssh_log.read_text().splitlines() if line]

    @staticmethod
    def hosts(lines):
        return [line.split("host=", 1)[1].split(" ", 1)[0] for line in lines]

    def assert_no_side_effects(self, docker_log):
        self.assertFalse(
            [line for line in docker_log.splitlines()
             if line.startswith(("stop ", "rm ", "rm -f"))],
            docker_log,
        )


class StopHostScopeTest(StopHarness):
    def test_all_named_hosts_stop_even_when_spark_count_is_smaller(self):
        """Six hosts named, SPARK_COUNT=4: all six are cleaned."""
        config = write_config(self.root / "six-hosts-count4.config",
                              SIX_HOSTS, spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_explicit_example_six_host_config_stops_all_six(self):
        """The reported --config example configuration must clean every rank."""
        result, lines, _ = self.run_stop(
            "--config", str(EXAMPLES / "tp3ep2-native.config"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_six_host_config_without_spark_count_stops_all_six(self):
        """A six-host file that omits SPARK_COUNT still cleans all six hosts."""
        config = write_config(self.root / "six-hosts-no-count.config", SIX_HOSTS)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_four_named_hosts_stay_four(self):
        """A genuinely four-host configuration does not invent extra ranks."""
        config = write_config(self.root / "four.config", FOUR_HOSTS, spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(FOUR_HOSTS))

    def test_wip_container_is_stopped_on_every_named_host(self):
        config = write_config(self.root / "six.config", SIX_HOSTS, spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        for host in SIX_HOSTS:
            self.assertTrue(
                any(f"host={host}" in line and "ds41rt-spark-expert-wip" in line
                    for line in lines),
                f"WIP container not stopped on {host}: {lines}",
            )

    def test_repeated_host_names_are_contacted_once_per_phase(self):
        config = write_config(
            self.root / "duplicate.config",
            ["ostrich", "dodo", "emu", "kiwi", "ostrich", "ostrich"],
            spark_count=4,
        )
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        counts = {host: self.hosts(lines).count(host) for host in set(self.hosts(lines))}
        self.assertEqual(set(counts), {"ostrich", "dodo", "emu", "kiwi"})
        # ostrich is named three times but must not be contacted more often
        # than any other single host.
        self.assertEqual(counts["ostrich"], counts["dodo"], counts)

    def test_blank_inactive_host_key_is_skipped(self):
        """A blank rank between named hosts must not become an SSH target."""
        config = self.root / "blank.config"
        config.write_text(
            "SPARK_COUNT=4\n"
            + "".join(
                f"SPARK_{index}_HOST={host}\nSPARK_{index}_LANE_A=10.55.0.{index + 1}\n"
                for index, host in enumerate(FOUR_HOSTS)
            )
            + "SPARK_4_HOST=\nSPARK_5_HOST=moa\nSPARK_5_LANE_A=10.55.0.6\n"
        )
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.hosts(lines)), set(FOUR_HOSTS + ["moa"]))
        self.assertNotIn("", self.hosts(lines))

    def test_zero_spark_config_stops_only_the_coordinator(self):
        config = self.root / "zero.config"
        config.write_text("SPARK_COUNT=0\nRTX_EXPERT_LAYERS=40\n")
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, [])


class StopConfigSafetyTest(StopHarness):
    def test_missing_config_fails_before_any_side_effect(self):
        result, lines, docker_log = self.run_stop(
            "--config", str(self.root / "does-not-exist.config"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("configuration file not found", result.stderr)
        self.assertEqual(lines, [])
        self.assert_no_side_effects(docker_log)

    def test_malformed_config_fails_before_any_side_effect(self):
        config = self.root / "malformed.config"
        config.write_text("SPARK_0_HOST=ostrich\nBOGUS_KEY=1\n")
        result, lines, docker_log = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown configuration key", result.stderr)
        self.assertEqual(lines, [])
        self.assert_no_side_effects(docker_log)

    def test_incomplete_launch_host_set_still_cleans_named_hosts(self):
        """Stop does not require the active rank count to be fully named."""
        config = self.root / "partial-hosts.config"
        config.write_text("SPARK_COUNT=4\nSPARK_0_HOST=ostrich\n")
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.hosts(lines)), {"ostrich"})

    def test_config_path_with_spaces_is_honored(self):
        directory = self.root / "dir with spaces"
        directory.mkdir()
        config = write_config(directory / "six hosts.config", SIX_HOSTS,
                              spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_config_without_value_is_rejected(self):
        result, lines, docker_log = self.run_stop("--config")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--config requires a configuration file", result.stderr)
        self.assertEqual(lines, [])
        self.assert_no_side_effects(docker_log)

    def test_unknown_argument_is_rejected(self):
        result, lines, docker_log = self.run_stop("--bogus")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown stop argument", result.stderr)
        self.assertEqual(lines, [])
        self.assert_no_side_effects(docker_log)


class StopFailureAggregationTest(StopHarness):
    def test_one_unreachable_host_does_not_skip_other_hosts_or_phases(self):
        config = write_config(self.root / "six.config", SIX_HOSTS, spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config), fail_host="rhea")
        self.assertNotEqual(result.returncode, 0)
        # Every configured host was still attempted.
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))
        # Every phase reached the failing host and the healthy hosts. The
        # release-container stop names the per-host container, so its presence
        # proves the release phase ran after the WIP phase failed.
        for host in SIX_HOSTS:
            self.assertTrue(
                any(f"host={host}" in line and
                    f"ds41rt-spark-expert-{host}-19441" in line for line in lines),
                f"release phase skipped for {host}: {lines}",
            )
        self.assertIn("could not be stopped", result.stderr)

    def test_every_configured_host_is_reported_on_failure(self):
        config = write_config(self.root / "six.config", SIX_HOSTS, spark_count=4)
        result, lines, _ = self.run_stop("--config", str(config), fail_host="moa")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("moa", result.stderr)


class StopConfigModeTest(StopHarness):
    """Stop mode drops launch usability but never syntax or value safety."""

    def load(self, config, mode=""):
        suffix = f" {mode}" if mode else ""
        return subprocess.run(
            ["bash", "-euc",
             f'source scripts/release-common.sh; release_load_config "$1"{suffix}; '
             'printf "loaded\\n"',
             "test", str(config)],
            cwd=ROOT, text=True, capture_output=True,
        )

    def raw_config(self, name, body):
        config = self.root / name
        config.write_text(body)
        return config

    def host_lines(self, hosts, *, lanes=True):
        return "".join(
            f"SPARK_{index}_HOST={host}\n"
            + (f"SPARK_{index}_LANE_A=10.55.0.{index + 1}\n" if lanes else "")
            for index, host in enumerate(hosts)
        )

    def test_default_mode_keeps_the_strict_launch_contract(self):
        config = self.raw_config(
            "six-no-topology.config",
            "SPARK_COUNT=6\n" + self.host_lines(SIX_HOSTS),
        )
        strict = self.load(config)
        self.assertEqual(strict.returncode, 2, strict.stdout)
        self.assertIn("SPARK_COUNT=6 requires explicit", strict.stderr)

    def test_six_hosts_without_topology_stop_all_six(self):
        """The parent's usability case: launch-invalid, cleanup-complete."""
        config = self.raw_config(
            "six-no-topology.config",
            "SPARK_COUNT=6\n" + self.host_lines(SIX_HOSTS),
        )
        relaxed = self.load(config, "stop")
        self.assertEqual(relaxed.returncode, 0, relaxed.stderr)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_unsupported_count_is_launch_invalid_but_stop_safe(self):
        config = self.raw_config(
            "count-five.config",
            "SPARK_COUNT=5\n" + self.host_lines(SIX_HOSTS),
        )
        self.assertEqual(self.load(config).returncode, 2)
        self.assertEqual(self.load(config, "stop").returncode, 0)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_missing_rails_are_launch_invalid_but_stop_safe(self):
        """Rails are launch-only; cleanup must not need LANE_A/B."""
        config = self.raw_config(
            "no-rails.config",
            "SPARK_COUNT=6\nSPARK_TP=3\nSPARK_EP=2\n" + self.host_lines(SIX_HOSTS, lanes=False),
        )
        strict = self.load(config)
        self.assertEqual(strict.returncode, 2, strict.stdout)
        self.assertIn("LANE_A must be an IPv4 address", strict.stderr)
        self.assertEqual(self.load(config, "stop").returncode, 0)
        result, lines, _ = self.run_stop("--config", str(config))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(set(self.hosts(lines))), sorted(SIX_HOSTS))

    def test_missing_hosts_for_the_count_still_clean_what_is_named(self):
        config = self.raw_config("partial.config", "SPARK_COUNT=4\nSPARK_0_HOST=ostrich\n")
        self.assertEqual(self.load(config).returncode, 2)
        self.assertEqual(self.load(config, "stop").returncode, 0)

    def test_stop_mode_rejects_malformed_syntax(self):
        for name, body, message in (
            ("unknown.config", "SPARK_0_HOST=ostrich\nBOGUS_KEY=1\n", "unknown configuration key"),
            ("no-equals.config", "SPARK_0_HOST\n", "invalid configuration line"),
            ("whitespace.config", "SPARK_0_HOST=two words\n", "unquoted whitespace"),
        ):
            with self.subTest(name=name):
                config = self.raw_config(name, body)
                result = self.load(config, "stop")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(message, result.stderr)

    def test_stop_mode_rejects_bad_value_domains(self):
        for name, body in (
            ("port.config", "SPARK_0_HOST=ostrich\nEXPERT_PORT=99999\n"),
            ("gpus.config", "SPARK_0_HOST=ostrich\nRTX_GPUS=3\n"),
            ("count-type.config", "SPARK_0_HOST=ostrich\nSPARK_COUNT=abc\n"),
            ("tp-type.config", "SPARK_0_HOST=ostrich\nSPARK_TP=abc\n"),
        ):
            with self.subTest(name=name):
                config = self.raw_config(name, body)
                result = self.load(config, "stop")
                self.assertEqual(result.returncode, 2, result.stdout)

    def test_stop_mode_rejects_unsafe_host_tokens_before_any_effect(self):
        for token in ('"bad host"', "'$(id)'", "a;b", "-leading",
                      "user@host", "2001:db8::1"):
            with self.subTest(token=token):
                config = self.raw_config(
                    "unsafe.config",
                    f'SPARK_COUNT=4\nSPARK_0_HOST=ostrich\nSPARK_5_HOST={token}\n',
                )
                result = self.load(config, "stop")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("not a safe host token", result.stderr)
                outcome, lines, docker_log = self.run_stop("--config", str(config))
                self.assertEqual(outcome.returncode, 2, outcome.stderr)
                self.assertEqual(lines, [])
                self.assert_no_side_effects(docker_log)

    def test_stop_mode_accepts_trailing_separator_aliases(self):
        """Docker-name-valid aliases the launcher accepts must stay stoppable."""
        for token in ("host_", "host-", "host."):
            with self.subTest(token=token):
                config = self.raw_config(
                    "trailing.config",
                    "SPARK_COUNT=4\n"
                    f"SPARK_0_HOST={token}\n"
                    "SPARK_1_HOST=dodo\nSPARK_2_HOST=emu\nSPARK_3_HOST=kiwi\n"
                    "SPARK_0_LANE_A=10.55.0.1\nSPARK_1_LANE_A=10.55.0.2\n"
                    "SPARK_2_LANE_A=10.55.0.3\nSPARK_3_LANE_A=10.55.0.4\n",
                )
                # Parity: the existing launcher already accepts the alias ...
                self.assertEqual(self.load(config).returncode, 0)
                # ... so stopping must not reject it either.
                self.assertEqual(self.load(config, "stop").returncode, 0)
                outcome, lines, _ = self.run_stop("--config", str(config))
                self.assertEqual(outcome.returncode, 0, outcome.stderr)
                self.assertIn(token, self.hosts(lines))

    def test_unknown_load_mode_is_rejected(self):
        config = self.raw_config("plain.config", "SPARK_0_HOST=ostrich\n")
        result = self.load(config, "bogus")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("unknown release_load_config mode", result.stderr)


class StopHelperScopeTest(unittest.TestCase):
    """Direct helper coverage: the widened scope is opt-in, not global."""

    def run_common(self, setup, command):
        return subprocess.run(
            ["bash", "-euc", f'source scripts/release-common.sh\n{setup}\n{command}',
             "test"],
            cwd=ROOT, text=True, capture_output=True,
        )

    HOST_VARS = textwrap.dedent("""\
        SPARK_COUNT=4
        SPARK_0_HOST=ostrich
        SPARK_1_HOST=dodo
        SPARK_2_HOST=emu
        SPARK_3_HOST=kiwi
        SPARK_4_HOST=rhea
        SPARK_5_HOST=moa
        EXPERT_PORT=19441
        ADDR=127.0.0.1:18000
        """)

    def test_active_rank_helpers_ignore_inactive_hosts(self):
        result = self.run_common(
            self.HOST_VARS,
            'release_hosts_csv; release_lane_a_csv',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], "ostrich,dodo,emu,kiwi")

    def test_select_stop_hosts_includes_every_named_host_in_rank_order(self):
        result = self.run_common(
            self.HOST_VARS,
            'release_select_stop_hosts; release_stop_hosts | paste -sd, -',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ostrich,dodo,emu,kiwi,rhea,moa\n")

    def test_out_of_range_host_key_fails_clearly(self):
        """A rank key the grammar cannot represent is reported, not skipped."""
        result = self.run_common(
            self.HOST_VARS + "SPARK_6_HOST=extra\n",
            'release_select_stop_hosts; echo "reached"',
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("unsupported Spark host key SPARK_6_HOST", result.stderr)
        self.assertNotIn("reached", result.stdout)

    def test_non_canonical_rank_key_fails_clearly(self):
        result = self.run_common(
            self.HOST_VARS + "SPARK_00_HOST=extra\n",
            "release_select_stop_hosts",
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("unsupported Spark host key SPARK_00_HOST", result.stderr)

    def test_stop_hosts_fall_back_to_active_ranks_without_selection(self):
        result = self.run_common(
            self.HOST_VARS,
            'release_stop_hosts | paste -sd, -',
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ostrich,dodo,emu,kiwi\n")

    def test_zero_spark_configuration_has_empty_stop_scope(self):
        result = self.run_common(
            "SPARK_COUNT=0\n",
            "release_select_stop_hosts; release_stop_hosts",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_stop_services_uses_active_ranks_for_launch_restart_callers(self):
        command = textwrap.dedent("""\
            release_stop_local_container() { :; }
            release_stop_host_api() { :; }
            release_stop_remote_containers() { echo "release:$1"; }
            release_stop_persistent_local_container() { :; }
            release_stop_persistent_remote_container() { echo "persistent:$1"; }
            release_stop_services coord spark
            release_stop_wip_containers
            """)
        result = self.run_common(self.HOST_VARS, command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            set(result.stdout.splitlines()),
            {"release:ostrich", "release:dodo", "release:emu", "release:kiwi",
             "persistent:ostrich", "persistent:dodo", "persistent:emu",
             "persistent:kiwi"},
        )

    def test_stop_services_uses_selected_hosts_for_stop_path(self):
        command = textwrap.dedent("""\
            release_select_stop_hosts
            release_stop_local_container() { :; }
            release_stop_host_api() { :; }
            release_stop_remote_containers() { echo "release:$1"; }
            release_stop_persistent_local_container() { :; }
            release_stop_persistent_remote_container() { echo "persistent:$1"; }
            release_stop_services coord spark
            release_stop_wip_containers
            """)
        result = self.run_common(self.HOST_VARS, command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            set(result.stdout.splitlines()),
            {"release:ostrich", "release:dodo", "release:emu", "release:kiwi",
             "release:rhea", "release:moa",
             "persistent:ostrich", "persistent:dodo", "persistent:emu",
             "persistent:kiwi", "persistent:rhea", "persistent:moa"},
        )

    def test_stop_wip_services_reports_unreachable_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ssh = root / "ssh"
            ssh.write_text("#!/usr/bin/env bash\nexit 255\n")
            ssh.chmod(0o755)
            script = textwrap.dedent("""\
                source scripts/release-common.sh
                SPARK_COUNT=1
                SPARK_0_HOST=ostrich
                EXPERT_PORT=19441
                ADDR=127.0.0.1:18000
                release_stop_wip_coordinator() { :; }
                release_stop_wip_services
                """)
            result = subprocess.run(
                ["bash", "-euc", script, "test"], cwd=ROOT, text=True,
                capture_output=True,
                env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"]),
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("failed to stop WIP process", result.stderr)


if __name__ == "__main__":
    unittest.main()
