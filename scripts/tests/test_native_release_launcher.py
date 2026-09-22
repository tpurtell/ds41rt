from __future__ import annotations

import os
import subprocess
import json
import shlex
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class NativeReleaseLauncherTest(unittest.TestCase):
    def test_explicit_dual_layer_boundary_covers_delegated_experts(self) -> None:
        for layout, layers, expected in [('2', 'auto', '20'), ('2', '17', '17'),
                                         ('2', '1', '1'), ('2', '40', '39'),
                                         ('1', '17', '0'), ('1', '0', '0')]:
            with self.subTest(layout=layout, layers=layers):
                result = subprocess.run(['bash', '-c',
                    'source scripts/release-common.sh; release_spark_first_layer "$1" "$2"',
                    'test', layout, layers], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
        result = subprocess.run(['bash', '-c',
            'source scripts/release-common.sh; release_spark_first_layer 2 0'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_spark_build_arguments_survive_ssh_empty_argument_elision(self) -> None:
        source = (ROOT / 'build.sh').read_text()
        block = source.split('echo "== building Spark development and inference images natively on $seed_host =="', 1)[1]
        invocation, remote = block.split("<<'REMOTE'", 1)
        preamble = remote.split('cd "$remote_dir"', 1)[0]
        # The optional source manifest and the optional V41 expert roles are
        # both carried behind non-empty sentinels. An empty earlier value must
        # not shift a later one, because OpenSSH joins argv into one command
        # string and does not preserve an empty argument.
        for digest, roles, expected in (
            ('', '', ['on', '', '']),
            ('a' * 64, '', ['on', 'a' * 64, '']),
            ('', 'tp2', ['on', '', 'tp2']),
            ('a' * 64, 'tp2;tp3', ['on', 'a' * 64, 'tp2;tp3']),
        ):
            with self.subTest(digest=digest, roles=roles):
                harness = f'''set -euo pipefail
# build.sh routes every remote step through release_ssh (host is its first
# argument); emulating it by re-running the joined command string locally is what
# reproduces OpenSSH's behaviour of collapsing an empty argument.
release_ssh() {{ shift 1; bash -c "$*"; }}
seed_host=fixture
remote_dir=/fixture
SPARK_EXPERT_DOCKER_DEV=dev
SPARK_EXPERT_DOCKER_INFERENCE=inference
engine_commit=engine
sparkinfer_commit=fork
release_version=v5
EXL3_PAIRED_TP4=on
source_manifest_sha256={shlex.quote(digest)}
spark_tp_roles={shlex.quote(roles)}
'''
                harness += invocation + "<<'REMOTE'" + preamble
                harness += 'printf "%s\\n" "$exl3_paired_tp4" "$source_manifest_sha256" "$spark_tp_roles"\nREMOTE\n'
                result = subprocess.run(['bash', '-c', harness],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected)

    def test_ssh_config_file_escaped_onto_child_command_line_is_never_reparsed(self) -> None:
        source = (ROOT / 'build.sh').read_text()
        block = source.split('echo "== building Spark development and inference images natively on $seed_host =="', 1)[1]
        invocation, remote = block.split("<<'REMOTE'", 1)
        preamble = remote.split('cd "$remote_dir"', 1)[0]
        # An ssh config path chosen for the build must reach ssh as its own argv
        # element and must never be re-spelled inside the command string the remote
        # shell executes: OpenSSH flattens that string, so a value that landed there
        # would be re-parsed by a second shell. The leg is a quoted heredoc, so the
        # only thing the remote shell receives is the positional argument vector.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'ds41rt-release.config'
            config.write_text("Include ~/.ssh/config\n")
            log = root / 'ssh.argv'
            stub_dir = root / 'bin'
            stub_dir.mkdir()
            (stub_dir / 'ssh').write_text(
                '#!/usr/bin/env bash\n'
                'set -euo pipefail\n'
                'opts=(); rest=()\n'
                'while [[ $# -gt 0 ]]; do\n'
                '  case "$1" in\n'
                '    -o|-F|-i|-l|-p) opts+=("$1" "$2"); shift 2 ;;\n'
                '    -*) opts+=("$1"); shift ;;\n'
                '    *) host="$1"; shift; rest+=("$@"); break ;;\n'
                '  esac\n'
                'done\n'
                '{ printf "H\\t%s\\n" "$host"\n'
                '  for token in ${opts[@]+"${opts[@]}"}; do printf "O\\t%s\\n" "$token"; done\n'
                '  for token in ${rest[@]+"${rest[@]}"}; do printf "C\\t%s\\n" "$token"; done; } '
                f'>>"{log}"\n'
                'exec bash -c "${rest[*]}"\n'
            )
            (stub_dir / 'ssh').chmod(0o755)
            harness = f'''set -euo pipefail
source scripts/release-common.sh
export DS41RT_RELEASE_SSH_CONFIG={shlex.quote(str(config))}
seed_host=fixture
remote_dir=/fixture
SPARK_EXPERT_DOCKER_DEV=dev
SPARK_EXPERT_DOCKER_INFERENCE=inference
engine_commit=engine
sparkinfer_commit=fork
release_version=v5
EXL3_PAIRED_TP4=on
source_manifest_sha256=
spark_tp_roles=
'''
            environment = dict(os.environ)
            environment['PATH'] = f"{stub_dir}:{environment['PATH']}"
            result = subprocess.run(
                ['bash', '-c', harness + invocation + "<<'REMOTE'" + preamble
                 + 'printf "%s\\n" "$exl3_paired_tp4" "$source_manifest_sha256" "$spark_tp_roles"\nREMOTE\n'],
                cwd=ROOT, text=True, capture_output=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = {'H': [], 'O': [], 'C': []}
            for line in log.read_text().splitlines():
                kind, _, token = line.partition('\t')
                records[kind].append(token)
            self.assertEqual(records['O'], ['-o', 'BatchMode=yes', '-F', str(config)])
            self.assertEqual(records['H'], ['fixture'])
            self.assertEqual(records['C'][:3], ['bash', '-s', '--'],
                             'the remote command is what OpenSSH joins into one string')
            self.assertFalse([t for t in records['C'] if str(config) in t],
                             'the config path must not appear in the remote command line')
            self.assertEqual(result.stdout.splitlines(), ['on', '', ''])

    def test_native_api_identity_is_independent_of_checkpoint_repository(self) -> None:
        for model, expected in [('deepseek-ai/DeepSeek-V4.1-Flash', 0),
                                ('wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1', 1)]:
            with self.subTest(model=model):
                result = subprocess.run(
                    ['bash', '-c', 'source scripts/release-common.sh; release_native_model_list_matches "$RELEASE_NATIVE_API_MODEL_ID"'],
                    cwd=ROOT, input=json.dumps({'object': 'list', 'data': [{'id': model}]}),
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_paired_build_setting_is_explicit_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'release.config'
            for setting in ('on', 'off', 'invalid'):
                with self.subTest(setting=setting):
                    config.write_text((ROOT / 'ds41rt.config').read_text()
                                      + f'\nEXL3_PAIRED_TP4={setting}\n')
                    result = subprocess.run(
                        ['bash', '-c', 'source scripts/release-common.sh; release_load_config "$1"; printf "%s" "$EXL3_PAIRED_TP4"',
                         'test', str(config)], cwd=ROOT, capture_output=True, text=True)
                    if setting == 'invalid':
                        self.assertEqual(result.returncode, 2)
                        self.assertIn('EXL3_PAIRED_TP4 must be on or off', result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, setting)

    def package_identity(self, manifest: dict) -> subprocess.CompletedProcess:
        return subprocess.run(
            ['bash', '-c', 'source scripts/release-common.sh; release_exl3_package_identity test-revision'],
            cwd=ROOT, input=json.dumps(manifest), text=True, capture_output=True,
        )

    def test_exl3_preflight_identity_binds_layout_and_package(self) -> None:
        manifest = dict(schema='ds41rt.exl3-package.v1', role='spark',
                        sparkinfer_revision='test-revision', files={'kernel': 'first'})
        disjoint = self.package_identity(manifest)
        self.assertEqual(disjoint.returncode, 0, disjoint.stderr)
        self.assertTrue(disjoint.stdout.startswith('disjoint:'))
        manifest['paired_tp4'] = True
        paired = self.package_identity(manifest)
        self.assertEqual(paired.returncode, 0, paired.stderr)
        self.assertTrue(paired.stdout.startswith('paired:'))
        self.assertNotEqual(disjoint.stdout, paired.stdout)
        manifest['files']['kernel'] = 'second'
        self.assertNotEqual(paired.stdout, self.package_identity(manifest).stdout)
        for field, value in [('paired_tp4', 'true'), ('paired_tp4', None),
                             ('role', 'coordinator'), ('schema', 'wrong'),
                             ('sparkinfer_revision', 'another-build')]:
            with self.subTest(field=field, value=value):
                result = self.package_identity({**manifest, field: value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('invalid Spark EXL3 package identity', result.stderr)

    def test_shell_is_valid_and_help_exposes_native_controls(self) -> None:
        subprocess.run(
            ["bash", "-n", "run.sh", "scripts/release-common.sh"],
            cwd=ROOT,
            check=True,
        )
        help_text = subprocess.run(
            ["./run.sh", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for option in (
            "--listen",
            "--rtx-gpus",
            "--concurrency",
            "--kv-pool-size",
            "--memory-reservation",
            "--prefix-cache-entries",
            "--max-context-tokens",
            "--max-output-tokens",
            "--prefill-batch-tokens",
            "--dspark",
            "--no-dspark",
        ):
            self.assertIn(option, help_text)

    def test_standard_release_defaults_are_native(self) -> None:
        script = r'''
source scripts/release-common.sh
release_load_config ds41rt.config
printf '%s\n' "$MODEL_ID" "$MODEL_REVISION" "$EXPERT_FORMAT" "$SPARKINFER_EXL3" \
  "$CONCURRENCY" "$PREFIX_CACHE_ENTRIES" "$MAX_CONTEXT_TOKENS" \
  "$MAX_OUTPUT_TOKENS" "$ADDR" "$EXPERT_PORT" "$RTX_GPUS"
'''
        values = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(
            values,
            [
                "deepseek-ai/DeepSeek-V4.1-Flash",
                "dba1be0a40aa45a94ad051997016db3960a90277",
                "native",
                "disable",
                "16",
                "20",
                "1048576",
                "393216",
                "0.0.0.0:8000",
                "19441",
                "auto",
            ],
        )

    def test_published_images_and_full_model_are_release_defaults(self) -> None:
        config = (ROOT / "ds41rt.config").read_text()
        self.assertIn("MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash", config)

        def value(key: str) -> str:
            lines = [line for line in config.splitlines() if line.startswith(f"{key}=")]
            self.assertEqual(len(lines), 1, f"{key} must be named exactly once")
            return lines[0].split("=", 1)[1]

        coordinator = value("COORDINATOR_DOCKER_INFERENCE")
        spark = value("SPARK_EXPERT_DOCKER_INFERENCE")
        # Both roles must name the same release: a mixed-version default pair
        # still passes each individual assertion, yet the launcher's engine
        # identity check then rejects the deployment at startup.
        self.assertTrue(coordinator.startswith("ghcr.io/"), coordinator)
        self.assertTrue(spark.startswith("ghcr.io/"), spark)
        self.assertEqual(coordinator.rsplit(":", 1)[1], spark.rsplit(":", 1)[1])
        # Every example config must name this same promoted pair. An example
        # that pins a per-topology local tag (a `*-candidate` reference built
        # only by that exact config) fails `run.sh`'s image check on a host that
        # only has the promoted release, which is what it is documenting.
        examples = sorted((ROOT / "examples" / "configs").glob("*.config"))
        self.assertTrue(examples, "the example directory must not be empty")
        for path in examples:
            with self.subTest(example=path.name):
                text = path.read_text()
                self.assertIn(f"COORDINATOR_DOCKER_INFERENCE={coordinator}", text)
                self.assertIn(f"SPARK_EXPERT_DOCKER_INFERENCE={spark}", text)
        # The documented pull commands must name the release the launcher uses.
        readme = (ROOT / "README.md").read_text()
        self.assertIn(f"docker pull {coordinator}", readme)
        self.assertIn(f"docker pull {spark}", readme)
        self.assertIn("The official full checkpoint remains the default", readme)
        self.assertNotIn("docker pull ghcr.io/tpurtell/ds41rt-coordinator:v3", readme)

    def test_launchers_use_ds41rt_container_names(self) -> None:
        combined = (ROOT / "build.sh").read_text() + (ROOT / "run.sh").read_text()
        common = (ROOT / "scripts/release-common.sh").read_text()
        self.assertNotIn("ds4rt", combined.lower())
        self.assertIn("ds41rt-coordinator", common)
        self.assertIn("ds41rt-spark-expert", common)
        self.assertIn("expertd-native", combined)
        self.assertIn("serve-native", combined)

    def test_standard_launch_records_runtime_placement(self) -> None:
        script = (ROOT / "run.sh").read_text()
        self.assertIn('RUST_LOG=${RUST_LOG:-info}', script)
        self.assertIn('--rtx-gpus "$RELEASE_RTX_GPUS"', script)
        self.assertIn('--first-layer "$first_layer"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES=$gpu_uuid_csv', script)
        self.assertIn("trap cleanup EXIT", script)

    def test_coordinator_release_contains_both_rtx_expert_interfaces(self) -> None:
        script = (ROOT / "scripts/build-release-artifacts.sh").read_text()
        self.assertIn('-DDS41RT_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('-DDS41RT_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot"', script)
        self.assertIn('"ds41rt_v41_local_expert_info"', script)
        self.assertIn('"ds41rt_v41_tp2_expert_info"', script)

    def test_invalid_direct_overrides_fail_before_external_checks(self) -> None:
        for args, message in (
            (["--concurrency", "17"], "CONCURRENCY must be in 1..16"),
            (["--prefix-cache-entries", "129"], "PREFIX_CACHE_ENTRIES must be in 0..128"),
            (["--max-context-tokens", "1048577"], "MAX_CONTEXT_TOKENS must be in 1..1048576"),
            (["--max-output-tokens", "393217"], "MAX_OUTPUT_TOKENS must be in 1..393216"),
            (["--rtx-gpus", "3"], "RTX_GPUS must be auto, 1, or 2"),
        ):
            result = subprocess.run(
                ["./run.sh", *args], cwd=ROOT, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(message, result.stderr)


class V10BuildTargetTest(unittest.TestCase):
    """The retained v10 build target and the promoted runtime default.

    `ds41rt.build-v10.config` is retained as an explicit historical release
    BUILD target (build.sh derives `release_version` from its coordinator tag).
    The runtime default is now promoted to `v11`, so the retained target
    deliberately differs from `ds41rt.config`; the promoted pair's own equality
    assertion lives in `test_promoted_build_config_matches_the_runtime_default`.

    The default-pair assertions are deliberately derived from `ds41rt.config`
    rather than hardcoded, so promoting the runtime default to a later release
    does not require rewriting this class: only the retained `v10` target below
    and the example-config expectation remain version-pinned.
    """

    BUILD_CONFIG = ROOT / "ds41rt.build-v10.config"

    def dry_run(self, config: Path | None) -> str:
        args = ["bash", "build.sh"]
        if config is not None:
            args += ["--config", str(config)]
        args += ["--dry-run"]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def assignments(self, path: Path) -> list[str]:
        return [line for line in path.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#")]

    def config_value(self, path: Path, key: str) -> str:
        for line in self.assignments(path):
            name, _, value = line.partition("=")
            if name.strip() == key:
                return value.strip()
        self.fail(f"{key} not found in {path}")

    def test_promoted_build_config_matches_the_runtime_default(self) -> None:
        # The promoted release pair: the v11 build target must equal the runtime
        # default exactly, so a plain `./build.sh` derives the v11 tag.
        base = self.assignments(ROOT / "ds41rt.config")
        target = self.assignments(ROOT / "ds41rt.build-v11.config")
        self.assertEqual(len(base), len(target))
        self.assertEqual(base, target)

    def test_default_build_reports_the_runtime_default_pair(self) -> None:
        """The default derives its tag from ds41rt.config, whatever it names."""
        default = self.dry_run(None)
        config = ROOT / "ds41rt.config"
        coordinator = self.config_value(config, "COORDINATOR_DOCKER_INFERENCE")
        spark = self.config_value(config, "SPARK_EXPERT_DOCKER_INFERENCE")
        tag = coordinator.rsplit(":", 1)[1]
        self.assertIn(f"release tag: {tag}", default)
        self.assertIn(f"coordinator image: {coordinator}", default)
        self.assertIn(f"spark image: {spark}", default)
        # The promoted default carries the universal role set.
        self.assertIn("tp2;tp3;tp6", default)

    def test_the_retained_v10_build_target_still_reports_v10(self) -> None:
        # CPU-only: --dry-run validates without touching Docker, SSH or images.
        # This is the explicit historical build target and must not drift with
        # the runtime promotion.
        v10 = self.dry_run(self.BUILD_CONFIG)
        self.assertIn("release tag: v10", v10)
        self.assertIn("coordinator image: ghcr.io/tpurtell/ds41rt-coordinator:v10", v10)
        self.assertIn("spark image: ghcr.io/tpurtell/ds41rt-spark-expert:v10", v10)
        self.assertIn("tp2;tp3;tp6", v10)

    def test_all_examples_use_the_promoted_pair(self) -> None:
        examples = sorted((ROOT / "examples" / "configs").glob("*.config"))
        self.assertTrue(examples, "the example directory must not be empty")
        for path in examples:
            with self.subTest(example=path.name):
                text = path.read_text()
                self.assertIn("COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v11", text)
                self.assertIn("SPARK_EXPERT_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-spark-expert:v11", text)


class V11ReleaseBuildTargetTest(unittest.TestCase):
    """The promoted v11 release build target.

    `ds41rt.build-v11.config` selects the `v11` tag for the release build and,
    after promotion, is identical to the runtime default; `run.sh` therefore
    derives the same `v11` pair with or without `--config
    ds41rt.build-v11.config`. That equality is asserted by the promoted
    build-target test above and by `scripts/tests/test_release_helpers.py`,
    owned by the release executor.
    """

    BUILD_CONFIG = ROOT / "ds41rt.build-v11.config"

    def dry_run(self, config: Path) -> str:
        result = subprocess.run(
            ["bash", "build.sh", "--config", str(config), "--dry-run"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_the_v11_target_reports_v11_and_the_universal_roles(self) -> None:
        if not self.BUILD_CONFIG.is_file():
            self.skipTest("ds41rt.build-v11.config is not present in this checkout")
        v11 = self.dry_run(self.BUILD_CONFIG)
        self.assertIn("release tag: v11", v11)
        self.assertIn("coordinator image: ghcr.io/tpurtell/ds41rt-coordinator:v11", v11)
        self.assertIn("spark image: ghcr.io/tpurtell/ds41rt-spark-expert:v11", v11)
        self.assertIn("tp2;tp3;tp6", v11)


if __name__ == "__main__":
    unittest.main()
