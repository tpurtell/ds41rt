from __future__ import annotations

import subprocess
import json
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
        for digest in ('', 'a' * 64):
            with self.subTest(digest=digest):
                # OpenSSH joins the command arguments for a remote shell. An
                # empty local argument is not retained as an empty remote one.
                harness = '''set -euo pipefail
ssh() { shift 3; bash -c "$*"; }
seed_host=fixture
remote_dir=/fixture
SPARK_EXPERT_DOCKER_DEV=dev
SPARK_EXPERT_DOCKER_INFERENCE=inference
engine_commit=engine
sparkinfer_commit=fork
release_version=v5
EXL3_PAIRED_TP4=on
source_manifest_sha256="$1"
'''
                harness += invocation + "<<'REMOTE'" + preamble
                harness += 'printf "%s\\n" "$exl3_paired_tp4" "$source_manifest_sha256"\nREMOTE\n'
                result = subprocess.run(['bash', '-c', harness, 'test', digest],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ['on', digest])

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

    def test_v7_published_images_and_full_model_are_release_defaults(self) -> None:
        config = (ROOT / "ds41rt.config").read_text()
        self.assertIn("MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash", config)
        # Both roles must name the same release: a mixed-version default pair
        # still passes each individual assertion, yet the launcher's engine
        # identity check then rejects the deployment at startup.
        self.assertIn("COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v7", config)
        self.assertIn("SPARK_EXPERT_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-spark-expert:v7", config)
        readme = (ROOT / "README.md").read_text()
        self.assertIn("docker pull ghcr.io/tpurtell/ds41rt-coordinator:v7", readme)
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


if __name__ == "__main__":
    unittest.main()
