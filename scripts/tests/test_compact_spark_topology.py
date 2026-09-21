"""CPU-only coverage: no real Docker, SSH, GPU, or serving operations."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'runs/v7q-a1/serve-diffbot.sh'


class CompactSparkTopologyTest(unittest.TestCase):
    def config(self, settings, command='printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE"'):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'config'
            config.write_text((ROOT / 'ds41rt.config').read_text() + '\n' + settings + '\n')
            return subprocess.run(['bash', '-euc',
                'source scripts/release-common.sh; release_load_config "$1"; ' + command,
                'test', str(config)], cwd=ROOT, text=True, capture_output=True)

    def test_compact_defaults_and_absolute_ceiling(self):
        base = 'SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto'
        for value, success in [('', True), ('32GiB', True), ('31GiB', True),
                               ('34359738368B', True), ('32768MiB', True),
                               ('34359738369B', False), ('0GiB', False),
                               ('32.000001GiB', False), ('50%', False)]:
            with self.subTest(value=value):
                result = self.config(base + '\nMEMORY_RESERVATION=' + value)
                self.assertEqual(result.returncode, 0 if success else 2, result.stderr)
                if success:
                    self.assertEqual(result.stdout.splitlines(), [value or '32GiB', '2GiB'])
        result = self.config(base + '\nKV_POOL_SIZE=1GiB')
        self.assertEqual(result.stdout.splitlines(), ['32GiB', '1GiB'])
        for settings in ['SPARK_COUNT=2', base + '\nRTX_GPUS=2',
                         base + '\nEXL3_PAIRED_TP4=on']:
            self.assertEqual(self.config(settings).returncode, 2)

    def test_compact_prefill_cap_is_logged_and_preserves_smaller_requests(self):
        base = 'SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto'
        for requested in (80, 128, 256, 2048, 4096):
            result = self.config(base + f'\nPREFILL_BATCH_TOKENS={requested}',
                                 'printf "%s" "$PREFILL_BATCH_TOKENS"')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, str(min(requested, 256)))
            self.assertEqual('caps PREFILL_BATCH_TOKENS' in result.stderr, requested > 256)
        result = self.config('', 'printf "%s" "$PREFILL_BATCH_TOKENS"')
        self.assertEqual(result.stdout, '2048')

    def test_only_active_hosts_and_rails_are_required(self):
        result = self.config('SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto\n'
                             'SPARK_2_HOST=\nSPARK_3_HOST=\nSPARK_2_LANE_A=\nSPARK_3_LANE_A=\n'
                             'SPARK_2_LANE_B=\nSPARK_3_LANE_B=',
                             'release_hosts_csv; release_lane_a_csv; release_lane_b_csv; release_expert_hosts_csv')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ostrich,dodo\n', result.stdout)
        self.assertNotIn('spark-2', result.stdout)

    def test_stop_helpers_scope_to_active_hosts(self):
        # Replace every external operation used by the stop helpers before calling.
        command = '''
release_stop_local_container() { :; }
release_stop_host_api() { :; }
release_stop_remote_containers() { echo "release:$1"; }
release_stop_persistent_local_container() { :; }
release_stop_persistent_remote_container() { echo "persistent:$1"; }
release_stop_services coord spark
release_stop_wip_containers
'''
        result = self.config('SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto', command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()),
                         {'release:ostrich', 'release:dodo', 'persistent:ostrich', 'persistent:dodo'})

    def helper(self, *args, **env):
        return subprocess.run(['bash', str(HELPER), *args], cwd=ROOT, text=True,
                              capture_output=True,
                              env=dict(os.environ, DRY_RUN='1', SPARK_COUNT='2', **env))

    @unittest.skipUnless(HELPER.exists(), 'manual WIP helper lives in ignored runs/')
    def test_wip_commands_have_world_two_peers_ceiling_and_first_layer(self):
        result = self.helper('start-experts', '3')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line[len('DRY RUN:'):]) for line in result.stdout.splitlines()
                    if line.startswith('DRY RUN:')]
        self.assertEqual(len(commands), 2)
        for rank, command in enumerate(commands):
            self.assertIn(f'--rank {rank} --world 2', command[-1])
            self.assertIn('--first-layer 3', command[-1])
            self.assertIn('--capacity 256', command[-1])
        result = self.helper('start-coordinator', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout.splitlines()[0][len('DRY RUN:'):])[-1]
        self.assertIn('--memory-reservation 32GiB', command)
        self.assertIn('--kv-pool-size 2GiB', command)
        self.assertIn('--prefill-batch-tokens 256', command)
        self.assertIn('--peers 10.55.0.1:19441,10.55.0.2:19441 --rtx-gpus 1', command)
        for args in [('start-coordinator', '2'),
                     ('start-coordinator', '1', '--memory-reservation', '80GiB')]:
            self.assertEqual(self.helper(*args).returncode, 2)
        self.assertEqual(self.helper('start-coordinator', '1', MEMORY_RESERVATION='33GiB').returncode, 2)

    def test_package_variants_cover_every_worker_capacity_and_shape(self):
        variants = [dict(directory=f'tp2-rank{rank}/m{capacity}', capacity=capacity,
                         intermediate=1152, experts=384, top_k=6, output_dtype='bf16', bits=[2,3])
                    for rank in range(2) for capacity in [1,16,80,256,1024,4096]]
        base = dict(compute=[12,1], variants=variants)
        def check(manifest, capacity=4096):
            return subprocess.run(['bash', '-euc',
                'source scripts/release-common.sh; release_validate_exl3_tp2_variants "$1" k23',
                'test', str(capacity)], cwd=ROOT, input=json.dumps(manifest), text=True, capture_output=True)
        self.assertEqual(check(base).returncode, 0)
        for field, value in [('intermediate', 576), ('experts', 383), ('top_k', 5),
                             ('output_dtype', 'fp32'), ('bits', [3,2]), ('capacity', 2)]:
            with self.subTest(field=field):
                candidate = json.loads(json.dumps(base))
                candidate['variants'][0][field] = value
                self.assertEqual(check(candidate).returncode, 2)
        self.assertEqual(check(dict(base, compute=[12,0])).returncode, 2)
        self.assertEqual(check(dict(base, variants=variants[1:])).returncode, 2)
        self.assertEqual(check(dict(base, variants=variants + variants[:1])).returncode, 2)
        small = dict(base, variants=[v for v in variants if v['capacity'] <= 80])
        self.assertEqual(check(small, 80).returncode, 0)
        self.assertEqual(check(small).returncode, 2)

    # --- v10 three-rank profiles: implicit compact TP3, explicit native bypass --

    COMPACT3 = 'SPARK_COUNT=3\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto'
    NATIVE3 = ('SPARK_COUNT=3\nEXPERT_FORMAT=native\nSPARKINFER_EXL3=disable\n'
               'SPARK_TP=3\nSPARK_EP=1')

    def test_compact_tp3_defaults_ceiling_and_prefill_cap(self):
        for value, success in [('', True), ('32GiB', True), ('31GiB', True),
                               ('34359738369B', False), ('50%', False)]:
            with self.subTest(value=value):
                result = self.config(self.COMPACT3 + '\nMEMORY_RESERVATION=' + value)
                self.assertEqual(result.returncode, 0 if success else 2, result.stderr)
                if success:
                    self.assertEqual(result.stdout.splitlines(), [value or '32GiB', '2GiB'])
        # A too-large prefill is clamped with a TP3-named note; smaller stands.
        result = self.config(self.COMPACT3 + '\nPREFILL_BATCH_TOKENS=2048',
                             'printf "%s" "$PREFILL_BATCH_TOKENS"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '256')
        self.assertIn('compact Spark TP3 caps PREFILL_BATCH_TOKENS=2048', result.stderr)
        result = self.config(self.COMPACT3 + '\nPREFILL_BATCH_TOKENS=128',
                             'printf "%s" "$PREFILL_BATCH_TOKENS"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '128')
        self.assertNotIn('caps PREFILL_BATCH_TOKENS', result.stderr)
        # Two-rank behavior is unchanged.
        result = self.config('SPARK_COUNT=2\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto',
                             'printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['32GiB', '2GiB'])

    def test_compact_tp3_rejects_wrong_format_single_rtx_and_paired(self):
        for settings in ['SPARK_COUNT=3',                       # native without keys
                         self.COMPACT3 + '\nRTX_GPUS=2',        # compact is one-RTX only
                         self.COMPACT3 + '\nEXL3_PAIRED_TP4=on' # disjoint packages only
                         ]:
            with self.subTest(settings=settings):
                self.assertEqual(self.config(settings).returncode, 2, settings)

    def test_explicit_native_tp3ep1_bypasses_every_compact_rule(self):
        # The bypass must hold for MEMORY_RESERVATION, KV, prefill and the
        # single-RTX rule: nothing in the compact block may touch this launch.
        result = self.config(self.NATIVE3,
                             'printf "%s\\n" "$MEMORY_RESERVATION" "$KV_POOL_SIZE" '
                             '"$PREFILL_BATCH_TOKENS" "$EXPERT_FORMAT"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['', '', '2048', 'native'])
        result = self.config(self.NATIVE3 + '\nMEMORY_RESERVATION=90%',
                             'printf "%s" "$MEMORY_RESERVATION"')
        self.assertEqual(result.returncode, 0, result.stderr)   # percentages stay legal
        self.assertEqual(result.stdout, '90%')
        result = self.config(self.NATIVE3 + '\nPREFILL_BATCH_TOKENS=4096',
                             'printf "%s" "$PREFILL_BATCH_TOKENS"')
        self.assertEqual(result.returncode, 0, result.stderr)   # no 256 cap
        self.assertEqual(result.stdout, '4096')
        self.assertNotIn('caps PREFILL_BATCH_TOKENS', result.stderr)
        result = self.config(self.NATIVE3 + '\nEXL3_PAIRED_TP4=on')
        self.assertEqual(result.returncode, 2, result.stderr)   # native-only guard
        self.assertIn('native topology', result.stderr)

    def test_malformed_partial_three_rank_topologies_are_rejected(self):
        for settings, message in [
            ('SPARK_COUNT=3\nSPARK_TP=3', 'set together or omitted together'),
            ('SPARK_COUNT=3\nSPARK_EP=1', 'set together or omitted together'),
            ('SPARK_COUNT=3\nSPARK_TP=3\nSPARK_EP=2', 'must equal SPARK_COUNT'),
            ('SPARK_COUNT=3\nSPARK_TP=1\nSPARK_EP=3', 'SPARK_TP must be 2, 3, 4, or 6'),
            ('SPARK_COUNT=3\nSPARK_TP=2\nSPARK_EP=1', 'must equal SPARK_COUNT'),
            ('SPARK_COUNT=3\nSPARK_TP=3\nSPARK_EP=1\nEXPERT_FORMAT=exl3\nSPARKINFER_EXL3=auto',
             'requires EXPERT_FORMAT=native'),
            ('SPARK_COUNT=5', 'SPARK_COUNT must be 0, 2, 3, 4, or 6'),
        ]:
            with self.subTest(settings=settings):
                result = self.config(settings)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

    def test_compact_tp3_package_variants_admission(self):
        # 2304/3 = 768-column shards for every rank and worker capacity; a
        # missing rank must fail admission even when ranks 0 and 1 are whole.
        variants = [dict(directory=f'tp3-rank{rank}/m{capacity}', capacity=capacity,
                         intermediate=768, experts=384, top_k=6, output_dtype='bf16', bits=[2, 3])
                    for rank in range(3) for capacity in [1, 16, 80, 256, 1024, 4096]]
        base = dict(compute=[12, 1], variants=variants)
        def check(manifest, capacity=4096):
            return subprocess.run(['bash', '-euc',
                'source scripts/release-common.sh; release_validate_exl3_compact_variants "$1" k23 3',
                'test', str(capacity)], cwd=ROOT, input=json.dumps(manifest), text=True,
                capture_output=True)
        passed = check(base)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        missing_rank2 = dict(base, variants=[v for v in variants
                                             if not v['directory'].startswith('tp3-rank2')])
        failed = check(missing_rank2)
        self.assertEqual(failed.returncode, 2)
        self.assertIn('lacks required TP3', failed.stderr)
        for field, value in [('intermediate', 1152), ('intermediate', 576), ('bits', [3, 2])]:
            with self.subTest(field=field, value=value):
                candidate = json.loads(json.dumps(base))
                candidate['variants'][0][field] = value
                self.assertEqual(check(candidate).returncode, 2)
        self.assertEqual(check(dict(base, compute=[12, 0])).returncode, 2)
        # Compact admission only exists for the compact degrees TP2 and TP3.
        rejected = subprocess.run(['bash', '-euc',
            'source scripts/release-common.sh; release_validate_exl3_compact_variants 256 k23 4',
            'test'], cwd=ROOT, input=json.dumps(base), text=True, capture_output=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn('supports TP2 or TP3', rejected.stderr)
        # A TP3 layout never passes the TP2 shim and vice versa.
        shim = subprocess.run(['bash', '-euc',
            'source scripts/release-common.sh; release_validate_exl3_tp2_variants 256 k23',
            'test'], cwd=ROOT, input=json.dumps(base), text=True, capture_output=True)
        self.assertEqual(shim.returncode, 2)

    def test_daemon_remote_arguments_preserve_world(self):
        source = (ROOT / 'run.sh').read_text()
        invocation, remote = source.split('echo "== starting native Spark experts =="', 1)[1].split("<<'REMOTE' &", 1)
        remote = remote.split('\nREMOTE', 1)[0]
        self.assertIn('"$SPARK_COUNT"', invocation)
        # The legacy ten-positional invocation must survive for both compact
        # worlds: the coordinator keeps the worker at the world it was given.
        for world in (2, 3):
            with self.subTest(world=world):
                result = subprocess.run(['bash', '-euc', 'docker() { printf "%s\\n" "$@"; }\n' +
                    remote.replace(' >/dev/null', ''), 'test', 'image', 'name', '1', '4096',
                    '107374182400', '19441', 'snapshot', 'fingerprint', '3', str(world)],
                    cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()
                self.assertEqual(arguments[arguments.index('--world') + 1], str(world))
                self.assertEqual(arguments[arguments.index('--first-layer') + 1], '3')
                # A legacy world-2/3 launch carries no explicit topology flags.
                self.assertNotIn('--spark-tp', arguments)
                self.assertNotIn('--spark-ep', arguments)


if __name__ == '__main__':
    unittest.main()
