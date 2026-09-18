"""Exercise the real launcher without contacting Docker, SSH or a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MOCK = r'''
import json, os, subprocess, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as log:
    log.write(json.dumps([tool, os.environ.get('MOCK_HOST'), args]) + '\n')
if tool == 'ssh':
    while args[0] == '-o': args = args[2:]
    host, *command = args
    env = dict(os.environ, MOCK_HOST=host)
    sys.exit(subprocess.run(command, input=sys.stdin.read(), text=True, env=env).returncode)
if tool == 'nvidia-smi':
    gpu = '0, GPU-00000000-0000-0000-0000-000000000000, 00000000:11:00.0'
    print(gpu + (', 97887, 97887' if 'memory.total' in ' '.join(args) else ''))
    sys.exit(0)
if tool == 'docker':
    if args == ['info']: sys.exit(0)
    if args[:2] == ['container', 'inspect']: sys.exit(1)
    if args[:2] == ['image', 'inspect']:
        if '-f' in args:
            print('fixture-engine' if 'org.opencontainers.image.revision' in ' '.join(args)
                  else os.environ['MOCK_REVISION'])
        sys.exit(0)
    if args[:1] == ['run'] and '--entrypoint' in args:
        entry = args[args.index('--entrypoint') + 1]
        assert '--gpus' not in args and '--network' in args
        if entry == '/bin/cat':
            # Legacy single-family image: read the package manifest directly.
            assert args[-1] == '/opt/ds41rt/lib/exl3/manifest.json'
        else:
            # Multi-family image: a shell probes the tier family first and
            # falls back to the legacy package manifest.
            assert entry == '/bin/sh'
            script = args[args.index('-c') + 1]
            assert 'exl3' in script and 'manifest.json' in script and 'cat' in script
        paired = os.environ['MOCK_LAYOUT'] == 'paired'
        if os.environ.get('MOCK_MISMATCH') == os.environ['MOCK_HOST']: paired = not paired
        print(json.dumps(dict(schema='ds41rt.exl3-package.v1', role='spark',
                              sparkinfer_revision=os.environ['MOCK_REVISION'], paired_tp4=paired)))
        sys.exit(0)
raise SystemExit('Unexpected external action: ' + tool + ' ' + repr(args))
'''


class Exl3ReleasePreflightTest(unittest.TestCase):
    def launch(self, *, exl3=True, paired=True, mismatch=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / 'bin'
            binary.mkdir()
            for tool in ('docker', 'ssh', 'nvidia-smi', 'curl', 'ss'):
                path = binary / tool
                path.write_text(f'#!{sys.executable}\n' + MOCK)
                path.chmod(0o755)
            snapshot = directory / 'hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277'
            snapshot.mkdir(parents=True)
            (snapshot / 'config.json').write_text(json.dumps(
                {'quantization_config': {'quant_method': 'exl3' if exl3 else 'fp4'}}))
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       HF_HOME=str(directory / 'hf'), MOCK_LOG=str(directory / 'calls.jsonl'),
                       MOCK_REVISION=json.loads((ROOT / 'third_party/sparkinfer.lock.json').read_text())['revision'],
                       MOCK_LAYOUT='paired' if paired else 'disjoint',
                       MOCK_MISMATCH='dodo' if mismatch else '')
            result = subprocess.run(['bash', 'run.sh', '--rtx-gpus', '1',
                                     '--restart' if mismatch else '--dry-run'],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in (directory / 'calls.jsonl').read_text().splitlines()]
            # Every Docker operation must be read-only inspection or the
            # disposable manifest reader. Never stop or launch a serving process.
            for tool, host, args in calls:
                if tool == 'docker':
                    self.assertTrue(args == ['info'] or args[:2] in
                                    (['image', 'inspect'], ['container', 'inspect']) or
                                    (args[0] == 'run' and '--entrypoint' in args), (host, args))
            return result, calls

    def test_matching_packages_pass_dry_run(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                result, calls = self.launch(paired=paired)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Spark EXL3 package: ' + ('paired:' if paired else 'disjoint:'), result.stdout)
                self.assertEqual(sum(tool == 'docker' and args[0] == 'run' for tool, _, args in calls), 4)

    def test_mismatch_rejects_restart_before_stopping_services(self):
        result, _ = self.launch(mismatch=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('Spark EXL3 packages differ across hosts', result.stderr)

    def test_full_model_skips_exl3_package_reads(self):
        result, calls = self.launch(exl3=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(tool == 'docker' and args[0] == 'run' for tool, _, args in calls))


if __name__ == '__main__':
    unittest.main()
