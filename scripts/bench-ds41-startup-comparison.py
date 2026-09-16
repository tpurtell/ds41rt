#!/usr/bin/env python3
"""Compare coordinator loading under fixed geometry and alternating binaries.

Only polls health; no inference or worker-startup performance is measured.
The caller owns stopping/restoring any existing coordinator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--before-commit', required=True)
    parser.add_argument('--after-commit', required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--peers', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, default=25)
    parser.add_argument('--port', type=int, default=18071)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory must be new')
    args.output.mkdir(parents=True)
    binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
    common = ['serve-native', '--snapshot', str(args.snapshot.resolve()),
              '--native-lib', str(args.native_lib.resolve()), '--peers', args.peers,
              '--rtx-gpus', '2', '--rtx-expert-layers', str(args.layers),
              '--listen', f'127.0.0.1:{args.port}', '--prefill-batch-tokens', '2048',
              '--concurrency', '16', '--prefix-cache-entries', '24',
              '--max-context-tokens', '1048576', '--max-output-tokens', '393216',
              '--dspark', '--dspark-draft-limit', '7']
    report = dict(scope=__doc__, before_commit=args.before_commit, after_commit=args.after_commit,
                  binary_sha256={k: sha(v) for k, v in binaries.items()},
                  native_sha256=sha(args.native_lib), arguments=common,
                  order=['before', 'after', 'after', 'before', 'before', 'after'],
                  warmup='one before launch excluded from summaries', samples=[], passed=False)
    report_path = args.output / 'report.json'

    def sample(label, index, warmup):
        try:
            with socket.create_connection(('127.0.0.1', args.port), timeout=.2):
                raise RuntimeError('benchmark port already in use')
        except (ConnectionRefusedError, TimeoutError):
            pass
        path = args.output / f'{index}-{label}.log'
        command = [str(binaries[label]), *common]
        result = dict(label=label, index=index, warmup=warmup, command=command)
        with path.open('w') as log:
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while time.monotonic() - started < args.timeout:
                    if process.poll() is not None:
                        raise RuntimeError(f'candidate exited before ready; see {path}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=.5) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.05)
                else:
                    raise TimeoutError(f'candidate startup timed out; see {path}')
                result['wall_seconds_to_health'] = time.monotonic() - started
                result['process_io'] = {k: int(v) for k, v in
                    (line.split(':') for line in Path(f'/proc/{process.pid}/io').read_text().splitlines())}
                result['process_status'] = Path(f'/proc/{process.pid}/status').read_text()
                result['gpu_at_ready'] = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,uuid,memory.used,power.limit,clocks.mem,clocks.sm', '--format=csv'], text=True)
                contents = path.read_text()
                owner = re.findall(r'serving owners ready elapsed_ms=(\d+)', contents)
                assert len(owner) == 1, 'missing or duplicate owner-ready observation'
                result['owner_seconds'] = int(owner[0]) / 1000
                pool = re.findall(r'dual RTX cache reservation[^\n]*global_bytes=(\d+)', contents)
                assert len(pool) == 1 and int(pool[0]) == 13094420480, 'KV pool changed'
                result['kv_global_bytes'] = int(pool[0])
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    result['forced_shutdown'] = True
                # Let the terminated CUDA context release its allocations.
                time.sleep(1)
        report['samples'].append(result)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ['label', 'index', 'warmup', 'owner_seconds',
                                                'wall_seconds_to_health', 'process_io']}), flush=True)

    sample('before', 0, True)
    for index, label in enumerate(report['order'], 1):
        sample(label, index, False)
    import statistics
    report['summaries'] = {}
    for label in binaries:
        rows = [s for s in report['samples'] if s['label'] == label and not s['warmup']]
        report['summaries'][label] = {metric: dict(median=statistics.median(values), min=min(values), max=max(values))
            for metric in ['owner_seconds', 'wall_seconds_to_health']
            for values in [[s[metric] for s in rows]]}
    report['passed'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
