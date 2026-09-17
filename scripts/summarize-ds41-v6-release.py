#!/usr/bin/env python3
"""Assemble native v6 measurements; reject incomplete or mismatched campaigns."""
import argparse
import csv
import io
import hashlib
import json
from pathlib import Path
import runpy
import re

HERE = Path(__file__).resolve().parent
BASES = [0, 32768, 65536, 131072, 262144]
SUFFIXES = [1024, 2048, 4096, 8192, 16384, 32768]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_settings(settings):
    rows = list(csv.DictReader(io.StringIO(settings), skipinitialspace=True))
    assert rows, 'missing GPU settings'
    for row in rows:
        assert float(row['power.limit [W]'].split()[0]) == 400.0, row
        assert int(row['clocks.max.memory [MHz]'].split()[0]) == 14001, row


def validate_prefill_samples(report):
    assert report['passed'] and report['repeats'] == 3
    assert report['bases'] == BASES and report['suffixes'] == SUFFIXES
    expected = {(base, suffix, repeat) for base in BASES for suffix in SUFFIXES
                for repeat in (1, 2, 3)}
    observed = [(row['base_context_tokens'], row['suffix_tokens'], row['repeat'])
                for row in report['samples'] if row['timed']]
    assert len(observed) == len(expected) and set(observed) == expected
    assert all(row['passed'] for row in report['samples'])


def assemble(directory):
    helpers = runpy.run_path(str(HERE / 'summarize-ds41-phase2-release.py'))
    checks = runpy.run_path(str(HERE / 'summarize-ds41-upstream-release.py'))
    corpus_path = HERE / 'fixtures/release-semantic-corpus.json'
    corpus = read(corpus_path)
    image = read(directory / 'coordinator-image.json')
    data = directory / 'performance'
    artifacts = set()
    layouts, launches, phases = {}, {}, {}
    corpus_hash, context_hash = digest(corpus_path), None
    for layout, count in [('single', 1), ('dual', 2)]:
        expected = {
            'dspark': [f'{layout}-decode.json', f'{layout}-prefill.json',
                       f'{layout}-retained-decode.json',
                       *[f'{layout}-{case}.json' for case in ['code', 'topic', 'counting']],
                       *[f'{layout}-mixed-{i}.json' for i in range(1, 4)]],
            'target': [f'{layout}-target-decode.json'],
            'retained_2k': [f'{layout}-retained-2k.json'],
        }
        for phase, names in expected.items():
            path = data / f'{layout}-{phase}-execution.json'
            execution = read(path)
            assert execution['passed'] and execution['completed_ns'], path
            validate_settings(execution['gpu_settings'])
            container = execution['container']
            assert container['Image'] == image['image_id'], path
            assert container['Config']['Labels']['org.opencontainers.image.revision'] == image['engine_commit']
            argv = container['Config']['Cmd']
            assert argv[argv.index('--rtx-gpus') + 1] == str(count)
            assert ('--dspark' in argv) == (phase != 'target')
            assert argv[argv.index('--prefix-cache-entries') + 1] == '20'
            assert argv[argv.index('--host-cache-bytes') + 1] == 'auto'
            assert not any(flag in argv for flag in ['--tp2-attention', '--tp2-query-projection',
                                                    '--tp2-output-projection', '--tp2-dspark-experts'])
            observed = []
            for command in execution['commands']:
                assert command['exit_code'] == 0 and command['completed_ns'], command
                cmd = command['command']
                output = data / Path(cmd[cmd.index('--output') + 1]).name
                observed.append(output.name)
                raw = read(output)
                assert raw['passed'] is True, output
                if 'corpus_sha256' in raw:
                    assert raw['corpus_sha256'] == corpus_hash, output
                if 'context_sha256' in raw:
                    if context_hash is None:
                        context_hash = raw['context_sha256']
                    assert raw['context_sha256'] == context_hash, output
                artifacts.add(output)
            assert sorted(observed) == sorted(names), (path, observed, names)
            phases[f'{layout}-{phase}'] = {
                'arguments': argv, 'image_id': container['Image'],
                'started_ns': execution['started_ns'], 'completed_ns': execution['completed_ns'],
                'commands': execution['commands'], 'gpu_settings': execution['gpu_settings'],
            }
            artifacts.add(path)
        launch_path = directory / f'{layout}-launch.json'
        launch = read(launch_path)
        assert launch['passed'] and launch['launch_exit_code'] == 0
        assert launch['coordinator']['Image'] == image['image_id']
        workers = launch['workers']
        assert len(workers) == 4
        assert len({worker['Image'] for worker in workers.values()}) == 1
        assert all(worker['State']['Running'] and
                   worker['Config']['Labels']['org.opencontainers.image.revision'] == image['engine_commit']
                   for worker in workers.values())
        server_log = directory / f'{layout}-launch-server.log'
        deployment = checks['summarize_deployment']({
            'layout': layout, 'arguments': launch['coordinator']['Config']['Cmd'],
            'workers': [{'args': worker['Config']['Cmd']} for worker in workers.values()],
        }, server_log)
        text = re.sub(r'\x1b\[[0-9;]*m', '', server_log.read_text())
        budgets = re.findall(r'automatic host cache budget.*?Budget \{ ([^}]+) \}', text)
        assert len(budgets) == 1, server_log
        capacity = {key.strip(): int(value) for key, value in
                    (entry.split(':', 1) for entry in budgets[0].split(','))}
        assert capacity['device_tokens'] == deployment['logical_pool_tokens']
        assert capacity['combined_tokens'] == capacity['device_tokens'] + capacity['host_tokens']
        argv = launch['coordinator']['Config']['Cmd']
        target = int(argv[argv.index('--prefix-cache-entries') + 1]) * int(argv[argv.index('--max-context-tokens') + 1])
        assert capacity['target_tokens'] == target and capacity['combined_tokens'] > target
        artifacts.add(server_log)
        gpu_uuids = [uuid for request in launch['coordinator']['HostConfig']['DeviceRequests']
                     for uuid in request['DeviceIDs']]
        assert len(gpu_uuids) == len(set(gpu_uuids)) == count
        launches[layout] = {'startup_seconds': launch['startup_seconds'], 'gpu_uuids': gpu_uuids,
                           'deployment': deployment, 'ram_capacity': capacity,
                           'gpu_memory': launch['gpu_memory'], 'placement': launch.get('placement'),
                           'worker_image_id': next(iter(workers.values()))['Image']}
        artifacts.add(launch_path)
        result = {}
        for mode, prefix in [('decode', ''), ('target_decode', 'target-')]:
            path = data / f'{layout}-{prefix}decode.json'
            checks['validate_decode_corpus'](read(path), corpus)
            result[mode] = helpers['summarize_decode'](path)
            assert result[mode]['repeats'] == 3
        result['concurrency'] = {case: helpers['summarize_concurrency'](data / f'{layout}-{case}.json')
                                 for case in ['code', 'topic', 'counting']}
        result['mixed'] = helpers['summarize_mixed']([data / f'{layout}-mixed-{i}.json' for i in range(1, 4)])
        validate_prefill_samples(read(data / f'{layout}-prefill.json'))
        result['prefill'] = helpers['summarize_prefill'](data / f'{layout}-prefill.json')
        assert result['prefill']['bases'] == BASES and result['prefill']['suffixes'] == SUFFIXES
        for name, suffix, contexts in [('retained_decode', 'retained-decode', BASES),
                                        ('retained_decode_2k', 'retained-2k', [2048])]:
            path = data / f'{layout}-{suffix}.json'
            checks['validate_retained_corpus'](read(path), corpus)
            result[name] = helpers['summarize_retained'](path)
            assert result[name]['contexts'] == contexts
        layouts[layout] = result
    assert launches['single']['worker_image_id'] == launches['dual']['worker_image_id']
    artifacts.add(directory / 'coordinator-image.json')
    return {'schema': 1, 'release': 'v6', 'scope': 'Native checkpoint performance; EXL3 results are historical and separate.',
            'engine_commit': image['engine_commit'], 'coordinator_image_id': image['image_id'],
            'binary_sha256': image['artifacts']['ds41rt'],
            'native_library_sha256': image['artifacts']['libds41rt_native.so'], 'corpus_sha256': corpus_hash,
            'context_sha256': context_hash, 'weighted_case_ids': corpus['weighted_case_ids'],
            'controls': {'repeats': 3, 'rtx_power_limit_watts': 400, 'rtx_memory': 'stock',
                         'thinking': 'per-case; reasoning code uses high effort', 'tp2_options': 'off'},
            'layouts': layouts, 'launches': launches, 'phases': phases,
            'artifacts': [{'path': str(path.relative_to(directory)), 'sha256': digest(path),
                           'bytes': path.stat().st_size} for path in sorted(artifacts)],
            'performance_matrix_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    report = assemble(args.input)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
