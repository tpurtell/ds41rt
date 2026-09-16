#!/usr/bin/env python3
"""Archive completed fixed-width calibration with exact parser reproduction.

Keep cost, draft-acceptance, and lane-round records from every content window.
Raw source hashes and remapped offsets preserve trace provenance. This does not
turn instrumented calibration into a release throughput measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--summary', required=True, type=Path)
    args = parser.parse_args()
    outputs = [path.resolve() for path in [args.stage, args.archive, args.summary]]
    if len(set(outputs)) != 3 or any(path.is_relative_to(outputs[0]) for path in outputs[1:]):
        parser.error('outputs must be distinct; archive and summary must be outside stage')
    if any(path.exists() for path in [args.stage, args.archive, args.summary]):
        parser.error('stage, archive and summary must all be new')
    if len({path.name for path in args.directories}) != len(args.directories):
        parser.error('source directory names must be unique')
    here = Path(__file__).resolve().parent
    costs = runpy.run_path(str(here / 'fit-ds41-placement-costs.py'))
    policy = runpy.run_path(str(here / 'summarize-ds41-native-policy.py'))
    args.stage.mkdir(parents=True)
    report = dict(scope=__doc__, configurations=[])
    markers = [b'ds41rt::cost_model:', b'native draft policy observation',
               b'native independent lane round', b'native scheduler round']
    for src in args.directories:
        for case in ['code', 'topic', 'code-reasoning', 'mixed']:
            if not json.loads((src / f'{case}.json').read_text())['passed']:
                raise ValueError(f'failed collection: {src}/{case}')
        if (src / 'coordinator-restored.log').read_text().strip() != 'ds41rt-coordinator':
            raise ValueError(f'collection not restored: {src}')
        dst = args.stage / src.name
        dst.mkdir()
        raw = (src / 'server.log').read_bytes()
        windows = json.loads((src / 'segments.json').read_text())
        filtered = bytearray()
        mapped = []
        acceptance = []
        for window in windows:
            original = raw[window['begin']:window['end']]
            begin = len(filtered)
            part = b''.join(line for line in original.splitlines(keepends=True)
                            if any(marker in line for marker in markers))
            filtered.extend(part)
            mapped.append(dict(workload=window['workload'], begin=begin, end=len(filtered)))
            before = policy['parse'](original.decode())
            after = policy['parse'](part.decode())
            if before != after:
                raise ValueError(f'acceptance records changed: {src}/{window["workload"]}')
            summary = policy['summarize'](*after)
            summary['workload'] = window['workload']
            saved = src / f'{window["workload"]}-acceptance.json'
            if saved.exists():
                metadata = json.loads(saved.read_text())
                for key, value in summary.items():
                    if json.dumps(metadata[key], sort_keys=True) != json.dumps(value, sort_keys=True):
                        raise ValueError(f'acceptance summary changed: {saved}/{key}')
                summary.update({key: metadata[key] for key in
                                ['model', 'rtx_gpus', 'draft_limit', 'draft_policy']})
            acceptance.append(summary)
        (dst / 'server.log').write_bytes(filtered)
        (dst / 'segments.json').write_text(json.dumps(mapped, indent=2)+'\n')
        provenance = dict(original_trace_sha256=sha(raw), filtered_trace_sha256=sha(filtered),
                          filter='Cost-model, draft-observation, and lane-round records in all original content windows.')
        (dst / 'source.json').write_text(json.dumps(provenance, indent=2)+'\n')
        # Preserve source-bound exclusions without silently broadening them.
        exclusion = src / 'excluded_non_verification_batches.json'
        if exclusion.exists():
            value = json.loads(exclusion.read_text())
            if value['trace_sha256'] != sha(raw):
                raise ValueError(f'stale exclusions: {src}')
            value['original_trace_sha256'] = value['trace_sha256']
            value['trace_sha256'] = sha(filtered)
            (dst / exclusion.name).write_text(json.dumps(value, indent=2)+'\n')
        original_records, _ = costs['observations'](src)
        archived_records, _ = costs['observations'](dst)
        if original_records != archived_records:
            raise ValueError(f'cost records changed: {src}')
        for path in src.iterdir():
            if path.is_file() and (path.suffix in ('.json', '.py', '.sh') or path.name.endswith('-restored.log')):
                if not (dst / path.name).exists():
                    (dst / path.name).write_bytes(path.read_bytes())
        report['configurations'].append(dict(directory=src.name, **provenance,
                                            cost_rounds=len(original_records), acceptance=acceptance))
        print(f'Exact cost and acceptance reproduction: {src.name} ({len(original_records)} rounds)', flush=True)
    for name in ['archive-ds41-calibration.py', 'fit-ds41-placement-costs.py', 'summarize-ds41-native-policy.py']:
        (args.stage / name).write_bytes((here / name).read_bytes())
    hashes = {str(path.relative_to(args.stage)): sha(path.read_bytes())
              for path in sorted(args.stage.rglob('*')) if path.is_file()}
    (args.stage / 'sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    with tarfile.open(args.archive, 'x:gz') as archive:
        for path in sorted(args.stage.iterdir()):
            archive.add(path, arcname=path.name)
    report['archive_sha256'] = sha(args.archive.read_bytes())
    report['archive'] = str(args.archive)
    args.summary.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
