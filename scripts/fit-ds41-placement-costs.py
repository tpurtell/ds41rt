#!/usr/bin/env python3
"""Fit exploratory placement-aware costs from fixed-prefix serving traces.

Fits odd draft widths and reports held-out even widths on code and mixed workloads.
Additional explicitly segmented workloads, such as topic, are entirely held out.
These use observed routes, not forecasts: route-forecast error is a separate gate.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def observations(directory):
    match = re.search(r'rtx([12])-k([1-7])$', directory.name)
    if not match:
        raise ValueError(f'expected directory ending in rtxN-kN: {directory}')
    gpus, width = map(int, match.groups())
    path = directory / 'server.log'
    trace_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    # Older engines could label prefill with stale captured decode routes.
    # Recovery requires an audited, exact-trace exclusion manifest. Never
    # silently drop an incomplete verification round or arbitrary leftovers.
    exclusion_path = directory / 'excluded_non_verification_batches.json'
    exclusions = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else None
    if exclusions and exclusions['trace_sha256'] != trace_hash:
        raise ValueError('non-verification exclusions belong to another trace')
    excluded = {int(x['batch']): x for x in exclusions['batches']} if exclusions else {}
    # The client writes its final code report after receiving the last response;
    # mixed requests are issued only after that client exits. Explicit segment
    # offsets take precedence when the collector recorded them.
    code_end = (directory / 'code.json').stat().st_mtime
    segments = json.loads((directory / 'segments.json').read_text()) if (directory / 'segments.json').exists() else None
    pending = defaultdict(dict)
    records = []
    seen_shapes = Counter()
    offset = 0
    for raw in path.open('rb'):
        begin = offset
        offset += len(raw)
        line = raw.decode()
        if 'ds41rt::cost_model:' not in line or 'verification cost forecast' in line:
            continue
        fields = dict(re.findall(r'(\w+)=("[^"]*"|[^ ]+)', line.strip()))
        batch = int(fields['batch'])
        if 'verification layer cost' in line:
            layer = int(fields['layer'])
            if layer in pending[batch]:
                raise ValueError(f'duplicate layer {batch}/{layer}')
            pending[batch][layer] = fields
            continue
        if batch in excluded:
            raise ValueError(f'cannot exclude a completed verification round: {batch}')
        layers = pending.pop(batch)
        rows = int(fields['rows'])
        verify_us = int(fields['verify_us'])
        if set(layers) != set(range(40)):
            raise ValueError(f'incomplete batch {batch}')
        grouped = {}
        elapsed = 0
        for layer, f in layers.items():
            if int(f['rows']) != rows:
                raise ValueError(f'row mismatch {batch}/{layer}')
            stages = ['produced_us', 'index_us', 'attention_us', 'experts_us', 'finish_us']
            total = sum(int(f[key]) for key in stages)
            if total != int(f['total_us']):
                raise ValueError(f'stage mismatch {batch}/{layer}')
            elapsed += total
            backend = f['routed_backend'].strip('"') + '_shared' + f['shared_tp']
            group = grouped.setdefault(backend, dict(layers=0, unique=0, groups16=0, expert_us=0))
            group['layers'] += 1
            group['unique'] += int(f['distinct_experts'])
            group['groups16'] += int(f['expert_groups_16'])
            group['expert_us'] += int(f['experts_us'])
        if elapsed > verify_us:
            raise ValueError(f'round duration mismatch {batch}')
        timestamp = datetime.fromisoformat(line.split()[0].replace('Z', '+00:00')).timestamp()
        workload = 'code' if timestamp <= code_end else 'mixed'
        if segments:
            workload = next(s['workload'] for s in segments if s['begin'] <= begin < s['end'])
        shape = (int(fields['lane']), rows)
        seen_shapes[shape] += 1
        records.append(dict(source=directory.name, batch=batch, gpus=gpus, width=width,
                            lane=shape[0], rows=rows, requests=int(fields['requests']),
                            workload=workload, warm=seen_shapes[shape] > 2,
                            verify_us=verify_us, groups=grouped,
                            other_us=verify_us-sum(g['expert_us'] for g in grouped.values())))
    if set(pending) != set(excluded) or any(
            set(pending[batch]) != set(entry['layers'])
            or not entry['reason'] for batch, entry in excluded.items()):
        raise ValueError(f'unfinished layer records in {path}')
    return records, trace_hash


def fit(x, y):
    """Nonnegative robust least squares; prevent a cheaper larger workload."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scale = np.maximum(np.linalg.norm(x, axis=0), 1)
    z = x / scale
    beta = np.zeros(z.shape[1])
    weights = np.ones(len(y))
    for _ in range(8):
        for _ in range(80):
            for j in range(z.shape[1]):
                denominator = np.dot(weights*z[:, j], z[:, j])
                if denominator:
                    residual = y-z@beta+z[:, j]*beta[j]
                    beta[j] = max(0, np.dot(weights*z[:, j], residual)/denominator)
        residual = abs(y-z@beta)
        threshold = max(100, float(np.median(residual))*2)
        weights = np.minimum(1, threshold/np.maximum(residual, 1))
    return beta / scale


def expert_features(record, group, variant):
    n, rows = group['layers'], record['rows']
    values = [n, n*rows, group['unique']]
    if variant == 'hinge16':
        values.append(n*max(0, rows-16))
    return values


def other_features(record, variant):
    values = [1, record['rows'], record['requests']]
    if variant == 'hinge16':
        values.append(max(0, record['rows']-16))
    return values


def error_summary(predicted, observed):
    if not observed:
        return None
    relative = (np.asarray(predicted)-observed)/observed
    return dict(rounds=len(observed), median_absolute_relative_error=float(np.median(abs(relative))),
                p90_absolute_relative_error=float(np.quantile(abs(relative), .9)),
                median_signed_relative_error=float(np.median(relative)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-profile', type=Path,
                        help='Optional experimental runtime profile; validation report remains separate')
    parser.add_argument('--variant', choices=['affine', 'hinge16'], default='affine')
    parser.add_argument('--training-workload', choices=['code', 'both'], default='both',
                        help='Code alone also holds out all mixed traffic; both trains on odd widths of each')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    if args.export_profile and (args.export_profile.exists() or args.export_profile == args.output):
        parser.error('profile must be a separate new output')
    records, hashes = [], {}
    for directory in args.directories:
        rows, hashes[directory.name] = observations(directory)
        records.extend(rows)
    warm = [r for r in records if r['warm'] and r['rows'] >= 2]
    training_workloads = {'code', 'mixed'} if args.training_workload == 'both' else {'code'}
    training = [r for r in warm if r['workload'] in training_workloads
                and r['width'] % 2 == 1]
    variants = {}
    for variant in ['affine', 'hinge16']:
        experts = {}
        for backend in sorted({b for r in training for b in r['groups']}):
            rows = [r for r in training if backend in r['groups']]
            experts[backend] = fit([expert_features(r, r['groups'][backend], variant) for r in rows],
                                   [r['groups'][backend]['expert_us'] for r in rows]).tolist()
        other = {}
        for gpus in sorted({r['gpus'] for r in training}):
            rows = [r for r in training if r['gpus'] == gpus]
            other[gpus] = fit([other_features(r, variant) for r in rows], [r['other_us'] for r in rows]).tolist()
        evaluations = {}
        for gpus in sorted(other):
            for workload in sorted({r['workload'] for r in warm}):
                for parity in [0, 1]:
                    rows = [r for r in warm if r['gpus'] == gpus and r['workload'] == workload and r['width'] % 2 == parity]
                    predicted = [float(np.dot(other[gpus], other_features(r, variant))) + sum(
                        float(np.dot(experts[b], expert_features(r, group, variant))) for b, group in r['groups'].items()) for r in rows]
                    old = [19864+803*r['rows']+636*sum(g['unique'] for g in r['groups'].values())/40 for r in rows]
                    key = f'rtx{gpus}_{workload}_{"even_holdout" if parity == 0 else "odd"}'
                    evaluations[key] = dict(training=parity == 1 and workload in training_workloads,
                                            candidate=error_summary(predicted, [r['verify_us'] for r in rows]),
                                            old=error_summary(old, [r['verify_us'] for r in rows]))
        variants[variant] = dict(experts=experts, other=other, evaluations=evaluations)
    report = dict(scope=__doc__, trace_sha256=hashes, rounds=len(records), warm_rounds=len(warm),
                  training_rounds=len(training), training_workload=args.training_workload, variants=variants,
                  limitations='Observed expert routes; no history prediction error included. Odd widths train; even widths are held out. Workload selection is explicit. Elapsed timing includes instrumentation and concurrent scheduling. Short-context corpus only; not a serving-default qualification.')
    report['excluded_non_verification_batches'] = {
        directory.name: json.loads((directory / 'excluded_non_verification_batches.json').read_text())
        for directory in args.directories if (directory / 'excluded_non_verification_batches.json').exists()}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    if args.export_profile:
        selected = variants[args.variant]
        def coefficients(values):
            return values + [0.] if len(values) == 3 else values
        profile = dict(version=1,
                       experts={k: coefficients(v) for k, v in selected['experts'].items()},
                       other={k: coefficients(v) for k, v in selected['other'].items()})
        with args.export_profile.open('x') as stream:
            json.dump(profile, stream, indent=2)
            stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
