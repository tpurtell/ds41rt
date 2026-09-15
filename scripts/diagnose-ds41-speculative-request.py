#!/usr/bin/env python3
"""Replay a saved request and summarize independent-lane debug timing.

Use an otherwise idle server with RUST_LOG=info,ds41rt::timing=debug.
Timing includes logging overhead; this is an acceptance diagnostic, not a TPS
qualification. The saved request is replayed without changing its prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    source = args.report.read_bytes()
    samples = [s for s in json.loads(source)['samples'] if s['case'] == args.case]
    if len(samples) != 1:
        parser.error('source must contain exactly one sample for the selected case')
    request = samples[0]['request']
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    begin = args.server_log.stat().st_size
    result = api['stream_case'](args.base_url, request)
    result.pop('events', None)
    # The final round log follows stream delivery; wait for the idle log to settle.
    for _ in range(20):
        size = args.server_log.stat().st_size
        time.sleep(.05)
        if args.server_log.stat().st_size == size:
            break
    with args.server_log.open('rb') as log:
        log.seek(begin)
        raw = log.read()
    rounds = []
    for line in raw.decode().splitlines():
        if 'native independent lane round' in line:
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            rounds.append({k: int(v) for k, v in re.findall(r'\b(\w+)=(\d+)\b', clean)})
    if not rounds or any(r.get('requests') != 1 for r in rounds):
        raise RuntimeError('expected isolated C1 independent-lane timing rounds')
    keys = ('proposed', 'accepted', 'emitted', 'draft_us', 'prepared_us', 'verify_us', 'total_us')
    if any(any(k not in r for k in keys) for r in rounds):
        raise RuntimeError('incomplete timing fields; use a daemon with draft_us logging')
    totals = {k: sum(r[k] for r in rounds) for k in keys}
    report = dict(scope=__doc__, case=args.case, request=request, result=result,
                  reference_sha256=hashlib.sha256(source).hexdigest(),
                  log_begin=begin, log_end=begin+len(raw),
                  log_sha256=hashlib.sha256(raw).hexdigest(), rounds=rounds,
                  totals=totals, emitted_per_round=totals['emitted']/len(rounds),
                  accepted_fraction=totals['accepted']/max(1, totals['proposed']),
                  median_us={k: statistics.median(r[k] for r in rounds)
                             for k in keys if k.endswith('_us')})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('case', 'emitted_per_round', 'accepted_fraction', 'median_us')}))


if __name__ == '__main__':
    main()
