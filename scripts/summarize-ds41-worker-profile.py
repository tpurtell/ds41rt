#!/usr/bin/env python3
"""Summarize opt-in Spark timings within conservatively trimmed workload windows."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import statistics

ANSI = re.compile(r'\x1b\[[0-9;]*m')
FIELDS = re.compile(r'\b(\w+)=([0-9]+(?:\.[0-9]+)?)')
METRICS = ('rows', 'distinct_experts', 'upload_us', 'enqueue_us', 'wait_us', 'gpu_us', 'total_us')


def timestamp(line):
    return datetime.fromisoformat(line.split()[0].replace('Z', '+00:00')).timestamp()


def summarize(rows):
    return dict(count=len(rows), metrics={key: dict(
        mean=statistics.mean(r[key] for r in rows),
        median=statistics.median(r[key] for r in rows)) for key in METRICS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trim-seconds', type=float, default=1.)
    args = parser.parse_args()
    assert args.trim_seconds > 0
    directory = args.directory
    clocks = json.loads((directory / 'clock-bounds.json').read_text())
    # Bounds measured after the run, not a historical clock synchronization proof.
    assert all(max(abs(s[k]) for k in ('offset_low_ms', 'offset_high_ms'))
               < args.trim_seconds * 1000 for samples in clocks.values() for s in samples)
    host = (directory / 'serve.log').read_bytes()
    workers = []
    for rank in range(4):
        entries = []
        for raw in (directory / f'rank{rank}.log').read_text().splitlines():
            line = ANSI.sub('', raw)
            if 'EXL3 worker execution ' not in line:
                continue
            fields = {k: float(v) for k, v in FIELDS.findall(line)}
            assert fields['executor_id'] == rank + 1
            assert all(fields[key] >= 0 for key in METRICS)
            entries.append((timestamp(line), fields))
        assert entries, f'no timings for rank {rank}'
        workers.append(entries)
    results = []
    for window in json.loads((directory / 'windows.json').read_text()):
        lines = ANSI.sub('', host[window['start_byte']:window['end_byte']].decode()).splitlines()
        times = [timestamp(line) for line in lines if re.match(r'^\d{4}-', line)]
        start, end = min(times) + args.trim_seconds, max(times) - args.trim_seconds
        assert start < end
        ranks = []
        for rank, entries in enumerate(workers):
            selected = [r for t, r in entries if start <= t <= end and r['rows'] <= 80]
            assert selected
            groups = defaultdict(list)
            for row in selected:
                capacity = next(c for c in (1, 16, 80) if row['rows'] <= c)
                groups[capacity].append(row)
            ranks.append(dict(rank=rank, intermediate_width=640 if rank < 2 else 512,
                              **summarize(selected),
                              by_capacity={c: summarize(rows) for c, rows in groups.items()}))
        results.append(dict(case=window['case'], concurrency=window['concurrency'], ranks=ranks))
    report = dict(scope=__doc__, trim_seconds=args.trim_seconds, clock_bounds=clocks,
        caveats=[
            'Only rows <= 80; includes warmup and any small prefill executions in these windows.',
            'Window assignment uses host/worker wall clocks; post-run SSH bounds fit the trim, but do not prove historical clock stability.',
            'GPU time is a CUDA event interval including wire decode, routing, expert computation, output reduction and possible submission gaps; not isolated kernel busy time.',
            'GPU time overlaps enqueue and wait; these columns must not be added together.',
            'Worker timing begins after connection polling, so excludes time waiting behind another connection.',
            'Rank averages are not request-matched maximums and cannot be subtracted from coordinator stage averages to measure transport latency.',
            'Tracing perturbs timing; these are optimization diagnostics, not release benchmarks.'
        ], results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for result in results:
        print(result['case'], result['concurrency'], [
            {k: round(rank['metrics'][k]['mean'], 2) for k in ('gpu_us', 'total_us')}
            for rank in result['ranks']])


if __name__ == '__main__':
    main()
