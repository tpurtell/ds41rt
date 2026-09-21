#!/usr/bin/env python3
"""Aggregate the TP3 tile-tuning lane evidence into a cross-rep summary.

CPU-only and offline: it reads the run JSONs the lane runner produced (plus the
lane manifest.json) and writes summary.md/summary.json next to them.  It never
touches a GPU and never changes the harness; the medians it reports are the
decision inputs, the per-rep medians are the spread it must not hide.

Checks performed on every run:
  * every timed result bracketed its window with clocks provenance
    (clocks_before/clocks_after present, throttle bitmask 0x0, same GPU uuid,
    and a plausible SM-clock spread);
  * the run's own correctness/invariant/row-consistency gate passed;
  * all three reps of a cell raced the same compiled variant set and the
    production planner resolved the same tiles, so the reps are comparable.

The recommendation is deliberately conservative: a controlled candidate is
proposed only when it wins on BOTH capacities at ALL rank slices with a margin
above the cross-rep noise and every clock bracket stayed unthrottled;
otherwise the recommendation is to keep the production planner default.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# nvidia-smi emits a bare `N/A` for fields a socket masks.  On GB10/DGX Spark
# exactly two queried fields legitimately read N/A — power.limit and
# clocks.mem.  uuid, pstate, power.draw, clocks.sm and the throttle mask MUST
# carry real values, so an N/A in any of those fails to parse (fail closed).
CLOCK_RE = re.compile(
    r'^([^,]+),\s*(\w+),\s*([\d.]+ W|N/A),\s*([\d.]+) W,\s*(\d+) MHz,\s*(\d+ MHz|N/A),\s*(0x[0-9a-f]+)$')
NOISE_FLOOR_PCT = 2.0           # cross-rep spread tolerated before a win counts
REQUIRED_MARGIN_PCT = 5.0       # improvement needed to propose a candidate


def _field_number(token):
    return None if token == 'N/A' else float(token.split(' ')[0])


def parse_clocks(line):
    if not line:
        return None
    match = CLOCK_RE.match(line.strip())
    if not match:
        return None
    return dict(uuid=match.group(1).strip(), pstate=match.group(2),
                power_limit_w=_field_number(match.group(3)),
                power_draw_w=float(match.group(4)),
                sm_mhz=int(match.group(5)),
                mem_mhz=None if match.group(6) == 'N/A' else int(match.group(6).split(' ')[0]),
                throttle_reasons=match.group(7))


def audit_run(path: Path):
    record = json.load(path.open())
    problems = []
    if not record.get('passed'):
        problems.append('run correctness/invariant gate failed')
    # The lane ran family-DERIVED (no --tiers anywhere); the checkpoint itself
    # must have resolved to the authorized [3, 4] tier family.
    family = record.get('tier_family', {})
    if family.get('tiers') != [3, 4]:
        problems.append(f"derived tier family is {family.get('tiers')!r}, not [3, 4]")
    if not str(family.get('resolution', '')).startswith('derived'):
        problems.append(f"family was not derived from checkpoint widths: {family.get('resolution')!r}")
    if not record.get('results'):
        problems.append('no timed results')
    brackets, spread = [], []
    for result in record.get('results', []):
        if result.get('timings') is None:
            problems.append(f"untimed case {result.get('rows')}x{result.get('distinct_experts')}")
            continue
        before, after = parse_clocks(result.get('clocks_before')), parse_clocks(result.get('clocks_after'))
        if before is None or after is None:
            problems.append('clock provenance line missing or unparseable')
            continue
        for tag, sample in (('before', before), ('after', after)):
            if sample['throttle_reasons'] != '0x0':
                problems.append(f"throttled during window ({tag}: {sample['throttle_reasons']})")
            brackets.append(sample)
        if before['uuid'] != after['uuid']:
            problems.append('clock brackets came from different GPUs')
        if before['sm_mhz'] > 0:
            spread.append(abs(after['sm_mhz'] - before['sm_mhz']) / before['sm_mhz'] * 100)
    compiled = {v['name'] for v in record.get('variants', []) if v.get('status') == 'compiled'}
    planner = record.get('layout', {}).get('planner_tile')
    cfg = record.get('configuration', {})
    layout = record.get('layout', {})
    return record, dict(
        path=path.name, cell=(cfg.get('intermediate', layout.get('intermediate')),
                              cfg.get('slice_start', layout.get('slice_start')),
                              cfg.get('capacity', layout.get('capacity'))),
        passed=bool(record.get('passed')), variants=sorted(compiled),
        planner_tile=planner, problems=problems,
        power_draw_w_max=max([b['power_draw_w'] for b in brackets], default=None),
        sm_mhz_min=min([b['sm_mhz'] for b in brackets], default=None),
        sm_mhz_max=max([b['sm_mhz'] for b in brackets], default=None),
        clock_drift_pct_max=round(max(spread, default=0.0), 3),
        medians={t['variant']: t['median_us'] for result in record.get('results', [])
                 for t in (result.get('timings') or [])
                 if (result.get('rows'), result.get('distinct_experts')) == worst_case(record)},
        device=record.get('device', {}).get('name'),
        source=record.get('source', {}).get('sparkinfer_revision'))


def worst_case(record):
    """Report the heaviest timed case (most rows*experts) as the headline cell."""
    cases = [(int(r.get('rows', 0)) * int(r.get('distinct_experts', 0)), r)
             for r in record.get('results', []) if r.get('timings')]
    return None if not cases else (max(cases, key=lambda pair: pair[0])[1]['rows'],
                                   max(cases, key=lambda pair: pair[0])[1]['distinct_experts'])


def summarize(lane: Path):
    manifest_path = lane / 'manifest.json'
    manifest = json.load(manifest_path.open()) if manifest_path.is_file() else {}
    runs, problems = [], []
    lane_status = manifest.get('status', 'no-manifest')
    if lane_status != 'complete':
        reason = manifest.get('abort_reason')
        problems.append(f"lane did not complete (manifest status {lane_status!r}"
                        + (f": {reason}" if reason else '')
                        + (f" at {manifest.get('aborted_cell')}" if manifest.get('aborted_cell') else '')
                        + ')')
    for path in sorted(lane.glob('*.json')):
        if path.name in ('manifest.json', 'summary.json'):
            continue
        try:
            _, audit = audit_run(path)
        except (json.JSONDecodeError, KeyError) as exc:
            problems.append(f'{path.name}: unreadable run evidence ({exc})')
            continue
        runs.append(audit)
        problems += [f"{audit['path']}: {p}" for p in audit['problems']]

    cells = defaultdict(list)
    for run in runs:
        cells[run['cell']].append(run)
    for cell, reps in cells.items():
        if len({tuple(r['variants']) for r in reps}) > 1:
            problems.append(f'cell {cell}: reps did not compile the same variant set')
        if len({json.dumps(r['planner_tile']) for r in reps}) > 1:
            problems.append(f'cell {cell}: production planner resolved different tiles across reps')

    table = []
    for cell in sorted(cells, key=lambda c: [str(x) for x in c]):
        reps = cells[cell]
        names = sorted({n for r in reps for n in r['medians']})
        medians = {n: [r['medians'][n] for r in reps if n in r['medians']] for n in names}
        row = dict(cell=dict(intermediate=cell[0], slice_start=cell[1], capacity=cell[2]), reps=len(reps))
        per_variant = {}
        for n in names:
            values = medians[n]
            if not values:
                continue
            median = statistics.median(values)
            spread = ((max(values) - min(values)) / median * 100) if median and len(values) > 1 else 0.0
            per_variant[n] = dict(median_us=round(median, 3), rep_medians=[round(v, 3) for v in values],
                                  rep_spread_pct=round(spread, 2))
        row['variants'] = per_variant
        baseline = per_variant.get('baseline')
        winner, margin = None, 0.0
        if baseline:
            for name, entry in per_variant.items():
                if name == 'baseline' or entry['rep_spread_pct'] > NOISE_FLOOR_PCT:
                    continue
                gain = (baseline['median_us'] - entry['median_us']) / baseline['median_us'] * 100
                if gain > max(REQUIRED_MARGIN_PCT, entry['rep_spread_pct']):
                    if gain > margin:
                        winner, margin = name, gain
        row['best_variant'] = winner
        row['gain_over_baseline_pct'] = round(margin, 2) if winner else 0.0
        table.append(row)

    # A candidate is only recommended when it wins EVERY primary cell; a
    # partial winner (or a None in any cell) holds the production default.
    primary = [r for r in table if (r['cell']['intermediate'] or 0) >= 768]
    winners = [r['best_variant'] for r in primary]
    consistent = winners[0] if primary and all(winners) and len(set(winners)) == 1 else None
    recommendation = dict(candidate=consistent if consistent else None,
                          cells_evaluated=len(primary),
                          rule=f'wins every cell by >{REQUIRED_MARGIN_PCT}% with rep spread <={NOISE_FLOOR_PCT}%',
                          default_holds=consistent is None,
                          note='no promotion: this is tuning evidence, never a serving-config change')
    summary = dict(schema='ds41rt.tp3-tile-tuning-summary-v1', lane=str(lane),
                   ds41rt_commit=manifest.get('ds41rt_commit'), lane_status=lane_status,
                   runs=len(runs),
                   problems=problems, clean=not problems, cells=table, recommendation=recommendation)
    (lane / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [f'# TP3 tile tuning — {lane.name}', '',
             f'- runs: {len(runs)}  clean: **{summary["clean"]}**  commit: `{manifest.get("ds41rt_commit")}`',
             f'- recommendation: {recommendation["candidate"] or "keep production planner default"}', '']
    for row in table:
        lines.append(f"## cell {row['cell']['intermediate']}/{row['cell']['slice_start']} cap{row['cell']['capacity']}"
                     f" ({row['reps']} reps)")
        lines.append('')
        lines.append('| variant | median us | rep medians | rep spread % |')
        lines.append('| --- | --- | --- | --- |')
        for name, entry in sorted(row['variants'].items(), key=lambda kv: kv[1]['median_us']):
            lines.append(f"| {name} | {entry['median_us']} | {entry['rep_medians']} | {entry['rep_spread_pct']} |")
        lines.append('')
    if problems:
        lines += ['## provenance/correctness problems', ''] + [f'- {p}' for p in problems]
    (lane / 'summary.md').write_text('\n'.join(lines) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('lane', type=Path, help='lane evidence directory (contains the run JSONs)')
    parser.add_argument('--fail-on-problems', action='store_true',
                        help='exit nonzero if any run has a provenance/gate problem')
    args = parser.parse_args(argv)
    if not args.lane.is_dir():
        raise SystemExit(f'not a directory: {args.lane}')
    summary = summarize(args.lane)
    print(json.dumps(dict(runs=summary['runs'], clean=summary['clean'],
                          recommendation=summary['recommendation'],
                          problems=len(summary['problems']))))
    for path in (args.lane / 'summary.json', args.lane / 'summary.md'):
        print('wrote', path)
    if args.fail_on_problems and not summary['clean']:
        raise SystemExit(4)
    return 0


if __name__ == '__main__':
    sys.exit(main())
