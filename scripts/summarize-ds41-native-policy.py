#!/usr/bin/env python3
"""Summarize instrumented native draft observations; do not predict serving TPS."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ANSI = re.compile(r'\x1b\[[0-9;]*m')
FIELDS = re.compile(r'(\w+)=(\[[^\]]*\]|[^ ]+)')


def parse(text):
    observations, rounds = [], []
    for line in ANSI.sub('', text).splitlines():
        if 'native draft policy observation' in line:
            fields = dict(FIELDS.findall(line))
            confidence = json.loads(fields['raw_confidence'])
            if not confidence or not all(math.isfinite(x) for x in confidence):
                raise ValueError('invalid confidence vector')
            rows, matched = int(fields['verifier_rows']), int(fields['matched_prefix'])
            if not 0 <= matched < rows <= len(confidence) + 1:
                raise ValueError('invalid verified prefix')
            observations.append(dict(
                request_id=int(fields['request_id']), lane=int(fields['lane']),
                generated=int(fields['generated']), rows=rows,
                matched=matched, confidence=confidence,
                constrained=fields['constrained'] == 'true',
                terminal=fields['eos'] == 'true' or fields['length_limit'] == 'true'))
        elif 'native scheduler round' in line or 'native independent lane round' in line:
            fields = dict(FIELDS.findall(line))
            if 'prepared_us' in fields:
                fields['prepare_us'] = fields['prepared_us']
            rounds.append({key: int(fields[key]) for key in
                ('requests', 'proposed', 'accepted', 'draft_us', 'prepare_us', 'verify_us', 'total_us')})
    return observations, rounds


def summarize(observations, rounds):
    calibration = []
    eligible = [x for x in observations if not x['constrained'] and not x['terminal']]
    for position in range(max((len(x['confidence']) for x in observations), default=0)):
        rows = [x for x in eligible
                if x['rows'] > position + 1 and x['matched'] >= position]
        bins = defaultdict(list)
        for row in rows:
            logit = row['confidence'][position]
            probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            bins[min(9, int(probability * 10))].append((probability, row['matched'] > position))
        calibration.append(dict(position=position + 1, samples=len(rows), bins=[dict(
            lower_probability=b / 10, samples=len(v), mean_confidence=statistics.mean(x[0] for x in v),
            observed_acceptance=statistics.mean(x[1] for x in v)) for b, v in sorted(bins.items())]))
    costs = defaultdict(list)
    for row in rounds:
        costs[row['requests'], row['requests'] + row['proposed']].append(row)
    proposed = sum(x['rows'] - 1 for x in eligible)
    accepted = sum(x['matched'] for x in eligible)
    return dict(scope=__doc__, observations=len(observations), rounds=len(rounds),
                acceptance=dict(observations=len(eligible),
                    excluded_terminal_or_constrained=len(observations)-len(eligible),
                    requests=len({x['request_id'] for x in eligible}),
                    proposed_drafts=proposed, accepted_drafts=accepted,
                    accepted_fraction=accepted/proposed if proposed else None,
                    mean_accepted_drafts=accepted/len(eligible) if eligible else None,
                    mean_verified_drafts=proposed/len(eligible) if eligible else None,
                    zero_acceptance_observations=sum(x['matched'] == 0 for x in eligible),
                    zero_acceptance_fraction=(sum(x['matched'] == 0 for x in eligible)/len(eligible)
                                              if eligible else None),
                    mean_emitted_tokens=((accepted+len(eligible))/len(eligible) if eligible else None),
                    emitted_token_contract='Nonterminal unconstrained greedy verification emits the matched draft prefix plus one target-selected token.',
                    verified_width_counts=dict(sorted(Counter(
                        x['rows']-1 for x in eligible).items())),
                    interpretation='Prefix acceptance among verified drafts, excluding terminal and constrained observations. Adaptive selection censors unverified drafts; this is not unconditional draft accuracy.'),
                calibration=calibration, costs=[dict(requests=n, verifier_rows=m, samples=len(rows),
                    **{key: dict(median=statistics.median(r[key] for r in rows),
                                 minimum=min(r[key] for r in rows), maximum=max(r[key] for r in rows))
                       for key in ('draft_us', 'verify_us', 'total_us')})
                    for (n, m), rows in sorted(costs.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    raw = args.trace.read_bytes()
    observations, rounds = parse(raw.decode())
    if not observations or not rounds:
        parser.error('trace is missing confidence or scheduler records')
    report = summarize(observations, rounds)
    report['trace_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
