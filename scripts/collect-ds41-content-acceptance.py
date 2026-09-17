#!/usr/bin/env python3
"""Collect C1 adaptive draft acceptance by release content type, separately from TPS.

Run against an otherwise idle server with
RUST_LOG=info,ds41rt::draft_policy=debug,ds41rt::timing=debug,ds41rt::lane_schedule=debug
and continuously
capture its logs to --trace. This script does not launch or alter the server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time


def summarize_acceptance(observations):
    """Separate grammar-selected targets; censor terminal budget/EOS cycles."""
    result = {}
    for constrained, name in ((False, 'unconstrained'), (True, 'grammar_constrained')):
        selected = [row for row in observations if row['constrained'] == constrained]
        eligible = [row for row in selected if not row['terminal']]
        proposed = sum(row['rows'] - 1 for row in eligible)
        accepted = sum(row['matched'] for row in eligible)
        result[name] = dict(
            observations=len(selected), nonterminal_observations=len(eligible),
            excluded_terminal_observations=len(selected) - len(eligible),
            requests=len({row['request_id'] for row in selected}),
            verified_drafts=proposed, accepted_drafts=accepted,
            acceptance=accepted / proposed if proposed else None,
            mean_verified_drafts=proposed / len(eligible) if eligible else None,
            mean_emitted_tokens=(accepted + len(eligible)) / len(eligible) if eligible else None,
            zero_acceptance_cycles=sum(row['matched'] == 0 for row in eligible),
            verified_width_counts={str(width): sum(row['rows'] - 1 == width for row in eligible)
                                   for width in sorted({row['rows'] - 1 for row in eligible})})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--rtx-gpus', type=int, choices=(1, 2), required=True)
    parser.add_argument('--nonce-seed', type=int, default=71001)
    parser.add_argument('--corpus', type=Path,
                        default=Path(__file__).with_name('fixtures') / 'release-semantic-corpus.json')
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error('output directory must be new')
    if not args.trace.is_file():
        parser.error('trace must be an existing, continuously captured server log')
    corpus = json.loads(args.corpus.read_text())
    cases = corpus['weighted_case_ids']
    assert len(cases) == len(set(cases))
    here = Path(__file__).parent
    policy = runpy.run_path(str(here / 'summarize-ds41-native-policy.py'))
    args.output_directory.mkdir(parents=True)
    report = dict(model=args.model_label, rtx_gpus=args.rtx_gpus, concurrency=1,
                  repeats=3, draft_policy='adaptive', cases={}, passed=False,
                  corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                  interpretation='Conditional prefix acceptance of adaptively selected drafts. '
                  'Terminal cycles excluded; grammar-constrained targets reported separately. '
                  'Reasoning and final-answer generation are both included. '
                  'Different model outputs can change cycle counts; not teacher-forced agreement. '
                  'Instrumented timings are not release throughput.')
    def save():
        (args.output_directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for index, case in enumerate(cases):
        start = args.trace.stat().st_size
        command = [sys.executable, str(here / 'bench-ds41-release-decode.py'),
                   '--base-url', args.base_url, '--tokenizer', str(args.tokenizer),
                   '--corpus', str(args.corpus), '--label', 'adaptive-content-acceptance',
                   '--nonce-seed', str(args.nonce_seed + index * 100), '--case', case,
                   '--repeats', '3', '--output', str(args.output_directory / f'{case}.json')]
        with (args.output_directory / f'{case}-client.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        client = json.loads((args.output_directory / f'{case}.json').read_text())
        assert client['passed'] and len(client['samples']) == 3
        # Snapshot publication follows completed generation, including a last
        # target-only token that has no draft-confidence observation. Require
        # all three publications rather than assuming every final cycle drafts.
        deadline = time.monotonic() + 10
        while True:
            with args.trace.open('rb') as stream:
                stream.seek(start)
                raw = stream.read()
            raw = raw[:raw.rfind(b'\n') + 1]
            text = policy['ANSI'].sub('', raw.decode())
            observations, rounds = policy['parse'](text)
            ids = {row['request_id'] for row in observations}
            completed = {int(dict(policy['FIELDS'].findall(line))['request_id'])
                         for line in text.splitlines()
                         if 'independent snapshot published' in line}
            if len(ids) == 3 and completed == ids:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'{case}: incomplete or contaminated trace: '
                                   f'{len(ids)} observed requests, {len(completed)} published snapshots')
            time.sleep(0.1)
        (args.output_directory / f'{case}-trace.log').write_bytes(raw)
        definition = corpus['cases'][case]
        report['cases'][case] = dict(
            thinking=definition.get('thinking', 'disabled'),
            reasoning_effort=definition.get('reasoning_effort'),
            max_tokens=definition['max_tokens'],
            trace_start=start, trace_end=start + len(raw),
            trace_sha256=hashlib.sha256(raw).hexdigest(), command=command,
            acceptance=summarize_acceptance(observations))
        save()
        print('PASS', case, flush=True)
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
