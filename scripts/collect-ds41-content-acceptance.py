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


def completion_markers(observations, text, rtx_gpus, fields):
    """Return the scheduler events that prove the captured case trace settled."""
    if rtx_gpus == 2:
        return ({int(dict(fields.findall(line))['request_id'])
                 for line in text.splitlines()
                 if 'independent snapshot published' in line},
                'published snapshots')
    # The synchronous scheduler does not publish an independent snapshot event,
    # and a request may finish on a target-only cycle with no terminal draft
    # observation. The caller has already received all three complete client
    # responses, so request IDs in a quiescent trace are the correct boundary.
    return ({row['request_id'] for row in observations}, 'settled request traces')


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
        # Dual-RTX independent lanes publish retained snapshots asynchronously,
        # so their publication log is the strongest trace-settlement marker.
        # The single-RTX scheduler retains synchronously and has no independent
        # publication event; its terminal policy observation is the last event
        # needed for acceptance accounting. In both cases the client has already
        # completed all three requests before this loop begins.
        deadline = time.monotonic() + 10
        previous_size = -1
        stable_polls = 0
        while True:
            with args.trace.open('rb') as stream:
                stream.seek(start)
                raw = stream.read()
            raw = raw[:raw.rfind(b'\n') + 1]
            if len(raw) == previous_size:
                stable_polls += 1
            else:
                previous_size = len(raw)
                stable_polls = 0
            text = policy['ANSI'].sub('', raw.decode())
            observations, rounds = policy['parse'](text)
            ids = {row['request_id'] for row in observations}
            completed, marker = completion_markers(
                observations, text, args.rtx_gpus, policy['FIELDS'])
            if args.rtx_gpus == 1 and stable_polls < 5:
                completed = set()
            if len(ids) == 3 and completed == ids:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'{case}: incomplete or contaminated trace: '
                                   f'{len(ids)} observed requests, {len(completed)} {marker}')
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
