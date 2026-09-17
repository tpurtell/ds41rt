#!/usr/bin/env python3
"""Collect exact target-model first-token IDs on fixed teacher-forced prompts.

Run against an otherwise idle target-only server built with the v5 top-1 diagnostic
trace in v41_native_serve/scheduler.rs. The collector never compares decoded text:
token IDs come from the target model's GPU argmax immediately after prefill.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
FIELDS = re.compile(r'(\w+)=(\[[^\]]*\]|[^ ]+)')
TRACE_MARKER = 'native prompt top-1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(base):
    base = base.rstrip('/')
    return base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')


def parse_top1(raw):
    rows = []
    for line in ANSI.sub('', raw.decode()).splitlines():
        if TRACE_MARKER not in line:
            continue
        fields = dict(FIELDS.findall(line))
        rows.append(dict(request_id=int(fields['request_id']),
                         prompt_tokens=int(fields['prompt_tokens']),
                         token_id=int(fields['token_id']), line=line))
    return rows


def read_complete_tail(path, start):
    with path.open('rb') as stream:
        stream.seek(start)
        raw = stream.read()
    return raw[:raw.rfind(b'\n') + 1] if b'\n' in raw else b''


def request_one(url, sample, timeout):
    body = dict(model=MODEL, messages=sample['messages'], thinking={'type':'disabled'},
                temperature=0, top_p=1, max_tokens=1, stream=False)
    encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    request = urllib.request.Request(url, data=encoded, headers={'Content-Type':'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{sample['id']}: HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    elapsed = time.perf_counter() - started
    if result.get('error'):
        raise RuntimeError(f"{sample['id']}: {result['error']}")
    choices = result.get('choices', [])
    if len(choices) != 1:
        raise RuntimeError(f"{sample['id']}: expected one choice, got {len(choices)}")
    usage = result.get('usage') or {}
    if usage.get('completion_tokens') != 1:
        raise RuntimeError(f"{sample['id']}: expected one completion token, got {usage}")
    return body, result, elapsed


def wilson(matches, samples, z=1.959963984540054):
    if samples == 0:
        return None
    p = matches / samples
    denominator = 1 + z*z/samples
    center = (p + z*z/(2*samples))/denominator
    half = z*math.sqrt(p*(1-p)/samples + z*z/(4*samples*samples))/denominator
    return [center-half, center+half]


def summary(rows):
    matches = sum(row['matches_reference'] for row in rows)
    return dict(samples=len(rows), matches=matches, agreement=matches/len(rows),
                wilson_95=wilson(matches, len(rows)),
                baseline_token_count=len({row['reference_token_id'] for row in rows}),
                candidate_token_count=len({row['token_id'] for row in rows}))


def validate_corpus(corpus):
    assert corpus['schema'] == 1 and corpus['protocol']['draft_model'] == 'disabled'
    samples = corpus['samples']
    assert len(samples) >= 100 and len({row['id'] for row in samples}) == len(samples)
    counts = Counter(row['category'] for row in samples)
    assert counts == Counter(corpus['category_counts'])
    assert min(counts.values()) >= 10
    for row in samples:
        assert set(row) == {'id','category','messages'}
        assert len(row['messages']) >= 2 and row['messages'][-1]['role'] == 'assistant'
        assert row['messages'][-1]['wo_eos'] is True and row['messages'][-1]['content']
    return samples


def load_reference(path, corpus_hash, ids):
    reference = json.loads(path.read_text())
    assert reference['passed'] and reference['corpus_sha256'] == corpus_hash
    assert reference['role'] == 'baseline' and reference['protocol']['target_only'] is True
    rows = {row['id']:row for row in reference['samples']}
    assert list(rows) == ids and all(isinstance(row['token_id'], int) for row in rows.values())
    return reference, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--trace', type=Path, required=True,
                        help='continuously captured diagnostic server log')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('fixtures')/'release-v5-top1-corpus.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--reference', type=Path,
                        help='baseline result; omit only for the full-model baseline')
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--checkpoint-revision', required=True)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--trace-timeout', type=float, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists')
    if not args.trace.is_file():
        parser.error('trace must already exist and be continuously captured')
    if not re.fullmatch(r'[0-9a-f]{64}', args.binary_sha256):
        parser.error('--binary-sha256 must be 64 lowercase hexadecimal digits')

    corpus = json.loads(args.corpus.read_text())
    samples = validate_corpus(corpus)
    corpus_hash = sha256(args.corpus)
    ids = [row['id'] for row in samples]
    reference, reference_rows = (None, None) if args.reference is None else load_reference(args.reference, corpus_hash, ids)
    report = dict(schema=1, passed=False, label=args.label,
                  role='baseline' if reference is None else 'candidate',
                  corpus=str(args.corpus), corpus_sha256=corpus_hash,
                  binary_sha256=args.binary_sha256, checkpoint_revision=args.checkpoint_revision,
                  protocol=dict(target_only=True, draft_model='disabled', concurrency=1,
                      temperature=0, top_p=1, max_tokens=1, thinking='disabled',
                      token_source='instrumented target-model scores.select(mask) after fixed-prompt prefill',
                      constrained=False, comparison='exact token ID on byte-identical OpenAI messages'),
                  reference=(None if reference is None else dict(path=str(args.reference),
                      sha256=sha256(args.reference), label=reference['label'],
                      binary_sha256=reference['binary_sha256'], checkpoint_revision=reference['checkpoint_revision'])),
                  started_ns=time.time_ns(), samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    overall_started = time.perf_counter()
    url = endpoint(args.base_url)
    last_request_id = None
    for index, sample in enumerate(samples):
        start = args.trace.stat().st_size
        body, response, seconds = request_one(url, sample, args.timeout)
        deadline = time.monotonic() + args.trace_timeout
        while True:
            raw = read_complete_tail(args.trace, start)
            records = parse_top1(raw)
            if len(records) == 1:
                break
            if len(records) > 1:
                raise RuntimeError(f"{sample['id']}: contaminated trace has {len(records)} top-1 records")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{sample['id']}: diagnostic trace record did not arrive")
            time.sleep(0.05)
        record = records[0]
        if last_request_id is not None and record['request_id'] != last_request_id + 1:
            raise RuntimeError(f"{sample['id']}: request IDs are not contiguous; server was not isolated")
        last_request_id = record['request_id']
        usage = response['usage']
        if usage['prompt_tokens'] != record['prompt_tokens']:
            raise RuntimeError(f"{sample['id']}: API/log prompt-token mismatch")
        base = reference_rows[sample['id']]['token_id'] if reference_rows else record['token_id']
        row = dict(id=sample['id'], category=sample['category'], token_id=record['token_id'],
                   reference_token_id=base, matches_reference=record['token_id'] == base,
                   request_id=record['request_id'], prompt_tokens=record['prompt_tokens'],
                   seconds=seconds, finish_reason=response['choices'][0].get('finish_reason'),
                   decoded_text=response['choices'][0].get('message',{}).get('content'),
                   request=body, usage=usage, trace_start=start, trace_end=start+len(raw),
                   trace_sha256=hashlib.sha256(raw).hexdigest(), trace_line=record['line'])
        report['samples'].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f"{index+1}/{len(samples)} {sample['id']} token={record['token_id']} "
              f"match={row['matches_reference']} {seconds:.3f}s", flush=True)
    report['elapsed_seconds'] = time.perf_counter() - overall_started
    if reference_rows:
        report['overall'] = summary(report['samples'])
        report['categories'] = {category:summary([row for row in report['samples'] if row['category']==category])
                                for category in sorted(corpus['category_counts'])}
        report['disagreements'] = [{key:row[key] for key in ('id','category','reference_token_id','token_id')}
                                   for row in report['samples'] if not row['matches_reference']]
    else:
        report['overall'] = None
        report['categories'] = None
        report['disagreements'] = []
    report['finished_ns'] = time.time_ns()
    report['passed'] = len(report['samples']) == len(samples)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
