#!/usr/bin/env python3
"""Run the nine-category and counting release decode workloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import time

from tokenizers import Tokenizer


def token_zero_nonces(count, seed, tokenizer):
    start, total = seed % (0x9FFF - 0x4E00 + 1), 0x9FFF - 0x4E00 + 1
    result, seen = [], set()
    for offset in range(total):
        marker = chr(0x4E00 + ((start + offset) % total))
        prefix = f"{marker} request nonce {seed}-{len(result)}.\n"
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        marker_ids = tokenizer.encode(marker, add_special_tokens=False).ids
        if not encoded or len(marker_ids) != 1 or encoded[0] != marker_ids[0] or encoded[0] in seen:
            continue
        seen.add(encoded[0])
        result.append(dict(prefix=prefix, marker=marker, first_content_token_id=encoded[0]))
        if len(result) == count:
            return result
    raise RuntimeError(f"tokenizer exposed only {len(result)} unique token-zero nonces; need {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-V4.1-Flash')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--corpus', type=Path,
                        default=Path(__file__).with_name('fixtures') / 'release-semantic-corpus.json')
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--nonce-seed', type=int, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--include-orchid', action='store_true', help='Historical diagnostic only')
    parser.add_argument('--include-counting', action='store_true')
    parser.add_argument('--counting-only', action='store_true')
    parser.add_argument('--api-key-env', help='Environment variable containing the API key; never saved')
    parser.add_argument('--remote-reference', action='store_true', help='Record provider cache counters without enforcing local cache policy')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if args.repeats < 1:
        parser.error('repeats must be positive')
    corpus = json.loads(args.corpus.read_text())
    selected = [] if args.counting_only else (args.case or corpus['weighted_case_ids'])
    if args.counting_only:
        args.include_counting = True
    unknown = set(selected) - set(corpus['weighted_case_ids'])
    if unknown:
        parser.error(f"unknown or non-weighted cases: {sorted(unknown)}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    nonces = token_zero_nonces(args.repeats * (len(selected) + int(args.include_orchid) + int(args.include_counting)),
                               args.nonce_seed, tokenizer)
    quality = runpy.run_path(str(Path(__file__).with_name('release_throughput_checks.py')))
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    report = dict(
        scope=__doc__, label=args.label, base_url=args.base_url, model=args.model,
        corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        repeats=args.repeats, nonce_seed=args.nonce_seed, selected_cases=selected,
        include_orchid=args.include_orchid, include_counting=args.include_counting,
        remote_reference=args.remote_reference, controls=dict(temperature=0, thinking='per-case; default disabled'),
        samples=[], repeat_summaries=[], passed=False,
    )

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        parser.error('API key environment variable is empty')
    nonce_index = 0
    for repeat in range(args.repeats):
        repeat_samples = []
        for case_id in selected + (['orchid'] if args.include_orchid else []) + (['counting'] if args.include_counting else []):
            nonce = nonces[nonce_index]
            nonce_index += 1
            if case_id == 'counting':
                definition = corpus['counting']
                prompt = definition['prompt']
                max_tokens, weight, response_format = definition['max_tokens'], 0.0, None
            elif case_id == 'orchid':
                definition = corpus['orchid']
                prompt = definition['prompt_template'].format(nonce=nonce['prefix'].strip())
                max_tokens, weight = definition['max_tokens'], 0.0
                response_format = None
            else:
                definition = corpus['cases'][case_id]
                prompt = nonce['prefix'] + definition['prompt']
                max_tokens, weight = definition['max_tokens'], definition['weight']
                response_format = None
                if definition['json_schema']:
                    response_format = dict(type='json_schema', json_schema=dict(
                        name='file_edit', strict=True, schema=corpus['structured_edit_schema']))
            body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                        thinking=dict(type=definition.get('thinking', 'disabled')), temperature=0, max_tokens=max_tokens,
                        stream=True, stream_options=dict(include_usage=True))
            if definition.get('reasoning_effort'):
                body['reasoning_effort'] = definition['reasoning_effort']
            if response_format:
                body['response_format'] = response_format
            sample = dict(repeat=repeat + 1, case=case_id, category=definition.get('category', 'low-entropy'),
                          weight=weight, nonce=None if case_id == 'counting' else nonce, request=body, started_ns=time.time_ns())
            report['samples'].append(sample)
            save()
            try:
                result = api['stream_case'](args.base_url, body, api_key=api_key)
                content = result.pop('text')
                events = result.pop('events')
                finish_event = next(event['event'] for event in reversed(events)
                                    if any(c.get('finish_reason') for c in event['event'].get('choices', [])))
                sample.update(result, content=content,
                              content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                              finish_reason=finish_event['choices'][0]['finish_reason'],
                              system_fingerprint=finish_event.get('system_fingerprint'),
                              cached_tokens=result['usage'].get('prompt_cache_hit_tokens',
                                  result['usage'].get('prompt_tokens_details', {}).get('cached_tokens')))
                if case_id == 'counting':
                    expected = [str(n) for n in range(1, definition['count_to'] + 1)]
                    sample['quality_contract_passed'] = [x.strip() for x in content.strip().split(',')] == expected
                    sample['quality_contract_issues'] = [] if sample['quality_contract_passed'] else ['incorrect counting sequence']
                elif case_id == 'orchid':
                    words = content.split()
                    sample['quality_contract_passed'] = words == ['orchid'] * definition['requested_repetitions']
                    sample['quality_contract_issues'] = ([] if sample['quality_contract_passed'] else
                        [f"expected {definition['requested_repetitions']} exact orchid words, observed {len(words)} tokens"])
                else:
                    sample.update(quality['check_output'](case_id, content))
                # The first user-content token is unique. A small hit may still
                # cover the invariant chat-template prefix before user content.
                sample['bounded_static_prefix_hit'] = (isinstance(sample['cached_tokens'], int) and 0 <= sample['cached_tokens'] <= 32)
                sample['reasoning_present'] = bool(sample.get('reasoning', '').strip())
                sample['serving_completed'] = bool(content.strip())
                sample['passed'] = (sample['serving_completed']
                    and (definition.get('thinking') != 'enabled' or sample['reasoning_present'])
                    and sample.get('quality_contract_passed', True)
                    and sample.get('objective_checks_passed') is not False
                    and (args.remote_reference or case_id == 'counting' or sample['bounded_static_prefix_hit']))
            except Exception as error:
                sample.update(error=repr(error), passed=False)
            save()
            repeat_samples.append(sample)
            print(args.label, repeat + 1, case_id, 'PASS' if sample['passed'] else 'FAIL', flush=True)
        weighted = [sample for sample in repeat_samples if sample['weight'] > 0 and 'finish_seconds' in sample]
        timed_tokens = sum(sample['weight'] * (sample['usage']['completion_tokens'] - 1) for sample in weighted)
        timed_seconds = sum(sample['weight'] *
            (sample['finish_seconds'] - sample['first_output_seconds']) for sample in weighted)
        summary = dict(repeat=repeat + 1, weighted_cases=len(weighted),
                       serving_completed=sum(sample.get('serving_completed', False) for sample in weighted),
                       objective_checks_passed=sum(sample.get('objective_checks_passed') is True for sample in weighted),
                       objective_checks_assessed=sum(sample.get('objective_checks_passed') is not None for sample in weighted),
                       weighted_observed_decode_tokens_per_second=(timed_tokens / timed_seconds
                           if timed_seconds > 0 and len(weighted) == len(selected) else None),
                       partial_weighted_observed_decode_tokens_per_second=(timed_tokens / timed_seconds
                           if timed_seconds > 0 else None),
                       complete_weighted_corpus=len(weighted) == len(selected) and bool(selected),
                       all_samples_passed=all(sample['passed'] for sample in repeat_samples))
        report['repeat_summaries'].append(summary)
        save()
    values = [item['weighted_observed_decode_tokens_per_second'] for item in report['repeat_summaries']
              if item['weighted_observed_decode_tokens_per_second'] is not None]
    report['median_weighted_observed_decode_tokens_per_second'] = statistics.median(values) if values else None
    report['passed'] = all(sample['passed'] for sample in report['samples'])
    save()
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
