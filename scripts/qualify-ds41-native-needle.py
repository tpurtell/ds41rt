#!/usr/bin/env python3
"""Retrieve deterministic needles at configured context lengths and prove exact reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from tokenizers import Tokenizer


MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'


def stream(base, body, timeout):
    started = time.perf_counter()
    request = urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    text = ''
    reasoning = ''
    usage = None
    fingerprint = None
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            fingerprint = event.get('system_fingerprint') or fingerprint
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text += delta.get('content') or ''
                reasoning += delta.get('reasoning_content') or ''
    assert done and usage, 'incomplete stream'
    return dict(text=text, reasoning_content=reasoning, usage=usage,
                system_fingerprint=fingerprint,
                elapsed_seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', action='append', required=True,
                        help='MODE=URL; repeat for target and dspark')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--contexts', type=int, nargs='+',
                        default=[32768, 131072, 524288, 1040000])
    parser.add_argument('--positions', type=float, nargs='+',
                        default=[0.1, 0.5, 0.9, 0.5])
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if len(args.contexts) != len(args.positions):
        parser.error('contexts and positions must have the same length')
    if max(args.contexts) > 1040000 or min(args.contexts) < 1024:
        parser.error('contexts must be in 1024..1040000')
    if any(not 0 < p < 1 for p in args.positions):
        parser.error('positions must be between zero and one')
    urls = dict(item.split('=', 1) for item in args.base_url)
    if not urls or set(urls) - {'target', 'dspark'}:
        parser.error('provide target=URL, dspark=URL, or both')

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    source = args.source.read_text()
    text = source
    while len(tokenizer.encode(text, add_special_tokens=False).ids) < max(args.contexts):
        text += '\n' + text
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    report = dict(scope=__doc__, model=MODEL, urls=urls,
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), cases=[], passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    for index, (context, position) in enumerate(zip(args.contexts, args.positions)):
        code = 'DS41-' + hashlib.sha256(f'{context}:{position}'.encode()).hexdigest()[:16].upper()
        discriminator = hashlib.sha256(
            f'{args.output.name}:{index}:{context}:{position}'.encode()).hexdigest()
        cut = round(context * position)
        prompt = (f'Qualification discriminator {discriminator}.\n\n'
            + tokenizer.decode(ids[:cut], skip_special_tokens=False)
            + f'\n\nThe unique retrieval code is {code}. Memorize it exactly.\n\n'
            + tokenizer.decode(ids[cut:context], skip_special_tokens=False)
            + '\n\nWhat is the unique retrieval code? Return only the code.')
        body = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
            thinking=dict(type='enabled'), reasoning_effort='high', temperature=0,
            max_tokens=128, stream=True, stream_options=dict(include_usage=True))
        case = dict(context_source_tokens=context, needle_position=position, needle=code,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), runs={})
        report['cases'].append(case)
        save()
        for mode, base in urls.items():
            cold = stream(base, body, args.timeout)
            exact = stream(base, body, args.timeout)
            case['runs'][mode] = dict(cold=cold, exact=exact)
            save()
            for name, run in [('cold', cold), ('exact', exact)]:
                answer = run['text'].strip()
                run['retrieved'] = answer == code
                assert run['retrieved'], (mode, context, name, answer, code)
                assert run['system_fingerprint'] == f'ds41rt-native-fp4-kv{"-dspark" if mode == "dspark" else ""}'
            assert cold['usage']['prompt_tokens_details']['cached_tokens'] <= 32
            assert exact['usage']['prompt_tokens_details']['cached_tokens'] == exact['usage']['prompt_tokens']
            save()
        case['passed'] = True
        print('PASS', context, position, code, flush=True)
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
