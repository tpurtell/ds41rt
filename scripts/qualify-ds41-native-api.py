#!/usr/bin/env python3
"""Small live native-API smoke/latency run; not a release performance gate.

The release accepts sampled requests (``temperature > 0`` with optional
``top_p``/``top_k``/``min_p``/``seed``); this script asserts one sampled request
is served and that its fixed seed replays exactly. Deeper sampling acceptance
(five vectors, invalid parameters, constrained decoding, signed seeds, request
isolation) lives in ``scripts/qualify-ds41-sampling-api.py``.
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'

class IncompleteStreamError(AssertionError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"incomplete SSE: done={record['done']}, first={record['first_content_seconds']}, finish={record['finish_seconds']}, usage={record['usage']}")

def payload(prompt, stream=False):
    return dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                thinking=dict(type='disabled'), temperature=0, max_tokens=96,
                stream=stream, **({'stream_options': {'include_usage': True}} if stream else {}))

def open_request(base, body, api_key=None):
    return urllib.request.urlopen(urllib.request.Request(base + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + api_key} if api_key else {})}), timeout=180)

def stream_case(base, body, cancel=False, api_key=None, on_first_content=None):
    start = time.perf_counter()
    events, text, first, finish, usage = [], '', None, None, None
    reasoning, first_output, finish_reason = '', None, None
    done = False
    with open_request(base, body, api_key=api_key) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            elapsed = time.perf_counter() - start
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            events.append(dict(seconds=elapsed, event=event))
            if event.get('error'):
                raise RuntimeError(f"SSE inference error: {event['error']}")
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                thought = choice.get('delta', {}).get('reasoning_content', '') or ''
                delta = choice.get('delta', {}).get('content', '') or ''
                if (thought or delta) and first_output is None:
                    first_output = elapsed
                reasoning += thought
                if delta:
                    if first is None:
                        first = elapsed
                        if on_first_content is not None:
                            on_first_content()
                    text += delta
                    if cancel:
                        return dict(cancelled_after_content=True, first_content_seconds=first, text=text)
                if choice.get('finish_reason'):
                    finish = elapsed
                    finish_reason = choice['finish_reason']
    if not (done and first_output is not None and finish is not None and usage):
        raise IncompleteStreamError(dict(done=done, text=text, reasoning=reasoning, first_content_seconds=first,
                                         first_output_seconds=first_output, finish_reason=finish_reason,
                                         finish_seconds=finish, usage=usage, events=events))
    # Completion tokens include reasoning. Time from first reasoning OR answer
    # delta so reasoning tokens never get charged only to final-answer time.
    tps = (usage['completion_tokens'] - 1) / (finish - first_output) if finish > first_output else None
    return dict(text=text, reasoning=reasoning, first_output_seconds=first_output,
                first_content_seconds=first, finish_seconds=finish, finish_reason=finish_reason,
                observed_decode_tokens_per_second=tps, usage=usage, events=events)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:18041')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    simple = payload('What is 2 + 2? Answer with just the number.')
    with open_request(args.base_url, simple) as response:
        first = json.load(response)
    assert first['choices'][0]['message']['content'].strip() == '4'
    assert first['choices'][0]['finish_reason'] == 'stop'
    long = payload('Count from 1 to 20, separated by commas. Output only the numbers.', True)
    runs = [stream_case(args.base_url, long) for _ in range(3)]
    expected = [str(i) for i in range(1, 21)]
    for run in runs:
        assert [s.strip() for s in run['text'].strip().split(',')] == expected, run['text']
    cancelled = stream_case(args.base_url, payload('Count from 1 to 1000.', True), cancel=True)
    with open_request(args.base_url, simple) as response:
        recovered = json.load(response)
    assert recovered['choices'][0]['message']['content'].strip() == '4'
    # The release serves sampled requests. Pin acceptance plus fixed-seed replay
    # on a short deterministic-looking prompt; the exact text is not asserted
    # because non-zero temperature is supposed to vary with the seed.
    sampled_payload = dict(simple, temperature=0.7, top_p=0.95, seed=0, max_tokens=32)
    with open_request(args.base_url, sampled_payload) as response:
        sampled_first = json.load(response)
    assert sampled_first['choices'][0]['message'].get('content') is not None
    with open_request(args.base_url, dict(sampled_payload)) as response:
        sampled_replay = json.load(response)
    sampled_text = (sampled_first['choices'][0]['message'].get('content') or '') + \
        (sampled_first['choices'][0]['message'].get('reasoning_content') or '')
    replay_text = (sampled_replay['choices'][0]['message'].get('content') or '') + \
        (sampled_replay['choices'][0]['message'].get('reasoning_content') or '')
    assert sampled_text, sampled_first
    assert replay_text == sampled_text, (sampled_text, replay_text)
    invalid = dict(simple, top_p=0.0)
    try:
        open_request(args.base_url, invalid)
        raise AssertionError('invalid sampling accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
    record = dict(model=MODEL, base_url=args.base_url, first_json=first,
                  streaming_request=long, streaming_runs=runs, cancellation=cancelled,
                  post_cancellation_json=recovered,
                  sampled_request=sampled_payload, sampled_json=sampled_first,
                  sampled_replay_identical=True, invalid_sampling_status=400,
                  scope='One client, greedy text, tiny prompts, three same-shape streams; first run warms. One sampled request with seed=0 is replayed once. Engine mode is identified by system_fingerprint. Not release throughput or broad quality qualification.')
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(json_content=first['choices'][0]['message']['content'],
                         stream_text=runs[-1]['text'],
                         ttft_seconds=[r['first_content_seconds'] for r in runs],
                         observed_decode_tps=[r['observed_decode_tokens_per_second'] for r in runs],
                         cancellation_recovered=True,
                         sampled_replay_identical=True), indent=2))

if __name__ == '__main__':
    main()
