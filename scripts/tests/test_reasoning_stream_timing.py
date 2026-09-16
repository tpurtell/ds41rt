import importlib.util
import io
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).parents[1] / 'qualify-ds41-native-api.py'
    spec = importlib.util.spec_from_file_location('native_api_timing', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('thinking', [True, False])
def test_completion_token_timing_includes_reasoning(thinking, monkeypatch):
    api = load()
    events = []
    ticks = [0]
    if thinking:
        events.append({'choices': [{'delta': {'reasoning_content': 'Consider the intervals.'}}]})
        ticks.append(1)
    events.extend([
        {'choices': [{'delta': {'content': 'def merge_intervals(): pass'}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'completion_tokens': 10}},
    ])
    ticks.extend([4, 5, 6, 7])
    stream = b''.join(b'data: ' + json.dumps(event).encode() + b'\n' for event in events) + b'data: [DONE]\n'
    monkeypatch.setattr(api, 'open_request', lambda *args, **kwargs: io.BytesIO(stream))
    timer = iter(ticks)
    monkeypatch.setattr(api.time, 'perf_counter', lambda: next(timer))
    result = api.stream_case('unused', {})
    assert result['first_content_seconds'] == 4
    assert result['first_output_seconds'] == (1 if thinking else 4)
    assert result['observed_decode_tokens_per_second'] == (2.25 if thinking else 9)
    assert bool(result['reasoning']) == thinking
    assert result['text'] == 'def merge_intervals(): pass'


def test_reasoning_budget_exhaustion_is_a_complete_stream_not_an_answer(monkeypatch):
    api = load()
    events = [
        {'choices': [{'delta': {'reasoning_content': 'Still considering the problem.'}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'length'}]},
        {'choices': [], 'usage': {'completion_tokens': 4096}},
    ]
    raw = b''.join(b'data: ' + json.dumps(event).encode() + b'\n' for event in events) + b'data: [DONE]\n'
    monkeypatch.setattr(api, 'open_request', lambda *args, **kwargs: io.BytesIO(raw))
    timer = iter([0, 1, 5, 6, 7])
    monkeypatch.setattr(api.time, 'perf_counter', lambda: next(timer))
    result = api.stream_case('unused', {})
    assert result['text'] == '' and result['reasoning']
    assert result['first_content_seconds'] is None
    assert result['first_output_seconds'] == 1
    assert result['finish_reason'] == 'length'
    assert result['observed_decode_tokens_per_second'] == 4095 / 4


def test_concurrent_failure_retains_reasoning_and_other_responses(tmp_path, monkeypatch):
    import runpy
    import sys
    runner = Path(__file__).parents[1] / 'bench-ds41-concurrent-api.py'
    original = runpy.run_path
    calls = iter(range(3))
    def stream(*args, **kwargs):
        index = next(calls)
        return dict(text='' if index == 1 else 'answer', reasoning=f'reasoning-{index}',
            first_content_seconds=None if index == 1 else 2, first_output_seconds=1,
            finish_seconds=3, finish_reason='length' if index == 1 else 'stop',
            observed_decode_tokens_per_second=4.5, events=[],
            usage=dict(prompt_tokens=10, prompt_cache_hit_tokens=10,completion_tokens=10,total_tokens=20))
    def modules(path):
        if str(path).endswith('qualify-ds41-native-api.py'):
            return dict(payload=lambda *args: {}, stream_case=stream)
        if str(path).endswith('release_throughput_checks.py'):
            return dict(check_output=lambda case,text: dict(response_nonempty=bool(text), objective_checks_passed=None))
        return original(path)
    monkeypatch.setattr(runpy, 'run_path', modules)
    output = tmp_path / 'failure.json'
    monkeypatch.setattr(sys, 'argv', [str(runner), '--case', 'code-reasoning', '--concurrency','2',
                                    '--repeats','1','--output',str(output)])
    with pytest.raises(SystemExit, match='all responses retained'):
        original(str(runner), run_name='__main__')
    report = json.loads(output.read_text())
    assert report['passed'] is False and report['phase'] == 'C2 repeat 1'
    assert len(report['failed_batch']) == 2
    assert report['failed_batch'][0]['result']['reasoning'] == 'reasoning-1'
    assert report['failed_batch'][0]['result']['finish_reason'] == 'length'
    assert report['failed_batch'][1]['passed'] is True
