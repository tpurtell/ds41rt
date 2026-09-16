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
