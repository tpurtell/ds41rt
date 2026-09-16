from pathlib import Path
import runpy

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'release_throughput_checks.py'))['check_output']

def test_prose_is_unscored_and_empty_output_still_detected():
    for case in ('fable', 'topic', 'hello', 'multilingual'):
        result = CHECK(case, 'Pages map to physical frames through a page table.')
        assert result['response_nonempty']
        assert result['objective_checks_passed'] is None
        assert result['objective_checks'] == {}
        assert not result['prose_quality_assessed']
    assert not CHECK('fable', ' \n')['response_nonempty']

def test_objective_errors_are_not_hidden_by_serving_success():
    assert CHECK('math', '240 * 0.75 * 1.08 = 194.40')['objective_checks_passed']
    assert not CHECK('math', '240 * 0.75 * 1.08 = 195.40')['objective_checks_passed']
    assert not CHECK('structured-json-schema', '{"incorrect": true}')['objective_checks_passed']
    assert not CHECK('code', '```python\ndef merge_intervals(:\n```')['objective_checks_passed']
    assert CHECK('code-reasoning', '```python\ndef merge_intervals(:\n```') == CHECK(
        'code', '```python\ndef merge_intervals(:\n```')
