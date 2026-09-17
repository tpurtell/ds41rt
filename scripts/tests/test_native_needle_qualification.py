import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'qualify-ds41-native-needle.py'
SPEC = importlib.util.spec_from_file_location('native_needle', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CODE = 'DS41-C3BB1109A6BFD569'


def run(text='', reasoning=''):
    return {
        'text': text,
        'reasoning_content': reasoning,
        'usage': {'prompt_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 0}},
        'system_fingerprint': 'ds41rt-native-fp4-kv-dspark',
    }


def test_retrieval_field_accepts_exact_output_or_standalone_reasoning_code():
    assert MODULE.retrieval_field(run(text=f'  {CODE}\n'), CODE) == 'text'
    assert MODULE.retrieval_field(
        run(reasoning=f'The requested code is `{CODE}`.'), CODE
    ) == 'reasoning_content'


def test_retrieval_field_rejects_partial_embedded_and_conflicting_output():
    assert MODULE.retrieval_field(run(reasoning=CODE[:-1]), CODE) is None
    assert MODULE.retrieval_field(run(reasoning=f'X{CODE}Y'), CODE) is None
    assert MODULE.retrieval_field(
        run(text='wrong', reasoning=f'The code is {CODE}.'), CODE
    ) is None


def test_validate_report_marks_reasoning_retrieval_and_exact_cache_passed():
    cold = run(reasoning=f'Found {CODE}.')
    exact = run(reasoning=f'Found {CODE}.')
    exact['usage']['prompt_tokens_details']['cached_tokens'] = 100
    report = {
        'passed': False,
        'cases': [{
            'context_source_tokens': 1_040_000,
            'needle': CODE,
            'passed': False,
            'runs': {'dspark': {'cold': cold, 'exact': exact}},
        }],
    }
    MODULE.validate_report(report)
    assert report['passed'] is True
    assert report['cases'][0]['passed'] is True
    assert cold['retrieval_field'] == 'reasoning_content'
    assert exact['retrieval_field'] == 'reasoning_content'
