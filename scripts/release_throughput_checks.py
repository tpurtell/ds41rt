"""Named output checks for throughput runs; prose quality is not auto-scored."""
from pathlib import Path
import runpy

_validate = runpy.run_path(str(Path(__file__).with_name('release_semantic_quality.py')))['validate_case_content']
CHECK_NAMES = {
    'code': 'python_structure',
    'code-reasoning': 'python_structure',
    'math': 'arithmetic_answer_and_calculation',
    'structured-json': 'json_edit_fields',
    'structured-json-schema': 'json_edit_fields',
}

def check_output(case_id, content):
    """Separate valid nonempty output from narrowly specified objective checks."""
    checks = {}
    if case_id in CHECK_NAMES:
        result = _validate('code' if case_id == 'code-reasoning' else case_id, content)
        checks[CHECK_NAMES[case_id]] = {
            'passed': result['quality_contract_passed'],
            'issues': result['quality_contract_issues'],
        }
    return {
        'response_nonempty': bool(content.strip()),
        'objective_checks': checks,
        'objective_checks_passed': all(c['passed'] for c in checks.values()) if checks else None,
        'prose_quality_assessed': False,
    }
