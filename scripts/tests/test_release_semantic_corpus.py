import ast
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]


def test_preserved_release_corpus_and_quality_contract():
    corpus_path = ROOT / 'scripts/fixtures/release-semantic-corpus.json'
    corpus = json.loads(corpus_path.read_text())
    assert len(corpus['weighted_case_ids']) == 9
    assert sum(corpus['cases'][name]['weight'] for name in corpus['weighted_case_ids']) == 8
    reasoning = corpus['cases']['code-reasoning']
    assert reasoning['prompt'] == corpus['cases']['code']['prompt']
    assert reasoning['thinking'] == 'enabled' and reasoning['reasoning_effort'] == 'high'
    assert corpus['orchid']['requested_repetitions'] == 100

    source_root = ROOT.parent / 'glmrt-release'
    assert corpus['source_repository'] == 'glmrt-release'
    for relative, digest in corpus['source_files'].items():
        assert hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == digest

    original = ast.parse((source_root / 'python/tools/bench_real_full_mtp_acceptance.py').read_text())
    preserved = ast.parse((ROOT / 'scripts/release_semantic_quality.py').read_text())
    original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
    for node in preserved.body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == ast.dump(original_functions[node.name])

    validate = runpy.run_path(str(ROOT / 'scripts/release_semantic_quality.py'))['validate_case_content']
    assert validate('math', '240 × 0.75 × 1.08 = 194.40')['quality_contract_passed']
    assert not validate('math', '240 × 0.75 × 1.08 = 194.00')['quality_contract_passed']
    valid = json.dumps(dict(path='src/cache.rs', operation='replace', line_start=41,
                            line_end=47, rationale='Remove a redundant copy.'))
    assert validate('structured-json-schema', valid)['quality_contract_passed']
    assert not validate('structured-json-schema', f'```json\n{valid}\n```')['quality_contract_passed']
