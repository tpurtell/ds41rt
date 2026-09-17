"""Synthetic layout checks only; historical numbers are never release inputs."""
import copy
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]
RENDER = runpy.run_path(str(ROOT/'scripts/render-ds41-v5-release-tables.py'))


def reports():
    old = json.loads((ROOT/'docs/release-v4-performance.json').read_text())
    result = {}
    for model in ('full', 'exl3'):
        value = copy.deepcopy(old)
        value['model'] = model
        value['weighted_case_ids'] = list(set(RENDER['CASES']) - {'counting'})
        value['controls']['case_controls'] = {'code-reasoning': {'thinking': 'enabled', 'reasoning_effort': 'high'}}
        for layout in ('single', 'dual'):
            for mode in ('decode', 'target_decode'):
                value['layouts'][layout][mode]['cases']['code-reasoning'] = dict(samples=3, median_tps=123., min_tps=122., max_tps=124.)
        result[model] = value
    return result


def test_four_configuration_tables_without_old_comparisons_or_peaks():
    data = reports()
    tools = json.loads((ROOT/'docs/sparkinfer-upstream-tool-eval-20260916.json').read_text())
    text = RENDER['render'](data, tools={key:tools for key in ('full','exl3','exl3-fp4ple')})
    assert text.count(' prefill matrix.**') == 4
    assert text.count(' content-type decode.**') == 2
    assert text.count(' concurrency scaling.**') == 2
    assert text.count('| Code with reasoning |') == 2
    memory = text.split('**Memory after readiness.**')[1].split('**Tool calling.**')[0]
    assert sum(line.startswith(('| Full ', '| EXL3 ')) for line in memory.splitlines()) == 6
    tools_section = text.split('**Tool calling.**')[1]
    assert sum(line.startswith(('| Full /', '| EXL3 /')) for line in tools_section.splitlines()) == 9
    assert 'Change from v3' not in text and 'short-prefill controls' not in text and 'Observed GPU peaks' not in text
    assert '400 W' in text and 'standard memory speed' in text and 'private-tail tokens' in text


@pytest.mark.parametrize('defect', ['missing_reasoning', 'corpus_mismatch', 'binary_mismatch', 'duplicate_concurrency'])
def test_incomplete_or_unmatched_measurements_are_rejected(defect):
    data = reports()
    if defect == 'missing_reasoning':
        del data['exl3']['layouts']['dual']['decode']['cases']['code-reasoning']
    elif defect == 'corpus_mismatch':
        data['exl3']['corpus_sha256'] = 'different'
    elif defect == 'binary_mismatch':
        data['exl3']['binary_sha256'] = 'different'
    else:
        rows = data['exl3']['layouts']['single']['concurrency']['code']['summaries']
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(AssertionError):
        RENDER['render'](data)


def test_acceptance_keeps_grammar_targets_separate_and_rejects_invalid_denominator():
    data = reports()
    rate = dict(nonterminal_observations=10, verified_drafts=20, accepted_drafts=10,
                acceptance=.5, mean_emitted_tokens=2.)
    acceptance = {(model, layout): {'cases': {
        case: {'acceptance': {'grammar_constrained' if case == 'structured-json-schema'
                             else 'unconstrained': copy.deepcopy(rate)}}
        for case in RENDER['CASES'] if case != 'counting'}}
        for model in ('full', 'exl3') for layout in ('single', 'dual')}
    text = RENDER['render'](data, acceptance=acceptance)
    section = text.split('**Adaptive draft acceptance by content.**')[1].split('**Deployment')[0]
    assert section.count('50.00% (2.00)') == 36
    assert 'Schema JSON (grammar-constrained)' in section
    assert 'not teacher-forced quant agreement' in section
    acceptance['exl3', 'dual']['cases']['code']['acceptance']['unconstrained']['verified_drafts'] = 0
    with pytest.raises(AssertionError):
        RENDER['render'](data, acceptance=acceptance)
