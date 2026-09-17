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


def test_acceptance_loader_reproduces_traces_and_rejects_tampering(tmp_path):
    import hashlib
    data = reports()
    collector = runpy.run_path(str(ROOT/'scripts/collect-ds41-content-acceptance.py'))
    policy = runpy.run_path(str(ROOT/'scripts/summarize-ds41-native-policy.py'))
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    write(tmp_path/'complete.json', {'results': [{'passed': True}] * 4})
    write(tmp_path/'restoration.json', {'errors': []})
    for model in ('full', 'exl3'):
        for count in (1, 2):
            directory = tmp_path/f'{model}-rtx{count}'
            write(directory/'deployment.json', {
                'binary_sha256': data[model]['binary_sha256'],
                'coordinator': {'Config': {'Labels': {
                    'org.opencontainers.image.revision': data[model]['engine_commit']}}}})
            cases = {}
            for case in set(RENDER['CASES']) - {'counting'}:
                thinking = 'enabled' if case == 'code-reasoning' else 'disabled'
                effort = 'high' if thinking == 'enabled' else None
                constrained = 'true' if case == 'structured-json-schema' else 'false'
                raw = '\n'.join(
                    f'native draft policy observation request_id={i} lane=0 generated=10 '
                    f'verifier_rows=3 matched_prefix=1 raw_confidence=[0,0] constrained={constrained} '
                    'eos=false length_limit=false' for i in (1, 2, 3)).encode()
                observations, _ = policy['parse'](raw.decode())
                cases[case] = dict(thinking=thinking, reasoning_effort=effort,
                    trace_sha256=hashlib.sha256(raw).hexdigest(),
                    acceptance=collector['summarize_acceptance'](observations))
                request = {'thinking': {'type': thinking}, 'reasoning_effort': effort}
                write(directory/f'content/{case}.json', {'passed': True, 'samples': [
                    dict(case=case, repeat=i, passed=True, request=request) for i in (1, 2, 3)]})
                (directory/f'content/{case}-trace.log').write_bytes(raw)
            write(directory/'content/summary.json', dict(passed=True, repeats=3, concurrency=1,
                  rtx_gpus=count, draft_policy='adaptive', cases=cases,
                  corpus_sha256=data[model]['corpus_sha256']))
    result = RENDER['load_acceptance'](tmp_path, data)
    assert len(result) == 4
    trace = tmp_path/'exl3-rtx2/content/code-reasoning-trace.log'
    trace.write_bytes(trace.read_bytes().replace(b'matched_prefix=1', b'matched_prefix=2'))
    with pytest.raises(AssertionError):
        RENDER['load_acceptance'](tmp_path, data)
