import copy
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / 'summarize-ds41-upstream-release.py'))


def measurement():
    corpus = dict(weighted_case_ids=['code', 'code-reasoning'], cases={
        'code': {}, 'code-reasoning': dict(thinking='enabled', reasoning_effort='high')})
    samples = []
    for case in ['code', 'code-reasoning', 'counting']:
        for repeat in [1, 2, 3]:
            request = dict(thinking={'type': 'enabled' if case == 'code-reasoning' else 'disabled'})
            if case == 'code-reasoning':
                request['reasoning_effort'] = 'high'
            samples.append(dict(case=case, repeat=repeat, request=request))
    return dict(samples=samples), corpus


def test_report_checks_actual_reasoning_request_and_complete_repeats():
    report, corpus = measurement()
    MODULE['validate_decode_corpus'](report, corpus)
    for change in ['missing_reasoning', 'thinking_disabled', 'effort_missing', 'repeat_duplicate']:
        broken = copy.deepcopy(report)
        if change == 'missing_reasoning':
            broken['samples'] = [s for s in broken['samples'] if s['case'] != 'code-reasoning']
        elif change == 'thinking_disabled':
            broken['samples'][3]['request']['thinking']['type'] = 'disabled'
        elif change == 'effort_missing':
            del broken['samples'][3]['request']['reasoning_effort']
        else:
            broken['samples'][0]['repeat'] = 2
        with pytest.raises(AssertionError):
            MODULE['validate_decode_corpus'](broken, corpus)


@pytest.mark.parametrize('field', ['encoder_layers', 'rtx_expert_layers'])
def test_deployment_reads_both_log_formats_and_rejects_partition_gap(tmp_path, field):
    path = tmp_path / 'server.log'
    path.write_text('dual RTX cache reservation after fixed allocations '
                    f'source_pages=[28736,28736,28736,57472] global_bytes=13094420480 '
                    f'runtime_headroom_bytes=838860800 {field}=20\n')
    metadata = dict(layout='dual', arguments=['--concurrency','16','--prefix-cache-entries','24'],
                    workers=[dict(args=['--device-budget-bytes','107374182400','--first-layer','20'])])
    result = MODULE['summarize_deployment'](metadata, path)
    assert result['rtx_expert_layers'] == result['spark_first_layer'] == 20
    assert result['logical_pool_tokens'] == 14680064
    metadata['workers'][0]['args'][-1] = '21'
    with pytest.raises(AssertionError, match='partition'):
        MODULE['summarize_deployment'](metadata, path)
