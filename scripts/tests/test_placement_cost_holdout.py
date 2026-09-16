"""Keep new content cohorts out of the code/mixed calibration fit."""
import importlib.util
import json
from pathlib import Path
import sys


def test_topic_is_held_out_even_at_odd_widths(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'fit-ds41-placement-costs.py'
    spec = importlib.util.spec_from_file_location('placement_costs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for width in [1, 2, 3, 4]:
        for workload in ['code', 'mixed', 'topic']:
            for requests in [1, 4, 8]:
                rows = requests * (width + 1)
                # Deliberately incompatible held-out costs would distort a fit
                # if topic were accidentally included in "both" training sets.
                factor = 100 if workload == 'topic' else 1
                group = dict(layers=40, unique=40 * rows, expert_us=factor * (400 + rows * 80))
                records.append(dict(gpus=2, width=width, workload=workload, warm=True,
                    rows=rows, requests=requests, groups={'spark_tp4_shared2': group},
                    other_us=200 + rows * 10, verify_us=group['expert_us'] + 200 + rows * 10))
    monkeypatch.setattr(module, 'observations', lambda directory: (records, 'fixture'))
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', [str(path), str(tmp_path / 'rtx2-k7'), '--output', str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert report['training_rounds'] == 12
    for variant in report['variants'].values():
        topic = variant['evaluations']['rtx2_topic_odd']
        assert topic['training'] is False
        assert topic['candidate']['rounds'] == 6
        assert topic['candidate']['median_absolute_relative_error'] > .9
        assert variant['evaluations']['rtx2_code_odd']['training'] is True
        assert variant['evaluations']['rtx2_mixed_odd']['training'] is True
        assert variant['evaluations']['rtx2_code_even_holdout']['training'] is False


def test_stale_capture_recovery_requires_exact_trace_and_layer_manifest(tmp_path):
    import hashlib
    import pytest
    path = Path(__file__).parents[1] / 'fit-ds41-placement-costs.py'
    spec = importlib.util.spec_from_file_location('placement_exclusions', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'rtx2-k5'
    directory.mkdir()
    (directory / 'code.json').write_text('{}')
    trace = b'2026-09-17T00:00:00Z DEBUG ds41rt::cost_model: verification layer cost batch=12 layer=0 rows=42\n'
    (directory / 'server.log').write_bytes(trace)
    with pytest.raises(ValueError, match='unfinished layer'):
        module.observations(directory)
    manifest = dict(trace_sha256=hashlib.sha256(trace).hexdigest(),
                    batches=[dict(batch=12, layers=[0], reason='Audited prefill artifact fixture')])
    exclusion = directory / 'excluded_non_verification_batches.json'
    exclusion.write_text(json.dumps(manifest))
    assert module.observations(directory)[0] == []
    manifest['batches'][0]['layers'] = [0, 1]
    exclusion.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unfinished layer'):
        module.observations(directory)
    manifest['batches'][0]['layers'] = [0]
    trace += b'2026-09-17T00:00:01Z DEBUG ds41rt::cost_model: verification round cost batch=12 rows=42 verify_us=1\n'
    (directory / 'server.log').write_bytes(trace)
    with pytest.raises(ValueError, match='another trace'):
        module.observations(directory)
    manifest['trace_sha256'] = hashlib.sha256(trace).hexdigest()
    exclusion.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cannot exclude a completed verification'):
        module.observations(directory)


def test_explicit_reasoning_segment_needs_no_code_report(tmp_path):
    path = Path(__file__).parents[1] / 'fit-ds41-placement-costs.py'
    spec = importlib.util.spec_from_file_location('placement_reasoning', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'reasoning-rtx2-k7'
    directory.mkdir()
    lines = [f'2026-09-17T00:00:00Z DEBUG ds41rt::cost_model: verification layer cost batch=1 layer={layer} rows=8 '
             'routed_backend="spark_tp4" shared_tp=2 distinct_experts=6 expert_groups_16=6 '
             'produced_us=1 index_us=1 attention_us=1 experts_us=1 finish_us=1 total_us=5\n'
             for layer in range(40)]
    lines.append('2026-09-17T00:00:01Z DEBUG ds41rt::cost_model: verification round cost batch=1 lane=0 rows=8 requests=1 verify_us=250\n')
    raw = ''.join(lines).encode()
    (directory / 'server.log').write_bytes(raw)
    (directory / 'segments.json').write_text(json.dumps([dict(workload='code-reasoning',begin=0,end=len(raw))]))
    records, _ = module.observations(directory)
    assert len(records) == 1 and records[0]['workload'] == 'code-reasoning'


def test_capped_row_basis_can_reduce_slope_without_reducing_total_cost():
    path = Path(__file__).parents[1] / 'fit-ds41-placement-costs.py'
    spec = importlib.util.spec_from_file_location('placement_basis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = []
    for rows in [2, 8, 16, 17, 32, 64]:
        record = dict(rows=rows, requests=1)
        features = module.other_features(record, 'capped16')
        assert features == [1, min(rows,16), 1, max(0,rows-16)]
        values.append(sum(a*b for a,b in zip(features,[100,8,1,2])))
        group = dict(layers=3, unique=18)
        assert module.expert_features(record, group, 'capped16') == [3,3*min(rows,16),18,3*max(0,rows-16)]
    assert all(a < b for a,b in zip(values,values[1:]))
