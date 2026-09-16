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
