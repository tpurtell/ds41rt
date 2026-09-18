"""Reports expose compact cells without inventing unmeasured results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class CompactReports(unittest.TestCase):
    def test_quant_report_reads_single_compact_measurements(self):
        report = load('render-ds41-v7-quant-reports')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = report.render('exl3', root)
            self.assertIn('32 GiB budget', missing)
            self.assertIn('not a measurement on RTX 5090', missing)
            self.assertIn('| Weighted decode | — | — | — |', missing)
            (root / 'single-exl3-dspark.json').write_text(json.dumps({
                'median_weighted_observed_decode_tokens_per_second': 42.5,
                'samples': [{'case': 'code', 'observed_decode_tokens_per_second': n} for n in [40,42,44]],
            }))
            (root / 'single-exl3-prefill.json').write_text(json.dumps({'passed': True, 'completed_ns': 123, 'cells': [{
                'base_context_tokens': 0, 'suffix_tokens': 1024,
                'median_effective_prefill_tokens_per_second': 1234,
            }]}))
            measured = report.render('exl3', root)
            self.assertIn('| Weighted decode | 42.50 | — | — |', measured)
            self.assertIn('| Best prefill | 1,234 | — | — |', measured)
            self.assertNotIn('not implemented', measured)
            (root / 'single-exl3-dspark.json').write_text(json.dumps({
                'samples': [{'case': 'code-reasoning', 'repeat': 3, 'passed': False,
                             'finish_reason': 'length', 'content': '',
                             'observed_decode_tokens_per_second': 42}],
            }))
            failed = report.render('exl3', root)
            self.assertIn('0/1 sample checks explicitly passed', failed)
            self.assertIn('final response empty', failed)
            self.assertIn('not a successful quality result', failed)

    def test_report_preserves_missing_columns_and_provenance(self):
        report = load('render-ds41-v7-quant-reports')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text = report.render('nvfp4', root)
            self.assertIn('| Case | 1 RTX dSpark | 2 RTX dSpark |', text)
            self.assertIn('| Code | — | — |', text)
            self.assertIn('not proof of the quant snapshot', text)
            self.assertIn('have not been built or published', text)
            self.assertIn('SHA-256 `unavailable`', text)
            self.assertLess(text.index('## Prefill'), text.index('## Quality and evaluation'))
            self.assertIn('target-only decode', text)
            sample = {'case': 'code-reasoning', 'repeat': 3, 'passed': False,
                      'finish_reason': 'length', 'content': '',
                      'usage': {'completion_tokens': 4096}, 'request': {'max_tokens': 4096},
                      'observed_decode_tokens_per_second': 42}
            (root / 'single-exl3-dspark.json').write_text(json.dumps({'samples': [sample]}))
            text = report.render('exl3', root)
            self.assertIn('Output tokens: 4096; request cap: 4096', text)
            self.assertIn('0/1 sample checks explicitly passed', text)
            self.assertIn('not a successful quality result', text)

    def test_headline_missing_note_follows_cells(self):
        report = load('render-ds41-v7-headline')
        self.assertIn('EXL3 1x weighted decode', report.render())
        for column, _q, _l, _p, values in report.CONFIGS:
            if column == 'EXL3 1x':
                values.update({metric: 42 for metric in report.METRICS})
        text = report.render()
        self.assertNotIn('EXL3 1x weighted decode', text)
        self.assertIn('simulates RTX 5090 memory capacity, not its performance', text)
        self.assertIn('NVFP4 1x prefill', text)

    def test_headline_never_falls_back_or_retains_previous_package(self):
        report = load('render-ds41-v7-headline')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs, documents = report.load_measurements(root / 'missing')
            for _column, _quant, _layout, provenance, values in configs:
                if provenance == 'v7':
                    self.assertTrue(all(value is None for value in values.values()))
            self.assertEqual(report.pairs(configs)['official']['1x']['Weighted decode'], 92)
            path = root / 'single-exl3-dspark.json'
            path.write_text(json.dumps({'samples': [
                {'case': 'code', 'observed_decode_tokens_per_second': 123.54, 'passed': True},
                {'case': 'counting', 'observed_decode_tokens_per_second': 163.55, 'passed': True},
            ], 'median_weighted_observed_decode_tokens_per_second': 88.103}))
            configs, documents = report.load_measurements(root)
            text = report.render(configs=configs, documents=documents)
            self.assertIn('123.54', text)
            self.assertIn('163.55', text)
            self.assertIn('88.10', text)
            self.assertEqual(report.pairs(configs)['exl3']['1x']['Weighted decode'], 88.103)
            path.unlink()
            configs, _documents = report.load_measurements(root)
            self.assertIsNone(report.pairs(configs)['exl3']['1x']['C1 code decode'])
            self.assertIn('historical v6', text)
            self.assertIn('not isolated second-GPU scaling', text)
            self.assertIn('not its performance', report.render(pending_note=False))

    def test_prefill_requires_both_success_and_completion(self):
        quant = load('render-ds41-v7-quant-reports')
        headline = load('render-ds41-v7-headline')
        cells = [{'base_context_tokens': 0, 'suffix_tokens': 1024,
                  'median_effective_prefill_tokens_per_second': 1234.56}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for status in ({}, {'passed': True}, {'completed_ns': 123},
                           {'passed': False, 'completed_ns': 123},
                           {'passed': None, 'completed_ns': 123},
                           {'passed': True, 'completed_ns': 0},
                           {'passed': True, 'completed_ns': 123}):
                with self.subTest(status=status):
                    document = dict(status, cells=cells)
                    complete = status.get('passed') is True and bool(status.get('completed_ns'))
                    for stem in ('single-exl3', 'dual-nvfp4'):
                        (root / f'{stem}-prefill.json').write_text(json.dumps(document))
                    self.assertEqual(quant.best_prefill(document), 1234.56 if complete else None)
                    configs, documents = headline.load_measurements(root)
                    for q, layout in (('exl3', '1x'), ('nvfp4', '2x')):
                        self.assertEqual(headline.pairs(configs)[q][layout]['Prefill'],
                                         1234.56 if complete else None)
                    text = quant.render('exl3', root)
                    if complete:
                        self.assertIn('| Best prefill | 1,235 | — | — |', text)
                    else:
                        self.assertIn('| Best prefill | — | — | — |', text)
                        self.assertIn('| 0K | 1,235 |', text)
                        self.assertIn('partial measurements below are provisional', text)
                        self.assertIn('partial measurements are provisional',
                                      headline.render(configs=configs, documents=documents))

    def test_unknown_decode_status_is_not_a_pass_and_reads_are_reused(self):
        quant = load('render-ds41-v7-quant-reports')
        headline = load('render-ds41-v7-headline')
        samples = [{'case': 'code', 'observed_decode_tokens_per_second': 42, **status}
                   for status in ({'passed': True}, {'passed': False}, {'passed': None}, {})]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'single-exl3-dspark.json').write_text(json.dumps({'samples': samples}))
            with mock.patch.object(quant, 'load', wraps=quant.load) as reader:
                text = quant.render('exl3', root)
                self.assertEqual(reader.call_count, 4)
            self.assertIn('1/4 sample checks explicitly passed; 1 failed; 2 unknown/unreported', text)
            with mock.patch.object(headline, 'load', wraps=headline.load) as reader:
                configs, documents = headline.load_measurements(root)
                text = headline.render(configs=configs, documents=documents)
                self.assertEqual(reader.call_count, 8)
            self.assertIn('1/4 decode sample checks explicitly passed; 1 failed; 2 unknown/unreported', text)

    def test_invalid_records_are_unavailable_without_crashing(self):
        quant = load('render-ds41-v7-quant-reports')
        headline = load('render-ds41-v7-headline')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for bad in ('{', 'null', '[]', '42'):
                with self.subTest(record=bad):
                    (root / 'single-exl3-dspark.json').write_text(bad)
                    self.assertIn('| Weighted decode | — | — | — |', quant.render('exl3', root))
                    configs, documents = headline.load_measurements(root)
                    self.assertIsNone(headline.pairs(configs)['exl3']['1x']['Weighted decode'])
                    headline.render(configs=configs, documents=documents)
                    with mock.patch('sys.argv', ['render', '--package', str(root),
                                                 '--output', str(root / 'headline.md')]):
                        headline.main()
                    self.assertTrue((root / 'headline.md').exists())

if __name__ == '__main__':
    unittest.main()
