"""Check native table coverage without reusing historical data in release output."""
import copy
import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / 'scripts/render-ds41-v6-release-tables.py'))


class V6TablesTest(unittest.TestCase):
    def fixture(self):
        # Historical native summaries supply realistic shapes only for this test.
        source = json.loads((ROOT / 'docs/release-v5-performance.json').read_text())['performance']['full']
        report = copy.deepcopy(source)
        report['release'] = 'v6'
        report['launches'] = {}
        for layout, count in [('single', 1), ('dual', 2)]:
            deployment = report['deployment'][layout]
            report['launches'][layout] = dict(
                startup_seconds=1.0, deployment=deployment,
                ram_capacity=dict(device_tokens=deployment['logical_pool_tokens'], host_tokens=1000,
                                  combined_tokens=deployment['logical_pool_tokens'] + 1000, pinned_bytes=2**30),
                gpu_uuids=[f'GPU-{i}' for i in range(count)],
                gpu_memory='uuid, memory.used [MiB], memory.free [MiB]\n' +
                           ''.join(f'GPU-{i}, 90000 MiB, 7250 MiB\n' for i in range(count)))
        return report

    def test_all_native_tables_and_token_capacities_are_present(self):
        result = MODULE['render'](self.fixture(), {})
        headers = [line for line in result.splitlines() if line.startswith('**')]
        self.assertEqual(len(headers), 10)
        for label in ['Content-type decode', '1 RTX prefill matrix', '2 RTX prefill matrix',
                      'Decode over retained context', 'Concurrency scaling', 'Mixed traffic',
                      'Deployment and cache capacity', 'Startup', 'Memory after readiness']:
            self.assertIn('**' + label + '.**', result)
        self.assertIn('Code with reasoning', result)
        self.assertIn('14,680,064 usable', result)
        self.assertIn('1 GiB pinned / 1,000 logical tokens', result)
        self.assertNotIn('EXL3 1 RTX', result)
        widths = set()
        for line in result.splitlines():
            if not line.startswith('|'):
                self.assertLessEqual(len(widths), 1)
                widths = set()
            else:
                widths.add(len(line.split('|')))

    def test_unqualified_or_wrong_release_rejected(self):
        for key, value in [('release', 'v5'), ('performance_matrix_passed', False)]:
            report = self.fixture()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                MODULE['render'](report, {})


if __name__ == '__main__':
    unittest.main()
