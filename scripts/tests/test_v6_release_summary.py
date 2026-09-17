"""Reject incomplete native release matrices and incorrect power controls."""
import copy
from pathlib import Path
import runpy
import tempfile
import unittest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'summarize-ds41-v6-release.py'))


class V6SummaryTest(unittest.TestCase):
    def test_no_historical_fallback_for_missing_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                MODULE['assemble'](Path(directory))

    def test_power_controls_are_evidence_not_constants(self):
        header = 'uuid, power.limit [W], clocks.max.memory [MHz]\n'
        MODULE['validate_settings'](header + 'GPU-0, 400.00 W, 14001 MHz\n')
        for body in ['', 'GPU-0, 300.00 W, 14001 MHz\n', 'GPU-0, 400.00 W, 15000 MHz\n']:
            with self.subTest(body=body), self.assertRaises(AssertionError):
                MODULE['validate_settings'](header + body)

    def test_prefill_requires_three_distinct_samples_per_cell(self):
        report = dict(passed=True, repeats=3, bases=MODULE['BASES'], suffixes=MODULE['SUFFIXES'],
                      samples=[dict(base_context_tokens=b, suffix_tokens=s, repeat=r, timed=True, passed=True)
                               for b in MODULE['BASES'] for s in MODULE['SUFFIXES'] for r in (1, 2, 3)])
        MODULE['validate_prefill_samples'](report)
        for mutation in ['missing', 'duplicate', 'failed', 'warmup']:
            candidate = copy.deepcopy(report)
            if mutation == 'missing': candidate['samples'].pop()
            elif mutation == 'duplicate': candidate['samples'][-1] = candidate['samples'][0]
            elif mutation == 'failed': candidate['samples'][-1]['passed'] = False
            else: candidate['samples'][-1]['timed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                MODULE['validate_prefill_samples'](candidate)


if __name__ == '__main__':
    unittest.main()
