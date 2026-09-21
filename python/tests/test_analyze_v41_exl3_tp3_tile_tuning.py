"""CPU tests for the offline tile-tuning lane analyzer (no GPU, synthetic lanes)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        'analyze_tp3_tuning_under_test', TOOLS / 'analyze_v41_exl3_tp3_tile_tuning.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load_analyzer()

GOOD_CLOCK = 'GPU-fake-uuid, P0, 150.00 W, 42.13 W, 1807 MHz, 3996 MHz, 0x0'


def make_record(cell, medians, *, throttle=GOOD_CLOCK, passed=True,
                variants=('baseline', 'k64-n256'), extra_light_case=False):
    timings = [dict(variant=name, tile=[64, 64, 64, 64], median_us=value,
                    samples_us=[value] * 6, min_us=value, max_us=value)
               for name, value in sorted(medians.items()) if name in variants]
    results = [dict(rows=16, distinct_experts=30, timings=timings,
                    clocks_before=throttle, clocks_after=throttle,
                    checks=[dict(passed=True)])]
    if extra_light_case:
        results.insert(0, dict(rows=1, distinct_experts=6, timings=[dict(
            variant=name, tile=[64, 64, 64, 64], median_us=value / 3,
            samples_us=[value / 3] * 6, min_us=value / 3, max_us=value / 3)
            for name, value in sorted(medians.items()) if name in variants],
            clocks_before=throttle, clocks_after=throttle,
            checks=[dict(passed=True)]))
    return dict(passed=passed,
                tier_family=dict(tiers=[3, 4],
                                 resolution='derived-from-checkpoint-trellis-widths-matching-global'),
                configuration=dict(intermediate=cell[0], slice_start=cell[1], capacity=cell[2]),
                layout=dict(intermediate=cell[0], slice_start=cell[1], capacity=cell[2],
                            planner_tile=[128, 128, 128, 128]),
                device=dict(name='fake-sm121'), source=dict(sparkinfer_revision='rev-x'),
                variants=[dict(name=name, status='compiled', tile=[64, 64, 64, 64])
                          for name in variants],
                results=results)


def write_lane(temp, records, manifest=True):
    lane = Path(temp) / 'lane'
    lane.mkdir()
    for name, record in records.items():
        (lane / name).write_text(json.dumps(record))
    if manifest:
        (lane / 'manifest.json').write_text(json.dumps(
            {'schema': 'ds41rt.tp3-tile-tuning-lane-v1', 'ds41rt_commit': 'c' * 40,
             'status': 'complete'}))
    return lane


class CleanLaneTests(unittest.TestCase):
    def _records(self):
        records = {}
        cells = {(768, 0, 16), (768, 768, 16), (768, 1536, 16)}
        for cell in cells:
            for rep, (base, cand) in enumerate([(100.0, 88.0), (99.0, 87.5), (101.0, 88.5)], 1):
                records[f'tp3-w768-s{cell[1]}-c{cell[2]}_rep{rep}.json'] = make_record(
                    cell, {'baseline': base, 'k64-n256': cand})
        # Legacy paired cell where the candidate LOSES: must not veto the primary
        # recommendation, because promotion only ever concerns the TP3 matrix.
        for rep, (base, cand) in enumerate([(100.0, 120.0), (100.0, 120.0), (100.0, 120.0)], 1):
            records[f'legacy-w640-s640-c80_rep{rep}.json'] = make_record(
                (640, 640, 80), {'baseline': base, 'k64-n256': cand})
        return records

    def test_winner_across_every_primary_cell_is_recommended(self):
        with tempfile.TemporaryDirectory() as temp:
            lane = write_lane(temp, self._records())
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertEqual(summary['runs'], 12)
            self.assertTrue(summary['clean'], summary['problems'])
            self.assertEqual(summary['recommendation']['candidate'], 'k64-n256')
            self.assertTrue(summary['recommendation']['default_holds'] is False)
            tp3 = {tuple(c['cell'].values()): c for c in summary['cells']}
            cell = tp3[(768, 0, 16)]
            self.assertEqual(cell['variants']['k64-n256']['median_us'], 88.0)
            self.assertEqual(cell['variants']['k64-n256']['rep_medians'], [88.0, 87.5, 88.5])
            self.assertLess(cell['variants']['k64-n256']['rep_spread_pct'], 2.0)
            self.assertTrue((lane / 'summary.md').is_file())

    def test_heads_up_the_summary_uses_the_heaviest_timed_case(self):
        # A lighter second timed case (value/3) must not become the headline.
        with tempfile.TemporaryDirectory() as temp:
            records = {}
            for rep in range(3):
                records[f'tp3-w768-s0-c16_rep{rep+1}.json'] = make_record(
                    (768, 0, 16), {'baseline': 100.0, 'k64-n256': 90.0},
                    extra_light_case=True)
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertEqual(summary['cells'][0]['variants']['k64-n256']['median_us'], 90.0)


class ProblemLaneTests(unittest.TestCase):
    def test_a_family_other_than_3_4_violates_the_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            record = make_record((768, 0, 16), {'baseline': 100.0, 'k64-n256': 88.0})
            record['tier_family'] = dict(tiers=[4, 5], resolution='declared')
            lane = write_lane(temp, {'tp3-w768-s0-c16_rep1.json': record})
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertTrue(any('not [3, 4]' in x for x in summary['problems']))
            self.assertTrue(any('not derived' in x or 'was not derived' in x
                                for x in summary['problems']))


    def _base(self):
        records = {}
        for rep, (base, cand) in enumerate([(100.0, 88.0), (99.0, 87.5), (101.0, 88.5)], 1):
            records[f'tp3-w768-s0-c16_rep{rep}.json'] = make_record(
                (768, 0, 16), {'baseline': base, 'k64-n256': cand})
        return records

    def test_throttle_bitmask_is_a_provenance_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            records = self._base()
            records['tp3-w768-s0-c16_rep3.json'] = make_record(
                (768, 0, 16), {'baseline': 101.0, 'k64-n256': 88.5},
                throttle='GPU-fake-uuid, P0, 150.00 W, 42.13 W, 1305 MHz, 3996 MHz, 0x4')
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertFalse(summary['clean'])
            self.assertTrue(any('throttl' in p.lower() for p in summary['problems']))
            with self.assertRaises(SystemExit) as caught:
                analyzer.main([str(lane), '--fail-on-problems'])
            self.assertEqual(caught.exception.code, 4)

    def test_failed_correctness_gate_is_reported_even_with_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            records = self._base()
            records['tp3-w768-s0-c16_rep2.json'] = make_record(
                (768, 0, 16), {'baseline': 99.0, 'k64-n256': 87.5}, passed=False)
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertTrue(any('correctness' in p for p in summary['problems']))

    def test_reps_must_have_compiled_the_same_variant_set(self):
        with tempfile.TemporaryDirectory() as temp:
            records = self._base()
            records['tp3-w768-s0-c16_rep3.json'] = make_record(
                (768, 0, 16), {'baseline': 101.0}, variants=('baseline',))
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertTrue(any('variant set' in p for p in summary['problems']))

    def test_unparseable_clock_line_is_a_hard_provenance_problem(self):
        with tempfile.TemporaryDirectory() as temp:
            records = self._base()
            records['tp3-w768-s0-c16_rep1.json'] = make_record(
                (768, 0, 16), {'baseline': 100.0, 'k64-n256': 88.0}, throttle='')
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertTrue(any('clock provenance' in p for p in summary['problems']))

    def test_no_consistent_winner_holds_the_production_default(self):
        with tempfile.TemporaryDirectory() as temp:
            records = {}
            # candidate wins cap16, loses cap80: recommendation must stay default.
            for rep in range(3):
                records[f'tp3-w768-s0-c16_rep{rep+1}.json'] = make_record(
                    (768, 0, 16), {'baseline': 100.0, 'k64-n256': 85.0})
                records[f'tp3-w768-s0-c80_rep{rep+1}.json'] = make_record(
                    (768, 0, 80), {'baseline': 300.0, 'k64-n256': 305.0})
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertIsNone(summary['recommendation']['candidate'])
            self.assertTrue(summary['recommendation']['default_holds'])



GB10_CLOCK = 'GPU-3740a9fc-56c0-4a6f-8d2e-9f1b4c7d0a11, P0, N/A, 11.57 W, 2235 MHz, N/A, 0x0'


class ClockParsingTests(unittest.TestCase):
    def test_real_gb10_line_parses_with_masked_fields_none(self):
        sample = analyzer.parse_clocks(GB10_CLOCK)
        self.assertIsNotNone(sample)
        self.assertEqual(sample['uuid'], 'GPU-3740a9fc-56c0-4a6f-8d2e-9f1b4c7d0a11')
        self.assertIsNone(sample['power_limit_w'])
        self.assertEqual(sample['power_draw_w'], 11.57)
        self.assertEqual(sample['sm_mhz'], 2235)
        self.assertIsNone(sample['mem_mhz'])
        self.assertEqual(sample['throttle_reasons'], '0x0')

    def test_na_in_a_required_field_is_not_provenance(self):
        for bad in (
            'GPU-x, P0, N/A, N/A, 2235 MHz, N/A, 0x0',        # power.draw N/A
            'GPU-x, P0, N/A, 11.57 W, N/A MHz, N/A, 0x0',     # clocks.sm N/A
            'GPU-x, P0, N/A, 11.57 W, 2235 MHz, N/A, N/A',    # throttle N/A
            ', P0, N/A, 11.57 W, 2235 MHz, N/A, 0x0',         # uuid missing
        ):
            with self.subTest(line=bad):
                self.assertIsNone(analyzer.parse_clocks(bad))

    def test_gb10_clock_lines_run_a_clean_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            records = {}
            for rep, (base, cand) in enumerate([(100.0, 88.0), (99.0, 87.5), (101.0, 88.5)], 1):
                records[f'tp3-w768-s0-c16_rep{rep}.json'] = make_record(
                    (768, 0, 16), {'baseline': base, 'k64-n256': cand}, throttle=GB10_CLOCK)
            lane = write_lane(temp, records)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertTrue(summary['clean'], summary['problems'])
            self.assertEqual(summary['recommendation']['candidate'], 'k64-n256')



class LaneManifestStatusTests(unittest.TestCase):
    def test_aborted_lane_manifest_is_surfaced_and_not_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            records = {}
            for rep in range(3):
                records[f'tp3-w768-s0-c16_rep{rep+1}.json'] = make_record(
                    (768, 0, 16), {'baseline': 100.0, 'k64-n256': 88.0})
            lane = write_lane(temp, records)
            manifest = json.loads((lane / 'manifest.json').read_text())
            manifest.update(status='aborted', abort_reason='throttle provenance failed',
                            aborted_cell='tp3-w768-s768-c16 rep2')
            (lane / 'manifest.json').write_text(json.dumps(manifest))
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertEqual(summary['lane_status'], 'aborted')
            self.assertEqual(summary['ds41rt_commit'], 'c' * 40)   # never None on abort now
            self.assertFalse(summary['clean'])
            self.assertTrue(any('did not complete' in x and 'throttle' in x
                                for x in summary['problems']))

    def test_missing_lane_manifest_reports_no_manifest_and_keeps_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            lane = write_lane(temp, {
                'tp3-w768-s0-c16_rep1.json': make_record(
                    (768, 0, 16), {'baseline': 100.0, 'k64-n256': 88.0})}, manifest=False)
            analyzer.main([str(lane)])
            summary = json.loads((lane / 'summary.json').read_text())
            self.assertEqual(summary['lane_status'], 'no-manifest')
            self.assertFalse(summary['clean'])
            self.assertEqual(summary['runs'], 1)


if __name__ == '__main__':
    unittest.main()
