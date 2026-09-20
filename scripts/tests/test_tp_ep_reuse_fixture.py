"""CPU-only validation tests for the deterministic TP×EP reuse fixture.

Loads `scripts/fixtures/tp-ep-reuse-m64-e384.json` through the standalone
validator and re-checks the exact counts, per-row distinctness, no-remainder
realization and whole-expert EP2/EP3 ownership. No GPU, build, benchmark edit or
remote operation.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "scripts" / "fixtures" / "tp-ep-reuse-m64-e384.json"
VALIDATOR = ROOT / "scripts" / "validate-tp-ep-reuse-fixture.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_tp_ep_reuse_fixture", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_validator()
FIXTURE_DATA = json.loads(FIXTURE.read_text())


class ReuseFixtureTest(unittest.TestCase):
    def test_fixture_validates_with_exact_counts(self):
        report = MODULE.validate(copy.deepcopy(FIXTURE_DATA))
        self.assertEqual(report["experts_active"], 48)
        self.assertEqual(report["routes"], 384)
        self.assertEqual(report["rows"], 64)
        self.assertEqual(report["topk"], 6)
        self.assertEqual(report["mean_rows_per_expert"], 8.0)
        self.assertEqual(report["tail_experts"], 7)
        self.assertEqual(report["tail_routes"], 173)
        self.assertTrue(report["tail_per_expert_synthetic"])
        self.assertTrue(report["no_remainder"])
        self.assertTrue(report["realized_by_greedy_highest_remaining"])

    def test_every_row_has_six_distinct_official_ids(self):
        routes = FIXTURE_DATA["routes"]
        self.assertEqual(len(routes), 64)
        for index, row in enumerate(routes):
            self.assertEqual(len(row), 6, index)
            self.assertEqual(len(set(row)), 6, index)
            self.assertTrue(all(0 <= expert < 384 for expert in row), index)

    def test_counts_are_fully_realized_with_no_remainder(self):
        declared = [0] * 384
        for entry in FIXTURE_DATA["expert_counts"]:
            declared[entry["id"]] = entry["rows"]
        realized = MODULE.counts_from_routes(FIXTURE_DATA["routes"], 384)
        self.assertEqual(realized, declared)
        regenerated = MODULE.realize_counts(declared, 64, 6)
        self.assertEqual(MODULE.counts_from_routes(regenerated, 384), declared)

    def test_whole_expert_lpt_ownership_ep2_and_ep3(self):
        declared = [0] * 384
        for entry in FIXTURE_DATA["expert_counts"]:
            declared[entry["id"]] = entry["rows"]
        ep2 = MODULE.lpt_ownership(declared, 2)
        self.assertEqual(ep2["group_experts"], [24, 24])
        # Equal expert counts, deliberately unequal routed rows.
        self.assertEqual(ep2["group_routes"], [185, 199])
        ep3 = MODULE.lpt_ownership(declared, 3)
        self.assertEqual(ep3["group_experts"], [16, 16, 16])
        self.assertEqual(ep3["group_routes"], [120, 124, 140])
        for plan in (ep2, ep3):
            self.assertEqual(sum(plan["assigned"]), 48)

    def test_balanced_rows_observed_checks_all_groups_and_is_never_a_guarantee(self):
        # First == last but the middle group differs: the old first/last check
        # would have wrongly reported balance for EP3.
        self.assertFalse(MODULE.balanced_rows_observed([120, 144, 120]))
        self.assertTrue(MODULE.balanced_rows_observed([120, 120, 120]))
        report = MODULE.validate(copy.deepcopy(FIXTURE_DATA))
        for groups in ("ep2", "ep3"):
            entry = report["ownership"][groups]
            self.assertFalse(entry["balanced_rows_observed"])
            self.assertFalse(entry["balanced_rows_guaranteed"])
            self.assertIn("benchmark stable ties", entry["tie_break"])
            self.assertIn("not the Rust coordinator's seeded tie-break", entry["tie_break"])

    def test_fixture_records_the_mean_aggregation_and_no_balance_claim(self):
        notes = FIXTURE_DATA["notes"]
        self.assertIn("ratio of medians", notes["mean_rows_per_expert"])
        self.assertIn("does not imply equal routed rows", notes["no_balanced_rows_guarantee"])
        footprint = FIXTURE_DATA["provenance"]["active_weight_footprint"]
        self.assertIn("not measured DRAM", footprint)

    def test_impossible_degree_sequence_is_rejected(self):
        counts = [6] + [0] * 383
        with self.assertRaises(ValueError):
            MODULE.realize_counts(counts, 1, 6)

    def test_malformed_fixtures_are_rejected(self):
        mutations = {}

        duplicate = copy.deepcopy(FIXTURE_DATA)
        duplicate["routes"][0][1] = duplicate["routes"][0][0]
        mutations["duplicate expert in a row"] = duplicate

        wrong_count = copy.deepcopy(FIXTURE_DATA)
        wrong_count["expert_counts"][0]["rows"] += 1
        mutations["declared count mismatch"] = wrong_count

        out_of_range = copy.deepcopy(FIXTURE_DATA)
        out_of_range["expert_counts"][0]["id"] = 384
        mutations["out-of-range expert id"] = out_of_range

        too_many = copy.deepcopy(FIXTURE_DATA)
        too_many["expert_counts"][0]["rows"] = 65
        mutations["count above row count"] = too_many

        bad_sum = copy.deepcopy(FIXTURE_DATA)
        bad_sum["expert_counts"][1]["rows"] += 1
        mutations["count sum mismatch"] = bad_sum

        bad_tail = copy.deepcopy(FIXTURE_DATA)
        bad_tail["tail_bin"]["routes"] = 172
        mutations["tail aggregate mismatch"] = bad_tail

        bad_schema = copy.deepcopy(FIXTURE_DATA)
        bad_schema["schema"] = "ds41rt.tp-ep-reuse-fixture.v0"
        mutations["schema mismatch"] = bad_schema

        fractional_rows = copy.deepcopy(FIXTURE_DATA)
        fractional_rows["rows"] = 64.5
        mutations["fractional rows scalar"] = fractional_rows

        fractional_count = copy.deepcopy(FIXTURE_DATA)
        fractional_count["expert_counts"][0]["rows"] = 1.5
        mutations["fractional expert count"] = fractional_count

        bool_id = copy.deepcopy(FIXTURE_DATA)
        bool_id["expert_counts"][0]["id"] = True
        mutations["bool expert id"] = bool_id

        fractional_route = copy.deepcopy(FIXTURE_DATA)
        fractional_route["routes"][0][0] = 1.5
        mutations["fractional route id"] = fractional_route

        for name, fixture in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                MODULE.validate(fixture)


if __name__ == "__main__":
    unittest.main()
