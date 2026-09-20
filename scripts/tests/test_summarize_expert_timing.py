"""CPU-only coverage for the Spark expert-timing log summarizer.

Replicated expert groups own only a subset of the canonical six routes per row
(unowned routes are masked to the 384 sentinel), so the summarizer must infer
the owned route total from the histogram instead of assuming ``rows * 6``.
Legacy single-group records must keep their old meaning, malformed records must
still be rejected, and an empty group must keep its distribution metadata with
null fractions rather than dividing by zero.

No GPU, container, service or build operation is performed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "summarize-ds41-expert-timing.py"


def load_summarizer():
    spec = importlib.util.spec_from_file_location("summarize_ds41_expert_timing", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_summarizer()


def native_record(rows, counts, tail_routes, active_experts=None, extra=None):
    """One debug line in the exact shape emitted by the daemon timing target."""
    if active_experts is None:
        active_experts = sum(counts)
    output_bytes = int(rows) * 10240 if isinstance(rows, int) else 20480
    fields = {
        "layer": 7,
        "executor_id": 17,
        "rows": rows,
        "active_experts": active_experts,
        "max_expert_rows": max((i + 1 for i, v in enumerate(counts) if v), default=0),
        "kernel_capacity": 80,
        "output_bytes": output_bytes,
        "upload_us": 1,
        "kernel_us": 2,
        "compact_us": 3,
        "execution_host_us": 4,
        "download_us": 5,
        "total_us": 6,
        "expert_rows_tail_routes": tail_routes,
    }
    if extra:
        fields.update(extra)
    body = " ".join(f"{key}={value}" for key, value in fields.items())
    histogram = "[" + ", ".join(str(value) for value in counts) + "]"
    return (
        "2026-01-01T00:00:00Z INFO ds41rt::expert_timing: "
        f"{body} expert_rows_histogram={histogram} native expert execution"
    )


def summary_for(lines):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "timing.log"
        path.write_text("\n".join(lines) + "\n")
        return MODULE.summarize([path])


class SummarizerTest(unittest.TestCase):
    def test_single_group_legacy_histogram_keeps_its_old_meaning(self):
        counts = [0] * 17
        counts[1] = 6  # six experts, each with two routed rows
        group = summary_for([native_record(2, counts, 0)])["groups"][0]
        self.assertEqual(group["owned_routes"], 12)
        self.assertEqual(group["total_request_routes"], 12)
        self.assertEqual(group["route_fraction_scope"], "owned routes inferred from histogram")
        fractions = [entry["route_fraction"] for entry in group["expert_row_distribution"]]
        self.assertAlmostEqual(sum(fractions), 1.0)

    def test_replicated_group_owns_a_subset_of_the_request_routes(self):
        # Two rows, six canonical routes per row (12), but this group owns 7:
        # five experts with one row and one expert with two rows.
        counts = [0] * 17
        counts[0] = 5
        counts[1] = 1
        group = summary_for([native_record(2, counts, 0)])["groups"][0]
        self.assertEqual(group["owned_routes"], 7)
        self.assertEqual(group["total_request_routes"], 12)
        self.assertLess(group["owned_routes"], group["total_request_routes"])
        fractions = [entry["route_fraction"] for entry in group["expert_row_distribution"]]
        self.assertAlmostEqual(sum(fractions), 1.0)
        # The repeated expert is legal: two rows can route to the same expert.
        self.assertEqual(group["metrics"]["max_expert_rows"]["max"], 2)

    def test_empty_group_keeps_metadata_with_null_fractions(self):
        counts = [0] * 17
        group = summary_for([native_record(2, counts, 0, active_experts=0)])["groups"][0]
        self.assertEqual(group["owned_routes"], 0)
        self.assertEqual(group["total_request_routes"], 12)
        self.assertEqual(len(group["expert_row_distribution"]), 17)
        self.assertTrue(all(entry["route_fraction"] is None for entry in group["expert_row_distribution"]))
        self.assertTrue(all(entry["expert_fraction"] is None for entry in group["expert_row_distribution"]))
        self.assertEqual(sum(entry["experts"] for entry in group["expert_row_distribution"]), 0)

    def test_tail_bin_carries_the_reported_tail_routes(self):
        counts = [0] * 17
        counts[16] = 1
        group = summary_for([native_record(20, counts, 20)])["groups"][0]
        self.assertEqual(group["owned_routes"], 20)
        tail = group["expert_row_distribution"][16]
        self.assertEqual(tail["expert_rows"], "17+")
        self.assertEqual(tail["routes"], 20)
        self.assertAlmostEqual(tail["route_fraction"], 1.0)

    def test_multiple_records_sum_owned_and_total_routes(self):
        subset = [0] * 17
        subset[0] = 5
        subset[1] = 1
        full = [0] * 17
        full[1] = 6
        group = summary_for([native_record(2, subset, 0), native_record(2, full, 0)])["groups"][0]
        self.assertEqual(group["owned_routes"], 7 + 12)
        self.assertEqual(group["total_request_routes"], 24)
        self.assertEqual(group["samples"], 2)

    def test_declared_owned_routes_must_match_when_present(self):
        counts = [0] * 17
        counts[0] = 5
        counts[1] = 1
        group = summary_for([native_record(2, counts, 0, extra={"owned_routes": 7})])["groups"][0]
        self.assertEqual(group["owned_routes"], 7)
        with self.assertRaises(ValueError):
            summary_for([native_record(2, counts, 0, extra={"owned_routes": 8})])

    def test_malformed_histograms_are_still_rejected(self):
        cases = []
        over_count = [0] * 17
        over_count[1] = 7  # owned 14 > rows * 6 = 12
        cases.append(("over-count", native_record(2, over_count, 0)))
        mismatched_active = [0] * 17
        mismatched_active[0] = 6
        cases.append(("active mismatch", native_record(2, mismatched_active, 0, active_experts=5)))
        bin_beyond_rows = [0] * 17
        bin_beyond_rows[2] = 1  # three rows for one expert on a two-row request
        cases.append(("bin beyond rows", native_record(2, bin_beyond_rows, 1)))
        tail_below_bound = [0] * 17
        tail_below_bound[16] = 1
        cases.append(("tail below bound", native_record(2, tail_below_bound, 0)))
        cases.append(("fractional active", native_record(2, [1] + [0] * 16, 0, active_experts=1.5)))
        cases.append(("fractional rows", native_record(2.5, [1] + [0] * 16, 0)))
        cases.append(("non-finite rows", native_record("1e309", [1] + [0] * 16, 0)))
        # A fractional or non-finite tail must be rejected before any int()
        # truncation, so the distribution denominator cannot disagree with the
        # validation value.
        cases.append(("fractional tail", native_record(2, [1] + [0] * 16, 0.5)))
        cases.append(("non-finite tail", native_record(2, [1] + [0] * 16, "1e309")))
        for name, line in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                summary_for([line])

    def test_no_histogram_record_still_summarizes(self):
        line = (
            "2026-01-01T00:00:00Z INFO ds41rt::timing: layer=1 rows=8 upload_us=1 "
            "kernel_us=2 compact_us=3 download_us=4 total_us=5 native expert execution"
        )
        group = summary_for([line])["groups"][0]
        self.assertNotIn("expert_row_distribution", group)
        self.assertNotIn("owned_routes", group)


if __name__ == "__main__":
    unittest.main()
