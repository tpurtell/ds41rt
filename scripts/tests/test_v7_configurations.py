"""Guard chart provenance, missing measurements and accounting boundaries."""
import json
from pathlib import Path
import re
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHART = runpy.run_path(str(ROOT / "scripts/render-ds41-v7-configurations.py"))


class V7ConfigurationsTests(unittest.TestCase):
    def test_embedded_compact_and_dual_source_values(self):
        evidence = json.loads((ROOT / "docs/release-v7-exl3-compact-evidence.json").read_text())
        configs = CHART["CONFIGS"]
        logs = evidence["logs"]
        compact = logs["compact-final-residency.log"]
        for value in configs[3]["devices"][0][2]:
            self.assertIn(str(value), compact)
        for worker in ["compact-final-ostrich.log", "compact-final-dodo.log"]:
            for value in configs[3]["devices"][1][2]:
                if value is not None:
                    self.assertIn(str(value), logs[worker])
        for device in configs[2]["devices"]:
            for value in device[2]:
                self.assertIn(str(value), logs["dual-regression.log"])
        # Rank peak is the expert loading plan; transport and KV are separate.
        self.assertIn("rank_peak_bytes", configs[2]["sources"][0]["fields"])

    def test_single_nvfp4_raw_source_when_available(self):
        path = ROOT / CHART["CONFIGS"][0]["sources"][0]["path"]
        if not path.exists():
            self.skipTest("historical local NVFP4 startup log not archived in checkout")
        log = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
        for value in CHART["CONFIGS"][0]["devices"][0][2]:
            self.assertIn(str(value), log)

    def test_unsourced_devices_are_blank_not_estimated(self):
        self.assertEqual(CHART["CONFIGS"][1]["sources"], [])
        for _label, _capacity, bands in CHART["CONFIGS"][1]["devices"]:
            self.assertTrue(all(value is None for value in bands))
        self.assertIsNone(CHART["CONFIGS"][3]["devices"][1][2][1])  # no Spark KV

    def test_missing_package_no_v7_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            values, _docs = CHART["load_measurements"](Path(directory))
            for column in ["EXL3 1x", "EXL3 2x", "NVFP4 1x", "NVFP4 2x"]:
                self.assertTrue(all(v is None for v in values[column].values()))
            svg = CHART["render"](Path(directory))
            CHART["self_check"](svg)
            self.assertIn("—", svg)
            self.assertIn("MXFP4 (v6)", svg)
            self.assertIn("NOT measured free", svg)
            self.assertIn("NOT RTX 5090 hardware", svg)

    def test_raw_medians_and_prefill_completion_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "single-exl3-dspark.json").write_text(json.dumps({"samples": [
                {"case": "code", "observed_decode_tokens_per_second": x, "passed": False}
                for x in [11, 50, 12]]}))
            (package / "single-exl3-prefill.json").write_text(json.dumps({
                "passed": False, "cells": [{"median_effective_prefill_tokens_per_second": 9999}]}))
            values, _docs = CHART["load_measurements"](package)
            self.assertEqual(values["EXL3 1x"]["C1 code decode"], 12)
            self.assertIsNone(values["EXL3 1x"]["Prefill"])
            svg = CHART["render"](package)
            CHART["self_check"](svg)
            self.assertIn("0/3 decode checks; failed samples included", svg)
            self.assertNotIn("9,999", svg)


if __name__ == "__main__":
    unittest.main()
