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
    def by_name(self, name):
        return next(c for c in CHART["CONFIGS"] if c["speed_columns"][0][1] == name)

    def test_embedded_compact_and_dual_source_values(self):
        evidence = json.loads((ROOT / "docs/release-v7-exl3-compact-evidence.json").read_text())
        published = (ROOT / "docs/release-v7-published-memory-evidence.log").read_text()
        compact = self.by_name("EXL3 5090+2-spark")
        for b in compact["devices"][0]["bands"][:3]:
            self.assertIn(str(b["bytes"]), published)
            self.assertEqual(b["status"], "logged")
        dual = self.by_name("EXL3 2x6000 0-spark")
        for device in dual["devices"]:
            for b in device["bands"][:3]:
                self.assertIn(str(b["bytes"]), evidence["logs"]["dual-regression.log"])
            self.assertEqual(device["bands"][0]["status"], "plan")

    def test_official_sources_and_padded_spark_geometry(self):
        launches = json.loads((ROOT / "docs/release-v6-performance.json").read_text())["launches"]
        for count, layout in [(1, "single"), (2, "dual")]:
            cfg = self.by_name(f"Official {count}x")
            dep = launches[layout]["deployment"]
            self.assertEqual(cfg["devices"][-1]["bands"][0]["bytes"],
                             384 * 5120 * 640 * 51 // 32 * dep["spark_layers"])
            for dev in cfg["devices"][:count]:
                self.assertEqual(dev["bands"][0]["bytes"],
                                 7219445760 * dep["rtx_expert_layers"] // count)
                self.assertEqual(dev["bands"][-1]["status"], "estimate")
            self.assertNotIn("NVFP4", str(cfg))

    def test_six_profiles_and_estimate_provenance(self):
        headline = runpy.run_path(str(ROOT / "scripts/render-ds41-v7-headline.py"))
        self.assertEqual([c["speed_columns"][0][1] for c in CHART["CONFIGS"]],
                         [c[0] for c in headline["CONFIGS"]])
        for cfg in CHART["CONFIGS"]:
            for dev in cfg["devices"]:
                self.assertTrue(dev["bands"])
                for b in dev["bands"]:
                    self.assertIsNotNone(b["bytes"])
                    self.assertTrue(b["source"])
                    if b["status"] == "estimate":
                        self.assertTrue(b["basis"])
                if "Spark" in dev["label"]:
                    self.assertEqual(dev["bands"][1]["bytes"], 0)
                if dev["occupied"] is not None:
                    self.assertEqual(sum(b["bytes"] for b in dev["bands"]), dev["occupied"])
        compact = self.by_name("EXL3 5090+2-spark")["devices"][0]
        self.assertLessEqual(compact["occupied"] + compact["headroom"], 32 * CHART["GIB"])

    def test_single_nvfp4_raw_source_when_available(self):
        path = ROOT / "runs/v7q-a1/single-nvfp4-coordinator.log"
        if not path.exists():
            self.skipTest("local historical startup log unavailable")
        log = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text())
        for b in self.by_name("NVFP4 1x")["devices"][0]["bands"][:3]:
            self.assertIn(str(b["bytes"]), log)

    def test_missing_package_no_v7_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            values, _docs = CHART["load_measurements"](Path(directory))
            for column in ["EXL3 5090+2-spark", "EXL3 2x6000 0-spark", "NVFP4 1x", "NVFP4 2x"]:
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
            self.assertEqual(values["EXL3 5090+2-spark"]["C1 code decode"], 12)
            self.assertIsNone(values["EXL3 5090+2-spark"]["Prefill"])
            svg = CHART["render"](package)
            CHART["self_check"](svg)
            self.assertIn("0/3 decode checks; failed samples included", svg)
            self.assertNotIn("9,999", svg)


if __name__ == "__main__":
    unittest.main()
