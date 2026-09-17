"""Ensure README and report receive the identical measured table block."""

from pathlib import Path
import runpy
import unittest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "update-ds41-v6-release-docs.py"))


class V6ReleaseDocsTest(unittest.TestCase):
    def test_exact_table_block_and_surrounding_readme_are_preserved(self):
        original = "# Product\n\n## Performance\n\nold tables\n\n## Getting started\n\nkeep this\n"
        tables = "RTX measurements use **400 W per card**.\n\n| A | B |\n|---|---:|\n| x | 1 |\n"
        updated = MODULE["update_readme"](original, tables)
        self.assertEqual(updated.count(tables.rstrip()), 1)
        self.assertIn("# Product", updated)
        self.assertIn("## Getting started\n\nkeep this", updated)
        self.assertNotIn("old tables", updated)

    def test_report_uses_same_tables_and_provenance(self):
        summary = dict(release="v6", performance_matrix_passed=True, engine_commit="engine",
                       sparkinfer_commit="spark", model_id="owner/model", model_revision="revision")
        tables = "RTX measurements use **400 W per card**.\n"
        report = MODULE["performance_report"](summary, tables)
        self.assertEqual(report.count(tables.rstrip()), 1)
        self.assertIn("`engine`", report)
        self.assertIn("`owner/model@revision`", report)


if __name__ == "__main__":
    unittest.main()
