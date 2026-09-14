from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draft_anchors import select_anchors, splitmix64


class AnchorTest(unittest.TestCase):
    def test_strata_coordinates_and_full_selection(self):
        records = [{"input_ids": tuple(range(length))} for length in (2, 9, 4, 1, 15)]
        selected = select_anchors(records, count=8)
        self.assertEqual(selected, select_anchors(records, count=8))
        self.assertEqual(selected["proposal_rows"], 40)
        flat, offset = [], 0
        for record, positions in zip(records, selected["positions"]):
            self.assertEqual(list(positions), sorted(set(positions)))
            for position in positions:
                self.assertGreaterEqual(position, 1)
                self.assertLess(position + 1, len(record["input_ids"]))
                flat.append(offset + position - 1)
            offset += max(len(record["input_ids"]) - 2, 0)
        for index, ordinal in enumerate(flat):
            self.assertTrue(index * offset // 8 <= ordinal < (index + 1) * offset // 8)
        full = select_anchors(records, count=offset)
        self.assertEqual(full["positions"], tuple(tuple(range(1, len(row["input_ids"]) - 1)) for row in records))
        with self.assertRaisesRegex(ValueError, "no automatic resizing"):
            select_anchors(records)
        self.assertEqual(splitmix64(0), 0xE220A8397B1DCDAF)
