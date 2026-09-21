"""CPU tests for the paired qualifier's compile geometry (review item H).

The GPU-only tool otherwise has no coverage; these pin the two corrections:
route blocks are a CEIL of whole MOE blocks, and the compile tier bits come
from the checkpoint-resolved family rather than the function defaults.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def load_paired():
    pinned = types.ModuleType('_pinned_sparkinfer')
    pinned.REVISION = 'pinned-for-tests'
    pinned.VERSION = '0.0.0'
    pinned.LOCK_DATA = {}
    # The paired tool imports load_weights from the bench harness by name.
    bench_spec = importlib.util.spec_from_file_location(
        'bench_v41_exl3_tiles', TOOLS / 'bench_v41_exl3_tiles.py')
    bench = importlib.util.module_from_spec(bench_spec)
    sys.modules['bench_v41_exl3_tiles'] = bench
    with patch.dict(sys.modules, {'_pinned_sparkinfer': pinned}):
        bench_spec.loader.exec_module(bench)
        spec = importlib.util.spec_from_file_location(
            'qualify_v41_exl3_paired_under_test', TOOLS / 'qualify_v41_exl3_paired.py')
        paired = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(paired)
    return paired


paired = load_paired()


class PackedRouteBlocksTests(unittest.TestCase):
    def test_blocks_round_up_to_whole_moe_blocks(self):
        for slots in (0, 8, 96, 97, 103, 104, 481):
            with self.subTest(slots=slots):
                self.assertEqual(paired.packed_route_blocks(slots, 8),
                                 (slots + 7) // 8)
        self.assertEqual(paired.packed_route_blocks(97, 8), 13)   # ceil, not floor 12

    def test_invalid_geometry_refuses(self):
        for slots, block in ((-1, 8), (10, 0), (10, -8)):
            with self.subTest(slots=slots, block=block), self.assertRaises(ValueError):
                paired.packed_route_blocks(slots, block)

    def test_the_floor_truncation_is_gone_from_the_source(self):
        source = (TOOLS / 'qualify_v41_exl3_paired.py').read_text()
        self.assertNotIn('[1]//8', source.replace(' ', ''))
        # ...and the compile now takes the family-resolved bits, not defaults.
        self.assertIn('family=fam', source)
        self.assertIn('tier0_bits=bits[0], tier1_bits=bits[1]', source)
        self.assertIn("fam.get('tiers')", source)


if __name__ == '__main__':
    unittest.main()
