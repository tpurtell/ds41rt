"""Exhaust the reviewed h16/d512 sparse-gather CUDA shared-memory mapping.

Transcribed from the frozen-image generated device kernel, not a replacement
kernel. The warning branch has only stores: its 32 participating lanes each
write 128 eight-BF16 vectors into a 64x512 shared tile before a block barrier.
"""
import unittest


class SparseGatherLayoutTest(unittest.TestCase):
    def test_each_shared_element_has_exactly_one_writer(self):
        offsets = []
        for thread in range(256):
            if (thread & 31) >> 2 != 0:
                continue
            for i in range(128):
                offset = (((i & 63) >> 3) * 4096 + (thread >> 5) * 512
                          + (thread & 3) * 128 + (i >> 6) * 64
                          + ((((i & 7) >> 2) + ((thread & 3) >> 1)) & 1) * 32
                          + ((((i & 3) >> 1) + (thread & 1)) & 1) * 16
                          + (((i >> 6) + (i & 1)) & 1) * 8)
                self.assertEqual(offset % 8, 0)
                offsets.extend(range(offset, offset + 8))
        self.assertEqual(len(offsets), 64 * 512)
        self.assertEqual(len(set(offsets)), len(offsets))
        self.assertEqual(set(offsets), set(range(64 * 512)))
