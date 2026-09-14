"""CPU tests load the dependency-free PLE reader without importing Torch."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "third_party/gptqmodel/gptqmodel/utils/ple_mmap.py"
spec = importlib.util.spec_from_file_location("ple_mmap", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PLETest(unittest.TestCase):
    def test_owned_rows_prefetch_release_and_bounds(self):
        header = json.dumps({
            "embed.weight": {"dtype": "F8_E4M3", "shape": [4, 32], "data_offsets": [0, 128]},
            "embed.scale": {"dtype": "F8_E8M0", "shape": [4, 1], "data_offsets": [128, 132]},
        }).encode()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "table.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(range(132)))
            with module.MappedPLETable(path, "embed", max_gather_rows=3) as table:
                table.prefetch([3, 1, 3])
                weights, scales = table.gather([3, 1, 3])
                table.release()
                self.assertEqual(weights, bytes(range(96, 128)) + bytes(range(32, 64)) + bytes(range(96, 128)))
                self.assertEqual(scales, bytes([131, 129, 131]))
                self.assertEqual(table.gather([]), (b"", b""))
                with self.assertRaises(IndexError):
                    table.gather([-1])
                with self.assertRaises(IndexError):
                    table.gather([4])
                with self.assertRaises(ValueError):
                    table.gather(iter([0] * 4))
            self.assertEqual(scales, bytes([131, 129, 131]))
            with self.assertRaises(RuntimeError):
                table.gather([0])


if __name__ == "__main__":
    unittest.main()
