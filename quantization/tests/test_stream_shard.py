from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_shard import read_header, repack


class StreamShardTest(unittest.TestCase):
    def test_subset_rename_and_byte_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.safetensors", root / "second.safetensors"
            a = torch.arange(42, dtype=torch.int8).reshape(6, 7)
            b = torch.randn(3, 5, dtype=torch.bfloat16)
            save_file({"a": a, "ignored": torch.ones(7)}, first)
            save_file({"b": b}, second)
            output = root / "output.safetensors"
            result = repack({"renamed": (first, "a"), "b": (second, "b")}, output, chunk_bytes=7)
            tensors = load_file(output)
            torch.testing.assert_close(tensors["renamed"], a, rtol=0, atol=0)
            torch.testing.assert_close(tensors["b"], b, rtol=0, atol=0)
            self.assertEqual(result["bytes"], output.stat().st_size)
            self.assertNotIn("sha256", result)
            self.assertEqual(set(read_header(output)[0]), {"renamed", "b"})
            with self.assertRaises(FileExistsError):
                repack({"renamed": (first, "a")}, output)
            self.assertFalse(list(root.glob("*.partial")))
