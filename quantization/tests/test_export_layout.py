from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_layout import plan_shards, tensor_group
from stream_shard import read_header, repack


class ExportLayoutTest(unittest.TestCase):
    def test_standard_shards_and_independent_ple_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = {f"layers.{layer}.engram.embed.{suffix}": torch.arange(size, dtype=torch.int8)
                       for layer in (1, 14) for suffix, size in (("weight", 100), ("scale", 8))}
            tensors["embed.weight"] = torch.ones(12, dtype=torch.bfloat16)
            source = root / "source.safetensors"
            save_file(tensors, source)
            header, _ = read_header(source)
            inventory = {name: dict(bytes=item["data_offsets"][1] - item["data_offsets"][0])
                         for name, item in header.items()}
            plan = plan_shards(inventory, target_bytes=32)
            self.assertEqual(len(plan["files"]), 5)
            self.assertEqual(plan["index"]["metadata"]["total_size"], 240)
            for filename, names in plan["files"].items():
                self.assertEqual(len({tensor_group(name) for name in names}), 1)
                repack({name: (source, name) for name in names}, root / filename)
                loaded = load_file(root / filename)
                self.assertEqual(set(loaded), set(names))
                for name in names:
                    self.assertEqual(plan["index"]["weight_map"][name], filename)
                    torch.testing.assert_close(loaded[name], tensors[name], rtol=0, atol=0)
            changed = plan_shards({**inventory, "extra.weight": dict(bytes=500)}, target_bytes=32)
            for name in inventory:
                if tensor_group(name) != "model":
                    self.assertEqual(plan["index"]["weight_map"][name], changed["index"]["weight_map"][name])
            with self.assertRaisesRegex(ValueError, "incomplete PLE"):
                plan_shards({"embed.weight": dict(bytes=10)})
            with self.assertRaisesRegex(ValueError, "unexpected PLE"):
                tensor_group("layers.2.engram.embed.weight")
