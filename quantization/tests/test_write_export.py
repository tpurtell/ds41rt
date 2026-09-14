import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stream_shard import read_header
from write_export import write_export, _write_shard


class WriteExportTest(unittest.TestCase):
    def test_complete_source_shard_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source.safetensors", Path(directory) / "linked.safetensors"
            name = "layers.1.engram.embed.weight"
            save_file({name: torch.ones(32, dtype=torch.int8)}, source)
            entries = {name: dict(path=str(source), tensor=name, dtype="I8", shape=[32], bytes=32)}
            with patch("write_export.repack", side_effect=AssertionError("must not copy")):
                self.assertEqual(_write_shard(destination, [name], entries), "hard-linked")
            self.assertEqual(source.stat().st_ino, destination.stat().st_ino)

    def test_interrupted_explicit_resume_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            tensors = {f"layers.{layer}.engram.embed.{suffix}": torch.ones(length, dtype=torch.int8)
                       for layer in (1, 14) for suffix, length in (("weight", 100), ("scale", 8))}
            tensors["embed.weight"] = torch.arange(12, dtype=torch.bfloat16)
            save_file(tensors, source)
            header, _ = read_header(source)
            inventory = dict(tensors={name: dict(path=str(source), tensor=name, dtype=item["dtype"],
                                                shape=item["shape"], bytes=item["data_offsets"][1]-item["data_offsets"][0])
                                      for name, item in header.items()}, tiers={})
            output, state = root / "artifact", root / "state"
            def stop(event):
                raise RuntimeError("injected interruption")
            kwargs = dict(identity={"test": 1}, target_bytes=32,
                          metadata={"config.json": {"model_type": "deepseek_v41"},
                                    "quantize_config.json": {"quant_method": "exl3"}})
            asset = root / "tokenizer.json"
            asset.write_bytes(b"unchanged asset\n")
            kwargs["assets"] = {"reference/tokenizer.json": dict(path=str(asset), bytes=asset.stat().st_size,
                                                               sha256=hashlib.sha256(asset.read_bytes()).hexdigest())}
            with self.assertRaisesRegex(RuntimeError, "injected"):
                write_export(inventory, output, state, progress=stop, **kwargs)
            self.assertFalse((output / "model.safetensors.index.json").exists())
            first, = output.iterdir()
            old_stat = first.stat()
            with self.assertRaisesRegex(ValueError, "explicit resume"):
                write_export(inventory, output, state, **kwargs)
            report = write_export(inventory, output, state, resume=True, **kwargs)
            self.assertEqual(report["tensors"], 5)
            self.assertEqual(first.stat().st_ino, old_stat.st_ino)
            self.assertEqual(first.stat().st_mtime_ns, old_stat.st_mtime_ns)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            for filename, value in kwargs["metadata"].items():
                self.assertEqual(json.loads((output / filename).read_text()), value)
            self.assertEqual((output / "reference/tokenizer.json").read_bytes(), asset.read_bytes())
            for name, filename in index["weight_map"].items():
                torch.testing.assert_close(load_file(output / filename)[name], tensors[name], rtol=0, atol=0)
            with patch("write_export.repack", side_effect=AssertionError("must not recopy")):
                self.assertEqual(write_export(inventory, output, state, resume=True, **kwargs), report)
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                write_export(inventory, output, state, resume=True, identity={"test": 2}, target_bytes=32)
            # Truncation must fail rather than silently replacing a published shard.
            with first.open("r+b") as stream:
                stream.truncate(first.stat().st_size - 1)
            with self.assertRaises(ValueError):
                write_export(inventory, output, state, resume=True, **kwargs)
