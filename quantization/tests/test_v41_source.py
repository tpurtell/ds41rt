"""Real checkpoint header contract, without allocating checkpoint weights."""
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import DeepseekV41Config
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Experts
from gptqmodel.utils.v41_source import V41Source, source_layer_key


SNAPSHOT = Path("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
                "dba1be0a40aa45a94ad051997016db3960a90277")


class V41SourceTest(unittest.TestCase):
    def test_packed_reload_rejects_native_mix_and_invalid_geometry(self):
        source = V41Source.__new__(V41Source)
        name = "layers.0.ffn.experts.0.w1"
        packed = dict(trellis=torch.zeros(320, 144, 48, dtype=torch.int16),
                      suh=torch.ones(5120, dtype=torch.float16),
                      svh=torch.ones(2304, dtype=torch.float16), mcg=torch.tensor(1, dtype=torch.int32))
        source._projection_buffers = {name: set(packed)}
        with patch.object(source, "tensor", side_effect=lambda key, device: packed[key.rsplit(".", 1)[1]]):
            self.assertEqual(set(source.packed_projection(name)), set(packed))
            source._projection_buffers[name].add("weight")
            with self.assertRaisesRegex(ValueError, "exactly four"):
                source.packed_projection(name)
            source._projection_buffers[name].remove("weight")
            packed["svh"] = torch.ones(2303, dtype=torch.float16)
            with self.assertRaisesRegex(ValueError, "geometry"):
                source.packed_projection(name)
            with self.assertRaisesRegex(ValueError, "restricted"):
                source.packed_projection("layers.0.ffn.shared_experts.w1")
            with self.assertRaisesRegex(ValueError, "outside"):
                source.packed_projection("mtp.3.ffn.experts.0.w1")

    def test_all_main_block_names_and_shapes(self):
        source = V41Source(SNAPSHOT)
        config = DeepseekV41Config.from_dict(source.config).get_text_config()
        config._experts_implementation = "eager"
        config._attn_implementation = "eager"
        shapes = {}
        for shard_name in set(source.weight_map.values()):
            with safe_open(SNAPSHOT / shard_name, framework="pt") as shard:
                for key in shard.keys():
                    item = shard.get_slice(key)
                    shape = item.get_shape()
                    if item.get_dtype() == "I8" and ".ffn.experts." in key and key.endswith(".weight"):
                        shape[-1] *= 2
                    shapes[key] = tuple(shape)
        mapped = set()
        for index in range(config.num_hidden_layers):
            with torch.device("meta"):
                block = DeepseekV41DecoderLayer(config, index)
                block.mlp.experts = DeepSeekV41Experts.from_fused(block.mlp.experts)
            for key, value in block.state_dict().items():
                name = source_layer_key(index, key)
                self.assertIn(name, shapes)
                self.assertEqual(tuple(value.shape), shapes[name], name)
                self.assertNotIn(name, mapped)
                mapped.add(name)
        expected = {key for key in source.weight_map if key.startswith("layers.")
                    and not key.endswith(".scale") and ".engram.embed." not in key}
        self.assertEqual(mapped, expected)

    def test_ple_full_read_rejected(self):
        source = V41Source(SNAPSHOT)
        for index in (1, 14):
            with self.assertRaises(ValueError):
                source.tensor(f"layers.{index}.engram.embed.weight")

    def test_dspark_core_names_shapes_and_routed_count(self):
        source = V41Source(SNAPSHOT)
        config = source.block_config("mtp")
        self.assertEqual((config.num_hidden_layers, config.n_routed_experts, config.num_experts_per_tok),
                         (3, 128, 3))
        shapes = {}
        shards = {shard for key, shard in source.weight_map.items() if key.startswith("mtp.")}
        for shard_name in shards:
            with safe_open(SNAPSHOT / shard_name, framework="pt") as shard:
                for key in shard.keys():
                    item = shard.get_slice(key)
                    shape = item.get_shape()
                    if item.get_dtype() == "I8" and ".ffn.experts." in key and key.endswith(".weight"):
                        shape[-1] *= 2
                    shapes[key] = tuple(shape)
        mapped = set()
        for stage in range(3):
            with torch.device("meta"):
                block = DeepseekV41DecoderLayer(config, stage)
                block.mlp.experts = DeepSeekV41Experts.from_fused(block.mlp.experts)
            self.assertIsNone(block.engram)
            self.assertIsNone(block.self_attn.indexer)
            for key, value in block.state_dict().items():
                name = source_layer_key(stage, key, "mtp")
                self.assertEqual(tuple(value.shape), shapes[name], name)
                mapped.add(name)
        core = {key for key in source.weight_map if key.startswith("mtp.")
                and key.split(".")[2] in ("attn", "ffn", "attn_norm", "ffn_norm")
                and not key.endswith(".scale")}
        core |= {key for key in source.weight_map if key.startswith("mtp.") and key.split(".")[2].startswith("hc_")}
        self.assertEqual(mapped, core)
        self.assertEqual(sum(".ffn.experts." in key for key in mapped), 3 * 128 * 3)


if __name__ == "__main__":
    unittest.main()
