"""Real checkpoint header contract, without allocating checkpoint weights."""
from pathlib import Path
import unittest

import torch
from safetensors import safe_open
from transformers import DeepseekV41Config
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41DecoderLayer
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Experts
from gptqmodel.utils.v41_source import V41Source, source_layer_key


SNAPSHOT = Path("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
                "dba1be0a40aa45a94ad051997016db3960a90277")


class V41SourceTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
