from pathlib import Path
import sys
import unittest
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_config import model_metadata
from mixed_recipe import NAMESPACES, layer_quotas
from validate_export import validate_inventory


class ExportConfigTest(unittest.TestCase):
    def test_complete_mixed_storage_and_native_metadata(self):
        tiers, tensors, native = {}, {}, {}
        for namespace, (prefix, layers, experts) in NAMESPACES.items():
            for layer in range(layers):
                quota = layer_quotas(namespace, layer)
                for expert in range(experts):
                    for projection in ("w1", "w3", "w2"):
                        name = f"{prefix}.{layer}.ffn.experts.{expert}.{projection}"
                        bits = 4 if expert < quota[projection] else 3
                        tiers[name] = bits
                        inputs, outputs = (2304, 5120) if projection == "w2" else (5120, 2304)
                        native[name + ".weight"] = dict(dtype="I8", shape=[outputs, inputs // 2], bytes=inputs * outputs // 2)
                        native[name + ".scale"] = dict(dtype="F8_E8M0", shape=[outputs // 32, inputs // 32], bytes=inputs * outputs // 1024)
                        for suffix, dtype, shape in (("trellis", "I16", [inputs // 16, outputs // 16, bits * 16]),
                                                     ("suh", "F16", [inputs]), ("svh", "F16", [outputs]), ("mcg", "I32", [])):
                            tensors[name + "." + suffix] = dict(dtype=dtype, shape=shape, bytes=math.prod(shape) * (4 if dtype == "I32" else 2))
        tensors["embed.weight"] = native["embed.weight"] = dict(dtype="BF16", shape=[2, 5120], bytes=20480)
        validate_inventory(tensors, native, tiers)
        tensors["embed.weight"] = {**tensors["embed.weight"], "bytes": 20482}
        with self.assertRaisesRegex(ValueError, "unchanged source"):
            validate_inventory(tensors, native, tiers)
        tensors["embed.weight"] = native["embed.weight"]
        source = dict(model_type="deepseek_v41", quantization_config={"quant_method": "fp8", "expert_dtype": "fp4"}, text_config={"hidden_size": 5120})
        output = model_metadata(source, dict(tiers=tiers, tensors=tensors), provenance={"run": "test"})
        config, external = output["config.json"], output["quantize_config.json"]
        self.assertEqual(config["quantization_config"]["bits"], 3)
        self.assertEqual(config["quantization_config"]["quant_method"], "exl3")
        self.assertEqual(len(external["tensor_storage"]), 47232)
        self.assertEqual(sum(entry["bits_per_weight"] == 4 for entry in external["tensor_storage"].values()), 11808)
        self.assertEqual(external["meta"]["ds41rt"]["native_quantization_config"], source["quantization_config"])
        self.assertEqual(source["quantization_config"]["quant_method"], "fp8")
        self.assertEqual(config["text_config"], source["text_config"])
        tensors["mtp.2.ffn.experts.127.w2.trellis"]["shape"][-1] = 64
        with self.assertRaisesRegex(ValueError, "selected tier"):
            model_metadata(source, dict(tiers=tiers, tensors=tensors), provenance={"run": "test"})
