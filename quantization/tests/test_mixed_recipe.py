from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixed_recipe import NAMESPACES, RATIO, layer_quotas, validate_source_tiers, mixed_policy


class MixedRecipeTest(unittest.TestCase):
    def test_fork_selector_policy_matches_every_block(self):
        from gptqmodel.utils.exl3_inline_mixed import inline_mixed_policy, INLINE_MIXED_META_KEY
        for namespace, (_, layers, experts) in NAMESPACES.items():
            policy = inline_mixed_policy({INLINE_MIXED_META_KEY: mixed_policy("/tmp/recipe-test", namespace)})
            for layer in range(layers):
                self.assertEqual(policy.layer_quotas(layer_index=layer, layer_count=layers,
                                                     experts_per_layer=experts), layer_quotas(namespace, layer))

    def test_complete_main_and_dspark_quotas(self):
        self.assertEqual(layer_quotas("base", 0), {"w1": 54, "w3": 90, "w2": 144})
        self.assertEqual(layer_quotas("mtp", 0), {"w1": 18, "w3": 30, "w2": 48})
        tiers = {}
        for namespace, (prefix, layers, experts) in NAMESPACES.items():
            for layer in range(layers):
                quotas = layer_quotas(namespace, layer)
                for expert in range(experts):
                    for projection in RATIO:
                        tiers[f"{prefix}.{layer}.ffn.experts.{expert}.{projection}"] = 4 if expert < quotas[projection] else 3
        validate_source_tiers(tiers)
        self.assertEqual(len(tiers), 47232)
        self.assertEqual(sum(tiers.values()) * 4, len(tiers) * 13)
        tiers["mtp.2.ffn.experts.0.w1"] = 3
        with self.assertRaises(ValueError):
            validate_source_tiers(tiers)
        tiers.pop("mtp.2.ffn.experts.0.w1")
        with self.assertRaises(ValueError):
            validate_source_tiers(tiers)
