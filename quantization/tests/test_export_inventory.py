from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_inventory import packed_entries, collect_selected
from mixed_recipe import NAMESPACES, layer_quotas
from gptqmodel.utils.v41_checkpoint import _save_state


class ExportInventoryTest(unittest.TestCase):
    def test_packed_offsets_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.safetensors"
            packed = dict(trellis=torch.zeros((320, 144, 48), dtype=torch.int16),
                          suh=torch.ones(5120, dtype=torch.float16),
                          svh=torch.ones(2304, dtype=torch.float16),
                          mcg=torch.tensor([1], dtype=torch.int32))
            provenance = {"artifact": "test"}
            _save_state(dict(packed=packed), path, provenance=provenance, kind="projection")
            entries = packed_entries(path, provenance=provenance, bits=3, projection="w1")
            self.assertEqual(set(entries), set(packed))
            self.assertEqual(sum(v["bytes"] for v in entries.values()), sum(t.numel() * t.element_size() for t in packed.values()))
            with self.assertRaisesRegex(ValueError, "provenance"):
                packed_entries(path, provenance={"artifact": "changed"}, bits=3, projection="w1")
            with self.assertRaisesRegex(ValueError, "geometry"):
                packed_entries(path, provenance=provenance, bits=4, projection="w1")

    def test_all_main_and_draft_selections(self):
        markers = {}
        for namespace, (_, layers, experts) in NAMESPACES.items():
            for layer in range(layers):
                root = f"blocks/{namespace}/{layer:03d}"
                markers[root + "/complete"] = {}
                quotas = layer_quotas(namespace, layer)
                for phase, projections in (("gate_up", ("w1", "w3")), ("down", ("w2",))):
                    markers[root + f"/{phase}/complete"] = dict(selected=[
                        (expert, projection, f"{root}/{phase}/expert-{expert:03d}/{projection}-k{4 if expert < quotas[projection] else 3}")
                        for expert in range(experts) for projection in projections])
        class Driver:
            def _load(self, key, kind):
                return markers.get(key)
        selected, tiers = collect_selected(Driver())
        self.assertEqual(len(selected), 47232)
        self.assertEqual(sum(bits == 4 for bits in tiers.values()), 11808)
        markers.pop("blocks/mtp/002/complete")
        with self.assertRaisesRegex(ValueError, "incomplete block"):
            collect_selected(Driver())
