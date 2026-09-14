from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coordinator import exclusive_run, run_namespace, quantize_namespaces


class CoordinatorTest(unittest.TestCase):
    def test_completed_namespace_resume_does_not_recreate_retired_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [dict(id="a", input_ids=(1, 2, 3, 4))]
            attestation = {"test": True}
            initial = dict(output_keys=("old-input",), output_provenance={}, next_layer=0)
            draft_initial = dict(output_keys=("old-draft",), output_provenance={}, next_layer=0)
            main, draft = {"stage": "main"}, {"stage": "draft"}
            from draft_anchors import select_anchors
            stored = {"inputs/inventory": dict(records=records, attestation=attestation),
                      "inputs/complete": initial, "draft-inputs/complete": draft_initial,
                      "draft-inputs/inventory": dict(selection=select_anchors(records, count=2), main_frontier=main)}
            driver = SimpleNamespace(journal=SimpleNamespace(root=Path(directory)),
                _load=lambda key, kind: stored.get(key),
                _publish=lambda key, kind, state, parents: stored.__setitem__(key, state))
            def forbidden():
                raise AssertionError("must not recreate consumed input adapters")
            with patch("coordinator.run_namespace", side_effect=[main, draft]) as run:
                result = quantize_namespaces(driver, records, attestation, main_adapters=forbidden,
                    draft_adapters=forbidden, native_kernels=None, anchor_count=2)
                self.assertEqual([call.args[1] for call in run.call_args_list], ["base", "mtp"])
                self.assertEqual(result["status"], "namespaces-quantized-export-pending")
                self.assertIn("quantization/complete", stored)

    def test_exclusive_lock_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with exclusive_run(root):
                with self.assertRaisesRegex(RuntimeError, "another coordinator"):
                    with exclusive_run(root):
                        pass
            with exclusive_run(root):
                pass

    def test_resume_after_block_commit_before_retirement(self):
        initial = dict(output_keys=("input",), output_provenance={"input": {}}, next_layer=0)
        records, processed, loaded, retired = {}, [], [], []
        latest = [None]
        fail = [True]
        class FakeWavefront:
            def __init__(self, driver):
                pass
            def latest(self, namespace):
                return latest[0]
            def process(self, block, namespace, layer, keys, *, input_provenance, replica=None):
                processed.append(layer)
                key = f"output-{layer}"
                latest[0] = dict(output_keys=(key,), output_provenance={key: {}}, next_layer=layer + 1)
                return latest[0]
        def load(layer, device, **kwargs):
            loaded.append((layer, kwargs["namespace"]))
            return object()
        def retire(driver, namespace, layer):
            if fail[0]:
                raise RuntimeError("retirement interrupted")
            retired.append(layer)
            return 0
        driver = SimpleNamespace(source=SimpleNamespace(load_decoded_block=load), device=torch.device("cpu"),
            _load=lambda key, kind: records.get(key),
            _publish=lambda key, kind, state, parents: records.__setitem__(key, state), progress=lambda event: None)
        with patch("coordinator.NAMESPACES", {"base": ("layers", 3, 4)}), \
             patch("coordinator.Wavefront", FakeWavefront), \
             patch("coordinator.retire_block_temporaries", side_effect=retire):
            with self.assertRaisesRegex(RuntimeError, "retirement interrupted"):
                run_namespace(driver, "base", initial, native_kernels=None)
            self.assertEqual(processed, [0])
            fail[0] = False
            result = run_namespace(driver, "base", initial, native_kernels=None)
            self.assertEqual(processed, [0, 1, 2])
            self.assertEqual(loaded, [(0, "layers"), (1, "layers"), (2, "layers")])
            self.assertEqual(retired, [0, 1, 2])
            self.assertEqual(result["next_layer"], 3)
            self.assertEqual(run_namespace(driver, "base", initial, native_kernels=None), result)
            self.assertEqual(processed, [0, 1, 2])
            with self.assertRaisesRegex(ValueError, "initial frontier changed"):
                run_namespace(driver, "base", {**initial, "output_keys": ("changed",)}, native_kernels=None)
