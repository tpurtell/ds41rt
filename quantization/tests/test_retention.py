from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from gptqmodel.utils.v41_checkpoint import save_frontier
from gptqmodel.utils.v41_replay import V41ReplayBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from retention import retire_block_temporaries, retire_namespace_frontier
from run_store import RunStore


class RetentionTest(unittest.TestCase):
    def final_fixture(self, directory, namespace):
        journal = RunStore(directory, {"test": "final-retention"})
        self.addCleanup(journal.close)
        driver = BlockDriver(None, journal, {"test": "final-retention"}, device="cpu")
        def frontier(key, layer, parents=()):
            prov = {"test": key}
            path = Path(directory) / (key.replace("/", "-") + ".safetensors")
            save_frontier(V41ReplayBatch(layer, torch.zeros(1, 2, 1, 2),
                          torch.ones(1, 2, 1), {}, {}, {}), path, provenance=prov)
            journal.record_file(key, "replay", path, parents=parents)
            return dict(next_layer=layer, output_keys=(key,), output_provenance={key: prov})
        complete = frontier("final-output", 40 if namespace == "base" else 3)
        complete_key = f"namespaces/{namespace}/complete"
        driver._publish(complete_key, "namespace", complete, complete["output_keys"])
        initial = {**complete, "next_layer": 0}
        driver._publish(f"namespaces/{namespace}/inputs", "namespace-inputs", initial, ())
        if namespace == "base":
            driver._publish("draft-inputs/inventory", "inventory", {"main_frontier": complete}, (complete_key,))
            downstream = frontier("draft-input", 0, ("draft-inputs/inventory", "final-output"))
            driver._publish("draft-inputs/complete", "inputs", downstream, downstream["output_keys"])
        return driver, complete, initial

    def test_final_frontier_retirement_and_namespace_resume(self):
        from coordinator import run_namespace
        for namespace in ("base", "mtp"):
            with self.subTest(namespace=namespace), tempfile.TemporaryDirectory() as directory:
                driver, complete, initial = self.final_fixture(directory, namespace)
                self.assertEqual(retire_namespace_frontier(driver, namespace), complete)
                with self.assertRaisesRegex(ValueError, "retired"):
                    driver.journal.get("final-output")
                with patch("coordinator.Wavefront", side_effect=AssertionError("must not replay")):
                    self.assertEqual(run_namespace(driver, namespace, initial, native_kernels=None), complete)
                if namespace == "base":
                    self.assertIsNotNone(driver.journal.get("draft-input"))

    def test_interrupted_final_retirement_resumes_without_loading_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, complete, _ = self.final_fixture(directory, "mtp")
            with patch.object(Path, "unlink", side_effect=OSError("interrupted deletion")):
                with self.assertRaisesRegex(OSError, "interrupted deletion"):
                    retire_namespace_frontier(driver, "mtp")
            self.assertIsNotNone(driver._load("namespaces/mtp/retirement", "namespace-retirement"))
            with patch("retention.load_frontier", side_effect=AssertionError("already authorized")):
                self.assertEqual(retire_namespace_frontier(driver, "mtp"), complete)
                self.assertEqual(retire_namespace_frontier(driver, "mtp"), complete)

    def test_corrupt_handoff_does_not_authorize_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            driver, _, _ = self.final_fixture(directory, "base")
            path = Path(directory) / driver.journal.get("draft-input")["path"]
            with path.open("r+b") as stream:
                stream.truncate(8)
            with self.assertRaises(ValueError):
                retire_namespace_frontier(driver, "base")
            self.assertIsNone(driver._load("namespaces/base/retirement", "namespace-retirement"))
            self.assertIsNotNone(driver.journal.get("final-output"))

    def test_keeps_selected_weights_and_current_frontier(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = RunStore(directory, {"test": "retention"})
            self.addCleanup(journal.close)
            driver = BlockDriver(None, journal, {"test": "retention"}, device="cpu")
            root = "blocks/base/000"
            driver._publish("inputs/0", "replay", {"old": True}, ())
            driver._publish(root + "/input-inventory", "inventory", {"keys": ("inputs/0",)}, ("inputs/0",))
            driver._publish(root + "/hessian", "hessian", {"H": torch.eye(2)}, ("inputs/0",))
            driver._publish(root + "/unused-k3", "projection", {"packed": {}}, (root + "/hessian",))
            selected = []
            for phase, projections in (("gate_up", ("w1", "w3")), ("down", ("w2",))):
                entries = []
                for projection in projections:
                    key = root + "/" + projection
                    driver._publish(key, "projection", {"packed": {}}, (root + "/hessian",))
                    entries.append((0, projection, key))
                    selected.append(key)
                driver._publish(root + f"/{phase}/complete", "phase", {"selected": tuple(entries)},
                                (root + "/unused-k3", *(entry[2] for entry in entries)))
            key = root + "/outputs/0"
            prov = {"test": "new-frontier"}
            path = Path(directory) / "output.safetensors"
            state = V41ReplayBatch(1, torch.zeros(1, 2, 1, 2), torch.ones(1, 2, 1), {}, {}, {})
            save_frontier(state, path, provenance=prov)
            journal.record_file(key, "replay", path, parents=(root + "/gate_up/complete", root + "/down/complete"))
            driver._publish(root + "/complete", "block", dict(output_keys=(key,), output_provenance={key: prov}),
                            (key, root + "/input-inventory"))
            self.assertGreater(retire_block_temporaries(driver, "base", 0), 0)
            for retained in [key, *selected]:
                self.assertIsNotNone(journal.get(retained))
            for retired in ["inputs/0", root + "/hessian", root + "/unused-k3"]:
                with self.assertRaisesRegex(ValueError, "retired"):
                    journal.get(retired)
            self.assertEqual(retire_block_temporaries(driver, "base", 0), 0)
