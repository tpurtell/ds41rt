from pathlib import Path
import sys
import tempfile
import unittest

import torch
from gptqmodel.utils.v41_checkpoint import save_frontier
from gptqmodel.utils.v41_replay import V41ReplayBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from retention import retire_block_temporaries
from run_store import RunStore


class RetentionTest(unittest.TestCase):
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
