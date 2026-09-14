from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from gptqmodel.utils.v41_checkpoint import save_frontier
from gptqmodel.utils.v41_dspark import V41DSparkInput
from gptqmodel.utils.v41_replay import V41ReplayBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from draft_inputs import prepare_draft_frontiers
from run_store import RunStore


class DraftInputTest(unittest.TestCase):
    def test_interrupted_handoff_keeps_joint_groups_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = RunStore(directory, {"test": "draft-handoff"})
            self.addCleanup(journal.close)
            driver = BlockDriver(None, journal, {"test": "draft-handoff"}, device="cpu")
            records = [dict(id=str(i), input_ids=tuple(range(8 + i))) for i in range(2)]
            driver._publish("inputs/inventory", "inventory", {"records": records}, ())
            main = dict(output_keys=("main/0", "main/1"), output_provenance={}, next_layer=40)
            for index, key in enumerate(main["output_keys"]):
                length = len(records[index]["input_ids"])
                prov = {"source": index}
                state = V41ReplayBatch(40, torch.zeros(1, length, 2, 8), torch.zeros(1, length, 2), {}, {}, {},
                                      (7, 3), {layer: torch.randn(1, length, 8) for layer in (7, 3)})
                path = Path(directory) / f"main-{index}.safetensors"
                save_frontier(state, path, provenance=prov)
                journal.record_file(key, "replay", path)
                main["output_provenance"][key] = prov
            driver._publish("namespaces/base/complete", "namespace", main, main["output_keys"])
            config = SimpleNamespace(dspark_target_layer_ids=[7, 3], dspark_block_size=5,
                                     dspark_noise_token_id=15, hc_mult=2)
            adapters = [V41DSparkInput(torch.nn.Embedding(16, 8), torch.nn.Linear(16, 8, bias=False),
                        torch.nn.LayerNorm(8), config) for _ in range(2)]
            adapters[1].load_state_dict(adapters[0].state_dict())
            events = []
            def interrupted(event):
                events.append(event)
                raise RuntimeError("interrupted after durable draft input")
            driver.progress = interrupted
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                prepare_draft_frontiers(driver, main, records, adapters, count=7)
            self.assertIsNone(journal.get("draft-inputs/complete"))
            driver.progress = events.append
            result = prepare_draft_frontiers(driver, main, records, adapters, count=7)
            self.assertEqual(len(events), 2)
            self.assertEqual(len(result["output_keys"]), 2)
            self.assertEqual(prepare_draft_frontiers(driver, main, records, adapters, count=7), result)
            self.assertEqual(len(events), 2)
            with self.assertRaisesRegex(ValueError, "selection or main frontier changed"):
                prepare_draft_frontiers(driver, main, records, adapters, count=8)
