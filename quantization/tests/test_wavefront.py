from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import DeepseekV41ForCausalLM
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41QModel
from gptqmodel.utils.v41_replay import V41ReplayBatch
from gptqmodel.utils.v41_checkpoint import save_frontier, load_frontier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from run_store import RunStore
from wavefront import Wavefront
from test_v41_primitives import tiny_config


class WavefrontTest(unittest.TestCase):
    @torch.no_grad()
    def test_output_interruption_resumes_only_uncommitted_batches(self):
        torch.manual_seed(83)
        model = DeepseekV41ForCausalLM(tiny_config()).eval()
        DeepSeekV41QModel.convert_model_structure(model)
        block = model.model.layers[0]
        states = [V41ReplayBatch.prepare(model, torch.randint(0, 128, (2, 17))) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"test": "wavefront"}
            journal = RunStore(root, identity)
            self.addCleanup(journal.close)
            keys, provenance = [], {}
            for index, state in enumerate(states):
                key = f"inputs/{index}"
                provenance[key] = {"input": index}
                path = root / (key + ".safetensors")
                save_frontier(state, path, provenance=provenance[key])
                journal.record_file(key, "replay", path)
                keys.append(key)
            events = []
            fail = [True]
            def progress(event):
                events.append(event)
                if fail[0] and event["event"] == "output_committed":
                    raise RuntimeError("interrupted after durable output")
            driver = BlockDriver(None, journal, identity, device="cpu", progress=progress)
            source_weight = block.mlp.experts[0].down_proj.weight.clone()
            def selected_phase(block, namespace, layer, routed_keys, *, routed_provenance):
                block.mlp.experts[0].down_proj.weight.copy_(source_weight * .5)
                key = "blocks/base/000/down/complete"
                if driver._load(key, "phase") is None:
                    driver._publish(key, "phase", {"mock_selected": True}, tuple(routed_keys))
                return key
            driver.run = selected_phase
            wavefront = Wavefront(driver)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                wavefront.process(block, "base", 0, keys, input_provenance=provenance)
            self.assertIsNone(journal.get("blocks/base/000/complete"))
            self.assertIsNotNone(journal.get("blocks/base/000/outputs/batch-000000"))
            fail[0] = False
            result = wavefront.process(block, "base", 0, keys, input_provenance=provenance)
            self.assertEqual(sum(event["event"] == "routed_committed" for event in events), 2)
            self.assertEqual(sum(event["event"] == "output_committed" for event in events), 2)
            for index, key in enumerate(result["output_keys"]):
                record = journal.get(key)
                loaded = load_frontier(root / record["path"], expected_sha256=record["sha256"],
                                       expected_provenance=result["output_provenance"][key])
                expected = states[index].advance(block, "cpu")
                torch.testing.assert_close(loaded.hidden, expected.hidden, rtol=0, atol=0)
                torch.testing.assert_close(loaded.pre_mix, expected.pre_mix, rtol=0, atol=0)
            before = len(events)
            self.assertEqual(wavefront.process(block, "base", 0, keys, input_provenance=provenance), result)
            self.assertEqual(len(events), before)
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                wavefront.process(block, "base", 0, list(reversed(keys)), input_provenance=provenance)
