import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recovery_identity import resolve_identity, authorize_input_recovery
from gptqmodel.utils import v41_checkpoint
from block_driver import BlockDriver
from run_store import RunStore


class RecoveryIdentityTest(unittest.TestCase):
    def fixture(self, root):
        original = dict(run_root=str(root), corpus="unchanged-corpus", coordinator_slots=[
            dict(device=f"cuda:{i}", gpu_uuid=str(i), image_digest="old-image", preflight_sha256="old-code") for i in range(2)])
        (root / "original.json").write_text(json.dumps(original))
        report = dict(event="checkpoint_memory_probe_passed", gc_disabled=True, workers=2, waves=32,
            post_warmup_spread_kib=120, byte_exact_inputs=["a", "b", "c"],
            serializer_sha256=hashlib.sha256(Path(v41_checkpoint.__file__).read_bytes()).hexdigest())
        payload = json.dumps(report).encode()
        (root / "memory.log").write_bytes(payload)
        current = copy.deepcopy(original)
        for slot in current["coordinator_slots"]:
            slot.update(image_digest="new-image", preflight_sha256="new-code")
        current["recovery"] = dict(schema="ds41rt-input-serializer-recovery-v1",
            original_manifest=str(root / "original.json"), memory_report=str(root / "memory.log"),
            memory_report_sha256=hashlib.sha256(payload).hexdigest())
        return original, current

    def test_only_qualified_execution_identity_may_change(self):
        with tempfile.TemporaryDirectory() as directory:
            original, current = self.fixture(Path(directory))
            preserved, evidence = resolve_identity(current)
            self.assertEqual(preserved, original)
            self.assertEqual(evidence["execution_manifest"], current)
            changed = copy.deepcopy(current)
            changed["corpus"] = "different"
            with self.assertRaisesRegex(ValueError, "changed model/corpus"):
                resolve_identity(changed)
            changed = copy.deepcopy(current)
            changed["coordinator_slots"][0]["gpu_uuid"] = "different"
            with self.assertRaisesRegex(ValueError, "topology"):
                resolve_identity(changed)
            (Path(directory) / "memory.log").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resolve_identity(current)

    def test_authorization_is_durable_and_only_before_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, current = self.fixture(root)
            _, evidence = resolve_identity(current)
            identity = {"runtime": original}
            journal = RunStore(root, identity)
            self.addCleanup(journal.close)
            driver = BlockDriver(None, journal, identity, device="cpu")
            driver._publish("inputs/inventory", "inventory", {"records": [1, 2, 3]}, ())
            for i in range(3):
                driver._publish(f"inputs/frontiers/{i}", "replay", {}, ("inputs/inventory",))
            # Even an unfinished block must prevent a new input-only migration.
            driver._publish("blocks/base/000/input-inventory", "inventory", {}, ("inputs/inventory",))
            with self.assertRaisesRegex(ValueError, "input-only"):
                authorize_input_recovery(driver, evidence)
            self.assertIsNone(driver._load("recovery/input-serializer-release-v1", "recovery"))

    def test_authorized_run_can_resume_after_blocks_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, current = self.fixture(root)
            _, evidence = resolve_identity(current)
            identity = {"runtime": original}
            journal = RunStore(root, identity)
            self.addCleanup(journal.close)
            driver = BlockDriver(None, journal, identity, device="cpu")
            driver._publish("inputs/inventory", "inventory", {"records": [1]}, ())
            driver._publish("inputs/frontiers/0", "replay", {}, ("inputs/inventory",))
            authorize_input_recovery(driver, evidence)
            driver._publish("blocks/base/000/input-inventory", "inventory", {}, ("inputs/inventory",))
            authorize_input_recovery(driver, evidence)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                authorize_input_recovery(driver, {**evidence, "serializer_sha256": "different"})
