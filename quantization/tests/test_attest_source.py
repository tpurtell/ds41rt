import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attest_source import attest, hash_file


class SourceAttestationTest(unittest.TestCase):
    def test_complete_inventory_and_content_addressed_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            payload = b"test weight shard"
            blob = root / hashlib.sha256(payload).hexdigest()
            blob.write_bytes(payload)
            (snapshot / "model-1.safetensors").symlink_to(blob)
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-1.safetensors"}}))
            output = root / "attestation.json"
            result = attest(snapshot, output, workers=2)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["files"]["model-1.safetensors"]["sha256"], blob.name)
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaisesRegex(ValueError, "must not modify"):
                attest(snapshot, snapshot / "report.json")
            (snapshot / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "shard inventory"):
                attest(snapshot, output)
            blob.write_bytes(b"wrong bytes")
            with self.assertRaisesRegex(ValueError, "blob mismatch"):
                hash_file(snapshot / "model-1.safetensors")
