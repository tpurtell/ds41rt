import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_quantization import read_attestation, verify_source, validate_manifest


class RuntimeTest(unittest.TestCase):
    def test_attestation_log_requires_one_passed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attestation.log"
            report = dict(schema="ds41rt-input-attestation-v1", status="passed")
            path.write_text("library banner\n" + json.dumps(report) + "\n")
            self.assertEqual(read_attestation(path), report)
            path.write_text(json.dumps(report) + "\n" + json.dumps(report))
            with self.assertRaises(ValueError):
                read_attestation(path)
            path.write_text(json.dumps(report, indent=2))
            self.assertEqual(read_attestation(path), report)

    def test_source_reuse_never_hashes_weight_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            shard = snapshot / "model.safetensors"
            save_file({"weight": torch.ones(4)}, shard)
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard.name}}))
            files = {path.name: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                     for path in snapshot.iterdir()}
            core = dict(schema="ds41rt-v41-source-attestation-v1", source_snapshot="snapshot", files=files,
                        tensor_count=1, shard_count=1)
            digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            report = root / "source.json"
            report.write_text(json.dumps({**core, "status": "passed", "manifest_sha256": digest}))
            original = hashlib.file_digest
            def bounded(stream, algorithm):
                self.assertFalse(str(stream.name).endswith(".safetensors"))
                return original(stream, algorithm)
            with patch("run_quantization.hashlib.file_digest", side_effect=bounded):
                self.assertEqual(verify_source(snapshot, report), digest)
            (snapshot / "unexpected.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "inventory"):
                verify_source(snapshot, report)

    def test_manifest_requires_complete_topology(self):
        manifest = dict(schema="ds41rt-quantization-runtime-v1", identity={"run": "test"},
                        endpoints=[dict(name=name, url="http://" + name, preflight_sha256="test", image_digest="test")
                                   for name in ("ostrich", "dodo", "emu", "kiwi")],
                        coordinator_slots=[dict(device=f"cuda:{i}", gpu_uuid=str(i), preflight_sha256="test", image_digest="test")
                                           for i in range(2)])
        for key in ("snapshot", "source_attestation", "corpus", "input_attestation", "run_root", "output", "export_state", "token_file"):
            manifest[key] = "/test/" + key
        validate_manifest(manifest)
        manifest["coordinator_slots"].pop()
        with self.assertRaisesRegex(ValueError, "both RTX"):
            validate_manifest(manifest)
