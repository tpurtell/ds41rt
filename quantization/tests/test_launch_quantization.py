import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launch_quantization import command, launch


class LaunchTest(unittest.TestCase):
    def fixture(self, directory):
        root, hf = Path(directory) / "run", Path(directory) / "hf"
        root.mkdir()
        snapshot = hf / "hub/models--source--model/snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        token = hf / "token"
        token.write_text("test-token")
        token.chmod(0o600)
        image = "sha256:" + "b" * 64
        manifest = dict(run_root=str(root), snapshot=str(snapshot), output=str(hf / "exports/model"),
            publication=dict(repo_id="wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1", cache_root=str(hf / "hub")),
            coordinator_slots=[dict(image_digest=image), dict(image_digest=image)])
        for key in ("corpus", "input_attestation", "source_attestation", "token_file", "export_state"):
            manifest[key] = str(root / key)
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest, image, hf

    def test_detached_command_uses_shared_export_cache_mount_and_readonly_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, image, hf = self.fixture(directory)
            args, _ = command(path, manifest, image, hf, "c" * 32, resume=True)
            self.assertIn("--detach", args)
            self.assertIn("--restart=no", args)
            self.assertIn("--resume", args)
            self.assertNotIn("--rm", args)
            self.assertIn(f"type=bind,src={hf},dst={hf}", args)
            source_root = Path(manifest["snapshot"]).parent.parent
            self.assertIn(f"type=bind,src={source_root},dst={source_root},readonly", args)
            self.assertNotIn("test-token", " ".join(args))
            manifest["output"] = str(Path(directory) / "separate-export")
            with self.assertRaisesRegex(ValueError, "staged inside"):
                command(path, manifest, image, hf, "c" * 32)

    def test_live_attempt_is_never_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, image, hf = self.fixture(directory)
            calls = []
            def execute(args, **kwargs):
                calls.append(args)
                if args[1:3] == ["image", "inspect"]:
                    return image
                if args[1] == "ps":
                    return "old-container"
                if args[1] == "inspect":
                    return json.dumps([dict(State=dict(Status="running"))])
                raise AssertionError("must not launch")
            with self.assertRaisesRegex(ValueError, "not terminal"):
                launch(path, image, hf, resume=True, execute=execute)
            self.assertFalse((Path(manifest["run_root"]) / "attempts").exists())

    def test_launch_records_intent_and_retains_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, image, hf = self.fixture(directory)
            def execute(args, **kwargs):
                if args[1:3] == ["image", "inspect"]:
                    return image
                if args[1] == "ps":
                    return ""
                if args[1] == "run":
                    raise RuntimeError("ambiguous Docker response")
                raise AssertionError(args)
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                launch(path, image, hf, execute=execute)
            intents = list((Path(manifest["run_root"]) / "attempts").glob("*-launch.json"))
            self.assertEqual(len(intents), 1)
            self.assertIn("--name", json.loads(intents[0].read_text())["command"])
