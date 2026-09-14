import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_assets import plan_assets, publish_asset, asset_target


class ExportAssetsTest(unittest.TestCase):
    def test_exact_source_assets_resume_and_card_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            files = {}
            for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE", "inference/kernel.py", "README.md", "config.json", "model.safetensors.index.json", "model.safetensors"):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = (name + "\r\nexact bytes\x00").encode()
                path.write_bytes(payload)
                files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            assets = plan_assets(source, dict(status="passed", source_snapshot="source", files=files))
            self.assertEqual(len(assets), 5)
            self.assertIn("README.source.md", assets)
            self.assertNotIn("README.md", assets)
            for name, record in assets.items():
                publish_asset(output, name, record)
                self.assertEqual((output / name).read_bytes(), Path(record["path"]).read_bytes())
                before = (output / name).stat()
                publish_asset(output, name, record)
                self.assertEqual((output / name).stat().st_mtime_ns, before.st_mtime_ns)
            (output / "LICENSE").write_bytes(b"x" * files["LICENSE"]["bytes"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                publish_asset(output, "LICENSE", assets["LICENSE"])
            for name in ("../outside", "/outside", "inference//kernel.py", "./file"):
                with self.assertRaises(ValueError):
                    asset_target(output, name)
            (output / "link").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                asset_target(output, "link/LICENSE")
