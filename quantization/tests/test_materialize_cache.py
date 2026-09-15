import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from materialize_cache import materialize_cache
from write_export import _fingerprint


class CacheTest(unittest.TestCase):
    def test_guarded_revision_advance_preserves_old_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            old = materialize_cache(output, cache, receipt)
            new = dict(receipt, commit='b' * 40)
            with self.assertRaisesRegex(ValueError, 'authorized previous'):
                materialize_cache(output, cache, new, previous_commit='c' * 40)
            result = materialize_cache(output, cache, new, previous_commit=receipt['commit'])
            self.assertEqual((cache / 'models--test--model/refs/main').read_text(), new['commit'])
            self.assertTrue(Path(old['snapshot']).is_dir())
            self.assertEqual(materialize_cache(output, cache, new, previous_commit=receipt['commit']), result)

    def test_explicit_owner_covers_refs_without_following_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            with patch('materialize_cache.os.chown') as chown:
                materialize_cache(output, cache, receipt, owner=(98765, 98765))
            self.assertTrue(any(str(call.args[0]).endswith('/refs/main') for call in chown.call_args_list))
            self.assertTrue(all(call.kwargs == {'follow_symlinks': False} for call in chown.call_args_list))
            with self.assertRaises(ValueError):
                materialize_cache(output, cache, receipt, owner=(-1, 0))

    def fixture(self, root):
        output = root / "export"
        output.mkdir()
        files = {}
        for name in ("model.safetensors", "config.json", "inference/kernel.py"):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            files[name] = dict(bytes=path.stat().st_size, local=_fingerprint(path),
                               blob=hashlib.sha256(name.encode()).hexdigest())
        receipt = dict(status="uploaded", repo_id="test/model", commit="a" * 40, files=files)
        return output, root / "hub", receipt

    def test_hard_links_offline_hf_lookup_and_resume(self):
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            first = materialize_cache(output, cache, receipt)
            self.assertEqual(materialize_cache(output, cache, receipt), first)
            for name in receipt["files"]:
                path = Path(hf_hub_download("test/model", name, cache_dir=cache, local_files_only=True))
                self.assertTrue(path.is_symlink())
                self.assertEqual(path.stat().st_ino, (output / name).stat().st_ino)
            self.assertEqual(first["payload_copied_bytes"], 0)

    def test_interrupted_snapshot_does_not_publish_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            with patch.object(Path, "symlink_to", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    materialize_cache(output, cache, receipt)
            self.assertFalse((cache / "models--test--model/refs/main").exists())
            self.assertEqual(materialize_cache(output, cache, receipt)["status"], "materialized")

    def test_identical_uploaded_assets_share_one_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            duplicate = output / "duplicate.json"
            duplicate.write_bytes((output / "config.json").read_bytes())
            receipt["files"][duplicate.name] = {**receipt["files"]["config.json"],
                                                 "local": _fingerprint(duplicate)}
            result = materialize_cache(output, cache, receipt)
            snapshot = Path(result["snapshot"])
            self.assertEqual((snapshot / "duplicate.json").resolve(), (snapshot / "config.json").resolve())
            self.assertEqual(materialize_cache(output, cache, receipt), result)

    def test_changed_local_file_rejected_before_cache_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output, cache, receipt = self.fixture(Path(directory))
            (output / "config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                materialize_cache(output, cache, receipt)
            self.assertFalse(cache.exists())
