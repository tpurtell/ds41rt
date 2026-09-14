import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upload_model import upload_artifact
from materialize_cache import materialize_cache


class FakeApi:
    def __init__(self):
        self.head = "a" * 40
        self.files = {}
        self.uploaded = []
        self.commits = 0
        self.fail_upload = False
        self.lose_commit_response = False

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha=self.head)

    def list_repo_tree(self, *args, **kwargs):
        return list(self.files.values())

    def preupload_lfs_files(self, repo, operations, **kwargs):
        for op in operations:
            if self.fail_upload and self.uploaded:
                raise RuntimeError("transport failed")
            op._should_ignore = False
            op._upload_mode = "lfs" if op.path_in_repo.endswith(".safetensors") else "regular"
            if op._upload_mode == "lfs":
                with op.as_file() as stream:
                    op.upload_info.sha256 = hashlib.sha256(stream.read()).digest()
                op._is_uploaded = True
            self.uploaded.append(op.path_in_repo)

    def create_commit(self, repo, operations, **kwargs):
        assert kwargs["parent_commit"] == self.head
        self.commits += 1
        for op in operations:
            if op._upload_mode == "lfs":
                assert op._is_uploaded and op.upload_info.is_hashed
                blob = op.upload_info.sha256.hex()
                lfs = dict(oid=blob, size=op.upload_info.size, pointerSize=130)
            else:
                with op.as_file() as stream:
                    payload = stream.read()
                blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                lfs = None
            self.files[op.path_in_repo] = RepoFile(path=op.path_in_repo, size=op.upload_info.size,
                                                  oid=blob, lfs=lfs)
        self.head = "b" * 40
        if self.lose_commit_response:
            raise RuntimeError("lost commit response")
        return SimpleNamespace(oid=self.head)


class UploadTest(unittest.TestCase):
    def fixture(self, root):
        output = root / "export"
        output.mkdir()
        for name in (".gitattributes", "README.md", "config.json", "model.safetensors.index.json", "model.safetensors"):
            (output / name).write_bytes(name.encode())
        return output, root / "state", FakeApi()

    def test_upload_receipt_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, state, api = self.fixture(root)
            receipt = upload_artifact(output, state, "test/model", api=api)
            self.assertEqual(receipt["status"], "uploaded")
            self.assertEqual(upload_artifact(output, state, "test/model", api=api, resume=True), receipt)
            self.assertEqual(api.commits, 1)
            self.assertEqual(len(api.uploaded), 5)
            self.assertEqual(materialize_cache(output, root / "hub", receipt)["status"], "materialized")

    def test_preupload_failure_reuses_completed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.fail_upload = True
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                upload_artifact(output, state, "test/model", api=api)
            self.assertEqual(api.commits, 0)
            api.fail_upload = False
            upload_artifact(output, state, "test/model", api=api, resume=True)
            self.assertEqual(len(api.uploaded), 5)

    def test_lost_commit_response_recovers_without_reupload_or_recommit(self):
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.lose_commit_response = True
            with self.assertRaisesRegex(RuntimeError, "lost commit response"):
                upload_artifact(output, state, "test/model", api=api)
            with patch.object(api, "preupload_lfs_files", side_effect=AssertionError("must not reupload")):
                receipt = upload_artifact(output, state, "test/model", api=api, resume=True)
            self.assertEqual(receipt["commit"], "b" * 40)
            self.assertEqual(api.commits, 1)

    def test_populated_remote_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output, state, api = self.fixture(Path(directory))
            api.files["existing"] = RepoFile(path="existing", size=3, oid="c" * 40)
            with self.assertRaisesRegex(ValueError, "populated"):
                upload_artifact(output, state, "test/model", api=api)
            self.assertEqual(api.uploaded, [])
