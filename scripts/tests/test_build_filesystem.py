"""CPU-only build filesystem guard regressions; never invokes Cargo."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "build_filesystem", Path(__file__).resolve().parents[1] / "assert-build-filesystem.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class BuildFilesystemTest(unittest.TestCase):
    def mount(self, kind="ext4", options="rw,relatime"):
        return subprocess.CompletedProcess([], 0, json.dumps({
            "filesystems": [{"fstype": kind, "options": options}]}), "")

    def test_scratch_rejected_before_mount_probe(self):
        with patch.object(guard.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "prohibited"):
                guard.check_path("/mnt/scratch/new/target")
            run.assert_not_called()

    def test_ntfs_container_alias_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for kind in ("ntfs3", "ntfs", "fuseblk", "fuse.ntfs-3g"):
                with self.subTest(kind=kind), patch.object(
                    guard.subprocess, "run", return_value=self.mount(kind)):
                    with self.assertRaisesRegex(ValueError, "prohibited"):
                        guard.check_path(root + "/not-created/target")

    def test_symlink_into_scratch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            link = Path(root) / "alias"
            link.symlink_to("/mnt/scratch")
            with self.assertRaisesRegex(ValueError, "prohibited"):
                guard.check_path(str(link / "target"))

    def test_ext4_nonexistent_target_uses_existing_parent(self):
        with tempfile.TemporaryDirectory() as root, patch.object(
            guard.subprocess, "run", return_value=self.mount()) as run:
            guard.check_path(root + "/new/target")
            self.assertIn(root, run.call_args.args[0])

    def test_readonly_and_unknown_fail_closed(self):
        for result in (self.mount(options="ro"), subprocess.CompletedProcess([], 0, '{"filesystems": []}', '')):
            with patch.object(guard.subprocess, "run", return_value=result):
                with self.assertRaises(ValueError):
                    guard.check_path("/tmp")


if __name__ == "__main__":
    unittest.main()
