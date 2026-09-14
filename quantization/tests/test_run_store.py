from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_store import RunStore


class RunStoreTest(unittest.TestCase):
    def test_dependencies_reopen_idempotence_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "input.bin", root / "routed.bin"
            first.write_bytes(b"input")
            second.write_bytes(b"routed")
            store = RunStore(root, {"source": "test", "recipe": "test"})
            self.addCleanup(store.close)
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                store.record_file("routed", "routed", second, parents=("input",))
            self.assertIsNone(store.get("routed"))
            self.assertEqual(second.read_bytes(), b"routed")
            record = store.record_file("input", "replay", first)
            self.assertEqual(store.record_file("input", "replay", first), record)
            routed = store.record_file("routed", "routed", second, parents=("input",))
            reopened = RunStore(root, {"source": "test", "recipe": "test"})
            self.addCleanup(reopened.close)
            self.assertEqual(reopened.get("routed"), routed)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                RunStore(root, {"source": "different"})
            second.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                reopened.get("routed")
            with self.assertRaisesRegex(ValueError, "already-committed"):
                store.record_file("routed", "routed", second, parents=("input",))

    def test_rejects_external_payload_and_self_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "run", {"test": True})
            self.addCleanup(store.close)
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            with self.assertRaises(ValueError):
                store.record_file("x", "routed", outside)
            with self.assertRaises(ValueError):
                store.record_file("x", "routed", outside, parents=("x",))
