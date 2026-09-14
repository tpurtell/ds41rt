"""Recovery gates for owned, atomic V4.1 replay frontiers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier
from gptqmodel.utils.v41_replay import V41ReplayBatch


class V41CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "frontier.safetensors"
        self.provenance = dict(source="test-source", corpus="test-corpus",
                               recipe="test-recipe", implementation="test-code")
        self.batch = V41ReplayBatch(
            3, torch.randn(1, 9, 2, 8).to(torch.bfloat16), torch.randn(1, 9, 2),
            dict(compress_kv=torch.randn(1, 1, 4, 8), candidates=torch.ones(1, 9, 4, dtype=torch.bool)),
            dict(position_embeddings=(torch.randn(1, 9, 4), torch.randn(1, 9, 4)),
                 past_key_values=None), {14: torch.randn(1, 9, 24, 8)})

    def load(self, digest, **kwargs):
        return load_frontier(self.path, expected_sha256=digest,
                             expected_provenance=kwargs.get("provenance", self.provenance))

    def test_owned_roundtrip_with_typed_keys_and_tuples(self):
        digest = save_frontier(self.batch, self.path, provenance=self.provenance)
        restored = self.load(digest)
        self.assertEqual(restored.next_layer, 3)
        torch.testing.assert_close(restored.hidden, self.batch.hidden, rtol=0, atol=0)
        self.assertIsInstance(restored.kwargs["position_embeddings"], tuple)
        self.assertEqual(set(restored.engram_rows), {14})
        torch.testing.assert_close(restored.engram_rows[14], self.batch.engram_rows[14], rtol=0, atol=0)
        # Loaded data survives replacement/unlink and cannot mutate the saved state.
        restored.hidden.zero_()
        torch.testing.assert_close(self.load(digest).hidden, self.batch.hidden, rtol=0, atol=0)
        self.path.unlink()
        self.assertTrue(torch.equal(restored.shared["candidates"], self.batch.shared["candidates"]))

    def test_corruption_and_identity_rejected(self):
        digest = save_frontier(self.batch, self.path, provenance=self.provenance)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load(digest, provenance={**self.provenance, "corpus": "different"})
        with self.path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(digest)

    def test_failed_publication_preserves_previous_frontier(self):
        digest = save_frontier(self.batch, self.path, provenance=self.provenance)
        self.batch.next_layer = 4
        with patch("gptqmodel.utils.v41_checkpoint.os.replace", side_effect=OSError("injected failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                save_frontier(self.batch, self.path, provenance=self.provenance)
        self.assertEqual(self.load(digest).next_layer, 3)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
