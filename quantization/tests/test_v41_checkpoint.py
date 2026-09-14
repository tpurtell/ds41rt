"""Recovery gates for owned, atomic V4.1 replay frontiers."""
from pathlib import Path
import gc
import tempfile
import unittest
import weakref
from unittest.mock import patch

import torch

from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier
from gptqmodel.utils.v41_checkpoint import load_routed_batch, save_routed_batch
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
from gptqmodel.utils.v41_replay import V41ReplayBatch


class V41CheckpointTest(unittest.TestCase):
    def test_saved_tensor_copies_release_without_cyclic_collection(self):
        from gptqmodel.utils import v41_checkpoint as checkpoint
        original = checkpoint.save_file
        references = []
        def observed(tensors, *args, **kwargs):
            references.extend(weakref.ref(value) for value in tensors.values())
            return original(tensors, *args, **kwargs)
        enabled = gc.isenabled()
        gc.disable()
        try:
            # Plain function replacement, not a mock retaining call arguments.
            with patch.object(checkpoint, "save_file", new=observed):
                for _ in range(16):
                    save_frontier(self.batch, self.path, provenance=self.provenance)
                    self.assertTrue(references)
                    self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()

    def test_loaded_tensors_release_without_cyclic_collection(self):
        digest = save_frontier(self.batch, self.path, provenance=self.provenance)
        enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(16):
                state = self.load(digest)
                references = [weakref.ref(state.hidden), weakref.ref(state.pre_mix),
                              weakref.ref(state.engram_rows[14])]
                del state
                self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()

    def test_routed_roundtrip_cannot_be_confused_with_replay(self):
        batch = V41RoutedBatch(torch.randn(3, 8), torch.randn(3, 4), torch.rand(3, 2),
                               torch.tensor([[0, 1], [1, 2], [2, 3]]))
        digest = save_routed_batch(batch, self.path, provenance=self.provenance)
        loaded = load_routed_batch(self.path, expected_sha256=digest, expected_provenance=self.provenance)
        for name in vars(batch):
            torch.testing.assert_close(getattr(loaded, name), getattr(batch, name), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "kind mismatch"):
            self.load(digest)
        batch.indices[0, 1] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            save_routed_batch(batch, self.path, provenance=self.provenance)
        # Validation failed before publication: previous durable bytes survive.
        load_routed_batch(self.path, expected_sha256=digest, expected_provenance=self.provenance)

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
