import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from gptqmodel.utils.v41_replay import V41ReplayBatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_driver import BlockDriver
from corpus_inputs import CONTRACT, prepare_frontiers, tokenize_corpus
from run_store import RunStore


class CorpusTest(unittest.TestCase):
    def test_attestation_and_interrupted_preparation(self):
        def tokenizer(text, **kwargs):
            return dict(input_ids=[1] + [ord(char) for char in text],
                        attention_mask=[1] * (len(text) + 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            rows = [dict(id=str(i), prompt="abc"[:i + 1]) for i in range(3)]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows))
            stream = b"".join(json.dumps(dict(id=row["id"], **tokenizer(row["prompt"])),
                              sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
            attestation = dict(status="passed", contract=CONTRACT, records=3, tokens=9, longest=4,
                               corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
                               token_stream_sha256=hashlib.sha256(stream).hexdigest())
            records = tokenize_corpus(corpus, tokenizer, attestation)
            with self.assertRaisesRegex(ValueError, "tokenized corpus"):
                tokenize_corpus(corpus, tokenizer, {**attestation, "tokens": 10})
            calls = []
            class Adapter:
                def prepare(self, ids):
                    calls.append(ids.tolist())
                    hidden = ids.float()[..., None, None]
                    return V41ReplayBatch(0, hidden, torch.ones_like(hidden[..., 0]), {}, {}, {})
            adapters = [Adapter(), Adapter()]
            journal = RunStore(root / "run", {"test": "corpus"})
            self.addCleanup(journal.close)
            def fail(event):
                raise RuntimeError("interrupted after input commit")
            driver = BlockDriver(None, journal, {"test": "corpus"}, device="cpu", progress=fail)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                prepare_frontiers(driver, records, attestation, adapters)
            self.assertIsNotNone(journal.get("inputs/frontiers/batch-000000"))
            self.assertIsNone(journal.get("inputs/complete"))
            first_calls = len(calls)
            driver.progress = lambda event: None
            result = prepare_frontiers(driver, records, attestation, adapters)
            self.assertEqual(len(calls), first_calls + 2)
            self.assertEqual(len(result["output_keys"]), 3)
            before = len(calls)
            self.assertEqual(prepare_frontiers(driver, records, attestation, adapters), result)
            self.assertEqual(len(calls), before)
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                prepare_frontiers(driver, records[::-1], attestation, adapters)
