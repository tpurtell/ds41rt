#!/usr/bin/env python3
"""Attest unchanged corpus text tokenization and V4.1 PLE hashes locally.

No model weights are loaded. Direct AutoTokenizer is a diagnostic comparator;
production inputs use Tokenicer. Corpus text is raw, with special tokens enabled,
without chat rendering, concatenation, truncation or padding.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
import transformers
import transformers.models.deepseek_v41.modeling_deepseek_v41 as modeling
from tokenicer import Tokenicer
from transformers import AutoTokenizer, DeepseekV41Config
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41NgramHashState


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def attest(snapshot, corpus):
    config = DeepseekV41Config.from_pretrained(snapshot, local_files_only=True)
    tokenizer = Tokenicer.load(str(snapshot), model_config=config, local_files_only=True,
                               trust_remote_code=False).tokenizer
    direct = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    probes = ["Hello, world!", " The THE thé  Ｔｈｅ\t\n", "你好，世界。日本語 한국어 العربية",
              "x = 1\n    return x\n", "<｜begin▁of▁sentence｜>test<｜end▁of▁sentence｜>", "🙂é\u0301\u0000"]
    samples = []
    for text in probes:
        actual, expected = tokenizer(text, add_special_tokens=True), direct(text, add_special_tokens=True)
        if dict(actual) != dict(expected):
            raise AssertionError("Tokenicer/direct tokenizer mismatch")
        samples.append(dict(raw_text=text, rendered_text=text, **dict(actual)))
    stream = hashlib.sha256()
    records = tokens = longest = 0
    with corpus.open() as source:
        for line in source:
            row = json.loads(line)
            text = row["prompt"]
            actual = tokenizer(text, add_special_tokens=True)
            expected = direct(text, add_special_tokens=True)
            if dict(actual) != dict(expected):
                raise AssertionError(f"corpus tokenizer mismatch at record {records}")
            entry = dict(id=row["id"], **dict(actual))
            stream.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            records += 1
            tokens += len(actual["input_ids"])
            longest = max(longest, len(actual["input_ids"]))
    sys.path.insert(0, str(snapshot / "inference"))
    import model as reference
    from engram import EngramLayout, NgramHashState
    args = reference.ModelArgs(**json.loads((snapshot / "inference/config.json").read_text()))
    args.max_batch_size, args.max_seq_len = 2, 1024
    oracle = NgramHashState(args, EngramLayout.from_args(args), direct)
    candidate = DeepseekV41NgramHashState(config.get_text_config())
    candidate.bind_tokenizer(tokenizer)
    for name in ("token_map", "primes", "offsets", "multipliers"):
        if not torch.equal(getattr(oracle, name), getattr(candidate, name)):
            raise AssertionError(f"PLE hash table differs: {name}")
    torch.manual_seed(41)
    ids = torch.randint(len(tokenizer), (2, 1024))
    for device in ("cpu", "cuda:0", "cuda:1"):
        oracle.to(device)
        candidate.to(device)
        for masked in (False, True):
            mask = torch.ones_like(ids, dtype=torch.bool, device=device) if masked else None
            if mask is not None:
                mask[:, ::17] = False
                mask[1, :9] = False
            expected = oracle(ids.to(device), 0, mask)
            actual = candidate(ids.to(device), mask, None)
            if not torch.equal(actual, expected):
                raise AssertionError(f"PLE hashes differ on {device}, masked={masked}")
    return dict(schema="ds41rt-input-attestation-v1", status="passed",
                contract="raw-text-add-special-tokens-no-chat-no-truncation-v1",
                corpus_sha256=digest(corpus), records=records, tokens=tokens, longest=longest,
                token_stream_sha256=stream.hexdigest(), samples=samples,
                tokenizer_class=type(tokenizer).__name__, special_tokens_map=tokenizer.special_tokens_map,
                versions={name: importlib.metadata.version(name) for name in ("transformers", "tokenicer", "tokenizers", "torch")},
                imported_transformers=dict(version=transformers.__version__,
                    module_path=transformers.__file__, modeling_sha256=digest(modeling.__file__)),
                source_files={name: digest(snapshot / name) for name in
                              ("config.json", "tokenizer.json", "tokenizer_config.json", "inference/engram.py", "inference/model.py")},
                compressed_vocab=args.engram_compressed_vocab_size,
                hash_cases="full token map and tables; 2x1024 IDs, masked/unmasked, CPU and both RTX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(attest(args.snapshot, args.corpus), ensure_ascii=False, sort_keys=True), flush=True)
