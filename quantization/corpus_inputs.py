"""Attested raw-text corpus and bounded, resumable initial RTX frontiers.

One original record per batch preserves corpus order without padding or packing.
Each adapter belongs to one worker; journal writes stay on the coordinator thread.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier


CONTRACT = "raw-text-add-special-tokens-no-chat-no-truncation-v1"


def tokenize_corpus(corpus, tokenizer, attestation):
    payload = Path(corpus).read_bytes()
    if (attestation.get("status") != "passed" or attestation.get("contract") != CONTRACT
            or hashlib.sha256(payload).hexdigest() != attestation.get("corpus_sha256")):
        raise ValueError("corpus does not match passed input attestation")
    records, seen, stream = [], set(), hashlib.sha256()
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row["id"], str) or row["id"] in seen or not isinstance(row["prompt"], str):
            raise ValueError("corpus requires unique string IDs and raw text prompts")
        seen.add(row["id"])
        encoded = dict(tokenizer(row["prompt"], add_special_tokens=True,
                                 truncation=False, padding=False))
        ids = encoded["input_ids"]
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid corpus token IDs")
        if encoded.get("attention_mask", [1] * len(ids)) != [1] * len(ids):
            raise ValueError("corpus records must be unpadded")
        stream.update(json.dumps(dict(id=row["id"], **encoded), sort_keys=True,
                                 separators=(",", ":")).encode() + b"\n")
        records.append(dict(id=row["id"], input_ids=tuple(ids)))
    if (not records or len(records) != attestation.get("records")
            or sum(len(row["input_ids"]) for row in records) != attestation.get("tokens")
            or max(len(row["input_ids"]) for row in records) != attestation.get("longest")
            or stream.hexdigest() != attestation.get("token_stream_sha256")):
        raise ValueError("tokenized corpus differs from input attestation")
    return records


def prepare_frontiers(driver, records, attestation, adapters):
    """Prepare at most one outstanding record per adapter; stop on any failure.

    Adapters must already be constructed on the two RTX devices by the caller.
    This function never loads a model or initializes a tokenizer itself.
    """
    if len(adapters) != 2 or adapters[0] is adapters[1]:
        raise ValueError("input preparation requires two independently owned RTX adapters")
    inventory_key = "inputs/inventory"
    inventory = dict(records=records, attestation=attestation, batching="one-record-unpadded-v1")
    previous = driver._load(inventory_key, "inventory")
    if previous is None:
        driver._publish(inventory_key, "inventory", inventory, ())
    elif previous != inventory:
        raise ValueError("input inventory changed during recovery")
    keys, provenance, missing = [], {}, []
    for ordinal, row in enumerate(records):
        key = f"inputs/frontiers/batch-{ordinal:06d}"
        prov = {**driver.identity, "artifact": key}
        record = driver.journal.get(key, verify=False)
        if record is not None:
            if record["kind"] != "replay":
                raise ValueError("initial frontier has incorrect artifact kind")
            state = load_frontier(driver.journal.root / record["path"],
                                  expected_sha256=record["sha256"], expected_provenance=prov)
            if state.next_layer != 0 or tuple(state.hidden.shape[:2]) != (1, len(row["input_ids"])):
                raise ValueError("initial frontier geometry differs from corpus")
            del state
        else:
            missing.append(ordinal)
        keys.append(key)
        provenance[key] = prov

    def prepare(ordinal, adapter):
        with torch.inference_mode():
            state = adapter.prepare(torch.tensor([records[ordinal]["input_ids"]], dtype=torch.long))
            if (state.next_layer != 0 or not torch.isfinite(state.hidden).all()
                    or not torch.isfinite(state.pre_mix).all()
                    or tuple(state.hidden.shape[:2]) != (1, len(records[ordinal]["input_ids"]))):
                raise ValueError("invalid prepared input frontier")
            key = keys[ordinal]
            path = driver.journal.root / (key + ".safetensors")
            save_frontier(state, path, provenance=provenance[key])
            return path

    # Fixed two-record windows bound pending CPU states and disk writers. No
    # worker ever touches SQLite or shares an adapter with another worker.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0, len(missing), 2):
            pending = [(ordinal, pool.submit(prepare, ordinal, adapters[slot]))
                       for slot, ordinal in enumerate(missing[offset:offset + 2])]
            for ordinal, future in pending:
                path = future.result()
                driver.journal.record_file(keys[ordinal], "replay", path, parents=(inventory_key,))
                driver.progress(dict(event="input_committed", key=keys[ordinal],
                                     tokens=len(records[ordinal]["input_ids"])))
    result = dict(output_keys=tuple(keys), output_provenance=provenance, next_layer=0)
    if driver._load("inputs/complete", "inputs") is None:
        driver._publish("inputs/complete", "inputs", result, (inventory_key, *keys))
    return result
