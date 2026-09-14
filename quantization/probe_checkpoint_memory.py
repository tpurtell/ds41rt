#!/usr/bin/env python3
"""Bounded serializer memory and byte compatibility with stopped-run inputs.

No model replay, quantization, persistent validation samples, or source-weight
hashing. Temporary rewritten recovery checkpoints are removed after the probe.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import weakref
from unittest.mock import patch

import torch
from gptqmodel.utils import v41_checkpoint as checkpoint
from stream_shard import read_header


def rss_kib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError("missing process RSS")


def run(journal_root):
    db = sqlite3.connect(f"file:{journal_root / 'run.sqlite'}?mode=ro", uri=True)
    try:
        identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        rows = db.execute("SELECT key, record FROM artifacts WHERE key LIKE 'inputs/frontiers/%' ORDER BY key").fetchall()
        if not rows:
            raise ValueError("no committed input frontiers")
        block_count = db.execute("SELECT count(*) FROM artifacts WHERE key LIKE 'blocks/%/complete'").fetchone()[0]
        print(json.dumps(dict(event="stopped_journal_inventory", input_frontiers=len(rows),
                              block_completion_records=block_count)), flush=True)
    finally:
        db.close()
    checked = []
    with tempfile.TemporaryDirectory(prefix="ds41rt-checkpoint-memory-") as directory:
        root = Path(directory)
        state = None
        for key, encoded in (rows[0], rows[len(rows) // 2], rows[-1]):
            record = json.loads(encoded)
            provenance = {**identity, "artifact": key}
            state = checkpoint._load_state(journal_root / record["path"], expected_sha256=record["sha256"],
                                           expected_provenance=provenance, kind="replay")
            # SQLite canonicalizes identity key order. Preserve the original
            # header's ordering when testing exact serialized bytes, not merely
            # semantic provenance equality (already checked by the loader).
            header, _ = read_header(journal_root / record["path"])
            original_provenance = json.loads(header["__metadata__"]["v41_frontier"])["provenance"]
            if original_provenance != provenance:
                raise AssertionError("original provenance differs from journal identity")
            provenance = original_provenance
            actual = checkpoint._save_state(state, root / "roundtrip.safetensors", provenance=provenance, kind="replay")
            if actual != record["sha256"]:
                raise AssertionError("serializer bytes changed for committed input")
            checked.append(key)
        original_writer = checkpoint.save_file
        references = []
        def observed(tensors, *args, **kwargs):
            references.extend(weakref.ref(value) for value in tensors.values())
            return original_writer(tensors, *args, **kwargs)
        def save(slot):
            with torch.inference_mode():
                return checkpoint._save_state(state, root / f"worker-{slot}.safetensors",
                                               provenance=provenance, kind="replay")
        readings = []
        enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            with patch.object(checkpoint, "save_file", new=observed), ThreadPoolExecutor(max_workers=2) as pool:
                for wave in range(32):
                    futures = [pool.submit(save, slot) for slot in range(2)]
                    for future in futures:
                        if future.result() != actual:
                            raise AssertionError("parallel serializer bytes changed")
                    if any(reference() is not None for reference in references):
                        raise AssertionError("serializer retained copied tensors with GC disabled")
                    references.clear()
                    readings.append(rss_kib())
            if max(readings[8:]) - min(readings[8:]) > 256 * 1024:
                raise AssertionError("serializer RSS did not remain bounded after warmup")
        finally:
            if enabled:
                gc.enable()
        print(json.dumps(dict(event="checkpoint_memory_probe_passed", gc_disabled=True,
            serializer_sha256=hashlib.sha256(Path(checkpoint.__file__).read_bytes()).hexdigest(),
            workers=2, waves=32, byte_exact_inputs=checked, rss_kib=readings,
            post_warmup_spread_kib=max(readings[8:]) - min(readings[8:]))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-root", type=Path, required=True)
    run(parser.parse_args().journal_root)
