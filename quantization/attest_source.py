#!/usr/bin/env python3
"""Hash the complete V4.1 checkpoint with bounded parallel reads and atomic output."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def hash_file(path):
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
        raise ValueError(f"source file changed while hashing: {path.name}")
    blob = path.resolve().name
    if path.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", blob) and blob != digest:
        raise ValueError(f"Hugging Face content-addressed blob mismatch: {path.name}")
    return dict(bytes=before.st_size, sha256=digest)


def attest(snapshot, output, workers=4):
    snapshot = snapshot.resolve(strict=True)
    if output.resolve().is_relative_to(snapshot):
        raise ValueError("source attestation must not modify the checkpoint directory")
    index_bytes = (snapshot / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("checkpoint index contains invalid shard paths")
    # Bind every checkpoint-provided non-weight file, including reference source
    # and tokenizer assets. Extra unindexed weight shards are rejected.
    files = sorted(path for path in snapshot.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix != ".pyc")
    actual_shards = {str(path.relative_to(snapshot)) for path in files if path.suffix == ".safetensors"}
    if actual_shards != set(shards):
        raise ValueError("checkpoint payload shard inventory differs from its index")
    records = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, record in zip(files, pool.map(hash_file, files)):
            name = str(path.relative_to(snapshot))
            records[name] = record
            print(json.dumps(dict(event="source_file_hashed", name=name, **record)), flush=True)
    if records["model.safetensors.index.json"]["sha256"] != hashlib.sha256(index_bytes).hexdigest():
        raise ValueError("checkpoint index changed during source attestation")
    core = dict(schema="ds41rt-v41-source-attestation-v1", source_snapshot=snapshot.name,
                tensor_count=len(index["weight_map"]), shard_count=len(shards), files=records)
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    result = {**core, "manifest_sha256": hashlib.sha256(encoded).hexdigest(), "status": "passed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix="source-attestation-", delete=False) as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, output)
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(dict(event="source_attestation_passed", sha256=result["manifest_sha256"],
                          bytes=sum(record["bytes"] for record in records.values()))), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("source hashing requires 1..8 bounded readers")
    attest(args.snapshot, args.output, args.workers)
