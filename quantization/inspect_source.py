#!/usr/bin/env python3
"""Inspect V4.1 shard headers without paging the enormous PLE payloads in."""

import argparse
import collections
import json
from pathlib import Path
import re
import struct


def inspect(snapshot: Path) -> dict:
    config = json.loads((snapshot / "config.json").read_text())
    if config["model_type"] != "deepseek_v41":
        raise ValueError("expected DeepSeek V4.1 source")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    tensors = {}
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = snapshot / name
        if path.parent != snapshot or not path.is_file():
            raise ValueError(f"missing or invalid shard: {name}")
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 128 * 1024 * 1024:
                raise ValueError(f"unbounded header: {name}")
            header = json.loads(stream.read(size))
        payload_start = 8 + size
        file_size = path.stat().st_size
        shards[name] = file_size
        for key, item in header.items():
            if key == "__metadata__":
                continue
            start, end = item["data_offsets"]
            if not 0 <= start <= end <= file_size - payload_start:
                raise ValueError(f"invalid offsets: {key}")
            if key in tensors or index["weight_map"].get(key) != name:
                raise ValueError(f"index/header mismatch: {key}")
            tensors[key] = dict(item, shard=name, bytes=end-start)
    if tensors.keys() != index["weight_map"].keys():
        raise ValueError("incomplete tensor inventory")
    projections = collections.Counter()
    for key in tensors:
        match = re.fullmatch(r"(.+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight", key)
        if match:
            projections[match[1]] += 1
            if key.removesuffix("weight") + "scale" not in tensors:
                raise ValueError(f"missing source scale: {key}")
    ple = {key: item for key, item in tensors.items() if ".engram.embed." in key}
    return {
        "snapshot": str(snapshot.resolve()),
        "verification_scope": "headers, index, sizes; payload hashes not verified",
        "source_bytes": sum(shards.values()),
        "tensor_count": len(tensors),
        "routed_projection_count": sum(projections.values()),
        "routed_projections_per_block": dict(sorted(projections.items())),
        "ple": ple,
        "ple_shard_other_tensors": {
            shard: [key for key, item in tensors.items()
                    if item["shard"] == shard and key not in ple]
            for shard in sorted({item["shard"] for item in ple.values()})
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.snapshot), indent=2))
