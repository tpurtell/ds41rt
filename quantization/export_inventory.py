"""Build a source-native routed-only EXL3 export inventory from committed phases.

The final model-loader/config contract is a separate gate. This inventory retains
checkpoint-native names for unchanged tensors and replaces each routed matrix
and its source scale with the four standard EXL3 buffers. No large source payload
is read, decoded, or hashed here.
"""
import json
from pathlib import Path
import re

from mixed_recipe import NAMESPACES, validate_source_tiers
from stream_shard import read_header


def source_entries(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    entries = {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise ValueError("invalid source shard path")
        header, _ = read_header(snapshot / filename)
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in entries or weight_map.get(name) != filename:
                raise ValueError("source index/header mismatch")
            entries[name] = dict(path=str(snapshot / filename), tensor=name,
                                 bytes=item["data_offsets"][1] - item["data_offsets"][0],
                                 dtype=item["dtype"], shape=item["shape"])
    if set(entries) != set(weight_map):
        raise ValueError("incomplete source tensor inventory")
    return entries


def _dict_nodes(node):
    if node[0] != "dict":
        raise ValueError("expected projection dictionary")
    result = {}
    for key, value in node[1]:
        if key[0] != "scalar" or not isinstance(key[1], str) or key[1] in result:
            raise ValueError("invalid projection dictionary key")
        result[key[1]] = value
    return result


def packed_entries(path, *, provenance, bits, projection):
    """Locate serialized buffers after the caller verifies the journal payload."""
    header, _ = read_header(path)
    manifest = json.loads(header["__metadata__"]["v41_frontier"])
    if (manifest.get("version") != 1 or manifest.get("kind") != "projection"
            or manifest.get("provenance") != provenance):
        raise ValueError("projection export provenance mismatch")
    packed = _dict_nodes(_dict_nodes(manifest["state"])["packed"])
    if set(packed) != {"trellis", "suh", "svh", "mcg"} or bits not in (3, 4):
        raise ValueError("invalid selected EXL3 buffer inventory")
    inputs, outputs = (2304, 5120) if projection == "w2" else (5120, 2304)
    expected = {"trellis": ("I16", [inputs // 16, outputs // 16, bits * 16]),
                "suh": ("F16", [inputs]), "svh": ("F16", [outputs]), "mcg": ("I32", [1])}
    result, used = {}, set()
    for suffix, node in packed.items():
        if node[0] != "tensor" or node[1] in used:
            raise ValueError("invalid or aliased packed tensor reference")
        used.add(node[1])
        item = header[node[1]]
        dtype, shape = expected[suffix]
        # The quantizer may store its single MCG integer as a scalar.
        if item["dtype"] != dtype or (item["shape"] != shape and not (suffix == "mcg" and item["shape"] == [])):
            raise ValueError("selected EXL3 geometry differs from V4.1 projection")
        result[suffix] = dict(path=str(path), tensor=node[1], dtype=dtype, shape=item["shape"],
                              bytes=item["data_offsets"][1] - item["data_offsets"][0])
    return result


def collect_selected(driver):
    """Bind all 43 completed blocks to their retained selected candidates."""
    selected, tiers = {}, {}
    for namespace, (prefix, layers, experts) in NAMESPACES.items():
        for layer in range(layers):
            root = f"blocks/{namespace}/{layer:03d}"
            if driver._load(root + "/complete", "block") is None:
                raise ValueError(f"cannot export incomplete block: {root}")
            for phase, projections in (("gate_up", ("w1", "w3")), ("down", ("w2",))):
                marker = driver._load(root + f"/{phase}/complete", "phase")
                if marker is None:
                    raise ValueError("missing selected phase")
                expected = {(expert, projection) for expert in range(experts) for projection in projections}
                observed = set()
                for expert, projection, key in marker["selected"]:
                    if (expert, projection) not in expected or (expert, projection) in observed:
                        raise ValueError("invalid selected projection inventory")
                    observed.add((expert, projection))
                    stem = f"{root}/{phase}/expert-{expert:03d}/{projection}-k"
                    if key not in (stem + "3", stem + "4"):
                        raise ValueError("selected candidate path differs from phase")
                    name = f"{prefix}.{layer}.ffn.experts.{expert}.{projection}"
                    selected[name] = key
                    tiers[name] = int(key[-1])
                if observed != expected:
                    raise ValueError("incomplete selected phase inventory")
    validate_source_tiers(tiers)
    return selected, tiers


def build_inventory(driver):
    selected, tiers = collect_selected(driver)
    entries = source_entries(driver.source.snapshot)
    routed = {name for name in entries if re.search(r"\.ffn\.experts\.", name)}
    if routed != {name + "." + suffix for name in selected for suffix in ("weight", "scale")}:
        raise ValueError("source routed inventory differs from complete recipe")
    for name, key in selected.items():
        for suffix in ("weight", "scale"):
            if entries.pop(name + "." + suffix, None) is None:
                raise ValueError("selected projection lacks source weight/scale pair")
        # Selected candidate files are small recovery artifacts, verified once
        # here. Do not load/reconstruct their tensors just to discover offsets.
        record = driver.journal.get(key)
        if record is None or record["kind"] != "projection":
            raise ValueError("missing selected projection payload")
        packed = packed_entries(driver.journal.root / record["path"],
                                provenance={**driver.identity, "artifact": key},
                                bits=tiers[name], projection=name.rsplit(".", 1)[1])
        for suffix, entry in packed.items():
            target = name + "." + suffix
            if target in entries:
                raise ValueError("EXL3 export tensor name collision")
            entries[target] = entry
    return dict(tensors=entries, tiers=tiers)
