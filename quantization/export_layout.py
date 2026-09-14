"""Standard HF shard planning with independently reusable V4.1 PLE groups.

Works from tensor descriptors only; never reads payloads or computes hashes.
The caller supplies the final tensor inventory after routed-weight replacement.
"""
import re


def tensor_group(name):
    match = re.fullmatch(r"layers\.(1|14)\.engram\.embed\.(weight|scale)", name)
    if match:
        return "ple-" + match[1]
    if ".engram.embed." in name:
        raise ValueError(f"unexpected PLE tensor: {name}")
    return "model"


def plan_shards(tensors, *, target_bytes=5_000_000_000):
    """Return file inventories and an HF index, preserving each tensor whole.

    Input maps final tensor names to descriptors containing payload `bytes`.
    Every PLE group must contain its weight and scale. Filenames are stable
    within each group, independent of changes to the other groups.
    """
    if type(target_bytes) is not int or target_bytes < 1 or not tensors:
        raise ValueError("nonempty inventory and positive shard target required")
    groups = {"model": [], "ple-1": [], "ple-14": []}
    for name, descriptor in sorted(tensors.items()):
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise ValueError("invalid tensor name")
        size = descriptor["bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("invalid tensor payload size")
        groups[tensor_group(name)].append(name)
    for layer in (1, 14):
        expected = {f"layers.{layer}.engram.embed.{suffix}" for suffix in ("weight", "scale")}
        if set(groups[f"ple-{layer}"]) != expected:
            raise ValueError(f"incomplete PLE group {layer}")
    files, weight_map = {}, {}
    for group, names in groups.items():
        shards, current, size = [], [], 0
        for name in names:
            length = tensors[name]["bytes"]
            if current and size + length > target_bytes:
                shards.append(current)
                current, size = [], 0
            current.append(name)
            size += length
        if current:
            shards.append(current)
        for ordinal, members in enumerate(shards, 1):
            filename = f"{group}-{ordinal:05d}-of-{len(shards):05d}.safetensors"
            files[filename] = members
            weight_map.update((name, filename) for name in members)
    return dict(files=files, index=dict(
        metadata=dict(total_size=sum(item["bytes"] for item in tensors.values())),
        weight_map=weight_map))
