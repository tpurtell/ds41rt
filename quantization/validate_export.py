"""Read-only complete export structure validation; never hashes weight shards.

This does not substitute for numerical replay or model-quality qualification.
The source snapshot and frozen export plan are required validation inputs.
"""
import hashlib
import json
from pathlib import Path

from export_assets import asset_target
from export_config import model_metadata
from export_inventory import source_entries
from export_layout import tensor_group
from mixed_recipe import validate_source_tiers
from write_export import verify_shard


def validate_inventory(actual, source, tiers):
    validate_source_tiers(tiers)
    removed = {name + "." + suffix for name in tiers for suffix in ("weight", "scale")}
    source_routed = {name for name in source if ".ffn.experts." in name}
    if source_routed != removed:
        raise ValueError("source routed inventory differs from full recipe")
    added = {name + "." + suffix for name in tiers for suffix in ("trellis", "suh", "svh", "mcg")}
    if set(actual) != (set(source) - removed) | added:
        raise ValueError("export must replace only the complete routed weight/scale inventory")
    for name in set(source) - removed:
        for key in ("dtype", "shape", "bytes"):
            if actual[name][key] != source[name][key]:
                raise ValueError(f"unchanged source descriptor differs: {name}")
    for name, bits in tiers.items():
        inputs, outputs = (2304, 5120) if name.endswith(".w2") else (5120, 2304)
        expected = {"trellis": ("I16", [inputs // 16, outputs // 16, bits * 16], inputs * outputs * bits // 8),
                    "suh": ("F16", [inputs], inputs * 2), "svh": ("F16", [outputs], outputs * 2),
                    "mcg": ("I32", [], 4)}
        for suffix, (dtype, shape, size) in expected.items():
            item = actual[name + "." + suffix]
            if (item["dtype"] != dtype or item["bytes"] != size
                    or (item["shape"] != shape and not (suffix == "mcg" and item["shape"] == [1]))):
                raise ValueError("EXL3 buffer geometry differs from selected tier")


def verify_files(output, plan):
    """Check actual files against frozen export plan without publishing anything."""
    output = Path(output).resolve(strict=True)
    if str(output) != plan["output"]:
        raise ValueError("artifact path differs from export plan")
    layout, entries = plan["layout"], plan["inventory"]["tensors"]
    expected_files = set(layout["files"]) | set(plan["metadata"]) | set(plan.get("assets", {})) | {"model.safetensors.index.json"}
    actual_files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file() or path.is_symlink()}
    if actual_files != expected_files:
        raise ValueError("export file inventory differs from frozen plan")
    index_path = asset_target(output, "model.safetensors.index.json")
    if json.loads(index_path.read_text()) != layout["index"]:
        raise ValueError("export index differs from frozen plan")
    covered, payload = set(), 0
    for filename, names in layout["files"].items():
        path = asset_target(output, filename)
        if len({tensor_group(name) for name in names}) != 1:
            raise ValueError("PLE files must not mix groups or ordinary weights")
        if covered.intersection(names):
            raise ValueError("tensor is assigned to multiple export shards")
        covered.update(names)
        if any(layout["index"]["weight_map"].get(name) != filename for name in names):
            raise ValueError("export index/shard assignment mismatch")
        verify_shard(path, names, entries)
        payload += sum(entries[name]["bytes"] for name in names)
    if covered != set(entries) or covered != set(layout["index"]["weight_map"]) or layout["index"]["metadata"]["total_size"] != payload:
        raise ValueError("export index does not cover the exact tensor payload")
    for filename, expected in plan["metadata"].items():
        if json.loads(asset_target(output, filename).read_text()) != expected:
            raise ValueError("export config differs from frozen plan")
    for filename, record in plan.get("assets", {}).items():
        path = asset_target(output, filename)
        if not 0 <= record["bytes"] <= 64 * 1024 * 1024 or path.stat().st_size != record["bytes"]:
            raise ValueError("export asset size differs")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("export asset checksum differs")
    return dict(files=len(expected_files), tensors=len(entries), payload_bytes=payload)


def validate_export(output, state_root, source_snapshot):
    plan = json.loads((Path(state_root) / "plan.json").read_text())
    counts = verify_files(output, plan)
    inventory = plan["inventory"]
    validate_inventory(inventory["tensors"], source_entries(source_snapshot), inventory["tiers"])
    source_config = json.loads((Path(source_snapshot) / "config.json").read_text())
    external = plan["metadata"]["quantize_config.json"]
    provenance = external["meta"]["ds41rt"]["provenance"]
    if model_metadata(source_config, inventory, provenance=provenance) != plan["metadata"]:
        raise ValueError("model/storage config differs from complete export recipe")
    return dict(schema="ds41rt-export-structure-validation-v1", status="passed", **counts,
                routed_projections=len(inventory["tiers"]), k4_projections=sum(bits == 4 for bits in inventory["tiers"].values()),
                verification="complete headers/index/config/native descriptors and small asset checksums",
                weight_payload_hashes=False, numerical_validation="separate-required-gate")
