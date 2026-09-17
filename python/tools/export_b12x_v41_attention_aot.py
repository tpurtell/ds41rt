#!/usr/bin/env python3
"""Export direct native FP4 attention and sink merge from verified SparkInfer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import _pinned_sparkinfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(32,64), default=64)
    args = parser.parse_args()
    stem = "v41_attention" if args.heads == 64 else "v41_attention_heads32"
    symbol = "ds41rt_" + stem
    # Runtime cached modules omit the retained IR needed by export_to_c.
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"
    import torch
    from b12x.attention._shared.mla.native_v41_aot import compile_native_v41_attention_aot

    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("native attention export requires SM120")
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / (stem+".json")
    manifest_path.unlink(missing_ok=True)
    compile_native_v41_attention_aot(args.heads).export_to_c(str(destination), stem, symbol)
    header = (destination / (stem+".h")).read_text()
    pointers = ["query", "descriptors", "metadata", "selected", "bounds", "sink", "partials", "lses", "output"]
    signature = re.search(r"static inline int32_t cute_dsl_"+symbol+r"_wrapper\(([^)]*)\)", header)
    expected = [symbol+"_Kernel_Module_t *module"] + ["void *"+name for name in pointers]
    expected += ["int32_t rows", "cudaStream_t stream"]
    if signature is None or re.sub(r"\s+", "", signature[1]) != re.sub(r"\s+", "", ",".join(expected)):
        raise ValueError("unexpected native attention pointer ABI")
    arguments = re.search(r"void \*args\[12\] = \{([^}]*)\}", header)
    if arguments is None or re.sub(r"\s+", "", arguments[1]) != ",".join("&"+x for x in pointers+["rows", "stream", "ret"]):
        raise ValueError("unexpected native attention generated argument order")
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in destination.glob(stem+".*") if p.is_file() and p != manifest_path}
    manifest = dict(schema=1, sparkinfer_revision=_pinned_sparkinfer.REVISION,
                    source_tree_sha256=_pinned_sparkinfer.LOCK_DATA["source_tree_sha256"],
                    capability=[12, 0], geometry=dict(heads=args.heads, head_dim=512, window=128, selected=512, splits=10),
                    descriptor_bytes=120, metadata_bytes_per_row=80,
                    partial_bytes_per_row=args.heads*10*512*2, lse_bytes_per_row=args.heads*10*4,
                    live_rows="runtime argument; caller validates allocated capacity",
                    format="FP8 window / FP4 indexed; separate value and scale planes",
                    abi=dict(pointers=pointers, i32=["rows"], stream="stream"), artifacts=artifacts)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
