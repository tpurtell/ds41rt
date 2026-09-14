#!/usr/bin/env python3
"""Full real dSpark block phase diagnostic; synthetic data, never publish output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from block_driver import BlockDriver
from run_store import RunStore
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
from gptqmodel.utils.v41_checkpoint import save_routed_batch


@torch.inference_mode()
def run(snapshot, root, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    source = V41Source(snapshot)
    block = source.load_decoded_block(0, device, namespace="mtp", native_kernels=kernel)
    identity = dict(diagnostic="full-mtp0-mixed-phases-v1", source=snapshot.name,
                    driver_sha256=hashlib.sha256(Path(__file__).with_name("block_driver.py").read_bytes()).hexdigest(),
                    seed=41, rows=64)
    journal = RunStore(root, identity)
    started = time.monotonic()
    def report(event):
        print(json.dumps({**event, "elapsed": time.monotonic() - started,
                          "gpu_allocated_bytes": torch.cuda.memory_allocated(device)}, sort_keys=True), flush=True)
    try:
        torch.manual_seed(41)
        hidden = torch.randn(1, 64, 5120, device=device, dtype=torch.bfloat16) * .1
        provenance = {**identity, "artifact": "routed"}
        if journal.get("routed", verify=False) is None:
            logits, weights, indices = block.mlp.gate(hidden)
            batch = V41RoutedBatch(hidden.reshape(-1, 5120).cpu(), logits.cpu(), weights.cpu(), indices.cpu())
            save_routed_batch(batch, root / "routed.safetensors", provenance=provenance)
            journal.record_file("routed", "routed", "routed.safetensors")
        driver = BlockDriver(source, journal, identity, device=device, subset_size=8, progress=report)
        committed = driver.run(block, "mtp", 0, ["routed"], routed_provenance={"routed": provenance})
        first, second = block.mlp(hidden), block.mlp(hidden)
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise AssertionError("completed mixed block is nonfinite or not repeatable")
        report(dict(event="passed", phase_commit=committed))
    finally:
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.snapshot, args.root, args.device)
