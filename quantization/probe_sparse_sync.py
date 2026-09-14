#!/usr/bin/env python3
"""Small native sparse-attention cases for compute-sanitizer race/sync checks.

Run against the checkpoint's unchanged kernel on each RTX. Covers tile tails,
multiple KV iterations, repeated indices, invalid slots and all-masked rows.
Numerical comparisons assess repeatability, not independent model correctness.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import torch


@torch.inference_mode()
def run(snapshot, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    kernel = importlib.import_module("kernel")
    torch.manual_seed(41)
    cases = []
    for topk in (1, 63, 64, 65, 129, 257):
        q = torch.randn(1, 4, 16, 512, dtype=torch.bfloat16, device=device)
        kv = torch.randn(1, 257, 512, dtype=torch.bfloat16, device=device)
        sinks = torch.randn(16, dtype=torch.float32, device=device)
        indices = torch.randint(257, (1, 4, topk), dtype=torch.int32, device=device)
        indices[:, 0] = -1
        indices[:, 1, ::3] = -1
        indices[:, 2] = 7
        expected = kernel.sparse_attn(q, kv, sinks, indices, 512 ** -.5)
        for _ in range(3):
            actual = kernel.sparse_attn(q, kv, sinks, indices, 512 ** -.5)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not torch.isfinite(expected).all() or torch.count_nonzero(expected[:, 0]):
            raise AssertionError("invalid sparse attention masked-row result")
        torch.cuda.synchronize(device)
        cases.append(topk)
    print(json.dumps(dict(event="sparse_sync_probe_passed", device=device, topk=cases,
                          repeats=4, heads=16, head_dim=512)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    run(**vars(parser.parse_args()))
