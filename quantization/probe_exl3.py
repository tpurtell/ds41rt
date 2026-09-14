#!/usr/bin/env python3
"""Diagnostic native capture -> actual EXL3 K3 search on one V4.1 matrix.

Uses synthetic MoE inputs, not the qualified corpus. Never publish its output.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

from gptqmodel.utils.v41_capture import V41Capture
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3


@torch.inference_mode()
def run(snapshot, device):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    source = V41Source(snapshot)
    block = source.load_decoded_block(0, device, native_kernels=kernel)
    torch.manual_seed(41)
    hidden = torch.randn(1, 256, 5120, device=device, dtype=torch.bfloat16) * .1
    capture = V41Capture(block, [0, 1, 2, 3], device=device)
    with capture:
        block.mlp(hidden)
    expert = max(capture.expert_ids, key=lambda index: capture.counts[index, "gate_up"])
    hessian, evidence = capture.projection(expert, "w1")
    if not hessian["count"]:
        raise AssertionError("probe has no naturally routed rows")
    del block, capture, hidden
    gc.collect()
    torch.cuda.empty_cache()
    name = f"layers.0.ffn.experts.{expert}.w1.weight"
    weight = source.decoded(name, device).T.contiguous().float()
    hessian["H"] = hessian["H"].to(device)
    args = dict(K=3, devices=[torch.device(device)], apply_out_scales=None,
                sigma_reg=0.025, seed=787, mcg=True)
    print(json.dumps(dict(event="search_start", projection=name, route_evidence=evidence)), flush=True)
    started = time.monotonic()
    quantized, error, tensors = quantize_exl3(weight, hessian, args, return_weight_q=True)
    if not torch.isfinite(quantized).all():
        raise AssertionError("nonfinite reconstructed projection")
    print(json.dumps(dict(event="search_passed", seconds=time.monotonic() - started,
                          proxy_error=float(error), metrics=args.get("error_metrics"),
                          tensors={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                   for key, value in tensors.items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.snapshot, args.device)
