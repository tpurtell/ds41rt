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
import tempfile
from contextlib import nullcontext

import torch

from gptqmodel.utils.v41_capture import V41Capture
from gptqmodel.utils.v41_source import V41Source
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from gptqmodel.utils.v41_mixed_replay import install_projection
from gptqmodel.utils.v41_recovery import V41Recovery
from safetensors.torch import save_file, load_file


@torch.inference_mode()
def run(snapshot, device, bits=3, recover=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import kernel
    source = V41Source(snapshot)
    block = source.load_decoded_block(0, device, native_kernels=kernel)
    torch.manual_seed(41)
    hidden = torch.randn(1, 256, 5120, device=device, dtype=torch.bfloat16) * .1
    capture = V41Capture(block, [0, 1, 2, 3], device=device)
    recovery = V41Recovery(block, capture.expert_ids) if recover else None
    with capture, recovery if recovery is not None else nullcontext():
        block.mlp(hidden)
    expert = max(capture.expert_ids, key=lambda index: capture.counts[index, "gate_up"])
    hessian, evidence = (capture.projection(expert, "w1") if recovery is None
                         else recovery.projection(capture, expert, "w1"))
    if not hessian["count"]:
        raise AssertionError("probe has no naturally routed rows")
    del capture, recovery
    gc.collect()
    torch.cuda.empty_cache()
    name = f"layers.0.ffn.experts.{expert}.w1.weight"
    weight = source.decoded(name, device).T.contiguous().float()
    hessian["H"] = hessian["H"].to(device)
    args = dict(K=bits, devices=[torch.device(device)], apply_out_scales=None,
                sigma_reg=0.025, seed=787, mcg=True)
    print(json.dumps(dict(event="search_start", bits=bits, projection=name, route_evidence=evidence)), flush=True)
    started = time.monotonic()
    quantized, error, tensors = quantize_exl3(weight, hessian, args, return_weight_q=True)
    if not torch.isfinite(quantized).all():
        raise AssertionError("nonfinite reconstructed projection")
    with tempfile.TemporaryDirectory(prefix="ds41rt-packed-probe-") as directory:
        path = str(Path(directory) / "candidate.safetensors")
        save_file({key: value.cpu().contiguous() for key, value in tensors.items()}, path)
        packed = load_file(path)
        replacement = install_projection(block, expert, "w1", packed, device=device)
        raw_difference = (replacement.weight.float().T - quantized.float()).abs().max().item()
        first = block.mlp(hidden)
        second = block.mlp(hidden)
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise AssertionError("mixed expert replay is nonfinite or not repeatable")
    print(json.dumps(dict(event="search_passed", seconds=time.monotonic() - started,
                          packed_replay_repeatable=True, raw_search_vs_packed_bf16_max_abs=raw_difference,
                          proxy_error=float(error), metrics=args.get("error_metrics"),
                          tensors={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                   for key, value in tensors.items()})), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bits", type=int, choices=(3, 4), default=3)
    parser.add_argument("--recover", action="store_true", help="exercise adjacent-rank plus identity top-up")
    args = parser.parse_args()
    run(args.snapshot, args.device, args.bits, args.recover)
