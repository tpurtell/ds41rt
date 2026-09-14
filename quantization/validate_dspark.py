#!/usr/bin/env python3
"""Real dSpark core-chain parity, including reference window seeding/wraparound.

Synthetic projected main features and draft embeddings isolate the three routed
blocks. This does not qualify main_proj, token embedding, or auxiliary heads.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch

from gptqmodel.utils.v41_source import V41Source, source_layer_key
from gptqmodel.utils.v41_replay import V41ReplayBatch, owned_tree


@torch.inference_mode()
def validate(snapshot, device, lengths):
    sys.path.insert(0, str(snapshot / "inference"))
    import model as reference
    import kernel

    def sparse(q, kv, sinks, indices, scale):
        return torch.cat([kernel.sparse_attn(q[:, :, i:i + 16].contiguous(), kv,
                                             sinks[i:i + 16].contiguous(), indices, scale)
                          for i in range(0, q.shape[2], 16)], dim=2)

    reference.sparse_attn = sparse
    source = V41Source(snapshot)
    args = reference.ModelArgs(**json.loads((snapshot / "inference/config.json").read_text()))
    args.max_batch_size = 1
    args.max_seq_len = max(lengths) + args.dspark_block_size + 1
    previous_dtype = torch.get_default_dtype()
    frontier = {}
    try:
        torch.set_default_dtype(torch.bfloat16)
        for stage in range(args.n_mtp_layers):
            candidate = source.load_decoded_block(stage, device, native_kernels=kernel, namespace="mtp")
            with torch.device("meta"):
                original = reference.DSparkBlock(args.n_layers + stage, args)
            # Scope is the core block, excluding input preparation/output heads.
            for name in ("main_proj", "main_norm", "norm", "markov_head", "confidence_head"):
                if hasattr(original, name):
                    delattr(original, name)
            weights = {source_layer_key(stage, key, "mtp").removeprefix(f"mtp.{stage}."): value
                       for key, value in candidate.state_dict().items()}
            original.load_state_dict(weights, strict=True, assign=True)
            for module in original.modules():
                if getattr(module, "scale", None) is not None and hasattr(module, "weight"):
                    module.weight.scale = module.scale
            with torch.device(device):
                reference.precompute_freqs_cis.cache_clear()
                original.attn.freqs_cis = reference.precompute_freqs_cis(
                    args.rope_head_dim, args.max_seq_len, 0, args.rope_theta,
                    args.rope_factor, args.beta_fast, args.beta_slow)
                for name, value in list(original.named_buffers()):
                    if value.is_meta:
                        owner, _, leaf = name.rpartition(".")
                        setattr(original.get_submodule(owner), leaf,
                                torch.zeros(value.shape, device=device, dtype=value.dtype))
                for length in lengths:
                    if stage == 0:
                        torch.manual_seed(1234 + length)
                        hidden = torch.randn(1, args.dspark_block_size, args.hc_mult, args.dim, device=device) * .1
                        pre = reference.make_identity_pre_mix(hidden, args.hc_mult)
                        main = torch.randn(1, length, args.dim, device=device) * .1
                        state = V41ReplayBatch(0, owned_tree(hidden, "cpu"), owned_tree(pre, "cpu"), {},
                                               {"main_x": owned_tree(main, "cpu")}, {})
                        ref_hidden, ref_pre = hidden, pre
                    else:
                        state, ref_hidden, ref_pre = frontier[length]
                        main = state.kwargs["main_x"].to(device)
                        ref_hidden, ref_pre = ref_hidden.to(device), ref_pre.to(device)
                    # Official prefill seeds history only; the final main token is
                    # then inserted by the actual draft execution at start_pos > 0.
                    original.attn(ref_hidden[:, :, 0], 0, main[:, :-1])
                    expected, expected_pre = original(ref_hidden, length - 1, ref_pre, main[:, -1:])
                    outgoing = state.advance(candidate, device)
                    report = dict(stage=stage, main_length=length,
                                  hidden_max_abs=(outgoing.hidden.float() - expected.cpu().float()).abs().max().item(),
                                  pre_max_abs=(outgoing.pre_mix - expected_pre.cpu()).abs().max().item())
                    print(json.dumps(report), flush=True)
                    if report["hidden_max_abs"] != 0 or report["pre_max_abs"] != 0:
                        raise AssertionError("dSpark core differs from checkpoint reference")
                    frontier[length] = outgoing, owned_tree(expected, "cpu"), owned_tree(expected_pre, "cpu")
            del candidate, original, weights
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2, 129, 257])
    args = parser.parse_args()
    if min(args.lengths) < 2:
        parser.error("draft execution requires at least two main-history positions")
    validate(args.snapshot.resolve(), args.device, args.lengths)
