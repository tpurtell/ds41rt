#!/usr/bin/env python3
"""Compare real native block 0 with the checkpoint's original implementation."""
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import DeepseekV41Config
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41RotaryEmbedding
from gptqmodel.utils.v41_source import V41Source, source_layer_key


@torch.inference_mode()
def validate(snapshot, device):
    sys.path.insert(0, str(snapshot / "inference"))
    import model as reference
    import kernel
    captures = {}

    # The released 64-head kernel requests 141312 shared-memory bytes, exceeding
    # SM120's per-block limit. Heads are independent: invoke the unchanged
    # reference kernel on 16-head groups, with the same KV/indices and sinks.
    def reference_attention(q, kv, sinks, indices, scale):
        captures["reference_q"] = q.detach().clone()
        captures["reference_kv"] = kv.detach().clone()
        result = torch.cat([kernel.sparse_attn(q[:, :, start:start + 16].contiguous(),
                                            kv, sinks[start:start + 16].contiguous(), indices, scale)
                          for start in range(0, q.shape[2], 16)], dim=2)
        captures["reference_core"] = result.detach().clone()
        return result

    reference.sparse_attn = reference_attention

    source = V41Source(snapshot)
    block = source.load_decoded_block(0, device, native_kernels=kernel)
    config = DeepseekV41Config.from_dict(source.config).get_text_config()
    args = reference.ModelArgs(**json.loads((snapshot / "inference/config.json").read_text()))
    args.max_batch_size = 1
    args.max_seq_len = 256
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            original = reference.Block(0, args)
        weights = {source_layer_key(0, name).removeprefix("layers.0."): value
                   for name, value in block.state_dict().items()}
        original.load_state_dict(weights, strict=True, assign=True)
        for module in original.modules():
            if getattr(module, "scale", None) is not None and hasattr(module, "weight"):
                module.weight.scale = module.scale
        reference.precompute_freqs_cis.cache_clear()
        with torch.device(device):
            original.attn.freqs_cis = reference.precompute_freqs_cis(
                args.rope_head_dim, args.max_seq_len, 0, args.rope_theta,
                args.rope_factor, args.beta_fast, args.beta_slow)
            original.attn.window_kv_cache = torch.zeros(1, args.window_size, args.head_dim)
            rotary = DeepSeekV41RotaryEmbedding(config)
        reports = []
        def capture(name):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    captures[name] = output.detach().clone()
            return hook
        for label, original_module, candidate_module in (
            ("attn_input", original.attn_norm, block.input_layernorm),
            ("attn_output", original.attn, block.self_attn),
            ("ffn_input", original.ffn_norm, block.post_attention_layernorm),
            ("ffn_output", original.ffn, block.mlp),
            ("qa", original.attn.wq_a, block.self_attn.q_a_proj),
            ("qn", original.attn.q_norm, block.self_attn.q_a_norm),
            ("qb", original.attn.wq_b, block.self_attn.q_b_proj),
            ("kv_norm", original.attn.kv_norm, block.self_attn.kv_norm),
            ("oa", original.attn.wo_b, block.self_attn.o_b_proj),
        ):
            original_module.register_forward_hook(capture("reference_" + label))
            candidate_module.register_forward_hook(capture("candidate_" + label))
        for length in (32, 129):
            torch.manual_seed(917 + length)
            hidden = torch.randn(1, length, args.hc_mult, args.dim, device=device,
                                 dtype=torch.bfloat16) * 0.1
            pre = torch.zeros(1, length, args.hc_mult, device=device, dtype=torch.float32)
            pre[..., 0] = 1
            positions = torch.arange(length, device=device).unsqueeze(0)
            pos_emb = {kind: rotary(hidden[:, :, 0], position_ids=positions, layer_type=kind)
                       for kind in ("main", "compress")}
            q = positions.unsqueeze(-1)
            k = positions.unsqueeze(-2)
            visible = (k <= q) & (k > q - args.window_size)
            mask = torch.where(visible.unsqueeze(1), 0., torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
            # The official runner executes under a default CUDA device context;
            # its cached window-index helper relies on that context.
            with torch.device(device):
                expected, expected_pre = original(hidden.clone(), 0, pre.clone(), None)
            actual, actual_pre = block(hidden.clone(), pre.clone(), None, None, shared={},
                                       position_embeddings=pos_emb, position_ids=positions,
                                       attention_mask=mask, padding_mask=None, past_key_values=None)
            report = {"length": length,
                      "hidden_max_abs": (actual.float() - expected.float()).abs().max().item(),
                      "hidden_relative_l2": ((actual.float() - expected.float()).norm() / expected.float().norm()).item(),
                      "pre_max_abs": (actual_pre - expected_pre).abs().max().item()}
            for label in ("attn_input", "attn_output", "ffn_input", "ffn_output", "qa", "qn", "qb", "kv_norm", "q", "kv", "core", "oa"):
                if "candidate_" + label in captures:
                    left, right = captures["reference_" + label].float(), captures["candidate_" + label].float()
                    report[label + "_relative_l2"] = ((left - right).norm() / left.norm()).item()
            reports.append(report)
            print(json.dumps(report), flush=True)
            if report["hidden_max_abs"] != 0 or report["pre_max_abs"] != 0:
                raise AssertionError("real V4.1 block differs from the reference")
        return reports
    finally:
        torch.set_default_dtype(old_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    validate(arguments.snapshot.resolve(), arguments.device)
