#!/usr/bin/env python3
"""Joint V4.1 draft parity against checkpoint blocks with adapted cache indexing.

Only the reference attention's scalar start_pos/cache interface is adapted for
different anchor positions in one batch. Its projection, normalization, rotary,
attention, output projection, mHC and expert operations stay reference-owned.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import MethodType

import torch
from gptqmodel.utils.v41_source import V41Source, source_layer_key


def joint_reference_attention(module, x, start_pos, main_x, *, reference, kernel, anchors):
    batch, drafts, _ = x.shape
    win, rd = module.window_size, module.rope_head_dim
    main_kv = module.kv_norm(module.wkv(main_x))
    reference.apply_rotary_emb(main_kv[..., -rd:], module.freqs_cis[:main_x.shape[1]])
    kernel.act_quant(main_kv, reference.fp8_block_size, reference.scale_fmt, reference.scale_dtype, True)
    qr = module.q_norm(module.wq_a(x))
    q = module.wq_b(qr).unflatten(-1, (module.n_local_heads, module.head_dim))
    kv = module.kv_norm(module.wkv(x))
    for index, position in enumerate(anchors):
        frequencies = module.freqs_cis[position + 1:position + 1 + drafts]
        reference.apply_rotary_emb(q[index:index + 1, ..., -rd:], frequencies)
        reference.apply_rotary_emb(kv[index:index + 1, ..., -rd:], frequencies)
    kernel.act_quant(kv, reference.fp8_block_size, reference.scale_fmt, reference.scale_dtype, True)
    window = torch.zeros(batch, win, module.head_dim, device=x.device, dtype=main_kv.dtype)
    indices = torch.full((batch, drafts, win + drafts), -1, device=x.device, dtype=torch.int32)
    for index, position in enumerate(anchors):
        # Literal cache writes are independent of the candidate's modulo gather.
        for token_position in range(position + 1):
            window[index, token_position % win] = main_kv[0, token_position]
        indices[index, :, :min(win, position + 1)] = torch.arange(min(win, position + 1), device=x.device)
        indices[index, :, win:] = win + torch.arange(drafts, device=x.device)
    bank = torch.cat((window, kv), dim=1)
    output = torch.cat([kernel.sparse_attn(q[:, :, head:head + 16].contiguous(), bank,
                        module.attn_sink[head:head + 16].contiguous(), indices, module.softmax_scale)
                        for head in range(0, module.n_local_heads, 16)], dim=2)
    for index, position in enumerate(anchors):
        reference.apply_rotary_emb(output[index:index + 1, ..., -rd:],
                                   module.freqs_cis[position + 1:position + 1 + drafts], True)
    output = output.view(batch, drafts, module.n_local_groups, -1)
    wo_a = module.wo_a.weight.view(module.n_local_groups, module.o_lora_rank, -1)
    output = torch.einsum("bsgd,grd->bsgr", output, wo_a)
    return module.wo_b(output.flatten(2))


@torch.inference_mode()
def validate(snapshot, device, anchors):
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(snapshot / "inference"))
    import model as reference
    import kernel
    source = V41Source(snapshot)
    args = reference.ModelArgs(**json.loads((snapshot / "inference/config.json").read_text()))
    args.max_batch_size = len(anchors)
    args.max_seq_len = max(anchors) + 7
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        adapter = source.load_dspark_input(device, native_kernels=kernel)
        torch.manual_seed(51)
        features = {index: torch.randn(1, max(anchors) + 2, args.dim, device=device) * .1
                    for index in adapter.target_layer_ids}
        ids = torch.randint(args.vocab_size, (1, max(anchors) + 2), device=device)
        state = adapter.prepare_joint(features, ids, positions=torch.tensor(anchors))
        expected_hidden = state.hidden.to(device)
        expected_pre = state.pre_mix.to(device)
        for stage in range(args.n_mtp_layers):
            candidate = source.load_decoded_block(stage, device, native_kernels=kernel, namespace="mtp")
            with torch.device("meta"):
                original = reference.DSparkBlock(args.n_layers + stage, args)
            if stage == 0:
                reference_input = torch.nn.Module()
                reference_input.main_proj = original.main_proj
                reference_input.main_norm = original.main_norm
                reference_input.embed = adapter.embed
                reference_input.block_size = args.dspark_block_size
                reference_input.noise_token_id = args.dspark_noise_token_id
                reference_input.hc_mult = args.hc_mult
                reference_input.load_state_dict(adapter.state_dict(), strict=True, assign=True)
                for module in reference_input.modules():
                    if getattr(module, "scale", None) is not None and hasattr(module, "weight"):
                        module.weight.scale = module.scale
                main_hidden = torch.cat([features[index][:, :max(anchors) + 1]
                                         for index in adapter.target_layer_ids], dim=-1)
                expected_hidden, expected_main = reference.DSparkBlock.forward_embed(
                    reference_input, main_hidden, ids[0, torch.tensor(anchors, device=device) + 1])
                expected_pre = reference.make_identity_pre_mix(expected_hidden, args.hc_mult)
                if not torch.equal(expected_hidden.cpu(), state.hidden) or not torch.equal(expected_main.cpu(), state.kwargs["main_x"]):
                    raise AssertionError("joint input adapter differs from checkpoint reference")
                print(json.dumps(dict(event="joint_input_exact", anchors=len(anchors))), flush=True)
                del reference_input, main_hidden, expected_main
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
                original.attn.freqs_cis = reference.precompute_freqs_cis(args.rope_head_dim, args.max_seq_len,
                    0, args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
            def forward(module, x, start_pos, main_x):
                return joint_reference_attention(module, x, start_pos, main_x,
                                                   reference=reference, kernel=kernel, anchors=anchors)
            original.attn.forward = MethodType(forward, original.attn)
            captured = {}
            handles = [original.attn.register_forward_hook(lambda m, a, o: captured.__setitem__("reference", o)),
                       candidate.self_attn.register_forward_hook(lambda m, a, o: captured.__setitem__("candidate", o[0]))]
            try:
                expected_hidden, expected_pre = original(expected_hidden, 1, expected_pre, state.kwargs["main_x"].to(device))
                outgoing = state.advance(candidate, device)
            finally:
                for handle in handles:
                    handle.remove()
            report = dict(stage=stage, anchors=len(anchors),
                attention_max_abs=(captured["reference"].float() - captured["candidate"].float()).abs().max().item(),
                hidden_max_abs=(expected_hidden.cpu().float() - outgoing.hidden.float()).abs().max().item(),
                pre_max_abs=(expected_pre.cpu() - outgoing.pre_mix).abs().max().item())
            print(json.dumps(report), flush=True)
            if any(report[key] != 0 for key in ("attention_max_abs", "hidden_max_abs", "pre_max_abs")):
                raise AssertionError("joint dSpark differs from independently adapted checkpoint reference")
            state = outgoing
            del candidate, original, weights, captured
            gc.collect()
        print(json.dumps(dict(event="joint_reference_passed", anchors=anchors)), flush=True)
    finally:
        torch.set_default_dtype(dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchors", type=int, nargs="+", default=[1, 7, 127, 128, 256])
    parser.add_argument("--anchor-count", type=int, help="use this many consecutive anchors starting at 1")
    args = parser.parse_args()
    anchors = list(range(1, args.anchor_count + 1)) if args.anchor_count is not None else args.anchors
    if not anchors or anchors[0] < 1 or anchors != sorted(set(anchors)):
        parser.error("anchors must be sorted, distinct and positive")
    validate(args.snapshot, args.device, anchors)
