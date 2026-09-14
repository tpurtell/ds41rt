#!/usr/bin/env python3
"""Compare real native blocks and optional consecutive replay against the reference."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext

import torch
from transformers import DeepseekV41Config
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41RotaryEmbedding, DeepSeekV41MappedEmbedding
from gptqmodel.utils.v41_source import V41Source, source_layer_key
from gptqmodel.utils.v41_replay import V41ReplayBatch, owned_tree


@torch.inference_mode()
def validate(snapshot, device, layer_index=0, lengths=(32, 129), frontier=None, capture_experts=None):
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
    block = source.load_decoded_block(layer_index, device, native_kernels=kernel)
    config = DeepseekV41Config.from_dict(source.config).get_text_config()
    args = reference.ModelArgs(**json.loads((snapshot / "inference/config.json").read_text()))
    args.max_batch_size = 1
    args.max_seq_len = max(lengths) + 1
    old_dtype = torch.get_default_dtype()
    mapped = None
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            original = reference.Block(layer_index, args, reference.EngramLayout.from_args(args))
        if original.engram is not None:
            # Compare the full Engram arithmetic with identical real gathered rows.
            # Lookup precision/lifetime is separately covered by the mapped tests.
            original.engram.embed = torch.nn.Identity()
            prefix = f"layers.{layer_index}.engram.embed"
            mapped = DeepSeekV41MappedEmbedding(snapshot / source.weight_map[prefix + ".weight"], prefix)
        weights = {source_layer_key(layer_index, name).removeprefix(f"layers.{layer_index}."): value
                   for name, value in block.state_dict().items()}
        for name, template in original.state_dict().items():
            if template.dtype == torch.float32 and weights[name].dtype == torch.bfloat16:
                weights[name] = weights[name].float()
        original.load_state_dict(weights, strict=True, assign=True)
        for module in original.modules():
            if getattr(module, "scale", None) is not None and hasattr(module, "weight"):
                module.weight.scale = module.scale
        reference.precompute_freqs_cis.cache_clear()
        with torch.device(device):
            original.attn.freqs_cis = reference.precompute_freqs_cis(
                args.rope_head_dim, args.max_seq_len,
                args.original_seq_len if original.attn.compress_ratio else 0,
                args.compress_rope_theta if original.attn.compress_ratio else args.rope_theta,
                args.rope_factor, args.beta_fast, args.beta_slow)
            for name, tensor in list(original.named_buffers()):
                if tensor.is_meta:
                    parent_name, _, leaf = name.rpartition(".")
                    parent = original.get_submodule(parent_name)
                    value = torch.full(tensor.shape, -torch.inf if "score_state" in name else 0,
                                       dtype=tensor.dtype, device=device)
                    setattr(parent, leaf, value)
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
        for length in lengths:
            torch.manual_seed(917 + length)
            hidden = torch.randn(1, length, args.hc_mult, args.dim, device=device,
                                 dtype=torch.bfloat16) * 0.1
            pre = torch.zeros(1, length, args.hc_mult, device=device, dtype=torch.float32)
            pre[..., 0] = 1
            incoming = None if frontier is None else frontier.get(length)
            if incoming is not None:
                if incoming["candidate"].next_layer != layer_index:
                    raise ValueError("chain frontier must precede this layer")
                hidden = incoming["candidate"].hidden.to(device)
                pre = incoming["candidate"].pre_mix.to(device)
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
                rows = None
                reference_hidden = (hidden.clone() if incoming is None
                                    else incoming["reference_hidden"].to(device))
                reference_pre = (pre.clone() if incoming is None
                                 else incoming["reference_pre"].to(device))
                reference.shared_attn = reference.SharedAttentionRuntime()
                if incoming is not None:
                    for name, value in incoming["reference_shared"].items():
                        setattr(reference.shared_attn, name, owned_tree(value, device))
                if mapped is not None:
                    hashes = torch.randint(mapped.num_embeddings,
                                           (1, length, (args.engram_max_ngram_size - 1) * args.engram_n_heads),
                                           device=device)
                    rows = mapped(hashes)
                    reference_hidden = original.engram(reference_hidden, rows.to(torch.bfloat16), None)
                expected, expected_pre = original(reference_hidden, 0, reference_pre, None)
            state = V41ReplayBatch(
                layer_index, owned_tree(hidden, "cpu"), owned_tree(pre, "cpu"),
                {} if incoming is None else incoming["candidate"].shared,
                owned_tree(dict(position_embeddings=pos_emb, position_ids=positions,
                                attention_mask=mask, padding_mask=None, past_key_values=None), "cpu"),
                {} if rows is None else {layer_index: owned_tree(rows, "cpu")})
            from gptqmodel.utils.v41_capture import V41Capture
            capture_state = (V41Capture(block, capture_experts, device=device)
                             if capture_experts else None)
            with capture_state if capture_state is not None else nullcontext():
                outgoing = state.advance(block, device)
            actual, actual_pre = outgoing.hidden.to(device), outgoing.pre_mix.to(device)
            candidate_shared = owned_tree(outgoing.shared, device)
            report = {"layer": layer_index, "length": length,
                      "hidden_max_abs": (actual.float() - expected.float()).abs().max().item(),
                      "hidden_relative_l2": ((actual.float() - expected.float()).norm() / expected.float().norm()).item(),
                      "pre_max_abs": (actual_pre - expected_pre).abs().max().item()}
            if capture_state is not None:
                from gptqmodel.utils.v41_routed_batch import V41RoutedBatch
                from gptqmodel.utils.v41_checkpoint import save_frontier, save_routed_batch, load_routed_batch
                from run_store import RunStore
                routed = V41RoutedBatch.from_replay(block, state, device)
                with tempfile.TemporaryDirectory(prefix="ds41rt-routed-gate-") as directory:
                    root = Path(directory)
                    provenance = dict(diagnostic="routed-roundtrip", layer=layer_index, length=length,
                                      snapshot=snapshot.name)
                    journal = RunStore(root, provenance)
                    try:
                        save_frontier(state, root / "input.safetensors", provenance=provenance)
                        journal.record_file("input", "replay", "input.safetensors")
                        save_routed_batch(routed, root / "routed.safetensors", provenance=provenance)
                        journal.record_file("routed", "routed", "routed.safetensors", parents=("input",))
                        record = journal.get("routed")
                        routed = load_routed_batch(root / record["path"], expected_sha256=record["sha256"],
                                                   expected_provenance=provenance)
                    finally:
                        journal.close()
                report["journaled_routed_reload"] = True
                report["direct_capture_exact"] = True
                for phase, projections in (("gate_up", ("w1", "w3")), ("down", ("w2",))):
                    direct = V41Capture(block, capture_experts, device=device, phase=phase)
                    direct.capture_routed(routed)
                    for expert in capture_experts:
                        for projection in projections:
                            left, le = direct.projection(expert, projection)
                            right, re = capture_state.projection(expert, projection)
                            if not torch.equal(left["H"], right["H"]) or left["count"] != right["count"] or le != re:
                                report["direct_capture_exact"] = False
                    del direct
                report["captured_experts"] = {}
                for expert in capture_experts:
                    for projection in ("w1", "w3", "w2"):
                        hessian, evidence = capture_state.projection(expert, projection)
                        if not torch.isfinite(hessian["H"]).all():
                            raise AssertionError("nonfinite captured Hessian")
                        report["captured_experts"][f"{expert}.{projection}"] = evidence
                del capture_state
            if original.attn.compress_ratio:
                count = length // original.attn.compress_ratio
                for name in ("compress_kv", "index_k"):
                    expected_shared = getattr(reference.shared_attn, name)[:1, :count]
                    actual_shared = candidate_shared[name][:, 0]
                    report[name + "_exact"] = torch.equal(expected_shared, actual_shared)
                ref_indices = reference.shared_attn.topk_idxs
                ref_indices = torch.where(ref_indices >= 0, ref_indices - length, -1)
                report["indices_exact"] = torch.equal(ref_indices, candidate_shared["topk_idx"])
                if reference.shared_attn.candidates is not None:
                    report["candidates_exact"] = torch.equal(reference.shared_attn.candidates,
                                                             candidate_shared["candidates"])
            for label in ("attn_input", "attn_output", "ffn_input", "ffn_output", "qa", "qn", "qb", "kv_norm", "q", "kv", "core", "oa"):
                if "candidate_" + label in captures:
                    left, right = captures["reference_" + label].float(), captures["candidate_" + label].float()
                    report[label + "_relative_l2"] = ((left - right).norm() / left.norm()).item()
            reports.append(report)
            print(json.dumps(report), flush=True)
            if report["hidden_max_abs"] != 0 or report["pre_max_abs"] != 0:
                raise AssertionError("real V4.1 block differs from the reference")
            if any(value is False for key, value in report.items() if key.endswith("_exact")):
                raise AssertionError("shared attention state differs from the reference")
            if frontier is not None:
                frontier[length] = dict(candidate=outgoing,
                                        reference_hidden=owned_tree(expected, "cpu"),
                                        reference_pre=owned_tree(expected_pre, "cpu"),
                                        reference_shared=owned_tree(vars(reference.shared_attn), "cpu"))
        return reports
    finally:
        if mapped is not None:
            mapped.close()
        torch.set_default_dtype(old_dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--through-layer", type=int,
                        help="validate consecutive blocks with owned CPU replay boundaries")
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 129])
    parser.add_argument("--capture-experts", type=int, nargs="+")
    arguments = parser.parse_args()
    if arguments.capture_experts:
        # Process startup, before reference or candidate execution; never toggle
        # global matmul precision inside concurrent capture workers.
        torch.backends.cuda.matmul.allow_tf32 = False
    last = arguments.layer if arguments.through_layer is None else arguments.through_layer
    if last < arguments.layer:
        parser.error("--through-layer must be at least --layer")
    frontier = {}
    for layer in range(arguments.layer, last + 1):
        validate(arguments.snapshot.resolve(), arguments.device, layer, arguments.lengths, frontier,
                 arguments.capture_experts)
        # Hooks create cycles in the diagnostic model; free them between blocks.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
