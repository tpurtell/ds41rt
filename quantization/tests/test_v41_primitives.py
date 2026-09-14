"""Numerical gates for the separate GPTQModel V4.1 integration."""
import ast
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch
from transformers import DeepseekV41ForCausalLM, DeepseekV41TextConfig
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41EngramEmbedding
from gptqmodel.models.definitions.deepseek_v41 import DeepSeekV41Expert, DeepSeekV41MappedEmbedding, DeepSeekV41QModel
from gptqmodel.utils.v41_replay import V41ReplayBatch
from gptqmodel.utils.v41_checkpoint import load_frontier, save_frontier
from gptqmodel.utils.v41_inputs import V41MainInput


def tiny_config():
    return DeepseekV41TextConfig(
        vocab_size=128, hidden_size=64, moe_intermediate_size=64,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, qk_rope_head_dim=8, q_lora_rank=32, o_groups=2,
        o_lora_rank=16, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, sliding_window=8, hc_mult=2,
        hc_sinkhorn_iters=3, index_n_heads=2, index_head_dim=32,
        index_topk=2, candidate_topk_blocks=4, candidate_block_size=4,
        compress_ratios=[0, 2, 2, 1, 1], engram_layer_ids=[],
        engram_num_embeddings=[], max_position_embeddings=128,
        dspark_noise_token_id=127, experts_implementation="eager",
    )


class V41PrimitivesTest(unittest.TestCase):
    @torch.no_grad()
    def test_main_input_joint_batch_without_decoder_allocation(self):
        config = tiny_config()
        model = DeepseekV41ForCausalLM(config).eval()
        ids = torch.randint(0, 128, (2, 17))
        expected = V41ReplayBatch.prepare(model, ids)
        adapter = V41MainInput(model.model.embed_tokens, None, model.model.rotary_emb, {}, config)
        actual = adapter.prepare(ids)
        torch.testing.assert_close(actual.hidden, expected.hidden, rtol=0, atol=0)
        torch.testing.assert_close(actual.pre_mix, expected.pre_mix, rtol=0, atol=0)
        torch.testing.assert_close(actual.kwargs["position_ids"], expected.kwargs["position_ids"], rtol=0, atol=0)
        self.assertEqual(actual.target_layer_ids, expected.target_layer_ids)
        self.assertIsNone(actual.kwargs["attention_mask"])
        self.assertEqual(actual.hidden.device.type, "cpu")
        for invalid in (ids[0], ids[:, :0], ids.float()):
            with self.assertRaises(ValueError):
                adapter.prepare(invalid)

    @torch.no_grad()
    def test_layer_replay_preserves_shared_state_and_mhc(self):
        torch.manual_seed(789)
        model = DeepseekV41ForCausalLM(tiny_config()).eval()
        DeepSeekV41QModel.convert_model_structure(model)
        ids = torch.randint(0, 128, (2, 33))
        reference = model(ids, use_cache=False).logits
        state = V41ReplayBatch.prepare(model, ids)
        target_inputs = {}
        handles = []
        for index in state.target_layer_ids:
            def capture(module, args, index=index):
                target_inputs[index] = args[0].mean(dim=2).clone()
            handles.append(model.model.layers[index].attn_hc.register_forward_pre_hook(capture))
        for layer in model.model.layers:
            first = state.advance(layer, "cpu")
            second = state.advance(layer, "cpu")
            torch.testing.assert_close(first.hidden, second.hidden, rtol=0, atol=0)
            torch.testing.assert_close(first.pre_mix, second.pre_mix, rtol=0, atol=0)
            # Resume each boundary from disk, then compare final full-forward logits.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "frontier.safetensors"
                provenance = {"test": "five-layer-replay"}
                digest = save_frontier(first, path, provenance=provenance)
                state = load_frontier(path, expected_sha256=digest, expected_provenance=provenance)
        hidden = model.model.layers[-1].hc_collapse(state.hidden, state.pre_mix)
        result = model.lm_head(model.model.norm(hidden))
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        self.assertEqual(state.next_layer, 5)
        for handle in handles:
            handle.remove()
        self.assertEqual(set(state.target_features), set(state.target_layer_ids))
        for index, value in state.target_features.items():
            torch.testing.assert_close(value, target_inputs[index], rtol=0, atol=0)
        self.assertTrue({"compress_kv", "index_k", "candidates", "topk_idx"} <= state.shared.keys())
        with self.assertRaises(ValueError):
            state.advance(model.model.layers[0], "cpu")

    @torch.no_grad()
    def test_expert_against_checkpoint_reference(self):
        # Execute the original definitions unchanged; unrelated TileLang attention
        # kernels are not needed for this floating-point expert comparison.
        source = Path("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/"
                      "dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py")
        tree = ast.parse(source.read_text())
        names = {"linear", "Linear", "Expert"}
        selected = ast.Module(body=[node for node in tree.body
                                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                    and node.name in names], type_ignores=[])
        scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional,
                     default_dtype=torch.bfloat16, fp8_block_size=32)
        exec(compile(selected, str(source), "exec"), scope)
        for device in ("cpu", "cuda:0", "cuda:1"):
            for dtype in (torch.float32, torch.bfloat16):
                torch.manual_seed(456)
                original = scope["Expert"](64, 96, dtype=dtype, swiglu_limit=1.5).to(device)
                ours = DeepSeekV41Expert(64, 96, 1.5, device=device, dtype=dtype)
                for src, dst in ((original.w1, ours.gate_proj),
                                 (original.w3, ours.up_proj), (original.w2, ours.down_proj)):
                    src.weight.normal_(std=0.2)
                    dst.weight.copy_(src.weight)
                x = torch.randn(17, 64, device=device, dtype=dtype)
                weights = torch.rand(17, device=device, dtype=torch.float32)
                torch.testing.assert_close(ours(x, weights), original(x, weights[:, None]),
                                           rtol=0, atol=0)

    @torch.no_grad()
    def test_defuse_full_five_layer_logits(self):
        torch.manual_seed(123)
        model = DeepseekV41ForCausalLM(tiny_config()).eval()
        ids = torch.randint(0, 128, (2, 33))
        before = model(ids, use_cache=False).logits
        self.assertTrue(DeepSeekV41QModel.convert_model_structure(model))
        after = model(ids, use_cache=False).logits
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertFalse(DeepSeekV41QModel.convert_model_structure(model))
        self.assertEqual(sum("mlp.experts." in key for key in model.state_dict()), 60)

    @torch.no_grad()
    def test_mapped_embedding_matches_transformers(self):
        torch.manual_seed(321)
        weight = torch.randn(17, 256).to(torch.float8_e4m3fn)
        scale = torch.randint(122, 132, (17, 8), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        raw_weight = bytes(weight.view(torch.uint8).flatten().tolist())
        raw_scale = bytes(scale.view(torch.uint8).flatten().tolist())
        header = json.dumps({
            "embed.weight": {"dtype": "F8_E4M3", "shape": [17, 256],
                             "data_offsets": [0, len(raw_weight)]},
            "embed.scale": {"dtype": "F8_E8M0", "shape": [17, 8],
                            "data_offsets": [len(raw_weight), len(raw_weight) + len(raw_scale)]},
        }).encode()
        oracle = DeepseekV41EngramEmbedding(17, 256)
        oracle.weight = torch.nn.Parameter(weight, requires_grad=False)
        oracle.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "table.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + raw_weight + raw_scale)
            mapped = DeepSeekV41MappedEmbedding(path, "embed", max_gather_rows=3)
            try:
                self.assertEqual(list(mapped.parameters()), [])
                self.assertEqual(list(mapped.buffers()), [])
                for device in ("cpu", "cuda:0", "cuda:1"):
                    ids = torch.tensor([[[16, 0, 1], [3, 3, 16]]], device=device)
                    actual = mapped(ids)
                    torch.testing.assert_close(actual, oracle(ids), rtol=0, atol=0)
                    self.assertEqual(mapped(torch.empty((0,), dtype=torch.long, device=device)).shape, (0, 256))
            finally:
                mapped.close()


if __name__ == "__main__":
    unittest.main()
