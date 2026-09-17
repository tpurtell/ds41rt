"""Upstream-ported quant-config validation invariants.

Ported from:
- ``vllm/tests/quantization/test_quantization_config_args.py``: quant config
  argument parsing/validation invariant classes (unknown method rejected,
  group-size bounds, bits-per-weight validation, mutually exclusive options).
- ``llama.cpp/tests/test-quant-type-selection.cpp``: per-arch tensor-type
  selection goldens, ported as ds41rt's EXL3 format/role selection table
  (projection tensor dtypes/shapes per integer tier, per-stem name routing).

ds41rt's quant-config validation surface is ``ds41rt_runtime.exl3_quantizer``
(source-checkpoint and artifact-plan validation),
``ds41rt_runtime.exl3_artifact_contract`` (GPTQModel-native EXL3 publication
and inline mixed-tier policy validation), and
``ds41rt_runtime.exl3_experts.validate_exl3_expert_snapshot`` (fail-closed
calibrated-snapshot validation). The invariants are ported against those.

Mapping notes / gaps:
- vLLM's ``QuantSpec`` name registry and ``targets`` pattern matching have no
  ds41rt analog (ds41rt has no online-quantization target surface); those
  upstream tests are recorded as unmapped rather than approximated.
- llama.cpp's remote per-arch snapshot goldens require fetching upstream GGUF
  metadata; the ds41rt analog is the deterministic projection selection table
  in ``_generated_projection_tensors`` plus the checkpoint-native/GPTQModel
  name-routing helpers, which are pinned as local goldens.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ds41rt_runtime.exl3_artifact_contract import (
    INLINE_MIXED_SCHEMA,
    INLINE_MIXED_SCORE,
    RECIPE as CONTRACT_RECIPE,
    RECIPE_K3,
    _expected_projection_encoded_bytes,
    _recipe_for_bits,
    validate_inline_mixed_policy,
)
from ds41rt_runtime.exl3_experts import validate_exl3_expert_snapshot
from ds41rt_runtime.exl3_quantizer import (
    EXL3_BITS,
    EXL3_CODEBOOK,
    EXL3_RECIPE,
    EXL3_SCHEMA,
    EXL3_SCHEMA_VERSION,
    EXL3_TENSOR_FORMAT,
    EXLLAMAV3_REPOSITORY,
    EXLLAMAV3_REVISION,
    EXLLAMAV3_SOURCE_TREE_SHA256,
    EXLLAMAV3_VERSION,
    EXPERT_TP_WORLD_SIZE,
    NATIVE_SOURCE_FORMAT,
    ModelShape,
    _generated_projection_tensors,
    build_artifact_plan,
    exl3_quantization_config,
    expert_projection_base,
    gptqmodel_expert_projection_base,
    read_native_model_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_config() -> dict:
    """Minimal valid native FP4/FP8 source checkpoint config."""
    return {
        "model_type": "deepseek_v4",
        "hidden_size": 256,
        "moe_intermediate_size": 1024,
        "num_hidden_layers": 2,
        "dspark_target_layer_ids": [1],
        "n_routed_experts": 8,
        "num_experts_per_tok": 6,
        "swiglu_limit": 1.0,
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
    }


def _exl3_quant_config(*, recipe: str = EXL3_RECIPE) -> dict:
    return {
        "quant_method": "exl3",
        "version": EXLLAMAV3_VERSION,
        "bits": float(EXL3_BITS),
        "codebook": EXL3_CODEBOOK,
        "calibration": {
            "method": "layerwise_simulated_isotropic",
            "device": "cuda:0",
            "rows": 4096,
            "seed": 1234,
            "hessian": "analytic_identity",
            "distribution": "rms_normalized_isotropic",
            "activation": "silu",
            "swiglu_limit": 1.0,
            "gate_clamp": [None, 1.0],
            "up_clamp": [-1.0, 1.0],
        },
        "ds41rt": {
            "schema": EXL3_SCHEMA,
            "schema_version": EXL3_SCHEMA_VERSION,
            "recipe": recipe,
            "calibrated": True,
            "scope": "routed_experts",
            "source_format": NATIVE_SOURCE_FORMAT,
            "tensor_format": EXL3_TENSOR_FORMAT,
            "expert_tp_world_size": EXPERT_TP_WORLD_SIZE,
            "quantizer_source": {
                "repository": EXLLAMAV3_REPOSITORY,
                "revision": EXLLAMAV3_REVISION,
                "source_tree_sha256": EXLLAMAV3_SOURCE_TREE_SHA256,
            },
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return snapshot


def _shape() -> ModelShape:
    return ModelShape(
        hidden_size=256,
        intermediate_size=1024,
        hidden_layers=2,
        dspark_layers=1,
        experts=8,
        swiglu_limit=1.0,
    )


# ---------------------------------------------------------------------------
# Unknown method rejected (vLLM: test_quant_spec_rejects_unknown_name and
# resolve_* shorthand invariants)
# ---------------------------------------------------------------------------


def test_unknown_source_quant_method_rejected(tmp_path: Path) -> None:
    # Ported invariant: an unknown quantization method must be rejected, not
    # silently mapped to a fallback. ds41rt's EXL3 converter only accepts the
    # native FP4/FP8 source checkpoint.
    config = _source_config()
    config["quantization_config"] = {"quant_method": "marlin"}
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="native FP4/FP8"):
        read_native_model_config(snapshot)


def test_missing_quant_method_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["quantization_config"] = {"quant_method": ""}
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="native FP4/FP8"):
        read_native_model_config(snapshot)


def test_unknown_model_type_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["model_type"] = "llama"
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="model_type"):
        read_native_model_config(snapshot)


def test_unknown_expert_dtype_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["expert_dtype"] = "bf16"
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="expert_dtype"):
        read_native_model_config(snapshot)


# ---------------------------------------------------------------------------
# Group-size bounds (vLLM group-size invariants mapped onto ds41rt's EXL3
# H128 / TP4 alignment requirements)
# ---------------------------------------------------------------------------


def test_hidden_size_below_h128_group_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["hidden_size"] = 100  # not divisible by EXL3 H128 (128)
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="divisible by EXL3 H128"):
        read_native_model_config(snapshot)


def test_intermediate_size_below_tp4_group_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["moe_intermediate_size"] = 128  # not divisible by TP4 * 128 = 512
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="divisible by TP4 H128"):
        read_native_model_config(snapshot)


def test_group_size_exact_boundary_accepted(tmp_path: Path) -> None:
    config = _source_config()
    config["hidden_size"] = 128
    config["moe_intermediate_size"] = 512
    snapshot = _write_config(tmp_path, config)
    _, shape = read_native_model_config(snapshot)
    assert shape.hidden_size == 128
    assert shape.intermediate_size == 512
    assert shape.total_layers == 3  # 2 hidden + 1 dSpark


def test_nonpositive_geometry_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["n_routed_experts"] = 0
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="no routed expert layers"):
        read_native_model_config(snapshot)


def test_nonfinite_swiglu_limit_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["swiglu_limit"] = float("nan")
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="swiglu_limit"):
        read_native_model_config(snapshot)


# ---------------------------------------------------------------------------
# Bits-per-weight validation (vLLM bits invariants mapped onto EXL3 integer
# tiers K2/K3)
# ---------------------------------------------------------------------------


def test_global_tier_outside_k2_k3_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="integer K2 or K3"):
        build_artifact_plan(missing, exl3_bits=4)


def test_global_tier_bool_rejected(tmp_path: Path) -> None:
    # bool is an int subclass in Python; it must not sneak through as tier 1.
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="integer K2 or K3"):
        build_artifact_plan(missing, exl3_bits=True)


def test_per_projection_fractional_tier_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="integer K2 or K3"):
        build_artifact_plan(
            missing, exl3_projection_bits={"layers.0.ffn.experts.0.w1": 2.5}
        )


def test_per_projection_string_tier_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="integer K2 or K3"):
        build_artifact_plan(
            missing, exl3_projection_bits={"layers.0.ffn.experts.0.w1": "3"}
        )


def test_recipe_for_bits_goldens() -> None:
    # Tier -> recipe goldens: renames must not silently rewire publication.
    assert _recipe_for_bits(2) == CONTRACT_RECIPE
    assert _recipe_for_bits(3) == RECIPE_K3
    with pytest.raises(ValueError, match="unsupported EXL3 integer tier"):
        _recipe_for_bits(4)


def test_encoded_projection_bytes_use_physical_tiers() -> None:
    # The packed-byte accounting must multiply by each module's physical tier.
    quant = {"tensor_storage": {"a": {"bits_per_weight": 2}}}
    hidden, intermediate = 256, 1024
    assert _expected_projection_encoded_bytes(
        quant, hidden_size=hidden, intermediate_size=intermediate
    ) == (hidden // 16) * (intermediate // 16) * 32 * 2 + (hidden + intermediate) * 2 + 4


def test_encoded_projection_bytes_reject_unknown_tiers() -> None:
    assert (
        _expected_projection_encoded_bytes(
            {"tensor_storage": {"a": {"bits_per_weight": 4}}},
            hidden_size=256,
            intermediate_size=1024,
        )
        is None
    )
    # A float tier must not pass the strict `type(bits) is int` check.
    assert (
        _expected_projection_encoded_bytes(
            {"tensor_storage": {"a": {"bits_per_weight": 2.0}}},
            hidden_size=256,
            intermediate_size=1024,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Mutually exclusive / unsupported option combinations
# ---------------------------------------------------------------------------


def test_nonpositive_shard_budget_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="max_shard_bytes must be positive"):
        build_artifact_plan(missing, max_shard_bytes=0)


def test_unknown_expert_tensor_layout_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="unsupported EXL3 expert tensor layout"):
        build_artifact_plan(missing, expert_tensor_layout="bogus")


def test_nonpositive_calibration_rows_rejected() -> None:
    with pytest.raises(ValueError, match="calibration_rows must be positive"):
        exl3_quantization_config(calibration_rows=0, seed=1)


def test_nonpositive_swiglu_limit_in_quant_config_rejected() -> None:
    with pytest.raises(ValueError, match="swiglu_limit must be positive and finite"):
        exl3_quantization_config(calibration_rows=16, seed=1, swiglu_limit=-1.0)


def test_nonfinite_swiglu_limit_in_quant_config_rejected() -> None:
    with pytest.raises(ValueError, match="swiglu_limit must be positive and finite"):
        exl3_quantization_config(
            calibration_rows=16, seed=1, swiglu_limit=math.inf
        )


def test_empty_recipe_rejected() -> None:
    with pytest.raises(ValueError, match="EXL3 recipe must be non-empty"):
        exl3_quantization_config(calibration_rows=16, seed=1, recipe="")


def test_inline_mixed_policy_requires_object() -> None:
    with pytest.raises(ValueError, match="not an object"):
        validate_inline_mixed_policy(["not", "a", "dict"])


def test_inline_mixed_policy_private_tier_plan_rejected_when_portable() -> None:
    value = _valid_inline_mixed_policy()
    value["tier_plan_root"] = "/private/path"
    with pytest.raises(ValueError, match="private tier-plan path"):
        validate_inline_mixed_policy(value, require_portable=True)


def test_inline_mixed_policy_rejects_nonreduced_bitrate() -> None:
    value = _valid_inline_mixed_policy()
    value["extra_bits"] = {"numerator": 2, "denominator": 4}
    value["target_bpw"] = "5/2"
    with pytest.raises(ValueError, match="not reduced"):
        validate_inline_mixed_policy(value)


def test_inline_mixed_policy_rejects_bitrate_at_or_above_one() -> None:
    value = _valid_inline_mixed_policy()
    value["extra_bits"] = {"numerator": 1, "denominator": 1}
    with pytest.raises(ValueError, match="invalid exact bitrate"):
        validate_inline_mixed_policy(value)


def test_inline_mixed_policy_rejects_target_mismatch() -> None:
    value = _valid_inline_mixed_policy()
    value["target_bpw"] = "8/3"  # true target from 1/3 extra bits is 7/3
    with pytest.raises(ValueError, match="target disagrees"):
        validate_inline_mixed_policy(value)


def test_inline_mixed_policy_valid_golden_accepted() -> None:
    value = _valid_inline_mixed_policy()
    assert validate_inline_mixed_policy(value) == value


def _valid_inline_mixed_policy() -> dict:
    return {
        "schema": INLINE_MIXED_SCHEMA,
        "schema_version": 1,
        "namespace": "base",
        "base_bits": 2,
        "upgrade_bits": 3,
        "extra_bits": {"numerator": 1, "denominator": 3},
        "target_bpw": "7/3",
        "score_kind": INLINE_MIXED_SCORE,
        "projection_ratio": {"w1": 2, "w3": 1, "w2": 1},
    }


# ---------------------------------------------------------------------------
# Fail-closed calibrated-snapshot validation (ds41rt's own contract, in the
# spirit of vLLM's resolve/validate invariants)
# ---------------------------------------------------------------------------


def test_unknown_recipe_rejected(tmp_path: Path) -> None:
    config = _source_config()
    config["quantization_config"] = _exl3_quant_config(
        recipe="deepseek_v4_exl3_trellis_9bpw_v99"
    )
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="recognized checkpoint-bound"):
        validate_exl3_expert_snapshot(snapshot)


def test_wrong_top_level_bits_rejected(tmp_path: Path) -> None:
    config = _source_config()
    quant = _exl3_quant_config()
    quant["bits"] = 3.0  # K3 tier in a K2 contract
    config["quantization_config"] = quant
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="quantization_config.bits"):
        validate_exl3_expert_snapshot(snapshot)


def test_wrong_codebook_rejected(tmp_path: Path) -> None:
    config = _source_config()
    quant = _exl3_quant_config()
    quant["codebook"] = "nvfp4"
    config["quantization_config"] = quant
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="quantization_config.codebook"):
        validate_exl3_expert_snapshot(snapshot)


def test_wrong_schema_version_rejected(tmp_path: Path) -> None:
    config = _source_config()
    quant = _exl3_quant_config()
    quant["ds41rt"]["schema_version"] = 2
    config["quantization_config"] = quant
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="ds41rt.schema_version"):
        validate_exl3_expert_snapshot(snapshot)


def test_tampered_quantizer_provenance_rejected(tmp_path: Path) -> None:
    config = _source_config()
    quant = _exl3_quant_config()
    quant["ds41rt"]["quantizer_source"]["revision"] = "f" * 40
    config["quantization_config"] = quant
    snapshot = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="provenance"):
        validate_exl3_expert_snapshot(snapshot)


def test_valid_minimal_exl3_config_accepted(tmp_path: Path) -> None:
    config = _source_config()
    config["quantization_config"] = _exl3_quant_config()
    snapshot = _write_config(tmp_path, config)
    validated = validate_exl3_expert_snapshot(snapshot)
    assert validated.bits == EXL3_BITS == 2
    assert validated.config.top_k == 6
    assert validated.config.global_experts == 8


# ---------------------------------------------------------------------------
# Format-selection table goldens (llama.cpp test-quant-type-selection.cpp):
# per-tier projection tensor dtypes/shapes and per-stem name routing.
# ---------------------------------------------------------------------------


def test_projection_tensor_selection_table_k2() -> None:
    tensors = {
        tensor.name: tensor
        for tensor in _generated_projection_tensors("expert.w1", 512, 1024, bits=2)
    }
    assert set(tensors) == {"expert.w1.trellis", "expert.w1.suh", "expert.w1.svh", "expert.w1.mcg"}
    trellis = tensors["expert.w1.trellis"]
    assert trellis.dtype == "I16"
    assert trellis.shape == (512 // 16, 1024 // 16, 16 * 2)
    assert trellis.nbytes == (512 // 16) * (1024 // 16) * 32 * 2
    assert tensors["expert.w1.suh"].dtype == "F16"
    assert tensors["expert.w1.suh"].shape == (512,)
    assert tensors["expert.w1.svh"].dtype == "F16"
    assert tensors["expert.w1.svh"].shape == (1024,)
    assert tensors["expert.w1.mcg"].dtype == "I32"
    assert tensors["expert.w1.mcg"].shape == ()
    assert tensors["expert.w1.mcg"].nbytes == 4


def test_projection_tensor_selection_table_k3() -> None:
    tensors = {
        tensor.name: tensor
        for tensor in _generated_projection_tensors("expert.w2", 1024, 512, bits=3)
    }
    trellis = tensors["expert.w2.trellis"]
    assert trellis.dtype == "I16"
    assert trellis.shape == (1024 // 16, 512 // 16, 16 * 3)
    assert trellis.nbytes == (1024 // 16) * (512 // 16) * 48 * 2
    assert tensors["expert.w2.suh"].shape == (1024,)
    assert tensors["expert.w2.svh"].shape == (512,)


def test_checkpoint_native_projection_name_routing() -> None:
    shape = _shape()
    # Hidden layers use the flat "layers.N" namespace.
    assert (
        expert_projection_base(shape, 0, 3, "w1")
        == "layers.0.ffn.experts.3.w1"
    )
    # dSpark blocks route to "mtp.N" with re-based indexing.
    assert (
        expert_projection_base(shape, 2, 5, "w2")
        == "mtp.0.ffn.experts.5.w2"
    )


def test_checkpoint_native_projection_bounds_enforced() -> None:
    shape = _shape()
    with pytest.raises(ValueError, match="outside"):
        expert_projection_base(shape, 3, 0, "w1")  # layer beyond total
    with pytest.raises(ValueError, match="outside"):
        expert_projection_base(shape, 0, 8, "w1")  # expert beyond range
    with pytest.raises(ValueError, match="unsupported routed projection"):
        expert_projection_base(shape, 0, 0, "w4")


def test_gptqmodel_projection_name_routing_goldens() -> None:
    shape = _shape()
    assert (
        gptqmodel_expert_projection_base(shape, 0, 1, "w1")
        == "model.layers.0.mlp.experts.1.gate_proj"
    )
    assert (
        gptqmodel_expert_projection_base(shape, 0, 1, "w3")
        == "model.layers.0.mlp.experts.1.up_proj"
    )
    assert (
        gptqmodel_expert_projection_base(shape, 2, 1, "w2")
        == "mtp.0.mlp.experts.1.down_proj"
    )
    with pytest.raises(ValueError, match="unsupported routed projection"):
        gptqmodel_expert_projection_base(shape, 0, 1, "w4")
