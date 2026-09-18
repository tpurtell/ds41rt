//! ModelOpt NVFP4 routed-expert contract for the native V4.1 engine.
//!
//! `nvidia/DeepSeek-V4.1-Flash-NVFP4` quantizes only the routed experts:
//! every backbone expert projection stores a packed E2M1 payload, an F8_E4M3
//! per-16-group scale plane, a per-tensor FP32 global weight scale and a
//! per-tensor FP32 activation scale (the W4A4 indication). Attention,
//! shared experts, embeddings, head, vision and the MTP draft experts stay
//! on the official FP8/MXFP4 layout, so the MTP path is unchanged.
use crate::{OfficialV41Config, OFFICIAL_V41_MODEL_ID};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

/// On-disk routed-expert tensor layout for an NVFP4 publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Nvfp4ExpertLayout {
    /// Packed E2M1 (U8) + F8_E4M3 per-16 scales + FP32 tensor scale +
    /// FP32 activation scale, one set per w1/w3/w2 projection.
    ModelOpt,
}

/// Validated NVFP4 expert geometry and quantization contract.
#[derive(Debug, Clone)]
pub struct V41Nvfp4Contract {
    pub config: OfficialV41Config,
    pub group_size: usize,
    pub layout: V41Nvfp4ExpertLayout,
}

const NVFP4_GROUP_SIZE: usize = 16;
const NVFP4_OFFICIAL_SCALE_BLOCK: [usize; 2] = [32, 32];
const NVFP4_IGNORED_PATTERNS: [&str; 4] = [
    "*.attn.*",
    "*.ffn.shared_experts.*",
    "head",
    "mtp.*",
];

/// True when a raw config.json declares the ModelOpt NVFP4 expert recipe.
pub fn is_v41_nvfp4_publication(quantization_config: &Value) -> bool {
    quantization_config
        .get("moe_quant_algo")
        .and_then(Value::as_str)
        == Some("NVFP4")
}

/// Read and validate the NVFP4 expert contract, returning `None` for every
/// checkpoint that does not declare it.
pub fn read_v41_nvfp4_contract(snapshot: &Path) -> Result<Option<V41Nvfp4Contract>> {
    let raw = crate::v41_exl3::read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let Some(quant) = raw.get("quantization_config") else {
        return Ok(None);
    };
    if !is_v41_nvfp4_publication(quant) {
        return Ok(None);
    }
    parse_v41_nvfp4_contract(raw).map(Some)
}

fn parse_v41_nvfp4_contract(mut config: Value) -> Result<V41Nvfp4Contract> {
    let quant = config["quantization_config"]
        .as_object()
        .context("NVFP4 publication requires a quantization_config object")?;
    ensure!(
        quant.get("quant_method").and_then(Value::as_str) == Some("fp8")
            && quant.get("quant_algo").and_then(Value::as_str) == Some("MIXED_PRECISION")
            && quant.get("moe_quant_algo").and_then(Value::as_str) == Some("NVFP4")
            && quant.get("expert_dtype").and_then(Value::as_str) == Some("fp4")
            && quant.get("activation_scheme").and_then(Value::as_str) == Some("dynamic")
            && quant.get("group_size").and_then(Value::as_u64) == Some(NVFP4_GROUP_SIZE as u64),
        "unsupported NVFP4 publication quantization contract"
    );
    ensure!(
        quant.get("weight_block_size") == Some(&serde_json::json!(NVFP4_OFFICIAL_SCALE_BLOCK)),
        "NVFP4 publication must keep the official FP8 block for non-routed tensors"
    );
    ensure!(
        quant.get("ignore") == Some(&serde_json::json!(NVFP4_IGNORED_PATTERNS)),
        "NVFP4 publication must ignore attention, shared experts, head and MTP"
    );
    // W4A4: both operands are 4-bit floats in 16-wide groups.
    let group = quant
        .get("config_groups")
        .and_then(|groups| groups.get("group_0"))
        .context("NVFP4 publication requires config_groups.group_0")?;
    ensure!(
        group.get("targets") == Some(&serde_json::json!(["Linear"])),
        "NVFP4 config_groups.group_0 must target Linear layers"
    );
    for operand in ["weights", "input_activations"] {
        let spec = group
            .get(operand)
            .with_context(|| format!("NVFP4 config_groups.group_0.{operand} missing"))?;
        ensure!(
            spec.get("num_bits").and_then(Value::as_u64) == Some(4)
                && spec.get("group_size").and_then(Value::as_u64) == Some(NVFP4_GROUP_SIZE as u64)
                && spec.get("type").and_then(Value::as_str) == Some("float"),
            "NVFP4 {operand} must be a 4-bit float in {NVFP4_GROUP_SIZE}-wide groups"
        );
    }
    let producer = quant
        .get("producer")
        .context("NVFP4 publication requires a producer identity")?;
    ensure!(
        producer.get("name").and_then(Value::as_str) == Some("modelopt"),
        "NVFP4 routed experts must come from ModelOpt"
    );
    ensure!(
        quant.get("kv_cache_quant_algo").map_or(true, Value::is_null),
        "NVFP4 publication must leave the KV cache unquantized"
    );

    // Substitute the canonical official FP8 block so the strict official
    // config validation covers every architecture and non-expert field.
    let quantized_layers = quant
        .get("quantized_layers")
        .and_then(Value::as_object)
        .context("NVFP4 publication requires quantized_layers")?
        .clone();
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": NVFP4_OFFICIAL_SCALE_BLOCK,
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    });
    let validated =
        OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, &serde_json::to_vec(&config)?)?;
    let text = validated.text();
    ensure!(
        quantized_layers.len() == text.num_hidden_layers,
        "NVFP4 publication must quantize every hidden layer's experts"
    );
    let mut experts = 0usize;
    for layer in 0..text.num_hidden_layers {
        let name = format!("layers.{layer}.ffn.experts");
        let entry = quantized_layers
            .get(&name)
            .with_context(|| format!("NVFP4 publication is missing {name}"))?;
        ensure!(
            entry.get("quant_algo").and_then(Value::as_str) == Some("NVFP4")
                && entry.get("group_size").and_then(Value::as_u64)
                    == Some(NVFP4_GROUP_SIZE as u64),
            "NVFP4 {name} must declare NVFP4 group {NVFP4_GROUP_SIZE}"
        );
        experts += 1;
    }
    ensure!(
        experts == text.num_hidden_layers,
        "NVFP4 routed-expert coverage mismatch"
    );
    Ok(V41Nvfp4Contract {
        config: validated,
        group_size: NVFP4_GROUP_SIZE,
        layout: V41Nvfp4ExpertLayout::ModelOpt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nvfp4_config() -> Value {
        let mut config: Value =
            serde_json::from_str(include_str!("official-v41-config.json")).unwrap();
        let layers: serde_json::Map<String, Value> = (0..40)
            .map(|layer| {
                (
                    format!("layers.{layer}.ffn.experts"),
                    serde_json::json!({"group_size": 16, "quant_algo": "NVFP4"}),
                )
            })
            .collect();
        config["quantization_config"] = serde_json::json!({
            "quant_method": "fp8",
            "quant_algo": "MIXED_PRECISION",
            "moe_quant_algo": "NVFP4",
            "expert_dtype": "fp4",
            "activation_scheme": "dynamic",
            "group_size": 16,
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32],
            "kv_cache_quant_algo": null,
            "ignore": NVFP4_IGNORED_PATTERNS,
            "producer": {"name": "modelopt", "version": "dsv4-nvfp4-experts"},
            "quantized_layers": Value::Object(layers),
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"},
                    "input_activations": {"dynamic": false, "group_size": 16, "num_bits": 4, "type": "float"}
                }
            }
        });
        config
    }

    #[test]
    fn accepts_the_declared_w4a4_expert_contract() {
        let contract = parse_v41_nvfp4_contract(nvfp4_config()).unwrap();
        assert_eq!(contract.group_size, 16);
        assert_eq!(contract.layout, V41Nvfp4ExpertLayout::ModelOpt);
        assert_eq!(contract.config.text().n_routed_experts, 384);
        assert!(contract
            .config
            .quantization()
            .quant_method
            .eq_ignore_ascii_case("fp8"));
    }

    #[test]
    fn rejects_contract_drift() {
        for (pointer, value, expected) in [
            (
                "/quantization_config/moe_quant_algo",
                serde_json::json!("MXFP4"),
                "is_v41_nvfp4_publication",
            ),
            (
                "/quantization_config/expert_dtype",
                serde_json::json!("fp8"),
                "unsupported NVFP4",
            ),
            (
                "/quantization_config/group_size",
                serde_json::json!(32),
                "unsupported NVFP4",
            ),
            (
                "/quantization_config/config_groups/group_0/input_activations/num_bits",
                serde_json::json!(8),
                "4-bit float",
            ),
            (
                "/quantization_config/ignore",
                serde_json::json!(["head"]),
                "must ignore",
            ),
        ] {
            let mut config = nvfp4_config();
            *config.pointer_mut(pointer).unwrap() = value;
            let result = parse_v41_nvfp4_contract(config);
            if expected == "is_v41_nvfp4_publication" {
                // The selector turns other MoE algorithms away before parsing.
                assert!(!is_v41_nvfp4_publication(
                    &serde_json::json!({"moe_quant_algo": "MXFP4"})
                ));
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(expected), "unexpected error: {error}");
            }
        }
        let mut missing = nvfp4_config();
        missing["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .remove("layers.39.ffn.experts");
        assert!(parse_v41_nvfp4_contract(missing)
            .unwrap_err()
            .to_string()
            .contains("every hidden layer"));
    }

    #[test]
    #[ignore = "requires DS41RT_NVFP4_SNAPSHOT pointing to a local ModelOpt NVFP4 publication"]
    fn real_publication_contract() {
        let path = std::env::var_os("DS41RT_NVFP4_SNAPSHOT").expect("DS41RT_NVFP4_SNAPSHOT");
        let contract = read_v41_nvfp4_contract(Path::new(&path)).unwrap().unwrap();
        assert_eq!(contract.group_size, 16);
        assert_eq!(contract.layout, V41Nvfp4ExpertLayout::ModelOpt);
        assert_eq!(contract.config.text().num_hidden_layers, 40);
        println!("validated ModelOpt NVFP4 expert contract");
    }
}
