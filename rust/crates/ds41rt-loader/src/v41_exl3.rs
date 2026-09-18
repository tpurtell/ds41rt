//! Validated routed-only EXL3 metadata for the native V4.1 engine.
use crate::{OfficialV41Config, SafetensorsTensorMetadata, OFFICIAL_V41_MODEL_ID};
use anyhow::{ensure, Context, Result};
use ds41rt_core::DType;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    ops::Range,
    path::Path,
};

pub const V41_EXL3_SCHEMA: &str = "ds41rt.v41-routed-exl3.v1";
const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3ProjectionKind {
    Gate,
    Up,
    Down,
}

/// Explicit resident layout; paired storage requires ownership-aware execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V41Exl3Partition {
    Disjoint,
    PairedTp4,
}

/// One projection's on-disk contract, before conversion into native kernel tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V41Exl3Projection {
    pub name: String,
    pub kind: V41Exl3ProjectionKind,
    pub bits: usize,
    pub input_features: usize,
    pub output_features: usize,
}

impl V41Exl3Projection {
    pub fn trellis_shape(&self) -> [usize; 3] {
        [
            self.input_features / 16,
            self.output_features / 16,
            16 * self.bits,
        ]
    }

    pub fn trellis_bytes(&self) -> usize {
        self.input_features * self.output_features * self.bits / 8
    }

    /// Check the physical safetensors header against the declared projection.
    /// The manifest alone is never proof of the actual checkpoint's layout.
    pub fn validate_tensor(&self, tensor: &SafetensorsTensorMetadata) -> Result<()> {
        let (prefix, suffix) = tensor
            .name
            .rsplit_once('.')
            .context("missing EXL3 tensor suffix")?;
        ensure!(
            prefix == self.name,
            "EXL3 tensor belongs to another projection"
        );
        let (dtype, shape, bytes) = match suffix {
            "trellis" => (
                DType::I16,
                self.trellis_shape().to_vec(),
                self.trellis_bytes(),
            ),
            "suh" => (
                DType::F16,
                vec![self.input_features],
                self.input_features * 2,
            ),
            "svh" => (
                DType::F16,
                vec![self.output_features],
                self.output_features * 2,
            ),
            "mcg" => (DType::I32, vec![], 4),
            _ => anyhow::bail!("unexpected EXL3 tensor {}", tensor.name),
        };
        ensure!(
            tensor.dtype == dtype && tensor.shape == shape && tensor.byte_length == bytes as u64,
            "EXL3 manifest/header mismatch for {}",
            tensor.name
        );
        Ok(())
    }

    /// Partition complete H128 rotation blocks. Equal TP4 slices of 2304
    /// channels would split blocks, so the first two ranks own one extra block.
    pub fn intermediate_partition(&self, world: usize, rank: usize) -> Result<Range<usize>> {
        self.intermediate_partition_with_layout(world, rank, V41Exl3Partition::Disjoint)
    }

    pub fn intermediate_partition_with_layout(
        &self,
        world: usize,
        rank: usize,
        layout: V41Exl3Partition,
    ) -> Result<Range<usize>> {
        ensure!(
            matches!(world, 1 | 2 | 4) && rank < world,
            "invalid EXL3 TP rank/world"
        );
        let intermediate = match self.kind {
            V41Exl3ProjectionKind::Down => self.input_features,
            _ => self.output_features,
        };
        ensure!(
            intermediate % 128 == 0,
            "EXL3 intermediate axis is not H128 aligned"
        );
        if layout == V41Exl3Partition::PairedTp4 {
            ensure!(
                world == 4 && intermediate == 2304,
                "paired EXL3 layout requires TP4 with 2304 intermediate channels"
            );
            let blocks = &ds41rt_core::EXL3_TP4_RESIDENT_BLOCKS[rank];
            return Ok(blocks.start * 128..blocks.end * 128);
        }
        let blocks = intermediate / 128;
        ensure!(blocks >= world, "EXL3 partition would be empty");
        let count = blocks / world + usize::from(rank < blocks % world);
        let start = rank * (blocks / world) + rank.min(blocks % world);
        Ok(start * 128..(start + count) * 128)
    }
}

/// How the checkpoint stores its MTP (dSpark draft) routed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41Exl3MtpExperts {
    /// DS41RT-staged snapshots quantize draft experts to EXL3 as well.
    Exl3,
    /// Raw publications keep draft experts at the official native FP4
    /// (MXFP4) source precision; the draft path must load native weights.
    Source,
}

#[derive(Debug)]
pub struct V41Exl3Manifest {
    /// Validated original non-routed model geometry and quantization contract.
    pub config: OfficialV41Config,
    pub projections: BTreeMap<String, V41Exl3Projection>,
    /// One checkpoint-wide family, shared by all target and draft layers.
    pub(crate) decoder_tiers: Vec<usize>,
    /// Retained for validation by the PLE storage path; never silently discarded.
    pub ple_quantization: Option<Value>,
    pub(crate) mtp_experts: V41Exl3MtpExperts,
}

impl V41Exl3Manifest {
    pub fn decoder_tiers(&self) -> &[usize] {
        &self.decoder_tiers
    }

    /// True when draft experts stayed at the native FP4 source precision.
    pub fn mtp_experts_are_source(&self) -> bool {
        self.mtp_experts == V41Exl3MtpExperts::Source
    }
}

pub(crate) fn decoder_family(
    projections: &BTreeMap<String, V41Exl3Projection>,
) -> Result<Vec<usize>> {
    let mut bits: BTreeSet<_> = projections.values().map(|p| p.bits).collect();
    ensure!(
        !bits.is_empty() && bits.iter().all(|b| (2..=5).contains(b)),
        "EXL3 decoder family must contain K2..K5 projections"
    );
    // Native mixed kernels retain an empty adjacent tier for uniform models.
    if bits.len() == 1 {
        let bit = *bits.first().unwrap();
        bits.insert(if bit == 5 { 4 } else { bit + 1 });
    }
    Ok(bits.into_iter().collect())
}

pub(crate) fn read_json(path: &Path, limit: u64) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "{} exceeds metadata size limit",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn read_v41_exl3_manifest(snapshot: &Path) -> Result<V41Exl3Manifest> {
    let config = read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let manifest_path = snapshot.join("quantize_config.json");
    if manifest_path.is_file() {
        let manifest = read_json(&manifest_path, MAX_MANIFEST_BYTES)?;
        parse_manifest(config, &manifest)
    } else {
        parse_raw_publication(config)
    }
}

/// First-class support for raw exllamav3-style V4.1 Flash EXL3 publications
/// that ship only a compact `config.json` quantization block (no
/// quantize_config.json staging manifest). The accepted contract is strict:
/// integer K2..K5 uniform routed experts with the MCG codebook, checkpoint-
/// native tensor naming, source-precision MTP draft experts, and the
/// official FP8 block for every non-routed tensor. Tensor payload layouts
/// are verified against the safetensors headers by the catalog, so the
/// config only needs to establish geometry-independent facts.
fn parse_raw_publication(mut config: Value) -> Result<V41Exl3Manifest> {
    let quant = config["quantization_config"]
        .as_object()
        .context("raw EXL3 publication requires a quantization_config object")?;
    ensure!(
        quant.get("quant_method").and_then(Value::as_str) == Some("exl3"),
        "raw EXL3 publication requires quant_method=exl3"
    );
    let bits = quant["bits"]
        .as_u64()
        .filter(|b| (2..=5).contains(b))
        .context("raw EXL3 publication requires integer bits in 2..=5")? as usize;
    ensure!(
        quant.get("codebook").and_then(Value::as_str) == Some("mcg"),
        "raw EXL3 publication requires the MCG codebook"
    );
    ensure!(
        quant.get("mtp_experts").and_then(Value::as_str) == Some("source"),
        "raw EXL3 publication requires mtp_experts=source"
    );
    // Optional exllamav3 storage hints, when present, must match the only
    // layout the engine consumes.
    for (key, wanted) in [
        ("out_scales", Value::from("never")),
        ("group_size", Value::from(-1)),
        ("desc_act", Value::from(false)),
        ("pack_dtype", Value::from("int32")),
    ] {
        if let Some(found) = quant.get(key) {
            ensure!(
                found == &wanted,
                "raw EXL3 publication has unsupported {key}={found}"
            );
        }
    }
    let native = quant
        .get("non_routed_quantization")
        .context("raw EXL3 publication requires non_routed_quantization")?;
    ensure!(
        native["quant_method"] == "deepseek_v4_fp8"
            && native["fmt"] == "e4m3"
            && native["activation_scheme"] == "dynamic"
            && native["scale_fmt"] == "ue8m0"
            && native["weight_block_size"] == serde_json::json!([32, 32])
            && native["expert_dtype"] == "fp4",
        "raw EXL3 publication has an unsupported non-routed FP8 contract"
    );
    // Every contract key must be understood; reject silent drift.
    for key in quant.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "quant_method"
                    | "bits"
                    | "codebook"
                    | "mtp_experts"
                    | "mtp_experts_start_layer"
                    | "weight_block_size"
                    | "non_routed_quantization"
                    | "out_scales"
                    | "group_size"
                    | "desc_act"
                    | "pack_dtype"
            ),
            "raw EXL3 publication has unexpected quantization_config key {key}"
        );
    }
    let mtp_start_layer = quant
        .get("mtp_experts_start_layer")
        .and_then(Value::as_u64);
    // Substitute the canonical official FP8 block so the strict official
    // config validation covers every non-quantization field.
    config["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    });
    let validated =
        OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, &serde_json::to_vec(&config)?)?;
    let text = validated.text();
    if let Some(start_layer) = mtp_start_layer {
        ensure!(
            start_layer == text.num_hidden_layers as u64,
            "raw EXL3 mtp_experts_start_layer {start_layer} disagrees with the architecture"
        );
    }
    let mut projections = BTreeMap::new();
    for layer in 0..text.num_hidden_layers {
        for expert in 0..text.n_routed_experts {
            for (stem, kind) in [
                ("w1", V41Exl3ProjectionKind::Gate),
                ("w3", V41Exl3ProjectionKind::Up),
                ("w2", V41Exl3ProjectionKind::Down),
            ] {
                let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                    (text.moe_intermediate_size, text.hidden_size)
                } else {
                    (text.hidden_size, text.moe_intermediate_size)
                };
                let name = format!("layers.{layer}.ffn.experts.{expert}.{stem}");
                projections.insert(
                    name.clone(),
                    V41Exl3Projection {
                        name,
                        kind,
                        bits,
                        input_features: input,
                        output_features: output,
                    },
                );
            }
        }
    }
    Ok(V41Exl3Manifest {
        decoder_tiers: decoder_family(&projections)?,
        config: validated,
        projections,
        ple_quantization: None,
        mtp_experts: V41Exl3MtpExperts::Source,
    })
}

fn parse_manifest(mut config: Value, manifest: &Value) -> Result<V41Exl3Manifest> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        ensure!(
            manifest.get(field).and_then(Value::as_str) == Some("exl3"),
            "EXL3 manifest requires {field}=exl3"
        );
    }
    ensure!(
        manifest["codebook"] == "mcg"
            && manifest["out_scales"] == "never"
            && manifest["group_size"] == -1
            && manifest["desc_act"] == false
            && manifest["pack_dtype"] == "int32",
        "unsupported EXL3 storage contract"
    );
    ensure!(
        manifest["bits"]
            .as_u64()
            .is_some_and(|v| (2..=5).contains(&v)),
        "EXL3 base bits must be an integer in 2..5"
    );
    let meta = manifest
        .pointer("/meta/ds41rt")
        .context("missing V4.1 EXL3 metadata")?;
    ensure!(
        meta["schema"] == V41_EXL3_SCHEMA && meta["tensor_naming"] == "checkpoint-native",
        "unsupported V4.1 EXL3 schema or namespace"
    );
    let compact = config["quantization_config"]
        .as_object()
        .context("missing compact EXL3 config")?;
    ensure!(
        !compact.is_empty() && compact.get("quant_method") == Some(&Value::from("exl3")),
        "config must declare EXL3 quantization"
    );
    for (key, value) in compact {
        ensure!(
            manifest.get(key) == Some(value),
            "compact/full EXL3 metadata disagree at {key}"
        );
    }
    config["quantization_config"] = meta["native_quantization_config"].clone();
    let ple_quantization = config
        .as_object_mut()
        .context("config must be an object")?
        .remove("ds41rt_ple_quantization");
    let validated =
        OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, &serde_json::to_vec(&config)?)?;
    let text = validated.text();
    let storage = manifest["tensor_storage"]
        .as_object()
        .context("missing EXL3 tensor_storage")?;
    let mut projections = BTreeMap::new();
    for (prefix, layers, experts) in [
        ("layers", text.num_hidden_layers, text.n_routed_experts),
        (
            "mtp",
            validated.dspark_compress_ratios().len(),
            text.dspark_n_routed_experts,
        ),
    ] {
        for layer in 0..layers {
            for expert in 0..experts {
                for (stem, kind) in [
                    ("w1", V41Exl3ProjectionKind::Gate),
                    ("w3", V41Exl3ProjectionKind::Up),
                    ("w2", V41Exl3ProjectionKind::Down),
                ] {
                    let name = format!("{prefix}.{layer}.ffn.experts.{expert}.{stem}");
                    let value = storage
                        .get(&name)
                        .with_context(|| format!("missing projection {name}"))?;
                    let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                        (text.moe_intermediate_size, text.hidden_size)
                    } else {
                        (text.hidden_size, text.moe_intermediate_size)
                    };
                    let projection = parse_projection(&name, kind, input, output, value)?;
                    projections.insert(name, projection);
                }
            }
        }
    }
    ensure!(
        storage.len() == projections.len(),
        "EXL3 manifest includes non-routed or unexpected projections"
    );
    Ok(V41Exl3Manifest {
        config: validated,
        decoder_tiers: decoder_family(&projections)?,
        projections,
        ple_quantization,
        mtp_experts: V41Exl3MtpExperts::Exl3,
    })
}

fn parse_projection(
    name: &str,
    kind: V41Exl3ProjectionKind,
    input: usize,
    output: usize,
    value: &Value,
) -> Result<V41Exl3Projection> {
    ensure!(
        value["quant_format"] == "exl3",
        "projection {name} is not EXL3"
    );
    let bits = value["bits_per_weight"]
        .as_u64()
        .filter(|b| (2..=5).contains(b))
        .with_context(|| format!("projection {name} must have integer K2..K5"))?
        as usize;
    ensure!(
        input > 0 && output > 0 && input % 128 == 0 && output % 128 == 0,
        "projection {name} requires H128-aligned geometry"
    );
    let projection = V41Exl3Projection {
        name: name.to_owned(),
        kind,
        bits,
        input_features: input,
        output_features: output,
    };
    let tensors = value["stored_tensors"]
        .as_object()
        .context("missing projection storage")?;
    ensure!(
        tensors.len() == 4,
        "projection {name} must contain exactly trellis/suh/svh/mcg"
    );
    for (suffix, dtype, shape) in [
        ("trellis", "int16", projection.trellis_shape().to_vec()),
        ("suh", "float16", vec![input]),
        ("svh", "float16", vec![output]),
        ("mcg", "int32", vec![]),
    ] {
        let key = format!("{name}.{suffix}");
        let tensor = tensors
            .get(&key)
            .with_context(|| format!("missing {key}"))?;
        ensure!(
            tensor["torch_dtype"] == dtype && tensor["shape"] == serde_json::to_value(shape)?,
            "invalid EXL3 dtype/shape for {key}"
        );
    }
    Ok(projection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_publication_config(bits: u64) -> Value {
        let mut config: Value =
            serde_json::from_str(include_str!("official-v41-config.json")).unwrap();
        config["quantization_config"] = serde_json::json!({
            "quant_method": "exl3",
            "bits": bits,
            "codebook": "mcg",
            "mtp_experts": "source",
            "mtp_experts_start_layer": 40,
            "weight_block_size": [32, 32],
            "non_routed_quantization": {
                "quant_method": "deepseek_v4_fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "scale_fmt": "ue8m0",
                "weight_block_size": [32, 32],
                "expert_dtype": "fp4",
            },
        });
        config
    }

    #[test]
    fn raw_publication_manifest_is_uniform_backbone_only() {
        let manifest = parse_raw_publication(raw_publication_config(2)).unwrap();
        assert!(manifest.mtp_experts_are_source());
        assert!(manifest.ple_quantization.is_none());
        assert_eq!(manifest.decoder_tiers(), &[2, 3]);
        assert_eq!(manifest.projections.len(), 40 * 384 * 3);
        assert!(manifest
            .projections
            .keys()
            .all(|name| name.starts_with("layers.")));
        let projection = &manifest.projections["layers.39.ffn.experts.383.w1"];
        assert_eq!(projection.bits, 2);
        assert_eq!(projection.trellis_shape(), [320, 144, 32]);
        let down = &manifest.projections["layers.0.ffn.experts.0.w2"];
        assert_eq!(down.trellis_shape(), [144, 320, 32]);
        assert!(manifest
            .config
            .quantization()
            .quant_method
            .eq_ignore_ascii_case("fp8"));
    }

    #[test]
    fn raw_publication_rejects_contract_drift() {
        for (pointer, value, expected) in [
            ("/quantization_config/bits", serde_json::json!(3.25), "integer bits"),
            (
                "/quantization_config/codebook",
                serde_json::json!("gptq"),
                "MCG codebook",
            ),
            (
                "/quantization_config/mtp_experts",
                serde_json::json!("exl3"),
                "mtp_experts=source",
            ),
            (
                "/quantization_config/non_routed_quantization/fmt",
                serde_json::json!("e5m2"),
                "non-routed FP8",
            ),
        ] {
            let mut config = raw_publication_config(2);
            *config.pointer_mut(pointer).unwrap() = value;
            let error = parse_raw_publication(config).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
        let mut extra = raw_publication_config(2);
        extra["quantization_config"]["surprise"] = serde_json::json!(true);
        assert!(parse_raw_publication(extra)
            .unwrap_err()
            .to_string()
            .contains("unexpected quantization_config key"));
        let mut wrong_start = raw_publication_config(2);
        wrong_start["quantization_config"]["mtp_experts_start_layer"] = serde_json::json!(39);
        assert!(parse_raw_publication(wrong_start)
            .unwrap_err()
            .to_string()
            .contains("mtp_experts_start_layer"));
    }

    #[test]
    #[ignore = "requires DS41RT_EXL3_RAW_SNAPSHOT pointing to a raw local publication"]
    fn raw_publication_checkpoint_manifest() {
        let path = std::env::var_os("DS41RT_EXL3_RAW_SNAPSHOT").expect("DS41RT_EXL3_RAW_SNAPSHOT");
        let manifest = read_v41_exl3_manifest(Path::new(&path)).unwrap();
        assert!(manifest.mtp_experts_are_source());
        assert_eq!(manifest.projections.len(), 46_080);
        assert_eq!(manifest.decoder_tiers(), &[2, 3]);
        assert!(manifest
            .projections
            .values()
            .all(|projection| projection.bits == 2));
        let index = read_json(
            &Path::new(&path).join("model.safetensors.index.json"),
            MAX_MANIFEST_BYTES,
        )
        .unwrap();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for shard in shards {
            for tensor in crate::read_safetensors_metadata(&Path::new(&path).join(shard)).unwrap() {
                if !tensor.name.starts_with("layers.") || !tensor.name.contains(".ffn.experts.") {
                    continue;
                }
                let (prefix, _) = tensor.name.rsplit_once('.').unwrap();
                manifest
                    .projections
                    .get(prefix)
                    .expect("unexpected routed tensor")
                    .validate_tensor(&tensor)
                    .unwrap();
                assert!(seen.insert(tensor.name), "duplicate routed tensor");
            }
        }
        assert_eq!(seen.len(), 4 * manifest.projections.len());
        println!(
            "validated {} raw K2 projections",
            manifest.projections.len()
        );
    }

    #[test]
    #[ignore = "requires DS41RT_EXL3_SNAPSHOT pointing to a published local snapshot"]
    fn published_checkpoint_manifest() {
        let path = std::env::var_os("DS41RT_EXL3_SNAPSHOT").expect("DS41RT_EXL3_SNAPSHOT");
        let manifest = read_v41_exl3_manifest(Path::new(&path)).unwrap();
        assert_eq!(manifest.projections.len(), 47_232);
        assert_eq!(manifest.decoder_tiers(), &[3, 4]);
        // For the published quant the common family equals every old local
        // family, so descriptor layouts and allocation sizes remain unchanged.
        let mut families = BTreeMap::<String, BTreeSet<usize>>::new();
        for p in manifest.projections.values() {
            let prefix = p.name.split(".ffn.experts.").next().unwrap();
            families.entry(prefix.into()).or_default().insert(p.bits);
        }
        assert_eq!(families.len(), 43);
        assert!(families
            .values()
            .all(|bits| bits.iter().copied().collect::<Vec<_>>() == [3, 4]));
        let mut counts = BTreeMap::new();
        for p in manifest.projections.values() {
            *counts.entry(p.bits).or_insert(0usize) += 1;
        }
        assert_eq!(counts, BTreeMap::from([(3, 35_424), (4, 11_808)]));
        let index = read_json(
            &Path::new(&path).join("model.safetensors.index.json"),
            MAX_MANIFEST_BYTES,
        )
        .unwrap();
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for shard in shards {
            for tensor in crate::read_safetensors_metadata(&Path::new(&path).join(shard)).unwrap() {
                if !tensor.name.contains(".ffn.experts.") {
                    continue;
                }
                let (prefix, _) = tensor.name.rsplit_once('.').unwrap();
                manifest
                    .projections
                    .get(prefix)
                    .expect("unexpected routed tensor")
                    .validate_tensor(&tensor)
                    .unwrap();
                assert!(seen.insert(tensor.name), "duplicate routed tensor");
            }
        }
        assert_eq!(seen.len(), 4 * manifest.projections.len());
        println!(
            "validated {} projections; tiers {:?}; PLE override={}",
            manifest.projections.len(),
            counts,
            manifest.ple_quantization.is_some()
        );
    }

    #[test]
    fn aligned_tp_slices_cover_the_complete_rotation_axis() {
        for kind in [
            V41Exl3ProjectionKind::Gate,
            V41Exl3ProjectionKind::Up,
            V41Exl3ProjectionKind::Down,
        ] {
            let (input, output) = if kind == V41Exl3ProjectionKind::Down {
                (2304, 5120)
            } else {
                (5120, 2304)
            };
            let p = V41Exl3Projection {
                name: "test".into(),
                kind,
                bits: 3,
                input_features: input,
                output_features: output,
            };
            for (world, widths) in [
                (1, vec![2304]),
                (2, vec![1152, 1152]),
                (4, vec![640, 640, 512, 512]),
            ] {
                let mut cursor = 0;
                for (rank, width) in widths.into_iter().enumerate() {
                    let range = p.intermediate_partition(world, rank).unwrap();
                    assert_eq!(range, cursor..cursor + width);
                    assert_eq!(range.start % 128, 0);
                    cursor = range.end;
                }
                assert_eq!(cursor, 2304);
                assert!(p.intermediate_partition(world, world).is_err());
            }
        }
    }

    #[test]
    fn validates_each_integer_tier_and_rejects_shape_or_rotation_drift() {
        let name = "layers.0.ffn.experts.0.w1";
        for bits in 2..=5 {
            let mut tensors = serde_json::Map::new();
            for (suffix, dtype, shape) in [
                ("trellis", "int16", vec![320, 144, 16 * bits]),
                ("suh", "float16", vec![5120]),
                ("svh", "float16", vec![2304]),
                ("mcg", "int32", vec![]),
            ] {
                tensors.insert(
                    format!("{name}.{suffix}"),
                    serde_json::json!({"torch_dtype": dtype, "shape": shape}),
                );
            }
            let value = serde_json::json!({"quant_format": "exl3", "bits_per_weight": bits, "stored_tensors": tensors});
            let parse =
                |v: &Value| parse_projection(name, V41Exl3ProjectionKind::Gate, 5120, 2304, v);
            assert_eq!(
                parse(&value).unwrap().trellis_bytes(),
                5120 * 2304 * bits / 8
            );
            for invalid in [
                serde_json::json!(1),
                serde_json::json!(6),
                serde_json::json!(3.25),
            ] {
                let mut bad = value.clone();
                bad["bits_per_weight"] = invalid;
                assert!(parse(&bad).is_err());
            }
            let mut bad = value.clone();
            bad["stored_tensors"][format!("{name}.suh")]["shape"] = serde_json::json!([2304]);
            assert!(parse(&bad).is_err());
            let mut bad = value.clone();
            bad["stored_tensors"][format!("{name}.trellis")]["shape"] =
                serde_json::json!([144, 320, 16 * bits]);
            assert!(parse(&bad).is_err());
        }
    }
}
