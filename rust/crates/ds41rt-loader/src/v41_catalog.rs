//! Official checkpoint storage and fixed RTX/Spark-TP4 placement contracts.
use crate::{
    read_official_v41_config, read_safetensors_metadata, OfficialV41Config,
    SafetensorsTensorMetadata,
};
use anyhow::{ensure, Context, Result};
use ds41rt_core::DType;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V41TensorPlacement {
    CoordinatorRtx,
    HostMappedEngram,
    /// Every Spark holds one intermediate-dimension quarter of every backbone expert.
    BackboneExpertTp4 {
        layer: usize,
        expert: usize,
        axis: usize,
    },
    /// Packed EXL3 projection; rotations and unequal aligned TP slices require
    /// the projection descriptor rather than the native FP4 slicing rule.
    BackboneExl3,
}

#[derive(Debug, Clone)]
pub struct V41Tensor {
    pub shard: String,
    pub metadata: SafetensorsTensorMetadata,
    pub placement: V41TensorPlacement,
}

#[derive(Debug)]
pub struct OfficialV41Catalog {
    config: OfficialV41Config,
    snapshot: PathBuf,
    tensors: Vec<V41Tensor>,
    exl3: Option<crate::V41Exl3Manifest>,
}

/// Bounded range reads for a validated, unsharded coordinator tensor.
pub struct V41CoordinatorTensorReader {
    file: File,
    offset: u64,
    bytes: u64,
}
impl V41CoordinatorTensorReader {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn read_into(&self, offset: u64, destination: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let end = offset
            .checked_add(u64::try_from(destination.len())?)
            .context("coordinator tensor read extent overflow")?;
        ensure!(
            end <= self.bytes,
            "coordinator tensor range exceeds payload"
        );
        self.file.read_exact_at(
            destination,
            self.offset
                .checked_add(offset)
                .context("coordinator tensor file offset overflow")?,
        )?;
        Ok(())
    }
}

impl OfficialV41Catalog {
    pub fn exl3(&self) -> Option<&crate::V41Exl3Manifest> {
        self.exl3.as_ref()
    }
    /// True when the draft (MTP) routed experts load as native FP4 weights:
    /// either the checkpoint is not EXL3 at all, or it is a raw EXL3
    /// publication that kept its draft experts at source precision.
    pub fn native_dspark_experts(&self) -> bool {
        self.exl3
            .as_ref()
            .is_none_or(|manifest| manifest.mtp_experts_are_source())
    }
    pub fn coordinator_tensor_reader(&self, name: &str) -> Result<V41CoordinatorTensorReader> {
        let tensor = self.tensor(name)?;
        ensure!(
            matches!(tensor.placement, V41TensorPlacement::CoordinatorRtx),
            "only unsharded RTX tensors support coordinator range reads"
        );
        Ok(V41CoordinatorTensorReader {
            file: File::open(self.snapshot.join(&tensor.shard))?,
            offset: tensor.metadata.byte_offset,
            bytes: tensor.metadata.byte_length,
        })
    }
    pub fn config(&self) -> &OfficialV41Config {
        &self.config
    }
    pub fn snapshot(&self) -> &Path {
        &self.snapshot
    }
    pub fn tensors(&self) -> &[V41Tensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&V41Tensor> {
        let index = self
            .tensors
            .binary_search_by(|tensor| tensor.metadata.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("unknown official tensor {name}"))?;
        Ok(&self.tensors[index])
    }

    pub fn device_tensor_bytes(&self, name: &str, spark_rank: Option<usize>) -> Result<u64> {
        let tensor = self.tensor(name)?;
        match tensor.placement {
            V41TensorPlacement::BackboneExl3 => {
                let rank = spark_rank.context("EXL3 backbone requires a Spark rank")?;
                self.exl3_tensor_bytes(name, 4, rank)
            }
            V41TensorPlacement::HostMappedEngram => {
                anyhow::bail!("engram tables must be mapped, not eagerly loaded")
            }
            V41TensorPlacement::CoordinatorRtx => {
                ensure!(
                    spark_rank.is_none(),
                    "RTX tensor {name} cannot be loaded as a Spark shard"
                );
                Ok(tensor.metadata.byte_length)
            }
            V41TensorPlacement::BackboneExpertTp4 { .. } => {
                ensure!(
                    spark_rank.is_some_and(|rank| rank < 4),
                    "backbone expert {name} requires a Spark TP rank in 0..4"
                );
                Ok(tensor.metadata.byte_length / 4)
            }
        }
    }

    /// Read the selected physical tensor/shard into caller-owned staging memory.
    /// Column-sharded W2 uses bounded scratch to coalesce file reads, then packs rows.
    pub fn read_device_tensor_into(
        &self,
        name: &str,
        spark_rank: Option<usize>,
        dst: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        if matches!(self.tensor(name)?.placement, V41TensorPlacement::BackboneExl3) {
            return self.read_exl3_tensor_into(name, 4,
                spark_rank.context("EXL3 backbone requires a Spark rank")?, dst, scratch);
        }
        let bytes = usize::try_from(self.device_tensor_bytes(name, spark_rank)?)?;
        let axis = match self.tensor(name)?.placement {
            V41TensorPlacement::BackboneExpertTp4 { axis, .. } => Some(axis),
            _ => None,
        };
        self.read_tensor_partition(name, spark_rank, 4, axis, bytes, dst, scratch)
    }

    pub(crate) fn read_backbone_tp2_into(
        &self, name: &str, rank: usize, dst: &mut [u8], scratch: &mut [u8],
    ) -> Result<usize> {
        let tensor = self.tensor(name)?;
        ensure!(rank < 2 && matches!(tensor.placement, V41TensorPlacement::BackboneExpertTp4 { .. }),
            "TP2 read requires a backbone expert and rank in 0..2");
        let bytes = usize::try_from(tensor.metadata.byte_length / 2)?;
        let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = tensor.placement else { unreachable!() };
        self.read_tensor_partition(name, Some(rank), 2, Some(axis), bytes, dst, scratch)
    }

    /// Read a row or column half of an ordinary coordinator matrix. Shapes and
    /// dtype remain native; column reads use caller-owned full-row scratch.
    pub fn read_coordinator_tp2_into(&self, name: &str, axis: usize, rank: usize,
        dst: &mut [u8], scratch: &mut [u8]) -> Result<usize> {
        let tensor = self.tensor(name)?;
        ensure!(matches!(tensor.placement, V41TensorPlacement::CoordinatorRtx)
            && tensor.metadata.shape.len() == 2 && axis < 2 && rank < 2,
            "TP2 coordinator reads require a matrix, axis 0/1, and rank 0/1");
        ensure!(tensor.metadata.shape[axis] % 2 == 0 && tensor.metadata.byte_length % 2 == 0,
            "coordinator matrix cannot be divided in half");
        let bytes = usize::try_from(tensor.metadata.byte_length / 2)?;
        self.read_tensor_partition(name, Some(rank), 2, Some(axis), bytes, dst, scratch)
    }

    fn read_tensor_partition(
        &self, name: &str, spark_rank: Option<usize>, partitions: usize, axis: Option<usize>, bytes: usize,
        dst: &mut [u8], scratch: &mut [u8],
    ) -> Result<usize> {
        use std::os::unix::fs::FileExt;
        ensure!(
            dst.len() >= bytes,
            "tensor staging buffer for {name} needs {bytes} bytes"
        );
        let tensor = self.tensor(name)?;
        let metadata = &tensor.metadata;
        let file = File::open(self.snapshot.join(&tensor.shard))?;
        match axis {
            Some(0) => {
                let offset = metadata
                    .byte_offset
                    .checked_add(
                        (bytes as u64)
                            .checked_mul(spark_rank.unwrap() as u64)
                            .context("TP row offset overflow")?,
                    )
                    .context("TP file offset overflow")?;
                file.read_exact_at(&mut dst[..bytes], offset)?;
            }
            Some(1) => {
                let rows = metadata.shape[0];
                let row_bytes = usize::try_from(metadata.byte_length / rows as u64)?;
                ensure!(
                    scratch.len() >= row_bytes,
                    "W2 scratch needs at least {row_bytes} bytes"
                );
                let shard_bytes = row_bytes / partitions;
                let column = spark_rank.unwrap() * shard_bytes;
                let rows_per_read = scratch.len() / row_bytes;
                for start in (0..rows).step_by(rows_per_read) {
                    let count = rows_per_read.min(rows - start);
                    let offset = metadata
                        .byte_offset
                        .checked_add(
                            (start as u64)
                                .checked_mul(row_bytes as u64)
                                .context("TP column row offset overflow")?,
                        )
                        .context("TP column file offset overflow")?;
                    file.read_exact_at(&mut scratch[..count * row_bytes], offset)?;
                    for row in 0..count {
                        dst[(start + row) * shard_bytes..(start + row + 1) * shard_bytes].copy_from_slice(
                            &scratch[row * row_bytes + column..row * row_bytes + column + shard_bytes],
                        );
                    }
                }
            }
            None => {
                file.read_exact_at(&mut dst[..bytes], metadata.byte_offset)?
            }
            _ => anyhow::bail!("unsupported device tensor placement"),
        }
        Ok(bytes)
    }

    /// # Safety
    /// Checkpoint files must remain immutable for the lifetime of the returned maps.
    pub unsafe fn map_engram(&self, layer: usize) -> Result<crate::EngramTable> {
        ensure!(
            self.config.text().engram_layer_ids.contains(&layer),
            "no engram table at layer {layer}"
        );
        let map = |suffix: &str| -> Result<crate::MappedRows> {
            let tensor = self.tensor(&format!("layers.{layer}.engram.embed.{suffix}"))?;
            let metadata = &tensor.metadata;
            unsafe {
                crate::MappedRows::open(
                    &self.snapshot.join(&tensor.shard),
                    metadata.byte_offset,
                    metadata.shape[0] as u64,
                    metadata.shape[1],
                )
            }
        };
        if let Some(ple) = self.exl3.as_ref().and_then(|m| m.ple_quantization.as_ref()) {
            use std::os::unix::fs::FileExt;
            let prefix = format!("layers.{layer}.engram.embed");
            let tensor = self.tensor(&format!("{prefix}.weight_scale_2"))?;
            let mut bytes = [0; 4];
            File::open(self.snapshot.join(&tensor.shard))?.read_exact_at(&mut bytes, tensor.metadata.byte_offset)?;
            let global = f32::from_le_bytes(bytes);
            let declared = ple["tensors"][&prefix]["global_scale"].as_f64().context("missing NVFP4 global scale")? as f32;
            ensure!(global.to_bits() == declared.to_bits(), "NVFP4 global scale differs from metadata");
            crate::EngramTable::new_nvfp4(map("weight")?, map("weight_scale")?, global)
        } else {
            crate::EngramTable::new(map("weight")?, map("scale")?)
        }
    }

    /// Physical checkpoint bytes only; packing, caches and execution scratch are additional.
    pub fn storage_budget(&self) -> Result<V41StorageBudget> {
        let mut budget = V41StorageBudget::default();
        for tensor in &self.tensors {
            let bytes = tensor.metadata.byte_length;
            match tensor.placement {
                V41TensorPlacement::CoordinatorRtx => {
                    budget.coordinator_bytes = budget
                        .coordinator_bytes
                        .checked_add(bytes)
                        .context("RTX weight byte overflow")?;
                    if tensor.metadata.name.starts_with("mtp.") {
                        budget.dspark_bytes = budget
                            .dspark_bytes
                            .checked_add(bytes)
                            .context("dSpark weight byte overflow")?;
                    }
                }
                V41TensorPlacement::HostMappedEngram => {
                    budget.host_mapped_bytes = budget
                        .host_mapped_bytes
                        .checked_add(bytes)
                        .context("engram byte overflow")?;
                }
                V41TensorPlacement::BackboneExpertTp4 { .. } => {
                    ensure!(
                        bytes % 4 == 0,
                        "TP4 tensor byte count is not divisible by four"
                    );
                    budget.per_spark_bytes = budget
                        .per_spark_bytes
                        .checked_add(bytes / 4)
                        .context("Spark weight byte overflow")?;
                    for rank_bytes in &mut budget.spark_rank_bytes {
                        *rank_bytes = rank_bytes.checked_add(bytes / 4).context("Spark rank budget overflow")?;
                    }
                }
                V41TensorPlacement::BackboneExl3 => {
                    for rank in 0..4 {
                        budget.spark_rank_bytes[rank] = budget.spark_rank_bytes[rank]
                            .checked_add(self.exl3_tensor_bytes(&tensor.metadata.name, 4, rank)?)
                            .context("EXL3 Spark rank budget overflow")?;
                    }
                }
            }
        }
        budget.per_spark_bytes = *budget.spark_rank_bytes.iter().max().unwrap();
        Ok(budget)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct V41StorageBudget {
    pub coordinator_bytes: u64,
    /// Included in coordinator_bytes; shared embedding/output tensors are counted only once.
    pub dspark_bytes: u64,
    pub per_spark_bytes: u64,
    /// Exact per-rank bytes; per_spark_bytes is their maximum for legacy callers.
    pub spark_rank_bytes: [u64; 4],
    pub host_mapped_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct Template {
    pattern: String,
    axes: Vec<Vec<usize>>,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ExpectedTensor {
    dtype: DType,
    shape: Vec<usize>,
    bytes: u64,
}

fn expected_tensors() -> Result<BTreeMap<String, ExpectedTensor>> {
    let templates: Vec<Template> = serde_json::from_str(include_str!("official-v41-tensors.json"))?;
    let mut expected = BTreeMap::new();
    for template in templates {
        let dtype = DType::from_safetensors(&template.dtype);
        let width = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            DType::F8E4M3 | DType::F8E8M0 | DType::I8 => 1,
            _ => anyhow::bail!("unsupported official tensor dtype {}", template.dtype),
        };
        let bytes = template
            .shape
            .iter()
            .try_fold(width, |n: u64, &dim| n.checked_mul(dim as u64))
            .context("tensor size overflow")?;
        let mut names = vec![template.pattern];
        for axis in template.axes {
            names = names
                .into_iter()
                .flat_map(|name| {
                    axis.iter()
                        .map(move |i| name.replacen("{}", &i.to_string(), 1))
                })
                .collect();
        }
        for name in names {
            ensure!(!name.contains("{}"), "incomplete tensor template {name}");
            ensure!(
                expected
                    .insert(
                        name.clone(),
                        ExpectedTensor {
                            dtype: dtype.clone(),
                            shape: template.shape.clone(),
                            bytes
                        }
                    )
                    .is_none(),
                "duplicate tensor template {name}"
            );
        }
    }
    ensure!(
        expected.len() == 96085,
        "incomplete official tensor contract"
    );
    Ok(expected)
}

fn placement(name: &str) -> V41TensorPlacement {
    // Keep the complete draft model local, including its distinct routed
    // experts. Decide this before applying any backbone offload rules.
    if name.starts_with("mtp.") {
        return V41TensorPlacement::CoordinatorRtx;
    }
    if name.contains(".engram.embed.") {
        return V41TensorPlacement::HostMappedEngram;
    }
    let parts: Vec<_> = name.split('.').collect();
    if parts.len() == 7 && parts[0] == "layers" && parts[2] == "ffn" && parts[3] == "experts" {
        V41TensorPlacement::BackboneExpertTp4 {
            layer: parts[1].parse().expect("validated official layer"),
            expert: parts[4].parse().expect("validated official expert"),
            axis: usize::from(parts[5] == "w2"),
        }
    } else {
        V41TensorPlacement::CoordinatorRtx
    }
}

/// Header-only inspection: never reads or eagerly allocates checkpoint tensor payloads.
pub fn read_official_v41_catalog(model_id: &str, snapshot: &Path) -> Result<OfficialV41Catalog> {
    let raw_config: serde_json::Value = crate::v41_exl3::read_json(&snapshot.join("config.json"), 1024 * 1024)?;
    let exl3 = if raw_config["quantization_config"]["quant_method"] == "exl3" {
        Some(crate::read_v41_exl3_manifest(snapshot)?)
    } else { None };
    let config = match &exl3 {
        Some(manifest) => manifest.config.clone(),
        None => read_official_v41_config(model_id, snapshot)?,
    };
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let mut bytes = Vec::new();
    File::open(snapshot.join("model.safetensors.index.json"))?
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 64 * 1024 * 1024,
        "checkpoint index exceeds sixty-four MiB"
    );
    let index: Index = serde_json::from_slice(&bytes)?;
    let mut expected = expected_tensors()?;
    if let Some(manifest) = &exl3 {
        // Raw publications keep their MTP draft experts at the native FP4
        // source precision, so only backbone expert tensors leave the
        // official contract; staged snapshots quantize drafts too.
        let keep_native_mtp = manifest.mtp_experts_are_source();
        expected.retain(|name, _| {
            !name.contains(".ffn.experts.") || (keep_native_mtp && name.starts_with("mtp."))
        });
        for projection in manifest.projections.values() {
            for (suffix, dtype, shape, bytes) in [
                ("trellis", DType::I16, projection.trellis_shape().to_vec(), projection.trellis_bytes()),
                ("suh", DType::F16, vec![projection.input_features], projection.input_features * 2),
                ("svh", DType::F16, vec![projection.output_features], projection.output_features * 2),
                ("mcg", DType::I32, vec![], 4),
            ] {
                expected.insert(format!("{}.{suffix}", projection.name),
                    ExpectedTensor { dtype, shape, bytes: bytes as u64 });
            }
        }
        if let Some(ple) = &manifest.ple_quantization {
            apply_nvfp4_ple_contract(&mut expected, ple, config.text().engram_layer_ids.as_slice())?;
        }
    }
    ensure!(
        index.weight_map.len() == expected.len(),
        "official tensor inventory count mismatch: expected {}, got {}",
        expected.len(),
        index.weight_map.len()
    );
    let shard_count = if exl3.is_some() {
        index.weight_map.values().collect::<BTreeSet<_>>().len()
    } else { 48 };
    let allowed_shards: BTreeSet<_> = (1..=shard_count)
        .map(|i| format!("model-{i:05}-of-{shard_count:05}.safetensors"))
        .collect();
    let mut shards: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, shard) in &index.weight_map {
        ensure!(
            expected.contains_key(name),
            "unexpected checkpoint tensor {name}"
        );
        ensure!(
            allowed_shards.contains(shard),
            "unsupported checkpoint shard {shard}"
        );
        shards
            .entry(shard.clone())
            .or_default()
            .insert(name.clone());
    }
    ensure!(
        shards.len() == shard_count,
        "checkpoint requires all {shard_count} shards"
    );
    let mut tensors = Vec::with_capacity(expected.len());
    for (shard, names) in shards {
        let path = snapshot.join(&shard);
        let metadata =
            read_safetensors_metadata(&path).with_context(|| format!("reading {shard}"))?;
        ensure!(
            metadata.len() == names.len(),
            "index/header tensor count mismatch in {shard}"
        );
        let mut intervals = Vec::with_capacity(metadata.len());
        for tensor in metadata {
            ensure!(
                names.contains(&tensor.name),
                "tensor {} appears in unexpected shard {shard}",
                tensor.name
            );
            let spec = &expected[&tensor.name];
            validate_tensor(&tensor, spec)?;
            let target = if exl3.is_some() && tensor.name.starts_with("layers.")
                && tensor.name.contains(".ffn.experts.") {
                V41TensorPlacement::BackboneExl3
            } else { placement(&tensor.name) };
            if let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = target {
                ensure!(
                    tensor.shape[axis] % 4 == 0,
                    "TP4 axis cannot be divided for {}",
                    tensor.name
                );
            }
            intervals.push((
                tensor.byte_offset,
                tensor
                    .byte_offset
                    .checked_add(tensor.byte_length)
                    .context("tensor end overflow")?,
            ));
            tensors.push(V41Tensor {
                shard: shard.clone(),
                metadata: tensor,
                placement: target,
            });
        }
        intervals.sort_unstable();
        let mut header_len = [0u8; 8];
        File::open(&path)?.read_exact(&mut header_len)?;
        let mut cursor = u64::from_le_bytes(header_len)
            .checked_add(8)
            .context("header offset overflow")?;
        for (start, end) in intervals {
            ensure!(
                start == cursor,
                "overlap or gap in {shard} at byte {cursor}"
            );
            cursor = end;
        }
        ensure!(
            cursor == path.metadata()?.len(),
            "unindexed trailing bytes in {shard}"
        );
    }
    tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
    Ok(OfficialV41Catalog {
        config,
        snapshot: snapshot.to_path_buf(),
        tensors,
        exl3,
    })
}

fn validate_tensor(tensor: &SafetensorsTensorMetadata, spec: &ExpectedTensor) -> Result<()> {
    ensure!(
        tensor.dtype == spec.dtype,
        "{} dtype mismatch: expected {:?}, got {:?}",
        tensor.name,
        spec.dtype,
        tensor.dtype
    );
    ensure!(
        tensor.shape == spec.shape,
        "{} shape mismatch: expected {:?}, got {:?}",
        tensor.name,
        spec.shape,
        tensor.shape
    );
    ensure!(
        tensor.byte_length == spec.bytes,
        "{} storage length mismatch: expected {}, got {}",
        tensor.name,
        spec.bytes,
        tensor.byte_length
    );
    Ok(())
}

fn apply_nvfp4_ple_contract(expected: &mut BTreeMap<String, ExpectedTensor>,
    ple: &serde_json::Value, layers: &[usize]) -> Result<()> {
    ensure!(ple["schema"] == "ds41rt.nvfp4-ple.v1" && ple["format"] == "nvfp4"
        && ple["block_size"] == 16 && ple["packing"] == "even-element-low-nibble"
        && ple["scale_layout"] == "row-major"
        && ple["reconstruction"] == "E2M1(weight) * FP8_E4M3(weight_scale) * FP32(weight_scale_2)",
        "unsupported NVFP4 PLE storage contract");
    let tables = ple["tensors"].as_object().context("missing NVFP4 PLE tables")?;
    ensure!(tables.len() == layers.len(), "NVFP4 PLE must cover exactly the original tables");
    for &layer in layers {
        let prefix = format!("layers.{layer}.engram.embed");
        let table = tables.get(&prefix).with_context(|| format!("missing {prefix} PLE metadata"))?;
        let original = expected.get(&format!("{prefix}.weight")).context("missing original PLE weight")?;
        let shape = original.shape.clone();
        ensure!(shape.len() == 2 && shape[1] % 16 == 0
            && table["logical_shape"] == serde_json::to_value(&shape)?
            && table["weight_dtype"] == "uint8" && table["weight_scale_dtype"] == "float8_e4m3fn"
            && table["weight_scale_2_dtype"] == "float32"
            && table["global_scale"].as_f64().is_some_and(|s| s.is_finite() && s > 0.0),
            "invalid NVFP4 PLE geometry or scale for {prefix}");
        let rows = shape[0]; let columns = shape[1];
        ensure!(expected.remove(&format!("{prefix}.scale")).is_some(), "missing original PLE scales");
        for (suffix, dtype, shape, bytes) in [
            ("weight", DType::U8, vec![rows, columns / 2], rows as u64 * columns as u64 / 2),
            ("weight_scale", DType::F8E4M3, vec![rows, columns / 16], rows as u64 * columns as u64 / 16),
            ("weight_scale_2", DType::F32, vec![], 4),
        ] {
            expected.insert(format!("{prefix}.{suffix}"), ExpectedTensor { dtype, shape, bytes });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_contract_keeps_native_types_and_local_draft_experts() {
        let tensors = expected_tensors().unwrap();
        assert_eq!(tensors["head.weight"].dtype, DType::Bf16);
        assert_eq!(tensors["layers.0.attn.wo_a.weight"].dtype, DType::F8E4M3);
        assert_eq!(tensors["layers.0.attn.wo_a.scale"].shape, [256, 128]);
        assert_eq!(
            tensors["mtp.0.ffn.experts.127.w1.weight"].shape,
            [2304, 2560]
        );
        assert!(!tensors.contains_key("mtp.0.ffn.experts.128.w1.weight"));
        assert_eq!(
            placement("mtp.0.ffn.experts.127.w1.weight"),
            V41TensorPlacement::CoordinatorRtx
        );
        assert_eq!(
            placement("layers.39.ffn.experts.383.w2.scale"),
            V41TensorPlacement::BackboneExpertTp4 {
                layer: 39,
                expert: 383,
                axis: 1
            }
        );
        assert_eq!(
            placement("layers.14.engram.embed.weight"),
            V41TensorPlacement::HostMappedEngram
        );
    }
    #[test]
    fn rejects_same_byte_count_with_wrong_representation() {
        let spec = ExpectedTensor {
            dtype: DType::I8,
            shape: vec![2304, 2560],
            bytes: 5898240,
        };
        let mut tensor = SafetensorsTensorMetadata {
            name: "expert".into(),
            dtype: DType::I8,
            shape: vec![2304, 2560],
            byte_offset: 8,
            byte_length: 5898240,
        };
        validate_tensor(&tensor, &spec).unwrap();
        tensor.shape = vec![2560, 2304];
        assert!(validate_tensor(&tensor, &spec).is_err());
        tensor.shape = spec.shape.clone();
        tensor.dtype = DType::F8E4M3;
        assert!(validate_tensor(&tensor, &spec).is_err());
        tensor.dtype = DType::I8;
        tensor.byte_length -= 1;
        assert!(validate_tensor(&tensor, &spec).is_err());
    }
    #[test]
    fn tp4_staging_reads_columns_and_rows_beyond_two_gib() {
        use std::os::unix::fs::FileExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture");
        let offset = (1u64 << 31) + 64;
        let file = File::create(&path).unwrap();
        file.set_len(offset + 32).unwrap();
        file.write_all_at(&(0u8..32).collect::<Vec<_>>(), offset)
            .unwrap();
        let name = "layers.0.ffn.experts.0.w2.weight";
        let mut catalog = OfficialV41Catalog {
            config: OfficialV41Config::from_json(
                crate::OFFICIAL_V41_MODEL_ID,
                include_bytes!("official-v41-config.json"),
            )
            .unwrap(),
            exl3: None,
            snapshot: dir.path().into(),
            tensors: vec![V41Tensor {
                shard: "fixture".into(),
                metadata: SafetensorsTensorMetadata {
                    name: name.into(),
                    dtype: DType::I8,
                    shape: vec![4, 8],
                    byte_offset: offset,
                    byte_length: 32,
                },
                placement: V41TensorPlacement::BackboneExpertTp4 {
                    layer: 0,
                    expert: 0,
                    axis: 1,
                },
            }],
        };
        for rank in 0..4 {
            let mut out = [255u8; 8];
            catalog
                .read_device_tensor_into(name, Some(rank), &mut out, &mut [0; 16])
                .unwrap();
            let expected: Vec<_> = (0..4)
                .flat_map(|r| [r * 8 + rank * 2, r * 8 + rank * 2 + 1])
                .map(|v| v as u8)
                .collect();
            assert_eq!(out.as_slice(), expected);
        }
        assert!(catalog.device_tensor_bytes(name, None).is_err());
        assert!(catalog.device_tensor_bytes(name, Some(4)).is_err());
        catalog.tensors[0].placement = V41TensorPlacement::BackboneExpertTp4 {
            layer: 0,
            expert: 0,
            axis: 0,
        };
        let mut out = [0; 8];
        catalog
            .read_device_tensor_into(name, Some(3), &mut out, &mut [])
            .unwrap();
        assert_eq!(out, [24, 25, 26, 27, 28, 29, 30, 31]);
        catalog.tensors[0].placement = V41TensorPlacement::CoordinatorRtx;
        for axis in 0..2 {
            for rank in 0..2 {
                let mut out = [255u8; 20];
                let bytes = catalog.read_coordinator_tp2_into(name, axis, rank, &mut out, &mut [0; 17]).unwrap();
                assert_eq!(bytes, 16);
                let expected: Vec<u8> = if axis == 0 {
                    (rank*16..(rank+1)*16).map(|v| v as u8).collect()
                } else {
                    (0..4).flat_map(|r| (r*8+rank*4..r*8+(rank+1)*4).map(|v| v as u8)).collect()
                };
                assert_eq!(&out[..16], expected);
                assert_eq!(&out[16..], &[255; 4]);
            }
        }
        assert!(catalog.read_coordinator_tp2_into(name, 1, 0, &mut [0; 16], &mut [0; 7]).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 2, &mut [0; 16], &mut []).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 2, 0, &mut [0; 16], &mut []).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 0, &mut [0; 15], &mut []).is_err());
        catalog.tensors[0].placement = V41TensorPlacement::HostMappedEngram;
        assert!(catalog.device_tensor_bytes(name, None).is_err());
        assert!(catalog.read_coordinator_tp2_into(name, 0, 0, &mut [0; 16], &mut []).is_err());
    }

    #[test]
    #[ignore = "requires DS41RT_EXL3_RAW_SNAPSHOT pointing to a raw local publication"]
    fn raw_publication_catalog_keeps_native_draft_experts() {
        let path = std::env::var_os("DS41RT_EXL3_RAW_SNAPSHOT").expect("DS41RT_EXL3_RAW_SNAPSHOT");
        let catalog =
            read_official_v41_catalog("diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000", Path::new(&path))
                .unwrap();
        assert!(catalog.exl3().is_some());
        assert!(catalog.native_dspark_experts());
        let mut backbone_exl3 = 0usize;
        let mut native_mtp = 0usize;
        for tensor in &catalog.tensors {
            match tensor.placement {
                V41TensorPlacement::BackboneExl3 => backbone_exl3 += 1,
                V41TensorPlacement::CoordinatorRtx
                    if tensor.metadata.name.starts_with("mtp.") && tensor.metadata.name.contains(".ffn.experts.") =>
                {
                    native_mtp += 1
                }
                _ => {}
            }
        }
        assert_eq!(backbone_exl3, 4 * 46_080);
        assert_eq!(native_mtp, 6 * 3 * 128);
        println!(
            "raw catalog: {} tensors, {backbone_exl3} EXL3 backbone, {native_mtp} native draft",
            catalog.tensors.len()
        );
    }
}

#[cfg(test)]
mod expert_staging_tests {
    use super::*;
    use crate::V41ExpertSelection;
    use std::os::unix::fs::FileExt;

    #[test]
    fn native_expert_staging_covers_all_tp_ranks_and_full_experts() {
        let dir = tempfile::tempdir().unwrap();
        let file = File::create(dir.path().join("fixture")).unwrap();
        let mut offset = (1u64 << 31) + 128;
        let specs = expected_tensors().unwrap();
        let mut tensors = Vec::new();
        let suffixes = [
            "w1.weight",
            "w3.weight",
            "w2.weight",
            "w1.scale",
            "w3.scale",
            "w2.scale",
        ];
        let mut payloads = Vec::new();
        for (slot, suffix) in suffixes.iter().enumerate() {
            let name = format!("layers.39.ffn.experts.383.{suffix}");
            let spec = &specs[&name];
            let cols = spec.shape[1];
            let payload: Vec<u8> = (0..spec.bytes as usize)
                .map(|index| ((index / cols * 7 + index % cols * 13 + slot * 19) & 255) as u8)
                .collect();
            file.write_all_at(&payload, offset).unwrap();
            for prefix in ["layers.39.ffn.experts.383", "mtp.2.ffn.experts.127"] {
                let name = format!("{prefix}.{suffix}");
                tensors.push(V41Tensor {
                    shard: "fixture".into(),
                    placement: placement(&name),
                    metadata: SafetensorsTensorMetadata {
                        name,
                        dtype: spec.dtype.clone(),
                        shape: spec.shape.clone(),
                        byte_offset: offset,
                        byte_length: spec.bytes,
                    },
                });
            }
            payloads.push(payload);
            offset += spec.bytes;
        }
        tensors.sort_by(|a, b| a.metadata.name.cmp(&b.metadata.name));
        let catalog = OfficialV41Catalog {
            config: OfficialV41Config::from_json(
                crate::OFFICIAL_V41_MODEL_ID,
                include_bytes!("official-v41-config.json"),
            )
            .unwrap(),
            exl3: None,
            snapshot: dir.path().into(),
            tensors,
        };
        let select = |rank| V41ExpertSelection::Backbone {
            layer: 39,
            expert: 383,
            rank,
        };
        for rank in 0..4 {
            let plan = catalog.expert_staging(select(rank)).unwrap();
            assert_eq!(plan.staging_bytes(), 4_700_160);
            assert_eq!(plan.intermediate_size(), 576);
            assert_eq!(plan.minimum_read_scratch_bytes(), 1152);
            // An odd number of physical rows exercises the final partial read batch.
            let mut scratch = vec![0; 1152 * 7 + 3];
            let mut staging = vec![205; plan.staging_bytes() + 32];
            assert!(plan
                .read_into(&mut staging[..plan.staging_bytes() - 1], &mut scratch)
                .is_err());
            assert!(plan.read_into(&mut staging, &mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&byte| byte == 205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging, &mut scratch).unwrap();
            for (slot, range) in plan.tensor_ranges().iter().enumerate() {
                assert_eq!(range.start % 16, 0);
                let source = &payloads[slot];
                let expected = if slot == 2 || slot == 5 {
                    let row = source.len() / 5120;
                    source
                        .chunks_exact(row)
                        .flat_map(|r| r[rank * row / 4..(rank + 1) * row / 4].iter().copied())
                        .collect::<Vec<_>>()
                } else {
                    source[rank * source.len() / 4..(rank + 1) * source.len() / 4].to_vec()
                };
                assert_eq!(&staging[range.clone()], expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..]
                .iter()
                .all(|&byte| byte == 205));
        }
        // Two RTX halves reconstruct the official row and column partitions.
        for rank in 0..2 {
            let plan = catalog.expert_staging(V41ExpertSelection::BackboneTp2 {
                layer: 39, expert: 383, rank,
            }).unwrap();
            assert_eq!(plan.intermediate_size(), 1152);
            assert_eq!(plan.staging_bytes(), 9_400_320);
            assert_eq!(plan.minimum_read_scratch_bytes(), 1152);
            let mut staging = vec![205; plan.staging_bytes() + 32];
            let mut scratch = vec![0; 1152 * 7 + 3];
            assert!(plan.read_into(&mut staging[..plan.staging_bytes()-1], &mut scratch).is_err());
            assert!(plan.read_into(&mut staging, &mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&v| v == 205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging, &mut scratch).unwrap();
            for (slot, range) in plan.tensor_ranges().iter().enumerate() {
                let source = &payloads[slot];
                let expected = if slot == 2 || slot == 5 {
                    let row = source.len() / 5120;
                    source.chunks_exact(row).flat_map(|r|
                        r[rank*row/2..(rank+1)*row/2].iter().copied()).collect::<Vec<_>>()
                } else {
                    source[rank*source.len()/2..(rank+1)*source.len()/2].to_vec()
                };
                assert_eq!(&staging[range.clone()], expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..].iter().all(|&v| v == 205));
        }
        // Full backbone reads must preserve every official byte, including W2
        // columns that the TP4 path normally slices into separate ranks.
        let full = catalog.expert_staging(V41ExpertSelection::BackboneFull {
            layer: 39, expert: 383,
        }).unwrap();
        assert_eq!(full.staging_bytes(), 18_800_640);
        assert_eq!(full.intermediate_size(), 2304);
        assert_eq!(full.minimum_read_scratch_bytes(), 0);
        let mut full_staging = vec![205; full.staging_bytes() + 32];
        assert!(full.read_into(&mut full_staging[..full.staging_bytes() - 1], &mut []).is_err());
        assert!(full_staging.iter().all(|&byte| byte == 205));
        full.prefetch().unwrap();
        full.read_into(&mut full_staging, &mut []).unwrap();
        for (range, expected) in full.tensor_ranges().iter().zip(&payloads) {
            assert_eq!(&full_staging[range.clone()], expected.as_slice());
        }
        assert!(full_staging[full.staging_bytes()..].iter().all(|&byte| byte == 205));
        let plan = catalog
            .expert_staging(V41ExpertSelection::Dspark {
                stage: 2,
                expert: 127,
            })
            .unwrap();
        assert_eq!(plan.staging_bytes(), 18_800_640);
        assert_eq!(plan.minimum_read_scratch_bytes(), 0);
        assert_eq!(plan.intermediate_size(), 2304);
        let mut staging = vec![0; plan.staging_bytes()];
        plan.prefetch().unwrap();
        plan.read_into(&mut staging, &mut []).unwrap();
        for (range, expected) in plan.tensor_ranges().iter().zip(&payloads) {
            assert_eq!(&staging[range.clone()], expected.as_slice());
        }
        // Draft halves use the same packed row/column split, while preserving
        // all 128 expert IDs independently on each RTX rank.
        for rank in 0..2 {
            let plan=catalog.expert_staging(V41ExpertSelection::DsparkTp2 {stage:2,expert:127,rank}).unwrap();
            assert_eq!(plan.intermediate_size(),1152);
            assert_eq!(plan.staging_bytes(),9_400_320);
            assert_eq!(plan.minimum_read_scratch_bytes(),1152);
            let mut staging=vec![205;plan.staging_bytes()+32];
            let mut scratch=vec![0;1152*7+3];
            assert!(plan.read_into(&mut staging[..plan.staging_bytes()-1],&mut scratch).is_err());
            assert!(plan.read_into(&mut staging,&mut scratch[..1151]).is_err());
            assert!(staging.iter().all(|&v|v==205));
            plan.prefetch().unwrap();
            plan.read_into(&mut staging,&mut scratch).unwrap();
            for (slot,range) in plan.tensor_ranges().iter().enumerate() {
                let source=&payloads[slot];
                let expected=if slot==2 || slot==5 {
                    let row=source.len()/5120;
                    source.chunks_exact(row).flat_map(|r|
                        r[rank*row/2..(rank+1)*row/2].iter().copied()).collect::<Vec<_>>()
                } else {source[rank*source.len()/2..(rank+1)*source.len()/2].to_vec()};
                assert_eq!(&staging[range.clone()],expected.as_slice());
            }
            assert!(staging[plan.staging_bytes()..].iter().all(|&v|v==205));
        }
        for selection in [
            select(4),
            V41ExpertSelection::DsparkTp2 {stage:3,expert:0,rank:0},
            V41ExpertSelection::DsparkTp2 {stage:0,expert:128,rank:0},
            V41ExpertSelection::DsparkTp2 {stage:0,expert:0,rank:2},
            V41ExpertSelection::BackboneTp2 { layer: 0, expert: 0, rank: 2 },
            V41ExpertSelection::BackboneTp2 { layer: 40, expert: 0, rank: 0 },
            V41ExpertSelection::BackboneTp2 { layer: 0, expert: 384, rank: 0 },
            V41ExpertSelection::BackboneFull { layer: 40, expert: 0 },
            V41ExpertSelection::BackboneFull { layer: 0, expert: 384 },
            V41ExpertSelection::Backbone {
                layer: 40,
                expert: 0,
                rank: 0,
            },
            V41ExpertSelection::Backbone {
                layer: 0,
                expert: 384,
                rank: 0,
            },
            V41ExpertSelection::Dspark {
                stage: 3,
                expert: 0,
            },
            V41ExpertSelection::Dspark {
                stage: 0,
                expert: 128,
            },
        ] {
            assert!(catalog.expert_staging(selection).is_err());
        }
        // Catalog metadata alone is insufficient after a file changes: propagate I/O failure.
        file.set_len(0).unwrap();
        assert!(full.read_into(&mut full_staging, &mut []).is_err());
        assert!(plan.read_into(&mut staging, &mut []).is_err());
    }
}
