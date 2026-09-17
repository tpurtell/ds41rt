//! Bounded official expert reads in the native packer's source order.
use crate::OfficialV41Catalog;
use anyhow::{ensure, Context, Result};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41ExpertSelection {
    Backbone {
        layer: usize,
        expert: usize,
        rank: usize,
    },
    /// Complete routed expert on one device; no TP slicing during the read.
    BackboneFull {
        layer: usize,
        expert: usize,
    },
    /// Half-width encoder expert on one member of an RTX pair.
    BackboneTp2 { layer: usize, expert: usize, rank: usize },
    Dspark {
        stage: usize,
        expert: usize,
    },
    /// Half-width routed draft expert on one member of an RTX pair.
    DsparkTp2 { stage: usize, expert: usize, rank: usize },
}

/// Borrows the validated catalog; storage is supplied by the caller and reusable.
/// Source order is W1,W3,W2,S1,S3,S2, matching the native CUDA packer.
pub struct V41ExpertStaging<'a> {
    catalog: &'a OfficialV41Catalog,
    selection: V41ExpertSelection,
    names: [String; 6],
    ranges: [Range<usize>; 6],
    bytes: usize,
    scratch_bytes: usize,
    intermediate: usize,
    rank: Option<usize>,
}

impl OfficialV41Catalog {
    pub fn expert_staging(&self, selection: V41ExpertSelection) -> Result<V41ExpertStaging<'_>> {
        ensure!(self.exl3().is_none(), "EXL3 experts require compressed projection staging, not the native FP4 packer");
        let config = self.config().text();
        let (prefix, rank, intermediate) = match selection {
            V41ExpertSelection::BackboneTp2 { layer, expert, rank } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                ensure!(rank < 2, "backbone TP2 rank must be in 0..2");
                (format!("layers.{layer}.ffn.experts.{expert}"), Some(rank), config.moe_intermediate_size / 2)
            }
            V41ExpertSelection::Backbone {
                layer,
                expert,
                rank,
            } => {
                ensure!(
                    layer < config.num_hidden_layers,
                    "backbone layer out of range"
                );
                ensure!(
                    expert < config.n_routed_experts,
                    "backbone expert out of range"
                );
                ensure!(rank < 4, "backbone TP rank must be in 0..4");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    Some(rank),
                    config.moe_intermediate_size / 4,
                )
            }
            V41ExpertSelection::BackboneFull { layer, expert } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    None,
                    config.moe_intermediate_size,
                )
            }
            V41ExpertSelection::DsparkTp2 { stage, expert, rank } => {
                ensure!(stage < config.num_nextn_predict_layers,"dSpark stage out of range");
                ensure!(expert < config.dspark_n_routed_experts,"dSpark expert out of range");
                ensure!(rank < 2,"dSpark TP2 rank must be in 0..2");
                (format!("mtp.{stage}.ffn.experts.{expert}"),Some(rank),config.moe_intermediate_size/2)
            }
            V41ExpertSelection::Dspark { stage, expert } => {
                ensure!(
                    stage < config.num_nextn_predict_layers,
                    "dSpark stage out of range"
                );
                ensure!(
                    expert < config.dspark_n_routed_experts,
                    "dSpark expert out of range"
                );
                (
                    format!("mtp.{stage}.ffn.experts.{expert}"),
                    None,
                    config.moe_intermediate_size,
                )
            }
        };
        let suffixes = [
            "w1.weight",
            "w3.weight",
            "w2.weight",
            "w1.scale",
            "w3.scale",
            "w2.scale",
        ];
        let names = suffixes.map(|suffix| format!("{prefix}.{suffix}"));
        let mut ranges = std::array::from_fn(|_| 0..0);
        let mut bytes = 0usize;
        let mut scratch_bytes = 0usize;
        for (slot, name) in names.iter().enumerate() {
            let size = usize::try_from(if matches!(selection, V41ExpertSelection::BackboneFull { .. }) {
                self.tensor(name)?.metadata.byte_length
            } else if matches!(selection, V41ExpertSelection::BackboneTp2 { .. } | V41ExpertSelection::DsparkTp2 { .. }) {
                self.tensor(name)?.metadata.byte_length / 2
            } else {
                self.device_tensor_bytes(name, rank)?
            })?;
            let expected = intermediate
                .checked_mul(config.hidden_size)
                .context("expert staging size overflow")?
                / if slot < 3 { 2 } else { 32 };
            ensure!(
                size == expected,
                "unexpected native expert extent for {name}"
            );
            let start = bytes
                .checked_add(15)
                .context("expert staging alignment overflow")?
                & !15;
            bytes = start
                .checked_add(size)
                .context("expert staging size overflow")?;
            ranges[slot] = start..bytes;
            if rank.is_some() && (slot == 2 || slot == 5) {
                let tensor = self.tensor(name)?;
                let row_bytes = tensor.metadata.byte_length / config.hidden_size as u64;
                scratch_bytes = scratch_bytes.max(usize::try_from(row_bytes)?);
            }
        }
        Ok(V41ExpertStaging {
            catalog: self,
            selection,
            names,
            ranges,
            bytes,
            scratch_bytes,
            intermediate,
            rank,
        })
    }
}

impl V41ExpertStaging<'_> {
    pub fn selection(&self) -> V41ExpertSelection {
        self.selection
    }
    pub fn intermediate_size(&self) -> usize {
        self.intermediate
    }
    pub fn staging_bytes(&self) -> usize {
        self.bytes
    }
    /// Minimum W2 read scratch; larger scratch coalesces more physical rows per read.
    pub fn minimum_read_scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    /// Offsets within host staging or a matching contiguous device staging allocation.
    pub fn tensor_ranges(&self) -> &[Range<usize>; 6] {
        &self.ranges
    }
    pub fn tensor_names(&self) -> &[String; 6] {
        &self.names
    }

    /// Advise only this expert's physical read ranges, without mapping or loading
    /// unrelated experts/engram tables; callers bound the lookahead window.
    pub fn prefetch(&self) -> Result<()> {
        use std::os::fd::AsRawFd;
        for (slot, (name, range)) in self.names.iter().zip(&self.ranges).enumerate() {
            let tensor = self.catalog.tensor(name)?;
            let mut offset = tensor.metadata.byte_offset;
            let bytes = match tensor.placement {
                _ if matches!(self.selection,V41ExpertSelection::DsparkTp2 { .. }) && slot!=2 && slot!=5 => {
                    offset=offset.checked_add(range.len() as u64*self.rank.unwrap() as u64)
                        .context("draft expert prefetch offset overflow")?;
                    range.len() as u64
                }
                crate::V41TensorPlacement::BackboneExpertTp4 { axis: 0, .. } if self.rank.is_some() => {
                    offset = offset
                        .checked_add((range.len() as u64) * self.rank.unwrap() as u64)
                        .context("expert prefetch offset overflow")?;
                    range.len() as u64
                }
                // Strided W2 column slices touch the complete source row span.
                _ => tensor.metadata.byte_length,
            };
            let file = std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?;
            let status = unsafe {
                libc::posix_fadvise(
                    file.as_raw_fd(),
                    i64::try_from(offset)?,
                    i64::try_from(bytes)?,
                    libc::POSIX_FADV_WILLNEED,
                )
            };
            ensure!(
                status == 0,
                "expert prefetch failed: {}",
                std::io::Error::from_raw_os_error(status)
            );
        }
        Ok(())
    }

    /// Read all six native tensors; only a successful return permits packing.
    /// On an I/O failure staging may contain partial data and must not be consumed.
    /// Checkpoint files must remain unchanged after catalog validation.
    pub fn read_into(&self, staging: &mut [u8], scratch: &mut [u8]) -> Result<()> {
        ensure!(
            staging.len() >= self.bytes,
            "expert staging requires {} bytes",
            self.bytes
        );
        ensure!(
            scratch.len() >= self.scratch_bytes,
            "expert read scratch requires {} bytes",
            self.scratch_bytes
        );
        for (slot, (name, range)) in self.names.iter().zip(&self.ranges).enumerate() {
            if matches!(self.selection, V41ExpertSelection::BackboneFull { .. }) {
                use std::os::unix::fs::FileExt;
                let tensor = self.catalog.tensor(name)?;
                std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
                    .read_exact_at(&mut staging[range.clone()], tensor.metadata.byte_offset)
                    .with_context(|| format!("staging full official backbone expert tensor {name}"))?;
            } else if let V41ExpertSelection::BackboneTp2 { rank, .. } = self.selection {
                self.catalog.read_backbone_tp2_into(name, rank, &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging TP2 backbone expert tensor {name}"))?;
            } else if let V41ExpertSelection::DsparkTp2 { rank, .. } = self.selection {
                let axis=usize::from(slot==2 || slot==5);
                self.catalog.read_coordinator_tp2_into(name,axis,rank,&mut staging[range.clone()],scratch)
                    .with_context(||format!("staging TP2 dSpark expert tensor {name}"))?;
            } else {
                self.catalog
                    .read_device_tensor_into(name, self.rank, &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging official expert tensor {name}"))?;
            }
        }
        Ok(())
    }
}
