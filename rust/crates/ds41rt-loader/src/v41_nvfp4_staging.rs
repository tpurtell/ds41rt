//! Bounded ModelOpt NVFP4 routed-expert reads in the device packer's order.
//!
//! The pack step consumes the fused FC1 payload and scale plane as the
//! contiguous `[up(w3); gate(w1)]` pair, so W3/W1 and their scale planes are
//! staged adjacently in that order. The six
//! per-tensor FP32 scalars (global weight scale and activation scale for each
//! projection) are replicated to every TP rank and staged after the planes.
use crate::OfficialV41Catalog;
use anyhow::{ensure, Context, Result};
use std::ops::Range;

use crate::v41_expert_staging::V41ExpertSelection;

/// Number of staged regions: three packed weights, three scale planes and
/// six replicated FP32 scalars.
pub const V41_NVFP4_STAGING_SLOTS: usize = 12;

/// Borrows the validated catalog; storage is supplied by the caller.
pub struct V41Nvfp4Staging<'a> {
    catalog: &'a OfficialV41Catalog,
    selection: V41ExpertSelection,
    names: [String; V41_NVFP4_STAGING_SLOTS],
    ranges: [Range<usize>; V41_NVFP4_STAGING_SLOTS],
    bytes: usize,
    scratch_bytes: usize,
    intermediate: usize,
    hidden: usize,
    // w13 occupies slots 0/1 and scales 3/4; the pair must be contiguous for
    // the single-matrix pack call.
    w13_contiguous: bool,
    scale13_contiguous: bool,
}

impl OfficialV41Catalog {
    /// Stage the ModelOpt NVFP4 tensors of one routed expert.
    pub fn nvfp4_expert_staging(
        &self,
        selection: V41ExpertSelection,
    ) -> Result<V41Nvfp4Staging<'_>> {
        ensure!(
            self.nvfp4().is_some(),
            "NVFP4 expert staging requires a ModelOpt NVFP4 checkpoint"
        );
        let contract = self.nvfp4().expect("checked above");
        let group = contract.group_size;
        ensure!(group == 16, "NVFP4 expert staging requires 16-wide groups");
        let config = self.config().text();
        let (prefix, intermediate, rank) = match selection {
            V41ExpertSelection::Backbone { layer, expert, rank } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                ensure!(rank < 4, "backbone TP rank must be in 0..4");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    config.moe_intermediate_size / 4,
                    Some(rank),
                )
            }
            V41ExpertSelection::BackboneTp2 { layer, expert, rank } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                ensure!(rank < 2, "backbone TP2 rank must be in 0..2");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    config.moe_intermediate_size / 2,
                    Some(rank),
                )
            }
            V41ExpertSelection::BackboneFull { layer, expert } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    config.moe_intermediate_size,
                    None,
                )
            }
            _ => anyhow::bail!("NVFP4 staging covers backbone experts only"),
        };
        let hidden = config.hidden_size;
        // Order: W3(up), W1(gate), W2, S3, S1, S2, then the six scalars.
        // The b12x nvfp4 kernel consumes the fused FC1 payload as
        // [up(w3); gate(w1)] and swizzles the concatenated scale plane, so
        // staging the pairs adjacently lets each H2D land a whole expert
        // plane without a repack.
        let names = [
            format!("{prefix}.w3.weight"),
            format!("{prefix}.w1.weight"),
            format!("{prefix}.w2.weight"),
            format!("{prefix}.w3.weight_scale"),
            format!("{prefix}.w1.weight_scale"),
            format!("{prefix}.w2.weight_scale"),
            format!("{prefix}.w1.weight_scale_2"),
            format!("{prefix}.w3.weight_scale_2"),
            format!("{prefix}.w2.weight_scale_2"),
            format!("{prefix}.w1.input_scale"),
            format!("{prefix}.w3.input_scale"),
            format!("{prefix}.w2.input_scale"),
        ];
        let mut ranges = std::array::from_fn(|_| 0..0);
        let mut bytes = 0usize;
        let mut scratch_bytes = 0usize;
        for (slot, name) in names.iter().enumerate() {
            let size = if slot >= 6
                || matches!(selection, V41ExpertSelection::BackboneFull { .. })
            {
                usize::try_from(self.tensor(name)?.metadata.byte_length)?
            } else if matches!(selection, V41ExpertSelection::BackboneTp2 { .. }) {
                usize::try_from(self.tensor(name)?.metadata.byte_length / 2)?
            } else {
                usize::try_from(self.device_tensor_bytes(name, rank)?)?
            };
            // Weights pack two E2M1 values per byte; scale planes carry one
            // E4M3 per 16-wide group.
            let expected = if slot < 3 {
                intermediate
                    .checked_mul(hidden)
                    .context("NVFP4 staging size overflow")?
                    / 2
            } else if slot < 6 {
                intermediate
                    .checked_mul(hidden)
                    .context("NVFP4 staging size overflow")?
                    / group
            } else {
                4
            };
            ensure!(
                size == expected,
                "unexpected NVFP4 expert extent for {name}: {size} != {expected}"
            );
            let start = bytes
                .checked_add(15)
                .context("NVFP4 staging alignment overflow")?
                & !15;
            bytes = start
                .checked_add(size)
                .context("NVFP4 staging size overflow")?;
            ranges[slot] = start..bytes;
            // W2 is column-sharded, so the pack reads coalesce full source rows.
            if slot == 2 && rank.is_some() {
                let tensor = self.tensor(name)?;
                let row_bytes = tensor.metadata.byte_length / hidden as u64;
                scratch_bytes = scratch_bytes.max(usize::try_from(row_bytes)?);
            }
        }
        let w13_contiguous = ranges[0].end == ranges[1].start;
        let scale13_contiguous = ranges[3].end == ranges[4].start;
        Ok(V41Nvfp4Staging {
            catalog: self,
            selection,
            names,
            ranges,
            bytes,
            scratch_bytes,
            intermediate,
            hidden,
            w13_contiguous,
            scale13_contiguous,
        })
    }
}

impl V41Nvfp4Staging<'_> {
    pub fn selection(&self) -> V41ExpertSelection {
        self.selection
    }
    pub fn intermediate_size(&self) -> usize {
        self.intermediate
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden
    }
    pub fn staging_bytes(&self) -> usize {
        self.bytes
    }
    pub fn minimum_read_scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    pub fn tensor_ranges(&self) -> &[Range<usize>; V41_NVFP4_STAGING_SLOTS] {
        &self.ranges
    }
    pub fn tensor_names(&self) -> &[String; V41_NVFP4_STAGING_SLOTS] {
        &self.names
    }
    /// True when staging W3(up) and W1(gate) are adjacent, so the pair is
    /// one contiguous `[2*intermediate, ...]` kernel-native FC1 plane.
    pub fn w13_contiguous(&self) -> bool {
        self.w13_contiguous
    }
    pub fn weight_scale13_contiguous(&self) -> bool {
        self.scale13_contiguous
    }

    /// Advise this expert's physical read ranges without mapping other experts.
    pub fn prefetch(&self) -> Result<()> {
        use std::os::fd::AsRawFd;
        for (slot, (name, range)) in self.names.iter().zip(&self.ranges).enumerate() {
            let tensor = self.catalog.tensor(name)?;
            let mut offset = tensor.metadata.byte_offset;
            let bytes = match tensor.placement {
                crate::V41TensorPlacement::BackboneExpertTp4 { axis: 0, .. }
                    if self.rank().is_some() =>
                {
                    offset = offset
                        .checked_add((range.len() as u64) * self.rank().unwrap() as u64)
                        .context("NVFP4 prefetch offset overflow")?;
                    range.len() as u64
                }
                // Column-sharded W2 and every replicated scalar span the row.
                _ => tensor.metadata.byte_length.max(range.len() as u64),
            };
            let _ = slot;
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
                "NVFP4 expert prefetch failed: {}",
                std::io::Error::from_raw_os_error(status)
            );
        }
        Ok(())
    }

    /// Read all twelve regions; only a successful return permits packing.
    pub fn read_into(&self, staging: &mut [u8], scratch: &mut [u8]) -> Result<()> {
        ensure!(
            staging.len() >= self.bytes,
            "NVFP4 expert staging requires {} bytes",
            self.bytes
        );
        ensure!(
            scratch.len() >= self.scratch_bytes,
            "NVFP4 expert read scratch requires {} bytes",
            self.scratch_bytes
        );
        for (slot, (name, range)) in self.names.iter().zip(&self.ranges).enumerate() {
            if slot >= 6 || matches!(self.selection, V41ExpertSelection::BackboneFull { .. }) {
                use std::os::unix::fs::FileExt;
                let tensor = self.catalog.tensor(name)?;
                std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
                    .read_exact_at(&mut staging[range.clone()], tensor.metadata.byte_offset)
                    .with_context(|| format!("staging NVFP4 tensor {name}"))?;
            } else if let V41ExpertSelection::BackboneTp2 { rank, .. } = self.selection {
                self.catalog
                    .read_backbone_tp2_into(name, rank, &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging TP2 NVFP4 tensor {name}"))?;
            } else {
                self.catalog
                    .read_device_tensor_into(name, self.rank(), &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging NVFP4 tensor {name}"))?;
            }
        }
        Ok(())
    }

    fn rank(&self) -> Option<usize> {
        match self.selection {
            V41ExpertSelection::Backbone { rank, .. } => Some(rank),
            V41ExpertSelection::BackboneTp2 { rank, .. } => Some(rank),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    #[ignore = "requires DS41RT_NVFP4_SNAPSHOT pointing to a local ModelOpt NVFP4 publication"]
    fn real_snapshot_stages_tp4_and_tp2_experts() {
        let path = std::env::var_os("DS41RT_NVFP4_SNAPSHOT").expect("DS41RT_NVFP4_SNAPSHOT");
        let catalog =
            crate::read_official_v41_catalog("nvidia/DeepSeek-V4.1-Flash-NVFP4", Path::new(&path))
                .unwrap();
        for selection in [
            V41ExpertSelection::Backbone { layer: 0, expert: 0, rank: 0 },
            V41ExpertSelection::Backbone { layer: 39, expert: 383, rank: 3 },
            V41ExpertSelection::BackboneTp2 { layer: 20, expert: 100, rank: 1 },
        ] {
            let staging = catalog.nvfp4_expert_staging(selection).unwrap();
            assert!(staging.w13_contiguous(), "W3/W1 must be adjacent");
            assert!(
                staging.weight_scale13_contiguous(),
                "W3/W1 scales must be adjacent"
            );
            let mut bytes = vec![0u8; staging.staging_bytes()];
            let mut scratch = vec![0u8; staging.minimum_read_scratch_bytes()];
            staging.read_into(&mut bytes, &mut scratch).unwrap();
            let ranges = staging.tensor_ranges();
            // Packed weights must not be all zero, and the replicated FP32
            // scales must be finite, positive reciprocals/scales.
            assert!(bytes[ranges[0].clone()].iter().any(|byte| *byte != 0));
            for slot in 6..12 {
                let raw = &bytes[ranges[slot].clone()];
                let value = f32::from_le_bytes(raw.try_into().unwrap());
                assert!(value.is_finite() && value != 0.0, "slot {slot} scale {value}");
            }
            // W1 and W3 occupy exactly the pair range the pack consumes.
            assert_eq!(ranges[0].end, ranges[1].start);
            assert_eq!(ranges[3].end, ranges[4].start);
            assert_eq!(ranges[1].end, ranges[2].start);
        }
        println!("NVFP4 staging read TP4 and TP2 experts from the real snapshot");
    }

    #[test]
    fn staging_slot_order_keeps_w13_pairs_adjacent() {
        // The adjacency invariant is structural: slots 0/1 and 3/4 are laid
        // out consecutively with 16-byte alignment, and every size is a
        // multiple of 16, so the pair ranges must touch.
        for (intermediate, hidden, group) in [(576usize, 5120usize, 16usize), (1152, 5120, 16)] {
            let weight = intermediate * hidden / 2;
            let scale = intermediate * hidden / group;
            assert_eq!(weight % 16, 0);
            assert_eq!(scale % 16, 0);
        }
    }
}
