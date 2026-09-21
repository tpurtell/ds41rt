//! Bounded official expert reads in the native packer's source order.
//!
//! Besides the fixed RTX/Spark TP4 and RTX-pair TP2 layouts, this module
//! stages an explicitly selected shard of one official native FP4/E8M0
//! backbone routed expert for a caller-chosen TP world of 2, 3 or 4. The
//! generic selection keeps the same six-region plan (`W1,W3,W2,S1,S3,S2`) and
//! the same packing meaning as the fixed paths; only the shard geometry is
//! parameterized.
use crate::{OfficialV41Catalog, V41TensorPlacement};
use anyhow::{ensure, Context, Result};
use ds41rt_core::DType;
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
    /// Explicitly sized TP shard of one official native FP4/E8M0 backbone
    /// routed expert. `world` is the shard count (2, 3, 4 or 6) and `rank` is
    /// the shard index. W1/W3 weights and both scale planes slice output rows;
    /// W2 weight and scale slice input columns. Only the official MXFP4
    /// (native FP4/E8M0) checkpoint is accepted: EXL3 and ModelOpt NVFP4
    /// publications keep their own compressed/NVFP4 staging.
    BackboneTp {
        layer: usize,
        expert: usize,
        rank: usize,
        world: usize,
    },
    Dspark {
        stage: usize,
        expert: usize,
    },
    /// Half-width routed draft expert on one member of an RTX pair.
    DsparkTp2 { stage: usize, expert: usize, rank: usize },
}

/// Validated geometry of one explicit TP shard of an official native FP4/E8M0
/// backbone routed expert.
///
/// The shard carries `moe_intermediate_size / world` intermediate values.
/// W1/W3 (and their per-32-value E8M0 scale planes) slice output rows, so the
/// slice boundary always falls on a whole scale group. W2 instead slices input
/// columns; because one byte holds two FP4 values and one E8M0 scale covers 32
/// input values, both the packed row slice and the scale row slice stay
/// byte/group aligned for every rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41BackboneTpGeometry {
    intermediate: usize,
    hidden: usize,
    world: usize,
    rank: usize,
    shard_intermediate: usize,
}

impl V41BackboneTpGeometry {
    /// Validate an explicit TP shard geometry against the official group-32
    /// packed-FP4 storage contract.
    pub fn new(
        moe_intermediate_size: usize,
        hidden_size: usize,
        world: usize,
        rank: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(world, 2 | 3 | 4 | 6),
            "backbone TP world must be 2, 3, 4 or 6, got {world}"
        );
        ensure!(
            rank < world,
            "backbone TP rank {rank} must be in 0..{world}"
        );
        ensure!(
            moe_intermediate_size > 0 && hidden_size > 0,
            "backbone expert geometry must be non-empty"
        );
        ensure!(
            hidden_size % 32 == 0,
            "backbone hidden size {hidden_size} does not align to 32-value scale groups"
        );
        ensure!(
            moe_intermediate_size % world == 0,
            "backbone intermediate size {moe_intermediate_size} is not divisible by TP world {world}"
        );
        let shard_intermediate = moe_intermediate_size / world;
        // W1/W3 output rows and their scale rows both advance by whole rows, so
        // only the W2 input-column slice needs explicit alignment: one E8M0
        // scale covers 32 input values and one byte packs two FP4 values.
        ensure!(
            shard_intermediate % 32 == 0,
            "backbone TP shard of {shard_intermediate} intermediate values does not align to 32-value scale groups"
        );
        ensure!(
            (moe_intermediate_size / 2) % world == 0,
            "backbone TP world {world} splits the packed FP4 W2 rows of {moe_intermediate_size} intermediate values"
        );
        // Every staged extent is derived from shard_intermediate * hidden_size;
        // reject geometries whose extents cannot be represented.
        shard_intermediate
            .checked_mul(hidden_size)
            .context("backbone TP shard extent overflow")?;
        Ok(Self {
            intermediate: moe_intermediate_size,
            hidden: hidden_size,
            world,
            rank,
            shard_intermediate,
        })
    }

    pub fn world(&self) -> usize {
        self.world
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Intermediate values carried by this shard (`moe_intermediate_size / world`).
    pub fn intermediate_size(&self) -> usize {
        self.shard_intermediate
    }

    pub fn full_intermediate_size(&self) -> usize {
        self.intermediate
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// Packed W1/W3 weight bytes per shard: `hidden/2` bytes per output row.
    pub fn w13_weight_shard_bytes(&self) -> usize {
        self.shard_intermediate * (self.hidden / 2)
    }

    /// E8M0 W1/W3 scale bytes per shard: one byte per 32 hidden values.
    pub fn w13_scale_shard_bytes(&self) -> usize {
        self.shard_intermediate * (self.hidden / 32)
    }

    /// Packed W2 weight bytes per shard; W2 rows span the full intermediate
    /// dimension, so this is the input-column slice of every row.
    pub fn w2_weight_shard_bytes(&self) -> usize {
        self.hidden * (self.shard_intermediate / 2)
    }

    /// E8M0 W2 scale bytes per shard: `shard_intermediate/32` groups per row.
    pub fn w2_scale_shard_bytes(&self) -> usize {
        self.hidden * (self.shard_intermediate / 32)
    }

    /// Bytes of one full source W2 weight row; also the W2 read scratch bound.
    pub fn w2_weight_row_bytes(&self) -> usize {
        self.intermediate / 2
    }

    /// Bytes of one full source W2 scale row; also the W2 scale scratch bound.
    pub fn w2_scale_row_bytes(&self) -> usize {
        self.intermediate / 32
    }

    /// Byte offset of this shard inside every packed W2 weight row.
    pub fn w2_weight_column_offset_bytes(&self) -> usize {
        self.rank * (self.shard_intermediate / 2)
    }

    /// Byte offset of this shard inside every W2 scale row.
    pub fn w2_scale_column_offset_bytes(&self) -> usize {
        self.rank * (self.shard_intermediate / 32)
    }
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
    geometry: Option<V41BackboneTpGeometry>,
}

impl OfficialV41Catalog {
    pub fn expert_staging(&self, selection: V41ExpertSelection) -> Result<V41ExpertStaging<'_>> {
        if self.exl3().is_some() {
            // Raw EXL3 publications keep draft experts at native FP4 source
            // precision; only those selections may use the native packer.
            ensure!(
                self.native_dspark_experts(),
                "EXL3 experts require compressed projection staging, not the native FP4 packer"
            );
            ensure!(
                matches!(selection, V41ExpertSelection::Dspark { .. }),
                "source-precision drafts are the only native experts in an EXL3 checkpoint"
            );
        }
        let config = self.config().text();
        let (prefix, rank, intermediate, geometry) = match selection {
            V41ExpertSelection::BackboneTp {
                layer,
                expert,
                rank,
                world,
            } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                ensure!(
                    self.exl3().is_none() && self.nvfp4().is_none(),
                    "generic backbone TP staging covers only the native FP4/E8M0 official checkpoint"
                );
                let geometry = V41BackboneTpGeometry::new(
                    config.moe_intermediate_size,
                    config.hidden_size,
                    world,
                    rank,
                )?;
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    Some(rank),
                    geometry.intermediate_size(),
                    Some(geometry),
                )
            }
            V41ExpertSelection::BackboneTp2 { layer, expert, rank } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                ensure!(rank < 2, "backbone TP2 rank must be in 0..2");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    Some(rank),
                    config.moe_intermediate_size / 2,
                    None,
                )
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
                    None,
                )
            }
            V41ExpertSelection::BackboneFull { layer, expert } => {
                ensure!(layer < config.num_hidden_layers, "backbone layer out of range");
                ensure!(expert < config.n_routed_experts, "backbone expert out of range");
                (
                    format!("layers.{layer}.ffn.experts.{expert}"),
                    None,
                    config.moe_intermediate_size,
                    None,
                )
            }
            V41ExpertSelection::DsparkTp2 { stage, expert, rank } => {
                ensure!(stage < config.num_nextn_predict_layers,"dSpark stage out of range");
                ensure!(expert < config.dspark_n_routed_experts,"dSpark expert out of range");
                ensure!(rank < 2,"dSpark TP2 rank must be in 0..2");
                (format!("mtp.{stage}.ffn.experts.{expert}"),Some(rank),config.moe_intermediate_size/2,None)
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
                    None,
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
            let size = if let Some(geometry) = geometry {
                // The generic selection never trusts the TP4-hardcoded catalog
                // helpers: every extent is re-derived from validated metadata.
                let tensor = self.tensor(name)?;
                ensure!(
                    matches!(tensor.placement, V41TensorPlacement::BackboneExpertTp4 { .. }),
                    "generic TP staging requires a routed backbone expert tensor {name}"
                );
                let expected_dtype = if slot < 3 { DType::I8 } else { DType::F8E8M0 };
                ensure!(
                    tensor.metadata.dtype == expected_dtype,
                    "generic TP staging requires native FP4/E8M0 storage for {name}"
                );
                let length = usize::try_from(tensor.metadata.byte_length)?;
                ensure!(
                    length > 0 && length % geometry.world() == 0,
                    "generic TP tensor {name} does not divide into {} shards",
                    geometry.world()
                );
                let size = length / geometry.world();
                let expected = match slot {
                    0 | 1 => geometry.w13_weight_shard_bytes(),
                    3 | 4 => geometry.w13_scale_shard_bytes(),
                    2 => geometry.w2_weight_shard_bytes(),
                    _ => geometry.w2_scale_shard_bytes(),
                };
                ensure!(
                    size == expected,
                    "unexpected native expert extent for {name}: {size} != {expected}"
                );
                size
            } else if matches!(selection, V41ExpertSelection::BackboneFull { .. }) {
                usize::try_from(self.tensor(name)?.metadata.byte_length)?
            } else if matches!(selection, V41ExpertSelection::BackboneTp2 { .. } | V41ExpertSelection::DsparkTp2 { .. }) {
                usize::try_from(self.tensor(name)?.metadata.byte_length / 2)?
            } else {
                usize::try_from(self.device_tensor_bytes(name, rank)?)?
            };
            if geometry.is_none() {
                let expected = intermediate
                    .checked_mul(config.hidden_size)
                    .context("expert staging size overflow")?
                    / if slot < 3 { 2 } else { 32 };
                ensure!(
                    size == expected,
                    "unexpected native expert extent for {name}"
                );
            }
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
            geometry,
        })
    }
}

impl OfficialV41Catalog {
    /// Read one explicitly sized TP shard of an official native FP4/E8M0
    /// backbone expert tensor.
    ///
    /// Axis 0 (W1/W3 weights and both scale planes) is a contiguous output-row
    /// slice. Axis 1 (W2 weight and scale) is a strided input-column slice:
    /// whole source rows are read through caller-owned scratch and each row's
    /// byte/group-aligned column window is copied out, so the file offset of
    /// every row is checked before the read.
    fn read_backbone_tp_into(
        &self,
        name: &str,
        geometry: V41BackboneTpGeometry,
        dst: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<usize> {
        use std::os::unix::fs::FileExt;
        let tensor = self.tensor(name)?;
        let V41TensorPlacement::BackboneExpertTp4 { axis, .. } = tensor.placement else {
            anyhow::bail!("generic TP read requires a routed backbone expert tensor {name}");
        };
        let metadata = &tensor.metadata;
        let length = usize::try_from(metadata.byte_length)?;
        ensure!(
            length > 0 && length % geometry.world() == 0,
            "expert tensor {name} does not divide into {} TP shards",
            geometry.world()
        );
        let bytes = length / geometry.world();
        ensure!(
            dst.len() >= bytes,
            "tensor staging buffer for {name} needs {bytes} bytes"
        );
        let file = std::fs::File::open(self.snapshot().join(&tensor.shard))?;
        match axis {
            0 => {
                let offset = metadata
                    .byte_offset
                    .checked_add(
                        u64::try_from(
                            bytes
                                .checked_mul(geometry.rank())
                                .context("generic TP row offset overflow")?,
                        )
                        .context("generic TP row offset overflow")?,
                    )
                    .context("generic TP file offset overflow")?;
                file.read_exact_at(&mut dst[..bytes], offset)
                    .with_context(|| format!("staging generic TP rows of {name}"))?;
            }
            1 => {
                ensure!(
                    metadata.shape.len() == 2 && metadata.shape[0] > 0,
                    "generic W2 column slice requires a matrix {name}"
                );
                let rows = metadata.shape[0];
                let row_bytes = length / rows;
                ensure!(
                    row_bytes > 0 && row_bytes % geometry.world() == 0,
                    "W2 rows of {name} do not divide into {} TP shards",
                    geometry.world()
                );
                let shard_bytes = row_bytes / geometry.world();
                ensure!(
                    rows.checked_mul(shard_bytes) == Some(bytes),
                    "W2 TP shard extent mismatch for {name}"
                );
                ensure!(
                    scratch.len() >= row_bytes,
                    "W2 scratch needs at least {row_bytes} bytes for {name}"
                );
                let column = geometry.rank() * shard_bytes;
                let rows_per_read = scratch.len() / row_bytes;
                for start in (0..rows).step_by(rows_per_read) {
                    let count = rows_per_read.min(rows - start);
                    let offset = metadata
                        .byte_offset
                        .checked_add(
                            u64::try_from(
                                start
                                    .checked_mul(row_bytes)
                                    .context("generic W2 row offset overflow")?,
                            )
                            .context("generic W2 row offset overflow")?,
                        )
                        .context("generic W2 file offset overflow")?;
                    file.read_exact_at(&mut scratch[..count * row_bytes], offset)
                        .with_context(|| format!("staging generic TP W2 rows of {name}"))?;
                    for row in 0..count {
                        dst[(start + row) * shard_bytes..(start + row + 1) * shard_bytes]
                            .copy_from_slice(
                                &scratch[row * row_bytes + column..row * row_bytes + column + shard_bytes],
                            );
                    }
                }
            }
            _ => anyhow::bail!("unsupported backbone expert TP axis {axis} for {name}"),
        }
        Ok(bytes)
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
    /// Validated explicit-TP geometry for `BackboneTp`; `None` for the fixed
    /// TP4/TP2/full selections, which keep their existing meaning.
    pub fn geometry(&self) -> Option<V41BackboneTpGeometry> {
        self.geometry
    }
    /// Shard count of an explicit `BackboneTp` selection, else `None`.
    pub fn tp_world(&self) -> Option<usize> {
        self.geometry.map(|geometry| geometry.world())
    }
    /// Validate an explicit TP shard geometry without a catalog, mirroring the
    /// checks `expert_staging` applies to `BackboneTp`.
    pub fn backbone_tp_geometry(
        moe_intermediate_size: usize,
        hidden_size: usize,
        world: usize,
        rank: usize,
    ) -> Result<V41BackboneTpGeometry> {
        V41BackboneTpGeometry::new(moe_intermediate_size, hidden_size, world, rank)
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
            } else if matches!(self.selection, V41ExpertSelection::BackboneTp { .. }) {
                let geometry = self.geometry.expect("generic TP plan carries geometry");
                self.catalog
                    .read_backbone_tp_into(name, geometry, &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging generic TP backbone expert tensor {name}"))?;
            } else {
                self.catalog
                    .read_device_tensor_into(name, self.rank, &mut staging[range.clone()], scratch)
                    .with_context(|| format!("staging official expert tensor {name}"))?;
            }
        }
        Ok(())
    }
}
