//! Native V4.1 expert residency; one GPU worker owns each layer and its buffers.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
pub(crate) mod coordinator;
pub(crate) mod paired;
pub(crate) mod local;
pub(crate) mod tp2;
pub(crate) mod tp2_ffn;
pub(crate) mod dspark;
mod execution;
pub(crate) mod exl3;
pub(crate) mod nvfp4;
pub(crate) mod service;
pub(crate) use execution::{ExpertExecution, ExpertExecutionBudget, HostExpertExchange};

use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{NativeLibrary, V41ExpertKernel, V41_EXPERT_POINTER_COUNT};
use ds41rt_loader::{OfficialV41Catalog, V41ExpertSelection};
use std::ffi::c_void;

// Bound disk concurrency and pinned staging independently of model layer count.
const EXPERT_READ_LANES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertLayer {
    Backbone { layer: usize, rank: usize },
    BackboneFull { layer: usize },
    BackboneTp2 { layer: usize, rank: usize },
    Dspark { stage: usize },
    DsparkTp2 { stage: usize, rank: usize },
}
impl ExpertLayer {
    fn expert(self, expert: usize) -> V41ExpertSelection {
        match self {
            Self::Backbone { layer, rank } => V41ExpertSelection::Backbone {
                layer,
                rank,
                expert,
            },
            Self::BackboneFull { layer } => V41ExpertSelection::BackboneFull { layer, expert },
            Self::BackboneTp2 { layer, rank } => V41ExpertSelection::BackboneTp2 { layer, expert, rank },
            Self::Dspark { stage } => V41ExpertSelection::Dspark { stage, expert },
            Self::DsparkTp2 { stage, rank } => V41ExpertSelection::DsparkTp2 { stage, expert, rank },
        }
    }
    fn role(self) -> u32 {
        match self {
            Self::Dspark { .. } => 0,
            Self::Backbone { .. } => 1,
            Self::BackboneFull { .. } => 2,
            Self::BackboneTp2 { .. } => 3,
            Self::DsparkTp2 { .. } => 4,
        }
    }
    fn info(self, library: &NativeLibrary, capacity: u32) -> Result<ds41rt_ffi::V41ExpertInfo> {
        if matches!(self, Self::BackboneFull { .. }) {
            library.v41_local_expert_info(capacity)
        } else if matches!(self, Self::BackboneTp2 { .. }) {
            library.v41_tp2_expert_info(capacity)
        } else if matches!(self,Self::DsparkTp2 { .. }) {
            library.v41_dspark_tp2_expert_info(capacity)
        } else { library.v41_expert_info(capacity) }
    }
    fn kernel(self, library: &NativeLibrary, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        if matches!(self, Self::BackboneFull { .. }) {
            library.v41_local_expert_kernel(capacity)
        } else if matches!(self, Self::BackboneTp2 { .. }) {
            library.v41_tp2_expert_kernel(capacity)
        } else if matches!(self,Self::DsparkTp2 { .. }) {
            library.v41_dspark_tp2_expert_kernel(capacity)
        } else { library.v41_expert_kernel(capacity) }
    }

    /// W4A4 NVFP4 family selection. Only backbone experts are quantized; the
    /// MTP draft path stays on the native family.
    fn nvfp4_kernel(self, library: &NativeLibrary, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        match self {
            Self::Backbone { .. } => library.v41_nvfp4_expert_kernel(capacity),
            Self::BackboneTp2 { .. } => library.v41_nvfp4_tp2_expert_kernel(capacity),
            // Raw publications keep draft experts at source MXFP4 precision.
            other => other.kernel(library, capacity),
        }
    }

    fn nvfp4_info(self, library: &NativeLibrary, capacity: u32) -> Result<ds41rt_ffi::V41ExpertInfo> {
        match self {
            Self::Backbone { .. } => library.v41_nvfp4_expert_info(capacity),
            Self::BackboneTp2 { .. } => library.v41_nvfp4_tp2_expert_info(capacity),
            other => other.info(library, capacity),
        }
    }

    /// True when this layer's routed experts use the W4A4 NVFP4 family.
    fn is_quantized_nvfp4(self, catalog: &OfficialV41Catalog) -> bool {
        catalog.nvfp4().is_some()
            && matches!(self, Self::Backbone { .. } | Self::BackboneTp2 { .. })
    }
}

/// Expert checkpoint format selected from the validated catalog. All three
/// families can coexist in one library; the daemon picks one per deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertFormat {
    /// Official checkpoint: MXFP4 routed experts with E8M0 K32 scales.
    Native,
    /// Staged or raw EXL3 trellis publications.
    Exl3,
    /// ModelOpt NVFP4 (W4A4) publications.
    Nvfp4,
}

impl ExpertFormat {
    pub(crate) fn of(catalog: &OfficialV41Catalog) -> Self {
        if catalog.exl3().is_some() {
            Self::Exl3
        } else if catalog.nvfp4().is_some() {
            Self::Nvfp4
        } else {
            Self::Native
        }
    }
    pub(crate) fn is_exl3(self) -> bool {
        matches!(self, Self::Exl3)
    }
    pub(crate) fn is_nvfp4(self) -> bool {
        matches!(self, Self::Nvfp4)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertLoadBudget {
    pub resident_bytes: usize,
    pub device_staging_bytes: usize,
    pub pinned_host_bytes: usize,
    pub read_scratch_bytes: usize,
}
impl ExpertLoadBudget {
    pub fn peak_device_bytes(self) -> Result<usize> {
        self.resident_bytes
            .checked_add(self.device_staging_bytes)
            .context("expert load budget overflow")
    }
}

/// Resident packed weights borrow the native library; no logical layer copy remains.
/// Captured graphs must be destroyed before this owner is dropped.
pub(crate) struct ExpertWeights<'a> {
    buffers: [DeviceAllocation<'a>; 4],
    layer: ExpertLayer,
    experts: usize,
    budget: ExpertLoadBudget,
    /// W4A4 resident storage. `None` for the native and EXL3 families.
    nvfp4: Option<nvfp4::Nvfp4Side<'a>>,
    format: ExpertFormat,
}
impl<'a> ExpertWeights<'a> {
    fn layout(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<(ExpertLoadBudget, [usize; 4], u32, usize)> {
        if layer.is_quantized_nvfp4(catalog) {
            return Self::nvfp4_layout(library, catalog, layer);
        }
        let first = catalog.expert_staging(layer.expert(0))?;
        // The NVFP4 family reports its own geometry (unpadded intermediate).
        let info = layer.nvfp4_info(library, 16)?;
        ensure!(
            info.role == layer.role(),
            "native expert role does not match layer placement"
        );
        ensure!(
            info.logical_intermediate as usize == first.intermediate_size(),
            "native expert intermediate mismatch"
        );
        let experts = info.experts as usize;
        let packer = library.v41_expert_packer(info.logical_intermediate)?;
        let strides = packer.packed_bytes().map(usize::try_from);
        let mut sizes = [0usize; 4];
        for (size, stride) in sizes.iter_mut().zip(strides) {
            *size = stride?
                .checked_mul(experts)
                .context("resident expert allocation overflow")?;
        }
        let resident_bytes = sizes.iter().try_fold(0usize, |sum, size| {
            sum.checked_add(*size)
                .context("resident expert byte overflow")
        })?;
        let budget = ExpertLoadBudget {
            resident_bytes,
            device_staging_bytes: first.staging_bytes(),
            pinned_host_bytes: first
                .staging_bytes()
                .checked_mul(EXPERT_READ_LANES)
                .context("expert pinned staging overflow")?,
            read_scratch_bytes: first
                .minimum_read_scratch_bytes()
                .checked_mul(64 * EXPERT_READ_LANES)
                .context("expert read scratch overflow")?,
        };
        Ok((budget, sizes, info.logical_intermediate, experts))
    }
    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        Ok(Self::layout(library, catalog, layer)?.0)
    }

    /// W4A4 layout: the four per-expert planes replace the packed W4A8 slabs.
    fn nvfp4_layout(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<(ExpertLoadBudget, [usize; 4], u32, usize)> {
        let (intermediate, experts) =
            nvfp4::Nvfp4Side::rank_planes(layer, catalog)?;
        let budget = nvfp4::Nvfp4Side::plan(library, catalog, layer)?;
        let sizes = nvfp4::plane_sizes(intermediate);
        Ok((budget, sizes, intermediate as u32, experts))
    }

    /// True when this layer's experts are W4A4 ModelOpt NVFP4.
    pub(crate) fn is_nvfp4(&self) -> bool {
        self.nvfp4.is_some()
    }

    pub(crate) fn format(&self) -> ExpertFormat {
        self.format
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        if layer.is_quantized_nvfp4(catalog) {
            return Self::load_nvfp4(library, catalog, layer, available_device_bytes);
        }
        let (budget, sizes, intermediate, experts) = Self::layout(library, catalog, layer)?;
        let packer = library.v41_expert_packer(intermediate)?;
        ensure!(budget.peak_device_bytes()? <= available_device_bytes,
            "expert layer needs {} device bytes including staging, budget is {available_device_bytes}", budget.peak_device_bytes()?);
        // Fail role/device checks and allocation admission before opening payloads.
        // The NVFP4 family loads its own variants; both report the same roles.
        let format = ExpertFormat::of(catalog);
        let _kernel = if format.is_nvfp4() {
            layer.nvfp4_kernel(library, 16)?
        } else {
            layer.kernel(library, 16)?
        };
        let mut owned = Vec::with_capacity(4);
        for size in sizes {
            owned.push(DeviceAllocation::new(library, size)?);
        }
        let buffers: [DeviceAllocation<'a>; 4] =
            owned.try_into().ok().expect("four packed buffers");
        let device_staging = DeviceAllocation::new(library, budget.device_staging_bytes)?;
        let mut hosts = (0..EXPERT_READ_LANES)
            .map(|_| HostAllocation::new(library, budget.pinned_host_bytes / EXPERT_READ_LANES))
            .collect::<Result<Vec<_>>>()?;
        let mut read_scratch = (0..EXPERT_READ_LANES)
            .map(|_| vec![0; budget.read_scratch_bytes / EXPERT_READ_LANES])
            .collect::<Vec<_>>();
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        // Read one bounded group in parallel while advising only the next group.
        // CPU readers borrow disjoint pinned byte slices; all CUDA calls remain
        // on this owning thread, after the scoped readers have joined.
        for expert in 0..EXPERT_READ_LANES.min(experts) {
            catalog.expert_staging(layer.expert(expert))?.prefetch()?;
        }
        for first in (0..experts).step_by(EXPERT_READ_LANES) {
            let end = (first + EXPERT_READ_LANES).min(experts);
            for future in end..(end + EXPERT_READ_LANES).min(experts) {
                catalog.expert_staging(layer.expert(future))?.prefetch()?;
            }
            let plans = (first..end)
                .map(|expert| catalog.expert_staging(layer.expert(expert)))
                .collect::<Result<Vec<_>>>()?;
            std::thread::scope(|scope| -> Result<()> {
                let mut readers = Vec::with_capacity(plans.len());
                for ((plan, host), scratch) in plans.iter().zip(&mut hosts).zip(&mut read_scratch) {
                    let bytes = host.bytes_mut();
                    readers.push(scope.spawn(move || plan.read_into(bytes, scratch)));
                }
                for reader in readers {
                    reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("expert read thread panicked"))??;
                }
                Ok(())
            })?;
            for (offset, (plan, host)) in plans.iter().zip(&hosts).enumerate() {
                let expert = first + offset;
                unsafe {
                    library.copy_host_buffer_h2d_async(
                        device_staging.buffer,
                        host.buffer,
                        plan.staging_bytes(),
                        stream.raw,
                    )?;
                    let sources = std::array::from_fn(|i| {
                        device_staging
                            .buffer
                            .ptr
                            .cast::<u8>()
                            .add(plan.tensor_ranges()[i].start)
                            .cast_const()
                    });
                    let destinations = std::array::from_fn(|i| {
                        buffers[i]
                            .buffer
                            .ptr
                            .cast::<u8>()
                            .add(expert * (sizes[i] / experts))
                    });
                    packer.pack(sources, destinations, stream.raw)?;
                    // The next copy reuses device staging on this same stream,
                    // after this pack. Distinct pinned inputs stay alive for the
                    // entire group. LoadStream::drop drains on partial failure.
                }
            }
            // Only CPU reuse of pinned staging requires host completion.
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
        }
        Ok(Self {
            buffers,
            layer,
            experts,
            budget,
            nvfp4: None,
            format: ExpertFormat::of(catalog),
        })
    }

    /// Load the W4A4 planes for one layer and adopt its buffers.
    fn load_nvfp4(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        let (buffers, side) = nvfp4::Nvfp4Side::load(library, catalog, layer, available_device_bytes)?;
        let budget = nvfp4::Nvfp4Side::plan(library, catalog, layer)?;
        let (_, experts) = nvfp4::Nvfp4Side::rank_planes(layer, catalog)?;
        Ok(Self {
            buffers,
            layer,
            experts,
            budget,
            nvfp4: Some(side),
            format: ExpertFormat::Nvfp4,
        })
    }
    pub fn budget(&self) -> ExpertLoadBudget {
        self.budget
    }

    /// Bind prepared weight carriers after scratch binding and before argument validation.
    /// Returned raw slots borrow this owner and must not outlive it, including graph replay.
    pub fn bind(
        &self,
        kernel: &V41ExpertKernel<'_>,
        slots: &mut [*mut c_void; V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        ensure!(
            kernel.info().role == self.layer.role()
                && kernel.info().experts as usize == self.experts,
            "expert weights do not match kernel role"
        );
        ensure!(
            !slots[34].is_null() && !slots[37].is_null(),
            "bind initialized scratch before weights"
        );
        if let Some(side) = &self.nvfp4 {
            // W4A4: fused FC1 payload and scale plane, FC2 payload and scale
            // plane, then the resident per-expert alphas and activation scales.
            for (slot, pointer) in [
                (22, self.buffers[0].buffer.ptr),
                (23, self.buffers[1].buffer.ptr),
                (24, self.buffers[2].buffer.ptr),
                (25, self.buffers[3].buffer.ptr),
                (37, side.input_scales.buffer.ptr),
                (38, side.alphas.buffer.ptr),
                (39, side.down_alphas.buffer.ptr),
                (40, side.down_input_scales.buffer.ptr),
            ] {
                slots[slot] = pointer;
            }
            return Ok(());
        }
        let [w13, s13, w2, s2] = std::array::from_fn(|i| self.buffers[i].buffer.ptr);
        for (slot, pointer) in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
        ] {
            slots[slot] = pointer;
        }
        Ok(())
    }
}
