//! W4A4 (ModelOpt NVFP4) routed-expert residency.
//!
//! The b12x nvfp4 kernel consumes the checkpoint payload unchanged in the
//! kernel-native `[up(w3); gate(w1)]` row order plus `[down(w2)]`, an
//! F8_128x4-swizzled E4M3 K16 scale plane per fused projection, and four
//! per-expert FP32 launch vectors. Loading therefore avoids any weight
//! repack: each expert plane is staged contiguously and lands with one
//! device-to-device copy, the scale planes are swizzled on device, and the
//! scalar vectors are computed on the host from the checkpoint's per-tensor
//! `weight_scale_2` and `input_scale`.
use super::{ExpertLayer, ExpertLoadBudget, EXPERT_READ_LANES};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;

/// Staging slot order produced by `V41Nvfp4Staging`.
const SLOT_UP_WEIGHT: usize = 0;
const SLOT_GATE_WEIGHT: usize = 1;
const SLOT_DOWN_WEIGHT: usize = 2;
const SLOT_UP_SCALE: usize = 3;
const SLOT_GATE_SCALE: usize = 4;
const SLOT_DOWN_SCALE: usize = 5;
const SLOT_SCALARS: usize = 6;
const SLOT_W1_WEIGHT_SCALE_2: usize = 6;
const SLOT_W3_WEIGHT_SCALE_2: usize = 7;
const SLOT_W2_WEIGHT_SCALE_2: usize = 8;
const SLOT_W1_INPUT_SCALE: usize = 9;
const SLOT_W3_INPUT_SCALE: usize = 10;
const SLOT_W2_INPUT_SCALE: usize = 11;
const HIDDEN: usize = 5120;

/// Resident NVFP4 routed expert weights for one layer on one rank.
pub(crate) struct Nvfp4Weights<'a> {
    /// b_w13, sfb_w13, b_down, sfb_down (experts concatenated).
    buffers: [DeviceAllocation<'a>; 4],
    /// Per-expert FC1 and FC2 runtime alphas bound to slots 38/39.
    alphas: DeviceAllocation<'a>,
    down_alphas: DeviceAllocation<'a>,
    /// Per-expert activation-scale vectors. Uploaded once per layer, then
    /// bound to the kernel's `input_global_scale` (slot 37) and `global_scale`
    /// (slot 40) pointers so no per-launch copy is needed.
    input_scales: DeviceAllocation<'a>,
    down_input_scales: DeviceAllocation<'a>,
    layer: ExpertLayer,
    experts: usize,
    budget: ExpertLoadBudget,
    intermediate: usize,
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

/// Per-expert fused FC1 payload, FC1 scale plane, FC2 payload and FC2 scale
/// plane extents for one rank's intermediate width.
fn plane_sizes(intermediate: usize) -> [usize; 4] {
    let hidden = HIDDEN;
    let fc1_rows = 2 * intermediate;
    let fc1_cols = hidden / 16;
    let down_cols = intermediate / 16;
    [
        fc1_rows * (hidden / 2),
        align_up(fc1_rows, 128) * align_up(fc1_cols, 4),
        hidden * (intermediate / 2),
        align_up(hidden, 128) * align_up(down_cols, 4),
    ]
}

impl<'a> Nvfp4Weights<'a> {
    fn rank_planes(layer: ExpertLayer, catalog: &OfficialV41Catalog) -> Result<(usize, usize)> {
        let text = catalog.config().text();
        let intermediate = match layer {
            ExpertLayer::Backbone { .. } => text.moe_intermediate_size / 4,
            ExpertLayer::BackboneTp2 { .. } => text.moe_intermediate_size / 2,
            ExpertLayer::BackboneFull { .. } => text.moe_intermediate_size,
            _ => anyhow::bail!("NVFP4 covers backbone routed experts only"),
        };
        ensure!(
            intermediate % 32 == 0,
            "NVFP4 rank intermediate must be a multiple of 32"
        );
        Ok((intermediate, text.n_routed_experts))
    }

    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        let (intermediate, experts) = Self::rank_planes(layer, catalog)?;
        let staging = catalog.nvfp4_expert_staging(layer.expert(0))?;
        let sizes = plane_sizes(intermediate);
        let resident_bytes = sizes
            .iter()
            .try_fold(0usize, |sum, size| {
                sum.checked_add(size.checked_mul(experts).context("NVFP4 resident overflow")?)
                    .context("NVFP4 resident byte overflow")
            })?
            .checked_add(experts * 4 * 2)
            .context("NVFP4 alpha vector overflow")?;
        Ok(ExpertLoadBudget {
            resident_bytes,
            device_staging_bytes: staging.staging_bytes(),
            pinned_host_bytes: staging
                .staging_bytes()
                .checked_mul(EXPERT_READ_LANES)
                .context("NVFP4 pinned staging overflow")?,
            read_scratch_bytes: staging
                .minimum_read_scratch_bytes()
                .checked_mul(64 * EXPERT_READ_LANES)
                .context("NVFP4 read scratch overflow")?,
        })
    }

    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        let (intermediate, experts) = Self::rank_planes(layer, catalog)?;
        let budget = Self::plan(library, catalog, layer)?;
        ensure!(
            budget.peak_device_bytes()? <= available_device_bytes,
            "NVFP4 expert layer needs {} device bytes including staging, budget is {available_device_bytes}",
            budget.peak_device_bytes()?
        );
        let sizes = plane_sizes(intermediate);
        let mut owned = Vec::with_capacity(4);
        for size in sizes {
            owned.push(DeviceAllocation::new(
                library,
                size.checked_mul(experts).context("NVFP4 slab overflow")?,
            )?);
        }
        let buffers: [DeviceAllocation<'a>; 4] =
            owned.try_into().ok().expect("four NVFP4 slabs");
        let alphas = DeviceAllocation::new(library, experts * 4)?;
        let down_alphas = DeviceAllocation::new(library, experts * 4)?;
        let input_scales_device = DeviceAllocation::new(library, experts * 4)?;
        let down_input_scales_device = DeviceAllocation::new(library, experts * 4)?;
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
        let mut input_scales = vec![0f32; experts];
        let mut down_input_scales = vec![0f32; experts];
        let mut alpha_values = vec![0f32; experts];
        let mut down_alpha_values = vec![0f32; experts];
        for expert in 0..EXPERT_READ_LANES.min(experts) {
            catalog.nvfp4_expert_staging(layer.expert(expert))?.prefetch()?;
        }
        for first in (0..experts).step_by(EXPERT_READ_LANES) {
            let end = (first + EXPERT_READ_LANES).min(experts);
            for future in end..(end + EXPERT_READ_LANES).min(experts) {
                catalog
                    .nvfp4_expert_staging(layer.expert(future))?
                    .prefetch()?;
            }
            let plans = (first..end)
                .map(|expert| catalog.nvfp4_expert_staging(layer.expert(expert)))
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
                        .map_err(|_| anyhow::anyhow!("NVFP4 expert read thread panicked"))??;
                }
                Ok(())
            })?;
            for (offset, (plan, host)) in plans.iter().zip(&hosts).enumerate() {
                let expert = first + offset;
                ensure!(
                    plan.w13_contiguous() && plan.weight_scale13_contiguous(),
                    "NVFP4 staging must keep the fused FC1 planes contiguous"
                );
                let ranges = plan.tensor_ranges();
                let staged = host.bytes();
                // Per-rank intermediate from the FC1 payload plane.
                let down_cols = intermediate / 16;
                unsafe {
                    library.copy_host_buffer_h2d_async(
                        device_staging.buffer,
                        host.buffer,
                        plan.staging_bytes(),
                        stream.raw,
                    )?;
                    // Contiguous staging span starting at a slot, with an
                    // explicit byte length for adjacent-pair views.
                    let staging_span = |start: usize, bytes: usize| -> Ds41rtDeviceBuffer {
                        let mut buffer = device_staging.buffer;
                        buffer.ptr = buffer.ptr.cast::<u8>().add(start).cast();
                        buffer.bytes = bytes;
                        buffer
                    };
                    let destination = |buffer: Ds41rtDeviceBuffer,
                                        stride: usize|
                     -> Ds41rtDeviceBuffer {
                        let mut out = buffer;
                        out.ptr = out.ptr.cast::<u8>().add(expert * stride).cast();
                        out.bytes = stride;
                        out
                    };
                    // Fused FC1 payload: [up(w3); gate(w1)] staged adjacently.
                    let fc1_bytes = ranges[SLOT_GATE_WEIGHT].end - ranges[SLOT_UP_WEIGHT].start;
                    library.copy_d2d_async(
                        destination(buffers[0].buffer, sizes[0]),
                        staging_span(ranges[SLOT_UP_WEIGHT].start, fc1_bytes),
                        fc1_bytes,
                        stream.raw,
                    )?;
                    let down_bytes = ranges[SLOT_DOWN_WEIGHT].len();
                    library.copy_d2d_async(
                        destination(buffers[2].buffer, sizes[2]),
                        staging_span(ranges[SLOT_DOWN_WEIGHT].start, down_bytes),
                        down_bytes,
                        stream.raw,
                    )?;
                    // Fused FC1 scale plane: swizzle the concatenated pair.
                    let fc1_scale_bytes =
                        ranges[SLOT_GATE_SCALE].end - ranges[SLOT_UP_SCALE].start;
                    let fc1_scale_rows = 2 * intermediate;
                    let fc1_scale_cols = HIDDEN / 16;
                    library.cuda_nvfp4_swizzle_scale_async(
                        staging_span(ranges[SLOT_UP_SCALE].start, fc1_scale_bytes),
                        destination(buffers[1].buffer, sizes[1]),
                        fc1_scale_rows,
                        fc1_scale_cols,
                        stream.raw,
                    )?;
                    let down_scale_bytes = ranges[SLOT_DOWN_SCALE].len();
                    library.cuda_nvfp4_swizzle_scale_async(
                        staging_span(ranges[SLOT_DOWN_SCALE].start, down_scale_bytes),
                        destination(buffers[3].buffer, sizes[3]),
                        HIDDEN,
                        down_cols,
                        stream.raw,
                    )?;
                }
                let scalar = |slot: usize| -> f32 {
                    f32::from_le_bytes(
                        staged[ranges[slot].clone()]
                            .try_into()
                            .expect("four scalar bytes"),
                    )
                };
                let w1_weight_scale_2 = scalar(SLOT_W1_WEIGHT_SCALE_2);
                let w3_weight_scale_2 = scalar(SLOT_W3_WEIGHT_SCALE_2);
                let w2_weight_scale_2 = scalar(SLOT_W2_WEIGHT_SCALE_2);
                let w1_input_scale = scalar(SLOT_W1_INPUT_SCALE);
                let w3_input_scale = scalar(SLOT_W3_INPUT_SCALE);
                let w2_input_scale = scalar(SLOT_W2_INPUT_SCALE);
                // The fused FC1 plane carries one alpha, so gate and up must
                // agree. Fail loudly rather than silently scaling one half.
                ensure!(
                    w1_weight_scale_2.to_bits() == w3_weight_scale_2.to_bits()
                        && w1_input_scale.to_bits() == w3_input_scale.to_bits(),
                    "NVFP4 expert {expert} has asymmetric gate/up scales"
                );
                ensure!(
                    w1_input_scale > 0.0 && w2_input_scale > 0.0,
                    "NVFP4 expert {expert} has a non-positive activation scale"
                );
                input_scales[expert] = 1.0 / w1_input_scale;
                down_input_scales[expert] = 1.0 / w2_input_scale;
                alpha_values[expert] = w1_weight_scale_2 * w1_input_scale;
                down_alpha_values[expert] = w2_weight_scale_2 * w2_input_scale;
                let _ = SLOT_SCALARS;
            }
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
        }
        // Per-expert vectors are tiny; upload once per layer on the load stream.
        {
            let mut host = HostAllocation::new(library, experts * 4)?;
            let upload = |values: &[f32],
                          destination: Ds41rtDeviceBuffer,
                          host: &mut HostAllocation<'_>|
             -> Result<()> {
                for (index, value) in values.iter().enumerate() {
                    host.bytes_mut()[index * 4..index * 4 + 4]
                        .copy_from_slice(&value.to_le_bytes());
                }
                unsafe {
                    library.copy_host_buffer_h2d_async(
                        destination,
                        host.buffer,
                        experts * 4,
                        stream.raw,
                    )
                }
            };
            upload(&alpha_values, alphas.buffer, &mut host)?;
            upload(&down_alpha_values, down_alphas.buffer, &mut host)?;
            upload(&input_scales, input_scales_device.buffer, &mut host)?;
            upload(&down_input_scales, down_input_scales_device.buffer, &mut host)?;
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
        }
        Ok(Self {
            buffers,
            alphas,
            down_alphas,
            input_scales: input_scales_device,
            down_input_scales: down_input_scales_device,
            layer,
            experts,
            budget,
            intermediate,
        })
    }

    pub fn budget(&self) -> ExpertLoadBudget {
        self.budget
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate
    }

    pub fn layer(&self) -> ExpertLayer {
        self.layer
    }

    /// Bind the resident planes, runtime alphas and per-expert activation
    /// scales. Slots 37/40 point at this layer's uploaded vectors instead of
    /// the shared scratch arrays, so the kernel reads the right values with no
    /// per-launch copy.
    pub fn bind(
        &self,
        kernel: &ds41rt_ffi::V41ExpertKernel<'_>,
        slots: &mut [*mut std::ffi::c_void; ds41rt_ffi::V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        ensure!(
            kernel.info().experts as usize == self.experts,
            "NVFP4 weights do not match kernel expert count"
        );
        // Engine slots: 22 b_w13, 23 sfb_w13 (the bridge aliases it for the
        // gate view), 24 b_down, 25 sfb_down, 37/40 activation scales,
        // 38/39 runtime alphas. The scratch bind still runs first so the
        // kernel-owned slots are valid.
        for (slot, pointer) in [
            (22, self.buffers[0].buffer.ptr),
            (23, self.buffers[1].buffer.ptr),
            (24, self.buffers[2].buffer.ptr),
            (25, self.buffers[3].buffer.ptr),
            (37, self.input_scales.buffer.ptr),
            (38, self.alphas.buffer.ptr),
            (39, self.down_alphas.buffer.ptr),
            (40, self.down_input_scales.buffer.ptr),
        ] {
            slots[slot] = pointer;
        }
        Ok(())
    }
}
