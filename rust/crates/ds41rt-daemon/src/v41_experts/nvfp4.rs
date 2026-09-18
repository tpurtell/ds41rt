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

/// Resident NVFP4 storage for one layer on one rank, embedded in
/// [`super::ExpertWeights`] so every existing wave, execution and service
/// path serves W4A4 without a second weights type.
pub(crate) struct Nvfp4Side<'a> {
    /// Per-expert FC1 and FC2 runtime alphas bound to slots 38/39.
    pub(crate) alphas: DeviceAllocation<'a>,
    pub(crate) down_alphas: DeviceAllocation<'a>,
    /// Per-expert activation-scale vectors. Uploaded once per layer, then
    /// bound to the kernel's `input_global_scale` (slot 37) and `global_scale`
    /// (slot 40) pointers so no per-launch copy is needed.
    pub(crate) input_scales: DeviceAllocation<'a>,
    pub(crate) down_input_scales: DeviceAllocation<'a>,
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

/// Per-expert fused FC1 payload, FC1 scale plane, FC2 payload and FC2 scale
/// plane extents for one rank's intermediate width.
pub(crate) fn plane_sizes(intermediate: usize) -> [usize; 4] {
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

impl<'a> Nvfp4Side<'a> {
    /// Per-rank intermediate width and routed-expert count for a layer.
    pub(crate) fn rank_planes(layer: ExpertLayer, catalog: &OfficialV41Catalog) -> Result<(usize, usize)> {
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

    pub(crate) fn plan(
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
            // Four per-expert f32 vectors: alphas, down alphas and the two
            // activation-scale planes.
            .checked_add(experts * 4 * 4)
            .context("NVFP4 scalar vector overflow")?;
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

    pub(crate) fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<([DeviceAllocation<'a>; 4], Self)> {
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
        Ok((
            buffers,
            Self {
                alphas,
                down_alphas,
                input_scales: input_scales_device,
                down_input_scales: down_input_scales_device,
            },
        ))
    }

}
