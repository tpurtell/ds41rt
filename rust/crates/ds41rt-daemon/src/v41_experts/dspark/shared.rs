//! Native shared-expert projections, packed scales and exclusive per-wave scratch.
use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

pub(super) fn packed_scale_bytes(library: &NativeLibrary) -> Result<usize> {
    let up = library.v41_fp8_matrix_info(16, 5120, 2304)?;
    let down = library.v41_fp8_matrix_info(16, 2304, 5120)?;
    usize::try_from((2 * up.packed_weight_scale_bytes + down.packed_weight_scale_bytes) * 3)
        .context("shared FFN packed scale budget overflow")
}
pub(super) fn pack_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &NativeRtxTensors<'a>,
) -> Result<[DeviceAllocation<'a>; 9]> {
    let up = library.v41_fp8_matrix_kernel(16, 5120, 2304)?;
    let down = library.v41_fp8_matrix_kernel(16, 2304, 5120)?;
    // The stream is declared after storage so error unwinding drains before free.
    let mut packed = Vec::with_capacity(9);
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for stage in 0..3 {
        for (name, kernel) in [("w1", &up), ("w3", &up), ("w2", &down)] {
            packed.push(DeviceAllocation::new(
                library,
                kernel.info().packed_weight_scale_bytes as usize,
            )?);
            unsafe {
                kernel.pack_scales(
                    tensors.get(&format!("mtp.{stage}.ffn.shared_experts.{name}.scale"))?,
                    packed.last().unwrap().buffer,
                    stream.raw,
                )?;
            }
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    packed
        .try_into()
        .ok()
        .context("expected nine shared FFN scale buffers")
}

pub(crate) struct DsparkSharedFfn<'weights, 'library> {
    weights: &'weights DsparkWeights<'library>,
    stage: usize,
    inner: crate::v41_shared_ffn::SharedFfn<'weights, 'library>,
}
impl<'library> DsparkWeights<'library> {
    pub fn shared_ffn(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkSharedFfn<'_, 'library>> {
        ensure!(stage < 3, "invalid shared FFN stage");
        Ok(DsparkSharedFfn {
            weights: self,
            stage,
            inner: crate::v41_shared_ffn::SharedFfn::new(
                self.library,
                &self.auxiliary,
                &format!("mtp.{stage}.ffn.shared_experts"),
                &self.shared_scales[stage * 3..stage * 3 + 3],
                capacity,
                budget,
            )?,
        })
    }
}
impl DsparkSharedFfn<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        crate::v41_shared_ffn::SharedFfn::device_bytes(library, capacity)
    }
    pub(super) fn matches_stage(&self, weights: &DsparkWeights<'_>, stage: usize) -> bool {
        self.stage == stage && std::ptr::eq(self.weights, weights)
    }
    pub(in crate::v41_experts) fn matches(
        &self,
        weights: &crate::v41_experts::ExpertWeights<'_>,
    ) -> bool {
        self.weights.full_expert(self.stage).is_some_and(|stage| std::ptr::eq(stage, weights))
    }
    /// Caller drains the stream before releasing the exclusive scratch borrow.
    pub(in crate::v41_experts) unsafe fn enqueue(
        &mut self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe { self.inner.enqueue(input, output, rows, stream) }
    }
}
