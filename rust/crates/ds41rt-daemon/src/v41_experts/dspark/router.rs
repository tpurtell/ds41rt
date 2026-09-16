//! Per-stage router scratch; writes directly into the expert wave's stable inputs.
use super::DsparkWeights;
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Router};
use std::ffi::c_void;

pub(crate) struct DsparkRouter<'weights, 'library> {
    // Keep the complete checkpoint owner borrowed while retaining tensor views.
    _weights: &'weights DsparkWeights<'library>,
    tensors: [Ds41rtDeviceBuffer; 2],
    kernel: V41Router<'library>,
    scores: DeviceAllocation<'library>,
    stage: usize,
    capacity: usize,
}
impl<'library> DsparkWeights<'library> {
    pub fn router(
        &self,
        stage: usize,
        capacity: usize,
        budget: usize,
    ) -> Result<DsparkRouter<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark router stage");
        let bytes = DsparkRouter::device_bytes(capacity)?;
        ensure!(bytes <= budget, "dSpark router exceeds budget");
        let tensors = [
            self.tensor(&format!("mtp.{stage}.ffn.gate.weight"))?,
            self.tensor(&format!("mtp.{stage}.ffn.gate.bias"))?,
        ];
        let library = self.library;
        Ok(DsparkRouter {
            _weights: self,
            tensors,
            kernel: library.v41_router()?,
            scores: DeviceAllocation::new(library, bytes)?,
            stage,
            capacity,
        })
    }
}
impl DsparkRouter<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid dSpark router capacity"
        );
        Ok(capacity * 128 * 4)
    }
    pub(super) fn matches_stage(&self, weights: &DsparkWeights<'_>, stage: usize) -> bool {
        self.stage == stage && std::ptr::eq(self._weights, weights)
    }
    pub(in crate::v41_experts) fn matches(
        &self,
        weights: &crate::v41_experts::ExpertWeights<'_>,
    ) -> bool {
        self._weights.full_expert(self.stage).is_some_and(|stage| std::ptr::eq(stage, weights))
    }
    /// Caller must drain the stream, including on launch failure, before releasing
    /// this borrow: the first kernel may already be using scores scratch.
    pub(in crate::v41_experts) unsafe fn enqueue(
        &mut self,
        inputs: [Ds41rtDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "dSpark router rows exceed capacity"
        );
        unsafe {
            self.kernel.launch(
                inputs[0],
                self.tensors[0],
                self.tensors[1],
                self.tensors[1],
                None,
                self.scores.buffer,
                inputs[1],
                inputs[2],
                rows,
                128,
                stream,
            )
        }
    }
}
