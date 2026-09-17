//! One complete dSpark FFN boundary on its owned expert wave stream.
use super::{DsparkRouter, DsparkSharedFfn, DsparkWeights, HcSublayer};
use super::expert_backend::DraftExperts;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

pub(crate) struct DsparkFfn<'weights, 'library> {
    // Drop drains this stream before any sibling workspace can be released.
    experts: DraftExperts<'weights, 'library>,
    boundary: HcSublayer<'weights, 'library>,
    router: DsparkRouter<'weights, 'library>,
    shared: DsparkSharedFfn<'weights, 'library>,
    device_bytes: usize,
    library: &'library NativeLibrary,
    graph: Option<(*mut c_void, u32)>,
}
impl<'library> DsparkWeights<'library> {
    pub fn ffn_bytes(&self, capacity: u32) -> Result<usize> {
        let library = self.library;
        let experts = self.expert_bytes(capacity)?;
        let hc = HcSublayer::device_bytes(capacity as usize)?;
        let router = DsparkRouter::device_bytes(capacity as usize)?;
        let shared = DsparkSharedFfn::device_bytes(library, capacity)?;
        [hc, router, shared]
            .into_iter()
            .try_fold(experts, |bytes, next| {
                bytes
                    .checked_add(next)
                    .context("dSpark FFN combined budget overflow")
            })
    }
    /// Replaces one stage's separate FFN mHC/router/shared/expert wave owners.
    /// These allocations are already included in DsparkBudget, not additional.
    pub fn ffn(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkFfn<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark FFN stage");
        let bytes = self.ffn_bytes(capacity)?;
        ensure!(bytes <= budget, "dSpark FFN exceeds device budget");
        let library = self.library;
        Ok(DsparkFfn {
            experts: self.expert_wave(stage, capacity)?,
            boundary: self.hc_sublayer(
                stage,
                false,
                capacity as usize,
                HcSublayer::device_bytes(capacity as usize)?,
            )?,
            router: self.router(
                stage,
                capacity as usize,
                DsparkRouter::device_bytes(capacity as usize)?,
            )?,
            shared: self.shared_ffn(
                stage,
                capacity,
                DsparkSharedFfn::device_bytes(library, capacity)?,
            )?,
            device_bytes: bytes,
            library,
            graph: None,
        })
    }
}
impl DsparkFfn<'_, '_> {
    #[cfg(test)]
    pub(super) fn expert_diagnostics(&self)->[Ds41rtDeviceBuffer;4] {
        let [hidden,ids,routing]=self.experts.inputs();
        [hidden,ids,routing,self.experts.output().expect("draft output")]
    }
    pub(super) fn stream(&self) -> *mut c_void {
        self.experts.stream()
    }
    pub(super) fn synchronize(&self) -> Result<()> {
        self.experts.synchronize()
    }
    pub(super) fn invalidate(&mut self) {
        self.boundary.invalidate();
    }
    /// The containing stage has drained its complete graph successfully.
    pub(super) unsafe fn complete_replay(&mut self, rows: u32) -> Result<[Ds41rtDeviceBuffer; 2]> {
        unsafe { self.boundary.complete_replay(rows as usize) }
    }

    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }
    /// BF16 residual [capacity,4,5120] and incoming FP32 pre-mix [capacity,4].
    /// The incoming mix is the preceding attention sublayer's generated pre.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.boundary.inputs()
    }

    /// # Safety
    /// Initialize both inputs with finite states and valid pre-mix coefficients on
    /// this device, finish producer writes, and serialize all reuse of borrowed views.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let launched = unsafe { self.enqueue(rows) };
        let drained = self.experts.synchronize();
        if let Err(error) = launched.and(drained) {
            self.boundary.invalidate();
            return Err(error);
        }
        unsafe { self.boundary.complete() }
    }
    /// Scratch owners and borrowed weights must live through stream completion.
    pub(super) unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.experts.stream()) }
    }
    pub(super) fn output_storage(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.boundary.output_storage()
    }
    /// The containing owner serializes and drains this supplied stream.
    pub(super) unsafe fn enqueue_on(&mut self, rows: u32, stream: *mut c_void) -> Result<()> {
        self.boundary.invalidate();
        let normalized = self.experts.inputs()[0];
        let result = self
            .experts
            .output()
            .context("dSpark FFN requires RTX output")?;
        unsafe {
            // Write normalization directly into the expert wave, avoiding D2D copies.
            self.boundary
                .enqueue_begin(rows as usize, Some(normalized), stream)?;
            self.experts
                .enqueue_draft_ffn_on(&mut self.router, &mut self.shared, rows, stream)?;
            self.boundary.enqueue_finish(Some(result), stream)
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; captures only owned addresses.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        ensure!(self.graph.is_none(), "dSpark FFN graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        self.boundary.invalidate();
        unsafe {
            self.library
                .cuda_graph_begin_capture(self.experts.stream())?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.library.cuda_graph_end_capture(self.experts.stream()) };
        self.boundary.invalidate();
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                    tracing::error!(%cleanup,"destroying failed dSpark FFN capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; rows must match capture.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.boundary.invalidate();
        let (graph, captured_rows) = self.graph.context("dSpark FFN graph was not captured")?;
        ensure!(
            rows == captured_rows,
            "dSpark FFN replay rows differ from capture"
        );
        let launched = unsafe { self.library.cuda_graph_launch(graph, self.experts.stream()) };
        let drained = self.experts.synchronize();
        launched.and(drained)?;
        unsafe { self.boundary.complete_replay(rows as usize) }
    }
    /// Completed residual streams and next pre-mix; invalid after a failed execute.
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.boundary.output()
    }
}
impl Drop for DsparkFfn<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.experts.synchronize() {
            tracing::error!(%error,"draining dSpark FFN boundary");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark FFN graph");
            }
        }
    }
}
