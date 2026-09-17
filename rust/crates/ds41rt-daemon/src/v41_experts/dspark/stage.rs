//! One complete attention/mHC/FFN draft stage, captured on its expert stream.
use super::{DsparkAttentionWave, DsparkFfn, DsparkWeights, HcSublayer};
use crate::v41_dspark_cache::{DsparkWindow, WindowLease, WindowRead};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

pub(crate) struct DsparkStage<'weights, 'library> {
    ffn: DsparkFfn<'weights, 'library>,
    boundary: HcSublayer<'weights, 'library>,
    attention: DsparkAttentionWave<'weights, 'library>,
    library: &'library NativeLibrary,
    graph: Option<(*mut c_void, usize, u64)>,
    requests: u32,
    width: usize,
    ready: bool,
}
impl<'library> DsparkWeights<'library> {
    pub fn stage_bytes(&self, requests: u32) -> Result<usize> {
        let capacity = DsparkAttentionWave::projection_capacity_with_width(requests, self.draft_width)?;
        let library = self.library;
        let attention = DsparkAttentionWave::device_bytes_with_width(library, requests, self.draft_width)?;
        self.ffn_bytes(capacity)?
            .checked_add(HcSublayer::device_bytes(capacity as usize)?)
            .and_then(|v| v.checked_add(attention))
            .context("dSpark combined stage budget overflow")
    }
    /// Replaces the separate attention/mHC/FFN owners already in the wave budget.
    pub fn stage(
        &self,
        stage: usize,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkStage<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark stage");
        ensure!(
            self.stage_bytes(requests)? <= budget,
            "dSpark stage exceeds budget"
        );
        let capacity = DsparkAttentionWave::projection_capacity_with_width(requests, self.draft_width)?;
        let library = self.library;
        Ok(DsparkStage {
            ffn: self.ffn(stage, capacity, self.ffn_bytes(capacity)?)?,
            boundary: self.hc_sublayer(
                stage,
                true,
                capacity as usize,
                HcSublayer::device_bytes(capacity as usize)?,
            )?,
            attention: self.attention_wave(
                stage,
                requests,
                DsparkAttentionWave::device_bytes_with_width(library, requests, self.draft_width)?,
            )?,
            library,
            graph: None,
            requests,
            width: self.draft_width,
            ready: false,
        })
    }
}
impl DsparkStage<'_, '_> {
    #[cfg(test)]
    pub(super) fn expert_diagnostics(&self)->[Ds41rtDeviceBuffer;4] {self.ffn.expert_diagnostics()}

    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.boundary.inputs()
    }
    pub(super) fn invalidate(&mut self) {
        self.ready = false;
        self.boundary.invalidate();
        self.ffn.invalidate();
    }
    pub(super) fn prepare(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<WindowRead> {
        self.invalidate();
        ensure!(
            !requests.is_empty() && requests.len() <= self.requests as usize,
            "invalid dSpark stage batch"
        );
        self.attention.prepare(window, requests)
    }
    unsafe fn enqueue(&mut self, read: &WindowRead, requests: usize) -> Result<()> {
        unsafe { self.enqueue_on(read, requests, self.ffn.stream()) }
    }
    pub(super) fn output_storage(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.ffn.output_storage()
    }
    pub(super) fn upload(
        &mut self,
        read: &WindowRead,
        requests: &[(WindowLease, u64)],
    ) -> Result<()> {
        self.attention.upload(read, requests)
    }
    pub(super) unsafe fn upload_on(&mut self, read: &WindowRead,
        requests: &[(WindowLease, u64)], stream: *mut c_void) -> Result<()> {
        unsafe { self.attention.upload_on(read, requests, stream) }
    }
    /// The containing owner retains and drains the supplied stream through all children.
    pub(super) unsafe fn enqueue_on(
        &mut self,
        read: &WindowRead,
        requests: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests as u32 * self.width as u32;
        unsafe {
            self.boundary
                .enqueue_begin(rows as usize, Some(self.attention.input()), stream)?;
            self.attention.enqueue_on(read, requests as u32, stream)?;
            self.boundary
                .enqueue_finish(Some(self.attention.output_storage()), stream)?;
            let source = self.boundary.output_storage();
            let destination = self.ffn.inputs();
            self.library.copy_d2d_async(
                destination[0],
                source[0],
                rows as usize * 40960,
                stream,
            )?;
            self.library
                .copy_d2d_async(destination[1], source[1], rows as usize * 16, stream)?;
            self.ffn.enqueue_on(rows, stream)
        }
    }
    /// # Safety
    /// Initialize finite residual [requests,K,4,5120] and incoming pre [requests,K,4];
    /// complete producer writes and serialize all raw buffer use through completion.
    /// Committed windows must contain accepted main-model tokens on this device.
    pub unsafe fn execute(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let read = self.prepare(window, requests)?;
        self.attention.upload(&read, requests)?;
        let launched = unsafe { self.enqueue(&read, requests.len()) };
        let drained = self.ffn.synchronize();
        if let Err(error) = launched.and(drained) {
            self.invalidate();
            return Err(error);
        }
        unsafe {
            self.ffn.complete_replay(requests.len() as u32 * self.width as u32)?;
        }
        self.ready = true;
        self.output()
    }
    /// # Safety
    /// Same initialization contract as execute; captured ring storage is bound to
    /// this unique window owner and revalidated before every replay.
    pub unsafe fn capture(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<()> {
        self.invalidate();
        ensure!(self.graph.is_none(), "dSpark stage already captured");
        unsafe {
            self.execute(window, requests)?;
        }
        let read = self.prepare(window, requests)?;
        unsafe {
            self.library.cuda_graph_begin_capture(self.ffn.stream())?;
        }
        let launched = unsafe { self.enqueue(&read, requests.len()) };
        let captured = unsafe { self.library.cuda_graph_end_capture(self.ffn.stream()) };
        self.invalidate();
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, requests.len(), read.owner));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same contract as execute; request count and cache owner must match capture.
    pub unsafe fn replay(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let read = self.prepare(window, requests)?;
        let (graph, count, owner) = self.graph.context("dSpark stage not captured")?;
        ensure!(
            count == requests.len() && owner == read.owner,
            "dSpark stage capture binding differs"
        );
        self.attention.upload(&read, requests)?;
        let launched = unsafe { self.library.cuda_graph_launch(graph, self.ffn.stream()) };
        let drained = self.ffn.synchronize();
        launched.and(drained)?;
        unsafe {
            self.ffn.complete_replay(count as u32 * self.width as u32)?;
        }
        self.ready = true;
        self.output()
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        ensure!(self.ready, "dSpark stage output incomplete");
        self.ffn.output()
    }
}
impl Drop for DsparkStage<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.ffn.synchronize() {
            tracing::error!(%error,"draining dSpark stage");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark stage graph");
            }
        }
    }
}
