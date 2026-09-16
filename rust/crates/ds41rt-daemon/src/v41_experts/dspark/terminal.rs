use super::{DsparkConfidence, DsparkMarkov, DsparkWeights};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::VocabularyHead;
use anyhow::{ensure, Context, Result};
use ds41rt_core::DsparkRng;
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41DraftStep, V41Hc, V41VocabularyProjection};
use std::ffi::c_void;
mod distributed;
pub(crate) use distributed::DistributedDsparkTerminal;

struct LocalHead<'w, 'a> {
    kernel: V41VocabularyProjection<'a>,
    _workspace: DeviceAllocation<'a>,
    weights: &'w VocabularyHead<'a>,
}

/// Width-dependent Markov/sample positions followed by raw confidence, on one
/// stream. All draft tensors use [position,request,...] order at live row count.
pub(crate) struct DsparkTerminal<'weights, 'library> {
    stream: LoadStream<'library>,
    markov: DsparkMarkov<'weights, 'library>,
    confidence: DsparkConfidence<'weights, 'library>,
    sample: V41DraftStep<'library>,
    hc: V41Hc<'library>,
    residual: DeviceAllocation<'library>,
    pre_mix: DeviceAllocation<'library>,
    local_head: Option<LocalHead<'weights, 'library>>,
    normalized: DeviceAllocation<'library>,
    weights: &'weights DsparkWeights<'library>,
    shared_logits: DeviceAllocation<'library>,
    adjusted_logits: DeviceAllocation<'library>,
    rng: DeviceAllocation<'library>,
    sampling_requests: Option<usize>,
    sampling_staging: HostAllocation<'library>,
    temperatures: DeviceAllocation<'library>,
    tokens: DeviceAllocation<'library>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    ready_requests: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    /// Collapse residual streams, normalize and project through the borrowed shared
    /// head, then apply Markov correction, sampling and confidence.
    pub fn terminal<'weights>(
        &'weights self,
        head: &'weights VocabularyHead<'library>,
        capacity: usize,
        budget: usize,
    ) -> Result<DsparkTerminal<'weights, 'library>> {
        self.terminal_storage(Some(head), capacity, budget)
    }
    fn terminal_storage<'weights>(
        &'weights self, head: Option<&'weights VocabularyHead<'library>>,
        capacity: usize, budget: usize,
    ) -> Result<DsparkTerminal<'weights, 'library>> {
        let bytes = DsparkTerminal::device_bytes_with_width(capacity, self.draft_width)?
            - if head.is_none() { V41VocabularyProjection::WORKSPACE_BYTES } else { 0 };
        ensure!(bytes <= budget, "dSpark terminal exceeds budget");
        let library = self.library;
        self.tensor("mtp.2.norm.weight")?;
        let local_head = head.map(|weights| -> Result<_> {
            weights.weight()?;
            let workspace = DeviceAllocation::new(library, V41VocabularyProjection::WORKSPACE_BYTES)?;
            let kernel = unsafe { library.v41_vocabulary_head(workspace.buffer)? };
            Ok(LocalHead { kernel, _workspace: workspace, weights })
        }).transpose()?;
        Ok(DsparkTerminal {
            local_head,
            normalized: DeviceAllocation::new(library, capacity * self.draft_width * 10240)?,
            weights: self,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            markov: self.markov(capacity, DsparkMarkov::device_bytes(capacity)?)?,
            confidence: self
                .confidence(capacity * self.draft_width, DsparkConfidence::device_bytes(capacity * self.draft_width)?)?,
            sample: library.v41_draft_step()?,
            hc: library.v41_hc()?,
            residual: DeviceAllocation::new(library, capacity * self.draft_width * 40960)?,
            pre_mix: DeviceAllocation::new(library, capacity * self.draft_width * 16)?,
            shared_logits: DeviceAllocation::new(library, capacity * self.draft_width * 129280 * 4)?,
            adjusted_logits: DeviceAllocation::new(library, capacity * self.draft_width * 129280 * 4)?,
            rng: DeviceAllocation::new(library, capacity * 16)?,
            sampling_requests: None,
            sampling_staging: HostAllocation::new(library, capacity * 20)?,
            temperatures: DeviceAllocation::new(library, capacity * 4)?,
            tokens: DeviceAllocation::new(library, capacity * (self.draft_width + 1) * 4)?,
            capacity,
            graph: None,
            ready_requests: None,
        })
    }
}
impl DsparkTerminal<'_, '_> {
    pub fn additional_bytes(capacity: usize) -> Result<usize> {
        Self::additional_bytes_with_width(capacity, 5)
    }
    pub fn additional_bytes_with_width(capacity: usize, width: usize) -> Result<usize> {
        ensure!(matches!(width, 5 | 7), "draft width must be five or seven");
        ensure!(
            (1..=16).contains(&capacity),
            "terminal request capacity must be 1 through 16"
        );
        Ok(
            capacity * (width * 129280 * 4 * 2 + (width + 2) * 4 + 16 + width * 10240 + width * (40960 + 16))
                + V41VocabularyProjection::WORKSPACE_BYTES,
        )
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Self::device_bytes_with_width(capacity, 5)
    }
    pub fn device_bytes_with_width(capacity: usize, width: usize) -> Result<usize> {
        Ok(Self::additional_bytes_with_width(capacity, width)?
            + DsparkMarkov::device_bytes(capacity)?
            + DsparkConfidence::device_bytes(capacity * width)?)
    }
    /// Stable inputs: BF16 residual [K,R,4,5120], FP32 incoming pre-mix [K,R,4]
    /// and anchor IDs [R]. Sampling is set via `prepare_sampling`.
    /// Use live R densely, without capacity padding between positions. Never free
    /// or retain after drop; finish all producer writes before execute/replay.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        let mut anchors = self.tokens.buffer;
        anchors.bytes = self.capacity * 4;
        [self.residual.buffer, self.pre_mix.buffer, anchors]
    }
    /// Atomically admit a batch's RNG ranges before reserving any. Once reserved,
    /// failures/cancellation consume them; replay intentionally retains the same
    /// draws until this method prepares a new attempt. Temperatures are validated
    /// here and uploaded with request-owned seed/range metadata after prior work.
    pub fn prepare_sampling(
        &mut self,
        rngs: &mut [&mut DsparkRng],
        temperatures: &[f32],
    ) -> Result<()> {
        self.synchronize()?;
        self.stage_sampling(rngs, temperatures)?;
        let bytes = self.sampling_staging.bytes_mut();
        let uploaded = (|| {
            self.stream.library.copy_h2d(self.rng.buffer, &bytes[..rngs.len() * 16])?;
            self.stream.library.copy_h2d(self.temperatures.buffer,
                &bytes[self.capacity * 16..self.capacity * 16 + rngs.len() * 4])
        })();
        if uploaded.is_err() { self.sampling_requests = None; }
        uploaded
    }
    /// CPU-only reservation. The containing chain excludes pending users of staging.
    pub(super) fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        self.ready_requests = None;
        self.sampling_requests = None;
        ensure!(
            !rngs.is_empty() && rngs.len() <= self.capacity && rngs.len() == temperatures.len(),
            "invalid terminal sampling request count"
        );
        ensure!(
            temperatures.iter().all(|t| t.is_finite() && *t >= 0.0),
            "invalid draft temperature"
        );
        ensure!(
            rngs.iter().all(|rng| rng.can_reserve_width(self.weights.draft_width)),
            "dSpark RNG exhausted"
        );
        let bytes = self.sampling_staging.bytes_mut();
        for (i, rng) in rngs.iter_mut().enumerate() {
            let reservation = rng.reserve_width(self.weights.draft_width).context("dSpark RNG exhausted")?;
            bytes[i * 16..i * 16 + 8].copy_from_slice(&reservation.seed.to_ne_bytes());
            bytes[i * 16 + 8..i * 16 + 16].copy_from_slice(&reservation.first_subsequence.to_ne_bytes());
        }
        for (i, temperature) in temperatures.iter().enumerate() {
            let offset = self.capacity * 16 + i * 4;
            bytes[offset..offset + 4].copy_from_slice(&temperature.to_ne_bytes());
        }
        self.sampling_requests = Some(rngs.len());
        Ok(())
    }
    /// The chain retains this pinned staging until its stream completes.
    pub(super) unsafe fn upload_sampling_on(&mut self, stream: *mut c_void) -> Result<()> {
        let count = self.sampling_requests.context("draft sampling not staged")?;
        let bytes = self.sampling_staging.bytes_mut();
        unsafe {
            self.stream.library.copy_h2d_async(self.rng.buffer, &bytes[..count * 16], stream)?;
            self.stream.library.copy_h2d_async(self.temperatures.buffer,
                &bytes[self.capacity * 16..self.capacity * 16 + count * 4], stream)
        }
    }
    fn slice(
        buffer: Ds41rtDeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            offset
                .checked_add(bytes)
                .context("terminal slice overflow")?
                <= buffer.bytes,
            "terminal slice exceeds allocation"
        );
        Ok(Ds41rtDeviceBuffer {
            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
            bytes,
            ..buffer
        })
    }
    pub(super) fn validate_sampling(&self, requests: usize) -> Result<()> {
        ensure!(
            requests > 0 && requests <= self.capacity,
            "terminal requests exceed capacity"
        );
        ensure!(
            self.sampling_requests == Some(requests),
            "sampling state does not match live requests"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, requests: usize) -> Result<()> {
        unsafe { self.enqueue_on(requests, self.stream.raw) }
    }
    pub(super) unsafe fn enqueue_on(&self, requests: usize, stream: *mut c_void) -> Result<()> {
        self.validate_sampling(requests)?;
        let head = self.local_head.as_ref().context("terminal requires external vocabulary projection")?;
        unsafe {
            self.enqueue_normalize_on(requests, stream)?;
            head.kernel.launch(self.normalized.buffer, head.weights.weight()?,
                self.shared_logits.buffer, requests * self.weights.draft_width, stream)?;
            self.enqueue_sampling_on(requests, stream)
        }
    }
    unsafe fn enqueue_normalize_on(&self, requests: usize, stream: *mut c_void) -> Result<()> {
        self.validate_sampling(requests)?;
        let library = self.stream.library;
        unsafe {
            self.hc.pre(
                self.residual.buffer,
                self.pre_mix.buffer,
                self.confidence.inputs()[0],
                requests * self.weights.draft_width,
                stream,
            )?;
            // This RNE variant matches the reference's one final BF16 rounding.
            library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.confidence.inputs()[0],
                self.weights.tensor("mtp.2.norm.weight")?,
                self.normalized.buffer,
                (requests * self.weights.draft_width) as i32,
                5120,
                1e-20,
                stream,
            )?;
        }
        Ok(())
    }
    unsafe fn enqueue_sampling_on(&self, requests: usize, stream: *mut c_void) -> Result<()> {
        self.validate_sampling(requests)?;
        let library = self.stream.library;
        let row_bytes = requests * 129280 * 4;
        unsafe {
            library.copy_d2d_async(
                self.markov.tokens(),
                self.tokens.buffer,
                requests * 4,
                stream,
            )?;
            for position in 0..self.weights.draft_width {
                self.markov.enqueue_on(requests, stream)?;
                let [embedding, bias] = self.markov.storage();
                let confidence_embedding = Self::slice(
                    self.confidence.inputs()[1],
                    position * requests * 512,
                    requests * 512,
                )?;
                library.copy_d2d_async(confidence_embedding, embedding, requests * 512, stream)?;
                let next = Self::slice(
                    self.tokens.buffer,
                    (position + 1) * requests * 4,
                    requests * 4,
                )?;
                self.sample.launch(
                    Self::slice(self.shared_logits.buffer, position * row_bytes, row_bytes)?,
                    bias,
                    self.rng.buffer,
                    self.temperatures.buffer,
                    Self::slice(self.adjusted_logits.buffer, position * row_bytes, row_bytes)?,
                    next,
                    requests,
                    position,
                    stream,
                )?;
                if position + 1 < self.weights.draft_width {
                    library.copy_d2d_async(self.markov.tokens(), next, requests * 4, stream)?;
                }
            }
            self.confidence.enqueue_on(requests * self.weights.draft_width, stream)?;
        }
        Ok(())
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Inputs must satisfy `inputs()` layout with initialized BF16 residuals and
    /// FP32 incoming pre-mix coefficients,
    /// valid anchors <129280, finite shared+Markov logits and
    /// sampling state prepared for these requests. Producers must be complete;
    /// no input writes may race execution.
    pub unsafe fn execute(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready_requests = None;
        unsafe {
            self.enqueue(requests)?;
        }
        self.synchronize()?;
        self.ready_requests = Some(requests);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warms the complete sequence before capture.
    pub unsafe fn capture(&mut self, requests: usize) -> Result<()> {
        ensure!(self.graph.is_none(), "terminal graph already captured");
        unsafe {
            self.execute(requests)?;
        }
        self.ready_requests = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.enqueue(requests) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, requests));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed terminal capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; prepare new sampling state for a fresh attempt.
    pub unsafe fn replay(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready_requests = None;
        ensure!(
            self.sampling_requests == Some(requests),
            "sampling state does not match replay requests"
        );
        let (graph, captured) = self.graph.context("terminal graph is not captured")?;
        ensure!(
            requests == captured,
            "terminal replay shape differs from capture"
        );
        unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)?;
        }
        self.synchronize()?;
        self.ready_requests = Some(requests);
        self.output()
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_requests = None;
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
    /// Tokens [K+1,R] (anchor then K drafts), corrected raw logits [K,R,V],
    /// raw confidence [K,R]. Borrowed until reuse/drop; no history is committed.
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        let requests = self
            .ready_requests
            .context("terminal output is not complete")?;
        self.output_storage(requests)
    }
    pub(super) fn output_storage(&self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.validate_sampling(requests)?;
        Ok([
            Self::slice(self.tokens.buffer, 0, requests * (self.weights.draft_width + 1) * 4)?,
            Self::slice(self.adjusted_logits.buffer, 0, requests * self.weights.draft_width * 129280 * 4)?,
            Self::slice(self.confidence.storage()[0], 0, requests * self.weights.draft_width * 4)?,
        ])
    }
}
impl Drop for DsparkTerminal<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining dSpark terminal");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying dSpark terminal graph");
            }
        }
    }
}
