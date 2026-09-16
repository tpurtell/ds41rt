use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41VocabularyProjection};
use std::ffi::c_void;

/// One sequential draft position's stable token, embedding and logits buffers.
/// Returned views are valid until this owner is reused; preserve embeddings for
/// the final confidence projection before advancing to the next draft position.
pub(crate) struct DsparkMarkov<'weights, 'library> {
    stream: LoadStream<'library>,
    kernel: V41VocabularyProjection<'library>,
    _workspace: DeviceAllocation<'library>,
    tokens: DeviceAllocation<'library>,
    embedding: DeviceAllocation<'library>,
    logits: DeviceAllocation<'library>,
    weights: &'weights DsparkWeights<'library>,
    graph: Option<(*mut c_void, usize)>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn markov(
        &self,
        capacity: usize,
        available_device_bytes: usize,
    ) -> Result<DsparkMarkov<'_, 'library>> {
        let bytes = DsparkMarkov::device_bytes(capacity)?;
        ensure!(
            bytes <= available_device_bytes,
            "dSpark Markov exceeds device budget"
        );
        let library = self.library;
        self.tensor("mtp.2.markov_head.embed.weight")?;
        self.tensor("mtp.2.markov_head.head.weight")?;
        let workspace = DeviceAllocation::new(library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let kernel = unsafe { library.v41_dspark_markov(workspace.buffer)? };
        Ok(DsparkMarkov {
            kernel,
            _workspace: workspace,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            tokens: DeviceAllocation::new(library, capacity * 4)?,
            embedding: DeviceAllocation::new(library, capacity * 512)?,
            logits: DeviceAllocation::new(library, capacity * 129280 * 4)?,
            weights: self,
            graph: None,
            capacity,
            ready_rows: None,
        })
    }
}
impl DsparkMarkov<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=16).contains(&capacity),
            "Markov request capacity must be 1 through 16"
        );
        Ok(capacity * (4 + 512 + 129280 * 4) + V41VocabularyProjection::WORKSPACE_BYTES)
    }
    /// Preceding token IDs [capacity], u32, each less than 129280. Never free or
    /// retain after drop; finish producer writes before execute/replay.
    pub fn tokens(&self) -> Ds41rtDeviceBuffer {
        self.tokens.buffer
    }
    fn validate_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "markov rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.stream.raw) }
    }
    pub(super) unsafe fn enqueue_on(&self, rows: usize, stream: *mut c_void) -> Result<()> {
        self.validate_rows(rows)?;
        unsafe {
            self.stream.library.cuda_embedding_lookup_bf16_async(
                self.weights.tensor("mtp.2.markov_head.embed.weight")?,
                self.tokens.buffer,
                self.embedding.buffer,
                rows,
                129280,
                256,
                stream,
            )?;
            // BF16 checkpoint weights and embeddings are exact FP32 values;
            // accumulate and expose FP32 logits without a BF16 output rounding.
            self.kernel.launch(
                self.embedding.buffer,
                self.weights.tensor("mtp.2.markov_head.head.weight")?,
                self.logits.buffer,
                rows,
                stream,
            )
        }
    }
    pub(super) fn storage(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.embedding.buffer, self.logits.buffer]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Owned token IDs must be initialized and less than 129280; producer work
    /// must be complete; input writes must not race this execution.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.ready_rows = None;
        self.validate_rows(rows)?;
        unsafe {
            self.enqueue(rows)?;
        }
        self.synchronize()?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warms the launch before graph capture.
    pub unsafe fn capture(&mut self, rows: usize) -> Result<()> {
        ensure!(self.graph.is_none(), "markov graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        self.ready_rows = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.enqueue(rows) };
        let capture = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, capture) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed markov capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; rows must match the captured shape.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.ready_rows = None;
        let (graph, captured_rows) = self.graph.context("markov graph is not captured")?;
        ensure!(
            rows == captured_rows,
            "markov replay shape differs from capture"
        );
        unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)?;
        }
        self.synchronize()?;
        self.ready_rows = Some(rows);
        self.output()
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_rows = None;
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
    /// BF16 preceding-token embeddings and FP32 logits bias, in matching row
    /// order, borrowed until reuse/drop. No sampling or logits addition here.
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let rows = self.ready_rows.context("Markov output is not complete")?;
        let mut embedding = self.embedding.buffer;
        embedding.bytes = rows * 512;
        let mut logits = self.logits.buffer;
        logits.bytes = rows * 129280 * 4;
        Ok([embedding, logits])
    }
}
impl Drop for DsparkMarkov<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining dSpark markov execution");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying dSpark markov graph");
            }
        }
    }
}
