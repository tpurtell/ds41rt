use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41DsparkConfidence};
use std::ffi::c_void;

/// Stable per-wave confidence inputs/output; borrowed weights outlive its graph.
pub(crate) struct DsparkConfidence<'weights, 'library> {
    stream: LoadStream<'library>,
    hidden: DeviceAllocation<'library>,
    markov: DeviceAllocation<'library>,
    output: DeviceAllocation<'library>,
    kernel: V41DsparkConfidence<'library>,
    weights: &'weights DsparkWeights<'library>,
    graph: Option<(*mut c_void, usize)>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn confidence(
        &self,
        capacity: usize,
        available_device_bytes: usize,
    ) -> Result<DsparkConfidence<'_, 'library>> {
        let bytes = DsparkConfidence::device_bytes(capacity)?;
        ensure!(
            bytes <= available_device_bytes,
            "dSpark confidence exceeds device budget"
        );
        let library = self.library;
        let kernel = library.v41_dspark_confidence()?;
        // Resolve the final-stage checkpoint weight before allocating workspace.
        self.tensor("mtp.2.confidence_head.proj.weight")?;
        Ok(DsparkConfidence {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            hidden: DeviceAllocation::new(library, capacity * 10240)?,
            markov: DeviceAllocation::new(library, capacity * 512)?,
            output: DeviceAllocation::new(library, capacity * 4)?,
            kernel,
            weights: self,
            graph: None,
            capacity,
            ready_rows: None,
        })
    }
}
impl DsparkConfidence<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid confidence capacity"
        );
        Ok(capacity * (10240 + 512 + 4))
    }
    /// BF16 hidden [capacity,5120] and preceding-token Markov embedding
    /// [capacity,256], with identical row order. Never free or retain after drop.
    /// Finish producer writes before execute/replay; serialize reuse per owner.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.hidden.buffer, self.markov.buffer]
    }
    fn validate_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "confidence rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.stream.raw) }
    }
    pub(super) unsafe fn enqueue_on(&self, rows: usize, stream: *mut c_void) -> Result<()> {
        self.validate_rows(rows)?;
        unsafe {
            self.kernel.launch(
                self.hidden.buffer,
                self.markov.buffer,
                self.weights.tensor("mtp.2.confidence_head.proj.weight")?,
                self.output.buffer,
                rows,
                stream,
            )
        }
    }
    pub(super) fn storage(&self) -> [Ds41rtDeviceBuffer; 1] {
        [self.output.buffer]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Both owned inputs must be initialized for these rows and producer work
    /// must be complete; input writes must not race this execution.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<Ds41rtDeviceBuffer> {
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
        ensure!(self.graph.is_none(), "confidence graph already captured");
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
                    tracing::error!(%cleanup, "destroying failed confidence capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; rows must match the captured shape.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        let (graph, captured_rows) = self.graph.context("confidence graph is not captured")?;
        ensure!(
            rows == captured_rows,
            "confidence replay shape differs from capture"
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
    /// Raw FP32 scores, without sigmoid; borrowed until reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("confidence output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows * 4;
        Ok(output)
    }
}
impl Drop for DsparkConfidence<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining dSpark confidence execution");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying dSpark confidence graph");
            }
        }
    }
}
