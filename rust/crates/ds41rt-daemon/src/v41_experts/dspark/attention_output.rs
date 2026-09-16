//! Inverse RoPE -> grouped FP8 wo_a -> native FP8 wo_b on one owned stream.
use super::{DsparkProjection, DsparkWeights, ProjectionKind};
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Plan,
};
use std::ffi::c_void;
pub(crate) struct DsparkAttentionOutput<'weights, 'library> {
    stream: LoadStream<'library>,
    grouped: V41Fp8Plan<'library>,
    grouped_scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    scales: Ds41rtDeviceBuffer,
    projection: DsparkProjection<'weights, 'library>,
    weight: Ds41rtDeviceBuffer,
    input: DeviceAllocation<'library>,
    frequencies: DeviceAllocation<'library>,
    capacity: u32,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn attention_output(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkAttentionOutput<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark attention output stage");
        let library = self.library;
        ensure!(
            DsparkAttentionOutput::device_bytes(library, capacity)? <= budget,
            "dSpark attention output exceeds budget"
        );
        let grouped = library.v41_fp8_matrix_plan(capacity, 32768, 8192)?;
        let grouped_scratch = DeviceAllocation::new(library, grouped.info().scratch_bytes as usize)?;
        let alpha = DeviceAllocation::new(library, 4)?;
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let initialized = unsafe { grouped.initialize_scratch(grouped_scratch.buffer, alpha.buffer, stream.raw) };
        initialized.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        let projection = self.projection(
            ProjectionKind::OutputB(stage),
            capacity,
            DsparkProjection::device_bytes(library, ProjectionKind::OutputB(stage), capacity)?,
        )?;
        Ok(DsparkAttentionOutput {
            stream,
            grouped,
            grouped_scratch,
            alpha,
            scales: self.grouped_output_scales[stage].buffer,
            projection,
            weight: self.auxiliary.get(&format!("mtp.{stage}.attn.wo_a.weight"))?,
            input: DeviceAllocation::new(library, capacity as usize * 65536)?,
            frequencies: DeviceAllocation::new(library, capacity as usize * 256)?,
            capacity,
            graph: None,
            ready: None,
        })
    }
}
impl DsparkAttentionOutput<'_, '_> {
    // Extra storage beyond the already-budgeted OutputB projection owner.
    pub fn additional_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid dSpark attention output capacity"
        );
        Ok(library.v41_fp8_matrix_plan_info(capacity, 32768, 8192)?.scratch_bytes as usize + 4 + capacity as usize * (65536 + 256))
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        DsparkProjection::device_bytes(library, ProjectionKind::OutputB(0), capacity)?
            .checked_add(Self::additional_bytes(library, capacity)?)
            .context("dSpark attention output storage overflow")
    }
    pub(super) fn output_storage(&self)->Ds41rtDeviceBuffer {self.projection.output_storage()}
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input.buffer
    }
    pub fn frequencies(&self) -> Ds41rtDeviceBuffer {
        self.frequencies.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Inputs/frequencies must be initialized on this device. A containing owner
    /// must drain the supplied stream before ending the exclusive borrow.
    pub(super) unsafe fn enqueue(&mut self, rows: u32, stream: *mut c_void) -> Result<()> {
        self.ready = None;
        ensure!(
            (1..=self.capacity).contains(&rows),
            "invalid dSpark attention output rows"
        );
        unsafe {
            self.grouped.launch_rope(
                self.input.buffer,
                self.frequencies.buffer,
                self.weight,
                self.scales,
                self.grouped_scratch.buffer,
                self.alpha.buffer,
                self.projection.input(),
                rows,
                stream,
            )?;
            self.projection.enqueue(
                self.projection.input(),
                self.projection.output_storage(),
                rows,
                stream,
            )
        }
    }
    /// # Safety
    /// Initialize finite BF16 attention outputs and per-row FP32 complex frequencies,
    /// finish producer writes, and serialize raw view reuse through completion.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        let launched = unsafe { self.enqueue(rows, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warmup and capture use owned storage only.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "dSpark attention output already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows, self.stream.raw) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup,"destroying failed attention output capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute, using the captured row count.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        self.ready = None;
        let (graph, captured) = self
            .graph
            .context("dSpark attention output was not captured")?;
        ensure!(
            rows == captured,
            "dSpark attention output replay rows differ"
        );
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready
            .context("dSpark attention output is incomplete")?;
        let mut output = self.projection.output_storage();
        output.bytes = rows as usize * 10240;
        Ok(output)
    }
}
impl Drop for DsparkAttentionOutput<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark attention output");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark attention output graph");
            }
        }
    }
}

pub(super) fn pack_grouped_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &crate::v41_tensors::NativeRtxTensors<'a>,
) -> Result<[DeviceAllocation<'a>; 3]> {
    let mut output = Vec::with_capacity(3);
    // Drain before destinations are released on any failed load.
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for stage in 0..3 {
        output.push(DeviceAllocation::new(library, 1048576)?);
        unsafe {
            library.v41_fp8_matrix_kernel(1, 32768, 8192)?.pack_scales(
                tensors.get(&format!("mtp.{stage}.attn.wo_a.scale"))?,
                output.last().unwrap().buffer,
                stream.raw,
            )?;
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    output
        .try_into()
        .ok()
        .context("expected three grouped output scale allocations")
}
