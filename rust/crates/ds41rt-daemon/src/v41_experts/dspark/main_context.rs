//! Shared main-hidden projection and three independent committed-KV producers.
use super::{DsparkProjection, DsparkWeights, ProjectionKind};
use crate::v41_dspark_cache::{DsparkWindow, WindowChunk, WindowWrite};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps};
use std::ffi::c_void;
mod proposal;
mod queued;
pub(crate) use proposal::prepare_commit_rows;
pub(crate) use proposal::MainProposal;

pub(crate) struct DsparkMainContext<'weights, 'library> {
    stream: LoadStream<'library>,
    main: DsparkProjection<'weights, 'library>,
    kv: [DsparkProjection<'weights, 'library>; 3],
    ops: V41AttentionOps<'library>,
    main_norm: Ds41rtDeviceBuffer,
    kv_norm: [Ds41rtDeviceBuffer; 3],
    normalized: DeviceAllocation<'library>,
    positions: DeviceAllocation<'library>,
    frequencies: DeviceAllocation<'library>,
    rotated: [DeviceAllocation<'library>; 3],
    capacity: u32,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
    pending_writes: Option<[WindowWrite; 3]>,
    commit_descriptors: [DeviceAllocation<'library>; 3],
    commit_staging: HostAllocation<'library>,
}
impl<'library> DsparkWeights<'library> {
    pub fn main_context(
        &self,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkMainContext<'_, 'library>> {
        let library = self.library;
        ensure!(
            DsparkMainContext::device_bytes(library, capacity)? <= budget,
            "dSpark main context exceeds budget"
        );
        let normalized = DeviceAllocation::new(library, capacity as usize * 10240)?;
        let kv = |stage| unsafe {
            self.projection_from(ProjectionKind::Kv(stage), capacity, normalized.buffer,
                DsparkProjection::external_input_bytes(library, ProjectionKind::Kv(stage), capacity)?)
        };
        Ok(DsparkMainContext {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            main: self.projection(ProjectionKind::Main, capacity,
                DsparkProjection::device_bytes(library, ProjectionKind::Main, capacity)?)?,
            kv: [
                kv(0)?, kv(1)?, kv(2)?,
            ],
            ops: library.v41_attention_ops()?,
            main_norm: self.tensor("mtp.0.main_norm.weight")?,
            kv_norm: [
                self.tensor("mtp.0.attn.kv_norm.weight")?,
                self.tensor("mtp.1.attn.kv_norm.weight")?,
                self.tensor("mtp.2.attn.kv_norm.weight")?,
            ],
            normalized,
            positions: DeviceAllocation::new(library, capacity as usize * 8)?,
            frequencies: DeviceAllocation::new(library, capacity as usize * 256)?,
            rotated: [
                DeviceAllocation::new(library, capacity as usize * 1024)?,
                DeviceAllocation::new(library, capacity as usize * 1024)?,
                DeviceAllocation::new(library, capacity as usize * 1024)?,
            ],
            capacity,
            graph: None,
            ready: None,
            pending_writes: None,
            commit_descriptors: [DeviceAllocation::new(library, 384)?, DeviceAllocation::new(library, 384)?, DeviceAllocation::new(library, 384)?],
            commit_staging: HostAllocation::new(library, capacity as usize*8 + 1152)?,
        })
    }
}
impl DsparkMainContext<'_, '_> {
    /// Extra storage beyond the main projection already in the wave plan;
    /// committed and private-draft KV use separate projection workspaces.
    pub fn additional_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid main context capacity"
        );
        Ok(1152 + capacity as usize * (10240 + 256 + 8 + 3 * 1024)
            + 3 * DsparkProjection::external_input_bytes(library, ProjectionKind::Kv(0), capacity)?)
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::additional_bytes(library, capacity)?
            .checked_add(DsparkProjection::device_bytes(
                library,
                ProjectionKind::Main,
                capacity,
            )?)
            .context("main context storage overflow")
    }
    /// BF16 [capacity,15360], concatenated taps in checkpoint target-layer order.
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.main.input()
    }
    /// # Safety
    /// Same packed requests/row order must be used for all three decoder taps.
    /// The prepared block and this owner remain live until the stream drains.
    pub unsafe fn enqueue_block_tap(
        &mut self,
        input: &crate::v41_block::PreparedBlockInput<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        self.ready = None;
        ensure!(
            input.previous_binding().layer() + 1 == input.layer
                && input.residual.bytes == input.tokens.len() * 40960
                && input.residual.device_id == self.input().device_id,
            "prepared dSpark tap binding or extent differs"
        );
        unsafe {
            self.enqueue_tap(
                input.residual,
                input.layer as u32,
                u32::try_from(input.tokens.len())?,
                stream,
            )
        }
    }
    /// Enqueue a decoder attention-input tap directly into the stable main input.
    /// No intermediate allocation or copy; can be part of the backbone graph.
    /// # Safety
    /// The caller provides the post-engram/pre-attention BF16 streams, preserves
    /// the same packed row order for all three layers, and keeps input and this
    /// owner live through stream completion or graph replay; all three tap writes
    /// must complete before execute/capture/replay consumes the main input.
    pub unsafe fn enqueue_tap(
        &mut self,
        input: Ds41rtDeviceBuffer,
        layer: u32,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(self.pending_writes.is_none(), "main-context commit still pending");
        self.ready = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "invalid main context tap rows"
        );
        unsafe { self.ops.tap(input, self.main.input(), rows, layer, stream) }
    }
    /// U64 [capacity] absolute committed main positions.
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.positions.buffer
    }
    fn prepare(&mut self, rows: u32) -> Result<()> {
        ensure!(self.pending_writes.is_none(), "main-context commit still pending");
        self.ready = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "invalid main context rows"
        );
        Ok(())
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        let stream = self.stream.raw;
        unsafe {
            self.ops
                .frequencies(self.positions.buffer, self.frequencies.buffer, rows, stream)?;
            self.main
                .enqueue(self.main.input(), self.main.output_storage(), rows, stream)?;
            self.ops.norm(
                self.main.output_storage(),
                self.main_norm,
                None,
                self.normalized.buffer,
                rows,
                5120,
                stream,
            )?;
            for stage in 0..3 {
                self.kv[stage].enqueue(
                    self.normalized.buffer,
                    self.kv[stage].output_storage(),
                    rows,
                    stream,
                )?;
                // Window writes perform the final official K32 quant/dequant
                // while scattering; do not quantize twice in this producer.
                self.ops.norm(
                    self.kv[stage].output_storage(),
                    self.kv_norm[stage],
                    Some(self.frequencies.buffer),
                    self.rotated[stage].buffer,
                    rows,
                    512,
                    stream,
                )?;
            }
        }
        Ok(())
    }
    /// Produce shared main context and all three stage KV planes from completed
    /// target taps, using the same flattened absolute positions for every stage.
    /// # Safety
    /// Input is finite BF16 [rows,15360], on this device, fully produced and
    /// disjoint from this owner's storage. All owners have exclusive GPU use.
    pub unsafe fn execute_input(
        &mut self,
        input: Ds41rtDeviceBuffer,
        positions: &[u64],
    ) -> Result<()> {
        self.ready = None;
        let rows = u32::try_from(positions.len())?;
        self.prepare(rows)?;
        ensure!(
            input.bytes == positions.len() * 30720
                && input.device_id == self.input().device_id
                && input.ptr != self.input().ptr,
            "target main-context input extent or device differs"
        );
        self.synchronize()?;
        self.stream
            .library
            .copy_d2d(self.input(), input, input.bytes)?;
        let bytes = positions
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>();
        self.stream.library.copy_h2d(self.positions(), &bytes)?;
        unsafe { self.execute(rows) }
    }
    /// # Safety
    /// Hidden inputs must be finite and positions initialized on this device;
    /// serialize writes and all borrowed outputs through completion/replay.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<()> {
        self.prepare(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        Ok(())
    }
    /// # Safety
    /// Same initialized-input and exclusive-use contract as execute.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.prepare(rows)?;
        ensure!(self.graph.is_none(), "main context already captured");
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => self.graph = Some((graph, rows)),
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                return Err(error);
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-input and exclusive-use contract as execute.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<()> {
        self.prepare(rows)?;
        let (graph, captured) = self.graph.context("main context not captured")?;
        ensure!(rows == captured, "main context replay rows differ");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        Ok(())
    }
    /// Normalized main hidden, shared by all three stages.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self.ready.context("main context is incomplete")?;
        let mut output = self.normalized.buffer;
        output.bytes = rows as usize * 10240;
        Ok(output)
    }
    #[cfg(test)]
    pub(crate) fn kv_output(&self, stage: usize) -> Result<Ds41rtDeviceBuffer> {
        let rows = self.ready.context("main context is incomplete")?;
        ensure!(stage < 3, "invalid dSpark KV stage");
        let mut output = self.rotated[stage].buffer;
        output.bytes = rows as usize * 1024;
        Ok(output)
    }
    /// Commit the same produced rows to each independently leased stage ring.
    /// Prevalidate all stages; invalidate every participating lease if a GPU
    /// failure leaves a partially updated multi-stage commit. Consumes readiness.
    /// # Safety
    /// Chunks describe accepted main tokens only, with correct positions and
    /// row mappings; windows are on this device. Reserved readers of disjoint slots may remain
    /// active; no unreserved raw consumer may overlap a written slot.
    pub unsafe fn commit(
        &mut self,
        windows: &mut [&mut DsparkWindow<'_>; 3],
        chunks: [&[WindowChunk]; 3],
    ) -> Result<()> {
        let rows = self.ready.take().context("main context is incomplete")?;
        for stage in 0..3 {
            windows[stage].validate_write(chunks[stage], rows)?;
        }
        // Corresponding stage leases can differ, but positions and row mappings
        // must describe the same batch before any device memory is changed.
        for stage in 1..3 {
            ensure!(
                chunks[stage].len() == chunks[0].len(),
                "main context stage batch differs"
            );
            for (a, b) in chunks[0].iter().zip(chunks[stage]) {
                ensure!(
                    windows[0].request_id(a.lease)? == windows[stage].request_id(b.lease)?,
                    "main context stage request differs"
                );
                ensure!(
                    a.position == b.position
                        && a.source_row == b.source_row
                        && a.tokens == b.tokens,
                    "main context stage row mapping differs"
                );
            }
        }
        let result = (|| -> Result<()> {
            for stage in 0..3 {
                self.stream.library.copy_d2d(
                    windows[stage].source(),
                    self.rotated[stage].buffer,
                    rows as usize * 1024,
                )?;
                unsafe {
                    windows[stage].write(chunks[stage])?;
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for stage in 0..3 {
                for chunk in chunks[stage] {
                    let _ = windows[stage].release(chunk.lease);
                }
            }
        }
        result
    }
}
impl Drop for DsparkMainContext<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark main context");
        }
        self.pending_writes = None; // Stream drained before slot reservations release.
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark main context graph");
            }
        }
    }
}
