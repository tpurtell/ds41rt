//! Native dSpark main/attention FP8 matrix ownership.
use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Plan};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug)]
pub(crate) enum ProjectionKind {
    Main,
    QueryA(usize),
    QueryB(usize),
    Kv(usize),
    OutputB(usize),
}
impl ProjectionKind {
    fn binding(self) -> Result<(usize, String, u32, u32)> {
        let (stage, offset, suffix, k, n) = match self {
            Self::Main => return Ok((0, "mtp.0.main_proj".into(), 15360, 5120)),
            Self::QueryA(s) => (s, 0, "wq_a", 5120, 1280),
            Self::QueryB(s) => (s, 1, "wq_b", 1280, 32768),
            Self::Kv(s) => (s, 2, "wkv", 5120, 512),
            Self::OutputB(s) => (s, 3, "wo_b", 8192, 5120),
        };
        ensure!(stage < 3, "invalid dSpark projection stage");
        Ok((
            1 + stage * 4 + offset,
            format!("mtp.{stage}.attn.{suffix}"),
            k,
            n,
        ))
    }
}
const KINDS: [ProjectionKind; 13] = [
    ProjectionKind::Main,
    ProjectionKind::QueryA(0),
    ProjectionKind::QueryB(0),
    ProjectionKind::Kv(0),
    ProjectionKind::OutputB(0),
    ProjectionKind::QueryA(1),
    ProjectionKind::QueryB(1),
    ProjectionKind::Kv(1),
    ProjectionKind::OutputB(1),
    ProjectionKind::QueryA(2),
    ProjectionKind::QueryB(2),
    ProjectionKind::Kv(2),
    ProjectionKind::OutputB(2),
];
pub(super) fn packed_bytes(library: &NativeLibrary) -> Result<usize> {
    KINDS.iter().try_fold(0usize, |bytes, kind| {
        let (_, _, k, n) = kind.binding()?;
        bytes
            .checked_add(
                library
                    .v41_fp8_matrix_info(16, k, n)?
                    .packed_weight_scale_bytes as usize,
            )
            .context("dSpark projection scale budget overflow")
    })
}
pub(super) fn wave_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
    KINDS.iter().try_fold(0usize, |bytes, kind| {
        bytes
            .checked_add(DsparkProjection::device_bytes(library, *kind, capacity)?)
            .context("dSpark projection wave budget overflow")
    })
}
pub(super) fn pack_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &NativeRtxTensors<'a>,
) -> Result<[DeviceAllocation<'a>; 13]> {
    let mut scales = Vec::with_capacity(13);
    // On error the stream drains before the vector releases pending destinations.
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for kind in KINDS {
        let (_, name, k, n) = kind.binding()?;
        let kernel = library.v41_fp8_matrix_kernel(16, k, n)?;
        scales.push(DeviceAllocation::new(
            library,
            kernel.info().packed_weight_scale_bytes as usize,
        )?);
        unsafe {
            kernel.pack_scales(
                tensors.get(&format!("{name}.scale"))?,
                scales.last().unwrap().buffer,
                stream.raw,
            )?;
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    scales
        .try_into()
        .ok()
        .context("expected thirteen dSpark projection scales")
}
pub(crate) struct DsparkProjection<'weights, 'library> {
    stream: LoadStream<'library>,
    kernel: V41Fp8Plan<'library>,
    _weights: &'weights DsparkWeights<'library>,
    weight: Ds41rtDeviceBuffer,
    scales: Ds41rtDeviceBuffer,
    scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    input: Ds41rtDeviceBuffer,
    _owned_input: Option<DeviceAllocation<'library>>,
    output: DeviceAllocation<'library>,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn projection(
        &self,
        kind: ProjectionKind,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        unsafe { self.projection_input(kind, capacity, None, budget) }
    }
    /// Borrow an existing producer buffer instead of allocating unused private input.
    /// # Safety
    /// Input remains on this device and live through this projection's destruction;
    /// the containing owner serializes producer writes and projection reads.
    pub unsafe fn projection_from(
        &self, kind: ProjectionKind, capacity: u32, input: Ds41rtDeviceBuffer, budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        unsafe { self.projection_input(kind, capacity, Some(input), budget) }
    }
    unsafe fn projection_input(
        &self, kind: ProjectionKind, capacity: u32, external: Option<Ds41rtDeviceBuffer>, budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        let library = self.library;
        ensure!(
            (if external.is_some() { DsparkProjection::external_input_bytes(library, kind, capacity)? }
                else { DsparkProjection::device_bytes(library, kind, capacity)? }) <= budget,
            "dSpark projection exceeds budget"
        );
        let (index, name, k, n) = kind.binding()?;
        let kernel = library.v41_fp8_matrix_plan(capacity, k, n)?;
        let owned_input = if external.is_none() { Some(DeviceAllocation::new(library, capacity as usize*k as usize*2)?) } else { None };
        let input = external.unwrap_or_else(|| owned_input.as_ref().unwrap().buffer);
        ensure!(!input.ptr.is_null() && input.bytes >= capacity as usize*k as usize*2
            && input.device_id == self.tensor(&format!("{name}.weight"))?.device_id,
            "projection input extent or device differs");
        let value = DsparkProjection {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            scratch: DeviceAllocation::new(library, kernel.info().scratch_bytes as usize)?,
            kernel,
            _weights: self,
            weight: self.tensor(&format!("{name}.weight"))?,
            scales: self.projection_scales[index].buffer,
            alpha: DeviceAllocation::new(library, 4)?,
            input,
            _owned_input: owned_input,
            output: DeviceAllocation::new(library, capacity as usize * n as usize * 2)?,
            graph: None,
            ready: None,
        };
        unsafe {
            value.kernel.initialize_scratch(
                value.scratch.buffer,
                value.alpha.buffer,
                value.stream.raw,
            )?;
        }
        value.synchronize()?;
        Ok(value)
    }
}
impl DsparkProjection<'_, '_> {
    pub fn device_bytes(
        library: &NativeLibrary,
        kind: ProjectionKind,
        capacity: u32,
    ) -> Result<usize> {
        let (_, _, k, n) = kind.binding()?;
        let info = library.v41_fp8_matrix_plan_info(capacity, k, n)?;
        usize::try_from(
            info.scratch_bytes + u64::from(capacity) * (u64::from(k) + u64::from(n)) * 2 + 4,
        )
        .context("dSpark projection workspace overflow")
    }
    pub fn external_input_bytes(library: &NativeLibrary, kind: ProjectionKind, capacity: u32) -> Result<usize> {
        let (_, _, k, _) = kind.binding()?;
        Self::device_bytes(library, kind, capacity)?
            .checked_sub(capacity as usize*k as usize*2).context("projection input budget underflow")
    }
    // Storage access for an exclusive containing owner that tracks completion itself.
    pub(super) fn output_storage(&self) -> Ds41rtDeviceBuffer {
        self.output.buffer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// For a containing attention owner: supplied views must be distinct initialized
    /// buffers on the stream device; drain before releasing this exclusive borrow.
    pub(super) unsafe fn enqueue(
        &mut self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        self.ready = None;
        unsafe {
            self.kernel.launch(
                input,
                self.weight,
                self.scales,
                self.scratch.buffer,
                self.alpha.buffer,
                output,
                rows,
                stream,
            )
        }
    }
    /// # Safety
    /// Initialize finite BF16 input rows and finish producer writes; serialize view reuse.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        let launched =
            unsafe { self.enqueue(self.input, self.output.buffer, rows, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same initialized input contract as execute; captures only owned addresses.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "dSpark projection graph already captured"
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
        let launched =
            unsafe { self.enqueue(self.input, self.output.buffer, rows, self.stream.raw) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup,"destroying failed projection capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute, with the captured row count.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        self.ready = None;
        let (graph, captured) = self.graph.context("dSpark projection was not captured")?;
        ensure!(
            rows == captured,
            "dSpark projection replay rows differ from capture"
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
            .context("dSpark projection output is incomplete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * self.kernel.info().output_dim as usize * 2;
        Ok(output)
    }
}
impl Drop for DsparkProjection<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark projection");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark projection graph");
            }
        }
    }
}
