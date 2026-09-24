//! Native engram projection carriers and fused residual gate ownership.
use super::EngramDeviceView;
use crate::{
    v41_memory::{DeviceAllocation, LoadStream},
    v41_tensors::NativeRtxTensors,
};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Plan};
use ds41rt_loader::OfficialV41Catalog;

pub(crate) struct EngramLayerWeights<'a> {
    tensors: NativeRtxTensors<'a>,
    packed_scales: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    layer_index: usize,
    names: [String; 4],
}
impl<'a> EngramLayerWeights<'a> {
    pub fn device_bytes(library:&NativeLibrary,catalog:&OfficialV41Catalog,layer_index:usize)->Result<usize> {
        let layer=*ds41rt_core::ENGRAM_LAYERS.get(layer_index).context("invalid engram layer index")?;
        let names=["wkv.weight","wkv.scale","q_weight","k_weight"].map(|suffix| format!("layers.{layer}.engram.{suffix}"));
        NativeRtxTensors::plan(catalog,&names)?.checked_add(library.v41_fp8_kernel(16)?.info().packed_weight_scale_bytes as usize)
            .context("engram resident budget overflow")
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer_index: usize,
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let layer = *ds41rt_core::ENGRAM_LAYERS
            .get(layer_index)
            .context("invalid engram layer index")?;
        let names = ["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
            .map(|suffix| format!("layers.{layer}.engram.{suffix}"));
        let kernel = library.v41_fp8_kernel(16)?;
        let packed_bytes = usize::try_from(kernel.info().packed_weight_scale_bytes)?;
        let required = NativeRtxTensors::plan(catalog, &names)?
            .checked_add(packed_bytes)
            .context("engram resident budget overflow")?;
        ensure!(
            required <= device_budget,
            "engram weights and packed scales exceed device budget"
        );
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names,
            device_budget - packed_bytes,
            staging_bytes,
        )?;
        let packed_scales = DeviceAllocation::new(library, packed_bytes)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            kernel.pack_scales(tensors.get(&names[1])?, packed_scales.buffer, stream.raw)?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(Self {
            tensors,
            packed_scales,
            library,
            layer_index,
            names,
        })
    }
    pub fn resident_bytes(&self) -> usize {
        self.tensors.resident_bytes() + self.packed_scales.buffer.bytes
    }
}

pub(crate) struct EngramGate<'weights, 'library> {
    stream: LoadStream<'library>,
    output: DeviceAllocation<'library>,
    residual: DeviceAllocation<'library>,
    embeddings: DeviceAllocation<'library>,
    text_mask: DeviceAllocation<'library>,
    graph: Option<(*mut std::ffi::c_void, usize)>,
    projected: DeviceAllocation<'library>,
    scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    kernel: V41Fp8Plan<'library>,
    weights: &'weights EngramLayerWeights<'library>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'weights, 'library> EngramGate<'weights, 'library> {
    pub fn layer(&self) -> usize {
        ds41rt_core::ENGRAM_LAYERS[self.weights.layer_index] as usize
    }
    /// # Safety
    /// Same input ownership and row-order contract as execute. Captures only
    /// this gate's owned buffers and recaptures when the live shape changes.
    pub unsafe fn execute_captured(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        if self.graph.as_ref().is_none_or(|(_, rows)| *rows != gathered.rows) {
            self.clear_graph()?;
            unsafe { self.capture(residual, gathered)?; }
        }
        unsafe { self.replay(residual, gathered) }
    }
    pub fn device_bytes(library:&NativeLibrary,capacity:usize)->Result<usize> {
        ensure!(capacity>0 && capacity<=4096,"invalid engram gate capacity");
        let scratch=library.v41_fp8_matrix_plan(capacity as u32,6144,25600)?.info().scratch_bytes as usize;
        (capacity*(40960*2+51200+24*512+1)).checked_add(scratch).and_then(|b| b.checked_add(4))
            .context("engram execution budget overflow")
    }
    pub fn new(
        weights: &'weights EngramLayerWeights<'library>,
        capacity: usize,
        device_budget: usize,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram gate capacity"
        );
        let kernel = weights.library.v41_fp8_matrix_plan(u32::try_from(capacity)?, 6144, 25600)?;
        let scratch_bytes = usize::try_from(kernel.info().scratch_bytes)?;
        let output_bytes = capacity * 4 * 5120 * 2;
        let projected_bytes = capacity * 5 * 5120 * 2;
        let input_bytes = output_bytes + capacity * 24 * 512 + capacity;
        let bytes = output_bytes
            .checked_add(projected_bytes)
            .and_then(|n| n.checked_add(scratch_bytes))
            .and_then(|n| n.checked_add(4 + input_bytes))
            .context("engram execution budget overflow")?;
        ensure!(
            bytes <= device_budget,
            "engram execution exceeds device budget"
        );
        let value = Self {
            stream: LoadStream {
                library: weights.library,
                raw: weights.library.cuda_stream_create()?,
            },
            output: DeviceAllocation::new(weights.library, output_bytes)?,
            residual: DeviceAllocation::new(weights.library, output_bytes)?,
            embeddings: DeviceAllocation::new(weights.library, capacity * 24 * 512)?,
            text_mask: DeviceAllocation::new(weights.library, capacity)?,
            graph: None,
            projected: DeviceAllocation::new(weights.library, projected_bytes)?,
            scratch: DeviceAllocation::new(weights.library, scratch_bytes)?,
            alpha: DeviceAllocation::new(weights.library, 4)?,
            kernel,
            weights,
            capacity,
            ready_rows: None,
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
    /// Quantize gathered rows by K32, project with native FP8 weights, then gate.
    ///
    /// # Safety
    /// Residual BF16 [rows,4,5120], gathered BF16 [rows,24,256] and text mask
    /// must belong to this device and remain valid through this call; producer
    /// writes must be complete. Inputs may use their corresponding `inputs()`
    /// destination; otherwise they must not alias owned workspace.
    pub unsafe fn execute(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        let launched = unsafe {
            self.stage_inputs(residual, gathered)?;
            self.enqueue(gathered.rows)
        };
        launched.and(self.synchronize())?;
        self.ready_rows = Some(gathered.rows);
        self.output()
    }
    /// Stable input destinations: residual BF16, embeddings BF16, text-mask bytes.
    /// Never free or retain after this owner drops; finish writes before execution.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        [
            self.residual.buffer,
            self.embeddings.buffer,
            self.text_mask.buffer,
        ]
    }
    unsafe fn stage_inputs(
        &self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<()> {
        self.synchronize()?;
        unsafe { self.enqueue_inputs(residual, gathered) }?;
        self.synchronize()
    }
    unsafe fn enqueue_inputs(&self, residual: Ds41rtDeviceBuffer, gathered: &EngramDeviceView) -> Result<()> {
        ensure!(
            gathered.layer_index == self.weights.layer_index,
            "engram gate layer mismatch"
        );
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity,
            "engram gate exceeds capacity"
        );
        let sources = [residual, gathered.embeddings, gathered.text_mask];
        let destinations = self.inputs();
        let sizes = [
            gathered.rows * 4 * 5120 * 2,
            gathered.rows * 24 * 512,
            gathered.rows,
        ];
        // Validate all views and cross-copy hazards before enqueuing any work.
        for (index, source) in sources.iter().enumerate() {
            ensure!(
                !source.ptr.is_null() && source.bytes >= sizes[index],
                "engram input is null or too small"
            );
            let start = source.ptr as usize;
            let end = start
                .checked_add(sizes[index])
                .context("engram input extent overflow")?;
            for (slot, destination) in destinations.iter().enumerate() {
                let dst = destination.ptr as usize;
                let dst_end = dst
                    .checked_add(destination.bytes)
                    .context("engram destination extent overflow")?;
                ensure!(
                    (slot == index && start == dst) || end <= dst || start >= dst_end,
                    "engram input overlaps another owned input destination"
                );
            }
        }
        let copies = (|| -> Result<()> {
            for index in 0..3 {
                if sources[index].ptr != destinations[index].ptr {
                    unsafe {
                        self.weights.library.copy_d2d_async(
                            destinations[index],
                            sources[index],
                            sizes[index],
                            self.stream.raw,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = copies { self.synchronize()?; return Err(error); }
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.kernel.launch(
                self.embeddings.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.packed_scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.projected.buffer,
                u32::try_from(rows)?,
                self.stream.raw,
            )?;
            self.weights.library.cuda_engram_gate_bf16_async(
                self.residual.buffer,
                self.projected.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.tensors.get(&self.weights.names[3])?,
                Some(self.text_mask.buffer),
                self.output.buffer,
                i32::try_from(rows)?,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-input contract as execute. Captures only owned addresses;
    /// the supplied external views need not remain alive after this call returns.
    pub unsafe fn capture(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<()> {
        ensure!(self.graph.is_none(), "engram graph is already captured");
        unsafe {
            self.execute(residual, gathered)?;
        }
        unsafe { self.capture_ready(gathered.rows) }
    }
    unsafe fn capture_ready(&mut self, rows: usize) -> Result<()> {
        self.ready_rows = None;
        unsafe {
            self.weights
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        // End on both paths so a failed enqueue cannot leave the stream capturing.
        let captured = unsafe { self.weights.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.weights.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed engram capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; row count must match capture.
    pub unsafe fn replay(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        let (graph, rows) = self.graph.context("engram graph has not been captured")?;
        ensure!(
            rows == gathered.rows,
            "engram replay row count differs from capture"
        );
        let launched = unsafe {
            self.stage_inputs(residual, gathered)?;
            self.weights
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// # Safety
    /// Retain gathered storage and exclusive residual ownership through completion
    /// or cancellation drain. The final residual copy is queued after the gate.
    pub async unsafe fn execute_into_cooperative(&mut self, residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView) -> Result<()> {
        self.ready_rows = None;
        let cold = self.graph.as_ref().is_none_or(|(_, rows)| *rows != gathered.rows);
        if cold { self.clear_graph()?; }
        let launched = (|| unsafe {
            crate::v41_memory::chain::join(self.weights.library, self.stream.raw)?;
            self.enqueue_inputs(residual, gathered)?;
            if cold { self.enqueue(gathered.rows) } else {
                self.weights.library.cuda_graph_launch(self.graph.unwrap().0, self.stream.raw)
            }
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        if cold {
            self.stream.wait().await?;
            unsafe { self.capture_ready(gathered.rows)?; }
            let launched = unsafe { self.weights.library.cuda_graph_launch(self.graph.unwrap().0, self.stream.raw) };
            if let Err(error) = launched { self.synchronize()?; return Err(error); }
        }
        let copied = unsafe { self.weights.library.copy_d2d_async(residual, self.output.buffer,
            gathered.rows * 40960, self.stream.raw) };
        if let Err(error) = copied { self.synchronize()?; return Err(error); }
        unsafe { crate::v41_memory::chain::finish_cooperative(&self.stream).await?; }
        self.ready_rows = Some(gathered.rows);
        Ok(())
    }
    #[cfg(test)]
    pub async unsafe fn check_cooperative(&mut self, residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView) -> Result<()> {
        use std::{future::Future, task::Poll};
        let lib = self.weights.library;
        let mut residual = residual; residual.bytes = gathered.rows * 40960;
        let read = |buffer: Ds41rtDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let original = read(residual)?;
        let expected = read(unsafe { self.execute_captured(residual, gathered)? })?;
        self.clear_graph()?;
        let cancelled = {
            let mut work = std::pin::pin!(unsafe { self.execute_into_cooperative(residual, gathered) });
            std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
        };
        if cancelled { assert!(self.output().is_err()); }
        for _ in 0..2 {
            lib.copy_h2d(residual, &original)?;
            unsafe { self.execute_into_cooperative(residual, gathered).await?; }
            assert_eq!(read(residual)?, expected);
        }
        lib.copy_h2d(residual, &original)?;
        eprintln!("PASS queued Engram gate {}: exact gate/residual parity, pending cancellation={cancelled}, reuse", self.layer());
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_rows = None;
        self.stream.require_complete()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.weights.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
    /// Borrowed BF16 residual; never free or retain across reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("engram gate output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows * 4 * 5120 * 2;
        Ok(output)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.weights
                .library
                .cuda_stream_synchronize(self.stream.raw)
        }
    }
}
impl Drop for EngramGate<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native engram residual gate");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.weights.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying native engram graph");
            }
        }
    }
}
