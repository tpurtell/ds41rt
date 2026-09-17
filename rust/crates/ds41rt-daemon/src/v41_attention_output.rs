//! Backbone inverse rotary, grouped FP8 wo_a and native FP8 wo_b.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_sparse_attention::SparseAttentionOutput;
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan,
};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
const ROW_BYTES: [usize; 5] = [65536, 16384, 10240, 8, 256];
pub(crate) struct AttentionOutputWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 4],
    tensors: NativeRtxTensors<'a>,
    grouped_scales: DeviceAllocation<'a>,
    scales: Option<DeviceAllocation<'a>>,
}
impl<'a> AttentionOutputWeights<'a> {
    fn names(layer: usize) -> Result<[String; 4]> {
        ensure!(layer < 40, "invalid backbone output layer");
        Ok([
            format!("layers.{layer}.attn.wo_a.weight"),
            format!("layers.{layer}.attn.wo_a.scale"),
            format!("layers.{layer}.attn.wo_b.weight"),
            format!("layers.{layer}.attn.wo_b.scale"),
        ])
    }
    /// Official FP8 weights and packed scales for both output projections.
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Self::device_bytes_with_split(library,catalog,layer,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,
        layer:usize,split_output_b:bool)->Result<usize> {
        let names=Self::names(layer)?;
        Ok(NativeRtxTensors::plan(catalog,&names[..if split_output_b {2} else {4}])?
            + library.v41_fp8_matrix_info(1,32768,8192)?.packed_weight_scale_bytes as usize
            + if split_output_b {0} else {library.v41_fp8_matrix_info(1,8192,5120)?.packed_weight_scale_bytes as usize})
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        Self::load_with_split(library,catalog,layer,budget,staging,false)
    }
    pub fn load_with_split(library:&'a NativeLibrary,catalog:&OfficialV41Catalog,layer:usize,
        budget:usize,staging:usize,split_output_b:bool)->Result<Self> {
        ensure!(
            Self::device_bytes_with_split(library, catalog, layer,split_output_b)? <= budget,
            "attention output weights exceed budget"
        );
        let names = Self::names(layer)?;
        let grouped_kernel = library.v41_fp8_matrix_kernel(1, 32768, 8192)?;
        let grouped_scales = DeviceAllocation::new(library, grouped_kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let kernel = library.v41_fp8_matrix_kernel(1, 8192, 5120)?;
        let scales=if split_output_b {None} else {
            Some(DeviceAllocation::new(library,kernel.info().packed_weight_scale_bytes as usize)?)
        };
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names[..if split_output_b {2} else {4}],
            budget - grouped_scales.buffer.bytes - scales.as_ref().map_or(0,|s|s.buffer.bytes),
            staging,
        )?;
        let launched = (|| unsafe {
            grouped_kernel.pack_scales(tensors.get(&names[1])?, grouped_scales.buffer, stream.raw)?;
            if let Some(scales)=&scales {
                kernel.pack_scales(tensors.get(&names[3])?,scales.buffer,stream.raw)?;
            }
            Ok(())
        })();
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            grouped_scales,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionOutputWave<'_, 'a>> {
        ensure!(
            AttentionOutputWave::device_bytes_with_split(self.library, capacity,self.scales.is_none())? <= budget,
            "attention output wave exceeds budget"
        );
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernel=self.scales.as_ref().map(|_|self.library.v41_fp8_matrix_plan(capacity,8192,5120)).transpose()?;
        let grouped = self.library.v41_fp8_matrix_plan(capacity, 32768, 8192)?;
        let grouped_scratch = DeviceAllocation::new(self.library, grouped.info().scratch_bytes as usize)?;
        let value = AttentionOutputWave {
            stream,
            grouped,
            grouped_scratch,
            scratch: kernel.as_ref().map(|k|DeviceAllocation::new(self.library,k.info().scratch_bytes as usize)).transpose()?,
            kernel,
            alpha: DeviceAllocation::new(self.library, 4)?,
            norm: self.library.v41_attention_ops()?,
            buffers: ROW_BYTES
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n * capacity as usize))
                .collect::<Result<Vec<_>>>()?,
            position_staging: HostAllocation::new(self.library, capacity as usize * 8)?,
            weights: self,
            capacity,
            graphs: LayerGraphs::new(self.library),
            ready: None,
            origin: None,
        };
        ensure!(
            value.b(0).device_id == self.grouped_scales.buffer.device_id,
            "output weight device differs"
        );
        let launched = (|| unsafe {
            value.grouped.initialize_scratch(value.grouped_scratch.buffer, value.alpha.buffer, value.stream.raw)?;
            if let (Some(kernel),Some(scratch))=(&value.kernel,&value.scratch) {
                kernel.initialize_scratch(scratch.buffer,value.alpha.buffer,value.stream.raw)?;
            }
            Ok(())
        })();
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
pub(crate) struct AttentionOutput<'a> {
    origin: Option<QueryBinding>,
    pub layer: usize,
    pub rows: usize,
    pub input: Ds41rtDeviceBuffer,
    pub grouped: Ds41rtDeviceBuffer,
    pub projected: Ds41rtDeviceBuffer,
    pub positions: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        let b = self
            .origin
            .context("attention projection has no query origin")?;
        ensure!(
            b.layer() == self.layer,
            "attention projection layer differs"
        );
        Ok(b)
    }
}
pub(crate) struct AttentionOutputWave<'w, 'a> {
    stream: LoadStream<'a>,
    grouped: V41Fp8Plan<'a>,
    grouped_scratch: DeviceAllocation<'a>,
    kernel: Option<V41Fp8Plan<'a>>,
    scratch: Option<DeviceAllocation<'a>>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    position_staging: HostAllocation<'a>,
    weights: &'w AttentionOutputWeights<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, AttentionOutputWeights<'a>>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> AttentionOutputWave<'w, 'a> {
    /// Reuse this lane's storage with another layer's weights. Existing output
    /// borrows must end first. Cached graphs retain their original weight owners.
    pub fn rebind(&mut self, weights: &'w AttentionOutputWeights<'a>) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            std::ptr::eq(self.stream.library, weights.library)
                && self.kernel.is_some()==weights.scales.is_some()
                && weights.grouped_scales.buffer.device_id == self.b(0).device_id,
            "attention rebound weight library or device differs"
        );
        self.stream.require_complete()?;
        self.weights = weights;
        Ok(())
    }
}
impl AttentionOutputWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::device_bytes_with_split(library,capacity,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,capacity:u32,split_output_b:bool)->Result<usize> {
        Ok(library.v41_fp8_matrix_plan_info(capacity, 32768, 8192)?.scratch_bytes as usize
            + 4
            + capacity as usize * ROW_BYTES.iter().sum::<usize>()
            + if split_output_b {0} else {library.v41_fp8_matrix_plan_info(capacity,8192,5120)?.scratch_bytes as usize})
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.b(3)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "attention output rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.stream.raw) }
    }
    unsafe fn enqueue_grouped_on(&mut self, rows: u32, stream: *mut std::ffi::c_void) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(3),
                self.b(4),
                rows,
                self.weights.layer as u32,
                stream,
            )?;
            self.grouped.launch_rope(
                self.b(0),
                self.b(4),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.grouped_scales.buffer,
                self.grouped_scratch.buffer,
                self.alpha.buffer,
                self.b(1),
                rows,
                stream,
            )?;
        }
        Ok(())
    }
    unsafe fn enqueue_on(&mut self,rows:u32,stream:*mut std::ffi::c_void)->Result<()> {
        ensure!(self.kernel.is_some(),"split output-B requires TP2 execution");
        unsafe {
            self.enqueue_grouped_on(rows,stream)?;
            self.kernel.as_ref().context("full output-B kernel absent")?.launch(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales.as_ref().context("full output-B scales absent")?.buffer,
                self.scratch.as_ref().context("full output-B scratch absent")?.buffer,
                self.alpha.buffer,
                self.b(2),
                rows,
                stream,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "attention output graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready_on(rows, self.stream.raw) }
    }
    unsafe fn capture_ready_on(&mut self, rows: u32, stream: *mut std::ffi::c_void) -> Result<()> {
        self.ready = None;
        self.origin = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(stream)?;
        }
        let launched = unsafe { self.enqueue_on(rows, stream) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(stream) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) = unsafe {
                    self.graphs
                        .insert(self.weights.layer, self.weights, rows, graph)
                } {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same matching initialized inputs as execute. Captured live row count is fixed.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .context("attention output graph missing")?;
        ensure!(count == rows, "attention output capture row count differs");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// No external writes race this wave or the completed attention output.
    pub unsafe fn execute_attention(
        &mut self,
        attention: &SparseAttentionOutput<'_>,
    ) -> Result<AttentionOutput<'_>> {
        self.ready = None;
        self.origin = None;
        let tokens = attention.tokens()?;
        ensure!(
            attention.layer == self.weights.layer
                && attention.rows == tokens.len()
                && attention.rows <= self.capacity as usize
                && attention.values.device_id == self.b(0).device_id,
            "attention output origin differs"
        );
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let executed = (|| -> Result<()> {
            unsafe {
                self.stream.library.copy_d2d_async(
                    self.b(0), attention.values, attention.values.bytes, self.stream.raw,
                )?;
                self.stream.library.copy_host_buffer_h2d_async(
                    self.positions(), self.position_staging.buffer, tokens.len() * 8, self.stream.raw,
                )?;
            }
            let rows = attention.rows as u32;
            if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
                unsafe { self.capture(rows)?; }
            }
            unsafe { self.replay(rows)?; }
            Ok(())
        })();
        if let Err(error) = executed {
            // Drain staged copies even if graph preparation/replay rejects them.
            self.synchronize()?;
            self.ready = None;
            self.origin = None;
            return Err(error);
        }
        self.origin = Some(attention.binding()?);
        self.output()
    }
    /// # Safety
    /// Attention is queued on stream. All owners, including this projection's
    /// graph and pinned staging, stay alive and exclusive until that stream is
    /// drained by the enclosing sparse-attention continuation, including errors.
    /// This method never publishes AttentionOutput or marks this wave ready.
    pub unsafe fn enqueue_attention(
        &mut self, attention: &crate::v41_sparse_attention::QueuedSparseAttention,
        tokens: &[u64], stream: *mut std::ffi::c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready = None;
        self.origin = None;
        ensure!(attention.layer == self.weights.layer && attention.rows == tokens.len()
            && attention.rows > 0 && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id,
            "queued attention output origin differs");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            self.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len() * 8, stream)?;
        }
        let rows = attention.rows as u32;
        if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
            // Cold setup uses the existing drained capture path; its input copies
            // must complete before capture's private-stream warmup can read them.
            unsafe { self.stream.library.cuda_stream_synchronize(stream)?; self.capture(rows)?; }
        }
        let (graph, count) = self.graphs.get_shape(self.weights.layer, self.weights, rows)
            .context("queued output graph missing")?;
        ensure!(count == rows, "queued output graph shape differs");
        unsafe { self.stream.library.cuda_graph_launch(graph, stream)?; }
        let mut output = self.b(2);
        output.bytes = attention.rows * ROW_BYTES[2];
        Ok(output)
    }
    /// # Safety
    /// Retain attention and this wave through the containing stream's completion.
    /// None means warmup is queued and finish_prepared must follow a cooperative wait.
    pub unsafe fn enqueue_attention_prepared(&mut self,
        attention: &crate::v41_sparse_attention::QueuedSparseAttention, tokens: &[u64],
        stream: *mut std::ffi::c_void) -> Result<Option<Ds41rtDeviceBuffer>> {
        self.ready = None;
        self.origin = None;
        ensure!(attention.layer == self.weights.layer && attention.rows == tokens.len()
            && attention.rows > 0 && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id,
            "queued attention output origin differs");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            self.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len() * 8, stream)?;
        }
        let rows = attention.rows as u32;
        if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
            unsafe { self.enqueue_on(rows, stream)?; }
            return Ok(None);
        }
        unsafe { self.replay_prepared(rows, stream).map(Some) }
    }
    /// # Safety
    /// Queued warmup on stream has completed; retain all storage through replay.
    pub unsafe fn finish_prepared(&mut self, rows: u32,
        stream: *mut std::ffi::c_void) -> Result<Ds41rtDeviceBuffer> {
        unsafe { self.capture_ready_on(rows, stream)?; self.replay_prepared(rows, stream) }
    }
    unsafe fn replay_prepared(&self, rows: u32, stream: *mut std::ffi::c_void) -> Result<Ds41rtDeviceBuffer> {
        let (graph, count) = self.graphs.get_shape(self.weights.layer, self.weights, rows)
            .context("prepared output graph missing")?;
        ensure!(count == rows, "prepared output graph shape differs");
        unsafe { self.stream.library.cuda_graph_launch(graph, stream)?; }
        let mut output = self.b(2); output.bytes = rows as usize * ROW_BYTES[2];
        Ok(output)
    }
    /// Prepare only mutable metadata outside a containing graph. The caller
    /// drains stream before releasing or reusing this wave or pinned staging.
    pub unsafe fn prepare_chain_graph(&mut self, tokens: &[u64], stream: *mut std::ffi::c_void) -> Result<()> {
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p| p < 1048576), "projection graph positions exceed context");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe { self.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
            tokens.len() * 8, stream) }
    }
    /// # Safety
    /// Matching sparse output is ordered on stream. Capture retains all weights
    /// and storage through graph destruction, and owners stay exclusive in flight.
    pub unsafe fn enqueue_chain_graph(&mut self, attention: &crate::v41_sparse_attention::QueuedSparseAttention,
        stream: *mut std::ffi::c_void) -> Result<Ds41rtDeviceBuffer> {
        ensure!(attention.layer == self.weights.layer && attention.rows > 0
            && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id, "projection graph origin differs");
        unsafe {
            self.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.enqueue_on(attention.rows as u32, stream)?;
        }
        let mut value = self.b(2); value.bytes = attention.rows * 10240; Ok(value)
    }
    /// # Safety
    /// Same capture/producer contract as enqueue_chain_graph. This queues only
    /// inverse rotary and grouped output-A; output-B must follow before FFN.
    pub unsafe fn enqueue_grouped_chain_graph(&mut self,
        attention:&crate::v41_sparse_attention::QueuedSparseAttention,
        stream:*mut std::ffi::c_void)->Result<Ds41rtDeviceBuffer> {
        ensure!(self.kernel.is_none() && attention.layer==self.weights.layer && attention.rows>0
            && attention.rows<=self.capacity as usize && attention.values.device_id==self.b(0).device_id,
            "split output prefix differs");
        unsafe {
            self.stream.library.copy_d2d_async(self.b(0),attention.values,attention.values.bytes,stream)?;
            self.enqueue_grouped_on(attention.rows as u32,stream)?;
        }
        Ok(Ds41rtDeviceBuffer {bytes:attention.rows*ROW_BYTES[1],..self.b(1)})
    }
    /// # Safety
    /// Grouped output-A has been queued on producer (or is complete when absent).
    /// Retain prefix producer, this wave and consumer storage until completion
    /// or drained cancellation. Consumer must enqueue only on the supplied stream.
    pub async unsafe fn finish_tp2_then<T>(&mut self,rows:u32,
        projection:&mut crate::v41_projection_tp2::Wave<'_, '_>,producer:Option<*mut std::ffi::c_void>,
        consume:impl FnOnce(Ds41rtDeviceBuffer,*mut std::ffi::c_void)->Result<T>)->Result<T> {
        self.validate(rows)?;
        ensure!(self.kernel.is_none() && projection.kind()==crate::v41_projection_tp2::Kind::OutputB
            && projection.output_device().id==self.b(0).device_id
            && std::ptr::eq(projection.output_device().library,self.stream.library),"output projection owner differs");
        unsafe { projection.execute_after(self.weights.layer,rows,
            Ds41rtDeviceBuffer {bytes:rows as usize*ROW_BYTES[1],..self.b(1)},producer,|output,stream| {
                self.stream.library.copy_d2d_async(self.b(2),output,output.bytes,stream)?;
                consume(Ds41rtDeviceBuffer {bytes:output.bytes,..self.b(2)},stream)
            }).await }
    }
    pub fn chain_graph_identity(&self) -> [usize; 2] {
        [self.b(0).ptr as usize, self.weights as *const _ as usize]
    }
    pub fn output(&self) -> Result<AttentionOutput<'_>> {
        let rows = self.ready.context("attention output unpublished")? as usize;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * ROW_BYTES[i];
            b
        };
        Ok(AttentionOutput {
            origin: self.origin,
            layer: self.weights.layer,
            rows,
            input: b(0),
            grouped: b(1),
            projected: b(2),
            positions: b(3),
            frequencies: b(4),
            _owner: PhantomData,
        })
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graphs.enable_small_shapes(); }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.weights.layer)?;
        }
        Ok(())
    }
}
impl Drop for AttentionOutputWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            tracing::error!(%error,"draining attention output graph");
        }
    }
}

#[cfg(test)]
mod tp2_tests {
    use super::*;
    use crate::v41_memory::device::{Allocation,Device};
    use crate::v41_projection_tp2::{Kind,Weights,Wave};
    #[test]
    #[ignore = "requires checkpoint, two GPUs and projection shard AOT"]
    fn checkpoint_tp2_output_preserves_grouped_rotary() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog=ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        let devices=[Device {library:&library,id:0},Device {library:&library,id:1}];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for (owner,layer) in [(0,2),(1,20)] {
            let device=devices[owner];let capacity=16;
            let weights=device.own(||AttentionOutputWeights::load(&library,&catalog,layer,
                AttentionOutputWeights::device_bytes(&library,&catalog,layer)?,1<<20))?;
            let split=device.own(||AttentionOutputWeights::load_with_split(&library,&catalog,layer,
                AttentionOutputWeights::device_bytes_with_split(&library,&catalog,layer,true)?,1<<20,true))?;
            assert!(split.scales.is_none() && split.tensors.get(&split.names[2]).is_err());
            let mut reference=device.own(||weights.wave(capacity,usize::MAX))?;
            let mut output=device.own(||split.wave(capacity,usize::MAX))?;
            assert!(output.scratch.is_none() && output.rebind(&weights).is_err());
            let halves=[vec![Weights::load(devices[0],&catalog,layer,Kind::OutputB,0,
                Weights::load_peak_device_bytes(Kind::OutputB))?],
                vec![Weights::load(devices[1],&catalog,layer,Kind::OutputB,1,
                Weights::load_peak_device_bytes(Kind::OutputB))?]];
            let mut projection=Wave::new([&halves[0],&halves[1]],capacity,owner,
                Wave::device_bytes(&library,Kind::OutputB,capacity,owner)?)?;
            let input=Allocation::new(device,capacity as usize*ROW_BYTES[0])?;
            let read=|buffer:Ds41rtDeviceBuffer|->Result<Vec<u8>> {
                let mut bytes=vec![0;buffer.bytes];device.run(||library.copy_d2h(&mut bytes,buffer))?;Ok(bytes)
            };
            for (cycle,rows) in [1,6,16,6,1].into_iter().enumerate() {
                let host:Vec<u8>=(0..capacity as usize*32768).flat_map(|i| {
                    let value=((i+cycle*5)%29) as f32/64.0-0.25;
                    ((value.to_bits()>>16) as u16).to_ne_bytes()
                }).collect();
                device.run(||library.copy_h2d(input.buffer,&host))?;
                let tokens:Vec<u64>=(0..rows).map(|i|(cycle*8192+i) as u64).collect();
                let attention=crate::v41_sparse_attention::QueuedSparseAttention {
                    values:Ds41rtDeviceBuffer {bytes:rows*ROW_BYTES[0],..input.buffer},rows,layer};
                let stream=reference.stream.raw;
                let expected=device.run(||unsafe {
                    reference.prepare_chain_graph(&tokens,stream)?;
                    let result=reference.enqueue_chain_graph(&attention,stream)?;
                    library.cuda_stream_synchronize(stream)?;read(result)
                })?;
                let stream=output.stream.raw;
                device.run(||unsafe {
                    output.prepare_chain_graph(&tokens,stream)?;
                    output.enqueue_grouped_chain_graph(&attention,stream)?;Ok(())
                })?;
                runtime.block_on(async {
                    let mut pending=std::pin::pin!(device.future(unsafe {
                        output.finish_tp2_then(rows as u32,&mut projection,Some(stream),|value,_|Ok(value))
                    }));
                    std::future::poll_fn(|cx| {
                        use std::future::Future;
                        let _=pending.as_mut().poll(cx);std::task::Poll::Ready(())
                    }).await;
                });
                let actual=runtime.block_on(device.future(unsafe {
                    output.finish_tp2_then(rows as u32,&mut projection,Some(stream),|value,_|Ok(value))
                }))?;
                let actual=read(actual)?;let mut max_error=0f32;let mut max_ulp=0u16;let mut changed=0usize;
                for index in [0,1,3,4] {
                    let slice=|buffer:Ds41rtDeviceBuffer|Ds41rtDeviceBuffer {bytes:rows*ROW_BYTES[index],..buffer};
                    assert_eq!(read(slice(reference.b(index)))?,read(slice(output.b(index)))?);
                }
                for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                    let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                    let (a,b)=(decode(a),decode(b));
                    ensure!(a.is_finite() && b.is_finite(),"nonfinite output projection");
                    let ulp=((a.to_bits()>>16) as u16).abs_diff((b.to_bits()>>16) as u16);
                    // The full M1 export uses split-K=2, while each half uses
                    // split-K=1. Different FP32 accumulation order can straddle
                    // a BF16 rounding boundary; allow one adjacent BF16 value.
                    ensure!((a-b).abs()<=1e-4 || ulp<=1,
                        "TP2 output differs beyond rounding: layer={layer} rows={rows} {a} vs {b}");
                    max_ulp=max_ulp.max(ulp);changed+=usize::from(a!=b);
                    max_error=max_error.max((a-b).abs());
                }
                eprintln!("TP2 complete output layer={layer} rows={rows} cycle={cycle} max_error={max_error} max_ulp={max_ulp} changed={changed}");

            }
        }
        Ok(())
    }
}
