//! Real backbone low-rank query projection, normalization and rotary graphs.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
const MATRICES: [(u32, u32); 2] = [(5120, 1280), (1280, 32768)];
const ROW_BYTES: [usize; 7] = [10240, 2560, 2560, 65536, 65536, 8, 256];
fn part(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
pub(crate) struct AttentionQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 5],
    tensors: NativeRtxTensors<'a>,
    scales: Vec<DeviceAllocation<'a>>,
    split_query_b: bool,
}
impl<'a> AttentionQueryWeights<'a> {
    fn names(layer: usize) -> Result<[String; 5]> {
        ensure!(layer < 40, "invalid backbone query layer");
        Ok([
            format!("layers.{layer}.attn.wq_a.weight"),
            format!("layers.{layer}.attn.wq_a.scale"),
            format!("layers.{layer}.attn.wq_b.weight"),
            format!("layers.{layer}.attn.wq_b.scale"),
            format!("layers.{layer}.attn.q_norm.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Self::device_bytes_with_split(library,catalog,layer,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,
        layer:usize,split_query_b:bool)->Result<usize> {
        let names=Self::names(layer)?;
        let selected:Vec<_>=names.iter().enumerate().filter(|(i,_)|!split_query_b || !matches!(i,2|3))
            .map(|(_,name)|name.clone()).collect();
        let mut bytes=NativeRtxTensors::plan(catalog,&selected)?;
        for (k,n) in MATRICES.into_iter().take(if split_query_b {1} else {2}) {
            bytes+=library.v41_fp8_matrix_info(1,k,n)?.packed_weight_scale_bytes as usize;
        }
        Ok(bytes)
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
        budget:usize,staging:usize,split_query_b:bool)->Result<Self> {
        ensure!(
            Self::device_bytes_with_split(library, catalog, layer,split_query_b)? <= budget,
            "attention query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let selected:Vec<_>=names.iter().enumerate().filter(|(i,_)|!split_query_b || !matches!(i,2|3))
            .map(|(_,name)|name.clone()).collect();
        let tensors = NativeRtxTensors::load(library, catalog, &selected, budget, staging)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut scales = vec![];
        for (i, (k, n)) in MATRICES.into_iter().take(if split_query_b {1} else {2}).enumerate() {
            let kernel = library.v41_fp8_matrix_kernel(1, k, n)?;
            let scale =
                DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
            let launched = unsafe {
                kernel.pack_scales(tensors.get(&names[i * 2 + 1])?, scale.buffer, stream.raw)
            };
            launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
            scales.push(scale);
        }
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
            split_query_b,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionQueryWave<'_, 'a>> {
        ensure!(
            AttentionQueryWave::device_bytes_with_split(self.library, capacity,self.split_query_b)? <= budget,
            "attention query wave exceeds budget"
        );
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernels = MATRICES
            .into_iter().take(if self.split_query_b {1} else {2})
            .map(|(k, n)| self.library.v41_fp8_matrix_plan(capacity, k, n))
            .collect::<Result<Vec<_>>>()?;
        let scratch = kernels
            .iter()
            .map(|k| DeviceAllocation::new(self.library, k.info().scratch_bytes as usize))
            .collect::<Result<Vec<_>>>()?;
        let value = AttentionQueryWave {
            stream,
            kernels,
            scratch,
            alpha: DeviceAllocation::new(self.library, 4)?,
            buffers: ROW_BYTES
                .into_iter().enumerate()
                .map(|(i, n)| {
                    if i == 4 && !cfg!(test) { Ok(None) }
                    else { DeviceAllocation::new(self.library, n * capacity as usize).map(Some) }
                })
                .collect::<Result<Vec<_>>>()?,
            norm: self.library.v41_attention_ops()?,
            position_staging: HostAllocation::new(self.library, capacity as usize * 8)?,
            weights: self,
            capacity,
            graphs: LayerGraphs::new(self.library),
            tp2_prefix_graphs: LayerGraphs::new(self.library),
            ready: None,
            binding: None,
            tokens: Vec::new(),
        };
        ensure!(
            value.b(0).device_id == self.tensors.get(&self.names[0])?.device_id,
            "query weight device differs"
        );
        for i in 0..value.kernels.len() {
            let launched = unsafe {
                value.kernels[i].initialize_scratch(
                    value.scratch[i].buffer,
                    value.alpha.buffer,
                    value.stream.raw,
                )
            };
            launched.and(value.synchronize())?;
        }
        Ok(value)
    }
}
pub(crate) struct AttentionQueryOutput<'a> {
    binding: Option<QueryBinding>,
    tokens: &'a [u64],
    pub layer: usize,
    pub rows: usize,
    pub hidden: Ds41rtDeviceBuffer,
    pub raw_rank: Ds41rtDeviceBuffer,
    pub normalized_rank: Ds41rtDeviceBuffer,
    #[cfg(test)]
    pub projected: Ds41rtDeviceBuffer,
    #[cfg(test)]
    pub qb_scratch: Option<Ds41rtDeviceBuffer>,
    pub rotated: Ds41rtDeviceBuffer,
    pub positions: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionQueryOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.binding.context("query has no token binding")
    }
    pub fn tokens(&self) -> Result<&[u64]> {
        self.binding()?;
        Ok(self.tokens)
    }
}
pub(crate) struct AttentionQueryWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernels: Vec<V41Fp8Plan<'a>>,
    scratch: Vec<DeviceAllocation<'a>>,
    alpha: DeviceAllocation<'a>,
    buffers: Vec<Option<DeviceAllocation<'a>>>,
    position_staging: HostAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w AttentionQueryWeights<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, AttentionQueryWeights<'a>>,
    tp2_prefix_graphs: LayerGraphs<'w, 'a, AttentionQueryWeights<'a>>,
    ready: Option<u32>,
    binding: Option<QueryBinding>,
    tokens: Vec<u64>,
}
impl<'w, 'a> AttentionQueryWave<'w, 'a> {
    /// Reuse this lane's storage with another layer's weights. Existing output
    /// borrows must end first. Cached graphs retain their original weight owners.
    pub fn rebind(&mut self, weights: &'w AttentionQueryWeights<'a>) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            std::ptr::eq(self.stream.library, weights.library)
                && self.kernels.len()==weights.scales.len()
                && weights.tensors.get(&weights.names[0])?.device_id == self.b(0).device_id,
            "attention rebound weight library or device differs"
        );
        self.stream.require_complete()?;
        self.weights = weights;
        Ok(())
    }
}
impl AttentionQueryWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::device_bytes_with_split(library,capacity,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,capacity:u32,split_query_b:bool)->Result<usize> {
        let mut bytes = 4 + capacity as usize * ROW_BYTES.iter().enumerate()
            .filter(|(i, _)| *i != 4 || cfg!(test)).map(|(_, bytes)| bytes).sum::<usize>();
        for (k, n) in MATRICES.into_iter().take(if split_query_b {1} else {2}) {
            bytes += library.v41_fp8_matrix_plan_info(capacity, k, n)?.scratch_bytes as usize;
        }
        Ok(bytes)
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].as_ref().expect("query buffer is allocated").buffer
    }
    pub fn layer(&self) -> usize {
        self.weights.layer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.b(5)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "query rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue_rank(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(5),
                self.b(6),
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.kernels[0].launch(
                self.b(0),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales[0].buffer,
                self.scratch[0].buffer,
                self.alpha.buffer,
                self.b(1),
                rows,
                self.stream.raw,
            )?;
            self.norm.norm(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[4])?,
                None,
                self.b(2),
                rows,
                1280,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        ensure!(!self.weights.split_query_b,"split query-B weights require TP2 execution");
        unsafe {
            self.enqueue_rank(rows)?;
            self.kernels[1].launch(
                self.b(2),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales[1].buffer,
                self.scratch[1].buffer,
                self.alpha.buffer,
                self.b(3),
                rows,
                self.stream.raw,
            )?;
            // Preserve the pre-RoPE tensor only for numerical trace fixtures.
            // Serving rotates the completed projection in place, saving one
            // 64-head BF16 buffer per lane without changing the kernel math.
            #[cfg(test)]
            self.stream.library.copy_d2d_async(self.b(4), self.b(3),
                rows as usize * ROW_BYTES[3], self.stream.raw)?;
            self.norm.rope(
                self.b(3),
                self.b(6),
                self.b(3),
                rows,
                64,
                false,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
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
        self.binding = None;
        self.tokens.clear();
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "attention query graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        unsafe { self.capture_ready(rows) }
    }
    /// Capture only after the input producer and warmup have completed.
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .context("attention query graph missing")?;
        ensure!(count == rows, "attention query capture row count differs");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(unsafe { crate::v41_memory::chain::finish(self.stream.library, self.stream.raw) })?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Hidden rows are finite attention inputs in the supplied token order, with
    /// producer writes drained. No external writes race this wave.
    pub unsafe fn execute_tokens(&mut self, tokens: &[u64]) -> Result<AttentionQueryOutput<'_>> {
        unsafe { self.execute_tokens_prepared(tokens, |_, _| Ok(())) }
    }
    /// Enqueue an input producer before the query graph on its owned stream.
    ///
    /// # Safety
    /// The producer must initialize finite hidden rows in token order, retain
    /// its allocations through this call, and use only the supplied stream.
    /// Success and producer/query errors both drain before releasing its borrow.
    /// Preparation is outside capture; cached graphs contain query operations only.
    pub(crate) unsafe fn execute_tokens_prepared(
        &mut self,
        tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, Ds41rtDeviceBuffer) -> Result<()>,
    ) -> Result<AttentionQueryOutput<'_>> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.capacity as usize
                && tokens.iter().all(|&p| p < 1048576),
            "invalid query tokens"
        );
        let binding = QueryBinding::new(self.weights.layer)?;
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let prepared_and_executed = (|| -> Result<()> {
            unsafe {
                crate::v41_memory::chain::join(self.stream.library, self.stream.raw)?;
                self.stream.library.copy_host_buffer_h2d_async(
                    self.positions(), self.position_staging.buffer, tokens.len() * 8, self.stream.raw,
                )?;
            }
            prepare(self.stream.raw, self.input())?;
            let rows = tokens.len() as u32;
            if self
                .graphs
                .get_shape(self.weights.layer, self.weights, rows)
                .is_none()
            {
                unsafe { self.capture(rows)?; }
            }
            unsafe { self.replay(rows)?; }
            Ok(())
        })();
        if let Err(error) = prepared_and_executed {
            // Even a partially submitted producer must finish before its owner
            // can be reset or reused. Successful replay already drained.
            let drained = self.synchronize();
            self.ready = None;
            self.binding = None;
            if let Err(drain_error) = drained {
                tracing::error!(%drain_error, "draining failed query preparation");
            }
            return Err(error);
        }
        self.tokens.extend_from_slice(tokens);
        self.binding = Some(binding);
        self.output()
    }
    /// # Safety
    /// Same producer/ownership contract as execute_tokens_prepared, retained
    /// across suspension. All producer and query work uses this one stream.
    pub(crate) async unsafe fn execute_tokens_prepared_cooperative(&mut self, tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, Ds41rtDeviceBuffer) -> Result<()>)
        -> Result<AttentionQueryOutput<'_>> {
        ensure!(tokens.len() <= self.capacity as usize, "query rows exceed capacity");
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p| p < 1048576), "invalid query tokens");
        let binding = QueryBinding::new(self.weights.layer)?;
        let rows = tokens.len() as u32;
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let graph = self.graphs.get_shape(self.weights.layer, self.weights, rows);
        let queued = (|| -> Result<()> { unsafe {
            crate::v41_memory::chain::join(self.stream.library, self.stream.raw)?;
            self.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len()*8, self.stream.raw)?;
            prepare(self.stream.raw, self.input())?;
            if let Some((graph, _)) = graph { self.stream.library.cuda_graph_launch(graph, self.stream.raw) }
            else { self.enqueue(rows) }
        } })();
        let drained = if queued.is_err() { self.stream.wait().await }
            else { unsafe { crate::v41_memory::chain::finish_cooperative(&self.stream).await } };
        queued.and(drained)?;
        if graph.is_none() {
            // Eager output is complete; capture only records the next execution.
            // Replaying now would duplicate work whenever a shape was evicted.
            unsafe { self.capture_ready(rows)?; }
        }
        self.ready = Some(rows);
        self.tokens.extend_from_slice(tokens);
        self.binding = Some(binding);
        self.output()
    }
    /// Query-A/norm stays on the layer owner; query-B uses both RTX shards.
    /// The projection owner appends rotary before its final wait, preserving
    /// the ordinary query output/binding contract for attention and the indexer.
    /// # Safety
    /// Same producer ownership contract as execute_tokens_prepared_cooperative.
    /// Projection weights must be this layer's checkpoint query-B shards and
    /// its gathered output device must match this query owner.
    pub(crate) async unsafe fn execute_tokens_tp2_prepared_cooperative(
        &mut self, tokens: &[u64], projection: &mut crate::v41_projection_tp2::Wave<'_, '_>,
        prepare: impl FnOnce(*mut std::ffi::c_void, Ds41rtDeviceBuffer)->Result<()>)
        ->Result<AttentionQueryOutput<'_>> {
        ensure!(tokens.len()<=self.capacity as usize,"query rows exceed capacity");
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p|p<1048576)
            && projection.kind()==crate::v41_projection_tp2::Kind::QueryB
            && projection.output_device().id==self.input().device_id
            && std::ptr::eq(projection.output_device().library,self.stream.library),"TP2 query origin differs");
        let rows=tokens.len() as u32;
        let binding=QueryBinding::new(self.weights.layer)?;
        for (dst,token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        struct Drain<'s,'a> { stream:&'s LoadStream<'a>, complete:bool }
        impl Drop for Drain<'_,'_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error)=unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
                        tracing::error!(%error,"draining interrupted TP2 query prefix");
                    }
                }
            }
        }
        let mut guard=Drain { stream:&self.stream,complete:false };
        let graph=self.tp2_prefix_graphs.get_shape(self.weights.layer,self.weights,rows);
        unsafe {
            self.stream.library.copy_host_buffer_h2d_async(self.positions(),self.position_staging.buffer,
                tokens.len()*8,self.stream.raw)?;
            prepare(self.stream.raw,self.input())?;
            if let Some((graph,_))=graph { self.stream.library.cuda_graph_launch(graph,self.stream.raw)?; }
            else { self.enqueue_rank(rows)?; }
            projection.execute_after(self.weights.layer,rows,self.b(2),Some(self.stream.raw),|projected,stream| {
                self.stream.library.copy_d2d_async(self.b(3),projected,rows as usize*ROW_BYTES[3],stream)?;
                #[cfg(test)]
                self.stream.library.copy_d2d_async(self.b(4),projected,rows as usize*ROW_BYTES[3],stream)?;
                self.norm.rope(self.b(3),self.b(6),self.b(3),rows,64,false,stream)
            }).await?;
        }
        if graph.is_none() {
            unsafe { self.stream.library.cuda_graph_begin_capture(self.stream.raw)?; }
            let queued=unsafe { self.enqueue_rank(rows) };
            let captured=unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
            match (queued,captured) {
                (Ok(()),Ok(graph))=> {
                    if let Err(error)=unsafe { self.tp2_prefix_graphs.insert(self.weights.layer,self.weights,rows,graph) } {
                        unsafe { self.stream.library.cuda_graph_exec_destroy(graph)?; }return Err(error);
                    }
                },
                (Err(error),Ok(graph))=> { unsafe { self.stream.library.cuda_graph_exec_destroy(graph)?; }return Err(error); },
                (Err(error),Err(_)) | (Ok(()),Err(error))=>return Err(error),
            }
        }
        guard.complete=true;drop(guard);
        self.ready=Some(rows);self.tokens.extend_from_slice(tokens);self.binding=Some(binding);
        self.output()
    }
    pub fn output(&self) -> Result<AttentionQueryOutput<'_>> {
        let rows = self.ready.context("attention query output unpublished")? as usize;
        let b = |i| part(self.b(i), 0, rows * ROW_BYTES[i]);
        Ok(AttentionQueryOutput {
            binding: self.binding,
            tokens: &self.tokens,
            layer: self.weights.layer,
            rows,
            hidden: b(0),
            raw_rank: b(1),
            normalized_rank: b(2),
            #[cfg(test)]
            projected: b(4),
            #[cfg(test)]
            qb_scratch: self.scratch.get(1).map(|scratch|scratch.buffer),
            rotated: b(3),
            positions: b(5),
            frequencies: b(6),
            _owner: PhantomData,
        })
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graphs.enable_small_shapes(); self.tp2_prefix_graphs.enable_small_shapes(); }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.weights.layer)?;
            self.tp2_prefix_graphs.remove(self.weights.layer)?;
        }
        Ok(())
    }
}
impl Drop for AttentionQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|()| unsafe {
                let first=self.graphs.clear();let second=self.tp2_prefix_graphs.clear();first.and(second)
            })
        {
            tracing::error!(%error,"draining attention query graph");
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
    fn checkpoint_tp2_query_preserves_rank_and_rotary() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog=ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        let devices=[Device { library:&library,id:0 },Device { library:&library,id:1 }];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let read=|device:Device,output:AttentionQueryOutput<'_>|->Result<Vec<Vec<u8>>> {
            [output.hidden,output.raw_rank,output.normalized_rank,output.projected,
                output.rotated,output.positions,output.frequencies].into_iter().map(|view| {
                    let mut bytes=vec![0;view.bytes];device.run(||library.copy_d2h(&mut bytes,view))?;Ok(bytes)
                }).collect()
        };
        for (owner,layer) in [(0,2),(1,20)] {
            let device=devices[owner];let capacity=16;
            let weights=device.own(||AttentionQueryWeights::load(&library,&catalog,layer,
                AttentionQueryWeights::device_bytes(&library,&catalog,layer)?,1<<20))?;
            let mut reference=device.own(||weights.wave(capacity,usize::MAX))?;
            let split_weights=device.own(||AttentionQueryWeights::load_with_split(&library,&catalog,layer,
                AttentionQueryWeights::device_bytes_with_split(&library,&catalog,layer,true)?,1<<20,true))?;
            assert!(split_weights.tensors.get(&split_weights.names[2]).is_err());
            assert_eq!(split_weights.scales.len(),1);
            let mut query=device.own(||split_weights.wave(capacity,usize::MAX))?;
            assert_eq!(query.scratch.len(),1);
            assert!(query.rebind(&weights).is_err());
            let halves=[vec![Weights::load(devices[0],&catalog,layer,Kind::QueryB,0,
                Weights::load_peak_device_bytes(Kind::QueryB))?],
                vec![Weights::load(devices[1],&catalog,layer,Kind::QueryB,1,
                Weights::load_peak_device_bytes(Kind::QueryB))?]];
            let mut projection=Wave::new([&halves[0],&halves[1]],capacity,owner,
                Wave::device_bytes(&library,Kind::QueryB,capacity,owner)?)?;
            let input=Allocation::new(device,capacity as usize*10240)?;
            query.enable_small_graph_shapes();
            for (cycle,rows) in [1,6,16,6,1].into_iter().enumerate() {
                let host:Vec<u8>=(0..capacity as usize*5120).flat_map(|i| {
                    let value=((i+cycle*3)%23) as f32/32.0-0.25;
                    ((value.to_bits()>>16) as u16).to_ne_bytes()
                }).collect();
                device.run(||library.copy_h2d(input.buffer,&host))?;
                let tokens:Vec<u64>=(0..rows).map(|i|(cycle*8192+i) as u64).collect();
                let prepare=|stream,destination|unsafe {
                    library.copy_d2d_async(destination,input.buffer,rows*10240,stream)
                };
                let full=runtime.block_on(device.future(unsafe {
                    reference.execute_tokens_prepared_cooperative(&tokens,prepare)
                }))?;
                let expected=read(device,full)?;
                // A dropped future cannot publish a partially completed query.
                let cancelled=runtime.block_on(async {
                    let mut pending=std::pin::pin!(device.future(unsafe {
                        query.execute_tokens_tp2_prepared_cooperative(&tokens,&mut projection,prepare)
                    }));
                    std::future::poll_fn(|cx| {
                        use std::future::Future;
                        std::task::Poll::Ready(pending.as_mut().poll(cx).is_pending())
                    }).await
                });
                if cancelled { assert!(query.output().is_err()); }
                let actual=runtime.block_on(device.future(unsafe {
                    query.execute_tokens_tp2_prepared_cooperative(&tokens,&mut projection,prepare)
                }))?;
                assert_eq!(actual.tokens()?,tokens);
                actual.binding()?;
                let actual=read(device,actual)?;
                for index in [0,1,2,5,6] { assert_eq!(actual[index],expected[index]); }
                let mut max_error=0f32;
                for index in [3,4] {
                    for (a,b) in actual[index].chunks_exact(2).zip(expected[index].chunks_exact(2)) {
                        let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                        let (a,b)=(decode(a),decode(b));
                        ensure!(a.is_finite() && b.is_finite() && (a-b).abs()<=1e-4,
                            "TP2 query differs: layer={layer} rows={rows} output={index} {a} vs {b}");
                        max_error=max_error.max((a-b).abs());
                    }
                }
                eprintln!("TP2 complete query layer={layer} rows={rows} cycle={cycle} max_error={max_error}");
            }
        }
        Ok(())
    }
}
