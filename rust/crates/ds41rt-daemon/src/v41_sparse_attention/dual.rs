//! Two local-head waves for one lane; no state is shared with another lane.
use super::*;
use crate::v41_memory::device::{Allocation, Device, DeviceOwner, Event};
use crate::v41_backbone_cache::{BackboneCache,CacheAttention,peer_inputs::PeerAttentionInputs};

pub(crate) struct DualAttentionWave<'a> {
    halves: [DeviceOwner<'a, CompactSparseAttentionWave<'a>>; 2],
    copies: [V41PeerCopy<'a>; 2],
    peer_done: Event<'a>,
    output: Allocation<'a>,
    inputs: PeerAttentionInputs<'a>,
    peer_sink: Allocation<'a>,
    capacity: usize,
    pending: Option<OwnedSubmission>,
    tails: [VecDeque<(*mut c_void, Vec<usize>)>; 40],
}
struct OwnedSubmission { cold:[bool;2],rows:usize,layer:usize }
impl<'a> DualAttentionWave<'a> {
    /// Budgets are in head-half order: the first device owns the gathered output.
    /// Includes private proposals, selection and sink storage on the peer.
    /// Committed cache replicas remain budgeted by the backbone bank.
    pub fn device_bytes(capacity: usize) -> Result<[usize; 2]> {
        let half = CompactSparseAttentionWave::device_bytes(capacity)?;
        Ok([half + capacity*64*1024, half+PeerAttentionInputs::device_bytes(capacity)?+128])
    }
    pub fn new(devices: [Device<'a>; 2], capacity: usize, budgets: [usize; 2]) -> Result<Self> {
        ensure!(devices[0].id != devices[1].id
            && std::ptr::eq(devices[0].library, devices[1].library), "dual attention devices differ");
        let needed = Self::device_bytes(capacity)?;
        ensure!(needed.iter().zip(budgets).all(|(n,b)| *n<=b), "dual attention exceeds budget");
        for i in 0..2 { devices[i].run(|| devices[i].library.cuda_enable_peer(devices[1-i].id))?; }
        let half = CompactSparseAttentionWave::device_bytes(capacity)?;
        Ok(Self {
            halves: [devices[0].own(|| CompactSparseAttentionWave::new(devices[0].library,capacity,half))?,
                devices[1].own(|| CompactSparseAttentionWave::new(devices[1].library,capacity,half))?],
            copies: [devices[0].run(|| devices[0].library.v41_peer_copy())?,
                devices[1].run(|| devices[1].library.v41_peer_copy())?],
            peer_done: Event::new(devices[1])?,
            output: Allocation::new(devices[0],capacity*64*1024)?, capacity,
            inputs: PeerAttentionInputs::new(devices[0],devices[1],capacity,PeerAttentionInputs::device_bytes(capacity)?)?,
            peer_sink: Allocation::new(devices[1],128)?,
            pending: None,
            tails: std::array::from_fn(|_| VecDeque::new()),
        })
    }
    pub fn reserve_decode_rows(&mut self, rows: usize) -> Result<()> {
        for half in &mut self.halves {
            let device=half.device;
            device.run(|| half.get_mut().reserve_decode_rows(rows))?;
        }
        Ok(())
    }
    pub fn enable_small_graph_shapes(&mut self) {
        for half in &mut self.halves { half.enable_small_graph_shapes(); }
    }
    pub fn drain(&mut self) -> Result<()> {
        // Always attempt both drains, even if the first device reports an error.
        let devices=[self.halves[0].device,self.halves[1].device];
        let first = devices[0].run(|| self.halves[0].drain_chain());
        let second = devices[1].run(|| self.halves[1].drain_chain());
        self.pending=None;
        first.and(second)
    }
    /// # Safety
    /// Same external lifetime contract as enqueue_cached. The enclosing lane
    /// must complete or drain this owner before releasing any input/consumer.
    pub unsafe fn enqueue_cached_owned(&mut self,query:&AttentionQueryOutput<'_>,sink:Ds41rtDeviceBuffer,
        bank:&BackboneCache<'_>,cache:&CacheAttention<'_>,selection:Option<&IndexSelectionOutput<'_>>)->Result<()> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        let mut submission=unsafe { self.enqueue_cached(query,sink,bank,cache,selection)? };
        let plan=OwnedSubmission { cold:submission.cold,rows:submission.rows,layer:submission.layer };
        submission.wave=None;
        drop(submission);
        self.pending=Some(plan);Ok(())
    }
    /// # Safety
    /// Preserve enqueue_cached_owned's owners and complete_then's consumer contract.
    pub async unsafe fn complete_owned_then<T>(&mut self,
        consume:impl FnOnce(QueuedSparseAttention,*mut c_void)->Result<T>)->Result<T> {
        let plan=self.pending.take().context("dual attention submission absent")?;
        let pending=PendingDualAttention { wave:Some(self),cold:plan.cold,rows:plan.rows,layer:plan.layer };
        unsafe { pending.complete_then(consume).await }
    }
    /// Complete this lane with a captured projection/FFN continuation. The
    /// enclosing backbone retains its weights and buffers until this wave drops.
    /// Cold tails warm once before capture; warm tails have only the existing
    /// final completion wait and never join another lane.
    /// # Safety
    /// Same ownership contract as complete_owned_then, plus retain all tail
    /// storage/weights until graph destruction and provide an exact identity.
    pub async unsafe fn complete_owned_tail(&mut self, tail: &mut dyn AttentionGraphTail) -> Result<()> {
        let plan=self.pending.as_ref().context("dual attention submission absent")?;
        let (layer,rows)=(plan.layer,plan.rows);
        let mut key=tail.identity();
        key.extend([rows,self.output.buffer.ptr as usize]);
        let cached=self.tails[layer].iter().position(|(_,identity)| identity==&key);
        let graph=cached.map(|index| {
            let entry=self.tails[layer].remove(index).unwrap();
            let graph=entry.0;self.tails[layer].push_back(entry);graph
        });
        let library=self.halves[0].device.library;
        let (attention,stream)=unsafe { self.complete_owned_then(|attention,stream| {
            tail.prepare(stream)?;
            if let Some(graph)=graph {
                tail.replay_state()?;
                library.cuda_graph_launch(graph,stream)?;
            } else { tail.enqueue(&attention,stream)?; }
            Ok((attention,stream))
        }).await? };
        if graph.is_some() { return Ok(()); }
        // The warmup is complete. Re-establish host-side block state, capture
        // the same operations and replay them before publishing FFN output.
        let mut guard=PendingDualAttention { wave:Some(self),cold:[false;2],rows,layer };
        let wave=guard.wave.as_deref_mut().unwrap();
        let owner=wave.halves[0].device;
        owner.run(||unsafe {
            tail.restore_warmup()?;
            library.cuda_graph_begin_capture(stream)?;
            let queued=tail.enqueue(&attention,stream);
            let captured=library.cuda_graph_end_capture(stream);
            let graph=match (queued,captured) {
                (Ok(()),Ok(graph))=>graph,
                (Err(error),Ok(graph))=> { library.cuda_graph_exec_destroy(graph)?;return Err(error); },
                (Err(error),Err(_)) | (Ok(()),Err(error))=>return Err(error),
            };
            // Match the attention owner's bounded shape retention policy.
            let limit=wave.halves[0].graph_limit;
            if wave.tails[layer].len()>=limit {
                let (old,_)=wave.tails[layer].pop_front().unwrap();
                if let Err(error)=library.cuda_graph_exec_destroy(old) {
                    library.cuda_graph_exec_destroy(graph)?;return Err(error);
                }
            }
            wave.tails[layer].push_back((graph,key));
            library.cuda_graph_launch(graph,stream)
        })?;
        owner.future(wave.halves[0].wait_chain()).await?;
        guard.wave=None;
        Ok(())
    }
    /// # Safety
    /// Original query/proposal/selection producers and committed replicas are
    /// complete. Retain bank, cache, query, selection and sink through returned
    /// completion or cancellation. This owner cannot be reused while consumers
    /// still read its previous gathered output.
    pub unsafe fn enqueue_cached<'s>(&'s mut self, query:&AttentionQueryOutput<'_>,
        sink:Ds41rtDeviceBuffer, bank:&BackboneCache<'_>,cache:&CacheAttention<'_>,
        selection:Option<&IndexSelectionOutput<'_>>)->Result<PendingDualAttention<'s,'a>> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        ensure!(query.rows>0 && query.rows<=self.capacity && sink.bytes>=256
            && sink.device_id==self.halves[0].device.id,"dual cached attention input differs");
        let mut pending=PendingDualAttention { wave:Some(self),cold:[false;2],rows:query.rows,layer:query.layer };
        // Guard precedes every copy; metadata views can drop on error only after
        // the underlying streams have drained through this pending owner.
        let wave=pending.wave.as_deref_mut().unwrap();
        let stream=wave.halves[1].chain_stream();
        let peer=wave.halves[1].device;
        let mirrored=unsafe { wave.inputs.enqueue(bank,cache,stream)? };
        let selected=selection.map(|s|unsafe { wave.inputs.selection(s,stream) }).transpose()?;
        peer.run(||unsafe { wave.copies[1].launch(wave.peer_sink.buffer,slice(sink,128,128),128,stream) })?;
        let (original,original_count)=cache.attention_requests_fixed()?;
        let (requests,count)=mirrored.attention_requests();
        let request_slices=[&original[..original_count],&requests[..count]];
        let selections=[selection,selected.as_ref()];
        let sinks=[slice(sink,0,128),wave.peer_sink.buffer];
        for i in 0..2 {
            let device=wave.halves[i].device;
            pending.cold[i]=device.run(||unsafe {
                wave.halves[i].enqueue_split_query_prepared(query,i*32,&wave.copies[i],
                    sinks[i],request_slices[i],selections[i],true,None)
            })?.is_none();
        }
        Ok(pending)
    }
    /// # Safety
    /// Both devices' query/cache/proposal/selection producers have completed or
    /// are ordered before these waves' streams. Requests have identical logical
    /// rows and reference their respective local replicas. Sinks are the matching
    /// checkpoint head halves. Retain every external allocation through complete
    /// or drop of the returned guard; this includes deferred cold preparation.
    pub unsafe fn enqueue<'s>(&'s mut self, query: &AttentionQueryOutput<'_>,
        sinks: [Ds41rtDeviceBuffer; 2], requests: [&[AttentionRequest<'_>]; 2],
        selections: [Option<&IndexSelectionOutput<'_>>; 2]) -> Result<PendingDualAttention<'s,'a>> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        ensure!(query.rows>0 && query.rows<=self.capacity, "dual attention rows exceed capacity");
        ensure!(requests[0].len()==requests[1].len() && requests[0].iter().zip(requests[1]).all(|(a,b)|
            a.window.binding==b.window.binding && a.positions==b.positions
            && a.source.map(|s|s.binding())==b.source.map(|s|s.binding())),
            "dual attention proposal identities differ");
        let mut pending = PendingDualAttention { wave: Some(self), cold: [false;2],
            rows: query.rows, layer: query.layer };
        let wave = pending.wave.as_deref_mut().unwrap();
        for i in 0..2 {
            let device = wave.halves[i].device;
            pending.cold[i] = device.run(|| unsafe {
                wave.halves[i].enqueue_split_query_prepared(query,i*32,&wave.copies[i],
                    sinks[i],requests[i],selections[i],true,None)
            })?.is_none();
        }
        Ok(pending)
    }
}

pub(crate) struct PendingDualAttention<'s,'a> {
    wave: Option<&'s mut DualAttentionWave<'a>>,
    cold: [bool;2],
    rows: usize,
    layer: usize,
}
impl PendingDualAttention<'_,'_> {
    pub async fn complete(self) -> Result<QueuedSparseAttention> {
        unsafe { self.complete_then(|output,_|Ok(output)).await }
    }
    /// Queue consumers after gathering and wait once for the entire lane chain.
    /// # Safety
    /// The callback enqueues only on the supplied stream, on the output device.
    /// External consumer allocations must remain alive through this future's
    /// completion or cancellation, including callback errors. The callback must
    /// not expose GPU results before completion and must not suspend.
    pub async unsafe fn complete_then<T>(mut self,
        consume:impl FnOnce(QueuedSparseAttention,*mut c_void)->Result<T>)->Result<T> {
        let wave = self.wave.as_deref_mut().unwrap();
        // Both warmups were submitted before either wait. Warm graph hits never
        // enter this branch and require no host rendezvous between the halves.
        for i in 0..2 {
            if self.cold[i] {
                let device = wave.halves[i].device;
                unsafe { device.future(wave.halves[i].finish_prepare(None)).await?; }
            }
        }
        let owner = wave.halves[0].device;
        let peer = wave.halves[1].device;
        let stream = wave.halves[0].chain_stream();
        let library = owner.library;
        owner.run(|| unsafe {
            wave.copies[0].launch_rows(wave.output.buffer,wave.halves[0].output.buffer,
                32*1024,self.rows,64*1024,32*1024,stream)
        })?;
        peer.run(|| unsafe { library.cuda_event_record(wave.peer_done.raw,wave.halves[1].chain_stream()) })?;
        owner.run(|| unsafe {
            library.cuda_stream_wait_event(stream,wave.peer_done.raw)?;
            wave.copies[0].launch_rows(slice(wave.output.buffer,32*1024,wave.output.buffer.bytes-32*1024),
                wave.halves[1].output.buffer,32*1024,self.rows,64*1024,32*1024,stream)
        })?;
        let values = slice(wave.output.buffer,0,self.rows*64*1024);
        let result=owner.run(||consume(QueuedSparseAttention { values,rows:self.rows,layer:self.layer },stream))?;
        owner.future(wave.halves[0].wait_chain()).await?;
        self.wave = None;
        Ok(result)
    }
}
impl Drop for PendingDualAttention<'_,'_> {
    fn drop(&mut self) {
        if let Some(wave) = self.wave.as_deref_mut() {
            if let Err(error) = wave.drain() { tracing::error!(%error,"draining dual attention on cancellation"); }
        }
    }
}
impl Drop for DualAttentionWave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() { tracing::error!(%error,"draining dual attention owner"); }
        let owner=self.halves[0].device;
        for entries in &mut self.tails {
            for (graph,_) in entries.drain(..) {
                if let Err(error)=owner.run(||unsafe { owner.library.cuda_graph_exec_destroy(graph) }) {
                    tracing::error!(%error,"destroying dual attention continuation graph");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires compact attention, SM peer copies and two CUDA GPUs"]
    fn dual_attention_gathers_both_head_halves() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for owner in 0..2 {
            let devices=[Device { library:&library,id:owner },Device { library:&library,id:1-owner }];
            let mut wave=DualAttentionWave::new(devices,4,DualAttentionWave::device_bytes(4)?)?;
            let mut buffers=Vec::new();
            for (i,device) in devices.into_iter().enumerate() {
                let values=Allocation::new(device,128*512)?;
                let scales=Allocation::new(device,128*16)?;
                let end=Allocation::new(device,8)?;
                let sink=Allocation::new(device,32*4)?;
                device.run(|| {
                    library.copy_h2d(scales.buffer,&vec![127;128*16])?;
                    library.copy_h2d(end.buffer,&[0;8])?;
                    let s=if i==0 { 0f32 } else { 3f32.ln() };
                    library.copy_h2d(sink.buffer,&s.to_ne_bytes().repeat(32))?;
                    library.copy_h2d(wave.halves[i].query.buffer,&vec![0;4*32*1024])?;
                    library.copy_h2d(wave.halves[i].replay_begins.buffer,&[0;32])?;
                    let metadata:Vec<u8>=(0..4).flat_map(|pos|
                        [0u64,0,4,pos,0,0,0,0,0,0].into_iter().flat_map(u64::to_ne_bytes)).collect();
                    library.copy_h2d(wave.halves[i].metadata.buffer,&metadata)
                })?;
                buffers.push((values,scales,end,sink));
            }
            for (encoded,value,cancel) in [(0x38,1f32,false),(0x40,2f32,true),(0x40,2f32,false),(0,0f32,false)] {
                for i in 0..2 {
                    let (values,scales,end,sink)=&buffers[i];
                    devices[i].run(|| {
                        library.copy_h2d(values.buffer,&vec![encoded;128*512])?;
                        let half=&wave.halves[i];
                        let window=V41SparseWindow { values:values.buffer,scales:scales.buffer,
                            proposals:values.buffer,proposal_scales:scales.buffer,end:end.buffer,
                            proposal_capacity:128,replay_begins:Some(half.replay_begins.buffer) };
                        unsafe { half.kernel.launch(half.query.buffer,sink.buffer,half.metadata.buffer,None,
                            &window,None,half.output.buffer,4,0,None,half.chain_stream()) }
                    })?;
                }
                let pending=PendingDualAttention { wave:Some(&mut wave),cold:[false;2],rows:4,layer:0 };
                if cancel { drop(pending); continue; }
                let output=runtime.block_on(pending.complete())?;
                let mut actual=vec![0;output.values.bytes];
                devices[0].run(||library.copy_d2h(&mut actual,output.values))?;
                for row in 0..4 { for half in 0..2 {
                    let n=(row+1) as f32;
                    let expected=value*n/(n+if half==0 {1.0} else {3.0});
                    for bytes in actual[(row*64+half*32)*1024..(row*64+half*32+32)*1024].chunks_exact(2) {
                        let observed=f32::from_bits(u32::from(u16::from_ne_bytes([bytes[0],bytes[1]]))<<16);
                        assert!((observed-expected).abs()<0.005,"owner={owner} row={row} half={half}: {observed} != {expected}");
                    }
                }}
            }
        }
        Ok(())
    }
}
