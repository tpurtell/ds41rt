//! Reusable private inputs for one lane's peer attention half.
use super::*;
use crate::v41_memory::{device::Allocation,proposal_replica::{ProposalReplica,ProposalFormat}};
use crate::v41_index_selection::IndexSelectionOutput;
use crate::v41_compressor::IndexBinding;
use ds41rt_ffi::{Ds41rtDeviceBuffer,V41PeerCopy};
use std::ffi::c_void;

pub(crate) struct PeerAttentionInputs<'a> {
    device:Device<'a>,
    window:ProposalReplica<'a>,
    sources:[ProposalReplica<'a>;4],
    source_bindings:[std::cell::Cell<[Option<IndexBinding>;16]>;4],
    selected:Allocation<'a>,
    copy:V41PeerCopy<'a>,
    capacity:usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires compact attention, SM peer copy, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn checkpoint_private_inputs_reach_peer_with_original_bindings()->Result<()> {
        let lib=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog=ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        let map=CachePlacement::encoder_decoder();
        for layer in [2,20] {
            let source=Device { library:&lib,id:map.attention(layer)? as i32 };
            let peer=Device { library:&lib,id:1-source.id };
            let mut bank=BackboneCache::new_replicated(&lib,map,2,[2;4],
                BackboneCache::replicated_device_bytes(map,2,[2;4])?)?;
            let lease=bank.begin_request(0,42)?;
            let batch=bank.plan(&[CacheWork { lease,tokens:5,kind:ExpertV2SourceKind::Prefill }])?;
            let ww=source.own(||crate::v41_window::WindowWeights::load(&lib,&catalog,layer,
                crate::v41_window::WindowWeights::device_bytes(&lib,&catalog,layer)?,1<<20))?;
            let cw=source.own(||crate::v41_compressor::CompressorWeights::load(&lib,&catalog,layer,
                crate::v41_compressor::CompressorWeights::device_bytes(&catalog,layer)?,1<<20))?;
            let mut window=source.own(||ww.wave(16,usize::MAX))?;
            let mut compressed=source.own(||cw.wave(16,usize::MAX))?;
            let qw=source.own(||crate::v41_attention_query::AttentionQueryWeights::load(&lib,&catalog,layer,
                crate::v41_attention_query::AttentionQueryWeights::device_bytes(&lib,&catalog,layer)?,1<<20))?;
            let mut query=source.own(||qw.wave(16,usize::MAX))?;
            let iw=source.own(||crate::v41_index_query::IndexQueryWeights::load(&lib,&catalog,layer,
                crate::v41_index_query::IndexQueryWeights::device_bytes(&lib,&catalog,layer)?,1<<20))?;
            let mut index=source.own(||iw.wave(16,usize::MAX))?;
            let mut selection=source.own(||crate::v41_index_selection::IndexSelectionWave::new(&lib,16,usize::MAX))?;
            let mut full=source.own(||crate::v41_sparse_attention::SparseAttentionWave::new(&lib,16,usize::MAX))?;
            let mut dual=crate::v41_sparse_attention::dual::DualAttentionWave::new([source,peer],16,
                crate::v41_sparse_attention::dual::DualAttentionWave::device_bytes(16)?)?;
            let sink=Allocation::new(source,256)?;
            source.run(||lib.copy_h2d(sink.buffer,&[0;256]))?;
            let runtime=tokio::runtime::Builder::new_current_thread().build()?;
            let inputs=PeerAttentionInputs::new(source,peer,16,PeerAttentionInputs::device_bytes(16)?)?;
            let stream=crate::v41_memory::device::Stream::new(peer)?;
            for seed in [0,7] {
                let host:Vec<u8>=(0..5*5120).flat_map(|i| {
                    let value=((i+seed)%17) as f32/32.0-0.25;
                    ((value.to_bits()>>16) as u16).to_ne_bytes()
                }).collect();
                source.run(|| {
                    lib.copy_h2d(window.input(),&host)?;lib.copy_h2d(compressed.input(),&host)?;
                    unsafe {
                        window.execute(bank.window(&batch,layer)?,&batch.window_chunks(layer)?)?;
                        compressed.execute(bank.source(&batch,layer)?,&batch.source_chunks(layer)?)?;
                    } Ok(())
                })?;
                let original=bank.attention(&batch,layer,&window,Some(&compressed))?;
                source.run(|| {
                    lib.copy_h2d(query.input(),&host)?;
                    unsafe { query.execute_tokens(&[0,1,2,3,4])?; }
                    Ok(())
                })?;
                let q=query.output()?;
                source.run(||unsafe {
                    let iq=index.execute_attention(&q)?;
                    selection.execute(&iq,&original.selection_requests()?,None)?; Ok(())
                })?;
                let selected=selection.output()?;
                let requests=original.attention_requests();
                let expected=source.run(||unsafe {
                    let output=full.execute_query(&q,sink.buffer,&requests,Some(&selected))?;
                    let mut bytes=vec![0;output.values.bytes];lib.copy_d2h(&mut bytes,output.values)?;Ok(bytes)
                })?;
                // Cancel before completing cold preparation or a warm replay,
                // then immediately reuse the same copies and both attention waves.
                drop(unsafe { dual.enqueue_cached(&q,sink.buffer,&bank,&original,Some(&selected))? });
                for _ in 0..2 {
                    let pending=unsafe { dual.enqueue_cached(&q,sink.buffer,&bank,&original,Some(&selected))? };
                    let output=runtime.block_on(pending.complete())?;
                    let mut actual=vec![0;output.values.bytes];
                    source.run(||lib.copy_d2h(&mut actual,output.values))?;
                    assert_eq!(actual.len(),expected.len());
                    let mut max_error=0f32;
                    for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                        let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                        let (a,b)=(decode(a),decode(b));
                        max_error=max_error.max((a-b).abs());
                        assert!(a.is_finite() && b.is_finite() && (a-b).abs()<=0.02+0.02*b.abs(),
                            "dual attention differs: layer={layer} seed={seed} actual={a} expected={b}");
                    }
                    eprintln!("dual cached attention layer={layer} seed={seed} max_abs_error={max_error}");
                    assert_eq!(actual,expected,"compact WMMA attention differs from full-head WMMA");
                }
                // Repeated consumption of the same source must preserve its copy.
                for _ in 0..2 {
                    let mirrored=unsafe { inputs.enqueue(&bank,&original,stream.raw)? };
                    stream.drain()?;
                    let (requests,n)=mirrored.attention_requests();assert_eq!(n,1);
                    let a=&original.windows[0];let b=requests[0].window;
                    assert_eq!(a.binding,b.binding);assert_eq!(a.metadata(4)?,b.metadata(4)?);
                    let compare=|a:Ds41rtDeviceBuffer,b:Ds41rtDeviceBuffer,offset:usize,bytes:usize|->Result<()> {
                        let slice=|buffer:Ds41rtDeviceBuffer|Ds41rtDeviceBuffer {
                            ptr:unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },bytes,..buffer };
                        let mut left=vec![0;bytes];let mut right=left.clone();
                        source.run(||lib.copy_d2h(&mut left,slice(a)))?;
                        peer.run(||lib.copy_d2h(&mut right,slice(b)))?;
                        assert_eq!(left,right);Ok(())
                    };
                    compare(a.values,b.values,0,5*512)?;compare(a.scales,b.scales,0,5*16)?;
                    let a=&original.sources[0];let b=requests[0].source.unwrap();
                    assert_eq!(a.binding(),b.binding());assert_eq!(a.metadata(4)?,b.metadata(4)?);
                    let metadata=a.metadata(4)?;
                    for row in 0..metadata[3] as usize {
                        let physical=metadata[4] as usize+row*metadata[5] as usize;
                        compare(a.kv_values,b.kv_values,physical*256,256)?;
                        compare(a.kv_scales,b.kv_scales,physical*32,32)?;
                    }
                }
            }
        }
        Ok(())
    }
}
pub(crate) struct PeerCacheAttention<'s> {
    windows:[Option<WindowProposal<'s>>;16],
    sources:[Option<IndexProposal<'s>>;16],
    positions:&'s [Vec<u64>],
}
impl PeerCacheAttention<'_> {
    /// Fixed stack storage; callers pass only the first returned count entries.
    pub fn attention_requests(&self)->([AttentionRequest<'_>;16],usize) {
        let n=self.positions.len();
        (std::array::from_fn(|i| {
            let i=if i<n {i} else {0};
            AttentionRequest { window:self.windows[i].as_ref().unwrap(),
                source:self.sources[i].as_ref(),positions:&self.positions[i] }
        }),n)
    }
}
impl<'a> PeerAttentionInputs<'a> {
    pub fn device_bytes(capacity:usize)->Result<usize> {
        Ok(ProposalReplica::device_bytes(capacity,ProposalFormat::WindowFp8)?
            +4*ProposalReplica::device_bytes(capacity,ProposalFormat::CompressedFp4)?+capacity*2048)
    }
    pub fn new(source:Device<'a>,peer:Device<'a>,capacity:usize,budget:usize)->Result<Self> {
        ensure!(Self::device_bytes(capacity)?<=budget,"peer attention inputs exceed budget");
        let source_buffer=||ProposalReplica::new(source,peer,capacity,ProposalFormat::CompressedFp4);
        Ok(Self { device:peer,window:ProposalReplica::new(source,peer,capacity,ProposalFormat::WindowFp8)?,
            sources:[source_buffer()?,source_buffer()?,source_buffer()?,source_buffer()?],
            source_bindings:std::array::from_fn(|_|std::cell::Cell::new([None;16])),selected:Allocation::new(peer,capacity*2048)?,
            copy:peer.run(||peer.library.v41_peer_copy())?,capacity })
    }
    /// # Safety
    /// Original producers and committed replica writes are complete or precede
    /// stream, which belongs to the peer device. Retain this owner, bank and
    /// original views through all consumers. Drain stream on any error/cancellation.
    /// No earlier attention may still consume this lane's window/selection buffers.
    pub unsafe fn enqueue<'s>(&'s self,bank:&'s BackboneCache<'_>,original:&'s CacheAttention<'_>,
        stream:*mut c_void)->Result<PeerCacheAttention<'s>> {
        let n=original.windows.len();
        ensure!(n>0 && n<=16 && original.positions.len()==n
            && (original.sources.is_empty() || original.sources.len()==n),"invalid peer attention requests");
        ensure!(original.positions.iter().map(Vec::len).sum::<usize>()<=self.capacity,
            "peer attention rows exceed capacity");
        let mut windows=std::array::from_fn(|_|None);
        let mut sources=std::array::from_fn(|_|None);
        for (i,proposal) in original.windows.iter().enumerate() {
            let state=bank.windows.get(proposal.layer).context("invalid peer window layer")?;
            let replica=state.replica_ref().context("peer window cache absent")?;
            ensure!(replica.device().id==self.device.id,"peer window device differs");
            unsafe { proposal.copy_peer(&self.window,stream)?; }
            let (values,scales)=self.window.buffers();
            windows[i]=Some(unsafe { replica.proposal(state,proposal,values,scales)? });
        }
        if let Some(first)=original.sources.first() {
            let index=SOURCES.iter().position(|&l|l==first.source_layer).context("invalid peer source layer")?;
            ensure!(original.sources.iter().all(|s|s.source_layer==first.source_layer),"peer source layers differ");
            let bindings=std::array::from_fn(|i|original.sources.get(i).map(IndexProposal::binding));
            let changed=self.source_bindings[index].get()!=bindings;
            // Recopy after any partially failed submission; publish the key last.
            if changed { self.source_bindings[index].set([None;16]); }
            let state=&bank.sources[index];
            let replica=state.replica_ref().context("peer source cache absent")?;
            ensure!(replica.device().id==self.device.id,"peer source device differs");
            let storage=&self.sources[index];
            for (i,proposal) in original.sources.iter().enumerate() {
                if changed { unsafe { proposal.copy_peer(storage,stream)?; } }
                let (values,scales)=storage.buffers();
                sources[i]=Some(unsafe { proposal.peer_attention(state,replica,values,scales)? });
            }
            self.source_bindings[index].set(bindings);
        }
        Ok(PeerCacheAttention { windows,sources,positions:&original.positions })
    }
    /// # Safety
    /// Same stream/owner contract as enqueue. Selection production is complete.
    pub unsafe fn selection<'s>(&'s self,original:&'s IndexSelectionOutput<'_>,stream:*mut c_void)
        ->Result<IndexSelectionOutput<'s>> {
        ensure!(original.rows>0 && original.rows<=self.capacity,"peer selection exceeds capacity");
        let selected=Ds41rtDeviceBuffer { bytes:original.rows*2048,..self.selected.buffer };
        self.device.run(||unsafe { self.copy.launch(selected,original.selected,selected.bytes,stream) })?;
        unsafe { original.peer_attention(selected) }
    }
}
