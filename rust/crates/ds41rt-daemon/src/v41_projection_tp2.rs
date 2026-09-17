//! Output-channel FP8 projection shards with lane-owned exchange storage.
use crate::v41_memory::{HostAllocation, device::{Allocation, Device, Event, Stream}};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Fp8Plan, V41PeerCopy};
use ds41rt_loader::OfficialV41Catalog;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind { QueryB, OutputB }
impl Kind {
    pub fn geometry(self) -> (usize, usize) {
        match self { Self::QueryB => (1280, 32768), Self::OutputB => (8192, 5120) }
    }
    fn name(self) -> &'static str { match self { Self::QueryB => "wq_b", Self::OutputB => "wo_b" } }
}

pub(crate) struct Weights<'a> {
    device: Device<'a>,
    layer: usize,
    rank: usize,
    kind: Kind,
    weight: Allocation<'a>,
    scales: Allocation<'a>,
}
impl<'a> Weights<'a> {
    pub fn device_bytes(kind: Kind) -> usize {
        let (k,n)=kind.geometry(); k*(n/2) + k*(n/2)/32
    }
    pub fn load_peak_device_bytes(kind: Kind) -> usize {
        let (k,n)=kind.geometry(); Self::device_bytes(kind) + k*(n/2)/1024
    }
    pub fn load(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize,
        kind: Kind, rank: usize, budget: usize) -> Result<Self> {
        ensure!(layer<40 && rank<2 && budget>=Self::load_peak_device_bytes(kind),
            "invalid projection shard layer, rank or budget");
        let (k,n)=kind.geometry(); let n=n/2;
        let prefix=format!("layers.{layer}.attn.{}",kind.name());
        let weight_name=format!("{prefix}.weight"); let scale_name=format!("{prefix}.scale");
        ensure!(catalog.tensor(&weight_name)?.metadata.shape==[2*n,k]
            && catalog.tensor(&scale_name)?.metadata.shape==[2*n/32,k/32],
            "projection checkpoint geometry differs");
        let stream=Stream::new(device)?;
        let weight=Allocation::new(device,k*n)?;
        let scales=Allocation::new(device,k*n/32)?;
        let raw_scales=Allocation::new(device,k*n/1024)?;
        let mut staging=HostAllocation::new(device.library,k*n)?;
        for (name,target) in [(&weight_name,&weight),(&scale_name,&raw_scales)] {
            let bytes=catalog.read_coordinator_tp2_into(name,0,rank,staging.bytes_mut(),&mut [])?;
            ensure!(bytes==target.buffer.bytes,"projection shard payload size differs");
            device.run(||device.library.copy_h2d(target.buffer,&staging.bytes_mut()[..bytes]))?;
        }
        let kernel=device.run(||device.library.v41_fp8_matrix_kernel(1,k as u32,n as u32))?;
        let queued=device.run(||unsafe { kernel.pack_scales(raw_scales.buffer,scales.buffer,stream.raw) });
        let drained=stream.drain();queued.and(drained)?;
        Ok(Self { device,layer,rank,kind,weight,scales })
    }
}

struct Rank<'w,'a> {
    weights: &'w [Weights<'a>],
    stream: Stream<'a>,
    plan: V41Fp8Plan<'a>,
    copy: V41PeerCopy<'a>,
    input: Allocation<'a>,
    output: Allocation<'a>,
    scratch: Allocation<'a>,
    alpha: Allocation<'a>,
}
impl<'w,'a> Rank<'w,'a> {
    fn new(weights: &'w [Weights<'a>], capacity: u32) -> Result<Self> {
        let first=weights.first().context("projection shard weights absent")?;
        let device=first.device; let (k,n)=first.kind.geometry();
        ensure!(weights.len()<=40 && weights.iter().all(|w|w.kind==first.kind
            && w.rank==first.rank && w.device.id==device.id
            && std::ptr::eq(w.device.library,device.library)),"projection weight bank differs");
        for (index,w) in weights.iter().enumerate() {
            ensure!(!weights[..index].iter().any(|other|other.layer==w.layer),"duplicate projection layer");
        }
        let plan=device.run(||device.library.v41_fp8_matrix_plan(capacity,k as u32,(n/2) as u32))?;
        let stream=Stream::new(device)?;
        let scratch=Allocation::new(device,plan.info().scratch_bytes as usize)?;
        let alpha=Allocation::new(device,16)?;
        let initialized=device.run(||unsafe { plan.initialize_scratch(scratch.buffer,alpha.buffer,stream.raw) });
        let drained=stream.drain();initialized.and(drained)?;
        Ok(Self { weights,stream,plan,copy:device.run(||device.library.v41_peer_copy())?,
            input:Allocation::new(device,capacity as usize*k*2)?,
            output:Allocation::new(device,capacity as usize*n)?,scratch,alpha })
    }
}

pub(crate) struct Wave<'w,'a> {
    ranks: [Rank<'w,'a>;2],
    output: Allocation<'a>,
    peer_done: Event<'a>,
    input_ready: [Event<'a>;2],
    owner: usize,
    capacity: u32,
    kind: Kind,
}
struct Drain<'s,'w,'a> { wave: &'s mut Wave<'w,'a>, complete: bool }
impl Drop for Drain<'_,'_,'_> {
    fn drop(&mut self) {
        if !self.complete {
            if let Err(error)=self.wave.drain() { tracing::error!(%error,"draining interrupted projection"); }
        }
    }
}
impl<'w,'a> Wave<'w,'a> {
    pub fn kind(&self) -> Kind { self.kind }
    pub fn output_device(&self) -> Device<'a> { self.output.device }
    pub fn device_bytes(library: &ds41rt_ffi::NativeLibrary,kind: Kind,capacity: u32,owner: usize) -> Result<[usize;2]> {
        ensure!(owner<2 && (1..=4096).contains(&capacity),"invalid projection wave geometry");
        let (k,n)=kind.geometry();
        let plan=library.v41_fp8_matrix_plan_info(capacity,k as u32,(n/2) as u32)?;
        let half=plan.scratch_bytes as usize+16+capacity as usize*(k*2+n);
        let mut bytes=[half;2];bytes[owner]+=capacity as usize*n*2;Ok(bytes)
    }
    pub fn new(weights: [&'w [Weights<'a>];2],capacity: u32,owner: usize,budgets: [usize;2]) -> Result<Self> {
        let a=weights[0].first().context("projection rank 0 absent")?;
        let b=weights[1].first().context("projection rank 1 absent")?;
        ensure!(a.rank==0 && b.rank==1 && a.kind==b.kind && a.device.id!=b.device.id
            && std::ptr::eq(a.device.library,b.device.library),"projection ranks differ");
        ensure!(weights[0].len()==weights[1].len() && weights[0].iter().zip(weights[1]).all(|(a,b)|a.layer==b.layer),
            "projection rank layers differ");
        let needed=Self::device_bytes(a.device.library,a.kind,capacity,owner)?;
        ensure!(needed.iter().zip(budgets).all(|(&need,budget)|need<=budget),"projection workspace exceeds budget");
        for (device,peer) in [(a.device,b.device),(b.device,a.device)] {
            device.run(||device.library.cuda_enable_peer(peer.id))?;
        }
        let devices=[a.device,b.device];let (_,n)=a.kind.geometry();
        Ok(Self { ranks:[Rank::new(weights[0],capacity)?,Rank::new(weights[1],capacity)?],
            output:Allocation::new(devices[owner],capacity as usize*n*2)?,
            peer_done:Event::new(devices[1-owner])?,
            input_ready:[Event::new(devices[0])?,Event::new(devices[1])?],owner,capacity,kind:a.kind })
    }
    fn drain(&self) -> Result<()> {
        let first=self.ranks[0].stream.drain();let second=self.ranks[1].stream.drain();first.and(second)
    }
    /// # Safety
    /// Input production is complete. Keep input live/unchanged until completion
    /// or cancellation drains both ranks. Consume the result before reusing wave.
    pub async unsafe fn execute(&mut self,layer: usize,rows: u32,input: Ds41rtDeviceBuffer) -> Result<Ds41rtDeviceBuffer> {
        unsafe { self.execute_after(layer,rows,input,None,|output,_|Ok(output)).await }
    }
    /// Enqueue after an optional producer stream and append a same-device
    /// consumer before the one final cooperative wait.
    /// # Safety
    /// Producer belongs to input.device_id; absent producer means input is ready.
    /// Retain producer/input/consumer storage through completion or cancellation.
    /// Consumer queues only on the supplied stream and must not publish results
    /// before this future completes. It runs on the gathered output's device.
    pub async unsafe fn execute_after<T>(&mut self,layer: usize,rows: u32,input: Ds41rtDeviceBuffer,
        producer: Option<*mut std::ffi::c_void>,
        consume: impl FnOnce(Ds41rtDeviceBuffer,*mut std::ffi::c_void)->Result<T>) -> Result<T> {
        let (k,n)=self.kind.geometry();
        ensure!(rows>0 && rows<=self.capacity && input.bytes>=rows as usize*k*2
            && !input.ptr.is_null() && self.ranks.iter().any(|r|r.stream.device.id==input.device_id),
            "projection input differs");
        let mut guard=Drain { wave:self,complete:false };let wave=&mut *guard.wave;
        let input_rank=wave.ranks.iter().position(|r|r.stream.device.id==input.device_id).unwrap();
        if let Some(producer)=producer {
            wave.ranks[input_rank].stream.device.run(||unsafe {
                wave.ranks[input_rank].stream.device.library.cuda_event_record(wave.input_ready[input_rank].raw,producer)
            })?;
        }
        for rank in &wave.ranks {
            let weights=rank.weights.iter().find(|w|w.layer==layer).context("projection layer absent")?;
            rank.stream.device.run(||unsafe {
                if producer.is_some() {
                    rank.stream.device.library.cuda_stream_wait_event(rank.stream.raw,wave.input_ready[input_rank].raw)?;
                }
                rank.copy.launch_rows(rank.input.buffer,input,k*2,rows as usize,k*2,k*2,rank.stream.raw)?;
                rank.plan.launch(rank.input.buffer,weights.weight.buffer,weights.scales.buffer,
                    rank.scratch.buffer,rank.alpha.buffer,rank.output.buffer,rows,rank.stream.raw)
            })?;
        }
        let owner=&wave.ranks[wave.owner];let peer=&wave.ranks[1-wave.owner];
        wave.peer_done.record(&peer.stream)?;
        owner.stream.device.run(||unsafe {
            owner.stream.device.library.cuda_stream_wait_event(owner.stream.raw,wave.peer_done.raw)?;
            for (i,rank) in wave.ranks.iter().enumerate() {
                let mut dst=wave.output.buffer;
                dst.ptr=dst.ptr.cast::<u8>().add(i*n).cast();dst.bytes-=i*n;
                owner.copy.launch_rows(dst,rank.output.buffer,n,rows as usize,n*2,n,owner.stream.raw)?;
            }Ok(())
        })?;
        let output=Ds41rtDeviceBuffer { bytes:rows as usize*n*2,..wave.output.buffer };
        let result=owner.stream.device.run(||consume(output,owner.stream.raw))?;
        owner.stream.wait().await?;
        guard.complete=true;Ok(result)
    }
}
impl Drop for Wave<'_,'_> {
    fn drop(&mut self) { if let Err(error)=self.drain() { tracing::error!(%error,"draining projection wave"); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_tensors::NativeRtxTensors;
    #[test]
    #[ignore = "requires checkpoint, projection shard AOT and two CUDA GPUs"]
    fn checkpoint_dual_projection_lanes_match_full() -> Result<()> {
        let lib=unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog=ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        let devices=[Device { library:&lib,id:0 },Device { library:&lib,id:1 }];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for kind in [Kind::QueryB,Kind::OutputB] {
            let (k,n)=kind.geometry();let capacity=16;
            let weights=[(0..2).map(|i|Weights::load(devices[0],&catalog,[2,20][i],kind,0,
                Weights::load_peak_device_bytes(kind))).collect::<Result<Vec<_>>>()?,
                (0..2).map(|i|Weights::load(devices[1],&catalog,[2,20][i],kind,1,
                Weights::load_peak_device_bytes(kind))).collect::<Result<Vec<_>>>()?];
            let banks=[weights[0].as_slice(),weights[1].as_slice()];
            assert!(Wave::new(banks,capacity,0,[0,0]).is_err());
            let mut lanes=[Wave::new(banks,capacity,0,Wave::device_bytes(&lib,kind,capacity,0)?)?,
                Wave::new(banks,capacity,1,Wave::device_bytes(&lib,kind,capacity,1)?)?];
            let inputs=[Allocation::new(devices[0],capacity as usize*k*2)?,
                Allocation::new(devices[1],capacity as usize*k*2)?];
            let plan=devices[0].run(||lib.v41_fp8_matrix_plan(capacity,k as u32,n as u32))?;
            let stream=Stream::new(devices[0])?;
            let scratch=Allocation::new(devices[0],plan.info().scratch_bytes as usize)?;
            let alpha=Allocation::new(devices[0],16)?;
            let output=Allocation::new(devices[0],capacity as usize*n*2)?;
            devices[0].run(||unsafe { plan.initialize_scratch(scratch.buffer,alpha.buffer,stream.raw) })?;
            stream.drain()?;
            for layer in [2,20] {
                let names=[format!("layers.{layer}.attn.{}.weight",kind.name()),
                    format!("layers.{layer}.attn.{}.scale",kind.name())];
                let full=devices[0].own(||NativeRtxTensors::load(&lib,&catalog,&names,
                    NativeRtxTensors::plan(&catalog,&names)?,1<<20))?;
                let scales=Allocation::new(devices[0],k*n/32)?;
                devices[0].run(||unsafe { lib.v41_fp8_matrix_kernel(1,k as u32,n as u32)?
                    .pack_scales(full.get().get(&names[1])?,scales.buffer,stream.raw) })?;
                stream.drain()?;
                for (rows,seed) in [(1,0),(6,7),(16,3),(1,5)] {
                    let host:Vec<u8>=(0..capacity as usize*k).flat_map(|i| {
                        let value=((i+seed)%31) as f32/32.0-0.5;
                        ((value.to_bits()>>16) as u16).to_ne_bytes()
                    }).collect();
                    for input in &inputs { input.device.run(||lib.copy_h2d(input.buffer,&host))?; }
                    devices[0].run(||unsafe { plan.launch(inputs[0].buffer,full.get().get(&names[0])?,scales.buffer,
                        scratch.buffer,alpha.buffer,output.buffer,rows,stream.raw) })?;
                    stream.drain()?;
                    let mut expected=vec![0;rows as usize*n*2];
                    devices[0].run(||lib.copy_d2h(&mut expected,Ds41rtDeviceBuffer { bytes:rows as usize*n*2,..output.buffer }))?;
                    // Poll once and discard an unpublished submission, then reuse
                    // the same owner; its guard drains any work already issued.
                    runtime.block_on(async {
                        let mut pending=std::pin::pin!(unsafe { lanes[0].execute(layer,rows,inputs[0].buffer) });
                        std::future::poll_fn(|cx| {
                            use std::future::Future;
                            let _=pending.as_mut().poll(cx);std::task::Poll::Ready(())
                        }).await;
                    });
                    let [left,right]=&mut lanes;
                    let (a,b)=runtime.block_on(async { tokio::join!(
                        unsafe { left.execute(layer,rows,inputs[1].buffer) },
                        unsafe { right.execute(layer,rows,inputs[0].buffer) }) });
                    for (device,result) in devices.into_iter().zip([a?,b?]) {
                        let mut actual=vec![0;expected.len()];device.run(||lib.copy_d2h(&mut actual,result))?;
                        let mut max_error=0f32;
                        for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                            let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                            let (a,b)=(decode(a),decode(b));
                            ensure!(a.is_finite() && b.is_finite() && (a-b).abs()<=1e-4,
                                "projection differs: {kind:?} layer={layer} rows={rows} {a} vs {b}");
                            max_error=max_error.max((a-b).abs());
                        }
                        eprintln!("projection owner {kind:?} layer={layer} rows={rows} gpu={} max_error={max_error}",device.id);
                    }
                }
            }
        }
        Ok(())
    }
}
