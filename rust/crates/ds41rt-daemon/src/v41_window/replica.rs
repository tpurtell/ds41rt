//! Local FP8 window storage for the peer attention heads.
use super::*;
use crate::v41_memory::device::{Allocation, Device};
use ds41rt_ffi::V41PeerCopy;

pub(crate) struct WindowReplica<'a> {
    copy: V41PeerCopy<'a>,
    values: Allocation<'a>,
    scales: Allocation<'a>,
    ends: Allocation<'a>,
    owner: u64,
}
impl<'a> WindowReplica<'a> {
    pub fn device(&self) -> Device<'a> { self.values.device }
    pub fn validate_owner(&self, state: &WindowState<'_>) -> Result<()> {
        ensure!(self.owner==state.owner,"foreign window replica"); Ok(())
    }
    pub fn new(state: &WindowState<'a>, device: Device<'a>) -> Result<Self> {
        ensure!(std::ptr::eq(state.ends.library,device.library)
            && state.values.buffer.device_id!=device.id,"window replica requires a peer device");
        ensure!(state.slots.iter().all(|slot|slot.request.is_none()) && state.writing.get()==0,
            "window replica requires no live requests");
        device.run(||device.library.cuda_enable_peer(state.values.buffer.device_id))?;
        let ends=Allocation::new(device,state.ends.buffer.bytes)?;
        device.run(||device.library.copy_h2d(ends.buffer,&vec![0;ends.buffer.bytes]))?;
        Ok(Self { copy:device.run(||device.library.v41_peer_copy())?,
            values:Allocation::new(device,state.values.buffer.bytes)?,
            scales:Allocation::new(device,state.scales.buffer.bytes)?,ends,owner:state.owner })
    }
    /// # Safety
    /// The peer stream follows the source's accepted ring writes and end update.
    /// Hold the source write reservation and all storage until publication
    /// completes. Use the lane's PeerPublication to cover errors/cancellation.
    pub unsafe fn copy_commit(&self,state:&WindowState<'_>,lease:WindowLease,end:u64,
        stream:*mut c_void)->Result<()> {
        ensure!(state.owner==self.owner,"foreign window replica");
        let slot=state.validate_identity(lease)?;
        ensure!(state.writing.get() & (1<<slot)!=0,"window replica commit is not reserved");
        let old=state.slots[slot].end;
        ensure!(end>=old && end<=1048576,"invalid window replica append extent");
        unsafe { self.copy_range(state,slot,old,end,stream) }
    }
    /// Publish a newly restored or reset ring, including an empty bounded replay.
    /// # Safety
    /// Source restore/reset is ordered before this peer stream. Retain the lease
    /// and both allocations until completion; drain before allowing consumers.
    pub unsafe fn copy_restored(&self,state:&WindowState<'_>,lease:WindowLease,
        stream:*mut c_void)->Result<()> {
        ensure!(state.owner==self.owner,"foreign window replica");
        let slot=state.validate(lease)?;
        unsafe { self.copy_range(state,slot,state.slots[slot].begin,state.slots[slot].end,stream) }
    }
    unsafe fn copy_range(&self,state:&WindowState<'_>,slot:usize,begin:u64,end:u64,
        stream:*mut c_void)->Result<()> {
        self.values.device.run(|| {
            for (offset,rows) in prefix::spans(begin,end) {
                for (destination,source,width) in [(self.values.buffer,state.values.buffer,512),
                    (self.scales.buffer,state.scales.buffer,16)] {
                    let base=(slot*128+offset)*width;
                    unsafe { self.copy.launch(slice(destination,base,rows*width),
                        slice(source,base,rows*width),rows*width,stream)?; }
                }
            }
            // An end becomes visible only after the complete accepted suffix.
            unsafe { self.copy.launch(slice(self.ends.buffer,slot*8,8),
                slice(state.ends.buffer,slot*8,8),8,stream) }
        })
    }
    /// # Safety
    /// Publication for this exact source lease has completed; keep the source
    /// lease and replica alive and immutable through all attention consumers.
    pub unsafe fn view<'s>(&'s self,state:&'s WindowState<'_>,lease:WindowLease)->Result<WindowCacheView<'s>> {
        ensure!(state.owner==self.owner,"foreign window replica");
        let slot=state.validate(lease)?;
        Ok(WindowCacheView { values:slice(self.values.buffer,slot*128*512,128*512),
            scales:slice(self.scales.buffer,slot*128*16,128*16),
            device_end:slice(self.ends.buffer,slot*8,8),end:state.slots[slot].end,
            begin:state.slots[slot].begin,_owner:PhantomData })
    }
    /// Bind peer storage to an existing proposal without creating a new snapshot.
    /// # Safety
    /// Both peer proposal planes contain this exact proposal's published rows at
    /// their original offsets. Keep their allocation owners alive and immutable
    /// until all consumers drain, including cold graph preparation and errors.
    /// The committed replica publication required by view must also be complete.
    pub unsafe fn proposal<'s>(&'s self,state:&'s WindowState<'_>,
        original:&'s WindowProposal<'_>,values:Ds41rtDeviceBuffer,
        scales:Ds41rtDeviceBuffer)->Result<WindowProposal<'s>> {
        let cache=unsafe { self.view(state,original.binding.lease)? };
        ensure!(original.layer==state.layer
            && state.request_id(original.binding.lease)?==original.request
            && cache.end==original.cache.end && cache.begin==original.cache.begin
            && original.first==cache.end,
            "window replica proposal snapshot differs");
        ensure!(values.device_id==cache.values.device_id && scales.device_id==values.device_id
            && original.capacity.checked_mul(512).is_some_and(|n|values.bytes>=n)
            && original.capacity.checked_mul(16).is_some_and(|n|scales.bytes>=n)
            && !values.ptr.is_null() && !scales.ptr.is_null(),
            "window replica proposal storage differs");
        Ok(WindowProposal { cache,values,scales,capacity:original.capacity,
            request:original.request,layer:original.layer,binding:original.binding,
            first:original.first,tokens:original.tokens,offset:original.offset,_wave:PhantomData })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::{device::Stream,peer_publication::PeerPublication};

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB with SM peer copy and two CUDA GPUs"]
    fn window_replica_wrap_restore_and_bounded_replay() -> Result<()> {
        let lib=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        for gpu in 0..2 {
            let owner=Device { library:&lib,id:gpu };
            let peer=Device { library:&lib,id:1-gpu };
            owner.run(|| {
                let mut state=WindowState::new(&lib,0,2,WindowState::device_bytes(0,2)?)?;
                let replica=WindowReplica::new(&state,peer)?;
                let producer=Stream::new(owner)?;
                let mut publication=PeerPublication::new(owner,peer)?;
                peer.run(|| {
                    lib.copy_h2d(replica.values.buffer,&vec![0xcd;replica.values.buffer.bytes])?;
                    lib.copy_h2d(replica.scales.buffer,&vec![0xcd;replica.scales.buffer.bytes])
                })?;
                let lease=state.begin_request(0,1)?;
                let mut expected_values=vec![0xcd;128*512];
                let mut expected_scales=vec![0xcd;128*16];
                let check=|state:&WindowState<'_>,lease,values:&[u8],scales:&[u8]|->Result<()> {
                    let view=unsafe { replica.view(state,lease)? };
                    peer.run(|| {
                        let mut actual=vec![0;values.len()];lib.copy_d2h(&mut actual,view.values)?;
                        assert_eq!(actual,values);
                        let mut actual=vec![0;scales.len()];lib.copy_d2h(&mut actual,view.scales)?;
                        assert_eq!(actual,scales);
                        let mut end=[0u8;8];lib.copy_d2h(&mut end,view.device_end)?;
                        assert_eq!(u64::from_ne_bytes(end),view.end);Ok(())
                    })
                };
                let mut append=|state:&mut WindowState<'_>,lease:WindowLease,new:u64,
                    values:&mut [u8],scales:&mut [u8]|->Result<()> {
                    let slot=state.validate(lease)?;
                    let old=state.slots[slot].end;
                    for position in old..new {
                        let row=position as usize%128;
                        let value=(position%251+1) as u8;
                        let scale=(position*3%251+1) as u8;
                        values[row*512..(row+1)*512].fill(value);
                        scales[row*16..(row+1)*16].fill(scale);
                        lib.copy_h2d(slice(state.values.buffer,(slot*128+row)*512,512),&vec![value;512])?;
                        lib.copy_h2d(slice(state.scales.buffer,(slot*128+row)*16,16),&vec![scale;16])?;
                    }
                    lib.copy_h2d(slice(state.ends.buffer,slot*8,8),&new.to_ne_bytes())?;
                    state.writing.set(1<<slot);
                    unsafe { publication.enqueue(producer.raw,|stream|replica.copy_commit(state,lease,new,stream))?; }
                    producer.drain()?;
                    state.writing.set(0);state.slots[slot].end=new;state.slots[slot].version+=1;
                    check(state,lease,values,scales)
                };
                for end in [5,125,130,400,401] {
                    append(&mut state,lease,end,&mut expected_values,&mut expected_scales)?;
                }
                drop(append);
                assert!(unsafe { replica.copy_commit(&state,lease,402,producer.raw) }.is_err());
                let storage=DeviceAllocation::new(&lib,WINDOW_PREFIX_BYTES)?;
                let prefix=unsafe { state.retain_prefix(lease,storage.buffer,producer.raw)? };
                producer.drain()?;
                let restored=state.begin_request(1,2)?;
                unsafe {
                    state.restore_prefix(restored,&prefix,storage.buffer,producer.raw)?;
                    publication.enqueue(producer.raw,|stream|replica.copy_restored(&state,restored,stream))?;
                }
                producer.drain()?;
                check(&state,restored,&expected_values,&expected_scales)?;
                state.release(lease)?;
                assert!(unsafe { replica.view(&state,lease) }.is_err());
                let replay=state.begin_request(0,3)?;
                state.begin_encoder_replay(replay,900)?;
                unsafe { publication.enqueue(producer.raw,|stream|replica.copy_restored(&state,replay,stream))?; }
                producer.drain()?;
                let view=unsafe { replica.view(&state,replay)? };
                assert_eq!((view.begin,view.end),(900,900));
                check(&state,replay,&expected_values,&expected_scales)?;
                let foreign=WindowState::new(&lib,1,2,WindowState::device_bytes(1,2)?)?;
                assert!(unsafe { replica.view(&foreign,replay) }.is_err());
                Ok(())
            })?;
        }
        Ok(())
    }
}
