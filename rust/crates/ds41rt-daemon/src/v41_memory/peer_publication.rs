//! Lane-owned producer -> peer -> producer ordering for replicated KV commits.
use super::device::{Device, Event, Stream};
use anyhow::{ensure, Result};
use std::ffi::c_void;

pub(crate) struct PeerPublication<'a> {
    peer: Stream<'a>,
    produced: Event<'a>,
    copied: Event<'a>,
}
impl<'a> PeerPublication<'a> {
    pub fn new(source: Device<'a>, destination: Device<'a>) -> Result<Self> {
        ensure!(source.id != destination.id && std::ptr::eq(source.library,destination.library),
            "peer publication requires distinct devices from one library");
        destination.run(|| destination.library.cuda_enable_peer(source.id))?;
        Ok(Self { peer: Stream::new(destination)?, produced: Event::new(source)?, copied: Event::new(destination)? })
    }

    /// Order peer copies after producer writes, and producer completion after
    /// peer copies. Normal enqueue performs no allocation, polling or host wait.
    /// The existing producer completion/cancellation guard covers the round trip.
    ///
    /// # Safety
    /// `producer` is a live stream on the source device. The callback only enqueues
    /// on the supplied peer stream. All referenced allocations, reservations and
    /// this owner survive producer completion, including cancellation. Drain the
    /// prior publication before reusing this owner on another producer stream.
    /// Use a distinct owner for each independent lane. Use SM peer copies here:
    /// DMA copies behind unresolved peer waits can couple lanes on bidirectional
    /// peer-access systems. On enqueue failure the
    /// peer stream drains before callback captures are released.
    pub unsafe fn enqueue(&mut self, producer: *mut c_void,
        mut copies: impl FnMut(*mut c_void) -> Result<()>) -> Result<()> {
        let library = self.peer.device.library;
        struct Drain<'s,'a> { stream: &'s Stream<'a>, armed: bool }
        impl Drop for Drain<'_,'_> {
            fn drop(&mut self) {
                if self.armed {
                    if let Err(error)=self.stream.drain() {
                        tracing::error!(%error,"draining failed peer publication");
                    }
                }
            }
        }
        let mut drain = Drain { stream: &self.peer, armed: true };
        self.produced.device.run(|| unsafe { library.cuda_event_record(self.produced.raw,producer) })?;
        self.peer.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.peer.raw,self.produced.raw)?;
            copies(self.peer.raw)?;
            library.cuda_event_record(self.copied.raw,self.peer.raw)
        })?;
        self.produced.device.run(|| unsafe { library.cuda_stream_wait_event(producer,self.copied.raw) })?;
        drain.armed=false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_memory::device::Allocation;
    use ds41rt_ffi::NativeLibrary;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, libcudart.so.13 and two CUDA GPUs"]
    fn peer_publication_keeps_lanes_independent_and_drains_failures() -> Result<()> {
        struct Runtime(*mut c_void);
        impl Drop for Runtime { fn drop(&mut self) { unsafe { libc::dlclose(self.0); } } }
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release=unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) { std::thread::yield_now(); }
        }
        let runtime=Runtime(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(),libc::RTLD_NOW) });
        ensure!(!runtime.0.is_null(),"CUDA runtime unavailable");
        let symbol=unsafe { libc::dlsym(runtime.0,c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(),"CUDA host callback unavailable");
        let launch: unsafe extern "C" fn(*mut c_void,unsafe extern "C" fn(*mut c_void),*mut c_void)->i32 =
            unsafe { std::mem::transmute(symbol) };
        let lib=unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        for gpu in 0..2 {
            let owner=Device { library:&lib,id:gpu };
            let peer=Device { library:&lib,id:1-gpu };
            let copy=peer.run(|| lib.v41_peer_copy())?;
            let source=[Allocation::new(owner,4096)?,Allocation::new(owner,4096)?];
            let destination=[Allocation::new(peer,4096)?,Allocation::new(peer,4096)?];
            let producers=[Stream::new(owner)?,Stream::new(owner)?];
            let mut publications=[Some(PeerPublication::new(owner,peer)?),Some(PeerPublication::new(owner,peer)?)];
            for lane in 0..2 {
                owner.run(|| lib.copy_h2d(source[lane].buffer,&vec![11+lane as u8;4096]))?;
            }
            let release=AtomicBool::new(false);
            struct Release<'a>(&'a AtomicBool);
            impl Drop for Release<'_> { fn drop(&mut self) { self.0.store(true,Ordering::Release); } }
            let _release=Release(&release);
            let gate=|| owner.run(|| {
                ensure!(unsafe { launch(producers[0].raw,hold,(&release as *const AtomicBool).cast_mut().cast()) }==0,
                    "cannot stall first producer");Ok(())
            });
            gate()?;
            for lane in 0..2 {
                unsafe { publications[lane].as_mut().unwrap().enqueue(producers[lane].raw,|stream|
                    copy.launch(destination[lane].buffer,source[lane].buffer,4096,stream))?; }
            }
            let deadline=std::time::Instant::now()+std::time::Duration::from_secs(3);
            while !owner.run(|| unsafe { lib.cuda_stream_query(producers[1].raw) })? {
                ensure!(std::time::Instant::now()<deadline,"second lane waited for stalled first lane from GPU {gpu}");
                std::thread::yield_now();
            }
            assert!(!owner.run(|| unsafe { lib.cuda_stream_query(producers[0].raw) })?);
            assert!(!release.load(Ordering::Acquire));
            release.store(true,Ordering::Release);
            producers[0].drain()?;
            for lane in 0..2 {
                let mut bytes=vec![0;4096];
                peer.run(|| lib.copy_d2h(&mut bytes,destination[lane].buffer))?;
                assert_eq!(bytes,vec![11+lane as u8;4096]);
            }
            // An enqueue error after issuing copies must drain captured storage.
            let result=unsafe { publications[1].as_mut().unwrap().enqueue(producers[1].raw,|stream| {
                copy.launch(destination[1].buffer,source[1].buffer,4096,stream)?;
                anyhow::bail!("injected publication failure")
            }) };
            assert!(result.is_err());
            assert!(peer.run(|| unsafe { lib.cuda_stream_query(publications[1].as_ref().unwrap().peer.raw) })?);
            // Dropping a queued publication waits for its live source dependency.
            release.store(false,Ordering::Release);gate()?;
            unsafe { publications[0].as_mut().unwrap().enqueue(producers[0].raw,|stream|
                copy.launch(destination[0].buffer,source[0].buffer,4096,stream))?; }
            std::thread::scope(|scope| {
                scope.spawn(|| { std::thread::sleep(std::time::Duration::from_millis(50));
                    release.store(true,Ordering::Release); });
                drop(publications[0].take());
                assert!(release.load(Ordering::Acquire));
            });
            producers[0].drain()?;
        }
        Ok(())
    }
}
