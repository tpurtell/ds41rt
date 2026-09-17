//! Pre-resolved operations used by asynchronous dual-device chains.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type AddFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, usize, *mut c_void) -> i32;
pub struct V41Bf16Add<'a> { _library: &'a NativeLibrary, launch: AddFn }
impl NativeLibrary {
    pub fn v41_bf16_add(&self) -> Result<V41Bf16Add<'_>> {
        Ok(V41Bf16Add { _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_add_tp2_shared_async")? } })
    }
}
impl V41Bf16Add<'_> {
    /// # Safety
    /// Buffers must be live on the current stream device, with producers ordered
    /// before this operation. No conflicting aliases may access them until done.
    pub unsafe fn launch(&self, a: Ds41rtDeviceBuffer, b: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer, count: usize, stream: *mut c_void) -> Result<()> {
        ensure!(count > 0 && count <= 4096*5120, "invalid BF16 addition extent");
        ensure!(a.device_id == output.device_id && b.device_id == output.device_id,
            "BF16 addition device mismatch");
        for buffer in [a,b,output] {
            ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 2 == 0 && buffer.bytes >= count*2,
                "invalid BF16 addition buffer");
        }
        let status = unsafe {
            (self.launch)(a.ptr.cast(), b.ptr.cast(), output.ptr.cast(), count, stream)
        };
        ensure!(status == 0, "shared TP2 addition failed with CUDA status {status}");
        Ok(())
    }
}

type PeerCopyFn = unsafe extern "C" fn(*mut c_void,*const c_void,u64,*mut c_void)->i32;
/// SM-issued peer copy, avoiding DMA queue coupling across lane-local event waits.
type PeerRowsFn = unsafe extern "C" fn(*mut c_void,*const c_void,u64,u64,u64,u64,*mut c_void)->i32;
pub struct V41PeerCopy<'a> { _library: &'a NativeLibrary, launch: PeerCopyFn, rows: PeerRowsFn }
impl NativeLibrary {
    pub fn v41_peer_copy(&self) -> Result<V41PeerCopy<'_>> {
        let initialize=unsafe { *self.lib.get::<unsafe extern "C" fn()->i32>(b"ds41rt_v41_peer_copy_initialize")? };
        ensure!(unsafe { initialize() }==0,"peer copy initialization failed");
        Ok(V41PeerCopy { _library:self,launch:unsafe { *self.lib.get(b"ds41rt_v41_peer_copy_async")? },
            rows:unsafe { *self.lib.get(b"ds41rt_v41_peer_copy_rows_async")? } })
    }
}
impl V41PeerCopy<'_> {
    /// # Safety
    /// Destination is on the current stream device; source peer access is enabled.
    /// Producers precede this stream, buffers are disjoint and live through
    /// completion, and no conflicting access occurs. Initialize before capture.
    pub unsafe fn launch(&self,destination: Ds41rtDeviceBuffer,source: Ds41rtDeviceBuffer,
        bytes: usize,stream:*mut c_void)->Result<()> {
        ensure!(destination.device_id>=0 && source.device_id>=0 && destination.device_id!=source.device_id,
            "peer copy needs distinct device owners");
        ensure!(bytes>0 && bytes<=destination.bytes && bytes<=source.bytes
            && destination.flags==0 && source.flags==0,"invalid peer copy extent or buffer flags");
        let status=unsafe { (self.launch)(destination.ptr,source.ptr,bytes as u64,stream) };
        ensure!(status==0,"SM peer copy failed with CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Same publication/lifetime rules as launch; local copies are also allowed.
    /// Each row has `width` bytes and begins at its buffer's respective pitch.
    pub unsafe fn launch_rows(&self,destination:Ds41rtDeviceBuffer,source:Ds41rtDeviceBuffer,
        width:usize,rows:usize,destination_pitch:usize,source_pitch:usize,stream:*mut c_void)->Result<()> {
        ensure!(destination.device_id>=0 && source.device_id>=0 && destination.flags==0 && source.flags==0,
            "invalid pitched copy device owner or flags");
        ensure!((1..=4096).contains(&rows) && width>0,"invalid pitched copy shape");
        for (buffer,pitch) in [(destination,destination_pitch),(source,source_pitch)] {
            let bytes=(rows-1).checked_mul(pitch).and_then(|n|n.checked_add(width));
            ensure!(pitch>=width && bytes.is_some_and(|n|n<=buffer.bytes),"pitched copy exceeds buffer");
        }
        let status=unsafe { (self.rows)(destination.ptr,source.ptr,width as u64,rows as u64,
            destination_pitch as u64,source_pitch as u64,stream) };
        ensure!(status==0,"SM pitched copy failed with CUDA status {status}");Ok(())
    }

}
