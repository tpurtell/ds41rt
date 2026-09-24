//! Native device/pinned allocation and stream owners shared by model components.
use anyhow::Result;
use ds41rt_ffi::{Ds41rtDeviceBuffer, Ds41rtHostBuffer, NativeLibrary};
use std::ffi::c_void;

#[path = "v41_memory/snapshot.rs"]
mod snapshot;
pub(crate) use snapshot::{SnapshotCopies, SnapshotPool, SnapshotStorage};
#[path = "v41_memory/download.rs"]
mod download;
pub(crate) use download::RowDownload;
#[path = "v41_memory/device.rs"]
pub(crate) mod device;
#[path = "v41_memory/peer_publication.rs"]
pub(crate) mod peer_publication;
#[path = "v41_memory/proposal_replica.rs"]
pub(crate) mod proposal_replica;
#[path = "v41_memory/chain.rs"]
pub(crate) mod chain;

pub(crate) struct DeviceAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: Ds41rtDeviceBuffer,
}
impl<'a> DeviceAllocation<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        Ok(Self {
            library,
            buffer: library.alloc_device_buffer(bytes)?,
        })
    }
}
impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.library.free_device_buffer(&mut self.buffer) {
            tracing::error!(%error, "freeing V4.1 device allocation");
        }
    }
}
pub(crate) struct HostAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: Ds41rtHostBuffer,
}
impl<'a> HostAllocation<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        let value = Self {
            library,
            buffer: library.alloc_host_buffer(bytes)?,
        };
        // Padding must also be initialized before copying the contiguous arena.
        unsafe {
            std::ptr::write_bytes(value.buffer.ptr.cast::<u8>(), 0, bytes);
        }
        Ok(value)
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr.cast::<u8>(), self.buffer.bytes) }
    }
    /// Read-only view of the same pinned bytes; readers must have joined.
    pub(crate) fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.ptr.cast::<u8>(), self.buffer.bytes) }
    }
}
impl Drop for HostAllocation<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.library.free_host_buffer(&mut self.buffer) {
            tracing::error!(%error, "freeing V4.1 pinned staging");
        }
    }
}
pub(crate) struct LoadStream<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) raw: *mut c_void,
}
impl LoadStream<'_> {
    /// Rebinding requires a completed owner, not a hidden host-thread wait.
    #[track_caller]
    pub(crate) fn require_complete(&self) -> Result<()> {
        let caller = std::panic::Location::caller();
        anyhow::ensure!(unsafe { self.library.cuda_stream_query(self.raw)? },
            "cannot rebind an unfinished V4.1 stream ({}:{})", caller.file(), caller.line());
        Ok(())
    }
    /// Yield the owner thread while retaining stream/buffer ownership. Cancellation
    /// and errors still drain before the caller can release queued input storage.
    pub(crate) async fn wait(&self) -> Result<()> {
        struct Drain<'s, 'a> { stream: &'s LoadStream<'a>, complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error) = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
                        tracing::error!(%error, "draining cancelled V4.1 stream wait");
                    }
                }
            }
        }
        let mut guard = Drain { stream: self, complete: false };
        while !unsafe { self.library.cuda_stream_query(self.raw)? } {
            tokio::task::yield_now().await;
        }
        guard.complete = true;
        Ok(())
    }
}
impl Drop for LoadStream<'_> {
    fn drop(&mut self) {
        // Owners must drop this stream before releasing buffers used by queued work.
        if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.raw) } {
            tracing::error!(%error, "draining V4.1 loading stream");
        }
        if let Err(error) = unsafe { self.library.cuda_stream_destroy(self.raw) } {
            tracing::error!(%error, "destroying V4.1 loading stream");
        }
    }
}
