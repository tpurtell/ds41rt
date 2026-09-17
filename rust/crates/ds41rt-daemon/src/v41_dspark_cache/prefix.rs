use super::*;
use crate::v41_memory::{SnapshotPool, SnapshotStorage};

pub(crate) struct DsparkPrefix<'a> {
    owner: u64,
    end: u64,
    ring: SnapshotStorage<'a>,
}
impl DsparkPrefix<'_> {
    pub fn end(&self) -> u64 {
        self.end
    }
}
fn slice(mut buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    buffer
}
impl<'a> DsparkWindow<'a> {
    pub fn reserve_prefixes(&mut self, slots: usize) -> Result<usize> {
        ensure!(self.prefix_pool.is_none() && self.slots.iter().all(|s| s.request.is_none()),
            "draft snapshot arena must be installed before admission");
        if slots > 0 { self.prefix_pool = Some(SnapshotPool::new(self.stream.library, V41DsparkCache::SLOT_BYTES, slots)?); }
        Ok(self.prefix_pool.as_ref().map_or(0, SnapshotPool::device_bytes))
    }
    pub fn retain_prefix(&mut self, lease: WindowLease) -> Result<DsparkPrefix<'a>> {
        let prefix = self.copy_prefix(lease, self.stream.raw)?;
        self.synchronize()?;
        Ok(prefix)
    }
    pub fn queue_prefix(&mut self, lane: usize, lease: WindowLease) -> Result<()> {
        let copies = self.prefix_copies.get(lane).context("invalid draft snapshot lane")?;
        ensure!(copies.pending.is_none(), "draft snapshot lane is occupied");
        let stream = copies.stream.raw;
        let slot = self.validate(lease)?;
        let mut used = [false; 16]; used[slot] = true;
        let reservation = self.access.reserve(used)?;
        let prefix = self.copy_prefix(lease, stream)?;
        self.prefix_copies[lane].pending = Some((lease, prefix, reservation));
        Ok(())
    }
    pub fn prefix_ready(&self, lane: usize, lease: WindowLease) -> Result<bool> {
        let copies = self.prefix_copies.get(lane).context("invalid draft snapshot lane")?;
        ensure!(copies.pending.as_ref().is_some_and(|(l, _, _)| *l == lease), "draft snapshot owner differs");
        self.validate(lease)?;
        copies.ready()
    }
    pub fn finish_prefix(&mut self, lane: usize, lease: WindowLease) -> Result<DsparkPrefix<'a>> {
        ensure!(self.prefix_ready(lane, lease)?, "draft snapshot copies are incomplete");
        Ok(self.prefix_copies[lane].pending.take().unwrap().1)
    }
    pub fn abort_prefix(&mut self, lane: usize) -> Result<()> {
        self.prefix_copies.get_mut(lane).context("invalid draft snapshot lane")?.abort()
    }
    fn copy_prefix(&self, lease: WindowLease, stream: *mut c_void) -> Result<DsparkPrefix<'a>> {
        let slot = self.validate(lease)?;
        self.access.readable(slot)?;
        let end = self.slots[slot]
            .end
            .context("cannot retain an unseeded draft window")?;
        let bytes = end.min(128) as usize * V41DsparkCache::ROW_BYTES;
        ensure!(bytes > 0, "cannot retain an empty draft window");
        let ring = SnapshotStorage::new(self.stream.library, bytes, self.prefix_pool.as_ref())?;
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                ring.buffer,
                slice(self.ring.buffer, slot * V41DsparkCache::SLOT_BYTES, bytes),
                bytes,
                stream,
            )
        };
        if let Err(error) = copied {
            unsafe { self.stream.library.cuda_stream_synchronize(stream)?; }
            return Err(error);
        }
        Ok(DsparkPrefix {
            owner: self.owner,
            end,
            ring,
        })
    }
    pub fn restore_prefix(&mut self, lease: WindowLease, prefix: &DsparkPrefix<'a>) -> Result<()> {
        let slot = self.validate(lease)?;
        self.access.writable(slot)?;
        ensure!(
            prefix.owner == self.owner && self.slots[slot].end.is_none(),
            "foreign draft prefix or nonfresh window"
        );
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                slice(
                    self.ring.buffer,
                    slot * V41DsparkCache::SLOT_BYTES,
                    prefix.ring.buffer.bytes,
                ),
                prefix.ring.buffer,
                prefix.ring.buffer.bytes,
                self.stream.raw,
            )
        };
        let drained = self.synchronize();
        if let Err(error) = copied.and(drained) {
            self.slots[slot].request = None;
            return Err(error);
        }
        self.slots[slot].end = Some(prefix.end);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_queued_draft_prefixes_preserve_peers_and_abort() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 128, usize::MAX)?;
        window.reserve_prefixes(3)?;
        for end in [5u64, 128, 129, 1_000_000] {
            let first = window.begin_request(0, 1)?;
            let second = window.begin_request(1, 2)?;
            window.slots[0].end = Some(end); window.slots[1].end = Some(end);
            let bytes = end.min(128) as usize * V41DsparkCache::ROW_BYTES;
            let original: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &original)?;
            lib.copy_h2d(slice(window.ring.buffer, V41DsparkCache::SLOT_BYTES, bytes), &vec![29; bytes])?;
            let direct = window.retain_prefix(first)?;
            window.queue_prefix(0, first)?;
            window.queue_prefix(1, second)?;
            assert!(window.release(first).is_err());
            assert!(window.access.writable(0).is_err());
            assert!(window.prefix_ready(0, second).is_err());
            assert!(window.queue_prefix(1, first).is_err());
            while !window.prefix_ready(1, second)? { std::thread::yield_now(); }
            let peer = window.finish_prefix(1, second)?;
            let mut actual = vec![0; bytes];
            lib.copy_d2h(&mut actual, peer.ring.buffer)?;
            assert_eq!(actual, vec![29; bytes]);
            drop(peer);
            window.queue_prefix(1, second)?;
            window.abort_prefix(1)?;
            window.release(second)?;
            assert!(window.release(first).is_err());
            while !window.prefix_ready(0, first)? { std::thread::yield_now(); }
            let queued = window.finish_prefix(0, first)?;
            lib.copy_d2h(&mut actual, queued.ring.buffer)?;
            assert_eq!(actual, original);
            let mut expected = vec![0; bytes];
            lib.copy_d2h(&mut expected, direct.ring.buffer)?;
            assert_eq!(actual, expected);
            window.release(first)?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_draft_prefix_survives_slot_reuse() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 128, usize::MAX)?;
        window.reserve_prefixes(1)?;
        for end in [5u64, 128, 129, 1000000] {
            let old = window.begin_request(0, 1)?;
            assert!(window.retain_prefix(old).is_err());
            window.slots[0].end = Some(end);
            let bytes = end.min(128) as usize * V41DsparkCache::ROW_BYTES;
            let original: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &original)?;
            let prefix = window.retain_prefix(old)?;
            window.release(old)?;
            let replacement = window.begin_request(0, 2)?;
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &vec![0xff; bytes])?;
            let resumed = window.begin_request(1, 3)?;
            window.restore_prefix(resumed, &prefix)?;
            let mut restored = vec![0; bytes];
            lib.copy_d2h(
                &mut restored,
                slice(window.ring.buffer, V41DsparkCache::SLOT_BYTES, bytes),
            )?;
            assert_eq!(restored, original);
            assert_eq!(window.committed_end(resumed)?, Some(end));
            assert!(window.validate(old).is_err());
            assert!(window.restore_prefix(resumed, &prefix).is_err());
            window.release(replacement)?;
            window.release(resumed)?;
        }
        Ok(())
    }
}

impl<'a> DsparkPrefix<'a> {
    pub fn parts(&self) -> (u64, u64, &SnapshotStorage<'a>) {
        (self.owner, self.end, &self.ring)
    }
    pub fn from_parts(owner: u64, end: u64, ring: SnapshotStorage<'a>) -> Self {
        Self { owner, end, ring }
    }
}
impl<'a> DsparkWindow<'a> {
    pub fn owner(&self) -> u64 {
        self.owner
    }
    pub fn prefix_pool(&self) -> Option<&SnapshotPool<'a>> {
        self.prefix_pool.as_ref()
    }
    pub fn library(&self) -> &'a NativeLibrary {
        self.stream.library
    }
}
