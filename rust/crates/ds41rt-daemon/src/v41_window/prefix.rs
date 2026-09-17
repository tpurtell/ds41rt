use super::*;

pub(crate) const WINDOW_PREFIX_BYTES: usize = 128 * 528 + 8;
pub(crate) struct WindowPrefix {
    owner: u64,
    end: u64,
    begin: u64,
}

// Valid ring positions can wrap once. Never copy uninitialized padding from a
// short request or the unpopulated beginning of a bounded decoder replay.
pub(super) fn spans(begin: u64, end: u64) -> impl Iterator<Item = (usize, usize)> {
    let first = begin.max(end.saturating_sub(128));
    let count = (end - first) as usize;
    let offset = first as usize % 128;
    let initial = count.min(128 - offset);
    [(offset, initial), (0, count-initial)].into_iter().filter(|&(_, rows)| rows != 0)
}

impl WindowState<'_> {
    /// # Safety
    /// Drain all producers first. Destination stays alive and exclusively owned
    /// until stream completion, including when this method returns an error.
    pub unsafe fn retain_prefix(
        &self,
        lease: WindowLease,
        destination: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<WindowPrefix> {
        let slot = self.validate(lease)?;
        ensure!(
            destination.bytes == WINDOW_PREFIX_BYTES,
            "window prefix storage size differs"
        );
        let live = self.slots[slot];
        for (offset, rows) in spans(live.begin, live.end) {
            for (source, width, base) in [
                (self.values.buffer, 512, 0),
                (self.scales.buffer, 16, 128 * 512),
            ] {
                unsafe {
                    self.ends.library.copy_d2d_async(
                        slice(destination, base + offset * width, rows * width),
                        slice(source, (slot * 128 + offset) * width, rows * width),
                        rows * width,
                        stream,
                    )?;
                }
            }
        }
        unsafe {
            self.ends.library.copy_d2d_async(
                slice(destination, 128 * 528, 8),
                slice(self.ends.buffer, slot * 8, 8),
                8,
                stream,
            )?;
        }
        Ok(WindowPrefix {
            owner: self.owner,
            end: live.end,
            begin: live.begin,
        })
    }

    /// # Safety
    /// Source and destination stay live until completion. Drain the stream before
    /// observing the restored request or releasing it after any error.
    /// Stream belongs to the current CUDA device; replica publication uses that
    /// device's ordering. Complete a restore before reusing its publication events
    /// on another stream.
    pub unsafe fn restore_prefix(
        &mut self,
        lease: WindowLease,
        prefix: &WindowPrefix,
        source: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        let slot = self.validate(lease)?;
        ensure!(
            prefix.owner == self.owner
                && self.slots[slot].end == 0
                && self.slots[slot].version == 0
                && source.bytes == WINDOW_PREFIX_BYTES,
            "foreign window prefix or nonfresh destination"
        );
        for (offset, rows) in spans(prefix.begin, prefix.end) {
            for (destination, width, base) in [
                (self.values.buffer, 512, 0),
                (self.scales.buffer, 16, 128 * 512),
            ] {
                unsafe {
                    self.ends.library.copy_d2d_async(
                        slice(destination, (slot * 128 + offset) * width, rows * width),
                        slice(source, base + offset * width, rows * width),
                        rows * width,
                        stream,
                    )?;
                }
            }
        }
        unsafe {
            self.ends.library.copy_d2d_async(
                slice(self.ends.buffer, slot * 8, 8),
                slice(source, 128 * 528, 8),
                8,
                stream,
            )?;
        }
        self.slots[slot].begin = prefix.begin;
        self.slots[slot].end = prefix.end;
        self.slots[slot].version = 1;
        unsafe { self.publish_replica_restore(lease,stream)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_window_spans_cover_only_initialized_ring_positions() {
        assert!(spans(0, 0).next().is_none());
        assert_eq!(spans(0, 5).collect::<Vec<_>>(), vec![(0, 5)]);
        assert_eq!(spans(0, 128).collect::<Vec<_>>(), vec![(0, 128)]);
        assert_eq!(spans(0, 130).collect::<Vec<_>>(), vec![(2, 126), (0, 2)]);
        assert_eq!(spans(125, 130).collect::<Vec<_>>(), vec![(125, 3), (0, 2)]);
        assert!(spans(1000, 1000).next().is_none());
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_window_prefix_restores_after_slot_reuse() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let device = crate::v41_memory::device::cache_test_device(&lib)?;
        let mut state = device.own(|| WindowState::new(&lib, 20, 2, usize::MAX))?;
        let saved = DeviceAllocation::new(&lib, WINDOW_PREFIX_BYTES)?;
        let stream = LoadStream {
            library: &lib,
            raw: lib.cuda_stream_create()?,
        };
        for (begin, end) in [(0, 5), (0, 128), (125, 130), (1000, 1133)] {
            let old = state.begin_request(0, 1)?;
            state.slots[0].begin = begin;
            state.slots[0].end = end;
            lib.copy_h2d(slice(state.ends.buffer, 0, 8), &end.to_ne_bytes())?;
            let view = state.view(old)?;
            assert_eq!(view.values.device_id,device.id);
            assert_eq!(saved.buffer.device_id,0);
            let values: Vec<u8> = (0..128 * 512).map(|i| (i % 251) as u8).collect();
            let scales: Vec<u8> = (0..128 * 16).map(|i| (i % 137) as u8).collect();
            lib.copy_h2d(view.values, &values)?;
            lib.copy_h2d(view.scales, &scales)?;
            let prefix = unsafe { state.retain_prefix(old, saved.buffer, stream.raw)? };
            unsafe {
                lib.cuda_stream_synchronize(stream.raw)?;
            }
            state.release(old)?;
            let replacement = state.begin_request(0, 2)?;
            let view = state.view(replacement)?;
            lib.copy_h2d(view.values, &vec![0xff; view.values.bytes])?;
            lib.copy_h2d(view.scales, &vec![0xff; view.scales.bytes])?;
            let resumed = state.begin_request(1, 3)?;
            unsafe {
                state.restore_prefix(resumed, &prefix, saved.buffer, stream.raw)?;
                lib.cuda_stream_synchronize(stream.raw)?;
            }
            let view = state.view(resumed)?;
            assert_eq!((view.begin, view.end), (begin, end));
            for (buffer, expected, width) in
                [(view.values, &values, 512), (view.scales, &scales, 16)]
            {
                for (offset, rows) in spans(begin, end) {
                    let mut bytes = vec![0; rows * width];
                    lib.copy_d2h(&mut bytes, slice(buffer, offset * width, rows * width))?;
                    assert_eq!(bytes, expected[offset * width..(offset + rows) * width]);
                }
            }
            assert!(state.validate(old).is_err());
            assert!(
                unsafe { state.restore_prefix(resumed, &prefix, saved.buffer, stream.raw) }
                    .is_err()
            );
            state.release(replacement)?;
            state.release(resumed)?;
        }
        Ok(())
    }
}

impl WindowPrefix {
    pub fn parts(&self) -> (u64, u64, u64) {
        (self.owner, self.end, self.begin)
    }
    pub fn from_parts(owner: u64, end: u64, begin: u64) -> Self {
        Self { owner, end, begin }
    }
}
