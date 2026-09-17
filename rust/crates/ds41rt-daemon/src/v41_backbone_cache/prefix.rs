use super::*;
use crate::v41_compressor::{CompressorPrefix, COMPRESSOR_PREFIX_BYTES};
use crate::v41_memory::{SnapshotPool, SnapshotStorage};
use crate::v41_window::{WindowPrefix, WINDOW_PREFIX_BYTES};
use ds41rt_ffi::Ds41rtDeviceBuffer;

pub(crate) struct BackbonePrefix<'a> {
    owner: u64,
    end: u64,
    tail: SnapshotStorage<'a>,
    windows: Vec<WindowPrefix>,
    sources: Vec<CompressorPrefix>,
}
impl BackbonePrefix<'_> {
    pub fn device_bytes() -> usize {
        40 * WINDOW_PREFIX_BYTES + 4 * COMPRESSOR_PREFIX_BYTES
    }
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
impl<'a> BackboneCache<'a> {
    pub fn install_prefix_pool(&mut self, pool: SnapshotPool<'a>) -> Result<()> {
        ensure!(self.prefix_pool.is_none() && self.requests.iter().all(Option::is_none),
            "snapshot arena must be installed before admission");
        self.prefix_pool = Some(pool);
        Ok(())
    }

    /// Attach a complete-group global prefix and initialize empty encoder
    /// windows at the bounded replay start. Decoder windows remain fresh.
    /// Subsequent replay reads the shared source pages without producing them.
    pub fn restore_encoder_prefix(
        &mut self,
        lease: CacheLease,
        prefix: &BackbonePrefix<'a>,
        end: u64,
        prompt_end: u64,
    ) -> Result<u64> {
        let request = self.request(lease)?;
        ensure!(
            prefix.owner == self.owner
                && end > 0
                && end % 2 == 0
                && end <= prefix.end
                && end <= prompt_end
                && prompt_end <= 1048576
                && request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty(),
            "invalid encoder prefix restore frontier or owner"
        );
        let start = end.saturating_sub(128);
        let windows = request.windows;
        let sources = request.sources;
        let result = (|| -> Result<()> {
            for layer in 0..20 {
                self.windows[layer].begin_encoder_replay(windows[layer], start)?;
            }
            for ((state, lease), saved) in self.sources.iter_mut().zip(sources).zip(&prefix.sources)
            {
                state.restore_compressed_prefix(lease, saved, end)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            if let Err(cleanup) = self.release(&[lease]) {
                tracing::error!(%cleanup, "releasing failed encoder prefix restore");
            }
            return Err(error);
        }
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated fresh admission");
        request.end = end;
        request.phase = CachePhase::EncoderReplay {
            prefix: end,
            target: prompt_end,
            end: start,
        };
        request.version = 1;
        self.committed_end(lease)?;
        Ok(start)
    }

    /// Snapshot only a fully committed request, after all producer/consumer
    /// streams have drained. Global KV/index pages remain shared; only bounded
    /// SWA and pending compressor state is copied into the retained GPU arena.
    pub fn retain_prefix(&mut self, lease: CacheLease, budget: usize) -> Result<BackbonePrefix<'a>> {
        let prefix = self.copy_prefix(lease, budget, self.prefix_stream.raw)?;
        unsafe { self.prefix_stream.library.cuda_stream_synchronize(self.prefix_stream.raw)?; }
        Ok(prefix)
    }
    pub fn queue_prefix(&mut self, lane: usize, lease: CacheLease, budget: usize) -> Result<()> {
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        ensure!(copies.pending.is_none(), "backbone snapshot lane is occupied");
        let stream = copies.stream.raw;
        let prefix = self.copy_prefix(lease, budget, stream)?;
        self.prefix_copies[lane].pending = Some((lease, prefix));
        Ok(())
    }
    pub fn prefix_ready(&self, lane: usize, lease: CacheLease) -> Result<bool> {
        let copies = self.prefix_copies.get(lane).context("invalid snapshot lane")?;
        ensure!(copies.pending.as_ref().is_some_and(|(l, _)| *l == lease), "backbone snapshot owner differs");
        self.request_identity(lease)?;
        copies.ready()
    }
    pub fn finish_prefix(&mut self, lane: usize, lease: CacheLease) -> Result<BackbonePrefix<'a>> {
        ensure!(self.prefix_ready(lane, lease)?, "backbone snapshot copies are incomplete");
        Ok(self.prefix_copies[lane].pending.take().unwrap().1)
    }
    pub fn abort_prefix(&mut self, lane: usize) -> Result<()> {
        self.prefix_copies.get_mut(lane).context("invalid snapshot lane")?.abort()
    }
    fn copy_prefix(
        &mut self,
        lease: CacheLease,
        budget: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<BackbonePrefix<'a>> {
        let end = self.committed_end(lease)?;
        let request = self.request(lease)?;
        ensure!(
            end > 0 && request.phase == CachePhase::Full && request.publication.is_empty(),
            "prefix retention requires a complete request frontier"
        );
        ensure!(
            BackbonePrefix::device_bytes() <= budget,
            "retained backbone tail exceeds budget"
        );
        let tail =
            SnapshotStorage::new(self.prefix_stream.library, BackbonePrefix::device_bytes(), self.prefix_pool.as_ref())?;
        let result = (|| -> Result<_> {
            let windows = self
                .windows
                .iter()
                .zip(request.windows)
                .enumerate()
                .map(|(i, (state, lease))| unsafe {
                    state.retain_prefix(
                        lease,
                        slice(tail.buffer, i * WINDOW_PREFIX_BYTES, WINDOW_PREFIX_BYTES),
                        stream,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let sources = self
                .sources
                .iter()
                .zip(request.sources)
                .enumerate()
                .map(|(i, (state, lease))| unsafe {
                    state.retain_prefix(
                        lease,
                        slice(
                            tail.buffer,
                            40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES,
                            COMPRESSOR_PREFIX_BYTES,
                        ),
                        stream,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((windows, sources))
        })();
        // No await occurs here. On a partial enqueue failure, drain before any
        // copied storage is returned to its arena. Success transfers ownership.
        let (windows, sources) = match result {
            Ok(saved) => saved,
            Err(error) => {
                unsafe { self.prefix_stream.library.cuda_stream_synchronize(stream)?; }
                return Err(error);
            }
        };
        Ok(BackbonePrefix {
            owner: self.owner,
            end,
            tail,
            windows,
            sources,
        })
    }

    /// Restore into a fresh admission. Any partially applied failure revokes
    /// the whole request after draining; no mixed-layer frontier is observable.
    pub fn restore_prefix(&mut self, lease: CacheLease, prefix: &BackbonePrefix<'a>) -> Result<()> {
        self.restore_retained_phase(lease, prefix, CachePhase::Full)
    }

    /// A new suffix covering the whole decoder window needs only the saved
    /// encoder rings and global sources. Preserve odd compressor carry while
    /// leaving decoder rings fresh for the final-window replay.
    pub fn restore_encoder_continuation(&mut self, lease: CacheLease,
        prefix: &BackbonePrefix<'a>, prompt_end: u64) -> Result<()> {
        ensure!(prompt_end <= 1048576 && prompt_end.saturating_sub(prefix.end) >= 128,
            "encoder continuation must cover the final decoder window");
        self.restore_retained_phase(lease, prefix, CachePhase::Encoder { target: prompt_end })
    }

    fn restore_retained_phase(&mut self, lease: CacheLease, prefix: &BackbonePrefix<'a>,
        phase: CachePhase) -> Result<()> {
        let request = self.request(lease)?;
        ensure!(
            prefix.owner == self.owner
                && request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty(),
            "foreign backbone prefix or nonfresh admission"
        );
        let windows = request.windows;
        let sources = request.sources;
        let result = (|| -> Result<()> {
            for (i, ((state, lease), saved)) in self
                .windows
                .iter_mut()
                .zip(windows)
                .zip(&prefix.windows)
                .take(phase.stage().windows().end)
                .enumerate()
            {
                unsafe {
                    state.restore_prefix(
                        lease,
                        saved,
                        slice(
                            prefix.tail.buffer,
                            i * WINDOW_PREFIX_BYTES,
                            WINDOW_PREFIX_BYTES,
                        ),
                        self.prefix_stream.raw,
                    )?;
                }
            }
            for (i, ((state, lease), saved)) in self
                .sources
                .iter_mut()
                .zip(sources)
                .zip(&prefix.sources)
                .enumerate()
            {
                unsafe {
                    state.restore_prefix(
                        lease,
                        saved,
                        slice(
                            prefix.tail.buffer,
                            40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES,
                            COMPRESSOR_PREFIX_BYTES,
                        ),
                        self.prefix_stream.raw,
                    )?;
                }
            }
            Ok(())
        })();
        let drained = unsafe {
            self.prefix_stream
                .library
                .cuda_stream_synchronize(self.prefix_stream.raw)
        };
        if let Err(error) = result.and(drained) {
            if let Err(cleanup) = self.release(&[lease]) {
                tracing::error!(%cleanup, "releasing failed prefix restore");
            }
            return Err(error);
        }
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated prefix admission");
        request.end = prefix.end;
        request.phase = phase;
        request.version = 1;
        self.committed_end(lease)?;
        Ok(())
    }
}

impl<'a> BackbonePrefix<'a> {
    /// Owner, frontier, the arena tail and the per-layer descriptors, for the host cache.
    pub fn parts(&self) -> (u64, u64, &SnapshotStorage<'a>, &[WindowPrefix], &[CompressorPrefix]) {
        (self.owner, self.end, &self.tail, &self.windows, &self.sources)
    }
    /// Rebuild from a host copy: `tail` holds the copied arena bytes.
    pub fn from_parts(
        owner: u64,
        end: u64,
        tail: SnapshotStorage<'a>,
        windows: Vec<WindowPrefix>,
        sources: Vec<CompressorPrefix>,
    ) -> Self {
        Self { owner, end, tail, windows, sources }
    }
}
