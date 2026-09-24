//! Queued accepted SWA writes retain their lane's producer and upload storage.
use super::*;

pub(super) struct WriteReservation {
    flags: std::rc::Rc<std::cell::Cell<u16>>,
    mask: u16,
}
impl Drop for WriteReservation {
    fn drop(&mut self) { self.flags.set(self.flags.get() & !self.mask); }
}
pub(super) struct PendingCommit {
    prepared: Prepared,
    ends: Vec<u64>,
    accepted: Vec<u32>,
    /// A batched multi-layer store runs on this stream instead of the wave's.
    external: Option<*mut c_void>,
    _reservation: WriteReservation,
}
/// One layer's part of a batched window commit (see `stage_batched_commit`).
pub(crate) struct BatchedWindowCommit {
    pub destinations: Vec<u64>,
    pub ends: Vec<(u64, u64)>,
    pub layer: ds41rt_ffi::V41KvStoreLayer,
    pub device: i32,
}
impl<'a> WindowWave<'_, 'a> {
    /// Configure before production. Each wave owns independent publication
    /// events, while waves using disjoint request slots share replica storage.
    pub fn enable_replica(&mut self, state: &WindowState<'a>,
        replica: std::rc::Rc<super::replica::WindowReplica<'a>>) -> Result<()> {
        ensure!(self.ready.is_none() && self.pending_query.is_none() && self.pending_commit.is_none()
            && self.replica.is_none(), "window replica requires an unused wave");
        replica.validate_owner(state)?;
        ensure!(state.layer==self.weights.layer && state.values.buffer.device_id==self.values.buffer.device_id
            && std::ptr::eq(state.ends.library,self.stream.library),
            "window replica producer differs");
        let source=crate::v41_memory::device::Device { library:self.stream.library,id:self.values.buffer.device_id };
        let publication=crate::v41_memory::peer_publication::PeerPublication::new(source,replica.device())?;
        self.replica=Some((replica,publication)); Ok(())
    }
    /// # Safety
    /// All consumers of this proposal have drained. Keep state and wave alive
    /// through completion; on any failure drain before invalidating its requests.
    pub unsafe fn enqueue_commit(&mut self, state: &WindowState<'_>, accepted: &[u32]) -> Result<()> {
        unsafe { self.enqueue_commit_on(state, accepted, None).map(|_| ()) }
    }
    /// Host-side part of enqueue_commit for a batched multi-layer store: the
    /// caller launches one store for every layer on `stream`, and this wave's
    /// completion is then observed on that stream. Replicated windows keep the
    /// per-layer path (their publication is ordered on the wave's stream).
    /// # Safety
    /// Same contract as enqueue_commit; the caller enqueues the store on `stream`
    /// before any poll/commit and drains `stream` before releasing this state.
    pub(crate) unsafe fn stage_batched_commit(&mut self, state: &WindowState<'_>, accepted: &[u32],
        stream: *mut c_void) -> Result<BatchedWindowCommit> {
        ensure!(self.replica.is_none() && state.replica.is_none(), "replicated windows commit per layer");
        unsafe { self.enqueue_commit_on(state, accepted, Some(stream)) }?
            .context("batched window commit staged no rows")
    }
    unsafe fn enqueue_commit_on(&mut self, state: &WindowState<'_>, accepted: &[u32],
        external: Option<*mut c_void>) -> Result<Option<BatchedWindowCommit>> {
        ensure!(self.pending_commit.is_none(), "window commit already pending");
        if let Some(configured)=&state.replica {
            ensure!(self.replica.as_ref().is_some_and(|(storage,_)|
                std::rc::Rc::ptr_eq(storage,&configured.storage)),"window producer replica is not configured");
        }
        if let Some((replica,_))=&self.replica { replica.validate_owner(state)?; }
        self.validate_ready(state)?;
        let p = self.ready.take().context("window output unpublished")?;
        ensure!(accepted.len() == p.chunks.len(), "window acceptance count differs");
        let mut destinations = vec![u64::MAX; p.rows];
        let mut ends = vec![];
        for (i, c) in p.chunks.iter().enumerate() {
            ensure!(
                accepted[i] <= c.tokens,
                "window accepted prefix exceeds proposal"
            );
            state.slots[c.lease.slot]
                .version
                .checked_add(1)
                .context("window version exhausted")?;
            let n = accepted[i] as usize;
            for j in n.saturating_sub(128)..n {
                destinations[p.offsets[i] + j] =
                    (c.lease.slot * 128) as u64 + (c.position + j as u64) % 128;
            }
            ends.push(c.position + n as u64);
        }
        let staging = self.staging.bytes_mut();
        for (i, d) in destinations.iter().enumerate() {
            staging[i * 8..i * 8 + 8].copy_from_slice(&d.to_ne_bytes());
        }
        for (i, end) in ends.iter().enumerate() {
            staging[p.rows * 8 + i * 8..p.rows * 8 + i * 8 + 8].copy_from_slice(&end.to_ne_bytes());
        }

        let mask = p.chunks.iter().fold(0u16, |mask, c| mask | (1 << c.lease.slot));
        ensure!(state.writing.get() & mask == 0, "window slot is being committed");
        state.writing.set(state.writing.get() | mask);
        let batched = external.map(|_| {
            let pending_ends = p.chunks.iter().zip(&ends).map(|(c, &end)| (c.lease.slot as u64, end)).collect();
            BatchedWindowCommit {
                destinations: destinations.clone(),
                ends: pending_ends,
                layer: ds41rt_ffi::V41KvStoreLayer {
                    values: self.values.buffer.ptr as u64,
                    scales: self.scales.buffer.ptr as u64,
                    cache: state.values.buffer.ptr as u64,
                    cache_scales: state.scales.buffer.ptr as u64,
                    ends: state.ends.buffer.ptr as u64,
                    capacity: (state.slot_count * 128) as u64,
                    // Filled in by the batch owner once its arena is laid out.
                    destinations: 0,
                    end_pairs: 0,
                },
                device: self.values.buffer.device_id,
            }
        });
        self.pending_commit = Some(PendingCommit { prepared: p, ends, accepted: accepted.to_vec(),
            external, _reservation: WriteReservation { flags: state.writing.clone(), mask } });
        if batched.is_some() { return Ok(batched); }
        let p = &self.pending_commit.as_ref().unwrap().prepared;
        (|| -> Result<()> {
            unsafe {
                self.stream.library.copy_h2d_async(
                    self.destinations.buffer,
                    &self.staging.bytes_mut()[..p.rows * 8],
                    self.stream.raw,
                )?;
                self.kv.store(
                    self.values.buffer,
                    self.scales.buffer,
                    self.destinations.buffer,
                    state.values.buffer,
                    state.scales.buffer,
                    p.rows,
                    state.slot_count * 128,
                    self.stream.raw,
                )?;
                for (i, c) in p.chunks.iter().enumerate() {
                    self.stream.library.copy_h2d_async(
                        slice(state.ends.buffer, c.lease.slot * 8, 8),
                        &self.staging.bytes_mut()[p.rows * 8 + i * 8..p.rows * 8 + i * 8 + 8],
                        self.stream.raw,
                    )?;
                }
                if let Some((replica,publication)) = self.replica.as_mut() {
                    let pending=self.pending_commit.as_ref().unwrap();
                    publication.enqueue(self.stream.raw,|stream| {
                        for (chunk,&end) in pending.prepared.chunks.iter().zip(&pending.ends) {
                            replica.copy_commit(state,chunk.lease,end,stream)?;
                        }
                        Ok(())
                    })?;
                }
            }
            Ok(())
        })().map(|()| None)
    }
    pub(crate) fn has_pending_commit(&self) -> bool { self.pending_commit.is_some() }
    pub(crate) fn has_replica(&self) -> bool { self.replica.is_some() }
    pub fn poll_commit(&self) -> Result<bool> {
        let Some(pending) = &self.pending_commit else { return Ok(true) };
        unsafe { self.stream.library.cuda_stream_query(pending.external.unwrap_or(self.stream.raw)) }
    }
    pub(super) fn validate_pending_commit(&self, state: &WindowState<'_>) -> Result<&Prepared> {
        let pending = self.pending_commit.as_ref().context("window commit absent")?;
        let p = &pending.prepared;
        ensure!(p.owner == state.owner, "queued window owner differs");
        for (i, c) in p.chunks.iter().enumerate() {
            let slot = state.validate_identity(c.lease)?;
            ensure!(state.writing.get() & (1 << slot) != 0
                && state.slots[slot].version == p.versions[i] && state.slots[slot].end == c.position,
                "queued window binding changed");
        }
        Ok(p)
    }
    /// Complete either an already-queued decode write or a direct prefill/C1 write.
    pub fn commit(&mut self, state: &mut WindowState<'_>, accepted: &[u32]) -> Result<()> {
        let direct = self.pending_commit.is_none();
        if direct {
            let launched = unsafe { self.enqueue_commit(state, accepted) };
            if let Err(error) = launched.and(self.synchronize()) {
                self.abort_commit(state)?;
                return Err(error);
            }
        }
        self.validate_pending_commit(state)?;
        ensure!(self.pending_commit.as_ref().unwrap().accepted == accepted, "queued window acceptance changed");
        ensure!(direct || self.poll_commit()?, "window commit incomplete");
        let pending = self.pending_commit.take().unwrap();
        for (c, end) in pending.prepared.chunks.iter().zip(&pending.ends) {
            state.slots[c.lease.slot].end = *end;
            state.slots[c.lease.slot].version += 1;
        }
        Ok(())
    }
    /// Drain before returning write slots to the bank; revoke any touched request.
    pub fn abort_commit(&mut self, state: &mut WindowState<'_>) -> Result<()> {
        if self.pending_commit.is_none() { return Ok(()); }
        let mut drained = self.synchronize();
        if let Some(stream) = self.pending_commit.as_ref().unwrap().external {
            drained = drained.and(unsafe { self.stream.library.cuda_stream_synchronize(stream) });
        }
        let pending = self.pending_commit.take().unwrap();
        let leases = pending.prepared.chunks.iter().map(|c| c.lease).collect::<Vec<_>>();
        drop(pending);
        let revoked = state.invalidate(&leases);
        drained.and(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn write_reservations_release_only_their_slots() {
        let flags = std::rc::Rc::new(std::cell::Cell::new(0b1010));
        let first = WriteReservation { flags: flags.clone(), mask: 0b0010 };
        let peer = WriteReservation { flags: flags.clone(), mask: 0b1000 };
        drop(first);
        assert_eq!(flags.get(), 0b1000);
        drop(peer);
        assert_eq!(flags.get(), 0);
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_CACHE_COMMIT_MODEL and CUDA"]
    fn native_queued_windows_match_direct_and_keep_peer_slots_independent() -> Result<()> {
        queued_windows(false)
    }
    #[test]
    #[ignore = "requires SM peer copy, DS41RT_CACHE_COMMIT_MODEL and two CUDA GPUs"]
    fn native_queued_windows_publish_replicas_before_commit_completion() -> Result<()> {
        queued_windows(true)
    }
    fn queued_windows(replicated: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let model = std::env::var("DS41RT_CACHE_COMMIT_MODEL")?;
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        let weights = WindowWeights::load(&lib, &catalog, 0,
            WindowWeights::device_bytes(&lib, &catalog, 0)?, 1024*1024)?;
        let mut state = WindowState::new(&lib, 0, 2, usize::MAX)?;
        let mut reference = WindowState::new(&lib, 0, 2, usize::MAX)?;
        let replica=if replicated {
            Some(state.enable_replica(
                crate::v41_memory::device::Device { library:&lib,id:1-lib.cuda_get_device()? })?)
        } else { None };
        let first = state.begin_request(0, 11)?;
        let second = state.begin_request(1, 22)?;
        let originals = [reference.begin_request(0, 11)?, reference.begin_request(1, 22)?];
        // Waves drop/drain before either state, including on assertion/error paths.
        let mut waves = [weights.wave(16, usize::MAX)?, weights.wave(16, usize::MAX)?];
        if let Some(replica)=&replica {
            for wave in &mut waves { wave.enable_replica(&state,replica.clone())?; }
        }
        let mut direct = weights.wave(16, usize::MAX)?;
        let chunk = |lease, position| WindowChunk { lease, position, tokens: 3 };
        let input: Vec<u8> = (0..3*5120).flat_map(|i| {
            let value = ((i % 19) as f32 / 19.0).to_bits();
            ((value >> 16) as u16).to_ne_bytes()
        }).collect();
        for (i, lease) in [first, second].into_iter().enumerate() {
            lib.copy_h2d(waves[i].input(), &input)?;
            lib.copy_h2d(direct.input(), &input)?;
            unsafe {
                waves[i].execute(&state, &[chunk(lease, 0)])?;
                direct.execute(&reference, &[chunk(originals[i], 0)])?;
            }
            direct.commit(&mut reference, &[i as u32+2])?;
        }
        unsafe { waves[0].enqueue_commit(&state, &[2])?; }
        assert_eq!(state.end(first)?, 0);
        assert!(state.view(first).is_err());
        assert!(state.release(first).is_err());
        state.view(second)?;
        unsafe { waves[1].enqueue_commit(&state, &[3])?; }
        assert!(state.view(second).is_err());
        waves[0].synchronize()?;
        assert!(waves[0].commit(&mut state, &[3]).is_err());
        waves[0].commit(&mut state, &[2])?;
        assert_eq!(state.end(first)?, 2);
        assert_eq!(state.end(second)?, 0);
        assert!(state.view(second).is_err());
        waves[1].synchronize()?;
        waves[1].commit(&mut state, &[3])?;
        for (i, lease) in [first, second].into_iter().enumerate() {
            let actual = state.view(lease)?;
            let expected = reference.view(originals[i])?;
            assert_eq!(actual.end, expected.end);
            for (a, b, width) in [(actual.values, expected.values, 512), (actual.scales, expected.scales, 16)] {
                let mut left = vec![0; (i+2)*width];
                let mut right = left.clone();
                lib.copy_d2h(&mut left, slice(a, 0, (i+2)*width))?;
                lib.copy_d2h(&mut right, slice(b, 0, (i+2)*width))?;
                assert_eq!(left, right);
            }
            if let Some(replica)=&replica {
                let peer=unsafe { replica.view(&state,lease)? };
                for (a,b,width) in [(actual.values,peer.values,512),(actual.scales,peer.scales,16)] {
                    let mut left=vec![0;(i+2)*width]; let mut right=left.clone();
                    lib.copy_d2h(&mut left,slice(a,0,(i+2)*width))?;
                    replica.device().run(||lib.copy_d2h(&mut right,slice(b,0,(i+2)*width)))?;
                    assert_eq!(left,right);
                }
                let mut end=[0;8];
                replica.device().run(||lib.copy_d2h(&mut end,peer.device_end))?;
                assert_eq!(u64::from_ne_bytes(end),actual.end);
            }
        }
        unsafe {
            waves[0].execute(&state, &[chunk(first, 2)])?;
            waves[0].enqueue_commit(&state, &[2])?;
        }
        waves[0].abort_commit(&mut state)?;
        assert!(state.request_id(first).is_err());
        assert_eq!(state.end(second)?, 3);
        let device_end = state.view(second)?.device_end;
        state.release(second)?;
        assert!(state.view(second).is_err());
        let mut end_bytes = [0; 8];
        lib.copy_d2h(&mut end_bytes, device_end)?;
        assert_eq!(u64::from_ne_bytes(end_bytes), 3); // Release needs no GPU clear.
        let replacement = state.begin_request(1, 33)?;
        lib.copy_d2h(&mut end_bytes, state.view(replacement)?.device_end)?;
        assert_eq!(u64::from_ne_bytes(end_bytes), 0);
        assert_eq!(state.end(replacement)?, 0);
        Ok(())
    }
}
