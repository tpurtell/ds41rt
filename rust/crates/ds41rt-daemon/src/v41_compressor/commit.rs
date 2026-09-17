//! Lane-owned compressed source publication; page claims survive GPU work.
use super::*;
pub(super) struct PendingCommit {
    prepared: Prepared,
    plan: source_cache::IndexPlan,
    completed: Vec<CompressorLatentRow>,
    accepted: Vec<u32>,
}
impl<'a> CompressorWave<'_, 'a> {
    /// Configure once before production. Each lane owns its publication events;
    /// replicas share the authoritative pool's physical page identities.
    pub fn enable_replica(&mut self, state: &CompressorState<'a>,
        replica: std::rc::Rc<SourceReplica<'a>>) -> Result<()> {
        ensure!(self.ready.is_none() && self.pending_query.is_none() && self.pending_commit.is_none()
            && self.replica.is_none(), "source replica requires an unused wave");
        replica.validate_owner(&state.index)?;
        ensure!(state.layer==self.weights.layer && state.index.kv_values.buffer.device_id==self.kv_values.buffer.device_id
            && std::ptr::eq(state.index.kv_values.library,self.stream.library), "source replica producer differs");
        let source=crate::v41_memory::device::Device { library:self.stream.library,id:self.kv_values.buffer.device_id };
        let publication=crate::v41_memory::peer_publication::PeerPublication::new(source,replica.device())?;
        self.replica=Some((replica,publication)); Ok(())
    }
    /// # Safety
    /// Proposal consumers are complete. Keep wave and state alive and drain on
    /// every error before dropping page claims or releasing participating slots.
    pub unsafe fn enqueue_commit(&mut self, state: &CompressorState<'_>, accepted: &[u32]) -> Result<()> {
        ensure!(self.pending_commit.is_none(), "compressor commit already pending");
        if let Some(configured)=&state.index.replica {
            ensure!(self.replica.as_ref().is_some_and(|(storage,_)|
                std::rc::Rc::ptr_eq(storage,&configured.storage)),"compressor producer replica is not configured");
        }
        if let Some((replica,_))=&self.replica { replica.validate_owner(&state.index)?; }
        let prepared = self
            .ready
            .take()
            .context("compressor has no proposal to commit")?;
        ensure!(
            prepared.owner == state.owner && accepted.len() == prepared.chunks.len(),
            "compressor commit binding differs"
        );
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                state.slots[slot].version == prepared.versions[i]
                    && state.slots[slot].end == chunk.position,
                "stale compressor proposal"
            );
            ensure!(
                accepted[i] <= chunk.tokens,
                "compressor acceptance exceeds proposal"
            );
            state.slots[slot]
                .version
                .checked_add(1)
                .context("compressor version exhausted")?;
        }
        let ratio = self.weights.ratio;
        let mut completed = vec![];
        let mut appends = vec![];
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let end = prepared.offsets[i] + accepted[i] as usize;
            completed.extend(
                prepared
                    .completed
                    .iter()
                    .filter(|r| {
                        r.source_row as usize >= prepared.offsets[i]
                            && (r.source_row as usize) < end
                    })
                    .copied(),
            );
            appends.push((
                chunk.lease.slot,
                chunk.position as usize / ratio,
                (chunk.position as usize + accepted[i] as usize) / ratio,
            ));
        }
        let plan = state.index.reserve(&appends)?;
        let mut destinations = vec![u64::MAX; prepared.rows];
        for row in &completed {
            destinations[row.source_row as usize] =
                state
                    .index
                    .destination(&plan, row.lease.slot, row.position as usize / ratio)?;
        }
        // Reuse pinned proposal metadata after execution has drained. Commit is
        // outside the proposal graph because acceptance is only known afterward.
        for (dst, value) in self.staging.bytes_mut()[..prepared.rows * 8]
            .chunks_exact_mut(8)
            .zip(&destinations)
        {
            dst.copy_from_slice(&value.to_ne_bytes());
        }
        let slice = |b: Ds41rtDeviceBuffer, row: usize| Ds41rtDeviceBuffer {
            ptr: unsafe { b.ptr.cast::<u8>().add(row * 2048).cast() },
            bytes: 2048,
            ..b
        };

        self.pending_commit = Some(PendingCommit { prepared, plan, completed, accepted: accepted.to_vec() });
        let pending = self.pending_commit.as_ref().unwrap();
        let prepared = &pending.prepared;
        let plan = &pending.plan;
        let completed = &pending.completed;
        (|| -> Result<()> {
            unsafe { state.index.copy_shared_tails(plan, self.stream.raw)?; }
            if !completed.is_empty() {
                unsafe {
                    self.stream.library.copy_h2d_async(
                        self.cache_destinations.buffer,
                        &self.staging.bytes_mut()[..prepared.rows * 8],
                        self.stream.raw,
                    )?;
                    self.kernel.index_store(
                        self.index_packed.buffer,
                        self.index_scales.buffer,
                        self.cache_destinations.buffer,
                        state.index.packed.buffer,
                        state.index.scales.buffer,
                        prepared.rows,
                        state.index.capacity,
                        self.stream.raw,
                    )?;
                    self.kv.store(
                        self.kv_values.buffer,
                        self.kv_scales.buffer,
                        self.cache_destinations.buffer,
                        state.index.kv_values.buffer,
                        state.index.kv_scales.buffer,
                        prepared.rows,
                        state.index.capacity,
                        self.stream.raw,
                    )?;
                }
            }
            if let Some(pending) = &state.pending {
                for (i, chunk) in prepared.chunks.iter().enumerate() {
                    let count = accepted[i] as usize;
                    if count > 0 && (chunk.position + count as u64) % 2 == 1 {
                        let row = prepared.offsets[i] + count - 1;
                        for (dst, src) in [
                            (pending[0].buffer, self.projected.buffer),
                            (pending[1].buffer, self.scores.as_ref().unwrap().buffer),
                        ] {
                            unsafe {
                                self.stream.library.copy_d2d_async(
                                    slice(dst, chunk.lease.slot),
                                    slice(src, row),
                                    2048,
                                    self.stream.raw,
                                )?;
                            }
                        }
                    }
                }
            }
            unsafe {
                state.index.upload_staged(plan, self.commit_staging.bytes_mut(), self.stream.raw)?;
                if let Some((replica,publication))=self.replica.as_mut() {
                    publication.enqueue(self.stream.raw,|stream|replica.copy_append(&state.index,plan,stream))?;
                }
            }
            Ok(())
        })()

    }
    pub fn poll_commit(&self) -> Result<bool> {
        if self.pending_commit.is_none() { return Ok(true); }
        unsafe { self.stream.library.cuda_stream_query(self.stream.raw) }
    }
    pub(super) fn validate_pending_commit(&self, state: &CompressorState<'_>) -> Result<&Prepared> {
        let pending = self.pending_commit.as_ref().context("source commit absent")?;
        let p = &pending.prepared;
        ensure!(p.owner == state.owner, "queued source owner differs");
        state.index.validate_plan(&pending.plan)?;
        for (i, chunk) in p.chunks.iter().enumerate() {
            let slot = state.validate_identity(chunk.lease)?;
            ensure!(state.slots[slot].version == p.versions[i] && state.slots[slot].end == chunk.position,
                "queued source binding changed");
        }
        Ok(p)
    }
    pub fn commit(&mut self, state: &mut CompressorState<'_>, accepted: &[u32]) -> Result<Vec<CompressorLatentRow>> {
        let direct = self.pending_commit.is_none();
        if direct {
            let launched = unsafe { self.enqueue_commit(state, accepted) };
            if let Err(error) = launched.and(self.synchronize()) {
                if let Err(cleanup) = self.abort_commit(state) { tracing::error!(%cleanup, "revoking failed source commit"); }
                return Err(error);
            }
        }
        self.validate_pending_commit(state)?;
        ensure!(self.pending_commit.as_ref().unwrap().accepted == accepted, "queued source acceptance changed");
        ensure!(direct || self.poll_commit()?, "source commit incomplete");
        let pending = self.pending_commit.take().unwrap();
        state.index.apply(pending.plan);
        for (chunk, &count) in pending.prepared.chunks.iter().zip(accepted) {
            state.slots[chunk.lease.slot].end = chunk.position + u64::from(count);
            state.slots[chunk.lease.slot].version += 1;
        }
        Ok(pending.completed)
    }
    pub fn abort_commit(&mut self, state: &mut CompressorState<'_>) -> Result<()> {
        if self.pending_commit.is_none() { return Ok(()); }
        let drained = self.synchronize();
        let pending = self.pending_commit.take().unwrap();
        let slots = pending.prepared.chunks.iter().map(|c| c.lease.slot).collect::<Vec<_>>();
        drop(pending); // Drain before returning claimed pages and write slots.
        let mut revoked = Ok(());
        for slot in slots {
            state.slots[slot].request = None;
            if let Err(error) = state.index.release(slot) { revoked = Err(error); }
        }
        drained.and(revoked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(state: &CompressorState<'_>, slot: usize, rows: usize, lib: &NativeLibrary) -> Result<Vec<u8>> {
        let index = state.index.view(slot, rows);
        let kv = state.index.kv_view(slot, rows);
        let mut bytes = Vec::new();
        for row in 0..rows {
            let physical = index.pages[row / 256] as usize*256 + row % 256;
            for (buffer, width) in [(index.packed, 64), (index.scales, 4),
                (kv.values, V41Kv::COMPRESSED_VALUE_BYTES), (kv.scales, V41Kv::COMPRESSED_SCALE_BYTES)] {
                let start = bytes.len(); bytes.resize(start+width, 0);
                let source = Ds41rtDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(physical*width).cast() }, bytes: width, ..buffer };
                lib.copy_d2h(&mut bytes[start..], source)?;
            }
        }
        Ok(bytes)
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_CACHE_COMMIT_MODEL and CUDA"]
    fn native_independent_source_commits_match_direct_values_carry_and_abort() -> Result<()> {
        source_commits(false)
    }
    #[test]
    #[ignore = "requires SM peer copy, DS41RT_CACHE_COMMIT_MODEL and two CUDA GPUs"]
    fn native_independent_source_commits_publish_peer_payloads_and_metadata() -> Result<()> {
        source_commits(true)
    }
    fn source_commits(replicated: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let model = std::env::var("DS41RT_CACHE_COMMIT_MODEL")?;
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        for layer in [2, 20] {
            let weights = CompressorWeights::load(&lib, &catalog, layer,
                CompressorWeights::device_bytes(&catalog, layer)?, 1024*1024)?;
            let mut state = CompressorState::new(&lib, layer, 2, 4, usize::MAX)?;
            let mut reference = CompressorState::new(&lib, layer, 2, 4, usize::MAX)?;
            let replica=if replicated {
                Some(state.enable_replica(
                    crate::v41_memory::device::Device { library:&lib,id:1-lib.cuda_get_device()? })?)
            } else { None };
            let leases = [state.begin_request(0, 11)?, state.begin_request(1, 22)?];
            let originals = [reference.begin_request(0, 11)?, reference.begin_request(1, 22)?];
            let mut waves = [weights.wave(16, usize::MAX)?, weights.wave(16, usize::MAX)?];
            if let Some(replica)=&replica {
                for wave in &mut waves { wave.enable_replica(&state,replica.clone())?; }
            }
            let mut direct = weights.wave(16, usize::MAX)?;
            let chunk = |lease, position| CompressorChunk { lease, position, tokens: 5 };
            for i in 0..2 {
                let input: Vec<u8> = (0..5*5120).flat_map(|j| {
                    let value = ((j % 19) as f32 * (i+1) as f32 / 19.0).to_bits();
                    ((value >> 16) as u16).to_ne_bytes()
                }).collect();
                lib.copy_h2d(waves[i].input(), &input)?; lib.copy_h2d(direct.input(), &input)?;
                unsafe {
                    waves[i].execute(&state, &[chunk(leases[i], 0)])?;
                    direct.execute(&reference, &[chunk(originals[i], 0)])?;
                }
                direct.commit(&mut reference, &[i as u32+3])?;
            }
            unsafe {
                waves[0].enqueue_commit(&state, &[3])?;
                waves[1].enqueue_commit(&state, &[4])?;
            }
            assert!(state.release(leases[0]).is_err());
            assert!(state.release(leases[1]).is_err());
            assert_eq!(state.committed_end(leases[0])?, 0);
            assert!(waves[0].clear_graph().is_err());
            waves[1].synchronize()?;
            waves[1].commit(&mut state, &[4])?;
            assert_eq!(state.committed_end(leases[1])?, 4);
            assert_eq!(state.committed_end(leases[0])?, 0);
            waves[0].synchronize()?;
            assert!(waves[0].commit(&mut state, &[2]).is_err());
            waves[0].commit(&mut state, &[3])?;
            for i in 0..2 {
                assert_eq!(read(&state, i, (i+3)/ratio(layer)?, &lib)?,
                    read(&reference, i, (i+3)/ratio(layer)?, &lib)?);
                if let Some(replica)=&replica {
                    let original=state.kv_cache(leases[i])?;
                    let peer=unsafe { replica.view(&state.index,i,original.rows)? };
                    for row in 0..original.rows {
                        let physical=original.pages[row/256] as usize*256+row%256;
                        for (a,b,width) in [(original.values,peer.values,V41Kv::COMPRESSED_VALUE_BYTES),
                            (original.scales,peer.scales,V41Kv::COMPRESSED_SCALE_BYTES)] {
                            let slice=|buffer:Ds41rtDeviceBuffer| Ds41rtDeviceBuffer {
                                ptr:unsafe { buffer.ptr.cast::<u8>().add(physical*width).cast() },bytes:width,..buffer };
                            let mut left=vec![0;width];let mut right=left.clone();
                            lib.copy_d2h(&mut left,slice(a))?;
                            replica.device().run(||lib.copy_d2h(&mut right,slice(b)))?;
                            assert_eq!(left,right);
                        }
                    }
                    let mut rows=[0;8];
                    replica.device().run(||lib.copy_d2h(&mut rows,peer.device_rows))?;
                    assert_eq!(u64::from_ne_bytes(rows),original.rows as u64);
                    let mut pages=vec![0;original.pages.len()*4];
                    let page_bytes=pages.len();
                    replica.device().run(||lib.copy_d2h(&mut pages,
                        Ds41rtDeviceBuffer { bytes:page_bytes,..peer.device_pages }))?;
                    assert_eq!(pages,original.pages.iter().flat_map(|p|p.to_ne_bytes()).collect::<Vec<_>>());
                }
            }
            if let (Some(actual), Some(expected)) = (&state.pending, &reference.pending) {
                for (a, b) in actual.iter().zip(expected) {
                    let mut left = vec![0; 2048]; let mut right = left.clone();
                    lib.copy_d2h(&mut left, Ds41rtDeviceBuffer { bytes: 2048, ..a.buffer })?;
                    lib.copy_d2h(&mut right, Ds41rtDeviceBuffer { bytes: 2048, ..b.buffer })?;
                    assert_eq!(left, right);
                }
            }
            unsafe {
                waves[0].execute(&state, &[chunk(leases[0], 3)])?;
                waves[0].enqueue_commit(&state, &[2])?;
            }
            let prefix = state.committed_proposal(leases[0], 0..3, reserve_source_snapshot()?)?;
            assert_eq!(prefix.cache.rows, 3 / ratio(layer)?);
            assert_eq!(prefix.metadata(2)?[1], 3 / ratio(layer)? as u64);
            assert!(state.committed_proposal(leases[0], 0..4, reserve_source_snapshot()?).is_err());
            if let Some(replica)=&replica {
                let peer=unsafe { prefix.peer_attention(&state,replica,prefix.kv_values,prefix.kv_scales)? };
                assert_eq!(peer.binding(),prefix.binding());
                assert_eq!(peer.metadata(2)?,prefix.metadata(2)?);
                assert_eq!(peer.kv_cache.rows,prefix.kv_cache.rows);
                assert_eq!(peer.kv_values.device_id,replica.device().id);
                assert_eq!(peer.kv_scales.device_id,replica.device().id);
                assert!(peer.metadata(3).is_err());
                // Ordinary mutable/current-proposal views still reject the writer.
                assert!(unsafe { replica.view(&state.index,0,prefix.kv_cache.rows) }.is_err());
            }
            drop(prefix);
            assert!(state.index_cache(leases[0]).is_err());
            assert!(state.release(leases[0]).is_err());
            waves[0].abort_commit(&mut state)?;
            assert!(state.request_id(leases[0]).is_err());
            assert_eq!(state.committed_end(leases[1])?, 4);
            assert_eq!(read(&state, 1, 4/ratio(layer)?, &lib)?, read(&reference, 1, 4/ratio(layer)?, &lib)?);
        }
        Ok(())
    }
}
