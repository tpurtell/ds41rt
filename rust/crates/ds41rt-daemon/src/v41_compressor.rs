//! CSA2 source weights, immutable proposal execution and accepted-prefix state.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Compressor, V41Kv};
use ds41rt_loader::OfficialV41Catalog;
use std::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};
mod source_cache;
mod commit;
use commit::PendingCommit;
mod prefix;
pub(crate) use prefix::{CompressorPrefix, COMPRESSOR_PREFIX_BYTES};
use source_cache::SourceCache;
pub(crate) use source_cache::{IndexCacheView, KvCacheView, SourcePoolExhausted};
pub(crate) use source_cache::replica::SourceReplica;
static NEXT_PROPOSAL: AtomicU64 = AtomicU64::new(1);
pub(crate) fn reserve_source_snapshot() -> Result<u64> {
    NEXT_PROPOSAL.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| anyhow::anyhow!("compressor proposal IDs exhausted"))
}
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
fn ratio(layer: usize) -> Result<usize> {
    match layer {
        2 | 8 | 14 => Ok(2),
        20 => Ok(1),
        _ => anyhow::bail!("layer is not a V4.1 compressor source"),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompressorLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressorChunk {
    pub lease: CompressorLease,
    pub position: u64,
    pub tokens: u32,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressorLatentRow {
    pub lease: CompressorLease,
    pub source_row: u32,
    /// First token represented by this latent, for later RoPE/cache addressing.
    pub position: u64,
}
#[derive(Clone, Copy, Default)]
struct Slot {
    generation: u64,
    version: u64,
    request: Option<u64>,
    end: u64,
}
pub(crate) struct CompressorState<'a> {
    index: SourceCache<'a>,
    pending: Option<[DeviceAllocation<'a>; 2]>,
    slots: [Slot; 16],
    slot_count: usize,
    layer: usize,
    owner: u64,
}
impl<'a> CompressorState<'a> {
    /// Fully provision every admitted slot, rounding each slot to physical pages.
    pub fn pages_for_context(layer: usize, slots: usize, context: usize) -> Result<usize> {
        ensure!((1..=16).contains(&slots), "invalid compressor slot count");
        ensure!((1..=1048576).contains(&context), "invalid compressor context");
        let pages = context.div_ceil(ratio(layer)?).div_ceil(source_cache::PAGE_ROWS) * slots;
        SourceCache::device_bytes(pages, slots)?;
        Ok(pages)
    }

    pub fn device_bytes(layer: usize, slots: usize, index_pages: usize) -> Result<usize> {
        ensure!((1..=16).contains(&slots), "invalid compressor slot count");
        Ok(SourceCache::device_bytes(index_pages, slots)?
            + if ratio(layer)? == 2 { slots * 4096 } else { 0 })
    }
    pub fn new(
        library: &'a NativeLibrary,
        layer: usize,
        slots: usize,
        index_pages: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(layer, slots, index_pages)? <= budget,
            "compressor state exceeds budget"
        );
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("compressor owner IDs exhausted"))?;
        Ok(Self {
            index: SourceCache::new(library, index_pages, slots)?,
            pending: if ratio(layer)? == 2 {
                Some([
                    DeviceAllocation::new(library, slots * 2048)?,
                    DeviceAllocation::new(library, slots * 2048)?,
                ])
            } else {
                None
            },
            slots: [Slot::default(); 16],
            slot_count: slots,
            layer,
            owner,
        })
    }
    pub fn begin_request(&mut self, slot: usize, request: u64) -> Result<CompressorLease> {
        ensure!(
            slot < self.slot_count && self.slots[slot].request.is_none(),
            "compressor slot unavailable"
        );
        ensure!(
            !self.slots.iter().any(|s| s.request == Some(request)),
            "duplicate compressor request"
        );
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .context("compressor generation exhausted")?;
        self.index.reset(slot)?;
        self.slots[slot] = Slot {
            generation,
            request: Some(request),
            ..Slot::default()
        };
        // New requests start at zero and cannot read stale pending device rows.
        Ok(CompressorLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    fn validate(&self, lease: CompressorLease) -> Result<usize> {
        let slot = self.validate_identity(lease)?;
        self.index.ensure_idle(slot)?;
        Ok(slot)
    }
    fn validate_identity(&self, lease: CompressorLease) -> Result<usize> {
        ensure!(
            lease.owner == self.owner && lease.slot < self.slot_count,
            "foreign compressor lease"
        );
        let slot = self.slots[lease.slot];
        ensure!(
            slot.request.is_some() && slot.generation == lease.generation,
            "stale compressor lease"
        );
        Ok(lease.slot)
    }
    pub fn request_id(&self, lease: CompressorLease) -> Result<u64> {
        self.slots[self.validate_identity(lease)?]
            .request
            .context("compressor request missing")
    }
    pub fn committed_end(&self, lease: CompressorLease) -> Result<u64> {
        Ok(self.slots[self.validate_identity(lease)?].end)
    }
    pub(crate) fn ensure_not_writing(&self, lease: CompressorLease) -> Result<()> {
        self.validate(lease).map(|_| ())
    }
    pub fn check_append_capacity(&self, work: &[(CompressorLease, u32)]) -> Result<()> {
        let ratio = ratio(self.layer)?;
        let appends = work.iter().map(|&(lease, tokens)| {
            let slot = self.validate(lease)?;
            let old = self.slots[slot].end as usize;
            Ok((slot, old / ratio, (old + tokens as usize) / ratio))
        }).collect::<Result<Vec<_>>>()?;
        self.index.reserve(&appends).map(|_| ())
    }
    pub fn index_cache(&self, lease: CompressorLease) -> Result<IndexCacheView<'_>> {
        let slot = self.validate(lease)?;
        Ok(self
            .index
            .view(slot, self.slots[slot].end as usize / ratio(self.layer)?))
    }
    pub fn kv_cache(&self, lease: CompressorLease) -> Result<KvCacheView<'_>> {
        let slot = self.validate(lease)?;
        Ok(self
            .index
            .kv_view(slot, self.slots[slot].end as usize / ratio(self.layer)?))
    }
    /// Borrow an immutable committed source range after its producer is published.
    /// No private rows are produced; ratio-two encoder sources and ratio-one
    /// decoder sources retain their per-query causal counts. The caller owns
    /// the snapshot identity across consumers. A committed-only causal prefix
    /// can survive append; its backing rows must remain immutable and all
    /// consumers must drain before request release or restoration.
    pub fn committed_proposal(&self, lease: CompressorLease, positions: std::ops::Range<u64>,
        snapshot: u64) -> Result<IndexProposal<'_>> {
        let end = self.committed_end(lease)?;
        let step = ratio(self.layer)? as u64;
        ensure!(snapshot != 0 && positions.start < positions.end
            && positions.end <= end, "invalid committed source range");
        // Appends only write beyond this committed prefix. Read its published
        // extent even while a follower owns the append reservation; mutation,
        // restoration and release continue to require an idle slot.
        let slot = self.validate_identity(lease)?;
        let rows = end as usize / step as usize;
        let cache = self.index.view(slot, rows);
        let kv_cache = self.index.kv_view(slot, rows);
        // Valid read-only backing for zero-length private overlays. Their
        // metadata count is zero, so no private entry can be selected.
        Ok(IndexProposal {
            binding: IndexBinding { snapshot, lease }, request: self.request_id(lease)?,
            source_layer: self.layer,
            kv_values: kv_cache.values, kv_scales: kv_cache.scales,
            packed: cache.packed, scales: cache.scales, capacity: 1,
            cache, kv_cache, first_token: positions.start, end_token: positions.end,
            start: end / step, count: 0, offset: 0, step,
            _wave: std::marker::PhantomData,
        })
    }
    /// All device consumers of this request's cache must have finished.
    pub fn release(&mut self, lease: CompressorLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.slots[slot].request = None;
        self.index.release(slot)
    }
    /// Invalidate participating requests after an external cache transaction fails.
    pub fn invalidate(&mut self, leases: &[CompressorLease]) -> Result<()> {
        let slots = leases
            .iter()
            .map(|&l| self.validate(l))
            .collect::<Result<Vec<_>>>()?;
        let mut result = Ok(());
        for slot in slots {
            self.slots[slot].request = None;
            let released = self.index.release(slot);
            result = result.and(released);
        }
        result
    }
}
pub(crate) struct CompressorWeights<'a> {
    library: &'a NativeLibrary,
    tensors: NativeRtxTensors<'a>,
    layer: usize,
    ratio: usize,
    names: Vec<String>,
}
impl<'a> CompressorWeights<'a> {
    fn names(layer: usize) -> Result<Vec<String>> {
        let suffixes = if ratio(layer)? == 2 {
            vec!["wkv", "norm", "wgate"]
        } else {
            vec!["wkv", "norm"]
        };
        let mut names = suffixes
            .into_iter()
            .map(|s| format!("layers.{layer}.attn.compressor.{s}.weight"))
            .collect::<Vec<_>>();
        names.extend([
            format!("layers.{layer}.attn.indexer.wk.weight"),
            format!("layers.{layer}.attn.indexer.k_norm.weight"),
        ]);
        Ok(names)
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        NativeRtxTensors::plan(catalog, &Self::names(layer)?)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging_bytes)?;
        Ok(Self {
            library,
            tensors,
            layer,
            ratio: ratio(layer)?,
            names,
        })
    }
    pub fn wave(&self, rows: usize, budget: usize) -> Result<CompressorWave<'_, 'a>> {
        ensure!(
            CompressorWave::device_bytes(self.layer, rows)? <= budget,
            "compressor wave exceeds budget"
        );
        let workspace = DeviceAllocation::new(self.library, V41Compressor::WORKSPACE_BYTES)?;
        ensure!(
            workspace.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "compressor weights device differs"
        );
        let kernel = unsafe { self.library.v41_compressor(workspace.buffer)? };
        Ok(CompressorWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            kernel,
            kv: self.library.v41_compressed_kv()?,
            _workspace: workspace,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            input: DeviceAllocation::new(self.library, rows * 10240)?,
            projected: DeviceAllocation::new(
                self.library,
                rows * 512 * if self.ratio == 2 { 4 } else { 2 },
            )?,
            scores: if self.ratio == 2 {
                Some(DeviceAllocation::new(self.library, rows * 2048)?)
            } else {
                None
            },
            descriptors: if self.ratio == 2 {
                Some(DeviceAllocation::new(self.library, rows * 8)?)
            } else {
                None
            },
            staging: HostAllocation::new(
                self.library,
                rows * if self.ratio == 2 { 16 } else { 8 },
            )?,
            positions: DeviceAllocation::new(self.library, rows * 8)?,
            frequencies: DeviceAllocation::new(self.library, rows * 256)?,
            index_projected: DeviceAllocation::new(self.library, rows * 256)?,
            index_key: DeviceAllocation::new(self.library, rows * 256)?,
            index_packed: DeviceAllocation::new(self.library, rows * 64)?,
            index_scales: DeviceAllocation::new(self.library, rows * 4)?,
            kv_values: DeviceAllocation::new(self.library, rows * V41Kv::COMPRESSED_VALUE_BYTES)?,
            kv_scales: DeviceAllocation::new(self.library, rows * V41Kv::COMPRESSED_SCALE_BYTES)?,
            cache_destinations: DeviceAllocation::new(self.library, rows * 8)?,
            output: DeviceAllocation::new(self.library, rows * 1024)?,
            capacity: rows,
            graph: None,
            retained_graphs: [None; 64],
            ready: None,
            pending_query: None,
            pending_commit: None,
            commit_staging: HostAllocation::new(self.library, rows*4+256)?,
        })
    }
}
struct Prepared {
    snapshot: u64,
    owner: u64,
    chunks: Vec<CompressorChunk>,
    versions: Vec<u64>,
    offsets: Vec<usize>,
    descriptors: Vec<u64>,
    completed: Vec<CompressorLatentRow>,
    rows: usize,
}
pub(crate) struct CompressorOutput<'a> {
    pub buffer: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    /// Normalized, rotated index key before FP4 cache encoding.
    pub index_key: Ds41rtDeviceBuffer,
    /// E2M1 [rows,64] and E8M0 [rows,4] proposals. Only completed, accepted
    /// rows may enter persistent index storage; rejected suffixes stay private.
    pub index_packed: Ds41rtDeviceBuffer,
    pub index_scales: Ds41rtDeviceBuffer,
    pub kv_values: Ds41rtDeviceBuffer,
    pub kv_scales: Ds41rtDeviceBuffer,
    pub completed: &'a [CompressorLatentRow],
}
/// Exact producing execution and request lease for candidate sharing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexBinding {
    snapshot: u64,
    lease: CompressorLease,
}
impl IndexBinding {
    pub fn same_request(self, other: Self) -> bool {
        self.lease == other.lease
    }
    pub fn same_pool(self, other: Self) -> bool {
        self.lease.owner == other.lease.owner
    }
}
/// Borrowed accepted-history and current-wave index view. Consumers must drain
/// before releasing these borrows. No proposed rows are published to the cache.
pub(crate) struct IndexProposal<'a> {
    binding: IndexBinding,
    request: u64,
    pub source_layer: usize,
    pub cache: IndexCacheView<'a>,
    pub kv_cache: KvCacheView<'a>,
    pub kv_values: Ds41rtDeviceBuffer,
    pub kv_scales: Ds41rtDeviceBuffer,
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub capacity: usize,
    first_token: u64,
    end_token: u64,
    start: u64,
    count: u64,
    offset: u64,
    step: u64,
    _wave: std::marker::PhantomData<&'a ()>,
}
impl IndexProposal<'_> {
    /// Retain authoritative index keys and identity while substituting the peer's
    /// FP4 attention payload. Index selection still runs on the original device.
    /// # Safety
    /// Replica publication for this source and proposal is complete or ordered
    /// before every consumer. Retain both proposal plane owners through consumers
    /// and cold graph preparation; no writes may race these borrowed views.
    pub unsafe fn peer_attention<'s>(&'s self, state: &'s CompressorState<'_>,
        replica: &'s SourceReplica<'_>, values: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer) -> Result<IndexProposal<'s>> {
        let slot = state.validate(self.binding.lease)?;
        ensure!(state.layer == self.source_layer && state.request_id(self.binding.lease)? == self.request
            && state.slots[slot].end == self.first_token,
            "peer compressed proposal snapshot differs");
        let kv_cache = unsafe { replica.view(&state.index, slot, self.kv_cache.rows)? };
        ensure!(kv_cache.pages == self.kv_cache.pages && kv_cache.rows == self.start as usize,
            "peer compressed proposal cache differs");
        ensure!(values.device_id == kv_cache.values.device_id && scales.device_id == values.device_id
            && self.capacity.checked_mul(V41Kv::COMPRESSED_VALUE_BYTES).is_some_and(|n| values.bytes >= n)
            && self.capacity.checked_mul(V41Kv::COMPRESSED_SCALE_BYTES).is_some_and(|n| scales.bytes >= n)
            && !values.ptr.is_null() && !scales.ptr.is_null(),
            "peer compressed proposal storage differs");
        Ok(IndexProposal { binding: self.binding, request: self.request, source_layer: self.source_layer,
            cache: IndexCacheView { packed: self.cache.packed, scales: self.cache.scales,
                pages: self.cache.pages, rows: self.cache.rows, device_pages: self.cache.device_pages,
                device_rows: self.cache.device_rows },
            kv_cache, kv_values: values, kv_scales: scales, packed: self.packed, scales: self.scales,
            capacity: self.capacity, first_token: self.first_token, end_token: self.end_token,
            start: self.start, count: self.count, offset: self.offset, step: self.step,
            _wave: std::marker::PhantomData })
    }
    pub fn request_id(&self) -> u64 {
        self.request
    }
    pub fn first_token(&self) -> u64 {
        self.first_token
    }
    pub fn binding(&self) -> IndexBinding {
        self.binding
    }
    /// Metadata uses slot zero because cache is a single-request page-table view.
    pub fn metadata(&self, position: u64) -> Result<[u64; 6]> {
        ensure!(
            position >= self.first_token && position < self.end_token,
            "query position is outside its compressor proposal"
        );
        Ok([
            0,
            (position + 1) / self.step,
            self.start,
            self.count,
            self.offset,
            self.step,
        ])
    }
}
pub(crate) struct CompressorWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernel: V41Compressor<'a>,
    kv: V41Kv<'a>,
    _workspace: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w CompressorWeights<'a>,
    input: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    scores: Option<DeviceAllocation<'a>>,
    descriptors: Option<DeviceAllocation<'a>>,
    staging: HostAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    index_projected: DeviceAllocation<'a>,
    index_key: DeviceAllocation<'a>,
    index_packed: DeviceAllocation<'a>,
    index_scales: DeviceAllocation<'a>,
    kv_values: DeviceAllocation<'a>,
    kv_scales: DeviceAllocation<'a>,
    cache_destinations: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize, u64)>,
    // Lane-local decode shapes; large prefill retains only the current graph.
    retained_graphs: [Option<(*mut c_void, usize, u64)>; 64],
    ready: Option<Prepared>,
    pending_query: Option<(Prepared, bool)>,
    pending_commit: Option<PendingCommit>,
    commit_staging: HostAllocation<'a>,
}
impl CompressorWave<'_, '_> {
    pub fn device_bytes(layer: usize, rows: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&rows),
            "invalid compressor row capacity"
        );
        Ok(V41Compressor::WORKSPACE_BYTES
            + rows
                * if ratio(layer)? == 2 {
                    10240 + 2048 + 2048 + 8 + 1024 + 264 + 512 + 68 + 8 + V41Kv::COMPRESSED_ROW_BYTES
                } else {
                    10240 + 1024 + 1024 + 264 + 512 + 68 + 8 + V41Kv::COMPRESSED_ROW_BYTES
                })
    }
    /// Packed BF16 [sum(chunk.tokens),5120], in chunk order. Finish all producer
    /// writes before execution, and keep this borrowed storage live until drop.
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(
        &mut self,
        state: &CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<Prepared> {
        ensure!(self.pending_query.is_none(), "cache query production pending");
        ensure!(self.pending_commit.is_none(), "compressor commit pending");
        self.ready = None;
        ensure!(
            state.layer == self.weights.layer,
            "compressor source layer differs"
        );
        ensure!(
            !chunks.is_empty() && chunks.len() <= state.slot_count,
            "invalid compressor batch count"
        );
        ensure!(
            state.index.packed.buffer.device_id == self.input.buffer.device_id,
            "compressor index cache device differs"
        );
        if let Some(pending) = &state.pending {
            ensure!(
                pending[0].buffer.device_id == self.input.buffer.device_id,
                "compressor state device differs"
            );
        }
        let snapshot = reserve_source_snapshot()?;
        let mut result = Prepared {
            snapshot,
            owner: state.owner,
            chunks: chunks.to_vec(),
            versions: vec![],
            offsets: vec![],
            descriptors: vec![],
            completed: vec![],
            rows: 0,
        };
        let mut seen = [false; 16];
        for chunk in chunks {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                !seen[slot] && chunk.tokens > 0,
                "duplicate or empty compressor chunk"
            );
            seen[slot] = true;
            ensure!(
                chunk.position == state.slots[slot].end,
                "compressor position is not the committed end"
            );
            chunk
                .position
                .checked_add(u64::from(chunk.tokens))
                .context("compressor position overflow")?;
            let end = result
                .rows
                .checked_add(chunk.tokens as usize)
                .context("compressor row overflow")?;
            ensure!(
                end <= self.capacity,
                "compressor batch exceeds row capacity"
            );
            ensure!(
                chunk.position + u64::from(chunk.tokens) <= 1048576,
                "compressor chunk exceeds model context"
            );
            result.versions.push(state.slots[slot].version);
            result.offsets.push(result.rows);
            for j in 0..chunk.tokens as usize {
                let pos = chunk.position + j as u64;
                let row = result.rows + j;
                if self.weights.ratio == 1 || pos % 2 == 1 {
                    result.completed.push(CompressorLatentRow {
                        lease: chunk.lease,
                        source_row: row as u32,
                        position: pos + 1 - self.weights.ratio as u64,
                    });
                }
                result.descriptors.push(if pos % 2 == 0 {
                    u64::MAX
                } else if j == 0 {
                    slot as u64
                } else {
                    (state.slot_count + row - 1) as u64
                });
            }
            result.rows = end;
        }
        Ok(result)
    }
    fn upload(&mut self, prepared: &Prepared) -> Result<()> {
        let offset = if self.descriptors.is_some() {
            prepared.rows * 8
        } else {
            0
        };
        let bytes = self.staging.bytes_mut();
        if let Some(device) = &self.descriptors {
            for (chunk, value) in bytes[..offset]
                .chunks_exact_mut(8)
                .zip(&prepared.descriptors)
            {
                chunk.copy_from_slice(&value.to_ne_bytes());
            }
            self.stream
                .library
                .copy_h2d(device.buffer, &bytes[..offset])?;
        }
        let positions = &mut bytes[offset..offset + prepared.rows * 8];
        positions.fill(0);
        for row in &prepared.completed {
            positions[row.source_row as usize * 8..row.source_row as usize * 8 + 8]
                .copy_from_slice(&row.position.to_ne_bytes());
        }
        self.stream
            .library
            .copy_h2d(self.positions.buffer, positions)?;
        Ok(())
    }
    fn upload_queued(&mut self, prepared: &Prepared) -> Result<()> {
        let host = self.staging.buffer;
        let offset = if self.descriptors.is_some() { prepared.rows * 8 } else { 0 };
        let bytes = self.staging.bytes_mut();
        for (dst, value) in bytes[..offset].chunks_exact_mut(8).zip(&prepared.descriptors) {
            dst.copy_from_slice(&value.to_ne_bytes());
        }
        let positions = &mut bytes[offset..offset + prepared.rows * 8];
        positions.fill(0);
        for row in &prepared.completed {
            positions[row.source_row as usize * 8..row.source_row as usize * 8 + 8]
                .copy_from_slice(&row.position.to_ne_bytes());
        }
        unsafe {
            if let Some(device) = &self.descriptors {
                self.stream.library.copy_host_buffer_h2d_async(device.buffer, host, offset, self.stream.raw)?;
            }
            let mut positions = host;
            positions.ptr = host.ptr.cast::<u8>().add(offset).cast();
            positions.bytes = prepared.rows * 8;
            self.stream.library.copy_host_buffer_h2d_async(self.positions.buffer, positions,
                prepared.rows * 8, self.stream.raw)
        }
    }
    unsafe fn enqueue(&self, state: &CompressorState<'_>, rows: usize) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.positions.buffer,
                self.frequencies.buffer,
                rows as u32,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.kernel.project(
                self.input.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.projected.buffer,
                rows,
                self.weights.ratio,
                self.stream.raw,
            )?;
            if let Some(pending) = &state.pending {
                self.kernel.project(
                    self.input.buffer,
                    self.weights.tensors.get(&self.weights.names[2])?,
                    self.scores.as_ref().unwrap().buffer,
                    rows,
                    2,
                    self.stream.raw,
                )?;
                self.kernel.pool(
                    self.projected.buffer,
                    self.scores.as_ref().unwrap().buffer,
                    pending[0].buffer,
                    pending[1].buffer,
                    self.descriptors.as_ref().unwrap().buffer,
                    self.weights.tensors.get(&self.weights.names[1])?,
                    self.output.buffer,
                    rows,
                    state.slot_count,
                    self.stream.raw,
                )?;
            } else {
                self.norm.norm(
                    self.projected.buffer,
                    self.weights.tensors.get(&self.weights.names[1])?,
                    None,
                    self.output.buffer,
                    rows as u32,
                    512,
                    self.stream.raw,
                )?;
            }
            self.kernel.index_project(
                self.output.buffer,
                self.weights
                    .tensors
                    .get(&self.weights.names[self.weights.names.len() - 2])?,
                self.index_projected.buffer,
                rows,
                self.stream.raw,
            )?;
            self.norm.norm(
                self.index_projected.buffer,
                self.weights
                    .tensors
                    .get(&self.weights.names[self.weights.names.len() - 1])?,
                Some(self.frequencies.buffer),
                self.index_key.buffer,
                rows as u32,
                128,
                self.stream.raw,
            )?;
            self.kernel.index_pack(
                self.index_key.buffer,
                self.index_packed.buffer,
                self.index_scales.buffer,
                rows,
                self.stream.raw,
            )?;
            self.kv.pack(
                self.output.buffer,
                Some(self.frequencies.buffer),
                self.kv_values.buffer,
                self.kv_scales.buffer,
                rows,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Query and admitted state slots stay alive and immutable until poll_query
    /// completes or abort_query drains. Peer work may use only disjoint slots.
    pub unsafe fn enqueue_query(&mut self, state: &CompressorState<'_>, chunks: &[CompressorChunk],
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>) -> Result<()> {
        let prepared = self.prepare(state, chunks)?;
        ensure!(query.binding()?.layer() == self.weights.layer && query.layer == self.weights.layer
            && query.rows == prepared.rows && query.hidden.bytes == prepared.rows * 10240
            && query.hidden.device_id == self.input.buffer.device_id
            && query.tokens()?.iter().copied().eq(chunks.iter().flat_map(|c|
                c.position..c.position + u64::from(c.tokens))), "queued cache query differs");
        self.select_graph(prepared.rows, state.owner, false)?;
        let capture = self.graph.is_none();
        let result = (|| -> Result<()> {
            unsafe { self.stream.library.copy_d2d_async(self.input.buffer, query.hidden,
                query.hidden.bytes, self.stream.raw)?; }
            self.upload_queued(&prepared)?;
            unsafe {
                if capture { self.enqueue(state, prepared.rows) }
                else { self.stream.library.cuda_graph_launch(self.graph.unwrap().0, self.stream.raw) }
            }
        })();
        self.pending_query = Some((prepared, capture));
        if let Err(error) = result { self.abort_query()?; return Err(error); }
        Ok(())
    }
    /// # Safety
    /// Preserve the enqueue_query ownership contract. Capture never suspends.
    pub unsafe fn poll_query(&mut self, state: &CompressorState<'_>) -> Result<bool> {
        let result = (|| -> Result<bool> {
            let (prepared, capture) = self.pending_query.as_ref().context("no queued cache query")?;
            ensure!(prepared.owner == state.owner, "queued cache owner differs");
            if !unsafe { self.stream.library.cuda_stream_query(self.stream.raw)? } { return Ok(false); }
            if *capture {
                unsafe { self.stream.library.cuda_graph_begin_capture(self.stream.raw)?; }
                let queued = unsafe { self.enqueue(state, prepared.rows) };
                let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
                let graph = match (queued, captured) {
                    (Ok(()), Ok(graph)) => graph,
                    (Err(error), Ok(graph)) => {
                        unsafe { self.stream.library.cuda_graph_exec_destroy(graph)?; }
                        return Err(error);
                    }
                    (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
                };
                self.graph = Some((graph, prepared.rows, state.owner));
                // Eager proposal is complete. Capture records future work without
                // committing cache state; publish the existing result below.
            }
            self.ready = Some(self.pending_query.take().unwrap().0);
            self.output(state)?;
            Ok(true)
        })();
        if result.is_err() { self.abort_query()?; }
        result
    }
    pub fn abort_query(&mut self) -> Result<()> {
        if self.pending_query.is_some() { self.synchronize()?; }
        self.pending_query = None;
        self.ready = None;
        Ok(())
    }
    /// Consume the exact normalized hidden rows used by a bound attention query.
    /// # Safety
    /// Query producers have completed; no external writes race query or cache
    /// storage. The batch's request identities correspond to these hidden rows.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        state: &'s CompressorState<'_>,
        chunks: &[CompressorChunk],
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
    ) -> Result<CompressorOutput<'s>> {
        self.ready = None;
        let prepared = self.prepare(state, chunks)?;
        ensure!(query.binding()?.layer() == self.weights.layer
            && query.layer == self.weights.layer
            && query.rows == prepared.rows
            && query.hidden.bytes == prepared.rows * 10240
            && query.hidden.device_id == self.input.buffer.device_id
            && query.tokens()?.iter().copied().eq(chunks.iter().flat_map(|c|
                c.position..c.position + u64::from(c.tokens))),
            "compressor query layer, rows or positions differ");
        self.synchronize()?;
        self.stream.library.copy_d2d(self.input.buffer, query.hidden, query.hidden.bytes)?;
        self.select_graph(prepared.rows, state.owner, true)?;
        if self.graph.is_none() {
            unsafe { self.capture(state, chunks)?; }
        }
        unsafe { self.replay(state, chunks) }
    }
    /// # Safety
    /// Finite input rows follow chunk order on this device, with producer writes
    /// complete. No external writes may race this wave or committed state.
    pub unsafe fn execute<'s>(
        &'s mut self,
        state: &'s CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<CompressorOutput<'s>> {
        let prepared = self.prepare(state, chunks)?;
        self.upload(&prepared)?;
        let launched = unsafe { self.enqueue(state, prepared.rows) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(prepared);
        self.output(state)
    }
    /// # Safety
    /// Same input contract as execute. Warmup is unpublished and never commits.
    pub unsafe fn capture(
        &mut self,
        state: &CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<()> {
        self.ready = None;
        ensure!(self.graph.is_none(), "compressor graph already captured");
        unsafe {
            self.execute(state, chunks)?;
        }
        let prepared = self.prepare(state, chunks)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(state, prepared.rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, prepared.rows, state.owner));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; state owner and live rows match capture.
    pub unsafe fn replay<'s>(
        &'s mut self,
        state: &'s CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<CompressorOutput<'s>> {
        let prepared = self.prepare(state, chunks)?;
        let (graph, rows, owner) = self.graph.context("compressor graph is not captured")?;
        ensure!(
            rows == prepared.rows && owner == state.owner,
            "compressor capture binding differs"
        );
        self.upload(&prepared)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(prepared);
        self.output(state)
    }
    pub fn output<'s>(&'s self, state: &'s CompressorState<'_>) -> Result<CompressorOutput<'s>> {
        let prepared = self
            .ready
            .as_ref()
            .context("compressor output incomplete")?;
        ensure!(
            prepared.owner == state.owner,
            "compressor output state differs"
        );
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                state.slots[slot].version == prepared.versions[i]
                    && state.slots[slot].end == chunk.position,
                "stale compressor output"
            );
        }
        let mut buffer = self.output.buffer;
        buffer.bytes = prepared.rows * 1024;
        let mut frequencies = self.frequencies.buffer;
        frequencies.bytes = prepared.rows * 256;
        let mut index_key = self.index_key.buffer;
        index_key.bytes = prepared.rows * 256;
        let mut index_packed = self.index_packed.buffer;
        index_packed.bytes = prepared.rows * 64;
        let mut index_scales = self.index_scales.buffer;
        index_scales.bytes = prepared.rows * 4;
        let mut kv_values = self.kv_values.buffer;
        kv_values.bytes = prepared.rows * V41Kv::COMPRESSED_VALUE_BYTES;
        let mut kv_scales = self.kv_scales.buffer;
        kv_scales.bytes = prepared.rows * V41Kv::COMPRESSED_SCALE_BYTES;
        Ok(CompressorOutput {
            kv_values,
            kv_scales,
            buffer,
            index_key,
            index_packed,
            index_scales,
            frequencies,
            completed: &prepared.completed,
        })
    }
    /// Validate the entire wave's owner/generations/versions before exposing a
    /// request's strided completed rows. Ratio-two incomplete rows are skipped.
    pub fn index_proposal<'s>(
        &'s self,
        state: &'s CompressorState<'_>,
        lease: CompressorLease,
    ) -> Result<IndexProposal<'s>> {
        let output = self.output(state)?;
        let prepared = self
            .ready
            .as_ref()
            .context("compressor output incomplete")?;
        let i = prepared
            .chunks
            .iter()
            .position(|c| c.lease == lease)
            .context("request is absent from compressor proposal")?;
        let chunk = prepared.chunks[i];
        let step = self.weights.ratio as u64;
        let end = chunk.position + u64::from(chunk.tokens);
        let start = chunk.position / step;
        let count = end / step - start;
        let offset = prepared.offsets[i] as u64
            + if step == 2 && chunk.position % 2 == 0 {
                1
            } else {
                0
            };
        Ok(IndexProposal {
            request: state.request_id(lease)?,
            binding: IndexBinding {
                snapshot: prepared.snapshot,
                lease,
            },
            source_layer: self.weights.layer,
            cache: state.index_cache(lease)?,
            kv_cache: state.kv_cache(lease)?,
            kv_values: output.kv_values,
            kv_scales: output.kv_scales,
            packed: output.index_packed,
            scales: output.index_scales,
            capacity: prepared.rows,
            first_token: chunk.position,
            end_token: end,
            start,
            count,
            offset,
            step,
            _wave: std::marker::PhantomData,
        })
    }
    /// Validate an enclosing cache transaction's exact request order/proposal.
    pub(crate) fn validate_batch(
        &self,
        state: &CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<()> {
        let prepared = if self.pending_commit.is_some() { self.validate_pending_commit(state)? }
            else { self.output(state)?; self.ready.as_ref().context("compressor output incomplete")? };
        ensure!(
            prepared.chunks.len() == chunks.len()
                && prepared.chunks.iter().zip(chunks).all(|(a, b)|
                    a.lease == b.lease && a.position == b.position && a.tokens == b.tokens),
            "compressor transaction proposal differs"
        );
        Ok(())
    }
    /// Switch only after prior launches finish. Graphs bind the cache owner and
    /// lane workspace, while current request descriptors are uploaded on replay.
    fn select_graph(&mut self, rows: usize, owner: u64, drain: bool) -> Result<()> {
        if self.graph.is_some_and(|(_, n, o)| n == rows && o == owner) { return Ok(()); }
        ensure!(self.pending_query.is_none() && self.pending_commit.is_none(),
            "cannot switch a pending cache producer graph");
        if self.graph.is_some_and(|(_, _, o)| o != owner)
            || self.retained_graphs.iter().flatten().any(|(_, _, o)| *o != owner) {
            return self.clear_graph_inner(drain);
        }
        self.ready = None;
        if drain { self.synchronize()?; } else { self.stream.require_complete()?; }
        if let Some(old) = self.graph.take() {
            if (1..=self.retained_graphs.len()).contains(&old.1) {
                let slot = &mut self.retained_graphs[old.1 - 1];
                ensure!(slot.is_none(), "duplicate retained cache producer shape");
                *slot = Some(old);
            } else {
                unsafe { self.stream.library.cuda_graph_exec_destroy(old.0)?; }
            }
        }
        if (1..=self.retained_graphs.len()).contains(&rows) {
            self.graph = self.retained_graphs[rows - 1].take();
        }
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.clear_graph_inner(true)
    }
    fn clear_graph_inner(&mut self, drain: bool) -> Result<()> {
        ensure!(self.pending_query.is_none(), "cannot clear pending cache query");
        ensure!(self.pending_commit.is_none(), "cannot reset a pending source commit");
        self.ready = None;
        if drain { self.synchronize()?; } else { self.stream.require_complete()?; }
        for (graph, _, _) in self.graph.take().into_iter()
            .chain(self.retained_graphs.iter_mut().filter_map(Option::take)) {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for CompressorWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() { tracing::error!(%error, "draining pending source commit"); }
        self.pending_query = None;
        self.pending_commit = None;
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining compressor graph");
        }
    }
}

impl<'a> CompressorState<'a> {
    /// The source page cache, for the host cache's page copies and allocations.
    pub fn source_cache(&self) -> &source_cache::SourceCache<'a> {
        &self.index
    }
}
