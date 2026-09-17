//! Shared helpers for the HC-5 suites: a fake device that hands out byte ranges, snapshot and
//! restore-target builders over the pool suites' layout, a cache over the stub engine, and the
//! content pattern the fidelity checks compare.
#![allow(dead_code)]
use ds41rt_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreTarget, StoreOutcome, StoreTicket,
};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Event, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::pool::{HostChunk, HostRange, PinnedMemory};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};

/// The payload the suites attach to a store.
pub type Payload = u64;

/// Device bytes the functional suites give the stub: enough for every range they hand out.
pub const DEVICE_BYTES: usize = 1 << 24;

/// A fake device that hands out byte ranges, wrapping to the start when the tail does not fit.
pub struct Device {
    capacity: usize,
    next: usize,
}

impl Device {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, next: 0 }
    }

    /// A fresh range of `bytes`, wrapping to the start when the tail does not fit.
    pub fn range(&mut self, bytes: usize) -> DeviceRange {
        assert!(bytes <= self.capacity, "range larger than the device");
        if self.next + bytes > self.capacity {
            self.next = 0;
        }
        let addr = self.next;
        self.next += bytes;
        DeviceRange {
            addr: addr as u64,
            bytes,
        }
    }

    /// `count` equal segments of `bytes` each.
    pub fn ranges(&mut self, count: usize, bytes: usize) -> Vec<DeviceRange> {
        (0..count).map(|_| self.range(bytes)).collect()
    }
}

/// The test config: `bytes` of pinned memory, `store` mode, one-token minimum, both kinds.
pub fn config(bytes: u64, store: StoreMode) -> Config {
    Config {
        bytes,
        chunk_bytes: CHUNK as u64,
        store,
        min_tokens: 1,
        ..Config::default()
    }
}

/// A cache over a stub engine with `config` and the pool suites' layout.
pub fn cache(
    config: Config,
    model: CopyModel,
    device_bytes: usize,
) -> HostCache<StubCopyEngine, Payload> {
    let engine = StubCopyEngine::new(model, device_bytes, config.bytes as usize);
    HostCache::new(config, layout(), engine).expect("cache")
}

/// A cache with the default copy model and [`DEVICE_BYTES`] of fake device memory.
pub fn default_cache(bytes: u64, store: StoreMode) -> HostCache<StubCopyEngine, Payload> {
    cache(config(bytes, store), CopyModel::default(), DEVICE_BYTES)
}

/// A snapshot of `pages` pages per compressor over `tokens`, with the layout's part sizes and
/// `generation` folded into every device identity so distinct snapshots do not share pages.
pub fn snapshot(
    device: &mut Device,
    kind: SnapshotKind,
    tokens: &[u32],
    pages: usize,
    has_draft: bool,
    generation: u32,
) -> DeviceSnapshot {
    let layout = layout();
    let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    for (compressor, list) in lists.iter_mut().enumerate() {
        for index in 0..pages {
            list.push(DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: (compressor * 100_000 + index) as u32,
                    generation,
                },
                segments: device.ranges(4, layout.page / 4),
            });
        }
    }
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind,
            tokens: tokens.to_vec(),
            end: tokens.len() as u32,
            has_draft,
        },
        pages: lists,
        tail: device.ranges(4, layout.tail / 4),
        draft: has_draft.then(|| device.ranges(2, layout.draft / 2)),
        scores: device.ranges(1, layout.scores),
    }
}

/// Fresh destinations mirroring `snapshot`'s shape, with `generation` as their new identity.
pub fn target(device: &mut Device, snapshot: &DeviceSnapshot, generation: u32) -> RestoreTarget {
    let pages = std::array::from_fn(|compressor| {
        snapshot.pages[compressor]
            .iter()
            .enumerate()
            .map(|(index, page)| DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: (compressor * 100_000 + index) as u32,
                    generation,
                },
                segments: page
                    .segments
                    .iter()
                    .map(|segment| device.range(segment.bytes))
                    .collect(),
            })
            .collect()
    });
    RestoreTarget {
        pages,
        tail: snapshot
            .tail
            .iter()
            .map(|segment| device.range(segment.bytes))
            .collect(),
        draft: snapshot.draft.as_ref().map(|draft| {
            draft
                .iter()
                .map(|segment| device.range(segment.bytes))
                .collect()
        }),
        scores: snapshot
            .scores
            .iter()
            .map(|segment| device.range(segment.bytes))
            .collect(),
    }
}

/// The device identities of a snapshot, per compressor, as `plan_store` wants them.
pub fn device_pages(snapshot: &DeviceSnapshot) -> [Vec<DevicePageId>; COMPRESSORS] {
    std::array::from_fn(|c| snapshot.pages[c].iter().map(|page| page.id).collect())
}

/// A deterministic byte pattern for a device range, so a copy's content is checkable.
pub fn pattern(range: DeviceRange) -> Vec<u8> {
    (0..range.bytes)
        .map(|offset| (range.addr as usize + offset) as u8)
        .collect()
}

/// Write [`pattern`] into every range of `snapshot`.
pub fn write_snapshot(engine: &mut StubCopyEngine, snapshot: &DeviceSnapshot) {
    for list in &snapshot.pages {
        for page in list {
            for segment in &page.segments {
                engine.write_device(*segment, &pattern(*segment));
            }
        }
    }
    for segment in &snapshot.tail {
        engine.write_device(*segment, &pattern(*segment));
    }
    if let Some(draft) = &snapshot.draft {
        for segment in draft {
            engine.write_device(*segment, &pattern(*segment));
        }
    }
    for segment in &snapshot.scores {
        engine.write_device(*segment, &pattern(*segment));
    }
}

/// The bytes currently in the device ranges of `snapshot`, in the order a restore writes them.
pub fn stored_bytes(engine: &StubCopyEngine, snapshot: &DeviceSnapshot) -> Vec<u8> {
    let mut bytes = Vec::new();
    for list in &snapshot.pages {
        for page in list {
            for segment in &page.segments {
                bytes.extend(engine.read_device(*segment));
            }
        }
    }
    for segment in &snapshot.tail {
        bytes.extend(engine.read_device(*segment));
    }
    if let Some(draft) = &snapshot.draft {
        for segment in draft {
            bytes.extend(engine.read_device(*segment));
        }
    }
    for segment in &snapshot.scores {
        bytes.extend(engine.read_device(*segment));
    }
    bytes
}

/// The bytes a restore wrote into `target`, in the same order as [`stored_bytes`].
pub fn restored_bytes(engine: &StubCopyEngine, target: &RestoreTarget) -> Vec<u8> {
    let mut bytes = Vec::new();
    for list in &target.pages {
        for page in list {
            for segment in &page.segments {
                bytes.extend(engine.read_device(*segment));
            }
        }
    }
    for segment in &target.tail {
        bytes.extend(engine.read_device(*segment));
    }
    if let Some(draft) = &target.draft {
        for segment in draft {
            bytes.extend(engine.read_device(*segment));
        }
    }
    for segment in &target.scores {
        bytes.extend(engine.read_device(*segment));
    }
    bytes
}

/// `count` key-space tokens.
pub fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// Advance the virtual clock far enough that every issued copy has completed.
pub fn settle(cache: &mut HostCache<StubCopyEngine, Payload>) {
    cache.engine_mut().advance(1_000_000_000);
}

/// The stub's modelled duration of one store of `snapshot` under `model`: since HC-8 the cache
/// issues a store as one coalesced batch, charged as one per-copy latency plus the transfer of the
/// summed bytes at the modelled bandwidth, every extent completing at that one time.
pub fn modelled_store_ns(model: CopyModel, snapshot: &DeviceSnapshot) -> u64 {
    let bytes: usize = snapshot
        .pages
        .iter()
        .flatten()
        .flat_map(|page| &page.segments)
        .chain(snapshot.tail.iter())
        .chain(snapshot.draft.iter().flatten())
        .chain(snapshot.scores.iter())
        .map(|segment| segment.bytes)
        .sum();
    model.per_copy_latency_ns + (bytes as f64 / model.d2h_bytes_per_ns).ceil() as u64
}

/// Store `snapshot` and make it resident, whichever mode the cache uses: `OnRetain` issues at
/// store time and `tick` commits; `OnEvict` issues and commits on the device-evict path.
pub fn store_resident(
    cache: &mut HostCache<StubCopyEngine, Payload>,
    snapshot: &DeviceSnapshot,
    payload: Payload,
) -> StoreTicket {
    match cache.store(snapshot, payload) {
        StoreOutcome::Issued(ticket) => {
            settle(cache);
            cache.tick();
            ticket
        }
        StoreOutcome::Deferred(ticket) => {
            assert!(matches!(
                cache.before_device_evict(Some(ticket)),
                EvictDecision::WaitedClean { .. }
            ));
            ticket
        }
        StoreOutcome::Skipped(reason) => panic!("store skipped: {reason:?}"),
    }
}

/// A stub that lets `skip` store-stream copies through and then fails the next one, so a
/// mid-plan issue failure can be exercised with earlier copies already in flight.
pub struct FailAfter {
    inner: StubCopyEngine,
    remaining: usize,
}

impl FailAfter {
    /// A stub that fails the `skip + 1`-th store-stream `d2h`.
    pub fn new(inner: StubCopyEngine, skip: usize) -> Self {
        Self {
            inner,
            remaining: skip,
        }
    }

    /// The wrapped stub, for clock control and content checks.
    pub fn inner_mut(&mut self) -> &mut StubCopyEngine {
        &mut self.inner
    }
}

impl PinnedMemory for FailAfter {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<HostChunk> {
        self.inner.allocate_chunk(bytes)
    }

    fn release_chunk(&mut self, chunk: HostChunk) -> anyhow::Result<()> {
        self.inner.release_chunk(chunk)
    }
}

impl CopyEngine for FailAfter {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        if stream == Stream::Store {
            if self.remaining == 0 {
                anyhow::bail!("injected mid-plan issue failure");
            }
            self.remaining -= 1;
        }
        self.inner.d2h(stream, src, dst)
    }

    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        self.inner.h2d(stream, src, dst)
    }

    fn submission_count(&self) -> u64 {
        self.inner.submission_count()
    }

    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        self.inner.record(stream)
    }

    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        self.inner.completed(event)
    }

    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        self.inner.wait(event, budget_ns)
    }

    fn now_ns(&self) -> u64 {
        self.inner.now_ns()
    }
}
