//! Criterion groups for packet HC-5: store issue cost and restore latency at the engine's sizes
//! (890 bytes/token), and lookup throughput at 10k resident snapshots. The stub's virtual clock
//! makes the modelled restore latency exact; the wall-clock measurement is the memcpy the stub
//! performs when the copy executes.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ds41rt_hostcache::cache::{DevicePage, DeviceSnapshot, HostCache, RestoreTarget};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyModel, DeviceRange, StubCopyEngine};
use ds41rt_hostcache::pool::testing::layout as small_layout;
use ds41rt_hostcache::pool::Layout;
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS, KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};

/// Pinned allocation granularity for the benches.
const CHUNK: u64 = 64 << 20;

/// Hands out non-overlapping device ranges.
struct Ranges {
    addr: u64,
}

impl Ranges {
    fn new() -> Self {
        Self { addr: 0 }
    }
    fn range(&mut self, bytes: usize) -> DeviceRange {
        let range = DeviceRange {
            addr: self.addr,
            bytes,
        };
        self.addr += bytes as u64;
        range
    }
}

/// A cache over the engine layout with `bytes` of pinned host memory.
fn engine_cache(bytes: u64) -> HostCache<StubCopyEngine, u64> {
    let config = Config {
        bytes,
        chunk_bytes: CHUNK,
        store: StoreMode::OnRetain,
        min_tokens: 1,
        ..Config::default()
    };
    let engine = StubCopyEngine::new(CopyModel::default(), bytes as usize, bytes as usize);
    HostCache::new(config, Layout::engine(0), engine).expect("cache")
}

/// A snapshot of `tokens` tokens at the engine's 890 bytes/token, spread over the compressors.
fn engine_snapshot(tokens: usize, generation: u32) -> DeviceSnapshot {
    let pages = tokens * KV_BYTES_PER_TOKEN / (PAGE_BYTES * COMPRESSORS);
    let mut ranges = Ranges::new();
    let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
    for (compressor, list) in lists.iter_mut().enumerate() {
        for index in 0..pages {
            list.push(DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: index as u32,
                    generation,
                },
                segments: vec![ranges.range(PAGE_BYTES)],
            });
        }
    }
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: (0..tokens as u32).collect(),
            end: tokens as u32,
            has_draft: false,
        },
        pages: lists,
        tail: vec![ranges.range(TAIL_BYTES)],
        draft: None,
        scores: vec![],
    }
}

/// The pinned bytes a snapshot of `tokens` tokens needs, plus headroom for the pool's chunks.
fn engine_quota(tokens: usize) -> u64 {
    let pages = tokens * KV_BYTES_PER_TOKEN / PAGE_BYTES;
    (pages * PAGE_BYTES + TAIL_BYTES) as u64 + 4 * CHUNK
}

/// Store issue cost: plan and enqueue every copy of a 100k- and a 1M-token snapshot.
fn store_issue(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_store_issue");
    for (name, tokens) in [("100k_tokens", 100_000usize), ("1m_tokens", 1_000_000)] {
        let quota = engine_quota(tokens);
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_function(name, |b| {
            b.iter_batched(
                || (engine_cache(quota), engine_snapshot(tokens, 1)),
                |(mut cache, snapshot)| {
                    let _ = cache.store(&snapshot, 0);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Restore latency: copy a resident 100k- and 1M-token snapshot back into fresh destinations.
fn restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_restore");
    for (name, tokens) in [("100k_tokens", 100_000usize), ("1m_tokens", 1_000_000)] {
        let quota = engine_quota(tokens);
        let mut cache = engine_cache(quota);
        let snapshot = engine_snapshot(tokens, 1);
        let _ = cache.store(&snapshot, 0);
        cache.engine_mut().advance(1_000_000_000);
        cache.tick();
        let key = cache.lookup(&snapshot.meta.tokens).expect("host hit").key;
        group.throughput(Throughput::Bytes((tokens * KV_BYTES_PER_TOKEN) as u64));
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let mut ranges = Ranges::new();
                    let pages = std::array::from_fn(|compressor| {
                        snapshot.pages[compressor]
                            .iter()
                            .map(|page| DevicePage {
                                id: DevicePageId {
                                    compressor: compressor as u8,
                                    page: page.id.page,
                                    generation: 2,
                                },
                                segments: page
                                    .segments
                                    .iter()
                                    .map(|segment| ranges.range(segment.bytes))
                                    .collect(),
                            })
                            .collect()
                    });
                    RestoreTarget {
                        pages,
                        tail: snapshot
                            .tail
                            .iter()
                            .map(|segment| ranges.range(segment.bytes))
                            .collect(),
                        draft: None,
                        scores: vec![],
                    }
                },
                |target| {
                    let outcome = cache.restore(key, &target);
                    assert!(
                        matches!(
                            outcome,
                            ds41rt_hostcache::cache::RestoreOutcome::Done { .. }
                        ),
                        "restore did not complete"
                    );
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Lookup throughput over 10k resident snapshots of the pool suites' layout.
fn lookup(c: &mut Criterion) {
    let quota = 8 * CHUNK;
    let config = Config {
        bytes: quota,
        chunk_bytes: CHUNK,
        store: StoreMode::OnRetain,
        min_tokens: 1,
        ..Config::default()
    };
    let engine = StubCopyEngine::new(CopyModel::default(), quota as usize, quota as usize);
    let mut cache = HostCache::new(config, small_layout(), engine).expect("cache");
    let mut ranges = Ranges::new();
    let mut tokens = Vec::new();
    for generation in 0..10_000u32 {
        let sequence: Vec<u32> = (0..8).chain(std::iter::once(generation)).collect();
        let mut lists: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        for (compressor, list) in lists.iter_mut().enumerate() {
            list.push(DevicePage {
                id: DevicePageId {
                    compressor: compressor as u8,
                    page: generation,
                    generation,
                },
                segments: vec![ranges.range(small_layout().page)],
            });
        }
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta {
                kind: SnapshotKind::Turn,
                tokens: sequence.clone(),
                end: sequence.len() as u32,
                has_draft: false,
            },
            pages: lists,
            tail: vec![ranges.range(small_layout().tail)],
            draft: None,
            scores: vec![ranges.range(small_layout().scores)],
        };
        let _ = cache.store(&snapshot, generation as u64);
        cache.engine_mut().advance(1_000_000_000);
        cache.tick();
        tokens = sequence;
    }
    let mut group = c.benchmark_group("cache_lookup");
    group.throughput(Throughput::Elements(1));
    group.bench_function("10k_resident", |b| {
        b.iter(|| cache.lookup(&tokens).expect("host hit"))
    });
    group.finish();
}

criterion_group!(benches, store_issue, restore, lookup);
criterion_main!(benches);
