//! Criterion groups for packet HC-2: lookup throughput at 10k resident snapshots, `plan_store`
//! cost for a 1M-token snapshot, and eviction throughput. Sizes use the crate's engine constants.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ds41rt_hostcache::snapshot::testing::{resident_snapshots, snapshots};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};

/// Lookups per second over 10k resident 4k-token snapshots sharing a 2k-token prefix.
fn lookup(c: &mut Criterion) {
    let (mut store, keys) = resident_snapshots(10_000, 4096, 2048);
    let tokens = store.get(keys[0]).expect("snapshot").meta.tokens.clone();
    let mut group = c.benchmark_group("snapshot_lookup");
    group.throughput(Throughput::Elements(1));
    group.bench_function("10k_resident_4k_tokens", |b| {
        let mut now = 0u64;
        b.iter(|| {
            now += 1;
            store.lookup(&tokens, now).expect("hit")
        });
    });
    group.finish();
}

/// Planning a 1M-token snapshot: one page slab per 100 tokens across four compressors.
fn plan_store(c: &mut Criterion) {
    let mut store = snapshots(1 << 28);
    let tokens: Vec<u32> = (0..1_000_000).collect();
    let pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|compressor| {
        (0..(1_000_000 / 100) as u32)
            .map(|page| DevicePageId {
                compressor: compressor as u8,
                page,
                generation: 0,
            })
            .collect()
    });
    let meta = SnapshotMeta {
        kind: SnapshotKind::Turn,
        tokens,
        end: 1_000_000,
        has_draft: true,
    };
    let mut group = c.benchmark_group("snapshot_plan_store");
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("1m_tokens", |b| {
        b.iter_batched(
            || meta.clone(),
            |meta| {
                let plan = store.plan_store(meta, &pages).expect("plan");
                store.abort_store(plan);
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// Evictions per second from 1k resident snapshots down to an empty store.
fn evict(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_evict");
    group.throughput(Throughput::Elements(1_000));
    group.bench_function("1k_snapshots", |b| {
        b.iter_batched(
            || resident_snapshots(1_000, 4096, 2048).0,
            |mut store| {
                store.evict_to(0);
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, lookup, plan_store, evict);
criterion_main!(benches);
