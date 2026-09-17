//! Criterion group for packet HC-9: the prefill hold decision at the recommended fleet
//! pace (500 ms). The guard runs at every prefill chunk boundary of every lane, so the
//! decision paths must stay flat: knob off (pure no-op), nothing pending (an empty pending
//! scan, one clock read), an overdue store whose wait completes immediately, and an overdue
//! store on a stalled stream (one budgeted wait that burns the pace on the virtual clock
//! and counts a timeout — the worst case). Throughput is hold decisions per second; the
//! debug-build floor test lives in `tests/concurrency_hc_9.rs`.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ds41rt_hostcache::cache::{DevicePage, DeviceSnapshot, HostCache, StoreOutcome};
use ds41rt_hostcache::config::{Config, StoreMode};
use ds41rt_hostcache::copy::{CopyFault, CopyModel, DeviceRange, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::testing::{layout, CHUNK};
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};

/// The recommended fleet pace (see the daemon's `--host-cache-store-pace-ms` help).
const PACE_NS: u64 = 500_000_000;

/// A cache at `pace` with `pending` store copies in flight; `stalled` wedges the store
/// stream so every copy (and event) stays pending, otherwise the copies land on the clock.
fn cache(pace: u64, pending: usize, stalled: bool) -> HostCache<StubCopyEngine, u64> {
    let config = Config {
        bytes: 4 * CHUNK as u64,
        chunk_bytes: CHUNK as u64,
        store: StoreMode::OnRetain,
        store_pace_ns: pace,
        min_tokens: 1,
        ..Config::default()
    };
    let engine = StubCopyEngine::new(CopyModel::default(), 1 << 24, config.bytes as usize);
    let mut cache = HostCache::new(config, layout(), engine).expect("cache");
    for generation in 0..pending {
        let probe = probe_snapshot(generation as u32);
        if stalled {
            cache
                .engine_mut()
                .inject(CopyFault::StreamStalls(Stream::Store));
        }
        assert!(matches!(
            cache.store(&probe, generation as u64),
            StoreOutcome::Issued(_)
        ));
    }
    cache
}

/// One probe snapshot (~26 KB of copies across 21 segments) with `generation` folded into
/// its device identities so repeated stores never share pages; addresses stay inside the
/// stub's 16 MiB fake device for the pending counts this bench uses.
fn probe_snapshot(generation: u32) -> DeviceSnapshot {
    let layout = layout();
    let offset = u64::from(generation) * (1 << 20);
    let mut addr = offset;
    let mut range = |bytes: usize| {
        let start = addr;
        addr += bytes as u64;
        DeviceRange { addr: start, bytes }
    };
    let pages: [Vec<DevicePage>; COMPRESSORS] = std::array::from_fn(|compressor| {
        vec![DevicePage {
            id: DevicePageId {
                compressor: compressor as u8,
                page: generation * COMPRESSORS as u32 + compressor as u32,
                generation,
            },
            segments: (0..4).map(|_| range(layout.page / 4)).collect(),
        }]
    });
    DeviceSnapshot {
        meta: SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: (0..8).collect(),
            end: 8,
            has_draft: false,
        },
        pages,
        tail: (0..4).map(|_| range(layout.tail / 4)).collect(),
        draft: None,
        scores: vec![range(layout.scores)],
    }
}

fn prefill_hold(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefill_hold_decision");
    group.throughput(Throughput::Elements(1));

    let mut knob_off = cache(0, 1, true);
    group.bench_function("knob_off_noop", |b| {
        b.iter(|| knob_off.prefill_hold().expect("hold"))
    });

    let mut nothing_pending = cache(PACE_NS, 0, false);
    group.bench_function("nothing_pending", |b| {
        b.iter(|| nothing_pending.prefill_hold().expect("hold"))
    });

    // One store past its pace whose copies already landed (the event completed; the store
    // is uncommitted until a tick): the decision waits on a completed event and returns
    // immediately — the hold fast path after a store finishes during earlier chunks.
    let mut wait_completes = cache(PACE_NS, 1, false);
    wait_completes.engine_mut().advance(2 * PACE_NS);
    group.bench_function("overdue_store_wait_completes", |b| {
        b.iter(|| wait_completes.prefill_hold().expect("hold"))
    });

    // One pending store on a stalled stream: each call burns one full pace on the virtual
    // clock and counts a timeout — the worst-case decision path.
    let mut overdue_stalled = cache(PACE_NS, 1, true);
    overdue_stalled.engine_mut().advance(2 * PACE_NS);
    group.bench_function("overdue_store_times_out", |b| {
        b.iter(|| overdue_stalled.prefill_hold().expect("hold"))
    });

    group.finish();
}

criterion_group!(benches, prefill_hold);
criterion_main!(benches);
