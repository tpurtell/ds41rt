//! Criterion groups for packet HC-8: submissions per snapshot. At the engine's sizes (54k /
//! 320k / 1M tokens) a snapshot is thousands of 1D copies; the cache coalesces the plan and
//! issues one batch. `coalesce_plan` measures the merge itself, `issue_1d` the old path (one
//! engine submission per segment), `issue_batch` the new path (one submission per snapshot)
//! through the stub, which charges a batch as one submission.
//!
//! The host geometry mirrors a fresh pool: page slabs claimed consecutively from 8 MiB chunks
//! (91 page slabs each), so per-compressor runs merge until a chunk boundary — the
//! fragmentation the plan must absorb.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ds41rt_hostcache::copy::{
    coalesce, CopyEngine, CopyModel, DeviceRange, Stream, StubCopyEngine,
};
use ds41rt_hostcache::pool::{HostRange, PinnedMemory};
use ds41rt_hostcache::{COMPRESSORS, KV_BYTES_PER_TOKEN, PAGE_BYTES, TAIL_BYTES};

/// The pool's chunk size at the engine's layout, and page slabs per chunk.
const CHUNK: usize = 8 << 20;
const PAGE_SLABS_PER_CHUNK: usize = CHUNK / PAGE_BYTES;

/// The snapshot sizes of record: tokens at the engine's 890 bytes/token.
const SIZES: [(&str, usize); 3] = [
    ("54k_tokens", 54_000),
    ("320k_tokens", 320_000),
    ("1m_tokens", 1_000_000),
];

/// A store-orientation copy plan for a snapshot of `tokens` tokens, shaped like the engine
/// allocates it: per compressor, pages of four address-contiguous segments; host side,
/// consecutive page slabs in 8 MiB chunks (a chunk boundary every 91 slabs), then the tail
/// slab in its own chunk (the tail is 2.7 MB, a class of its own). Returns the plan and the
/// snapshot's byte size.
fn plan(tokens: usize) -> (Vec<(DeviceRange, HostRange)>, u64) {
    let pages_per_compressor = tokens * KV_BYTES_PER_TOKEN / (PAGE_BYTES * COMPRESSORS);
    let mut pairs = Vec::with_capacity(pages_per_compressor * COMPRESSORS * 4 + 1);
    let mut addr = 0u64;
    let mut slab = 0usize;
    for _compressor in 0..COMPRESSORS {
        for _page in 0..pages_per_compressor {
            for segment in 0..4 {
                pairs.push((
                    DeviceRange {
                        addr,
                        bytes: PAGE_BYTES / 4,
                    },
                    HostRange {
                        chunk: (slab / PAGE_SLABS_PER_CHUNK) as u32,
                        offset: (slab % PAGE_SLABS_PER_CHUNK) * PAGE_BYTES
                            + segment * (PAGE_BYTES / 4),
                        bytes: PAGE_BYTES / 4,
                    },
                ));
                addr += (PAGE_BYTES / 4) as u64;
            }
            slab += 1;
        }
    }
    // The tail slab lives in a fresh chunk of its own class.
    pairs.push((
        DeviceRange {
            addr,
            bytes: TAIL_BYTES,
        },
        HostRange {
            chunk: (slab / PAGE_SLABS_PER_CHUNK + 1) as u32,
            offset: 0,
            bytes: TAIL_BYTES,
        },
    ));
    let bytes = addr + TAIL_BYTES as u64;
    (pairs, bytes)
}

/// The page-slab chunk count a plan's host side spans (page slabs plus the tail's chunk).
fn host_chunks(tokens: usize) -> usize {
    let pages_per_compressor = tokens * KV_BYTES_PER_TOKEN / (PAGE_BYTES * COMPRESSORS);
    let slabs = pages_per_compressor * COMPRESSORS;
    slabs / PAGE_SLABS_PER_CHUNK + 2
}

/// The batch charge model of record.
fn model() -> CopyModel {
    CopyModel::default()
}

/// The merge itself: plan coalescing cost at each snapshot size, with the merged extent count
/// as the throughput story (reported in the criterion output).
fn plan_coalesce(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_plan");
    for (name, tokens) in SIZES {
        let (pairs, bytes) = plan(tokens);
        let merged = coalesce(&pairs).len();
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(name, |b| {
            b.iter_batched(
                || pairs.clone(),
                |plan| {
                    let merged = coalesce(&plan);
                    criterion::black_box(merged.len());
                },
                BatchSize::LargeInput,
            )
        });
        println!(
            "{name}: {} 1D copies -> {merged} merged extents",
            pairs.len()
        );
    }
    group.finish();
}

/// The old path: one engine submission per segment of the 1D plan.
fn issue_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_issue_1d");
    for (name, tokens) in SIZES {
        let (pairs, bytes) = plan(tokens);
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let mut engine = StubCopyEngine::new(
                        model(),
                        bytes as usize,
                        (bytes + CHUNK as u64) as usize,
                    );
                    let chunk = engine
                        .allocate_chunk((bytes + CHUNK as u64) as usize)
                        .expect("chunk");
                    (engine, chunk)
                },
                |(mut engine, chunk)| {
                    for &(device, _) in &pairs {
                        engine
                            .d2h(
                                Stream::Store,
                                device,
                                HostRange {
                                    chunk: chunk.id,
                                    offset: 0,
                                    bytes: device.bytes,
                                },
                            )
                            .expect("1D issue");
                    }
                    criterion::black_box(engine.submission_count());
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// The new path: the coalesced plan as one batch — one submission per snapshot. The plan's
/// multi-chunk host geometry is flattened into one oversized chunk (offset = chunk index *
/// CHUNK + offset), preserving exactly where chunk boundaries break contiguity.
fn issue_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalesce_issue_batch");
    for (name, tokens) in SIZES {
        let (pairs, bytes) = plan(tokens);
        let host_bytes = host_chunks(tokens) * CHUNK;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let mut engine = StubCopyEngine::new(model(), bytes as usize, host_bytes);
                    let chunk = engine.allocate_chunk(host_bytes).expect("chunk");
                    let host_pairs: Vec<(DeviceRange, HostRange)> = pairs
                        .iter()
                        .map(|&(device, host)| {
                            (
                                device,
                                HostRange {
                                    chunk: chunk.id,
                                    offset: host.chunk as usize * CHUNK + host.offset,
                                    bytes: host.bytes,
                                },
                            )
                        })
                        .collect();
                    (engine, coalesce(&host_pairs))
                },
                |(mut engine, merged)| {
                    engine
                        .d2h_many(Stream::Store, &merged)
                        .expect("batch issue");
                    criterion::black_box(engine.submission_count());
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, plan_coalesce, issue_1d, issue_batch);
criterion_main!(benches);
