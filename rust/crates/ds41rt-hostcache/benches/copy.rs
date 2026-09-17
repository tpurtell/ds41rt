//! Criterion benches for the copy engine (packet HC-3): the cost of issuing one copy and the
//! cost of advancing the clock through a large pending queue.
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ds41rt_hostcache::copy::{CopyEngine, CopyModel, DeviceRange, Stream, StubCopyEngine};
use ds41rt_hostcache::pool::{HostRange, PinnedMemory};

const DEVICE_BYTES: usize = 1 << 20;
const HOST_BYTES: usize = 1 << 20;
const PAGE: usize = ds41rt_hostcache::PAGE_BYTES;
const PENDING: usize = 10_000;
const COPY_BYTES: usize = 64;

fn engine() -> StubCopyEngine {
    StubCopyEngine::new(CopyModel::default(), DEVICE_BYTES, HOST_BYTES)
}

fn bench_issue(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy/issue");
    group.throughput(Throughput::Elements(1));
    group.bench_function("page_d2h", |b| {
        // PerIteration keeps the fresh-engine setup and its drop out of the measurement.
        b.iter_batched(
            || {
                let mut engine = engine();
                let chunk = engine.allocate_chunk(PAGE).unwrap();
                engine.write_device(
                    DeviceRange {
                        addr: 0,
                        bytes: PAGE,
                    },
                    &vec![0u8; PAGE],
                );
                (engine, chunk)
            },
            |(mut engine, chunk)| {
                engine
                    .d2h(
                        Stream::Store,
                        DeviceRange {
                            addr: 0,
                            bytes: PAGE,
                        },
                        HostRange {
                            chunk: chunk.id,
                            offset: 0,
                            bytes: PAGE,
                        },
                    )
                    .unwrap();
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn bench_advance(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy/advance");
    group.throughput(Throughput::Elements(PENDING as u64));
    group.bench_function("10k_pending", |b| {
        b.iter_batched(
            || {
                let mut engine = engine();
                let chunk = engine.allocate_chunk(PENDING * COPY_BYTES).unwrap();
                engine.write_device(
                    DeviceRange {
                        addr: 0,
                        bytes: PENDING * COPY_BYTES,
                    },
                    &vec![0u8; PENDING * COPY_BYTES],
                );
                for i in 0..PENDING {
                    engine
                        .d2h(
                            Stream::Store,
                            DeviceRange {
                                addr: (i * COPY_BYTES) as u64,
                                bytes: COPY_BYTES,
                            },
                            HostRange {
                                chunk: chunk.id,
                                offset: i * COPY_BYTES,
                                bytes: COPY_BYTES,
                            },
                        )
                        .unwrap();
                }
                engine
            },
            |mut engine| engine.advance(u64::MAX),
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_issue, bench_advance);
criterion_main!(benches);
