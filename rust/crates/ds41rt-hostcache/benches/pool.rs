//! Criterion benches for the pinned slab pool (HC-1): per-class take+give_back throughput and
//! the cost of claiming a fresh chunk.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ds41rt_hostcache::pool::testing::FakePinned;
use ds41rt_hostcache::pool::{Class, Layout, Slab, SlabPool};
use std::hint::black_box;

/// The default `Config::chunk_bytes`; large enough to carve every engine class.
const CHUNK: usize = 256 << 20;
const QUOTA: u64 = 1 << 30;

fn layout() -> Layout {
    Layout::engine(4096)
}

fn pool() -> SlabPool {
    let mut mem = FakePinned::new(usize::MAX);
    SlabPool::new(QUOTA, CHUNK, layout(), &mut mem).expect("bench pool")
}

fn take(pool: &mut SlabPool, class: Class) -> Slab {
    pool.take(class)
        .expect("bench pool is sized to satisfy every take")
}

fn take_give_back(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_take_give_back");
    group.throughput(Throughput::Elements(1));
    for class in Class::ALL {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{class:?}")),
            &class,
            |b, &class| {
                let mut pool = pool();
                // Hold one slab so the chunk stays carved and the fast path is measured.
                let keeper = take(&mut pool, class);
                b.iter(|| {
                    let slab = take(&mut pool, class);
                    black_box(pool.location(slab));
                    pool.give_back(slab);
                });
                pool.give_back(keeper);
            },
        );
    }
    group.finish();
}

fn chunk_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_chunk_claim");
    group.throughput(Throughput::Elements(1));
    group.bench_function("page_claim_release", |b| {
        let mut pool = pool();
        b.iter(|| {
            let slab = take(&mut pool, Class::Page);
            pool.give_back(slab);
        });
    });
    group.finish();
}

criterion_group!(benches, take_give_back, chunk_claim);
criterion_main!(benches);
