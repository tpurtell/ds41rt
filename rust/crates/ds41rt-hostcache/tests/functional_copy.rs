//! Functional suite for the copy engine (packet HC-3): per-stream ordering, stream independence,
//! completion timing against the model, content fidelity, the late-overwrite hazard, event
//! semantics, faults, the host allocation limit, and proptest properties over random sequences.
use ds41rt_hostcache::copy::testing::Shadow;
use ds41rt_hostcache::copy::{
    CopyEngine, CopyFault, CopyModel, DeviceRange, Event, Stream, StubCopyEngine,
};
use ds41rt_hostcache::pool::{HostChunk, HostRange, PinnedMemory};
use proptest::prelude::*;

const DEVICE_BYTES: usize = 1 << 12;
const HOST_BYTES: usize = 1 << 12;
const STRIDE: usize = 64;
const SLOTS: usize = DEVICE_BYTES / STRIDE;

fn engine() -> StubCopyEngine {
    StubCopyEngine::new(CopyModel::default(), DEVICE_BYTES, HOST_BYTES)
}

fn dev(addr: u64, bytes: usize) -> DeviceRange {
    DeviceRange { addr, bytes }
}

fn host(chunk: u32, offset: usize, bytes: usize) -> HostRange {
    HostRange {
        chunk,
        offset,
        bytes,
    }
}

#[test]
fn per_stream_ordering() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    engine.write_device(dev(100, 100), &[2u8; 100]);
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();
    engine
        .d2h(Stream::Store, dev(100, 100), host(chunk.id, 100, 100))
        .unwrap();
    assert_eq!(engine.pending(Stream::Store), 2);

    // 100 bytes at 25 B/ns is 4 ns, so the first copy completes at 10_004.
    engine.advance(10_004);
    assert_eq!(engine.pending(Stream::Store), 1);
    assert_eq!(engine.read_host(host(chunk.id, 0, 100)), vec![1u8; 100]);
    assert_eq!(engine.read_host(host(chunk.id, 100, 100)), vec![0u8; 100]);

    // The second starts at the first's completion and completes at 20_008.
    engine.advance(10_004);
    assert_eq!(engine.pending(Stream::Store), 0);
    assert_eq!(engine.read_host(host(chunk.id, 100, 100)), vec![2u8; 100]);
}

#[test]
fn streams_are_independent() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 1000), &[7u8; 1000]);
    // Store: 1000 bytes -> 40 ns + 10_000 = 10_040.
    engine
        .d2h(Stream::Store, dev(0, 1000), host(chunk.id, 0, 1000))
        .unwrap();
    // Restore: 100 bytes -> 4 ns + 10_000 = 10_004.
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(2000, 100))
        .unwrap();
    assert_eq!(engine.pending(Stream::Store), 1);
    assert_eq!(engine.pending(Stream::Restore), 1);

    engine.advance(10_004);
    assert_eq!(engine.pending(Stream::Store), 1);
    assert_eq!(engine.pending(Stream::Restore), 0);

    engine.advance(36);
    assert_eq!(engine.pending(Stream::Store), 0);
}

#[test]
fn completion_timing_matches_model() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 250), &[3u8; 250]);
    engine
        .d2h(Stream::Store, dev(0, 250), host(chunk.id, 0, 250))
        .unwrap();
    let event = engine.record(Stream::Store).unwrap();

    // 250 bytes at 25 B/ns is 10 ns, plus the 10_000 ns latency.
    assert!(!engine.wait(event, 10_009).unwrap());
    assert_eq!(engine.now_ns(), 10_009);
    assert!(engine.wait(event, 1).unwrap());
    assert_eq!(engine.now_ns(), 10_010);
    assert!(engine.completed(event).unwrap());
}

/// The two directions charge their own bandwidth: 250 bytes cost 10 ns d2h at 25 B/ns but 25 ns
/// h2d at 10 B/ns, so swapping the rates fails the hand-computed completion times.
#[test]
fn asymmetric_bandwidth_is_pinned() {
    let model = CopyModel {
        d2h_bytes_per_ns: 25.0,
        h2d_bytes_per_ns: 10.0,
        per_copy_latency_ns: 10_000,
    };
    let mut engine = StubCopyEngine::new(model, DEVICE_BYTES, HOST_BYTES);
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 250), &[1u8; 250]);
    engine
        .d2h(Stream::Store, dev(0, 250), host(chunk.id, 0, 250))
        .unwrap();
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 250), dev(2000, 250))
        .unwrap();
    let d2h = engine.record(Stream::Store).unwrap();
    let h2d = engine.record(Stream::Restore).unwrap();

    // d2h: 250 B at 25 B/ns = 10 ns -> 10_010. h2d: 250 B at 10 B/ns = 25 ns -> 10_025.
    assert!(!engine.wait(d2h, 10_009).unwrap());
    assert!(engine.wait(d2h, 1).unwrap());
    assert_eq!(engine.now_ns(), 10_010);
    assert!(!engine.completed(h2d).unwrap());
    assert!(!engine.wait(h2d, 14).unwrap());
    assert_eq!(engine.now_ns(), 10_024);
    assert!(engine.wait(h2d, 1).unwrap());
    assert_eq!(engine.now_ns(), 10_025);
}

/// Copies on different streams that complete at the same instant execute in issue order: the
/// store issued first reads the device before the restore overwrites it.
#[test]
fn cross_stream_ties_break_by_issue_order() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    // Both copies are 100 bytes at 25 B/ns and start at zero, so both complete at 10_004.
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(0, 100))
        .unwrap();
    engine.advance(10_004);

    // Store ran first, so the host holds the original device bytes and the restore writes them
    // back; had the restore run first the device would hold the zeroed host chunk.
    assert_eq!(engine.read_host(host(chunk.id, 0, 100)), vec![1u8; 100]);
    assert_eq!(engine.read_device(dev(0, 100)), vec![1u8; 100]);
}

/// Unknown events and out-of-range chunk ids are errors, not panics or silent successes.
#[test]
fn unknown_event_and_chunk_are_errors() {
    let mut engine = engine();
    assert!(engine.completed(Event(0)).is_err());
    assert!(engine.wait(Event(0), 0).is_err());
    assert!(engine.release_chunk(HostChunk { id: 7, bytes: 1 }).is_err());
}

#[test]
fn content_fidelity_round_trip() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    let pattern: Vec<u8> = (0..1000).map(|i| (i % 251) as u8).collect();
    engine.write_device(dev(0, 1000), &pattern);
    engine
        .d2h(Stream::Store, dev(0, 1000), host(chunk.id, 0, 1000))
        .unwrap();
    engine.advance(10_040);
    assert_eq!(engine.read_host(host(chunk.id, 0, 1000)), pattern);

    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 1000), dev(2000, 1000))
        .unwrap();
    engine.advance(10_040);
    assert_eq!(engine.read_device(dev(2000, 1000)), pattern);
}

#[test]
fn late_overwrite_is_visible() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();
    // The source changes before the copy executes, so the copy must see the new bytes.
    engine.write_device(dev(0, 100), &[2u8; 100]);
    engine.advance(10_004);
    assert_eq!(engine.read_host(host(chunk.id, 0, 100)), vec![2u8; 100]);
}

#[test]
fn advance_executes_in_completion_order() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 1000), &[1u8; 1000]);
    // Store completes at 10_040; Restore, issued later, completes at 10_004.
    engine
        .d2h(Stream::Store, dev(0, 1000), host(chunk.id, 0, 1000))
        .unwrap();
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(0, 100))
        .unwrap();
    engine.advance(10_040);

    // Restore ran first and zeroed the first 100 device bytes; Store then copied that.
    let copied = engine.read_host(host(chunk.id, 0, 1000));
    assert_eq!(&copied[..100], &[0u8; 100]);
    assert_eq!(&copied[100..], &[1u8; 900]);
}

#[test]
fn event_semantics() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();

    // An event with no copies before it is complete immediately.
    let empty = engine.record(Stream::Store).unwrap();
    assert!(engine.completed(empty).unwrap());

    engine.write_device(dev(0, 100), &[1u8; 100]);
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();
    let event = engine.record(Stream::Store).unwrap();
    assert!(!engine.completed(event).unwrap());
    engine.advance(10_004);
    assert!(engine.completed(event).unwrap());
}

#[test]
fn wait_over_budget_times_out() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();
    let event = engine.record(Stream::Store).unwrap();

    assert!(!engine.wait(event, 5_000).unwrap());
    assert_eq!(engine.now_ns(), 5_000);
    assert!(engine.wait(event, 5_004).unwrap());
    assert_eq!(engine.now_ns(), 10_004);
}

#[test]
fn issue_fault_fires_once() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);

    engine.inject(CopyFault::IssueFails(Stream::Store));
    assert!(engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .is_err());
    // The fault is consumed, so the next issue succeeds.
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();

    // A fault armed for one stream does not touch the other.
    engine.inject(CopyFault::IssueFails(Stream::Store));
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(2000, 100))
        .unwrap();
    assert!(engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .is_err());
}

/// A fired stall wedges the stream: the stalled event and every later copy and event on that
/// stream never complete, while the other stream proceeds.
#[test]
fn stream_stall_wedges_the_stream() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
        .unwrap();

    engine.inject(CopyFault::StreamStalls(Stream::Store));
    let stalled = engine.record(Stream::Store).unwrap();
    assert!(!engine.completed(stalled).unwrap());
    assert!(!engine.wait(stalled, 1_000_000).unwrap());

    // A later copy on the wedged stream is accepted but never executes.
    engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 100, 100))
        .unwrap();
    let later = engine.record(Stream::Store).unwrap();
    assert!(!engine.completed(later).unwrap());
    assert!(!engine.wait(later, 1_000_000).unwrap());
    assert_eq!(engine.pending(Stream::Store), 1);
    assert_eq!(engine.read_host(host(chunk.id, 100, 100)), vec![0u8; 100]);

    // The other stream is unaffected.
    engine
        .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(2000, 100))
        .unwrap();
    let healthy = engine.record(Stream::Restore).unwrap();
    assert!(engine.wait(healthy, 1_000_000).unwrap());
    assert_eq!(engine.read_device(dev(2000, 100)), vec![1u8; 100]);
}

#[test]
fn host_allocation_limit() {
    let mut engine = StubCopyEngine::new(CopyModel::default(), 1024, 1000);
    let first = engine.allocate_chunk(600).unwrap();
    let second = engine.allocate_chunk(400).unwrap();
    assert!(engine.allocate_chunk(1).is_err());

    engine.release_chunk(first).unwrap();
    let third = engine.allocate_chunk(600).unwrap();
    assert_ne!(first.id, third.id);
    assert!(engine.release_chunk(first).is_err());

    engine.release_chunk(second).unwrap();
    engine.release_chunk(third).unwrap();
}

#[test]
fn out_of_bounds_issues_fail() {
    let mut engine = StubCopyEngine::new(CopyModel::default(), 100, 100);
    let chunk = engine.allocate_chunk(100).unwrap();
    assert!(engine
        .d2h(Stream::Store, dev(50, 100), host(chunk.id, 0, 100))
        .is_err());
    assert!(engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 50, 100))
        .is_err());
    assert!(engine
        .d2h(Stream::Store, dev(0, 100), host(99, 0, 100))
        .is_err());
    assert!(engine
        .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 50))
        .is_err());
}

#[test]
fn pending_counts_track_execution() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(4096).unwrap();
    engine.write_device(dev(0, 100), &[1u8; 100]);
    for _ in 0..5 {
        engine
            .d2h(Stream::Store, dev(0, 100), host(chunk.id, 0, 100))
            .unwrap();
    }
    for _ in 0..3 {
        engine
            .h2d(Stream::Restore, host(chunk.id, 0, 100), dev(2000, 100))
            .unwrap();
    }
    assert_eq!(engine.pending(Stream::Store), 5);
    assert_eq!(engine.pending(Stream::Restore), 3);

    engine.advance(10_004);
    assert_eq!(engine.pending(Stream::Store), 4);
    assert_eq!(engine.pending(Stream::Restore), 2);

    engine.advance(100_000);
    assert_eq!(engine.pending(Stream::Store), 0);
    assert_eq!(engine.pending(Stream::Restore), 0);
}

/// A broken fast path must fail `cargo test`, not a human: 100k issues in a debug build.
#[test]
fn issue_floor_100k_within_2s() {
    let mut engine = engine();
    let chunk = engine.allocate_chunk(HOST_BYTES).unwrap();
    engine.write_device(dev(0, 64), &[1u8; 64]);
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        engine
            .d2h(Stream::Store, dev(0, 64), host(chunk.id, 0, 64))
            .unwrap();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "100k issues took {elapsed:?}, over the 2 s floor"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Random issue/advance sequences must keep the engine's observable state equal to an
    /// independent model: clock, pending counts, both memories, and per-stream event prefixes.
    #[test]
    fn random_sequences_match_the_shadow(ops in prop::collection::vec(any::<u8>(), 1..80)) {
        let model = CopyModel::default();
        let mut engine = StubCopyEngine::new(model, DEVICE_BYTES, HOST_BYTES);
        let chunk = engine.allocate_chunk(HOST_BYTES).unwrap();
        let mut shadow = Shadow::new(model, DEVICE_BYTES, HOST_BYTES);
        let mut events: Vec<(Stream, Event)> = Vec::new();
        let mut slot = 0usize;

        for (step, op) in ops.iter().enumerate() {
            match op % 4 {
                0 => {
                    let bytes = 1 + (step * 7) % STRIDE;
                    let addr = (slot % SLOTS) * STRIDE;
                    slot += 1;
                    let pattern: Vec<u8> = (0..bytes).map(|i| (i as u8) ^ (step as u8)).collect();
                    engine.write_device(dev(addr as u64, bytes), &pattern);
                    shadow.write_device(addr, &pattern);
                    engine
                        .d2h(Stream::Store, dev(addr as u64, bytes), host(chunk.id, addr, bytes))
                        .unwrap();
                    shadow.issue(Stream::Store, true, addr, addr, bytes);
                }
                1 => {
                    let bytes = 1 + (step * 11) % STRIDE;
                    let addr = (slot % SLOTS) * STRIDE;
                    slot += 1;
                    engine
                        .h2d(Stream::Restore, host(chunk.id, addr, bytes), dev(addr as u64, bytes))
                        .unwrap();
                    shadow.issue(Stream::Restore, false, addr, addr, bytes);
                }
                2 => {
                    let nanos = (step as u64 * 37) % 20_000;
                    engine.advance(nanos);
                    shadow.advance(nanos);
                }
                _ => {
                    let stream = if step % 2 == 0 { Stream::Store } else { Stream::Restore };
                    let event = engine.record(stream).unwrap();
                    let shadow_event = shadow.record(stream);
                    prop_assert_eq!(event.0 as usize, shadow_event);
                    events.push((stream, event));
                }
            }

            prop_assert_eq!(engine.now_ns(), shadow.now);
            prop_assert_eq!(engine.pending(Stream::Store), shadow.pending(Stream::Store));
            prop_assert_eq!(engine.pending(Stream::Restore), shadow.pending(Stream::Restore));
            let device = engine.read_device(dev(0, DEVICE_BYTES));
            prop_assert_eq!(device.as_slice(), shadow.device.as_slice());
            let host_bytes = engine.read_host(host(chunk.id, 0, HOST_BYTES));
            prop_assert_eq!(host_bytes.as_slice(), shadow.host.as_slice());
        }

        for stream in [Stream::Store, Stream::Restore] {
            let mut incomplete = false;
            for (event_stream, event) in &events {
                if *event_stream != stream {
                    continue;
                }
                if engine.completed(*event).unwrap() {
                    prop_assert!(!incomplete, "event completed after an incomplete one on {stream:?}");
                } else {
                    incomplete = true;
                }
            }
        }
    }

    /// On one stream, completions measured in issue order are non-decreasing.
    #[test]
    fn single_stream_completions_are_monotonic(
        sizes in prop::collection::vec(1usize..4096, 1..32),
        gaps in prop::collection::vec(0u64..5000, 1..32),
    ) {
        let mut engine = StubCopyEngine::new(CopyModel::default(), DEVICE_BYTES, HOST_BYTES);
        let chunk = engine.allocate_chunk(HOST_BYTES).unwrap();
        let count = sizes.len().min(gaps.len());
        let mut last = 0u64;
        for i in 0..count {
            engine.advance(gaps[i]);
            let bytes = sizes[i];
            engine.write_device(dev(0, bytes), &vec![i as u8; bytes]);
            engine
                .d2h(Stream::Store, dev(0, bytes), host(chunk.id, 0, bytes))
                .unwrap();
            let event = engine.record(Stream::Store).unwrap();
            prop_assert!(engine.wait(event, u64::MAX).unwrap());
            let completion = engine.now_ns();
            prop_assert!(completion >= last, "completion {completion} < previous {last}");
            last = completion;
        }
    }
}
