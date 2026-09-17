//! Counters and gauges, exported under `host_cache` whether or not the cache is on. Counters are
//! monotonic; gauges are the current state. Single-threaded: the daemon publishes a `Snapshot`
//! per tick for its metrics endpoint.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    pub stores_issued: u64,
    pub stores_completed: u64,
    /// Completed stores that replaced a resident snapshot with the same kind and tokens.
    pub stores_replaced: u64,
    pub stores_failed: u64,
    /// Failed issues whose already-issued copies did not drain within the copy budget, so the
    /// plan's slabs stay held.
    pub store_drain_timeouts: u64,
    /// Sum of the completion latencies of every completed store copy (see
    /// `store_latency_buckets`).
    pub store_latency_sum_ns: u64,
    /// Latency histogram of completed store copies in nanoseconds: bounds
    /// `RESTORE_BUCKETS_NS`, plus an overflow bucket. One entry per completed store, recorded
    /// at the commit a completion drives — observed by `tick`, or by `before_device_evict`
    /// waiting a pending copy clean — so a completion that replaced a resident snapshot still
    /// lands exactly once. Failed copies and drain-timeout drops never commit and record
    /// nothing.
    pub store_latency_buckets: [u64; RESTORE_BUCKETS_NS.len() + 1],
    pub stores_skipped: u64,
    pub store_bytes: u64,
    pub pages_copied: u64,
    pub pages_shared: u64,
    pub lookups: u64,
    pub host_hits: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub restore_timeouts: u64,
    pub restore_failures: u64,
    /// Latency histogram of completed restores in nanoseconds: bounds `RESTORE_BUCKETS_NS`.
    pub restore_latency_buckets: [u64; RESTORE_BUCKETS_NS.len() + 1],
    pub restore_latency_sum_ns: u64,
    pub evict_waits: u64,
    pub evict_wait_ns: u64,
    pub evict_drops_uncached: u64,
    /// Prefill pacing holds (`HostCache::prefill_hold` calls that waited on the oldest
    /// pending store's event). One per hold call that found an overdue store, whether the
    /// wait completed or timed out. Zero whenever `store_pace_ns` is 0: with the knob off
    /// the pacing hold is a no-op and never counts (store metrics may still move, because
    /// the daemon's wrapper observes completions with `tick` on every prefill chunk
    /// regardless of the knob). Invariant: `prefill_hold_timeouts <= prefill_holds`.
    pub prefill_holds: u64,
    /// Sum of the wall nanoseconds every hold waited: each summand is at most
    /// `store_pace_ns` (exact on the stub; on engines that poll, at most one poll quantum
    /// more — see `HostCache::prefill_hold`), so `prefill_hold_ns_sum <= prefill_holds *
    /// store_pace_ns`. The completing store's own latency — which naturally includes the
    /// hold time it observed — is recorded by the store-latency histogram exactly once at
    /// its commit, never here, so a hold is never double-counted.
    pub prefill_hold_ns_sum: u64,
    /// Holds whose budgeted wait ran out before the store's event completed. Counted,
    /// never an error: the hold returns regardless and the request is never failed.
    pub prefill_hold_timeouts: u64,
    /// Device snapshots the engine evicted while the cache was on: incremented exactly once per
    /// `HostCache::before_device_evict` call — every snapshot the engine evicts from a device
    /// bank or via `make_room` — before any budget wait and whatever the outcome (host copy
    /// already complete, waited clean, dropped uncached, budget timeout). Invariants:
    /// `device_evictions >= evict_waits` and `device_evictions >= evict_drops_uncached`, since
    /// both are counted on paths this counter has already passed. A disabled cache counts
    /// nothing: the hook is a no-op and never executes.
    pub device_evictions: u64,
    pub host_evictions: u64,
    pub host_evicted_bytes: u64,
    /// Engine submissions issued so far: one per 1D copy and one per coalesced batch, so the
    /// fleet report can show submissions per snapshot before and after coalescing. Gauge,
    /// refreshed with the other pool gauges; engines without submission instrumentation
    /// report 0.
    pub copy_submissions: u64,
    pub resident_snapshots: u64,
    pub bytes_used: u64,
    pub quota_bytes: u64,
}

/// Upper bounds of the restore latency buckets (1 ms … 1 s), plus an overflow bucket.
pub const RESTORE_BUCKETS_NS: [u64; 7] = [
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
];

/// The live counters. `record_restore` and `record_store` place a latency in its bucket;
/// everything else is a plain increment so the facade's code reads as what it counts.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    snapshot: Snapshot,
}

/// The histogram bucket a latency of `ns` nanoseconds belongs to: the first bound it does not
/// exceed, or the overflow bucket past the last bound.
fn latency_bucket(ns: u64) -> usize {
    RESTORE_BUCKETS_NS
        .iter()
        .position(|&bound| ns <= bound)
        .unwrap_or(RESTORE_BUCKETS_NS.len())
}

impl Metrics {
    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
    }
    pub fn get_mut(&mut self) -> &mut Snapshot {
        &mut self.snapshot
    }
    pub fn record_restore(&mut self, latency_ns: u64, bytes: u64) {
        let s = &mut self.snapshot;
        s.restores += 1;
        s.restore_bytes += bytes;
        s.restore_latency_sum_ns += latency_ns;
        s.restore_latency_buckets[latency_bucket(latency_ns)] += 1;
    }
    pub fn record_store(&mut self, latency_ns: u64) {
        let s = &mut self.snapshot;
        s.store_latency_sum_ns += latency_ns;
        s.store_latency_buckets[latency_bucket(latency_ns)] += 1;
    }
    /// Count one prefill pacing hold: `ns` waited (at most the configured pace, plus at
    /// most one poll quantum on engines that poll) and whether the wait timed out. A store
    /// whose copy completes during the hold keeps its latency for the store histogram's
    /// single commit-time entry; only the wait itself is here.
    pub fn record_prefill_hold(&mut self, ns: u64, timed_out: bool) {
        let s = &mut self.snapshot;
        s.prefill_holds += 1;
        s.prefill_hold_ns_sum += ns;
        s.prefill_hold_timeouts += u64::from(timed_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_bucket_places_bounds_and_overflow() {
        assert_eq!(latency_bucket(0), 0);
        assert_eq!(latency_bucket(1), 0);
        // A bound belongs to its own bucket: buckets are inclusive upper bounds.
        for (index, &bound) in RESTORE_BUCKETS_NS.iter().enumerate() {
            assert_eq!(latency_bucket(bound), index);
            if index + 1 < RESTORE_BUCKETS_NS.len() {
                assert_eq!(latency_bucket(bound + 1), index + 1);
            }
        }
        assert_eq!(latency_bucket(u64::MAX), RESTORE_BUCKETS_NS.len());
    }

    #[test]
    fn record_store_mirrors_record_restore_bucket_placement() {
        let mut metrics = Metrics::default();
        for &latency in &[0, 1_000_000, 999_999, 10_000_000, 1_000_000_001] {
            metrics.record_restore(latency, 0);
            metrics.record_store(latency);
            let snapshot = metrics.snapshot();
            let restore: u64 = snapshot.restore_latency_buckets.iter().sum();
            let store: u64 = snapshot.store_latency_buckets.iter().sum();
            assert_eq!(restore, store, "latency {latency} placed differently");
        }
    }

    /// `record_prefill_hold` (packet HC-9) keeps its counters ordered and the sum within
    /// `holds * pace` for every mix of completions and timeouts.
    #[test]
    fn record_prefill_hold_keeps_its_invariants() {
        let mut metrics = Metrics::default();
        for &(ns, timed_out) in &[
            (0, false),
            (1_000_000, false),
            (50_000_000, true),
            (100_000_000, true),
        ] {
            metrics.record_prefill_hold(ns, timed_out);
            let snapshot = metrics.snapshot();
            assert!(snapshot.prefill_hold_timeouts <= snapshot.prefill_holds);
            assert!(snapshot.prefill_hold_ns_sum <= snapshot.prefill_holds * 100_000_000);
        }
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.prefill_holds, 4);
        assert_eq!(snapshot.prefill_hold_timeouts, 2);
        assert_eq!(snapshot.prefill_hold_ns_sum, 151_000_000);
    }
}
