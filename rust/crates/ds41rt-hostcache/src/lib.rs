//! Host-RAM write-behind cache of DS41RT retained snapshots.
//!
//! The coordinator retains completed prompts and completed turns as snapshots in two
//! device-resident radix banks that share the device KV pool with active requests. Under pool
//! pressure the engine evicts them and a returning conversation is prefilled from cold. This
//! crate keeps evicted snapshots in pinned host memory and restores them through the engine's
//! own fill path in milliseconds instead of seconds. Design of record:
//! `dsv41-flash-tp4-engram/research/afd-hostcache-design.md` in the recipes repo.
//!
//! Everything here is host-only and single-threaded: the cache lives on the scheduler thread,
//! the GPU copies asynchronously through a [`copy::CopyEngine`], and completion is polled per
//! tick. The daemon supplies the CUDA copy engine and three hooks; every suite in this crate
//! runs against [`copy::StubCopyEngine`] on a virtual clock. Budgeted waits (the prefill
//! pacing hold, eviction waits) are bounded by their configured budget; on engines that poll
//! between probes, a timed-out wait is bounded by the budget plus at most one poll quantum,
//! because the inter-poll sleep is clipped to the remaining budget.
//!
//! Module map: [`config`] (the knobs), [`pool`] (pinned slab pool), [`copy`] (copy engine trait
//! and stub), [`snapshot`] (host snapshots, exact page sharing, the reuse radix, eviction),
//! [`cache`] (the facade the engine calls), [`metrics`], [`sim`] (the simulator that drives the
//! functional, performance, concurrency and stability suites).

pub mod cache;
pub mod budget;
pub mod config;
pub mod copy;
pub mod metrics;
pub mod pool;
pub mod sim;
pub mod snapshot;

pub use ds41rt_core::prefix::SnapshotKind;

/// Bytes of one compressor source row (FP4 KV).
pub const SOURCE_ROW_BYTES: usize = 356;
/// Rows per source page.
pub const PAGE_ROWS: usize = 256;
/// Bytes of one source page: the unit of sharing and of copying.
pub const PAGE_BYTES: usize = SOURCE_ROW_BYTES * PAGE_ROWS;
/// Tokens per compressor group; five pages per group across the four compressors.
pub const GROUP_TOKENS: usize = 512;
pub const PAGES_PER_GROUP: usize = 5;
/// Compressor banks per snapshot; page lists are indexed by bank.
pub const COMPRESSORS: usize = 4;
/// Bytes per token of source pages, the number the capacity arithmetic uses.
pub const KV_BYTES_PER_TOKEN: usize = PAGES_PER_GROUP * PAGE_BYTES / GROUP_TOKENS;
/// One backbone window prefix and how many a tail carries.
pub const WINDOW_PREFIX_BYTES: usize = 128 * 528 + 8;
pub const WINDOW_PREFIXES: usize = 40;
/// One compressor prefix; one per compressor in the tail.
pub const COMPRESSOR_PREFIX_BYTES: usize = 4096;
/// Bytes of a snapshot's backbone tail (one arena slot).
pub const TAIL_BYTES: usize =
    WINDOW_PREFIXES * WINDOW_PREFIX_BYTES + COMPRESSORS * COMPRESSOR_PREFIX_BYTES;
/// dSpark draft rings.
pub const DRAFT_RING_BYTES: usize = 67_584;
pub const DRAFT_RINGS: usize = 3;
pub const DRAFT_BYTES: usize = DRAFT_RINGS * DRAFT_RING_BYTES;
/// Longest context the engine serves.
pub const MAX_CONTEXT_TOKENS: u32 = 1_048_576;
/// Tokens a partial match replays before the retained frontier (the engine's rule).
pub const REPLAY_WINDOW_TOKENS: usize = 128;

#[cfg(test)]
mod constants {
    use super::*;
    #[test]
    fn engine_sizes() {
        assert_eq!(PAGE_BYTES, 91_136);
        assert_eq!(KV_BYTES_PER_TOKEN, 890);
        assert_eq!(TAIL_BYTES, 2_720_064);
        assert_eq!(DRAFT_BYTES, 202_752);
    }
}
