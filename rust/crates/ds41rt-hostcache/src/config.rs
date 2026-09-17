//! The knobs. Every field is a CLI flag in the daemon (`--host-cache-<name>`, env
//! `DS41RT_HOST_CACHE_<NAME>`), parsed once at boot, logged, and exported with the metrics.
use crate::MAX_CONTEXT_TOKENS;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// When the device→host copy of a retained snapshot is issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoreMode {
    /// At retention, on the store stream: eviction later drops a clean snapshot for free.
    OnRetain,
    /// Only when the engine evicts the snapshot: no copies until pressure, at the price of a
    /// bounded wait on the eviction path.
    OnEvict,
}

/// Which retention banks the cache serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kinds {
    pub prompt: bool,
    pub turn: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Pinned host memory for the cache; zero leaves every engine path untouched.
    pub bytes: u64,
    /// Pinned allocation and registration granularity.
    pub chunk_bytes: u64,
    pub store: StoreMode,
    /// Longest the device-evict path waits for an in-flight store before dropping uncached.
    pub copy_budget_ns: u64,
    /// Longest a restore waits before the request falls through to prefill.
    pub restore_budget_ns: u64,
    /// How old the oldest pending store copy may grow before a prefill chunk boundary
    /// pauses to let it finish: at each `HostCache::prefill_hold` call, if the oldest
    /// in-flight store is older than this, the cache waits on its event for at most this
    /// long again. Zero disables the guard: the call is a no-op and moves no hold metric.
    /// The fleet-recommended value is 500 ms (see the daemon's flag help).
    pub store_pace_ns: u64,
    /// Snapshots outside `[min_tokens, max_tokens]` are not cached.
    pub min_tokens: u32,
    pub max_tokens: u32,
    pub kinds: Kinds,
}

impl Default for Config {
    /// The documented defaults: off, 256 MiB chunks, store on retain, 1 s / 500 ms budgets,
    /// 512 tokens minimum, both kinds.
    fn default() -> Self {
        Self {
            bytes: 0,
            chunk_bytes: 256 << 20,
            store: StoreMode::OnRetain,
            copy_budget_ns: 1_000_000_000,
            restore_budget_ns: 500_000_000,
            store_pace_ns: 0,
            min_tokens: 512,
            max_tokens: MAX_CONTEXT_TOKENS,
            kinds: Kinds {
                prompt: true,
                turn: true,
            },
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConfigError {
    #[error("chunk_bytes must be positive and at most bytes ({bytes})")]
    Chunk { bytes: u64 },
    #[error("min_tokens ({min}) exceeds max_tokens ({max})")]
    Tokens { min: u32, max: u32 },
}

impl Config {
    pub fn enabled(&self) -> bool {
        self.bytes > 0
    }
    /// Every invariant a boot must check before allocating: chunk fits the quota, token bounds
    /// are ordered. An empty (disabled) config is always valid.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.enabled() && (self.chunk_bytes == 0 || self.chunk_bytes > self.bytes) {
            return Err(ConfigError::Chunk { bytes: self.bytes });
        }
        if self.min_tokens > self.max_tokens {
            return Err(ConfigError::Tokens {
                min: self.min_tokens,
                max: self.max_tokens,
            });
        }
        Ok(())
    }
}
