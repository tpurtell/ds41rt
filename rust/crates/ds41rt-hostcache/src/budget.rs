//! Startup sizing for logical GPU + RAM capacity. A retained RAM copy is a
//! second representation of the same tokens, not another cached conversation.
//! The capacity below describes storage after reclaiming such GPU copies; it
//! does not promise that every retained conversation is simultaneously resident.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use crate::{pool::Layout, GROUP_TOKENS, PAGES_PER_GROUP};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Budget {
    pub device_tokens: u64,
    pub target_tokens: u64,
    pub host_tokens: u64,
    pub combined_tokens: u64,
    pub staging_tokens: u64,
    pub pinned_bytes: u64,
}

/// `device_tokens` excludes the device allocator's private-tail/COW headroom
/// and counts TP replicas once. Reserve one full snapshot for incoming stores,
/// separate chunks for each slab class, and boundary pages for both banks.
/// None of those bytes count toward the advertised logical token capacity.
pub fn plan(device_tokens: u64, leading_slots: u32, max_context: u32,
    chunk_bytes: u64, draft: bool) -> Result<Budget> {
    ensure!(leading_slots <= 128 && (1..=crate::MAX_CONTEXT_TOKENS).contains(&max_context),
        "invalid host cache capacity target");
    let target = u64::from(leading_slots) * u64::from(max_context);
    let mut budget = Budget { device_tokens, target_tokens: target, host_tokens: 0,
        combined_tokens: device_tokens, staging_tokens: 0, pinned_bytes: 0 };
    if leading_slots == 0 || device_tokens > target { return Ok(budget); }
    ensure!(chunk_bytes > 0, "host cache chunk must be positive");
    let group = GROUP_TOKENS as u64;
    // Strictly exceed the requested leading edge, even on an exact boundary.
    let groups = (target - device_tokens) / group + 1;
    let staging_groups = u64::from(max_context).div_ceil(group);
    let snapshots = 2 * u64::from(leading_slots) + 2;
    let page_groups = groups + staging_groups + snapshots;
    let layout = Layout::engine(0);
    let chunks = |slabs: u64, slab_bytes: usize| -> Result<u64> {
        let per_chunk = chunk_bytes / slab_bytes as u64;
        ensure!(per_chunk > 0, "host cache chunk is smaller than a snapshot slab");
        Ok(slabs.div_ceil(per_chunk))
    };
    let mut count = chunks(page_groups * PAGES_PER_GROUP as u64, layout.page)?
        + chunks(snapshots, layout.tail)?;
    if draft { count += chunks(snapshots, layout.draft)?; }
    budget.host_tokens = groups * group;
    budget.combined_tokens = device_tokens + budget.host_tokens;
    budget.staging_tokens = staging_groups * group;
    budget.pinned_bytes = count.checked_mul(chunk_bytes).context("host cache budget overflow")?;
    Ok(budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    const M: u64 = 1 << 20;
    #[test]
    fn planned_bytes_fit_payload_staging_and_both_snapshot_banks_in_real_slab_pool() -> Result<()> {
        use crate::pool::{testing::FakePinned, Class, SlabPool};
        let p = plan(14*M, 20, M as u32, 256*M, true)?;
        let mut memory = FakePinned::new(usize::MAX);
        let mut pool = SlabPool::new(p.pinned_bytes, (256*M) as usize, Layout::engine(0), &mut memory)?;
        let snapshots = 42;
        let pages = (p.host_tokens / 512 + p.staging_tokens / 512 + snapshots) * 5;
        for _ in 0..pages { pool.take(Class::Page)?; }
        for _ in 0..snapshots { pool.take(Class::Tail)?; pool.take(Class::Draft)?; }
        assert!(pool.bytes_used() <= p.pinned_bytes);
        pool.release_all(&mut memory)?;
        assert_eq!(memory.live_chunks(), 0);
        Ok(())
    }
    #[test]
    fn sizing_excludes_staging_and_slab_fragmentation_from_token_claims() -> Result<()> {
        for gpu in [0, 7*M, 14*M, 20*M] {
            let p = plan(gpu, 20, M as u32, 256*M, true)?;
            assert!(p.combined_tokens > 20*M);
            assert!(p.combined_tokens <= 20*M + GROUP_TOKENS as u64);
            assert_eq!(p.staging_tokens, M);
            assert_eq!(p.pinned_bytes % (256*M), 0);
            assert!(p.pinned_bytes > (p.host_tokens + p.staging_tokens) * crate::KV_BYTES_PER_TOKEN as u64);
        }
        assert_eq!(plan(21*M, 20, M as u32, 256*M, true)?.pinned_bytes, 0);
        assert_eq!(plan(0, 0, M as u32, 256*M, true)?.pinned_bytes, 0);
        Ok(())
    }
    #[test]
    fn larger_draft_and_chunk_overheads_never_reduce_available_capacity() -> Result<()> {
        let plain = plan(14*M, 20, M as u32, 256*M, false)?;
        let draft = plan(14*M, 20, M as u32, 256*M, true)?;
        assert!(draft.pinned_bytes > plain.pinned_bytes);
        assert_eq!(draft.host_tokens, plain.host_tokens);
        for chunk in [4*M, 8*M, 64*M, 256*M] {
            let p = plan(1024, 20, 8193, chunk, true)?;
            assert!(p.combined_tokens > 20*8193);
            assert_eq!(p.staging_tokens, 8704);
        }
        assert!(plan(0, 20, M as u32, 1024, true).is_err());
        assert!(plan(0, 129, M as u32, 256*M, true).is_err());
        assert!(plan(0, 20, 0, 256*M, true).is_err());
        Ok(())
    }
}
