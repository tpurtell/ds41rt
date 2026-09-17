//! Startup accounting for colocated compressed sources on a pair of GPUs.
//! Budgets are residual bytes after all non-global allocations and headroom.
//! No GPU allocation, device selection, or serving-loop policy occurs here.

// Dual-RTX owners preallocate execution storage before sizing the pool.
// Keep room for request-time graphs and transient admission allocations.
pub(crate) const RUNTIME_HEADROOM: usize = 800 * 1024 * 1024;
const DEFAULT_POOL_TOKENS: usize = 14 * 1_048_576;

/// CUDA module/stream setup follows the deferred expert loading phase. Keep
/// this separate from the existing request-time graph/admission headroom.
pub(crate) const EXPERT_SETUP_HEADROOM: usize = 64 * 1024 * 1024;

/// Prefix budgets include resident weights and peak transient loading storage.
/// Both ranks must fit independently; spare bytes cannot cross the PCIe link.
pub(crate) fn expert_layers(requested: super::LocalLayers, prefix_peak: &[[usize; 2]],
    available: [usize; 2]) -> anyhow::Result<usize> {
    use anyhow::ensure;
    ensure!(prefix_peak.len() == 40 && prefix_peak.windows(2).all(|w|
        w[0].iter().zip(w[1]).all(|(&a, b)| a <= b)), "invalid TP2 prefix budgets");
    let fits = |count: usize| prefix_peak[count-1].iter().zip(available).all(|(&used, free)| used <= free);
    let layers = match requested {
        super::LocalLayers::Count(count) => {
            ensure!((1..=40).contains(&count), "dual RTX expert layers must be 1..=40");
            count
        }
        super::LocalLayers::Auto => (20..=40).rev().find(|&count| fits(count))
            .ok_or_else(|| anyhow::anyhow!("minimum TP2 expert placement does not fit reserved memory"))?,
    };
    ensure!(fits(layers), "requested TP2 expert layers exceed the per-GPU memory budget");
    Ok(layers)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SourcePoolPlan {
    pub groups: usize,
    pub pages: [usize; 4],
    pub device_bytes: [usize; 2],
    pub unused_bytes: [usize; 2],
}

impl SourcePoolPlan {
    pub fn new(
        available: [usize; 2],
        owners: [usize; 4],
        page_bytes: usize,
        minimum_groups: usize,
        maximum_groups: usize,
        exact_global_bytes: Option<usize>,
    ) -> Result<Self, &'static str> {
        if page_bytes == 0 || minimum_groups == 0 || minimum_groups > maximum_groups {
            return Err("invalid compressed-pool geometry");
        }
        let ratios = [1usize, 1, 1, 2];
        let mut per_device = [0usize; 2];
        for (owner, ratio) in owners.into_iter().zip(ratios) {
            let bytes = page_bytes.checked_mul(ratio).ok_or("source page size overflow")?;
            let sum = per_device.get_mut(owner).ok_or("source GPU must be zero or one")?;
            *sum = sum.checked_add(bytes).ok_or("source group size overflow")?;
        }
        let group_bytes = page_bytes.checked_mul(5).ok_or("global group size overflow")?;
        let fits = per_device.iter().zip(available).filter(|(bytes, _)| **bytes != 0)
            .map(|(bytes, budget)| budget / bytes).min().ok_or("no source owners")?
            .min(maximum_groups);
        let groups = match exact_global_bytes {
            // Match the existing planner: byte requests round down to complete
            // groups, never borrow unused bytes from the other device.
            Some(bytes) => bytes / group_bytes,
            None => fits,
        };
        if groups < minimum_groups || groups > fits {
            return Err("compressed pool does not fit per-GPU capacity or minimum admission");
        }
        let device_bytes = per_device.map(|bytes| bytes * groups);
        Ok(Self {
            groups,
            pages: [groups, groups, groups, groups.checked_mul(2).ok_or("page count overflow")?],
            device_bytes,
            unused_bytes: [available[0] - device_bytes[0], available[1] - device_bytes[1]],
        })
    }
}

/// Complete cache allocation plan, evaluated after fixed owners and snapshot
/// arenas are live. The byte ceiling applies independently to each GPU.
#[derive(Debug)]
pub(crate) struct PoolPlan {
    pub pages: [usize; 4],
    pub global_bytes: usize,
    pub cache_bytes: [usize; 2],
    pub occupied_before: [usize; 2],
    pub reservation_bytes: [usize; 2],
    pub unused_bytes: [usize; 2],
    pub desired_groups: usize,
}
impl PoolPlan {
    pub fn new(placement: crate::v41_backbone_cache::CachePlacement,
        slots: usize, context: usize, retained_turns: usize, _snapshot_bytes: usize,
        exact: Option<super::ByteSize>, reservation: Option<super::Reservation>,
        memory: [(usize, usize); 2]) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        use super::{BackboneCache, GROUP_BYTES, MAX_GROUPS};
        ensure!((1..=16).contains(&slots), "invalid concurrency limit");
        ensure!((1..=1_048_576).contains(&context), "invalid pool context limit");
        ensure!(retained_turns <= 128, "invalid retained-turn limit");
        let mut occupied_before = [0; 2];
        let mut reservation_bytes = [0; 2];
        let mut available = [0; 2];
        for (gpu, (free, total)) in memory.into_iter().enumerate() {
            ensure!(total > 0 && free <= total, "invalid GPU {gpu} memory information");
            occupied_before[gpu] = total - free;
            reservation_bytes[gpu] = reservation.map(|r| r.bytes(total)).transpose()?.unwrap_or(total);
            available[gpu] = reservation_bytes[gpu].checked_sub(occupied_before[gpu])
                .and_then(|n| n.checked_sub(RUNTIME_HEADROOM))
                .with_context(|| format!("GPU {gpu} reservation leaves no cache space after fixed owners and runtime headroom"))?;
        }
        let cache_bytes = |groups| BackboneCache::distributed_device_bytes(placement, slots,
            [groups, groups, groups, groups * 2]);
        let fits = |groups| -> anyhow::Result<bool> {
            Ok(cache_bytes(groups)?.iter().zip(available).all(|(&used, free)| used <= free))
        };
        let minimum = 2 * slots + 2 * retained_turns;
        ensure!(fits(minimum)?, "distributed cache cannot fit minimum admission and copy-on-write tails");
        let (mut low, mut high) = (minimum, MAX_GROUPS);
        while low < high {
            let mid = (low + high).div_ceil(2);
            if fits(mid)? { low = mid; } else { high = mid - 1; }
        }
        let desired_groups = if let Some(exact) = exact {
            exact.0 / GROUP_BYTES
        } else if reservation.is_some() {
            low
        } else {
            // Snapshot arenas are already charged to fixed occupancy. Cap the
            // default shared pool at 14M tokens, leaving runtime graph space;
            // concurrency and per-request context limits remain independent.
            // Explicit byte/reservation overrides retain their existing meaning.
            ((context.div_ceil(512) * slots).min(DEFAULT_POOL_TOKENS / 512)
                + slots + 2 * retained_turns).max(minimum)
        };
        ensure!(desired_groups >= minimum && desired_groups <= MAX_GROUPS,
            "distributed KV pool is outside admission or physical page limits");
        // The default targets the existing aggregate context capacity but may
        // shrink to the tighter card. Explicit byte requests must fit exactly
        // after whole-group rounding; never silently shrink an override.
        ensure!(exact.is_none() || desired_groups <= low,
            "explicit KV pool needs {desired_groups} groups but the tighter GPU fits {low}");
        let groups = desired_groups.min(low);
        let cache_bytes = cache_bytes(groups)?;
        Ok(Self { pages: [groups, groups, groups, groups * 2],
            global_bytes: groups * GROUP_BYTES, cache_bytes, occupied_before, reservation_bytes,
            unused_bytes: std::array::from_fn(|gpu| available[gpu] - cache_bytes[gpu]), desired_groups })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_expert_placement_preserves_pool_and_obeys_tighter_rank() -> anyhow::Result<()> {
        let totals = [101_973_491_712usize, 101_970_345_984];
        // Measured 20-layer EXL3 fixed occupancy, before its KV allocation.
        let occupied = [76_599_787_520usize, 78_055_211_008];
        let memory = std::array::from_fn(|gpu| (totals[gpu] - occupied[gpu], totals[gpu]));
        let placement = crate::v41_backbone_cache::CachePlacement::encoder_decoder();
        let pool = PoolPlan::new(placement, 16, 1_048_576, 24, 146_150_400, None, None, memory)?;
        let layer_bytes = 2_789_290_000usize;
        let prefix: Vec<_> = (1..=40).map(|n| [n*layer_bytes; 2]).collect();
        let available = std::array::from_fn(|gpu| prefix[19][gpu] + pool.unused_bytes[gpu] - EXPERT_SETUP_HEADROOM);
        assert_eq!(expert_layers(super::super::LocalLayers::Auto, &prefix, available)?, 25);
        assert!(expert_layers(super::super::LocalLayers::Count(26), &prefix, available).is_err());
        let mut after = memory;
        for gpu in 0..2 { after[gpu].0 -= 5*layer_bytes + EXPERT_SETUP_HEADROOM; }
        let retained = PoolPlan::new(placement, 16, 1_048_576, 24, 146_150_400,
            Some(super::super::ByteSize(pool.global_bytes)), None, after)?;
        assert_eq!(retained.pages, pool.pages);
        assert_eq!(expert_layers(super::super::LocalLayers::Auto, &prefix,
            [available[0], 24*layer_bytes])?, 24);
        assert!(expert_layers(super::super::LocalLayers::Auto, &prefix,
            [available[0], 20*layer_bytes-1]).is_err());
        // Explicit placement can return expert memory to replicated KV without
        // changing the existing automatic launcher boundary before negotiation.
        assert_eq!(expert_layers(super::super::LocalLayers::Count(17), &prefix,
            [17*layer_bytes, 18*layer_bytes])?, 17);
        assert!(expert_layers(super::super::LocalLayers::Count(18), &prefix,
            [17*layer_bytes, 18*layer_bytes]).is_err());
        Ok(())
    }

    #[test]
    fn measured_dual_budget_reserves_fourteen_full_contexts_and_graph_headroom() -> anyhow::Result<()> {
        let totals = [101_973_491_712usize, 101_970_345_984];
        let occupied = [92_070_477_824usize, 94_956_158_976];
        let memory = std::array::from_fn(|gpu| (totals[gpu] - occupied[gpu], totals[gpu]));
        let plan = PoolPlan::new(crate::v41_backbone_cache::CachePlacement::encoder_decoder(),
            16, 1_048_576, 24, 146_150_400, None, None, memory)?;
        assert_eq!(plan.pages, [28_736, 28_736, 28_736, 57_472]);
        assert_eq!(plan.global_bytes, 13_094_420_480);
        assert!(plan.unused_bytes[1] >= 2 * 373_293_056);
        // The new default is not a hard cap on explicit user pool requests.
        let explicit = PoolPlan::new(crate::v41_backbone_cache::CachePlacement::encoder_decoder(),
            16, 1_048_576, 24, 146_150_400, Some(super::super::ByteSize(14_960_885_760)), None, memory)?;
        assert_eq!(explicit.pages, [32_832, 32_832, 32_832, 65_664]);
        let smaller = PoolPlan::new(crate::v41_backbone_cache::CachePlacement::encoder_decoder(),
            8, 1_048_576, 24, 146_150_400, None, None, memory)?;
        assert_eq!(smaller.pages[0], 8 * 2048 + 8 + 48);
        for gpu in 0..2 {
            assert!(occupied[gpu] + plan.cache_bytes[gpu] + RUNTIME_HEADROOM <= totals[gpu]);
        }
        Ok(())
    }

    #[test]
    fn complete_cache_respects_each_card_and_counts_non_global_storage() -> anyhow::Result<()> {
        use crate::v41_backbone_cache::{BackboneCache, CachePlacement};
        use super::super::{ByteSize, Reservation, GROUP_BYTES};
        let map = CachePlacement::encoder_decoder();
        let bytes = BackboneCache::distributed_device_bytes(map, 16, [100, 100, 100, 200])?;
        assert!(bytes.iter().sum::<usize>() > 100 * GROUP_BYTES);
        let memory = [(bytes[0] + RUNTIME_HEADROOM, 96 << 30), (20 << 30, 96 << 30)];
        let plan = PoolPlan::new(map, 16, 1_048_576, 24, 0, None,
            Some(Reservation::Percent(100_000_000)), memory)?;
        assert_eq!(plan.pages, [100, 100, 100, 200]);
        assert_eq!(plan.cache_bytes, bytes);
        assert_eq!(plan.unused_bytes[0], 0);
        assert!(plan.unused_bytes[1] > 0);
        assert!(PoolPlan::new(map, 16, 1_048_576, 24, 0,
            Some(ByteSize(101 * GROUP_BYTES)), None, memory).is_err());
        let rounded = PoolPlan::new(map, 16, 1_048_576, 24, 0,
            Some(ByteSize(101 * GROUP_BYTES - 1)), None, memory)?;
        assert_eq!(rounded.pages, plan.pages);
        let default = PoolPlan::new(map, 16, 1_048_576, 24, 0, None, None, memory)?;
        assert_eq!(default.pages, plan.pages);
        assert!(default.desired_groups > default.pages[0]);
        Ok(())
    }

    #[test]
    fn complete_cache_handles_second_card_limit_and_invalid_budgets() -> anyhow::Result<()> {
        use crate::v41_backbone_cache::{BackboneCache, CachePlacement};
        use super::super::Reservation;
        let map = CachePlacement::encoder_decoder();
        let bytes = BackboneCache::distributed_device_bytes(map, 2, [64, 64, 64, 128])?;
        let memory = [(20 << 30, 96 << 30), (bytes[1] + RUNTIME_HEADROOM, 96 << 30)];
        let plan = PoolPlan::new(map, 2, 4096, 24, 0, None,
            Some(Reservation::Percent(100_000_000)), memory)?;
        assert_eq!(plan.pages[0], 64);
        assert_eq!(plan.unused_bytes[1], 0);
        assert!(PoolPlan::new(map, 2, 4096, 24, 0, None,
            Some(Reservation::Percent(99_000_000)), memory).is_err());
        for invalid in [[(1, 0); 2], [(2, 1); 2], [(RUNTIME_HEADROOM, 96 << 30); 2]] {
            assert!(PoolPlan::new(map, 2, 4096, 24, 0, None, None, invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn asymmetric_sources_obey_the_tighter_card() {
        let p = SourcePoolPlan::new([3_000, 10_000], [0, 0, 0, 1], 100, 1, 100, None).unwrap();
        assert_eq!(p.pages, [10, 10, 10, 20]);
        assert_eq!(p.device_bytes, [3_000, 2_000]);
        assert_eq!(p.unused_bytes, [0, 8_000]);
        assert!(SourcePoolPlan::new([3_000, 10_000], [0, 0, 0, 1], 100, 1, 100, Some(5_500)).is_err());
    }

    #[test]
    fn swapped_devices_and_ratio_one_source_limit() {
        let p = SourcePoolPlan::new([800, 30_000], [1, 1, 1, 0], 100, 4, 100, None).unwrap();
        assert_eq!(p.groups, 4);
        assert_eq!(p.device_bytes, [800, 1_200]);
        assert!(SourcePoolPlan::new([799, 30_000], [1, 1, 1, 0], 100, 4, 100, None).is_err());
    }

    #[test]
    fn exact_rounding_and_physical_limit() {
        let p = SourcePoolPlan::new([10_000; 2], [0, 0, 0, 1], 100, 1, 8, Some(4_499)).unwrap();
        assert_eq!(p.groups, 8);
        assert!(SourcePoolPlan::new([10_000; 2], [0, 0, 0, 1], 100, 1, 8, Some(4_500)).is_err());
    }

    #[test]
    fn invalid_geometry_rejects_without_arithmetic_wrap() {
        for (owners, page, min, max) in [([0, 0, 0, 2], 100, 1, 8),
            ([0, 0, 0, 1], 0, 1, 8), ([0, 0, 0, 1], usize::MAX, 1, 8),
            ([0, 0, 0, 1], 100, 0, 8), ([0, 0, 0, 1], 100, 9, 8)] {
            assert!(SourcePoolPlan::new([usize::MAX; 2], owners, page, min, max, None).is_err());
        }
    }
}
