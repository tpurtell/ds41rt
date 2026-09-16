//! Candidate paired H128 ownership planner. Not yet connected to serving.
//!
//! Costs use a caller-selected common integer unit (for example estimated ns).
//! The caller includes weight streaming and routed-row reuse in each marginal
//! block cost; this module deliberately does not invent a hardware cost model.
use crate::Ds41rtError;
use std::ops::Range;

/// Four mandatory H128 blocks and one duplicated boundary block per rank.
/// Ranks 1 and 3 store their optional block first; ranks 0 and 2 store it last.
pub const EXL3_TP4_RESIDENT_BLOCKS: [Range<usize>; 4] = [0..5, 4..9, 9..14, 13..18];

#[derive(Clone, Copy, Debug, Default)]
pub struct Exl3BoundaryCost {
    pub active: bool,
    /// Marginal cost of this expert's boundary block in each of the two pairs.
    pub pair_cost: [u64; 2],
}

pub struct Exl3Tp4OwnershipPlan<'a> {
    /// Bit 0 selects rank 1 instead of 0; bit 1 selects rank 3 instead of 2.
    /// Inactive experts have value 255 and must never be dispatched.
    pub owners: &'a [u8],
    pub rank_cost: [u64; 4],
}

/// One instance per lane/batch owner. Planning does not allocate on success.
pub struct Exl3Tp4OwnershipPlanner {
    order: Vec<usize>,
    owners: Vec<u8>,
}

impl Exl3Tp4OwnershipPlanner {
    pub fn new(experts: usize) -> Self {
        Self {
            order: (0..experts).collect(),
            owners: vec![255; experts],
        }
    }

    /// Assign largest marginal blocks first, independently in each pair.
    /// `mandatory_cost` accounts for the four always-owned blocks on each rank.
    /// The returned borrow prevents reuse of this planner while its plan lives.
    pub fn plan(
        &mut self,
        costs: &[Exl3BoundaryCost],
        mandatory_cost: [u64; 4],
        tie_seed: usize,
    ) -> Result<Exl3Tp4OwnershipPlan<'_>, Ds41rtError> {
        if costs.len() != self.owners.len() {
            return Err(rejected("ownership expert extent differs"));
        }
        self.owners.fill(255);
        for (owner, cost) in self.owners.iter_mut().zip(costs) {
            if cost.active {
                *owner = 0;
            }
        }
        let mut rank_cost = mandatory_cost;
        for pair in 0..2 {
            // Unstable sort is in-place. Expert ID makes equal-cost ordering
            // deterministic even after an earlier batch permuted the scratch.
            self.order.sort_unstable_by(|&a, &b| {
                costs[b]
                    .active
                    .cmp(&costs[a].active)
                    .then_with(|| costs[b].pair_cost[pair].cmp(&costs[a].pair_cost[pair]))
                    .then_with(|| a.cmp(&b))
            });
            for &expert in &self.order {
                if !costs[expert].active {
                    break;
                }
                let first = pair * 2;
                let second = first + 1;
                let side = match rank_cost[first].cmp(&rank_cost[second]) {
                    std::cmp::Ordering::Less => 0,
                    std::cmp::Ordering::Greater => 1,
                    std::cmp::Ordering::Equal => (tie_seed ^ expert ^ pair) & 1,
                };
                let rank = first + side;
                rank_cost[rank] = rank_cost[rank]
                    .checked_add(costs[expert].pair_cost[pair])
                    .ok_or_else(|| rejected("ownership cost overflow"))?;
                self.owners[expert] |= (side as u8) << pair;
            }
        }
        Ok(Exl3Tp4OwnershipPlan {
            owners: &self.owners,
            rank_cost,
        })
    }
}

/// Active local block range within the fixed five-block physical allocation.
/// Physical tensor strides remain five blocks regardless of this range.
pub fn exl3_tp4_active_blocks(rank: usize, owners: u8) -> Option<Range<usize>> {
    if rank >= 4 || owners > 3 {
        return None;
    }
    let side = rank & 1;
    let selected = ((owners >> (rank / 2)) & 1) as usize;
    Some(if side == selected {
        0..5
    } else if side == 0 {
        0..4
    } else {
        1..5
    })
}

fn rejected(reason: &str) -> Ds41rtError {
    Ds41rtError::ExpertRoutePlanRejected {
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ownership_combination_covers_each_global_block_exactly_once() {
        for owners in 0..4 {
            let mut coverage = [0; 18];
            for (rank, resident) in EXL3_TP4_RESIDENT_BLOCKS.iter().enumerate() {
                let active = exl3_tp4_active_blocks(rank, owners).unwrap();
                assert!(active.len() == 4 || active.len() == 5);
                for local in active {
                    coverage[resident.start + local] += 1;
                }
            }
            assert_eq!(coverage, [1; 18]);
        }
        assert!(exl3_tp4_active_blocks(4, 0).is_none());
        assert!(exl3_tp4_active_blocks(0, 255).is_none());
    }

    #[test]
    fn six_equal_experts_balance_twenty_seven_blocks_per_rank() {
        let costs = [Exl3BoundaryCost {
            active: true,
            pair_cost: [1, 1],
        }; 6];
        let mut planner = Exl3Tp4OwnershipPlanner::new(6);
        for seed in 0..8 {
            let plan = planner.plan(&costs, [24; 4], seed).unwrap();
            assert_eq!(plan.rank_cost, [27; 4]);
            assert!(plan.owners.iter().all(|&owner| owner < 4));
        }
    }

    #[test]
    fn weighted_pairs_reuse_and_inactive_experts_do_not_share_decisions() {
        let costs = [
            Exl3BoundaryCost {
                active: true,
                pair_cost: [9, 1],
            },
            Exl3BoundaryCost {
                active: true,
                pair_cost: [5, 5],
            },
            Exl3BoundaryCost {
                active: true,
                pair_cost: [4, 8],
            },
            Exl3BoundaryCost {
                active: false,
                pair_cost: [u64::MAX; 2],
            },
        ];
        let mut planner = Exl3Tp4OwnershipPlanner::new(4);
        let expected = planner.plan(&costs, [0; 4], 0).unwrap().owners.to_vec();
        assert_eq!(
            planner.plan(&costs, [0; 4], 0).unwrap().rank_cost,
            [9, 9, 6, 8]
        );
        let empty = [Exl3BoundaryCost::default(); 4];
        assert_eq!(planner.plan(&empty, [0; 4], 0).unwrap().owners, &[255; 4]);
        assert_eq!(planner.plan(&costs, [0; 4], 0).unwrap().owners, expected);
        let plan = planner.plan(&costs, [100, 0, 0, 100], 0).unwrap();
        assert_eq!(plan.rank_cost, [100, 18, 14, 100]);
        assert_eq!(plan.owners[3], 255);
    }

    #[test]
    fn rejected_input_can_be_followed_by_a_clean_plan() {
        let mut planner = Exl3Tp4OwnershipPlanner::new(1);
        let costs = [Exl3BoundaryCost {
            active: true,
            pair_cost: [1; 2],
        }];
        assert!(planner.plan(&[], [0; 4], 0).is_err());
        assert!(planner.plan(&costs, [u64::MAX; 4], 0).is_err());
        assert_eq!(
            planner.plan(&costs, [0; 4], 0).unwrap().rank_cost,
            [1, 0, 0, 1]
        );
    }
}
