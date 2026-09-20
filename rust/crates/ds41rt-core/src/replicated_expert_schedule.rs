//! Deterministic CPU-only replicated expert-group scheduling for the routed FFN.
//!
//! The official backbone routes each token to top-6 of 384 routed experts per
//! layer. A deployment may hold `1`, `2` or `3` replicated expert groups that
//! each carry an identical copy of every routed expert. This module decides,
//! per layer/request, which group executes each *active* expert. It is a pure
//! CPU planner: it has no protocol, transport, topology or GPU dependency, and
//! it does not bind any wire format. The internal tensor-parallel width of a
//! group is irrelevant to the scheduling decision.
//!
//! Policy: whole-expert, largest-processing-time-first (LPT) greedy.
//!
//! 1. Every expert with at least one routed row is *active*; an expert with zero
//!    routed rows is left at the [`ReplicatedExpertGroupId::INACTIVE`] sentinel.
//! 2. Each active expert receives an estimated integer cost from the configured
//!    [`ReplicatedExpertCostModel`] (weight-residency cost plus activation-tile
//!    cost). Costs are model inputs, not measured coefficients.
//! 3. Active experts are ordered by descending estimated cost, with a
//!    deterministic tie break derived from a *local* tie seed, and each is
//!    assigned to the group with the smallest current predicted load.
//!
//! Whole-expert assignment preserves weight reuse: an expert's weights are
//! executed once, in one group, for all of its routed rows. The policy cannot
//! split a single dominant expert across groups, so a request whose routes
//! concentrate on one expert is bounded below by that expert's full cost even
//! when several groups are idle. Hot-expert splitting is deliberately not
//! implemented here; it would be a separate, opt-in change justified by
//! benchmarks.
//!
//! Determinism and allocation:
//!
//! - Output is a pure function of `(config, route_counts, tie_seed,
//!   expert_capacity)`. No process-global or cross-lane state is read.
//! - The plan borrows the scheduler's preallocated buffers. A successful plan
//!   whose histogram length is within the planned capacity performs no heap
//!   allocation. `plan` rejects a larger histogram instead of growing; callers
//!   grow explicitly with [`ReplicatedExpertScheduler::reserve_expert_capacity`].
//! - The caller copies or encodes the borrowed assignment into a request-owned
//!   buffer before the next plan reuses the scheduler.

use crate::Ds41rtError;

/// Maximum number of replicated expert groups this planner supports.
pub const MAX_REPLICATED_EXPERT_GROUPS: u8 = 3;

/// Encoded form of [`ReplicatedExpertGroupId::INACTIVE`].
pub const INACTIVE_REPLICATED_EXPERT_GROUP: u8 = u8::MAX;

/// Deterministic integer cost model for one whole routed expert.
///
/// All values are abstract *cost units* (CU). They are not microseconds, bytes,
/// FLOPs or any other physical quantity, and this module ships no calibrated
/// production coefficients. Choose values that make the relative ordering and
/// the group loads meaningful for the deployment being planned; the scheduler
/// only relies on the ordering and on exact integer arithmetic.
///
/// The estimated cost of an active expert with `routed_rows` routed rows is:
///
/// ```text
/// expert_weight_cost + ceil(routed_rows / tile_rows) * tile_cost
/// ```
///
/// The weight term is charged **once per active expert**, never once per routed
/// row, because whole-expert assignment executes each expert's weights once.
/// The activation term models the padded routed-row tile work on a group.
/// The production-shaped default is the pure-weight form of this expression:
/// a uniform per-expert weight cost plus `ceil(count / tile_rows) * tile_cost`
/// with no additional terms. Whether specific numbers are feasible for a
/// deployment must be established by measurement; this module intentionally
/// exposes the model instead of assuming coefficients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedExpertCostModel {
    /// Cost of executing one active expert's resident weights, in cost units.
    pub expert_weight_cost: u64,
    /// Cost of one routed activation tile, in cost units.
    pub tile_cost: u64,
    /// Routed rows per activation tile. Must be non-zero.
    pub tile_rows: u32,
}

impl ReplicatedExpertCostModel {
    /// Creates a model. See [`ReplicatedExpertCostModel::validate`].
    pub const fn new(expert_weight_cost: u64, tile_cost: u64, tile_rows: u32) -> Self {
        Self {
            expert_weight_cost,
            tile_cost,
            tile_rows,
        }
    }

    /// Rejects a model that cannot produce a positive cost for an active expert
    /// or that has a zero tile height.
    pub fn validate(&self) -> Result<(), Ds41rtError> {
        if self.tile_rows == 0 {
            return Err(rejected("cost model tile_rows must be non-zero"));
        }
        if self.expert_weight_cost == 0 && self.tile_cost == 0 {
            return Err(rejected(
                "cost model must give active experts a positive cost \
                 (expert_weight_cost and tile_cost are both zero)",
            ));
        }
        Ok(())
    }

    /// Estimated cost of an expert with `routed_rows` routed rows.
    ///
    /// A zero-row expert is inactive and costs nothing; this is the only case
    /// where the weight term is not charged.
    pub fn expert_cost(&self, routed_rows: u64) -> Result<u64, Ds41rtError> {
        self.validate()?;
        if routed_rows == 0 {
            return Ok(0);
        }
        let tiles = routed_rows.div_ceil(self.tile_rows as u64);
        let activation_cost = tiles
            .checked_mul(self.tile_cost)
            .ok_or_else(|| rejected("cost model activation-tile cost overflow"))?;
        self.expert_weight_cost
            .checked_add(activation_cost)
            .ok_or_else(|| rejected("cost model expert cost overflow"))
    }
}

/// Static planner configuration. Per-request inputs (histogram and tie seed)
/// are passed to [`ReplicatedExpertScheduler::plan`] so one scheduler can serve
/// a whole lane without rebuilding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedExpertScheduleConfig {
    /// Replicated groups to distribute experts across: `1`, `2` or `3`.
    pub group_count: u8,
    /// Integer cost model. See [`ReplicatedExpertCostModel`].
    pub cost_model: ReplicatedExpertCostModel,
}

impl ReplicatedExpertScheduleConfig {
    pub const fn new(group_count: u8, cost_model: ReplicatedExpertCostModel) -> Self {
        Self {
            group_count,
            cost_model,
        }
    }

    /// Rejects a zero or unsupported group count and an invalid cost model.
    pub fn validate(&self) -> Result<(), Ds41rtError> {
        if self.group_count == 0 {
            return Err(rejected("group_count must be non-zero"));
        }
        if self.group_count > MAX_REPLICATED_EXPERT_GROUPS {
            return Err(rejected(format!(
                "group_count {} exceeds the supported maximum {MAX_REPLICATED_EXPERT_GROUPS}",
                self.group_count
            )));
        }
        self.cost_model.validate()
    }
}

/// Target group for one expert, or the inactive sentinel.
///
/// The encoded byte is `0..group_count` for an assigned expert and
/// [`INACTIVE_REPLICATED_EXPERT_GROUP`] for an expert with no routed rows, so a
/// caller can copy the assignment straight into a request-owned byte buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ReplicatedExpertGroupId(u8);

impl ReplicatedExpertGroupId {
    /// Sentinel for an expert with no routed rows in this plan.
    pub const INACTIVE: Self = Self(INACTIVE_REPLICATED_EXPERT_GROUP);

    /// Builds an active group id for `index`, or `None` if it is out of range.
    pub const fn active(index: u8) -> Option<Self> {
        if index < MAX_REPLICATED_EXPERT_GROUPS {
            Some(Self(index))
        } else {
            None
        }
    }

    /// True when this id names a real replicated group.
    pub const fn is_active(self) -> bool {
        self.0 != INACTIVE_REPLICATED_EXPERT_GROUP
    }

    /// Group index when active.
    pub const fn index(self) -> Option<u8> {
        if self.is_active() {
            Some(self.0)
        } else {
            None
        }
    }

    /// Raw byte form: group index, or [`INACTIVE_REPLICATED_EXPERT_GROUP`].
    pub const fn encoded(self) -> u8 {
        self.0
    }
}

/// Predicted per-group load for one plan, in the same cost units as the model.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplicatedExpertGroupLoad {
    /// Sum of estimated expert costs assigned to this group (predicted
    /// makespan driver; the maximum over groups is the predicted makespan).
    pub cost: u64,
    /// Number of active experts assigned to this group.
    pub expert_count: u32,
    /// Sum of routed rows over the experts assigned to this group.
    pub routed_rows: u64,
}

/// Borrowed assignment and predicted loads for one histogram.
///
/// All slices borrow the producing [`ReplicatedExpertScheduler`]; the borrow
/// ends when this value is dropped, so the scheduler cannot be replanned while
/// a plan is live.
#[derive(Clone, Copy, Debug)]
pub struct ReplicatedExpertGroupPlan<'a> {
    assignment: &'a [ReplicatedExpertGroupId],
    expert_cost: &'a [u64],
    group_loads: &'a [ReplicatedExpertGroupLoad],
    total_cost: u64,
    group_count: u8,
    tie_seed: u64,
}

impl<'a> ReplicatedExpertGroupPlan<'a> {
    /// Expert-id-indexed target group, one entry per histogram entry. Experts
    /// with no routed rows hold [`ReplicatedExpertGroupId::INACTIVE`].
    pub fn assignment(&self) -> &'a [ReplicatedExpertGroupId] {
        self.assignment
    }

    /// Expert-id-indexed estimated cost; zero exactly for inactive experts.
    pub fn expert_cost(&self) -> &'a [u64] {
        self.expert_cost
    }

    /// Per-group predicted cost/load, `group_count` entries in group order.
    pub fn group_loads(&self) -> &'a [ReplicatedExpertGroupLoad] {
        self.group_loads
    }

    /// Sum of all active expert costs.
    pub fn total_cost(&self) -> u64 {
        self.total_cost
    }

    /// Number of replicated groups in this plan.
    pub fn group_count(&self) -> u8 {
        self.group_count
    }

    /// Tie seed used to order equal-cost experts and equal-load groups.
    pub fn tie_seed(&self) -> u64 {
        self.tie_seed
    }

    /// Assignment of one expert id, or `None` when the id is out of range.
    pub fn group_for_expert(&self, expert_id: usize) -> Option<ReplicatedExpertGroupId> {
        self.assignment.get(expert_id).copied()
    }
}

/// Lane-owned, reusable replicated expert-group planner.
///
/// A scheduler owns its scratch and output buffers. It reads no global state,
/// so two lanes planning concurrently produce independent results, and a plan
/// is a pure function of its inputs.
pub struct ReplicatedExpertScheduler {
    config: ReplicatedExpertScheduleConfig,
    planned_expert_capacity: usize,
    assignment: Vec<ReplicatedExpertGroupId>,
    expert_cost: Vec<u64>,
    tie_rank: Vec<u64>,
    order: Vec<usize>,
    group_loads: Vec<ReplicatedExpertGroupLoad>,
}

impl ReplicatedExpertScheduler {
    /// Creates a scheduler with preallocated capacity for `expert_capacity`
    /// histogram entries. `expert_capacity` is the production expert count
    /// (384 for the official backbone), not the routed-row count.
    pub fn new(
        config: ReplicatedExpertScheduleConfig,
        expert_capacity: usize,
    ) -> Result<Self, Ds41rtError> {
        config.validate()?;
        let group_load_slots = config.group_count as usize;
        Ok(Self {
            config,
            planned_expert_capacity: expert_capacity,
            assignment: Vec::with_capacity(expert_capacity),
            expert_cost: Vec::with_capacity(expert_capacity),
            tie_rank: Vec::with_capacity(expert_capacity),
            order: Vec::with_capacity(expert_capacity),
            group_loads: Vec::with_capacity(group_load_slots),
        })
    }

    /// Static configuration this scheduler was built with.
    pub fn config(&self) -> ReplicatedExpertScheduleConfig {
        self.config
    }

    /// Histogram length this scheduler can plan without allocating.
    pub fn expert_capacity(&self) -> usize {
        self.planned_expert_capacity
    }

    /// Grows the preallocated capacity so later plans of up to `capacity`
    /// experts are allocation-free. This is the only sizing operation that
    /// allocates, and it is explicit so no request path allocates implicitly.
    pub fn reserve_expert_capacity(&mut self, capacity: usize) -> Result<(), Ds41rtError> {
        self.config.validate()?;
        if capacity > self.planned_expert_capacity {
            reserve_to(&mut self.assignment, capacity);
            reserve_to(&mut self.expert_cost, capacity);
            reserve_to(&mut self.tie_rank, capacity);
            reserve_to(&mut self.order, capacity);
            self.planned_expert_capacity = capacity;
        }
        Ok(())
    }

    /// Plans whole-expert assignment for `route_counts`.
    ///
    /// `route_counts[expert_id]` is the number of routed rows that selected
    /// that expert for this layer/request; the official backbone passes 384
    /// entries. `tie_seed` must be derived from local layer/request identity
    /// only, e.g. with [`replicated_expert_tie_seed`].
    ///
    /// On success every active expert appears exactly once in the returned
    /// assignment, and no expert contribution is duplicated. On error the
    /// scheduler is unchanged for planning purposes: a later valid plan behaves
    /// as if the failed call had not happened.
    pub fn plan(
        &mut self,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Result<ReplicatedExpertGroupPlan<'_>, Ds41rtError> {
        self.config.validate()?;
        let expert_count = route_counts.len();
        if expert_count > self.planned_expert_capacity {
            return Err(rejected(format!(
                "histogram length {expert_count} exceeds planned expert capacity {}; \
                 call reserve_expert_capacity outside the request path",
                self.planned_expert_capacity
            )));
        }
        let group_count = self.config.group_count as usize;

        self.assignment.clear();
        self.assignment
            .resize(expert_count, ReplicatedExpertGroupId::INACTIVE);
        self.expert_cost.clear();
        self.expert_cost.resize(expert_count, 0);
        self.tie_rank.clear();
        self.tie_rank.resize(expert_count, 0);
        self.order.clear();
        self.group_loads.clear();
        self.group_loads
            .resize(group_count, ReplicatedExpertGroupLoad::default());

        let model = self.config.cost_model;
        let mut total_cost = 0_u64;
        for (expert_id, &routed_rows) in route_counts.iter().enumerate() {
            if routed_rows == 0 {
                continue;
            }
            let cost = model.expert_cost(routed_rows as u64)?;
            self.expert_cost[expert_id] = cost;
            total_cost = total_cost
                .checked_add(cost)
                .ok_or_else(|| rejected("plan total estimated cost overflow"))?;
            self.tie_rank[expert_id] = expert_tie_rank(tie_seed, expert_id);
            self.order.push(expert_id);
        }

        {
            let order = &mut self.order;
            let expert_cost = &self.expert_cost;
            let tie_rank = &self.tie_rank;
            order.sort_unstable_by(|left, right| {
                expert_cost[*right]
                    .cmp(&expert_cost[*left])
                    .then_with(|| tie_rank[*left].cmp(&tie_rank[*right]))
                    .then_with(|| left.cmp(right))
            });
        }

        let mut group_rank = [0_u64; MAX_REPLICATED_EXPERT_GROUPS as usize];
        for (group, rank) in group_rank.iter_mut().enumerate().take(group_count) {
            *rank = group_tie_rank(tie_seed, group);
        }

        for order_index in 0..self.order.len() {
            let expert_id = self.order[order_index];
            let mut best_group = 0_usize;
            for group in 1..group_count {
                if group_load_better(
                    self.group_loads[group].cost,
                    group_rank[group],
                    group,
                    self.group_loads[best_group].cost,
                    group_rank[best_group],
                    best_group,
                ) {
                    best_group = group;
                }
            }
            self.assignment[expert_id] = ReplicatedExpertGroupId::active(best_group as u8)
                .ok_or_else(|| rejected("plan selected a group outside the supported range"))?;
            let load = &mut self.group_loads[best_group];
            load.cost = load
                .cost
                .checked_add(self.expert_cost[expert_id])
                .ok_or_else(|| rejected("plan group cost overflow"))?;
            load.expert_count = load
                .expert_count
                .checked_add(1)
                .ok_or_else(|| rejected("plan group expert count overflow"))?;
            load.routed_rows = load
                .routed_rows
                .checked_add(route_counts[expert_id] as u64)
                .ok_or_else(|| rejected("plan group routed-row count overflow"))?;
        }

        Ok(ReplicatedExpertGroupPlan {
            assignment: &self.assignment,
            expert_cost: &self.expert_cost,
            group_loads: &self.group_loads,
            total_cost,
            group_count: self.config.group_count,
            tie_seed,
        })
    }
}

/// Deterministic tie seed from purely local scheduling identity.
///
/// The seed depends only on the layer id and request id supplied by the caller.
/// It never reads process-global, GPU, transport or cross-lane state, so two
/// independent lanes cannot perturb each other's tie breaks.
pub const fn replicated_expert_tie_seed(layer_id: u32, request_id: u64) -> u64 {
    mix64(mix64(request_id) ^ (layer_id as u64))
}

/// Opt-in tie-seed derivation for the replicated whole-expert planner.
///
/// Both modes feed the same [`replicated_expert_tie_seed`] mix. `Dispatch`
/// (default, and the only production mode) uses the caller's per-dispatch
/// request id, reproducing historical behavior byte for byte. `Layer` pins the
/// request component to [`TIE_SEED_FIXED_REQUEST_ID`], so the seed depends on
/// the layer alone; this makes two dispatches of the same work at the same layer
/// produce the same ownership when their route histogram is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicatedExpertTieSeedMode {
    /// Historical behavior: seed from `(layer_id, dispatch request_id)`.
    Dispatch,
    /// Directional control: seed from `(layer_id, TIE_SEED_FIXED_REQUEST_ID)`.
    Layer,
}

/// Request-id component used by [`ReplicatedExpertTieSeedMode::Layer`].
///
/// Zero is a fixed sentinel, not a real request id: it is hashed through the
/// existing `mix64(request_id)` leg, so `layer` mode is a documented special
/// case of the same function rather than a second seed algorithm.
pub const TIE_SEED_FIXED_REQUEST_ID: u64 = 0;

impl ReplicatedExpertTieSeedMode {
    /// Parse an opt-in environment value.
    ///
    /// `None` (unset) selects [`Self::Dispatch`]. Only the exact lowercase
    /// tokens `dispatch` and `layer` are accepted; every other value, including
    /// an empty string, is rejected so a typo fails closed at startup instead of
    /// silently changing ownership.
    pub fn from_env_value(value: Option<&str>) -> Result<Self, Ds41rtError> {
        match value {
            None | Some("dispatch") => Ok(Self::Dispatch),
            Some("layer") => Ok(Self::Layer),
            Some(other) => Err(Ds41rtError::UnknownReplicatedExpertTieSeed(other.to_owned())),
        }
    }

    /// Stable lowercase name for logs and tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dispatch => "dispatch",
            Self::Layer => "layer",
        }
    }
}

/// Tie seed for `mode`. `Dispatch` is byte-identical to
/// [`replicated_expert_tie_seed`]. `Layer` reuses the same mix with the request
/// component pinned to [`TIE_SEED_FIXED_REQUEST_ID`].
///
/// Neither mode makes the resulting assignment independent of the route
/// histogram: a changed batching/route composition can still repartition the
/// groups, so this is not a universal bit-reproducibility guarantee.
pub const fn replicated_expert_tie_seed_for(
    mode: ReplicatedExpertTieSeedMode,
    layer_id: u32,
    request_id: u64,
) -> u64 {
    match mode {
        ReplicatedExpertTieSeedMode::Dispatch => replicated_expert_tie_seed(layer_id, request_id),
        ReplicatedExpertTieSeedMode::Layer => {
            replicated_expert_tie_seed(layer_id, TIE_SEED_FIXED_REQUEST_ID)
        }
    }
}

fn expert_tie_rank(seed: u64, expert_id: usize) -> u64 {
    mix64(seed ^ mix64(expert_id as u64))
}

fn group_tie_rank(seed: u64, group: usize) -> u64 {
    // Distinct tag so equal-cost expert ranks and equal-load group ranks do not
    // share a permutation.
    mix64(seed ^ mix64(0x4752_4F55_505F_5449 ^ group as u64))
}

fn group_load_better(
    candidate_cost: u64,
    candidate_rank: u64,
    candidate_group: usize,
    current_cost: u64,
    current_rank: u64,
    current_group: usize,
) -> bool {
    candidate_cost < current_cost
        || (candidate_cost == current_cost
            && (candidate_rank < current_rank
                || (candidate_rank == current_rank && candidate_group < current_group)))
}

fn reserve_to<T>(values: &mut Vec<T>, capacity: usize) {
    if values.capacity() < capacity {
        // `Vec::reserve` takes an additional count relative to `len`, not a new
        // absolute capacity. Reserving `capacity - capacity()` would under-grow
        // whenever `len` is non-zero, so reserve the shortfall against `len`.
        // The guard plus the `len <= capacity()` invariant makes this positive.
        values.reserve(capacity - values.len());
    }
}

/// SplitMix64 finalizer: deterministic, allocation-free, no external state.
const fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn rejected(reason: impl Into<String>) -> Ds41rtError {
    Ds41rtError::ExpertRoutePlanRejected {
        reason: format!("replicated expert schedule: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    type Snapshot = (Vec<u8>, Vec<(u64, u32, u64)>, u64);

    fn model(expert_weight_cost: u64, tile_cost: u64, tile_rows: u32) -> ReplicatedExpertCostModel {
        ReplicatedExpertCostModel::new(expert_weight_cost, tile_cost, tile_rows)
    }

    fn config(
        group_count: u8,
        cost_model: ReplicatedExpertCostModel,
    ) -> ReplicatedExpertScheduleConfig {
        ReplicatedExpertScheduleConfig::new(group_count, cost_model)
    }

    fn scheduler(
        group_count: u8,
        cost_model: ReplicatedExpertCostModel,
        expert_capacity: usize,
    ) -> ReplicatedExpertScheduler {
        ReplicatedExpertScheduler::new(config(group_count, cost_model), expert_capacity).unwrap()
    }

    fn snapshot(
        scheduler: &mut ReplicatedExpertScheduler,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Snapshot {
        let plan = scheduler.plan(route_counts, tie_seed).unwrap();
        (
            plan.assignment()
                .iter()
                .map(|group| group.encoded())
                .collect(),
            plan.group_loads()
                .iter()
                .map(|load| (load.cost, load.expert_count, load.routed_rows))
                .collect(),
            plan.total_cost(),
        )
    }

    /// Encoded per-expert group assignment for one plan.
    fn assignment_of(
        scheduler: &mut ReplicatedExpertScheduler,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Vec<u8> {
        scheduler
            .plan(route_counts, tie_seed)
            .unwrap()
            .assignment()
            .iter()
            .map(|group| group.encoded())
            .collect()
    }

    fn base4_counts(mut encoded: usize, expert_count: usize) -> Vec<u32> {
        (0..expert_count)
            .map(|_| {
                let count = (encoded % 4) as u32;
                encoded /= 4;
                count
            })
            .collect()
    }

    fn brute_force_optimal_max(costs: &[u64], group_count: usize) -> u64 {
        if costs.is_empty() {
            return 0;
        }
        let mut best = u64::MAX;
        let mut loads = vec![0_u64; group_count];
        for assignment in 0..group_count.pow(costs.len() as u32) {
            loads.fill(0);
            let mut code = assignment;
            for &cost in costs {
                let group = code % group_count;
                code /= group_count;
                loads[group] += cost;
            }
            best = best.min(loads.iter().copied().max().unwrap_or(0));
        }
        best
    }

    /// Deliberately naive, test-only baseline: assign active experts to groups
    /// round-robin in ascending expert order. It is not a production strategy.
    fn round_robin_baseline_max_load(
        route_counts: &[u32],
        group_count: usize,
        cost_model: ReplicatedExpertCostModel,
    ) -> u64 {
        let mut loads = vec![0_u64; group_count];
        let mut next_group = 0_usize;
        for &routed_rows in route_counts {
            if routed_rows == 0 {
                continue;
            }
            loads[next_group % group_count] += cost_model.expert_cost(routed_rows as u64).unwrap();
            next_group += 1;
        }
        loads.into_iter().max().unwrap_or(0)
    }

    #[test]
    fn six_single_row_experts_balance_across_replicated_groups() {
        let route_counts = [1_u32; 6];
        let cost_model = model(100, 10, 1);

        let mut two_groups = scheduler(2, cost_model, route_counts.len());
        let plan = two_groups.plan(&route_counts, 7).unwrap();
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.expert_count)
                .collect::<Vec<_>>(),
            vec![3, 3]
        );
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.cost)
                .collect::<Vec<_>>(),
            vec![3 * 110, 3 * 110]
        );
        assert_eq!(plan.total_cost(), 6 * 110);

        let mut three_groups = scheduler(3, cost_model, route_counts.len());
        let plan = three_groups.plan(&route_counts, 7).unwrap();
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.expert_count)
                .collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
        assert_eq!(
            plan.group_loads()
                .iter()
                .map(|load| load.cost)
                .collect::<Vec<_>>(),
            vec![2 * 110, 2 * 110, 2 * 110]
        );
        assert!(plan.assignment().iter().all(|group| group.is_active()));
    }

    #[test]
    fn weight_cost_is_charged_once_per_active_expert_not_per_row() {
        let cost_model = model(500, 3, 2);
        let mut planner = scheduler(2, cost_model, 3);
        let plan = planner.plan(&[5, 0, 0], 1).unwrap();

        // ceil(5 / 2) = 3 activation tiles; the 500 weight cost appears once.
        assert_eq!(plan.expert_cost()[0], 500 + 3 * 3);
        assert_ne!(plan.expert_cost()[0], 5 * 500 + 3 * 3);
        assert_eq!(plan.total_cost(), 509);
        assert_eq!(
            plan.group_loads()[plan.group_for_expert(0).unwrap().index().unwrap() as usize].cost,
            509
        );

        // A pure-weight model charges the weight once for any row count.
        let mut pure_weight = scheduler(1, model(1000, 1, 8), 2);
        let plan = pure_weight.plan(&[8, 0], 1).unwrap();
        assert_eq!(plan.expert_cost()[0], 1001);
        assert_ne!(plan.expert_cost()[0], 8 * 1000);
    }

    #[test]
    fn tile_costs_use_ceiling_boundaries() {
        let cost_model = model(100, 7, 4);
        let mut planner = scheduler(2, cost_model, 9);
        let plan = planner.plan(&[0, 1, 2, 3, 4, 5, 7, 8, 9], 3).unwrap();

        assert_eq!(
            plan.expert_cost(),
            &[0, 107, 107, 107, 107, 114, 114, 114, 121]
        );
        assert!(!plan.group_for_expert(0).unwrap().is_active());
        assert!(plan.group_for_expert(1).unwrap().is_active());
        assert_eq!(cost_model.expert_cost(0).unwrap(), 0);
        assert_eq!(cost_model.expert_cost(4).unwrap(), 107);
        assert_eq!(cost_model.expert_cost(5).unwrap(), 114);
    }

    #[test]
    fn empty_and_all_zero_histograms_are_inactive() {
        let mut planner = scheduler(2, model(10, 1, 1), 6);

        {
            let plan = planner.plan(&[], 5).unwrap();
            assert!(plan.assignment().is_empty());
            assert_eq!(plan.total_cost(), 0);
            assert_eq!(
                plan.group_loads(),
                &[
                    ReplicatedExpertGroupLoad::default(),
                    ReplicatedExpertGroupLoad::default()
                ]
            );
            assert!(plan.group_for_expert(0).is_none());
        }

        let all_zero = [0_u32; 6];
        let (assignment, loads, total) = snapshot(&mut planner, &all_zero, 5);
        assert!(assignment
            .iter()
            .all(|raw| *raw == INACTIVE_REPLICATED_EXPERT_GROUP));
        assert_eq!(loads, vec![(0, 0, 0), (0, 0, 0)]);
        assert_eq!(total, 0);
    }

    #[test]
    fn single_group_receives_every_active_expert() {
        let mut planner = scheduler(1, model(4, 2, 2), 4);
        // Costs: count 2 -> 4 + 1*2 = 6, count 3 -> 4 + 2*2 = 8, count 1 -> 6.
        let (assignment, loads, total) = snapshot(&mut planner, &[2, 0, 3, 1], 9);

        assert_eq!(assignment[0], 0);
        assert_eq!(assignment[1], INACTIVE_REPLICATED_EXPERT_GROUP);
        assert_eq!(assignment[2], 0);
        assert_eq!(assignment[3], 0);
        assert_eq!(loads, vec![(20, 3, 6)]);
        assert_eq!(total, 20);
    }

    #[test]
    fn seeded_tie_breaks_are_deterministic_and_seed_sensitive() {
        let route_counts = [1_u32; 6];
        let mut planner = scheduler(2, model(100, 10, 1), 6);
        let first = snapshot(&mut planner, &route_counts, 111);
        let second = snapshot(&mut planner, &route_counts, 111);
        assert_eq!(first, second);
        assert_eq!(
            first.0,
            snapshot(&mut scheduler(2, model(100, 10, 1), 6), &route_counts, 111).0
        );

        let mut distinct = BTreeSet::new();
        for seed in 0..16_u64 {
            let mut fresh = scheduler(2, model(100, 10, 1), 6);
            distinct.insert(snapshot(&mut fresh, &route_counts, seed).0);
        }
        assert!(
            distinct.len() > 1,
            "tie seed must influence equal-cost assignment"
        );
    }

    #[test]
    fn tie_seed_is_a_pure_function_of_local_identity() {
        assert_eq!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(7, 42)
        );
        assert_ne!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(8, 42)
        );
        assert_ne!(
            replicated_expert_tie_seed(7, 42),
            replicated_expert_tie_seed(7, 43)
        );
    }

    #[test]
    fn tie_seed_mode_parser_defaults_to_dispatch_and_fails_closed() {
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(None).unwrap(),
            ReplicatedExpertTieSeedMode::Dispatch
        );
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(Some("dispatch")).unwrap(),
            ReplicatedExpertTieSeedMode::Dispatch
        );
        assert_eq!(
            ReplicatedExpertTieSeedMode::from_env_value(Some("layer")).unwrap(),
            ReplicatedExpertTieSeedMode::Layer
        );
        for invalid in [
            "", "Dispatch", "LAYER", "layer ", " layer", "fixed", "histogram", "logical", "0",
        ] {
            assert!(
                ReplicatedExpertTieSeedMode::from_env_value(Some(invalid)).is_err(),
                "invalid value {invalid:?} must fail closed"
            );
        }
        assert_eq!(ReplicatedExpertTieSeedMode::Dispatch.as_str(), "dispatch");
        assert_eq!(ReplicatedExpertTieSeedMode::Layer.as_str(), "layer");
    }

    #[test]
    fn dispatch_mode_is_byte_identical_to_the_legacy_seed() {
        for (layer, request) in [(0_u32, 1_u64), (7, 42), (39, u64::MAX)] {
            assert_eq!(
                replicated_expert_tie_seed_for(
                    ReplicatedExpertTieSeedMode::Dispatch,
                    layer,
                    request
                ),
                replicated_expert_tie_seed(layer, request)
            );
        }
    }

    #[test]
    fn layer_mode_is_stable_across_request_ids() {
        for layer in [0_u32, 7, 39] {
            let expected = replicated_expert_tie_seed(layer, TIE_SEED_FIXED_REQUEST_ID);
            for request in [0_u64, 1, 42, u64::MAX] {
                assert_eq!(
                    replicated_expert_tie_seed_for(
                        ReplicatedExpertTieSeedMode::Layer,
                        layer,
                        request
                    ),
                    expected
                );
            }
        }
    }

    #[test]
    fn layer_mode_assignment_is_stable_across_request_ids_for_fixed_histogram() {
        // Uniform costs make every ordering decision a tie, so only the seed can
        // change the assignment. Layer mode must ignore every request id.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        let seed = |request: u64| {
            replicated_expert_tie_seed_for(ReplicatedExpertTieSeedMode::Layer, 5, request)
        };
        let base = assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed(1));
        for request in [0_u64, 2, 99, u64::MAX] {
            assert_eq!(
                assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed(request)),
                base
            );
        }
    }

    #[test]
    fn dispatch_mode_assignment_varies_with_request_id_for_fixed_histogram() {
        // This is the sensitivity the `layer` control removes; show it exists on
        // the same tie-heavy histogram with a deterministic search.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        let assignment = |request: u64| {
            let seed = replicated_expert_tie_seed_for(
                ReplicatedExpertTieSeedMode::Dispatch,
                5,
                request,
            );
            assignment_of(&mut scheduler(2, cost_model, 4), &histogram, seed)
        };
        let base = assignment(1);
        assert!(
            (2..64).any(|request| assignment(request) != base),
            "dispatch-mode owners must depend on the request id"
        );
    }

    #[test]
    fn histogram_changes_repartition_even_in_layer_mode() {
        // Honest limitation guard: layer mode is not histogram-invariant.
        let cost_model = model(1, 0, 16);
        let seed = replicated_expert_tie_seed_for(ReplicatedExpertTieSeedMode::Layer, 5, 0);
        let assignment = |histogram: &[u32]| {
            let mut s = ReplicatedExpertScheduler::new(config(2, cost_model), histogram.len())
                .unwrap();
            let plan = s.plan(histogram, seed).unwrap();
            plan.assignment().iter().map(|g| g.encoded()).collect::<Vec<u8>>()
        };
        let base = assignment(&[3, 3, 3, 3]);
        let changed = [[1_u32, 0, 0, 0], [0, 2, 0, 2], [4, 1, 0, 0], [0, 0, 0, 9]]
            .iter()
            .any(|candidate| assignment(candidate) != base);
        assert!(changed, "histogram changes must be able to repartition owners");
    }

    #[test]
    fn single_group_ownership_is_seed_and_request_independent() {
        // EP1 negative: with one group the seed cannot affect ownership.
        let cost_model = model(1, 0, 16);
        let histogram = [3_u32, 3, 3, 3];
        for mode in [
            ReplicatedExpertTieSeedMode::Dispatch,
            ReplicatedExpertTieSeedMode::Layer,
        ] {
            for request in [1_u64, 42, u64::MAX] {
                let seed = replicated_expert_tie_seed_for(mode, 5, request);
                let assignment =
                    assignment_of(&mut scheduler(1, cost_model, 4), &histogram, seed);
                assert!(
                    assignment.iter().all(|&group| group == 0),
                    "single-group ownership must always be group 0"
                );
            }
        }
    }

    #[test]
    fn expert_permutation_preserves_group_load_multiset() {
        // Distinct route counts give distinct estimated costs, so LPT ordering
        // is permutation-invariant apart from tie ranks; compare the sorted
        // load distribution rather than per-group labels.
        let cost_model = model(10, 1, 1);
        let route_counts = [9_u32, 5, 3, 1];
        let permuted = [3_u32, 9, 1, 5];

        let (_, base_loads, base_total) =
            snapshot(&mut scheduler(2, cost_model, 4), &route_counts, 5);
        let (_, permuted_loads, permuted_total) =
            snapshot(&mut scheduler(2, cost_model, 4), &permuted, 5);

        let mut base_sorted = base_loads;
        let mut permuted_sorted = permuted_loads;
        base_sorted.sort_unstable();
        permuted_sorted.sort_unstable();
        assert_eq!(base_sorted, permuted_sorted);
        assert_eq!(base_total, permuted_total);
        assert_eq!(base_total, 4 * 10 + 18);
    }

    #[test]
    fn independent_schedulers_keep_independent_scratch() {
        let route_counts = [4_u32, 4, 4, 4];
        let mut lane_a = scheduler(2, model(3, 1, 1), 4);
        let mut lane_b = scheduler(3, model(5, 1, 1), 4);

        let a_first = snapshot(&mut lane_a, &route_counts, 21);
        let b_first = snapshot(&mut lane_b, &route_counts, 22);
        let a_second = snapshot(&mut lane_a, &route_counts, 21);
        let b_second = snapshot(&mut lane_b, &route_counts, 22);

        assert_eq!(a_first, a_second);
        assert_eq!(b_first, b_second);
        assert_ne!(a_first.1.len(), b_first.1.len());
        assert_eq!(
            a_first,
            snapshot(&mut scheduler(2, model(3, 1, 1), 4), &route_counts, 21)
        );
        assert_eq!(
            b_first,
            snapshot(&mut scheduler(3, model(5, 1, 1), 4), &route_counts, 22)
        );
    }

    #[test]
    fn plan_borrows_outputs_and_preserves_inputs() {
        let route_counts = [3_u32, 0, 2];
        let before = route_counts;
        let mut planner = scheduler(2, model(5, 2, 2), 3);

        // The plan holds an exclusive borrow of the scheduler for its whole
        // lifetime, so the scheduler cannot be replanned while it is live. The
        // scope below records the borrowed slice addresses as plain integers.
        let (assignment_ptr, expert_cost_ptr, group_loads_ptr) = {
            let plan = planner.plan(&route_counts, 11).unwrap();
            assert_eq!(plan.tie_seed(), 11);
            assert_eq!(plan.group_count(), 2);
            assert_eq!(
                plan.group_for_expert(1).unwrap().encoded(),
                INACTIVE_REPLICATED_EXPERT_GROUP
            );
            (
                plan.assignment().as_ptr() as usize,
                plan.expert_cost().as_ptr() as usize,
                plan.group_loads().as_ptr() as usize,
            )
        };

        assert_eq!(route_counts, before);
        assert_eq!(assignment_ptr, planner.assignment.as_ptr() as usize);
        assert_eq!(expert_cost_ptr, planner.expert_cost.as_ptr() as usize);
        assert_eq!(group_loads_ptr, planner.group_loads.as_ptr() as usize);
    }

    #[test]
    fn invalid_configurations_and_overflow_are_rejected_then_recovered() {
        // Group-count bounds and zero-cost models are rejected at construction.
        assert!(ReplicatedExpertScheduler::new(config(0, model(1, 1, 1)), 4).is_err());
        assert!(ReplicatedExpertScheduler::new(
            config(MAX_REPLICATED_EXPERT_GROUPS + 1, model(1, 1, 1)),
            4
        )
        .is_err());
        assert!(ReplicatedExpertScheduler::new(config(2, model(1, 1, 0)), 4).is_err());
        assert!(ReplicatedExpertScheduler::new(config(2, model(0, 0, 1)), 4).is_err());
        assert!(model(1, 1, 0).expert_cost(1).is_err());
        assert!(model(0, 0, 1).expert_cost(1).is_err());
        assert_eq!(model(1, 1, 1).expert_cost(0).unwrap(), 0);

        // Group id sentinel boundary.
        assert!(ReplicatedExpertGroupId::active(MAX_REPLICATED_EXPERT_GROUPS - 1).is_some());
        assert!(ReplicatedExpertGroupId::active(MAX_REPLICATED_EXPERT_GROUPS).is_none());
        assert!(!ReplicatedExpertGroupId::INACTIVE.is_active());
        assert_eq!(ReplicatedExpertGroupId::INACTIVE.index(), None);
        assert_eq!(
            ReplicatedExpertGroupId::INACTIVE.encoded(),
            INACTIVE_REPLICATED_EXPERT_GROUP
        );

        // Per-expert cost overflow, then recovery on the same scheduler.
        let mut overflow = scheduler(2, model(u64::MAX, 1, 1), 4);
        assert!(overflow.plan(&[1, 0, 0, 0], 3).is_err());
        let recovered = snapshot(&mut overflow, &[0, 0, 0, 0], 3);
        assert!(recovered
            .0
            .iter()
            .all(|raw| *raw == INACTIVE_REPLICATED_EXPERT_GROUP));
        assert_eq!(recovered.2, 0);

        // Total-cost overflow, then a valid plan on the same scheduler.
        let half = u64::MAX / 2;
        let mut total_overflow = scheduler(2, model(half, 0, 1), 4);
        assert!(total_overflow.plan(&[1, 1, 1, 0], 3).is_err());
        let recovered = snapshot(&mut total_overflow, &[1, 1, 0, 0], 3);
        assert_eq!(recovered.2, half * 2);

        // Capacity overflow is explicit and recoverable via reserve.
        let mut capacity = scheduler(2, model(1, 1, 1), 2);
        assert!(capacity.plan(&[1, 1, 1, 1], 3).is_err());
        assert_eq!(capacity.expert_capacity(), 2);
        capacity.reserve_expert_capacity(4).unwrap();
        assert_eq!(capacity.expert_capacity(), 4);
        let recovered = snapshot(&mut capacity, &[1, 1, 1, 1], 3);
        assert_eq!(recovered.2, 8);
    }

    #[test]
    fn reserve_expert_capacity_is_honored_before_and_after_planning() {
        // Before the first plan every buffer is empty. A reserve that used the
        // old capacity instead of len would under-grow here and then reallocate
        // during the first larger plan.
        let mut fresh = scheduler(2, model(3, 1, 1), 2);
        fresh.reserve_expert_capacity(6).unwrap();
        assert_eq!(fresh.expert_capacity(), 6);
        assert!(fresh.assignment.capacity() >= 6);
        assert!(fresh.expert_cost.capacity() >= 6);
        assert!(fresh.tie_rank.capacity() >= 6);
        assert!(fresh.order.capacity() >= 6);

        let counts = [1_u32; 6];
        let _ = fresh.plan(&counts, 2).unwrap();
        let fresh_capacities = (
            fresh.assignment.capacity(),
            fresh.expert_cost.capacity(),
            fresh.tie_rank.capacity(),
            fresh.order.capacity(),
        );
        let fresh_expected = snapshot(&mut fresh, &counts, 2);
        for _ in 0..64 {
            assert_eq!(snapshot(&mut fresh, &counts, 2), fresh_expected);
        }
        assert_eq!(
            fresh_capacities,
            (
                fresh.assignment.capacity(),
                fresh.expert_cost.capacity(),
                fresh.tie_rank.capacity(),
                fresh.order.capacity(),
            ),
            "a plan within the reserved capacity must not reallocate"
        );

        // After a short plan the buffers have non-zero len, which is the case a
        // reserve measured against the old capacity miscounts.
        let mut used = scheduler(2, model(3, 1, 1), 4);
        let _ = used.plan(&[1, 1], 1).unwrap();
        assert_eq!(used.expert_capacity(), 4);
        used.reserve_expert_capacity(8).unwrap();
        assert_eq!(used.expert_capacity(), 8);
        assert!(used.assignment.capacity() >= 8);
        assert!(used.expert_cost.capacity() >= 8);
        assert!(used.tie_rank.capacity() >= 8);
        assert!(used.order.capacity() >= 8);

        let counts = [1_u32; 8];
        let _ = used.plan(&counts, 3).unwrap();
        let used_capacities = (
            used.assignment.capacity(),
            used.expert_cost.capacity(),
            used.tie_rank.capacity(),
            used.order.capacity(),
            used.group_loads.capacity(),
        );
        let used_expected = snapshot(&mut used, &counts, 3);
        for _ in 0..64 {
            assert_eq!(snapshot(&mut used, &counts, 3), used_expected);
        }
        assert_eq!(
            used_capacities,
            (
                used.assignment.capacity(),
                used.expert_cost.capacity(),
                used.tie_rank.capacity(),
                used.order.capacity(),
                used.group_loads.capacity(),
            ),
            "a plan at the reserved capacity must not reallocate"
        );
    }

    #[test]
    fn whole_expert_policy_keeps_dominant_expert_unsplit() {
        // One expert owns most routed rows. Whole-expert assignment must place
        // its full cost in exactly one group; the predicted makespan cannot
        // drop below that cost, which is the documented policy limitation.
        let route_counts = [64_u32, 1, 1, 1];
        let mut planner = scheduler(2, model(0, 1, 1), 4);
        let plan = planner.plan(&route_counts, 9).unwrap();

        let dominant_group = plan.group_for_expert(0).unwrap();
        assert!(dominant_group.is_active());
        assert_eq!(plan.expert_cost()[0], 64);
        assert_eq!(
            plan.assignment()
                .iter()
                .filter(|group| **group == dominant_group)
                .count(),
            1
        );
        let dominant_load = plan.group_loads()[dominant_group.index().unwrap() as usize];
        assert!(dominant_load.cost >= 64);
        assert_eq!(plan.total_cost(), 67);
        assert_eq!(
            plan.group_loads().iter().map(|load| load.cost).sum::<u64>(),
            plan.total_cost()
        );
    }

    #[test]
    fn exhaustive_small_histograms_cover_each_active_expert_once() {
        for group_count in 1..=MAX_REPLICATED_EXPERT_GROUPS {
            for expert_count in 0..=4_usize {
                for encoded in 0..4_usize.pow(expert_count as u32) {
                    let route_counts = base4_counts(encoded, expert_count);
                    let cost_model = model(7, 3, 2);
                    let mut planner = scheduler(group_count, cost_model, expert_count);
                    let seed = 1234 + encoded as u64 + expert_count as u64;
                    let plan = planner.plan(&route_counts, seed).unwrap();

                    let mut active_costs = Vec::new();
                    let mut cost_by_group = vec![0_u64; group_count as usize];
                    let mut experts_by_group = vec![0_u32; group_count as usize];
                    let mut rows_by_group = vec![0_u64; group_count as usize];
                    for (expert_id, &routed_rows) in route_counts.iter().enumerate() {
                        let group = plan.group_for_expert(expert_id).unwrap();
                        if routed_rows == 0 {
                            assert!(!group.is_active());
                            assert_eq!(plan.expert_cost()[expert_id], 0);
                            continue;
                        }
                        assert!(group.is_active());
                        let group_index = group.index().unwrap() as usize;
                        assert!(group_index < group_count as usize);
                        let expected = 7 + (routed_rows as u64).div_ceil(2) * 3;
                        assert_eq!(plan.expert_cost()[expert_id], expected);
                        active_costs.push(expected);
                        cost_by_group[group_index] += expected;
                        experts_by_group[group_index] += 1;
                        rows_by_group[group_index] += routed_rows as u64;
                    }

                    // Every active expert is covered exactly once.
                    assert_eq!(
                        experts_by_group.iter().sum::<u32>(),
                        route_counts.iter().filter(|count| **count > 0).count() as u32
                    );
                    assert_eq!(
                        rows_by_group.iter().sum::<u64>(),
                        route_counts.iter().map(|count| *count as u64).sum::<u64>()
                    );
                    for (group, load) in plan.group_loads().iter().enumerate() {
                        assert_eq!(load.cost, cost_by_group[group]);
                        assert_eq!(load.expert_count, experts_by_group[group]);
                        assert_eq!(load.routed_rows, rows_by_group[group]);
                    }
                    assert_eq!(plan.total_cost(), cost_by_group.iter().sum::<u64>());
                    assert_eq!(plan.total_cost(), plan.expert_cost().iter().sum::<u64>());

                    // LPT is 4/3-competitive; assert the loose bound and never
                    // assert optimality.
                    let optimal = brute_force_optimal_max(&active_costs, group_count as usize);
                    let greedy = plan
                        .group_loads()
                        .iter()
                        .map(|load| load.cost)
                        .max()
                        .unwrap_or(0);
                    assert!(
                        greedy * 3 <= optimal * 4,
                        "greedy {greedy} exceeded 4/3 of optimal {optimal} \
                         for groups={group_count} counts={route_counts:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn greedy_lpt_matches_or_beats_test_only_round_robin_baseline() {
        let route_counts = [8_u32, 1, 1, 1, 1];
        let cost_model = model(0, 1, 1);
        let mut planner = scheduler(2, cost_model, route_counts.len());
        let (_, loads, _) = snapshot(&mut planner, &route_counts, 4);

        let greedy_max = loads.iter().map(|load| load.0).max().unwrap();
        let baseline_max = round_robin_baseline_max_load(&route_counts, 2, cost_model);
        assert_eq!(greedy_max, 8);
        assert_eq!(baseline_max, 10);
        assert!(greedy_max <= baseline_max);
    }

    #[test]
    fn production_shape_repeated_plans_are_allocation_stable() {
        // Cheap CPU regression that doubles as a microbenchmark shape check: it
        // makes no timing or GPU-speedup claim. 384 experts matches the official
        // backbone. Cost values are placeholders, not calibrated coefficients.
        let route_counts: Vec<u32> = (0..384)
            .map(|expert_id| ((expert_id * 37 + 11) % 9) as u32)
            .collect();
        let cost_model = model(2304, 1536, 64);
        let mut planner = scheduler(3, cost_model, 384);

        // Warm up so any first-use growth is done before capacity is recorded.
        let _ = planner.plan(&route_counts, 42).unwrap();
        let capacities = (
            planner.assignment.capacity(),
            planner.expert_cost.capacity(),
            planner.tie_rank.capacity(),
            planner.order.capacity(),
            planner.group_loads.capacity(),
        );
        let expected = snapshot(&mut planner, &route_counts, 42);

        for _ in 0..512 {
            assert_eq!(snapshot(&mut planner, &route_counts, 42), expected);
        }
        assert_eq!(
            capacities,
            (
                planner.assignment.capacity(),
                planner.expert_cost.capacity(),
                planner.tie_rank.capacity(),
                planner.order.capacity(),
                planner.group_loads.capacity(),
            ),
            "a successful plan within capacity must not reallocate"
        );
    }
}
