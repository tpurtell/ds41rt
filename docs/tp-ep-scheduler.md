# Replicated expert-group scheduler (TP/EP)

`ds41rt-core::replicated_expert_schedule` is a deterministic, CPU-only planner
that assigns each routed expert of one layer/request to one of `1`, `2` or `3`
replicated expert groups. It exists to decouple expert-to-group scheduling from
transport, topology and protocol decisions: the planner produces an expert to
group assignment and predicted per-group cost/load, and integration code copies
or encodes that assignment into a request-owned buffer.

This module is **new, additive and not yet wired into serving**. Its evidence is
unit-test evidence only; no throughput, GPU, latency or numerical claim is made
here.

## Scope

- Official backbone shape: 384 routed experts per layer, top-6 routing. The
  planner accepts any histogram length; 384 is the production entry count.
- The deployment may hold `group_count = 1`, `2` or `3` identical expert groups.
  The internal tensor-parallel width of a group (TP2/TP3) is irrelevant to pure
  scheduling and is deliberately not modelled.
- Input is a per-expert routed-row histogram: `route_counts[expert_id]` is the
  number of routed rows that selected that expert for this layer/request.
- Output is a borrowed assignment (`ReplicatedExpertGroupId` per expert id) plus
  per-group predicted cost/load, so the caller copies the assignment into a
  request-owned buffer before the next plan reuses the scheduler.

### Non-goals

- No wire format, message layout, transport or serialization is defined here.
- No topology, host, device or TP-rank binding is defined here.
- No hot-expert splitting. Whole-expert assignment cannot split one dominant
  expert across groups (see [Whole-expert policy](#whole-expert-policy)).
- No measured or calibrated production coefficients are shipped. The cost model
  is an explicit caller input.

## API

All types are re-exported from the crate root (`ds41rt_core::...`).

```rust
pub const MAX_REPLICATED_EXPERT_GROUPS: u8 = 3;
pub const INACTIVE_REPLICATED_EXPERT_GROUP: u8 = u8::MAX;

pub struct ReplicatedExpertCostModel {
    pub expert_weight_cost: u64,
    pub tile_cost: u64,
    pub tile_rows: u32,
}

impl ReplicatedExpertCostModel {
    pub const fn new(expert_weight_cost: u64, tile_cost: u64, tile_rows: u32) -> Self;
    pub fn validate(&self) -> Result<(), Ds41rtError>;
    pub fn expert_cost(&self, routed_rows: u64) -> Result<u64, Ds41rtError>;
}

pub struct ReplicatedExpertScheduleConfig {
    pub group_count: u8,
    pub cost_model: ReplicatedExpertCostModel,
}

impl ReplicatedExpertScheduleConfig {
    pub const fn new(group_count: u8, cost_model: ReplicatedExpertCostModel) -> Self;
    pub fn validate(&self) -> Result<(), Ds41rtError>;
}

pub struct ReplicatedExpertGroupId(u8);

impl ReplicatedExpertGroupId {
    pub const INACTIVE: Self;                       // encoded u8::MAX
    pub const fn active(index: u8) -> Option<Self>; // None unless index < 3
    pub const fn is_active(self) -> bool;
    pub const fn index(self) -> Option<u8>;
    pub const fn encoded(self) -> u8;
}

pub struct ReplicatedExpertGroupLoad {
    pub cost: u64,        // sum of estimated expert costs assigned here
    pub expert_count: u32,
    pub routed_rows: u64,
}

pub struct ReplicatedExpertGroupPlan<'a> { /* borrowed */ }

impl<'a> ReplicatedExpertGroupPlan<'a> {
    pub fn assignment(&self) -> &'a [ReplicatedExpertGroupId];
    pub fn expert_cost(&self) -> &'a [u64];
    pub fn group_loads(&self) -> &'a [ReplicatedExpertGroupLoad];
    pub fn total_cost(&self) -> u64;
    pub fn group_count(&self) -> u8;
    pub fn tie_seed(&self) -> u64;
    pub fn group_for_expert(&self, expert_id: usize) -> Option<ReplicatedExpertGroupId>;
}

pub struct ReplicatedExpertScheduler { /* preallocated scratch */ }

impl ReplicatedExpertScheduler {
    pub fn new(config: ReplicatedExpertScheduleConfig, expert_capacity: usize)
        -> Result<Self, Ds41rtError>;
    pub fn config(&self) -> ReplicatedExpertScheduleConfig;
    pub fn expert_capacity(&self) -> usize;
    pub fn reserve_expert_capacity(&mut self, capacity: usize) -> Result<(), Ds41rtError>;
    pub fn plan(
        &mut self,
        route_counts: &[u32],
        tie_seed: u64,
    ) -> Result<ReplicatedExpertGroupPlan<'_>, Ds41rtError>;
}

pub const fn replicated_expert_tie_seed(layer_id: u32, request_id: u64) -> u64;
```

### Integration recipe

```rust
use ds41rt_core::{
    replicated_expert_tie_seed, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, ReplicatedExpertGroupId,
};

// Once per lane / stage, outside the request path:
let config = ReplicatedExpertScheduleConfig::new(
    2,
    ReplicatedExpertCostModel::new(/* expert_weight_cost */ w, /* tile_cost */ t, /* tile_rows */ k),
);
let mut scheduler = ReplicatedExpertScheduler::new(config, 384)?; // 384 production experts

// Per layer/request, after routing produced `route_counts: [u32; 384]`:
let seed = replicated_expert_tie_seed(layer_id, request_id);
let group_for_request = {
    let plan = scheduler.plan(&route_counts, seed)?;
    // Copy/encode into a request-owned buffer; do not retain the borrow.
    let mut encoded = [ReplicatedExpertGroupId::INACTIVE.encoded(); 384];
    for (slot, group) in encoded.iter_mut().zip(plan.assignment()) {
        *slot = group.encoded();
    }
    // ``plan`` is dropped here; the scheduler is free for the next request.
    encoded
};
```

- `assignment()[expert_id]` is expert-id-indexed and always has
  `route_counts.len()` entries.
- Active experts (count > 0) hold an active group id; every other entry holds
  `ReplicatedExpertGroupId::INACTIVE` (`encoded() == u8::MAX`).
- The borrow checker guarantees the plan cannot outlive the scheduler or coexist
  with another plan, so integration must copy before replanning.
- Use one scheduler per lane. Two lanes cannot affect each other's results.

## Cost model and units

`expert_cost` returns abstract integer **cost units** (CU), not microseconds,
bytes, FLOPs or any measured quantity:

```text
expert_cost(routed_rows) =
    0                                                    if routed_rows == 0
    expert_weight_cost + ceil(routed_rows / tile_rows) * tile_cost   otherwise
```

- The weight term is charged **once per active expert**, never once per routed
  row. Whole-expert assignment executes each expert's weights once for all of
  its routed rows, and the tests pin this behaviour.
- The activation term models padded routed-row tiles on a group.
- The production-shaped default is exactly this pure-weight form: a uniform
  per-expert weight cost plus `ceil(count / tile_rows) * tile_cost`. The
  official geometry has one expert size, so a uniform weight cost is retained.

**There are no calibrated defaults.** Whether specific values are feasible for a
deployment must be established by measurement; that measurement is out of scope
here. The model is exposed precisely so that callers do not inherit assumed
coefficients. Cost units are consistent within one plan; only ratios between
plans with the same model are comparable.

Model validation rejects `tile_rows == 0` and rejects a model where
`expert_weight_cost == 0 && tile_cost == 0`, which would give every active expert
zero cost and make the scheduling policy meaningless. A histogram with no active
experts legitimately plans to total cost `0`.

## Algorithm

1. Mark every expert with `route_counts[expert_id] > 0` active.
2. Estimate each active expert's cost with the configured model.
3. Sort active experts by **descending** estimated cost. Equal costs are ordered
   by a deterministic rank derived from the plan's tie seed, then by expert id.
4. Greedily assign each expert, in that order, to the group with the smallest
   current predicted cost; equal group loads are broken by a seeded rank, then
   by group index (classic LPT greedy).

`group_loads()[g].cost` is the predicted load of group `g`; the maximum over
groups is the predicted makespan driver. `total_cost` is the sum of all active
expert costs; `sum(group_loads.cost) == total_cost` always.

## Tie seed locality

`tie_seed` is a per-plan argument, not planner state, so a lane can replan for
each layer/request without rebuilding the scheduler. Derive it from purely local
identity with `replicated_expert_tie_seed(layer_id, request_id)`. The module
reads no process-global, GPU, transport or cross-lane state, so tie breaks from
one lane cannot perturb another, and the same `(config, route_counts, tie_seed,
expert_capacity)` always yields the same assignment.

## Whole-expert policy

The scheduler assigns **whole experts**. An expert's weights are executed once,
in exactly one group, for all of its routed rows, which preserves weight reuse.
A consequence is that a single dominant expert cannot be split: if routed rows
concentrate on one expert, its full estimated cost lands in one group and the
predicted makespan cannot fall below that cost even when other groups are idle.

Hot-expert splitting is intentionally **not** implemented. It would change the
execution contract and must be a separate, opt-in feature justified by
measurements; it is not a scheduling-only change.

## Determinism, allocation and errors

- **Deterministic**: output is a pure function of the inputs above.
- **Allocation-free on success**: `new` preallocates `expert_capacity` entries.
  A successful plan whose histogram length is within capacity does not allocate;
  the regression test plans 384 experts 512 times and asserts all five scratch
  buffers keep their capacity.
- **Explicit sizing**: `plan` rejects a histogram longer than the planned
  capacity instead of allocating mid-plan. Callers grow with
  `reserve_expert_capacity` outside the request path.
- **Input preservation**: the histogram is a shared slice and is never mutated.
- **Rejections**, all reported as `Ds41rtError::ExpertRoutePlanRejected`:
  `group_count == 0`, `group_count > 3`, `tile_rows == 0`, a model with both cost
  terms zero, histogram length above planned capacity, and per-expert, group-cost
  or total-cost integer overflow. `ReplicatedExpertGroupId::active` rejects an
  index `>= 3`.
- **Recoverability**: after any rejected plan the scheduler is unpoisoned; a
  later valid plan on the same scheduler behaves as if the failed call had not
  happened. The tests exercise overflow, then recovery on the same instance.

## Test evidence

Focused command (isolated target directory so concurrent agent builds are not
disturbed):

```bash
CARGO_TARGET_DIR=runs/tp-ep-scheduler/target \
  cargo test --manifest-path rust/Cargo.toml -p ds41rt-core replicated_expert_schedule
```

Result at the time of writing: **15 passed, 0 failed**. Unit tests only; no GPU,
serving or performance claim.

| Test | Evidence |
| --- | --- |
| `six_single_row_experts_balance_across_replicated_groups` | Six unique M1 (one routed row) experts across two groups yields 3/3 experts and equal cost; across three groups yields 2/2/2. |
| `weight_cost_is_charged_once_per_active_expert_not_per_row` | Repeated rows charge the weight term once, not per row. |
| `tile_costs_use_ceiling_boundaries` | `ceil(count / tile_rows)` boundaries for counts 0..9. |
| `empty_and_all_zero_histograms_are_inactive` | Empty and all-zero histograms produce no active expert and zero load. |
| `single_group_receives_every_active_expert` | `group_count = 1` assigns every active expert to group 0. |
| `seeded_tie_breaks_are_deterministic_and_seed_sensitive` | Same seed reproduces exactly; different seeds produce different equal-cost assignments. |
| `tie_seed_is_a_pure_function_of_local_identity` | Seed depends only on layer/request inputs. |
| `expert_permutation_preserves_group_load_multiset` | With distinct costs, permuting expert ids preserves the sorted per-group load distribution and total. |
| `independent_schedulers_keep_independent_scratch` | Interleaved planners keep independent results. |
| `plan_borrows_outputs_and_preserves_inputs` | Output slices alias planner buffers; the input histogram is unchanged. |
| `invalid_configurations_and_overflow_are_rejected_then_recovered` | Invalid groups/model, cost overflow, total overflow and capacity overflow reject, and later valid plans recover. |
| `whole_expert_policy_keeps_dominant_expert_unsplit` | A dominant expert is assigned once and its full cost stays in one group. |
| `exhaustive_small_histograms_cover_each_active_expert_once` | Enumerates all histograms with up to four experts and counts 0..3 across 1/2/3 groups: exact coverage once, exact group sums, and the loose 4/3 LPT bound against a brute-force optimum. Never asserts greedy optimality. |
| `greedy_lpt_matches_or_beats_test_only_round_robin_baseline` | Greedy LPT is compared against a test-only round-robin baseline on a skewed fixture. |
| `production_shape_repeated_plans_are_allocation_stable` | 384-expert shape repeatedly planned with all scratch capacities asserted stable. |

The round-robin route-count baseline exists only as a test helper
(`round_robin_baseline_max_load`); it is not part of the production API or
complexity.

## Follow-ups (not in this change)

- Calibrate `expert_weight_cost`, `tile_cost` and `tile_rows` per deployment and
  record the measurement.
- Wire the encoded assignment into the request-owned expert dispatch buffer and
  bind it to groups at the topology layer.
- Revisit hot-expert splitting as a separate opt-in only if benchmarks justify
  it.
