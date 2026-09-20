//! Coordinator-level integration proof for the frozen native replicated-group
//! path, using only the public transport API and the public core scheduler.
//!
//! Pipeline exercised per physical rank:
//! canonical top-6 histogram -> real `ReplicatedExpertScheduler` LPT assignment
//! -> `with_native_group_owners` -> wire `encode()` -> worker
//! `parse_native_group` -> `copy_native_group_routes_into` host mask.
//!
//! The exact-once property is proven at the TP-shard level: a route assigned to
//! one group is delivered to exactly `TP` physical ranks (every rank of its
//! owning group), not once globally. Non-owning ranks receive sentinel id 384
//! with zero weight.
//!
//! The integer weighted bookkeeping below is **dispatch instrumentation only**:
//! it counts, per expert, which ranks received a route and with which weight.
//! It is NOT a coordinator numerical reference. Each TP rank computes a distinct
//! intermediate-dimension partial of the same route, so the real FFN output is
//! the FP32 sum of those different partials, not `weight * TP`. No GPU or FFN
//! arithmetic is simulated; the proof is integer-exact over exactly
//! representable weights and asserts delivery multiplicity only.
//!
//! Chunked response coverage is checked for the N=4 and N=6 layouts so the
//! transport plane count matches the topology.

use std::collections::BTreeSet;

use anyhow::{ensure, Result};
use ds41rt_core::{
    replicated_expert_tie_seed, ReplicatedExpertCostModel, ReplicatedExpertScheduleConfig,
    ReplicatedExpertScheduler, INACTIVE_REPLICATED_EXPERT_GROUP,
};
use ds41rt_transport::v41_expert::{
    V41BackboneRequest, V41NativeOwnerRouteWord, V41SparkTopology, V41Tp4ChunkReceiver,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_UNASSIGNED_EXPERT_ID, V41_PARTIAL_ROW_BYTES,
    V41_ROUTED_EXPERTS, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16,
};
use ds41rt_transport::{
    ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2Dtype, ExpertV2SourceKind,
};

const EXPERTS: usize = V41_ROUTED_EXPERTS;
const MAX_ROWS: u32 = 4096;
const MAX_FRAME: usize = 200_000;
const HIDDEN_BYTES: usize = 5120 * 2;

/// Every approved layout: World 2, 3, 4 and 6.
const TOPOLOGIES: [V41SparkTopology; 6] = [
    V41SparkTopology::NATIVE_TP2_EP1,
    V41SparkTopology::NATIVE_TP3_EP1,
    V41SparkTopology::NATIVE_TP4_EP1,
    V41SparkTopology::NATIVE_TP2_EP2,
    V41SparkTopology::NATIVE_TP3_EP2,
    V41SparkTopology::NATIVE_TP2_EP3,
];
const ROWS: [usize; 4] = [1, 2, 8, 16];

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Fixture {
    request: ExpertProtocolV2Request,
    experts: Vec<u32>,
    weights: Vec<f32>,
}

/// Deterministic canonical top-6 fixture. Row 0 pins boundary experts 0 and 383;
/// later rows are a seeded squared-uniform skewed draw with six distinct experts.
fn fixture(rows: usize, seed: u64) -> Result<Fixture> {
    let mut rng = Rng(seed);
    let descriptors: Vec<ExpertProtocolV2RowDescriptor> = (0..rows)
        .map(|row| ExpertProtocolV2RowDescriptor {
            row_id: row as u64,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 1000 + row as u64,
            token_position: row as u64,
            route_offset: (row * 6) as u32,
            route_count: 6,
        })
        .collect();
    let mut experts = Vec::with_capacity(rows * 6);
    let mut weights = Vec::with_capacity(rows * 6);
    for row in 0..rows {
        let mut chosen = [0u32; 6];
        if row == 0 {
            chosen = [0, 383, 1, 2, 3, 4];
        } else {
            for slot in 0..6 {
                loop {
                    let u = rng.unit();
                    let expert = ((u * u * EXPERTS as f64) as u32).min(EXPERTS as u32 - 1);
                    if !chosen[..slot].contains(&expert) {
                        chosen[slot] = expert;
                        break;
                    }
                }
            }
        }
        for &expert in &chosen {
            experts.push(expert);
            // Small integer weights keep the dispatch accounting integer-exact;
            // they are labels for delivery counting, not FFN arithmetic.
            weights.push(((rng.next_u64() % 5) + 1) as f32);
        }
    }
    let routes: Vec<ExpertProtocolV2RouteEntry> = (0..rows * 6)
        .map(|index| ExpertProtocolV2RouteEntry {
            row_index: (index / 6) as u32,
            expert_id: experts[index],
            gate_weight: weights[index],
        })
        .collect();
    let mut request = ExpertProtocolV2Request::new(
        7,
        17,
        39,
        5120,
        ExpertV2Dtype::Bf16,
        descriptors,
        routes,
        vec![0u8; rows * HIDDEN_BYTES],
    )?;
    request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(Fixture {
        request,
        experts,
        weights,
    })
}

fn route_counts(fixture: &Fixture) -> [u32; EXPERTS] {
    let mut counts = [0u32; EXPERTS];
    for &expert in &fixture.experts {
        counts[expert as usize] += 1;
    }
    counts
}

/// Real core scheduler, runtime-default-shaped model `(1, 0, 16)`, and a
/// determinism re-check.
fn lpt_owners(counts: &[u32; EXPERTS], ep: u8) -> Result<Vec<u8>> {
    let model = ReplicatedExpertCostModel::new(1, 0, 16);
    let config = ReplicatedExpertScheduleConfig::new(ep, model);
    let mut scheduler = ReplicatedExpertScheduler::new(config, EXPERTS)?;
    let seed = replicated_expert_tie_seed(39, 7);
    let owners: Vec<u8> = {
        let plan = scheduler.plan(counts, seed)?;
        plan.assignment().iter().map(|group| group.encoded()).collect()
    };
    let again: Vec<u8> = {
        let plan = scheduler.plan(counts, seed)?;
        plan.assignment().iter().map(|group| group.encoded()).collect()
    };
    ensure!(owners == again, "LPT assignment must be deterministic");
    Ok(owners)
}

/// Full wire + ownership + exactly-once proof for one topology/fixture/assignment.
fn assert_wire_roundtrip(
    topology: V41SparkTopology,
    fixture: &Fixture,
    owners: &[u8],
) -> Result<()> {
    ensure!(owners.len() == EXPERTS);
    let mut flagged = fixture.request.clone();
    flagged.with_native_group_owners(owners, topology)?;
    ensure!(flagged.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0);
    let frame = flagged.encode()?;
    // A canonical consumer must reject the ownership frame.
    ensure!(V41BackboneRequest::parse(&frame, MAX_ROWS).is_err());

    let ranks = topology.world_size();
    ensure!(ranks == topology.tp() as usize * topology.ep() as usize);
    let executors = topology.executor_ids();
    ensure!(executors.len() == ranks);
    ensure!(executors.iter().all(|id| *id != 0));
    let mut per_group = vec![0usize; topology.ep() as usize];
    for rank in 0..ranks {
        per_group[topology.group(rank)? as usize] += 1;
    }
    ensure!(
        per_group.iter().all(|count| *count == topology.tp() as usize),
        "every replicated group holds exactly TP physical ranks"
    );

    let route_count = fixture.experts.len();
    let mut owned_per_route = vec![0usize; route_count];
    let mut dispatch_weight = vec![0i64; EXPERTS];
    for rank in 0..ranks {
        let native = V41BackboneRequest::parse_native_group(&frame, MAX_ROWS, topology)?;
        ensure!(native.is_native_group());
        ensure!(native.native_topology() == Some(topology));
        ensure!(native.rows() as usize == route_count / 6);
        let group = topology.group(rank)?;
        let mut ids = vec![-1i32; route_count];
        let mut weights = vec![-1.0f32; route_count];
        native.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
        for index in 0..route_count {
            let expert = fixture.experts[index];
            let owner = owners[expert as usize];
            ensure!(owner < topology.group_count(), "active expert owner out of range");
            if owner == group {
                ensure!(ids[index] == expert as i32, "owned route must keep its true expert id");
                ensure!(
                    weights[index].to_bits() == fixture.weights[index].to_bits(),
                    "owned route must keep its exact FP32 weight"
                );
                owned_per_route[index] += 1;
                dispatch_weight[expert as usize] += fixture.weights[index] as i64;
            } else {
                ensure!(
                    ids[index] == V41_NATIVE_UNASSIGNED_EXPERT_ID,
                    "non-owning rank must receive the sentinel expert id"
                );
                ensure!(weights[index] == 0.0, "non-owning rank must receive zero weight");
            }
        }
    }

    // Exactly once per TP shard: each route reaches exactly TP physical ranks,
    // all inside its single owning group; no other group owns it.
    for index in 0..route_count {
        let owner = owners[fixture.experts[index] as usize];
        let expected = if owner == INACTIVE_REPLICATED_EXPERT_GROUP {
            0
        } else {
            topology.tp() as usize
        };
        ensure!(
            owned_per_route[index] == expected,
            "route {index} delivered to {} ranks, expected {expected}",
            owned_per_route[index]
        );
    }

    // Dispatch instrumentation only, not a numerical reference: account for the
    // per-rank route deliveries per expert and check the delivery multiplicity is
    // TP for every route. Each TP rank emits a different intermediate partial, so
    // this must not be read as the coordinator's FFN sum.
    let mut expected_dispatch_weight = vec![0i64; EXPERTS];
    for index in 0..route_count {
        let expert = fixture.experts[index] as usize;
        if owners[expert] != INACTIVE_REPLICATED_EXPERT_GROUP {
            expected_dispatch_weight[expert] +=
                fixture.weights[index] as i64 * topology.tp() as i64;
        }
    }
    ensure!(
        dispatch_weight == expected_dispatch_weight,
        "per-expert dispatch accounting must match TP delivery multiplicity"
    );
    Ok(())
}

#[test]
fn wire_roundtrip_is_exactly_once_per_tp_shard_for_approved_topologies() -> Result<()> {
    for topology in TOPOLOGIES {
        for &rows in &ROWS {
            let fixture = fixture(rows, 0x5EED_0000 + rows as u64)?;
            for row in 0..rows {
                let slice = &fixture.experts[row * 6..row * 6 + 6];
                let unique: BTreeSet<u32> = slice.iter().copied().collect();
                ensure!(unique.len() == 6, "top-6 must be six distinct experts");
                ensure!(slice.iter().all(|expert| (*expert as usize) < EXPERTS));
            }
            ensure!(
                fixture.experts.contains(&0) && fixture.experts.contains(&383),
                "fixture must exercise boundary experts 0 and 383"
            );
            let counts = route_counts(&fixture);
            let owners = lpt_owners(&counts, topology.group_count())?;
            for (expert, &count) in counts.iter().enumerate() {
                if count > 0 {
                    ensure!(owners[expert] < topology.group_count());
                } else {
                    ensure!(owners[expert] == INACTIVE_REPLICATED_EXPERT_GROUP);
                }
            }
            assert_wire_roundtrip(topology, &fixture, &owners)?;
        }
    }
    Ok(())
}

#[test]
fn ep1_owner_zero_flag_and_single_owner_bit() -> Result<()> {
    for topology in [
        V41SparkTopology::NATIVE_TP2_EP1,
        V41SparkTopology::NATIVE_TP3_EP1,
        V41SparkTopology::NATIVE_TP4_EP1,
    ] {
        ensure!(topology.group_count() == 1);
        for rows in [1usize, 8] {
            let fixture = fixture(rows, 0xE91_0000 + rows as u64)?;
            let counts = route_counts(&fixture);
            let owners = lpt_owners(&counts, 1)?;
            for (expert, &count) in counts.iter().enumerate() {
                if count > 0 {
                    ensure!(owners[expert] == 0, "EP1 owner must be group 0");
                }
            }
            let mut flagged = fixture.request.clone();
            flagged.with_native_group_owners(&owners, topology)?;
            let frame = flagged.encode()?;
            let decoded = ExpertProtocolV2Request::decode(&frame)?;
            ensure!(decoded.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0);
            for route in &decoded.routes {
                let word = V41NativeOwnerRouteWord::decode(route.expert_id, 1)?;
                ensure!(word.owner == 0);
                ensure!(
                    (route.expert_id >> 9) & 0b111 == 1,
                    "EP1 must set only the owner bit for group 0"
                );
            }
            assert_wire_roundtrip(topology, &fixture, &owners)?;
        }
    }
    Ok(())
}

#[test]
fn inactive_group_ranks_are_sentinel_only_zero_rows() -> Result<()> {
    for topology in [
        V41SparkTopology::NATIVE_TP2_EP2,
        V41SparkTopology::NATIVE_TP3_EP2,
        V41SparkTopology::NATIVE_TP2_EP3,
    ] {
        let fixture = fixture(8, 0x1AC7_0000)?;
        let counts = route_counts(&fixture);
        // Concentrate every active expert in group 0 so groups 1.. are idle.
        let mut owners = vec![INACTIVE_REPLICATED_EXPERT_GROUP; EXPERTS];
        for (expert, &count) in counts.iter().enumerate() {
            if count > 0 {
                owners[expert] = 0;
            }
        }
        let mut flagged = fixture.request.clone();
        flagged.with_native_group_owners(&owners, topology)?;
        let frame = flagged.encode()?;
        for rank in 0..topology.world_size() {
            let group = topology.group(rank)?;
            let native = V41BackboneRequest::parse_native_group(&frame, MAX_ROWS, topology)?;
            let mut ids = vec![0i32; fixture.experts.len()];
            let mut weights = vec![1.0f32; fixture.experts.len()];
            native.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
            if group == 0 {
                ensure!(ids.iter().all(|id| *id != V41_NATIVE_UNASSIGNED_EXPERT_ID));
                ensure!(weights.iter().all(|weight| *weight >= 0.0));
            } else {
                ensure!(
                    ids.iter().all(|id| *id == V41_NATIVE_UNASSIGNED_EXPERT_ID),
                    "idle group rank must be sentinel-only"
                );
                ensure!(
                    weights.iter().all(|weight| *weight == 0.0),
                    "idle group rank must carry zero weight rows"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn chunked_response_coverage_matches_n_physical_planes() -> Result<()> {
    for topology in [
        V41SparkTopology::NATIVE_TP4_EP1,
        V41SparkTopology::NATIVE_TP2_EP2,
        V41SparkTopology::NATIVE_TP3_EP2,
    ] {
        let rows = 8usize;
        let fixture = fixture(rows, 0xC0FF_EE00)?;
        let counts = route_counts(&fixture);
        let owners = lpt_owners(&counts, topology.group_count())?;
        let mut flagged = fixture.request.clone();
        flagged.with_native_group_owners(&owners, topology)?;
        let frame = flagged.encode()?;
        let native = V41BackboneRequest::parse_native_group(&frame, MAX_ROWS, topology)?;
        let executors = topology.executor_ids();
        let ranks = topology.world_size();
        ensure!(ranks == executors.len());

        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, MAX_FRAME)?;
        ensure!(receiver.world_size() == ranks);
        ensure!(!receiver.complete());
        // Feed every rank in reverse, one row per chunk.
        for rank in (0..ranks).rev() {
            let executor_id = executors[rank];
            let mut indices = [0u32; 1];
            for row in 0..rows as u32 {
                let payload = vec![rank as u8 + 1; V41_PARTIAL_ROW_BYTES as usize];
                let chunk = native
                    .response_chunk(executor_id, row, &payload, &mut indices, MAX_FRAME)?
                    .to_owned()?
                    .encode()?;
                receiver.push(&chunk, |actual_rank, start, bytes| {
                    ensure!(actual_rank == rank && start == row);
                    ensure!(bytes.iter().all(|byte| *byte == rank as u8 + 1));
                    Ok(())
                })?;
            }
            ensure!(
                receiver.complete() == (rank == 0),
                "coverage must complete only after the last physical rank"
            );
        }
        ensure!(receiver.complete());
        ensure!(receiver.received_rows_slice() == &[rows as u32; 6][..ranks]);
    }
    Ok(())
}
