//! Native replicated expert-group (TP×EP) topology and ownership contract.
//!
//! The official native Spark path historically ran `TP=4, EP=1`. Replicated
//! expert groups keep every routed expert in every group, tensor-parallel
//! sharded inside the group, and assign each *route* to exactly one group:
//! `physical_rank = group * TP + tp_rank`, so `group = physical_rank / TP` and
//! `tp_rank = physical_rank % TP`. There are no placeholder ranks: an
//! unsupported `TP×EP` is rejected instead of padded.
//!
//! The wire contract is deliberately additive. The 12-byte route entry is
//! unchanged: bits `0..8` still carry the true expert id `0..383`, bits `9..11`
//! carry the one-hot owning-group bitmap (`bit 9 + group`, exactly one bit set),
//! and bits `12..31` are reserved and rejected. A new request flag selects the
//! contract, so canonical and paired EXL3 consumers reject it and only a worker
//! admitted with a matching topology may decode the owner bits. Route order,
//! FP32 gate weights, the canonical six unique experts per token and the hidden
//! payload are left byte-identical; only the route word's owner field is added.
//!
//! Topology is also bound into response identity: each supported `TP×EP` owns a
//! disjoint executor-id namespace (see [`V41SparkTopology::executor_id`]), so a
//! stale worker from another topology cannot satisfy a receiver that was built
//! for this one, including the same-world-size `TP3EP2` vs `TP2EP3` pair.
//!
//! All physical rank planes stay compact BF16 `[M, 5120]` partials that return
//! to the coordinator; this module adds no Spark-to-Spark reduction. The
//! coordinator sums the physical rank planes in order in FP32 (or per group, as
//! its assembler chooses), adds the shared expert once and rounds once to BF16.
//!
//! The owner assignment is caller-provided (`owners[expert_id]`), for example
//! from `ds41rt_core::replicated_expert_schedule`: active experts hold a group
//! index `0..EP`, untouched experts hold
//! `ds41rt_core::INACTIVE_REPLICATED_EXPERT_GROUP` (`255`). Transport does not
//! depend on the scheduler.

use anyhow::{bail, ensure, Result};

/// Selects the native replicated-group ownership contract. Distinct from the
/// paired EXL3 request flag (`1 << 17`). Only the native compact BF16 request
/// contract may combine with it, and it never appears in a response.
pub const V41_NATIVE_GROUP_REQUEST_FLAG: u32 = 1 << 18;

/// Routed backbone experts in the official checkpoint.
pub const V41_ROUTED_EXPERTS: usize = 384;

/// Largest supported number of replicated groups (EP).
pub const V41_MAX_NATIVE_GROUPS: u8 = 3;

/// Expert id the worker hands to the kernel for a route its group does not own.
/// The existing native kernel maps an out-of-range id to inverse `-1` and
/// contributes zero, so the coordinator sums the same row span on every rank.
pub const V41_NATIVE_UNASSIGNED_EXPERT_ID: i32 = 384;

/// Expert with no routed row in the caller-provided assignment.
pub const V41_NATIVE_INACTIVE_OWNER: u8 = u8::MAX;

const EXPERT_MASK: u32 = 511;
const OWNER_SHIFT: u32 = 9;
const OWNER_MASK: u32 = 0b111;
const WORD_MASK: u32 = 0xfff;

/// Validated native Spark `TP×EP` topology.
///
/// The supported set is exactly `TP2EP1`, `TP3EP1`, the legacy `TP4EP1`,
/// `TP2EP2`, `TP3EP2`, `TP2EP3` and the pure `TP6EP1`; anything else is
/// rejected. `TP×EP` is the physical rank count, with no dummy ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct V41SparkTopology {
    tp: u8,
    ep: u8,
}

impl V41SparkTopology {
    /// Two physical ranks, one group of TP2.
    pub const NATIVE_TP2_EP1: Self = Self { tp: 2, ep: 1 };
    /// Three physical ranks, one group of TP3.
    pub const NATIVE_TP3_EP1: Self = Self { tp: 3, ep: 1 };
    /// Legacy production layout: four physical ranks, one group of TP4.
    pub const NATIVE_TP4_EP1: Self = Self { tp: 4, ep: 1 };
    /// Four physical ranks: two replicated groups of TP2.
    pub const NATIVE_TP2_EP2: Self = Self { tp: 2, ep: 2 };
    /// Six physical ranks: two replicated groups of TP3.
    pub const NATIVE_TP3_EP2: Self = Self { tp: 3, ep: 2 };
    /// Six physical ranks: three replicated groups of TP2.
    pub const NATIVE_TP2_EP3: Self = Self { tp: 2, ep: 3 };
    /// Six physical ranks: one unreplicated group of TP6. Each rank holds one
    /// disjoint intermediate-dimension slice of every routed expert
    /// (2304/6 = 384 per rank, no storage padding), and the coordinator sums the
    /// six compact BF16 planes. This is not an expert-parallel layout: no expert
    /// is duplicated and every rank sees every routed expert's own rows.
    pub const NATIVE_TP6_EP1: Self = Self { tp: 6, ep: 1 };

    /// Validates a `TP×EP` pair. Unsupported pairs fail; no rank is invented.
    pub fn new(tp: u8, ep: u8) -> Result<Self> {
        ensure!(tp >= 1, "native Spark TP degree must be at least one");
        ensure!(ep >= 1, "native Spark group count must be at least one");
        ensure!(ep <= V41_MAX_NATIVE_GROUPS, "native Spark group count exceeds {}", V41_MAX_NATIVE_GROUPS);
        let topology = Self { tp, ep };
        topology.executor_base()?;
        Ok(topology)
    }

    /// Tensor-parallel degree inside one group.
    pub const fn tp(self) -> u8 {
        self.tp
    }

    /// Number of replicated expert groups (`EP`).
    pub const fn ep(self) -> u8 {
        self.ep
    }

    /// Group count (`EP`), the number of valid owner values.
    pub const fn group_count(self) -> u8 {
        self.ep
    }

    /// Physical Spark ranks: `TP * EP`.
    pub const fn world_size(self) -> usize {
        self.tp as usize * self.ep as usize
    }

    /// Group index for a physical rank, `global_rank / TP`.
    pub fn group(self, global_rank: usize) -> Result<u8> {
        ensure!(global_rank < self.world_size(), "native Spark global rank out of range");
        Ok((global_rank / self.tp as usize) as u8)
    }

    /// Tensor-parallel rank inside its group, `global_rank % TP`.
    pub fn tp_rank(self, global_rank: usize) -> Result<u8> {
        ensure!(global_rank < self.world_size(), "native Spark global rank out of range");
        Ok((global_rank % self.tp as usize) as u8)
    }

    /// Group-major physical rank from `(group, tp_rank)`.
    pub fn global_rank(self, group: u8, tp_rank: u8) -> Result<usize> {
        ensure!(group < self.ep, "native Spark group out of range");
        ensure!(tp_rank < self.tp, "native Spark TP rank out of range");
        Ok(group as usize * self.tp as usize + tp_rank as usize)
    }

    /// Canonical executor identity for a physical rank.
    ///
    /// Namespaces are disjoint across topologies and preserve the legacy
    /// identities (`TP4EP1` = 1..=4, `TP2EP1` = 5..=6), so a response from a
    /// worker admitted for another topology is rejected by identity. This binds
    /// topology and rank, not checkpoint or deployment identity.
    pub fn executor_id(self, global_rank: usize) -> Result<u64> {
        ensure!(global_rank < self.world_size(), "native Spark global rank out of range");
        Ok(self.executor_base()? + global_rank as u64)
    }

    /// Canonical executor identities for every physical rank, in rank order.
    pub fn executor_ids(self) -> Vec<u64> {
        (0..self.world_size())
            .map(|rank| self.executor_id(rank).expect("validated topology rank fits its namespace"))
            .collect()
    }

    /// Physical rank named by an executor identity, or `None` for another
    /// topology's (or an unknown) identity.
    pub fn rank_of_executor(self, executor_id: u64) -> Option<usize> {
        let base = self.executor_base().ok()?;
        let offset = executor_id.checked_sub(base)?;
        if offset < self.world_size() as u64 {
            Some(offset as usize)
        } else {
            None
        }
    }

    fn executor_base(self) -> Result<u64> {
        // Aligns the two same-world six-rank replicated layouts with the
        // architecture audit's proposal (TP3EP2 = 11..=16, TP2EP3 = 21..=26),
        // keeps the pure six-rank TP6EP1 in its own 27..=32 namespace next to
        // them, and keeps every other namespace disjoint, including the legacy
        // 1..=4 and 5..=6.
        match (self.tp, self.ep) {
            (4, 1) => Ok(1),
            (2, 1) => Ok(5),
            (3, 1) => Ok(7),
            (3, 2) => Ok(11),
            (2, 2) => Ok(17),
            (2, 3) => Ok(21),
            (6, 1) => Ok(27),
            (tp, ep) => bail!(
                "unsupported native Spark TP×EP topology TP{tp}EP{ep}: \
                 supported are TP2EP1, TP3EP1, TP4EP1, TP2EP2, TP3EP2, TP2EP3 and TP6EP1"
            ),
        }
    }
}

/// Owner-tagged native route word carried in the existing 12-byte route entry.
///
/// The wire keeps the paired.rs style one-bit-per-group owner field, widened to
/// the three groups `0..=2`: bits `0..8` are the true expert, bits `9..11` are a
/// one-hot owner bitmap with **exactly one** bit set (bit `9 + group`), and bits
/// `12..31` are reserved and rejected. The API exposes the owning group index
/// (`owner: 0..EP`); the bitmap is the wire form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41NativeOwnerRouteWord {
    /// True routed expert, `0..384`.
    pub expert_id: u32,
    /// Replicated group that owns every route to this expert, `0..EP`.
    pub owner: u8,
}

impl V41NativeOwnerRouteWord {
    /// Encodes the word with a one-hot owner bit. The topology's actual `EP`
    /// bound is enforced by [`Self::decode`] and by the request builder.
    pub fn encode(self) -> Result<u32> {
        ensure!((self.expert_id as usize) < V41_ROUTED_EXPERTS, "native owner expert id out of range");
        ensure!(self.owner < V41_MAX_NATIVE_GROUPS, "native owner field must fit three bits");
        Ok(self.expert_id | (1u32 << (OWNER_SHIFT + u32::from(self.owner))))
    }

    /// Decodes and validates against the worker's `group_count` (`EP`).
    ///
    /// The owner field must contain exactly one group bit, and that group must be
    /// below `group_count`; a multi-bit or empty owner is rejected so no route
    /// can be counted by more than one group.
    pub fn decode(word: u32, group_count: u8) -> Result<Self> {
        ensure!(
            group_count >= 1 && group_count <= V41_MAX_NATIVE_GROUPS,
            "native group count {group_count} is outside 1..={V41_MAX_NATIVE_GROUPS}"
        );
        ensure!(word & !WORD_MASK == 0, "native owner route has reserved bits set");
        let expert_id = word & EXPERT_MASK;
        ensure!((expert_id as usize) < V41_ROUTED_EXPERTS, "native owner expert id out of range");
        let owners = (word >> OWNER_SHIFT) & OWNER_MASK;
        ensure!(
            owners.count_ones() == 1,
            "native owner word must name exactly one replicated group"
        );
        let owner = owners.trailing_zeros() as u8;
        ensure!(owner < group_count, "native owner {owner} is not below group count {group_count}");
        Ok(Self { expert_id, owner })
    }
}

/// Batch-local ownership validation; no allocation on success.
///
/// Every occurrence of an expert must carry the same owner, and every active
/// expert must be owned by exactly one group. Observing the same word twice is
/// idempotent.
#[derive(Clone)]
pub struct V41NativeOwnershipBatch {
    owners: [u8; V41_ROUTED_EXPERTS],
}

impl Default for V41NativeOwnershipBatch {
    fn default() -> Self {
        Self {
            owners: [V41_NATIVE_INACTIVE_OWNER; V41_ROUTED_EXPERTS],
        }
    }
}

impl V41NativeOwnershipBatch {
    /// Clears all observations without allocating.
    pub fn clear(&mut self) {
        self.owners.fill(V41_NATIVE_INACTIVE_OWNER);
    }

    /// Decodes one route word and requires it to agree with earlier occurrences.
    pub fn observe(&mut self, word: u32, group_count: u8) -> Result<V41NativeOwnerRouteWord> {
        let route = V41NativeOwnerRouteWord::decode(word, group_count)?;
        let previous = &mut self.owners[route.expert_id as usize];
        ensure!(
            *previous == V41_NATIVE_INACTIVE_OWNER || *previous == route.owner,
            "native expert has conflicting group ownership within a batch"
        );
        *previous = route.owner;
        Ok(route)
    }

    /// Owner observed for an expert, or `None` if it was never routed.
    pub fn owner_of(&self, expert_id: usize) -> Option<u8> {
        self.owners
            .get(expert_id)
            .copied()
            .filter(|owner| *owner != V41_NATIVE_INACTIVE_OWNER)
    }
}

impl crate::ExpertProtocolV2Request {
    /// Re-encodes every route with its caller-provided group owner and sets
    /// [`V41_NATIVE_GROUP_REQUEST_FLAG`].
    ///
    /// `owners[expert_id]` is the owning group `0..topology.group_count()`, or
    /// [`V41_NATIVE_INACTIVE_OWNER`] for an expert the caller never routed;
    /// a routed expert with an inactive or out-of-range owner is rejected. The
    /// route order, the exact FP32 gate weights, the canonical six unique
    /// experts per token and the hidden payload are unchanged; only the route
    /// word gains its owner bits.
    pub fn with_native_group_owners(
        &mut self,
        owners: &[u8],
        topology: V41SparkTopology,
    ) -> Result<()> {
        ensure!(
            owners.len() == V41_ROUTED_EXPERTS,
            "native group ownership needs one entry per routed expert"
        );
        ensure!(
            !self.stream_data_enabled() && !self.stream_plan_enabled() && !self.precompile_warmup_enabled(),
            "native group ownership is only available for regular native batches"
        );
        ensure!(
            self.header.flags & super::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0,
            "native group ownership requires the native compact BF16 request contract"
        );
        ensure!(
            self.header.flags & super::V41_EXL3_PAIRED_REQUEST_FLAG == 0,
            "native group ownership cannot combine with paired EXL3 admission"
        );
        ensure!(
            self.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG == 0,
            "request already carries native group ownership"
        );
        // Fully validate the canonical shape *before* touching any route word, so
        // a malformed batch is returned byte-identical and can be corrected and
        // retried. 4096 is the largest native transport capacity.
        super::V41BackboneRequest::validate_owned(self, 4096)?;
        for route in &self.routes {
            let expert = route.expert_id;
            ensure!(
                (expert as usize) < V41_ROUTED_EXPERTS,
                "native group route expert id out of range"
            );
            ensure!(
                owners[expert as usize] < topology.group_count(),
                "routed native expert {expert} has no valid group owner"
            );
        }
        // Every fallible check has passed; the re-encode below cannot fail for a
        // validated canonical route set (`expert < 384`, `owner < EP <= 3`).
        for route in &mut self.routes {
            let expert = route.expert_id;
            route.expert_id = V41NativeOwnerRouteWord {
                expert_id: expert,
                owner: owners[expert as usize],
            }
            .encode()?;
        }
        self.header.flags |= V41_NATIVE_GROUP_REQUEST_FLAG;
        self.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_expert::{
        tests::request, V41BackboneRequest, V41Tp4ChunkReceiver, V41Tp4Planes, V41Tp4Roce,
        V41Tp4Tcp, V41_EXL3_PAIRED_REQUEST_FLAG, V41_PARTIAL_ROW_BYTES,
    };
    use crate::{ExpertProtocolV2Request, ExpertProtocolV2Response, TcpTransportConfig};
    use std::collections::BTreeSet;
    use std::net::SocketAddr;

    const ALL: [V41SparkTopology; 7] = [
        V41SparkTopology::NATIVE_TP2_EP1,
        V41SparkTopology::NATIVE_TP3_EP1,
        V41SparkTopology::NATIVE_TP4_EP1,
        V41SparkTopology::NATIVE_TP2_EP2,
        V41SparkTopology::NATIVE_TP3_EP2,
        V41SparkTopology::NATIVE_TP2_EP3,
        V41SparkTopology::NATIVE_TP6_EP1,
    ];

    fn owners_for(topology: V41SparkTopology) -> Vec<u8> {
        (0..V41_ROUTED_EXPERTS)
            .map(|expert| (expert % topology.group_count() as usize) as u8)
            .collect()
    }

    fn native_request(
        rows: u32,
        topology: V41SparkTopology,
        owners: &[u8],
    ) -> Result<ExpertProtocolV2Request> {
        let mut request = request(rows);
        request.with_native_group_owners(owners, topology)?;
        Ok(request)
    }

    fn frame_config() -> TcpTransportConfig {
        TcpTransportConfig {
            timing: false,
            timeout: std::time::Duration::from_secs(1),
            max_frame_bytes: 200_000,
        }
    }

    #[test]
    fn topology_mapping_is_group_major_and_namespaces_are_disjoint() {
        let expected: [(V41SparkTopology, usize, [u64; 6]); 7] = [
            (V41SparkTopology::NATIVE_TP2_EP1, 2, [5, 6, 0, 0, 0, 0]),
            (V41SparkTopology::NATIVE_TP3_EP1, 3, [7, 8, 9, 0, 0, 0]),
            (V41SparkTopology::NATIVE_TP4_EP1, 4, [1, 2, 3, 4, 0, 0]),
            (V41SparkTopology::NATIVE_TP2_EP2, 4, [17, 18, 19, 20, 0, 0]),
            (V41SparkTopology::NATIVE_TP3_EP2, 6, [11, 12, 13, 14, 15, 16]),
            (V41SparkTopology::NATIVE_TP2_EP3, 6, [21, 22, 23, 24, 25, 26]),
            (V41SparkTopology::NATIVE_TP6_EP1, 6, [27, 28, 29, 30, 31, 32]),
        ];
        let mut seen = BTreeSet::new();
        for (topology, world, ids) in expected {
            assert_eq!(topology.world_size(), world);
            assert_eq!(topology.group_count(), topology.ep());
            assert_eq!(topology.executor_ids(), &ids[..world]);
            for rank in 0..world {
                assert_eq!(
                    topology.group(rank).unwrap() as usize,
                    rank / topology.tp() as usize
                );
                assert_eq!(
                    topology.tp_rank(rank).unwrap() as usize,
                    rank % topology.tp() as usize
                );
                assert_eq!(
                    topology
                        .global_rank(
                            topology.group(rank).unwrap(),
                            topology.tp_rank(rank).unwrap()
                        )
                        .unwrap(),
                    rank
                );
                assert_eq!(topology.executor_id(rank).unwrap(), ids[rank]);
                assert_eq!(topology.rank_of_executor(ids[rank]), Some(rank));
                assert!(seen.insert(ids[rank]));
            }
            assert!(topology.group(world).is_err());
            assert!(topology.tp_rank(world).is_err());
            assert!(topology.executor_id(world).is_err());
            assert!(topology.global_rank(topology.ep(), 0).is_err());
            assert!(topology.global_rank(0, topology.tp()).is_err());
            assert_eq!(topology.rank_of_executor(0), None);
            assert_eq!(topology.rank_of_executor(99), None);
        }
        // Same-size six-rank topologies never accept each other's workers.
        assert_eq!(V41SparkTopology::NATIVE_TP3_EP2.rank_of_executor(20), None);
        assert_eq!(V41SparkTopology::NATIVE_TP2_EP3.rank_of_executor(14), None);
        // Pure TP6EP1 shares the six-rank world size but not the identities of
        // either replicated six-rank layout, in both directions.
        assert_eq!(V41SparkTopology::NATIVE_TP6_EP1.rank_of_executor(21), None);
        assert_eq!(V41SparkTopology::NATIVE_TP6_EP1.rank_of_executor(11), None);
        assert_eq!(V41SparkTopology::NATIVE_TP2_EP3.rank_of_executor(27), None);
        assert_eq!(V41SparkTopology::NATIVE_TP3_EP2.rank_of_executor(27), None);
        // A four-rank TP2×EP2 receiver never accepts legacy TP4 identities.
        assert_eq!(V41SparkTopology::NATIVE_TP2_EP2.rank_of_executor(1), None);
    }

    #[test]
    fn unsupported_topologies_are_rejected_without_dummy_ranks() {
        for (tp, ep) in [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 2),
            (4, 3),
            (3, 3),
            (2, 4),
            (5, 1),
            (6, 2),
            (6, 3),
            (255, 1),
            (2, 255),
        ] {
            assert!(
                V41SparkTopology::new(tp, ep).is_err(),
                "TP{tp}EP{ep} must be rejected"
            );
        }
        for (tp, ep) in [(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3), (6, 1)] {
            assert!(V41SparkTopology::new(tp, ep).is_ok(), "TP{tp}EP{ep}");
        }
    }

    #[test]
    fn pure_tp6_is_one_unreplicated_group_over_six_ranks() {
        let topology = V41SparkTopology::new(6, 1).unwrap();
        assert_eq!(topology, V41SparkTopology::NATIVE_TP6_EP1);
        assert_eq!(topology.tp(), 6);
        assert_eq!(topology.ep(), 1);
        assert_eq!(topology.world_size(), 6);
        // Every rank is its own TP shard of the single group, so ownership has
        // exactly one legal value and cannot be inactive.
        assert_eq!(topology.group_count(), 1);
        for rank in 0..6 {
            assert_eq!(topology.group(rank).unwrap(), 0);
            assert_eq!(topology.tp_rank(rank).unwrap() as usize, rank);
            assert_eq!(topology.executor_id(rank).unwrap(), 27 + rank as u64);
        }
        // Owners are all group 0; routing weight and route order are untouched.
        let owners = owners_for(topology);
        assert!(owners.iter().all(|owner| *owner == 0));
        let request = native_request(4, topology, &owners).unwrap();
        assert_ne!(request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
        assert_eq!(request.routes.len(), 4 * 6);
        for route in &request.routes {
            let decoded = V41NativeOwnerRouteWord::decode(route.expert_id, 1).unwrap();
            assert_eq!(decoded.owner, 0);
            assert_eq!(decoded.expert_id as usize, route.expert_id as usize & 0x1ff);
        }
        // The owned batch is validated by the native-group contract itself; the
        // canonical/paired consumers (and the float-plane reducer that stops at
        // four ranks) must keep rejecting it.
        assert!(V41BackboneRequest::validate_owned_native_group(&request, 4096, topology).is_ok());
        assert!(V41BackboneRequest::validate_owned(&request, 4096).is_err());
        let frame = request.encode().unwrap();
        assert!(V41BackboneRequest::parse(&frame, 4096).is_err());
    }

    #[test]
    fn owner_words_round_trip_and_reject_reserved_or_out_of_range_bits() {
        for expert_id in 0..V41_ROUTED_EXPERTS as u32 {
            for owner in 0..V41_MAX_NATIVE_GROUPS {
                let word = V41NativeOwnerRouteWord { expert_id, owner }
                    .encode()
                    .unwrap();
                assert_eq!(word & !WORD_MASK, 0);
                assert_eq!(word & EXPERT_MASK, expert_id);
                // One-hot owner bitmap: bit 9 + owner is the only owner bit set.
                assert_eq!(word >> OWNER_SHIFT, 1u32 << owner);
                for group_count in 1..=V41_MAX_NATIVE_GROUPS {
                    let decoded = V41NativeOwnerRouteWord::decode(word, group_count);
                    if owner < group_count {
                        assert_eq!(
                            decoded.unwrap(),
                            V41NativeOwnerRouteWord { expert_id, owner }
                        );
                    } else {
                        assert!(decoded.is_err());
                    }
                }
            }
        }
        // Bits 12..31 are reserved.
        for bit in 12..32 {
            assert!(V41NativeOwnerRouteWord::decode(1 << bit, 3).is_err());
        }
        // Empty and multi-bit owner fields never name exactly one group.
        assert!(V41NativeOwnerRouteWord::decode(0, 3).is_err());
        assert!(V41NativeOwnerRouteWord::decode(0b011 << OWNER_SHIFT, 3).is_err());
        assert!(V41NativeOwnerRouteWord::decode(0b101 << OWNER_SHIFT, 3).is_err());
        assert!(V41NativeOwnerRouteWord::decode(0b111 << OWNER_SHIFT, 3).is_err());
        for expert_id in V41_ROUTED_EXPERTS as u32..512 {
            assert!(
                V41NativeOwnerRouteWord::decode(expert_id | (1 << OWNER_SHIFT), 3).is_err()
            );
        }
        assert!(V41NativeOwnerRouteWord {
            expert_id: V41_ROUTED_EXPERTS as u32,
            owner: 0
        }
        .encode()
        .is_err());
        assert!(V41NativeOwnerRouteWord {
            expert_id: 0,
            owner: V41_MAX_NATIVE_GROUPS
        }
        .encode()
        .is_err());
        for group_count in [0u8, 4, 255] {
            assert!(V41NativeOwnerRouteWord::decode(0, group_count).is_err());
        }
    }

    #[test]
    fn every_supported_topology_encodes_and_unwraps_exactly_once() -> Result<()> {
        for topology in ALL {
            let canonical = request(3);
            let owners = owners_for(topology);
            let canonical_frame = canonical.encode()?;
            let request = native_request(3, topology, &owners)?;
            assert_ne!(request.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
            assert_eq!(request.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG, 0);
            assert_eq!(request.rows, canonical.rows);
            assert_eq!(request.hidden_payload, canonical.hidden_payload);

            let frame = request.encode()?;
            // Only the owner bits change: the frame keeps its exact size.
            assert_eq!(frame.len(), canonical_frame.len());
            let parsed = V41BackboneRequest::parse_native_group(&frame, 3, topology)?;
            assert!(parsed.is_native_group());
            assert_eq!(parsed.native_topology(), Some(topology));
            assert_eq!(parsed.hidden(), canonical.hidden_payload.as_ref());

            let count = request.routes.len();
            let mut assigned = vec![0u8; count];
            for group in 0..topology.group_count() {
                let mut ids = vec![-1i32; count];
                let mut weights = vec![-1.0f32; count];
                parsed.copy_native_group_routes_into(&mut ids, &mut weights, group)?;
                for (index, route) in canonical.routes.iter().enumerate() {
                    if owners[route.expert_id as usize] == group {
                        assert_eq!(ids[index], route.expert_id as i32);
                        assert_eq!(weights[index].to_bits(), route.gate_weight.to_bits());
                        assigned[index] += 1;
                    } else {
                        assert_eq!(ids[index], V41_NATIVE_UNASSIGNED_EXPERT_ID);
                        assert_eq!(weights[index], 0.0);
                    }
                }
                // Reusing the same buffers for a second group is scratch-safe.
                let mut again = vec![0i32; count];
                parsed.copy_native_group_routes_into(&mut again, &mut weights, group)?;
                assert_eq!(again, ids);
                assert!(parsed
                    .copy_native_group_routes_into(&mut ids, &mut weights, topology.group_count())
                    .is_err());
            }
            // Each route is owned by exactly one group; every active expert too.
            assert!(assigned.iter().all(|count| *count == 1));
            let mut batch = V41NativeOwnershipBatch::default();
            for route in &request.routes {
                batch.observe(route.expert_id, topology.group_count())?;
            }
            assert_eq!(
                batch.owner_of(0),
                Some(owners[0]),
                "active expert ownership is available after validation"
            );

            // Canonical and paired consumers reject the new flag.
            assert!(V41BackboneRequest::parse(&frame, 3).is_err());
            assert!(V41BackboneRequest::parse_paired(&frame, 3).is_err());
            assert!(V41BackboneRequest::validate_owned(&request, 3).is_err());
            assert!(V41BackboneRequest::validate_owned_paired(&request, 3).is_err());
            // The unowned canonical frame cannot be parsed as native group.
            assert!(V41BackboneRequest::parse_native_group(&canonical_frame, 3, topology).is_err());
        }
        Ok(())
    }

    #[test]
    fn inactive_groups_and_inactive_experts_are_zero_weight_sentinels() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP2_EP2;
        let canonical = request(2);
        // Concentrate every routed expert in group 0 so group 1 is inactive.
        let mut owners = vec![V41_NATIVE_INACTIVE_OWNER; V41_ROUTED_EXPERTS];
        for route in &canonical.routes {
            owners[route.expert_id as usize] = 0;
        }
        let request = native_request(2, topology, &owners)?;
        let frame = request.encode()?;
        let parsed = V41BackboneRequest::parse_native_group(&frame, 2, topology)?;
        let count = request.routes.len();
        let mut idle_ids = vec![0i32; count];
        let mut idle_weights = vec![1.0f32; count];
        parsed.copy_native_group_routes_into(&mut idle_ids, &mut idle_weights, 1)?;
        assert!(idle_ids.iter().all(|id| *id == V41_NATIVE_UNASSIGNED_EXPERT_ID));
        assert!(idle_weights.iter().all(|weight| *weight == 0.0));
        Ok(())
    }

    #[test]
    fn request_builder_rejects_missing_owner_conflicts_and_oversized_owners() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP2_EP2;
        let canonical = request(2);

        let mut short = canonical.clone();
        assert!(short
            .with_native_group_owners(&[0u8; V41_ROUTED_EXPERTS - 1], topology)
            .is_err());

        let mut inactive = canonical.clone();
        let mut owners = vec![0u8; V41_ROUTED_EXPERTS];
        owners[0] = V41_NATIVE_INACTIVE_OWNER;
        assert!(inactive.with_native_group_owners(&owners, topology).is_err());

        let mut oversized = canonical.clone();
        let mut owners = vec![0u8; V41_ROUTED_EXPERTS];
        owners[0] = topology.group_count();
        assert!(oversized.with_native_group_owners(&owners, topology).is_err());

        let mut paired = canonical.clone();
        paired.header.flags |= V41_EXL3_PAIRED_REQUEST_FLAG;
        assert!(paired
            .with_native_group_owners(&vec![0u8; V41_ROUTED_EXPERTS], topology)
            .is_err());

        let mut encoded = native_request(2, topology, &vec![0u8; V41_ROUTED_EXPERTS])?;
        assert!(encoded
            .with_native_group_owners(&vec![0u8; V41_ROUTED_EXPERTS], topology)
            .is_err());

        // A canonical (non-compact) request cannot adopt the contract.
        let mut compactless = canonical.clone();
        compactless.header.flags &= !crate::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        assert!(compactless
            .with_native_group_owners(&vec![0u8; V41_ROUTED_EXPERTS], topology)
            .is_err());

        // A partial mutation must be rejected by the parser, never accepted.
        let mut conflict = native_request(3, topology, &vec![0u8; V41_ROUTED_EXPERTS])?;
        let first = conflict.routes[0].expert_id;
        conflict.routes[6].expert_id = (first & EXPERT_MASK) | (1 << (OWNER_SHIFT + 1));
        assert!(V41BackboneRequest::parse_native_group(&conflict.encode()?, 3, topology).is_err());
        Ok(())
    }

    #[test]
    fn native_parser_rejects_owner_overflow_reserved_bits_duplicates_and_bad_weights() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP2_EP1; // EP=1: only owner 0 is legal.
        let owners = vec![0u8; V41_ROUTED_EXPERTS];
        let valid = native_request(2, topology, &owners)?;

        let mut overflow = valid.clone();
        overflow.routes[0].expert_id =
            (overflow.routes[0].expert_id & EXPERT_MASK) | (1 << (OWNER_SHIFT + 1));
        assert!(V41BackboneRequest::parse_native_group(&overflow.encode()?, 2, topology).is_err());

        let mut reserved = valid.clone();
        reserved.routes[0].expert_id |= 1 << 12;
        assert!(V41BackboneRequest::parse_native_group(&reserved.encode()?, 2, topology).is_err());

        let mut duplicate = valid.clone();
        duplicate.routes[1].expert_id =
            (duplicate.routes[0].expert_id & EXPERT_MASK) | (1 << OWNER_SHIFT);
        assert!(V41BackboneRequest::parse_native_group(&duplicate.encode()?, 2, topology).is_err());

        let mut negative = valid.clone();
        negative.routes[0].gate_weight = -0.5;
        assert!(V41BackboneRequest::parse_native_group(&negative.encode()?, 2, topology).is_err());
        Ok(())
    }

    /// The request flag and canonical shape are identical for same-EP, different-TP
    /// topologies; only the topology-bound executor namespace distinguishes them.
    #[test]
    fn same_group_count_requests_are_separated_by_the_executor_namespace() -> Result<()> {
        let tp2 = V41SparkTopology::NATIVE_TP2_EP1;
        let tp3 = V41SparkTopology::NATIVE_TP3_EP1;
        let owners = owners_for(tp2);
        let request = native_request(2, tp2, &owners)?;
        let frame = request.encode()?;
        // The owner-only wire contract cannot tell the two same-EP topologies apart.
        assert!(V41BackboneRequest::parse_native_group(&frame, 2, tp3).is_ok());
        // The receiver, bound to TP3's namespace, rejects the TP2 workers' ids.
        let native = V41BackboneRequest::parse_native_group(&frame, 2, tp3)?;
        let mut receiver =
            V41Tp4ChunkReceiver::new_ranks(&native, &tp3.executor_ids(), 200_000)?;
        let payload = vec![0; native.plane_bytes()?];
        for stale in tp2.executor_ids() {
            let response = native.response(stale, &payload)?.to_owned()?.encode()?;
            assert!(receiver
                .push(&response, |_, _, _| panic!("stale same-EP worker"))
                .is_err());
        }
        Ok(())
    }

    #[test]
    fn six_rank_receiver_covers_every_plane_and_rejects_stale_or_foreign_responses() -> Result<()> {
        for (topology, other) in [
            (V41SparkTopology::NATIVE_TP3_EP2, V41SparkTopology::NATIVE_TP2_EP3),
            (V41SparkTopology::NATIVE_TP2_EP3, V41SparkTopology::NATIVE_TP3_EP2),
            (V41SparkTopology::NATIVE_TP6_EP1, V41SparkTopology::NATIVE_TP2_EP3),
            (V41SparkTopology::NATIVE_TP2_EP3, V41SparkTopology::NATIVE_TP6_EP1),
        ] {
            let owners = owners_for(topology);
            let request = native_request(2, topology, &owners)?;
            let frame = request.encode()?;
            let native = V41BackboneRequest::parse_native_group(&frame, 2, topology)?;
            let executors = topology.executor_ids();
            let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, 200_000)?;
            assert_eq!(receiver.received_rows(), [0; 4]);
            assert_eq!(receiver.received_rows_slice(), [0u32; 6].as_slice());
            // A world of the wrong size cannot be admitted for this topology.
            assert!(
                V41Tp4ChunkReceiver::new_ranks(&native, &executors[..5], 200_000).is_err()
            );

            let payload = vec![0x11; native.plane_bytes()?];
            let mut stale = native.response(executors[0], &payload)?.to_owned()?;
            stale.header.request_id += 1;
            assert!(receiver
                .push(&stale.encode()?, |_, _, _| panic!("stale response"))
                .is_err());
            let foreign = native
                .response(other.executor_ids()[0], &payload)?
                .to_owned()?;
            assert!(receiver
                .push(&foreign.encode()?, |_, _, _| panic!("foreign response"))
                .is_err());
            assert!(receiver
                .push(&frame[..frame.len() - 1], |_, _, _| panic!("truncated response"))
                .is_err());

            for rank in (0..6).rev() {
                let response = native.response(executors[rank], &payload)?.to_owned()?.encode()?;
                assert_eq!(
                    receiver
                        .push(&response, |actual, start, bytes| {
                            assert_eq!((actual, start), (rank, 0));
                            assert_eq!(bytes, payload);
                            Ok(())
                        })
                        .unwrap(),
                    rank
                );
                assert!(receiver
                    .push(&response, |_, _, _| panic!("duplicate response"))
                    .is_err());
                assert_eq!(receiver.complete(), rank == 0);
            }
            assert_eq!(receiver.received_rows_slice(), [2u32; 6].as_slice());
        }
        Ok(())
    }

    #[test]
    fn six_rank_chunked_coverage_rejects_reordered_overlapping_and_bad_final_markers() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP2_EP3;
        let owners = owners_for(topology);
        let request = native_request(3, topology, &owners)?;
        let frame = request.encode()?;
        let native = V41BackboneRequest::parse_native_group(&frame, 3, topology)?;
        let executors = topology.executor_ids();
        let budget = 200_000;
        let mut receiver = V41Tp4ChunkReceiver::new_ranks(&native, &executors, budget)?;

        let chunk = |rank: usize, start: u32, rows: u32| -> Result<Vec<u8>> {
            let payload = vec![rank as u8 + 1; rows as usize * V41_PARTIAL_ROW_BYTES as usize];
            let mut indices = vec![0u32; rows as usize];
            native
                .response_chunk(executors[rank], start, &payload, &mut indices, budget)?
                .to_owned()?
                .encode()
        };

        // Row overlap is rejected without committing coverage.
        let second = chunk(0, 1, 2)?;
        assert!(receiver
            .push(&second, |_, _, _| panic!("reordered rows"))
            .is_err());
        // An early final marker is rejected.
        let mut early = ExpertProtocolV2Response::decode(&chunk(0, 0, 2)?)?;
        early.header.flags &= !crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        assert!(receiver
            .push(&early.encode()?, |_, _, _| panic!("early final marker"))
            .is_err());
        // A missing final marker is rejected.
        let mut late = ExpertProtocolV2Response::decode(&chunk(0, 2, 1)?)?;
        late.header.flags |= crate::EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS;
        assert!(receiver
            .push(&late.encode()?, |_, _, _| panic!("missing final marker"))
            .is_err());

        for rank in [5usize, 0, 3, 1, 4, 2] {
            for (start, rows) in [(0u32, 2u32), (2, 1)] {
                let frame = chunk(rank, start, rows)?;
                let expected_payload =
                    vec![rank as u8 + 1; rows as usize * V41_PARTIAL_ROW_BYTES as usize];
                receiver.push(&frame, |actual, first, bytes| {
                    assert_eq!((actual, first), (rank, start));
                    assert_eq!(bytes, expected_payload);
                    Ok(())
                })?;
            }
        }
        assert!(receiver.complete());
        assert_eq!(receiver.received_rows_slice(), [3u32; 6].as_slice());
        assert_eq!(receiver.received_rows(), [3; 4]);
        Ok(())
    }

    #[test]
    fn planes_collection_supports_three_and_six_ranks_but_legacy_accessor_stays_four() -> Result<()> {
        for topology in [V41SparkTopology::NATIVE_TP3_EP1, V41SparkTopology::NATIVE_TP3_EP2] {
            let owners = owners_for(topology);
            let request = native_request(1, topology, &owners)?;
            let frame = request.encode()?;
            let native = V41BackboneRequest::parse_native_group(&frame, 1, topology)?;
            let executors = topology.executor_ids();
            let world = topology.world_size();
            let payloads: Vec<Vec<u8>> = (0..world)
                .map(|rank| vec![rank as u8 + 1; native.plane_bytes().unwrap()])
                .collect();
            let responses: Vec<Vec<u8>> = (0..world)
                .map(|rank| {
                    native
                        .response(executors[rank], &payloads[rank])
                        .unwrap()
                        .to_owned()
                        .unwrap()
                        .encode()
                        .unwrap()
                })
                .collect();
            let mut planes = V41Tp4Planes::new_ranks(&native, &executors)?;
            assert!(!planes.complete());
            assert_eq!(planes.world_size(), world);
            assert!(planes.planes().is_err());
            for rank in (0..world).rev() {
                assert_eq!(planes.insert(&responses[rank])?, rank);
                assert!(planes.insert(&responses[rank]).is_err());
            }
            assert!(planes.complete());
            for rank in 0..world {
                assert_eq!(planes.plane(rank)?, payloads[rank].as_slice());
            }
            assert!(planes.plane(world).is_err());
        }
        Ok(())
    }

    #[test]
    fn owned_topology_receiver_requires_the_flag_and_the_matching_world() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP3_EP2;
        let owners = owners_for(topology);
        let owned = native_request(2, topology, &owners)?;
        let executors = topology.executor_ids();
        let mut receiver =
            V41Tp4ChunkReceiver::from_owned_topology(&owned, 2, &executors, topology, 200_000)?;
        assert!(!receiver.complete());
        assert!(V41Tp4ChunkReceiver::from_owned_topology(
            &owned,
            2,
            &executors[..4],
            topology,
            200_000
        )
        .is_err());
        // The legacy owned constructor cannot decode ownership without a topology.
        assert!(
            V41Tp4ChunkReceiver::from_owned_ranks(&owned, 2, &executors, 200_000).is_err()
        );
        // A canonical owned request is not a native-group request.
        let canonical = request(2);
        assert!(V41Tp4ChunkReceiver::from_owned_topology(
            &canonical,
            2,
            &executors,
            topology,
            200_000
        )
        .is_err());
        let owned_frame = owned.encode()?;
        let native = V41BackboneRequest::parse_native_group(&owned_frame, 2, topology)?;
        let payload = vec![0; native.plane_bytes()?];
        for (rank, id) in executors.iter().enumerate() {
            let chunk = native.response(*id, &payload)?.to_owned()?.encode()?;
            receiver.push(&chunk, |actual, _, _| {
                assert_eq!(actual, rank);
                Ok(())
            })?;
        }
        assert!(receiver.complete());
        Ok(())
    }

    #[test]
    fn generic_roce_constructors_validate_world_topology_and_identity() -> Result<()> {
        let peers: Vec<SocketAddr> = (0..6)
            .map(|index| format!("127.0.0.1:{}", 24_000 + index).parse())
            .collect::<std::result::Result<_, _>>()?;
        let config = frame_config();
        for topology in ALL {
            let world = topology.world_size();
            let mut client =
                V41Tp4Roce::new_topology(topology, &peers[..world], 80, config.clone())?;
            assert_eq!(client.world_size(), world);
            assert_eq!(client.topology(), Some(topology));
            assert_eq!(client.capacity(), 80);
            client.reset_connections();
        }
        let three = V41Tp4Roce::new_ranks(&peers[..3], &[7, 8, 9], 80, config.clone())?;
        assert_eq!(three.world_size(), 3);
        assert_eq!(three.topology(), None);

        // Unsupported world sizes, mismatched topology peers, duplicates.
        assert!(V41Tp4Roce::new_ranks(&peers[..5], &[1, 2, 3, 4, 5], 80, config.clone()).is_err());
        assert!(V41Tp4Roce::new_ranks(&peers[..2], &[1, 2, 3], 80, config.clone()).is_err());
        assert!(V41Tp4Roce::new_topology(
            V41SparkTopology::NATIVE_TP3_EP2,
            &peers[..4],
            80,
            config.clone()
        )
        .is_err());
        assert!(V41Tp4Roce::new_topology(
            V41SparkTopology::NATIVE_TP2_EP2,
            &peers[..3],
            80,
            config.clone()
        )
        .is_err());
        assert!(V41Tp4Roce::new_ranks(&[peers[0]; 3], &[7, 8, 9], 80, config.clone()).is_err());
        assert!(V41Tp4Roce::new_ranks(&peers[..3], &[7, 7, 9], 80, config.clone()).is_err());
        assert!(V41Tp4Roce::new_ranks(&peers[..3], &[7, 0, 9], 80, config.clone()).is_err());
        assert!(V41Tp4Roce::new_ranks(&peers[..6], &[7, 8, 9, 10, 11, 12], 0, config).is_err());
        Ok(())
    }

    #[test]
    fn generic_protocol_flag_validator_bounds_the_new_flag() -> Result<()> {
        use crate::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let compact = EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        // Native group flag alone, with the paired flag, or with foreign flags.
        for flags in [
            V41_NATIVE_GROUP_REQUEST_FLAG,
            V41_NATIVE_GROUP_REQUEST_FLAG | V41_EXL3_PAIRED_REQUEST_FLAG | compact,
            V41_NATIVE_GROUP_REQUEST_FLAG | compact | (1 << 9),
            V41_NATIVE_GROUP_REQUEST_FLAG | compact | (1 << 20),
            V41_NATIVE_GROUP_REQUEST_FLAG | crate::EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
        ] {
            let mut request = request(1);
            request.header.flags = flags;
            assert!(request.encode().is_err(), "flags 0x{flags:08x} must be rejected");
        }
        // The exact pair compact+native (plus optional debug checksum) is legal.
        let mut valid = request(1);
        valid.with_native_group_owners(
            &owners_for(V41SparkTopology::NATIVE_TP2_EP1),
            V41SparkTopology::NATIVE_TP2_EP1,
        )?;
        assert_ne!(valid.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
        let valid_frame = valid.encode()?;

        // A response still may not carry the request-only flag.
        let native = V41BackboneRequest::parse_native_group(
            &valid_frame,
            1,
            V41SparkTopology::NATIVE_TP2_EP1,
        )?;
        let mut response = native
            .response(
                V41SparkTopology::NATIVE_TP2_EP1.executor_id(0)?,
                &vec![0; native.plane_bytes()?],
            )?
            .to_owned()?;
        assert_eq!(response.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
        response.header.flags |= V41_NATIVE_GROUP_REQUEST_FLAG;
        assert!(response.encode().is_err());
        Ok(())
    }

    #[test]
    fn failed_owner_encoding_leaves_the_request_byte_identical() -> Result<()> {
        let topology = V41SparkTopology::NATIVE_TP2_EP2;
        let owners = vec![0u8; V41_ROUTED_EXPERTS];

        let mut cases: Vec<(ExpertProtocolV2Request, Vec<u8>)> = Vec::new();
        // Malformed canonical shapes that `ExpertProtocolV2Request::new` accepts.
        let mut duplicate = request(2);
        duplicate.routes[1].expert_id = duplicate.routes[0].expert_id;
        cases.push((duplicate, owners.clone()));
        let mut span = request(2);
        span.rows[0].route_offset = 1;
        cases.push((span, owners.clone()));
        let mut weight = request(2);
        weight.routes[0].gate_weight = -0.5;
        cases.push((weight, owners.clone()));
        let mut expert = request(2);
        expert.routes[0].expert_id = V41_ROUTED_EXPERTS as u32;
        cases.push((expert, owners.clone()));
        // Owner-assignment failures.
        let mut inactive = owners.clone();
        inactive[0] = V41_NATIVE_INACTIVE_OWNER;
        cases.push((request(2), inactive));
        let mut oversized = owners.clone();
        oversized[0] = topology.group_count();
        cases.push((request(2), oversized));

        for (mut bad, bad_owners) in cases {
            let before = bad.encode()?;
            let before_flags = bad.header.flags;
            let before_routes = bad.routes.clone();
            assert!(bad.with_native_group_owners(&bad_owners, topology).is_err());
            assert_eq!(
                bad.header.flags, before_flags,
                "a rejected assignment must not set the native group flag"
            );
            assert_eq!(
                bad.routes, before_routes,
                "a rejected assignment must not rewrite route words"
            );
            assert_eq!(
                bad.encode()?,
                before,
                "a rejected assignment must leave the frame byte-identical"
            );
        }

        // A corrected assignment can be retried on the same request object.
        let mut retry = request(2);
        let mut bad_owners = vec![0u8; V41_ROUTED_EXPERTS];
        bad_owners[0] = topology.group_count();
        assert!(retry.with_native_group_owners(&bad_owners, topology).is_err());
        retry.with_native_group_owners(&owners, topology)?;
        assert_ne!(retry.header.flags & V41_NATIVE_GROUP_REQUEST_FLAG, 0);
        assert!(V41BackboneRequest::parse_native_group(&retry.encode()?, 2, topology).is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn topology_bound_transports_fail_closed_on_flag_mismatch() -> Result<()> {
        let peers: Vec<SocketAddr> = (0..4)
            .map(|index| format!("127.0.0.1:{}", 26_000 + index).parse())
            .collect::<std::result::Result<_, _>>()?;
        let config = frame_config();
        let topology = V41SparkTopology::NATIVE_TP4_EP1;
        let flagged = native_request(2, topology, &owners_for(topology))?;
        let canonical = request(2);

        // Legacy (topology-free) transports reject ownership before any I/O.
        let mut legacy_roce = V41Tp4Roce::new_ranks(&peers, &[1, 2, 3, 4], 80, config.clone())?;
        assert!(legacy_roce.dispatch(&flagged).await.is_err());
        let mut legacy_tcp = V41Tp4Tcp::new_ranks(&peers, &[1, 2, 3, 4], 80, config.clone())?;
        assert!(legacy_tcp.dispatch(&flagged).await.is_err());

        // Topology-bound transports require the flag; no silent canonical fallback.
        let mut bound_roce = V41Tp4Roce::new_topology(topology, &peers, 80, config.clone())?;
        assert!(bound_roce.dispatch(&canonical).await.is_err());
        let mut bound_tcp = V41Tp4Tcp::new_topology(topology, &peers, 80, config.clone())?;
        assert!(bound_tcp.dispatch(&canonical).await.is_err());
        Ok(())
    }
}
