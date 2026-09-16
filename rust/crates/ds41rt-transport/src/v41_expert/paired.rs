//! Paired EXL3 route-word contract, requiring explicit paired native admission.
//!
//! Keep the existing 12-byte route entry: bits 0..8 identify the expert,
//! bits 9..10 select the two boundary owners. Every other bit is reserved.
//! The request flag and loaded paired layout must agree before decoding these
//! words. In particular, an all-zero ownership pattern still needs the flag.
use anyhow::{ensure, Result};

/// Requires explicit paired admission; ordinary native workers reject this flag.
pub const V41_EXL3_PAIRED_REQUEST_FLAG: u32 = 1 << 17;
const EXPERT_MASK: u32 = 511;
const OWNERS_SHIFT: u32 = 9;
const WORD_MASK: u32 = 2047;
const EXPERTS: usize = 384;
const INACTIVE: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41PairedRouteWord {
    pub expert_id: u32,
    /// Bit 0 selects rank 1 over 0; bit 1 selects rank 3 over 2.
    pub owners: u8,
}

impl V41PairedRouteWord {
    pub fn encode(self) -> Result<u32> {
        ensure!(
            self.expert_id < EXPERTS as u32,
            "paired expert ID out of range"
        );
        ensure!(
            self.owners < 4,
            "paired ownership must contain exactly two bits"
        );
        Ok(self.expert_id | (u32::from(self.owners) << OWNERS_SHIFT))
    }

    pub fn decode(word: u32) -> Result<Self> {
        ensure!(word & !WORD_MASK == 0, "paired route has reserved bits set");
        let expert_id = word & EXPERT_MASK;
        ensure!(expert_id < EXPERTS as u32, "paired expert ID out of range");
        Ok(Self {
            expert_id,
            owners: (word >> OWNERS_SHIFT) as u8,
        })
    }
}

/// Batch-local validation and unpacking; no allocation on success. One owner
/// assignment applies to every occurrence of an expert across the whole batch.
pub struct V41PairedOwnershipBatch {
    owners: [u8; EXPERTS],
}

impl Default for V41PairedOwnershipBatch {
    fn default() -> Self {
        Self {
            owners: [INACTIVE; EXPERTS],
        }
    }
}

impl V41PairedOwnershipBatch {
    pub fn clear(&mut self) {
        self.owners.fill(INACTIVE);
    }

    pub fn observe(&mut self, word: u32) -> Result<V41PairedRouteWord> {
        let route = V41PairedRouteWord::decode(word)?;
        let previous = &mut self.owners[route.expert_id as usize];
        ensure!(
            *previous == INACTIVE || *previous == route.owners,
            "paired expert has conflicting ownership within a batch"
        );
        *previous = route.owners;
        Ok(route)
    }

    /// Expand to the kernel's stable int32 descriptor row. Extra padded expert
    /// slots and inactive experts are zero; route validation must exclude them.
    pub fn write_local_ownership(&self, rank: usize, output: &mut [i32]) -> Result<()> {
        ensure!(rank < 4, "paired Spark rank out of range");
        ensure!(output.len() >= EXPERTS, "paired ownership output too small");
        output.fill(0);
        for (slot, &owners) in output.iter_mut().zip(&self.owners) {
            if owners != INACTIVE {
                *slot = i32::from(usize::from((owners >> (rank / 2)) & 1) == rank % 2);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_legal_word_round_trips_and_reserved_bits_are_rejected() {
        for expert_id in 0..384 {
            for owners in 0..4 {
                let route = V41PairedRouteWord { expert_id, owners };
                let word = route.encode().unwrap();
                assert_eq!(V41PairedRouteWord::decode(word).unwrap(), route);
                for bit in 11..32 {
                    assert!(V41PairedRouteWord::decode(word | (1 << bit)).is_err());
                }
            }
        }
        for id in 384..512 {
            assert!(V41PairedRouteWord::decode(id).is_err());
        }
        assert!(V41PairedRouteWord {
            expert_id: 384,
            owners: 0
        }
        .encode()
        .is_err());
        assert!(V41PairedRouteWord {
            expert_id: 0,
            owners: 4
        }
        .encode()
        .is_err());
    }

    #[test]
    fn repeated_experts_keep_one_owner_and_reuse_clears_padded_metadata() {
        let mut batch = V41PairedOwnershipBatch::default();
        for owners in 0..4 {
            let route = V41PairedRouteWord {
                expert_id: owners as u32,
                owners,
            };
            for _ in 0..20 {
                assert_eq!(batch.observe(route.encode().unwrap()).unwrap(), route);
            }
        }
        assert!(batch
            .observe(
                V41PairedRouteWord {
                    expert_id: 0,
                    owners: 3
                }
                .encode()
                .unwrap()
            )
            .is_err());
        let mut total = [[0; 4]; 2];
        let mut output = [99; 768];
        for rank in 0..4 {
            batch.write_local_ownership(rank, &mut output).unwrap();
            for expert in 0..4 {
                total[rank / 2][expert] += output[expert];
            }
            assert!(output[4..].iter().all(|&value| value == 0));
        }
        assert_eq!(total, [[1; 4]; 2]);
        assert!(batch.write_local_ownership(4, &mut output).is_err());
        assert!(batch.write_local_ownership(0, &mut output[..383]).is_err());
        batch.clear();
        batch.write_local_ownership(0, &mut output).unwrap();
        assert_eq!(output, [0; 768]);
        assert!(batch
            .observe(
                V41PairedRouteWord {
                    expert_id: 0,
                    owners: 3
                }
                .encode()
                .unwrap()
            )
            .is_ok());
    }

    #[test]
    fn paired_frames_preserve_size_routes_and_responses_and_reject_conflicts() {
        use crate::v41_expert::V41BackboneRequest;
        let original = crate::v41_expert::tests::request(3);
        let ordinary = original.encode().unwrap();
        assert!(V41BackboneRequest::parse_paired(&ordinary, 3).is_err());
        for pattern in 0..4 {
            let mut request = original.clone();
            request.header.flags |= V41_EXL3_PAIRED_REQUEST_FLAG;
            for route in &mut request.routes {
                route.expert_id = V41PairedRouteWord {
                    expert_id: route.expert_id,
                    owners: ((route.expert_id + pattern) % 4) as u8,
                }
                .encode()
                .unwrap();
            }
            let frame = request.encode().unwrap();
            assert_eq!(frame.len(), ordinary.len());
            assert!(V41BackboneRequest::parse(&frame, 3).is_err());
            V41BackboneRequest::validate_owned_paired(&request, 3).unwrap();
            let native = V41BackboneRequest::parse_paired(&frame, 3).unwrap();
            let mut ids = [0; 18];
            let mut weights = [0.; 18];
            assert!(native.copy_routes_into(&mut ids, &mut weights).is_err());
            for rank in 0..4 {
                let mut ownership = [99; 768];
                native
                    .copy_paired_routes_into(&mut ids, &mut weights, rank, &mut ownership)
                    .unwrap();
                for (index, route) in original.routes.iter().enumerate() {
                    assert_eq!(ids[index], route.expert_id as i32);
                    assert_eq!(weights[index].to_bits(), route.gate_weight.to_bits());
                    let owners = (route.expert_id + pattern) % 4;
                    assert_eq!(
                        ownership[route.expert_id as usize],
                        i32::from(((owners >> (rank / 2)) & 1) == (rank % 2) as u32)
                    );
                }
                assert!(ownership[384..].iter().all(|&v| v == 0));
            }
            let partials = vec![0; native.plane_bytes().unwrap()];
            let response = native.response(1, &partials).unwrap();
            assert_eq!(response.header.flags, original.header.flags);
            response.validate().unwrap();
            request.routes[6].expert_id ^= 1 << OWNERS_SHIFT;
            assert!(V41BackboneRequest::validate_owned_paired(&request, 3).is_err());
            assert!(V41BackboneRequest::parse_paired(&request.encode().unwrap(), 3).is_err());
        }
    }

    #[test]
    fn current_native_frame_contract_still_rejects_paired_admission() {
        let mut request = crate::v41_expert::tests::request(2);
        let original = request.encode().unwrap();
        // Decoding a candidate word must not mutate the ordinary native frame.
        for route in &request.routes {
            let decoded = V41PairedRouteWord::decode(route.expert_id).unwrap();
            assert_eq!(decoded.owners, 0);
        }
        assert_eq!(request.encode().unwrap(), original);
        assert!(crate::v41_expert::V41BackboneRequest::validate_owned(&request, 2).is_ok());
        request.header.flags |= V41_EXL3_PAIRED_REQUEST_FLAG;
        let paired_frame = request.encode().unwrap();
        assert!(crate::v41_expert::V41BackboneRequest::parse(&paired_frame, 2).is_err());
        assert!(crate::v41_expert::V41BackboneRequest::parse_paired(&paired_frame, 2).is_ok());
        assert!(crate::v41_expert::V41BackboneRequest::validate_owned(&request, 2).is_err());
    }
}
