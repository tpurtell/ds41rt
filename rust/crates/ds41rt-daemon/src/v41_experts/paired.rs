//! Lane-owned coordinator scratch; ownership travels in the existing route word.
use anyhow::{ensure, Context, Result};
use ds41rt_core::{Exl3BoundaryCost, Exl3Tp4OwnershipPlanner};
use ds41rt_transport::{ExpertProtocolV2Request, v41_expert::{V41BackboneRequest, V41PairedRouteWord, V41_EXL3_PAIRED_REQUEST_FLAG}};

/// Cost units must agree across weights and routed-row terms. The caller supplies
/// calibrated costs; this adapter does not assume a bandwidth or compute ratio.
#[derive(Clone, Copy, Default, serde::Deserialize)]
pub(crate) struct BoundaryCostModel {
    pub weight: [u64; 2],
    pub per_row: [u64; 2],
}

/// Explicit experimental profile; weights and per-row costs share one unit.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PairedProfile {
    schema: String,
    layers: Vec<Vec<BoundaryCostModel>>,
}
impl PairedProfile {
    pub(crate) fn from_env(compressed: bool) -> Result<Option<std::rc::Rc<Self>>> {
        let Some(path) = std::env::var_os("DS41RT_EXL3_PAIRED_COST_PROFILE") else { return Ok(None); };
        ensure!(compressed, "paired EXL3 profile requires an EXL3 checkpoint");
        let profile: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        profile.validate()?;
        Ok(Some(std::rc::Rc::new(profile)))
    }
    fn validate(&self) -> Result<()> {
        ensure!(self.schema == "ds41rt.exl3-paired-cost.v1", "unsupported paired cost schema");
        ensure!(self.layers.len() == 40 && self.layers.iter().all(|layer| layer.len() == 384),
            "paired profile requires 40 layers of 384 expert costs");
        for model in self.layers.iter().flatten() {
            for pair in 0..2 {
                let cost = model.per_row[pair].checked_mul(4096)
                    .and_then(|rows| model.weight[pair].checked_add(rows))
                    .context("paired profile cost overflow")?;
                ensure!(model.weight[pair] > 0 || model.per_row[pair] > 0, "paired profile needs positive costs");
                ensure!(cost.checked_mul(5 * 384).is_some(), "paired profile batch cost overflow");
            }
        }
        Ok(())
    }
    pub(crate) fn layer(&self, layer: usize) -> Result<&[BoundaryCostModel; 384]> {
        self.layers.get(layer).context("paired profile layer missing")?.as_slice().try_into()
            .map_err(|_| anyhow::anyhow!("paired profile expert extent mismatch"))
    }
}

pub(crate) struct PairedAssignment {
    planner: Exl3Tp4OwnershipPlanner,
    counts: [u32; 384],
    costs: [Exl3BoundaryCost; 384],
}
impl PairedAssignment {
    pub(crate) fn new() -> Self {
        Self { planner: Exl3Tp4OwnershipPlanner::new(384), counts: [0; 384], costs: [Exl3BoundaryCost::default(); 384] }
    }

    /// Validate and plan completely before mutating the request. On success the
    /// request owns all decisions, so this scratch can immediately serve another
    /// batch. No other lane or GPU operation is consulted.
    pub(crate) fn encode(&mut self, request: &mut ExpertProtocolV2Request,
        models: &[BoundaryCostModel; 384], tie_seed: usize) -> Result<[u64; 4]> {
        V41BackboneRequest::validate_owned(request, 4096)?;
        self.counts.fill(0);
        self.costs.fill(Exl3BoundaryCost::default());
        for route in &request.routes {
            self.counts[route.expert_id as usize] += 1;
        }
        let mut mandatory = [0u64; 4];
        for (expert, &count) in self.counts.iter().enumerate() {
            if count == 0 { continue; }
            let mut pair_cost = [0; 2];
            for pair in 0..2 {
                pair_cost[pair] = models[expert].per_row[pair].checked_mul(u64::from(count))
                    .and_then(|rows| models[expert].weight[pair].checked_add(rows))
                    .context("paired boundary cost overflow")?;
                ensure!(pair_cost[pair] > 0, "active paired expert needs positive cost");
                let base = pair_cost[pair].checked_mul(4).context("paired mandatory cost overflow")?;
                for rank in pair * 2..pair * 2 + 2 {
                    mandatory[rank] = mandatory[rank].checked_add(base).context("paired batch cost overflow")?;
                }
            }
            self.costs[expert] = Exl3BoundaryCost { active: true, pair_cost };
        }
        let plan = self.planner.plan(&self.costs, mandatory, tie_seed)?;
        // All IDs and owner values were checked above; packing cannot fail now.
        for route in &mut request.routes {
            route.expert_id = V41PairedRouteWord { expert_id: route.expert_id,
                owners: plan.owners[route.expert_id as usize] }.encode()?;
        }
        request.header.flags |= V41_EXL3_PAIRED_REQUEST_FLAG;
        Ok(plan.rank_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_transport::{ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind};
    fn request() -> ExpertProtocolV2Request {
        let mut request = ExpertProtocolV2Request::new(1, 1, 30, 5120, ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..3).map(|r| ExpertProtocolV2RowDescriptor { row_id:r as u64, source_kind:ExpertV2SourceKind::Decode,
                source_request_id:r as u64 + 1, token_position:0, route_offset:r*6, route_count:6 }).collect(),
            (0..18).map(|r| ExpertProtocolV2RouteEntry { row_index:r/6, expert_id:r%6, gate_weight:1.0/6.0 }).collect(),
            vec![0; 3*5280]).unwrap();
        request.header.flags |= ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        request
    }
    #[test]
    fn paired_profile_validates_full_extent_and_safe_cost_bounds() -> Result<()> {
        let mut profile = PairedProfile { schema: "ds41rt.exl3-paired-cost.v1".into(),
            layers: vec![vec![BoundaryCostModel { weight:[10;2], per_row:[2;2] };384];40] };
        profile.validate()?;
        assert!(profile.layer(40).is_err());
        profile.layers[39][383].weight[1] = u64::MAX;
        assert!(profile.validate().is_err());
        profile.layers[39][383] = BoundaryCostModel::default();
        assert!(profile.validate().is_err());
        profile.layers[39][383] = BoundaryCostModel { weight:[10;2], per_row:[2;2] };
        profile.layers[39].pop();
        assert!(profile.validate().is_err());
        assert!(profile.layer(39).is_err());
        Ok(())
    }

    #[test]
    fn paired_assignment_balances_reuse_and_owns_decisions() -> Result<()> {
        let mut scratch = PairedAssignment::new();
        let models = [BoundaryCostModel { weight:[10;2], per_row:[2;2] };384];
        let mut first = request();
        let original = first.encode()?;
        assert_eq!(scratch.encode(&mut first, &models, 0)?, [27*16;4]);
        V41BackboneRequest::validate_owned_paired(&first, 3)?;
        let encoded = first.encode()?;
        assert_eq!(original.len(), encoded.len());
        let mut second = request();
        scratch.encode(&mut second, &models, 1)?;
        assert_eq!(first.encode()?, encoded);
        assert_ne!(first.encode()?, second.encode()?);
        let mut bad = request();
        let before = bad.encode()?;
        let overflow = [BoundaryCostModel { weight:[u64::MAX;2], per_row:[1;2] };384];
        assert!(scratch.encode(&mut bad, &overflow, 0).is_err());
        assert_eq!(bad.encode()?, before);
        assert!(scratch.encode(&mut bad, &[BoundaryCostModel::default();384], 0).is_err());
        assert_eq!(bad.encode()?, before);
        scratch.encode(&mut bad, &models, 0)?;
        assert_eq!(bad.encode()?, encoded);
        Ok(())
    }
}
