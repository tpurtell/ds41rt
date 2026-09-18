//! Placement-aware verification costs. Profiles are calibrated offline; loading
//! one performs no GPU work and never waits for the other decode lane.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use crate::v41_experts::coordinator::NativeTp4Wave;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    version: u32,
    // Per layer: intercept, rows, distinct experts, max(rows - 16, 0).
    experts: BTreeMap<String, [f64; 4]>,
    // Per round: intercept, rows, requests, max(rows - 16, 0).
    other: BTreeMap<String, [f64; 4]>,
}

pub(super) struct Model {
    layers: [[f64; 4]; 40],
    other: [f64; 4],
}

impl Model {
    /// Single-RTX uses the calibrated placement profile by default. Explicit
    /// profiles override either layout; "legacy" restores the original formula.
    pub fn from_environment(transport: &NativeTp4Wave<'_>) -> Result<Option<Self>> {
        let path = std::env::var_os("DS41RT_ADAPTIVE_COST_PROFILE");
        if path.as_deref() == Some(std::ffi::OsStr::new("legacy")) { return Ok(None); }
        let placement: [String; 40] = std::array::from_fn(|layer| {
            let backend = if transport.has_tp2_layer(layer) { "rtx_tp2" }
                else if transport.has_local_layer(layer) { "rtx_local" }
                else if transport.spark_world() == 2 { "spark_tp2" } else { "spark_tp4" };
            let shared_tp = if transport.has_tp2_shared_layer(layer) { 2 } else { 1 };
            format!("{backend}_shared{shared_tp}")
        });
        let gpus = if (0..40).any(|layer| transport.has_tp2_shared_layer(layer)
            || transport.has_tp2_layer(layer)) { 2 } else { 1 };
        let bytes = match &path {
            Some(path) => std::fs::read(path).context("reading adaptive cost profile")?,
            // The built-in measurements are TP4, not TP2. Use the legacy
            // adaptive heuristic until a Spark TP2 calibration is supplied.
            None if gpus == 1 && transport.spark_world() == 4 => include_bytes!("cost-profile.json").to_vec(),
            None => return Ok(None),
        };
        let model = Self::parse(&bytes, &placement, gpus)?;
        tracing::info!(profile=?path, gpus, placement=?placement,
            "placement-aware adaptive costs loaded");
        Ok(Some(model))
    }

    fn parse(bytes: &[u8], placement: &[String; 40], gpus: usize) -> Result<Self> {
        let profile: Profile = serde_json::from_slice(bytes).context("parsing adaptive cost profile")?;
        ensure!(profile.version == 1, "unsupported adaptive cost profile version");
        ensure!(profile.experts.values().chain(profile.other.values()).flatten()
            .all(|v| v.is_finite() && *v >= 0.), "adaptive cost coefficients must be finite and nonnegative");
        let mut layers = [[0.; 4]; 40];
        for (layer, backend) in placement.iter().enumerate() {
            layers[layer] = *profile.experts.get(backend)
                .with_context(|| format!("adaptive cost profile missing installed backend {backend}"))?;
        }
        let other = *profile.other.get(&gpus.to_string())
            .context("adaptive cost profile missing GPU layout")?;
        ensure!(other[0] + layers.iter().map(|c| c[0]).sum::<f64>() > 0.,
            "adaptive cost profile needs a positive base cost");
        let model = Self { layers, other };
        ensure!(model.verify_us(64, 8, &[384; 40]).is_finite(),
            "adaptive cost coefficients overflow the decode extent");
        Ok(model)
    }

    pub fn verify_us(&self, rows: usize, requests: usize, unique: &[usize; 40]) -> f64 {
        let rows = rows as f64;
        let extra = (rows - 16.).max(0.);
        self.other[0] + self.other[1]*rows + self.other[2]*requests as f64 + self.other[3]*extra
            + self.layers.iter().zip(unique).map(|(c, &unique)|
                c[0] + c[1]*rows + c[2]*unique as f64 + c[3]*extra).sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const PROFILE: &[u8] = br#"{"version":1,"experts":{"spark_tp4_shared1":[10,2,4,1],"rtx_local_shared1":[5,1,1,0]},"other":{"1":[100,3,7,0]}}"#;

    #[test]
    fn moving_one_layer_changes_only_its_backend_cost() -> Result<()> {
        let mut placement = std::array::from_fn(|_| "spark_tp4_shared1".to_owned());
        let original = Model::parse(PROFILE, &placement, 1)?;
        let mut unique = [2; 40];
        unique[7] = 11;
        placement[7] = "rtx_local_shared1".into();
        let moved = Model::parse(PROFILE, &placement, 1)?;
        for rows in [2usize, 6, 16, 32, 64] {
            let difference = 5. + rows as f64 + 3.*11. + rows.saturating_sub(16) as f64;
            assert_eq!(original.verify_us(rows, 2, &unique)-moved.verify_us(rows, 2, &unique), difference);
        }
        assert!(moved.verify_us(32, 4, &unique) >= moved.verify_us(16, 4, &unique));
        Ok(())
    }

    #[test]
    fn builtin_profile_covers_single_rtx_residency_changes() -> Result<()> {
        for local_layers in 0..=40 {
            let placement = std::array::from_fn(|layer| if layer < local_layers {
                "rtx_local_shared1".to_owned()
            } else { "spark_tp4_shared1".to_owned() });
            let model = Model::parse(include_bytes!("cost-profile.json"), &placement, 1)?;
            assert!(model.verify_us(64, 8, &[384; 40]).is_finite());
        }
        Ok(())
    }

    #[test]
    fn profiles_cannot_silently_misprice_missing_backends_or_layouts() {
        let mut placement = std::array::from_fn(|_| "spark_tp4_shared1".to_owned());
        assert!(Model::parse(PROFILE, &placement, 2).is_err());
        placement[20] = "spark_tp4_shared2".into();
        assert!(Model::parse(PROFILE, &placement, 1).is_err());
        placement[20] = "spark_tp4_shared1".into();
        let invalid = String::from_utf8(PROFILE.to_vec()).unwrap().replace("[10,2,4,1]", "[10,-2,4,1]");
        assert!(Model::parse(invalid.as_bytes(), &placement, 1).is_err());
    }
}
