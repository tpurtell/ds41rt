//! Placement-aware verification costs. Profiles are calibrated offline; loading
//! one performs no GPU work and never waits for the other decode lane.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;
use crate::v41_experts::coordinator::NativeTp4Wave;
use ds41rt_transport::v41_expert::V41SparkTopology;

/// Remote backend label for one layer. The label must describe the actual
/// Spark topology: a four-rank TP2×EP2 launch is not `spark_tp4`.
fn remote_backend_label(topology: Option<V41SparkTopology>, world: usize) -> String {
    match topology {
        Some(topology) => format!("spark_tp{}ep{}", topology.tp(), topology.ep()),
        None if world == 2 => "spark_tp2".to_owned(),
        None => "spark_tp4".to_owned(),
    }
}

/// The built-in calibrations measure the legacy single-RTX TP4×EP1 layout only.
/// Any explicit replicated topology, or a multi-RTX layout, needs an explicit
/// profile rather than a silently reused TP4 table.
fn builtin_profile_applies(gpus: usize, topology: Option<V41SparkTopology>, world: usize) -> bool {
    gpus == 1 && topology.is_none() && world == 4
}

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
    /// Single-RTX uses the checkpoint-format calibration by default. Explicit
    /// profiles override either layout; "legacy" restores the original formula.
    pub fn from_environment(transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<Option<Self>> {
        let path = std::env::var_os("DS41RT_ADAPTIVE_COST_PROFILE");
        if path.as_deref() == Some(std::ffi::OsStr::new("legacy")) { return Ok(None); }
        let placement: [String; 40] = std::array::from_fn(|layer| {
            let backend = if transport.has_tp2_layer(layer) { "rtx_tp2".to_owned() }
                else if transport.has_local_layer(layer) { "rtx_local".to_owned() }
                else { remote_backend_label(transport.native_topology(), transport.spark_world()) };
            let shared_tp = if transport.has_tp2_shared_layer(layer) { 2 } else { 1 };
            format!("{backend}_shared{shared_tp}")
        });
        let gpus = if (0..40).any(|layer| transport.has_tp2_shared_layer(layer)
            || transport.has_tp2_layer(layer)) { 2 } else { 1 };
        let bytes = match &path {
            Some(path) => std::fs::read(path).context("reading adaptive cost profile")?,
            // The built-in measurements are legacy single-RTX TP4×EP1. An
            // explicit TP×EP topology has no shipped calibration yet, so it uses
            // the legacy adaptive heuristic until a truthful profile is supplied.
            None if builtin_profile_applies(gpus, transport.native_topology(), transport.spark_world()) =>
                Self::builtin(nvfp4).to_vec(),
            None => return Ok(None),
        };
        let model = Self::parse(&bytes, &placement, gpus)?;
        tracing::info!(profile=?path, nvfp4, gpus, placement=?placement,
            "placement-aware adaptive costs loaded");
        Ok(Some(model))
    }

    fn builtin(nvfp4: bool) -> &'static [u8] {
        if nvfp4 { include_bytes!("cost-profile-nvfp4.json") }
        else { include_bytes!("cost-profile.json") }
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
            for nvfp4 in [false, true] {
                let model = Model::parse(Model::builtin(nvfp4), &placement, 1)?;
                assert!(model.verify_us(64, 8, &[384; 40]).is_finite());
            }
        }
        Ok(())
    }

    #[test]
    fn checkpoint_format_selects_distinct_calibration() -> Result<()> {
        let placement = std::array::from_fn(|_| "spark_tp4_shared1".to_owned());
        let native = Model::parse(Model::builtin(false), &placement, 1)?;
        let nvfp4 = Model::parse(Model::builtin(true), &placement, 1)?;
        assert_eq!(Model::builtin(false), include_bytes!("cost-profile.json"));
        assert_ne!(native.verify_us(6, 1, &[24; 40]), nvfp4.verify_us(6, 1, &[24; 40]));
        Ok(())
    }

    #[test]
    fn remote_backend_labels_report_the_actual_topology() {
        for (tp, ep) in [(2u8, 1u8), (3, 1), (4, 1), (2, 2), (3, 2), (2, 3)] {
            let topology = V41SparkTopology::new(tp, ep).unwrap();
            assert_eq!(
                remote_backend_label(Some(topology), topology.world_size()),
                format!("spark_tp{tp}ep{ep}")
            );
        }
        // A four-rank TP2×EP2 launch must not be labelled legacy TP4.
        assert_eq!(remote_backend_label(Some(V41SparkTopology::new(2, 2).unwrap()), 4), "spark_tp2ep2");
        assert_ne!(remote_backend_label(Some(V41SparkTopology::new(2, 2).unwrap()), 4), "spark_tp4");
        // Legacy labels are preserved exactly.
        assert_eq!(remote_backend_label(None, 4), "spark_tp4");
        assert_eq!(remote_backend_label(None, 2), "spark_tp2");
    }

    #[test]
    fn builtin_tp4_profile_never_covers_an_explicit_topology() {
        let tp2ep2 = Some(V41SparkTopology::new(2, 2).unwrap());
        let tp3ep2 = Some(V41SparkTopology::new(3, 2).unwrap());
        let tp4ep1 = Some(V41SparkTopology::new(4, 1).unwrap());
        assert!(builtin_profile_applies(1, None, 4));
        assert!(!builtin_profile_applies(1, None, 2));
        assert!(!builtin_profile_applies(2, None, 4));
        // A single-RTX six-rank TP3×EP2 launch is not the legacy TP4 layout even
        // though the RTX side is one GPU.
        assert!(!builtin_profile_applies(1, tp3ep2, 6));
        for topology in [tp2ep2, tp3ep2, tp4ep1] {
            assert!(!builtin_profile_applies(1, topology, topology.unwrap().world_size()));
        }
        // Explicit profiles still use the truthful topology label.
        let placement = std::array::from_fn(|_| "spark_tp2ep2_shared1".to_owned());
        let profile = br#"{"version":1,"experts":{"spark_tp2ep2_shared1":[10,2,4,1]},"other":{"1":[100,3,7,0]}}"#;
        let model = Model::parse(profile, &placement, 1).unwrap();
        assert!(model.verify_us(16, 2, &[384; 40]).is_finite());
    }

    #[test]
    fn profiles_cannot_silently_misprice_missing_backends_or_layouts() {        let mut placement = std::array::from_fn(|_| "spark_tp4_shared1".to_owned());
        assert!(Model::parse(PROFILE, &placement, 2).is_err());
        placement[20] = "spark_tp4_shared2".into();
        assert!(Model::parse(PROFILE, &placement, 1).is_err());
        placement[20] = "spark_tp4_shared1".into();
        let invalid = String::from_utf8(PROFILE.to_vec()).unwrap().replace("[10,2,4,1]", "[10,-2,4,1]");
        assert!(Model::parse(invalid.as_bytes(), &placement, 1).is_err());
    }
}
