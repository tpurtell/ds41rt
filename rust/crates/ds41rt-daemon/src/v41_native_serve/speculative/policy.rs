//! Binding for the online verification-length policy: the installed expert
//! placement, round observations from captured routes/timings, and the
//! `/v1/stats` export. The policy itself lives in `ds41rt_core` and performs
//! no device work.
use anyhow::{ensure, Result};
use crate::v41_experts::coordinator::NativeTp4Wave;
use ds41rt_core::{DsparkLayerClass, DsparkPlacement, DsparkPolicy, DSPARK_LAYERS};
use std::sync::Mutex;
use std::time::Instant;

const HIDDEN: f64 = 5120.;
const INTERMEDIATE: f64 = 2304.;

/// Bytes one device reads for one routed expert: gate, up and down slices of
/// `hidden x intermediate / tp` 4-bit values plus their UE8M0 (MXFP4, per 32)
/// or E4M3 (NVFP4, per 16) block scales. This is the standardized base for the
/// official checkpoint; other formats' differences land in the fitted bandwidth.
pub(super) fn expert_slice_bytes(tp: usize, nvfp4: bool) -> f64 {
    let scale_block = if nvfp4 { 16. } else { 32. };
    3. * HIDDEN * (INTERMEDIATE / tp as f64) * (0.5 + 1. / scale_block)
}

/// Resource class and per-device expert bytes of every installed layer.
pub(super) fn placement(transport: &NativeTp4Wave<'_>, nvfp4: bool) -> Result<DsparkPlacement> {
    let remote_tp = transport.native_topology()
        .map_or(transport.spark_world(), |topology| topology.tp() as usize);
    ensure!(remote_tp > 0, "Spark tensor-parallel width is zero");
    let class = std::array::from_fn(|layer| if transport.has_local_layer(layer) {
        DsparkLayerClass::Local } else { DsparkLayerClass::Remote });
    let bytes = std::array::from_fn(|layer| {
        let tp = if transport.has_tp2_layer(layer) { 2 }
            else if transport.has_local_layer(layer) { 1 } else { remote_tp };
        expert_slice_bytes(tp, nvfp4)
    });
    DsparkPlacement::new(class, bytes).map_err(anyhow::Error::msg)
}

/// Per-layer wall time from consecutive FFN completions; layer 0 has no
/// predecessor and stays in the round residual.
pub(super) fn layer_us(done: &[Option<Instant>]) -> [Option<f64>; DSPARK_LAYERS] {
    let mut out = [None; DSPARK_LAYERS];
    for layer in 1..DSPARK_LAYERS.min(done.len()) {
        if let (Some(previous), Some(current)) = (done[layer - 1], done[layer]) {
            if current >= previous {
                out[layer] = Some(current.duration_since(previous).as_secs_f64() * 1e6);
            }
        }
    }
    out
}

pub(super) fn sigmoid(x: f32) -> f64 {
    let x = f64::from(x);
    if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
}

static SNAPSHOT: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    SNAPSHOT.lock().ok().and_then(|slot| slot.clone()).unwrap_or(serde_json::Value::Null)
}

pub(super) fn publish(policy: &DsparkPolicy, draft_limit: usize) {
    let stats = policy.stats();
    let placement = policy.placement();
    let fit = |shared: bool| {
        let snapshot = policy.cost_snapshot(shared);
        let classes: Vec<_> = [DsparkLayerClass::Local, DsparkLayerClass::Remote].iter()
            .zip(snapshot.layers).map(|(class, c)| serde_json::json!({
                "class": class.label(),
                "intercept_us": c[0], "us_per_row": c[1], "us_per_mb": c[2],
                // µs per MB → GB/s: 1 MB / (c µs) = 1000 / c GB/s.
                "effective_gb_per_s": if c[2] > 0. { 1000. / c[2] } else { f64::INFINITY },
                "samples": c[3], "residual_scale_us": c[4],
            })).collect();
        serde_json::json!({
            "warm": policy.warm(shared),
            "layers": classes,
            "round": { "intercept_us": snapshot.round[0], "us_per_row": snapshot.round[1],
                "us_per_request": snapshot.round[2], "samples": snapshot.round[3],
                "residual_scale_us": snapshot.round[4] },
        })
    };
    let reliability: Vec<_> = (0..7).map(|p| serde_json::json!({
        "position": p + 1,
        "reached": stats.position_reached[p],
        "mean_confidence": if stats.position_reached[p] > 0 {
            stats.position_confidence[p] / stats.position_reached[p] as f64 } else { 0. },
        "accept_rate": if stats.position_reached[p] > 0 {
            stats.position_accepted[p] as f64 / stats.position_reached[p] as f64 } else { 0. },
    })).collect();
    let local_layers = (0..DSPARK_LAYERS).filter(|&l| placement.class(l) == DsparkLayerClass::Local).count();
    let value = serde_json::json!({
        "mode": if policy.fixed() { "fixed" } else { "bandwidth" },
        "draft_limit": draft_limit,
        "local_layers": local_layers,
        "remote_expert_bytes": placement.expert_bytes(DSPARK_LAYERS - 1),
        "rounds": stats.rounds,
        "selected_rounds": stats.selected_rounds,
        "verified_rows": stats.verified_rows,
        "verified_drafts": stats.verified_drafts,
        "accepted_drafts": stats.accepted_drafts,
        "request_rounds": stats.emitted_requests,
        "draft_rows_histogram": stats.draft_rows,
        "confidence_reliability": reliability,
        "prediction": {
            "rounds": stats.predicted_rounds,
            "mean_error_us": if stats.predicted_rounds > 0 {
                stats.prediction_error_us / stats.predicted_rounds as f64 } else { 0. },
            "mean_abs_relative_error": if stats.observed_us > 0. {
                stats.prediction_abs_error_us / stats.observed_us } else { 0. },
        },
        "solo": fit(false),
        "shared": fit(true),
    });
    if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(value); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn official_spark_tp4_slice_matches_the_packed_geometry() {
        // 3 x (1,474,560 FP4 bytes + 92,160 UE8M0 scale bytes).
        assert_eq!(super::expert_slice_bytes(4, false), 4_700_160.);
        assert_eq!(super::expert_slice_bytes(1, false), 18_800_640.);
        assert_eq!(super::expert_slice_bytes(2, false), 9_400_320.);
        assert_eq!(super::expert_slice_bytes(4, true), 4_976_640.);
    }
    #[test]
    fn layer_times_come_from_consecutive_completions() {
        let start = std::time::Instant::now();
        let at = |us| Some(start + std::time::Duration::from_micros(us));
        let mut done = vec![None; 40];
        done[0] = at(100); done[1] = at(400); done[2] = at(1000); done[4] = at(2000);
        let us = super::layer_us(&done);
        assert_eq!(us[0], None);
        assert!((us[1].unwrap() - 300.).abs() < 1e-6);
        assert!((us[2].unwrap() - 600.).abs() < 1e-6);
        assert_eq!(us[3], None);
        assert_eq!(us[4], None);
    }
}
