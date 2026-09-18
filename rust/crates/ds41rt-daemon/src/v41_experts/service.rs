//! Native GPU resources and inference QPs share one owning thread.
mod local;
mod backend;

use super::{ExpertLayer, ExpertWeights, HostExpertExchange};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};
use ds41rt_transport::v41_expert::V41BackboneRequest;
use std::{path::PathBuf, sync::mpsc, thread};

pub(crate) async fn run(args: crate::cli::NativeExpertDaemonArgs) -> Result<()> {
    let config = NativeExpertServiceConfig {
        library: args.native_lib,
        exl3_aot_dir: args.exl3_aot_dir,
        snapshot: args.snapshot,
        rank: args.rank as usize,
        first_layer: args.first_layer as usize,
        capacity: args.capacity,
        device_budget: args.device_budget_bytes,
        max_frame_bytes: args.max_frame_bytes,
    };
    tokio::task::spawn_blocking(move || local::run(config, &args.listen))
        .await
        .context("native local RoCE owner failed")?
}

pub(crate) struct NativeExpertServiceConfig {
    pub library: PathBuf,
    pub exl3_aot_dir: Option<PathBuf>,
    pub snapshot: PathBuf,
    pub rank: usize,
    pub first_layer: usize,
    pub capacity: u32,
    pub device_budget: usize,
    pub max_frame_bytes: usize,
}

fn load_weights<'a>(
    library: &'a NativeLibrary,
    config: &NativeExpertServiceConfig,
) -> Result<(backend::Weights<'a>, usize)> {
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &config.snapshot)?;
    ensure!(config.first_layer < 40, "native first layer must be 0..39");
    if catalog.exl3().is_some() { return backend::load_exl3(library, &catalog, config); }
    // NVFP4 backbone experts load through the format-aware ExpertWeights path.
    let mut resident = 0usize;
    let mut staging = 0usize;
    for layer in config.first_layer..40 {
        let plan = ExpertWeights::plan(
            library,
            &catalog,
            ExpertLayer::Backbone {
                layer,
                rank: config.rank,
            },
        )?;
        resident = resident
            .checked_add(plan.resident_bytes)
            .context("resident budget overflow")?;
        staging = staging.max(plan.device_staging_bytes);
    }
    ensure!(
        resident
            .checked_add(staging)
            .context("loading budget overflow")?
            <= config.device_budget,
        "native TP weights and staging exceed device budget"
    );
    // The Spark plans its own role: TP4 shard for NVFP4, TP4 padded for native.
    let nvfp4 = catalog.nvfp4().is_some();
    let workspace = ExpertWeights::plan_execution(
        ExpertLayer::Backbone {
            layer: 0,
            rank: config.rank,
        },
        library,
        config.capacity,
        nvfp4,
    )?
    .total()?;
    ensure!(
        resident
            .checked_add(workspace)
            .context("execution budget overflow")?
            <= config.device_budget,
        "native TP weights and execution workspace exceed device budget"
    );
    let mut weights = Vec::with_capacity(40 - config.first_layer);
    let mut remaining = config.device_budget;
    for layer in config.first_layer..40 {
        let started = std::time::Instant::now();
        let weight = ExpertWeights::load(
            library,
            &catalog,
            ExpertLayer::Backbone {
                layer,
                rank: config.rank,
            },
            remaining,
        )?;
        remaining = remaining
            .checked_sub(weight.budget().resident_bytes)
            .context("resident budget exhausted")?;
        tracing::info!(
            rank = config.rank,
            layer,
            resident_bytes = weight.budget().resident_bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "native expert layer loaded"
        );
        weights.push(weight);
    }
    Ok((backend::Weights::Full(weights), remaining))
}

impl NativeExpertServiceConfig {
    /// Resolve this rank's EXL3 AOT package for the running checkpoint's
    /// decoder tiers (multi-family images) with the legacy single-family
    /// location as fallback. An explicit --exl3-aot-dir is used verbatim.
    fn exl3_directory_for(&self, tiers: &[usize]) -> PathBuf {
        self.exl3_aot_dir.clone().unwrap_or_else(|| {
            crate::v41_experts::exl3::aot_layout_directory(
                &self.library,
                tiers,
                &format!("tp4-rank{}", self.rank),
            )
        })
    }
}
