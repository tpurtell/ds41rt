//! Inspect the official catalog and fixed storage placement without loading weights.
use anyhow::{Context, Result};
use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};
use std::path::PathBuf;
fn main() -> Result<()> {
    let snapshot = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("usage: v41_catalog SNAPSHOT")?,
    );
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let budget = catalog.storage_budget()?;
    println!(
        "{}",
        serde_json::json!({
            "tensors": catalog.tensors().len(),
            "coordinator_checkpoint_bytes": budget.coordinator_bytes,
            "dspark_checkpoint_bytes": budget.dspark_bytes,
            "per_spark_checkpoint_bytes": budget.per_spark_bytes,
            "spark_rank_checkpoint_bytes": budget.spark_rank_bytes,
            "routed_exl3": catalog.exl3().is_some(),
            "host_mapped_bytes": budget.host_mapped_bytes,
        })
    );
    Ok(())
}
