//! Inspect one compressed layer plan; optionally read every bounded staging job.
use anyhow::{Context, Result};
use ds41rt_loader::{
    read_official_v41_catalog, V41Exl3Layer, V41Exl3Partition, OFFICIAL_V41_MODEL_ID,
};
use sha2::{Digest, Sha256};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 5,
        "usage: v41_exl3_residency SNAPSHOT LAYER|mtp:STAGE WORLD RANK [--read] [--paired-tp4]"
    );
    anyhow::ensure!(
        args[5..]
            .iter()
            .all(|s| s == "--read" || s == "--paired-tp4"),
        "unknown inspection option"
    );
    let layout = if args[5..].iter().any(|s| s == "--paired-tp4") {
        V41Exl3Partition::PairedTp4
    } else {
        V41Exl3Partition::Disjoint
    };
    let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, Path::new(&args[1]))?;
    let layer = if let Some(stage) = args[2].strip_prefix("mtp:") {
        V41Exl3Layer::Dspark(stage.parse()?)
    } else {
        V41Exl3Layer::Backbone(args[2].parse()?)
    };
    let plan = catalog
        .exl3()
        .context("EXL3 checkpoint required")?
        .residency_with_layout(layer, args[3].parse()?, args[4].parse()?, layout)?;
    let mut hashes = Vec::new();
    if args[5..].iter().any(|s| s == "--read") {
        let mut staging = vec![0; plan.staging_bytes()];
        let mut scratch = vec![0; plan.scratch_bytes(&catalog)?];
        for (index, job) in plan.loads.iter().enumerate() {
            let bytes = plan.read_into(&catalog, index, &mut staging, &mut scratch)?;
            hashes.push(serde_json::json!({"tensor":job.tensor,
                "sha256":format!("{:x}", Sha256::digest(&staging[..bytes]))}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"resident_bytes":plan.bytes(),
        "staging_bytes":plan.staging_bytes(),"scratch_bytes":plan.scratch_bytes(&catalog)?,
        "plan":plan,"read_hashes":hashes})
    );
    Ok(())
}
