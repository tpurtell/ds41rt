//! Gather a bounded real-checkpoint PLE fixture through the serving loader.
use anyhow::{Context, Result};
use ds41rt_loader::{read_official_v41_catalog, EngramBatchStaging};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let snapshot = Path::new(args.get(1).context("missing snapshot")?);
    let output = Path::new(args.get(2).context("missing output directory")?);
    let catalog = read_official_v41_catalog("deepseek-ai/DeepSeek-V4.1-Flash", snapshot)?;
    std::fs::create_dir_all(output)?;
    let history = ds41rt_core::EngramHistory::new(0)?;
    let batch = history.prepare(0, &[Some(7), None, Some(31), Some(9182)], 4)?;
    let mut staging = EngramBatchStaging::new(16)?;
    let mut results = Vec::new();
    for (index, layer) in [1, 14].into_iter().enumerate() {
        // These immutable checkpoint snapshots also back the serving process.
        let table = unsafe { catalog.map_engram(layer)? };
        let view = staging.gather(&table, &[&batch, &batch], index)?;
        std::fs::write(output.join(format!("layer{layer}-weights.bin")), view.weights)?;
        std::fs::write(output.join(format!("layer{layer}-scales.bin")), view.scales)?;
        results.push(serde_json::json!({"layer":layer, "encoding":format!("{:?}", view.encoding),
            "global_scale":view.global_scale, "hash_rows":view.weights.len()/view.encoding.weight_bytes(),
            "weights_bytes":view.weights.len(), "scales_bytes":view.scales.len(),
            "text_mask":view.text_mask}));
    }
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(output.join("manifest.json"), &json)?;
    println!("{json}");
    Ok(())
}
