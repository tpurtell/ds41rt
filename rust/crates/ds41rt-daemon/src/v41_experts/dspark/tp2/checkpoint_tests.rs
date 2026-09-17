use super::{pair::Pair, rank::Weights};
use crate::v41_experts::{ExpertLayer, ExpertWeights};
use crate::v41_memory::device::{Device, Stream};
use anyhow::{ensure, Result};
use ds41rt_ffi::NativeLibrary;

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|v| f32::from_bits(u32::from(u16::from_ne_bytes([v[0], v[1]])) << 16))
        .collect()
}

#[test]
#[ignore = "requires checkpoint, both RTX GPUs and about 7 GiB free on RTX1"]
fn cuda_dspark_tp2_checkpoint_experts_and_graph() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    let devices = [
        Device {
            library: &lib,
            id: 0,
        },
        Device {
            library: &lib,
            id: 1,
        },
    ];
    let owner = devices[1];
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
    )?;
    let weights = Weights::load_pair(devices, &catalog, [16usize << 30; 2])?;
    eprintln!(
        "draft TP2 resident bytes: {:?}",
        weights.each_ref().map(|w| w.resident_bytes())
    );
    let mut pair = Pair::new(weights, 256)?;
    let root = Stream::new(owner)?;
    for stage in 0..3 {
        let full = owner.own(|| {
            ExpertWeights::load(&lib, &catalog, ExpertLayer::Dspark { stage }, 8usize << 30)
        })?;
        let mut reference = owner.own(|| full.execution(256, 2usize << 30))?;
        for (rows, include_shared) in [1u32, 7, 16, 65, 112, 1]
            .into_iter()
            .flat_map(|rows| [false, true].map(|shared| (rows, shared)))
        {
            let inputs = reference.inputs();
            let shared = reference.shared().unwrap();
            // Warm first so capture only encounters already initialized modules.
            for input in inputs {
                owner.run(|| lib.copy_h2d(input, &vec![0; input.bytes]))?;
            }
            owner.run(|| lib.copy_h2d(shared, &vec![0; shared.bytes]))?;
            unsafe {
                pair.enqueue(
                    stage,
                    rows,
                    inputs,
                    include_shared.then_some(shared),
                    root.raw,
                )?;
            }
            root.drain()?;
            pair.drain_peer()?;
            let graph = owner.run(|| unsafe {
                lib.cuda_graph_begin_capture(root.raw)?;
                let queued = pair.enqueue(
                    stage,
                    rows,
                    inputs,
                    include_shared.then_some(shared),
                    root.raw,
                );
                let captured = lib.cuda_graph_end_capture(root.raw);
                match (queued, captured) {
                    (Ok(()), Ok(graph)) => Ok(graph),
                    (Err(e), Ok(graph)) => {
                        lib.cuda_graph_exec_destroy(graph)?;
                        Err(e)
                    }
                    (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
                }
            })?;
            let checked = (|| -> Result<()> {
                for seed in [1usize, 19, 83] {
                    let hidden: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            bf16((((i * 37 + seed * 11) % 257) as f32 - 128.0) * 0.003)
                                .to_ne_bytes()
                        })
                        .collect();
                    let ids: Vec<u8> = (0..rows as usize * 3)
                        .flat_map(|i| {
                            (((i / 3 * 7 + i % 3 * 41 + seed) % 128) as i32).to_ne_bytes()
                        })
                        .collect();
                    let routing: Vec<u8> = (0..rows as usize * 3)
                        .flat_map(|i| (0.15 + (i % 3) as f32 * 0.1).to_ne_bytes())
                        .collect();
                    let shared_data: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| bf16(((i + seed) % 17) as f32 * 0.001).to_ne_bytes())
                        .collect();
                    for (buffer, data) in inputs.into_iter().zip([hidden, ids, routing]) {
                        owner.run(|| lib.copy_h2d(buffer, &data))?;
                    }
                    owner.run(|| lib.copy_h2d(shared, &shared_data))?;
                    owner.run(|| unsafe {
                        reference.launch(rows, include_shared)?;
                        reference.synchronize()
                    })?;
                    owner.run(|| unsafe { lib.cuda_graph_launch(graph, root.raw) })?;
                    root.drain()?;
                    pair.drain_peer()?;
                    let mut actual = vec![0; rows as usize * 10240];
                    let mut expected = actual.clone();
                    owner.run(|| lib.copy_d2h(&mut actual, pair.output()))?;
                    owner.run(|| lib.copy_d2h(&mut expected, reference.output().unwrap()))?;
                    let (actual, expected) = (values(&actual), values(&expected));
                    let mut error = 0f64;
                    let mut norm = 0f64;
                    let mut max = 0f32;
                    let mut exact = 0;
                    for (&a, &b) in actual.iter().zip(&expected) {
                        ensure!(a.is_finite() && b.is_finite(), "nonfinite draft output");
                        let d = (a - b).abs();
                        max = max.max(d);
                        error += f64::from(d).powi(2);
                        norm += f64::from(b).powi(2);
                        exact += usize::from(a == b);
                    }
                    let relative = (error / norm.max(1e-30)).sqrt();
                    eprintln!("draft TP2 stage={stage} rows={rows} shared={include_shared} seed={seed} relative_rms={relative:.7} max_abs={max:.7} exact={exact}/{}",actual.len());
                    ensure!(
                        relative < 0.0001,
                        "draft TP2 differs from full expert: relative RMS {relative}"
                    );
                }
                Ok(())
            })();
            let drained = root.drain().and(pair.drain_peer());
            owner.run(|| unsafe { lib.cuda_graph_exec_destroy(graph) })?;
            drained?;
            checked?;
        }
    }
    Ok(())
}
