//! Independent B12x fixtures exercised through complete resident draft weights.
use super::*;
use ds41rt_ffi::NativeLibrary;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires two RTX GPUs, EXL3 snapshot/package and dSpark B12x fixtures"]
fn exl3_dspark_resident_stages_match_b12x() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    let snapshot = PathBuf::from(std::env::var("DS41RT_SNAPSHOT")?);
    let package = PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
    let fixtures = PathBuf::from(std::env::var("DS41RT_EXL3_FIXTURE")?);
    let catalog =
        ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &snapshot)?;
    for device in [0, 1] {
        lib.cuda_set_device(device)?;
        let owner = DsparkWeights::load_with_width(
            &lib,
            &catalog,
            256,
            2,
            32usize << 30,
            16 << 20,
            7,
            Some(&package),
        )?;
        for stage in 0..3 {
            let mut lanes = [
                owner.expert_wave(stage, 256)?,
                owner.expert_wave(stage, 256)?,
            ];
            for capacity in [1usize, 16, 80, 256] {
                let directory = fixtures.join(format!("stage{stage}/m{capacity}"));
                let meta: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(directory.join("fixture.json"))?)?;
                let export: serde_json::Value = serde_json::from_slice(&std::fs::read(
                    package.join(format!("m{capacity}/v41_exl3.json")),
                )?)?;
                ensure!(
                    meta["layer"] == format!("mtp.{stage}")
                        && meta["capacity"] == capacity
                        && meta["width"] == 2304
                        && meta["slice_start"] == 0
                        && meta["topk"] == 3
                        && meta["canonical_routes"] == true
                        && meta["input_format"] == "bf16"
                        && meta["output_dtype"] == "bf16"
                        && meta["reference_experts"] == 6
                        && meta["snapshot_revision"].as_str()
                            == snapshot.file_name().and_then(|v| v.to_str()),
                    "dSpark fixture geometry or identity mismatch"
                );
                for key in ["tile", "direct", "output_dtype"] {
                    ensure!(
                        meta[key] == export[key],
                        "dSpark reference policy mismatch: {key}"
                    );
                }
                let mut data = Vec::new();
                for (name, stride) in [
                    ("input", 10240),
                    ("ids", 12),
                    ("weights", 12),
                    ("expected", 10240),
                ] {
                    let bytes = std::fs::read(directory.join(format!("{name}.bin")))?;
                    ensure!(
                        bytes.len() == capacity * stride
                            && meta["artifacts"][name]["bytes"].as_u64()
                                == Some(bytes.len() as u64)
                            && meta["artifacts"][name]["sha256"].as_str()
                                == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
                        "corrupt dSpark fixture {name}"
                    );
                    data.push(bytes);
                }
                ensure!(
                    data[3]
                        .chunks_exact(2)
                        .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80)
                        && data[3].iter().any(|&v| v != 0),
                    "nonfinite or zero reference"
                );
                for rows in [
                    capacity,
                    1,
                    capacity.min(3),
                    capacity.saturating_sub(1).max(1),
                    capacity,
                ] {
                    let mut outputs = Vec::new();
                    for (lane, wave) in lanes.iter_mut().enumerate() {
                        let DraftExperts::Exl3(wave) = wave else {
                            anyhow::bail!("expected EXL3 draft owner")
                        };
                        let mut inputs = wave.inputs.each_ref().map(|a| a.buffer);
                        let order: Vec<_> = (0..rows)
                            .map(|row| if lane == 0 { row } else { rows - 1 - row })
                            .collect();
                        for (index, stride) in [10240, 12, 12].into_iter().enumerate() {
                            let reordered: Vec<_> = order
                                .iter()
                                .flat_map(|&row| {
                                    data[index][row * stride..(row + 1) * stride]
                                        .iter()
                                        .copied()
                                })
                                .collect();
                            inputs[index].bytes = reordered.len();
                            lib.copy_h2d(inputs[index], &reordered)?;
                        }
                        let state = wave
                            .states
                            .iter_mut()
                            .find(|v| v.capacity() == capacity)
                            .unwrap();
                        let output =
                            unsafe { state.launch_layer(stage, inputs, rows, wave.stream.raw)? };
                        outputs.push((output, order));
                    }
                    // Both lanes launch before either is drained.
                    for wave in &lanes {
                        wave.synchronize()?;
                    }
                    for (lane, (output, order)) in outputs.into_iter().enumerate() {
                        let mut actual = vec![0; output.bytes];
                        lib.copy_d2h(&mut actual, output)?;
                        let expected: Vec<_> = order
                            .iter()
                            .flat_map(|&row| {
                                data[3][row * 10240..(row + 1) * 10240].iter().copied()
                            })
                            .collect();
                        ensure!(actual == expected, "dSpark B12x mismatch device={device} stage={stage} capacity={capacity} rows={rows} lane={lane}");
                    }
                }
                eprintln!("PASS dSpark B12x device={device} stage={stage} capacity={capacity}: full 128-expert residency, six reference experts, changed rows and independent lane orders");
            }
        }
    }
    lib.cuda_set_device(0)?;
    Ok(())
}
