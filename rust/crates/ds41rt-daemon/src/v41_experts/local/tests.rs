use super::*;
use sha2::{Digest, Sha256};

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

#[test]
fn local_activation_contract_selects_bf16_only_for_nvfp4() -> Result<()> {
    for rows in [1, 3, 16, 80, 4096] {
        let bf16 = Ds41rtDeviceBuffer { ptr: 0x1000usize as *mut c_void,
            bytes: rows as usize * 10240, device_id: 0, flags: 0 };
        let fp8 = Ds41rtDeviceBuffer { ptr: 0x2000usize as *mut c_void,
            bytes: rows as usize * 5280, device_id: 0, flags: 0 };
        for (nvfp4, expected) in [(false, fp8), (true, bf16)] {
            let format = LocalInputFormat::for_nvfp4(nvfp4);
            let selected = format.select(rows, bf16, fp8)?;
            assert_eq!(selected.ptr, expected.ptr);
            assert_eq!(selected.bytes, expected.bytes);
            assert!(format.select(rows, fp8, bf16).is_err());
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires RTX, full EXL3 snapshot, TP1 package and six-expert FP8-wire fixture"]
fn exl3_local_capacity_lanes_and_shared_sum_match_reference() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        Path::new(&std::env::var("DS41RT_EXL3_SNAPSHOT")?),
    )?;
    let directory = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
    let fixture = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_FIXTURE")?);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
    ensure!(
        meta["canonical_routes"] == true
            && meta["input_format"] == "fp8_k32"
            && meta["output_dtype"] == "fp32"
            && meta["layer"] == "layers.0"
            && meta["slice_start"] == 0
            && meta["width"] == 2304
            && meta["capacity"] == 16,
        "unexpected local expert fixture"
    );
    let payloads = ["input", "ids", "weights", "expected"]
        .map(|name| -> Result<Vec<u8>> {
            let data = std::fs::read(fixture.join(format!("{name}.bin")))?;
            ensure!(
                Some(data.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                    && Some(format!("{:x}", Sha256::digest(&data)).as_str())
                        == meta["artifacts"][name]["sha256"].as_str(),
                "corrupt local fixture"
            );
            Ok(data)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let one_meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("m1/fixture.json"))?)?;
    ensure!(
        one_meta["capacity"] == 1
            && one_meta["width"] == 2304
            && one_meta["input_format"] == "fp8_k32"
            && one_meta["output_dtype"] == "fp32",
        "unexpected single-row reference"
    );
    let mut one_expected = Vec::new();
    for (index, name) in ["input", "ids", "weights", "expected"].iter().enumerate() {
        let bytes = std::fs::read(fixture.join(format!("m1/{name}.bin")))?;
        ensure!(
            Some(bytes.len() as u64) == one_meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
                    == one_meta["artifacts"][name]["sha256"].as_str(),
            "corrupt single-row fixture"
        );
        if index < 3 {
            ensure!(
                payloads[index].starts_with(&bytes),
                "single-row reference input differs"
            );
        } else {
            one_expected = bytes;
        }
    }
    for (capacity, fixture_meta) in [(1, &one_meta), (16, &meta)] {
        let export: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join(format!("m{capacity}/v41_exl3.json")),
        )?)?;
        for key in ["tile", "direct", "output_dtype"] {
            ensure!(
                export[key] == fixture_meta[key],
                "reference/kernel policy mismatch"
            );
        }
    }
    let plan = Exl3Weights::plan(&catalog, ExpertLayer::BackboneFull { layer: 0 })?;
    let workspace = LocalExpertWave::exl3_device_bytes(&directory, 16)?;
    ensure!(
        lib.cuda_memory_info()?.0 > plan.resident_bytes + workspace * 2 + (256 << 20),
        "insufficient GPU headroom for local qualification"
    );
    let weights = Rc::new(vec![Exl3Weights::load(
        &lib,
        &catalog,
        ExpertLayer::BackboneFull { layer: 0 },
        plan.peak_device_bytes()?,
    )?]);
    let inputs = payloads[..3]
        .iter()
        .map(|data| -> Result<_> {
            let allocation = DeviceAllocation::new(&lib, data.len())?;
            lib.copy_h2d(allocation.buffer, data)?;
            Ok(allocation)
        })
        .collect::<Result<Vec<_>>>()?;
    let shared = (0..2)
        .map(|lane| -> Result<_> {
            let bytes: Vec<u8> = (0..16 * 5120)
                .flat_map(|i| {
                    bf16((i % 17) as f32 * 0.0625 * if lane == 0 { 1.0 } else { -1.0 })
                        .to_le_bytes()
                })
                .collect();
            let allocation = DeviceAllocation::new(&lib, bytes.len())?;
            lib.copy_h2d(allocation.buffer, &bytes)?;
            Ok((allocation, bytes))
        })
        .collect::<Result<Vec<_>>>()?;
    // Both owners drop before their borrowed inputs, and drain before scratch.
    let mut lanes = [
        unsafe { LocalExpertWave::new_exl3(&lib, weights.clone(), &directory, 16, workspace)? },
        unsafe { LocalExpertWave::new_exl3(&lib, weights, &directory, 16, workspace)? },
    ];
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    for rows in [1u32, 3, 16, 1] {
        let views = std::array::from_fn(|i| Ds41rtDeviceBuffer {
            bytes: rows as usize * if i == 0 { 5280 } else { 24 },
            ..inputs[i].buffer
        });
        for lane in 0..2 {
            lib.copy_h2d(lanes[lane].output.buffer, &vec![0xa5; 16 * 10240])?;
            let shared_view = Ds41rtDeviceBuffer {
                bytes: rows as usize * 10240,
                ..shared[lane].0.buffer
            };
            assert!(unsafe { lanes[lane].enqueue_buffers(1, rows, views, shared_view) }.is_err());
            let mut short = views;
            short[0].bytes -= 1;
            assert!(unsafe { lanes[lane].enqueue_buffers(0, rows, short, shared_view) }.is_err());
            unsafe {
                lanes[lane].enqueue_buffers(0, rows, views, shared_view)?;
            }
        }
        for lane in 0..2 {
            runtime.block_on(lanes[lane].stream.wait())?;
            let mut actual = vec![0; 16 * 10240];
            lib.copy_d2h(&mut actual, lanes[lane].output.buffer)?;
            for i in 0..rows as usize * 5120 {
                let expected_routed = if rows == 1 {
                    &one_expected
                } else {
                    &payloads[3]
                };
                let routed = f32::from_le_bytes(expected_routed[i * 4..i * 4 + 4].try_into()?);
                let shared_bits = u16::from_le_bytes(shared[lane].1[i * 2..i * 2 + 2].try_into()?);
                // Match the existing compact BF16 routed boundary, then shared addition.
                let compact = f32::from_bits((bf16(routed) as u32) << 16);
                let expected = bf16(compact + f32::from_bits((shared_bits as u32) << 16));
                let got = u16::from_le_bytes(actual[i * 2..i * 2 + 2].try_into()?);
                ensure!(got == expected, "local EXL3 differs at lane {lane}, rows {rows}, element {i}: {got:#x} != {expected:#x}");
            }
            ensure!(
                actual[rows as usize * 10240..].iter().all(|&b| b == 0xa5),
                "local output tail overwritten"
            );
        }
        // Check the FP32 expert epilogue separately from the established BF16
        // routed-output boundary in the serving reducer.
        if let Backend::Exl3 { states, .. } = &mut lanes[0].backend {
            let state = states
                .iter_mut()
                .find(|s| s.capacity() >= rows as usize)
                .unwrap();
            let values =
                unsafe { state.launch_layer(0, views, rows as usize, lanes[0].stream.raw)? };
            unsafe {
                lib.cuda_stream_synchronize(lanes[0].stream.raw)?;
            }
            let mut actual = vec![0; values.bytes];
            lib.copy_d2h(&mut actual, values)?;
            let expected = if rows == 1 {
                &one_expected
            } else {
                &payloads[3]
            };
            ensure!(
                actual == expected[..actual.len()],
                "FP32 routed epilogue differs from matching B12x tile"
            );
        }
        println!("rows={rows}: independent lanes, FP32 routed + signed BF16 shared, guards and rejection recovery passed");
    }
    println!("resident_bytes={} workspace_per_lane={workspace}; full 384 experts resident, six routed reference experts", plan.resident_bytes);
    Ok(())
}
