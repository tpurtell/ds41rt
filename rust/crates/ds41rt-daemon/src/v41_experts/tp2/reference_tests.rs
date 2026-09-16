use super::*;
use ds41rt_ffi::NativeLibrary;
use sha2::{Digest, Sha256};

struct Reference {
    inputs: Vec<Vec<u8>>,
    expected: Vec<u8>,
}
fn read_reference(
    directory: &Path,
    package: &Path,
    rank: usize,
    capacity: usize,
    revision: &str,
) -> Result<Reference> {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("fixture.json"))?)?;
    ensure!(
        meta["layer"] == "layers.0"
            && meta["slice_start"] == rank * 1152
            && meta["width"] == 1152
            && meta["capacity"] == capacity
            && meta["topk"] == 6
            && meta["canonical_routes"] == true
            && meta["input_format"] == "fp8_k32"
            && meta["output_dtype"] == "fp32"
            && meta["snapshot_revision"] == revision,
        "unexpected TP2 reference geometry or identity"
    );
    let export: serde_json::Value = serde_json::from_slice(&std::fs::read(
        package.join(format!("m{capacity}/v41_exl3.json")),
    )?)?;
    for key in ["tile", "direct", "output_dtype"] {
        ensure!(
            meta[key] == export[key],
            "TP2 reference/kernel policy mismatch"
        );
    }
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for (name, width) in [
        ("input", 5280),
        ("ids", 24),
        ("weights", 24),
        ("expected", 5120 * 4),
    ] {
        let data = std::fs::read(directory.join(format!("{name}.bin")))?;
        ensure!(
            data.len() == capacity * width
                && Some(data.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&data)).as_str())
                    == meta["artifacts"][name]["sha256"].as_str(),
            "corrupt TP2 reference payload"
        );
        if name == "expected" {
            expected = data;
        } else {
            inputs.push(data);
        }
    }
    Ok(Reference { inputs, expected })
}
fn rounded_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

#[test]
#[ignore = "requires two RTX GPUs, EXL3 snapshot, TP2 package and matched rank references"]
fn exl3_tp2_rank_outputs_and_peer_sums_match_b12x() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
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
    let snapshot = PathBuf::from(std::env::var("DS41RT_SNAPSHOT")?);
    let package = PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
    let fixture = PathBuf::from(std::env::var("DS41RT_EXL3_FIXTURE")?);
    let revision = snapshot.file_name().unwrap().to_str().unwrap();
    let references = (0..2)
        .map(|rank| {
            [1, 16]
                .into_iter()
                .map(|capacity| {
                    read_reference(
                        &fixture.join(format!("rank{rank}/m{capacity}")),
                        &package,
                        rank,
                        capacity,
                        revision,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let baseline = &references[0][1].inputs;
    for rank in &references {
        for reference in rank {
            for (base, input) in baseline.iter().zip(&reference.inputs) {
                ensure!(
                    base.starts_with(input),
                    "TP2 references did not use identical inputs/routes"
                );
            }
        }
    }
    let catalog =
        ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let weights = RankWeights::load_exl3_pair(
        devices, &catalog, 1, [4_000_000_000; 2], &package,
    )?.map(Rc::new);
    let inputs = (0..4)
        .map(|index| -> Result<_> {
            let device = devices[index % 2];
            Ok((
                [
                    Allocation::new(device, 16 * 5280)?,
                    Allocation::new(device, 16 * 24)?,
                    Allocation::new(device, 16 * 24)?,
                ],
                Stream::new(device)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    // Wave owners drain before releasing any of the inputs above.
    let mut first = ExpertWave::new(weights.clone(), 16)?;
    let mut second = ExpertWave::new(weights, 16)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    for (step, rows) in [16usize, 1, 3, 16, 1].into_iter().enumerate() {
        let row_order: [Vec<usize>; 2] = std::array::from_fn(|lane| {
            (0..rows)
                .map(|r| {
                    if lane == 0 {
                        (r + step) % rows
                    } else {
                        rows - 1 - r
                    }
                })
                .collect()
        });
        for lane in 0..2 {
            for rank in 0..2 {
                for (field, width) in [5280, 24, 24].into_iter().enumerate() {
                    let mut data = vec![0xa5; 16 * width];
                    for (r, &source) in row_order[lane].iter().enumerate() {
                        data[r * width..(r + 1) * width].copy_from_slice(
                            &baseline[field][source * width..(source + 1) * width],
                        );
                    }
                    devices[rank]
                        .run(|| lib.copy_h2d(inputs[lane * 2 + rank].0[field].buffer, &data))?;
                }
            }
        }
        let make_inputs = |lane: usize| -> [RankInputs<'_, '_>; 2] {
            std::array::from_fn(|rank| {
                let (buffers, stream) = &inputs[lane * 2 + rank];
                RankInputs {
                    wire: Ds41rtDeviceBuffer {
                        bytes: rows * 5280,
                        ..buffers[0].buffer
                    },
                    ids: Ds41rtDeviceBuffer {
                        bytes: rows * 24,
                        ..buffers[1].buffer
                    },
                    routing: Ds41rtDeviceBuffer {
                        bytes: rows * 24,
                        ..buffers[2].buffer
                    },
                    producer: stream,
                }
            })
        };
        let (x, y) = runtime.block_on(async {
            tokio::join!(
                unsafe { first.execute(0, rows as u32, 0, make_inputs(0)) },
                unsafe { second.execute(0, rows as u32, 1, make_inputs(1)) }
            )
        });
        for (lane, (wave, output)) in [(&first, x?), (&second, y?)].into_iter().enumerate() {
            let mut expected_ranks = Vec::new();
            for rank in 0..2 {
                let reference = &references[rank][usize::from(rows > 1)].expected;
                let expected: Vec<u8> = row_order[lane]
                    .iter()
                    .flat_map(|&source| {
                        reference[source * 5120 * 4..(source + 1) * 5120 * 4]
                            .iter()
                            .copied()
                    })
                    .collect();
                let mut actual = vec![0; rows * 5120 * 4];
                let view = Ds41rtDeviceBuffer {
                    bytes: actual.len(),
                    ..wave.ranks[rank].output.buffer
                };
                devices[rank].run(|| lib.copy_d2h(&mut actual, view))?;
                ensure!(
                    actual == expected,
                    "TP2 FP32 rank {rank} lane {lane} rows {rows} differs from B12x"
                );
                expected_ranks.push(expected);
            }
            let expected: Vec<u8> = expected_ranks[0]
                .chunks_exact(4)
                .zip(expected_ranks[1].chunks_exact(4))
                .flat_map(|(a, b)| {
                    rounded_bf16(
                        f32::from_le_bytes(a.try_into().unwrap())
                            + f32::from_le_bytes(b.try_into().unwrap()),
                    )
                    .to_le_bytes()
                })
                .collect();
            let mut actual = vec![0; output.bytes];
            devices[lane].run(|| lib.copy_d2h(&mut actual, output))?;
            ensure!(
                actual == expected,
                "TP2 peer reduction lane {lane} rows {rows} differs from FP32-sum/BF16 reference"
            );
        }
        ensure!(
            lib.cuda_get_device()? == 0,
            "TP2 reference check leaked device scope"
        );
        println!("rows={rows} step={step}: both FP32 ranks and opposite BF16 peer sums bitwise equal; distinct lane row orders");
    }
    Ok(())
}
