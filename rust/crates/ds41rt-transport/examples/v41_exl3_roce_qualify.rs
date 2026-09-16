//! Real four-Spark expert responses against hashed B12x rank fixtures.
//! <fixture-root> <peer0> <peer1> <peer2> <peer3>
use anyhow::{ensure, Context, Result};
use ds41rt_transport::{v41_expert::*, *};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Duration};

struct Fixture {
    data: Vec<Vec<u8>>,
}
fn fixture(root: &Path, rank: usize, capacity: usize) -> Result<Fixture> {
    let dir = root.join(format!("rank{rank}/m{capacity}"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("fixture.json"))?)?;
    ensure!(
        meta["layer"] == "layers.39"
            && meta["capacity"] == capacity
            && meta["topk"] == 6
            && meta["canonical_routes"] == true
            && meta["input_format"] == "fp8_k32"
            && meta["output_dtype"] == "bf16"
            && meta["slice_start"] == [0, 640, 1280, 1792][rank]
            && meta["width"] == [640, 640, 512, 512][rank]
            && meta["snapshot_revision"] == "cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88",
        "fixture geometry/identity mismatch"
    );
    let mut data = Vec::new();
    for (name, stride) in [
        ("input", 5280),
        ("ids", 24),
        ("weights", 24),
        ("expected", 10240),
    ] {
        let bytes = std::fs::read(dir.join(format!("{name}.bin")))?;
        ensure!(
            bytes.len() == capacity * stride
                && meta["artifacts"][name]["bytes"].as_u64() == Some(bytes.len() as u64)
                && meta["artifacts"][name]["sha256"].as_str()
                    == Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "corrupt fixture {name}"
        );
        data.push(bytes);
    }
    ensure!(
        data[3].iter().any(|&b| b != 0)
            && data[3]
                .chunks_exact(2)
                .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80),
        "invalid reference output"
    );
    Ok(Fixture { data })
}
fn request(f: &Fixture, rows: u32, id: u64, reverse: bool) -> Result<ExpertProtocolV2Request> {
    let order: Vec<_> = (0..rows as usize)
        .map(|r| if reverse { rows as usize - 1 - r } else { r })
        .collect();
    let hidden = order
        .iter()
        .flat_map(|&r| f.data[0][r * 5280..(r + 1) * 5280].iter().copied())
        .collect();
    let mut routes = Vec::new();
    for (row, &source) in order.iter().enumerate() {
        for slot in 0..6 {
            let at = (source * 6 + slot) * 4;
            routes.push(ExpertProtocolV2RouteEntry {
                row_index: row as u32,
                expert_id: u32::from_le_bytes(f.data[1][at..at + 4].try_into().unwrap()),
                gate_weight: f32::from_le_bytes(f.data[2][at..at + 4].try_into().unwrap()),
            });
        }
    }
    let mut r = ExpertProtocolV2Request::new(
        id,
        17,
        39,
        5120,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32,
        (0..rows)
            .map(|row| ExpertProtocolV2RowDescriptor {
                row_id: row as u64,
                source_kind: if rows <= 16 {
                    ExpertV2SourceKind::Decode
                } else {
                    ExpertV2SourceKind::Prefill
                },
                source_request_id: 100 + row as u64,
                token_position: 7,
                route_offset: row * 6,
                route_count: 6,
            })
            .collect(),
        routes,
        hidden,
    )?;
    r.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(r)
}
async fn run_wave(
    tp: &mut V41Tp4Roce,
    req: &ExpertProtocolV2Request,
    refs: &[Fixture],
    reverse: bool,
    check_retained: bool,
) -> Result<()> {
    let rows = req.rows.len();
    let mut seen = vec![false; rows * 4];
    let mut held = Vec::new();
    tp.dispatch(req)
        .await?
        .receive_owned(|rank, start, payload| {
            let bytes = payload.as_ref();
            ensure!(bytes.len() % 10240 == 0, "invalid response extent");
            for (offset, row) in bytes.chunks_exact(10240).enumerate() {
                let dest = start as usize + offset;
                ensure!(
                    dest < rows && !seen[rank * rows + dest],
                    "duplicate/out-of-range response"
                );
                let source = if reverse { rows - 1 - dest } else { dest };
                ensure!(
                    row == &refs[rank].data[3][source * 10240..(source + 1) * 10240],
                    "rank {rank} row {dest} differs from B12x"
                );
                seen[rank * rows + dest] = true;
            }
            held.push(payload);
            Ok(())
        })
        .await?;
    ensure!(seen.iter().all(|v| *v), "missing response rows");
    ensure!(
        held.iter().any(|p| p.retains_receive_slot()),
        "expected retained receive slots"
    );
    if check_retained {
        let hashes: Vec<_> = held.iter().map(|p| Sha256::digest(p.as_ref())).collect();
        ensure!(
            tp.dispatch(req).await.is_err(),
            "retained response was overwritten"
        );
        for (payload, hash) in held.iter().zip(hashes) {
            ensure!(
                Sha256::digest(payload.as_ref()) == hash,
                "retained payload changed during reset"
            );
        }
    }
    drop(held);
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 6, "fixture-root and four peers required");
    let peers = args[2..]
        .iter()
        .map(|v| v.parse())
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();
    let make = || {
        V41Tp4Roce::new(
            peers,
            [1, 2, 3, 4],
            80,
            TcpTransportConfig {
                timeout: Duration::from_secs(30),
                max_frame_bytes: 2 * 1024 * 1024,
            },
        )
    };
    let mut lanes = [make()?, make()?];
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        for (cycle,rows) in [1u32,3,16,17,80,1].into_iter().enumerate() {
            let capacity = if rows==1 {1} else if rows<=16 {16} else {80};
            let refs = (0..4).map(|rank| fixture(Path::new(&args[1]),rank,capacity)).collect::<Result<Vec<_>>>()?;
            for rank in 1..4 { ensure!(refs[rank].data[..3]==refs[0].data[..3], "rank inputs differ"); }
            let requests = [request(&refs[0],rows,cycle as u64*2+1,false)?,request(&refs[0],rows,cycle as u64*2+2,true)?];
            let [first,second] = &mut lanes;
            let (a,b) = tokio::join!(run_wave(first,&requests[0],&refs,false,cycle==1),run_wave(second,&requests[1],&refs,true,cycle==1));
            a.context("lane 0")?;b.context("lane 1")?;
            if cycle == 2 { drop(lanes[0].dispatch(&requests[0]).await?); }
            eprintln!("PASS four-Spark EXL3 rows={rows} capacity={capacity}: both lanes, exact rank responses, retained-slot protection and reconnect/recovery");
        }
        Ok(())
    })
}
