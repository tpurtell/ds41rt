use super::*;
use crate::{v41_experts::ExpertLayer, v41_memory::HostAllocation};
use ds41rt_transport::{
    v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, ExpertProtocolV2Request,
    ExpertProtocolV2ResponseView, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
    ExpertV2SourceKind, EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
};
use sha2::{Digest, Sha256};

fn reference(fixture: &Path, aot: &Path, capacity: usize) -> Result<Vec<Vec<u8>>> {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
    let export: serde_json::Value = serde_json::from_slice(&std::fs::read(
        aot.join(format!("m{capacity}/v41_exl3.json")),
    )?)?;
    ensure!(
        meta["canonical_routes"] == true
            && meta["input_format"] == "fp8_k32"
            && meta["output_dtype"] == "bf16"
            && meta["layer"] == "layers.0"
            && meta["slice_start"] == 1280
            && meta["width"] == 512
            && meta["capacity"] == capacity,
        "unexpected worker fixture"
    );
    for key in ["tile", "direct", "output_dtype"] {
        ensure!(meta[key] == export[key], "fixture/export policy mismatch");
    }
    let mut payloads = Vec::new();
    for name in ["input", "ids", "weights", "expected"] {
        let bytes = std::fs::read(fixture.join(format!("{name}.bin")))?;
        ensure!(
            Some(bytes.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
                    == meta["artifacts"][name]["sha256"].as_str(),
            "corrupt worker fixture"
        );
        payloads.push(bytes);
    }
    Ok(payloads)
}

#[test]
fn exl3_worker_capacity_bounds() -> Result<()> {
    assert_eq!(Exl3Worker::capacities(1)?, vec![1]);
    assert_eq!(Exl3Worker::capacities(16)?, vec![1, 16]);
    assert_eq!(Exl3Worker::capacities(40)?, vec![1, 16, 80]);
    assert_eq!(
        Exl3Worker::capacities(4096)?,
        vec![1, 16, 80, 256, 1024, 4096]
    );
    assert!(Exl3Worker::capacities(0).is_err());
    assert!(Exl3Worker::capacities(4097).is_err());
    Ok(())
}

#[test]
#[ignore = "requires CUDA, native EXL3 wire decoder, full snapshot, AOT and canonical wire fixture"]
fn exl3_worker_mapped_and_chunked_responses_match_reference() -> Result<()> {
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        Path::new(&std::env::var("DS41RT_EXL3_SNAPSHOT")?),
    )?;
    let aot = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
    let fixture = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_FIXTURE")?);
    let layer = ExpertLayer::Backbone { layer: 0, rank: 2 };
    let budget = Exl3Weights::plan(&catalog, layer)?;
    let (free, _) = lib.cuda_memory_info()?;
    ensure!(
        free > budget.resident_bytes + Exl3Worker::plan(&aot, 4096)? + (256 << 20),
        "insufficient GPU headroom"
    );
    let weights = Rc::new(vec![Exl3Weights::load(
        &lib,
        &catalog,
        layer,
        budget.resident_bytes,
    )?]);
    let workspace = Exl3Worker::plan(&aot, 4096)?;
    let mut worker = Exl3Worker::new(&lib, weights, &aot, 4096, workspace)?;
    let mut exchange = HostExpertExchange::new(4096)?;
    let mut row_indices = vec![0; 4096];
    assert!(worker.bind_layer(1).is_err());
    worker.bind_layer(0)?;
    assert_eq!(
        worker
            .executions
            .iter()
            .map(|e| e.capacity())
            .collect::<Vec<_>>(),
        vec![1, 16, 80, 256, 1024, 4096]
    );
    for rows in [1u32, 3, 16, 17, 80, 81, 256, 257, 1024, 1025, 4096, 1] {
        let capacity = [1usize, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&c| c >= rows as usize)
            .unwrap();
        let payloads = reference(&fixture.join(format!("m{capacity}")), &aot, capacity)?;
        let mut owned = ExpertProtocolV2Request::new(
            91,
            17,
            0,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows)
                .map(|r| ExpertProtocolV2RowDescriptor {
                    row_id: r as u64,
                    source_kind: ExpertV2SourceKind::Prefill,
                    source_request_id: 1,
                    token_position: r as u64,
                    route_offset: r * 6,
                    route_count: 6,
                })
                .collect(),
            (0..rows as usize * 6)
                .map(|r| ExpertProtocolV2RouteEntry {
                    row_index: r as u32 / 6,
                    expert_id: u32::from_le_bytes(
                        payloads[1][r * 4..r * 4 + 4].try_into().unwrap(),
                    ),
                    gate_weight: f32::from_le_bytes(
                        payloads[2][r * 4..r * 4 + 4].try_into().unwrap(),
                    ),
                })
                .collect(),
            payloads[0][..rows as usize * 5280].to_vec(),
        )?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let frame = owned.encode()?;
        let request = V41BackboneRequest::parse(&frame, 4096)?;
        let bytes = request.plane_bytes()?;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut host = HostAllocation::new(&lib, prefix + bytes + 64)?;
        host.bytes_mut().fill(0xa5);
        let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;
        let short = Ds41rtDeviceBuffer {
            bytes: prefix + bytes - 1,
            ..alias
        };
        assert!(
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, short)? }.is_none()
        );
        assert!(host.bytes_mut().iter().all(|&b| b == 0xa5));
        assert!(
            unsafe { worker.execute_mapped_request(&request, 2, &mut exchange, alias) }.is_err()
        );
        assert!(host.bytes_mut().iter().all(|&b| b == 0xa5));
        let response =
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, alias)? }.unwrap();
        assert_eq!(response.header.executor_id, 3);
        assert_eq!(response.partial_output_payload.bytes, bytes);
        assert!(host.bytes_mut()[..prefix].iter().all(|&b| b == 0xa5));
        assert!(host.bytes_mut()[prefix + bytes..]
            .iter()
            .all(|&b| b == 0xa5));
        assert_eq!(
            &host.bytes_mut()[prefix..prefix + bytes],
            &payloads[3][..bytes]
        );
        // Debug-checksum requests must use host encoding; force multiple frames.
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        let frame = owned.encode()?;
        let request = V41BackboneRequest::parse(&frame, 4096)?;
        assert!(
            unsafe { worker.execute_mapped_request(&request, 3, &mut exchange, alias)? }.is_none()
        );
        let max_frame =
            ds41rt_transport::EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN + 3 * (10240 + 4);
        let mut actual = Vec::new();
        let mut chunks = 0;
        worker.execute_host_chunks(
            &request,
            3,
            &mut exchange,
            &mut row_indices,
            max_frame,
            |response| {
                let encoded = response.to_owned()?.encode()?;
                assert!(encoded.len() <= max_frame);
                let parsed = ExpertProtocolV2ResponseView::parse(&encoded)?;
                if let Some(indices) = response.row_indices {
                    assert_eq!(indices[0] as usize, actual.len() / 10240);
                }
                actual.extend_from_slice(parsed.partial_output_payload());
                chunks += 1;
                Ok(())
            },
        )?;
        assert_eq!(actual, payloads[3][..bytes]);
        assert_eq!(chunks, rows.div_ceil(3));
        // Sink failure occurs after GPU completion and does not poison later work.
        assert!(worker
            .execute_host_chunks(
                &request,
                3,
                &mut exchange,
                &mut row_indices,
                max_frame,
                |_| anyhow::bail!("intentional send failure")
            )
            .is_err());
        owned.header.layer_id = 1;
        let bad_frame = owned.encode()?;
        let bad = V41BackboneRequest::parse(&bad_frame, 4096)?;
        assert!(worker
            .execute_host_chunks(&bad, 3, &mut exchange, &mut row_indices, max_frame, |_| Ok(
                ()
            ))
            .is_err());
        println!("PASS EXL3 worker maximum=4096 selected_capacity={capacity} rows={rows}: native mapped BF16 output equals B12x, guards, chunked checksum fallback, bounds/identity/layer rejection and sink-failure recovery");
    }
    println!("EXL3 worker workspace payload: {workspace} bytes; 384 resident experts, six routed IDs, TP4 rank 2; host architecture {}", std::env::consts::ARCH);
    Ok(())
}
