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

#[test]
fn paired_worker_loading_rejects_wrong_rank_and_mixed_capacities() -> Result<()> {
    use ds41rt_loader::V41Exl3Partition;
    let root = tempfile::tempdir()?;
    let original = serde_json::json!({
        "schema":"ds41rt.v41-exl3-aot.v1", "output_dtype":"bf16", "sparkinfer_revision":"test",
        "hidden":5120,"intermediate":640,"experts":384,"capacity":80,"top_k":6,
        "bits":[3,4],"swiglu_limit":10.0,"direct":false,"sms":48,"blocks_per_sm":1,
        "buffers":{},"objects":[],"trellis_lut":{"file":"lut","bytes":16,"sha256":"test"}
    });
    let write = |capacity, value: &serde_json::Value| -> Result<()> {
        let directory = root.path().join(format!("m{capacity}"));
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(value)?)?;
        Ok(())
    };
    for c in [1, 16, 80] { write(c, &original)?; }
    for rank in 0..4 {
        assert_eq!(Exl3Worker::partition(root.path(), 80, rank)?, V41Exl3Partition::Disjoint);
    }
    for (boundary, ranks) in [("last", [0, 2]), ("first", [1, 3])] {
        let mut paired = original.clone();
        paired["paired_boundary"] = boundary.into();
        paired["descriptor_rows"] = 4.into();
        paired["native_info_version"] = 3.into();
        for c in [1, 16, 80] { write(c, &paired)?; }
        for rank in ranks {
            assert_eq!(Exl3Worker::partition(root.path(), 80, rank)?, V41Exl3Partition::PairedTp4);
            assert!(Exl3Worker::partition(root.path(), 80, rank ^ 1).is_err());
        }
        write(80, &original)?;
        assert!(Exl3Worker::partition(root.path(), 80, ranks[0]).is_err());
        assert_eq!(Exl3Worker::partition(root.path(), 16, ranks[0])?, V41Exl3Partition::PairedTp4);
    }
    assert!(Exl3Worker::partition(root.path(), 80, 4).is_err());
    Ok(())
}

#[test]
#[ignore = "requires paired Spark package, paired reference fixtures, snapshot and CUDA"]
fn paired_worker_mapped_and_chunked_match_reference() -> Result<()> {
    use ds41rt_loader::V41Exl3Partition;
    use ds41rt_transport::v41_expert::{V41PairedRouteWord, V41_EXL3_PAIRED_REQUEST_FLAG};
    let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
    lib.cuda_set_device(0)?;
    let snapshot = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_SNAPSHOT")?);
    let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &snapshot)?;
    let package = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_PAIRED_PACKAGE")?);
    let fixtures = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_PAIRED_FIXTURES")?);
    for (rank, boundary) in [(0, "last"), (1, "first")] {
        let fixture = fixtures.join(format!("paired-fixture-{boundary}"));
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.join("fixture.json"))?)?;
        ensure!(meta["layer"] == "layers.30" && meta["paired_rank"] == rank
            && meta["paired_boundary"] == boundary && meta["capacity"] == 80
            && meta["input_format"] == "fp8_k32" && meta["canonical_routes"] == true
            && meta["snapshot_revision"].as_str() == snapshot.file_name().and_then(|v| v.to_str()), "paired fixture identity mismatch");
        let read = |name: &str| -> Result<Vec<u8>> {
            let bytes = std::fs::read(fixture.join(format!("{name}.bin")))?;
            ensure!(Some(bytes.len() as u64) == meta["artifacts"][name]["bytes"].as_u64()
                && Some(format!("{:x}", Sha256::digest(&bytes)).as_str()) == meta["artifacts"][name]["sha256"].as_str(), "paired fixture checksum mismatch");
            Ok(bytes)
        };
        let input = read("input")?; let ids = read("ids")?; let routing = read("weights")?;
        let owners = read("owners")?; let expected = read("expected")?;
        let aot = package.join(format!("tp4-rank{rank}"));
        assert_eq!(Exl3Worker::partition(&aot, 80, rank)?, V41Exl3Partition::PairedTp4);
        let layer = ExpertLayer::Backbone { layer:30, rank };
        let budget = Exl3Weights::plan_with_layout(&catalog, layer, V41Exl3Partition::PairedTp4)?;
        let weights = Rc::new(vec![Exl3Weights::load_with_layout(&lib, &catalog, layer,
            budget.resident_bytes, V41Exl3Partition::PairedTp4)?]);
        let mut worker = Exl3Worker::new(&lib, weights, &aot, 80, Exl3Worker::plan(&aot, 80)?)?;
        let mut exchange = HostExpertExchange::new(80)?;
        let mut request = ExpertProtocolV2Request::new(91,17,30,5120,ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..80).map(|r| ExpertProtocolV2RowDescriptor { row_id:r as u64, source_kind:ExpertV2SourceKind::Prefill,
                source_request_id:1,token_position:r as u64,route_offset:r*6,route_count:6 }).collect(),
            (0..480).map(|r| {
                let id = u32::from_le_bytes(ids[r*4..r*4+4].try_into().unwrap());
                let owner = u32::from_le_bytes(owners[id as usize*4..id as usize*4+4].try_into().unwrap());
                Ok(ExpertProtocolV2RouteEntry { row_index:r as u32/6,
                    expert_id:V41PairedRouteWord { expert_id:id, owners:u8::try_from(owner)? }.encode()?,
                    gate_weight:f32::from_le_bytes(routing[r*4..r*4+4].try_into().unwrap()) })
            }).collect::<Result<Vec<_>>>()?, input)?;
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 | V41_EXL3_PAIRED_REQUEST_FLAG;
        let frame = request.encode()?;
        let parsed = V41BackboneRequest::parse_paired(&frame,80)?;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut host = HostAllocation::new(&lib,prefix+expected.len()+64)?;
        host.bytes_mut().fill(0xa5);
        let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;
        let response = unsafe { worker.execute_mapped_request(&parsed,rank as u64+1,&mut exchange,alias)? }.context("paired mapped response absent")?;
        assert_eq!(response.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG,0);
        assert_eq!(&host.bytes_mut()[prefix..prefix+expected.len()], expected.as_slice());
        assert!(host.bytes_mut()[..prefix].iter().all(|&v| v==0xa5));
        assert!(host.bytes_mut()[prefix+expected.len()..].iter().all(|&v| v==0xa5));
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        let frame = request.encode()?;
        let parsed = V41BackboneRequest::parse_paired(&frame,80)?;
        let mut indices = [0;3]; let mut actual = Vec::new();
        worker.execute_host_chunks(&parsed,rank as u64+1,&mut exchange,&mut indices,
            ds41rt_transport::EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN+3*(10240+4), |response| {
                response.validate()?;
                assert_eq!(response.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG,0);
                actual.extend_from_slice(response.partial_output_payload); Ok(())
            })?;
        assert_eq!(actual,expected);
        eprintln!("paired rank {rank} ({boundary}): mapped and chunked outputs match reference exactly");
    }
    Ok(())
}
