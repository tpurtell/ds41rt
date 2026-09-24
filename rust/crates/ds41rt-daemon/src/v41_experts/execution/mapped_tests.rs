use super::*;
use ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
use crate::v41_experts::ExpertLayer;
use crate::v41_memory::HostAllocation;
use ds41rt_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor,
    ExpertProtocolV2RouteEntry, ExpertV2Dtype, ExpertV2SourceKind,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN};

#[test]
fn real_registered_output_matches_device_compaction_and_preserves_guards() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_MAPPED_EXPERT_LIBRARY") else { return Ok(()); };
    let lib = unsafe { NativeLibrary::load(std::path::Path::new(&path)) }?;
    let snapshot = std::env::var_os("DS41RT_MAPPED_EXPERT_MODEL").context("missing model")?;
    let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&snapshot))?;
    let weights = ExpertWeights::load(&lib, &catalog, ExpertLayer::Backbone { layer: 0, rank: 0 }, 16 << 30)?;
    let mut execution = weights.execution(4096, 16 << 30)?;
    let mut exchange = HostExpertExchange::new(4096)?;
    for rows in [1u32, 16, 80, 256, 2048, 4096] {
        let mut hidden = Vec::new();
        for _ in 0..rows { hidden.extend(vec![0x38; 5120]); hidden.extend(vec![127; 160]); }
        let mut owned = ExpertProtocolV2Request::new(91, 17, 0, 5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            (0..rows).map(|r| ExpertProtocolV2RowDescriptor { row_id: r as u64,
                source_kind: ExpertV2SourceKind::Prefill, source_request_id: 1,
                token_position: r as u64, route_offset: r * 6, route_count: 6 }).collect(),
            (0..rows * 6).map(|r| ExpertProtocolV2RouteEntry { row_index: r / 6,
                expert_id: (r / 6 + r % 6 * 63) % 384, gate_weight: 0.1234567 }).collect(), hidden)?;
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        let frame = owned.encode()?;
        let request = ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame, 4096)?;
        let bytes = request.plane_bytes()?;
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let mut host = HostAllocation::new(&lib, prefix + bytes + 64)?;
        host.bytes_mut().fill(0xa5);
        let alias = lib.cuda_host_buffer_device_alias(host.buffer)?;
        let short = Ds41rtDeviceBuffer { bytes: prefix + bytes - 1, ..alias };
        assert!(unsafe { execution.execute_mapped_request(&request, 1, &mut exchange, short, None)? }.is_none());
        assert!(host.bytes_mut().iter().all(|&b| b == 0xa5));
        let response = unsafe { execution.execute_mapped_request(&request, 1, &mut exchange, alias, None)? }.unwrap();
        assert_eq!(response.partial_output_payload.bytes, bytes);
        assert!(host.bytes_mut()[..prefix].iter().all(|&b| b == 0xa5));
        assert!(host.bytes_mut()[prefix + bytes..].iter().all(|&b| b == 0xa5));
        // Compact the identical FP32 result to device storage: no second atomic
        // expert execution can change summation order in this exact comparison.
        let device = execution.compact_output.as_ref().unwrap().buffer;
        let reducer = execution.compact_reducer.as_ref().unwrap();
        let (kernel, slots, _) = execution.execution_state(rows);
        unsafe {
            if kernel.accumulates_tokens() {
                reducer.compact_tokens(slots[41].cast(), device.ptr.cast(), rows, execution.stream.raw)?;
            } else {
                reducer.compact(execution.route_partials(rows)?.ptr.cast(), device.ptr.cast(), rows, execution.stream.raw)?;
            }
        }
        execution.synchronize()?;
        let mut expected = vec![0; bytes];
        lib.copy_d2h(&mut expected, Ds41rtDeviceBuffer { bytes, ..device })?;
        assert_eq!(&host.bytes_mut()[prefix..prefix + bytes], expected.as_slice());
        // Exercise the refactored host fallback too. Compare its downloaded
        // bytes with its own device result, avoiding a second-execution atomic comparison.
        let fallback = execution.execute_host_request(&request, 1, &mut exchange)?;
        assert_eq!(fallback.header, response.header);
        let fallback_bytes = fallback.partial_output_payload.to_vec();
        lib.copy_d2h(&mut expected, Ds41rtDeviceBuffer { bytes, ..device })?;
        assert_eq!(fallback_bytes, expected);
        assert!(expected.iter().any(|&b| b != 0), "fixture must exercise nonzero outputs");
        owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        let frame = owned.encode()?;
        let request = ds41rt_transport::v41_expert::V41BackboneRequest::parse(&frame, 4096)?;
        assert!(unsafe { execution.execute_mapped_request(&request, 1, &mut exchange, alias, None)? }.is_none());
        eprintln!("PASS registered output rows={rows}: exact compaction, guards and fallback");
    }
    Ok(())
}
