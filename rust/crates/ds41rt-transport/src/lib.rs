use anyhow::{bail, Result};
use ds41rt_core::{ExpertRequest, ExpertResponse};
use std::io::ErrorKind;
use std::time::Duration;

mod capabilities;
mod debug_json;
mod host_batch_set;
pub mod protocol_v2;
pub mod v41_expert;
mod protocol_v2_tcp;
mod synthetic;
mod verbs;

pub use capabilities::{
    inproc_capabilities, tcp_capabilities, verbs_host_app_transport_blocker, verbs_host_available,
    verbs_host_capabilities, verbs_host_preflight, VerbsHostPreflight,
    VERBS_HOST_APP_TRANSPORT_BLOCKER, VERBS_HOST_APP_TRANSPORT_STATUS,
    VERBS_HOST_PREFLIGHT_ONLY_PROTOCOL,
};
pub use debug_json::{
    debug_json_tcp_roundtrip, serve_synthetic_debug_json_tcp, serve_synthetic_tcp, tcp_roundtrip,
    DEBUG_JSON_EXPERT_PROTOCOL_LABEL, DEBUG_JSON_EXPERT_PROTOCOL_VERSION,
    DEBUG_JSON_FRAME_PROTOCOL,
};
#[cfg(test)]
pub(crate) use debug_json::{encode_frame, handle_synthetic_connection, read_frame, FrameKind};
pub use host_batch_set::{
    protocol_v2_stream_ingress_rows, protocol_v2_verbs_host_execution_lanes,
    tcp_protocol_v2_host_batch_set_bf16_dispatch,
    tcp_protocol_v2_host_batch_set_bf16_dispatch_with_graph_pool,
    tcp_protocol_v2_host_batch_set_bf16_payload_dispatch,
    tcp_protocol_v2_host_batch_set_bf16_payload_dispatch_with_graph_pool,
    verbs_host_protocol_v2_host_batch_set_bf16_dispatch,
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch,
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_structural_stats,
    TcpProtocolV2HostBatchSetBf16PayloadDispatch, TcpProtocolV2HostBatchSetDispatch,
    TcpProtocolV2HostBatchSetDispatchStats, TcpProtocolV2HostBatchSetPersistentClient,
    TcpProtocolV2HostBatchTarget, VerbsHostProtocolV2HostBatchSetBf16PayloadChunk,
    VerbsHostProtocolV2HostBatchSetPayloadStart, VerbsHostProtocolV2HostBatchSetPersistentClient,
    VerbsHostProtocolV2ReducedIdentityPayloadPending,
    VerbsHostProtocolV2ReducedIdentityPayloadStart, MAX_VERBS_HOST_EXECUTION_LANES,
};
pub use protocol_v2::{
    expert_protocol_v2_compact_id, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2FrameArena,
    ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request, ExpertProtocolV2RequestHeader,
    ExpertProtocolV2RequestView, ExpertProtocolV2Response, ExpertProtocolV2ResponseHeader,
    ExpertProtocolV2ResponseRef, ExpertProtocolV2ResponseView, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2RowDescriptor, ExpertProtocolV2Status, ExpertProtocolV2StreamPlan,
    ExpertProtocolV2StreamRouteGroup, ExpertProtocolV2WireStats, ExpertV2Dtype, ExpertV2SourceKind,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM, EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
    EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES,
    EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION, EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA,
    EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL, EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN,
    EXPERT_PROTOCOL_V2_FRAME_PROTOCOL, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
    EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
};
#[cfg(test)]
pub(crate) use protocol_v2_tcp::handle_protocol_v2_synthetic_connection;
pub use protocol_v2_tcp::{
    serve_protocol_v2_tcp_listener_with_executor, serve_protocol_v2_tcp_with_executor,
    serve_synthetic_protocol_v2_tcp, serve_synthetic_protocol_v2_tcp_listener,
    tcp_protocol_v2_expert_request_roundtrip, tcp_protocol_v2_roundtrip,
    tcp_protocol_v2_roundtrip_arena_response_view, tcp_protocol_v2_roundtrip_response_view,
    TcpProtocolV2PersistentClient, TcpProtocolV2PendingChunks,
};
pub use synthetic::{
    expert_response_from_protocol_v2_response, protocol_v2_echo_loopback_response,
    protocol_v2_request_from_expert_request, protocol_v2_route_dependent_synthetic_response,
    protocol_v2_synthetic_response, synthetic_expert_response, EchoExecutor,
    ProtocolV2ExecutorResponseRef, ProtocolV2ExpertExecutor, ProtocolV2RequestDevicePayload,
    SyntheticRouteExecutor, PROTOCOL_V2_ECHO_EXECUTOR, PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
    SYNTHETIC_EXPERT_KERNEL,
};
pub use verbs::{
    LocalVerbsExpertConnection,
    serve_protocol_v2_verbs_host_with_executor, serve_synthetic_verbs_host,
    verbs_host_protocol_v2_endpoint_plan, verbs_host_protocol_v2_expert_request_roundtrip,
    verbs_host_protocol_v2_handshake_contract, verbs_host_protocol_v2_round_trip_plan,
    verbs_host_protocol_v2_roundtrip, verbs_host_validate_protocol_v2_handshake,
    VerbsHostMappedRdmaPollStats, VerbsHostMappedRdmaRing, VerbsHostMappedRdmaRingConfig,
    VerbsHostMappedRdmaSlot, VerbsHostNativeEndpointDescriptor, VerbsHostProtocolV2EndpointPlan,
    VerbsHostProtocolV2HandshakeContract, VerbsHostProtocolV2HandshakeValidation,
    VerbsHostProtocolV2PendingResponse, VerbsHostProtocolV2PersistentClient,
    VerbsHostProtocolV2ResponseChunk, VerbsHostProtocolV2ResponsePayload,
    VerbsHostProtocolV2ResponseStreamStats, VerbsHostProtocolV2RoundTripPlan,
    VerbsHostRcEndpointDescriptor,
};

pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const DEFAULT_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TcpTransportConfig {
    pub timeout: Duration,
    pub max_frame_bytes: usize,
    /// Per-chunk transport diagnostics. Resolved once from the process
    /// environment at startup so the progress/poll path never calls `getenv`.
    pub timing: bool,
}

impl Default for TcpTransportConfig {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
            timing: false,
        }
    }
}

/// Protocol-v2 per-chunk transport diagnostics flag, read from the process
/// environment. Call this once at startup and thread the value (see
/// `TcpTransportConfig::timing` and the expert connection's `timing` field):
/// per-chunk progress and poll paths must never reach `getenv`.
pub fn protocol_v2_timing_from_env() -> bool {
    match std::env::var("DS41RT_PROTOCOL_V2_TCP_TIMING") {
        Ok(value) => matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

pub async fn inproc_roundtrip(request: &ExpertRequest) -> Result<ExpertResponse> {
    synthetic_expert_response(request)
}

pub async fn protocol_v2_inproc_roundtrip(
    request: &ExpertProtocolV2Request,
) -> Result<ExpertProtocolV2Response> {
    protocol_v2_synthetic_response(request)
}

pub async fn protocol_v2_inproc_expert_request_roundtrip(
    request: &ExpertRequest,
) -> Result<ExpertResponse> {
    let protocol_v2_request = protocol_v2_request_from_expert_request(request)?;
    let protocol_v2_response = protocol_v2_synthetic_response(&protocol_v2_request)?;
    expert_response_from_protocol_v2_response(request, &protocol_v2_response)
}

pub async fn protocol_v2_inproc_roundtrip_arena_response_view<'a>(
    request: &ExpertProtocolV2Request,
    arena: &'a mut ExpertProtocolV2FrameArena,
) -> Result<ExpertProtocolV2ResponseView<'a>> {
    let response = {
        let request_view = arena.encode_request_view(request)?;
        synthetic::protocol_v2_synthetic_response_from_view(&request_view)?
    };
    arena.response_buffer_mut().encode_response(&response)?;
    let view = ExpertProtocolV2ResponseView::parse(arena.response_frame())?;
    if view.header.request_id != request.header.request_id {
        bail!(
            "ProtocolV2 inproc response request_id {} did not match request_id {}",
            view.header.request_id,
            request.header.request_id
        );
    }
    if view.header.placement_version != request.header.placement_version {
        bail!(
            "ProtocolV2 inproc response placement_version {} did not match request placement_version {}",
            view.header.placement_version,
            request.header.placement_version
        );
    }
    if view.header.layer_id != request.header.layer_id {
        bail!(
            "ProtocolV2 inproc response layer_id {} did not match request layer_id {}",
            view.header.layer_id,
            request.header.layer_id
        );
    }
    Ok(view)
}

pub(crate) fn is_connection_closed(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                ErrorKind::UnexpectedEof
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
                    | ErrorKind::BrokenPipe
            )
        })
    })
}

#[cfg(test)]
mod connection_closed_tests {
    use super::is_connection_closed;
    use std::io::{Error, ErrorKind};

    #[test]
    fn connection_closed_includes_write_side_disconnects() {
        for kind in [
            ErrorKind::UnexpectedEof,
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionAborted,
            ErrorKind::BrokenPipe,
        ] {
            let error = anyhow::Error::new(Error::from(kind));
            assert!(is_connection_closed(&error), "{kind:?} should be retryable");
        }
    }
}

#[cfg(test)]
mod tests;
