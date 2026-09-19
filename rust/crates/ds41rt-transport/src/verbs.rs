mod registered_response;
use registered_response::{RegisteredResponseFrame, RegisteredResponseRing};
mod local;
mod local_client;
pub(crate) use local_client::LocalTp4Client;
pub use local::LocalVerbsExpertConnection;
use anyhow::{bail, Context, Result};
use ds41rt_core::{ExpertRequest, ExpertResponse};
use ds41rt_ffi::{
    c_char_array_to_string, Ds41rtDeviceBuffer, Ds41rtHostBuffer, Ds41rtRdmaRcCompletionStats,
    Ds41rtRdmaRcEndpointBufferView, Ds41rtRdmaRcEndpointInfo, NativeLibrary,
    DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST, DS41RT_HOST_BUFFER_FLAG_MAPPED,
    DS41RT_HOST_BUFFER_FLAG_PINNED,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::env;
use std::ffi::c_void;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::synthetic::{
    expert_response_from_protocol_v2_response, protocol_v2_request_from_expert_request,
    ProtocolV2ExecutorResponseRef, ProtocolV2ExpertExecutor, ProtocolV2RequestDevicePayload,
    SyntheticRouteExecutor,
};
use crate::{
    is_connection_closed, verbs_host_preflight, ExpertProtocolV2FrameBuffer,
    ExpertProtocolV2Request, ExpertProtocolV2RequestView, ExpertProtocolV2Response,
    ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseView, ExpertProtocolV2Status,
    ExpertV2Dtype, TcpTransportConfig, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};

const VERBS_HOST_RECV_WR_ID: u64 = 0x7256_1001;
const VERBS_HOST_SEND_WR_ID: u64 = 0x7256_1002;
const VERBS_HOST_RDMA_RING_DEPTH: usize = 8;
const VERBS_HOST_MAPPED_RDMA_RING_MAX_DEPTH: usize = 32;
const VERBS_HOST_RDMA_RING_SLOT_BYTES: usize = 8 * 1024 * 1024;
const VERBS_HOST_LANE_FANOUT_COMMAND_SPIN: Duration = Duration::from_millis(1);
const PROTOCOL_V2_TCP_TIMING_ENV: &str = "DS41RT_PROTOCOL_V2_TCP_TIMING";
const VERBS_HOST_RDMA_DEVICE_MAP_ENV: &str = "DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP";
static VERBS_HOST_PSN_COUNTER: AtomicU32 = AtomicU32::new(1);

fn parse_verbs_host_rdma_device_map(raw: &str, local_ip: IpAddr) -> Result<Option<String>> {
    let mut selected = None;
    let mut configured_ips = Vec::new();
    for entry in raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let (raw_ip, raw_device) = entry.split_once('=').with_context(|| {
            format!(
                "invalid {VERBS_HOST_RDMA_DEVICE_MAP_ENV} entry {entry:?}; expected local-ip=device"
            )
        })?;
        let ip = raw_ip.trim().parse::<IpAddr>().with_context(|| {
            format!("invalid local IP in {VERBS_HOST_RDMA_DEVICE_MAP_ENV} entry {entry:?}")
        })?;
        let device = raw_device.trim();
        anyhow::ensure!(
            !device.is_empty(),
            "empty RDMA device in {VERBS_HOST_RDMA_DEVICE_MAP_ENV} entry {entry:?}"
        );
        anyhow::ensure!(
            !configured_ips.contains(&ip),
            "duplicate local IP {ip} in {VERBS_HOST_RDMA_DEVICE_MAP_ENV}"
        );
        configured_ips.push(ip);
        if ip == local_ip {
            selected = Some(device.to_owned());
        }
    }
    Ok(selected)
}

fn verbs_host_rdma_device_for_stream(stream: &TcpStream) -> Result<Option<String>> {
    let Ok(raw) = env::var(VERBS_HOST_RDMA_DEVICE_MAP_ENV) else {
        return Ok(None);
    };
    let local_addr = stream
        .local_addr()
        .context("reading verbs-host control connection local address")?;
    parse_verbs_host_rdma_device_map(&raw, local_addr.ip())
}

struct VerbsHostCudaStream {
    library: Arc<NativeLibrary>,
    raw: *mut c_void,
}

impl VerbsHostCudaStream {
    fn create(library: Arc<NativeLibrary>) -> Result<Self> {
        let raw = library.cuda_stream_create()?;
        Ok(Self { library, raw })
    }
}

impl Drop for VerbsHostCudaStream {
    fn drop(&mut self) {
        let _ = unsafe { self.library.cuda_stream_destroy(self.raw) };
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerbsHostProtocolV2EndpointPlan {
    pub protocol: String,
    pub data_plane: String,
    pub control_plane: String,
    pub memory: String,
    pub polling: String,
    pub request_frame_bytes: usize,
    pub response_frame_bytes: usize,
    pub request_logical_payload_bytes: usize,
    pub response_logical_payload_bytes: usize,
    pub request_rows: usize,
    pub response_rows: usize,
    pub registration_alignment_bytes: usize,
    pub request_registered_span_bytes: usize,
    pub response_registered_span_bytes: usize,
    pub total_registered_span_bytes: usize,
    pub request_registration_slack_bytes: usize,
    pub response_registration_slack_bytes: usize,
    pub request_registered_span_aligned: bool,
    pub response_registered_span_aligned: bool,
    pub queue_pairs_per_peer: usize,
    pub send_work_requests_per_roundtrip: usize,
    pub recv_work_requests_per_roundtrip: usize,
    pub scatter_gather_entries_per_message: usize,
    pub requires_peer_qp_num: bool,
    pub requires_peer_psn: bool,
    pub requires_peer_gid: bool,
    pub requires_registered_host_buffers: bool,
    pub app_transport_implemented: bool,
    pub app_transport_blocker: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerbsHostProtocolV2HandshakeContract {
    pub protocol: String,
    pub control_plane: String,
    pub descriptor_fields: Vec<String>,
    pub client_role: String,
    pub server_role: String,
    pub client_send_frame_bytes: usize,
    pub client_recv_frame_bytes: usize,
    pub server_send_frame_bytes: usize,
    pub server_recv_frame_bytes: usize,
    pub client_send_registered_span_bytes: usize,
    pub client_recv_registered_span_bytes: usize,
    pub server_send_registered_span_bytes: usize,
    pub server_recv_registered_span_bytes: usize,
    pub requires_peer_qp_num: bool,
    pub requires_peer_psn: bool,
    pub requires_peer_gid: bool,
    pub requires_registered_host_buffers: bool,
    pub descriptor_validation_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerbsHostRcEndpointDescriptor {
    pub role: String,
    pub host: String,
    pub port_num: u32,
    pub qp_num: u32,
    pub psn: u32,
    pub gid_hex: String,
    pub send_frame_bytes: usize,
    pub recv_frame_bytes: usize,
    pub send_registered_span_bytes: usize,
    pub recv_registered_span_bytes: usize,
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_sge: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerbsHostNativeEndpointDescriptor {
    pub port_num: u32,
    pub qp_num: u32,
    pub psn: u32,
    pub lid: u32,
    pub active_mtu: u32,
    pub gid_hex: String,
    pub send_frame_bytes: usize,
    pub recv_frame_bytes: usize,
    pub send_registered_span_bytes: usize,
    pub recv_registered_span_bytes: usize,
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_sge: u32,
    pub device_name: String,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerbsHostProtocolV2HandshakeValidation {
    pub protocol: String,
    pub control_plane: String,
    pub client_host: String,
    pub server_host: String,
    pub client_sends_request: bool,
    pub server_sends_response: bool,
    pub peer_qp_num_present: bool,
    pub peer_psn_present: bool,
    pub peer_gid_present: bool,
    pub registered_spans_match_endpoint_plan: bool,
    pub validated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerbsHostProtocolV2RoundTripPlan {
    pub protocol: String,
    pub data_plane: String,
    pub control_plane: String,
    pub memory: String,
    pub polling: String,
    pub client_host: String,
    pub server_host: String,
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub request_frame_bytes: usize,
    pub response_frame_bytes: usize,
    pub request_logical_payload_bytes: usize,
    pub response_logical_payload_bytes: usize,
    pub request_rows: usize,
    pub response_rows: usize,
    pub request_registered_span_bytes: usize,
    pub response_registered_span_bytes: usize,
    pub client_send_frame_bytes: usize,
    pub client_recv_frame_bytes: usize,
    pub server_send_frame_bytes: usize,
    pub server_recv_frame_bytes: usize,
    pub client_send_registered_span_bytes: usize,
    pub client_recv_registered_span_bytes: usize,
    pub server_send_registered_span_bytes: usize,
    pub server_recv_registered_span_bytes: usize,
    pub client_send_work_requests: usize,
    pub client_recv_work_requests: usize,
    pub server_send_work_requests: usize,
    pub server_recv_work_requests: usize,
    pub total_work_requests: usize,
    pub scatter_gather_entries_per_message: usize,
    pub request_frame_fits_registered_span: bool,
    pub response_frame_fits_registered_span: bool,
    pub request_response_headers_match: bool,
    pub endpoints_validated: bool,
    pub registered_spans_match_endpoint_plan: bool,
    pub app_transport_implemented: bool,
    pub app_transport_blocker: String,
}

pub fn verbs_host_protocol_v2_endpoint_plan(
    request_frame: &[u8],
    response_frame: &[u8],
    registration_alignment_bytes: usize,
) -> Result<VerbsHostProtocolV2EndpointPlan> {
    if registration_alignment_bytes == 0 {
        bail!("verbs-host ProtocolV2 endpoint registration alignment must be non-zero");
    }
    let request_view = ExpertProtocolV2RequestView::parse(request_frame)
        .context("verbs-host endpoint plan request frame is not valid ProtocolV2")?;
    let response_view = ExpertProtocolV2ResponseView::parse(response_frame)
        .context("verbs-host endpoint plan response frame is not valid ProtocolV2")?;
    let request_frame_bytes = request_view.wire_stats().wire_bytes;
    let response_frame_bytes = response_view.wire_stats().wire_bytes;
    let request_registered_span_bytes =
        align_up(request_frame_bytes, registration_alignment_bytes)?;
    let response_registered_span_bytes =
        align_up(response_frame_bytes, registration_alignment_bytes)?;

    let capabilities = crate::verbs_host_capabilities();

    Ok(VerbsHostProtocolV2EndpointPlan {
        protocol: "ExpertProtocolV2".to_owned(),
        data_plane: "rc-qp-send-recv".to_owned(),
        control_plane: "tcp-qp-gid-psn-handshake".to_owned(),
        memory: "registered-host-frame-arenas".to_owned(),
        polling: "busy-poll-cq".to_owned(),
        request_frame_bytes,
        response_frame_bytes,
        request_logical_payload_bytes: request_view.wire_stats().logical_payload_bytes,
        response_logical_payload_bytes: response_view.wire_stats().logical_payload_bytes,
        request_rows: request_view.header.row_count as usize,
        response_rows: response_view.header.row_count as usize,
        registration_alignment_bytes,
        request_registered_span_bytes,
        response_registered_span_bytes,
        total_registered_span_bytes: request_registered_span_bytes
            .checked_add(response_registered_span_bytes)
            .context("verbs-host ProtocolV2 endpoint registered span overflow")?,
        request_registration_slack_bytes: request_registered_span_bytes
            .checked_sub(request_frame_bytes)
            .context("verbs-host request registered span smaller than frame")?,
        response_registration_slack_bytes: response_registered_span_bytes
            .checked_sub(response_frame_bytes)
            .context("verbs-host response registered span smaller than frame")?,
        request_registered_span_aligned: request_registered_span_bytes
            % registration_alignment_bytes
            == 0,
        response_registered_span_aligned: response_registered_span_bytes
            % registration_alignment_bytes
            == 0,
        queue_pairs_per_peer: 1,
        send_work_requests_per_roundtrip: 2,
        recv_work_requests_per_roundtrip: 2,
        scatter_gather_entries_per_message: 1,
        requires_peer_qp_num: true,
        requires_peer_psn: true,
        requires_peer_gid: true,
        requires_registered_host_buffers: true,
        app_transport_implemented: capabilities.app_transport_implemented,
        app_transport_blocker: capabilities.app_transport_status,
    })
}

pub fn verbs_host_protocol_v2_round_trip_plan(
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
    handshake_validation: &VerbsHostProtocolV2HandshakeValidation,
    request_frame: &[u8],
    response_frame: &[u8],
) -> Result<VerbsHostProtocolV2RoundTripPlan> {
    if !handshake_validation.validated {
        bail!("verbs-host ProtocolV2 round trip requires a validated handshake");
    }
    if handshake_validation.protocol != endpoint_plan.protocol {
        bail!(
            "verbs-host ProtocolV2 round trip handshake protocol {} does not match endpoint plan {}",
            handshake_validation.protocol,
            endpoint_plan.protocol
        );
    }
    if handshake_validation.control_plane != endpoint_plan.control_plane {
        bail!(
            "verbs-host ProtocolV2 round trip handshake control_plane {} does not match endpoint plan {}",
            handshake_validation.control_plane,
            endpoint_plan.control_plane
        );
    }
    if !handshake_validation.client_sends_request {
        bail!("verbs-host ProtocolV2 round trip requires client_sends_request");
    }
    if !handshake_validation.server_sends_response {
        bail!("verbs-host ProtocolV2 round trip requires server_sends_response");
    }
    if !handshake_validation.peer_qp_num_present {
        bail!("verbs-host ProtocolV2 round trip requires peer qp_num descriptors");
    }
    if !handshake_validation.peer_psn_present {
        bail!("verbs-host ProtocolV2 round trip requires peer psn descriptors");
    }
    if !handshake_validation.peer_gid_present {
        bail!("verbs-host ProtocolV2 round trip requires peer gid descriptors");
    }
    if !handshake_validation.registered_spans_match_endpoint_plan {
        bail!("verbs-host ProtocolV2 round trip requires registered spans to match endpoint plan");
    }
    if endpoint_plan.send_work_requests_per_roundtrip != 2 {
        bail!(
            "verbs-host ProtocolV2 round trip endpoint plan send_work_requests_per_roundtrip {} does not match required request+response sends",
            endpoint_plan.send_work_requests_per_roundtrip
        );
    }
    if endpoint_plan.recv_work_requests_per_roundtrip != 2 {
        bail!(
            "verbs-host ProtocolV2 round trip endpoint plan recv_work_requests_per_roundtrip {} does not match required request+response receives",
            endpoint_plan.recv_work_requests_per_roundtrip
        );
    }
    if endpoint_plan.scatter_gather_entries_per_message != 1 {
        bail!(
            "verbs-host ProtocolV2 round trip currently requires one SGE per frame, got {}",
            endpoint_plan.scatter_gather_entries_per_message
        );
    }

    let request_view = ExpertProtocolV2RequestView::parse(request_frame)
        .context("verbs-host round trip request frame is not valid ProtocolV2")?;
    let response_view = ExpertProtocolV2ResponseView::parse(response_frame)
        .context("verbs-host round trip response frame is not valid ProtocolV2")?;
    let request_stats = request_view.wire_stats();
    let response_stats = response_view.wire_stats();

    validate_round_trip_usize(
        "request frame bytes",
        request_stats.wire_bytes,
        endpoint_plan.request_frame_bytes,
    )?;
    validate_round_trip_usize(
        "response frame bytes",
        response_stats.wire_bytes,
        endpoint_plan.response_frame_bytes,
    )?;
    validate_round_trip_usize(
        "request logical payload bytes",
        request_stats.logical_payload_bytes,
        endpoint_plan.request_logical_payload_bytes,
    )?;
    validate_round_trip_usize(
        "response logical payload bytes",
        response_stats.logical_payload_bytes,
        endpoint_plan.response_logical_payload_bytes,
    )?;
    validate_round_trip_usize(
        "request rows",
        request_view.header.row_count as usize,
        endpoint_plan.request_rows,
    )?;
    validate_round_trip_usize(
        "response rows",
        response_view.header.row_count as usize,
        endpoint_plan.response_rows,
    )?;

    if response_view.header.request_id != request_view.header.request_id {
        bail!(
            "verbs-host ProtocolV2 round trip response request_id {} does not match request {}",
            response_view.header.request_id,
            request_view.header.request_id
        );
    }
    if response_view.header.placement_version != request_view.header.placement_version {
        bail!(
            "verbs-host ProtocolV2 round trip response placement_version {} does not match request {}",
            response_view.header.placement_version,
            request_view.header.placement_version
        );
    }
    if response_view.header.layer_id != request_view.header.layer_id {
        bail!(
            "verbs-host ProtocolV2 round trip response layer_id {} does not match request {}",
            response_view.header.layer_id,
            request_view.header.layer_id
        );
    }
    if response_view.header.row_count != request_view.header.row_count {
        bail!(
            "verbs-host ProtocolV2 round trip response row_count {} does not match request {}",
            response_view.header.row_count,
            request_view.header.row_count
        );
    }

    let request_frame_fits_registered_span =
        request_stats.wire_bytes <= endpoint_plan.request_registered_span_bytes;
    if !request_frame_fits_registered_span {
        bail!(
            "verbs-host ProtocolV2 round trip request frame bytes {} exceed registered span {}",
            request_stats.wire_bytes,
            endpoint_plan.request_registered_span_bytes
        );
    }
    let response_frame_fits_registered_span =
        response_stats.wire_bytes <= endpoint_plan.response_registered_span_bytes;
    if !response_frame_fits_registered_span {
        bail!(
            "verbs-host ProtocolV2 round trip response frame bytes {} exceed registered span {}",
            response_stats.wire_bytes,
            endpoint_plan.response_registered_span_bytes
        );
    }

    Ok(VerbsHostProtocolV2RoundTripPlan {
        protocol: endpoint_plan.protocol.clone(),
        data_plane: endpoint_plan.data_plane.clone(),
        control_plane: endpoint_plan.control_plane.clone(),
        memory: endpoint_plan.memory.clone(),
        polling: endpoint_plan.polling.clone(),
        client_host: handshake_validation.client_host.clone(),
        server_host: handshake_validation.server_host.clone(),
        request_id: request_view.header.request_id,
        placement_version: request_view.header.placement_version,
        layer_id: request_view.header.layer_id,
        request_frame_bytes: request_stats.wire_bytes,
        response_frame_bytes: response_stats.wire_bytes,
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        response_logical_payload_bytes: response_stats.logical_payload_bytes,
        request_rows: request_view.header.row_count as usize,
        response_rows: response_view.header.row_count as usize,
        request_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        response_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        client_send_frame_bytes: endpoint_plan.request_frame_bytes,
        client_recv_frame_bytes: endpoint_plan.response_frame_bytes,
        server_send_frame_bytes: endpoint_plan.response_frame_bytes,
        server_recv_frame_bytes: endpoint_plan.request_frame_bytes,
        client_send_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        client_recv_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        server_send_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        server_recv_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        client_send_work_requests: 1,
        client_recv_work_requests: 1,
        server_send_work_requests: 1,
        server_recv_work_requests: 1,
        total_work_requests: 4,
        scatter_gather_entries_per_message: endpoint_plan.scatter_gather_entries_per_message,
        request_frame_fits_registered_span,
        response_frame_fits_registered_span,
        request_response_headers_match: true,
        endpoints_validated: handshake_validation.validated,
        registered_spans_match_endpoint_plan: handshake_validation
            .registered_spans_match_endpoint_plan,
        app_transport_implemented: endpoint_plan.app_transport_implemented,
        app_transport_blocker: endpoint_plan.app_transport_blocker.clone(),
    })
}

pub fn verbs_host_protocol_v2_handshake_contract(
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
) -> VerbsHostProtocolV2HandshakeContract {
    VerbsHostProtocolV2HandshakeContract {
        protocol: endpoint_plan.protocol.clone(),
        control_plane: endpoint_plan.control_plane.clone(),
        descriptor_fields: [
            "role",
            "host",
            "port_num",
            "qp_num",
            "psn",
            "gid_hex",
            "send_frame_bytes",
            "recv_frame_bytes",
            "send_registered_span_bytes",
            "recv_registered_span_bytes",
            "max_send_wr",
            "max_recv_wr",
            "max_sge",
        ]
        .iter()
        .map(|field| (*field).to_owned())
        .collect(),
        client_role: "client".to_owned(),
        server_role: "server".to_owned(),
        client_send_frame_bytes: endpoint_plan.request_frame_bytes,
        client_recv_frame_bytes: endpoint_plan.response_frame_bytes,
        server_send_frame_bytes: endpoint_plan.response_frame_bytes,
        server_recv_frame_bytes: endpoint_plan.request_frame_bytes,
        client_send_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        client_recv_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        server_send_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        server_recv_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        requires_peer_qp_num: endpoint_plan.requires_peer_qp_num,
        requires_peer_psn: endpoint_plan.requires_peer_psn,
        requires_peer_gid: endpoint_plan.requires_peer_gid,
        requires_registered_host_buffers: endpoint_plan.requires_registered_host_buffers,
        descriptor_validation_available: true,
    }
}

pub fn verbs_host_validate_protocol_v2_handshake(
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
    client: &VerbsHostRcEndpointDescriptor,
    server: &VerbsHostRcEndpointDescriptor,
) -> Result<VerbsHostProtocolV2HandshakeValidation> {
    validate_endpoint_descriptor(endpoint_plan, client, "client")?;
    validate_endpoint_descriptor(endpoint_plan, server, "server")?;
    if client.host == server.host {
        bail!("verbs-host ProtocolV2 handshake client/server hosts must differ");
    }

    validate_endpoint_direction(
        endpoint_plan,
        client,
        endpoint_plan.request_frame_bytes,
        endpoint_plan.response_frame_bytes,
        endpoint_plan.request_registered_span_bytes,
        endpoint_plan.response_registered_span_bytes,
        "client",
    )?;
    validate_endpoint_direction(
        endpoint_plan,
        server,
        endpoint_plan.response_frame_bytes,
        endpoint_plan.request_frame_bytes,
        endpoint_plan.response_registered_span_bytes,
        endpoint_plan.request_registered_span_bytes,
        "server",
    )?;

    Ok(VerbsHostProtocolV2HandshakeValidation {
        protocol: endpoint_plan.protocol.clone(),
        control_plane: endpoint_plan.control_plane.clone(),
        client_host: client.host.clone(),
        server_host: server.host.clone(),
        client_sends_request: true,
        server_sends_response: true,
        peer_qp_num_present: endpoint_plan.requires_peer_qp_num
            && client.qp_num != 0
            && server.qp_num != 0,
        peer_psn_present: endpoint_plan.requires_peer_psn
            && client.psn <= 0x00ff_ffff
            && server.psn <= 0x00ff_ffff,
        peer_gid_present: endpoint_plan.requires_peer_gid
            && valid_gid_hex(&client.gid_hex)
            && valid_gid_hex(&server.gid_hex),
        registered_spans_match_endpoint_plan: true,
        validated: true,
    })
}

pub async fn verbs_host_protocol_v2_expert_request_roundtrip(
    addr: SocketAddr,
    request: &ExpertRequest,
    config: TcpTransportConfig,
) -> Result<ExpertResponse> {
    let protocol_v2_request = protocol_v2_request_from_expert_request(request)?;
    let protocol_v2_response =
        verbs_host_protocol_v2_roundtrip(addr, &protocol_v2_request, config).await?;
    expert_response_from_protocol_v2_response(request, &protocol_v2_response)
}

pub async fn verbs_host_protocol_v2_roundtrip(
    addr: SocketAddr,
    request: &ExpertProtocolV2Request,
    config: TcpTransportConfig,
) -> Result<ExpertProtocolV2Response> {
    let request = request.clone();
    tokio::task::spawn_blocking(move || {
        verbs_host_protocol_v2_roundtrip_blocking(addr, request, config)
    })
    .await
    .context("verbs-host ProtocolV2 roundtrip worker panicked")?
}

enum VerbsHostProtocolV2PersistentCommand {
    Roundtrip {
        request: ExpertProtocolV2Request,
        response_tx: tokio::sync::oneshot::Sender<Result<Vec<u8>>>,
    },
    RoundtripChunks {
        request: ExpertProtocolV2Request,
        stream_id: usize,
        chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
        response_tx: tokio::sync::oneshot::Sender<Result<VerbsHostProtocolV2ResponseStreamStats>>,
    },
    Reset,
    Shutdown,
}

struct VerbsHostProtocolV2CqPollRequest {
    library: Arc<NativeLibrary>,
    endpoint_handle: usize,
    max_send_completions: u32,
    max_recv_completions: u32,
    completed_send_completions: u32,
    poll_iterations: u32,
    deadline: Instant,
    response_tx: mpsc::SyncSender<Result<Ds41rtRdmaRcCompletionStats>>,
}

struct VerbsHostProtocolV2CqWaiter {
    response_tx: mpsc::SyncSender<Result<Ds41rtRdmaRcCompletionStats>>,
    response_rx: mpsc::Receiver<Result<Ds41rtRdmaRcCompletionStats>>,
}

impl VerbsHostProtocolV2CqWaiter {
    fn new() -> Self {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        Self {
            response_tx,
            response_rx,
        }
    }
}

enum VerbsHostProtocolV2CqHarvesterCommand {
    Poll(VerbsHostProtocolV2CqPollRequest),
    Shutdown,
}

pub(crate) struct VerbsHostProtocolV2CqHarvester {
    execution_lane: u32,
    tx: mpsc::Sender<VerbsHostProtocolV2CqHarvesterCommand>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
}

impl VerbsHostProtocolV2CqHarvester {
    pub(crate) fn new(execution_lane: u32) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name(format!("verbs-v2-cq-l{execution_lane}"))
            .spawn(move || verbs_host_protocol_v2_cq_harvester_worker(execution_lane, rx))
            .with_context(|| {
                format!("spawning verbs-host CQ harvester for lane {execution_lane}")
            })?;
        Ok(Arc::new(Self {
            execution_lane,
            tx,
            join: Mutex::new(Some(join)),
        }))
    }

    fn wait_for_response(
        &self,
        endpoint: &NativeRdmaEndpoint,
        waiter: &VerbsHostProtocolV2CqWaiter,
        max_send_completions: u32,
        max_recv_completions: u32,
        timeout: Duration,
    ) -> Result<Ds41rtRdmaRcCompletionStats> {
        anyhow::ensure!(
            max_recv_completions > 0,
            "verbs-host CQ harvester requires a receive completion target"
        );
        let deadline = Instant::now()
            .checked_add(timeout)
            .context("verbs-host CQ harvester deadline overflow")?;
        self.tx
            .send(VerbsHostProtocolV2CqHarvesterCommand::Poll(
                VerbsHostProtocolV2CqPollRequest {
                    library: Arc::clone(&endpoint.library),
                    endpoint_handle: endpoint.info.handle as usize,
                    max_send_completions,
                    max_recv_completions,
                    completed_send_completions: 0,
                    poll_iterations: 0,
                    deadline,
                    response_tx: waiter.response_tx.clone(),
                },
            ))
            .with_context(|| {
                format!(
                    "submitting verbs-host CQ poll to lane {} harvester",
                    self.execution_lane
                )
            })?;
        waiter.response_rx.recv().with_context(|| {
            format!(
                "receiving verbs-host CQ poll result from lane {} harvester",
                self.execution_lane
            )
        })?
    }
}

impl Drop for VerbsHostProtocolV2CqHarvester {
    fn drop(&mut self) {
        let _ = self
            .tx
            .send(VerbsHostProtocolV2CqHarvesterCommand::Shutdown);
        let Some(join) = self.join.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if thread::current().id() != join.thread().id() {
            let _ = join.join();
        }
    }
}

fn verbs_host_protocol_v2_cq_harvester_worker(
    execution_lane: u32,
    rx: mpsc::Receiver<VerbsHostProtocolV2CqHarvesterCommand>,
) {
    let mut active = VecDeque::<VerbsHostProtocolV2CqPollRequest>::new();
    let mut shutdown = false;
    while !shutdown {
        if active.is_empty() {
            match rx.recv() {
                Ok(VerbsHostProtocolV2CqHarvesterCommand::Poll(request)) => {
                    active.push_back(request)
                }
                Ok(VerbsHostProtocolV2CqHarvesterCommand::Shutdown) | Err(_) => break,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(VerbsHostProtocolV2CqHarvesterCommand::Poll(request)) => {
                    active.push_back(request)
                }
                Ok(VerbsHostProtocolV2CqHarvesterCommand::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }

        let scan_count = active.len();
        let mut made_progress = false;
        for _ in 0..scan_count {
            let request = active
                .pop_front()
                .expect("verbs-host CQ harvester scan count matches active requests");
            if Instant::now() >= request.deadline {
                let _ = request.response_tx.send(Err(anyhow::anyhow!(
                    "verbs-host lane {execution_lane} CQ harvester timed out waiting for a response completion"
                )));
                continue;
            }
            let result = request.library.rdma_rc_endpoint_try_poll(
                request.endpoint_handle as *mut c_void,
                request.max_send_completions,
                request.max_recv_completions,
            );
            match result {
                Ok(mut stats) if stats.recv_completions > 0 => {
                    made_progress = true;
                    stats.send_completions = stats
                        .send_completions
                        .saturating_add(request.completed_send_completions);
                    stats.poll_iterations = stats
                        .poll_iterations
                        .saturating_add(request.poll_iterations);
                    let _ = request.response_tx.send(Ok(stats));
                }
                Ok(stats) if stats.send_completions > 0 => {
                    made_progress = true;
                    let mut request = request;
                    request.max_send_completions = request
                        .max_send_completions
                        .saturating_sub(stats.send_completions);
                    request.completed_send_completions = request
                        .completed_send_completions
                        .saturating_add(stats.send_completions);
                    request.poll_iterations = request
                        .poll_iterations
                        .saturating_add(stats.poll_iterations);
                    active.push_back(request);
                }
                Ok(stats) => {
                    let mut request = request;
                    request.poll_iterations = request
                        .poll_iterations
                        .saturating_add(stats.poll_iterations);
                    active.push_back(request);
                }
                Err(error) => {
                    made_progress = true;
                    let _ = request.response_tx.send(Err(error));
                }
            }
        }
        if !made_progress {
            std::hint::spin_loop();
        }
    }
    while let Some(request) = active.pop_front() {
        let _ = request.response_tx.send(Err(anyhow::anyhow!(
            "verbs-host lane {execution_lane} CQ harvester shut down with a response pending"
        )));
    }
}

struct VerbsHostProtocolV2PinnedResponseFrame {
    library: Arc<NativeLibrary>,
    buffer: Ds41rtHostBuffer,
    len: usize,
}

// The native allocation is portable CUDA-pinned host storage. Ownership of a
// completed response frame moves from the verbs worker to the scheduler and
// back through the recycle channel, but the pointer is never accessed after
// the frame is recycled.
unsafe impl Send for VerbsHostProtocolV2PinnedResponseFrame {}
// A completed frame is immutable while it is exposed through a payload. Drop
// requires unique ownership before the allocation can return to the pool.
unsafe impl Sync for VerbsHostProtocolV2PinnedResponseFrame {}

impl VerbsHostProtocolV2PinnedResponseFrame {
    fn new(library: Arc<NativeLibrary>, len: usize) -> Result<Self> {
        let buffer = library
            .alloc_host_buffer(len)
            .context("allocating pinned verbs-host response frame")?;
        Ok(Self {
            library,
            buffer,
            len,
        })
    }

    fn resize(&mut self, len: usize) -> Result<()> {
        anyhow::ensure!(
            len <= self.buffer.bytes,
            "pinned verbs-host response frame bytes {len} exceed capacity {}",
            self.buffer.bytes
        );
        self.len = len;
        Ok(())
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.ptr.cast::<u8>(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr.cast::<u8>(), self.len) }
    }

    fn payload_host_buffer(&self, start: usize, end: usize) -> Option<Ds41rtHostBuffer> {
        if self.buffer.flags & DS41RT_HOST_BUFFER_FLAG_PINNED == 0 {
            return None;
        }
        Some(Ds41rtHostBuffer {
            ptr: unsafe { self.buffer.ptr.cast::<u8>().add(start).cast() },
            bytes: end - start,
            flags: self.buffer.flags,
        })
    }
}

impl Drop for VerbsHostProtocolV2PinnedResponseFrame {
    fn drop(&mut self) {
        if self.buffer.ptr.is_null() {
            return;
        }
        let _ = self.library.free_host_buffer(&mut self.buffer);
        self.buffer = Ds41rtHostBuffer::default();
    }
}

enum VerbsHostProtocolV2ResponseFrame {
    Bytes(Vec<u8>),
    Pinned(VerbsHostProtocolV2PinnedResponseFrame),
    Registered(RegisteredResponseFrame),
}

impl AsRef<[u8]> for VerbsHostProtocolV2ResponseFrame {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Pinned(frame) => frame.as_slice(),
            Self::Registered(frame) => frame.as_slice(),
        }
    }
}

impl AsMut<[u8]> for VerbsHostProtocolV2ResponseFrame {
    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::Pinned(frame) => frame.as_mut_slice(),
            Self::Registered(frame) => frame.as_mut_slice(),
        }
    }
}

pub struct VerbsHostProtocolV2ResponsePayload {
    bytes: Option<Vec<u8>>,
    pinned_frame: Option<VerbsHostProtocolV2PinnedResponseFrame>,
    registered_frame: Option<RegisteredResponseFrame>,
    payload_start: usize,
    payload_end: usize,
    recycle_tx: Option<mpsc::Sender<Vec<u8>>>,
    pinned_recycle_tx: Option<mpsc::Sender<VerbsHostProtocolV2PinnedResponseFrame>>,
}

impl VerbsHostProtocolV2ResponsePayload {
    #[cfg(test)]
    fn from_frame(
        bytes: Vec<u8>,
        payload_start: usize,
        payload_end: usize,
        recycle_tx: mpsc::Sender<Vec<u8>>,
    ) -> Result<Self> {
        if payload_start > payload_end || payload_end > bytes.len() {
            bail!(
                "ProtocolV2 response payload range {payload_start}..{payload_end} exceeds frame bytes {}",
                bytes.len()
            );
        }
        Ok(Self {
            bytes: Some(bytes),
            pinned_frame: None,
            registered_frame: None,
            payload_start,
            payload_end,
            recycle_tx: Some(recycle_tx),
            pinned_recycle_tx: None,
        })
    }

    fn from_response_frame(
        frame: VerbsHostProtocolV2ResponseFrame,
        payload_start: usize,
        payload_end: usize,
        recycle_tx: mpsc::Sender<Vec<u8>>,
        pinned_recycle_tx: mpsc::Sender<VerbsHostProtocolV2PinnedResponseFrame>,
    ) -> Result<Self> {
        let frame_len = frame.as_ref().len();
        if payload_start > payload_end || payload_end > frame_len {
            bail!(
                "ProtocolV2 response payload range {payload_start}..{payload_end} exceeds frame bytes {frame_len}"
            );
        }
        Ok(match frame {
            VerbsHostProtocolV2ResponseFrame::Bytes(bytes) => Self {
                bytes: Some(bytes),
                pinned_frame: None,
                registered_frame: None,
                payload_start,
                payload_end,
                recycle_tx: Some(recycle_tx),
                pinned_recycle_tx: None,
            },
            VerbsHostProtocolV2ResponseFrame::Pinned(frame) => Self {
                bytes: None,
                pinned_frame: Some(frame),
                registered_frame: None,
                payload_start,
                payload_end,
                recycle_tx: None,
                pinned_recycle_tx: Some(pinned_recycle_tx),
            },
            VerbsHostProtocolV2ResponseFrame::Registered(frame) => Self {
                bytes: None, pinned_frame: None, registered_frame: Some(frame),
                payload_start, payload_end, recycle_tx: None, pinned_recycle_tx: None,
            },
        })
    }

    pub fn from_owned(bytes: Vec<u8>) -> Self {
        let payload_end = bytes.len();
        Self {
            bytes: Some(bytes),
            pinned_frame: None,
            registered_frame: None,
            payload_start: 0,
            payload_end,
            recycle_tx: None,
            pinned_recycle_tx: None,
        }
    }

    /// Return the exact payload subview when the streamed response frame is
    /// backed by CUDA-pinned host storage (a copied frame or an owned receive
    /// slot). The view remains valid only while this payload object is alive.
    pub fn pinned_host_buffer(&self) -> Option<Ds41rtHostBuffer> {
        if let Some(frame) = &self.registered_frame {
            return Some(frame.payload_host_buffer(self.payload_start, self.payload_end));
        }
        self.pinned_frame
            .as_ref()?
            .payload_host_buffer(self.payload_start, self.payload_end)
    }

    /// True when this owner retains an unposted receive slot, rather than a
    /// copied response frame. Releasing it permits the next local dispatch.
    pub fn retains_receive_slot(&self) -> bool { self.registered_frame.is_some() }

    pub fn into_vec(mut self) -> Vec<u8> {
        if self.recycle_tx.is_none()
            && self.payload_start == 0
            && self.payload_end == self.bytes.as_ref().map_or(0, Vec::len)
        {
            return self.bytes.take().unwrap_or_default();
        }
        self.as_ref().to_vec()
    }
}

impl AsRef<[u8]> for VerbsHostProtocolV2ResponsePayload {
    fn as_ref(&self) -> &[u8] {
        if let Some(bytes) = self.bytes.as_deref() {
            &bytes[self.payload_start..self.payload_end]
        } else if let Some(frame) = self.pinned_frame.as_ref() {
            &frame.as_slice()[self.payload_start..self.payload_end]
        } else if let Some(frame) = &self.registered_frame {
            &frame.as_slice()[self.payload_start..self.payload_end]
        } else {
            &[]
        }
    }
}

impl std::fmt::Debug for VerbsHostProtocolV2ResponsePayload {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerbsHostProtocolV2ResponsePayload")
            .field("bytes", &self.as_ref().len())
            .field("frame_backed", &self.recycle_tx.is_some())
            .finish()
    }
}

impl PartialEq for VerbsHostProtocolV2ResponsePayload {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for VerbsHostProtocolV2ResponsePayload {}

impl Drop for VerbsHostProtocolV2ResponsePayload {
    fn drop(&mut self) {
        if let Some(mut bytes) = self.bytes.take() {
            if let Some(recycle_tx) = &self.recycle_tx {
                bytes.clear();
                let _ = recycle_tx.send(bytes);
            }
        }
        if let Some(frame) = self.pinned_frame.take() {
            if let Some(recycle_tx) = &self.pinned_recycle_tx {
                let _ = recycle_tx.send(frame);
            }
        }
    }
}

#[cfg(test)]
#[test]
fn verbs_response_payload_recycles_frame_storage() -> Result<()> {
    let (recycle_tx, recycle_rx) = mpsc::channel();
    let payload =
        VerbsHostProtocolV2ResponsePayload::from_frame(vec![9_u8, 1, 2, 3, 8], 1, 4, recycle_tx)?;
    assert_eq!(payload.as_ref(), &[1, 2, 3]);
    drop(payload);
    let recycled = recycle_rx.recv()?;
    assert!(recycled.is_empty());
    assert!(recycled.capacity() >= 5);

    let owned = VerbsHostProtocolV2ResponsePayload::from_owned(vec![4, 5, 6]);
    assert_eq!(owned.into_vec(), vec![4, 5, 6]);
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "requires CUDA native library"]
fn verbs_response_payload_exposes_and_recycles_pinned_subview() -> Result<()> {
    let library = load_verbs_host_native_library()?;
    let mut frame = VerbsHostProtocolV2PinnedResponseFrame::new(library, 5)?;
    frame.as_mut_slice().copy_from_slice(&[9, 1, 2, 3, 8]);
    let frame_ptr = frame.buffer.ptr;
    let (bytes_recycle_tx, _bytes_recycle_rx) = mpsc::channel();
    let (pinned_recycle_tx, pinned_recycle_rx) = mpsc::channel();
    let payload = VerbsHostProtocolV2ResponsePayload::from_response_frame(
        VerbsHostProtocolV2ResponseFrame::Pinned(frame),
        1,
        4,
        bytes_recycle_tx,
        pinned_recycle_tx,
    )?;
    assert_eq!(payload.as_ref(), &[1, 2, 3]);
    let pinned = payload
        .pinned_host_buffer()
        .context("pinned response payload did not expose a host buffer")?;
    assert_eq!(pinned.ptr, unsafe { frame_ptr.cast::<u8>().add(1).cast() });
    assert_eq!(pinned.bytes, 3);
    assert_ne!(pinned.flags & DS41RT_HOST_BUFFER_FLAG_PINNED, 0);
    drop(payload);
    let recycled = pinned_recycle_rx.recv()?;
    assert_eq!(recycled.buffer.ptr, frame_ptr);
    assert_eq!(recycled.as_slice(), &[9, 1, 2, 3, 8]);
    Ok(())
}

#[derive(Debug)]
pub struct VerbsHostProtocolV2ResponseChunk {
    pub stream_id: usize,
    pub header: ExpertProtocolV2ResponseHeader,
    pub row_indices: Option<Vec<u32>>,
    pub partial_output_payload: VerbsHostProtocolV2ResponsePayload,
    pub wire_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerbsHostProtocolV2ResponseStreamStats {
    pub response_frames: usize,
    pub response_wire_bytes: usize,
    pub response_executor_id: u64,
}

pub(crate) struct VerbsHostProtocolV2LaneFanoutResponse {
    pub response_stats_by_stream: Vec<VerbsHostProtocolV2ResponseStreamStats>,
    pub chunks: Vec<VerbsHostProtocolV2ResponseChunk>,
}

pub(crate) struct VerbsHostProtocolV2LaneFanoutPending {
    response_rx: tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2LaneFanoutResponse>>,
}

impl VerbsHostProtocolV2LaneFanoutPending {
    pub(crate) fn wait(self) -> Result<VerbsHostProtocolV2LaneFanoutResponse> {
        self.response_rx
            .blocking_recv()
            .context("receiving verbs-host lane fanout completion")?
    }
}

#[derive(Clone)]
pub(crate) struct VerbsHostProtocolV2LaneFanoutClient {
    inner: Arc<VerbsHostProtocolV2LaneFanoutClientInner>,
}

struct VerbsHostProtocolV2LaneFanoutClientInner {
    tx: mpsc::Sender<VerbsHostProtocolV2LaneFanoutCommand>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
}

enum VerbsHostProtocolV2LaneFanoutCommand {
    Roundtrip {
        requests: Vec<ExpertProtocolV2Request>,
        response_tx: tokio::sync::oneshot::Sender<Result<VerbsHostProtocolV2LaneFanoutResponse>>,
    },
    Reset,
    Shutdown,
}

#[derive(Clone)]
pub struct VerbsHostProtocolV2PersistentClient {
    inner: Arc<VerbsHostProtocolV2PersistentClientInner>,
}

pub struct VerbsHostProtocolV2PendingResponse {
    addr: SocketAddr,
    response_rx: tokio::sync::oneshot::Receiver<Result<Vec<u8>>>,
}

impl VerbsHostProtocolV2PendingResponse {
    pub fn wait_response_frame(self) -> Result<Vec<u8>> {
        self.response_rx.blocking_recv().with_context(|| {
            format!(
                "receiving blocking verbs-host ProtocolV2 persistent response from {}",
                self.addr
            )
        })?
    }

    pub fn wait(self) -> Result<ExpertProtocolV2Response> {
        let frame = self.wait_response_frame()?;
        ExpertProtocolV2Response::decode(&frame)
            .context("decoding blocking persistent verbs-host ProtocolV2 response frame")
    }

    async fn receive_response_frame(self) -> Result<Vec<u8>> {
        self.response_rx.await.with_context(|| {
            format!(
                "receiving verbs-host ProtocolV2 persistent roundtrip response from {}",
                self.addr
            )
        })?
    }
}

struct VerbsHostProtocolV2PersistentClientInner {
    addr: SocketAddr,
    execution_lane: u32,
    tx: mpsc::Sender<VerbsHostProtocolV2PersistentCommand>,
    chunk_sequence_lock: tokio::sync::Mutex<()>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
}

impl VerbsHostProtocolV2PersistentClient {
    pub fn new(addr: SocketAddr, config: TcpTransportConfig) -> Result<Self> {
        Self::new_with_execution_lane(addr, config, 0)
    }

    pub fn new_with_execution_lane(
        addr: SocketAddr,
        config: TcpTransportConfig,
        execution_lane: u32,
    ) -> Result<Self> {
        Self::new_with_execution_lane_and_cq_harvester(addr, config, execution_lane, None)
    }

    pub(crate) fn new_with_execution_lane_and_cq_harvester(
        addr: SocketAddr,
        config: TcpTransportConfig,
        execution_lane: u32,
        cq_harvester: Option<Arc<VerbsHostProtocolV2CqHarvester>>,
    ) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name(format!("verbs-v2-client-{addr}-lane-{execution_lane}"))
            .spawn(move || {
                verbs_host_protocol_v2_persistent_client_worker(
                    addr,
                    config,
                    execution_lane,
                    cq_harvester,
                    rx,
                )
            })
            .with_context(|| {
                format!(
                    "spawning verbs-host ProtocolV2 persistent client {addr} lane {execution_lane}"
                )
            })?;
        Ok(Self {
            inner: Arc::new(VerbsHostProtocolV2PersistentClientInner {
                addr,
                execution_lane,
                tx,
                chunk_sequence_lock: tokio::sync::Mutex::new(()),
                join: Mutex::new(Some(join)),
            }),
        })
    }

    pub async fn roundtrip(
        &self,
        request: &ExpertProtocolV2Request,
    ) -> Result<ExpertProtocolV2Response> {
        let frame = self.roundtrip_response_frame(request).await?;
        ExpertProtocolV2Response::decode(&frame)
            .context("decoding persistent verbs-host ProtocolV2 response frame")
    }

    pub async fn roundtrip_response_frame(
        &self,
        request: &ExpertProtocolV2Request,
    ) -> Result<Vec<u8>> {
        let timing_enabled = protocol_v2_transport_timing_enabled();
        let total_started = timing_enabled.then(Instant::now);
        let enqueue_started = timing_enabled.then(Instant::now);
        let pending = self.enqueue_roundtrip(request.clone())?;
        let enqueue_ms = elapsed_ms_optional(enqueue_started);
        let await_started = timing_enabled.then(Instant::now);
        let response = pending.receive_response_frame().await?;
        let await_ms = elapsed_ms_optional(await_started);
        if timing_enabled {
            eprintln!(
                "protocol_v2_verbs_persistent_client_command_timing addr={} execution_lane={} request_id={} layer_id={} rows={} routes={} response_wire_bytes={} enqueue_ms={:.3} await_ms={:.3} total_ms={:.3}",
                self.inner.addr,
                self.inner.execution_lane,
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                response.len(),
                enqueue_ms,
                await_ms,
                elapsed_ms_optional(total_started)
            );
        }
        Ok(response)
    }

    pub fn enqueue_roundtrip(
        &self,
        request: ExpertProtocolV2Request,
    ) -> Result<VerbsHostProtocolV2PendingResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.inner
            .tx
            .send(VerbsHostProtocolV2PersistentCommand::Roundtrip {
                request,
                response_tx,
            })
            .with_context(|| {
                format!(
                    "sending verbs-host ProtocolV2 persistent roundtrip command to {}",
                    self.inner.addr
                )
            })?;
        Ok(VerbsHostProtocolV2PendingResponse {
            addr: self.inner.addr,
            response_rx,
        })
    }

    pub async fn roundtrip_response_chunks(
        &self,
        request: ExpertProtocolV2Request,
        stream_id: usize,
        chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
    ) -> Result<VerbsHostProtocolV2ResponseStreamStats> {
        let response_rx = self.enqueue_response_chunks(request, stream_id, chunk_tx)?;
        response_rx.await.with_context(|| {
            format!(
                "receiving verbs-host ProtocolV2 persistent streaming completion from {}",
                self.inner.addr
            )
        })?
    }

    pub async fn roundtrip_response_chunk_sequence(
        &self,
        requests: Vec<ExpertProtocolV2Request>,
        stream_id: usize,
        chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
    ) -> Result<VerbsHostProtocolV2ResponseStreamStats> {
        anyhow::ensure!(
            !requests.is_empty(),
            "verbs-host ProtocolV2 streamed-ingress sequence must contain requests"
        );
        let response_receivers = {
            let _sequence = self.inner.chunk_sequence_lock.lock().await;
            let mut response_receivers = Vec::with_capacity(requests.len());
            for request in requests {
                response_receivers.push(self.enqueue_response_chunks(
                    request,
                    stream_id,
                    chunk_tx.clone(),
                )?);
            }
            response_receivers
        };

        let mut response_frames = 0_usize;
        let mut response_wire_bytes = 0_usize;
        let mut response_executor_id = None;
        for response_rx in response_receivers {
            let stats = response_rx.await.with_context(|| {
                format!(
                    "receiving verbs-host ProtocolV2 persistent streamed-ingress completion from {}",
                    self.inner.addr
                )
            })??;
            response_frames = response_frames
                .checked_add(stats.response_frames)
                .context("streamed-ingress response frame count overflow")?;
            response_wire_bytes = response_wire_bytes
                .checked_add(stats.response_wire_bytes)
                .context("streamed-ingress response byte count overflow")?;
            if let Some(expected) = response_executor_id {
                anyhow::ensure!(
                    stats.response_executor_id == expected,
                    "streamed-ingress response executor changed from {expected} to {}",
                    stats.response_executor_id
                );
            } else {
                response_executor_id = Some(stats.response_executor_id);
            }
        }
        Ok(VerbsHostProtocolV2ResponseStreamStats {
            response_frames,
            response_wire_bytes,
            response_executor_id: response_executor_id
                .context("streamed-ingress sequence returned no executor identity")?,
        })
    }

    pub(crate) fn enqueue_response_chunks(
        &self,
        request: ExpertProtocolV2Request,
        stream_id: usize,
        chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<VerbsHostProtocolV2ResponseStreamStats>>>
    {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.inner
            .tx
            .send(VerbsHostProtocolV2PersistentCommand::RoundtripChunks {
                request,
                stream_id,
                chunk_tx,
                response_tx,
            })
            .with_context(|| {
                format!(
                    "sending verbs-host ProtocolV2 persistent streaming roundtrip command to {}",
                    self.inner.addr
                )
            })?;
        Ok(response_rx)
    }

    pub fn reset(&self) {
        let _ = self
            .inner
            .tx
            .send(VerbsHostProtocolV2PersistentCommand::Reset);
    }
}

impl Drop for VerbsHostProtocolV2PersistentClientInner {
    fn drop(&mut self) {
        let _ = self.tx.send(VerbsHostProtocolV2PersistentCommand::Shutdown);
        let Some(join) = self.join.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if thread::current().id() != join.thread().id() {
            let _ = join.join();
        }
    }
}

impl VerbsHostProtocolV2LaneFanoutClient {
    pub(crate) fn new(
        addrs: Vec<SocketAddr>,
        config: TcpTransportConfig,
        execution_lane: u32,
    ) -> Result<Self> {
        anyhow::ensure!(
            !addrs.is_empty(),
            "verbs-host lane fanout requires at least one target"
        );
        let (tx, rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name(format!("verbs-v2-fanout-lane-{execution_lane}"))
            .spawn(move || {
                verbs_host_protocol_v2_lane_fanout_worker(addrs, config, execution_lane, rx)
            })
            .with_context(|| format!("spawning verbs-host lane {execution_lane} fanout worker"))?;
        Ok(Self {
            inner: Arc::new(VerbsHostProtocolV2LaneFanoutClientInner {
                tx,
                join: Mutex::new(Some(join)),
            }),
        })
    }

    pub(crate) fn enqueue(
        &self,
        requests: Vec<ExpertProtocolV2Request>,
    ) -> Result<VerbsHostProtocolV2LaneFanoutPending> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.inner
            .tx
            .send(VerbsHostProtocolV2LaneFanoutCommand::Roundtrip {
                requests,
                response_tx,
            })
            .context("submitting verbs-host lane fanout")?;
        Ok(VerbsHostProtocolV2LaneFanoutPending { response_rx })
    }

    pub(crate) fn reset(&self) {
        let _ = self
            .inner
            .tx
            .send(VerbsHostProtocolV2LaneFanoutCommand::Reset);
    }
}

impl Drop for VerbsHostProtocolV2LaneFanoutClientInner {
    fn drop(&mut self) {
        let _ = self.tx.send(VerbsHostProtocolV2LaneFanoutCommand::Shutdown);
        let Some(join) = self.join.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if thread::current().id() != join.thread().id() {
            let _ = join.join();
        }
    }
}

fn verbs_host_protocol_v2_lane_fanout_worker(
    addrs: Vec<SocketAddr>,
    config: TcpTransportConfig,
    execution_lane: u32,
    rx: mpsc::Receiver<VerbsHostProtocolV2LaneFanoutCommand>,
) {
    let mut sessions = (0..addrs.len()).map(|_| None).collect::<Vec<_>>();
    let mut spin_until = None;
    loop {
        let command = loop {
            match rx.try_recv() {
                Ok(command) => break Some(command),
                Err(mpsc::TryRecvError::Disconnected) => break None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
            if spin_until.is_some_and(|deadline| Instant::now() < deadline) {
                std::hint::spin_loop();
                continue;
            }
            break rx.recv().ok();
        };
        let Some(command) = command else {
            break;
        };
        match command {
            VerbsHostProtocolV2LaneFanoutCommand::Roundtrip {
                requests,
                response_tx,
            } => {
                let result = verbs_host_protocol_v2_lane_fanout_roundtrip(
                    &addrs,
                    &config,
                    execution_lane,
                    &mut sessions,
                    requests,
                );
                if result.is_err() {
                    sessions.iter_mut().for_each(|session| *session = None);
                }
                let _ = response_tx.send(result);
                spin_until = Instant::now().checked_add(VERBS_HOST_LANE_FANOUT_COMMAND_SPIN);
            }
            VerbsHostProtocolV2LaneFanoutCommand::Reset => {
                sessions.iter_mut().for_each(|session| *session = None);
                spin_until = None;
            }
            VerbsHostProtocolV2LaneFanoutCommand::Shutdown => break,
        }
    }
}

fn verbs_host_protocol_v2_lane_fanout_roundtrip(
    addrs: &[SocketAddr],
    config: &TcpTransportConfig,
    execution_lane: u32,
    sessions: &mut [Option<VerbsHostProtocolV2PersistentClientSession>],
    requests: Vec<ExpertProtocolV2Request>,
) -> Result<VerbsHostProtocolV2LaneFanoutResponse> {
    anyhow::ensure!(
        requests.len() == addrs.len() && sessions.len() == addrs.len(),
        "verbs-host lane fanout received {} requests for {} targets and {} sessions",
        requests.len(),
        addrs.len(),
        sessions.len()
    );
    let host_count = requests.len();
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut response_receivers = Vec::with_capacity(host_count);
    let mut pending_by_host = (0..host_count)
        .map(|_| VecDeque::new())
        .collect::<Vec<VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>>>();

    for (host_index, request) in requests.into_iter().enumerate() {
        if sessions[host_index]
            .as_ref()
            .map(|session| session.fits(&request))
            .transpose()?
            == Some(false)
        {
            sessions[host_index] = None;
        }
        if sessions[host_index].is_none() {
            sessions[host_index] = Some(VerbsHostProtocolV2PersistentClientSession::connect(
                addrs[host_index],
                config,
                &request,
                execution_lane,
                None,
            )?);
        }
        let timing = sessions[host_index]
            .as_mut()
            .expect("verbs-host lane fanout session connected above")
            .post_chunk_request(&request, config)?;
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        pending_by_host[host_index].push_back(VerbsHostProtocolV2PendingChunkRoundtrip::new(
            VerbsHostProtocolV2QueuedChunkCommand {
                request,
                stream_id: host_index,
                chunk_tx: chunk_tx.clone(),
                response_tx,
            },
            timing,
        ));
        response_receivers.push(response_rx);
    }
    drop(chunk_tx);

    let deadline = Instant::now()
        .checked_add(config.timeout)
        .context("verbs-host lane fanout deadline overflow")?;
    let busy_poll_started = Instant::now();
    let timing_enabled = config.timing;
    while pending_by_host.iter().any(|pending| !pending.is_empty()) {
        if Instant::now() >= deadline {
            bail!(
                "verbs-host lane {execution_lane} fanout timed out with {} hosts pending",
                pending_by_host
                    .iter()
                    .filter(|pending| !pending.is_empty())
                    .count()
            );
        }
        let mut made_progress = false;
        for host_index in 0..host_count {
            if pending_by_host[host_index].is_empty() {
                continue;
            }
            made_progress |= sessions[host_index]
                .as_mut()
                .context("verbs-host lane fanout lost an active session")?
                .try_progress_chunk_requests_with_timing(
                    &mut pending_by_host[host_index],
                    config,
                    timing_enabled,
                )?;
        }
        if !made_progress {
            if busy_poll_started.elapsed() < Duration::from_millis(1) {
                std::hint::spin_loop();
                continue;
            }
            let host_index = pending_by_host
                .iter()
                .position(|pending| !pending.is_empty())
                .context("verbs-host lane fanout lost its pending host")?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            let session = sessions[host_index]
                .as_mut()
                .context("verbs-host lane fanout lost its event-wait session")?;
            let stats = session.endpoint.poll_stats(0, 1, remaining)?;
            session.apply_chunk_completion_stats(
                &mut pending_by_host[host_index],
                config,
                stats,
            )?;
        }
    }

    let response_stats_by_stream = response_receivers
        .into_iter()
        .map(|response_rx| {
            response_rx
                .blocking_recv()
                .context("receiving verbs-host lane fanout stream completion")?
        })
        .collect::<Result<Vec<_>>>()?;
    let response_frames = response_stats_by_stream
        .iter()
        .map(|stats| stats.response_frames)
        .sum::<usize>();
    let mut chunks = Vec::with_capacity(response_frames);
    for _ in 0..response_frames {
        chunks.push(
            chunk_rx
                .try_recv()
                .context("verbs-host lane fanout completed without its response chunk")?,
        );
    }
    anyhow::ensure!(
        chunk_rx.try_recv().is_err(),
        "verbs-host lane fanout produced more chunks than its stream completions"
    );
    Ok(VerbsHostProtocolV2LaneFanoutResponse {
        response_stats_by_stream,
        chunks,
    })
}

fn verbs_host_protocol_v2_persistent_client_worker(
    addr: SocketAddr,
    config: TcpTransportConfig,
    execution_lane: u32,
    cq_harvester: Option<Arc<VerbsHostProtocolV2CqHarvester>>,
    rx: mpsc::Receiver<VerbsHostProtocolV2PersistentCommand>,
) {
    let mut session = None;
    let mut deferred = None;
    loop {
        let command = match deferred.take() {
            Some(command) => command,
            None => match rx.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        match command {
            VerbsHostProtocolV2PersistentCommand::Roundtrip {
                request,
                response_tx,
            } => {
                let result = verbs_host_protocol_v2_persistent_client_roundtrip(
                    addr,
                    &config,
                    execution_lane,
                    cq_harvester.as_ref(),
                    &mut session,
                    &request,
                );
                if result.is_err() {
                    session = None;
                }
                let _ = response_tx.send(result);
            }
            VerbsHostProtocolV2PersistentCommand::RoundtripChunks {
                request,
                stream_id,
                chunk_tx,
                response_tx,
            } => {
                let exit = verbs_host_protocol_v2_persistent_client_chunk_pipeline(
                    addr,
                    &config,
                    execution_lane,
                    cq_harvester.as_ref(),
                    &mut session,
                    VerbsHostProtocolV2QueuedChunkCommand {
                        request,
                        stream_id,
                        chunk_tx,
                        response_tx,
                    },
                    &rx,
                );
                match exit {
                    VerbsHostProtocolV2ChunkPipelineExit::Deferred(command) => {
                        deferred = Some(command);
                    }
                    VerbsHostProtocolV2ChunkPipelineExit::Idle => {}
                    VerbsHostProtocolV2ChunkPipelineExit::Disconnected => break,
                }
            }
            VerbsHostProtocolV2PersistentCommand::Reset => {
                session = None;
            }
            VerbsHostProtocolV2PersistentCommand::Shutdown => break,
        }
    }
}

struct VerbsHostProtocolV2QueuedChunkCommand {
    request: ExpertProtocolV2Request,
    stream_id: usize,
    chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
    response_tx: tokio::sync::oneshot::Sender<Result<VerbsHostProtocolV2ResponseStreamStats>>,
}

enum VerbsHostProtocolV2ChunkPipelineExit {
    Deferred(VerbsHostProtocolV2PersistentCommand),
    Idle,
    Disconnected,
}

struct VerbsHostProtocolV2PendingChunkRoundtrip {
    request: ExpertProtocolV2Request,
    stream_id: usize,
    chunk_tx: tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
    response_tx: tokio::sync::oneshot::Sender<Result<VerbsHostProtocolV2ResponseStreamStats>>,
    assembler: ProtocolV2ResponseChunkAssembler,
    response_frames: usize,
    response_wire_bytes: usize,
    total_started: Option<Instant>,
    encode_ms: f64,
    expected_response_ms: f64,
    send_ms: f64,
    poll_ms: f64,
    copy_recv_ms: f64,
    post_recv_ms: f64,
    parse_ms: f64,
}

struct VerbsHostProtocolV2ChunkSubmissionTiming {
    total_started: Option<Instant>,
    encode_ms: f64,
    expected_response_ms: f64,
    send_ms: f64,
}

impl VerbsHostProtocolV2PendingChunkRoundtrip {
    fn new(
        command: VerbsHostProtocolV2QueuedChunkCommand,
        timing: VerbsHostProtocolV2ChunkSubmissionTiming,
    ) -> Self {
        let assembler = ProtocolV2ResponseChunkAssembler::validation_only(&command.request);
        Self {
            request: command.request,
            stream_id: command.stream_id,
            chunk_tx: command.chunk_tx,
            response_tx: command.response_tx,
            assembler,
            response_frames: 0,
            response_wire_bytes: 0,
            total_started: timing.total_started,
            encode_ms: timing.encode_ms,
            expected_response_ms: timing.expected_response_ms,
            send_ms: timing.send_ms,
            poll_ms: 0.0,
            copy_recv_ms: 0.0,
            post_recv_ms: 0.0,
            parse_ms: 0.0,
        }
    }

    fn fail(self, message: &str) {
        let _ = self
            .response_tx
            .send(Err(anyhow::anyhow!(message.to_owned())));
    }
}

fn verbs_host_protocol_v2_persistent_client_chunk_pipeline(
    addr: SocketAddr,
    config: &TcpTransportConfig,
    execution_lane: u32,
    cq_harvester: Option<&Arc<VerbsHostProtocolV2CqHarvester>>,
    session: &mut Option<VerbsHostProtocolV2PersistentClientSession>,
    first: VerbsHostProtocolV2QueuedChunkCommand,
    rx: &mpsc::Receiver<VerbsHostProtocolV2PersistentCommand>,
) -> VerbsHostProtocolV2ChunkPipelineExit {
    let mut queued = VecDeque::from([first]);
    let mut pending = VecDeque::<VerbsHostProtocolV2PendingChunkRoundtrip>::new();
    let mut deferred = None;
    let mut disconnected = false;

    loop {
        while deferred.is_none() && !disconnected {
            match rx.try_recv() {
                Ok(VerbsHostProtocolV2PersistentCommand::RoundtripChunks {
                    request,
                    stream_id,
                    chunk_tx,
                    response_tx,
                }) => queued.push_back(VerbsHostProtocolV2QueuedChunkCommand {
                    request,
                    stream_id,
                    chunk_tx,
                    response_tx,
                }),
                Ok(command) => deferred = Some(command),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => disconnected = true,
            }
        }

        while let Some(command) = queued.front() {
            if session.is_none() {
                match VerbsHostProtocolV2PersistentClientSession::connect(
                    addr,
                    config,
                    &command.request,
                    execution_lane,
                    cq_harvester.cloned(),
                ) {
                    Ok(connected) => *session = Some(connected),
                    Err(error) => {
                        let command = queued.pop_front().expect("queued command is present");
                        let _ = command.response_tx.send(Err(error));
                        continue;
                    }
                }
            }
            let current = session
                .as_ref()
                .expect("persistent verbs-host session connected above");
            match current.fits(&command.request) {
                Ok(true) => {}
                Ok(false) if pending.is_empty() => {
                    *session = None;
                    continue;
                }
                Ok(false) => break,
                Err(error) => {
                    let command = queued.pop_front().expect("queued command is present");
                    let _ = command.response_tx.send(Err(error));
                    continue;
                }
            }
            if pending.len()
                == session
                    .as_ref()
                    .expect("persistent verbs-host session exists")
                    .request_ring
                    .depth
            {
                break;
            }
            let command = queued.pop_front().expect("queued command is present");
            let submission = session
                .as_mut()
                .expect("persistent verbs-host session exists")
                .post_chunk_request(&command.request, config);
            match submission {
                Ok(timing) => {
                    pending.push_back(VerbsHostProtocolV2PendingChunkRoundtrip::new(
                        command, timing,
                    ));
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    let _ = command.response_tx.send(Err(error));
                    fail_verbs_host_chunk_pipeline_pending(&mut pending, &message);
                    *session = None;
                    break;
                }
            }
        }

        if pending.is_empty() {
            if !queued.is_empty() {
                continue;
            }
            return match deferred {
                Some(command) => VerbsHostProtocolV2ChunkPipelineExit::Deferred(command),
                None if disconnected => VerbsHostProtocolV2ChunkPipelineExit::Disconnected,
                None => VerbsHostProtocolV2ChunkPipelineExit::Idle,
            };
        }

        let progress = session
            .as_mut()
            .expect("persistent verbs-host session exists while requests are pending")
            .try_progress_chunk_requests(&mut pending, config);
        match progress {
            Ok(true) => {}
            Ok(false) => {
                let progress = session
                    .as_mut()
                    .expect("persistent verbs-host session exists while requests are pending")
                    .wait_progress_chunk_requests(&mut pending, config);
                if let Err(error) = progress {
                    let message = format!("{error:#}");
                    fail_verbs_host_chunk_pipeline_pending(&mut pending, &message);
                    *session = None;
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                fail_verbs_host_chunk_pipeline_pending(&mut pending, &message);
                *session = None;
            }
        }
    }
}

fn fail_verbs_host_chunk_pipeline_pending(
    pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
    message: &str,
) {
    while let Some(request) = pending.pop_front() {
        request.fail(message);
    }
}

fn verbs_host_protocol_v2_persistent_client_roundtrip(
    addr: SocketAddr,
    config: &TcpTransportConfig,
    execution_lane: u32,
    cq_harvester: Option<&Arc<VerbsHostProtocolV2CqHarvester>>,
    session: &mut Option<VerbsHostProtocolV2PersistentClientSession>,
    request: &ExpertProtocolV2Request,
) -> Result<Vec<u8>> {
    match verbs_host_protocol_v2_persistent_client_roundtrip_once(
        addr,
        config,
        execution_lane,
        cq_harvester,
        session,
        request,
    ) {
        Ok(frame) => Ok(frame),
        Err(error) if is_verbs_host_protocol_v2_persistent_retryable_error(&error) => {
            if config.timing {
                eprintln!(
                    "protocol_v2_verbs_persistent_client_retry addr={} request_id={} layer_id={} rows={} routes={} error={:#}",
                    addr,
                    request.header.request_id,
                    request.header.layer_id,
                    request.header.row_count,
                    request.header.route_count,
                    error
                );
            }
            tracing::warn!(
                addr = %addr,
                error = %error,
                "verbs-host ProtocolV2 persistent client reconnecting after stale session"
            );
            *session = None;
            verbs_host_protocol_v2_persistent_client_roundtrip_once(
                addr,
                config,
                execution_lane,
                cq_harvester,
                session,
                request,
            )
        }
        Err(error) => Err(error),
    }
}

fn verbs_host_protocol_v2_persistent_client_roundtrip_once(
    addr: SocketAddr,
    config: &TcpTransportConfig,
    execution_lane: u32,
    cq_harvester: Option<&Arc<VerbsHostProtocolV2CqHarvester>>,
    session: &mut Option<VerbsHostProtocolV2PersistentClientSession>,
    request: &ExpertProtocolV2Request,
) -> Result<Vec<u8>> {
    let timing_enabled = config.timing;
    let total_started = timing_enabled.then(Instant::now);
    if session
        .as_ref()
        .map(|session| session.fits(request))
        .transpose()?
        == Some(false)
    {
        *session = None;
    }
    let had_session = session.is_some();
    let mut connect_ms = 0.0_f64;
    if session.is_none() {
        let connect_started = timing_enabled.then(Instant::now);
        *session = Some(VerbsHostProtocolV2PersistentClientSession::connect(
            addr,
            config,
            request,
            execution_lane,
            cq_harvester.cloned(),
        )?);
        connect_ms = elapsed_ms_optional(connect_started);
    }
    let frame = session
        .as_mut()
        .expect("persistent verbs-host session is initialized above")
        .roundtrip_response_frame(request, config)?;
    if timing_enabled {
        eprintln!(
            "protocol_v2_verbs_persistent_client_session_timing addr={} execution_lane={} request_id={} layer_id={} rows={} routes={} had_session={} connect_ms={:.3} total_ms={:.3}",
            addr,
            execution_lane,
            request.header.request_id,
            request.header.layer_id,
            request.header.row_count,
            request.header.route_count,
            had_session,
            connect_ms,
            elapsed_ms_optional(total_started)
        );
    }
    Ok(frame)
}

fn is_verbs_host_protocol_v2_persistent_retryable_error(error: &anyhow::Error) -> bool {
    is_connection_closed(error)
        || is_verbs_host_control_plane_closed(error)
        || is_protocol_v2_response_request_id_mismatch(error)
        || is_verbs_host_rdma_completion_error(error)
}

fn is_verbs_host_control_plane_closed(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("verbs-host ProtocolV2 control plane closed")
    })
}

fn is_protocol_v2_response_request_id_mismatch(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.starts_with("ProtocolV2 response request_id ")
            && message.contains(" did not match request_id ")
    })
}

fn is_verbs_host_rdma_completion_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("ds41rt_rdma_rc_endpoint_poll returned status")
            || message.contains("RDMA RC endpoint completion returned non-success status")
    })
}

fn is_verbs_host_rdma_poll_timeout(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("RDMA RC endpoint timed out waiting for completions")
    })
}

struct VerbsHostProtocolV2PersistentClientSession {
    _stream: TcpStream,
    addr: SocketAddr,
    endpoint: Arc<NativeRdmaEndpoint>,
    retained_response_ring: Option<RegisteredResponseRing>,
    cq_harvester: Option<Arc<VerbsHostProtocolV2CqHarvester>>,
    cq_waiter: Option<VerbsHostProtocolV2CqWaiter>,
    request_capacity_wire_bytes: usize,
    response_capacity_wire_bytes: usize,
    request_ring: VerbsHostRdmaRing,
    response_ring: VerbsHostRdmaRing,
    request_send_sequence: usize,
    request_send_in_flight: usize,
    response_recv_sequence: usize,
    request_frame: ExpertProtocolV2FrameBuffer,
    response_frame_pool: Vec<Vec<u8>>,
    response_frame_recycle_tx: mpsc::Sender<Vec<u8>>,
    response_frame_recycle_rx: mpsc::Receiver<Vec<u8>>,
    pinned_response_frame_pool: Vec<VerbsHostProtocolV2PinnedResponseFrame>,
    pinned_response_frame_recycle_tx: mpsc::Sender<VerbsHostProtocolV2PinnedResponseFrame>,
    pinned_response_frame_recycle_rx: mpsc::Receiver<VerbsHostProtocolV2PinnedResponseFrame>,
}

struct ProtocolV2ResponseChunkAssembler {
    request_row_count: usize,
    stream_frame: bool,
    row_sharded_reduction: bool,
    header: Option<ExpertProtocolV2ResponseHeader>,
    debug_checksum_enabled: Option<bool>,
    partial_output_payload: Option<Vec<u8>>,
    completed_rows: Vec<bool>,
    stream_completed_rows: Vec<usize>,
    completed_row_count: usize,
    final_chunk_received: bool,
}

impl ProtocolV2ResponseChunkAssembler {
    fn new(request: &ExpertProtocolV2Request) -> Self {
        Self::with_payload_assembly(request, true)
    }

    fn validation_only(request: &ExpertProtocolV2Request) -> Self {
        Self::with_payload_assembly(request, false)
    }

    fn with_payload_assembly(request: &ExpertProtocolV2Request, assemble_payload: bool) -> Self {
        let request_row_count = request.header.row_count as usize;
        let stream_frame = request.stream_plan_enabled() || request.stream_data_enabled();
        Self {
            request_row_count,
            stream_frame,
            row_sharded_reduction: request.spark_row_sharded_reduction_enabled(),
            header: None,
            debug_checksum_enabled: None,
            partial_output_payload: assemble_payload.then(Vec::new),
            completed_rows: if stream_frame {
                Vec::new()
            } else {
                vec![false; request_row_count]
            },
            stream_completed_rows: Vec::new(),
            completed_row_count: 0,
            final_chunk_received: false,
        }
    }

    fn accept(
        &mut self,
        request: &ExpertProtocolV2Request,
        chunk: &ExpertProtocolV2ResponseView<'_>,
    ) -> Result<()> {
        if self.final_chunk_received {
            bail!("ProtocolV2 response chunk arrived after the final chunk");
        }
        validate_response_matches_request(&chunk.header, request)?;
        if chunk.debug_checksum_enabled() {
            chunk.verify_checksum()?;
        }
        if self.stream_frame {
            return self.accept_stream_frame(chunk);
        }
        if chunk.header.row_count == 0 {
            anyhow::ensure!(
                request.spark_reduction_enabled(),
                "ProtocolV2 response chunk must contain at least one row"
            );
            anyhow::ensure!(
                self.partial_output_payload.is_none(),
                "Spark reduction follower acknowledgements require streaming response validation"
            );
            anyhow::ensure!(
                !chunk.more_chunks(),
                "Spark reduction follower acknowledgement cannot mark more chunks"
            );
            self.validate_or_initialize_metadata(chunk)?;
            self.final_chunk_received = true;
            return Ok(());
        }
        if !chunk.row_indexed() && chunk.header.row_count as usize != self.request_row_count {
            bail!(
                "ProtocolV2 non-indexed response row count {} did not match request row count {}",
                chunk.header.row_count,
                self.request_row_count
            );
        }

        self.validate_or_initialize_metadata(chunk)?;
        for chunk_row in 0..chunk.header.row_count as usize {
            let request_row = chunk.request_row_index(chunk_row)? as usize;
            if request_row >= self.request_row_count {
                bail!(
                    "ProtocolV2 response request row index {request_row} exceeded request row count {}",
                    self.request_row_count
                );
            }
            if self.completed_rows[request_row] {
                bail!("ProtocolV2 response request row index {request_row} was emitted twice");
            }
            let stride = chunk.header.output_row_stride_bytes as usize;
            let output_start = request_row
                .checked_mul(stride)
                .context("ProtocolV2 assembled response row offset overflow")?;
            let output_end = output_start
                .checked_add(stride)
                .context("ProtocolV2 assembled response row range overflow")?;
            if let Some(partial_output_payload) = &mut self.partial_output_payload {
                partial_output_payload[output_start..output_end]
                    .copy_from_slice(chunk.partial_output_row_payload(chunk_row)?);
            }
            self.completed_rows[request_row] = true;
            self.completed_row_count += 1;
        }

        if chunk.more_chunks() {
            if self.completed_row_count == self.request_row_count {
                bail!("ProtocolV2 response marked more chunks after completing every request row");
            }
        } else {
            self.final_chunk_received = true;
            if self.completed_row_count != self.request_row_count && !self.row_sharded_reduction {
                bail!(
                    "ProtocolV2 final response chunk completed {} of {} request rows",
                    self.completed_row_count,
                    self.request_row_count
                );
            }
            if self.row_sharded_reduction && self.completed_row_count == 0 {
                bail!("ProtocolV2 row-sharded reduction returned no rows");
            }
        }
        Ok(())
    }

    fn accept_stream_frame(&mut self, chunk: &ExpertProtocolV2ResponseView<'_>) -> Result<()> {
        if self.partial_output_payload.is_some() {
            bail!("ProtocolV2 streamed-ingress frames only support response validation");
        }
        if chunk.header.row_count > 0 && !chunk.row_indexed() {
            bail!("ProtocolV2 streamed-ingress response rows must carry logical row indices");
        }
        if chunk.header.row_count == 0 && chunk.more_chunks() {
            bail!("ProtocolV2 streamed-ingress zero-row acknowledgement cannot mark more chunks");
        }
        self.validate_or_initialize_metadata(chunk)?;
        for chunk_row in 0..chunk.header.row_count as usize {
            let request_row = chunk.request_row_index(chunk_row)? as usize;
            if self.stream_completed_rows.contains(&request_row) {
                bail!(
                    "ProtocolV2 streamed-ingress response row index {request_row} was emitted twice in one frame"
                );
            }
            self.stream_completed_rows.push(request_row);
            self.completed_row_count += 1;
        }
        if !chunk.more_chunks() {
            self.final_chunk_received = true;
        }
        Ok(())
    }

    fn validate_or_initialize_metadata(
        &mut self,
        chunk: &ExpertProtocolV2ResponseView<'_>,
    ) -> Result<()> {
        let debug_checksum_enabled = chunk.debug_checksum_enabled();
        let Some(header) = &self.header else {
            let payload_bytes = self
                .request_row_count
                .checked_mul(chunk.header.output_row_stride_bytes as usize)
                .context("ProtocolV2 assembled response payload byte count overflow")?;
            if let Some(partial_output_payload) = &mut self.partial_output_payload {
                partial_output_payload.resize(payload_bytes, 0);
            }
            self.header = Some(chunk.header.clone());
            self.debug_checksum_enabled = Some(debug_checksum_enabled);
            return Ok(());
        };

        if header.output_dim != chunk.header.output_dim
            || header.output_dtype != chunk.header.output_dtype
            || header.output_row_stride_bytes != chunk.header.output_row_stride_bytes
            || header.status != chunk.header.status
            || header.executor_id != chunk.header.executor_id
            || self.debug_checksum_enabled != Some(debug_checksum_enabled)
        {
            bail!("ProtocolV2 response chunk metadata changed within one request");
        }
        Ok(())
    }

    fn finish(self) -> Result<ExpertProtocolV2Response> {
        if self.stream_frame {
            bail!("ProtocolV2 streamed-ingress frames cannot be assembled as monolithic responses");
        }
        let partial_output_payload = self
            .partial_output_payload
            .context("ProtocolV2 response assembler was configured for validation only")?;
        if !self.final_chunk_received {
            bail!("ProtocolV2 response ended without a final chunk");
        }
        let header = self
            .header
            .context("ProtocolV2 response ended without any chunks")?;
        let response = ExpertProtocolV2Response::new_with_output_stride(
            header.request_id,
            header.placement_version,
            header.layer_id,
            self.request_row_count as u32,
            header.output_dim,
            header.output_dtype,
            header.output_row_stride_bytes,
            header.status,
            partial_output_payload,
        )?
        .with_executor_id(header.executor_id);
        Ok(if self.debug_checksum_enabled == Some(true) {
            response.with_debug_checksum()
        } else {
            response
        })
    }

    fn finish_validation(self) -> Result<ExpertProtocolV2ResponseHeader> {
        if !self.final_chunk_received {
            bail!("ProtocolV2 response ended without a final chunk");
        }
        self.header
            .context("ProtocolV2 response ended without any chunks")
    }
}

impl VerbsHostProtocolV2PersistentClientSession {
    fn connect(
        addr: SocketAddr,
        config: &TcpTransportConfig,
        request: &ExpertProtocolV2Request,
        execution_lane: u32,
        cq_harvester: Option<Arc<VerbsHostProtocolV2CqHarvester>>,
    ) -> Result<Self> {
        Self::connect_impl(addr, config, request, execution_lane, cq_harvester, false)
    }

    fn connect_local(addr: SocketAddr, config: &TcpTransportConfig,
                     request: &ExpertProtocolV2Request) -> Result<Self> {
        Self::connect_impl(addr, config, request, 0, None, true)
    }

    fn connect_impl(addr: SocketAddr, config: &TcpTransportConfig,
                    request: &ExpertProtocolV2Request, execution_lane: u32,
                    cq_harvester: Option<Arc<VerbsHostProtocolV2CqHarvester>>,
                    retain_final_response: bool) -> Result<Self> {
        verbs_host_preflight()?;
        let native_path = verbs_host_native_library_path().context(
            "native library not found; set DS41RT_NATIVE_LIB or build native/libds41rt_native.so with RDMA",
        )?;
        let library = Arc::new(unsafe { NativeLibrary::load(&native_path) }?);
        let request_wire_bytes = request.wire_stats().wire_bytes;
        let expected_response_wire_bytes = verbs_host_expected_response_wire_bytes(request)?;
        let (request_ring, response_ring) =
            verbs_host_persistent_rings(config, request_wire_bytes, expected_response_wire_bytes)?;
        let request_capacity_wire_bytes = request_ring.slot_capacity_bytes;
        let response_capacity_wire_bytes = response_ring.slot_capacity_bytes;
        let request_registered_span_bytes = request_ring.registered_span_bytes;
        let response_registered_span_bytes = response_ring.registered_span_bytes;
        let endpoint = if retain_final_response {
            NativeRdmaEndpoint::create_from_wire_bytes_mapped_for_ring(
                Arc::clone(&library), "client", request_capacity_wire_bytes,
                response_capacity_wire_bytes, request_registered_span_bytes,
                response_registered_span_bytes, next_local_psn("client"), response_ring.depth,
            )?
        } else {
            NativeRdmaEndpoint::create_from_wire_bytes(
                Arc::clone(&library), "client", request_capacity_wire_bytes,
                response_capacity_wire_bytes, request_registered_span_bytes,
                response_registered_span_bytes, next_local_psn("client"),
            )?
        };
        let peer = addr.to_string();
        let mut stream = connect_control_stream(&peer, config.timeout)?;
        configure_control_stream(&stream, config.timeout)?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let (client_host, _server_host) =
            distinct_endpoint_hosts(local_control_host("client"), peer.clone());
        let start = VerbsHostProtocolV2PersistentStart {
            message: "protocol_v2_persistent_start".to_owned(),
            execution_lane,
            request_capacity_wire_bytes,
            response_capacity_wire_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            ring_depth: request_ring.depth,
            request_slot_stride_bytes: request_ring.slot_stride_bytes,
            response_slot_stride_bytes: response_ring.slot_stride_bytes,
            client_endpoint: endpoint.verbs_descriptor("client", &client_host),
            client_native_endpoint: endpoint.native_descriptor(),
        };
        write_control(&mut stream, &start)?;
        let ready: VerbsHostProtocolV2PersistentReady = read_control(&mut reader)?;
        if ready.message != "protocol_v2_persistent_ready" {
            bail!(
                "verbs-host ProtocolV2 persistent client received invalid ready message {}",
                ready.message
            );
        }
        validate_persistent_endpoint_capacity(
            &start.client_endpoint,
            "client",
            request_capacity_wire_bytes,
            response_capacity_wire_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            request_ring.depth,
        )?;
        validate_persistent_endpoint_capacity(
            &ready.server_endpoint,
            "server",
            response_capacity_wire_bytes,
            request_capacity_wire_bytes,
            response_registered_span_bytes,
            request_registered_span_bytes,
            response_ring.depth,
        )?;
        if config.timing {
            let client_native = endpoint.native_descriptor();
            eprintln!(
                "protocol_v2_verbs_persistent_client_connect addr={} ring_depth={} request_capacity={} request_stride={} request_span={} response_capacity={} response_stride={} response_span={} client_device={} client_gid={} client_status=\"{}\" server_device={} server_gid={} server_status=\"{}\"",
                addr,
                request_ring.depth,
                request_capacity_wire_bytes,
                request_ring.slot_stride_bytes,
                request_ring.registered_span_bytes,
                response_capacity_wire_bytes,
                response_ring.slot_stride_bytes,
                response_ring.registered_span_bytes,
                client_native.device_name,
                client_native.gid_hex,
                client_native.status,
                ready.server_native_endpoint.device_name,
                ready.server_native_endpoint.gid_hex,
                ready.server_native_endpoint.status
            );
        }
        endpoint.connect(&ready.server_native_endpoint)?;
        for slot in 0..response_ring.depth {
            endpoint.post_recv_at(
                response_ring.slot_offset(slot),
                response_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
        }
        let (response_frame_recycle_tx, response_frame_recycle_rx) = mpsc::channel();
        let (pinned_response_frame_recycle_tx, pinned_response_frame_recycle_rx) = mpsc::channel();
        let retained_response_ring = if retain_final_response {
            Some(RegisteredResponseRing::new(&endpoint, response_ring)?)
        } else { None };
        Ok(Self {
            _stream: stream,
            addr,
            endpoint: Arc::new(endpoint),
            retained_response_ring,
            cq_waiter: cq_harvester
                .as_ref()
                .map(|_| VerbsHostProtocolV2CqWaiter::new()),
            cq_harvester,
            request_capacity_wire_bytes,
            response_capacity_wire_bytes,
            request_ring,
            response_ring,
            request_send_sequence: 0,
            request_send_in_flight: 0,
            response_recv_sequence: 0,
            request_frame: ExpertProtocolV2FrameBuffer::with_capacity(request_capacity_wire_bytes),
            response_frame_pool: vec![Vec::new()],
            response_frame_recycle_tx,
            response_frame_recycle_rx,
            pinned_response_frame_pool: Vec::new(),
            pinned_response_frame_recycle_tx,
            pinned_response_frame_recycle_rx,
        })
    }

    fn take_response_frame(
        &mut self,
        response_wire_bytes: usize,
        pinned: bool,
    ) -> Result<VerbsHostProtocolV2ResponseFrame> {
        if pinned {
            self.pinned_response_frame_pool
                .extend(self.pinned_response_frame_recycle_rx.try_iter());
            if let Some(index) = self
                .pinned_response_frame_pool
                .iter()
                .position(|frame| frame.buffer.bytes >= response_wire_bytes)
            {
                let mut frame = self.pinned_response_frame_pool.swap_remove(index);
                frame.resize(response_wire_bytes)?;
                return Ok(VerbsHostProtocolV2ResponseFrame::Pinned(frame));
            }
            // Replace one undersized recycled frame instead of retaining an
            // allocation for every response size encountered over process
            // lifetime. Concurrent in-flight frames still determine the
            // stable high-water mark.
            drop(self.pinned_response_frame_pool.pop());
            return Ok(VerbsHostProtocolV2ResponseFrame::Pinned(
                VerbsHostProtocolV2PinnedResponseFrame::new(
                    Arc::clone(&self.endpoint.library),
                    response_wire_bytes,
                )?,
            ));
        }
        self.response_frame_pool
            .extend(self.response_frame_recycle_rx.try_iter());
        let mut response_frame = self.response_frame_pool.pop().unwrap_or_default();
        response_frame.resize(response_wire_bytes, 0);
        Ok(VerbsHostProtocolV2ResponseFrame::Bytes(response_frame))
    }

    fn recycle_response_frame(&mut self, response_frame: VerbsHostProtocolV2ResponseFrame) {
        match response_frame {
            VerbsHostProtocolV2ResponseFrame::Bytes(mut response_frame) => {
                response_frame.clear();
                self.response_frame_pool.push(response_frame);
            }
            VerbsHostProtocolV2ResponseFrame::Pinned(response_frame) => {
                self.pinned_response_frame_pool.push(response_frame);
            }
            VerbsHostProtocolV2ResponseFrame::Registered(frame) => drop(frame),
        }
    }

    fn fits(&self, request: &ExpertProtocolV2Request) -> Result<bool> {
        let request_wire_bytes = request.wire_stats().wire_bytes;
        let response_wire_bytes = verbs_host_expected_response_wire_bytes(request)?;
        Ok(request_wire_bytes <= self.request_capacity_wire_bytes
            && response_wire_bytes <= self.response_capacity_wire_bytes)
    }

    fn post_chunk_request(
        &mut self,
        request: &ExpertProtocolV2Request,
        config: &TcpTransportConfig,
    ) -> Result<VerbsHostProtocolV2ChunkSubmissionTiming> {
        if let Some(ring) = &mut self.retained_response_ring {
            ring.reclaim_before_request(&self.endpoint, self.response_ring)?;
        }
        let timing_enabled = config.timing;
        let total_started = timing_enabled.then(Instant::now);
        let encode_started = timing_enabled.then(Instant::now);
        let request_prefix = self.request_frame.encode_request_prefix(request)?;
        let request_wire_bytes = request_prefix
            .len()
            .checked_add(request.hidden_payload.len())
            .context("persistent verbs-host ProtocolV2 request byte count overflow")?;
        let encode_ms = elapsed_ms_optional(encode_started);
        let expected_started = timing_enabled.then(Instant::now);
        let expected_response_wire_bytes = verbs_host_expected_response_wire_bytes(request)?;
        let expected_response_ms = elapsed_ms_optional(expected_started);
        if request_wire_bytes > config.max_frame_bytes
            || request_wire_bytes > self.request_capacity_wire_bytes
        {
            bail!(
                "persistent verbs-host ProtocolV2 request frame length {} exceeds capacity {}",
                request_wire_bytes,
                self.request_capacity_wire_bytes.min(config.max_frame_bytes)
            );
        }
        if expected_response_wire_bytes > config.max_frame_bytes
            || expected_response_wire_bytes > self.response_capacity_wire_bytes
        {
            bail!(
                "persistent verbs-host ProtocolV2 expected response frame length {} exceeds capacity {}",
                expected_response_wire_bytes,
                self.response_capacity_wire_bytes.min(config.max_frame_bytes)
            );
        }
        if self.request_send_in_flight == self.request_ring.depth {
            let stats = self
                .endpoint
                .poll_stats(1, 0, config.timeout)
                .context("waiting for a persistent verbs-host request send slot")?;
            let send_completions = stats.send_completions as usize;
            anyhow::ensure!(
                send_completions > 0 && send_completions <= self.request_send_in_flight,
                "persistent verbs-host reclaimed {send_completions} sends with {} in flight",
                self.request_send_in_flight
            );
            self.request_send_in_flight -= send_completions;
        }
        let send_started = timing_enabled.then(Instant::now);
        let request_send_slot = self.request_send_sequence % self.request_ring.depth;
        self.endpoint.send_parts_at(
            request_prefix,
            &request.hidden_payload,
            self.request_ring.slot_offset(self.request_send_sequence),
            VERBS_HOST_SEND_WR_ID + request_send_slot as u64,
        )?;
        self.request_send_sequence = self.request_send_sequence.wrapping_add(1);
        self.request_send_in_flight += 1;
        Ok(VerbsHostProtocolV2ChunkSubmissionTiming {
            total_started,
            encode_ms,
            expected_response_ms,
            send_ms: elapsed_ms_optional(send_started),
        })
    }

    fn try_progress_chunk_requests(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        config: &TcpTransportConfig,
    ) -> Result<bool> {
        self.try_progress_chunk_requests_with_timing(
            pending,
            config,
            config.timing,
        )
    }

    fn try_progress_chunk_requests_with_timing(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        config: &TcpTransportConfig,
        timing_enabled: bool,
    ) -> Result<bool> {
        let poll_started = timing_enabled.then(Instant::now);
        let stats = self.endpoint.try_poll(
            self.request_send_in_flight as u32,
            self.response_ring.depth as u32,
        )?;
        if let Some(front) = pending.front_mut() {
            front.poll_ms += elapsed_ms_optional(poll_started);
        }
        let progressed = stats.send_completions > 0 || stats.recv_completions > 0;
        self.apply_chunk_completion_stats(pending, config, stats)?;
        Ok(progressed)
    }

    fn wait_progress_chunk_requests(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        config: &TcpTransportConfig,
    ) -> Result<()> {
        let timing_enabled = config.timing;
        if let (Some(harvester), Some(waiter)) = (&self.cq_harvester, &self.cq_waiter) {
            let poll_started = timing_enabled.then(Instant::now);
            let stats = harvester.wait_for_response(
                &self.endpoint,
                waiter,
                self.request_send_in_flight as u32,
                self.response_ring.depth as u32,
                config.timeout,
            )?;
            if let Some(front) = pending.front_mut() {
                front.poll_ms += elapsed_ms_optional(poll_started);
            }
            self.apply_chunk_completion_stats(pending, config, stats)?;
            return Ok(());
        }
        let started = Instant::now();
        loop {
            if self.try_progress_chunk_requests_with_timing(pending, config, timing_enabled)? {
                return Ok(());
            }
            if started.elapsed() >= config.timeout {
                bail!(
                    "persistent verbs-host active request timed out after {:?} with {} requests pending",
                    config.timeout,
                    pending.len()
                );
            }
            std::hint::spin_loop();
        }
    }

    fn apply_chunk_completion_stats(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        config: &TcpTransportConfig,
        stats: Ds41rtRdmaRcCompletionStats,
    ) -> Result<()> {
        let send_completions = stats.send_completions as usize;
        if send_completions > self.request_send_in_flight {
            bail!(
                "persistent verbs-host observed {} send completions with only {} sends in flight",
                send_completions,
                self.request_send_in_flight
            );
        }
        self.request_send_in_flight -= send_completions;
        for _ in 0..stats.recv_completions {
            self.accept_chunk_response_frame(pending, config)?;
        }
        Ok(())
    }

    fn accept_chunk_response_frame(
        &mut self,
        pending: &mut VecDeque<VerbsHostProtocolV2PendingChunkRoundtrip>,
        config: &TcpTransportConfig,
    ) -> Result<()> {
        let timing_enabled = config.timing;
        let response_recv_slot = self.response_recv_sequence % self.response_ring.depth;
        let response_recv_offset = self.response_ring.slot_offset(self.response_recv_sequence);
        let copy_started = timing_enabled.then(Instant::now);
        let mut response_header = [0_u8; EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN];
        self.endpoint.copy_recv_at(
            &mut response_header,
            response_recv_offset,
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
        )?;
        let response_wire_bytes =
            ExpertProtocolV2Response::wire_bytes_from_header(&response_header)?;
        if response_wire_bytes > config.max_frame_bytes
            || response_wire_bytes > self.response_capacity_wire_bytes
        {
            bail!(
                "persistent verbs-host ProtocolV2 response chunk length {} exceeds capacity {}",
                response_wire_bytes,
                self.response_capacity_wire_bytes
                    .min(config.max_frame_bytes)
            );
        }
        let retained = match &mut self.retained_response_ring {
            Some(ring) => ring.try_retain_final(&self.endpoint, self.response_ring,
                                              self.response_recv_sequence, response_wire_bytes)?,
            None => None,
        };
        let response_frame = if let Some(frame) = retained {
            VerbsHostProtocolV2ResponseFrame::Registered(frame)
        } else {
            let mut frame = self.take_response_frame(response_wire_bytes, true)?;
            self.endpoint.copy_recv_at(frame.as_mut(), response_recv_offset, response_wire_bytes)?;
            frame
        };
        let copy_recv_ms = elapsed_ms_optional(copy_started);
        let post_recv_started = timing_enabled.then(Instant::now);
        if !matches!(&response_frame, VerbsHostProtocolV2ResponseFrame::Registered(_)) {
            self.endpoint.post_recv_at(response_recv_offset, self.response_ring.slot_capacity_bytes,
                                      VERBS_HOST_RECV_WR_ID + response_recv_slot as u64)?;
        }
        self.response_recv_sequence = self.response_recv_sequence.wrapping_add(1);
        let post_recv_ms = elapsed_ms_optional(post_recv_started);

        let parse_started = timing_enabled.then(Instant::now);
        let view = ExpertProtocolV2ResponseView::parse(response_frame.as_ref())
            .context("validating pipelined persistent verbs-host ProtocolV2 response chunk")?;
        let more_chunks = view.more_chunks();
        let front = pending
            .front_mut()
            .context("persistent verbs-host response arrived without a pending request")?;
        front.assembler.accept(&front.request, &view)?;
        let (header, row_indices, payload_start, payload_end) =
            response_chunk_metadata_from_view(&view, response_frame.as_ref())?;
        let partial_output_payload = VerbsHostProtocolV2ResponsePayload::from_response_frame(
            response_frame,
            payload_start,
            payload_end,
            self.response_frame_recycle_tx.clone(),
            self.pinned_response_frame_recycle_tx.clone(),
        )?;
        front
            .chunk_tx
            .send(VerbsHostProtocolV2ResponseChunk {
                stream_id: front.stream_id,
                header,
                row_indices,
                partial_output_payload,
                wire_bytes: response_wire_bytes,
            })
            .context("forwarding pipelined persistent verbs-host ProtocolV2 response chunk")?;
        front.response_frames += 1;
        front.response_wire_bytes = front
            .response_wire_bytes
            .checked_add(response_wire_bytes)
            .context("persistent verbs-host received response byte count overflow")?;
        front.copy_recv_ms += copy_recv_ms;
        front.post_recv_ms += post_recv_ms;
        front.parse_ms += elapsed_ms_optional(parse_started);
        if more_chunks {
            return Ok(());
        }

        let completed = pending
            .pop_front()
            .expect("persistent verbs-host pending request is present");
        let response_executor_id = completed.assembler.finish_validation()?.executor_id;
        if timing_enabled {
            eprintln!(
                "protocol_v2_verbs_persistent_client_roundtrip_timing addr={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_frames={} received_wire_bytes={} assembled_wire_bytes=0 encode_ms={:.3} expected_response_ms={:.3} post_recv_ms={:.3} send_ms={:.3} poll_ms={:.3} copy_recv_ms={:.3} parse_ms={:.3} assemble_ms=0.000 total_ms={:.3} pipelined=true",
                self.addr,
                completed.request.header.request_id,
                completed.request.header.layer_id,
                completed.request.header.row_count,
                completed.request.header.route_count,
                completed.request.wire_stats().wire_bytes,
                completed.response_frames,
                completed.response_wire_bytes,
                completed.encode_ms,
                completed.expected_response_ms,
                completed.post_recv_ms,
                completed.send_ms,
                completed.poll_ms,
                completed.copy_recv_ms,
                completed.parse_ms,
                elapsed_ms_optional(completed.total_started)
            );
        }
        let _ = completed
            .response_tx
            .send(Ok(VerbsHostProtocolV2ResponseStreamStats {
                response_frames: completed.response_frames,
                response_wire_bytes: completed.response_wire_bytes,
                response_executor_id,
            }));
        Ok(())
    }

    fn roundtrip_response_frame(
        &mut self,
        request: &ExpertProtocolV2Request,
        config: &TcpTransportConfig,
    ) -> Result<Vec<u8>> {
        let (response, _) = self.roundtrip_response_frames(request, config, None)?;
        response.context("persistent verbs-host ProtocolV2 response assembly was disabled")
    }

    fn roundtrip_response_frames(
        &mut self,
        request: &ExpertProtocolV2Request,
        config: &TcpTransportConfig,
        mut stream: Option<(
            usize,
            &tokio::sync::mpsc::UnboundedSender<VerbsHostProtocolV2ResponseChunk>,
            &mut usize,
        )>,
    ) -> Result<(Option<Vec<u8>>, VerbsHostProtocolV2ResponseStreamStats)> {
        let timing_enabled = config.timing;
        let streaming = stream.is_some();
        let total_started = timing_enabled.then(Instant::now);
        let encode_started = timing_enabled.then(Instant::now);
        let request_prefix = self.request_frame.encode_request_prefix(request)?;
        let request_wire_bytes = request_prefix
            .len()
            .checked_add(request.hidden_payload.len())
            .context("persistent verbs-host ProtocolV2 request byte count overflow")?;
        let encode_ms = elapsed_ms_optional(encode_started);
        let expected_started = timing_enabled.then(Instant::now);
        let expected_response_wire_bytes = verbs_host_expected_response_wire_bytes(request)?;
        let expected_response_ms = elapsed_ms_optional(expected_started);
        if request_wire_bytes > config.max_frame_bytes {
            bail!(
                "persistent verbs-host ProtocolV2 request frame length {} exceeds max frame bytes {}",
                request_wire_bytes,
                config.max_frame_bytes
            );
        }
        if expected_response_wire_bytes > config.max_frame_bytes {
            bail!(
                "persistent verbs-host ProtocolV2 expected response frame length {} exceeds max frame bytes {}",
                expected_response_wire_bytes,
                config.max_frame_bytes
            );
        }
        if request_wire_bytes > self.request_capacity_wire_bytes {
            bail!(
                "persistent verbs-host ProtocolV2 request frame length {} exceeds endpoint request capacity {}",
                request_wire_bytes,
                self.request_capacity_wire_bytes
            );
        }
        if expected_response_wire_bytes > self.response_capacity_wire_bytes {
            bail!(
                "persistent verbs-host ProtocolV2 response frame length {} exceeds endpoint response capacity {}",
                expected_response_wire_bytes,
                self.response_capacity_wire_bytes
            );
        }
        let request_frame_bytes = request_wire_bytes;
        let mut post_recv_ms = 0.0_f64;
        let send_started = timing_enabled.then(Instant::now);
        let request_send_slot = self.request_send_sequence % self.request_ring.depth;
        self.endpoint.send_parts_at(
            request_prefix,
            &request.hidden_payload,
            self.request_ring.slot_offset(self.request_send_sequence),
            VERBS_HOST_SEND_WR_ID + request_send_slot as u64,
        )?;
        self.request_send_sequence = self.request_send_sequence.wrapping_add(1);
        let send_ms = elapsed_ms_optional(send_started);
        let mut poll_ms = 0.0_f64;
        let mut copy_recv_ms = 0.0_f64;
        let mut parse_ms = 0.0_f64;
        let mut response_frames = 0_usize;
        let mut received_wire_bytes = 0_usize;
        let mut assembler = if streaming {
            ProtocolV2ResponseChunkAssembler::validation_only(request)
        } else {
            ProtocolV2ResponseChunkAssembler::new(request)
        };
        loop {
            let poll_started = timing_enabled.then(Instant::now);
            self.endpoint
                .poll(u32::from(response_frames == 0), 1, config.timeout)?;
            poll_ms += elapsed_ms_optional(poll_started);

            let copy_started = timing_enabled.then(Instant::now);
            let response_recv_slot = self.response_recv_sequence % self.response_ring.depth;
            let response_recv_offset = self.response_ring.slot_offset(self.response_recv_sequence);
            let mut response_header = [0_u8; EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN];
            self.endpoint.copy_recv_at(
                &mut response_header,
                response_recv_offset,
                EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
            )?;
            let response_wire_bytes =
                ExpertProtocolV2Response::wire_bytes_from_header(&response_header)?;
            if response_wire_bytes > config.max_frame_bytes
                || response_wire_bytes > self.response_capacity_wire_bytes
            {
                bail!(
                    "persistent verbs-host ProtocolV2 response chunk length {} exceeds capacity {}",
                    response_wire_bytes,
                    self.response_capacity_wire_bytes
                        .min(config.max_frame_bytes)
                );
            }
            let mut response_frame = self.take_response_frame(response_wire_bytes, streaming)?;
            self.endpoint.copy_recv_at(
                response_frame.as_mut(),
                response_recv_offset,
                response_wire_bytes,
            )?;
            copy_recv_ms += elapsed_ms_optional(copy_started);
            let post_recv_started = timing_enabled.then(Instant::now);
            self.endpoint.post_recv_at(
                response_recv_offset,
                self.response_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + response_recv_slot as u64,
            )?;
            self.response_recv_sequence = self.response_recv_sequence.wrapping_add(1);
            post_recv_ms += elapsed_ms_optional(post_recv_started);

            let parse_started = timing_enabled.then(Instant::now);
            let view = ExpertProtocolV2ResponseView::parse(response_frame.as_ref())
                .context("validating persistent verbs-host ProtocolV2 response chunk")?;
            let more_chunks = view.more_chunks();
            assembler.accept(request, &view)?;
            if let Some((stream_id, chunk_tx, emitted_frames)) = stream.as_mut() {
                let (header, row_indices, payload_start, payload_end) =
                    response_chunk_metadata_from_view(&view, response_frame.as_ref())?;
                let partial_output_payload =
                    VerbsHostProtocolV2ResponsePayload::from_response_frame(
                        response_frame,
                        payload_start,
                        payload_end,
                        self.response_frame_recycle_tx.clone(),
                        self.pinned_response_frame_recycle_tx.clone(),
                    )?;
                chunk_tx
                    .send(VerbsHostProtocolV2ResponseChunk {
                        stream_id: *stream_id,
                        header,
                        row_indices,
                        partial_output_payload,
                        wire_bytes: response_wire_bytes,
                    })
                    .context("forwarding persistent verbs-host ProtocolV2 response chunk")?;
                **emitted_frames += 1;
            } else {
                self.recycle_response_frame(response_frame);
            }
            parse_ms += elapsed_ms_optional(parse_started);
            response_frames += 1;
            received_wire_bytes = received_wire_bytes
                .checked_add(response_wire_bytes)
                .context("persistent verbs-host received response byte count overflow")?;
            if !more_chunks {
                break;
            }
        }
        let assemble_started = timing_enabled.then(Instant::now);
        let (response, response_executor_id) = if streaming {
            (None, assembler.finish_validation()?.executor_id)
        } else {
            let response = assembler.finish()?.encode()?;
            let executor_id = ExpertProtocolV2ResponseView::parse(&response)?
                .header
                .executor_id;
            (Some(response), executor_id)
        };
        let assemble_ms = elapsed_ms_optional(assemble_started);
        if timing_enabled {
            eprintln!(
                "protocol_v2_verbs_persistent_client_roundtrip_timing addr={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_frames={} received_wire_bytes={} assembled_wire_bytes={} encode_ms={:.3} expected_response_ms={:.3} post_recv_ms={:.3} send_ms={:.3} poll_ms={:.3} copy_recv_ms={:.3} parse_ms={:.3} assemble_ms={:.3} total_ms={:.3}",
                self.addr,
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                request_frame_bytes,
                response_frames,
                received_wire_bytes,
                response.as_ref().map_or(0, Vec::len),
                encode_ms,
                expected_response_ms,
                post_recv_ms,
                send_ms,
                poll_ms,
                copy_recv_ms,
                parse_ms,
                assemble_ms,
                elapsed_ms_optional(total_started)
            );
        }
        Ok((
            response,
            VerbsHostProtocolV2ResponseStreamStats {
                response_frames,
                response_wire_bytes: received_wire_bytes,
                response_executor_id,
            },
        ))
    }
}

fn response_chunk_metadata_from_view(
    view: &ExpertProtocolV2ResponseView<'_>,
    response_frame: &[u8],
) -> Result<(
    ExpertProtocolV2ResponseHeader,
    Option<Vec<u32>>,
    usize,
    usize,
)> {
    let row_indices = if view.row_indexed() {
        Some(
            (0..view.header.row_count as usize)
                .map(|row_index| view.request_row_index(row_index))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let frame_start = response_frame.as_ptr() as usize;
    let payload = view.partial_output_payload();
    let payload_start = (payload.as_ptr() as usize)
        .checked_sub(frame_start)
        .context("ProtocolV2 response payload starts before its frame")?;
    let payload_end = payload_start
        .checked_add(payload.len())
        .context("ProtocolV2 response payload range overflows usize")?;
    anyhow::ensure!(
        payload_end <= response_frame.len(),
        "ProtocolV2 response payload range {payload_start}..{payload_end} exceeds frame bytes {}",
        response_frame.len()
    );
    Ok((view.header.clone(), row_indices, payload_start, payload_end))
}

pub async fn serve_synthetic_verbs_host(addr: &str) -> Result<()> {
    serve_protocol_v2_verbs_host_with_executor(addr, Arc::new(SyntheticRouteExecutor)).await
}

pub async fn serve_protocol_v2_verbs_host_with_executor(
    addr: &str,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
) -> Result<()> {
    let addr = addr.to_owned();
    tokio::task::spawn_blocking(move || serve_protocol_v2_verbs_host_blocking(&addr, executor))
        .await
        .context("verbs-host ProtocolV2 server worker panicked")?
}

fn verbs_host_protocol_v2_roundtrip_blocking(
    addr: SocketAddr,
    request: ExpertProtocolV2Request,
    config: TcpTransportConfig,
) -> Result<ExpertProtocolV2Response> {
    verbs_host_preflight()?;
    let native_path = verbs_host_native_library_path().context(
        "native library not found; set DS41RT_NATIVE_LIB or build native/libds41rt_native.so with RDMA",
    )?;
    let library = Arc::new(unsafe { NativeLibrary::load(&native_path) }?);
    let expected_response = verbs_host_expected_response_for_request(&request)?;
    let request_frame = request.encode()?;
    let expected_response_frame = expected_response.encode()?;
    if request_frame.len() > config.max_frame_bytes {
        bail!(
            "verbs-host ProtocolV2 request frame length {} exceeds max frame bytes {}",
            request_frame.len(),
            config.max_frame_bytes
        );
    }
    if expected_response_frame.len() > config.max_frame_bytes {
        bail!(
            "verbs-host ProtocolV2 expected response frame length {} exceeds max frame bytes {}",
            expected_response_frame.len(),
            config.max_frame_bytes
        );
    }
    let endpoint_plan = verbs_host_protocol_v2_endpoint_plan(
        &request_frame,
        &expected_response_frame,
        crate::verbs_host_capabilities().preferred_alignment,
    )?;
    let endpoint = NativeRdmaEndpoint::create(
        Arc::clone(&library),
        "client",
        &endpoint_plan,
        next_local_psn("client"),
    )?;
    let peer = addr.to_string();
    let mut stream = connect_control_stream(&peer, config.timeout)?;
    configure_control_stream(&stream, config.timeout)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let (client_host, _server_host) =
        distinct_endpoint_hosts(local_control_host("client"), peer.clone());
    let start = VerbsHostProtocolV2RunStart {
        message: "protocol_v2_run_start".to_owned(),
        request_id: request.header.request_id,
        request_wire_bytes: endpoint_plan.request_frame_bytes,
        response_wire_bytes: endpoint_plan.response_frame_bytes,
        request_registered_span_bytes: endpoint_plan.request_registered_span_bytes,
        response_registered_span_bytes: endpoint_plan.response_registered_span_bytes,
        client_endpoint: endpoint.verbs_descriptor("client", &client_host),
        client_native_endpoint: endpoint.native_descriptor(),
    };
    write_control(&mut stream, &start)?;
    let ready: VerbsHostProtocolV2RunReady = read_control(&mut reader)?;
    if ready.message != "protocol_v2_run_ready" || ready.request_id != request.header.request_id {
        bail!("verbs-host ProtocolV2 client received invalid run_ready message");
    }
    let validation = verbs_host_validate_protocol_v2_handshake(
        &endpoint_plan,
        &start.client_endpoint,
        &ready.server_endpoint,
    )?;
    let _roundtrip = verbs_host_protocol_v2_round_trip_plan(
        &endpoint_plan,
        &validation,
        &request_frame,
        &expected_response_frame,
    )?;
    endpoint.connect(&ready.server_native_endpoint)?;
    let recv_ready: VerbsHostProtocolV2RecvReady = read_control(&mut reader)?;
    if recv_ready.message != "protocol_v2_recv_ready"
        || recv_ready.request_id != request.header.request_id
    {
        bail!("verbs-host ProtocolV2 client received invalid recv_ready message");
    }
    endpoint.post_recv(endpoint_plan.response_frame_bytes)?;
    endpoint.send(&request_frame)?;
    endpoint.poll(1, 1, config.timeout)?;
    let mut response_frame = vec![0_u8; endpoint_plan.response_frame_bytes];
    endpoint.copy_recv(&mut response_frame, endpoint_plan.response_frame_bytes)?;
    let response = ExpertProtocolV2Response::decode(&response_frame)
        .context("decoding verbs-host ProtocolV2 response frame")?;
    validate_response_matches_request(&response.header, &request)?;
    Ok(response)
}

fn serve_protocol_v2_verbs_host_blocking(
    addr: &str,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
) -> Result<()> {
    verbs_host_preflight()?;
    let native_path = verbs_host_native_library_path().context(
        "native library not found; set DS41RT_NATIVE_LIB or build native/libds41rt_native.so with RDMA",
    )?;
    let library = Arc::new(unsafe { NativeLibrary::load(&native_path) }?);
    let bind_addr = if addr.contains(':') {
        addr.to_owned()
    } else {
        format!("{addr}:9100")
    };
    let listener = TcpListener::bind(&bind_addr)
        .with_context(|| format!("binding verbs-host ProtocolV2 expert service to {bind_addr}"))?;
    tracing::info!(
        addr = %bind_addr,
        executor = executor.name(),
        "verbs-host ProtocolV2 expert service listening"
    );
    for accepted in listener.incoming() {
        let stream = accepted.context("accepting verbs-host ProtocolV2 control connection")?;
        let executor = Arc::clone(&executor);
        let library = Arc::clone(&library);
        thread::spawn(move || {
            if let Err(error) = handle_verbs_host_protocol_v2_connection(stream, executor, library)
            {
                eprintln!("verbs-host ProtocolV2 expert connection closed with error: {error:#}");
                tracing::warn!(
                    error = %error,
                    "verbs-host ProtocolV2 expert connection closed with error"
                );
            }
        });
    }
    Ok(())
}

fn handle_verbs_host_protocol_v2_connection(
    mut stream: TcpStream,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
    library: Arc<NativeLibrary>,
) -> Result<()> {
    configure_control_stream(&stream, default_control_timeout())?;
    let mut reader = BufReader::new(stream.try_clone()?);
    loop {
        let value = match read_control_value(&mut reader) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("control plane closed") => return Ok(()),
            Err(error) => return Err(error),
        };
        let message = control_message(&value)?;
        if message == "mapped_rdma_ring_start" {
            let start: VerbsHostMappedRdmaRingStart = serde_json::from_value(value)
                .context("decoding mapped RDMA ring start control message")?;
            return handle_mapped_protocol_v2_ring_connection(stream, executor, library, start);
        }
        if message == "protocol_v2_persistent_start" {
            let start: VerbsHostProtocolV2PersistentStart = serde_json::from_value(value)
                .context("decoding verbs-host ProtocolV2 persistent start control message")?;
            return handle_verbs_host_protocol_v2_persistent_connection(
                stream, reader, executor, library, start,
            );
        }
        if message != "protocol_v2_run_start" {
            bail!("verbs-host ProtocolV2 server expected protocol_v2_run_start, got {message}");
        }
        let start: VerbsHostProtocolV2RunStart = serde_json::from_value(value)
            .context("decoding verbs-host ProtocolV2 run start control message")?;
        let rdma_device = verbs_host_rdma_device_for_stream(&stream)?;
        let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_on_device(
            Arc::clone(&library),
            "server",
            start.request_wire_bytes,
            start.response_wire_bytes,
            start.request_registered_span_bytes,
            start.response_registered_span_bytes,
            next_local_psn("server"),
            rdma_device.as_deref(),
        )?;
        let (_client_host, server_host) = distinct_endpoint_hosts(
            start.client_endpoint.host.clone(),
            local_control_host("server"),
        );
        let server_endpoint = endpoint.verbs_descriptor("server", &server_host);
        validate_control_endpoint_metadata(&start.client_endpoint, "client")?;
        validate_control_endpoint_metadata(&server_endpoint, "server")?;
        let ready = VerbsHostProtocolV2RunReady {
            message: "protocol_v2_run_ready".to_owned(),
            request_id: start.request_id,
            server_endpoint,
            server_native_endpoint: endpoint.native_descriptor(),
        };
        write_control(&mut stream, &ready)?;
        endpoint.connect(&start.client_native_endpoint)?;
        endpoint.post_recv(start.request_wire_bytes)?;
        write_control(
            &mut stream,
            &VerbsHostProtocolV2RecvReady {
                message: "protocol_v2_recv_ready".to_owned(),
                request_id: start.request_id,
            },
        )?;
        endpoint.poll(0, 1, default_control_timeout())?;
        let mut request_frame = vec![0_u8; start.request_wire_bytes];
        endpoint.copy_recv(&mut request_frame, start.request_wire_bytes)?;
        let request = ExpertProtocolV2RequestView::parse(&request_frame)
            .context("parsing verbs-host ProtocolV2 request frame")?;
        if request.header.request_id != start.request_id {
            bail!(
                "verbs-host ProtocolV2 request_id {} did not match control request_id {}",
                request.header.request_id,
                start.request_id
            );
        }
        let response = executor.execute_with_identity(&request)?;
        let mut response_frame = ExpertProtocolV2FrameBuffer::new();
        let response_prefix = response_frame.encode_response_prefix(&response)?;
        let response_wire_bytes = response_prefix
            .len()
            .checked_add(response.partial_output_payload.len())
            .context("verbs-host ProtocolV2 response byte count overflow")?;
        if response_wire_bytes != start.response_wire_bytes {
            bail!(
                "verbs-host ProtocolV2 executor response frame bytes {} did not match registered response frame bytes {}; dynamic response sizing is not supported by this RDMA handshake",
                response_wire_bytes,
                start.response_wire_bytes
            );
        }
        endpoint.send_parts_at(
            response_prefix,
            &response.partial_output_payload,
            0,
            VERBS_HOST_SEND_WR_ID,
        )?;
        endpoint.poll(1, 0, default_control_timeout())?;
    }
}

fn handle_mapped_protocol_v2_ring_connection(
    stream: TcpStream,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
    library: Arc<NativeLibrary>,
    start: VerbsHostMappedRdmaRingStart,
) -> Result<()> {
    let transport = TcpTransportConfig::default();
    let response_library = Arc::clone(&library);
    let mut ring = VerbsHostMappedRdmaRing::accept_started(stream, &transport, start, library)?;
    let mut response_frame = ExpertProtocolV2FrameBuffer::new();
    loop {
        let request_slot = match ring.wait_recv_slot() {
            Ok(slot) => slot,
            Err(error) if is_verbs_host_rdma_poll_timeout(&error) => continue,
            Err(error) => return Err(error),
        };
        let request_capacity = request_slot.capacity_bytes;
        let request_storage = unsafe {
            std::slice::from_raw_parts(request_slot.host_ptr.cast_const(), request_capacity)
        };
        let request_wire_bytes = persistent_protocol_v2_request_wire_bytes_from_header(
            &request_storage[..EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN],
            request_capacity,
        )?;
        let request = ExpertProtocolV2RequestView::parse(&request_storage[..request_wire_bytes])
            .context("parsing mapped RDMA ring ProtocolV2 request")?;
        let device_payload = protocol_v2_request_device_payload(
            &request_storage[..request_wire_bytes],
            &request,
            request_slot.device_buffer,
            None,
            0,
        )?;
        let mut emitted = 0_usize;
        executor.execute_streaming_device_payload_with_identity(
            &request,
            device_payload,
            &mut |response| {
                let response_slot = ring.reserve_send_slot()?;
                let response_wire_bytes = match response {
                    ProtocolV2ExecutorResponseRef::Host(response) => {
                        let response_prefix =
                            response_frame.encode_borrowed_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.len())
                            .context("mapped RDMA ring response byte count overflow")?;
                        anyhow::ensure!(
                            wire_bytes <= response_slot.capacity_bytes,
                            "mapped RDMA ring response bytes {wire_bytes} exceed capacity {}",
                            response_slot.capacity_bytes
                        );
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                response_prefix.as_ptr(),
                                response_slot.host_ptr,
                                response_prefix.len(),
                            );
                            std::ptr::copy_nonoverlapping(
                                response.partial_output_payload.as_ptr(),
                                response_slot.host_ptr.add(response_prefix.len()),
                                response.partial_output_payload.len(),
                            );
                        }
                        wire_bytes
                    }
                    ProtocolV2ExecutorResponseRef::Device(response) => {
                        let response_prefix =
                            response_frame.encode_device_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.bytes)
                            .context("mapped RDMA ring device response byte count overflow")?;
                        anyhow::ensure!(
                            wire_bytes <= response_slot.capacity_bytes,
                            "mapped RDMA ring device response bytes {wire_bytes} exceed capacity {}",
                            response_slot.capacity_bytes
                        );
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                response_prefix.as_ptr(),
                                response_slot.host_ptr,
                                response_prefix.len(),
                            );
                        }
                        let destination = protocol_v2_device_buffer_slice(
                            response_slot.device_buffer,
                            response_prefix.len(),
                            response.partial_output_payload.bytes,
                            "mapped RDMA ring response payload",
                        )?;
                        response_library.copy_d2d(
                            destination,
                            response.partial_output_payload,
                            response.partial_output_payload.bytes,
                        )?;
                        wire_bytes
                    }
                };
                ring.post_reserved_send(response_wire_bytes)?;
                emitted = emitted
                    .checked_add(1)
                    .context("mapped RDMA ring emitted response count overflow")?;
                Ok(())
            },
        )?;
        anyhow::ensure!(emitted > 0, "mapped RDMA ring executor emitted no response");
        ring.release_recv_slot(request_slot.sequence)?;
    }
}

fn protocol_v2_request_device_payload(
    request_storage: &[u8],
    request: &ExpertProtocolV2RequestView<'_>,
    frame: Ds41rtDeviceBuffer,
    response_slot: Option<Ds41rtDeviceBuffer>,
    execution_lane: u32,
) -> Result<ProtocolV2RequestDevicePayload> {
    let storage_start = request_storage.as_ptr() as usize;
    let hidden_start = request.hidden_payload().as_ptr() as usize;
    anyhow::ensure!(
        hidden_start >= storage_start,
        "ProtocolV2 hidden payload starts before its mapped frame"
    );
    let hidden_offset = hidden_start - storage_start;
    let hidden_end = hidden_offset
        .checked_add(request.hidden_payload().len())
        .context("ProtocolV2 mapped hidden payload end overflow")?;
    anyhow::ensure!(
        hidden_end <= request_storage.len() && hidden_end <= frame.bytes,
        "ProtocolV2 mapped hidden payload [{hidden_offset}, {hidden_end}) exceeds frame bytes {}",
        frame.bytes
    );
    Ok(ProtocolV2RequestDevicePayload {
        execution_lane,
        response_slot,
        hidden_payload: Ds41rtDeviceBuffer {
            ptr: unsafe { frame.ptr.cast::<u8>().add(hidden_offset).cast() },
            bytes: request.hidden_payload().len(),
            device_id: frame.device_id,
            flags: frame.flags,
        },
    })
}

fn protocol_v2_device_buffer_slice(
    buffer: Ds41rtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
    label: &str,
) -> Result<Ds41rtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .with_context(|| format!("{label} end overflow"))?;
    anyhow::ensure!(
        end <= buffer.bytes,
        "{label} [{offset_bytes}, {end}) exceeds {} bytes",
        buffer.bytes
    );
    Ok(Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn device_buffers_match(left: Ds41rtDeviceBuffer, right: Ds41rtDeviceBuffer) -> bool {
    left.ptr == right.ptr
        && left.bytes == right.bytes
        && left.device_id == right.device_id
        && left.flags == right.flags
}

fn handle_verbs_host_protocol_v2_persistent_connection(
    stream: TcpStream,
    _reader: BufReader<TcpStream>,
    executor: Arc<dyn ProtocolV2ExpertExecutor>,
    library: Arc<NativeLibrary>,
    start: VerbsHostProtocolV2PersistentStart,
) -> Result<()> {
    let mut connection = LocalVerbsExpertConnection::initialize(stream, library, start)?;
    loop {
        if !connection.poll(|request, payload, emit| {
            executor.execute_streaming_device_payload_with_identity(request, payload, emit)
        })? { std::hint::spin_loop(); }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostProtocolV2RunStart {
    message: String,
    request_id: u64,
    request_wire_bytes: usize,
    response_wire_bytes: usize,
    request_registered_span_bytes: usize,
    response_registered_span_bytes: usize,
    client_endpoint: VerbsHostRcEndpointDescriptor,
    client_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostProtocolV2RunReady {
    message: String,
    request_id: u64,
    server_endpoint: VerbsHostRcEndpointDescriptor,
    server_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostProtocolV2PersistentStart {
    message: String,
    #[serde(default)]
    execution_lane: u32,
    request_capacity_wire_bytes: usize,
    response_capacity_wire_bytes: usize,
    request_registered_span_bytes: usize,
    response_registered_span_bytes: usize,
    ring_depth: usize,
    request_slot_stride_bytes: usize,
    response_slot_stride_bytes: usize,
    client_endpoint: VerbsHostRcEndpointDescriptor,
    client_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostProtocolV2PersistentReady {
    message: String,
    server_endpoint: VerbsHostRcEndpointDescriptor,
    server_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostProtocolV2RecvReady {
    message: String,
    request_id: u64,
}

struct NativeRdmaEndpoint {
    library: Arc<NativeLibrary>,
    info: Ds41rtRdmaRcEndpointInfo,
}

impl NativeRdmaEndpoint {
    fn create(
        library: Arc<NativeLibrary>,
        role: &str,
        endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
        local_psn: u32,
    ) -> Result<Self> {
        Self::create_from_wire_bytes(
            library,
            role,
            endpoint_plan.request_frame_bytes,
            endpoint_plan.response_frame_bytes,
            endpoint_plan.request_registered_span_bytes,
            endpoint_plan.response_registered_span_bytes,
            local_psn,
        )
    }

    fn create_from_wire_bytes(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
    ) -> Result<Self> {
        Self::create_from_wire_bytes_on_device(
            library,
            role,
            request_frame_bytes,
            response_frame_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            local_psn,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_from_wire_bytes_on_device(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
        rdma_device: Option<&str>,
    ) -> Result<Self> {
        Self::create_from_wire_bytes_with_buffer_flags(
            library,
            role,
            request_frame_bytes,
            response_frame_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            local_psn,
            0,
            VERBS_HOST_RDMA_RING_DEPTH as u32,
            rdma_device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_from_wire_bytes_mapped_on_device(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
        rdma_device: Option<&str>,
    ) -> Result<Self> {
        Self::create_from_wire_bytes_with_buffer_flags(
            library,
            role,
            request_frame_bytes,
            response_frame_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            local_psn,
            DS41RT_HOST_BUFFER_FLAG_PINNED | DS41RT_HOST_BUFFER_FLAG_MAPPED,
            8,
            rdma_device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_from_wire_bytes_mapped_for_ring(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
        ring_depth: usize,
    ) -> Result<Self> {
        let queue_depth = u32::try_from(ring_depth.max(VERBS_HOST_RDMA_RING_DEPTH))
            .context("mapped RDMA ring depth exceeds u32")?;
        Self::create_from_wire_bytes_with_buffer_flags(
            library,
            role,
            request_frame_bytes,
            response_frame_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            local_psn,
            DS41RT_HOST_BUFFER_FLAG_PINNED | DS41RT_HOST_BUFFER_FLAG_MAPPED,
            queue_depth,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_from_wire_bytes_mapped_for_ring_on_device(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
        ring_depth: usize,
        rdma_device: Option<&str>,
    ) -> Result<Self> {
        let queue_depth = u32::try_from(ring_depth.max(VERBS_HOST_RDMA_RING_DEPTH))
            .context("mapped RDMA ring depth exceeds u32")?;
        Self::create_from_wire_bytes_with_buffer_flags(
            library,
            role,
            request_frame_bytes,
            response_frame_bytes,
            request_registered_span_bytes,
            response_registered_span_bytes,
            local_psn,
            DS41RT_HOST_BUFFER_FLAG_PINNED | DS41RT_HOST_BUFFER_FLAG_MAPPED,
            queue_depth,
            rdma_device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_from_wire_bytes_with_buffer_flags(
        library: Arc<NativeLibrary>,
        role: &str,
        request_frame_bytes: usize,
        response_frame_bytes: usize,
        request_registered_span_bytes: usize,
        response_registered_span_bytes: usize,
        local_psn: u32,
        host_buffer_flags: u64,
        queue_depth: u32,
        rdma_device: Option<&str>,
    ) -> Result<Self> {
        let port_num = verbs_host_ib_port_num()?;
        let (send_frame_bytes, recv_frame_bytes, send_span, recv_span) = match role {
            "client" => (
                request_frame_bytes,
                response_frame_bytes,
                request_registered_span_bytes,
                response_registered_span_bytes,
            ),
            "server" => (
                response_frame_bytes,
                request_frame_bytes,
                response_registered_span_bytes,
                request_registered_span_bytes,
            ),
            other => bail!("unsupported verbs-host RDMA endpoint role: {other}"),
        };
        let info = if let Some(rdma_device) = rdma_device {
            library.rdma_rc_endpoint_create_on_device_with_buffer_flags(
                rdma_device,
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_span,
                recv_span,
                queue_depth,
                queue_depth,
                1,
                host_buffer_flags,
            )?
        } else if host_buffer_flags == 0 {
            library.rdma_rc_endpoint_create(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_span,
                recv_span,
                queue_depth,
                queue_depth,
                1,
            )?
        } else {
            library.rdma_rc_endpoint_create_with_buffer_flags(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_span,
                recv_span,
                queue_depth,
                queue_depth,
                1,
                host_buffer_flags,
            )?
        };
        Ok(Self { library, info })
    }

    fn send_buffer_view(&self) -> Result<Ds41rtRdmaRcEndpointBufferView> {
        self.library
            .rdma_rc_endpoint_buffer_view(self.info.handle, false)
    }

    fn recv_buffer_view(&self) -> Result<Ds41rtRdmaRcEndpointBufferView> {
        self.library
            .rdma_rc_endpoint_buffer_view(self.info.handle, true)
    }

    fn native_descriptor(&self) -> VerbsHostNativeEndpointDescriptor {
        VerbsHostNativeEndpointDescriptor {
            port_num: self.info.port_num,
            qp_num: self.info.qp_num,
            psn: self.info.psn,
            lid: self.info.lid,
            active_mtu: self.info.active_mtu,
            gid_hex: c_char_array_to_string(&self.info.gid_hex),
            send_frame_bytes: self.info.send_frame_bytes,
            recv_frame_bytes: self.info.recv_frame_bytes,
            send_registered_span_bytes: self.info.send_registered_span_bytes,
            recv_registered_span_bytes: self.info.recv_registered_span_bytes,
            max_send_wr: self.info.max_send_wr,
            max_recv_wr: self.info.max_recv_wr,
            max_sge: self.info.max_sge,
            device_name: c_char_array_to_string(&self.info.device_name),
            status: c_char_array_to_string(&self.info.status),
        }
    }

    fn verbs_descriptor(&self, role: &str, host: &str) -> VerbsHostRcEndpointDescriptor {
        VerbsHostRcEndpointDescriptor {
            role: role.to_owned(),
            host: host.to_owned(),
            port_num: self.info.port_num,
            qp_num: self.info.qp_num,
            psn: self.info.psn,
            gid_hex: c_char_array_to_string(&self.info.gid_hex),
            send_frame_bytes: self.info.send_frame_bytes,
            recv_frame_bytes: self.info.recv_frame_bytes,
            send_registered_span_bytes: self.info.send_registered_span_bytes,
            recv_registered_span_bytes: self.info.recv_registered_span_bytes,
            max_send_wr: self.info.max_send_wr,
            max_recv_wr: self.info.max_recv_wr,
            max_sge: self.info.max_sge,
        }
    }

    fn connect(&self, peer: &VerbsHostNativeEndpointDescriptor) -> Result<()> {
        self.library.rdma_rc_endpoint_connect(
            self.info.handle,
            peer.qp_num,
            peer.psn,
            peer.lid,
            &peer.gid_hex,
        )
    }

    fn post_recv(&self, bytes: usize) -> Result<()> {
        self.post_recv_at(0, bytes, VERBS_HOST_RECV_WR_ID)
    }

    fn post_recv_at(&self, offset_bytes: usize, bytes: usize, wr_id: u64) -> Result<()> {
        self.library
            .rdma_rc_endpoint_post_recv_at(self.info.handle, offset_bytes, bytes, wr_id)
    }

    fn send(&self, frame: &[u8]) -> Result<()> {
        self.send_at(frame, 0, VERBS_HOST_SEND_WR_ID)
    }

    fn send_at(&self, frame: &[u8], offset_bytes: usize, wr_id: u64) -> Result<()> {
        self.library
            .rdma_rc_endpoint_send_at(self.info.handle, frame, offset_bytes, wr_id)
    }

    fn post_send_at(&self, offset_bytes: usize, bytes: usize, wr_id: u64) -> Result<()> {
        self.library
            .rdma_rc_endpoint_post_send_at(self.info.handle, offset_bytes, bytes, wr_id)
    }

    fn send_parts_at(
        &self,
        prefix: &[u8],
        payload: &[u8],
        offset_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        self.library.rdma_rc_endpoint_send_parts_at(
            self.info.handle,
            prefix,
            payload,
            offset_bytes,
            wr_id,
        )
    }

    fn poll(&self, send: u32, recv: u32, active_timeout: Duration) -> Result<()> {
        self.poll_stats(send, recv, active_timeout)?;
        Ok(())
    }

    fn poll_stats(
        &self,
        send: u32,
        recv: u32,
        active_timeout: Duration,
    ) -> Result<Ds41rtRdmaRcCompletionStats> {
        self.library.rdma_rc_endpoint_poll(
            self.info.handle,
            send,
            recv,
            u32::MAX,
            active_poll_timeout_ms(active_timeout),
        )
    }

    fn try_poll(&self, send: u32, recv: u32) -> Result<Ds41rtRdmaRcCompletionStats> {
        self.library
            .rdma_rc_endpoint_try_poll(self.info.handle, send, recv)
    }

    fn copy_recv(&self, out: &mut [u8], bytes: usize) -> Result<()> {
        self.copy_recv_at(out, 0, bytes)
    }

    fn copy_recv_at(&self, out: &mut [u8], offset_bytes: usize, bytes: usize) -> Result<()> {
        self.library
            .rdma_rc_endpoint_copy_recv_at(self.info.handle, out, offset_bytes, bytes)
    }
}

impl Drop for NativeRdmaEndpoint {
    fn drop(&mut self) {
        if !self.info.handle.is_null() {
            let _ = self.library.rdma_rc_endpoint_destroy(self.info.handle);
            self.info.handle = std::ptr::null_mut();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerbsHostMappedRdmaRingConfig {
    pub slot_capacity_bytes: usize,
    pub depth: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct VerbsHostMappedRdmaPollStats {
    pub send_completions: usize,
    pub recv_completions: usize,
    pub poll_iterations: u64,
}

impl VerbsHostMappedRdmaPollStats {
    fn from_native(stats: Ds41rtRdmaRcCompletionStats) -> Self {
        Self {
            send_completions: stats.send_completions as usize,
            recv_completions: stats.recv_completions as usize,
            poll_iterations: stats.poll_iterations as u64,
        }
    }

    fn merge(&mut self, other: Self) {
        self.send_completions += other.send_completions;
        self.recv_completions += other.recv_completions;
        self.poll_iterations += other.poll_iterations;
    }
}

impl VerbsHostMappedRdmaRingConfig {
    pub fn new(slot_capacity_bytes: usize, depth: usize) -> Result<Self> {
        let alignment = crate::verbs_host_capabilities().preferred_alignment;
        VerbsHostRdmaRing::new_with_max_depth(
            slot_capacity_bytes,
            alignment,
            depth,
            VERBS_HOST_MAPPED_RDMA_RING_MAX_DEPTH,
        )?;
        Ok(Self {
            slot_capacity_bytes,
            depth,
        })
    }

    fn layout(self) -> Result<VerbsHostRdmaRing> {
        VerbsHostRdmaRing::new_with_max_depth(
            self.slot_capacity_bytes,
            crate::verbs_host_capabilities().preferred_alignment,
            self.depth,
            VERBS_HOST_MAPPED_RDMA_RING_MAX_DEPTH,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VerbsHostMappedRdmaSlot {
    pub sequence: u64,
    pub slot_index: usize,
    pub capacity_bytes: usize,
    pub host_ptr: *mut u8,
    pub device_buffer: Ds41rtDeviceBuffer,
}

// The endpoint owns the mapped allocation. Slots are borrowed addresses and may
// move between worker threads, but they must not outlive their ring.
unsafe impl Send for VerbsHostMappedRdmaSlot {}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostMappedRdmaRingStart {
    message: String,
    slot_capacity_bytes: usize,
    slot_stride_bytes: usize,
    ring_depth: usize,
    registered_span_bytes: usize,
    client_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerbsHostMappedRdmaRingReady {
    message: String,
    server_native_endpoint: VerbsHostNativeEndpointDescriptor,
}

pub struct VerbsHostMappedRdmaRing {
    _control_stream: TcpStream,
    endpoint: NativeRdmaEndpoint,
    layout: VerbsHostRdmaRing,
    send_view: Ds41rtRdmaRcEndpointBufferView,
    recv_view: Ds41rtRdmaRcEndpointBufferView,
    send_sequence: u64,
    send_in_flight: usize,
    reserved_send_sequence: Option<u64>,
    recv_completion_sequence: u64,
    recv_release_sequence: u64,
}

// All endpoint operations require `&mut self`; the native QP and borrowed views
// therefore retain single-threaded ownership when the ring moves to a worker.
unsafe impl Send for VerbsHostMappedRdmaRing {}

impl VerbsHostMappedRdmaRing {
    pub fn connect(
        peer: &str,
        transport: &TcpTransportConfig,
        config: VerbsHostMappedRdmaRingConfig,
    ) -> Result<Self> {
        Self::connect_inner(peer, transport, config, None)
    }

    pub fn connect_on_device(
        peer: &str,
        transport: &TcpTransportConfig,
        config: VerbsHostMappedRdmaRingConfig,
        rdma_device: &str,
    ) -> Result<Self> {
        anyhow::ensure!(
            !rdma_device.trim().is_empty(),
            "mapped RDMA ring device name is empty"
        );
        Self::connect_inner(peer, transport, config, Some(rdma_device))
    }

    fn connect_inner(
        peer: &str,
        transport: &TcpTransportConfig,
        config: VerbsHostMappedRdmaRingConfig,
        rdma_device: Option<&str>,
    ) -> Result<Self> {
        verbs_host_preflight()?;
        anyhow::ensure!(
            config.slot_capacity_bytes <= transport.max_frame_bytes,
            "mapped RDMA ring slot capacity {} exceeds transport maximum {}",
            config.slot_capacity_bytes,
            transport.max_frame_bytes
        );
        let layout = config.layout()?;
        let library = load_verbs_host_native_library()?;
        let endpoint = match rdma_device {
            Some(rdma_device) => {
                NativeRdmaEndpoint::create_from_wire_bytes_mapped_for_ring_on_device(
                    Arc::clone(&library),
                    "client",
                    layout.slot_capacity_bytes,
                    layout.slot_capacity_bytes,
                    layout.registered_span_bytes,
                    layout.registered_span_bytes,
                    next_local_psn("client"),
                    layout.depth,
                    Some(rdma_device),
                )?
            }
            None => NativeRdmaEndpoint::create_from_wire_bytes_mapped_for_ring(
                Arc::clone(&library),
                "client",
                layout.slot_capacity_bytes,
                layout.slot_capacity_bytes,
                layout.registered_span_bytes,
                layout.registered_span_bytes,
                next_local_psn("client"),
                layout.depth,
            )?,
        };
        let mut stream = connect_control_stream(peer, transport.timeout)?;
        configure_control_stream(&stream, transport.timeout)?;
        let mut reader = BufReader::new(stream.try_clone()?);
        write_control(
            &mut stream,
            &VerbsHostMappedRdmaRingStart {
                message: "mapped_rdma_ring_start".to_owned(),
                slot_capacity_bytes: layout.slot_capacity_bytes,
                slot_stride_bytes: layout.slot_stride_bytes,
                ring_depth: layout.depth,
                registered_span_bytes: layout.registered_span_bytes,
                client_native_endpoint: endpoint.native_descriptor(),
            },
        )?;
        let ready: VerbsHostMappedRdmaRingReady = read_control(&mut reader)?;
        anyhow::ensure!(
            ready.message == "mapped_rdma_ring_ready",
            "mapped RDMA ring client received unexpected ready message {}",
            ready.message
        );
        validate_mapped_ring_endpoint(&ready.server_native_endpoint, layout, "server")?;
        endpoint.connect(&ready.server_native_endpoint)?;
        Self::finish(stream, endpoint, layout)
    }

    pub fn accept(stream: TcpStream, transport: &TcpTransportConfig) -> Result<Self> {
        verbs_host_preflight()?;
        configure_control_stream(&stream, transport.timeout)?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let start: VerbsHostMappedRdmaRingStart = read_control(&mut reader)?;
        let library = load_verbs_host_native_library()?;
        Self::accept_started(stream, transport, start, library)
    }

    fn accept_started(
        mut stream: TcpStream,
        transport: &TcpTransportConfig,
        start: VerbsHostMappedRdmaRingStart,
        library: Arc<NativeLibrary>,
    ) -> Result<Self> {
        let rdma_device = verbs_host_rdma_device_for_stream(&stream)?;
        anyhow::ensure!(
            start.message == "mapped_rdma_ring_start",
            "mapped RDMA ring server received unexpected start message {}",
            start.message
        );
        anyhow::ensure!(
            start.slot_capacity_bytes <= transport.max_frame_bytes,
            "mapped RDMA ring slot capacity {} exceeds transport maximum {}",
            start.slot_capacity_bytes,
            transport.max_frame_bytes
        );
        let layout = VerbsHostRdmaRing::from_wire_with_max_depth(
            start.slot_capacity_bytes,
            start.slot_stride_bytes,
            start.ring_depth,
            start.registered_span_bytes,
            VERBS_HOST_MAPPED_RDMA_RING_MAX_DEPTH,
        )?;
        validate_mapped_ring_endpoint(&start.client_native_endpoint, layout, "client")?;
        let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_mapped_for_ring_on_device(
            library,
            "server",
            layout.slot_capacity_bytes,
            layout.slot_capacity_bytes,
            layout.registered_span_bytes,
            layout.registered_span_bytes,
            next_local_psn("server"),
            layout.depth,
            rdma_device.as_deref(),
        )?;
        endpoint.connect(&start.client_native_endpoint)?;
        for slot in 0..layout.depth {
            endpoint.post_recv_at(
                layout.slot_offset(slot),
                layout.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
        }
        write_control(
            &mut stream,
            &VerbsHostMappedRdmaRingReady {
                message: "mapped_rdma_ring_ready".to_owned(),
                server_native_endpoint: endpoint.native_descriptor(),
            },
        )?;
        Self::from_initialized(stream, endpoint, layout)
    }

    fn finish(
        stream: TcpStream,
        endpoint: NativeRdmaEndpoint,
        layout: VerbsHostRdmaRing,
    ) -> Result<Self> {
        for slot in 0..layout.depth {
            endpoint.post_recv_at(
                layout.slot_offset(slot),
                layout.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
        }
        Self::from_initialized(stream, endpoint, layout)
    }

    fn from_initialized(
        stream: TcpStream,
        endpoint: NativeRdmaEndpoint,
        layout: VerbsHostRdmaRing,
    ) -> Result<Self> {
        let send_view = endpoint.send_buffer_view()?;
        let recv_view = endpoint.recv_buffer_view()?;
        validate_mapped_endpoint_buffer_view(send_view, layout, "send")?;
        validate_mapped_endpoint_buffer_view(recv_view, layout, "receive")?;
        Ok(Self {
            _control_stream: stream,
            endpoint,
            layout,
            send_view,
            recv_view,
            send_sequence: 0,
            send_in_flight: 0,
            reserved_send_sequence: None,
            recv_completion_sequence: 0,
            recv_release_sequence: 0,
        })
    }

    pub fn config(&self) -> VerbsHostMappedRdmaRingConfig {
        VerbsHostMappedRdmaRingConfig {
            slot_capacity_bytes: self.layout.slot_capacity_bytes,
            depth: self.layout.depth,
        }
    }

    pub fn reserve_send_slot(&mut self) -> Result<VerbsHostMappedRdmaSlot> {
        self.reserve_send_slot_with_stats().map(|(slot, _)| slot)
    }

    pub fn reserve_send_slot_with_stats(
        &mut self,
    ) -> Result<(VerbsHostMappedRdmaSlot, VerbsHostMappedRdmaPollStats)> {
        anyhow::ensure!(
            self.reserved_send_sequence.is_none(),
            "mapped RDMA ring already has a reserved send slot"
        );
        let mut poll_stats = self.reclaim_send_completions_with_stats()?;
        if self.send_in_flight == self.layout.depth {
            let stats = self.endpoint.poll_stats(1, 0, default_control_timeout())?;
            poll_stats.merge(VerbsHostMappedRdmaPollStats::from_native(stats));
            self.send_in_flight -= 1;
        }
        let sequence = self.send_sequence;
        self.reserved_send_sequence = Some(sequence);
        Ok((
            mapped_ring_slot(self.send_view, self.layout, sequence)?,
            poll_stats,
        ))
    }

    pub fn post_reserved_send(&mut self, bytes: usize) -> Result<()> {
        let sequence = *self
            .reserved_send_sequence
            .as_ref()
            .context("mapped RDMA ring has no reserved send slot")?;
        anyhow::ensure!(
            bytes > 0 && bytes <= self.layout.slot_capacity_bytes,
            "mapped RDMA ring send bytes {bytes} must be in 1..={}",
            self.layout.slot_capacity_bytes
        );
        self.reserved_send_sequence = None;
        let slot_index = sequence as usize % self.layout.depth;
        self.endpoint.post_send_at(
            self.layout.slot_offset(sequence as usize),
            bytes,
            VERBS_HOST_SEND_WR_ID + slot_index as u64,
        )?;
        self.send_sequence = self.send_sequence.wrapping_add(1);
        self.send_in_flight += 1;
        Ok(())
    }

    pub fn send_copy(&mut self, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= self.layout.slot_capacity_bytes,
            "mapped RDMA ring payload bytes {} must be in 1..={}",
            bytes.len(),
            self.layout.slot_capacity_bytes
        );
        let slot = self.reserve_send_slot()?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), slot.host_ptr, bytes.len());
        }
        self.post_reserved_send(bytes.len())
    }

    pub fn reclaim_send_completions(&mut self) -> Result<usize> {
        self.reclaim_send_completions_with_stats()
            .map(|stats| stats.send_completions)
    }

    pub fn reclaim_send_completions_with_stats(&mut self) -> Result<VerbsHostMappedRdmaPollStats> {
        if self.send_in_flight == 0 {
            return Ok(VerbsHostMappedRdmaPollStats::default());
        }
        let stats = self.endpoint.try_poll(self.send_in_flight as u32, 0)?;
        let completed = stats.send_completions as usize;
        anyhow::ensure!(
            completed <= self.send_in_flight,
            "mapped RDMA ring reclaimed {completed} sends with only {} in flight",
            self.send_in_flight
        );
        self.send_in_flight -= completed;
        Ok(VerbsHostMappedRdmaPollStats::from_native(stats))
    }

    pub fn flush_sends(&mut self) -> Result<()> {
        self.flush_sends_with_stats().map(|_| ())
    }

    pub fn flush_sends_with_stats(&mut self) -> Result<VerbsHostMappedRdmaPollStats> {
        anyhow::ensure!(
            self.reserved_send_sequence.is_none(),
            "mapped RDMA ring cannot flush while a send slot is reserved"
        );
        let mut poll_stats = VerbsHostMappedRdmaPollStats::default();
        while self.send_in_flight > 0 {
            let stats = self.endpoint.poll_stats(1, 0, default_control_timeout())?;
            poll_stats.merge(VerbsHostMappedRdmaPollStats::from_native(stats));
            self.send_in_flight -= 1;
        }
        Ok(poll_stats)
    }

    pub fn try_recv_slot(&mut self) -> Result<Option<VerbsHostMappedRdmaSlot>> {
        self.try_recv_slot_with_stats().map(|(slot, _)| slot)
    }

    pub fn try_recv_slot_with_stats(
        &mut self,
    ) -> Result<(
        Option<VerbsHostMappedRdmaSlot>,
        VerbsHostMappedRdmaPollStats,
    )> {
        self.ensure_recv_capacity()?;
        let stats = self.endpoint.try_poll(0, 1)?;
        let poll_stats = VerbsHostMappedRdmaPollStats::from_native(stats);
        if stats.recv_completions == 0 {
            return Ok((None, poll_stats));
        }
        Ok((Some(self.take_completed_recv_slot()?), poll_stats))
    }

    pub fn wait_recv_slot(&mut self) -> Result<VerbsHostMappedRdmaSlot> {
        self.wait_recv_slot_with_stats().map(|(slot, _)| slot)
    }

    pub fn wait_recv_slot_with_stats(
        &mut self,
    ) -> Result<(VerbsHostMappedRdmaSlot, VerbsHostMappedRdmaPollStats)> {
        self.wait_recv_slot_with_timeout_and_stats(Duration::from_secs(1))
    }

    pub fn wait_recv_slot_with_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<VerbsHostMappedRdmaSlot> {
        self.wait_recv_slot_with_timeout_and_stats(timeout)
            .map(|(slot, _)| slot)
    }

    pub fn wait_recv_slot_with_timeout_and_stats(
        &mut self,
        timeout: Duration,
    ) -> Result<(VerbsHostMappedRdmaSlot, VerbsHostMappedRdmaPollStats)> {
        anyhow::ensure!(
            !timeout.is_zero(),
            "mapped RDMA receive timeout must be positive"
        );
        self.ensure_recv_capacity()?;
        let stats = self.endpoint.poll_stats(0, 1, timeout)?;
        Ok((
            self.take_completed_recv_slot()?,
            VerbsHostMappedRdmaPollStats::from_native(stats),
        ))
    }

    fn ensure_recv_capacity(&self) -> Result<()> {
        let outstanding = self
            .recv_completion_sequence
            .wrapping_sub(self.recv_release_sequence);
        anyhow::ensure!(
            outstanding < self.layout.depth as u64,
            "mapped RDMA ring has {} unreleased receive slots",
            outstanding
        );
        Ok(())
    }

    fn take_completed_recv_slot(&mut self) -> Result<VerbsHostMappedRdmaSlot> {
        let sequence = self.recv_completion_sequence;
        self.recv_completion_sequence = self.recv_completion_sequence.wrapping_add(1);
        mapped_ring_slot(self.recv_view, self.layout, sequence)
    }

    pub fn release_recv_slot(&mut self, sequence: u64) -> Result<()> {
        anyhow::ensure!(
            sequence == self.recv_release_sequence,
            "mapped RDMA ring receive slot {sequence} released out of order; expected {}",
            self.recv_release_sequence
        );
        anyhow::ensure!(
            sequence != self.recv_completion_sequence,
            "mapped RDMA ring receive slot {sequence} has not completed"
        );
        let slot_index = sequence as usize % self.layout.depth;
        self.endpoint.post_recv_at(
            self.layout.slot_offset(sequence as usize),
            self.layout.slot_capacity_bytes,
            VERBS_HOST_RECV_WR_ID + slot_index as u64,
        )?;
        self.recv_release_sequence = self.recv_release_sequence.wrapping_add(1);
        Ok(())
    }
}

fn load_verbs_host_native_library() -> Result<Arc<NativeLibrary>> {
    let native_path = verbs_host_native_library_path().context(
        "native library not found; set DS41RT_NATIVE_LIB or build native/libds41rt_native.so with RDMA",
    )?;
    Ok(Arc::new(unsafe { NativeLibrary::load(&native_path) }?))
}

fn validate_mapped_ring_endpoint(
    endpoint: &VerbsHostNativeEndpointDescriptor,
    layout: VerbsHostRdmaRing,
    role: &str,
) -> Result<()> {
    anyhow::ensure!(
        endpoint.send_frame_bytes == layout.slot_capacity_bytes
            && endpoint.recv_frame_bytes == layout.slot_capacity_bytes,
        "mapped RDMA ring {role} endpoint frame capacities do not match the ring"
    );
    anyhow::ensure!(
        endpoint.send_registered_span_bytes == layout.registered_span_bytes
            && endpoint.recv_registered_span_bytes == layout.registered_span_bytes,
        "mapped RDMA ring {role} endpoint registered spans do not match the ring"
    );
    anyhow::ensure!(
        endpoint.max_send_wr as usize >= layout.depth
            && endpoint.max_recv_wr as usize >= layout.depth,
        "mapped RDMA ring {role} endpoint work-request capacity is below depth {}",
        layout.depth
    );
    Ok(())
}

fn validate_mapped_endpoint_buffer_view(
    view: Ds41rtRdmaRcEndpointBufferView,
    layout: VerbsHostRdmaRing,
    label: &str,
) -> Result<()> {
    anyhow::ensure!(
        !view.host_ptr.is_null() && !view.device_ptr.is_null(),
        "mapped RDMA ring {label} buffer has no host or CUDA address"
    );
    anyhow::ensure!(
        view.bytes == layout.registered_span_bytes,
        "mapped RDMA ring {label} buffer bytes {} do not match span {}",
        view.bytes,
        layout.registered_span_bytes
    );
    let expected_flags = DS41RT_HOST_BUFFER_FLAG_PINNED | DS41RT_HOST_BUFFER_FLAG_MAPPED;
    anyhow::ensure!(
        view.host_flags & expected_flags == expected_flags,
        "mapped RDMA ring {label} buffer flags {:#x} are not pinned and mapped",
        view.host_flags
    );
    Ok(())
}

fn mapped_ring_slot(
    view: Ds41rtRdmaRcEndpointBufferView,
    layout: VerbsHostRdmaRing,
    sequence: u64,
) -> Result<VerbsHostMappedRdmaSlot> {
    let slot_index = sequence as usize % layout.depth;
    let offset = layout.slot_offset(sequence as usize);
    anyhow::ensure!(
        offset <= view.bytes && layout.slot_capacity_bytes <= view.bytes - offset,
        "mapped RDMA ring slot {slot_index} exceeds its registered buffer"
    );
    Ok(VerbsHostMappedRdmaSlot {
        sequence,
        slot_index,
        capacity_bytes: layout.slot_capacity_bytes,
        host_ptr: unsafe { view.host_ptr.cast::<u8>().add(offset) },
        device_buffer: Ds41rtDeviceBuffer {
            ptr: unsafe { view.device_ptr.cast::<u8>().add(offset).cast() },
            bytes: layout.slot_capacity_bytes,
            device_id: view.device_id,
            flags: DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST,
        },
    })
}

fn verbs_host_expected_response_for_request(
    request: &ExpertProtocolV2Request,
) -> Result<ExpertProtocolV2Response> {
    let output_payload_bytes = (request.header.row_count as usize)
        .checked_mul(request.header.hidden_row_stride_bytes as usize)
        .context("verbs-host expected response payload byte count overflow")?;
    let response = ExpertProtocolV2Response::new_with_output_stride(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        request.header.row_count,
        request.header.hidden_dim,
        request.header.hidden_dtype,
        request.header.hidden_row_stride_bytes,
        ExpertProtocolV2Status::Ok,
        vec![0_u8; output_payload_bytes],
    )?;
    Ok(if request.debug_checksum_enabled() {
        response.with_debug_checksum()
    } else {
        response
    })
}

fn verbs_host_expected_response_wire_bytes(request: &ExpertProtocolV2Request) -> Result<usize> {
    let hidden_dim = request.header.hidden_dim as usize;
    let bf16_row_bytes = ExpertV2Dtype::Bf16.row_bytes(hidden_dim)?;
    let negotiated_row_bytes = if request.nvfp4_e2m1_fp8_e4m3_response_enabled() {
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(hidden_dim)?
    } else if request.fp8_e4m3_row_scaled_response_enabled() {
        ExpertV2Dtype::Fp8E4m3RowScaled.row_bytes(hidden_dim)?
    } else {
        request.header.hidden_row_stride_bytes as usize
    };
    // Executors may promote packed ingress to BF16 output. Capacity planning is
    // conservative so a session negotiated by a stream-plan frame can carry
    // any later completion slice.
    let output_row_bytes = negotiated_row_bytes
        .max(request.header.hidden_row_stride_bytes as usize)
        .max(bf16_row_bytes);
    let output_payload_bytes = (request.header.row_count as usize)
        .checked_mul(output_row_bytes)
        .context("verbs-host expected response payload byte count overflow")?;
    let row_index_bytes = (request.header.row_count as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .context("verbs-host expected response row-index byte count overflow")?;
    let header_bytes = if request.debug_checksum_enabled() {
        EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
    } else {
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
    };
    header_bytes
        // Persistent CUDA executors may complete a request as one row-indexed device response.
        // Reserve that prefix even when the request itself does not carry a streaming flag.
        .checked_add(row_index_bytes)
        .context("verbs-host expected response prefix byte count overflow")?
        .checked_add(output_payload_bytes)
        .context("verbs-host expected response wire byte count overflow")
}

fn validate_response_matches_request(
    response: &ExpertProtocolV2ResponseHeader,
    request: &ExpertProtocolV2Request,
) -> Result<()> {
    if response.request_id != request.header.request_id {
        bail!(
            "verbs-host ProtocolV2 response request_id {} did not match request_id {}",
            response.request_id,
            request.header.request_id
        );
    }
    if response.placement_version != request.header.placement_version {
        bail!(
            "verbs-host ProtocolV2 response placement_version {} did not match request placement_version {}",
            response.placement_version,
            request.header.placement_version
        );
    }
    if response.layer_id != request.header.layer_id {
        bail!(
            "verbs-host ProtocolV2 response layer_id {} did not match request layer_id {}",
            response.layer_id,
            request.header.layer_id
        );
    }
    Ok(())
}

fn validate_control_endpoint_metadata(
    endpoint: &VerbsHostRcEndpointDescriptor,
    expected_role: &str,
) -> Result<()> {
    if endpoint.role != expected_role {
        bail!(
            "verbs-host ProtocolV2 control endpoint role {} did not match expected {expected_role}",
            endpoint.role
        );
    }
    if endpoint.qp_num == 0 {
        bail!("verbs-host ProtocolV2 control endpoint qp_num must be non-zero");
    }
    if endpoint.psn > 0x00ff_ffff {
        bail!("verbs-host ProtocolV2 control endpoint psn exceeds 24 bits");
    }
    if !valid_gid_hex(&endpoint.gid_hex) {
        bail!("verbs-host ProtocolV2 control endpoint gid_hex must be 32 hex characters");
    }
    Ok(())
}

fn connect_control_stream(peer: &str, timeout: Duration) -> Result<TcpStream> {
    let mut addrs = peer
        .to_socket_addrs()
        .with_context(|| format!("resolving verbs-host ProtocolV2 peer {peer}"))?;
    let first = addrs
        .next()
        .with_context(|| format!("verbs-host ProtocolV2 peer {peer} resolved no addresses"))?;
    TcpStream::connect_timeout(&first, timeout)
        .with_context(|| format!("connecting verbs-host ProtocolV2 control plane to {peer}"))
}

fn configure_control_stream(stream: &TcpStream, timeout: Duration) -> Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(())
}

fn verbs_host_control_plane_closed(stream: &TcpStream) -> Result<bool> {
    let prior_timeout = stream
        .read_timeout()
        .context("reading verbs-host control-plane timeout")?;
    stream
        .set_read_timeout(Some(Duration::from_millis(1)))
        .context("setting verbs-host control-plane liveness timeout")?;
    let mut byte = [0_u8; 1];
    let peek_result = stream.peek(&mut byte);
    stream
        .set_read_timeout(prior_timeout)
        .context("restoring verbs-host control-plane timeout")?;
    match peek_result {
        Ok(0) => Ok(true),
        Ok(_) => Ok(false),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
            ) =>
        {
            Ok(false)
        }
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::NotConnected
            ) =>
        {
            Ok(true)
        }
        Err(error) => Err(error).context("peeking verbs-host control-plane liveness"),
    }
}

fn write_control<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

fn read_control<T: DeserializeOwned>(reader: &mut BufReader<TcpStream>) -> Result<T> {
    Ok(serde_json::from_value(read_control_value(reader)?)?)
}

fn read_control_value(reader: &mut BufReader<TcpStream>) -> Result<serde_json::Value> {
    let mut line = String::new();
    let bytes = reader.read_line(&mut line)?;
    if bytes == 0 {
        bail!("verbs-host ProtocolV2 control plane closed");
    }
    Ok(serde_json::from_str(line.trim_end())?)
}

fn control_message(value: &serde_json::Value) -> Result<&str> {
    value
        .get("message")
        .and_then(serde_json::Value::as_str)
        .context("verbs-host ProtocolV2 control message missing string message field")
}

fn distinct_endpoint_hosts(mut client_host: String, mut server_host: String) -> (String, String) {
    if client_host == server_host {
        client_host.push_str("-client");
        server_host.push_str("-server");
    }
    (client_host, server_host)
}

fn local_control_host(role: &str) -> String {
    env::var("HOSTNAME")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("local-{role}"))
}

fn next_local_psn(role: &str) -> u32 {
    let base = match role {
        "client" => 0x110000,
        "server" => 0x220000,
        _ => 0x330000,
    };
    let offset = VERBS_HOST_PSN_COUNTER.fetch_add(1, Ordering::Relaxed) & 0x0fff;
    base + offset
}

fn verbs_host_native_library_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("DS41RT_NATIVE_LIB") {
        return Some(PathBuf::from(path));
    }
    for candidate in [
        "native/build-cuda-sm120/libds41rt_native.so",
        "native/build-cuda/libds41rt_native.so",
        "native/build/libds41rt_native.so",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

fn verbs_host_ib_port_num() -> Result<u32> {
    parse_env_u32("DS41RT_VERBS_APP_IB_PORT_NUM", 1)
}

fn protocol_v2_transport_timing_enabled() -> bool {
    env::var(PROTOCOL_V2_TCP_TIMING_ENV)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms_optional(started: Option<Instant>) -> f64 {
    started.map(elapsed_ms).unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerbsHostRdmaRing {
    slot_capacity_bytes: usize,
    slot_stride_bytes: usize,
    depth: usize,
    registered_span_bytes: usize,
}

impl VerbsHostRdmaRing {
    fn new(slot_capacity_bytes: usize, alignment: usize, depth: usize) -> Result<Self> {
        Self::new_with_max_depth(
            slot_capacity_bytes,
            alignment,
            depth,
            VERBS_HOST_RDMA_RING_DEPTH,
        )
    }

    fn new_with_max_depth(
        slot_capacity_bytes: usize,
        alignment: usize,
        depth: usize,
        max_depth: usize,
    ) -> Result<Self> {
        if slot_capacity_bytes == 0 {
            bail!("persistent verbs-host RDMA ring slot capacity must be non-zero");
        }
        if depth == 0 || depth > max_depth {
            bail!("persistent verbs-host RDMA ring depth {depth} must be in 1..={max_depth}");
        }
        let slot_stride_bytes = align_up(slot_capacity_bytes, alignment)?;
        let registered_span_bytes = slot_stride_bytes
            .checked_mul(depth)
            .context("persistent verbs-host RDMA ring registered span overflow")?;
        Ok(Self {
            slot_capacity_bytes,
            slot_stride_bytes,
            depth,
            registered_span_bytes,
        })
    }

    fn slot_offset(self, sequence: usize) -> usize {
        (sequence % self.depth) * self.slot_stride_bytes
    }

    fn from_wire(
        slot_capacity_bytes: usize,
        slot_stride_bytes: usize,
        depth: usize,
        registered_span_bytes: usize,
    ) -> Result<Self> {
        Self::from_wire_with_max_depth(
            slot_capacity_bytes,
            slot_stride_bytes,
            depth,
            registered_span_bytes,
            VERBS_HOST_RDMA_RING_DEPTH,
        )
    }

    fn from_wire_with_max_depth(
        slot_capacity_bytes: usize,
        slot_stride_bytes: usize,
        depth: usize,
        registered_span_bytes: usize,
        max_depth: usize,
    ) -> Result<Self> {
        let expected = Self::new_with_max_depth(
            slot_capacity_bytes,
            crate::verbs_host_capabilities().preferred_alignment,
            depth,
            max_depth,
        )?;
        anyhow::ensure!(
            slot_stride_bytes == expected.slot_stride_bytes,
            "persistent verbs-host RDMA ring slot stride {slot_stride_bytes} did not match expected {}",
            expected.slot_stride_bytes
        );
        anyhow::ensure!(
            registered_span_bytes == expected.registered_span_bytes,
            "persistent verbs-host RDMA ring registered span {registered_span_bytes} did not match expected {}",
            expected.registered_span_bytes
        );
        Ok(expected)
    }
}

fn verbs_host_persistent_rings(
    config: &TcpTransportConfig,
    minimum_request_wire_bytes: usize,
    minimum_response_wire_bytes: usize,
) -> Result<(VerbsHostRdmaRing, VerbsHostRdmaRing)> {
    if config.max_frame_bytes == 0 {
        bail!("persistent verbs-host ProtocolV2 max frame bytes must be non-zero");
    }
    if minimum_request_wire_bytes > config.max_frame_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 request frame length {minimum_request_wire_bytes} exceeds max frame bytes {}",
            config.max_frame_bytes
        );
    }
    if minimum_response_wire_bytes > config.max_frame_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 expected response frame length {minimum_response_wire_bytes} exceeds max frame bytes {}",
            config.max_frame_bytes
        );
    }
    let alignment = crate::verbs_host_capabilities().preferred_alignment;
    let requested_slot_bytes = parse_env_usize(
        "DS41RT_VERBS_HOST_RING_SLOT_BYTES",
        VERBS_HOST_RDMA_RING_SLOT_BYTES,
    )?;
    let depth = parse_env_usize("DS41RT_VERBS_HOST_RING_DEPTH", VERBS_HOST_RDMA_RING_DEPTH)?;
    let request_capacity = requested_slot_bytes
        .max(minimum_request_wire_bytes)
        .min(config.max_frame_bytes);
    let response_capacity = requested_slot_bytes
        .max(minimum_response_wire_bytes)
        .min(config.max_frame_bytes);
    Ok((
        VerbsHostRdmaRing::new(request_capacity, alignment, depth)?,
        VerbsHostRdmaRing::new(response_capacity, alignment, depth)?,
    ))
}

fn default_control_timeout() -> Duration {
    Duration::from_secs(60)
}

fn active_poll_timeout_ms(timeout: Duration) -> u32 {
    let requested = timeout.as_millis().max(1);
    let default = u128::from(30_000_u32);
    requested.max(default).min(u128::from(u32::MAX)) as u32
}

fn parse_env_u32(name: &str, default: u32) -> Result<u32> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<u32>()
            .with_context(|| format!("invalid {name} value {value}")),
        _ => Ok(default),
    }
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<usize>()
            .with_context(|| format!("invalid {name} value {value}")),
        _ => Ok(default),
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - remainder)
        .context("verbs-host ProtocolV2 endpoint aligned span overflow")
}

fn validate_endpoint_descriptor(
    endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
    endpoint: &VerbsHostRcEndpointDescriptor,
    expected_role: &str,
) -> Result<()> {
    if endpoint.role != expected_role {
        bail!(
            "verbs-host ProtocolV2 handshake endpoint role {} does not match expected {expected_role}",
            endpoint.role
        );
    }
    if endpoint.host.trim().is_empty() {
        bail!("verbs-host ProtocolV2 handshake endpoint host is empty");
    }
    if endpoint.port_num == 0 {
        bail!("verbs-host ProtocolV2 handshake endpoint port_num must be non-zero");
    }
    if endpoint.qp_num == 0 {
        bail!("verbs-host ProtocolV2 handshake endpoint qp_num must be non-zero");
    }
    if endpoint.psn > 0x00ff_ffff {
        bail!("verbs-host ProtocolV2 handshake endpoint psn exceeds 24 bits");
    }
    if !valid_gid_hex(&endpoint.gid_hex) {
        bail!("verbs-host ProtocolV2 handshake endpoint gid_hex must be 32 hex characters");
    }
    if endpoint.max_send_wr == 0 {
        bail!("verbs-host ProtocolV2 handshake endpoint max_send_wr must be non-zero");
    }
    if endpoint.max_recv_wr == 0 {
        bail!("verbs-host ProtocolV2 handshake endpoint max_recv_wr must be non-zero");
    }
    if endpoint.max_sge < endpoint_plan.scatter_gather_entries_per_message as u32 {
        bail!(
            "verbs-host ProtocolV2 handshake endpoint max_sge {} is smaller than required {}",
            endpoint.max_sge,
            endpoint_plan.scatter_gather_entries_per_message
        );
    }
    Ok(())
}

fn validate_endpoint_direction(
    _endpoint_plan: &VerbsHostProtocolV2EndpointPlan,
    endpoint: &VerbsHostRcEndpointDescriptor,
    expected_send_frame_bytes: usize,
    expected_recv_frame_bytes: usize,
    expected_send_registered_span_bytes: usize,
    expected_recv_registered_span_bytes: usize,
    role: &str,
) -> Result<()> {
    if endpoint.send_frame_bytes != expected_send_frame_bytes {
        bail!(
            "verbs-host ProtocolV2 handshake {role} send_frame_bytes {} does not match expected {expected_send_frame_bytes}",
            endpoint.send_frame_bytes
        );
    }
    if endpoint.recv_frame_bytes != expected_recv_frame_bytes {
        bail!(
            "verbs-host ProtocolV2 handshake {role} recv_frame_bytes {} does not match expected {expected_recv_frame_bytes}",
            endpoint.recv_frame_bytes
        );
    }
    if endpoint.send_registered_span_bytes != expected_send_registered_span_bytes {
        bail!(
            "verbs-host ProtocolV2 handshake {role} send_registered_span_bytes {} does not match expected {expected_send_registered_span_bytes}",
            endpoint.send_registered_span_bytes
        );
    }
    if endpoint.recv_registered_span_bytes != expected_recv_registered_span_bytes {
        bail!(
            "verbs-host ProtocolV2 handshake {role} recv_registered_span_bytes {} does not match expected {expected_recv_registered_span_bytes}",
            endpoint.recv_registered_span_bytes
        );
    }
    Ok(())
}

fn validate_persistent_endpoint_capacity(
    endpoint: &VerbsHostRcEndpointDescriptor,
    expected_role: &str,
    expected_send_frame_bytes: usize,
    expected_recv_frame_bytes: usize,
    expected_send_registered_span_bytes: usize,
    expected_recv_registered_span_bytes: usize,
    expected_ring_depth: usize,
) -> Result<()> {
    validate_control_endpoint_metadata(endpoint, expected_role)?;
    if endpoint.send_frame_bytes != expected_send_frame_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} send_frame_bytes {} did not match expected {expected_send_frame_bytes}",
            endpoint.send_frame_bytes
        );
    }
    if endpoint.recv_frame_bytes != expected_recv_frame_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} recv_frame_bytes {} did not match expected {expected_recv_frame_bytes}",
            endpoint.recv_frame_bytes
        );
    }
    if endpoint.send_registered_span_bytes != expected_send_registered_span_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} send_registered_span_bytes {} did not match expected {expected_send_registered_span_bytes}",
            endpoint.send_registered_span_bytes
        );
    }
    if endpoint.recv_registered_span_bytes != expected_recv_registered_span_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} recv_registered_span_bytes {} did not match expected {expected_recv_registered_span_bytes}",
            endpoint.recv_registered_span_bytes
        );
    }
    if endpoint.send_frame_bytes > endpoint.send_registered_span_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} send frame capacity exceeds registered span"
        );
    }
    if endpoint.recv_frame_bytes > endpoint.recv_registered_span_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} recv frame capacity exceeds registered span"
        );
    }
    if endpoint.max_send_wr < expected_ring_depth as u32 {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} max_send_wr {} is smaller than ring depth {expected_ring_depth}",
            endpoint.max_send_wr
        );
    }
    if endpoint.max_recv_wr < expected_ring_depth as u32 {
        bail!(
            "persistent verbs-host ProtocolV2 {expected_role} max_recv_wr {} is smaller than ring depth {expected_ring_depth}",
            endpoint.max_recv_wr
        );
    }
    Ok(())
}

fn persistent_protocol_v2_request_wire_bytes_from_header(
    header: &[u8],
    request_capacity_wire_bytes: usize,
) -> Result<usize> {
    if header.len() < EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN {
        bail!(
            "persistent verbs-host ProtocolV2 request header too short: {}",
            header.len()
        );
    }
    let wire_bytes = u64::from_le_bytes(
        header[76..84]
            .try_into()
            .expect("wire bytes slice length is fixed"),
    );
    let wire_bytes = usize::try_from(wire_bytes)
        .context("persistent verbs-host ProtocolV2 request wire bytes overflowed usize")?;
    if wire_bytes < EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN {
        bail!(
            "persistent verbs-host ProtocolV2 request wire bytes {} shorter than header {}",
            wire_bytes,
            EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
        );
    }
    if wire_bytes > request_capacity_wire_bytes {
        bail!(
            "persistent verbs-host ProtocolV2 request wire bytes {} exceeded capacity {}",
            wire_bytes,
            request_capacity_wire_bytes
        );
    }
    Ok(wire_bytes)
}

fn validate_round_trip_usize(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!(
            "verbs-host ProtocolV2 round trip {name} {actual} did not match endpoint plan {expected}"
        );
    }
    Ok(())
}

fn valid_gid_hex(value: &str) -> bool {
    value.len() == 32 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

#[cfg(test)]
mod persistent_tests {
    use super::*;
    use crate::{ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};

    #[test]
    fn shared_cq_harvester_starts_and_stops_without_registered_qps() {
        drop(VerbsHostProtocolV2CqHarvester::new(3).unwrap());
    }

    #[test]
    fn rdma_device_map_selects_device_by_control_destination_ip() {
        let raw = "10.55.0.2=rocep1s0f0,10.55.0.252=roceP2p1s0f0";

        assert_eq!(
            parse_verbs_host_rdma_device_map(raw, "10.55.0.2".parse().unwrap()).unwrap(),
            Some("rocep1s0f0".to_owned())
        );
        assert_eq!(
            parse_verbs_host_rdma_device_map(raw, "10.55.0.252".parse().unwrap()).unwrap(),
            Some("roceP2p1s0f0".to_owned())
        );
        assert_eq!(
            parse_verbs_host_rdma_device_map(raw, "127.0.0.1".parse().unwrap()).unwrap(),
            None
        );
    }

    #[test]
    fn rdma_device_map_rejects_duplicate_local_ips() {
        let error = parse_verbs_host_rdma_device_map(
            "10.55.0.2=rail0,10.55.0.2=rail1",
            "10.55.0.2".parse().unwrap(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("duplicate local IP"));
    }

    #[test]
    fn persistent_control_plane_liveness_distinguishes_idle_and_closed_sockets() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();

        assert!(!verbs_host_control_plane_closed(&server).unwrap());
        client.shutdown(std::net::Shutdown::Both).unwrap();
        assert!(verbs_host_control_plane_closed(&server).unwrap());
    }

    #[test]
    fn persistent_endpoint_capacity_accepts_fixed_registered_arena() {
        let endpoint = persistent_endpoint_descriptor("client", 64 * 1024 * 1024);

        validate_persistent_endpoint_capacity(
            &endpoint,
            "client",
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            8,
        )
        .unwrap();
    }

    #[test]
    fn persistent_endpoint_capacity_rejects_frame_larger_than_registered_span() {
        let mut endpoint = persistent_endpoint_descriptor("server", 64 * 1024 * 1024);
        endpoint.send_registered_span_bytes = 1024;

        let err = validate_persistent_endpoint_capacity(
            &endpoint,
            "server",
            64 * 1024 * 1024,
            64 * 1024 * 1024,
            1024,
            64 * 1024 * 1024,
            8,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("send frame capacity exceeds registered span"));
    }

    #[test]
    fn persistent_ring_reuses_the_old_arena_as_eight_slots() {
        let ring = VerbsHostRdmaRing::new(8 * 1024 * 1024, 4096, 8).unwrap();

        assert_eq!(ring.slot_stride_bytes, 8 * 1024 * 1024);
        assert_eq!(ring.registered_span_bytes, 64 * 1024 * 1024);
        assert_eq!(ring.slot_offset(0), 0);
        assert_eq!(ring.slot_offset(7), 56 * 1024 * 1024);
        assert_eq!(ring.slot_offset(8), 0);
    }

    #[test]
    fn mapped_ring_allows_benchmark_depth_thirty_two_only() {
        let config = VerbsHostMappedRdmaRingConfig::new(8192, 32).unwrap();
        assert_eq!(config.layout().unwrap().depth, 32);

        let err = VerbsHostMappedRdmaRingConfig::new(8192, 33)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1..=32"));

        let err = VerbsHostRdmaRing::new(8192, 4096, 9)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1..=8"));
    }

    #[test]
    fn persistent_ring_rejects_wire_stride_mismatch() {
        let err = VerbsHostRdmaRing::from_wire(8192, 12_288, 8, 98_304)
            .unwrap_err()
            .to_string();

        assert!(err.contains("slot stride"));
    }

    #[test]
    fn persistent_retryable_error_detects_closed_control_plane() {
        let err = anyhow::anyhow!("verbs-host ProtocolV2 control plane closed")
            .context("reading persistent verbs-host control message");

        assert!(is_verbs_host_protocol_v2_persistent_retryable_error(&err));
    }

    #[test]
    fn persistent_retryable_error_detects_connection_loss() {
        let err = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "socket closed",
        ))
        .context("writing persistent verbs-host control message");

        assert!(is_verbs_host_protocol_v2_persistent_retryable_error(&err));
    }

    #[test]
    fn persistent_retryable_error_detects_stale_response_request_id() {
        let err = anyhow::anyhow!("ProtocolV2 response request_id 41 did not match request_id 42")
            .context("validating persistent verbs-host response");

        assert!(is_verbs_host_protocol_v2_persistent_retryable_error(&err));
    }

    #[test]
    fn persistent_retryable_error_detects_rdma_completion_failure() {
        let err = anyhow::anyhow!(
            "ds41rt_rdma_rc_endpoint_poll returned status 7: RDMA RC endpoint completion returned non-success status"
        )
        .context("finishing scheduler sparse routed ProtocolV2 TCP BF16 payload batch");

        assert!(is_verbs_host_protocol_v2_persistent_retryable_error(&err));
    }

    #[test]
    fn persistent_retryable_error_rejects_capacity_error() {
        let err = anyhow::anyhow!(
            "persistent verbs-host ProtocolV2 request frame length 1025 exceeds endpoint request capacity 1024"
        );

        assert!(!is_verbs_host_protocol_v2_persistent_retryable_error(&err));
    }

    #[test]
    fn persistent_request_wire_bytes_reads_protocol_v2_header() {
        let mut header = vec![0_u8; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
        header[76..84].copy_from_slice(&(4096_u64).to_le_bytes());

        let wire_bytes =
            persistent_protocol_v2_request_wire_bytes_from_header(&header, 8192).unwrap();

        assert_eq!(wire_bytes, 4096);
    }

    #[test]
    fn persistent_request_wire_bytes_rejects_capacity_overrun() {
        let mut header = vec![0_u8; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
        header[76..84].copy_from_slice(&(8193_u64).to_le_bytes());

        let err = persistent_protocol_v2_request_wire_bytes_from_header(&header, 8192)
            .unwrap_err()
            .to_string();

        assert!(err.contains("exceeded capacity"));
    }

    #[test]
    fn persistent_request_wire_bytes_rejects_short_frame() {
        let mut header = vec![0_u8; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
        header[76..84]
            .copy_from_slice(&((EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN - 1) as u64).to_le_bytes());

        let err = persistent_protocol_v2_request_wire_bytes_from_header(&header, 8192)
            .unwrap_err()
            .to_string();

        assert!(err.contains("shorter than header"));
    }

    #[test]
    fn persistent_response_chunk_assembler_restores_request_row_order() {
        let request = persistent_chunk_request(3);
        let first = persistent_response_chunk(&request, vec![2], vec![9, 10, 11, 12], true);
        let final_chunk =
            persistent_response_chunk(&request, vec![0, 1], vec![1, 2, 3, 4, 5, 6, 7, 8], false);
        let first_frame = first.encode().unwrap();
        let final_frame = final_chunk.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::new(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&first_frame).unwrap(),
            )
            .unwrap();
        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&final_frame).unwrap(),
            )
            .unwrap();
        let response = assembler.finish().unwrap();

        assert!(!response.row_indexed());
        assert_eq!(response.header.row_count, 3);
        assert_eq!(response.header.executor_id, 77);
        assert_eq!(
            response.partial_output_payload,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }

    #[test]
    fn persistent_response_chunk_assembler_rejects_duplicate_rows() {
        let request = persistent_chunk_request(2);
        let first = persistent_response_chunk(&request, vec![0], vec![1, 2, 3, 4], true);
        let duplicate = persistent_response_chunk(&request, vec![0], vec![5, 6, 7, 8], false);
        let first_frame = first.encode().unwrap();
        let duplicate_frame = duplicate.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::new(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&first_frame).unwrap(),
            )
            .unwrap();
        let error = assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&duplicate_frame).unwrap(),
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("emitted twice"));
    }

    #[test]
    fn persistent_response_chunk_assembler_rejects_incomplete_final_chunk() {
        let request = persistent_chunk_request(2);
        let final_chunk = persistent_response_chunk(&request, vec![1], vec![5, 6, 7, 8], false);
        let final_frame = final_chunk.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::new(&request);

        let error = assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&final_frame).unwrap(),
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("completed 1 of 2"));
    }

    #[test]
    fn row_sharded_response_validation_accepts_an_indexed_partition() {
        let request = persistent_chunk_request(10).with_spark_row_sharded_reduction();
        let partition = persistent_response_chunk(
            &request,
            vec![3, 4, 5],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            false,
        );
        let frame = partition.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&frame).unwrap(),
            )
            .unwrap();
        assert_eq!(assembler.finish_validation().unwrap().row_count, 3);
    }

    #[test]
    fn persistent_response_chunk_validation_does_not_assemble_payload() {
        let request = persistent_chunk_request(2);
        let first = persistent_response_chunk(&request, vec![1], vec![5, 6, 7, 8], true);
        let final_chunk = persistent_response_chunk(&request, vec![0], vec![1, 2, 3, 4], false);
        let first_frame = first.encode().unwrap();
        let final_frame = final_chunk.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&first_frame).unwrap(),
            )
            .unwrap();
        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&final_frame).unwrap(),
            )
            .unwrap();
        let header = assembler.finish_validation().unwrap();

        assert_eq!(header.row_count, 1);
        assert_eq!(header.executor_id, 77);
    }

    #[test]
    fn streamed_ingress_plan_accepts_zero_row_acknowledgement() {
        let base = persistent_chunk_request(3);
        let request = ExpertProtocolV2Request::new_stream_plan_with_hidden_stride(
            base.header.request_id,
            base.header.placement_version,
            base.header.layer_id,
            base.header.hidden_dim,
            base.header.hidden_dtype,
            base.header.hidden_row_stride_bytes,
            base.rows,
            base.routes,
            vec![1],
        )
        .unwrap();
        let acknowledgement = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            0,
            request.header.hidden_dim,
            ExpertV2Dtype::Bf16,
            4,
            ExpertProtocolV2Status::Ok,
            Vec::new(),
        )
        .unwrap()
        .with_executor_id(77);
        let frame = acknowledgement.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&frame).unwrap(),
            )
            .unwrap();
        assert_eq!(assembler.finish_validation().unwrap().row_count, 0);
    }

    #[test]
    fn regular_spark_reduction_accepts_follower_acknowledgement_for_streaming_dispatch() {
        let request = persistent_chunk_request(1).with_spark_reduction();
        let acknowledgement = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            0,
            request.header.hidden_dim,
            ExpertV2Dtype::Bf16,
            4,
            ExpertProtocolV2Status::Ok,
            Vec::new(),
        )
        .unwrap()
        .with_executor_id(77);
        let frame = acknowledgement.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&frame).unwrap(),
            )
            .unwrap();
        assert_eq!(assembler.finish_validation().unwrap().row_count, 0);
    }

    #[test]
    fn streamed_ingress_data_accepts_completed_logical_rows_outside_chunk() {
        let request = ExpertProtocolV2Request::new_stream_data(
            42,
            7,
            13,
            2,
            ExpertV2Dtype::Bf16,
            4,
            4,
            2,
            vec![0; 8],
            true,
        )
        .unwrap();
        let response = persistent_response_chunk(&request, vec![9], vec![1, 2, 3, 4], false);
        let frame = response.encode().unwrap();
        let mut assembler = ProtocolV2ResponseChunkAssembler::validation_only(&request);

        assembler
            .accept(
                &request,
                &ExpertProtocolV2ResponseView::parse(&frame).unwrap(),
            )
            .unwrap();
        assert_eq!(assembler.finish_validation().unwrap().row_count, 1);
    }

    #[test]
    fn expected_response_wire_bytes_cover_row_indexed_device_output() {
        let request = persistent_chunk_request(3);
        for request in [request.clone(), request.with_debug_checksum()] {
            let row_indices = (0..request.header.row_count).collect::<Vec<_>>();
            let expected = verbs_host_expected_response_for_request(&request)
                .unwrap()
                .with_row_indices(row_indices, false)
                .unwrap();
            assert_eq!(
                verbs_host_expected_response_wire_bytes(&request).unwrap(),
                expected.wire_stats().wire_bytes
            );
        }
    }

    #[test]
    fn mapped_request_device_payload_preserves_hidden_offset() {
        let frame = persistent_chunk_request(3).encode().unwrap();
        let request = ExpertProtocolV2RequestView::parse(&frame).unwrap();
        let hidden_offset = request.hidden_payload().as_ptr() as usize - frame.as_ptr() as usize;
        let device_base = 0x10_0000_usize;

        let payload = protocol_v2_request_device_payload(
            &frame,
            &request,
            Ds41rtDeviceBuffer {
                ptr: device_base as *mut std::ffi::c_void,
                bytes: frame.len(),
                device_id: 3,
                flags: DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST,
            },
            None,
            7,
        )
        .unwrap();

        assert_eq!(
            payload.hidden_payload.ptr as usize,
            device_base + hidden_offset
        );
        assert_eq!(payload.hidden_payload.bytes, request.hidden_payload().len());
        assert_eq!(payload.hidden_payload.device_id, 3);
        assert!(payload.response_slot.is_none());
        assert_eq!(payload.execution_lane, 7);
        assert_eq!(
            payload.hidden_payload.flags,
            DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST
        );
    }

    fn persistent_chunk_request(row_count: usize) -> ExpertProtocolV2Request {
        ExpertProtocolV2Request::new(
            42,
            7,
            13,
            2,
            ExpertV2Dtype::Bf16,
            (0..row_count)
                .map(|row| ExpertProtocolV2RowDescriptor {
                    row_id: row as u64,
                    source_kind: ExpertV2SourceKind::Prefill,
                    source_request_id: 100,
                    token_position: row as u64,
                    route_offset: 0,
                    route_count: 0,
                })
                .collect(),
            Vec::new(),
            vec![0; row_count * 4],
        )
        .unwrap()
    }

    fn persistent_response_chunk(
        request: &ExpertProtocolV2Request,
        row_indices: Vec<u32>,
        payload: Vec<u8>,
        more_chunks: bool,
    ) -> ExpertProtocolV2Response {
        ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            row_indices.len() as u32,
            2,
            ExpertV2Dtype::Bf16,
            4,
            ExpertProtocolV2Status::Ok,
            payload,
        )
        .unwrap()
        .with_row_indices(row_indices, more_chunks)
        .unwrap()
        .with_executor_id(77)
    }

    fn persistent_endpoint_descriptor(
        role: &str,
        capacity: usize,
    ) -> VerbsHostRcEndpointDescriptor {
        VerbsHostRcEndpointDescriptor {
            role: role.to_owned(),
            host: "test-host".to_owned(),
            port_num: 1,
            qp_num: 0x1234,
            psn: 0x010203,
            gid_hex: "00000000000000000000ffff0a000001".to_owned(),
            send_frame_bytes: capacity,
            recv_frame_bytes: capacity,
            send_registered_span_bytes: capacity,
            recv_registered_span_bytes: capacity,
            max_send_wr: 8,
            max_recv_wr: 8,
            max_sge: 1,
        }
    }
}
