//! Persistent QP sessions polled directly by a thread-local GPU owner.
use super::*;

pub struct LocalVerbsExpertConnection {
    stream: TcpStream,
    library: Arc<NativeLibrary>,
    endpoint: NativeRdmaEndpoint,
    start: VerbsHostProtocolV2PersistentStart,
    request_ring: VerbsHostRdmaRing,
    response_ring: VerbsHostRdmaRing,
    request_recv_view: Ds41rtRdmaRcEndpointBufferView,
    response_send_view: Ds41rtRdmaRcEndpointBufferView,
    response_frame: ExpertProtocolV2FrameBuffer,
    request_recv_sequence: usize,
    response_send_sequence: usize,
    response_send_in_flight: usize,
    response_copy_stream: Option<VerbsHostCudaStream>,
    last_activity: Instant,
    last_liveness: Instant,
    /// Startup-resolved diagnostics flag; see `protocol_v2_timing_from_env`.
    timing: bool,
}
// No operation can race: polling requires &mut self, and all registered views
// remain owned by the endpoint. Like VerbsHostMappedRdmaRing, a session may move
// from its bootstrap thread to the GPU owner, but is never shared concurrently.
unsafe impl Send for LocalVerbsExpertConnection {}

impl LocalVerbsExpertConnection {
    /// Bootstrap only; call on an admission thread, then transfer to the GPU owner.
    /// `timing` is the startup-resolved diagnostics flag; it is not re-read from
    /// the environment on the poll path.
    pub fn accept(stream: TcpStream, max_frame_bytes: usize, timing: bool) -> Result<Self> {
        verbs_host_preflight()?;
        configure_control_stream(&stream, default_control_timeout())?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let start: VerbsHostProtocolV2PersistentStart = read_control(&mut reader)?;
        anyhow::ensure!(
            start.message == "protocol_v2_persistent_start",
            "native owner requires persistent RoCE bootstrap"
        );
        anyhow::ensure!(
            start.request_capacity_wire_bytes <= max_frame_bytes
                && start.response_capacity_wire_bytes <= max_frame_bytes,
            "peer registration exceeds native frame budget"
        );
        anyhow::ensure!(
            start.ring_depth <= 8,
            "native peer ring depth exceeds admission budget"
        );
        let path =
            verbs_host_native_library_path().context("native RoCE library not configured")?;
        let library = Arc::new(unsafe { NativeLibrary::load(&path) }?);
        Self::initialize(stream, library, start, timing)
    }

    pub(super) fn initialize(
        mut stream: TcpStream,
        library: Arc<NativeLibrary>,
        start: VerbsHostProtocolV2PersistentStart,
        timing: bool,
    ) -> Result<Self> {
        let rdma_device = verbs_host_rdma_device_for_stream(&stream)?;
        let request_ring = VerbsHostRdmaRing::from_wire(
            start.request_capacity_wire_bytes,
            start.request_slot_stride_bytes,
            start.ring_depth,
            start.request_registered_span_bytes,
        )?;
        let response_ring = VerbsHostRdmaRing::from_wire(
            start.response_capacity_wire_bytes,
            start.response_slot_stride_bytes,
            start.ring_depth,
            start.response_registered_span_bytes,
        )?;
        let endpoint = NativeRdmaEndpoint::create_from_wire_bytes_mapped_on_device(
            Arc::clone(&library),
            "server",
            start.request_capacity_wire_bytes,
            start.response_capacity_wire_bytes,
            start.request_registered_span_bytes,
            start.response_registered_span_bytes,
            next_local_psn("server"),
            rdma_device.as_deref(),
        )?;
        let request_recv_view = endpoint.recv_buffer_view()?;
        let response_send_view = endpoint.send_buffer_view()?;
        validate_mapped_endpoint_buffer_view(
            request_recv_view,
            request_ring,
            "persistent receive",
        )?;
        validate_mapped_endpoint_buffer_view(response_send_view, response_ring, "persistent send")?;
        let (_client_host, server_host) = distinct_endpoint_hosts(
            start.client_endpoint.host.clone(),
            local_control_host("server"),
        );
        let server_endpoint = endpoint.verbs_descriptor("server", &server_host);
        validate_persistent_endpoint_capacity(
            &start.client_endpoint,
            "client",
            start.request_capacity_wire_bytes,
            start.response_capacity_wire_bytes,
            start.request_registered_span_bytes,
            start.response_registered_span_bytes,
            request_ring.depth,
        )?;
        validate_persistent_endpoint_capacity(
            &server_endpoint,
            "server",
            start.response_capacity_wire_bytes,
            start.request_capacity_wire_bytes,
            start.response_registered_span_bytes,
            start.request_registered_span_bytes,
            response_ring.depth,
        )?;
        endpoint.connect(&start.client_native_endpoint)?;
        for slot in 0..request_ring.depth {
            endpoint.post_recv_at(
                request_ring.slot_offset(slot),
                request_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64,
            )?;
        }
        if timing {
            let server_native_endpoint = endpoint.native_descriptor();
            eprintln!(
            "protocol_v2_verbs_persistent_server_connect ring_depth={} request_capacity={} request_stride={} request_span={} response_capacity={} response_stride={} response_span={} server_device={} server_gid={} server_status=\"{}\" client_device={} client_gid={} client_status=\"{}\"",
            request_ring.depth,
            start.request_capacity_wire_bytes,
            request_ring.slot_stride_bytes,
            request_ring.registered_span_bytes,
            start.response_capacity_wire_bytes,
            response_ring.slot_stride_bytes,
            response_ring.registered_span_bytes,
            server_native_endpoint.device_name,
            server_native_endpoint.gid_hex,
            server_native_endpoint.status,
            start.client_native_endpoint.device_name,
            start.client_native_endpoint.gid_hex,
            start.client_native_endpoint.status
        );
        }
        write_control(
            &mut stream,
            &VerbsHostProtocolV2PersistentReady {
                message: "protocol_v2_persistent_ready".to_owned(),
                server_endpoint,
                server_native_endpoint: endpoint.native_descriptor(),
            },
        )?;

        Ok(Self {
            stream,
            library,
            endpoint,
            start,
            request_ring,
            response_ring,
            request_recv_view,
            response_send_view,
            response_frame: ExpertProtocolV2FrameBuffer::new(),
            request_recv_sequence: 0,
            response_send_sequence: 0,
            response_send_in_flight: 0,
            response_copy_stream: None,
            last_activity: Instant::now(),
            last_liveness: Instant::now(),
            timing,
        })
    }

    /// Execute at most one request. The callback must finish all GPU access to
    /// request/response slots before returning, including on errors. Idle returns
    /// immediately so other QPs can progress on the same GPU owner.
    pub fn poll<F>(&mut self, mut execute: F) -> Result<bool>
    where
        F: FnMut(
            &ExpertProtocolV2RequestView<'_>,
            ProtocolV2RequestDevicePayload,
            &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
        ) -> Result<()>,
    {
        let timing_enabled = self.timing;
        let total_started = timing_enabled.then(Instant::now);
        let poll_recv_started = timing_enabled.then(Instant::now);
        let stats = self.endpoint.try_poll(0, 1)?;
        if stats.recv_completions == 0 {
            if self.last_activity.elapsed() >= Duration::from_secs(1)
                && self.last_liveness.elapsed() >= Duration::from_secs(1)
            {
                self.last_liveness = Instant::now();
                anyhow::ensure!(
                    !verbs_host_control_plane_closed(&self.stream)?,
                    "native RoCE peer closed"
                );
            }
            return Ok(false);
        }
        self.last_activity = Instant::now();
        let endpoint = &self.endpoint;
        let library = &self.library;
        let start = &self.start;
        let request_ring = self.request_ring;
        let response_ring = self.response_ring;
        let request_recv_view = self.request_recv_view;
        let response_send_view = self.response_send_view;
        let response_frame = &mut self.response_frame;
        let response_copy_stream = &mut self.response_copy_stream;
        let mut request_recv_sequence = self.request_recv_sequence;
        let mut response_send_sequence = self.response_send_sequence;
        let mut response_send_in_flight = self.response_send_in_flight;
        let result = (|| -> Result<()> {
            let poll_recv_ms = elapsed_ms_optional(poll_recv_started);
            let request_recv_slot = mapped_ring_slot(
                request_recv_view,
                request_ring,
                request_recv_sequence as u64,
            )?;
            let request_storage = unsafe {
                std::slice::from_raw_parts(
                    request_recv_slot.host_ptr.cast_const(),
                    request_recv_slot.capacity_bytes,
                )
            };
            let copy_header_ms = 0.0_f64;
            let request_wire_bytes = persistent_protocol_v2_request_wire_bytes_from_header(
                &request_storage[..EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN],
                start.request_capacity_wire_bytes,
            )?;
            let resize_ms = 0.0_f64;
            let copy_request_ms = 0.0_f64;
            let mut post_recv_ms = 0.0_f64;
            let parse_started = timing_enabled.then(Instant::now);
            let request =
                ExpertProtocolV2RequestView::parse(&request_storage[..request_wire_bytes])
                    .context("parsing persistent verbs-host ProtocolV2 request frame")?;
            let parse_ms = elapsed_ms_optional(parse_started);
            let request_id = request.header.request_id;
            let layer_id = request.header.layer_id;
            let row_count = request.header.row_count;
            let route_count = request.header.route_count;
            if timing_enabled {
                eprintln!(
                "protocol_v2_verbs_persistent_server_received execution_lane={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_capacity={} poll_recv_ms={:.3} copy_header_ms={:.3} resize_ms={:.3} copy_request_ms={:.3} post_recv_ms={:.3} parse_ms={:.3}",
                start.execution_lane,
                request_id,
                layer_id,
                row_count,
                route_count,
                request_wire_bytes,
                start.response_capacity_wire_bytes,
                poll_recv_ms,
                copy_header_ms,
                resize_ms,
                copy_request_ms,
                post_recv_ms,
                parse_ms
            );
            }
            let mut response_frames = 0_usize;
            let mut response_wire_bytes = 0_usize;
            let mut encode_ms = 0.0_f64;
            let mut send_ms = 0.0_f64;
            let mut poll_send_ms = 0.0_f64;
            let mut direct_device_response_frames = 0_usize;
            let mut final_response_emitted = false;
            if response_send_in_flight == response_ring.depth {
                let poll_send_started = timing_enabled.then(Instant::now);
                endpoint.poll(1, 0, default_control_timeout())?;
                poll_send_ms += elapsed_ms_optional(poll_send_started);
                response_send_in_flight -= 1;
            }
            let first_response_slot = mapped_ring_slot(
                response_send_view,
                response_ring,
                response_send_sequence as u64,
            )?;
            let device_payload = protocol_v2_request_device_payload(
                &request_storage[..request_wire_bytes],
                &request,
                request_recv_slot.device_buffer,
                Some(first_response_slot.device_buffer),
                start.execution_lane,
            )?;
            let execute_started = timing_enabled.then(Instant::now);
            execute(&request, device_payload, &mut |response| {
                if final_response_emitted {
                    bail!(
                        "persistent verbs-host executor emitted a response after the final chunk"
                    );
                }
                if response_send_in_flight == response_ring.depth {
                    let poll_send_started = timing_enabled.then(Instant::now);
                    endpoint.poll(1, 0, default_control_timeout())?;
                    poll_send_ms += elapsed_ms_optional(poll_send_started);
                    response_send_in_flight -= 1;
                }
                let response_has_more = response.more_chunks();
                let response_send_slot = response_send_sequence % response_ring.depth;
                let response_send_offset = response_ring.slot_offset(response_send_sequence);
                let encode_started = timing_enabled.then(Instant::now);
                let response_wire_bytes_for_chunk = match response {
                    ProtocolV2ExecutorResponseRef::Host(response) => {
                        let response_prefix =
                            response_frame.encode_borrowed_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.len())
                            .context("persistent verbs-host response byte count overflow")?;
                        encode_ms += elapsed_ms_optional(encode_started);
                        anyhow::ensure!(
                            wire_bytes <= start.response_capacity_wire_bytes,
                            "persistent verbs-host ProtocolV2 executor response frame bytes {wire_bytes} exceeded response capacity {}",
                            start.response_capacity_wire_bytes
                        );
                        let send_started = timing_enabled.then(Instant::now);
                        endpoint.send_parts_at(
                            response_prefix,
                            response.partial_output_payload,
                            response_send_offset,
                            VERBS_HOST_SEND_WR_ID + response_send_slot as u64,
                        )?;
                        send_ms += elapsed_ms_optional(send_started);
                        wire_bytes
                    }
                    ProtocolV2ExecutorResponseRef::Device(response) => {
                        let response_prefix =
                            response_frame.encode_device_response_prefix(&response)?;
                        let wire_bytes = response_prefix
                            .len()
                            .checked_add(response.partial_output_payload.bytes)
                            .context("persistent verbs-host device response byte count overflow")?;
                        encode_ms += elapsed_ms_optional(encode_started);
                        anyhow::ensure!(
                            wire_bytes <= start.response_capacity_wire_bytes,
                            "persistent verbs-host ProtocolV2 executor device response frame bytes {wire_bytes} exceeded response capacity {}",
                            start.response_capacity_wire_bytes
                        );
                        let slot = mapped_ring_slot(
                            response_send_view,
                            response_ring,
                            response_send_sequence as u64,
                        )?;
                        let send_started = timing_enabled.then(Instant::now);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                response_prefix.as_ptr(),
                                slot.host_ptr,
                                response_prefix.len(),
                            );
                        }
                        let destination = protocol_v2_device_buffer_slice(
                            slot.device_buffer,
                            response_prefix.len(),
                            response.partial_output_payload.bytes,
                            "persistent verbs-host response payload",
                        )?;
                        if device_buffers_match(destination, response.partial_output_payload) {
                            direct_device_response_frames += 1;
                        } else {
                            if response_copy_stream.is_none() {
                                *response_copy_stream =
                                    Some(VerbsHostCudaStream::create(Arc::clone(&library))?);
                            }
                            let copy_stream = response_copy_stream
                                .as_ref()
                                .expect("response copy stream was created above")
                                .raw;
                            unsafe {
                                library.copy_d2d_async(
                                    destination,
                                    response.partial_output_payload,
                                    response.partial_output_payload.bytes,
                                    copy_stream,
                                )?;
                                library.cuda_stream_synchronize(copy_stream)?;
                            }
                        }
                        endpoint.post_send_at(
                            response_send_offset,
                            wire_bytes,
                            VERBS_HOST_SEND_WR_ID + response_send_slot as u64,
                        )?;
                        send_ms += elapsed_ms_optional(send_started);
                        wire_bytes
                    }
                };
                response_send_sequence = response_send_sequence.wrapping_add(1);
                response_send_in_flight += 1;
                response_frames += 1;
                response_wire_bytes = response_wire_bytes
                    .checked_add(response_wire_bytes_for_chunk)
                    .context("persistent verbs-host response wire byte count overflow")?;
                final_response_emitted = !response_has_more;
                Ok(())
            })?;
            let execute_ms = elapsed_ms_optional(execute_started);
            if response_frames == 0 || !final_response_emitted {
                bail!("persistent verbs-host executor did not emit a final response chunk");
            }
            let post_recv_started = timing_enabled.then(Instant::now);
            endpoint.post_recv_at(
                request_ring.slot_offset(request_recv_sequence),
                request_ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + request_recv_slot.slot_index as u64,
            )?;
            request_recv_sequence = request_recv_sequence.wrapping_add(1);
            post_recv_ms = elapsed_ms_optional(post_recv_started);
            if timing_enabled {
                eprintln!(
                "protocol_v2_verbs_persistent_server_roundtrip_timing execution_lane={} request_id={} layer_id={} rows={} routes={} request_wire_bytes={} response_frames={} direct_device_response_frames={} response_wire_bytes={} executor={} poll_recv_ms={:.3} copy_header_ms={:.3} resize_ms={:.3} copy_request_ms={:.3} post_recv_ms={:.3} parse_ms={:.3} execute_ms={:.3} encode_ms={:.3} send_ms={:.3} poll_send_ms={:.3} total_ms={:.3}",
                start.execution_lane,
                request_id,
                layer_id,
                row_count,
                route_count,
                request_wire_bytes,
                response_frames,
                direct_device_response_frames,
                response_wire_bytes,
                "local-owner",
                poll_recv_ms,
                copy_header_ms,
                resize_ms,
                copy_request_ms,
                post_recv_ms,
                parse_ms,
                execute_ms,
                encode_ms,
                send_ms,
                poll_send_ms,
                elapsed_ms_optional(total_started)
            );
            }
            Ok(())
        })();
        self.request_recv_sequence = request_recv_sequence;
        self.response_send_sequence = response_send_sequence;
        self.response_send_in_flight = response_send_in_flight;
        result.map(|()| true)
    }
}
