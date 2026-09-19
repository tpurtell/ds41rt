use super::bench_rdma_reduce::{run_reduction_follower, run_reduction_root};
use crate::cli::BenchRdmaRingArgs;
use anyhow::{bail, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, Ds41rtRouteShardReductionBuffers, NativeLibrary, DS41RT_ROUTE_SHARD_LOCAL_F32,
    DS41RT_ROUTE_SHARD_WIRE_BF16, DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
    DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
};
use ds41rt_transport::{
    TcpTransportConfig, VerbsHostMappedRdmaPollStats, VerbsHostMappedRdmaRing,
    VerbsHostMappedRdmaRingConfig, VerbsHostMappedRdmaSlot,
};
use serde::Serialize;
use std::collections::VecDeque;
use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

const TIMELINE_HEADER_BYTES: usize = 64;
const TIMELINE_MAGIC: &[u8; 8] = b"GLMQTLN1";
const TIMELINE_VERSION: u16 = 1;
const TIMELINE_REQUEST_KIND: u16 = 1;
const TIMELINE_RESPONSE_KIND: u16 = 2;

#[derive(Debug, Serialize)]
struct MappedRdmaRingReport {
    role: String,
    address: String,
    slot_bytes: usize,
    depth: usize,
    window: usize,
    gpu_echo: bool,
    warmup_iterations: usize,
    iterations: usize,
    elapsed_ms: f64,
    roundtrips_per_second: f64,
    bidirectional_payload_gbps: f64,
    amortized_roundtrip_us: Option<f64>,
    estimated_one_way_latency_us: Option<f64>,
    p50_roundtrip_us: Option<f64>,
    p95_roundtrip_us: Option<f64>,
    p99_roundtrip_us: Option<f64>,
}

#[derive(Debug, Serialize)]
struct MappedRdmaFanoutReport {
    role: String,
    peers: Vec<String>,
    wire_codec: String,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    slot_bytes: usize,
    warmup_iterations: usize,
    iterations: usize,
    elapsed_ms: f64,
    owner_steps_per_second: f64,
    average_owner_step_us: f64,
    mapped_reduce_kernel_us: f64,
}

#[derive(Debug, Default, Serialize)]
struct TimelineCqReport {
    poll_calls: usize,
    poll_iterations: u64,
    send_completions: usize,
    recv_completions: usize,
}

impl TimelineCqReport {
    fn record(&mut self, stats: VerbsHostMappedRdmaPollStats) {
        if stats == VerbsHostMappedRdmaPollStats::default() {
            return;
        }
        self.poll_calls += 1;
        self.poll_iterations += stats.poll_iterations;
        self.send_completions += stats.send_completions;
        self.recv_completions += stats.recv_completions;
    }
}

#[derive(Debug, Clone, Serialize)]
struct TimelineSummary {
    samples: usize,
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

#[derive(Debug, Serialize)]
struct MappedRdmaQueueTimelineReport {
    benchmark: String,
    role: String,
    address: String,
    network_label: String,
    pre_firmware: bool,
    request_bytes: usize,
    response_bytes: usize,
    slot_bytes: usize,
    depth: usize,
    window: usize,
    compute_delay_us: u64,
    validation_iterations: usize,
    warmup_iterations: usize,
    iterations: usize,
    elapsed_ms: f64,
    roundtrips_per_second: f64,
    bidirectional_payload_gbps: f64,
    thread_cpu_ms: Option<f64>,
    thread_cpu_fraction: Option<f64>,
    wraparound_generations: usize,
    full_payloads_validated: usize,
    sampled_payloads_validated: usize,
    generation_checks: usize,
    credit_exhaustions: usize,
    max_outstanding: usize,
    max_queue_depth: usize,
    cq: TimelineCqReport,
    reserve_send: Option<TimelineSummary>,
    post_send: Option<TimelineSummary>,
    receive_wait: Option<TimelineSummary>,
    request_validate: Option<TimelineSummary>,
    queue_wait: Option<TimelineSummary>,
    compute_placeholder: Option<TimelineSummary>,
    response_validate: Option<TimelineSummary>,
    roundtrip: Option<TimelineSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimelineHeader {
    kind: u16,
    sequence: u64,
    generation: u64,
    slot_index: u32,
    frame_bytes: u32,
    payload_seed: u32,
}

#[derive(Debug)]
struct QueuedTimelineRequest {
    slot: VerbsHostMappedRdmaSlot,
    header: TimelineHeader,
    received_at: Instant,
    validate_us: f64,
}

#[derive(Debug, Default)]
struct TimelineMeasurements {
    cq: TimelineCqReport,
    reserve_send_us: Vec<f64>,
    post_send_us: Vec<f64>,
    receive_wait_us: Vec<f64>,
    request_validate_us: Vec<f64>,
    queue_wait_us: Vec<f64>,
    compute_placeholder_us: Vec<f64>,
    response_validate_us: Vec<f64>,
    roundtrip_us: Vec<f64>,
    full_payloads_validated: usize,
    sampled_payloads_validated: usize,
    generation_checks: usize,
    credit_exhaustions: usize,
    max_outstanding: usize,
    max_queue_depth: usize,
}

pub(crate) fn run_bench_rdma_ring(args: BenchRdmaRingArgs) -> Result<()> {
    anyhow::ensure!(
        args.slot_bytes >= std::mem::size_of::<u64>(),
        "slot bytes are too small"
    );
    anyhow::ensure!(args.iterations > 0, "iterations must be non-zero");
    anyhow::ensure!(args.window > 0, "window must be non-zero");
    anyhow::ensure!(args.window <= args.depth, "window cannot exceed ring depth");
    if matches!(args.mode.as_str(), "timeline-server" | "timeline-client") {
        let request_bytes = timeline_request_bytes(&args);
        let response_bytes = timeline_response_bytes(&args);
        anyhow::ensure!(
            request_bytes >= TIMELINE_HEADER_BYTES && request_bytes <= args.slot_bytes,
            "timeline request bytes must be in {TIMELINE_HEADER_BYTES}..={}",
            args.slot_bytes
        );
        anyhow::ensure!(
            response_bytes >= TIMELINE_HEADER_BYTES && response_bytes <= args.slot_bytes,
            "timeline response bytes must be in {TIMELINE_HEADER_BYTES}..={}",
            args.slot_bytes
        );
        anyhow::ensure!(
            args.iterations >= args.depth.saturating_mul(2),
            "timeline iterations must cover at least two ring generations"
        );
    }
    let ring_config = VerbsHostMappedRdmaRingConfig::new(args.slot_bytes, args.depth)?;
    let transport = TcpTransportConfig { timing: false,
        timeout: Duration::from_millis(args.timeout_ms),
        max_frame_bytes: args.slot_bytes,
    };
    match args.mode.as_str() {
        "server" => run_server(&args, &transport),
        "client" => run_client(&args, &transport, ring_config),
        "fanout-client" => run_fanout_client(&args, &transport, ring_config),
        "reduce-root" => run_reduction_root(&args, &transport, ring_config),
        "reduce-follower" => run_reduction_follower(&args, &transport),
        "timeline-server" => run_timeline_server(&args, &transport),
        "timeline-client" => run_timeline_client(&args, &transport, ring_config),
        other => bail!(
            "unsupported mapped RDMA ring benchmark mode {other}; use server, client, fanout-client, reduce-root, reduce-follower, timeline-server, or timeline-client"
        ),
    }
}

fn run_fanout_client(
    args: &BenchRdmaRingArgs,
    transport: &TcpTransportConfig,
    config: VerbsHostMappedRdmaRingConfig,
) -> Result<()> {
    const HEADER_BYTES: usize = 64;
    let peers = args
        .peers
        .as_deref()
        .context("mapped RDMA fanout client requires --peers HOST:PORT,...")?
        .split(',')
        .map(str::trim)
        .filter(|peer| !peer.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        peers.len() == 3,
        "mapped RDMA fanout benchmark requires exactly three peers"
    );
    anyhow::ensure!(
        args.rows > 0 && args.row_width > 0,
        "rows and row width must be positive"
    );
    anyhow::ensure!(
        args.kernel_iterations > 0,
        "kernel iterations must be positive"
    );
    let (wire_dtype, peer_row_stride_bytes) = wire_codec(&args.wire_codec, args.row_width)?;
    let payload_bytes = args
        .rows
        .checked_mul(peer_row_stride_bytes)
        .context("mapped RDMA fanout payload byte count overflow")?;
    let required_slot_bytes = HEADER_BYTES
        .checked_add(payload_bytes)
        .context("mapped RDMA fanout slot byte count overflow")?;
    anyhow::ensure!(
        args.slot_bytes >= required_slot_bytes,
        "mapped RDMA fanout slot bytes {} are below required {required_slot_bytes}",
        args.slot_bytes
    );

    let mut rings = peers
        .iter()
        .map(|peer| VerbsHostMappedRdmaRing::connect(peer, transport, config))
        .collect::<Result<Vec<_>>>()?;
    let native = load_native_library(args.native_lib.as_deref())?;
    let values = args
        .rows
        .checked_mul(args.row_width)
        .context("mapped RDMA fanout value count overflow")?;
    let local_values = vec![1.0_f32; values];
    let mut local = native.alloc_device_buffer(values * std::mem::size_of::<f32>())?;
    native.copy_h2d(local, f32_bytes(&local_values))?;

    let mut mapped_reduce_kernel_us = None;
    run_fanout_phase(
        &native,
        &mut rings,
        local,
        args,
        wire_dtype,
        peer_row_stride_bytes,
        HEADER_BYTES,
        0,
        args.warmup_iterations,
        &mut mapped_reduce_kernel_us,
    )?;
    let started = Instant::now();
    run_fanout_phase(
        &native,
        &mut rings,
        local,
        args,
        wire_dtype,
        peer_row_stride_bytes,
        HEADER_BYTES,
        args.warmup_iterations as u64,
        args.iterations,
        &mut mapped_reduce_kernel_us,
    )?;
    let elapsed = started.elapsed();
    for ring in &mut rings {
        ring.flush_sends()?;
    }
    let mut output = vec![0_u8; values * std::mem::size_of::<f32>()];
    native.copy_d2h(&mut output, local)?;
    for value in bytes_to_f32(&output) {
        anyhow::ensure!(
            (value - 1.0).abs() < 1.0e-6,
            "fanout reduction output was {value}"
        );
    }
    native.free_device_buffer(&mut local)?;

    let elapsed_seconds = elapsed.as_secs_f64();
    println!(
        "{}",
        serde_json::to_string_pretty(&MappedRdmaFanoutReport {
            role: "fanout-client".to_owned(),
            peers,
            wire_codec: args.wire_codec.clone(),
            rows: args.rows,
            row_width: args.row_width,
            peer_row_stride_bytes,
            slot_bytes: args.slot_bytes,
            warmup_iterations: args.warmup_iterations,
            iterations: args.iterations,
            elapsed_ms: elapsed_seconds * 1000.0,
            owner_steps_per_second: args.iterations as f64 / elapsed_seconds,
            average_owner_step_us: elapsed_seconds * 1e6 / args.iterations as f64,
            mapped_reduce_kernel_us: mapped_reduce_kernel_us
                .context("mapped reduction kernel timing was not captured")?,
        })?
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fanout_phase(
    native: &NativeLibrary,
    rings: &mut [VerbsHostMappedRdmaRing],
    local: Ds41rtDeviceBuffer,
    args: &BenchRdmaRingArgs,
    wire_dtype: u32,
    peer_row_stride_bytes: usize,
    header_bytes: usize,
    marker_base: u64,
    iterations: usize,
    mapped_reduce_kernel_us: &mut Option<f64>,
) -> Result<()> {
    let peer_payload_bytes = args.rows * peer_row_stride_bytes;
    for iteration in 0..iterations {
        let marker = marker_base.wrapping_add(iteration as u64);
        for ring in rings.iter_mut() {
            let slot = ring.reserve_send_slot()?;
            unsafe {
                std::ptr::write_unaligned(slot.host_ptr.cast::<u64>(), marker.to_le());
            }
            ring.post_reserved_send(header_bytes + peer_payload_bytes)?;
        }
        let received = rings
            .iter_mut()
            .map(VerbsHostMappedRdmaRing::wait_recv_slot)
            .collect::<Result<Vec<_>>>()?;
        for slot in &received {
            let actual = unsafe { u64::from_le(std::ptr::read_unaligned(slot.host_ptr.cast())) };
            anyhow::ensure!(
                actual == marker,
                "fanout response marker {actual} did not match {marker}"
            );
        }
        let mut peer_buffers = [Ds41rtDeviceBuffer::default(); 3];
        for (dst, slot) in peer_buffers.iter_mut().zip(&received) {
            *dst = device_buffer_slice(slot.device_buffer, header_bytes, peer_payload_bytes)?;
        }
        let buffers = Ds41rtRouteShardReductionBuffers {
            local,
            peers: peer_buffers,
            output_f32: local,
        };
        if mapped_reduce_kernel_us.is_none() {
            *mapped_reduce_kernel_us = Some(time_mapped_reduction_kernel(
                native,
                &buffers,
                args,
                peer_row_stride_bytes,
                wire_dtype,
            )?);
        }
        unsafe {
            native.cuda_reduce_route_shards_to_f32_async(
                &buffers,
                args.rows,
                args.row_width,
                peer_row_stride_bytes,
                DS41RT_ROUTE_SHARD_LOCAL_F32,
                wire_dtype,
                3,
                std::ptr::null_mut(),
            )?;
            native.cuda_stream_synchronize(std::ptr::null_mut())?;
        }
        for (ring, slot) in rings.iter_mut().zip(received) {
            ring.release_recv_slot(slot.sequence)?;
        }
    }
    Ok(())
}

fn time_mapped_reduction_kernel(
    native: &NativeLibrary,
    buffers: &Ds41rtRouteShardReductionBuffers,
    args: &BenchRdmaRingArgs,
    peer_row_stride_bytes: usize,
    wire_dtype: u32,
) -> Result<f64> {
    let start = native.cuda_event_create()?;
    let end = native.cuda_event_create()?;
    let result = unsafe {
        native.cuda_event_record(start, std::ptr::null_mut())?;
        for _ in 0..args.kernel_iterations {
            native.cuda_reduce_route_shards_to_f32_async(
                buffers,
                args.rows,
                args.row_width,
                peer_row_stride_bytes,
                DS41RT_ROUTE_SHARD_LOCAL_F32,
                wire_dtype,
                3,
                std::ptr::null_mut(),
            )?;
        }
        native.cuda_event_record(end, std::ptr::null_mut())?;
        native.cuda_event_synchronize(end)?;
        let elapsed_ms = native.cuda_event_elapsed_ms(start, end)?;
        native.cuda_event_destroy(end)?;
        native.cuda_event_destroy(start)?;
        Ok::<_, anyhow::Error>(elapsed_ms as f64 * 1000.0 / args.kernel_iterations as f64)
    };
    result
}

fn wire_codec(codec: &str, row_width: usize) -> Result<(u32, usize)> {
    match codec {
        "bf16" => Ok((
            DS41RT_ROUTE_SHARD_WIRE_BF16,
            row_width
                .checked_mul(std::mem::size_of::<u16>())
                .context("BF16 row stride overflow")?,
        )),
        "fp8" => Ok((
            DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
            row_width
                .checked_add(std::mem::size_of::<f32>())
                .context("FP8 row stride overflow")?,
        )),
        "nvfp4" => {
            anyhow::ensure!(
                row_width > 0 && row_width % 16 == 0,
                "NVFP4 width must be a multiple of 16"
            );
            Ok((
                DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
                row_width / 2 + row_width / 16,
            ))
        }
        other => bail!("unsupported mapped RDMA wire codec {other}"),
    }
}

fn device_buffer_slice(
    buffer: Ds41rtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .context("device buffer slice overflow")?;
    anyhow::ensure!(
        end <= buffer.bytes,
        "device buffer slice exceeds its allocation"
    );
    Ok(Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_to_f32(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
}

fn run_timeline_server(args: &BenchRdmaRingArgs, transport: &TcpTransportConfig) -> Result<()> {
    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("binding mapped RDMA timeline server to {}", args.listen))?;
    let (stream, peer) = listener
        .accept()
        .context("accepting mapped RDMA timeline client")?;
    let mut ring = VerbsHostMappedRdmaRing::accept(stream, transport)?;
    let request_bytes = timeline_request_bytes(args);
    let response_bytes = timeline_response_bytes(args);
    let validation_iterations = args.depth;
    let measured_start = validation_iterations
        .checked_add(args.warmup_iterations)
        .context("timeline measured sequence overflow")?;
    let total = measured_start
        .checked_add(args.iterations)
        .context("timeline server iteration count overflow")?;
    let mut initialized_response_slots = vec![false; args.depth];
    let mut queue = VecDeque::with_capacity(args.depth);
    let mut measurements = TimelineMeasurements::default();
    let mut processed = 0_usize;
    let mut measured_started = None;
    let mut measured_cpu_started = None;

    while processed < total {
        if queue.is_empty() {
            let wait_started = Instant::now();
            let (slot, stats) = ring.wait_recv_slot_with_stats()?;
            let received_at = Instant::now();
            let queued = decode_timeline_request(
                slot,
                request_bytes,
                args.depth,
                received_at,
                processed < validation_iterations,
            )?;
            let measured = queued.header.sequence as usize >= measured_start;
            if measured {
                measurements.cq.record(stats);
                measurements
                    .receive_wait_us
                    .push(wait_started.elapsed().as_secs_f64() * 1e6);
            }
            record_validation(
                &mut measurements,
                measured || processed < validation_iterations,
                processed < validation_iterations,
            );
            queue.push_back(queued);
        }

        while queue.len() < args.depth {
            let poll_started = Instant::now();
            let (slot, stats) = ring.try_recv_slot_with_stats()?;
            let Some(slot) = slot else {
                if processed >= measured_start {
                    measurements.cq.record(stats);
                }
                break;
            };
            let received_at = Instant::now();
            let sequence = processed + queue.len();
            let queued = decode_timeline_request(
                slot,
                request_bytes,
                args.depth,
                received_at,
                sequence < validation_iterations,
            )?;
            let measured = queued.header.sequence as usize >= measured_start;
            if measured {
                measurements.cq.record(stats);
                measurements
                    .receive_wait_us
                    .push(poll_started.elapsed().as_secs_f64() * 1e6);
            }
            record_validation(
                &mut measurements,
                measured || sequence < validation_iterations,
                sequence < validation_iterations,
            );
            queue.push_back(queued);
        }

        if processed >= measured_start {
            measurements.max_queue_depth = measurements.max_queue_depth.max(queue.len());
        }
        let request = queue
            .pop_front()
            .context("timeline server queue unexpectedly empty")?;
        let sequence = request.header.sequence as usize;
        anyhow::ensure!(
            sequence == processed,
            "timeline server received sequence {sequence}, expected {processed}"
        );
        let measured = sequence >= measured_start;
        if sequence == measured_start {
            measured_started = Some(Instant::now());
            measured_cpu_started = thread_cpu_time();
        }
        if measured {
            measurements.request_validate_us.push(request.validate_us);
            measurements
                .queue_wait_us
                .push(request.received_at.elapsed().as_secs_f64() * 1e6);
        }

        let compute_started = Instant::now();
        if args.compute_delay_us > 0 {
            thread::sleep(Duration::from_micros(args.compute_delay_us));
        }
        if measured {
            measurements
                .compute_placeholder_us
                .push(compute_started.elapsed().as_secs_f64() * 1e6);
        }

        let reserve_started = Instant::now();
        let (send, reserve_stats) = ring.reserve_send_slot_with_stats()?;
        if measured {
            measurements.cq.record(reserve_stats);
            measurements
                .reserve_send_us
                .push(reserve_started.elapsed().as_secs_f64() * 1e6);
        }
        anyhow::ensure!(
            send.sequence == request.header.sequence,
            "timeline response send sequence {} did not match request {}",
            send.sequence,
            request.header.sequence
        );
        let response_seed = timeline_payload_seed(TIMELINE_RESPONSE_KIND, send.slot_index);
        let response_frame =
            unsafe { std::slice::from_raw_parts_mut(send.host_ptr, response_bytes) };
        if !initialized_response_slots[send.slot_index] {
            fill_timeline_payload(response_frame, response_seed);
            initialized_response_slots[send.slot_index] = true;
        }
        encode_timeline_header(
            response_frame,
            TimelineHeader {
                kind: TIMELINE_RESPONSE_KIND,
                sequence: send.sequence,
                generation: send.sequence / args.depth as u64,
                slot_index: send.slot_index as u32,
                frame_bytes: u32::try_from(response_bytes)
                    .context("timeline response bytes exceed u32")?,
                payload_seed: response_seed,
            },
        )?;
        let post_started = Instant::now();
        ring.post_reserved_send(response_bytes)?;
        if measured {
            measurements
                .post_send_us
                .push(post_started.elapsed().as_secs_f64() * 1e6);
        }
        ring.release_recv_slot(request.slot.sequence)?;
        processed += 1;
        if processed == validation_iterations || processed == measured_start {
            ring.flush_sends()?;
        }
    }

    let flush_stats = ring.flush_sends_with_stats()?;
    measurements.cq.record(flush_stats);
    let measured_started =
        measured_started.context("timeline server never entered measured phase")?;
    let elapsed = measured_started.elapsed();
    let thread_cpu_ms = elapsed_thread_cpu_ms(measured_cpu_started);
    print_timeline_report(
        args,
        "timeline-server",
        peer.to_string(),
        request_bytes,
        response_bytes,
        validation_iterations,
        elapsed,
        thread_cpu_ms,
        measurements,
    )
}

fn run_timeline_client(
    args: &BenchRdmaRingArgs,
    transport: &TcpTransportConfig,
    config: VerbsHostMappedRdmaRingConfig,
) -> Result<()> {
    let peer = args
        .peer
        .as_deref()
        .context("mapped RDMA timeline client requires --peer HOST:PORT")?;
    let mut ring = VerbsHostMappedRdmaRing::connect(peer, transport, config)?;
    let request_bytes = timeline_request_bytes(args);
    let response_bytes = timeline_response_bytes(args);
    let validation_iterations = args.depth;
    let mut initialized_request_slots = vec![false; args.depth];
    let mut measurements = TimelineMeasurements::default();
    run_timeline_client_phase(
        &mut ring,
        args,
        request_bytes,
        response_bytes,
        0,
        validation_iterations,
        true,
        false,
        &mut initialized_request_slots,
        &mut measurements,
    )?;
    run_timeline_client_phase(
        &mut ring,
        args,
        request_bytes,
        response_bytes,
        validation_iterations as u64,
        args.warmup_iterations,
        false,
        false,
        &mut initialized_request_slots,
        &mut measurements,
    )?;
    let measured_base = validation_iterations
        .checked_add(args.warmup_iterations)
        .context("timeline measured sequence overflow")? as u64;
    let measured_cpu_started = thread_cpu_time();
    let measured_started = Instant::now();
    run_timeline_client_phase(
        &mut ring,
        args,
        request_bytes,
        response_bytes,
        measured_base,
        args.iterations,
        false,
        true,
        &mut initialized_request_slots,
        &mut measurements,
    )?;
    let flush_stats = ring.flush_sends_with_stats()?;
    measurements.cq.record(flush_stats);
    let elapsed = measured_started.elapsed();
    let thread_cpu_ms = elapsed_thread_cpu_ms(measured_cpu_started);
    print_timeline_report(
        args,
        "timeline-client",
        peer.to_owned(),
        request_bytes,
        response_bytes,
        validation_iterations,
        elapsed,
        thread_cpu_ms,
        measurements,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_timeline_client_phase(
    ring: &mut VerbsHostMappedRdmaRing,
    args: &BenchRdmaRingArgs,
    request_bytes: usize,
    response_bytes: usize,
    sequence_base: u64,
    iterations: usize,
    full_validation: bool,
    measured: bool,
    initialized_request_slots: &mut [bool],
    measurements: &mut TimelineMeasurements,
) -> Result<()> {
    let mut sent = 0_usize;
    let mut received = 0_usize;
    let mut send_started = vec![None; iterations];
    while received < iterations {
        while sent < iterations && sent - received < args.window {
            let reserve_started = Instant::now();
            let (slot, stats) = ring.reserve_send_slot_with_stats()?;
            if measured {
                measurements.cq.record(stats);
                measurements
                    .reserve_send_us
                    .push(reserve_started.elapsed().as_secs_f64() * 1e6);
            }
            let sequence = sequence_base.wrapping_add(sent as u64);
            anyhow::ensure!(
                slot.sequence == sequence,
                "timeline request send sequence {} did not match {sequence}",
                slot.sequence
            );
            let seed = timeline_payload_seed(TIMELINE_REQUEST_KIND, slot.slot_index);
            let frame = unsafe { std::slice::from_raw_parts_mut(slot.host_ptr, request_bytes) };
            if !initialized_request_slots[slot.slot_index] {
                fill_timeline_payload(frame, seed);
                initialized_request_slots[slot.slot_index] = true;
            }
            encode_timeline_header(
                frame,
                TimelineHeader {
                    kind: TIMELINE_REQUEST_KIND,
                    sequence,
                    generation: sequence / args.depth as u64,
                    slot_index: slot.slot_index as u32,
                    frame_bytes: u32::try_from(request_bytes)
                        .context("timeline request bytes exceed u32")?,
                    payload_seed: seed,
                },
            )?;
            send_started[sent] = Some(Instant::now());
            let post_started = Instant::now();
            ring.post_reserved_send(request_bytes)?;
            if measured {
                measurements
                    .post_send_us
                    .push(post_started.elapsed().as_secs_f64() * 1e6);
            }
            sent += 1;
            measurements.max_outstanding = measurements.max_outstanding.max(sent - received);
        }
        if measured && sent < iterations && sent - received == args.window {
            measurements.credit_exhaustions += 1;
        }

        let wait_started = Instant::now();
        let (slot, stats) = ring.wait_recv_slot_with_stats()?;
        if measured {
            measurements.cq.record(stats);
            measurements
                .receive_wait_us
                .push(wait_started.elapsed().as_secs_f64() * 1e6);
        }
        let expected_sequence = sequence_base.wrapping_add(received as u64);
        let validate_started = Instant::now();
        validate_timeline_slot(
            slot,
            response_bytes,
            TIMELINE_RESPONSE_KIND,
            expected_sequence,
            args.depth,
            full_validation,
        )?;
        if measured {
            measurements
                .response_validate_us
                .push(validate_started.elapsed().as_secs_f64() * 1e6);
            measurements.roundtrip_us.push(
                send_started[received]
                    .context("timeline request has no post timestamp")?
                    .elapsed()
                    .as_secs_f64()
                    * 1e6,
            );
        }
        record_validation(measurements, measured || full_validation, full_validation);
        ring.release_recv_slot(slot.sequence)?;
        received += 1;
    }
    let stats = ring.flush_sends_with_stats()?;
    if measured {
        measurements.cq.record(stats);
    }
    Ok(())
}

fn decode_timeline_request(
    slot: VerbsHostMappedRdmaSlot,
    frame_bytes: usize,
    depth: usize,
    received_at: Instant,
    full_validation: bool,
) -> Result<QueuedTimelineRequest> {
    let validate_started = Instant::now();
    let frame = unsafe { std::slice::from_raw_parts(slot.host_ptr, frame_bytes) };
    let header = decode_timeline_header(frame)?;
    validate_timeline_header(
        header,
        TIMELINE_REQUEST_KIND,
        header.sequence,
        frame_bytes,
        depth,
    )?;
    anyhow::ensure!(
        slot.slot_index == header.slot_index as usize,
        "timeline request receive slot {} did not match header slot {}",
        slot.slot_index,
        header.slot_index
    );
    validate_timeline_payload(frame, header.payload_seed, full_validation)?;
    Ok(QueuedTimelineRequest {
        slot,
        header,
        received_at,
        validate_us: validate_started.elapsed().as_secs_f64() * 1e6,
    })
}

fn validate_timeline_slot(
    slot: VerbsHostMappedRdmaSlot,
    frame_bytes: usize,
    kind: u16,
    expected_sequence: u64,
    depth: usize,
    full_validation: bool,
) -> Result<()> {
    let frame = unsafe { std::slice::from_raw_parts(slot.host_ptr, frame_bytes) };
    let header = decode_timeline_header(frame)?;
    validate_timeline_header(header, kind, expected_sequence, frame_bytes, depth)?;
    anyhow::ensure!(
        slot.slot_index == header.slot_index as usize,
        "timeline receive slot {} did not match header slot {}",
        slot.slot_index,
        header.slot_index
    );
    validate_timeline_payload(frame, header.payload_seed, full_validation)
}

fn validate_timeline_header(
    header: TimelineHeader,
    kind: u16,
    expected_sequence: u64,
    frame_bytes: usize,
    depth: usize,
) -> Result<()> {
    anyhow::ensure!(header.kind == kind, "timeline frame kind mismatch");
    anyhow::ensure!(
        header.sequence == expected_sequence,
        "timeline frame sequence {} did not match {expected_sequence}",
        header.sequence
    );
    anyhow::ensure!(
        header.generation == expected_sequence / depth as u64,
        "timeline frame generation {} did not match sequence {expected_sequence}",
        header.generation
    );
    anyhow::ensure!(
        header.slot_index as usize == expected_sequence as usize % depth,
        "timeline frame slot {} did not match sequence {expected_sequence}",
        header.slot_index
    );
    anyhow::ensure!(
        header.frame_bytes as usize == frame_bytes,
        "timeline frame bytes {} did not match {frame_bytes}",
        header.frame_bytes
    );
    anyhow::ensure!(
        header.payload_seed == timeline_payload_seed(kind, header.slot_index as usize),
        "timeline frame payload seed mismatch"
    );
    Ok(())
}

fn encode_timeline_header(frame: &mut [u8], header: TimelineHeader) -> Result<()> {
    anyhow::ensure!(
        frame.len() >= TIMELINE_HEADER_BYTES,
        "timeline frame is smaller than its header"
    );
    frame[..TIMELINE_HEADER_BYTES].fill(0);
    frame[0..8].copy_from_slice(TIMELINE_MAGIC);
    frame[8..10].copy_from_slice(&TIMELINE_VERSION.to_le_bytes());
    frame[10..12].copy_from_slice(&header.kind.to_le_bytes());
    frame[12..16].copy_from_slice(&(TIMELINE_HEADER_BYTES as u32).to_le_bytes());
    frame[16..24].copy_from_slice(&header.sequence.to_le_bytes());
    frame[24..32].copy_from_slice(&header.generation.to_le_bytes());
    frame[32..36].copy_from_slice(&header.slot_index.to_le_bytes());
    frame[36..40].copy_from_slice(&header.frame_bytes.to_le_bytes());
    frame[40..44].copy_from_slice(&header.payload_seed.to_le_bytes());
    Ok(())
}

fn decode_timeline_header(frame: &[u8]) -> Result<TimelineHeader> {
    anyhow::ensure!(
        frame.len() >= TIMELINE_HEADER_BYTES,
        "timeline frame is smaller than its header"
    );
    anyhow::ensure!(
        &frame[0..8] == TIMELINE_MAGIC,
        "timeline frame magic mismatch"
    );
    anyhow::ensure!(
        u16::from_le_bytes(frame[8..10].try_into()?) == TIMELINE_VERSION,
        "timeline frame version mismatch"
    );
    anyhow::ensure!(
        u32::from_le_bytes(frame[12..16].try_into()?) as usize == TIMELINE_HEADER_BYTES,
        "timeline frame header size mismatch"
    );
    Ok(TimelineHeader {
        kind: u16::from_le_bytes(frame[10..12].try_into()?),
        sequence: u64::from_le_bytes(frame[16..24].try_into()?),
        generation: u64::from_le_bytes(frame[24..32].try_into()?),
        slot_index: u32::from_le_bytes(frame[32..36].try_into()?),
        frame_bytes: u32::from_le_bytes(frame[36..40].try_into()?),
        payload_seed: u32::from_le_bytes(frame[40..44].try_into()?),
    })
}

fn fill_timeline_payload(frame: &mut [u8], seed: u32) {
    for (index, byte) in frame[TIMELINE_HEADER_BYTES..].iter_mut().enumerate() {
        *byte = timeline_payload_byte(seed, index);
    }
}

fn validate_timeline_payload(frame: &[u8], seed: u32, full: bool) -> Result<()> {
    let payload = &frame[TIMELINE_HEADER_BYTES..];
    if full {
        for (index, actual) in payload.iter().copied().enumerate() {
            anyhow::ensure!(
                actual == timeline_payload_byte(seed, index),
                "timeline payload mismatch at byte {index}"
            );
        }
        return Ok(());
    }
    if payload.is_empty() {
        return Ok(());
    }
    let sample_count = payload.len().min(32);
    for sample in 0..sample_count {
        let index = sample * (payload.len() - 1) / sample_count.max(1);
        anyhow::ensure!(
            payload[index] == timeline_payload_byte(seed, index),
            "timeline sampled payload mismatch at byte {index}"
        );
    }
    Ok(())
}

fn timeline_payload_seed(kind: u16, slot_index: usize) -> u32 {
    0x9e37_79b9_u32
        ^ (kind as u32).wrapping_mul(0x85eb_ca6b)
        ^ (slot_index as u32).wrapping_mul(0xc2b2_ae35)
}

fn timeline_payload_byte(seed: u32, index: usize) -> u8 {
    let mut value = seed ^ (index as u32).wrapping_mul(0x45d9_f3b);
    value ^= value >> 16;
    value = value.wrapping_mul(0x45d9_f3b);
    value ^= value >> 16;
    value as u8
}

fn timeline_request_bytes(args: &BenchRdmaRingArgs) -> usize {
    args.request_bytes.unwrap_or(args.slot_bytes)
}

fn timeline_response_bytes(args: &BenchRdmaRingArgs) -> usize {
    args.response_bytes.unwrap_or(args.slot_bytes)
}

fn record_validation(measurements: &mut TimelineMeasurements, record: bool, full: bool) {
    if !record {
        return;
    }
    measurements.generation_checks += 1;
    if full {
        measurements.full_payloads_validated += 1;
    } else {
        measurements.sampled_payloads_validated += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn print_timeline_report(
    args: &BenchRdmaRingArgs,
    role: &str,
    address: String,
    request_bytes: usize,
    response_bytes: usize,
    validation_iterations: usize,
    elapsed: Duration,
    thread_cpu_ms: Option<f64>,
    measurements: TimelineMeasurements,
) -> Result<()> {
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_ms = elapsed_seconds * 1e3;
    let report = MappedRdmaQueueTimelineReport {
        benchmark: "mapped-rdma-queue-timeline".to_owned(),
        role: role.to_owned(),
        address,
        network_label: args.network_label.clone(),
        pre_firmware: args.network_label == "pre-firmware",
        request_bytes,
        response_bytes,
        slot_bytes: args.slot_bytes,
        depth: args.depth,
        window: args.window,
        compute_delay_us: args.compute_delay_us,
        validation_iterations,
        warmup_iterations: args.warmup_iterations,
        iterations: args.iterations,
        elapsed_ms,
        roundtrips_per_second: args.iterations as f64 / elapsed_seconds,
        bidirectional_payload_gbps: args.iterations as f64
            * (request_bytes + response_bytes) as f64
            * 8.0
            / elapsed_seconds
            / 1e9,
        thread_cpu_ms,
        thread_cpu_fraction: thread_cpu_ms.map(|cpu_ms| cpu_ms / elapsed_ms),
        wraparound_generations: args.iterations.div_ceil(args.depth),
        full_payloads_validated: measurements.full_payloads_validated,
        sampled_payloads_validated: measurements.sampled_payloads_validated,
        generation_checks: measurements.generation_checks,
        credit_exhaustions: measurements.credit_exhaustions,
        max_outstanding: measurements.max_outstanding,
        max_queue_depth: measurements.max_queue_depth,
        cq: measurements.cq,
        reserve_send: summarize_us(measurements.reserve_send_us),
        post_send: summarize_us(measurements.post_send_us),
        receive_wait: summarize_us(measurements.receive_wait_us),
        request_validate: summarize_us(measurements.request_validate_us),
        queue_wait: summarize_us(measurements.queue_wait_us),
        compute_placeholder: summarize_us(measurements.compute_placeholder_us),
        response_validate: summarize_us(measurements.response_validate_us),
        roundtrip: summarize_us(measurements.roundtrip_us),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn summarize_us(mut values: Vec<f64>) -> Option<TimelineSummary> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((values.len() - 1) as f64 * fraction).round() as usize;
        values[index]
    };
    Some(TimelineSummary {
        samples: values.len(),
        mean_us: values.iter().sum::<f64>() / values.len() as f64,
        p50_us: percentile(0.50),
        p95_us: percentile(0.95),
        p99_us: percentile(0.99),
        max_us: *values.last().expect("non-empty timeline samples"),
    })
}

#[cfg(target_os = "linux")]
fn thread_cpu_time() -> Option<Duration> {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock_id: i32, timespec: *mut Timespec) -> i32;
    }
    const CLOCK_THREAD_CPUTIME_ID: i32 = 3;
    let mut value = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut value) } != 0
        || value.tv_sec < 0
        || value.tv_nsec < 0
    {
        return None;
    }
    Some(Duration::new(value.tv_sec as u64, value.tv_nsec as u32))
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_time() -> Option<Duration> {
    None
}

fn elapsed_thread_cpu_ms(started: Option<Duration>) -> Option<f64> {
    let started = started?;
    Some(thread_cpu_time()?.checked_sub(started)?.as_secs_f64() * 1e3)
}

fn run_server(args: &BenchRdmaRingArgs, transport: &TcpTransportConfig) -> Result<()> {
    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("binding mapped RDMA ring benchmark to {}", args.listen))?;
    let (stream, peer) = listener
        .accept()
        .context("accepting mapped RDMA ring benchmark client")?;
    let mut ring = VerbsHostMappedRdmaRing::accept(stream, transport)?;
    let native = args
        .gpu_echo
        .then(|| load_native_library(args.native_lib.as_deref()))
        .transpose()?;
    let total = args
        .warmup_iterations
        .checked_add(args.iterations)
        .context("mapped RDMA ring benchmark iteration count overflow")?;
    let started = Instant::now();
    for _ in 0..total {
        let received = ring.wait_recv_slot()?;
        let send = ring.reserve_send_slot()?;
        if let Some(native) = &native {
            unsafe {
                native.copy_d2d_async(
                    send.device_buffer,
                    received.device_buffer,
                    args.slot_bytes,
                    std::ptr::null_mut(),
                )?;
                native.cuda_stream_synchronize(std::ptr::null_mut())?;
            }
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(received.host_ptr, send.host_ptr, args.slot_bytes);
            }
        }
        ring.release_recv_slot(received.sequence)?;
        ring.post_reserved_send(args.slot_bytes)?;
    }
    ring.flush_sends()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&MappedRdmaRingReport {
            role: "server".to_owned(),
            address: peer.to_string(),
            slot_bytes: args.slot_bytes,
            depth: args.depth,
            window: args.window,
            gpu_echo: args.gpu_echo,
            warmup_iterations: args.warmup_iterations,
            iterations: args.iterations,
            elapsed_ms,
            roundtrips_per_second: 0.0,
            bidirectional_payload_gbps: 0.0,
            amortized_roundtrip_us: None,
            estimated_one_way_latency_us: None,
            p50_roundtrip_us: None,
            p95_roundtrip_us: None,
            p99_roundtrip_us: None,
        })?
    );
    Ok(())
}

fn run_client(
    args: &BenchRdmaRingArgs,
    transport: &TcpTransportConfig,
    config: VerbsHostMappedRdmaRingConfig,
) -> Result<()> {
    let peer = args
        .peer
        .as_deref()
        .context("mapped RDMA ring client requires --peer HOST:PORT")?;
    let mut ring = VerbsHostMappedRdmaRing::connect(peer, transport, config)?;
    run_client_phase(
        &mut ring,
        args.slot_bytes,
        args.window,
        0,
        args.warmup_iterations,
        false,
    )?;
    let marker_base = args.warmup_iterations as u64;
    let started = Instant::now();
    let latencies = run_client_phase(
        &mut ring,
        args.slot_bytes,
        args.window,
        marker_base,
        args.iterations,
        args.window == 1,
    )?;
    let elapsed = started.elapsed();
    ring.flush_sends()?;
    let elapsed_seconds = elapsed.as_secs_f64();
    let elapsed_ms = elapsed_seconds * 1000.0;
    let roundtrips_per_second = args.iterations as f64 / elapsed_seconds;
    let bidirectional_payload_gbps =
        args.iterations as f64 * args.slot_bytes as f64 * 2.0 * 8.0 / elapsed_seconds / 1e9;
    let amortized_roundtrip_us = elapsed_seconds * 1e6 / args.iterations as f64;
    let (p50, p95, p99) = latency_percentiles(latencies);
    println!(
        "{}",
        serde_json::to_string_pretty(&MappedRdmaRingReport {
            role: "client".to_owned(),
            address: peer.to_owned(),
            slot_bytes: args.slot_bytes,
            depth: args.depth,
            window: args.window,
            gpu_echo: args.gpu_echo,
            warmup_iterations: args.warmup_iterations,
            iterations: args.iterations,
            elapsed_ms,
            roundtrips_per_second,
            bidirectional_payload_gbps,
            amortized_roundtrip_us: Some(amortized_roundtrip_us),
            estimated_one_way_latency_us: (args.window == 1)
                .then_some(amortized_roundtrip_us * 0.5),
            p50_roundtrip_us: p50,
            p95_roundtrip_us: p95,
            p99_roundtrip_us: p99,
        })?
    );
    Ok(())
}

fn run_client_phase(
    ring: &mut VerbsHostMappedRdmaRing,
    slot_bytes: usize,
    window: usize,
    marker_base: u64,
    iterations: usize,
    record_latency: bool,
) -> Result<Vec<f64>> {
    let mut sent = 0_usize;
    let mut received = 0_usize;
    let mut latencies = Vec::with_capacity(record_latency.then_some(iterations).unwrap_or(0));
    let mut send_started = Vec::with_capacity(record_latency.then_some(iterations).unwrap_or(0));
    while received < iterations {
        while sent < iterations && sent - received < window {
            let marker = marker_base.wrapping_add(sent as u64);
            let slot = ring.reserve_send_slot()?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    marker.to_le_bytes().as_ptr(),
                    slot.host_ptr,
                    std::mem::size_of::<u64>(),
                );
            }
            if record_latency {
                send_started.push(Instant::now());
            }
            ring.post_reserved_send(slot_bytes)?;
            sent += 1;
        }
        let slot = ring.wait_recv_slot()?;
        let mut marker_bytes = [0_u8; std::mem::size_of::<u64>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                slot.host_ptr,
                marker_bytes.as_mut_ptr(),
                marker_bytes.len(),
            );
        }
        let actual = u64::from_le_bytes(marker_bytes);
        let expected = marker_base.wrapping_add(received as u64);
        anyhow::ensure!(
            actual == expected,
            "mapped RDMA ring response marker {actual} did not match {expected}"
        );
        if record_latency {
            latencies.push(send_started[received].elapsed().as_secs_f64() * 1e6);
        }
        ring.release_recv_slot(slot.sequence)?;
        received += 1;
    }
    Ok(latencies)
}

fn latency_percentiles(mut values: Vec<f64>) -> (Option<f64>, Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None, None);
    }
    values.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = ((values.len() - 1) as f64 * fraction).round() as usize;
        values[index]
    };
    (
        Some(percentile(0.50)),
        Some(percentile(0.95)),
        Some(percentile(0.99)),
    )
}

fn load_native_library(path: Option<&Path>) -> Result<NativeLibrary> {
    let path = path
        .map(Path::to_path_buf)
        .or_else(|| env::var_os("DS41RT_NATIVE_LIB").map(PathBuf::from))
        .context("GPU echo requires --native-lib or DS41RT_NATIVE_LIB")?;
    unsafe { NativeLibrary::load(&path) }
        .with_context(|| format!("loading native library {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_frame_roundtrips_across_ring_generations() -> Result<()> {
        let depth = 8;
        let sequence = 37_u64;
        let slot_index = sequence as usize % depth;
        let seed = timeline_payload_seed(TIMELINE_REQUEST_KIND, slot_index);
        let mut frame = vec![0_u8; 4096];
        fill_timeline_payload(&mut frame, seed);
        let expected = TimelineHeader {
            kind: TIMELINE_REQUEST_KIND,
            sequence,
            generation: sequence / depth as u64,
            slot_index: slot_index as u32,
            frame_bytes: frame.len() as u32,
            payload_seed: seed,
        };
        encode_timeline_header(&mut frame, expected)?;

        let actual = decode_timeline_header(&frame)?;
        assert_eq!(actual, expected);
        validate_timeline_header(actual, TIMELINE_REQUEST_KIND, sequence, frame.len(), depth)?;
        validate_timeline_payload(&frame, seed, true)
    }

    #[test]
    fn timeline_frame_rejects_stale_generation_and_corruption() -> Result<()> {
        let depth = 4;
        let sequence = 9_u64;
        let slot_index = sequence as usize % depth;
        let seed = timeline_payload_seed(TIMELINE_RESPONSE_KIND, slot_index);
        let mut frame = vec![0_u8; 256];
        let frame_bytes = frame.len();
        fill_timeline_payload(&mut frame, seed);
        encode_timeline_header(
            &mut frame,
            TimelineHeader {
                kind: TIMELINE_RESPONSE_KIND,
                sequence,
                generation: 1,
                slot_index: slot_index as u32,
                frame_bytes: frame_bytes as u32,
                payload_seed: seed,
            },
        )?;
        let stale = decode_timeline_header(&frame)?;
        let error =
            validate_timeline_header(stale, TIMELINE_RESPONSE_KIND, sequence, frame.len(), depth)
                .expect_err("stale generation must be rejected");
        assert!(error.to_string().contains("generation"));

        frame[75] ^= 0xff;
        let error = validate_timeline_payload(&frame, seed, true)
            .expect_err("payload corruption must be rejected");
        assert!(error.to_string().contains("payload mismatch"));
        Ok(())
    }

    #[test]
    fn timeline_summary_uses_order_statistics() {
        let summary = summarize_us(vec![100.0, 1.0, 10.0, 1000.0]).expect("summary");
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.p50_us, 100.0);
        assert_eq!(summary.p95_us, 1000.0);
        assert_eq!(summary.max_us, 1000.0);
    }
}
