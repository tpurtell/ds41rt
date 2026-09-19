use anyhow::{bail, Context, Result};
#[cfg(test)]
use ds41rt_core::DS4_FLASH_HIDDEN_SIZE;
use ds41rt_transport::{
    expert_protocol_v2_compact_id, ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request,
    ExpertProtocolV2Response, ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseView,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertProtocolV2Status,
    ExpertV2Dtype, ExpertV2SourceKind, TcpTransportConfig, VerbsHostProtocolV2PersistentClient,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use serde::Serialize;
use std::net::{SocketAddr, ToSocketAddrs};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

use crate::cli::BenchProtocolV2TcpArgs;

const PROTOCOL_LABEL: &str = "expert_protocol_v2_binary";
const PRECOMPILE_REQUEST_FLAG_LABEL: &str = "precompile_warmup";

pub(crate) async fn run_bench_protocol_v2_tcp(args: BenchProtocolV2TcpArgs) -> Result<()> {
    let rows = benchmark_protocol_v2_tcp_rows(args).await?;
    for row in rows {
        println!("{}", serde_json::to_string(&row)?);
    }
    Ok(())
}

async fn benchmark_protocol_v2_tcp_rows(args: BenchProtocolV2TcpArgs) -> Result<Vec<BenchmarkRow>> {
    let config = BenchConfig::from_args(args)?;
    let mut rows = Vec::new();
    let mut request_id = config.request_id_start;
    let mut client = ProtocolV2BenchClient::connect(&config).await?;
    run_protocol_v2_warmup(&mut client, &config, &mut request_id).await?;
    if config.warmup_only {
        return Ok(rows);
    }

    for row_count in config.roundtrip_rows.clone() {
        let source_kind = source_kind_for_row_count(row_count);
        let iterations = iterations_for_row_count(row_count, &config);
        let measurement = measure_repeated_roundtrips(
            &mut client,
            &config,
            &mut request_id,
            row_count,
            source_kind,
            iterations,
        )
        .await?;
        rows.push(BenchmarkRow::roundtrip(
            &config,
            row_count,
            source_kind,
            measurement,
        ));
    }

    if config.roundtrip_only {
        return Ok(rows);
    }

    let measurement = measure_chain(
        &mut client,
        &config,
        &mut request_id,
        1,
        ExpertV2SourceKind::Decode,
        "tcp_expert_75_hop_chain",
    )
    .await?;
    rows.push(BenchmarkRow::chain(
        &config,
        "tcp_expert_75_hop_chain",
        1,
        ExpertV2SourceKind::Decode,
        measurement,
    ));

    for row_count in config.mtp_chain_rows.clone() {
        let measurement = measure_chain(
            &mut client,
            &config,
            &mut request_id,
            row_count,
            ExpertV2SourceKind::MtpVerify,
            "tcp_expert_75_layer_mtp_chain",
        )
        .await?;
        rows.push(BenchmarkRow::chain(
            &config,
            "tcp_expert_75_layer_mtp_chain",
            row_count,
            ExpertV2SourceKind::MtpVerify,
            measurement,
        ));
    }

    for row_count in config.prefill_roundtrip_rows.clone() {
        let iterations = iterations_for_row_count(row_count, &config);
        let measurement = measure_repeated_roundtrips(
            &mut client,
            &config,
            &mut request_id,
            row_count,
            ExpertV2SourceKind::Prefill,
            iterations,
        )
        .await?;
        rows.push(BenchmarkRow::prefill_roundtrip(
            &config,
            row_count,
            measurement,
        ));
    }

    for row_count in config.prefill_chain_rows.clone() {
        let measurement = measure_chain(
            &mut client,
            &config,
            &mut request_id,
            row_count,
            ExpertV2SourceKind::Prefill,
            "tcp_expert_75_layer_prefill_chain",
        )
        .await?;
        rows.push(BenchmarkRow::prefill_chain(&config, row_count, measurement));
    }

    Ok(rows)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchTransport {
    Tcp,
    VerbsHost,
}

impl BenchTransport {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "tcp" => Ok(Self::Tcp),
            "verbs-host" => Ok(Self::VerbsHost),
            other => bail!("--transport must be tcp or verbs-host, got {other}"),
        }
    }
}

#[derive(Debug, Clone)]
struct BenchConfig {
    addr: String,
    addr_socket: SocketAddr,
    transport: BenchTransport,
    addr_label: String,
    target: String,
    request_id_start: u64,
    hops: usize,
    iterations: usize,
    large_iterations: usize,
    warmup_iterations: usize,
    warmup_rows: usize,
    warmup_timeout_ms: u64,
    warmup_timeout: Duration,
    warmup_only: bool,
    roundtrip_only: bool,
    roundtrip_rows: Vec<usize>,
    mtp_chain_rows: Vec<usize>,
    prefill_roundtrip_rows: Vec<usize>,
    prefill_chain_rows: Vec<usize>,
    layer_id: u32,
    expert_id: u32,
    hidden_dim: usize,
    expert_ids: Vec<u32>,
    routes_per_row: usize,
    spark_reduction: bool,
    nvfp4_fp8_roundtrip: bool,
    expected_executor: Option<String>,
    expected_executor_id: Option<u64>,
    require_expected_executor: bool,
    timeout_ms: u64,
    timeout: Duration,
    max_frame_bytes: usize,
}

impl BenchConfig {
    fn from_args(args: BenchProtocolV2TcpArgs) -> Result<Self> {
        if args.hops == 0 {
            bail!("--hops must be non-zero");
        }
        if args.request_id_start == u64::MAX {
            bail!("--request-id-start must leave room for at least one request");
        }
        if args.iterations == 0 {
            bail!("--iterations must be non-zero");
        }
        if args.large_iterations == 0 {
            bail!("--large-iterations must be non-zero");
        }
        if args.warmup_iterations > 0 && args.warmup_rows == 0 {
            bail!("--warmup-rows must be non-zero when --warmup-iterations is set");
        }
        if args.warmup_only && args.warmup_iterations == 0 {
            bail!("--warmup-only requires --warmup-iterations to be non-zero");
        }
        if args.max_frame_bytes < EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN {
            bail!("--max-frame-bytes is smaller than the ProtocolV2 response header");
        }
        if args.hidden_dim == 0 {
            bail!("--hidden-dim must be non-zero");
        }
        if args.routes_per_row == 0 {
            bail!("--routes-per-row must be non-zero");
        }
        let expert_ids = parse_expert_ids(
            args.expert_ids.as_deref(),
            args.expert_id,
            args.routes_per_row,
        )?;
        let expected_executor_id = args
            .expected_executor
            .as_deref()
            .map(expert_protocol_v2_compact_id);
        let warmup_timeout_ms = args.warmup_timeout_ms.unwrap_or(args.timeout_ms);
        let addr_socket = resolve_addr(&args.addr)?;
        Ok(Self {
            addr: args.addr.clone(),
            addr_socket,
            transport: BenchTransport::parse(&args.transport)?,
            addr_label: args.addr,
            target: args.target,
            request_id_start: args.request_id_start,
            hops: args.hops,
            iterations: args.iterations,
            large_iterations: args.large_iterations,
            warmup_iterations: args.warmup_iterations,
            warmup_rows: args.warmup_rows,
            warmup_timeout_ms,
            warmup_timeout: Duration::from_millis(warmup_timeout_ms),
            warmup_only: args.warmup_only,
            roundtrip_only: args.roundtrip_only,
            roundtrip_rows: parse_row_counts(&args.roundtrip_rows, "--roundtrip-rows")?,
            mtp_chain_rows: parse_row_counts(&args.mtp_chain_rows, "--mtp-chain-rows")?,
            prefill_roundtrip_rows: parse_row_counts(
                &args.prefill_roundtrip_rows,
                "--prefill-roundtrip-rows",
            )?,
            prefill_chain_rows: parse_row_counts(&args.prefill_chain_rows, "--prefill-chain-rows")?,
            layer_id: args.layer_id,
            expert_id: args.expert_id,
            hidden_dim: args.hidden_dim,
            expert_ids,
            routes_per_row: args.routes_per_row,
            spark_reduction: args.spark_reduction,
            nvfp4_fp8_roundtrip: args.nvfp4_fp8_roundtrip,
            expected_executor: args.expected_executor,
            expected_executor_id,
            require_expected_executor: args.require_expected_executor,
            timeout_ms: args.timeout_ms,
            timeout: Duration::from_millis(args.timeout_ms),
            max_frame_bytes: args.max_frame_bytes,
        })
    }

    fn precompile_shape_count(&self) -> Option<usize> {
        (self.warmup_iterations > 0).then(|| protocol_v2_warmup_shapes(self).len())
    }

    fn precompile_protocol_invocations(&self) -> Option<usize> {
        self.precompile_shape_count()
            .map(|shape_count| shape_count * self.warmup_iterations)
    }
}

fn resolve_addr(addr: &str) -> Result<SocketAddr> {
    addr.to_socket_addrs()
        .with_context(|| format!("resolving ProtocolV2 benchmark address {addr}"))?
        .next()
        .with_context(|| format!("ProtocolV2 benchmark address {addr} resolved to no addresses"))
}

fn parse_row_counts(value: &str, label: &str) -> Result<Vec<usize>> {
    let mut rows = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let row_count = part
            .parse::<usize>()
            .with_context(|| format!("parsing {label} entry {part:?}"))?;
        if row_count == 0 {
            bail!("{label} entries must be non-zero");
        }
        rows.push(row_count);
    }
    if rows.is_empty() {
        bail!("{label} must include at least one row count");
    }
    Ok(rows)
}

fn parse_expert_ids(
    value: Option<&str>,
    default_expert_id: u32,
    routes_per_row: usize,
) -> Result<Vec<u32>> {
    let Some(value) = value else {
        return Ok(vec![default_expert_id; routes_per_row]);
    };
    let ids = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .with_context(|| format!("parsing --expert-ids entry {part:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if ids.len() < routes_per_row {
        bail!(
            "--expert-ids must contain at least --routes-per-row={} entries, got {}",
            routes_per_row,
            ids.len()
        );
    }
    Ok(ids)
}

#[derive(Default)]
struct ProtocolV2Buffers {
    request: ExpertProtocolV2FrameBuffer,
    response: Vec<u8>,
}

async fn run_protocol_v2_warmup(
    client: &mut ProtocolV2BenchClient,
    config: &BenchConfig,
    request_id: &mut u64,
) -> Result<()> {
    if config.warmup_iterations == 0 {
        return Ok(());
    }
    for shape in protocol_v2_warmup_shapes(config) {
        let mut request = protocol_v2_request_for_row_count(
            *request_id,
            shape.row_count,
            shape.source_kind,
            config.hidden_dim,
            config.layer_id,
            &config.expert_ids,
            config.routes_per_row,
            config.spark_reduction,
            config.nvfp4_fp8_roundtrip,
        )?
        .with_precompile_warmup();
        for _ in 0..config.warmup_iterations {
            *request_id += 1;
            request.header.request_id = *request_id;
            for row in &mut request.rows {
                row.source_request_id = *request_id;
            }
            client
                .roundtrip(config, &request, config.warmup_timeout)
                .await?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WarmupShape {
    row_count: usize,
    source_kind: ExpertV2SourceKind,
}

fn protocol_v2_warmup_shapes(config: &BenchConfig) -> Vec<WarmupShape> {
    let mut shapes = Vec::new();
    push_unique_warmup_shape(
        &mut shapes,
        config.warmup_rows,
        source_kind_for_row_count(config.warmup_rows),
    );
    for row_count in &config.roundtrip_rows {
        push_unique_warmup_shape(
            &mut shapes,
            *row_count,
            source_kind_for_row_count(*row_count),
        );
    }
    push_unique_warmup_shape(&mut shapes, 1, ExpertV2SourceKind::Decode);
    for row_count in &config.mtp_chain_rows {
        push_unique_warmup_shape(&mut shapes, *row_count, ExpertV2SourceKind::MtpVerify);
    }
    for row_count in &config.prefill_roundtrip_rows {
        push_unique_warmup_shape(&mut shapes, *row_count, ExpertV2SourceKind::Prefill);
    }
    for row_count in &config.prefill_chain_rows {
        push_unique_warmup_shape(&mut shapes, *row_count, ExpertV2SourceKind::Prefill);
    }
    shapes
}

fn push_unique_warmup_shape(
    shapes: &mut Vec<WarmupShape>,
    row_count: usize,
    source_kind: ExpertV2SourceKind,
) {
    if !shapes
        .iter()
        .any(|shape| shape.row_count == row_count && shape.source_kind == source_kind)
    {
        shapes.push(WarmupShape {
            row_count,
            source_kind,
        });
    }
}

enum ProtocolV2BenchClient {
    Tcp {
        stream: TcpStream,
        buffers: ProtocolV2Buffers,
    },
    VerbsHost {
        client: VerbsHostProtocolV2PersistentClient,
    },
}

impl ProtocolV2BenchClient {
    async fn connect(config: &BenchConfig) -> Result<Self> {
        match config.transport {
            BenchTransport::Tcp => Ok(Self::Tcp {
                stream: connect_tcp(config).await?,
                buffers: ProtocolV2Buffers::default(),
            }),
            BenchTransport::VerbsHost => Ok(Self::VerbsHost {
                client: VerbsHostProtocolV2PersistentClient::new(
                    config.addr_socket,
                    TcpTransportConfig { timing: false,
                        timeout: config.timeout.max(config.warmup_timeout),
                        max_frame_bytes: config.max_frame_bytes,
                    },
                )?,
            }),
        }
    }

    async fn roundtrip(
        &mut self,
        config: &BenchConfig,
        request: &ExpertProtocolV2Request,
        request_timeout: Duration,
    ) -> Result<RoundtripMeta> {
        match self {
            Self::Tcp { stream, buffers } => {
                roundtrip_once_tcp(stream, buffers, config, request, request_timeout).await
            }
            Self::VerbsHost { client } => roundtrip_once_verbs_host(client, config, request).await,
        }
    }
}

async fn connect_tcp(config: &BenchConfig) -> Result<TcpStream> {
    let stream = timeout(config.timeout, TcpStream::connect(config.addr.as_str()))
        .await
        .context("timed out connecting TCP ProtocolV2 benchmark target")?
        .with_context(|| format!("connecting TCP ProtocolV2 benchmark target {}", config.addr))?;
    stream
        .set_nodelay(true)
        .context("setting TCP_NODELAY for TCP ProtocolV2 benchmark client")?;
    Ok(stream)
}

async fn measure_repeated_roundtrips(
    client: &mut ProtocolV2BenchClient,
    config: &BenchConfig,
    request_id: &mut u64,
    row_count: usize,
    source_kind: ExpertV2SourceKind,
    iterations: usize,
) -> Result<Measurement> {
    let mut request = protocol_v2_request_for_row_count(
        *request_id,
        row_count,
        source_kind,
        config.hidden_dim,
        config.layer_id,
        &config.expert_ids,
        config.routes_per_row,
        config.spark_reduction,
        config.nvfp4_fp8_roundtrip,
    )?;
    let mut samples = Vec::with_capacity(iterations);
    let mut last_meta = None;
    for iteration in 0..iterations {
        *request_id += 1;
        request.header.request_id = *request_id;
        for row in &mut request.rows {
            row.source_request_id = *request_id;
        }
        let start = Instant::now();
        let meta = client.roundtrip(config, &request, config.timeout).await?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        last_meta = Some(meta);
    }
    let last_meta = last_meta.context("roundtrip benchmark produced no samples")?;
    Ok(Measurement::from_samples(samples, 1, last_meta))
}

async fn measure_chain(
    client: &mut ProtocolV2BenchClient,
    config: &BenchConfig,
    request_id: &mut u64,
    row_count: usize,
    source_kind: ExpertV2SourceKind,
    label: &str,
) -> Result<Measurement> {
    let mut request = protocol_v2_request_for_row_count(
        *request_id,
        row_count,
        source_kind,
        config.hidden_dim,
        config.layer_id,
        &config.expert_ids,
        config.routes_per_row,
        config.spark_reduction,
        config.nvfp4_fp8_roundtrip,
    )?;
    let start = Instant::now();
    let mut last_meta = None;
    for _ in 0..config.hops {
        *request_id += 1;
        request.header.request_id = *request_id;
        for row in &mut request.rows {
            row.source_request_id = *request_id;
        }
        last_meta = Some(client.roundtrip(config, &request, config.timeout).await?);
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let last_meta = last_meta.with_context(|| format!("{label} produced no hops"))?;
    Ok(Measurement::from_total(total_ms, config.hops, last_meta))
}

async fn roundtrip_once_tcp(
    stream: &mut TcpStream,
    buffers: &mut ProtocolV2Buffers,
    config: &BenchConfig,
    request: &ExpertProtocolV2Request,
    request_timeout: Duration,
) -> Result<RoundtripMeta> {
    let request_frame = buffers.request.encode_request(request)?;
    if request_frame.len() > config.max_frame_bytes {
        bail!(
            "ProtocolV2 request frame {} exceeds max frame bytes {}",
            request_frame.len(),
            config.max_frame_bytes
        );
    }
    timeout(request_timeout, stream.write_all(request_frame))
        .await
        .context("timed out writing ProtocolV2 benchmark request")?
        .context("writing ProtocolV2 benchmark request")?;
    timeout(request_timeout, stream.flush())
        .await
        .context("timed out flushing ProtocolV2 benchmark request")?
        .context("flushing ProtocolV2 benchmark request")?;

    read_response_frame(stream, buffers, config, request_timeout).await?;
    let view = ExpertProtocolV2ResponseView::parse(&buffers.response)
        .context("parsing ProtocolV2 benchmark response")?;
    validate_response(&view, request)?;
    validate_executor(&view, config)?;

    let request_stats = request.wire_stats();
    let response_stats = view.wire_stats();
    Ok(RoundtripMeta {
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        request_wire_bytes: request_stats.wire_bytes,
        response_logical_payload_bytes: response_stats.logical_payload_bytes,
        response_wire_bytes: response_stats.wire_bytes,
        executor_id: view.header.executor_id,
        output_checksum: response_payload_checksum(
            view.header.output_dtype,
            view.partial_output_payload(),
        )?,
        output_hash_fnv1a64: fnv1a64(view.partial_output_payload()),
    })
}

async fn roundtrip_once_verbs_host(
    client: &mut VerbsHostProtocolV2PersistentClient,
    config: &BenchConfig,
    request: &ExpertProtocolV2Request,
) -> Result<RoundtripMeta> {
    if request.spark_reduction_enabled() {
        return roundtrip_once_verbs_host_reduction(client, config, request).await;
    }

    let response = client.roundtrip(request).await?;
    validate_response_header(&response.header, request)?;
    validate_executor_id(response.header.executor_id, config)?;

    let request_stats = request.wire_stats();
    let response_stats = response.wire_stats();
    Ok(RoundtripMeta {
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        request_wire_bytes: request_stats.wire_bytes,
        response_logical_payload_bytes: response_stats.logical_payload_bytes,
        response_wire_bytes: response_stats.wire_bytes,
        executor_id: response.header.executor_id,
        output_checksum: response_payload_checksum(
            response.header.output_dtype,
            response.partial_output_payload.as_ref(),
        )?,
        output_hash_fnv1a64: fnv1a64(response.partial_output_payload.as_ref()),
    })
}

async fn roundtrip_once_verbs_host_reduction(
    client: &mut VerbsHostProtocolV2PersistentClient,
    config: &BenchConfig,
    request: &ExpertProtocolV2Request,
) -> Result<RoundtripMeta> {
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::unbounded_channel();
    let response_stats = client
        .roundtrip_response_chunks(request.clone(), 0, chunk_tx)
        .await?;
    validate_executor_id(response_stats.response_executor_id, config)?;

    let mut response_frames = 0_usize;
    let mut response_wire_bytes = 0_usize;
    let mut response_logical_payload_bytes = 0_usize;
    let mut output_checksum = 0.0_f64;
    let mut output_hash_fnv1a64 = FNV1A64_OFFSET_BASIS;
    while let Some(chunk) = chunk_rx.recv().await {
        response_frames = response_frames
            .checked_add(1)
            .context("ProtocolV2 streamed benchmark response frame count overflow")?;
        response_wire_bytes = response_wire_bytes
            .checked_add(chunk.wire_bytes)
            .context("ProtocolV2 streamed benchmark response byte count overflow")?;
        response_logical_payload_bytes = response_logical_payload_bytes
            .checked_add(chunk.partial_output_payload.as_ref().len())
            .context("ProtocolV2 streamed benchmark response payload byte count overflow")?;
        output_checksum += response_payload_checksum(
            chunk.header.output_dtype,
            chunk.partial_output_payload.as_ref(),
        )?;
        output_hash_fnv1a64 =
            fnv1a64_update(output_hash_fnv1a64, chunk.partial_output_payload.as_ref());
    }
    anyhow::ensure!(
        response_frames == response_stats.response_frames,
        "ProtocolV2 streamed benchmark received {response_frames} frames but transport reported {}",
        response_stats.response_frames
    );
    anyhow::ensure!(
        response_wire_bytes == response_stats.response_wire_bytes,
        "ProtocolV2 streamed benchmark received {response_wire_bytes} bytes but transport reported {}",
        response_stats.response_wire_bytes
    );

    let request_stats = request.wire_stats();
    Ok(RoundtripMeta {
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        request_wire_bytes: request_stats.wire_bytes,
        response_logical_payload_bytes,
        response_wire_bytes,
        executor_id: response_stats.response_executor_id,
        output_checksum,
        output_hash_fnv1a64,
    })
}

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(FNV1A64_OFFSET_BASIS, bytes)
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn response_payload_checksum(dtype: ExpertV2Dtype, bytes: &[u8]) -> Result<f64> {
    if dtype == ExpertV2Dtype::Bf16 {
        return bf16_payload_checksum(bytes);
    }
    Ok(bytes.iter().map(|value| f64::from(*value)).sum())
}

fn bf16_payload_checksum(bytes: &[u8]) -> Result<f64> {
    anyhow::ensure!(
        bytes.len() % std::mem::size_of::<u16>() == 0,
        "ProtocolV2 benchmark BF16 response has odd byte count {}",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|value| {
            let bits = u16::from_le_bytes([value[0], value[1]]);
            f32::from_bits((bits as u32) << 16) as f64
        })
        .sum())
}

async fn read_response_frame(
    stream: &mut TcpStream,
    buffers: &mut ProtocolV2Buffers,
    config: &BenchConfig,
    request_timeout: Duration,
) -> Result<()> {
    buffers.response.clear();
    buffers
        .response
        .resize(EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, 0);
    timeout(request_timeout, stream.read_exact(&mut buffers.response))
        .await
        .context("timed out reading ProtocolV2 response header")?
        .context("reading ProtocolV2 response header")?;
    let wire_bytes = ExpertProtocolV2Response::wire_bytes_from_header(&buffers.response)
        .context("reading ProtocolV2 response wire byte count")?;
    if wire_bytes < EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN {
        bail!("ProtocolV2 response wire bytes smaller than response header");
    }
    if wire_bytes > config.max_frame_bytes {
        bail!(
            "ProtocolV2 response frame {} exceeds max frame bytes {}",
            wire_bytes,
            config.max_frame_bytes
        );
    }
    buffers.response.resize(wire_bytes, 0);
    timeout(
        request_timeout,
        stream.read_exact(&mut buffers.response[EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN..]),
    )
    .await
    .context("timed out reading ProtocolV2 response payload")?
    .context("reading ProtocolV2 response payload")?;
    Ok(())
}

fn validate_response(
    response: &ExpertProtocolV2ResponseView<'_>,
    request: &ExpertProtocolV2Request,
) -> Result<()> {
    validate_response_header(&response.header, request)
}

fn validate_response_header(
    response: &ExpertProtocolV2ResponseHeader,
    request: &ExpertProtocolV2Request,
) -> Result<()> {
    if response.request_id != request.header.request_id {
        bail!(
            "ProtocolV2 response request_id {} did not match request_id {}",
            response.request_id,
            request.header.request_id
        );
    }
    if response.placement_version != request.header.placement_version {
        bail!(
            "ProtocolV2 response placement_version {} did not match request placement_version {}",
            response.placement_version,
            request.header.placement_version
        );
    }
    if response.layer_id != request.header.layer_id {
        bail!(
            "ProtocolV2 response layer_id {} did not match request layer_id {}",
            response.layer_id,
            request.header.layer_id
        );
    }
    if response.row_count != request.header.row_count {
        bail!(
            "ProtocolV2 response row_count {} did not match request row_count {}",
            response.row_count,
            request.header.row_count
        );
    }
    if response.status != ExpertProtocolV2Status::Ok {
        bail!("ProtocolV2 response status was {:?}", response.status);
    }
    Ok(())
}

fn validate_executor(
    response: &ExpertProtocolV2ResponseView<'_>,
    config: &BenchConfig,
) -> Result<()> {
    validate_executor_id(response.header.executor_id, config)
}

fn validate_executor_id(executor_id: u64, config: &BenchConfig) -> Result<()> {
    let Some(expected_executor_id) = config.expected_executor_id else {
        return Ok(());
    };
    if config.require_expected_executor && executor_id != expected_executor_id {
        bail!(
            "ProtocolV2 response executor_id {} did not match expected {} ({})",
            executor_id,
            expected_executor_id,
            config.expected_executor.as_deref().unwrap_or("unknown")
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RoundtripMeta {
    request_logical_payload_bytes: usize,
    request_wire_bytes: usize,
    response_logical_payload_bytes: usize,
    response_wire_bytes: usize,
    executor_id: u64,
    output_checksum: f64,
    output_hash_fnv1a64: u64,
}

#[derive(Debug)]
struct Measurement {
    samples_ms: Vec<f64>,
    hops: usize,
    total_ms: f64,
    meta: RoundtripMeta,
}

impl Measurement {
    fn from_samples(samples_ms: Vec<f64>, hops: usize, meta: RoundtripMeta) -> Self {
        let total_ms = samples_ms.iter().copied().sum::<f64>() / samples_ms.len() as f64;
        Self {
            samples_ms,
            hops,
            total_ms,
            meta,
        }
    }

    fn from_total(total_ms: f64, hops: usize, meta: RoundtripMeta) -> Self {
        Self {
            samples_ms: vec![total_ms / hops as f64],
            hops,
            total_ms,
            meta,
        }
    }

    fn avg_ms(&self) -> f64 {
        self.total_ms / self.hops as f64
    }

    fn min_ms(&self) -> f64 {
        self.samples_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    fn max_ms(&self) -> f64 {
        self.samples_ms
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn percentile_ms(&self, percentile: f64) -> f64 {
        debug_assert!((0.0..=1.0).contains(&percentile));
        let mut samples = self.samples_ms.clone();
        samples.sort_by(f64::total_cmp);
        let position = percentile * (samples.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        samples[lower] + (samples[upper] - samples[lower]) * position.fract()
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkRow {
    benchmark: String,
    protocol: &'static str,
    measured_timeout_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cold_start_excluded: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_protocol: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_request_flag: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_shape_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_iterations_per_shape: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_protocol_invocations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    precompile_timeout_ms: Option<u64>,
    target: String,
    addr: String,
    source_kind: &'static str,
    hidden_dim: usize,
    layer_id: u32,
    expert_id: u32,
    executor_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_executor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_executor_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_matches_expected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    row_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logical_payload_bytes: Option<usize>,
    request_frame_bytes: usize,
    response_frame_bytes: usize,
    response_logical_payload_bytes: usize,
    output_checksum: f64,
    output_hash_fnv1a64: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    hops: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_ms: Option<f64>,
    avg_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p50_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_prefill_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_verify_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aggregate_logical_gbps: Option<f64>,
}

impl BenchmarkRow {
    fn roundtrip(
        config: &BenchConfig,
        row_count: usize,
        source_kind: ExpertV2SourceKind,
        measurement: Measurement,
    ) -> Self {
        let mut row = Self::base(config, "tcp_expert_roundtrip", source_kind, &measurement);
        row.row_count = Some(row_count);
        row.payload_bytes = Some(measurement.meta.request_logical_payload_bytes);
        row.logical_payload_bytes = Some(measurement.meta.request_logical_payload_bytes);
        row.iterations = Some(measurement.samples_ms.len());
        row.min_ms = Some(measurement.min_ms());
        row.p50_ms = Some(measurement.percentile_ms(0.50));
        row.p95_ms = Some(measurement.percentile_ms(0.95));
        row.p99_ms = Some(measurement.percentile_ms(0.99));
        row.max_ms = Some(measurement.max_ms());
        row.aggregate_logical_gbps = Some(aggregate_logical_gbps(
            measurement.meta.request_logical_payload_bytes,
            1,
            measurement.avg_ms(),
        ));
        row
    }

    fn prefill_roundtrip(config: &BenchConfig, row_count: usize, measurement: Measurement) -> Self {
        let mut row = Self::roundtrip(config, row_count, ExpertV2SourceKind::Prefill, measurement);
        row.benchmark = "tcp_expert_prefill_roundtrip".to_owned();
        row.hops = Some(1);
        row.total_ms = Some(row.avg_ms);
        row.effective_prefill_tokens_per_sec = Some(row_count as f64 / (row.avg_ms / 1000.0));
        row
    }

    fn chain(
        config: &BenchConfig,
        benchmark: &str,
        row_count: usize,
        source_kind: ExpertV2SourceKind,
        measurement: Measurement,
    ) -> Self {
        let mut row = Self::base(config, benchmark, source_kind, &measurement);
        row.row_count = Some(row_count);
        row.payload_bytes = Some(measurement.meta.request_logical_payload_bytes);
        row.logical_payload_bytes = Some(measurement.meta.request_logical_payload_bytes);
        row.hops = Some(measurement.hops);
        row.total_ms = Some(measurement.total_ms);
        row.aggregate_logical_gbps = Some(aggregate_logical_gbps(
            measurement.meta.request_logical_payload_bytes,
            measurement.hops,
            measurement.total_ms,
        ));
        if source_kind == ExpertV2SourceKind::MtpVerify {
            row.effective_verify_tokens_per_sec =
                Some(row_count as f64 / (measurement.total_ms / 1000.0));
        }
        row
    }

    fn prefill_chain(config: &BenchConfig, row_count: usize, measurement: Measurement) -> Self {
        let mut row = Self::chain(
            config,
            "tcp_expert_75_layer_prefill_chain",
            row_count,
            ExpertV2SourceKind::Prefill,
            measurement,
        );
        row.effective_prefill_tokens_per_sec =
            Some(row_count as f64 / (row.total_ms.unwrap_or(row.avg_ms) / 1000.0));
        row
    }

    fn base(
        config: &BenchConfig,
        benchmark: &str,
        source_kind: ExpertV2SourceKind,
        measurement: &Measurement,
    ) -> Self {
        let executor_matches_expected = config
            .expected_executor_id
            .map(|expected| measurement.meta.executor_id == expected);
        Self {
            benchmark: benchmark.to_owned(),
            protocol: PROTOCOL_LABEL,
            measured_timeout_ms: config.timeout_ms,
            cold_start_excluded: (config.warmup_iterations > 0).then_some(true),
            precompile_protocol: (config.warmup_iterations > 0).then_some(PROTOCOL_LABEL),
            precompile_request_flag: (config.warmup_iterations > 0)
                .then_some(PRECOMPILE_REQUEST_FLAG_LABEL),
            precompile_shape_count: config.precompile_shape_count(),
            precompile_iterations_per_shape: (config.warmup_iterations > 0)
                .then_some(config.warmup_iterations),
            precompile_protocol_invocations: config.precompile_protocol_invocations(),
            precompile_timeout_ms: (config.warmup_iterations > 0)
                .then_some(config.warmup_timeout_ms),
            target: config.target.clone(),
            addr: config.addr_label.clone(),
            source_kind: source_kind_label(source_kind),
            hidden_dim: config.hidden_dim,
            layer_id: config.layer_id,
            expert_id: config.expert_id,
            executor_id: measurement.meta.executor_id,
            expected_executor: config.expected_executor.clone(),
            expected_executor_id: config.expected_executor_id,
            executor_matches_expected,
            row_count: None,
            payload_bytes: None,
            logical_payload_bytes: None,
            request_frame_bytes: measurement.meta.request_wire_bytes,
            response_frame_bytes: measurement.meta.response_wire_bytes,
            response_logical_payload_bytes: measurement.meta.response_logical_payload_bytes,
            output_checksum: measurement.meta.output_checksum,
            output_hash_fnv1a64: measurement.meta.output_hash_fnv1a64,
            hops: None,
            iterations: None,
            total_ms: None,
            avg_ms: measurement.avg_ms(),
            min_ms: None,
            p50_ms: None,
            p95_ms: None,
            p99_ms: None,
            max_ms: None,
            effective_prefill_tokens_per_sec: None,
            effective_verify_tokens_per_sec: None,
            aggregate_logical_gbps: None,
        }
    }
}

fn aggregate_logical_gbps(logical_payload_bytes: usize, hops: usize, total_ms: f64) -> f64 {
    ((logical_payload_bytes * hops * 2) as f64 * 8.0) / (total_ms / 1000.0) / 1e9
}

fn iterations_for_row_count(row_count: usize, config: &BenchConfig) -> usize {
    if row_count >= 256 {
        config.large_iterations
    } else {
        config.iterations
    }
}

fn protocol_v2_request_for_row_count(
    request_id: u64,
    row_count: usize,
    source_kind: ExpertV2SourceKind,
    hidden_dim: usize,
    layer_id: u32,
    expert_ids: &[u32],
    routes_per_row: usize,
    spark_reduction: bool,
    nvfp4_fp8_roundtrip: bool,
) -> Result<ExpertProtocolV2Request> {
    if routes_per_row == 0 {
        bail!("routes_per_row must be non-zero");
    }
    if expert_ids.len() < routes_per_row {
        bail!("expert_ids must contain at least routes_per_row entries");
    }
    let rows = (0..row_count)
        .map(|idx| ExpertProtocolV2RowDescriptor {
            row_id: idx as u64,
            source_kind,
            source_request_id: request_id,
            token_position: idx as u64,
            route_offset: (idx * routes_per_row) as u32,
            route_count: routes_per_row as u32,
        })
        .collect::<Vec<_>>();
    let routes = (0..row_count)
        .flat_map(|idx| {
            (0..routes_per_row).map(move |route_idx| {
                let expert_id = expert_ids[(idx * routes_per_row + route_idx) % expert_ids.len()];
                ExpertProtocolV2RouteEntry {
                    row_index: idx as u32,
                    expert_id,
                    gate_weight: 1.0 / routes_per_row as f32,
                }
            })
        })
        .collect::<Vec<_>>();
    let low_precision_roundtrip = spark_reduction || nvfp4_fp8_roundtrip;
    let hidden_dtype = if low_precision_roundtrip {
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
    } else {
        ExpertV2Dtype::Bf16
    };
    let hidden_payload = if low_precision_roundtrip {
        deterministic_hidden_payload_nvfp4(row_count, hidden_dim)?
    } else {
        deterministic_hidden_payload_bf16(row_count, hidden_dim)?
    };
    let request = ExpertProtocolV2Request::new(
        request_id,
        1,
        layer_id,
        hidden_dim as u32,
        hidden_dtype,
        rows,
        routes,
        hidden_payload,
    )?;
    let request = if spark_reduction {
        request
            .with_fp8_e4m3_row_scaled_response()
            .with_spark_reduction()
    } else if nvfp4_fp8_roundtrip {
        request.with_fp8_e4m3_row_scaled_response()
    } else {
        request
    };
    Ok(request)
}

fn deterministic_hidden_payload_nvfp4(row_count: usize, hidden_dim: usize) -> Result<Vec<u8>> {
    let row_bytes = ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(hidden_dim)?;
    let packed_bytes = hidden_dim / 2;
    let scale_bytes = hidden_dim / 16;
    anyhow::ensure!(
        hidden_dim > 0 && hidden_dim % 16 == 0 && row_bytes == packed_bytes + scale_bytes,
        "deterministic NVFP4 hidden shape is invalid: hidden={hidden_dim} row_bytes={row_bytes}"
    );
    let payload_bytes = row_count
        .checked_mul(row_bytes)
        .context("deterministic NVFP4 hidden payload byte count overflow")?;
    let mut payload = vec![0_u8; payload_bytes];
    for row in 0..row_count {
        let row_payload = &mut payload[row * row_bytes..(row + 1) * row_bytes];
        for (packed_index, packed) in row_payload[..packed_bytes].iter_mut().enumerate() {
            let code = |lane: usize| {
                let pattern = row
                    .wrapping_mul(17)
                    .wrapping_add(packed_index.wrapping_mul(2))
                    .wrapping_add(lane);
                let magnitude = 1 + pattern % 7;
                (magnitude | (((pattern / 7) & 1) << 3)) as u8
            };
            *packed = code(0) | (code(1) << 4);
        }
        // E4M3 0x18 is 2^-4. Combined with the E2M1 codes above, this
        // exercises nonzero expert math while keeping activations in roughly
        // the same range as the BF16 probe.
        row_payload[packed_bytes..packed_bytes + scale_bytes].fill(0x18);
    }
    Ok(payload)
}

fn deterministic_hidden_payload_bf16(row_count: usize, hidden_dim: usize) -> Result<Vec<u8>> {
    let value_count = row_count
        .checked_mul(hidden_dim)
        .context("deterministic hidden value count overflow")?;
    let mut payload = Vec::with_capacity(
        value_count
            .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
            .context("deterministic hidden payload byte count overflow")?,
    );
    for row in 0..row_count {
        for col in 0..hidden_dim {
            let lane = ((row as u32)
                .wrapping_mul(17)
                .wrapping_add((col as u32).wrapping_mul(31))
                .wrapping_add(13)
                % 257) as f32;
            let value = (lane - 128.0) / 256.0;
            payload.extend_from_slice(&f32_to_bf16_bytes(value));
        }
    }
    Ok(payload)
}

fn f32_to_bf16_bytes(value: f32) -> [u8; 2] {
    ((value.to_bits() >> 16) as u16).to_le_bytes()
}

fn source_kind_for_row_count(row_count: usize) -> ExpertV2SourceKind {
    match row_count {
        1 => ExpertV2SourceKind::Decode,
        2 | 3 | 4 | 5 | 6 | 8 => ExpertV2SourceKind::MtpVerify,
        _ => ExpertV2SourceKind::Prefill,
    }
}

fn source_kind_label(source_kind: ExpertV2SourceKind) -> &'static str {
    match source_kind {
        ExpertV2SourceKind::Decode => "decode",
        ExpertV2SourceKind::Prefill => "prefill",
        ExpertV2SourceKind::MtpVerify => "mtp_verify",
        ExpertV2SourceKind::Benchmark => "benchmark",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_response_checksum_decodes_payload_values() -> Result<()> {
        let bytes = [1.0_f32, -2.0, 0.5]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(bf16_payload_checksum(&bytes)?, -0.5);
        assert!(bf16_payload_checksum(&bytes[..bytes.len() - 1]).is_err());
        Ok(())
    }

    #[test]
    fn compact_response_checksum_and_percentiles_are_codec_safe() -> Result<()> {
        assert_eq!(
            response_payload_checksum(ExpertV2Dtype::Fp8E4m3RowScaled, &[1, 2, 255])?,
            258.0
        );
        let measurement = Measurement::from_samples(
            vec![100.0, 1.0, 4.0, 2.0, 3.0],
            1,
            RoundtripMeta {
                request_logical_payload_bytes: 0,
                request_wire_bytes: 0,
                response_logical_payload_bytes: 0,
                response_wire_bytes: 0,
                executor_id: 0,
                output_checksum: 0.0,
                output_hash_fnv1a64: FNV1A64_OFFSET_BASIS,
            },
        );
        assert_eq!(measurement.percentile_ms(0.50), 3.0);
        assert!((measurement.percentile_ms(0.95) - 80.8).abs() < f64::EPSILON * 128.0);
        Ok(())
    }

    #[test]
    fn expert_id_pattern_cycles_without_changing_routes_per_row() -> Result<()> {
        let request = protocol_v2_request_for_row_count(
            17,
            3,
            ExpertV2SourceKind::Prefill,
            DS4_FLASH_HIDDEN_SIZE,
            3,
            &[0, 4, 8, 12],
            2,
            false,
            false,
        )?;
        assert_eq!(request.routes.len(), 6);
        assert_eq!(
            request
                .routes
                .iter()
                .map(|route| route.expert_id)
                .collect::<Vec<_>>(),
            vec![0, 4, 8, 12, 0, 4]
        );
        assert!(request.rows.iter().all(|row| row.route_count == 2));
        Ok(())
    }

    #[test]
    fn spark_reduction_request_matches_flash_tp4_wire_shape() -> Result<()> {
        let request = protocol_v2_request_for_row_count(
            18,
            1,
            ExpertV2SourceKind::Decode,
            DS4_FLASH_HIDDEN_SIZE,
            0,
            &[0, 1, 2, 3, 4, 5],
            6,
            true,
            false,
        )?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Nvfp4E2m1Fp8E4m3);
        assert_eq!(request.routes.len(), 6);
        assert_eq!(
            request.hidden_payload.len(),
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(DS4_FLASH_HIDDEN_SIZE)?
        );
        assert!(request.fp8_e4m3_row_scaled_response_enabled());
        assert!(request.spark_reduction_enabled());
        Ok(())
    }

    #[test]
    fn nvfp4_fp8_roundtrip_request_uses_direct_low_precision_wire_shape() -> Result<()> {
        let request = protocol_v2_request_for_row_count(
            19,
            8,
            ExpertV2SourceKind::MtpVerify,
            DS4_FLASH_HIDDEN_SIZE,
            0,
            &[0, 1, 2, 3, 4, 5],
            6,
            false,
            true,
        )?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Nvfp4E2m1Fp8E4m3);
        assert_eq!(request.routes.len(), 48);
        assert_eq!(
            request.hidden_payload.len(),
            8 * ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(DS4_FLASH_HIDDEN_SIZE)?
        );
        assert!(request.hidden_payload.iter().any(|&byte| byte != 0));
        assert!(request.fp8_e4m3_row_scaled_response_enabled());
        assert!(!request.spark_reduction_enabled());
        Ok(())
    }

    use ds41rt_transport::{
        serve_protocol_v2_tcp_listener_with_executor, serve_synthetic_protocol_v2_tcp_listener,
        ExpertProtocolV2RequestView, ExpertProtocolV2Response, ProtocolV2ExpertExecutor,
        SyntheticRouteExecutor, PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::net::TcpListener;

    #[test]
    fn parses_row_counts() {
        assert_eq!(parse_row_counts("1, 2,4", "--rows").unwrap(), vec![1, 2, 4]);
        assert!(parse_row_counts("", "--rows").is_err());
        assert!(parse_row_counts("0", "--rows").is_err());
    }

    #[test]
    fn protocol_v2_warmup_shapes_cover_each_measured_protocol_shape_once() {
        let args = BenchProtocolV2TcpArgs {
            addr: "127.0.0.1:9".to_owned(),
            transport: "tcp".to_owned(),
            target: "shape-test".to_owned(),
            request_id_start: 1,
            hops: 75,
            iterations: 1,
            large_iterations: 1,
            warmup_iterations: 1,
            warmup_rows: 64,
            warmup_timeout_ms: Some(120_000),
            warmup_only: false,
            roundtrip_only: false,
            roundtrip_rows: "1,4,16".to_owned(),
            mtp_chain_rows: "4,5".to_owned(),
            prefill_roundtrip_rows: "16,32".to_owned(),
            prefill_chain_rows: "32".to_owned(),
            layer_id: 0,
            expert_id: 0,
            hidden_dim: DS4_FLASH_HIDDEN_SIZE,
            expert_ids: None,
            routes_per_row: 1,
            spark_reduction: false,
            nvfp4_fp8_roundtrip: false,
            expected_executor: Some(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR.to_owned()),
            require_expected_executor: true,
            timeout_ms: 5_000,
            max_frame_bytes: 64 * 1024 * 1024,
        };
        let config = BenchConfig::from_args(args).unwrap();

        let shapes = protocol_v2_warmup_shapes(&config);

        assert_eq!(config.precompile_shape_count(), Some(6));
        assert_eq!(config.precompile_protocol_invocations(), Some(6));
        assert_eq!(
            shapes,
            vec![
                WarmupShape {
                    row_count: 64,
                    source_kind: ExpertV2SourceKind::Prefill,
                },
                WarmupShape {
                    row_count: 1,
                    source_kind: ExpertV2SourceKind::Decode,
                },
                WarmupShape {
                    row_count: 4,
                    source_kind: ExpertV2SourceKind::MtpVerify,
                },
                WarmupShape {
                    row_count: 16,
                    source_kind: ExpertV2SourceKind::Prefill,
                },
                WarmupShape {
                    row_count: 5,
                    source_kind: ExpertV2SourceKind::MtpVerify,
                },
                WarmupShape {
                    row_count: 32,
                    source_kind: ExpertV2SourceKind::Prefill,
                },
            ]
        );
    }

    #[tokio::test]
    async fn benchmarks_binary_protocol_v2_tcp_frames() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = serve_synthetic_protocol_v2_tcp_listener(listener).await;
        });
        let args = BenchProtocolV2TcpArgs {
            addr: addr.to_string(),
            transport: "tcp".to_owned(),
            target: "local-protocol-smoke".to_owned(),
            request_id_start: 1,
            hops: 2,
            iterations: 1,
            large_iterations: 1,
            warmup_iterations: 0,
            warmup_rows: 1,
            warmup_timeout_ms: None,
            warmup_only: false,
            roundtrip_only: false,
            roundtrip_rows: "1".to_owned(),
            mtp_chain_rows: "2".to_owned(),
            prefill_roundtrip_rows: "2".to_owned(),
            prefill_chain_rows: "2".to_owned(),
            layer_id: 0,
            expert_id: 0,
            hidden_dim: DS4_FLASH_HIDDEN_SIZE,
            expert_ids: None,
            routes_per_row: 1,
            spark_reduction: false,
            nvfp4_fp8_roundtrip: false,
            expected_executor: Some(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR.to_owned()),
            require_expected_executor: true,
            timeout_ms: 5000,
            max_frame_bytes: 64 * 1024 * 1024,
        };

        let rows = benchmark_protocol_v2_tcp_rows(args).await.unwrap();
        server.abort();

        assert_eq!(rows.len(), 5);
        assert!(rows.iter().all(
            |row| row.protocol == PROTOCOL_LABEL && row.executor_matches_expected == Some(true)
        ));
        assert!(rows.iter().all(|row| row.measured_timeout_ms == 5000
            && row.cold_start_excluded.is_none()
            && row.precompile_request_flag.is_none()
            && row.precompile_protocol_invocations.is_none()));
        let serialized = serde_json::to_value(&rows[0]).unwrap();
        assert_eq!(serialized["measured_timeout_ms"], 5000);
        assert!(serialized.get("precompile_request_flag").is_none());
        assert!(serialized.get("precompile_protocol_invocations").is_none());
        assert!(rows
            .iter()
            .any(|row| row.benchmark == "tcp_expert_75_layer_prefill_chain"));
    }

    #[tokio::test]
    async fn protocol_v2_warmup_frames_are_not_reported_as_measurements() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = serve_synthetic_protocol_v2_tcp_listener(listener).await;
        });
        let args = BenchProtocolV2TcpArgs {
            addr: addr.to_string(),
            transport: "tcp".to_owned(),
            target: "local-protocol-warmup-smoke".to_owned(),
            request_id_start: 1,
            hops: 1,
            iterations: 2,
            large_iterations: 2,
            warmup_iterations: 3,
            warmup_rows: 1,
            warmup_timeout_ms: Some(5000),
            warmup_only: false,
            roundtrip_only: false,
            roundtrip_rows: "1".to_owned(),
            mtp_chain_rows: "2".to_owned(),
            prefill_roundtrip_rows: "16".to_owned(),
            prefill_chain_rows: "16".to_owned(),
            layer_id: 0,
            expert_id: 0,
            hidden_dim: DS4_FLASH_HIDDEN_SIZE,
            expert_ids: None,
            routes_per_row: 1,
            spark_reduction: false,
            nvfp4_fp8_roundtrip: false,
            expected_executor: Some(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR.to_owned()),
            require_expected_executor: true,
            timeout_ms: 5000,
            max_frame_bytes: 64 * 1024 * 1024,
        };

        let rows = benchmark_protocol_v2_tcp_rows(args).await.unwrap();
        server.abort();

        assert_eq!(rows.len(), 5);
        let roundtrip = rows
            .iter()
            .find(|row| row.benchmark == "tcp_expert_roundtrip")
            .expect("roundtrip benchmark row");
        assert_eq!(roundtrip.iterations, Some(2));
        let prefill_roundtrip = rows
            .iter()
            .find(|row| row.benchmark == "tcp_expert_prefill_roundtrip")
            .expect("prefill roundtrip benchmark row");
        assert_eq!(prefill_roundtrip.iterations, Some(2));
        assert!(rows.iter().all(
            |row| row.protocol == PROTOCOL_LABEL && row.executor_matches_expected == Some(true)
        ));
        assert!(rows.iter().all(|row| row.measured_timeout_ms == 5000
            && row.cold_start_excluded == Some(true)
            && row.precompile_protocol == Some(PROTOCOL_LABEL)
            && row.precompile_request_flag == Some(PRECOMPILE_REQUEST_FLAG_LABEL)
            && row.precompile_shape_count == Some(3)
            && row.precompile_iterations_per_shape == Some(3)
            && row.precompile_protocol_invocations == Some(9)
            && row.precompile_timeout_ms == Some(5000)));
        let serialized = serde_json::to_value(roundtrip).unwrap();
        assert_eq!(serialized["cold_start_excluded"], true);
        assert_eq!(serialized["precompile_protocol"], PROTOCOL_LABEL);
        assert_eq!(
            serialized["precompile_request_flag"],
            PRECOMPILE_REQUEST_FLAG_LABEL
        );
        assert_eq!(serialized["precompile_protocol_invocations"], 9);
        assert_eq!(serialized["precompile_timeout_ms"], 5000);
        assert_eq!(serialized["measured_timeout_ms"], 5000);
    }

    #[tokio::test]
    async fn warmup_only_sends_precompile_frames_without_measurement_rows() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let precompile_request_count = Arc::new(AtomicUsize::new(0));
        let hot_request_count = Arc::new(AtomicUsize::new(0));
        let executor = Arc::new(CountingSyntheticExecutor {
            request_count: Arc::clone(&request_count),
            precompile_request_count: Arc::clone(&precompile_request_count),
            hot_request_count: Arc::clone(&hot_request_count),
        });
        let server = tokio::spawn(async move {
            let _ = serve_protocol_v2_tcp_listener_with_executor(listener, executor).await;
        });
        let args = BenchProtocolV2TcpArgs {
            addr: addr.to_string(),
            transport: "tcp".to_owned(),
            target: "local-protocol-warmup-only".to_owned(),
            request_id_start: 1,
            hops: 1,
            iterations: 1,
            large_iterations: 1,
            warmup_iterations: 1,
            warmup_rows: 1,
            warmup_timeout_ms: Some(5000),
            warmup_only: true,
            roundtrip_only: false,
            roundtrip_rows: "1".to_owned(),
            mtp_chain_rows: "2".to_owned(),
            prefill_roundtrip_rows: "16".to_owned(),
            prefill_chain_rows: "16".to_owned(),
            layer_id: 0,
            expert_id: 0,
            hidden_dim: DS4_FLASH_HIDDEN_SIZE,
            expert_ids: None,
            routes_per_row: 1,
            spark_reduction: false,
            nvfp4_fp8_roundtrip: false,
            expected_executor: Some(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR.to_owned()),
            require_expected_executor: true,
            timeout_ms: 5000,
            max_frame_bytes: 64 * 1024 * 1024,
        };

        let rows = benchmark_protocol_v2_tcp_rows(args).await.unwrap();
        server.abort();

        assert!(rows.is_empty());
        assert_eq!(request_count.load(Ordering::SeqCst), 3);
        assert_eq!(precompile_request_count.load(Ordering::SeqCst), 3);
        assert_eq!(hot_request_count.load(Ordering::SeqCst), 0);
    }

    struct CountingSyntheticExecutor {
        request_count: Arc<AtomicUsize>,
        precompile_request_count: Arc<AtomicUsize>,
        hot_request_count: Arc<AtomicUsize>,
    }

    impl ProtocolV2ExpertExecutor for CountingSyntheticExecutor {
        fn name(&self) -> &'static str {
            PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR
        }

        fn execute(
            &self,
            request: &ExpertProtocolV2RequestView<'_>,
        ) -> Result<ExpertProtocolV2Response> {
            self.request_count.fetch_add(1, Ordering::SeqCst);
            if request.precompile_warmup_enabled() {
                self.precompile_request_count.fetch_add(1, Ordering::SeqCst);
            } else {
                self.hot_request_count.fetch_add(1, Ordering::SeqCst);
            }
            SyntheticRouteExecutor.execute(request)
        }
    }
}
