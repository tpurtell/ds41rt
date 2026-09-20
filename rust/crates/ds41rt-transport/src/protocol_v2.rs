use anyhow::{bail, Context, Result};
use bytes::Bytes;
use ds41rt_ffi::Ds41rtDeviceBuffer;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"DS41RTE3";
const VERSION: u16 = 3;
const REQUEST_KIND: u16 = 1;
const RESPONSE_KIND: u16 = 2;
pub const EXPERT_PROTOCOL_V2_FRAME_PROTOCOL: &str = "ExpertProtocolV2";
pub const EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN: usize = 96;
pub const EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN: usize = 96;
pub const EXPERT_PROTOCOL_V2_REQUEST_DEBUG_HEADER_LEN: usize = 128;
pub const EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN: usize = 128;
pub const EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN: usize = 40;
pub const EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN: usize = 12;
pub const EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM: u32 = 1 << 0;
pub const EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP: u32 = 1 << 1;
pub const EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES: u32 = 1 << 2;
pub const EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS: u32 = 1 << 3;
pub const EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED: u32 = 1 << 4;
pub const EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3: u32 = 1 << 5;
pub const EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN: u32 = 1 << 6;
pub const EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA: u32 = 1 << 7;
pub const EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL: u32 = 1 << 8;
pub const EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION: u32 = 1 << 9;
// Native V4.1 compact BF16 rank partials; echoed in responses.
pub const EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16: u32 = 1 << 16;
pub const EXPERT_PROTOCOL_V2_FLAG_SPARK_ROW_SHARDED_REDUCTION: u32 = 1 << 11;
const EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_SHIFT: u32 = 12;
const EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_MASK: u32 =
    0x0f << EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_SHIFT;
pub const EXPERT_PROTOCOL_V2_MAX_SPARK_COLLECTIVE_PARTS: usize = 15;
const CHECKSUM_LEN: usize = 32;
const REQUEST_CHECKSUM_OFFSET: usize = 92;
const RESPONSE_EXECUTOR_ID_OFFSET: usize = 80;
const RESPONSE_CHECKSUM_OFFSET: usize = 88;

mod batch;
mod request;
mod response;
mod stream;
mod view;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertV2Dtype {
    Bf16 = 1,
    F16 = 2,
    Fp8Debug = 3,
    Nvfp4E2m1Fp8E4m3 = 4,
    /// E4M3 values followed by one little-endian FP32 dequantization scale per row.
    Fp8E4m3RowScaled = 5,
    /// Unrounded native expert partial sums.
    F32 = 6,
    /// Each row stores E4M3 payload bytes, then one UE8M0 scale byte per
    /// contiguous 32 values. Distinct from FP32 row-scaled E4M3.
    /// Native V4.1 producers quantize BF16-rounded values with an amax floor
    /// of 1e-4; consumers must preserve the supplied payload and scales.
    Fp8E4m3Ue8m0K32 = 7,
}

impl ExpertV2Dtype {
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::Bf16 | Self::F16 => 2,
            Self::Fp8Debug
            | Self::Nvfp4E2m1Fp8E4m3
            | Self::Fp8E4m3RowScaled
            | Self::Fp8E4m3Ue8m0K32 => 1,
        }
    }

    pub fn row_bytes(self, elements: usize) -> Result<usize> {
        match self {
            Self::F32 => elements
                .checked_mul(4)
                .context("FP32 row byte count overflow"),
            Self::Bf16 | Self::F16 => elements
                .checked_mul(2)
                .context("16-bit row byte count overflow"),
            Self::Fp8Debug => Ok(elements),
            Self::Fp8E4m3Ue8m0K32 => {
                if elements == 0 || elements % 32 != 0 {
                    bail!("E4M3/UE8M0 K32 row width must be a nonzero multiple of 32");
                }
                elements
                    .checked_add(elements / 32)
                    .context("E4M3/UE8M0 K32 row byte count overflow")
            }
            Self::Nvfp4E2m1Fp8E4m3 => {
                if elements == 0 || elements % 16 != 0 {
                    bail!(
                        "NVFP4 E2M1 + FP8 E4M3 row width must be a nonzero multiple of 16, got {elements}"
                    );
                }
                elements
                    .checked_div(2)
                    .and_then(|packed| packed.checked_add(elements / 16))
                    .context("NVFP4 E2M1 + FP8 E4M3 row byte count overflow")
            }
            Self::Fp8E4m3RowScaled => elements
                .checked_add(std::mem::size_of::<f32>())
                .context("row-scaled FP8 E4M3 row byte count overflow"),
        }
    }

    fn from_u16(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::Bf16),
            2 => Ok(Self::F16),
            3 => Ok(Self::Fp8Debug),
            4 => Ok(Self::Nvfp4E2m1Fp8E4m3),
            5 => Ok(Self::Fp8E4m3RowScaled),
            6 => Ok(Self::F32),
            7 => Ok(Self::Fp8E4m3Ue8m0K32),
            other => bail!("unknown ExpertProtocolV2 dtype {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertV2SourceKind {
    Decode = 1,
    Prefill = 2,
    MtpVerify = 3,
    Benchmark = 4,
}

impl ExpertV2SourceKind {
    fn from_u16(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::Decode),
            2 => Ok(Self::Prefill),
            3 => Ok(Self::MtpVerify),
            4 => Ok(Self::Benchmark),
            other => bail!("unknown ExpertProtocolV2 source kind {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProtocolV2Status {
    Ok = 0,
    Error = 1,
}

impl ExpertProtocolV2Status {
    fn from_u16(value: u16) -> Result<Self> {
        match value {
            0 => Ok(Self::Ok),
            1 => Ok(Self::Error),
            other => bail!("unknown ExpertProtocolV2 response status {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertProtocolV2WireStats {
    pub logical_payload_bytes: usize,
    pub wire_bytes: usize,
}

#[derive(Debug, Default)]
pub struct ExpertProtocolV2FrameBuffer {
    bytes: Vec<u8>,
}

impl ExpertProtocolV2FrameBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    pub fn encode_request(&mut self, request: &ExpertProtocolV2Request) -> Result<&[u8]> {
        request.encode_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub fn encode_regular_forwarded_request(
        &mut self,
        request: &ExpertProtocolV2RequestView<'_>,
        response_dtype: ExpertV2Dtype,
    ) -> Result<&[u8]> {
        request.encode_regular_forwarded_into(response_dtype, &mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub(crate) fn encode_request_prefix(
        &mut self,
        request: &ExpertProtocolV2Request,
    ) -> Result<&[u8]> {
        request.encode_prefix_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub fn encode_response(&mut self, response: &ExpertProtocolV2Response) -> Result<&[u8]> {
        response.encode_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub(crate) fn encode_response_prefix(
        &mut self,
        response: &ExpertProtocolV2Response,
    ) -> Result<&[u8]> {
        response.encode_prefix_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub(crate) fn encode_borrowed_response_prefix(
        &mut self,
        response: &ExpertProtocolV2ResponseRef<'_>,
    ) -> Result<&[u8]> {
        response.encode_prefix_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub(crate) fn encode_device_response_prefix(
        &mut self,
        response: &ExpertProtocolV2DeviceResponseRef<'_>,
    ) -> Result<&[u8]> {
        response.encode_prefix_into(&mut self.bytes)?;
        Ok(self.as_slice())
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut Vec<u8> {
        &mut self.bytes
    }
}

pub fn expert_protocol_v2_compact_id(value: &str) -> u64 {
    let digest = Sha256::digest(value.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProtocolV2RequestHeader {
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub row_count: u32,
    pub hidden_dim: u32,
    pub hidden_dtype: ExpertV2Dtype,
    pub hidden_row_stride_bytes: u32,
    pub route_count: u32,
    pub row_descriptor_bytes: u32,
    pub route_bytes: u32,
    pub hidden_payload_bytes: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProtocolV2RowDescriptor {
    pub row_id: u64,
    pub source_kind: ExpertV2SourceKind,
    pub source_request_id: u64,
    pub token_position: u64,
    pub route_offset: u32,
    pub route_count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertProtocolV2RouteEntry {
    pub row_index: u32,
    pub expert_id: u32,
    pub gate_weight: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProtocolV2ResponseHeader {
    pub request_id: u64,
    pub placement_version: u64,
    pub layer_id: u32,
    pub row_count: u32,
    pub output_dim: u32,
    pub output_dtype: ExpertV2Dtype,
    pub output_row_stride_bytes: u32,
    pub output_payload_bytes: u64,
    pub status: ExpertProtocolV2Status,
    pub flags: u32,
    pub executor_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertProtocolV2Request {
    pub header: ExpertProtocolV2RequestHeader,
    pub rows: Vec<ExpertProtocolV2RowDescriptor>,
    pub routes: Vec<ExpertProtocolV2RouteEntry>,
    pub hidden_payload: Bytes,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertProtocolV2Response {
    pub header: ExpertProtocolV2ResponseHeader,
    pub row_indices: Option<Vec<u32>>,
    pub partial_output_payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertProtocolV2ResponseRef<'a> {
    pub header: ExpertProtocolV2ResponseHeader,
    pub row_indices: Option<&'a [u32]>,
    pub partial_output_payload: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct ExpertProtocolV2DeviceResponseRef<'a> {
    pub header: ExpertProtocolV2ResponseHeader,
    pub row_indices: Option<&'a [u32]>,
    pub partial_output_payload: Ds41rtDeviceBuffer,
}

pub use stream::{ExpertProtocolV2StreamPlan, ExpertProtocolV2StreamRouteGroup};
pub use view::{ExpertProtocolV2RequestView, ExpertProtocolV2ResponseView};

#[derive(Debug, Default)]
pub struct ExpertProtocolV2FrameArena {
    request_buffer: ExpertProtocolV2FrameBuffer,
    response_buffer: ExpertProtocolV2FrameBuffer,
}

impl ExpertProtocolV2FrameArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacities(request_capacity: usize, response_capacity: usize) -> Self {
        Self {
            request_buffer: ExpertProtocolV2FrameBuffer::with_capacity(request_capacity),
            response_buffer: ExpertProtocolV2FrameBuffer::with_capacity(response_capacity),
        }
    }

    pub fn request_frame(&self) -> &[u8] {
        self.request_buffer.as_slice()
    }

    pub fn response_frame(&self) -> &[u8] {
        self.response_buffer.as_slice()
    }

    pub fn request_capacity(&self) -> usize {
        self.request_buffer.capacity()
    }

    pub fn response_capacity(&self) -> usize {
        self.response_buffer.capacity()
    }

    pub fn request_ptr(&self) -> *const u8 {
        self.request_buffer.as_ptr()
    }

    pub fn response_ptr(&self) -> *const u8 {
        self.response_buffer.as_ptr()
    }

    pub fn request_buffer_mut(&mut self) -> &mut ExpertProtocolV2FrameBuffer {
        &mut self.request_buffer
    }

    pub fn response_buffer_mut(&mut self) -> &mut ExpertProtocolV2FrameBuffer {
        &mut self.response_buffer
    }

    pub fn encode_request_view(
        &mut self,
        request: &ExpertProtocolV2Request,
    ) -> Result<ExpertProtocolV2RequestView<'_>> {
        self.request_buffer.encode_request(request)?;
        ExpertProtocolV2RequestView::parse(self.request_buffer.as_slice())
    }

    pub fn encode_response_view(
        &mut self,
        response: &ExpertProtocolV2Response,
    ) -> Result<ExpertProtocolV2ResponseView<'_>> {
        self.response_buffer.encode_response(response)?;
        ExpertProtocolV2ResponseView::parse(self.response_buffer.as_slice())
    }
}

fn debug_checksum_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM != 0
}

fn precompile_warmup_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP != 0
}

fn response_fp8_e4m3_row_scaled_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED != 0
}

fn response_nvfp4_e2m1_fp8_e4m3_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3 != 0
}

fn stream_plan_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN != 0
}

fn stream_data_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA != 0
}

fn stream_final_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL != 0
}

fn spark_reduction_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION != 0
}

fn spark_row_sharded_reduction_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_SPARK_ROW_SHARDED_REDUCTION != 0
}

fn spark_collective_part_count(flags: u32) -> usize {
    let encoded = (flags & EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_MASK)
        >> EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_SHIFT;
    usize::try_from(encoded).expect("four-bit Spark collective part count fits usize")
}

fn response_row_indices_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES != 0
}

fn response_more_chunks_enabled(flags: u32) -> bool {
    flags & EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS != 0
}

fn request_header_len_from_flags(flags: u32) -> usize {
    if debug_checksum_enabled(flags) {
        EXPERT_PROTOCOL_V2_REQUEST_DEBUG_HEADER_LEN
    } else {
        EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
    }
}

fn response_header_len_from_flags(flags: u32) -> usize {
    if debug_checksum_enabled(flags) {
        EXPERT_PROTOCOL_V2_RESPONSE_DEBUG_HEADER_LEN
    } else {
        EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
    }
}

fn validate_flags(flags: u32, label: &str) -> Result<()> {
    let allowed = match label {
        "request" => {
            crate::v41_expert::V41_EXL3_PAIRED_REQUEST_FLAG
                | crate::v41_expert::V41_NATIVE_GROUP_REQUEST_FLAG
                | EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
                | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16
                | EXPERT_PROTOCOL_V2_FLAG_PRECOMPILE_WARMUP
                | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED
                | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_NVFP4_E2M1_FP8_E4M3
                | EXPERT_PROTOCOL_V2_FLAG_STREAM_PLAN
                | EXPERT_PROTOCOL_V2_FLAG_STREAM_DATA
                | EXPERT_PROTOCOL_V2_FLAG_STREAM_FINAL
                | EXPERT_PROTOCOL_V2_FLAG_SPARK_REDUCTION
                | EXPERT_PROTOCOL_V2_FLAG_SPARK_ROW_SHARDED_REDUCTION
                | EXPERT_PROTOCOL_V2_SPARK_COLLECTIVE_PART_COUNT_MASK
        }
        "response" => {
            EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
                | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16
                | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_ROW_INDICES
                | EXPERT_PROTOCOL_V2_FLAG_RESPONSE_MORE_CHUNKS
        }
        _ => EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
    };
    if label == "request" && flags & crate::v41_expert::V41_EXL3_PAIRED_REQUEST_FLAG != 0 {
        let paired_allowed = crate::v41_expert::V41_EXL3_PAIRED_REQUEST_FLAG
            | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 | EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        if flags & !paired_allowed != 0 || flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 == 0 {
            bail!("paired EXL3 requires the native compact request contract");
        }
    }
    // Native replicated-group ownership: same compact request contract as the
    // canonical native batch, never combined with paired EXL3 admission, and
    // never present in a response.
    if label == "request" && flags & crate::v41_expert::V41_NATIVE_GROUP_REQUEST_FLAG != 0 {
        let native_allowed = crate::v41_expert::V41_NATIVE_GROUP_REQUEST_FLAG
            | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 | EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM;
        if flags & crate::v41_expert::V41_EXL3_PAIRED_REQUEST_FLAG != 0 {
            bail!("native group ownership cannot combine with paired EXL3 admission");
        }
        if flags & !native_allowed != 0 || flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 == 0 {
            bail!("native group ownership requires the native compact request contract");
        }
    }
    let unknown = flags & !allowed;
    if unknown != 0 {
        bail!("ExpertProtocolV2 {label} has unknown flags 0x{unknown:08x}");
    }
    if label == "request"
        && response_fp8_e4m3_row_scaled_enabled(flags)
        && response_nvfp4_e2m1_fp8_e4m3_enabled(flags)
    {
        bail!("ExpertProtocolV2 request selects multiple low-precision response dtypes");
    }
    if label == "request" {
        if stream_plan_enabled(flags) && stream_data_enabled(flags) {
            bail!("ExpertProtocolV2 request cannot be both a stream plan and stream data frame");
        }
        if stream_final_enabled(flags) && !stream_data_enabled(flags) {
            bail!("ExpertProtocolV2 stream-final flag requires a stream data frame");
        }
        if spark_row_sharded_reduction_enabled(flags) && !spark_reduction_enabled(flags) {
            bail!("ExpertProtocolV2 row-sharded reduction requires Spark reduction");
        }
        let collective_parts = spark_collective_part_count(flags);
        if collective_parts == 1 {
            bail!("ExpertProtocolV2 striped Spark collective requires at least two parts");
        }
        if collective_parts > 0 && !spark_row_sharded_reduction_enabled(flags) {
            bail!("ExpertProtocolV2 striped Spark collective requires row-sharded reduction");
        }
        if precompile_warmup_enabled(flags)
            && (stream_plan_enabled(flags) || stream_data_enabled(flags))
        {
            bail!("ExpertProtocolV2 precompile warmup cannot use streamed ingress frames");
        }
    }
    Ok(())
}

fn validate_header_len(header_len: usize, expected_header_len: usize) -> Result<()> {
    if header_len != expected_header_len {
        bail!(
            "ExpertProtocolV2 header length mismatch: header={} expected={expected_header_len}",
            header_len
        );
    }
    Ok(())
}

fn validate_common_header(bytes: &[u8], expected_kind: u16) -> Result<usize> {
    if &bytes[..8] != MAGIC {
        bail!("invalid ExpertProtocolV2 magic");
    }
    let version = read_u16(bytes, 8, "version")?;
    if version != VERSION {
        bail!("unsupported ExpertProtocolV2 version {version}");
    }
    let kind = read_u16(bytes, 10, "kind")?;
    if kind != expected_kind {
        bail!("unexpected ExpertProtocolV2 message kind {kind}");
    }
    Ok(read_u32(bytes, 12, "header_len")? as usize)
}

fn encode_row_descriptor(out: &mut Vec<u8>, row: &ExpertProtocolV2RowDescriptor) {
    push_u64(out, row.row_id);
    push_u16(out, row.source_kind as u16);
    push_u16(out, 0);
    push_u64(out, row.source_request_id);
    push_u64(out, row.token_position);
    push_u32(out, row.route_offset);
    push_u32(out, row.route_count);
    push_u32(out, 0);
}

fn decode_row_descriptor(bytes: &[u8], offset: usize) -> Result<ExpertProtocolV2RowDescriptor> {
    Ok(ExpertProtocolV2RowDescriptor {
        row_id: read_u64(bytes, offset, "row_id")?,
        source_kind: ExpertV2SourceKind::from_u16(read_u16(bytes, offset + 8, "source_kind")?)?,
        source_request_id: read_u64(bytes, offset + 12, "source_request_id")?,
        token_position: read_u64(bytes, offset + 20, "token_position")?,
        route_offset: read_u32(bytes, offset + 28, "route_offset")?,
        route_count: read_u32(bytes, offset + 32, "route_count")?,
    })
}

fn encode_route_entry(out: &mut Vec<u8>, route: &ExpertProtocolV2RouteEntry) {
    push_u32(out, route.row_index);
    push_u32(out, route.expert_id);
    out.extend_from_slice(&route.gate_weight.to_le_bytes());
}

fn decode_route_entry(bytes: &[u8], offset: usize) -> Result<ExpertProtocolV2RouteEntry> {
    Ok(ExpertProtocolV2RouteEntry {
        row_index: read_u32(bytes, offset, "row_index")?,
        expert_id: read_u32(bytes, offset + 4, "expert_id")?,
        gate_weight: f32::from_le_bytes(
            bytes
                .get(offset + 8..offset + 12)
                .context("reading FP32 gate_weight")?
                .try_into()
                .unwrap(),
        ),
    })
}

fn payload_checksum(
    rows: &[ExpertProtocolV2RowDescriptor],
    routes: &[ExpertProtocolV2RouteEntry],
    hidden_payload: &[u8],
) -> [u8; CHECKSUM_LEN] {
    let mut hasher = Sha256::new();
    let mut scratch = Vec::with_capacity(EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN);
    for row in rows {
        scratch.clear();
        encode_row_descriptor(&mut scratch, row);
        hasher.update(&scratch);
    }
    scratch.reserve(EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN);
    for route in routes {
        scratch.clear();
        encode_route_entry(&mut scratch, route);
        hasher.update(&scratch);
    }
    hasher.update(hidden_payload);
    hasher.finalize().into()
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize, field: &str) -> Result<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .with_context(|| format!("reading {field}"))?
            .try_into()
            .unwrap(),
    ))
}

fn read_u32(bytes: &[u8], offset: usize, field: &str) -> Result<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .with_context(|| format!("reading {field}"))?
            .try_into()
            .unwrap(),
    ))
}

fn read_u64(bytes: &[u8], offset: usize, field: &str) -> Result<u64> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .with_context(|| format!("reading {field}"))?
            .try_into()
            .unwrap(),
    ))
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
    value
        .try_into()
        .with_context(|| format!("{label} exceeds u32"))
}

fn checked_usize(value: u64, label: &str) -> Result<usize> {
    value
        .try_into()
        .with_context(|| format!("{label} exceeds usize"))
}

#[cfg(test)]
mod tests;
