//! Complete native backbone batches carried by revision-3 frames.
//! A wire output row is one BF16 [5120] rank partial, after local route summation.
use crate::{
    ExpertProtocolV2RequestView, ExpertProtocolV2ResponseHeader, ExpertProtocolV2ResponseRef,
    ExpertProtocolV2ResponseView, ExpertProtocolV2Status, ExpertV2Dtype,
    EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM,
};
use anyhow::{ensure, Context, Result};

pub use crate::protocol_v2::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;

mod chunks;
pub use chunks::V41Tp4ChunkReceiver;
mod roce;
pub use roce::{V41Tp4Roce, V41Tp4RocePending};
mod tcp;
pub use tcp::{V41Tp4Pending, V41Tp4Tcp};
mod paired;
pub use paired::{V41PairedOwnershipBatch, V41PairedRouteWord, V41_EXL3_PAIRED_REQUEST_FLAG};
mod native_group;
pub use native_group::{
    V41NativeOwnerRouteWord, V41NativeOwnershipBatch, V41SparkTopology, V41_MAX_NATIVE_GROUPS,
    V41_NATIVE_GROUP_REQUEST_FLAG, V41_NATIVE_INACTIVE_OWNER, V41_NATIVE_UNASSIGNED_EXPERT_ID,
    V41_ROUTED_EXPERTS,
};

pub const V41_HIDDEN: u32 = 5120;
pub const V41_BACKBONE_TOPK: u32 = 6;
pub const V41_PARTIAL_ROW_BYTES: u32 = V41_HIDDEN * 2;

/// Bind native Spark responses to their tensor-parallel topology without a wire
/// ABI change. TP4 retains executor IDs 1..=4; TP2 uses the disjoint namespace
/// 5..=6, so a two-peer coordinator rejects stale TP4 rank-0/rank-1 workers.
/// Three-rank groups — the implicit single-RTX EXL3 compact profile — use the
/// `TP3EP1` namespace 7..=9, exactly as the two-rank compact group already
/// shares the `TP2EP1` namespace. This identifies topology and rank, not
/// checkpoint or deployment identity.
///
/// This helper covers only the implicit worlds that predate [`V41SparkTopology`].
/// Every explicit layout — including pure `TP6EP1` — must use
/// [`V41SparkTopology::executor_id`], which owns the wider disjoint
/// namespaces and rejects a legacy identity.
pub fn v41_spark_executor_id(world: usize, rank: usize) -> Result<u64> {
    let base = match world {
        4 => 1,
        2 => 5,
        3 => 7,
        _ => anyhow::bail!(
            "native Spark executor requires an implicit world of 2, 3 or 4; \
             an explicit topology must use V41SparkTopology::executor_id"
        ),
    };
    ensure!(rank < world, "native Spark executor rank {rank} is outside world {world}");
    Ok(rank as u64 + base)
}

// Shared by wire parsing on workers and validation of owned coordinator requests.
fn validate_canonical_body(
    header: &crate::ExpertProtocolV2RequestHeader,
    max_rows: u32,
    row_at: impl Fn(usize) -> Result<crate::ExpertProtocolV2RowDescriptor>,
    mut route_at: impl FnMut(usize) -> Result<crate::ExpertProtocolV2RouteEntry>,
) -> Result<()> {
    ensure!(
        header.row_count > 0 && header.row_count <= max_rows,
        "native batch exceeds admitted row capacity"
    );
    ensure!(header.layer_id < 40, "native backbone layer out of range");
    ensure!(
        header.hidden_dim == V41_HIDDEN
            && matches!(
                header.hidden_dtype,
                ExpertV2Dtype::Bf16 | ExpertV2Dtype::Fp8E4m3Ue8m0K32
            )
            && header.hidden_row_stride_bytes as usize
                == header.hidden_dtype.row_bytes(V41_HIDDEN as usize)?,
        "native backbone needs contiguous BF16 or E4M3/UE8M0 K32 hidden rows"
    );
    ensure!(
        header.route_count
            == header
                .row_count
                .checked_mul(6)
                .context("route count overflow")?,
        "native backbone needs six routes per token"
    );
    for row_index in 0..header.row_count {
        let row = row_at(row_index as usize)?;
        ensure!(
            row.route_offset == row_index * 6 && row.route_count == 6,
            "noncanonical native row route span"
        );
        let mut ids = [u32::MAX; 6];
        for slot in 0..6usize {
            let route = route_at(row_index as usize * 6 + slot)?;
            ensure!(
                route.row_index == row_index && route.expert_id < 384,
                "invalid native expert route"
            );
            ensure!(
                route.gate_weight.is_finite() && route.gate_weight >= 0.0,
                "invalid native FP32 routing weight"
            );
            ensure!(
                !ids[..slot].contains(&route.expert_id),
                "duplicate native expert route in one row"
            );
            ids[slot] = route.expert_id;
        }
    }
    Ok(())
}

fn validate_canonical(
    header: &crate::ExpertProtocolV2RequestHeader,
    max_rows: u32,
    row_at: impl Fn(usize) -> Result<crate::ExpertProtocolV2RowDescriptor>,
    route_at: impl FnMut(usize) -> Result<crate::ExpertProtocolV2RouteEntry>,
) -> Result<()> {
    ensure!(
        header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0,
        "native request requires compact BF16 response agreement"
    );
    ensure!(
        header.flags
            & !(EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
                | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16)
            == 0,
        "native complete batches cannot use legacy reduction/compression or stream flags"
    );
    validate_canonical_body(header, max_rows, row_at, route_at)
}

/// Ownership-aware canonical admission for a worker bound to one replicated
/// `TP×EP` topology. The flag and the topology's group count are both required,
/// every active expert must keep one owner across the whole batch, and only a
/// native compact batch may use the contract.
fn validate_native_group(
    header: &crate::ExpertProtocolV2RequestHeader,
    max_rows: u32,
    topology: V41SparkTopology,
    row_at: impl Fn(usize) -> Result<crate::ExpertProtocolV2RowDescriptor>,
    mut route_at: impl FnMut(usize) -> Result<crate::ExpertProtocolV2RouteEntry>,
) -> Result<()> {
    ensure!(
        header.flags & V41_NATIVE_GROUP_REQUEST_FLAG != 0,
        "native group worker requires the native group request flag"
    );
    ensure!(
        header.flags & V41_EXL3_PAIRED_REQUEST_FLAG == 0,
        "native group requests cannot use paired EXL3 admission"
    );
    ensure!(
        header.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0,
        "native group request requires compact BF16 response agreement"
    );
    ensure!(
        header.flags
            & !(EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
                | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16
                | V41_NATIVE_GROUP_REQUEST_FLAG)
            == 0,
        "native group requests cannot use legacy reduction/compression or stream flags"
    );
    let mut batch = V41NativeOwnershipBatch::default();
    validate_canonical_body(header, max_rows, row_at, |index| {
        let route = route_at(index)?;
        let decoded = batch.observe(route.expert_id, topology.group_count())?;
        Ok(crate::ExpertProtocolV2RouteEntry {
            expert_id: decoded.expert_id,
            ..route
        })
    })
}

fn validate_paired(
    header: &crate::ExpertProtocolV2RequestHeader, max_rows: u32,
    row_at: impl Fn(usize) -> Result<crate::ExpertProtocolV2RowDescriptor>,
    route_at: impl Fn(usize) -> Result<crate::ExpertProtocolV2RouteEntry>,
) -> Result<()> {
    ensure!(header.flags & V41_EXL3_PAIRED_REQUEST_FLAG != 0, "paired worker requires paired request");
    let mut canonical = header.clone();
    canonical.flags &= !V41_EXL3_PAIRED_REQUEST_FLAG;
    let mut batch = V41PairedOwnershipBatch::default();
    validate_canonical(&canonical, max_rows, row_at, |index| {
        let mut route = route_at(index)?;
        route.expert_id = batch.observe(route.expert_id)?.expert_id;
        Ok(route)
    })
}

/// Validated canonical row-major routing; the same request must reach every TP rank.
pub struct V41BackboneRequest<'a> {
    view: ExpertProtocolV2RequestView<'a>,
    /// Present only for a request admitted under the native replicated-group
    /// contract, which binds the group count used to decode owner bits.
    native_topology: Option<V41SparkTopology>,
}
impl<'a> V41BackboneRequest<'a> {
    pub fn parse(frame: &'a [u8], max_rows: u32) -> Result<Self> {
        let view = ExpertProtocolV2RequestView::parse(frame)?;
        validate_canonical(&view.header, max_rows, |i| view.row(i), |i| view.route(i))?;
        Ok(Self { view, native_topology: None })
    }
    /// Explicit admission for a worker bound to paired EXL3 resident weights.
    pub fn parse_paired(frame: &'a [u8], max_rows: u32) -> Result<Self> {
        let view = ExpertProtocolV2RequestView::parse(frame)?;
        validate_paired(&view.header, max_rows, |i| view.row(i), |i| view.route(i))?;
        Ok(Self { view, native_topology: None })
    }
    /// Explicit admission for a worker bound to one replicated `TP×EP`
    /// topology. All ranks receive the same encoded owner assignment; each rank
    /// unpacks only its own group's routes.
    pub fn parse_native_group(
        frame: &'a [u8],
        max_rows: u32,
        topology: V41SparkTopology,
    ) -> Result<Self> {
        let view = ExpertProtocolV2RequestView::parse(frame)?;
        validate_native_group(&view.header, max_rows, topology, |i| view.row(i), |i| view.route(i))?;
        Ok(Self { view, native_topology: Some(topology) })
    }
    pub fn is_paired(&self) -> bool { self.view.header.flags & V41_EXL3_PAIRED_REQUEST_FLAG != 0 }
    pub fn is_native_group(&self) -> bool { self.native_topology.is_some() }
    /// Topology this request was admitted under, if it carries native ownership.
    pub fn native_topology(&self) -> Option<V41SparkTopology> { self.native_topology }

    pub fn validate_owned_paired(request: &crate::ExpertProtocolV2Request, max_rows: u32) -> Result<()> {
        request.validate()?;
        validate_paired(&request.header, max_rows,
            |i| Ok(request.rows[i].clone()), |i| Ok(request.routes[i].clone()))
    }
    /// Ownership-aware check of a locally owned native group request; the caller
    /// must supply the topology its workers are bound to.
    pub fn validate_owned_native_group(
        request: &crate::ExpertProtocolV2Request,
        max_rows: u32,
        topology: V41SparkTopology,
    ) -> Result<()> {
        request.validate()?;
        validate_native_group(&request.header, max_rows, topology,
            |i| Ok(request.rows[i].clone()), |i| Ok(request.routes[i].clone()))
    }
    /// Check a locally owned request using the worker's canonical contract,
    /// without serializing or copying its activation payload.
    pub fn validate_owned(request: &crate::ExpertProtocolV2Request, max_rows: u32) -> Result<()> {
        request.validate()?;
        validate_canonical(&request.header, max_rows,
            |i| Ok(request.rows[i].clone()), |i| Ok(request.routes[i].clone()))
    }
    pub fn rows(&self) -> u32 {
        self.view.header.row_count
    }
    pub fn layer(&self) -> u32 {
        self.view.header.layer_id
    }
    /// Require the representation advertised by the bound native kernel before
    /// copying bytes into its input allocation. Wire parsing alone cannot do this.
    pub fn require_input_dtype(&self, native_dtype: u32) -> Result<()> {
        ensure!(
            self.view.header.hidden_dtype as u32 == native_dtype,
            "request input representation does not match native expert kernel"
        );
        Ok(())
    }
    pub fn hidden(&self) -> &'a [u8] {
        self.view.hidden_payload()
    }
    pub fn plane_bytes(&self) -> Result<usize> {
        (self.rows() as usize)
            .checked_mul(V41_PARTIAL_ROW_BYTES as usize)
            .context("route plane byte overflow")
    }
    /// Fill caller-owned GPU-upload arrays without reordering or rounding routes.
    pub fn copy_routes_into(&self, ids: &mut [i32], weights: &mut [f32]) -> Result<()> {
        ensure!(!self.is_paired(), "paired routes require ownership-aware unpacking");
        ensure!(!self.is_native_group(), "native group routes require ownership-aware unpacking");
        let count = self.view.header.route_count as usize;
        ensure!(
            ids.len() >= count && weights.len() >= count,
            "native routing buffers are too short"
        );
        for index in 0..count {
            let route = self.view.route(index)?;
            ids[index] = route.expert_id as i32;
            weights[index] = route.gate_weight;
        }
        Ok(())
    }
    pub fn copy_paired_routes_into(&self, ids: &mut [i32], weights: &mut [f32], rank: usize, ownership: &mut [i32]) -> Result<()> {
        ensure!(self.is_paired(), "disjoint routes cannot use paired unpacking");
        let count = self.view.header.route_count as usize;
        ensure!(ids.len() >= count && weights.len() >= count, "paired routing buffers too short");
        let mut batch = V41PairedOwnershipBatch::default();
        for index in 0..count {
            let route = self.view.route(index)?;
            ids[index] = batch.observe(route.expert_id)?.expert_id as i32;
            weights[index] = route.gate_weight;
        }
        batch.write_local_ownership(rank, ownership)
    }
    /// Unpack one replicated group's routes for a worker bound to `topology`.
    ///
    /// A route owned by `group` keeps its true expert id and exact FP32 gate
    /// weight; every other route receives [`V41_NATIVE_UNASSIGNED_EXPERT_ID`]
    /// and a zero weight, which the existing kernel reduces to zero through its
    /// invalid-id inverse `-1` path. The batch-wide ownership check rejects a
    /// request whose encoded owners conflict, so a route can never be counted
    /// twice. Buffers are reused; no allocation occurs on success.
    pub fn copy_native_group_routes_into(
        &self,
        ids: &mut [i32],
        weights: &mut [f32],
        group: u8,
    ) -> Result<()> {
        let topology = self
            .native_topology
            .context("native group routes require a topology-admitted request")?;
        ensure!(group < topology.group_count(), "native group index out of range");
        let count = self.view.header.route_count as usize;
        ensure!(
            ids.len() >= count && weights.len() >= count,
            "native group routing buffers are too short"
        );
        let mut batch = V41NativeOwnershipBatch::default();
        for index in 0..count {
            let route = self.view.route(index)?;
            let decoded = batch.observe(route.expert_id, topology.group_count())?;
            if decoded.owner == group {
                ids[index] = decoded.expert_id as i32;
                weights[index] = route.gate_weight;
            } else {
                ids[index] = V41_NATIVE_UNASSIGNED_EXPERT_ID;
                weights[index] = 0.0;
            }
        }
        Ok(())
    }
    fn response_header(&self, executor_id: u64) -> Result<ExpertProtocolV2ResponseHeader> {
        ensure!(executor_id != 0, "native response needs an executor identity");
        let header = &self.view.header;
        Ok(ExpertProtocolV2ResponseHeader {
                request_id: header.request_id,
                placement_version: header.placement_version,
                layer_id: header.layer_id,
                row_count: self.rows(),
                output_dim: V41_HIDDEN,
                output_dtype: ExpertV2Dtype::Bf16,
                output_row_stride_bytes: V41_PARTIAL_ROW_BYTES,
                output_payload_bytes: self.plane_bytes()? as u64,
                status: ExpertProtocolV2Status::Ok,
                // Request-only admission flags never appear in a response.
                flags: header.flags
                    & !(V41_EXL3_PAIRED_REQUEST_FLAG | V41_NATIVE_GROUP_REQUEST_FLAG),
                executor_id,
            })
    }
    pub fn response_device(&self, executor_id: u64, output: ds41rt_ffi::Ds41rtDeviceBuffer)
        -> Result<crate::ExpertProtocolV2DeviceResponseRef<'static>> {
        let response = crate::ExpertProtocolV2DeviceResponseRef {
            header: self.response_header(executor_id)?, row_indices: None,
            partial_output_payload: output,
        };
        response.validate()?;
        Ok(response)
    }
    pub fn permits_device_response(&self) -> bool {
        self.view.header.flags & EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM == 0
    }
    pub fn response<'p>(
        &self,
        executor_id: u64,
        partials: &'p [u8],
    ) -> Result<ExpertProtocolV2ResponseRef<'p>> {
        ensure!(
            executor_id != 0,
            "native response needs an executor identity"
        );
        ensure!(
            partials.len() == self.plane_bytes()?,
            "native route plane extent mismatch"
        );
        let response = ExpertProtocolV2ResponseRef {
            header: self.response_header(executor_id)?,
            row_indices: None,
            partial_output_payload: partials,
        };
        response.validate()?;
        Ok(response)
    }
}

// Fixed-size rank identities keep per-wave validation allocation-free.
enum V41Executors {
    Tp2([u64; 2]),
    Tp3([u64; 3]),
    Tp4([u64; 4]),
    Tp6([u64; 6]),
}
impl V41Executors {
    fn new(executors: &[u64]) -> Result<Self> {
        Ok(match executors.len() {
            2 => Self::Tp2(executors.try_into().expect("two executors")),
            3 => Self::Tp3(executors.try_into().expect("three executors")),
            4 => Self::Tp4(executors.try_into().expect("four executors")),
            6 => Self::Tp6(executors.try_into().expect("six executors")),
            other => anyhow::bail!(
                "native TP/EP requires two, three, four or six executors, got {other}"
            ),
        })
    }
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Tp2(ids) => ids,
            Self::Tp3(ids) => ids,
            Self::Tp4(ids) => ids,
            Self::Tp6(ids) => ids,
        }
    }
    fn len(&self) -> usize { self.as_slice().len() }
}

/// Collects complete native rank planes in rank order, independently of arrival
/// order, for every validated world size (2, 3, 4 or 6). Payloads are borrowed;
/// keep their frame storage alive until GPU copies finish. Request IDs must
/// uniquely identify in-flight waves within a placement version.
pub struct V41Tp4Planes<'a> {
    request_id: u64,
    placement_version: u64,
    layer: u32,
    rows: u32,
    executors: V41Executors,
    planes: [Option<&'a [u8]>; 6],
}
impl<'a> V41Tp4Planes<'a> {
    pub fn new(request: &V41BackboneRequest<'_>, executors: [u64; 4]) -> Result<Self> {
        Self::from_header(&request.view.header, executors)
    }
    /// Generic constructor for the validated physical rank counts 2, 3, 4 and 6.
    pub fn new_ranks(request: &V41BackboneRequest<'_>, executors: &[u64]) -> Result<Self> {
        if let Some(topology) = request.native_topology() {
            ensure!(
                executors.len() == topology.world_size(),
                "native group executor count does not match its topology"
            );
        }
        Self::from_header_ranks(&request.view.header, executors)
    }
    fn from_header(h: &crate::ExpertProtocolV2RequestHeader, executors: [u64; 4]) -> Result<Self> {
        Self::from_header_ranks(h, &executors)
    }
    fn from_header_ranks(h: &crate::ExpertProtocolV2RequestHeader, executors: &[u64]) -> Result<Self> {
        ensure!(
            matches!(executors.len(), 2 | 3 | 4 | 6),
            "native TP/EP requires two, three, four or six executors"
        );
        for (rank, id) in executors.iter().enumerate() {
            ensure!(
                *id != 0 && !executors[..rank].contains(id),
                "native TP requires distinct nonzero executor identities"
            );
        }
        Ok(Self {
            request_id: h.request_id,
            placement_version: h.placement_version,
            layer: h.layer_id,
            rows: h.row_count,
            executors: V41Executors::new(executors)?,
            planes: [None; 6],
        })
    }
    /// Rejections leave the collection unchanged.
    pub fn insert(&mut self, frame: &'a [u8]) -> Result<usize> {
        let response = ExpertProtocolV2ResponseView::parse(frame)?;
        let h = &response.header;
        let rank = self.response_rank(h)?;
        ensure!(
            h.flags
                & !(EXPERT_PROTOCOL_V2_FLAG_DEBUG_CHECKSUM
                    | EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16)
                == 0,
            "native route planes must be complete and unindexed"
        );
        ensure!(
            h.row_count == self.rows,
            "native TP route plane row count mismatch"
        );
        ensure!(self.planes[rank].is_none(), "duplicate native TP response");
        self.planes[rank] = Some(response.partial_output_payload());
        Ok(rank)
    }
    fn response_rank(&self, h: &ExpertProtocolV2ResponseHeader) -> Result<usize> {
        ensure!(
            h.flags & EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16 != 0,
            "native response lacks compact BF16 agreement"
        );
        ensure!(
            h.request_id == self.request_id
                && h.placement_version == self.placement_version
                && h.layer_id == self.layer,
            "stale or mismatched native TP response"
        );
        ensure!(
            h.status == ExpertProtocolV2Status::Ok,
            "native expert execution failed"
        );
        ensure!(
            h.output_dim == V41_HIDDEN
                && h.output_dtype == ExpertV2Dtype::Bf16
                && h.output_row_stride_bytes == V41_PARTIAL_ROW_BYTES,
            "native TP route plane geometry mismatch"
        );
        self.executors
            .as_slice()
            .iter()
            .position(|id| *id == h.executor_id)
            .context("unknown native TP executor")
    }
    /// Physical ranks represented by this collection.
    pub fn world_size(&self) -> usize {
        self.executors.len()
    }
    pub fn complete(&self) -> bool {
        self.planes[..self.executors.len()].iter().all(Option::is_some)
    }
    /// Legacy four-plane accessor. Worlds 2, 3 and 6 use [`Self::plane`].
    pub fn planes(&self) -> Result<[&'a [u8]; 4]> {
        ensure!(
            self.world_size() == 4,
            "four-plane accessor requires exactly four executors"
        );
        ensure!(self.complete(), "native TP response set is incomplete");
        Ok([
            self.planes[0].expect("complete native TP rank plane"),
            self.planes[1].expect("complete native TP rank plane"),
            self.planes[2].expect("complete native TP rank plane"),
            self.planes[3].expect("complete native TP rank plane"),
        ])
    }
    /// One complete rank plane, for any validated world size.
    pub fn plane(&self, rank: usize) -> Result<&'a [u8]> {
        ensure!(rank < self.world_size(), "native TP rank out of range");
        self.planes[rank].context("native TP rank plane is not complete")
    }
}

#[cfg(test)]
mod tcp_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
        ExpertV2SourceKind, EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
    };
    #[test]
    fn spark_executor_namespaces_reject_other_world_responses() -> Result<()> {
        let tp4 = [0, 1, 2, 3].map(|rank| v41_spark_executor_id(4, rank).unwrap());
        let tp2 = [0, 1].map(|rank| v41_spark_executor_id(2, rank).unwrap());
        // The implicit three-rank EXL3 compact group answers in the TP3EP1
        // namespace, exactly as the two-rank group shares TP2EP1's 5..=6.
        let tp3 = [0, 1, 2].map(|rank| v41_spark_executor_id(3, rank).unwrap());
        assert_eq!(tp4, [1, 2, 3, 4]);
        assert_eq!(tp2, [5, 6]);
        assert_eq!(tp3, [7, 8, 9]);
        assert_eq!(tp3.to_vec(), V41SparkTopology::new(3, 1)?.executor_ids());
        // Every implicit namespace stays disjoint from the other worlds'.
        for (world, rank) in [
            (0, 0),
            (1, 0),
            // Explicit layouts (e.g. pure TP6EP1) must go through the topology
            // namespace, never this implicit two/three/four-rank helper.
            (6, 0),
            (6, 5),
            (2, 2),
            (3, 3),
            (4, 4),
            (usize::MAX, 0),
            (2, usize::MAX),
        ] {
            assert!(v41_spark_executor_id(world, rank).is_err());
        }
        let frame = request(1).encode()?;
        let native = V41BackboneRequest::parse(&frame, 1)?;
        let payload = vec![0; V41_PARTIAL_ROW_BYTES as usize];
        let mut two = V41Tp4ChunkReceiver::new_tp2(&native, tp2, 200_000)?;
        let mut four = V41Tp4ChunkReceiver::new(&native, tp4, 200_000)?;
        for id in tp4 {
            let response = native.response(id, &payload)?.to_owned()?.encode()?;
            assert!(two.push(&response, |_, _, _| panic!("TP4 response reached TP2 sink")).is_err());
        }
        for (rank, id) in tp2.into_iter().enumerate() {
            let response = native.response(id, &payload)?.to_owned()?.encode()?;
            assert!(four.push(&response, |_, _, _| panic!("TP2 response reached TP4 sink")).is_err());
            two.push(&response, |actual, start, _| {
                assert_eq!((actual, start), (rank, 0)); Ok(())
            })?;
            assert_eq!(two.complete(), rank == 1);
        }
        assert_eq!(four.received_rows(), [0; 4]);
        Ok(())
    }

    pub(super) fn request(rows: u32) -> ExpertProtocolV2Request {
        let mut request = ExpertProtocolV2Request::new(
            91,
            17,
            39,
            5120,
            ExpertV2Dtype::Bf16,
            (0..rows)
                .map(|r| ExpertProtocolV2RowDescriptor {
                    row_id: r as u64,
                    source_kind: ExpertV2SourceKind::Decode,
                    source_request_id: 100 + r as u64,
                    token_position: 7,
                    route_offset: r * 6,
                    route_count: 6,
                })
                .collect(),
            (0..rows * 6)
                .map(|r| ExpertProtocolV2RouteEntry {
                    row_index: r / 6,
                    expert_id: (r % 6) * 63,
                    gate_weight: if r == 0 { f32::from_bits(1) } else { 0.1234567 },
                })
                .collect(),
            vec![0; rows as usize * 5120 * 2],
        )
        .unwrap();
        request.header.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        request
    }
    #[test]
    fn fp8_k32_wire_roundtrip_preserves_rows_and_bf16_worker_rejects_it() {
        for rows in [1, 16, 80] {
            let base = request(rows);
            let mut payload = Vec::new();
            for row in 0..rows {
                // Distinguish payload from scales and adjacent row boundaries.
                payload.extend((0..5120).map(|i| ((i + row * 7) % 127) as u8));
                payload.extend((0..160).map(|i| (105 + (i + row) % 20) as u8));
            }
            let mut input = ExpertProtocolV2Request::new(
                base.header.request_id,
                base.header.placement_version,
                base.header.layer_id,
                5120,
                ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                base.rows,
                base.routes,
                payload.clone(),
            )
            .unwrap();
            input.header.flags = base.header.flags;
            let frame = input.with_debug_checksum().encode().unwrap();
            let owned = ExpertProtocolV2Request::decode(&frame).unwrap();
            let view = ExpertProtocolV2RequestView::parse(&frame).unwrap();
            assert_eq!(owned.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
            assert_eq!(owned.header.hidden_row_stride_bytes, 5280);
            assert_eq!(view.hidden_payload(), payload);
            assert_eq!(owned.hidden_payload.as_ref(), payload);
            let native = V41BackboneRequest::parse(&frame, rows).unwrap();
            assert!(native.require_input_dtype(1).is_err());
            native.require_input_dtype(7).unwrap();
            assert_eq!(native.hidden(), payload);
            assert!(ExpertProtocolV2Request::decode(&frame[..frame.len() - 1]).is_err());
            assert!(ExpertProtocolV2RequestView::parse(&frame[..frame.len() - 1]).is_err());
        }
        let dtype = ExpertV2Dtype::Fp8E4m3Ue8m0K32;
        assert_eq!(dtype.row_bytes(32).unwrap(), 33);
        assert_eq!(dtype.row_bytes(5120).unwrap(), 5280);
        for width in [0, 31, 33, usize::MAX, usize::MAX - 31] {
            assert!(dtype.row_bytes(width).is_err());
        }
    }

    #[test]
    fn native_routes_preserve_order_and_fp32_bits() {
        let owned = request(16).with_debug_checksum();
        let frame = owned.encode().unwrap();
        let native = V41BackboneRequest::parse(&frame, 16).unwrap();
        let (mut ids, mut weights) = (vec![-1; 100], vec![-1.; 100]);
        native.copy_routes_into(&mut ids, &mut weights).unwrap();
        for (i, route) in owned.routes.iter().enumerate() {
            assert_eq!(ids[i], route.expert_id as i32);
            assert_eq!(weights[i].to_bits(), route.gate_weight.to_bits());
        }
        assert_eq!(&ids[96..], &[-1; 4]);
        assert_eq!(&weights[96..], &[-1.; 4]);
        assert_eq!(native.hidden(), owned.hidden_payload.as_ref());
        assert_eq!(native.plane_bytes().unwrap(), 163_840);
        assert!(native
            .copy_routes_into(&mut ids[..95], &mut weights)
            .is_err());
        assert!(V41BackboneRequest::parse(&frame, 15).is_err());
    }
    #[test]
    fn native_request_rejects_invalid_routing_and_legacy_modes() {
        for kind in 0..7 {
            let mut owned = request(1);
            match kind {
                0 => owned.routes[1].expert_id = owned.routes[0].expert_id,
                1 => owned.routes[0].expert_id = 384,
                2 => owned.routes[0].gate_weight = -0.5,
                3 => owned.header.layer_id = 40,
                4 => owned.header.flags |= EXPERT_PROTOCOL_V2_FLAG_RESPONSE_FP8_E4M3_ROW_SCALED,
                6 => owned.header.flags = 0,
                _ => {
                    owned.routes.pop();
                    owned.rows[0].route_count = 5;
                    owned.header.route_count = 5;
                    owned.header.route_bytes = 60;
                }
            }
            assert!(V41Tp4ChunkReceiver::from_owned(&owned, 16, [11,22,33,44], 1 << 20).is_err());
            match owned.encode() {
                Ok(frame) => assert!(V41BackboneRequest::parse(&frame, 16).is_err()),
                Err(_) => {}
            }
        }
    }
    #[test]
    fn owned_receiver_validates_capacity_extent_and_response_identity() {
        let executors = [11,22,33,44];
        let budget = 64 << 20;
        for rows in [1,80,1024,4096] {
            let owned = request(rows);
            let frame = owned.encode().unwrap();
            let native = V41BackboneRequest::parse(&frame, 4096).unwrap();
            let mut receiver = V41Tp4ChunkReceiver::from_owned(&owned, 4096, executors, budget).unwrap();
            assert!(V41Tp4ChunkReceiver::from_owned(&owned, rows-1, executors, budget).is_err());
            assert!(V41Tp4ChunkReceiver::from_owned(&owned, 4096, executors, frame.len()-1).is_err());
            assert!(V41Tp4ChunkReceiver::from_owned(&owned, 4096, [11;4], budget).is_err());
            let payload = vec![0; native.plane_bytes().unwrap()];
            for rank in [3,0,2,1] {
                let response = native.response(executors[rank], &payload).unwrap().to_owned().unwrap().encode().unwrap();
                receiver.push(&response, |r, start, bytes| {
                    assert_eq!((r,start,bytes.len()), (rank,0,payload.len())); Ok(())
                }).unwrap();
            }
            assert!(receiver.complete());
            let mut malformed = owned.clone();
            malformed.hidden_payload = bytes::Bytes::new();
            assert!(V41Tp4ChunkReceiver::from_owned(&malformed, 4096, executors, budget).is_err());
        }
    }
    #[test]
    fn tp4_assembly_checks_identity_geometry_duplicates_and_arrival_order() {
        let frame = request(1).encode().unwrap();
        let native = V41BackboneRequest::parse(&frame, 16).unwrap();
        let executors = [11, 22, 33, 44];
        assert!(V41Tp4Planes::new(&native, [11, 11, 33, 44]).is_err());
        assert!(V41Tp4Planes::new(&native, [0, 22, 33, 44]).is_err());
        let payloads: Vec<_> = (0..4)
            .map(|rank| vec![rank as u8 + 1; native.plane_bytes().unwrap()])
            .collect();
        let responses: Vec<_> = (0..4)
            .map(|rank| {
                native
                    .response(executors[rank], &payloads[rank])
                    .unwrap()
                    .to_owned()
                    .unwrap()
                    .encode()
                    .unwrap()
            })
            .collect();
        let mut collector = V41Tp4Planes::new(&native, executors).unwrap();
        assert!(collector.planes().is_err());
        for rank in [3, 1, 0, 2] {
            assert_eq!(collector.insert(&responses[rank]).unwrap(), rank);
            assert!(collector.insert(&responses[rank]).is_err());
        }
        for (plane, expected) in collector.planes().unwrap().into_iter().zip(&payloads) {
            assert_eq!(plane, expected);
        }
        for kind in 0..7 {
            let mut bad = native
                .response(11, &payloads[0])
                .unwrap()
                .to_owned()
                .unwrap();
            match kind {
                0 => bad.header.request_id += 1,
                1 => bad.header.placement_version += 1,
                2 => bad.header.layer_id -= 1,
                3 => bad.header.executor_id = 99,
                4 => {
                    bad.header.output_dtype = ExpertV2Dtype::F16;
                }
                6 => bad.header.flags = 0,
                _ => {
                    bad.header.output_dim = 2560;
                }
            }
            let frame = bad.encode().unwrap();
            let mut c = V41Tp4Planes::new(&native, executors).unwrap();
            assert!(c.insert(&frame).is_err());
            assert!(!c.complete());
            assert_eq!(c.insert(&responses[0]).unwrap(), 0);
        }
    }
}
