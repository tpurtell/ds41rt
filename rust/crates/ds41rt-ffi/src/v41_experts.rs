//! Native V4.1 expert AOT launch handles; weight/scratch ownership stays with the caller.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;
use std::ptr::NonNull;

pub const V41_EXPERT_POINTER_COUNT: usize = 44;

/// Native slot 41 representation; dtype and route axis are independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V41ExpertOutputKind {
    Fp32Routes,
    Fp32Tokens,
    Bf16Routes,
}
impl V41ExpertOutputKind {
    fn from_native(kind: u32) -> Result<Self> {
        match kind {
            0 => Ok(Self::Fp32Routes),
            1 => Ok(Self::Fp32Tokens),
            2 => Ok(Self::Bf16Routes),
            _ => anyhow::bail!("unsupported V4.1 expert output kind {kind}"),
        }
    }
    pub fn row_bytes(self, topk: u32) -> usize {
        5120 * match self {
            Self::Fp32Routes => topk as usize * 4,
            Self::Fp32Tokens => 4,
            Self::Bf16Routes => topk as usize * 2,
        }
    }
}

/// Pointer order is checked against generated C headers by the AOT exporter.
#[repr(usize)]
#[derive(Debug, Clone, Copy)]
pub enum V41ExpertPointer {
    Hidden = 0,
    RouteIds,
    RouteWeights,
    PackedA,
    Sfa,
    PackedAStorage,
    ScaleStorage,
    Intermediate,
    BarrierCount,
    BarrierEpoch,
    PairHead,
    ProducersDoneCount,
    AllWorkPublished,
    TaskHead,
    TaskTail,
    TaskReady,
    TaskExpert,
    TaskMTile,
    TaskSliceBegin,
    TaskSliceCount,
    TaskValidRows,
    TileWriteCount,
    WeightW13,
    ScaleW13,
    WeightDown,
    ScaleDown,
    ScaleW13Mx,
    ScaleDownMx,
    ResidualW13,
    ResidualDown,
    W13Repacked,
    W13ScaleRepacked,
    DownRepacked,
    DownScaleRepacked,
    RowCounts,
    ExpertWriteRows,
    ExpertTileBase,
    InputGlobalScale,
    Alpha,
    DownAlpha,
    GlobalScale,
    RoutePartials,
    TokenMap,
    TokenWeights,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct V41ExpertInfo {
    pub abi_version: u32,
    pub role: u32,
    pub experts: u32,
    pub hidden_size: u32,
    pub logical_intermediate: u32,
    pub kernel_intermediate: u32,
    pub topk: u32,
    pub capacity_rows: u32,
    pub scratch_bytes: u64,
    pub max_rows: i32,
    pub rows_padded: i32,
    pub max_tasks: i32,
    pub max_phys_tiles: i32,
    pub max_active_clusters: i32,
    /// ABI 2: 1 for BF16, 7 for row E4M3 payload followed by UE8M0 K32 scales.
    pub input_dtype: u32,
}

impl V41ExpertInfo {
    pub fn input_row_bytes(&self) -> Result<usize> {
        let hidden = self.hidden_size as usize;
        match self.input_dtype {
            1 => hidden.checked_mul(2).context("expert BF16 row overflow"),
            7 if hidden > 0 && hidden % 32 == 0 => hidden
                .checked_add(hidden / 32)
                .context("expert FP8 K32 row overflow"),
            _ => anyhow::bail!("unsupported expert input dtype or width"),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct V41ExpertLaunchArgs {
    pub tensors: [*mut c_void; V41_EXPERT_POINTER_COUNT],
    pub num_tokens: i32,
    pub max_rows: i32,
    pub scatter_rows: i32,
    pub rows_padded: i32,
    pub max_tasks: i32,
    pub max_phys_tiles: i32,
    pub max_active_clusters: i32,
    pub stream: *mut c_void,
}

impl V41ExpertLaunchArgs {
    pub fn new(
        info: &V41ExpertInfo,
        tensors: [*mut c_void; V41_EXPERT_POINTER_COUNT],
        rows: u32,
        stream: *mut c_void,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= info.capacity_rows,
            "expert launch exceeds planned token capacity"
        );
        ensure!(
            tensors.iter().all(|pointer| !pointer.is_null()),
            "expert launch contains a null tensor slot"
        );
        Ok(Self {
            tensors,
            num_tokens: i32::try_from(rows)?,
            max_rows: info.max_rows,
            scatter_rows: i32::try_from(
                rows.checked_mul(info.topk)
                    .context("route count overflow")?,
            )?,
            rows_padded: info.rows_padded,
            max_tasks: info.max_tasks,
            max_phys_tiles: info.max_phys_tiles,
            max_active_clusters: info.max_active_clusters,
            stream,
        })
    }
}

type InfoFn = unsafe extern "C" fn(i32, *mut V41ExpertInfo) -> i32;
type InitializeFn = unsafe extern "C" fn(i32, *mut *mut c_void) -> i32;
type LaunchFn = unsafe extern "C" fn(*mut c_void, *const V41ExpertLaunchArgs) -> i32;
type PackedSizesFn = unsafe extern "C" fn(u32, *mut u64) -> i32;
type PackFn = unsafe extern "C" fn(*const *const u8, *const *mut u8, u32, *mut c_void) -> i32;

/// Native packer accept-list. Every accepted extent is a multiple of 32 so the
/// K/32 UE8M0 scale axis is exact; 576 (Spark TP4) is storage-padded to 640.
/// 384 (pure `TP6EP1`) needs no padding: it is already 128-aligned.
pub fn v41_pack_intermediate_supported(intermediate: u32) -> bool {
    matches!(intermediate, 384 | 576 | 768 | 1152 | 2304)
}

/// Physical-rank counts the compact reduction contract defines: legacy 2 and 4
/// plus replicated-group 3 and 6. Library-independent; use
/// [`V41CompactReducer::supports_rank_count`] for what the loaded library can
/// actually execute.
pub fn v41_rank_count_supported(ranks: u32) -> bool {
    matches!(ranks, 2 | 3 | 4 | 6)
}

/// Official geometry for one `(family, role)` pair. `kernel_intermediate`
/// distinguishes the padded and exact Spark TP4 AOT exports.
fn expected_expert_geometry(
    nvfp4: bool,
    role: u32,
    kernel_intermediate: u32,
) -> Option<(u32, u32, u32, u32)> {
    match (nvfp4, role) {
        // Both exact source layout and zero-padded Spark AOTs are valid.
        (true, 1) if kernel_intermediate == 640 => Some((384, 576, 640, 6)),
        (true, 1) => Some((384, 576, 576, 6)),
        (true, 2) => Some((384, 2304, 2304, 6)),
        (true, 3) => Some((384, 1152, 1152, 6)),
        (false, 0) => Some((128, 2304, 2304, 3)),
        (false, 1) => Some((384, 576, 640, 6)),
        (false, 2) => Some((384, 2304, 2304, 6)),
        (false, 3) => Some((384, 1152, 1152, 6)),
        (false, 4) => Some((128, 1152, 1152, 3)),
        // Replicated-group Spark shards: TP2 has no storage padding, TP3's 768
        // is already 128-aligned.
        (false, 5) => Some((384, 1152, 1152, 6)),
        (false, 6) => Some((384, 768, 768, 6)),
        // Pure TP6EP1: one disjoint 384-column intermediate slice per rank with
        // no storage padding (384 is already 128-aligned).
        (false, 7) => Some((384, 384, 384, 6)),
        _ => None,
    }
}

/// Per-expert checkpoint staging avoids a second full layer of logical weights.
pub struct V41ExpertPacker<'a> {
    _library: &'a NativeLibrary,
    pack: PackFn,
    intermediate: u32,
    bytes: [u64; 4],
}

impl V41ExpertPacker<'_> {
    /// Per-expert byte strides: W13, W13 scales, W2, W2 scales.
    pub fn packed_bytes(&self) -> [u64; 4] {
        self.bytes
    }

    /// # Safety
    /// Sources must be contiguous native CUDA W1,W3,W2,S1,S3,S2 for one expert,
    /// with hidden size 5120 and this packer's logical intermediate size.
    /// Destinations must have the advertised byte sizes and 16-byte alignment,
    /// be mutually disjoint and not overlap any source, on the same device.
    /// All buffers must remain valid through stream completion; order input
    /// copies before this operation and expert execution after it.
    pub unsafe fn pack(
        &self,
        sources: [*const u8; 6],
        destinations: [*mut u8; 4],
        stream: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            (self.pack)(
                sources.as_ptr(),
                destinations.as_ptr(),
                self.intermediate,
                stream,
            )
        };
        ensure!(
            status == 0,
            "V4.1 expert packing failed with CUDA status {status}"
        );
        Ok(())
    }
}
type BindScratchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut *mut c_void) -> i32;
type InitScratchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void) -> i32;
type ReduceFn = unsafe extern "C" fn(
    *const *const f32,
    *const u16,
    *mut u16,
    u32,
    u32,
    u32,
    *mut c_void,
) -> i32;

/// Preloaded, allocation-free reduction entry point for native expert outputs.
pub struct V41RouteReducer<'a> {
    _library: &'a NativeLibrary,
    reduce: ReduceFn,
}

type CompactFn = unsafe extern "C" fn(*const f32, *mut u16, u32, *mut c_void) -> i32;
type CompactBf16RoutesFn = unsafe extern "C" fn(*const u16, *mut u16, u32, *mut c_void) -> i32;
type ReduceTp2Bf16RoutesFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, u32, *mut c_void) -> i32;
type ReduceCompactFn =
    unsafe extern "C" fn(*const *const u16, *const u16, *mut u16, u32, *mut c_void) -> i32;
type ReduceCompactPlanesFn =
    unsafe extern "C" fn(*const *const u16, *const u16, *mut u16, u32, u32, *mut c_void) -> i32;

/// Compact BF16 backbone returns; independent from diagnostic per-route reduction.
pub struct V41CompactReducer<'a> {
    _library: &'a NativeLibrary,
    compact: CompactFn,
    compact_tokens: Option<CompactFn>,
    compact_bf16_routes: Option<CompactBf16RoutesFn>,
    reduce: ReduceCompactFn,
    reduce_tp2: Option<ReduceCompactFn>,
    reduce_planes: Option<ReduceCompactPlanesFn>,
}
impl V41CompactReducer<'_> {
    /// Whether the loaded library can reduce `ranks` physical-rank BF16 planes.
    ///
    /// 4 is always available (the 4-plane entry point is required to construct
    /// this handle); 2 depends on the optional TP2 entry point and 3/6 depend on
    /// the optional N-plane entry point. Check this during admission, before any
    /// allocation or readiness publication, so an older library fails fast
    /// instead of at the first reduction. Any other value is unsupported.
    pub fn supports_rank_count(&self, ranks: u32) -> bool {
        match ranks {
            2 => self.reduce_tp2.is_some(),
            4 => true,
            3 | 6 => self.reduce_planes.is_some(),
            _ => false,
        }
    }

    /// Admission gate for a planned physical-rank count: succeeds only when the
    /// loaded library can execute it, with a descriptive error otherwise.
    pub fn require_rank_count(&self, ranks: u32) -> Result<()> {
        ensure!(
            v41_rank_count_supported(ranks),
            "unsupported physical rank count {ranks}; expected 2, 3, 4 or 6"
        );
        ensure!(
            self.supports_rank_count(ranks),
            "native library cannot reduce {ranks} physical-rank planes; rebuild it with the matching compact reducer entry point"
        );
        Ok(())
    }

    /// The physical-rank counts this loaded library can reduce, in ascending
    /// order. Useful for startup logging and fallback rejection.
    pub fn available_rank_counts(&self) -> Vec<u32> {
        [2u32, 3, 4, 6]
            .into_iter()
            .filter(|ranks| self.supports_rank_count(*ranks))
            .collect()
    }

    /// Sum six local FP32 routes and round the rank partial once to BF16.
    /// # Safety
    /// Routes are CUDA FP32 [rows,6,5120], output CUDA BF16 [rows,5120].
    /// They must not overlap. Both allocations and this library must outlive
    /// stream completion and graph replays, with producer writes ordered first.
    pub unsafe fn compact(
        &self,
        routes: *const f32,
        output: *mut u16,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status = unsafe { (self.compact)(routes, output, rows, stream) };
        ensure!(
            status == 0,
            "V4.1 route compaction failed with CUDA status {status}"
        );
        Ok(())
    }

    /// Sum six deterministic BF16 routes in FP32, then round once to BF16.
    /// # Safety
    /// Input is BF16 [rows,6,5120], output BF16 [rows,5120], disjoint and
    /// live on the current device through stream completion.
    pub unsafe fn compact_bf16_routes(&self, routes: *const u16, output: *mut u16,
        rows: u32, stream: *mut c_void) -> Result<()> {
        let function = self.compact_bf16_routes.context("native BF16 route compaction unavailable")?;
        let status = unsafe { function(routes, output, rows, stream) };
        ensure!(status == 0, "V4.1 BF16 route compaction failed with CUDA status {status}");
        Ok(())
    }

    /// Round an already accumulated FP32 token vector once to BF16.
    /// # Safety
    /// Input is CUDA FP32 [rows,5120], output CUDA BF16 [rows,5120].
    /// Both are disjoint, on the current device, and live through stream completion.
    pub unsafe fn compact_tokens(&self, tokens: *const f32, output: *mut u16,
        rows: u32, stream: *mut c_void) -> Result<()> {
        let function = self.compact_tokens.context("native token compaction unavailable")?;
        let status = unsafe { function(tokens, output, rows, stream) };
        ensure!(status == 0, "V4.1 token compaction failed with CUDA status {status}");
        Ok(())
    }

    /// Sum exactly two compact BF16 partials in FP32, add optional BF16 shared,
    /// then round once to BF16. Older native libraries remain TP4-compatible but
    /// return an error here if the optional TP2 entry point is absent.
    /// # Safety
    /// Planes, output and optional shared are CUDA BF16 [rows,5120] on the
    /// current device, with 1 <= rows <= 4096. Output cannot overlap planes and
    /// may alias shared only exactly. Storage and library must outlive stream
    /// completion/replay; order all producer writes before this operation.
    pub unsafe fn reduce_tp2(
        &self,
        planes: [*const u16; 2],
        shared: *const u16,
        output: *mut u16,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let function = self.reduce_tp2.context("native TP2 compact BF16 reduction unavailable")?;
        let status = unsafe { function(planes.as_ptr(), shared, output, rows, stream) };
        ensure!(status == 0, "V4.1 TP2 compact reduction failed with CUDA status {status}");
        Ok(())
    }

    /// Sum four compact partials in TP rank order, then add shared and round to BF16.
    /// # Safety
    /// Planes, output and optional shared are CUDA BF16 [rows,5120] on the
    /// current device. Output cannot overlap planes and may alias shared only
    /// exactly. Storage and library must outlive stream completion/replay;
    /// order all producer writes before this operation.
    pub unsafe fn reduce(
        &self,
        planes: [*const u16; 4],
        shared: *const u16,
        output: *mut u16,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status = unsafe { (self.reduce)(planes.as_ptr(), shared, output, rows, stream) };
        ensure!(
            status == 0,
            "V4.1 compact reduction failed with CUDA status {status}"
        );
        Ok(())
    }

    /// Sum `ranks` physical-rank BF16 [rows,5120] partial planes in rank order in
    /// FP32, add the optional BF16 shared expert exactly once, then round once to
    /// BF16. This is the replicated-group contract: every physical rank returns
    /// one plane directly to the coordinator, with no group-local reduction.
    /// # Safety
    /// `planes[0..ranks]` must be live non-null CUDA BF16 [rows,5120] views on the
    /// current device; `planes[ranks..6]` must be null. Output must not overlap any
    /// plane and may alias `shared` only exactly. 1 <= rows <= 4096 and ranks is
    /// 2, 3, 4 or 6. Storage and this library must outlive stream completion and
    /// every captured graph replay, with producer writes ordered first. The six
    /// pointers travel to the kernel by value; no device pointer array is used.
    pub unsafe fn reduce_planes(
        &self,
        planes: [*const u16; 6],
        ranks: u32,
        shared: *const u16,
        output: *mut u16,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            matches!(ranks, 2 | 3 | 4 | 6),
            "N-plane reduction requires ranks 2, 3, 4 or 6"
        );
        let function = self
            .reduce_planes
            .context("native N-plane compact BF16 reduction unavailable")?;
        let status = unsafe { function(planes.as_ptr(), shared, output, rows, ranks, stream) };
        ensure!(
            status == 0,
            "V4.1 {ranks}-plane compact reduction failed with CUDA status {status}"
        );
        Ok(())
    }
}

impl V41RouteReducer<'_> {
    /// # Safety
    /// Active planes must be contiguous CUDA FP32 [rows,topk,5120] in identical
    /// route order; output and optional shared must be CUDA BF16 [rows,5120].
    /// All storage must be on the current device and remain valid through stream
    /// completion and graph replays, with writes ordered before this operation.
    /// Output must not overlap planes; shared may alias output only exactly.
    /// Unused plane slots must be null. The library must outlive captured graphs.
    pub unsafe fn launch(
        &self,
        planes: [*const f32; 4],
        shared: *const u16,
        output: *mut u16,
        rows: u32,
        ranks: u32,
        topk: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status =
            unsafe { (self.reduce)(planes.as_ptr(), shared, output, rows, ranks, topk, stream) };
        ensure!(
            status == 0,
            "V4.1 route reduction failed with CUDA status {status}"
        );
        Ok(())
    }
}

pub struct V41ExpertKernel<'a> {
    // Keep all exported code and its CUDA modules loaded until this handle drops.
    _library: &'a NativeLibrary,
    handle: NonNull<c_void>,
    launch: LaunchFn,
    bind_scratch: BindScratchFn,
    initialize_scratch: InitScratchFn,
    info: V41ExpertInfo,
    output_kind: V41ExpertOutputKind,
}

impl NativeLibrary {
    pub fn v41_compact_reducer(&self) -> Result<V41CompactReducer<'_>> {
        Ok(V41CompactReducer {
            _library: self,
            compact: unsafe {
                *self
                    .lib
                    .get::<CompactFn>(b"ds41rt_v41_compact_routes_bf16_async")?
            },
            compact_tokens: unsafe { self.lib.get::<CompactFn>(b"ds41rt_v41_compact_tokens_bf16_async").ok().map(|f| *f) },
            compact_bf16_routes: unsafe { self.lib.get::<CompactBf16RoutesFn>(b"ds41rt_v41_compact_bf16_routes_async").ok().map(|f| *f) },
            reduce: unsafe {
                *self
                    .lib
                    .get::<ReduceCompactFn>(b"ds41rt_v41_reduce_compact_bf16_async")?
            },
            reduce_tp2: unsafe {
                self.lib.get::<ReduceCompactFn>(b"ds41rt_v41_reduce_tp2_compact_bf16_async").ok().map(|f| *f)
            },
            reduce_planes: unsafe {
                self.lib
                    .get::<ReduceCompactPlanesFn>(b"ds41rt_v41_reduce_compact_bf16_planes_async")
                    .ok()
                    .map(|f| *f)
            },
        })
    }

    pub fn v41_expert_packer(&self, intermediate: u32) -> Result<V41ExpertPacker<'_>> {
        ensure!(
            v41_pack_intermediate_supported(intermediate),
            "unsupported V4.1 pack intermediate {intermediate}; expected 384 (Spark TP6), 576 (Spark TP4), 768 (Spark TP3), 1152 (TP2) or 2304 (full)"
        );
        let sizes = unsafe {
            self.lib
                .get::<PackedSizesFn>(b"ds41rt_v41_expert_packed_sizes")?
        };
        let pack = unsafe { *self.lib.get::<PackFn>(b"ds41rt_v41_pack_expert_async")? };
        let mut bytes = [0; 4];
        let status = unsafe { sizes(intermediate, bytes.as_mut_ptr()) };
        ensure!(
            status == 0,
            "V4.1 packed sizes failed with CUDA status {status}"
        );
        Ok(V41ExpertPacker {
            _library: self,
            pack,
            intermediate,
            bytes,
        })
    }

    pub fn v41_route_reducer(&self) -> Result<V41RouteReducer<'_>> {
        let reduce = unsafe {
            *self
                .lib
                .get::<ReduceFn>(b"ds41rt_v41_reduce_routes_async")?
        };
        Ok(V41RouteReducer {
            _library: self,
            reduce,
        })
    }

    pub fn v41_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 0)
    }

    pub fn v41_local_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 2)
    }

    pub fn v41_tp2_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 3)
    }

    pub fn v41_dspark_tp2_expert_info(&self,capacity:u32)->Result<V41ExpertInfo> {
        self.expert_info_for(capacity,4)
    }

    /// Replicated-group Spark TP2 shard metadata (intermediate 1152, SM121).
    pub fn v41_spark_tp2_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 8)
    }

    /// Replicated-group Spark TP3 shard metadata (intermediate 768, SM121).
    pub fn v41_spark_tp3_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 9)
    }

    /// Pure `TP6EP1` Spark shard metadata (intermediate 384, SM121). Six
    /// disjoint intermediate slices of every routed expert, no storage padding.
    pub fn v41_spark_tp6_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 11)
    }

    fn expert_info_for(&self, capacity: u32, interface: u8) -> Result<V41ExpertInfo> {
        // 5..7 select the W4A4 ModelOpt NVFP4 family (RTX TP2, Spark TP4, full
        // RTX); 0..4 select the native W4A8 family; 8/9/11 select the native
        // replicated-group Spark TP2/TP3 and pure TP6 shards. All may live in
        // one library.
        let name: &[u8] = match interface {
            2 => b"ds41rt_v41_local_expert_info",
            3 => b"ds41rt_v41_tp2_expert_info",
            4 => b"ds41rt_v41_dspark_tp2_expert_info",
            5 => b"ds41rt_v41_nvfp4_tp2_expert_info",
            6 => b"ds41rt_v41_nvfp4_expert_info",
            7 => b"ds41rt_v41_nvfp4_local_expert_info",
            8 => b"ds41rt_v41_spark_tp2_expert_info",
            9 => b"ds41rt_v41_spark_tp3_expert_info",
            11 => b"ds41rt_v41_spark_tp6_expert_info",
            _ => b"ds41rt_v41_expert_info",
        };
        let function = unsafe { self.lib.get::<InfoFn>(name) }
            .with_context(|| format!("native library lacks expert interface {interface}; enable its AOT build"))?;
        let mut info = V41ExpertInfo::default();
        let status = unsafe { function(i32::try_from(capacity)?, &mut info) };
        ensure!(
            status == 0,
            "V4.1 expert metadata failed with CUDA status {status}"
        );
        ensure!(
            matches!(info.abi_version, 2 | 3) && info.hidden_size == 5120,
            "unsupported V4.1 native expert ABI"
        );
        let nvfp4 = matches!(interface, 5 | 6 | 7);
        ensure!(
            if nvfp4 {
                // W4A4 consumes BF16 hidden rows; the FP4 quantization happens
                // inside the kernel.
                info.input_dtype == 1
            } else {
                info.input_dtype == 1 || (matches!(info.role, 1 | 2 | 3 | 4 | 5 | 6 | 7) && info.input_dtype == 7)
            },
            "unsupported native expert input representation"
        );
        if matches!(interface, 8 | 9 | 11) {
            // Replicated-group and pure TP6 Spark shards are always exported
            // from the native FP8 K32 wire format; reject a BF16-hidden
            // artifact explicitly.
            ensure!(
                info.input_dtype == 7,
                "Spark TP2/TP3/TP6 require the native FP8 K32 input representation"
            );
        }
        let expected = expected_expert_geometry(nvfp4, info.role, info.kernel_intermediate)
            .ok_or_else(|| anyhow::anyhow!("unknown V4.1 expert role {} for interface {interface}", info.role))?;
        ensure!(
            (
                info.experts,
                info.logical_intermediate,
                info.kernel_intermediate,
                info.topk
            ) == expected,
            "native V4.1 expert geometry does not match the official checkpoint"
        );
        ensure!(
            info.capacity_rows == capacity && info.scratch_bytes > 0,
            "native expert capacity does not match requested variant"
        );
        let expected_role = match interface {
            0 => None,
            5 => Some(3),
            6 => Some(1),
            7 => Some(2),
            8 => Some(5),
            9 => Some(6),
            11 => Some(7),
            other => Some(u32::from(other)),
        };
        match expected_role {
            None => ensure!(info.role <= 1, "expert entry has incompatible role"),
            Some(role) => ensure!(info.role == role, "expert entry has incompatible role"),
        }
        Ok(info)
    }

    /// Load the chosen variant on the current CUDA device before graph capture.
    pub fn v41_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 0)
    }

    pub fn v41_local_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 2)
    }

    pub fn v41_tp2_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 3)
    }

    pub fn v41_dspark_tp2_expert_kernel(&self,capacity:u32)->Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity,4)
    }

    /// Replicated-group Spark TP2 shard kernels (native FP8 K32, SM121).
    pub fn v41_spark_tp2_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 8)
    }

    /// Replicated-group Spark TP3 shard kernels (native FP8 K32, SM121).
    pub fn v41_spark_tp3_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 9)
    }

    /// Pure `TP6EP1` Spark shard kernels (native FP8 K32, SM121): six disjoint
    /// intermediate slices of every routed expert, no storage padding.
    pub fn v41_spark_tp6_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 11)
    }

    /// W4A4 ModelOpt NVFP4 expert kernels. The family publishes BF16
    /// token-major route planes and is selected from the checkpoint format.
    pub fn v41_nvfp4_tp2_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 5)
    }

    /// Metadata for planning NVFP4 workspace before loading kernels.
    pub fn v41_nvfp4_tp2_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 5)
    }

    /// Spark TP4 metadata for the W4A4 family.
    pub fn v41_nvfp4_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 6)
    }

    /// Full-width RTX backbone kernels for the single-card W4A4 placement.
    pub fn v41_nvfp4_local_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 7)
    }

    pub fn v41_nvfp4_local_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        self.expert_info_for(capacity, 7)
    }

    pub fn v41_nvfp4_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        self.expert_kernel_for(capacity, 6)
    }

    fn expert_kernel_for(&self, capacity: u32, interface: u8) -> Result<V41ExpertKernel<'_>> {
        let info = self.expert_info_for(capacity, interface)?;
        // One symbol prefix per family/role. The NVFP4 family exposes its own
        // prefix so all expert formats coexist in one library.
        let prefix: &str = match interface {
            2 => "ds41rt_v41_local",
            3 => "ds41rt_v41_tp2",
            4 => "ds41rt_v41_dspark_tp2",
            5 => "ds41rt_v41_nvfp4_tp2",
            6 => "ds41rt_v41_nvfp4",
            7 => "ds41rt_v41_nvfp4_local",
            8 => "ds41rt_v41_spark_tp2",
            9 => "ds41rt_v41_spark_tp3",
            11 => "ds41rt_v41_spark_tp6",
            _ => "ds41rt_v41",
        };
        let symbol = |operation: &str| format!("{prefix}_expert_{operation}").into_bytes();
        let initialize = unsafe {
            self.lib
                .get::<InitializeFn>(&symbol("initialize"))?
        };
        let launch = unsafe { *self.lib.get::<LaunchFn>(&symbol("launch"))? };
        let bind_scratch = unsafe { *self.lib.get::<BindScratchFn>(&symbol("bind_scratch"))? };
        let initialize_scratch = unsafe {
            *self.lib.get::<InitScratchFn>(&symbol("initialize_scratch_async"))?
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { initialize(i32::try_from(capacity)?, &mut handle) };
        ensure!(
            status == 0,
            "V4.1 expert initialization failed with CUDA status {status}"
        );
        let nvfp4 = matches!(interface, 5 | 6 | 7);
        let output_kind = if info.abi_version == 3 || nvfp4 {
            type OutputKindFn = unsafe extern "C" fn(i32, *mut u32) -> i32;
            let query = unsafe { self.lib.get::<OutputKindFn>(&symbol("output_kind"))? };
            let mut kind = u32::MAX;
            let status = unsafe { query(i32::try_from(capacity)?, &mut kind) };
            ensure!(status == 0, "V4.1 output layout query failed: {status}");
            let kind = V41ExpertOutputKind::from_native(kind)?;
            if nvfp4 {
                ensure!(kind == V41ExpertOutputKind::Bf16Routes && info.abi_version == 2,
                    "NVFP4 requires deterministic BF16 route output");
                unsafe {
                    self.lib.get::<CompactBf16RoutesFn>(b"ds41rt_v41_compact_bf16_routes_async")?;
                    self.lib.get::<ReduceTp2Bf16RoutesFn>(b"ds41rt_v41_reduce_tp2_bf16_routes_async")?;
                }
            } else {
                ensure!(kind == V41ExpertOutputKind::Fp32Tokens && matches!(info.role, 1 | 2 | 3 | 5 | 6 | 7),
                    "unsupported V4.1 ABI 3 output layout");
                unsafe { self.lib.get::<CompactFn>(b"ds41rt_v41_compact_tokens_bf16_async")?; }
            }
            kind
        } else { V41ExpertOutputKind::Fp32Routes };
        Ok(V41ExpertKernel {
            _library: self,
            handle: NonNull::new(handle).context("native expert returned a null kernel handle")?,
            launch,
            bind_scratch,
            initialize_scratch,
            info,
            output_kind,
        })
    }
}

impl V41ExpertKernel<'_> {
    pub fn accumulates_tokens(&self) -> bool { self.output_kind == V41ExpertOutputKind::Fp32Tokens }

    pub fn output_kind(&self) -> V41ExpertOutputKind { self.output_kind }

    pub fn info(&self) -> &V41ExpertInfo {
        &self.info
    }

    /// Bind the exported scratch views; external tensor slots remain unchanged.
    /// # Safety
    /// Storage must be a live, aligned CUDA allocation of at least `bytes` bytes.
    /// The resulting pointers borrow storage and do not extend its lifetime.
    pub unsafe fn bind_scratch(
        &self,
        storage: *mut c_void,
        bytes: u64,
        tensors: &mut [*mut c_void; V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        let status = unsafe {
            (self.bind_scratch)(self.handle.as_ptr(), storage, bytes, tensors.as_mut_ptr())
        };
        ensure!(
            status == 0,
            "V4.1 scratch binding failed with CUDA status {status}"
        );
        Ok(())
    }

    /// Initialize the native recipe's scratch once before use and graph capture.
    /// # Safety
    /// Storage must be exclusively owned CUDA memory of at least `bytes` bytes
    /// on this kernel's device; it must remain alive until stream completion.
    /// Order this initialization before every launch that uses the storage.
    pub unsafe fn initialize_scratch(
        &self,
        storage: *mut c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        let status =
            unsafe { (self.initialize_scratch)(self.handle.as_ptr(), storage, bytes, stream) };
        ensure!(
            status == 0,
            "V4.1 scratch initialization failed with CUDA status {status}"
        );
        Ok(())
    }

    /// # Safety
    /// Every slot must reference correctly typed/packed CUDA storage of the
    /// capacity described by this variant's AOT manifest on its initialized device;
    /// scratch must be initialized and exclusively owned for this launch, and all
    /// buffers plus this kernel/library must remain alive through completion and
    /// every captured graph replay, with stream ordering enforced by the caller.
    pub unsafe fn launch(&self, args: &V41ExpertLaunchArgs) -> Result<()> {
        let status = unsafe { (self.launch)(self.handle.as_ptr(), args) };
        ensure!(
            status == 0,
            "V4.1 expert launch failed with CUDA status {status}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_kind_preserves_route_axis_and_dtype() {
        for (raw, kind, bytes) in [
            (0, V41ExpertOutputKind::Fp32Routes, 6 * 5120 * 4),
            (1, V41ExpertOutputKind::Fp32Tokens, 5120 * 4),
            (2, V41ExpertOutputKind::Bf16Routes, 6 * 5120 * 2),
        ] {
            assert_eq!(V41ExpertOutputKind::from_native(raw).unwrap(), kind);
            assert_eq!(kind.row_bytes(6), bytes);
        }
        assert!(V41ExpertOutputKind::from_native(3).is_err());
    }

    #[test]
    fn packed_intermediate_accept_list_covers_spark_tp_degrees() {
        for intermediate in [384u32, 576, 768, 1152, 2304] {
            assert!(v41_pack_intermediate_supported(intermediate));
        }
        for intermediate in [0u32, 1, 128, 385, 577, 640, 2303, 4096] {
            assert!(!v41_pack_intermediate_supported(intermediate));
        }
    }

    #[test]
    fn expected_geometry_covers_spark_tp2_tp3_tp6_without_touching_rtx_tp2() {
        // Replicated-group Spark shards: TP2 1152 and TP3 768 are unpadded, and
        // pure TP6 384 stores exactly its 2304/6 slice with no padding.
        assert_eq!(expected_expert_geometry(false, 5, 1152), Some((384, 1152, 1152, 6)));
        assert_eq!(expected_expert_geometry(false, 6, 768), Some((384, 768, 768, 6)));
        assert_eq!(expected_expert_geometry(false, 7, 384), Some((384, 384, 384, 6)));
        // Role 7 exists only in the native FP8 family: the W4A4 table and an
        // unknown role id both fail closed instead of matching by accident.
        assert_eq!(expected_expert_geometry(true, 7, 384), None);
        assert_eq!(expected_expert_geometry(false, 9, 384), None);
        // Historical Spark TP4 padding and RTX TP2 role semantics are unchanged.
        assert_eq!(expected_expert_geometry(false, 1, 640), Some((384, 576, 640, 6)));
        assert_eq!(expected_expert_geometry(false, 3, 1152), Some((384, 1152, 1152, 6)));
        // Unknown or cross-family pairs still fail closed.
        assert_eq!(expected_expert_geometry(false, 7, 768), None);
        assert_eq!(expected_expert_geometry(true, 5, 1152), None);
    }

    #[test]
    fn rank_count_contract_is_two_three_four_six() {
        for ranks in [2u32, 3, 4, 6] {
            assert!(v41_rank_count_supported(ranks));
        }
        for ranks in [0u32, 1, 5, 7, 12] {
            assert!(!v41_rank_count_supported(ranks));
        }
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA; reducer rank-count capability"]
    fn compact_reducer_reports_available_rank_counts() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let reducer = library.v41_compact_reducer()?;
        // A reducer handle always exposes the 4-plane entry point.
        assert!(reducer.supports_rank_count(4));
        assert!(reducer.require_rank_count(4).is_ok());
        // Unsupported counts are rejected before any allocation or launch.
        assert!(!reducer.supports_rank_count(5));
        assert!(reducer.require_rank_count(5).is_err());
        assert!(reducer.require_rank_count(0).is_err());
        // Whatever the library reports must agree with the contract set.
        for ranks in reducer.available_rank_counts() {
            assert!(v41_rank_count_supported(ranks));
            assert!(reducer.require_rank_count(ranks).is_ok());
        }
        // The current build must expose the replicated-group 3/6 path.
        assert!(
            reducer.supports_rank_count(3) && reducer.supports_rank_count(6),
            "current library is missing the N-plane compact reducer"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA; nonzero deterministic BF16 route oracle"]
    fn bf16_route_reducers_match_exact_nonzero_oracle() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let compact = library.v41_compact_reducer()?;
        let tp2 = library.v41_tp2_expert_reducer()?;
        // Nonuniform rows, columns, routes and ranks catch prefix-only copies,
        // dtype reinterpretation, wrong strides and omitted top-k reduction.
        let capacity = 80usize;
        let mut rank0 = library.alloc_device_buffer(capacity * 6 * 5120 * 2)?;
        let mut rank1 = library.alloc_device_buffer(capacity * 6 * 5120 * 2)?;
        let mut output = library.alloc_device_buffer(capacity * 5120 * 2)?;
        let result = (|| -> Result<()> {
            for rows in [1usize, 16, 80, 3, 1] {
                let mut sources = [Vec::new(), Vec::new()];
                for (rank, source) in sources.iter_mut().enumerate() {
                    for row in 0..rows {
                        for route in 0..6 {
                            for col in 0..5120 {
                                let value = if rank == 0 {
                                    (row % 7 + route + col % 3 + 1) as f32
                                } else {
                                    (route as i32 - 2) as f32
                                };
                                source.extend_from_slice(&((value.to_bits() >> 16) as u16).to_ne_bytes());
                            }
                        }
                    }
                }
                library.copy_h2d(rank0, &sources[0])?;
                library.copy_h2d(rank1, &sources[1])?;
                for ranks in [1usize, 2] {
                    unsafe {
                        if ranks == 1 {
                            compact.compact_bf16_routes(rank0.ptr.cast(), output.ptr.cast(), rows as u32, std::ptr::null_mut())?;
                        } else {
                            tp2.reduce_bf16_routes(rank0, rank1, output, rows as u32, std::ptr::null_mut())?;
                        }
                        library.cuda_stream_synchronize(std::ptr::null_mut())?;
                    }
                    let mut actual = vec![0u8; rows * 5120 * 2];
                    library.copy_d2h(&mut actual, output)?;
                    for (i, bytes) in actual.chunks_exact(2).enumerate() {
                        let row = i / 5120;
                        let col = i % 5120;
                        let expected = (6 * (row % 7 + col % 3 + 1) + 15 + if ranks == 2 { 3 } else { 0 }) as f32;
                        assert_eq!(u16::from_ne_bytes([bytes[0], bytes[1]]), (expected.to_bits() >> 16) as u16);
                    }
                }
            }
            Ok(())
        })();
        library.free_device_buffer(&mut rank0)?;
        library.free_device_buffer(&mut rank1)?;
        library.free_device_buffer(&mut output)?;
        result
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA; local BF16 route/shared oracle"]
    fn local_bf16_routes_preserve_compact_boundary() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let reducer = library.v41_local_expert_reducer()?;
        let capacity = 80usize;
        let mut routes = library.alloc_device_buffer(capacity * 6 * 5120 * 2)?;
        let mut fp32_routes = library.alloc_device_buffer(capacity * 6 * 5120 * 4)?;
        let mut shared = library.alloc_device_buffer(capacity * 5120 * 2)?;
        let mut output = library.alloc_device_buffer(capacity * 5120 * 2)?;
        let bf16 = |value: f32| {
            let bits = value.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
        };
        let value = |row: usize, route: usize, col: usize| -> f32 {
            if col % 7 == 0 { match route { 0 => 256., 1 => 1., _ => 0. } }
            else if col % 7 == 1 { match route { 0 => 256., 1 | 2 => 1., _ => 0. } }
            else { (row as i32 % 7 + route as i32 - col as i32 % 3) as f32 * 0.25 }
        };
        let result = (|| -> Result<()> {
            for rows in [1usize, 16, 80, 3, 1] {
                let mut source = Vec::new();
                for row in 0..rows { for route in 0..6 { for col in 0..5120 {
                    source.extend_from_slice(&bf16(value(row, route, col)).to_ne_bytes());
                } } }
                library.copy_h2d(routes, &source)?;
                let floats: Vec<u8> = source.chunks_exact(2).flat_map(|b|
                    f32::from_bits((u16::from_ne_bytes([b[0], b[1]]) as u32) << 16).to_ne_bytes()).collect();
                library.copy_h2d(fp32_routes, &floats)?;
                let shared_bytes: Vec<u8> = (0..rows * 5120).flat_map(|_| bf16(1.).to_ne_bytes()).collect();
                for mode in 0..3 {
                    library.copy_h2d(shared, &shared_bytes)?;
                    library.copy_h2d(output, &vec![0xa5; output.bytes])?;
                    let (s, out) = match mode {
                        0 => (std::ptr::null(), output),
                        1 => (shared.ptr.cast::<u16>() as *const u16, output),
                        _ => (shared.ptr.cast::<u16>() as *const u16, shared),
                    };
                    unsafe {
                        assert!(reducer.finish_bf16_routes(routes.ptr.cast(), s, out.ptr.cast(), 0, std::ptr::null_mut()).is_err());
                        assert!(reducer.finish_bf16_routes(routes.ptr.cast(), s, routes.ptr.cast(), rows as u32, std::ptr::null_mut()).is_err());
                        reducer.finish_bf16_routes(routes.ptr.cast(), s, out.ptr.cast(), rows as u32, std::ptr::null_mut())?;
                        library.cuda_stream_synchronize(std::ptr::null_mut())?;
                    }
                    let mut actual = vec![0u8; out.bytes];
                    library.copy_d2h(&mut actual, out)?;
                    for i in 0..rows * 5120 {
                        let sum: f32 = (0..6).map(|route| value(i / 5120, route, i % 5120)).sum();
                        let compact = f32::from_bits((bf16(sum) as u32) << 16);
                        let expected = bf16(compact + if mode == 0 { 0. } else { 1. });
                        assert_eq!(u16::from_ne_bytes(actual[i*2..i*2+2].try_into()?), expected,
                            "rows={rows} mode={mode} element={i}");
                    }
                    if mode < 2 { assert!(actual[rows * 10240..].iter().all(|&b| b == 0xa5)); }
                    // Keep the existing MXFP4/EXL3 FP32-route reducer numerically
                    // identical to the BF16 path for exactly representable inputs.
                    library.copy_h2d(shared, &shared_bytes)?;
                    unsafe {
                        reducer.finish(fp32_routes.ptr.cast(), s, out.ptr.cast(), rows as u32, false, std::ptr::null_mut())?;
                        library.cuda_stream_synchronize(std::ptr::null_mut())?;
                    }
                    let mut legacy = vec![0u8; rows * 10240];
                    library.copy_d2h(&mut legacy, out)?;
                    assert_eq!(legacy, actual[..legacy.len()]);
                }
            }
            Ok(())
        })();
        library.free_device_buffer(&mut routes)?;
        library.free_device_buffer(&mut fp32_routes)?;
        library.free_device_buffer(&mut shared)?;
        library.free_device_buffer(&mut output)?;
        result
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA; TP2/TP4 compact BF16 oracle"]
    fn compact_planes_sum_in_fp32_before_shared_and_final_rounding() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let reducer = library.v41_compact_reducer()?;
        let capacity = 80usize;
        let mut planes = (0..4).map(|_| library.alloc_device_buffer(capacity * 10240))
            .collect::<Result<Vec<_>>>()?;
        let mut shared = library.alloc_device_buffer(capacity * 10240)?;
        let mut output = library.alloc_device_buffer(capacity * 10240)?;
        let bf16 = |value: f32| {
            let bits = value.to_bits();
            ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
        };
        let value = |rank: usize, i: usize| -> f32 {
            match i % 8 {
                // 256+1+shared(1) must be 258, not round(257)+1 = 256.
                0 => [256., 1., 2., -1.][rank],
                1 => [-256., -1., -2., 1.][rank],
                // FP32 rank order, not a tree sum or BF16 intermediate sum.
                2 => [16777216., 1., -16777216., 1.][rank],
                3 => [1., 0.00390625, 0.0078125, -0.00390625][rank],
                _ => (i as i32 / 5120 % 7 + rank as i32 * 3 - i as i32 % 11) as f32 * 0.25,
            }
        };
        let result = (|| -> Result<()> {
            for rows in [1usize, 16, 80, 3, 1] {
                for (rank, plane) in planes.iter().enumerate() {
                    let source: Vec<u8> = (0..rows * 5120)
                        .flat_map(|i| bf16(value(rank, i)).to_ne_bytes()).collect();
                    library.copy_h2d(*plane, &source)?;
                }
                for ranks in [2usize, 4] { for mode in 0..3 {
                    let shared_bytes: Vec<u8> = (0..capacity * 5120)
                        .flat_map(|_| bf16(1.).to_ne_bytes()).collect();
                    library.copy_h2d(shared, &shared_bytes)?;
                    library.copy_h2d(output, &vec![0xa5; output.bytes])?;
                    let s = if mode == 0 { std::ptr::null() } else { shared.ptr.cast::<u16>().cast_const() };
                    let out = if mode == 2 { shared } else { output };
                    unsafe {
                        if ranks == 2 {
                            reducer.reduce_tp2(std::array::from_fn(|r| planes[r].ptr.cast::<u16>().cast_const()),
                                s, out.ptr.cast(), rows as u32, std::ptr::null_mut())?;
                        } else {
                            reducer.reduce(std::array::from_fn(|r| planes[r].ptr.cast::<u16>().cast_const()),
                                s, out.ptr.cast(), rows as u32, std::ptr::null_mut())?;
                        }
                        library.cuda_stream_synchronize(std::ptr::null_mut())?;
                    }
                    let mut actual = vec![0u8; out.bytes];
                    library.copy_d2h(&mut actual, out)?;
                    for i in 0..rows * 5120 {
                        let mut sum = value(0, i);
                        for rank in 1..ranks { sum += value(rank, i); }
                        if mode != 0 { sum += 1.; }
                        assert_eq!(u16::from_ne_bytes(actual[i*2..i*2+2].try_into()?), bf16(sum),
                            "TP{ranks} rows={rows} mode={mode} element={i}");
                    }
                    if mode != 2 { assert!(actual[rows * 10240..].iter().all(|&b| b == 0xa5)); }
                    else { assert_eq!(actual[rows * 10240..], shared_bytes[rows * 10240..]); }
                } }
            }
            // Validation must reject invalid geometry, null/misaligned planes,
            // routed/output aliasing, and partial shared/output overlap prelaunch.
            let p = std::array::from_fn(|r| planes[r].ptr.cast::<u16>().cast_const());
            let out = output.ptr.cast::<u16>();
            unsafe {
                for rows in [0, 4097] {
                    assert!(reducer.reduce_tp2(p, std::ptr::null(), out, rows, std::ptr::null_mut()).is_err());
                }
                assert!(reducer.reduce_tp2([p[0], std::ptr::null()], std::ptr::null(), out, 1, std::ptr::null_mut()).is_err());
                assert!(reducer.reduce_tp2([p[0], p[1].cast::<u8>().add(1).cast()], std::ptr::null(), out, 1, std::ptr::null_mut()).is_err());
                assert!(reducer.reduce_tp2(p, std::ptr::null(), planes[0].ptr.cast(), 1, std::ptr::null_mut()).is_err());
                assert!(reducer.reduce_tp2(p, out.add(1), out, 1, std::ptr::null_mut()).is_err());
            }
            Ok(())
        })();
        for plane in &mut planes { library.free_device_buffer(plane)?; }
        library.free_device_buffer(&mut shared)?;
        library.free_device_buffer(&mut output)?;
        result
    }

    #[test]
    fn expert_input_storage_tracks_encoded_representation() {
        let mut info = V41ExpertInfo {
            hidden_size: 5120,
            input_dtype: 1,
            ..Default::default()
        };
        assert_eq!(info.input_row_bytes().unwrap(), 10240);
        info.input_dtype = 7;
        assert_eq!(info.input_row_bytes().unwrap(), 5280);
        info.hidden_size = 5119;
        assert!(info.input_row_bytes().is_err());
        info.input_dtype = 3;
        assert!(info.input_row_bytes().is_err());
    }

    #[test]
    fn native_abi_layout_and_capacity_checks() {
        assert_eq!(std::mem::size_of::<V41ExpertInfo>(), 64);
        assert_eq!(std::mem::offset_of!(V41ExpertInfo, scratch_bytes), 32);
        assert_eq!(std::mem::offset_of!(V41ExpertInfo, input_dtype), 60);
        assert_eq!(std::mem::size_of::<V41ExpertLaunchArgs>(), 392);
        assert_eq!(std::mem::offset_of!(V41ExpertLaunchArgs, stream), 384);
        assert_eq!(
            V41ExpertPointer::TokenWeights as usize + 1,
            V41_EXPERT_POINTER_COUNT
        );
        let info = V41ExpertInfo {
            capacity_rows: 16,
            topk: 6,
            ..Default::default()
        };
        let slots = [NonNull::<u8>::dangling().as_ptr().cast(); 44];
        assert!(V41ExpertLaunchArgs::new(&info, slots, 17, std::ptr::null_mut()).is_err());
        assert!(V41ExpertLaunchArgs::new(&info, slots, 0, std::ptr::null_mut()).is_err());
        assert!(V41ExpertLaunchArgs::new(
            &info,
            [std::ptr::null_mut(); 44],
            16,
            std::ptr::null_mut()
        )
        .is_err());
        assert_eq!(
            V41ExpertLaunchArgs::new(&info, slots, 16, std::ptr::null_mut())
                .unwrap()
                .scatter_rows,
            96
        );
    }
}

type InputQuantFn = unsafe extern "C" fn(*mut c_void, *const u16, *mut u8, u32, *mut c_void) -> i32;

/// Preinitialized runtime-row quantizer writing the exact expert wire layout.
pub struct V41ExpertInputQuantizer<'a> {
    _library: &'a NativeLibrary,
    handle: NonNull<c_void>,
    launch: InputQuantFn,
}
impl NativeLibrary {
    pub fn v41_expert_input_quantizer(&self) -> Result<V41ExpertInputQuantizer<'_>> {
        type Init = unsafe extern "C" fn(*mut *mut c_void) -> i32;
        let init = unsafe {
            self.lib
                .get::<Init>(b"ds41rt_v41_expert_input_quant_initialize")?
        };
        let launch = unsafe {
            *self
                .lib
                .get::<InputQuantFn>(b"ds41rt_v41_expert_input_quantize_async")?
        };
        let mut raw = std::ptr::null_mut();
        let status = unsafe { init(&mut raw) };
        ensure!(
            status == 0,
            "expert input quantizer initialization failed: {status}"
        );
        Ok(V41ExpertInputQuantizer {
            _library: self,
            handle: NonNull::new(raw).context("null input quantizer handle")?,
            launch,
        })
    }
}
impl V41ExpertInputQuantizer<'_> {
    /// # Safety
    /// Initialized finite BF16 input and exclusive output are on the current
    /// device, nonoverlapping, and remain valid until this stream completes.
    pub unsafe fn launch(
        &self,
        input: crate::Ds41rtDeviceBuffer,
        output: crate::Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows)
                && input.device_id == output.device_id
                && input.bytes >= rows as usize * 10240
                && output.bytes >= rows as usize * 5280,
            "invalid expert input quantization buffers"
        );
        let status = unsafe {
            (self.launch)(
                self.handle.as_ptr(),
                input.ptr.cast(),
                output.ptr.cast(),
                rows,
                stream,
            )
        };
        ensure!(status == 0, "expert input quantization failed: {status}");
        Ok(())
    }
}


type FinishLocalFn = unsafe extern "C" fn(*const f32, *const u16, *mut u16, u32, u32, *mut c_void) -> i32;
type ReduceTp2Fn = unsafe extern "C" fn(*const f32, *const f32, *mut u16, u32, u32, *mut c_void) -> i32;
pub struct V41Tp2ExpertReducer<'a> {
    _library: &'a NativeLibrary,
    reduce: ReduceTp2Fn,
    reduce_bf16_routes: Option<ReduceTp2Bf16RoutesFn>,
    sum_routes: Option<SumTp2RoutesFn>,
}
type SumTp2RoutesFn = unsafe extern "C" fn(*const f32, *mut f32, u32, *mut c_void) -> i32;
impl NativeLibrary {
    pub fn v41_tp2_expert_reducer(&self) -> Result<V41Tp2ExpertReducer<'_>> {
        Ok(V41Tp2ExpertReducer { _library: self,
            reduce: unsafe { *self.lib.get::<ReduceTp2Fn>(b"ds41rt_v41_reduce_tp2_experts_async")? },
            reduce_bf16_routes: unsafe { self.lib.get::<ReduceTp2Bf16RoutesFn>(b"ds41rt_v41_reduce_tp2_bf16_routes_async").ok().map(|f| *f) },
            sum_routes: unsafe { self.lib.get::<SumTp2RoutesFn>(b"ds41rt_v41_sum_tp2_routes_async").ok().map(|f| *f) } })
    }
}
impl V41Tp2ExpertReducer<'_> {
    pub fn supports_route_sums(&self) -> bool { self.sum_routes.is_some() }
    /// FP32 token sums of one rank's six FP32 route planes, route order 0..5.
    /// # Safety
    /// `routes` is FP32 [rows,6,5120] and `sums` FP32 [rows,5120] on the current
    /// device, disjoint, ordered on this stream and live through completion.
    pub unsafe fn sum_routes(&self, routes: Ds41rtDeviceBuffer, sums: Ds41rtDeviceBuffer, rows: u32,
        stream: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid TP2 route-sum rows");
        let count = rows as usize * 5120;
        ensure!(routes.device_id == sums.device_id && routes.bytes >= count * 24 && sums.bytes >= count * 4,
            "TP2 route-sum buffers have incompatible device or extent");
        let function = self.sum_routes.context("native TP2 route sums unavailable")?;
        let status = unsafe { function(routes.ptr.cast(), sums.ptr.cast(), rows, stream) };
        ensure!(status == 0, "TP2 route sums failed with CUDA status {status}");
        Ok(())
    }
    /// Sum corresponding BF16 route pairs in FP32, then routes, rounding once.
    /// # Safety
    /// Inputs are BF16 [rows,6,5120]; output is BF16 [rows,5120]. All buffers
    /// are disjoint, ordered on this stream and live through completion.
    pub unsafe fn reduce_bf16_routes(&self, rank0: Ds41rtDeviceBuffer, rank1: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer, rows: u32, stream: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid TP2 reduction rows");
        let count = rows as usize * 5120;
        ensure!(rank0.device_id == output.device_id && rank1.device_id == output.device_id
            && rank0.bytes >= count * 12 && rank1.bytes >= count * 12 && output.bytes >= count * 2,
            "TP2 BF16 route buffers have incompatible device or extent");
        let function = self.reduce_bf16_routes.context("native TP2 BF16 route reduction unavailable")?;
        let status = unsafe { function(rank0.ptr.cast(), rank1.ptr.cast(), output.ptr.cast(), rows, stream) };
        ensure!(status == 0, "TP2 BF16 route reduction failed with CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Both FP32 rank buffers and the BF16 destination must remain live on the
    /// current device through stream completion. Producers (including peer copy)
    /// must be ordered before this launch. No conflicting aliases are permitted.
    pub unsafe fn reduce(&self, rank0: Ds41rtDeviceBuffer, rank1: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer, rows: u32, token_sums: bool, stream: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid TP2 reduction rows");
        let count = rows as usize * 5120;
        let bytes = count * if token_sums { 4 } else { 24 };
        ensure!(rank0.device_id == output.device_id && rank1.device_id == output.device_id
            && rank0.bytes >= bytes && rank1.bytes >= bytes && output.bytes >= count * 2,
            "TP2 reduction buffers have incompatible device or extent");
        let status = unsafe { (self.reduce)(rank0.ptr.cast(), rank1.ptr.cast(), output.ptr.cast(),
            rows, u32::from(token_sums), stream) };
        ensure!(status == 0, "TP2 expert reduction failed with CUDA status {status}");
        Ok(())
    }
}

type FinishLocalBf16RoutesFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, u32, *mut c_void) -> i32;
pub struct V41LocalExpertReducer<'a> {
    _library: &'a NativeLibrary,
    finish: FinishLocalFn,
    finish_bf16_routes: Option<FinishLocalBf16RoutesFn>,
}
impl NativeLibrary {
    pub fn v41_local_expert_reducer(&self) -> Result<V41LocalExpertReducer<'_>> {
        Ok(V41LocalExpertReducer { _library: self,
            finish: unsafe { *self.lib.get::<FinishLocalFn>(b"ds41rt_v41_finish_local_experts_async")? },
            finish_bf16_routes: unsafe { self.lib.get::<FinishLocalBf16RoutesFn>(b"ds41rt_v41_finish_local_bf16_routes_async").ok().map(|f| *f) } })
    }
}
impl V41LocalExpertReducer<'_> {
    /// Sum six BF16 routes in FP32, round to BF16, then add optional shared FFN.
    /// # Safety
    /// Complete BF16 routes [rows,6,5120] and optional shared [rows,5120] must
    /// remain live on the current device through stream completion. Output must
    /// not overlap routes; it may alias shared only exactly.
    pub unsafe fn finish_bf16_routes(&self, routed: *const u16, shared: *const u16,
        output: *mut u16, rows: u32, stream: *mut c_void) -> Result<()> {
        let function = self.finish_bf16_routes.context("native local BF16 route reduction unavailable")?;
        let status = unsafe { function(routed, shared, output, rows, stream) };
        ensure!(status == 0, "local BF16 expert reduction failed with CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Complete FP32 routes/token sums and optional BF16 shared input must be
    /// live on the current CUDA device, ordered before this operation. Output
    /// cannot overlap routed input, and may alias shared only exactly. Owners
    /// must remain alive through stream completion.
    pub unsafe fn finish(&self, routed: *const f32, shared: *const u16,
        output: *mut u16, rows: u32, token_sums: bool, stream: *mut c_void) -> Result<()> {
        let status = unsafe { (self.finish)(routed, shared, output, rows, u32::from(token_sums), stream) };
        ensure!(status == 0, "local expert reduction failed with CUDA status {status}");
        Ok(())
    }
}
