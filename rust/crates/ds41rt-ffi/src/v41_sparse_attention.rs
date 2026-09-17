//! Mixed FP8 window / FP4 paged-source attention with private proposal overlays.
use crate::{Ds41rtDeviceBuffer, NativeLibrary, V41Kv};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;
#[repr(C)]
#[derive(Clone, Copy)]
struct RawView {
    values: [*const u8; 4],
    scales: [*const u8; 4],
    window_end: *const u64,
    pages: *const u32,
    source_end: *const u64,
    window_proposal_capacity: u64,
    source_capacity: u64,
    source_proposal_capacity: u64,
    page_stride: u32,
    compressed: u32,
}
const _: [(); 120] = [(); std::mem::size_of::<RawView>()];
pub struct V41SparseSource {
    // Architectural compressed E2M1 values and E4M3 group-16 scales.
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub proposals: Ds41rtDeviceBuffer,
    pub proposal_scales: Ds41rtDeviceBuffer,
    pub pages: Ds41rtDeviceBuffer,
    pub end: Ds41rtDeviceBuffer,
    pub capacity: usize,
    pub proposal_capacity: usize,
    pub page_stride: usize,
}
pub struct V41SparseWindow {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub proposals: Ds41rtDeviceBuffer,
    pub proposal_scales: Ds41rtDeviceBuffer,
    pub end: Ds41rtDeviceBuffer,
    pub proposal_capacity: usize,
    /// Optional device U64 [rows] lower bounds for bounded decoder SWA replay.
    pub replay_begins: Option<Ds41rtDeviceBuffer>,
}
type Launch = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const u64,
    *const i32,
    *mut u16,
    i32,
    i32,
    *const RawView,
    *mut c_void,
) -> i32;
type SplitLaunch = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const u64,
    *const i32,
    *mut u16,
    i32,
    i32,
    *const RawView,
    *mut c_void,
    *mut f32,
    u64,
    i32,
) -> i32;
type BoundedLaunch = unsafe extern "C" fn(
    *const u16, *const f32, *const u64, *const i32, *mut u16, i32, i32,
    *const RawView, *mut c_void, *const u64, *mut f32, u64, i32,
) -> i32;
type BatchValidate = unsafe extern "C" fn(
    *const u16, *const f32, *const u64, *const i32, *mut u16, i32,
    *const RawView, *const RawView, *const u64, *mut f32, u64, i32, i32,
) -> i32;
type BatchLaunch = unsafe extern "C" fn(
    *const u16, *const f32, *const u64, *const i32, *mut u16, i32,
    *const RawView, *mut c_void, *const u64, *mut f32, i32, i32,
) -> i32;
/// Host-validated row descriptors. They do not own any referenced allocations.
pub struct V41SparseBatch {
    views: Vec<RawView>,
    compressed: i32,
    aot: bool,
    heads: usize,
}
impl V41SparseBatch {
    /// Include in captured graph identity: device descriptors may change their
    /// alignment/format eligibility between replays without changing shape.
    pub fn backend_key(&self) -> usize { usize::from(self.aot) | (usize::from(self.heads == 32) << 1) }
    pub fn bytes(&self) -> &[u8] {
        // repr(C), 120 bytes with no padding, all fields initialized by checked_view.
        unsafe { std::slice::from_raw_parts(self.views.as_ptr().cast(), self.views.len() * 120) }
    }
}
pub struct V41SparseAttention<'a> {
    _library: &'a NativeLibrary,
    heads: usize,
    launch: Option<Launch>,
    split_launch: Option<SplitLaunch>,
    bounded_launch: Option<BoundedLaunch>,
    batch_validate: Option<BatchValidate>,
    batch_launch: Option<BatchLaunch>,
    batch_aot_launch: Option<BatchLaunch>,
}
impl NativeLibrary {
    pub fn v41_sparse_attention(&self) -> Result<V41SparseAttention<'_>> {
        let initialize = unsafe {
            *self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_sparse_attention_initialize")?
        };
        let status = unsafe { initialize() };
        ensure!(
            status == 0,
            "sparse attention initialization status {status}"
        );
        Ok(V41SparseAttention {
            _library: self,
            heads: 64,
            launch: Some(unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention")? }),
            split_launch: Some(unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention_split")? }),
            batch_validate: unsafe { self.lib.get(b"ds41rt_v41_sparse_attention_batch_validate").ok().map(|symbol| *symbol) },
            batch_launch: unsafe { self.lib.get(b"ds41rt_v41_sparse_attention_batch").ok().map(|symbol| *symbol) },
            batch_aot_launch: unsafe { self.lib.get(b"ds41rt_v41_sparse_attention_batch_aot").ok().map(|symbol| *symbol) },
            bounded_launch: unsafe { self.lib.get(b"ds41rt_v41_sparse_attention_bounded").ok().map(|symbol| *symbol) },
        })
    }
    /// Prepare compact local-head attention on the current device before capture.
    /// Single-request launches require replay_begins (zeros allow the full window).
    pub fn v41_sparse_attention_heads32(&self) -> Result<V41SparseAttention<'_>> {
        let initialize = unsafe {
            *self.lib.get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_sparse_attention_heads32_initialize")?
        };
        ensure!(unsafe { initialize() } == 0, "compact sparse attention initialization failed");
        Ok(V41SparseAttention {
            _library: self,
            heads: 32,
            launch: None,
            split_launch: None,
            bounded_launch: Some(unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention_heads32_bounded")? }),
            batch_validate: Some(unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention_heads32_batch_validate")? }),
            batch_launch: Some(unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention_heads32_batch")? }),
            batch_aot_launch: unsafe { self.lib.get(b"ds41rt_v41_sparse_attention_heads32_batch_aot").ok().map(|symbol| *symbol) },
        })
    }

}
impl V41SparseAttention<'_> {
    pub fn heads(&self) -> usize { self.heads }

    /// Required split workspace for this prepared local-head geometry.
    pub fn scratch_bytes(&self, rows: usize, parts: usize) -> Result<usize> {
        Self::scratch_bytes_for_heads(rows, parts, self.heads)
    }

    /// Compatibility helper for the original 64-head path.
    pub fn split_scratch_bytes(rows: usize, parts: usize) -> Result<usize> {
        Self::scratch_bytes_for_heads(rows, parts, 64)
    }

    fn scratch_bytes_for_heads(rows: usize, parts: usize, heads: usize) -> Result<usize> {
        ensure!(heads == 32 || heads == 64, "invalid sparse attention head count");
        ensure!(
            (1..=4096).contains(&rows) && (1..=10).contains(&parts),
            "invalid sparse attention split shape"
        );
        Ok(rows * parts * heads * 514 * 4)
    }

    fn checked_view(
        query: Ds41rtDeviceBuffer, sink: Ds41rtDeviceBuffer, metadata: Ds41rtDeviceBuffer,
        selected: Option<Ds41rtDeviceBuffer>, window: &V41SparseWindow,
        source: Option<&V41SparseSource>, output: Ds41rtDeviceBuffer,
        rows: usize, window_width: usize, heads: usize,
    ) -> Result<RawView> {
        ensure!(
            (1..=4096).contains(&rows) && window_width <= 128 && (heads == 32 || heads == 64),
            "invalid sparse attention shape"
        );
        ensure!(
            (1..=4096).contains(&window.proposal_capacity),
            "invalid window proposal capacity"
        );
        ensure!(
            source.is_some() == selected.is_some(),
            "source and selection must be paired"
        );
        let check = |b: Ds41rtDeviceBuffer, n: usize| -> Result<()> {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == query.device_id,
                "sparse attention buffer is null, undersized or on another device"
            );
            Ok(())
        };
        for (b, n) in [
            (query, rows * heads * 1024),
            (sink, heads * 4),
            (metadata, rows * 80),
            (output, rows * heads * 1024),
            (window.values, 128 * 512),
            (window.scales, 128 * 16),
            (window.end, 8),
            (window.proposals, window.proposal_capacity * 512),
            (window.proposal_scales, window.proposal_capacity * 16),
        ] {
            check(b, n)?;
        }
        let mut view = RawView {
            values: [
                window.values.ptr.cast(),
                window.proposals.ptr.cast(),
                std::ptr::null(),
                std::ptr::null(),
            ],
            scales: [
                window.scales.ptr.cast(),
                window.proposal_scales.ptr.cast(),
                std::ptr::null(),
                std::ptr::null(),
            ],
            window_end: window.end.ptr.cast(),
            pages: std::ptr::null(),
            source_end: std::ptr::null(),
            window_proposal_capacity: window.proposal_capacity as u64,
            source_capacity: 0,
            source_proposal_capacity: 0,
            page_stride: 0,
            compressed: 0,
        };
        if let Some(s) = source {
            ensure!(
                (1..=67108864).contains(&s.capacity)
                    && (1..=4096).contains(&s.proposal_capacity)
                    && (1..=4096).contains(&s.page_stride),
                "invalid sparse source shape"
            );
            for (b, n) in [
                (s.values, s.capacity * V41Kv::COMPRESSED_VALUE_BYTES),
                (s.scales, s.capacity * V41Kv::COMPRESSED_SCALE_BYTES),
                (s.proposals, s.proposal_capacity * V41Kv::COMPRESSED_VALUE_BYTES),
                (s.proposal_scales, s.proposal_capacity * V41Kv::COMPRESSED_SCALE_BYTES),
                (s.pages, s.page_stride * 4),
                (s.end, 8),
                (
                    selected.expect("checked paired source selection"),
                    rows * 2048,
                ),
            ] {
                check(b, n)?;
            }
            view.values[2] = s.values.ptr.cast();
            view.values[3] = s.proposals.ptr.cast();
            view.scales[2] = s.scales.ptr.cast();
            view.scales[3] = s.proposal_scales.ptr.cast();
            view.pages = s.pages.ptr.cast();
            view.source_end = s.end.ptr.cast();
            view.source_capacity = s.capacity as u64;
            view.source_proposal_capacity = s.proposal_capacity as u64;
            view.page_stride = s.page_stride as u32;
            view.compressed = 2;
        }
        Ok(view)
    }
    pub fn supports_batch(&self) -> bool {
        self.batch_validate.is_some() && self.batch_launch.is_some()
    }
    pub fn batch_descriptor_bytes(rows: usize) -> Result<usize> {
        ensure!((1..=64).contains(&rows), "invalid sparse batch rows");
        Ok(rows * 120)
    }
    /// Validate the complete batch before descriptor upload and EVERY graph replay.
    /// Host descriptor preparation does not read device memory or enqueue work.
    pub fn prepare_batch(
        &self, query: Ds41rtDeviceBuffer, sink: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer, selected: Option<Ds41rtDeviceBuffer>,
        requests: &[(&V41SparseWindow, Option<&V41SparseSource>, usize)],
        output: Ds41rtDeviceBuffer, descriptors: Ds41rtDeviceBuffer,
        bounds: Ds41rtDeviceBuffer, scratch: Ds41rtDeviceBuffer,
    ) -> Result<V41SparseBatch> {
        ensure!(requests.len() > 1 && requests.len() <= 16, "invalid sparse batch requests");
        ensure!(requests.iter().all(|(_, _, n)| (1..=16).contains(n)), "sparse batch requires split rows");
        let rows: usize = requests.iter().map(|(_, _, n)| n).sum();
        let descriptor_bytes = Self::batch_descriptor_bytes(rows)?;
        let compressed = if selected.is_some() { 2 } else { 0 };
        let parts = if compressed != 0 { 10 } else { 2 };
        for (buffer, bytes) in [(descriptors, descriptor_bytes), (bounds, rows * 8),
            (scratch, self.scratch_bytes(rows, parts)?)] {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes && buffer.device_id == query.device_id,
                "sparse batch buffer is null, undersized or on another device");
        }
        let mut views = Vec::with_capacity(rows);
        for &(window, source, count) in requests {
            let view = Self::checked_view(query, sink, metadata, selected, window, source, output, rows, 0, self.heads)?;
            views.extend(std::iter::repeat(view).take(count));
        }
        let validate = self.batch_validate.context("native library lacks sparse batch validation")?;
        let status = unsafe { validate(query.ptr.cast(), sink.ptr.cast(), metadata.ptr.cast(),
            selected.map_or(std::ptr::null(), |b| b.ptr.cast()), output.ptr.cast(), rows as i32,
            views.as_ptr(), descriptors.ptr.cast(), bounds.ptr.cast(), scratch.ptr.cast(),
            scratch.bytes as u64, parts as i32, compressed) };
        ensure!(status == 0, "native sparse batch validation status {status}");
        let aot = self.batch_aot_launch.is_some() && compressed == 2
            && scratch.ptr as usize % 16 == 0
            && views.iter().all(|view| view.values.iter().chain(view.scales.iter())
                .all(|pointer| *pointer as usize % 16 == 0));
        Ok(V41SparseBatch { views, compressed, aot, heads: self.heads })
    }
    /// # Safety
    /// All buffers and dimensions must match prepare_batch; upload batch.bytes()
    /// to descriptors on this stream first. Hold every referenced allocation live
    /// and immutable through completion/replay. Revalidate current descriptors
    /// and bindings on EVERY graph replay; the graph does not validate for you.
    pub unsafe fn launch_batch(
        &self, batch: &V41SparseBatch, query: Ds41rtDeviceBuffer, sink: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer, selected: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer, descriptors: Ds41rtDeviceBuffer,
        bounds: Ds41rtDeviceBuffer, scratch: Ds41rtDeviceBuffer, stream: *mut c_void,
    ) -> Result<()> {
        ensure!(batch.heads == self.heads, "sparse batch head geometry differs from prepared kernel");
        let launch = if batch.aot { self.batch_aot_launch } else { self.batch_launch }
            .context("native library lacks selected sparse batch attention")?;
        let status = unsafe { launch(query.ptr.cast(), sink.ptr.cast(), metadata.ptr.cast(),
            selected.map_or(std::ptr::null(), |b| b.ptr.cast()), output.ptr.cast(), batch.views.len() as i32,
            descriptors.ptr.cast(), stream, bounds.ptr.cast(), scratch.ptr.cast(),
            if batch.compressed != 0 { 10 } else { 2 }, batch.compressed) };
        ensure!(status == 0, "native sparse batch status {status}");
        Ok(())
    }

    /// # Safety
    /// Rotated BF16 queries [rows,heads(),512], finite FP32 sinks [heads()], U64 metadata
    /// [rows,10] and optional I32 source IDs [rows,512] follow the native header.
    /// FP8 window and FP4 source KV dequantize to finite BF16. Caller binds every row to one request's
    /// live window/source leases and proposal snapshots; causal source lengths
    /// and unique IDs are correct. Inputs remain immutable on the initialized
    /// stream device through completion/replay, with disjoint BF16 output.
    /// Width zero derives each query's own causal window length, keeping its
    /// softmax tiles independent of other rows in the launch. Optional
    /// split scratch is disjoint from all inputs/output and remains live through
    /// replay; partitions alter softmax rounding and require separate qualification.
    pub unsafe fn launch(
        &self,
        query: Ds41rtDeviceBuffer,
        sink: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer,
        selected: Option<Ds41rtDeviceBuffer>,
        window: &V41SparseWindow,
        source: Option<&V41SparseSource>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        window_width: usize,
        split: Option<(Ds41rtDeviceBuffer, usize)>,
        stream: *mut c_void,
    ) -> Result<()> {
        let view = Self::checked_view(query, sink, metadata, selected, window, source, output, rows, window_width, self.heads)?;
        let check = |b: Ds41rtDeviceBuffer, n: usize| -> Result<()> {
            ensure!(!b.ptr.is_null() && b.bytes >= n && b.device_id == query.device_id,
                "sparse attention buffer is null, undersized or on another device");
            Ok(())
        };
        let status = if let Some(bounds) = window.replay_begins {
            check(bounds, rows * 8)?;
            let launch = self.bounded_launch.context("native library lacks bounded decoder attention")?;
            let (partial, bytes, parts) = if let Some((scratch, parts)) = split {
                check(scratch, self.scratch_bytes(rows, parts)?)?;
                (scratch.ptr.cast(), scratch.bytes as u64, parts as i32)
            } else {
                (std::ptr::null_mut(), 0, 1)
            };
            unsafe {
                launch(query.ptr.cast(), sink.ptr.cast(), metadata.ptr.cast(),
                    selected.map_or(std::ptr::null(), |b| b.ptr.cast()), output.ptr.cast(),
                    rows as i32, window_width as i32, &view, stream, bounds.ptr.cast(),
                    partial, bytes, parts)
            }
        } else if let Some((scratch, parts)) = split {
            check(scratch, self.scratch_bytes(rows, parts)?)?;
            unsafe {
                (self.split_launch.context("compact attention requires replay bounds")?)(
                    query.ptr.cast(),
                    sink.ptr.cast(),
                    metadata.ptr.cast(),
                    selected.map_or(std::ptr::null(), |b| b.ptr.cast()),
                    output.ptr.cast(),
                    rows as i32,
                    window_width as i32,
                    &view,
                    stream,
                    scratch.ptr.cast(),
                    scratch.bytes as u64,
                    parts as i32,
                )
            }
        } else {
            unsafe {
                (self.launch.context("compact attention requires replay bounds")?)(
                    query.ptr.cast(),
                    sink.ptr.cast(),
                    metadata.ptr.cast(),
                    selected.map_or(std::ptr::null(), |b| b.ptr.cast()),
                    output.ptr.cast(),
                    rows as i32,
                    window_width as i32,
                    &view,
                    stream,
                )
            }
        };
        ensure!(status == 0, "native sparse attention status {status}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(bytes: usize) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer { ptr: 0x1000usize as *mut c_void, bytes, device_id: 0, flags: 0 }
    }

    #[test]
    fn compact_geometry_validates_exact_extents_and_device_ownership() {
        let window = V41SparseWindow {
            values: buffer(128*512), scales: buffer(128*16),
            proposals: buffer(512), proposal_scales: buffer(16), end: buffer(8),
            proposal_capacity: 1, replay_begins: None,
        };
        let check = |query, sink, output, heads| {
            V41SparseAttention::checked_view(query, sink, buffer(80), None,
                &window, None, output, 1, 0, heads)
        };
        assert!(check(buffer(32768), buffer(128), buffer(32768), 32).is_ok());
        assert!(check(buffer(32768), buffer(128), buffer(32768), 64).is_err());
        assert!(check(buffer(32767), buffer(128), buffer(32768), 32).is_err());
        assert!(check(buffer(32768), buffer(127), buffer(32768), 32).is_err());
        assert!(check(buffer(32768), buffer(128), buffer(32767), 32).is_err());
        let mut other_device = buffer(32768);
        other_device.device_id = 1;
        assert!(check(buffer(32768), buffer(128), other_device, 32).is_err());
        assert!(check(buffer(65536), buffer(256), buffer(65536), 64).is_ok());
        assert!(check(buffer(65536), buffer(256), buffer(65536), 16).is_err());
    }

    #[test]
    fn scratch_and_graph_identity_distinguish_local_heads() -> Result<()> {
        for rows in [1, 16, 64, 4096] {
            let full = V41SparseAttention::split_scratch_bytes(rows, 10)?;
            assert_eq!(V41SparseAttention::scratch_bytes_for_heads(rows, 10, 32)?, full/2);
        }
        assert!(V41SparseAttention::scratch_bytes_for_heads(4097, 10, 32).is_err());
        assert!(V41SparseAttention::scratch_bytes_for_heads(64, 11, 32).is_err());
        let mut keys = Vec::new();
        for heads in [64, 32] { for aot in [false, true] {
            let batch = V41SparseBatch { views: Vec::new(), compressed: 2, aot, heads };
            keys.push(batch.backend_key());
        }}
        assert_eq!(keys, [0, 1, 2, 3]);
        Ok(())
    }
}
