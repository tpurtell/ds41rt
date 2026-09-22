//! Final target mHC collapse, RMS norm and the vocabulary shared with dSpark.
pub(crate) mod distributed;
pub(crate) mod distributed_target;
use crate::v41_attention_binding::QueryBinding;
use crate::v41_block::BlockOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::{Context, Result, ensure};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, Ds41rtV41SamplerRow, NativeLibrary, V41Hc, V41VocabularyProjection,
};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};
const STRIDES: [usize; 7] = [40960, 16, 10240, 10240, 517120, 4, 4];

/// Vocabulary width of the official checkpoint. Only used to size the mask,
/// bitmap and histogram arenas owned by [`TargetSamplingWave`]; the K1 kernel
/// discovers the width from the buffer it is given (contract §1).
const SAMPLING_VOCAB: usize = 129_280;

/// Packed mask words per row (`ceil(vocab/32)`, design §5.2).
const SAMPLING_MASK_WORDS: usize = SAMPLING_VOCAB.div_ceil(32);

/// Chunk-6 radix histogram arena, allocated once now so a later chunk needs no
/// new allocation. 2048 buckets per row (design §11.1).
const SAMPLING_HISTOGRAM_BUCKETS: usize = 2048;

/// The largest mask arena the wave can address (`mask_row` is a `u32`).
const SAMPLING_MAX_MASK_ROWS: usize = 128;

/// Per-row work for one target-sampler launch.
///
/// `greedy` is the *host* resolution of `temperature < 1e-5 || top_k == 1`.
/// The kernel re-derives it anyway, so this is only what tells the host which
/// rows still need the CPU path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TargetSamplingRowRequest {
    pub(crate) row: Ds41rtV41SamplerRow,
    pub(crate) greedy: bool,
}

/// Device-selected rows for one target wave, in batch-row order.
///
/// `scores` is the raw maximum logit and is only meaningful where the row was
/// greedy; `status_detail` has already had the device's `0xFFFFFFFF`
/// "no detail" sentinel normalized to 0 (approved design deviation).
pub(crate) struct SampledTargetRows {
    pub(crate) ids: Vec<u32>,
    pub(crate) scores: Vec<f32>,
    pub(crate) status: Vec<u32>,
    pub(crate) status_detail: Vec<u32>,
    /// The head's vocabulary-projection buffer these ids were selected from.
    /// Retained so a caller can download a specific row without re-running the
    /// pass; it is a view, not an owner.
    pub(crate) logits: Ds41rtDeviceBuffer,
}

impl SampledTargetRows {
    pub(crate) fn rows(&self) -> usize {
        self.ids.len()
    }
}

/// The device's "no detail" sentinel normalized to 0, matching the ABI note in
/// `v41_sampling_gpu.h` (approved design deviation).
fn normalize_status_detail(value: u32) -> u32 {
    if value == ds41rt_ffi::DS41RT_V41_SAMPLER_NO_DETAIL {
        0
    } else {
        value
    }
}

/// Reusable per-projection device storage for the v4.1 GPU target-sampler.
///
/// Allocated once with the [`TargetHeadWave`] and freed with it: the zero-
/// allocation contract of design §11.1. Everything is stream-ordered on the
/// head stream, so no new host synchronization is introduced — the sampler's
/// small D2H is enqueued before the drain the scheduler already performs.
struct TargetSamplingWave<'a> {
    library: &'a NativeLibrary,
    capacity: usize,
    mask_words: usize,
    /// 64-byte per-row parameter blocks. `Pinned` holds the staging copy.
    param_device: DeviceAllocation<'a>,
    param_pinned: HostAllocation<'a>,
    /// Packed `capacity * mask_words` mask arena plus its staging copy.
    mask_device: DeviceAllocation<'a>,
    mask_pinned: HostAllocation<'a>,
    /// Per-row K1 reductions.
    scratch: DeviceAllocation<'a>,
    /// Top-k membership bitmap arena (chunk 3, allocated now).
    bitmap: DeviceAllocation<'a>,
    /// Chunk-6 radix histogram arena (allocated now).
    histogram: DeviceAllocation<'a>,
    ids_device: DeviceAllocation<'a>,
    status_device: DeviceAllocation<'a>,
    detail_device: DeviceAllocation<'a>,
    scores_device: DeviceAllocation<'a>,
    total_device: DeviceAllocation<'a>,
    nucleus_device: DeviceAllocation<'a>,
    ids_pinned: HostAllocation<'a>,
    status_pinned: HostAllocation<'a>,
    detail_pinned: HostAllocation<'a>,
    scores_pinned: HostAllocation<'a>,
    /// The parameter blocks the last upload staged, so the launch can pass them
    /// by value to the FFI wrapper without re-borrowing the pinned buffer.
    params: Vec<Ds41rtV41SamplerRow>,
    /// Rows of the arenas that the last upload actually filled.
    param_rows: usize,
    mask_rows: usize,
}

/// The `(mask buffer, mask_words_per_row)` pair for one sampled launch.
///
/// The two must travel together: when no row needs a mask, `upload()` stages
/// nothing and leaves `mask_rows == 0`, and both the FFI validator
/// (`mask_words_per_row must be 0 when no mask buffer is supplied`) and the
/// kernel (`MASK_WIDTH`) reject a non-zero width with a null arena. The wave's
/// own `mask_words` (4040) is always non-zero, so passing it unconditionally
/// made every all-unconstrained round — a traced all-greedy round, or a
/// constrained member whose grammar needs no mask at any row — fail its lane.
fn mask_launch_arguments(mask_rows: usize, mask_device: Ds41rtDeviceBuffer, mask_words: usize,
) -> (Option<Ds41rtDeviceBuffer>, usize) {
    if mask_rows == 0 {
        (None, 0)
    } else {
        (Some(mask_device), mask_words)
    }
}

impl<'a> TargetSamplingWave<'a> {
    fn mask_arena_bytes(capacity: usize) -> usize {
        capacity * SAMPLING_MASK_WORDS * 4
    }
    /// Device bytes the sampler adds to the head wave budget (design §11.1).
    fn device_bytes(capacity: usize) -> usize {
        capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES
            + Self::mask_arena_bytes(capacity)
            + capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_SCRATCH_BYTES
            + Self::mask_arena_bytes(capacity)
            + capacity * SAMPLING_HISTOGRAM_BUCKETS * 4
            + capacity * 4
            + capacity * 4
            + capacity * 4
            + capacity * 4
            + capacity * 8
    }
    fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Self> {
        ensure!(
            (1..=SAMPLING_MAX_MASK_ROWS).contains(&capacity),
            "sampling capacity must be 1..{SAMPLING_MAX_MASK_ROWS}"
        );
        Ok(Self {
            library,
            capacity,
            mask_words: SAMPLING_MASK_WORDS,
            param_device: DeviceAllocation::new(
                library,
                capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES,
            )?,
            param_pinned: HostAllocation::new(
                library,
                capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES,
            )?,
            mask_device: DeviceAllocation::new(library, Self::mask_arena_bytes(capacity))?,
            mask_pinned: HostAllocation::new(library, Self::mask_arena_bytes(capacity))?,
            scratch: DeviceAllocation::new(
                library,
                capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_SCRATCH_BYTES,
            )?,
            bitmap: DeviceAllocation::new(library, Self::mask_arena_bytes(capacity))?,
            histogram: DeviceAllocation::new(
                library,
                capacity * SAMPLING_HISTOGRAM_BUCKETS * 4,
            )?,
            ids_device: DeviceAllocation::new(library, capacity * 4)?,
            status_device: DeviceAllocation::new(library, capacity * 4)?,
            detail_device: DeviceAllocation::new(library, capacity * 4)?,
            scores_device: DeviceAllocation::new(library, capacity * 4)?,
            total_device: DeviceAllocation::new(library, capacity * 4)?,
            nucleus_device: DeviceAllocation::new(library, capacity * 4)?,
            ids_pinned: HostAllocation::new(library, capacity * 4)?,
            status_pinned: HostAllocation::new(library, capacity * 4)?,
            detail_pinned: HostAllocation::new(library, capacity * 4)?,
            scores_pinned: HostAllocation::new(library, capacity * 4)?,
            params: Vec::new(),
            param_rows: 0,
            mask_rows: 0,
        })
    }

    /// Fill the pinned parameter staging from the per-row requests.
    ///
    /// `mask_staging` is the host's packed `rows * mask_words` arena; a row
    /// whose `flags` has `NO_MASK` set is unconstrained and its arena slice is
    /// not read. Only the rows that are actually masked are copied to the
    /// device.
    ///
    /// The §5.3 remainder rule is applied here, immediately before the upload:
    /// every bit `>= vocab` of a masked row's final word is cleared, so a
    /// whole-word reader cannot mistake an out-of-range bit for a real token.
    fn upload(
        &mut self,
        requests: &[TargetSamplingRowRequest],
        mask_staging: Option<&[u32]>,
        mask_words: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let rows = requests.len();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "sampling rows exceed the wave capacity"
        );
        ensure!(
            mask_words == self.mask_words,
            "sampling mask width differs from the arena"
        );
        if let Some(mask_staging) = mask_staging {
            ensure!(
                mask_staging.len() == rows * self.mask_words,
                "sampling mask staging extent differs"
            );
        }
        {
            let staging = self.param_pinned.bytes_mut();
            for (slot, request) in requests.iter().enumerate() {
                ensure!(
                    request.row.output_row as usize == slot,
                    "sampling output_row must equal the batch slot"
                );
                let offset = slot * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES;
                staging[offset..offset + ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES]
                    .copy_from_slice(bytemuck_bytes(&request.row));
            }
        }
        let mut mask_rows = 0;
        if let Some(mask_staging) = mask_staging {
            let staging = self.mask_pinned.bytes_mut();
            for (slot, request) in requests.iter().enumerate() {
                if request.row.flags & ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK != 0 {
                    continue;
                }
                let target_row = request.row.mask_row as usize;
                ensure!(
                    target_row < rows,
                    "sampling mask_row is outside the uploaded mask rows"
                );
                let source = slot * self.mask_words;
                let target = target_row * self.mask_words;
                let mut words = mask_staging[source..source + self.mask_words].to_vec();
                ds41rt_ffi::ds41rt_v41_sampler_clear_remainder(&mut words, SAMPLING_VOCAB);
                staging[target * 4..(target + self.mask_words) * 4]
                    .copy_from_slice(bytemuck_slice(&words));
                mask_rows = mask_rows.max(target_row + 1);
            }
        }
        let param_bytes = rows * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES;
        unsafe {
            self.library.copy_host_buffer_h2d_async(
                self.param_device.buffer,
                self.param_pinned.buffer,
                param_bytes,
                stream,
            )?;
            if mask_rows > 0 {
                self.library.copy_host_buffer_h2d_async(
                    self.mask_device.buffer,
                    self.mask_pinned.buffer,
                    mask_rows * self.mask_words * 4,
                    stream,
                )?;
            }
        }
        self.params = requests.iter().map(|request| request.row).collect();
        self.param_rows = rows;
        self.mask_rows = mask_rows;
        Ok(())
    }

    fn launch(&mut self, logits: Ds41rtDeviceBuffer, rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!(
            rows == self.param_rows && logits.bytes == rows * STRIDES[4],
            "sampling launch shape differs from the uploaded rows"
        );
        let (masks, mask_words_per_row) =
            mask_launch_arguments(self.mask_rows, self.mask_device.buffer, self.mask_words);
        // K1 reads the parameter block on device, so launch the buffer
        // `upload()` filled; the host copy is only the validator's input.
        let params = self.params.clone();
        unsafe {
            self.library.cuda_v41_target_sample_async(
                logits,
                rows,
                SAMPLING_VOCAB,
                SAMPLING_VOCAB,
                &params,
                self.param_device.buffer,
                masks,
                mask_words_per_row,
                self.ids_device.buffer,
                self.status_device.buffer,
                self.detail_device.buffer,
                self.scores_device.buffer,
                None,
                None,
                self.scratch.buffer,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.ids_pinned.buffer,
                self.ids_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.status_pinned.buffer,
                self.status_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.detail_pinned.buffer,
                self.detail_device.buffer,
                rows * 4,
                stream,
            )?;
            self.library.copy_d2h_host_buffer_async(
                self.scores_pinned.buffer,
                self.scores_device.buffer,
                rows * 4,
                stream,
            )?;
        }
        self.params = params;
        Ok(())
    }

    fn output(&mut self, logits: Ds41rtDeviceBuffer, rows: usize) -> Result<SampledTargetRows> {
        ensure!(
            rows == self.param_rows,
            "sampling output rows differ from the uploaded rows"
        );
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes[..rows * 4]
                .chunks_exact(4)
                .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
                .collect()
        };
        let status = words(self.status_pinned.bytes());
        let detail: Vec<u32> =
            words(self.detail_pinned.bytes()).into_iter().map(normalize_status_detail).collect();
        Ok(SampledTargetRows {
            ids: words(self.ids_pinned.bytes()),
            scores: self.scores_pinned.bytes()[..rows * 4]
                .chunks_exact(4)
                .map(|word| f32::from_ne_bytes(word.try_into().unwrap()))
                .collect(),
            status,
            status_detail: detail,
            logits,
        })
    }
}

/// Raw bytes of a plain-old-data value, for the pinned staging copy.
fn bytemuck_bytes<T: Copy>(value: &T) -> &[u8] {
    // Safety: `T` is `Copy` and `Ds41rtV41SamplerRow` is `repr(C)` with no
    // padding holes that Rust would leave uninitialized (every field is written
    // by `Default`), so reading its object representation is defined.
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

fn bytemuck_slice<T: Copy>(values: &[T]) -> &[u8] {
    // Safety: as above; `T` is a plain integer type.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

pub(crate) struct TargetHeadWeights<'a> {
    library: &'a NativeLibrary,
    norm: NativeRtxTensors<'a>,
}
impl<'a> TargetHeadWeights<'a> {
    pub fn device_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &["norm.weight".into()])?;
        ensure!(bytes == 10240, "unexpected target norm geometry");
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog)? <= budget,
            "target norm exceeds budget"
        );
        Ok(Self {
            library,
            norm: NativeRtxTensors::load(
                library,
                catalog,
                &["norm.weight".into()],
                budget,
                staging,
            )?,
        })
    }
    pub fn wave<'w>(
        &'w self,
        head: &'w VocabularyHead<'a>,
        capacity: usize,
        budget: usize,
    ) -> Result<TargetHeadWave<'w, 'a>> {
        ensure!(
            TargetHeadWave::device_bytes(capacity)? <= budget,
            "target head wave exceeds budget"
        );
        let workspace =
            DeviceAllocation::new(self.library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let projection = unsafe { self.library.v41_vocabulary_head(workspace.buffer)? };
        let buffers = STRIDES
            .into_iter()
            .map(|n| DeviceAllocation::new(self.library, n * capacity))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            head.weight()?.device_id == buffers[0].buffer.device_id
                && self.norm.get("norm.weight")?.device_id == buffers[0].buffer.device_id,
            "target head weight device differs"
        );
        Ok(TargetHeadWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            projection,
            _workspace: workspace,
            hc: self.library.v41_hc()?,
            buffers,
            weights: self,
            head,
            capacity,
            graph: None,
            ready: None,
            origin: None,
            selected: Vec::new(),
            tokens: Vec::new(),
            greedy_staging: HostAllocation::new(self.library, capacity * 8)?,
            greedy_ready: false,
            sampling: TargetSamplingWave::new(self.library, capacity)?,
            download: crate::v41_memory::RowDownload::new(self.library, capacity * STRIDES[4])?,
        })
    }
}
pub(crate) struct TargetLogits<'a> {
    pub rows: usize,
    pub collapsed: Ds41rtDeviceBuffer,
    pub normalized: Ds41rtDeviceBuffer,
    pub logits: Ds41rtDeviceBuffer,
    pub selected_rows: &'a [usize],
    pub token_positions: &'a [u64],
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl TargetLogits<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin.context("target logits have no block origin")
    }
}
pub(crate) struct TargetHeadWave<'w, 'a> {
    download: crate::v41_memory::RowDownload<'a>,
    stream: LoadStream<'a>,
    projection: V41VocabularyProjection<'a>,
    _workspace: DeviceAllocation<'a>,
    hc: V41Hc<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w TargetHeadWeights<'a>,
    head: &'w VocabularyHead<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    ready: Option<usize>,
    origin: Option<QueryBinding>,
    selected: Vec<usize>,
    tokens: Vec<u64>,
    greedy_staging: HostAllocation<'a>,
    greedy_ready: bool,
    sampling: TargetSamplingWave<'a>,
}
impl TargetHeadWave<'_, '_> {
    /// Up to 16 decode/prefill-last rows or 80 verification rows per head wave.
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=80).contains(&capacity),
            "target head capacity must be 1..80"
        );
        Ok(V41VocabularyProjection::WORKSPACE_BYTES
            + capacity * STRIDES.iter().sum::<usize>()
            + TargetSamplingWave::device_bytes(capacity))
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.greedy_ready = false;
        self.ready = None;
        self.origin = None;
        self.selected.clear();
        self.tokens.clear();
    }
    fn validate(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "target head rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.hc
                .pre(self.b(0), self.b(1), self.b(2), rows, self.stream.raw)?;
            self.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.b(2),
                self.weights.norm.get("norm.weight")?,
                self.b(3),
                rows as i32,
                5120,
                1e-20,
                self.stream.raw,
            )?;
            self.projection.launch(
                self.b(3),
                self.head.weight()?,
                self.b(4),
                rows,
                self.stream.raw,
            )?;
            Ok(())
        }
    }
    /// # Safety
    /// Residual and pre inputs are finite and initialized, with exclusive storage
    /// until the call drains. Raw outputs carry no block identity.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same initialized-input contract as execute; capture drains its warmup.
    pub unsafe fn capture(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(self.graph.is_none(), "target head graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready(rows) }
    }
    unsafe fn capture_ready(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same inputs as execute; live row count must match the captured graph.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let (graph, count) = self.graph.context("target head graph missing")?;
        ensure!(rows == count, "target head captured rows differ");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            offset <= b.bytes && bytes <= b.bytes - offset,
            "target head row slice exceeds buffer"
        );
        b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
        b.bytes = bytes;
        Ok(b)
    }
    /// # Safety
    /// Completed layer-39 residual/pre storage remains immutable until all row
    /// copies drain. Selection order defines output order; each row occurs once.
    unsafe fn copy_block(
        &mut self,
        block: &BlockOutput<'_>,
        selected: &[usize],
    ) -> Result<()> {
        self.invalidate();
        ensure!(
            block.layer == 39
                && block.binding().layer() == 39
                && block.tokens.len() <= 4096
                && !selected.is_empty()
                && selected.len() <= self.capacity
                && selected.iter().all(|&i| i < block.tokens.len())
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len()
                && block.residual.bytes == block.tokens.len() * 40960
                && block.pre.bytes == block.tokens.len() * 16
                && block.residual.device_id == self.b(0).device_id
                && block.pre.device_id == self.b(0).device_id,
            "target head block or selected rows differ"
        );
        let copied = (|| -> Result<()> {
            let mut first = 0;
            while first < selected.len() {
                let mut count = 1;
                while first + count < selected.len()
                    && selected[first + count] == selected[first] + count
                {
                    count += 1;
                }
                for (i, src, stride) in [(0, block.residual, 40960), (1, block.pre, 16)] {
                    unsafe {
                        self.stream.library.copy_d2d_async(
                            Self::slice(self.b(i), first * stride, count * stride)?,
                            Self::slice(src, selected[first] * stride, count * stride)?,
                            count * stride,
                            self.stream.raw,
                        )?;
                    }
                }
                first += count;
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.synchronize()?;
            return Err(error);
        }
        Ok(())
    }
    unsafe fn capture_block_head(&mut self, rows: usize) -> Result<()> {
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            self.clear_graph()?;
            unsafe { self.capture(rows)?; }
        }
        Ok(())
    }
    async unsafe fn prepare_head_cooperative(&mut self, rows: usize) -> Result<()> {
        // copy_block has already queued inputs on this stream. Warmup consumes
        // them in order, then capture executes without suspension.
        let launched = unsafe { self.enqueue(rows) };
        let drained = self.stream.wait().await;
        launched.and(drained)?;
        self.clear_graph()?;
        unsafe { self.capture_ready(rows) }
    }
    fn publish_block(&mut self, block: &BlockOutput<'_>, selected: &[usize]) {
        self.ready = Some(selected.len());
        self.origin = Some(block.binding());
        self.selected.extend_from_slice(selected);
        self.tokens.extend(selected.iter().map(|&i| block.tokens[i]));
    }
    /// Completed block inputs remain immutable through the drained head pass.
    pub unsafe fn execute_block(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        self.synchronize()?;
        unsafe { self.capture_block_head(selected.len())?; self.replay(selected.len())?; }
        self.publish_block(block, selected);
        self.output()
    }
    /// Same ownership contract, yielding the owner thread during GPU completion.
    /// First-use warmup and steady replay both complete cooperatively.
    pub async unsafe fn execute_block_cooperative(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        // Warmup and replay consume input copies on this same stream.
        if self.graph.as_ref().is_none_or(|(_, n)| *n != selected.len()) {
            unsafe { self.prepare_head_cooperative(selected.len()).await?; }
        }
        self.invalidate();
        let graph = self.graph.context("target head graph missing")?.0;
        let launched = unsafe { self.stream.library.cuda_graph_launch(graph, self.stream.raw) };
        let drained = self.stream.wait().await;
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.output()
    }
    /// Keep logits on the device, downloading only checked greedy IDs/scores.
    pub async unsafe fn execute_block_greedy(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], cooperative: bool) -> Result<()> {
        unsafe { self.copy_block(block, selected)?; }
        let rows = selected.len();
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            if cooperative { unsafe { self.prepare_head_cooperative(rows).await?; } }
            else { self.synchronize()?; unsafe { self.capture_block_head(rows)?; } }
        }
        self.invalidate();
        let graph = self.graph.context("target head graph missing")?.0;
        let logits = Self::slice(self.b(4), 0, rows * STRIDES[4])?;
        let indices = Self::slice(self.b(5), 0, rows * 4)?;
        let scores = Self::slice(self.b(6), 0, rows * 4)?;
        let launched = (|| -> Result<()> { unsafe {
            let lib = self.stream.library;
            lib.cuda_graph_launch(graph, self.stream.raw)?;
            lib.cuda_logits_argmax_checked_f32_async(logits, indices, scores, rows, 129280, self.stream.raw)?;
            let host = self.greedy_staging.bytes_mut();
            lib.copy_d2h_async(&mut host[..rows*4], indices, self.stream.raw)?;
            lib.copy_d2h_async(&mut host[rows*4..rows*8], scores, self.stream.raw)?;
            Ok(())
        } })();
        let drained = if cooperative { self.stream.wait().await } else { self.synchronize() };
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.greedy_ready = true;
        Ok(())
    }
    /// Run the head and then the v4.1 target-sampler (K1) on the same stream,
    /// downloading only `rows × (4 + 4 + 4 + 4)` bytes (ids, raw scores, status,
    /// status detail) plus, in a second call, the rows the caller asks for.
    ///
    /// This call downloads no logits: the caller decides which rows still need
    /// them and asks for exactly those through
    /// [`Self::download_sampled_rows`] (none, for an untraced all-greedy round;
    /// every row, for a traced one). The compact greedy lane
    /// ([`Self::execute_block_greedy`]) is untouched.
    ///
    /// Enqueue order is `copy_block` → graph launch → sampler → the small D2H,
    /// all before the drain the scheduler already performs, so no new host
    /// synchronization is introduced (design §11.2).
    pub async unsafe fn execute_block_sampled(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], requests: &[TargetSamplingRowRequest], mask_staging: Option<&[u32]>,
        mask_words: usize, cooperative: bool) -> Result<SampledTargetRows> {
        ensure!(
            requests.len() == selected.len(),
            "sampling requests must match the selected rows"
        );
        unsafe { self.copy_block(block, selected)?; }
        let rows = selected.len();
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            if cooperative { unsafe { self.prepare_head_cooperative(rows).await?; } }
            else { self.synchronize()?; unsafe { self.capture_block_head(rows)?; } }
        }
        self.invalidate();
        let graph = self.graph.context("target head graph missing")?.0;
        let logits = Self::slice(self.b(4), 0, rows * STRIDES[4])?;
        let staged = self
            .sampling
            .upload(requests, mask_staging, mask_words, self.stream.raw);
        let launched = staged.and_then(|()| {
            let launched = (|| -> Result<()> { unsafe {
                self.stream.library.cuda_graph_launch(graph, self.stream.raw)?;
                self.sampling.launch(logits, rows, self.stream.raw)
            } })();
            launched
        });
        let drained = if cooperative { self.stream.wait().await } else { self.synchronize() };
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.sampling.output(logits, rows)
    }
    /// Download the full logits of exactly the rows the caller names, packed in
    /// the same order.
    ///
    /// Chunk-1 callers name a row here for exactly two reasons: the round is
    /// traced (`SamplingRound::trace_rows` downloads every row so
    /// `ds41rt::logit_trace` can log `top_two`), or the row is a stochastic
    /// member of a CPU-path round. An untraced all-greedy round names none, so
    /// on that path a greedy row costs only its 4-byte id/score/status entries
    /// instead of a 517,120-byte row transfer (design §5.5, §8.2). A round
    /// containing any stochastic member still downloads every member's rows,
    /// greedy ones included; that qualifier is chunk 2's to remove.
    pub async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        ensure!(!selection.is_empty(), "sampled row download selection is empty");
        unsafe { self.download.rows(rows.logits, STRIDES[4], selection).await }
    }
    pub fn greedy_output(&mut self) -> Result<Vec<(u32, f32)>> {
        ensure!(self.greedy_ready, "compact head output unpublished");
        let rows = self.ready.context("compact head rows unpublished")?;
        let host = self.greedy_staging.bytes_mut();
        Ok((0..rows).map(|i| (
            u32::from_ne_bytes(host[i*4..i*4+4].try_into().unwrap()),
            f32::from_ne_bytes(host[rows*4+i*4..rows*4+i*4+4].try_into().unwrap()),
        )).collect())
    }
    pub fn output(&self) -> Result<TargetLogits<'_>> {
        let rows = self.ready.context("target logits unpublished")?;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * STRIDES[i];
            b
        };
        Ok(TargetLogits {
            rows,
            collapsed: b(2),
            normalized: b(3),
            logits: b(4),
            selected_rows: &self.selected,
            token_positions: &self.tokens,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    pub async fn download_rows(&mut self, rows: &[usize]) -> Result<Vec<u8>> {
        let logits = self.output()?.logits;
        // The head remains exclusively borrowed through transfer completion.
        unsafe { self.download.rows(logits, STRIDES[4], rows).await }
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.stream.require_complete()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for TargetHeadWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e,"draining target head");
        }
    }
}

#[cfg(test)]
mod sampling_budget_tests {
    use super::*;

    /// D-new: an all-unconstrained round stages no mask rows, and the launch
    /// pair must then be `(None, 0)`. Passing the wave's own non-zero
    /// `mask_words` with a null arena is what the FFI validator and the kernel
    /// both reject, which failed a whole lane for any all-greedy round that was
    /// not compact (a traced round, or a constrained member whose grammar needs
    /// no mask at any row).
    #[test]
    fn an_all_unconstrained_round_launches_without_a_mask_or_a_mask_width() {
        let device = Ds41rtDeviceBuffer {
            ptr: 0x1000 as *mut std::ffi::c_void,
            bytes: 8 * SAMPLING_MASK_WORDS * 4,
            ..Default::default()
        };
        // No staged rows: no buffer and a zero width.
        let (masks, words) = mask_launch_arguments(0, device, SAMPLING_MASK_WORDS);
        assert!(masks.is_none(), "an unstaged round must launch no mask buffer");
        assert_eq!(words, 0, "an unstaged round must launch mask_words_per_row = 0");

        // Staged rows: the buffer and the wave's width, unchanged.
        let (masks, words) = mask_launch_arguments(3, device, SAMPLING_MASK_WORDS);
        let masks = masks.expect("a staged round launches its mask buffer");
        assert_eq!(masks.ptr, device.ptr);
        assert_eq!(masks.bytes, device.bytes);
        assert_eq!(words, SAMPLING_MASK_WORDS);
        assert_eq!(words, 4040, "the shipped mask width");
    }

    /// The sampler's workspace is charged to the head wave so capacity
    /// budgeting stays honest (design §5.5, §11.1). The formula is pinned here
    /// so a field added without updating `device_bytes` fails loudly.
    #[test]
    fn sampling_workspace_bytes_follow_the_documented_formula() {
        for capacity in [1usize, 6, 48, 64, 80] {
            let param = capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_PARAM_BYTES;
            let arena = capacity * SAMPLING_MASK_WORDS * 4;
            let scratch = capacity * ds41rt_ffi::DS41RT_V41_SAMPLER_SCRATCH_BYTES;
            let histogram = capacity * SAMPLING_HISTOGRAM_BUCKETS * 4;
            let rows = capacity * 4;
            let expected = param + arena + scratch + arena + histogram + rows + rows + rows
                + rows + capacity * 8;
            assert_eq!(TargetSamplingWave::device_bytes(capacity), expected);
            // Pinned call sites: the production capacities are 48 and 64.
        }
        // The documented §11.1 total at capacity 80 is ~4.06 MiB; the arena is
        // allocated twice (device + top-k bitmap) and the histogram once.
        assert_eq!(TargetSamplingWave::device_bytes(80), 80 * 64 + 80 * 4040 * 4 + 80 * 64
            + 80 * 4040 * 4 + 80 * 2048 * 4 + 80 * 4 + 80 * 4 + 80 * 4 + 80 * 4 + 80 * 8);
        assert!(TargetSamplingWave::device_bytes(80) < 5 * 1024 * 1024);
    }
}
