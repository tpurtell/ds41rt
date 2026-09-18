//! Per-wave stable expert I/O, scratch, stream and graph ownership.
use super::{DeviceAllocation, ExpertWeights, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41CompactReducer, V41ExpertKernel, V41ExpertLaunchArgs,
    V41RouteReducer,
};
use std::ffi::c_void;
mod timing;
use timing::ExpertTiming;
use super::ExpertLayer;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertExecutionBudget {
    pub scratch_bytes: usize,
    pub decode_scratch_bytes: usize,
    pub small_scratch_bytes: usize,
    pub hidden_bytes: usize,
    pub routing_bytes: usize,
    pub output_and_shared_bytes: usize,
}
impl ExpertExecutionBudget {
    pub fn total(self) -> Result<usize> {
        [
            self.scratch_bytes,
            self.decode_scratch_bytes,
            self.small_scratch_bytes,
            self.hidden_bytes,
            self.routing_bytes,
            self.output_and_shared_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |sum, n| {
            sum.checked_add(n)
                .context("expert execution budget overflow")
        })
    }
}

/// Dedicated decode state shares the wave's immutable weights and input buffers.
/// Its arena and kernel are initialized before serving or graph capture.
struct DecodeExecution<'library> {
    kernel: V41ExpertKernel<'library>,
    scratch: DeviceAllocation<'library>,
    slots: [*mut c_void; 44],
}

/// One exclusive wave's buffers and graph borrow immutable resident weights.
/// Upstream producers and transport must obey this owner's stream/lifetime contract.
pub(crate) struct ExpertExecution<'weights, 'library> {
    // Destroy stream before allocations; Drop drains and destroys the graph first.
    stream: LoadStream<'library>,
    _weights: &'weights ExpertWeights<'library>,
    library: &'library NativeLibrary,
    kernel: V41ExpertKernel<'library>,
    decode: Option<DecodeExecution<'library>>,
    small: Option<DecodeExecution<'library>>,
    reducer: V41RouteReducer<'library>,
    timing: Option<ExpertTiming<'library>>,
    scratch: DeviceAllocation<'library>,
    hidden: DeviceAllocation<'library>,
    ids: DeviceAllocation<'library>,
    routing: DeviceAllocation<'library>,
    compact_reducer: Option<V41CompactReducer<'library>>,
    compact_output: Option<DeviceAllocation<'library>>,
    output: Option<DeviceAllocation<'library>>,
    shared: Option<DeviceAllocation<'library>>,
    slots: [*mut c_void; 44],
    graph: Option<(*mut c_void, u32, bool)>,
    budget: ExpertExecutionBudget,
}
impl<'library> ExpertWeights<'library> {
    pub fn execution_budget(&self, capacity: u32) -> Result<ExpertExecutionBudget> {
        let library = self.buffers[0].library;
        Self::plan_execution(self.layer, library, capacity, self.is_nvfp4())
    }
    pub(super) fn plan_execution(
        layer: ExpertLayer,
        library: &NativeLibrary,
        capacity: u32,
        nvfp4: bool,
    ) -> Result<ExpertExecutionBudget> {
        let info = layer.select_info(library, capacity, nvfp4)?;
        let hidden = (capacity as usize)
            .checked_mul(info.input_row_bytes()?)
            .context("hidden buffer overflow")?;
        let routing = (capacity as usize)
            .checked_mul(info.topk as usize * 8)
            .context("routing buffer overflow")?;
        Ok(ExpertExecutionBudget {
            scratch_bytes: usize::try_from(info.scratch_bytes)?,
            decode_scratch_bytes: if info.role == 1 && capacity > 1 {
                usize::try_from(library.v41_expert_info(1)?.scratch_bytes)?
            } else {
                0
            },
            small_scratch_bytes: if info.role == 1 && capacity > 80 {
                usize::try_from(library.v41_expert_info(80)?.scratch_bytes)?
            } else {
                0
            },
            hidden_bytes: hidden,
            routing_bytes: routing,
            output_and_shared_bytes: (capacity as usize)
                .checked_mul(5120 * 2 * if info.role == 0 { 2 } else { 1 })
                .context("output buffer overflow")?,
        })
    }
    pub fn execution(
        &self,
        capacity: u32,
        available_device_bytes: usize,
    ) -> Result<ExpertExecution<'_, 'library>> {
        let library = self.buffers[0].library;
        let budget = self.execution_budget(capacity)?;
        ensure!(
            budget.total()? <= available_device_bytes,
            "expert execution buffers exceed device budget"
        );
        let kernel = self.layer.select_kernel(library, capacity, self.is_nvfp4())?;
        let reducer = library.v41_route_reducer()?;
        let timing = if kernel.info().role == 1
            && tracing::enabled!(target: "ds41rt::expert_timing", tracing::Level::DEBUG)
        {
            Some(ExpertTiming::new(library)?)
        } else {
            None
        };
        let scratch = DeviceAllocation::new(library, budget.scratch_bytes)?;
        let hidden = DeviceAllocation::new(library, budget.hidden_bytes)?;
        let ids = DeviceAllocation::new(library, budget.routing_bytes / 2)?;
        let routing = DeviceAllocation::new(library, budget.routing_bytes / 2)?;
        let output = if kernel.info().role == 0 {
            Some(DeviceAllocation::new(library, budget.hidden_bytes)?)
        } else {
            None
        };
        let compact_output = if kernel.info().role == 1 {
            Some(DeviceAllocation::new(
                library,
                budget.output_and_shared_bytes,
            )?)
        } else {
            None
        };
        let compact_reducer = if kernel.info().role == 1 {
            Some(library.v41_compact_reducer()?)
        } else {
            None
        };
        let shared = if kernel.info().role == 0 {
            Some(DeviceAllocation::new(library, budget.hidden_bytes)?)
        } else {
            None
        };
        let mut slots = [std::ptr::null_mut(); 44];
        unsafe {
            kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?;
        }
        self.bind(&kernel, &mut slots)?;
        slots[0] = hidden.buffer.ptr;
        slots[1] = ids.buffer.ptr;
        slots[2] = routing.buffer.ptr;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            kernel.initialize_scratch(
                scratch.buffer.ptr,
                scratch.buffer.bytes as u64,
                stream.raw,
            )?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        let prepare_small = |planned_capacity, scratch_bytes| -> Result<Option<DecodeExecution<'library>>> {
            if scratch_bytes == 0 { return Ok(None); }
            let decode_kernel =
                self.layer.select_kernel(library, planned_capacity, self.is_nvfp4())?;
            ensure!(
                decode_kernel.info().input_dtype == kernel.info().input_dtype,
                "decode and grouped expert input formats differ"
            );
            let decode_scratch = DeviceAllocation::new(library, scratch_bytes)?;
            let mut decode_slots = slots;
            unsafe {
                decode_kernel.bind_scratch(
                    decode_scratch.buffer.ptr,
                    decode_scratch.buffer.bytes as u64,
                    &mut decode_slots,
                )?;
            }
            self.bind(&decode_kernel, &mut decode_slots)?;
            unsafe {
                let initialized = decode_kernel.initialize_scratch(
                    decode_scratch.buffer.ptr,
                    decode_scratch.buffer.bytes as u64,
                    stream.raw,
                );
                // Drain even on initialization failure before releasing its arena.
                let drained = library.cuda_stream_synchronize(stream.raw);
                initialized.and(drained)?;
            }
            Ok(Some(DecodeExecution {
                kernel: decode_kernel,
                scratch: decode_scratch,
                slots: decode_slots,
            }))
        };
        let decode = prepare_small(1, budget.decode_scratch_bytes)?;
        let small = prepare_small(80, budget.small_scratch_bytes)?;
        Ok(ExpertExecution {
            stream,
            _weights: self,
            library,
            kernel,
            decode,
            small,
            reducer,
            timing,
            compact_reducer,
            compact_output,
            scratch,
            hidden,
            ids,
            routing,
            output,
            shared,
            slots,
            graph: None,
            budget,
        })
    }
}
impl<'weights, 'library> ExpertExecution<'weights, 'library> {
    /// Reuse a wave's workspace across resident layers after its prior work drains.
    /// Captured graphs retain weight addresses and cannot be rebound.
    pub fn bind_layer(&mut self, weights: &'weights ExpertWeights<'library>) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "cannot rebind a captured expert graph"
        );
        ensure!(
            std::ptr::eq(self.library, weights.buffers[0].library),
            "expert layer belongs to a different native library"
        );
        self.synchronize()?;
        let mut slots = self.slots;
        weights.bind(&self.kernel, &mut slots)?;
        for decode in [&mut self.decode, &mut self.small].into_iter().flatten() {
            let mut decode_slots = decode.slots;
            weights.bind(&decode.kernel, &mut decode_slots)?;
            decode.slots = decode_slots;
        }
        self.slots = slots;
        self._weights = weights;
        Ok(())
    }

    pub fn budget(&self) -> ExpertExecutionBudget {
        self.budget
    }
    pub fn stream(&self) -> *mut c_void {
        self.stream.raw
    }
    /// Borrowed hidden in the native kernel's input format, I32 route IDs and
    /// FP32 route weights; never free these views. dSpark always uses BF16.
    /// Enqueue producers on stream() or provide event ordering before launch/replay.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        [self.hidden.buffer, self.ids.buffer, self.routing.buffer]
    }
    pub fn shared(&self) -> Option<Ds41rtDeviceBuffer> {
        self.shared.as_ref().map(|b| b.buffer)
    }
    pub fn output(&self) -> Option<Ds41rtDeviceBuffer> {
        self.output.as_ref().map(|b| b.buffer)
    }
    /// Borrow the output state selected for a launch of exactly `rows` rows.
    /// A one-row view is not a prefix of the multi-row state's output.
    pub fn route_partials(&self, rows: u32) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid expert output row count"
        );
        let (kernel, slots, scratch) = self.execution_state(rows);
        ensure!(!kernel.accumulates_tokens(), "expert output is token accumulation, not route planes");
        Ok(Ds41rtDeviceBuffer {
            ptr: slots[41],
            bytes: rows as usize * kernel.info().topk as usize * 5120 * 4,
            device_id: scratch.buffer.device_id,
            flags: 0,
        })
    }
    fn execution_state(
        &self,
        rows: u32,
    ) -> (
        &V41ExpertKernel<'library>,
        &[*mut c_void; 44],
        &DeviceAllocation<'library>,
    ) {
        if rows == 1 {
            if let Some(decode) = &self.decode {
                return (&decode.kernel, &decode.slots, &decode.scratch);
            }
        }
        if let Some(small) = &self.small {
            if rows <= small.kernel.info().capacity_rows {
                return (&small.kernel, &small.slots, &small.scratch);
            }
        }
        (&self.kernel, &self.slots, &self.scratch)
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }

    /// Run routing, shared FFN and the three routed experts on one wave stream.
    /// The existing reducer adds shared BF16 output after expert accumulation.
    /// # Safety
    /// Hidden states must be finite and initialized with producer writes ordered
    /// on this stream; serialize wave use and finish external readers before reuse.
    pub unsafe fn ffn_draft(
        &mut self,
        router: &mut super::dspark::DsparkRouter<'_, '_>,
        shared: &mut super::dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        let launched = unsafe { self.enqueue_draft_ffn(router, shared, rows) };
        let drained = self.synchronize();
        launched.and(drained)
    }

    /// Caller must drain this wave's stream before releasing router/shared scratch.
    pub(super) unsafe fn enqueue_draft_ffn(
        &mut self,
        router: &mut super::dspark::DsparkRouter<'_, '_>,
        shared: &mut super::dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        unsafe { self.enqueue_draft_ffn_on(router, shared, rows, self.stream.raw) }
    }
    /// Containing owner must drain the supplied stream before releasing scratch.
    pub(super) unsafe fn enqueue_draft_ffn_on(
        &mut self,
        router: &mut super::dspark::DsparkRouter<'_, '_>,
        shared: &mut super::dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            router.matches(self._weights) && shared.matches(self._weights),
            "dSpark FFN stage owners differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid dSpark FFN rows"
        );
        let output = self
            .shared
            .as_ref()
            .context("dSpark FFN requires coordinator output")?
            .buffer;
        unsafe {
            router.enqueue(self.inputs(), rows as usize, stream)?;
            shared.enqueue(self.hidden.buffer, output, rows, stream)?;
            self.launch_on(rows, true, stream)
        }
    }

    /// Compute the dSpark shared expert directly into this wave's shared output.
    /// # Safety
    /// Hidden states must be finite and initialized, with producer writes ordered
    /// on this stream; external consumers must finish before output reuse.
    pub unsafe fn shared_draft(
        &mut self,
        shared: &mut super::dspark::DsparkSharedFfn<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        ensure!(
            shared.matches(self._weights),
            "shared FFN and expert stage weights differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid shared FFN rows"
        );
        let output = self
            .shared
            .as_ref()
            .context("shared FFN requires coordinator output")?
            .buffer;
        let launched = unsafe { shared.enqueue(self.hidden.buffer, output, rows, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)
    }

    /// Route the current hidden states through this exact dSpark stage's gate.
    /// Drains the stream before returning so the router scratch can be reused.
    /// # Safety
    /// Hidden states must be initialized with finite router logits on this device;
    /// external producers/readers must finish or be ordered on this stream.
    pub unsafe fn route_draft(
        &mut self,
        router: &mut super::dspark::DsparkRouter<'_, '_>,
        rows: u32,
    ) -> Result<()> {
        ensure!(
            router.matches(self._weights),
            "router and expert stage weights differ"
        );
        ensure!(
            rows > 0 && rows <= self.kernel.info().capacity_rows,
            "invalid router rows"
        );
        let launched = unsafe { router.enqueue(self.inputs(), rows as usize, self.stream.raw) };
        let drained = self.synchronize();
        launched.and(drained)
    }

    /// # Safety
    /// Initialize hidden in the kernel-advertised input format, in-range I32
    /// expert IDs and finite nonnegative
    /// FP32 routing weights for `rows` before this operation, with stream ordering.
    /// Initialize shared BF16 output too if requested. External readers/writers must
    /// finish before storage is reused; all operations belong to this GPU worker.
    pub unsafe fn launch(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        unsafe { self.launch_on(rows, include_shared, self.stream.raw) }
    }
    unsafe fn launch_on(
        &mut self,
        rows: u32,
        include_shared: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            !include_shared || self.shared.is_some(),
            "shared output belongs on coordinator RTX"
        );
        let (kernel, slots, _) = self.execution_state(rows);
        let args = V41ExpertLaunchArgs::new(kernel.info(), *slots, rows, stream)?;
        unsafe {
            kernel.launch(&args)?;
        }
        if let Some(output) = &self.output {
            unsafe {
                self.reducer.launch(
                    [
                        slots[41].cast(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    ],
                    if include_shared {
                        self.shared.as_ref().unwrap().buffer.ptr.cast()
                    } else {
                        std::ptr::null()
                    },
                    output.buffer.ptr.cast(),
                    rows,
                    1,
                    3,
                    stream,
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-buffer and ordering contract as launch; captures fixed rows
    /// and shared-output participation. Input values may change between replays.
    pub unsafe fn capture(&mut self, rows: u32, include_shared: bool) -> Result<()> {
        ensure!(
            self.graph.is_none(),
            "expert execution already has a captured graph"
        );
        unsafe {
            self.launch(rows, include_shared)?;
        }
        self.synchronize()?;
        unsafe {
            self.library.cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.launch(rows, include_shared) };
        // End capture on both paths so an error cannot leave the stream capturing.
        let captured = unsafe { self.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows, include_shared));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) => Err(error),
            (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// The captured rows' inputs (and optional shared output) must satisfy launch's
    /// contract, with writes ordered before replay and reads ordered after it.
    pub unsafe fn replay(&mut self) -> Result<()> {
        let (graph, _, _) = self.graph.context("expert graph has not been captured")?;
        unsafe { self.library.cuda_graph_launch(graph, self.stream.raw) }
    }
}
impl Drop for ExpertExecution<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining V4.1 expert execution");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying V4.1 expert execution graph");
            }
        }
    }
}

/// Reusable host exchange for the TCP fallback; RDMA can consume device route views.
pub(crate) struct HostExpertExchange {
    pub(super) ids: Vec<i32>,
    pub(super) routing: Vec<f32>,
    pub(super) partials: Vec<u8>,
}
impl HostExpertExchange {
    pub fn new(capacity: u32) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "unsupported native host exchange capacity"
        );
        let routes = capacity as usize * 6;
        Ok(Self {
            ids: vec![0; routes],
            routing: vec![0.0; routes],
            partials: vec![0; capacity as usize * 5120 * 2],
        })
    }
}
impl ExpertExecution<'_, '_> {
    /// Execute once and deliver bounded frames using caller-owned index scratch.
    /// The sink must finish consuming each borrowed chunk before returning.
    /// A send failure aborts the wave; callers must not retry it on the same
    /// partially delivered response stream without resetting receiver state.
    pub fn execute_host_chunks<F>(
        &mut self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        mut sink: F,
    ) -> Result<()>
    where
        F: FnMut(ds41rt_transport::ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(
            row_indices.len() >= chunk_rows as usize,
            "response row-index scratch is too short"
        );
        let response = self.execute_host_request(request, executor_id, exchange)?;
        let stride = ds41rt_transport::v41_expert::V41_PARTIAL_ROW_BYTES as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            let payload =
                &response.partial_output_payload[start as usize * stride..end as usize * stride];
            sink(request.response_chunk(
                executor_id,
                start,
                payload,
                row_indices,
                max_frame_bytes,
            )?)?;
        }
        Ok(())
    }

    /// Write the complete rank plane into the transport's registered send slot.
    /// Return None before execution when bounded frames or checksums need fallback.
    /// # Safety
    /// The slot is GPU-accessible on this device and exclusively owned until the
    /// caller emits the returned response; the transport retains it through send completion.
    pub unsafe fn execute_mapped_request(&mut self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, slot: Ds41rtDeviceBuffer,
    ) -> Result<Option<ds41rt_transport::ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = ds41rt_transport::EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes { return Ok(None); }
        let output = Ds41rtDeviceBuffer {
            ptr: unsafe { slot.ptr.cast::<u8>().add(prefix).cast() }, bytes, ..slot
        };
        // Validate the destination descriptor before launching any GPU work.
        let response = request.response_device(executor_id, output)?;
        self.execute_request_output(request, executor_id, exchange, Some(output))?;
        Ok(Some(response))
    }

    /// Host fallback from a validated wire request to a compact BF16 rank response.
    /// The borrowed response prevents reuse of exchange storage until encoding/send ends.
    pub fn execute_host_request<'a>(
        &mut self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &'a mut HostExpertExchange,
    ) -> Result<ds41rt_transport::ExpertProtocolV2ResponseRef<'a>> {
        self.execute_request_output(request, executor_id, exchange, None)?;
        request.response(executor_id, &exchange.partials[..request.plane_bytes()?])
    }

    fn execute_request_output(&mut self,
        request: &ds41rt_transport::v41_expert::V41BackboneRequest<'_>, executor_id: u64,
        exchange: &mut HostExpertExchange, destination: Option<Ds41rtDeviceBuffer>,
    ) -> Result<()> {
        let super::ExpertLayer::Backbone { layer, .. } = self._weights.layer else {
            anyhow::bail!("backbone requests cannot execute on RTX dSpark weights");
        };
        ensure!(
            request.layer() as usize == layer,
            "request does not match resident expert layer"
        );
        ensure!(executor_id != 0, "native response needs executor identity");
        ensure!(
            request.rows() <= self.kernel.info().capacity_rows,
            "request exceeds native execution capacity"
        );
        request.require_input_dtype(self.kernel.info().input_dtype)?;
        let routes = request.rows() as usize * 6;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "host exchange is too small"
        );
        request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        let started = self.timing.as_ref().map(|_| std::time::Instant::now());
        self.synchronize()?;
        self.library
            .copy_h2d(self.hidden.buffer, request.hidden())?;
        // Supported hosts and CUDA use the checkpoint/wire little-endian byte order.
        ensure!(
            cfg!(target_endian = "little"),
            "native host exchange requires little-endian storage"
        );
        unsafe {
            self.library.copy_h2d(
                self.ids.buffer,
                std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4),
            )?;
            self.library.copy_h2d(
                self.routing.buffer,
                std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4),
            )?;
        }
        let uploaded_us = started.map(|t| t.elapsed().as_micros() as u64);
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(0, self.stream.raw)?;
            }
        }
        unsafe {
            self.launch(request.rows(), false)?;
        }
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(1, self.stream.raw)?;
            }
        }
        let output = destination.unwrap_or(self
            .compact_output
            .as_ref()
            .context("missing compact output")?
            .buffer);
        unsafe {
            let reducer = self.compact_reducer.as_ref().context("missing compact reducer")?;
            let (kernel, slots, _) = self.execution_state(request.rows());
            if kernel.accumulates_tokens() {
                reducer.compact_tokens(slots[41].cast(), output.ptr.cast(), request.rows(), self.stream.raw)?;
            } else {
                reducer.compact(self.route_partials(request.rows())?.ptr.cast(),
                    output.ptr.cast(), request.rows(), self.stream.raw)?;
            }
        }
        if let Some(timing) = &self.timing {
            unsafe {
                timing.record(2, self.stream.raw)?;
            }
        }
        self.synchronize()?;
        let executed_us = started.map(|t| t.elapsed().as_micros() as u64);
        if destination.is_none() {
            self.library.copy_d2h(&mut exchange.partials[..bytes], Ds41rtDeviceBuffer { bytes, ..output })?;
        }
        if let (Some(timing), Some(started)) = (&self.timing, started) {
            let total_us = started.elapsed().as_micros() as u64;
            let (kernel_us, compact_us) = unsafe { timing.elapsed_us()? };
            let mut histogram = [0u32; 384];
            for &expert in &exchange.ids[..routes] {
                histogram[expert as usize] += 1;
            }
            let active_experts = histogram.iter().filter(|&&count| count != 0).count();
            // Exact expert-local M=1..16; final bin is M>=17. These counts
            // describe routed rows before the kernel pads them to its M tile.
            let mut expert_rows_histogram = [0u32; 17];
            let mut expert_rows_tail_routes = 0u32;
            for &count in &histogram {
                if count > 0 {
                    expert_rows_histogram[count.min(17) as usize - 1] += 1;
                    if count > 16 {
                        expert_rows_tail_routes += count;
                    }
                }
            }
            let unique_expert_weight_bytes =
                self._weights.budget().resident_bytes / 384 * active_experts;
            tracing::debug!(target: "ds41rt::expert_timing",
                layer, executor_id, rows=request.rows(), active_experts, direct_registered_output=destination.is_some(),
                kernel_capacity=self.execution_state(request.rows()).0.info().capacity_rows,
                max_expert_rows=histogram.iter().copied().max().unwrap_or(0),
                ?expert_rows_histogram, expert_rows_tail_routes,
                unique_expert_weight_bytes, output_bytes=bytes,
                upload_us=uploaded_us.unwrap(), kernel_us, compact_us,
                execution_host_us=executed_us.unwrap()-uploaded_us.unwrap(),
                download_us=total_us-executed_us.unwrap(), total_us,
                "native expert execution");
        }
        Ok(())
    }
}

#[cfg(test)]
mod mapped_tests;
