//! One coordinator wave owns TP route planes through final native reduction.
use super::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41CompactReducer};
use ds41rt_transport::{
    v41_expert::{V41Tp4RocePending, V41Tp4Roce, V41_PARTIAL_ROW_BYTES},
    ExpertProtocolV2Request, VerbsHostProtocolV2ResponsePayload,
};

pub(crate) struct NativeTp4Wave<'a> {
    transport: V41Tp4Roce,
    // Drop drains the stream before fields release any GPU allocations.
    stream: LoadStream<'a>,
    planes: [DeviceAllocation<'a>; 4],
    upload_frames: Vec<VerbsHostProtocolV2ResponsePayload>,
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    reducer: V41CompactReducer<'a>,
    ready_rows: Option<u32>,
    local: Option<super::local::LocalExpertWave<'a>>,
    tp2: Option<Box<super::tp2_ffn::Wave<'a>>>,
    paired: Option<Box<(super::paired::PairedAssignment, std::rc::Rc<super::paired::PairedProfile>)>>,
}
impl<'a> NativeTp4Wave<'a> {
    pub(crate) fn install_paired(&mut self, profile: std::rc::Rc<super::paired::PairedProfile>) -> Result<()> {
        ensure!(self.paired.is_none(), "paired assignment already installed");
        self.paired = Some(Box::new((super::paired::PairedAssignment::new(), profile)));
        Ok(())
    }
    pub(crate) fn prepare_remote_request(&mut self, request: &mut crate::v41_backbone_router::BoundExpertRequest) -> Result<()> {
        if let Some(paired) = &mut self.paired {
            request.assign_paired(&mut paired.0, &paired.1)?;
        }
        Ok(())
    }

    pub fn install_local(&mut self, wave: super::local::LocalExpertWave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.local = Some(wave);
        Ok(())
    }
    pub fn has_local_layer(&self, layer: usize) -> bool {
        self.local.as_ref().is_some_and(|wave| wave.contains(layer)) || self.has_tp2_layer(layer)
    }
    pub fn install_tp2(&mut self, wave: super::tp2_ffn::Wave<'a>) -> Result<()> {
        ensure!(self.local.is_none() && self.tp2.is_none(), "local expert lane already installed");
        self.tp2 = Some(Box::new(wave));
        Ok(())
    }
    pub fn has_tp2_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains(layer))
    }
    pub fn has_tp2_shared_layer(&self, layer: usize) -> bool {
        self.tp2.as_ref().is_some_and(|wave| wave.contains_shared(layer))
    }
    /// # Safety
    /// Input and router producers have completed. Both borrowed outputs remain
    /// immutable through this operation, including cancellation draining.
    pub async unsafe fn execute_tp2_ffn(&mut self, input: &crate::v41_block::FfnInput<'_>,
        routed: &crate::v41_backbone_router::RouterOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        ensure!(binding == input.binding() && input.layer == routed.layer
            && input.tokens.len() == routed.rows as usize && input.tokens == routed.tokens,
            "TP2 FFN input/router identity mismatch");
        let values = unsafe { self.tp2.as_mut().context("TP2 expert lane missing")?
            .execute(input.layer, routed.rows, input.values, routed.expert_input, routed.ids, routed.routing).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared buffers stay live and unmodified through drain.
    pub unsafe fn execute_local_ffn(&mut self,
        routed: &crate::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?.execute(routed, shared)? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding: routed.binding()?, _owner: std::marker::PhantomData })
    }
    /// # Safety
    /// Completed router/shared inputs remain immutable until completion or drain.
    pub async unsafe fn execute_local_ffn_cooperative(&mut self,
        routed: &crate::v41_backbone_router::RouterOutput<'_>,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'_>> {
        self.ready_rows = None;
        let binding = routed.binding()?;
        let values = unsafe { self.local.as_mut().context("local expert lane missing")?
            .execute_cooperative(routed, shared).await? };
        self.ready_rows = Some(routed.rows);
        Ok(NativeFfnOutput { values, binding, _owner: std::marker::PhantomData })
    }
    /// Admission invalidates prior output; completed RoCE sessions remain reusable.
    pub fn begin_request(&mut self) {
        self.ready_rows = None;
    }
    pub fn reset_connections(&mut self) {
        self.ready_rows = None;
        self.transport.reset_connections();
    }
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid native TP wave capacity"
        );
        (capacity as usize)
            .checked_mul(4 * V41_PARTIAL_ROW_BYTES as usize + 2 * 5120 * 2)
            .context("native TP wave budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        transport: V41Tp4Roce,
        available_bytes: usize,
    ) -> Result<Self> {
        let capacity = transport.capacity();
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "native TP wave exceeds device budget"
        );
        let reducer = library.v41_compact_reducer()?;
        let plane_bytes = capacity as usize * V41_PARTIAL_ROW_BYTES as usize;
        let mut planes = Vec::with_capacity(4);
        for _ in 0..4 {
            planes.push(DeviceAllocation::new(library, plane_bytes)?);
        }
        let planes = planes.try_into().ok().expect("four native TP planes");
        let hidden_bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            transport,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            planes,
            upload_frames: Vec::with_capacity(capacity as usize * 4),
            shared: DeviceAllocation::new(library, hidden_bytes)?,
            output: DeviceAllocation::new(library, hidden_bytes)?,
            library,
            reducer,
            ready_rows: None,
            local: None,
            tp2: None,
            paired: None,
        })
    }
    /// RoCE execution with optional host BF16 shared-expert contribution.
    /// All GPU copies finish before frame storage can be reused, and reduction
    /// finishes before the borrowed output view is exposed. Cancellation leaves
    /// output unavailable until an entirely successful subsequent execution.
    pub async fn execute(
        &mut self,
        request: &ExpertProtocolV2Request,
        shared: Option<&[u8]>,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        self.synchronize()?;
        let rows = request.header.row_count;
        ensure!(
            rows > 0 && rows <= self.transport.capacity(),
            "native wave exceeds capacity"
        );
        let hidden_bytes = rows as usize * 5120 * 2;
        if let Some(shared) = shared {
            ensure!(
                shared.len() == hidden_bytes,
                "native shared output has wrong BF16 geometry"
            );
            self.library.copy_h2d(self.shared.buffer, shared)?;
        }
        self.execute_prepared(request, shared.is_some()).await
    }
    /// # Safety
    /// Shared device values are complete and immutable through the copy. The
    /// request and shared result must derive from the same actual block input.
    pub async unsafe fn execute_ffn<'w>(
        &'w mut self,
        request: &crate::v41_backbone_router::BoundExpertRequest,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        self.ready_rows = None;
        validate_shared(
            request,
            shared,
            self.shared.buffer,
            self.transport.capacity(),
        )?;
        unsafe { self.dispatch_ffn(request).await?.finish(shared).await }
    }
    /// Enqueue all four expert requests before returning. The caller can then run
    /// shared FFN work on RTX while the Spark workers execute the routed experts.
    pub async fn dispatch_ffn<'w, 'r>(
        &'w mut self,
        request: &'r crate::v41_backbone_router::BoundExpertRequest,
    ) -> Result<NativePendingFfn<'w, 'a, 'r>> {
        self.ready_rows = None;
        // Previous output or cancellation cleanup completed this wave.
        self.stream.require_complete()?;
        let header = &request.request().header;
        ensure!(
            header.layer_id as usize == request.binding().layer()
                && header.row_count > 0
                && header.row_count <= self.transport.capacity(),
            "native dispatched FFN identity or rows differ"
        );
        let capacity = self.transport.capacity();
        let pending = self.transport.dispatch(request.request()).await?;
        Ok(NativePendingFfn {
            pending,
            request,
            capacity,
            library: self.library,
            stream: &self.stream,
            planes: &self.planes,
            upload_frames: &mut self.upload_frames,
            shared: self.shared.buffer,
            output: self.output.buffer,
            reducer: &self.reducer,
            ready_rows: &mut self.ready_rows,
            tp2: self.tp2.as_deref_mut(),
        })
    }

    async fn execute_prepared(
        &mut self,
        request: &ExpertProtocolV2Request,
        has_shared: bool,
    ) -> Result<Ds41rtDeviceBuffer> {
        let rows = request.header.row_count;
        let library = self.library;
        let planes = &self.planes;
        self.transport
            .execute(request, |rank, first_row, bytes| {
                copy_chunk(library, planes, rank, first_row, bytes)
            })
            .await?;
        reduce_planes(
            self.library,
            &self.reducer,
            &self.stream,
            &self.planes,
            self.output.buffer,
            has_shared.then_some(self.shared.buffer),
            rows,
        )?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// Borrowed device view; never free it or retain it across wave reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("native TP wave output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 5120 * 2;
        Ok(output)
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for NativeTp4Wave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native coordinator TP wave");
        }
    }
}

/// Complete ordered TP4 reduction plus the shared expert, borrowed until consumed.
pub(crate) struct NativeFfnOutput<'a> {
    pub values: Ds41rtDeviceBuffer,
    binding: crate::v41_attention_binding::QueryBinding,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl NativeFfnOutput<'_> {
    pub fn binding(&self) -> crate::v41_attention_binding::QueryBinding {
        self.binding
    }
}

fn validate_shared(
    request: &crate::v41_backbone_router::BoundExpertRequest,
    shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    destination: Ds41rtDeviceBuffer,
    capacity: u32,
) -> Result<()> {
    let header = &request.request().header;
    ensure!(
        request.binding() == shared.binding()?
            && header.layer_id as usize == shared.layer
            && header.row_count == shared.rows
            && header.row_count > 0
            && header.row_count <= capacity
            && shared.values.bytes == header.row_count as usize * 10240
            && shared.values.device_id == destination.device_id,
        "native TP shared contribution differs from routed request"
    );
    Ok(())
}
fn copy_chunk(
    library: &NativeLibrary,
    planes: &[DeviceAllocation<'_>; 4],
    rank: usize,
    first_row: u32,
    bytes: &[u8],
) -> Result<()> {
    library.copy_h2d(chunk_destination(planes, rank, first_row, bytes.len())?, bytes)
}
fn chunk_destination(
    planes: &[DeviceAllocation<'_>; 4],
    rank: usize,
    first_row: u32,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    ensure!(rank < 4, "native route rank exceeds TP4");
    let offset = (first_row as usize)
        .checked_mul(V41_PARTIAL_ROW_BYTES as usize)
        .context("native route chunk offset overflow")?;
    let end = offset
        .checked_add(bytes)
        .context("native route chunk extent overflow")?;
    ensure!(
        end <= planes[rank].buffer.bytes,
        "native route chunk exceeds destination"
    );
    let mut destination = planes[rank].buffer;
    destination.ptr = unsafe { destination.ptr.cast::<u8>().add(offset).cast() };
    destination.bytes = bytes;
    Ok(destination)
}

/// Retain received pinned frames until uploads finish. Drop also drains on
/// cancellation or errors, before recycling their host storage.
struct PlaneUploads<'s, 'a> {
    library: &'a NativeLibrary,
    stream: &'s LoadStream<'a>,
    planes: &'s [DeviceAllocation<'a>; 4],
    frames: &'s mut Vec<VerbsHostProtocolV2ResponsePayload>,
    rows: u32,
    pending: bool,
}
impl PlaneUploads<'_, '_> {
    fn copy(&mut self, rank: usize, first_row: u32, payload: VerbsHostProtocolV2ResponsePayload) -> Result<()> {
        let bytes = payload.as_ref();
        let destination = chunk_destination(self.planes, rank, first_row, bytes.len())?;
        let offset = first_row as usize * V41_PARTIAL_ROW_BYTES as usize;
        ensure!(offset + bytes.len() <= self.rows as usize * 10240,
            "native upload exceeds live rows");
        if payload.pinned_host_buffer().is_none() {
            return self.library.copy_h2d(destination, bytes);
        }
        ensure!(self.frames.len() < self.frames.capacity(), "native retained frame capacity exhausted");
        // Transfer ownership before enqueue, including a possible partial enqueue error.
        self.frames.push(payload);
        self.pending = true;
        unsafe { self.library.copy_h2d_async(destination, self.frames.last().unwrap().as_ref(), self.stream.raw) }
    }
}
impl Drop for PlaneUploads<'_, '_> {
    fn drop(&mut self) {
        if self.pending {
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(self.stream.raw) } {
                tracing::error!(%error, "draining interrupted native rank uploads");
            }
        }
        self.frames.clear();
    }
}
unsafe fn enqueue_reduce_planes(reducer: &V41CompactReducer<'_>, stream: &LoadStream<'_>,
    planes: &[DeviceAllocation<'_>; 4], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    unsafe { reducer.reduce(
        std::array::from_fn(|rank| planes[rank].buffer.ptr.cast::<u16>().cast_const()),
        shared.map_or(std::ptr::null(), |b| b.ptr.cast()), output.ptr.cast(), rows, stream.raw) }
}
fn reduce_planes(library: &NativeLibrary, reducer: &V41CompactReducer<'_>,
    stream: &LoadStream<'_>, planes: &[DeviceAllocation<'_>; 4], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    let launched = unsafe { enqueue_reduce_planes(reducer, stream, planes, output, shared, rows) };
    launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })
}
/// Retain planes, upload frames, output and shared input through completion.
async unsafe fn reduce_planes_cooperative(reducer: &V41CompactReducer<'_>,
    stream: &LoadStream<'_>, planes: &[DeviceAllocation<'_>; 4], output: Ds41rtDeviceBuffer,
    shared: Option<Ds41rtDeviceBuffer>, rows: u32) -> Result<()> {
    let launched = unsafe { enqueue_reduce_planes(reducer, stream, planes, output, shared, rows) };
    let drained = stream.wait().await;
    launched.and(drained)
}

/// Borrows every mutable reduction buffer and owns all unread response sockets.
/// Dropping before completion leaves the wave unpublished and closes the sockets.
pub(crate) struct NativePendingFfn<'w, 'a, 'r> {
    pending: V41Tp4RocePending<'w, 'r>,
    request: &'r crate::v41_backbone_router::BoundExpertRequest,
    capacity: u32,
    library: &'a NativeLibrary,
    stream: &'w LoadStream<'a>,
    planes: &'w [DeviceAllocation<'a>; 4],
    upload_frames: &'w mut Vec<VerbsHostProtocolV2ResponsePayload>,
    shared: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    reducer: &'w V41CompactReducer<'a>,
    ready_rows: &'w mut Option<u32>,
    tp2: Option<&'w mut super::tp2_ffn::Wave<'a>>,
}
impl<'w> NativePendingFfn<'w, '_, '_> {
    /// # Safety
    /// Shared values hold the completed contribution for this exact request and
    /// remain immutable until the final reduction drains. Producers must be drained.
    pub async unsafe fn finish(
        self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>,
    ) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, false).await }
    }
    /// # Safety
    /// Same input retention as finish; cancellation drains before releasing frames.
    pub async unsafe fn finish_cooperative(self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>) -> Result<NativeFfnOutput<'w>> {
        unsafe { self.finish_inner(shared, true).await }
    }
    /// Run shared TP2 after Spark dispatch, reduce on the transport GPU, and
    /// return the result to the block's GPU. The pending owner retains every lane workspace.
    /// # Safety
    /// Input is the completed normalized FFN input for the dispatched request;
    /// its storage remains immutable through completion or cancellation drain.
    pub async unsafe fn finish_tp2(mut self, input: &crate::v41_block::FfnInput<'_>) -> Result<NativeFfnOutput<'w>> {
        let header = &self.request.request().header;
        ensure!(self.request.binding() == input.binding() && header.layer_id as usize == input.layer
            && header.row_count as usize == input.tokens.len()
            && matches!(input.values.device_id, 0 | 1),
            "TP2 shared input/request/device differs");
        let values = unsafe { self.tp2.as_mut().context("TP2 shared workspace missing")?
            .execute_shared_on(input.layer,header.row_count,input.values,self.output.device_id as usize).await? };
        let destination = input.values.device_id;
        if destination != self.output.device_id {
            let device = crate::v41_memory::device::Device { library: self.library, id: self.output.device_id };
            device.future(unsafe { self.finish_values(values,true,Some(destination)) }).await
        } else {
            unsafe { self.finish_values(values,true,Some(destination)).await }
        }
    }
    async unsafe fn finish_inner(self,
        shared: &crate::v41_backbone_shared::SharedOutput<'_>, cooperative: bool) -> Result<NativeFfnOutput<'w>> {
        validate_shared(self.request, shared, self.shared, self.capacity)?;
        unsafe { self.finish_values(shared.values,cooperative,None).await }
    }
    async unsafe fn finish_values(mut self, values: Ds41rtDeviceBuffer, cooperative: bool,
        destination: Option<i32>) -> Result<NativeFfnOutput<'w>> {
        ensure!(values.device_id == self.output.device_id
            && values.bytes == self.request.request().header.row_count as usize * 10240,
            "shared reduction device/extent differs");
        let timing = std::time::Instant::now();
        // The shared owner remains borrowed until reduction drains, so consume
        // its completed device output directly instead of copying it first.
        let shared_copy_us = 0u64;
        let mut uploads = PlaneUploads {
            library: self.library,
            stream: self.stream,
            planes: self.planes,
            frames: self.upload_frames,
            rows: self.request.request().header.row_count,
            pending: false,
        };
        let mut upload_us = 0u64;
        self.pending
            .receive_owned(|rank, first_row, payload| {
                let copy_start = std::time::Instant::now();
                let result = uploads.copy(rank, first_row, payload);
                upload_us += copy_start.elapsed().as_micros() as u64;
                result
            })
            .await?;
        let received_us = timing.elapsed().as_micros() as u64;
        let rows = self.request.request().header.row_count;
        if cooperative {
            unsafe { reduce_planes_cooperative(self.reducer, self.stream, self.planes,
                self.output, Some(values), rows).await?; }
        } else {
            reduce_planes(self.library, self.reducer, self.stream, self.planes,
                self.output, Some(values), rows)?;
        }
        uploads.pending = false; // uploads and reduction completed on the same stream.
        tracing::debug!(target: "ds41rt::timing", layer=self.request.request().header.layer_id, rows, shared_copy_us, upload_us, receive_us=received_us-shared_copy_us-upload_us, reduce_us=timing.elapsed().as_micros() as u64-received_us, "target collection");
        *self.ready_rows = Some(rows);
        let mut values = self.output;
        values.bytes = rows as usize * 10240;
        if let Some(device) = destination.filter(|&device| device != values.device_id) {
            values = unsafe { self.tp2.as_mut().context("TP2 return workspace missing")?
                .return_result(values, device as usize, rows).await? };
        }
        Ok(NativeFfnOutput {
            values,
            binding: self.request.binding(),
            _owner: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;

    #[test]
    fn pageable_rank_upload_fallback_matches_sync_and_checks_bounds() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_PLANE_UPLOAD_LIBRARY") else {
            eprintln!("skip GPU rank upload test: DS41RT_PLANE_UPLOAD_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let planes = (0..4)
            .map(|_| DeviceAllocation::new(&library, 4096 * 10240))
            .collect::<Result<Vec<_>>>()?
            .try_into().ok().expect("four planes");
        let mut frames = Vec::with_capacity(4096 * 4);
        let shared = DeviceAllocation::new(&library, 4096 * 10240)?;
        let output = DeviceAllocation::new(&library, 4096 * 10240)?;
        let reducer = library.v41_compact_reducer()?;
        let frames_address = frames.as_ptr();
        for rows in [1u32, 6, 80, 81, 256, 1024, 4096, 6] {
            let bytes = rows as usize * 10240;
            let shared_bytes: Vec<u8> = (0..bytes / 2)
                .flat_map(|_| 0x3f00u16.to_ne_bytes()).collect();
            library.copy_h2d(shared.buffer, &shared_bytes)?;
            let payloads: Vec<Vec<u8>> = (0..4).map(|rank| {
                (0..bytes / 2).flat_map(|i| {
                    let value = rank as f32 + 1.0 + (i % 31) as f32 / 32.0;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect()
            }).collect();
            for (rank, payload) in payloads.iter().enumerate() {
                copy_chunk(&library, &planes, rank, 0, payload)?;
            }
            reduce_planes(&library, &reducer, &stream, &planes,
                output.buffer, Some(shared.buffer), rows)?;
            let mut expected = vec![0u8; bytes];
            library.copy_d2h(&mut expected, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            for plane in &planes { library.copy_h2d(plane.buffer, &vec![0; bytes])?; }
            {
                let mut uploads = PlaneUploads {
                    library: &library, stream: &stream, planes: &planes,
                    frames: &mut frames, rows, pending: false,
                };
                for first in (0..rows).step_by(3) {
                    let end = (first + 3).min(rows);
                    for rank in [3, 1, 0, 2] {
                        uploads.copy(rank, first,
                            VerbsHostProtocolV2ResponsePayload::from_owned(payloads[rank][first as usize * 10240..end as usize * 10240].to_vec()))?;
                    }
                }
                let runtime = tokio::runtime::Builder::new_current_thread().build()?;
                runtime.block_on(async {
                    use std::{future::Future, task::Poll};
                    let cancelled = {
                        let mut work = std::pin::pin!(unsafe { reduce_planes_cooperative(&reducer, &stream,
                            &planes, output.buffer, Some(shared.buffer), rows) });
                        std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
                    };
                    stream.require_complete()?;
                    unsafe { reduce_planes_cooperative(&reducer, &stream, &planes,
                        output.buffer, Some(shared.buffer), rows).await?; }
                    eprintln!("PASS cooperative TP reduction rows={rows} pending_cancel={cancelled} reuse=true");
                    Ok::<_, anyhow::Error>(())
                })?;
                stream.require_complete()?;
                uploads.pending = false;
            }
            let mut actual = vec![0u8; bytes];
            library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes, ..output.buffer })?;
            assert_eq!(actual, expected, "rows={rows}");
            assert_eq!(frames.as_ptr(), frames_address);
            assert!(frames.is_empty());
            // Pageable fallback completes synchronously, including before a later bounds error.
            for error in [false, true] {
                library.copy_h2d(planes[0].buffer, &vec![0; 10240])?;
                {
                    let mut uploads = PlaneUploads {
                        library: &library, stream: &stream, planes: &planes,
                        frames: &mut frames, rows: 1, pending: false,
                    };
                    uploads.copy(0, 0, VerbsHostProtocolV2ResponsePayload::from_owned(payloads[0][..10240].to_vec()))?;
                    if error { assert!(uploads.copy(4, 0, VerbsHostProtocolV2ResponsePayload::from_owned(payloads[0][..10240].to_vec())).is_err()); }
                }
                assert!(frames.is_empty());
                let mut actual = vec![0; 10240];
                library.copy_d2h(&mut actual, Ds41rtDeviceBuffer { bytes: 10240, ..planes[0].buffer })?;
                assert_eq!(actual, payloads[0][..10240]);
            }
            eprintln!("PASS rows={rows}: pageable interleaved chunks/reduction exact, stable owner capacity, bounds errors");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires four idle live Spark workers and CUDA native library"]
    fn retained_roce_uploads_drain_before_recycling_live() -> Result<()> {
        use ds41rt_transport::{ExpertProtocolV2RowDescriptor, ExpertProtocolV2RouteEntry,
            ExpertV2SourceKind, ExpertV2Dtype, TcpTransportConfig};
        let library = unsafe { NativeLibrary::load(std::env::var("DS41RT_PLANE_UPLOAD_LIBRARY")?)? };
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let peers = std::env::var("DS41RT_LIVE_ROCE_PEERS")?.split(',')
            .map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?
            .try_into().map_err(|_| anyhow::anyhow!("four peers required"))?;
        let mut client = V41Tp4Roce::new(peers, [1,2,3,4], 4096, TcpTransportConfig {
            timeout: std::time::Duration::from_secs(30), max_frame_bytes: 64 << 20,
        })?;
        let planes = (0..4).map(|_| DeviceAllocation::new(&library, 4096 * 10240))
            .collect::<Result<Vec<_>>>()?.try_into().ok().expect("four planes");
        let mut frames = Vec::with_capacity(4096 * 4);
        let frame_address = frames.as_ptr();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async {
            let mut request_id = 900_000;
            for rows in [1u32,6,80,81,256,1024,4096,6] {
                let mut hidden = Vec::with_capacity(rows as usize * 5280);
                for row in 0..rows {
                    hidden.extend((0..5120).map(|i| 0x30 + ((i+row)%8) as u8));
                    hidden.extend([120;160]);
                }
                let mut request = ExpertProtocolV2Request::new(request_id, 17, 39, 5120,
                    ExpertV2Dtype::Fp8E4m3Ue8m0K32,
                    (0..rows).map(|r| ExpertProtocolV2RowDescriptor {
                        row_id: r as u64, source_kind: ExpertV2SourceKind::Prefill,
                        source_request_id: request_id, token_position: r as u64,
                        route_offset: r*6, route_count: 6,
                    }).collect(),
                    (0..rows).flat_map(|r| (0..6).map(move |j| ExpertProtocolV2RouteEntry {
                        row_index: r, expert_id: (r*7+j)%384, gate_weight: 1.0/6.0,
                    })).collect(), hidden)?;
                request.header.flags = ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
                for fail in [false,true,false] {
                    request_id += 1;
                    request.header.request_id = request_id;
                    let mut expected: Vec<(usize,u32,Vec<u8>)> = Vec::new();
                    {
                        let mut uploads = PlaneUploads { library: &library, stream: &stream,
                            planes: &planes, frames: &mut frames, rows, pending: false };
                        let result = client.dispatch(&request).await?.receive_owned(|rank, first, payload| {
                            ensure!(payload.pinned_host_buffer().is_some(), "live payload is not pinned");
                            ensure!(payload.retains_receive_slot(), "final response is not a retained receive slot");
                            expected.push((rank, first, payload.as_ref().to_vec()));
                            uploads.copy(rank, first, payload)?;
                            ensure!(uploads.pending && !uploads.frames.is_empty(), "upload ownership missing");
                            if fail { anyhow::bail!("injected failure after pinned upload enqueue"); }
                            Ok(())
                        }).await;
                        assert_eq!(result.is_err(), fail);
                        // QP teardown must not free payloads owned by an active upload.
                        if fail { client.reset_connections(); }
                        // Drop simulates cancellation after enqueue and must synchronize
                        // before returning retained storage to the transport pool.
                    }
                    assert!(frames.is_empty());
                    assert_eq!(frames.as_ptr(), frame_address);
                    assert!(!expected.is_empty());
                    if !fail {
                        assert_eq!(expected.iter().map(|(_,_,b)| b.len()).sum::<usize>(), rows as usize * 4 * 10240);
                    }
                    for (rank, first, bytes) in expected {
                        let mut actual = vec![0;bytes.len()];
                        library.copy_d2h(&mut actual, chunk_destination(&planes, rank, first, bytes.len())?)?;
                        assert_eq!(actual, bytes, "rows={rows} fail={fail} rank={rank}");
                    }
                }
                eprintln!("PASS retained rows={rows}: real pinned frames, exact device copies, failure/reset/drop drain and recovery");
            }
            Ok(())
        })
    }

}
