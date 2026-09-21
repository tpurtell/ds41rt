//! Spark request adapter: preserve the compact BF16 wire response and write
//! directly into registered send storage when the transport permits it.
use super::{
    execution::{Exl3Execution, Exl3InputFormat, Exl3Workspace},
    Exl3Weights,
};
use crate::{
    v41_experts::HostExpertExchange,
    v41_memory::{DeviceAllocation, LoadStream},
};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_transport::{
    v41_expert::V41BackboneRequest, ExpertProtocolV2DeviceResponseRef, ExpertProtocolV2ResponseRef,
    ExpertV2Dtype, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
};
use std::{path::Path, rc::Rc};

pub(crate) struct Exl3Worker<'a> {
    // Drain the stream before dropping kernels, weights or input allocations.
    stream: LoadStream<'a>,
    // Opt-in diagnostics: allocate events once; the default path never records them.
    timing: Option<[crate::v41_memory::device::Event<'a>; 2]>,
    executions: Vec<Exl3Execution<'a>>,
    capacity: usize,
    inputs: [DeviceAllocation<'a>; 3],
    paired_upload: Option<Vec<i32>>,
    ownership_words: usize,
    library: &'a NativeLibrary,
    first_layer: usize,
    layer_count: usize,
    layer: usize,
    executor_id: u64,
}

impl<'a> Exl3Worker<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        let capacities = [1, 16, 80, 256, 1024, 4096];
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid EXL3 worker capacity"
        );
        let maximum = capacities.into_iter().find(|&c| c >= capacity).unwrap();
        Ok(capacities.into_iter().filter(|&c| c <= maximum).collect())
    }

    /// Validate every capacity before allocating or reading resident weights.
    pub(crate) fn partition(directory: &Path, capacity: u32, rank: usize) -> Result<ds41rt_loader::V41Exl3Partition> {
        use ds41rt_ffi::V41Exl3Layout;
        use ds41rt_loader::V41Exl3Partition;
        ensure!(rank < 4, "EXL3 worker rank must be 0..3");
        let mut selected = None;
        for c in Self::capacities(capacity)? {
            let layout = Exl3Execution::artifact_layout(&directory.join(format!("m{c}")))?;
            ensure!(selected.is_none_or(|previous| previous == layout), "EXL3 capacity artifacts disagree on partition");
            ensure!(layout == V41Exl3Layout::Disjoint || layout == if rank % 2 == 0 {
                V41Exl3Layout::PairedLast
            } else { V41Exl3Layout::PairedFirst }, "EXL3 artifact boundary does not match worker rank");
            selected = Some(layout);
        }
        Ok(if selected == Some(V41Exl3Layout::Disjoint) { V41Exl3Partition::Disjoint }
            else { V41Exl3Partition::PairedTp4 })
    }

    pub(crate) fn plan(directory: &Path, capacity: u32) -> Result<usize> {
        let directories: Vec<_> = Self::capacities(capacity)?.into_iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        let ownership_bytes = Exl3Execution::ownership_bytes(&directories[0])?;
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(capacity as usize * (5280 + 6 * 8))
            .and_then(|bytes| bytes.checked_add(ownership_bytes))
            .context("EXL3 worker workspace budget overflow")
    }

    pub(crate) fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        capacity: u32,
        available_bytes: usize,
    ) -> Result<Self> {
        let first = weights.first().context("EXL3 worker has no layers")?;
        let ds41rt_loader::V41Exl3Layer::Backbone(first_layer) = first.layout.layer else {
            anyhow::bail!("EXL3 Spark worker requires backbone layers");
        };
        let rank = first.layout.rank;
        ensure!(
            matches!(first.layout.world, 2 | 3 | 4) && rank < first.layout.world,
            "EXL3 worker requires implicit Spark TP2, TP3 or TP4 weights"
        );
        for (index, weight) in weights.iter().enumerate() {
            ensure!(
                matches!(weight.layout.layer, ds41rt_loader::V41Exl3Layer::Backbone(layer)
                if layer == first_layer + index),
                "EXL3 worker layers must be contiguous"
            );
        }
        let layer_count = weights.len();
        let budget = Self::plan(directory, capacity)?;
        ensure!(
            budget <= available_bytes,
            "EXL3 worker workspace exceeds device budget"
        );
        let mut executions = Vec::new();
        let capacities = Self::capacities(capacity)?;
        let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
        let arena = Exl3Workspace::new(library, &directories)?;
        for c in capacities {
            let execution = unsafe {
                Exl3Execution::with_shared_workspace(
                    library,
                    weights.clone(),
                    &directory.join(format!("m{c}")),
                    Exl3InputFormat::Fp8K32,
                    Some(arena.clone()),
                )?
            };
            ensure!(
                execution.capacity() == c as usize && execution.output_element_bytes() == 2,
                "EXL3 Spark export must match planned capacity and BF16 response precision"
            );
            executions.push(execution);
        }
        let ownership_words = if first.layout.layout == ds41rt_loader::V41Exl3Partition::PairedTp4 {
            first.layout.experts * first.layout.tiers.len()
        } else { 0 };
        let paired_upload = (ownership_words > 0).then(|| vec![0; capacity as usize * 6 + ownership_words]);
        let inputs = [
            DeviceAllocation::new(library, capacity as usize * 5280)?,
            DeviceAllocation::new(library, (capacity as usize * 6 + ownership_words) * 4)?,
            DeviceAllocation::new(library, capacity as usize * 6 * 4)?,
        ];
        ensure!(
            executions
                .iter()
                .map(|e| e.workspace_bytes())
                .sum::<usize>()
                + inputs.iter().map(|b| b.buffer.bytes).sum::<usize>()
                + arena.bytes()
                == budget,
            "EXL3 worker workspace plan mismatch"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let timing = if std::env::var_os("DS41RT_EXL3_WORKER_TIMING").is_some() {
            let device = crate::v41_memory::device::Device {
                library,
                id: library.cuda_get_device()?,
            };
            Some([
                crate::v41_memory::device::Event::new(device)?,
                crate::v41_memory::device::Event::new(device)?,
            ])
        } else {
            None
        };
        Ok(Self {
            stream,
            timing,
            executions,
            capacity: capacity as usize,
            inputs,
            paired_upload,
            ownership_words,
            library,
            first_layer,
            layer_count,
            layer: 0,
            executor_id: ds41rt_transport::v41_expert::v41_spark_executor_id(first.layout.world, rank)?,
        })
    }

    pub(crate) fn is_paired(&self) -> bool { self.paired_upload.is_some() }

    pub(crate) fn bind_layer(&mut self, layer: usize) -> Result<()> {
        ensure!(
            layer < self.layer_count,
            "requested EXL3 layer is not resident"
        );
        self.layer = layer;
        Ok(())
    }

    fn execute(
        &mut self,
        request: &V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        destination: Option<Ds41rtDeviceBuffer>,
    ) -> Result<()> {
        ensure!(
            request.layer() as usize == self.first_layer + self.layer,
            "request does not match selected EXL3 layer"
        );
        ensure!(
            executor_id == self.executor_id,
            "EXL3 response executor identity mismatch"
        );
        ensure!(
            request.rows() > 0 && request.rows() as usize <= self.capacity,
            "EXL3 request exceeds capacity"
        );
        ensure!(request.is_paired() == self.is_paired(), "EXL3 request/resident layout mismatch");
        request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32)?;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "EXL3 host exchange is too small"
        );
        let started = self.timing.as_ref().map(|_| std::time::Instant::now());
        let routes = request.rows() as usize * 6;
        if let Some(upload) = &mut self.paired_upload {
            let (ids, tail) = upload.split_at_mut(routes);
            request.copy_paired_routes_into(ids, &mut exchange.routing, self.executor_id as usize - 1,
                &mut tail[..self.ownership_words])?;
        } else {
            request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        }
        ensure!(
            cfg!(target_endian = "little"),
            "native exchange requires little-endian storage"
        );
        // Every previous response completed this stream before returning.
        self.library
            .copy_h2d(self.inputs[0].buffer, request.hidden())?;
        unsafe {
            self.library.copy_h2d(
                self.inputs[1].buffer,
                match &self.paired_upload {
                    Some(upload) => std::slice::from_raw_parts(upload.as_ptr().cast::<u8>(), (routes + self.ownership_words) * 4),
                    None => std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4),
                },
            )?;
            self.library.copy_h2d(
                self.inputs[2].buffer,
                std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4),
            )?;
        }
        let uploaded_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if let Some([start, _]) = &self.timing {
            unsafe { self.library.cuda_event_record(start.raw, self.stream.raw)?; }
        }
        let inputs = std::array::from_fn(|i| self.inputs[i].buffer);
        // Modules and storage are resolved before accepting requests. This
        // bounded selection performs no allocation, loading or compilation.
        let execution = self
            .executions
            .iter_mut()
            .find(|e| e.capacity() >= request.rows() as usize)
            .context("missing preloaded EXL3 worker capacity")?;
        let output = unsafe {
            if self.ownership_words > 0 {
                let ownership = Ds41rtDeviceBuffer {
                    ptr: self.inputs[1].buffer.ptr.cast::<u8>().add(routes * 4).cast(),
                    bytes: self.ownership_words * 4,
                    ..self.inputs[1].buffer
                };
                execution.launch_paired_layer_into(self.layer, inputs, request.rows() as usize,
                    self.stream.raw, destination, ownership)?
            } else { match destination {
                Some(output) => execution.launch_layer_into(
                    self.layer,
                    inputs,
                    request.rows() as usize,
                    self.stream.raw,
                    output,
                )?,
                None => execution.launch_layer(
                    self.layer,
                    inputs,
                    request.rows() as usize,
                    self.stream.raw,
                )?,
            } }
        };
        let enqueued_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if let Some([_, end]) = &self.timing {
            unsafe { self.library.cuda_event_record(end.raw, self.stream.raw)?; }
        }
        unsafe {
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        let completed_us = started.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        if destination.is_none() {
            self.library
                .copy_d2h(&mut exchange.partials[..bytes], output)?;
        }
        if let (Some(started), Some([start, end])) = (started, &self.timing) {
            // This interval overlaps host enqueue/wait and includes all stream work,
            // including routing and output reduction, not just the expert kernel.
            let gpu_us = unsafe { self.library.cuda_event_elapsed_ms(start.raw, end.raw)? } * 1000.;
            let mut seen = [false; 384];
            let ids = self.paired_upload.as_deref().unwrap_or(&exchange.ids);
            for &id in &ids[..routes] {
                if let Some(value) = seen.get_mut(id as usize) { *value = true; }
            }
            tracing::info!(target: "ds41rt::worker_timing", executor_id, layer=request.layer(), rows=request.rows(),
                distinct_experts=seen.iter().filter(|&&v| v).count(), mapped=destination.is_some(),
                upload_us=uploaded_us, enqueue_us=enqueued_us-uploaded_us,
                wait_us=completed_us-enqueued_us, gpu_us, total_us=started.elapsed().as_micros() as u64,
                "EXL3 worker execution");
        }
        Ok(())
    }

    /// # Safety
    /// The transport exclusively owns a GPU-accessible send slot on this device
    /// and retains it through the response send completion.
    pub(crate) unsafe fn execute_mapped_request(
        &mut self,
        request: &V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        slot: Ds41rtDeviceBuffer,
    ) -> Result<Option<ExpertProtocolV2DeviceResponseRef<'static>>> {
        let prefix = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN;
        let bytes = request.plane_bytes()?;
        if !request.permits_device_response() || slot.bytes < prefix + bytes {
            return Ok(None);
        }
        ensure!(!slot.ptr.is_null(), "null EXL3 response slot");
        let output = Ds41rtDeviceBuffer {
            ptr: slot.ptr.cast::<u8>().add(prefix).cast(),
            bytes,
            ..slot
        };
        let response = request.response_device(executor_id, output)?;
        self.execute(request, executor_id, exchange, Some(output))?;
        Ok(Some(response))
    }

    pub(crate) fn execute_host_chunks<F>(
        &mut self,
        request: &V41BackboneRequest<'_>,
        executor_id: u64,
        exchange: &mut HostExpertExchange,
        row_indices: &mut [u32],
        max_frame_bytes: usize,
        mut sink: F,
    ) -> Result<()>
    where
        F: FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    {
        let chunk_rows = request.response_chunk_rows(max_frame_bytes)?;
        ensure!(
            row_indices.len() >= chunk_rows as usize,
            "response row-index scratch is too short"
        );
        self.execute(request, executor_id, exchange, None)?;
        let stride = ds41rt_transport::v41_expert::V41_PARTIAL_ROW_BYTES as usize;
        for start in (0..request.rows()).step_by(chunk_rows as usize) {
            let end = start.saturating_add(chunk_rows).min(request.rows());
            sink(request.response_chunk(
                executor_id,
                start,
                &exchange.partials[start as usize * stride..end as usize * stride],
                row_indices,
                max_frame_bytes,
            )?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
