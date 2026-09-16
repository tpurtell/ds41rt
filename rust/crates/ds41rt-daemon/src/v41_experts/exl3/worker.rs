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
    executions: Vec<Exl3Execution<'a>>,
    capacity: usize,
    inputs: [DeviceAllocation<'a>; 3],
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

    pub(crate) fn plan(directory: &Path, capacity: u32) -> Result<usize> {
        let directories: Vec<_> = Self::capacities(capacity)?.into_iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(capacity as usize * (5280 + 6 * 8))
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
            rank < 4 && first.layout.world == 4,
            "EXL3 worker requires Spark TP4 weights"
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
        let inputs = [
            DeviceAllocation::new(library, capacity as usize * 5280)?,
            DeviceAllocation::new(library, capacity as usize * 6 * 4)?,
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
        Ok(Self {
            stream,
            executions,
            capacity: capacity as usize,
            inputs,
            library,
            first_layer,
            layer_count,
            layer: 0,
            executor_id: rank as u64 + 1,
        })
    }

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
        request.require_input_dtype(ExpertV2Dtype::Fp8E4m3Ue8m0K32 as u32)?;
        let bytes = request.plane_bytes()?;
        ensure!(
            exchange.partials.len() >= bytes,
            "EXL3 host exchange is too small"
        );
        request.copy_routes_into(&mut exchange.ids, &mut exchange.routing)?;
        ensure!(
            cfg!(target_endian = "little"),
            "native exchange requires little-endian storage"
        );
        let routes = request.rows() as usize * 6;
        // Every previous response completed this stream before returning.
        self.library
            .copy_h2d(self.inputs[0].buffer, request.hidden())?;
        unsafe {
            self.library.copy_h2d(
                self.inputs[1].buffer,
                std::slice::from_raw_parts(exchange.ids.as_ptr().cast::<u8>(), routes * 4),
            )?;
            self.library.copy_h2d(
                self.inputs[2].buffer,
                std::slice::from_raw_parts(exchange.routing.as_ptr().cast::<u8>(), routes * 4),
            )?;
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
            match destination {
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
            }
        };
        unsafe {
            self.library.cuda_stream_synchronize(self.stream.raw)?;
        }
        if destination.is_none() {
            self.library
                .copy_d2h(&mut exchange.partials[..bytes], output)?;
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
