//! One lane's local routed-expert scratch; both lanes share immutable weights.
use super::exl3::{
    execution::{Exl3Execution, Exl3InputFormat, Exl3Workspace},
    Exl3Weights,
};
use super::{DeviceAllocation, ExpertLayer, ExpertWeights, LoadStream};
use crate::{v41_backbone_router::RouterOutput, v41_backbone_shared::SharedOutput};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs, V41LocalExpertReducer,
};
use std::{ffi::c_void, path::Path, rc::Rc};

/// Activation contract of the resident kernel, independent of remote wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalInputFormat {
    Fp8K32,
    Bf16,
}
impl LocalInputFormat {
    fn for_nvfp4(nvfp4: bool) -> Self {
        if nvfp4 { Self::Bf16 } else { Self::Fp8K32 }
    }
    fn row_bytes(self) -> usize {
        match self { Self::Fp8K32 => 5280, Self::Bf16 => 10240 }
    }
    fn select(self, rows: u32, bf16: Ds41rtDeviceBuffer, fp8: Ds41rtDeviceBuffer) -> Result<Ds41rtDeviceBuffer> {
        let input = match self { Self::Bf16 => bf16, Self::Fp8K32 => fp8 };
        ensure!(input.bytes == rows as usize * self.row_bytes(),
            "local expert activation extent differs from {self:?} kernel input");
        Ok(input)
    }
}

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}
enum Backend<'a> {
    Full {
        states: Vec<State<'a>>,
        _scratch: DeviceAllocation<'a>,
        weights: Rc<Vec<ExpertWeights<'a>>>,
    },
    Exl3 {
        states: Vec<Exl3Execution<'a>>,
        layers: usize,
    },
}
pub(crate) struct LocalExpertWave<'a> {
    // Drain before scratch, weights or output can be released on any exit.
    stream: LoadStream<'a>,
    backend: Backend<'a>,
    output: DeviceAllocation<'a>,
    reducer: V41LocalExpertReducer<'a>,
    capacity: u32,
    input_format: LocalInputFormat,
}
impl<'a> LocalExpertWave<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!(
            matches!(capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
            "invalid local expert capacity"
        );
        let mut capacities: Vec<_> = [1, 16, 80].into_iter().filter(|&c| c < capacity).collect();
        capacities.push(capacity);
        Ok(capacities)
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::device_bytes_for(library, capacity, false)
    }
    /// The W4A4 family reports its own scratch plan and uses its own kernels.
    pub fn device_bytes_for(
        library: &NativeLibrary,
        capacity: u32,
        nvfp4: bool,
    ) -> Result<usize> {
        let arena = Self::capacities(capacity)?
            .into_iter()
            .try_fold(0usize, |largest, c| {
                let info = if nvfp4 {
                    library.v41_nvfp4_local_expert_info(c)?
                } else {
                    library.v41_local_expert_info(c)?
                };
                Ok::<_, anyhow::Error>(largest.max(usize::try_from(info.scratch_bytes)?))
            })?;
        arena
            .checked_add(capacity as usize * 10240)
            .context("local expert workspace size overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<ExpertWeights<'a>>>,
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            !weights.is_empty() && weights.len() <= 40,
            "local experts require complete resident layers"
        );
        for (layer, weight) in weights.iter().enumerate() {
            ensure!(
                weight.layer == ExpertLayer::BackboneFull { layer }
                    && std::ptr::eq(weight.buffers[0].library, library),
                "local layer ownership differs"
            );
        }
        let nvfp4 = weights.first().is_some_and(|weight| weight.is_nvfp4());
        ensure!(weights.iter().all(|weight| weight.is_nvfp4() == nvfp4),
            "local expert layers mix activation formats");
        ensure!(
            Self::device_bytes_for(library, capacity, nvfp4)? <= budget,
            "local expert workspace exceeds budget"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let scratch = DeviceAllocation::new(
            library,
            Self::device_bytes_for(library, capacity, nvfp4)? - capacity as usize * 10240,
        )?;
        let mut states = Vec::new();
        for c in Self::capacities(capacity)? {
            let kernel = if nvfp4 {
                library.v41_nvfp4_local_expert_kernel(c)?
            } else {
                library.v41_local_expert_kernel(c)?
            };
            ensure!(kernel.info().input_row_bytes()? == LocalInputFormat::for_nvfp4(nvfp4).row_bytes(),
                "local expert kernel activation format differs from resident weights");
            let mut slots = [std::ptr::null_mut(); 44];
            unsafe {
                kernel.bind_scratch(scratch.buffer.ptr, scratch.buffer.bytes as u64, &mut slots)?;
            }
            let initialized = unsafe {
                kernel.initialize_scratch(
                    scratch.buffer.ptr,
                    scratch.buffer.bytes as u64,
                    stream.raw,
                )
            };
            let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
            initialized.and(drained)?;
            states.push(State { kernel, slots });
        }
        Ok(Self {
            stream,
            backend: Backend::Full {
                states,
                _scratch: scratch,
                weights,
            },
            input_format: LocalInputFormat::for_nvfp4(nvfp4),
            output: DeviceAllocation::new(library, capacity as usize * 10240)?,
            reducer: library.v41_local_expert_reducer()?,
            capacity,
        })
    }
    pub fn exl3_device_bytes(directory: &Path, capacity: u32) -> Result<usize> {
        let directories: Vec<_> = Self::capacities(capacity)?.into_iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(capacity as usize * 10240)
            .context("local EXL3 workspace overflow")
    }
    /// Trusted native package; all modules are bound before request processing.
    pub unsafe fn new_exl3(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            !weights.is_empty() && weights.len() <= 40,
            "local EXL3 requires resident layers"
        );
        for (layer, weight) in weights.iter().enumerate() {
            ensure!(
                matches!(weight.layout.layer, ds41rt_loader::V41Exl3Layer::Backbone(index) if index == layer)
                    && weight.layout.world == 1
                    && weight.layout.rank == 0,
                "local EXL3 layer ownership differs"
            );
        }
        ensure!(
            Self::exl3_device_bytes(directory, capacity)? <= budget,
            "local EXL3 workspace exceeds budget"
        );
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut states = Vec::new();
        let capacities = Self::capacities(capacity)?;
        let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
        let arena = Exl3Workspace::new(library, &directories)?;
        for c in capacities {
            let state = unsafe {
                Exl3Execution::with_shared_workspace(
                    library,
                    weights.clone(),
                    &directory.join(format!("m{c}")),
                    Exl3InputFormat::Fp8K32,
                    Some(arena.clone()),
                )?
            };
            ensure!(
                state.capacity() == c as usize && state.output_element_bytes() == 4,
                "local EXL3 requires matching capacity and FP32 output"
            );
            states.push(state);
        }
        ensure!(arena.bytes() + states.iter().map(|state| state.workspace_bytes()).sum::<usize>()
            + capacity as usize * 10240 == Self::exl3_device_bytes(directory, capacity)?,
            "local EXL3 shared workspace plan mismatch");
        Ok(Self {
            stream,
            backend: Backend::Exl3 {
                states,
                layers: weights.len(),
            },
            input_format: LocalInputFormat::Fp8K32,
            output: DeviceAllocation::new(library, capacity as usize * 10240)?,
            reducer: library.v41_local_expert_reducer()?,
            capacity,
        })
    }
    pub fn contains(&self, layer: usize) -> bool {
        layer
            < match &self.backend {
                Backend::Full { weights, .. } => weights.len(),
                Backend::Exl3 { layers, .. } => *layers,
            }
    }

    /// # Safety
    /// Router and shared outputs are complete and remain borrowed until this
    /// method drains, including on partial launch failure. No external mutation.
    pub unsafe fn execute(
        &mut self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<Ds41rtDeviceBuffer> {
        let rows = routed.rows;
        unsafe {
            self.enqueue(routed, shared)?;
        }
        unsafe {
            self.stream
                .library
                .cuda_stream_synchronize(self.stream.raw)?;
        }
        Ok(self.output_rows(rows))
    }
    /// # Safety
    /// Same retained input contract as execute; cancellation drains this stream.
    pub async unsafe fn execute_cooperative(
        &mut self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<Ds41rtDeviceBuffer> {
        unsafe {
            self.enqueue(routed, shared)?;
        }
        self.stream.wait().await?;
        Ok(self.output_rows(routed.rows))
    }
    fn output_rows(&self, rows: u32) -> Ds41rtDeviceBuffer {
        let mut output = self.output.buffer;
        output.bytes = rows as usize * 10240;
        output
    }
    unsafe fn enqueue(
        &mut self,
        routed: &RouterOutput<'_>,
        shared: &SharedOutput<'_>,
    ) -> Result<()> {
        let rows = routed.rows;
        ensure!(
            self.contains(routed.layer) && rows > 0 && rows <= self.capacity,
            "local expert layer/rows are not resident or exceed capacity"
        );
        ensure!(
            routed.binding()? == shared.binding()?
                && routed.layer == shared.layer
                && rows == shared.rows,
            "local router and shared expert binding differ"
        );
        unsafe {
            self.enqueue_buffers(
                routed.layer,
                rows,
                [self.input_format.select(rows, routed.input, routed.expert_input)?, routed.ids, routed.routing],
                shared.values,
            )
        }
    }

    // Caller checks block provenance and retains input owners through the drain.
    unsafe fn enqueue_buffers(
        &mut self,
        layer: usize,
        rows: u32,
        inputs: [Ds41rtDeviceBuffer; 3],
        shared: Ds41rtDeviceBuffer,
    ) -> Result<()> {
        ensure!(
            self.contains(layer) && rows > 0 && rows <= self.capacity,
            "local expert layer/rows are not resident or exceed capacity"
        );
        for (buffer, bytes) in [
            (inputs[0], rows as usize * self.input_format.row_bytes()),
            (inputs[1], rows as usize * 24),
            (inputs[2], rows as usize * 24),
            (shared, rows as usize * 10240),
        ] {
            ensure!(
                buffer.bytes == bytes && buffer.device_id == self.output.buffer.device_id,
                "local expert input extent or device differs"
            );
        }
        let launched = (|| -> Result<()> {
            match &mut self.backend {
                Backend::Exl3 { states, .. } => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.capacity() >= rows as usize)
                        .context("local EXL3 capacity state missing")?;
                    let values = unsafe {
                        state.launch_layer(
                            layer,
                            [inputs[0], inputs[1], inputs[2]],
                            rows as usize,
                            self.stream.raw,
                        )?
                    };
                    unsafe {
                        self.reducer.finish(
                            values.ptr.cast(),
                            shared.ptr.cast(),
                            self.output.buffer.ptr.cast(),
                            rows,
                            true,
                            self.stream.raw,
                        )
                    }
                }
                Backend::Full {
                    states, weights, ..
                } => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.kernel.info().capacity_rows >= rows)
                        .context("local expert capacity state missing")?;
                    weights[layer].bind(&state.kernel, &mut state.slots)?;
                    state.slots[0] = inputs[0].ptr;
                    state.slots[1] = inputs[1].ptr;
                    state.slots[2] = inputs[2].ptr;
                    let info = state.kernel.info();
                    let args = V41ExpertLaunchArgs {
                        tensors: state.slots,
                        num_tokens: rows as i32,
                        max_rows: info.max_rows,
                        scatter_rows: rows as i32 * 6,
                        rows_padded: info.rows_padded,
                        max_tasks: info.max_tasks,
                        max_phys_tiles: info.max_phys_tiles,
                        max_active_clusters: info.max_active_clusters,
                        stream: self.stream.raw,
                    };
                    // Shared arena remains exclusive through this drain, including failure.

                    unsafe {
                        state.kernel.launch(&args)?;
                    }
                    unsafe {
                        match state.kernel.output_kind() {
                            ds41rt_ffi::V41ExpertOutputKind::Bf16Routes => self.reducer.finish_bf16_routes(
                                state.slots[41].cast(), shared.ptr.cast(),
                                self.output.buffer.ptr.cast(), rows, self.stream.raw),
                            ds41rt_ffi::V41ExpertOutputKind::Fp32Routes |
                            ds41rt_ffi::V41ExpertOutputKind::Fp32Tokens => self.reducer.finish(
                                state.slots[41].cast(), shared.ptr.cast(),
                                self.output.buffer.ptr.cast(), rows,
                                state.kernel.accumulates_tokens(), self.stream.raw),
                        }
                    }
                }
            }
        })();
        if let Err(error) = launched {
            unsafe {
                self.stream
                    .library
                    .cuda_stream_synchronize(self.stream.raw)?;
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "local/tests.rs"]
mod tests;
