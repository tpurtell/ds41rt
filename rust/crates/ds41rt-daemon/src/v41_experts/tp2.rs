//! One lane's preallocated peer-copy and TP2 reduction chain.
use super::exl3::{
    execution::{Exl3Execution, Exl3InputFormat, Exl3Workspace},
    Exl3Weights,
};
use super::{ExpertLayer, ExpertWeights};
use crate::v41_memory::device::DeviceOwner;
use crate::v41_memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Tp2ExpertReducer};
use ds41rt_ffi::{V41ExpertKernel, V41ExpertLaunchArgs};
use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    path::{Path, PathBuf},
    rc::Rc,
};

/// Routed output layout a TP2 rank kernel publishes through slot 41.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tp2RoutedLayout {
    /// Six FP32 route planes per token (`[rows, 6, 5120]`).
    Fp32Routes,
    /// FP32 token sums (`[rows, 5120]`).
    Fp32Tokens,
    /// Deterministic NVFP4 BF16 routes (`[rows, 6, 5120]`).
    Bf16Routes,
}

impl Tp2RoutedLayout {
    fn element_bytes(self) -> usize {
        match self {
            Self::Fp32Routes => 6 * 4,
            Self::Fp32Tokens => 4,
            Self::Bf16Routes => 6 * 2,
        }
    }
    fn token_sums(self) -> bool {
        matches!(self, Self::Fp32Tokens)
    }
}

impl Default for Tp2RoutedLayout {
    fn default() -> Self {
        Self::Fp32Routes
    }
}

/// All encoder layers for one rank. Legacy weight allocations are created and
/// destroyed inside their owning device scope, never across an async yield.
enum RankStorage<'a> {
    Full(Vec<ExpertWeights<'a>>),
    Exl3 {
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: PathBuf,
    },
}
pub(crate) struct RankWeights<'a> {
    device: Device<'a>,
    weights: ManuallyDrop<RankStorage<'a>>,
}
impl<'a> RankWeights<'a> {
    pub fn device(&self) -> Device<'a> {
        self.device
    }
    pub fn layers(&self) -> usize {
        match &*self.weights {
            RankStorage::Full(weights) => weights.len(),
            RankStorage::Exl3 { weights, .. } => weights.len(),
        }
    }
    fn bytes_per_expert(&self, layer: usize) -> usize {
        match &*self.weights {
            RankStorage::Full(weights) => weights[layer].budget().resident_bytes / 384,
            RankStorage::Exl3 { weights, .. } => {
                weights[layer].budget.resident_bytes / weights[layer].layout.experts
            }
        }
    }
    pub fn load_exl3(
        device: Device<'a>,
        catalog: &ds41rt_loader::OfficialV41Catalog,
        layers: usize,
        budget: usize,
        directory: &Path,
    ) -> Result<Self> {
        ensure!(
            (1..=40).contains(&layers) && matches!(device.id, 0 | 1),
            "invalid TP2 EXL3 placement"
        );
        let weights = device.run(|| {
            let mut weights = Vec::with_capacity(layers);
            let mut remaining = budget;
            for layer in 0..layers {
                let weight = Exl3Weights::load(
                    device.library,
                    catalog,
                    ExpertLayer::BackboneTp2 {
                        layer,
                        rank: device.id as usize,
                    },
                    remaining,
                )?;
                remaining = remaining
                    .checked_sub(weight.budget.resident_bytes)
                    .ok_or_else(|| anyhow::anyhow!("TP2 EXL3 weights exceed budget"))?;
                weights.push(weight);
            }
            Ok(RankStorage::Exl3 {
                weights: Rc::new(weights),
                directory: directory.to_owned(),
            })
        })?;
        Ok(Self {
            device,
            weights: ManuallyDrop::new(weights),
        })
    }
    /// Load the two slices of each layer consecutively so rank 1 can reuse
    /// source pages read for rank 0 before advancing the file working set.
    /// Keep partial ranks owned throughout: errors release each allocation
    /// on its own device, including failures while loading the other rank.
    pub fn load_exl3_pair(
        devices: [Device<'a>; 2],
        catalog: &ds41rt_loader::OfficialV41Catalog,
        layers: usize,
        budgets: [usize; 2],
        directory: &Path,
    ) -> Result<[Self; 2]> {
        ensure!(
            (1..=40).contains(&layers) && devices[0].id == 0 && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid TP2 EXL3 rank pair"
        );
        let mut ranks = devices.map(|device| Self {
            device,
            weights: ManuallyDrop::new(RankStorage::Exl3 {
                weights: Rc::new(Vec::with_capacity(layers)),
                directory: directory.to_owned(),
            }),
        });
        let mut remaining = budgets;
        for layer in 0..layers {
            for rank in 0..2 {
                devices[rank].run(|| {
                    let weight = Exl3Weights::load(
                        devices[rank].library, catalog,
                        ExpertLayer::BackboneTp2 { layer, rank }, remaining[rank],
                    )?;
                    remaining[rank] = remaining[rank]
                        .checked_sub(weight.budget.resident_bytes)
                        .ok_or_else(|| anyhow::anyhow!("TP2 EXL3 weights exceed budget"))?;
                    let RankStorage::Exl3 { weights, .. } = &mut *ranks[rank].weights else {
                        unreachable!("pair initialized with EXL3 storage")
                    };
                    Rc::get_mut(weights).expect("unpublished rank weights").push(weight);
                    Ok(())
                })?;
            }
        }
        Ok(ranks)
    }
    pub fn load(
        device: Device<'a>,
        catalog: &ds41rt_loader::OfficialV41Catalog,
        layers: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=40).contains(&layers) && matches!(device.id, 0 | 1),
            "invalid TP2 encoder placement"
        );
        let weights = device.run(|| {
            let mut loaded = Vec::with_capacity(layers);
            let mut remaining = budget;
            for layer in 0..layers {
                let weight = ExpertWeights::load(
                    device.library,
                    catalog,
                    ExpertLayer::BackboneTp2 {
                        layer,
                        rank: device.id as usize,
                    },
                    remaining,
                )?;
                remaining = remaining
                    .checked_sub(weight.budget.resident_bytes)
                    .ok_or_else(|| anyhow::anyhow!("TP2 weights exceed budget"))?;
                loaded.push(weight);
            }
            Ok(loaded)
        })?;
        Ok(Self {
            device,
            weights: ManuallyDrop::new(RankStorage::Full(weights)),
        })
    }

}
impl Drop for RankWeights<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| {
            unsafe {
                ManuallyDrop::drop(&mut self.weights);
            }
            Ok(())
        }) {
            tracing::error!(%error, "releasing TP2 rank weights");
        }
    }
}

struct RankState<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}
pub(crate) struct RankInputs<'s, 'a> {
    pub wire: Ds41rtDeviceBuffer,
    pub ids: Ds41rtDeviceBuffer,
    pub routing: Ds41rtDeviceBuffer,
    pub producer: &'s Stream<'a>,
}
pub(crate) struct ExpertWave<'a> {
    reductions: [PeerReduction<'a>; 2],
    ranks: [RankWave<'a>; 2],
}
impl<'a> ExpertWave<'a> {
    /// Routed BF16 inputs use the already broadcast shared-expert values;
    /// native W4A8 and EXL3 consume the separate FP8 wire representation.
    pub fn uses_bf16_input(&self) -> bool {
        match &self.ranks[0].backend {
            RankBackend::Full { states, .. } => states[0].kernel.info().input_dtype == 1,
            RankBackend::Exl3(_) => false,
        }
    }

    /// Per GPU, per lane; immutable expert weights and CUDA state are separate.
    pub fn device_bytes(library: &ds41rt_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        RankWave::device_bytes(library, capacity)?
            .checked_add(PeerReduction::device_bytes(capacity)?)
            .ok_or_else(|| anyhow::anyhow!("TP2 expert workspace overflow"))
    }
    /// W4A4 NVFP4 per-lane workspace: kernel scratch plus the BF16 rank
    /// partial, peer staging and reduction output.
    pub fn nvfp4_device_bytes(library: &ds41rt_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        let scratch = RankWave::kernel_capacities(capacity)?
            .into_iter()
            .map(|c| {
                Ok(usize::try_from(
                    library.v41_nvfp4_tp2_expert_info(c)?.scratch_bytes,
                )?)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let reduction = PeerReduction::device_bytes(capacity)?;
        scratch
            // Rank output: six BF16 routes per token.
            .checked_add(capacity as usize * 5120 * 6 * 2)
            .and_then(|bytes| bytes.checked_add(reduction))
            .ok_or_else(|| anyhow::anyhow!("TP2 NVFP4 workspace overflow"))
    }
    pub fn exl3_device_bytes(directory: &Path, capacity: u32) -> Result<usize> {
        let directories: Vec<_> = RankWave::kernel_capacities(capacity)?.into_iter()
            .map(|c| directory.join(format!("m{c}"))).collect();
        Exl3Workspace::plan(&directories, Exl3InputFormat::Fp8K32)?
            .checked_add(PeerReduction::device_bytes(capacity)? + capacity as usize * 5120 * 4)
            .ok_or_else(|| anyhow::anyhow!("TP2 EXL3 workspace overflow"))
    }
    pub fn new(weights: [Rc<RankWeights<'a>>; 2], capacity: u32) -> Result<Self> {
        ensure!(
            weights[0].device.id == 0
                && weights[1].device.id == 1
                && std::ptr::eq(weights[0].device.library, weights[1].device.library)
                && weights[0].layers() == weights[1].layers()
                && matches!(
                    (&*weights[0].weights, &*weights[1].weights),
                    (RankStorage::Full(_), RankStorage::Full(_))
                        | (RankStorage::Exl3 { .. }, RankStorage::Exl3 { .. })
                ),
            "TP2 rank pair mismatch"
        );
        Ok(Self {
            reductions: [
                PeerReduction::new(weights[1].device, weights[0].device, capacity)?,
                PeerReduction::new(weights[0].device, weights[1].device, capacity)?,
            ],
            ranks: [
                RankWave::new(weights[0].clone(), capacity)?,
                RankWave::new(weights[1].clone(), capacity)?,
            ],
        })
    }
    /// # Safety
    /// Input producers own all writes; no conflicting input aliases until return.
    /// Each request lane must own a separate ExpertWave. The returned output view
    /// is valid until the next execution or destruction of this owner.
    pub async unsafe fn execute(
        &mut self,
        layer: usize,
        rows: u32,
        destination: usize,
        input: [RankInputs<'_, 'a>; 2],
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(destination < 2, "invalid TP2 output device");
        // Errors between rank launches must drain the already enqueued rank too.
        struct Drain<'s, 'a> {
            ranks: &'s mut [RankWave<'a>; 2],
            complete: bool,
        }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    for rank in self.ranks.iter() {
                        if let Err(error) = rank.stream.drain() {
                            tracing::error!(%error, "draining cancelled TP2 rank");
                        }
                    }
                }
            }
        }
        let mut guard = Drain {
            ranks: &mut self.ranks,
            complete: false,
        };
        let mut layouts = [Tp2RoutedLayout::default(); 2];
        for rank in 0..2 {
            layouts[rank] = unsafe {
                guard.ranks[rank].enqueue(
                    layer,
                    rows,
                    input[rank].wire,
                    input[rank].ids,
                    input[rank].routing,
                    input[rank].producer,
                )?
            };
        }
        ensure!(layouts[0] == layouts[1], "TP2 rank output layout mismatch");
        let local = &guard.ranks[destination];
        let remote = &guard.ranks[1 - destination];
        let output = unsafe {
            self.reductions[destination]
                .reduce(
                    &local.output,
                    &remote.output,
                    &local.stream,
                    &remote.stream,
                    rows,
                    layouts[0],
                )
                .await?
        };
        for rank in guard.ranks.iter() {
            if let Some([start, end]) = &rank.timing {
                // Reduction completion already proves both producers complete.
                // Timing reads add no device wait or cross-lane dependency.
                let gpu_ms = rank.stream.device.run(|| unsafe {
                    rank.stream
                        .device
                        .library
                        .cuda_event_elapsed_ms(start.raw, end.raw)
                })?;
                tracing::debug!(target: "ds41rt::timing", layer, rows,
                    gpu=rank.stream.device.id, gpu_us=gpu_ms * 1000.0,
                    bytes_per_expert=rank.weights.bytes_per_expert(layer),
                    "TP2 routed rank GPU execution");
            }
        }
        guard.complete = true;
        Ok(output)
    }
}

enum RankBackend<'a> {
    Full {
        states: Vec<RankState<'a>>,
        _scratch: Allocation<'a>,
    },
    Exl3(DeviceOwner<'a, Vec<Exl3Execution<'a>>>),
}
pub(crate) struct RankWave<'a> {
    pub stream: Stream<'a>,
    ready: Event<'a>,
    timing: Option<[Event<'a>; 2]>,
    backend: RankBackend<'a>,
    weights: Rc<RankWeights<'a>>,
    pub output: Allocation<'a>,
    capacity: u32,
}
impl<'a> RankWave<'a> {
    fn kernel_capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!((1..=4096).contains(&capacity), "invalid TP2 rank capacity");
        let compiled = [1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&c| c >= capacity)
            .unwrap();
        Ok([1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .filter(|&c| c <= compiled)
            .collect())
    }

    pub fn device_bytes(library: &ds41rt_ffi::NativeLibrary, capacity: u32) -> Result<usize> {
        let scratch = Self::kernel_capacities(capacity)?
            .into_iter()
            .map(|c| {
                Ok(usize::try_from(
                    library.v41_tp2_expert_info(c)?.scratch_bytes,
                )?)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        scratch
            .checked_add(capacity as usize * 5120 * 6 * 4)
            .ok_or_else(|| anyhow::anyhow!("TP2 rank workspace overflow"))
    }
    pub fn new(weights: Rc<RankWeights<'a>>, capacity: u32) -> Result<Self> {
        let device = weights.device;
        if let RankStorage::Exl3 {
            weights: compressed,
            directory,
        } = &*weights.weights
        {
            let stream = Stream::new(device)?;
            let states = device.own(|| {
                let capacities = Self::kernel_capacities(capacity)?;
                let directories: Vec<_> = capacities.iter().map(|c| directory.join(format!("m{c}"))).collect();
                let arena = Exl3Workspace::new(device.library, &directories)?;
                capacities
                    .into_iter()
                    .map(|c| {
                        let state = unsafe {
                            Exl3Execution::with_shared_workspace(
                                device.library,
                                compressed.clone(),
                                &directory.join(format!("m{c}")),
                                Exl3InputFormat::Fp8K32,
                                Some(arena.clone()),
                            )?
                        };
                        ensure!(
                            state.capacity() == c as usize && state.output_element_bytes() == 4,
                            "TP2 EXL3 requires matching capacity and FP32 sums"
                        );
                        Ok(state)
                    })
                    .collect::<Result<Vec<_>>>()
            })?;
            let timing = if std::env::var_os("DS41RT_TP2_TIMING").is_some() {
                Some([Event::new(device)?, Event::new(device)?])
            } else {
                None
            };
            return Ok(Self {
                stream,
                ready: Event::new(device)?,
                timing,
                backend: RankBackend::Exl3(states),
                weights,
                output: Allocation::new(device, capacity as usize * 5120 * 4)?,
                capacity,
            });
        }
        let nvfp4 = match &*weights.weights {
            RankStorage::Full(layers) => layers.first().is_some_and(|w| w.is_nvfp4()),
            RankStorage::Exl3 { .. } => false,
        };
        let capacities = Self::kernel_capacities(capacity)?;
        let kernels = device.run(|| {
            capacities
                .iter()
                .map(|&c| {
                    if nvfp4 {
                        device.library.v41_nvfp4_tp2_expert_kernel(c)
                    } else {
                        device.library.v41_tp2_expert_kernel(c)
                    }
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let bytes = kernels
            .iter()
            .map(|k| k.info().scratch_bytes as usize)
            .max()
            .unwrap();
        let stream = Stream::new(device)?;
        let scratch = Allocation::new(device, bytes)?;
        let mut states = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            let mut slots = [std::ptr::null_mut(); 44];
            let initialized = device.run(|| unsafe {
                kernel.bind_scratch(scratch.buffer.ptr, bytes as u64, &mut slots)?;
                kernel.initialize_scratch(scratch.buffer.ptr, bytes as u64, stream.raw)
            });
            let drained = stream.drain();
            initialized.and(drained)?;
            states.push(RankState { kernel, slots });
        }
        let timing = if std::env::var_os("DS41RT_TP2_TIMING").is_some() {
            Some([Event::new(device)?, Event::new(device)?])
        } else {
            None
        };
        Ok(Self {
            stream,
            ready: Event::new(device)?,
            timing,
            backend: RankBackend::Full {
                states,
                _scratch: scratch,
            },
            weights,
            // W4A4 publishes six BF16 routes per token; the native family
            // publishes FP32 route planes or token sums.
            output: Allocation::new(
                device,
                capacity as usize * 5120 * if nvfp4 { 6 * 2 } else { 6 * 4 },
            )?,
            capacity,
        })
    }
    /// # Safety
    /// Previous wave use is complete. All inputs are ordered on `producer` and
    /// retained without mutation until this stream completes. Callers must drain
    /// both rank streams on partial enqueue failure before releasing inputs.
    pub unsafe fn enqueue(
        &mut self,
        layer: usize,
        rows: u32,
        wire: Ds41rtDeviceBuffer,
        ids: Ds41rtDeviceBuffer,
        routing: Ds41rtDeviceBuffer,
        producer: &Stream<'a>,
    ) -> Result<Tp2RoutedLayout> {
        ensure!(
            rows > 0 && rows <= self.capacity && layer < self.weights.layers(),
            "TP2 layer/rows not resident"
        );
        let device = self.weights.device;
        let input_row_bytes = match &self.backend {
            RankBackend::Full { states, .. } => states[0].kernel.info().input_row_bytes()?,
            RankBackend::Exl3(_) => 5280,
        };
        for (buffer, width) in [(wire, input_row_bytes), (ids, 24), (routing, 24)] {
            ensure!(
                buffer.device_id == device.id && buffer.bytes >= rows as usize * width,
                "TP2 rank input device or extent differs"
            );
        }
        self.ready.record(producer)?;
        let queued = device.run(|| unsafe {
            device
                .library
                .cuda_stream_wait_event(self.stream.raw, self.ready.raw)?;
            if let Some([start, _]) = &self.timing {
                device
                    .library
                    .cuda_event_record(start.raw, self.stream.raw)?;
            }
            let token_sums = match (&mut self.backend, &*self.weights.weights) {
                (RankBackend::Exl3(states), RankStorage::Exl3 { .. }) => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.capacity() >= rows as usize)
                        .unwrap();
                    state
                        .launch_layer_into(
                            layer,
                            [wire, ids, routing],
                            rows as usize,
                            self.stream.raw,
                            self.output.buffer,
                        )
                        .map_err(|error| {
                            error.context(format!(
                                "EXL3 TP2 GPU {} layer {layer} rows {rows} capacity {}",
                                device.id,
                                state.capacity()
                            ))
                        })?;
                    Tp2RoutedLayout::Fp32Tokens
                }
                (RankBackend::Full { states, .. }, RankStorage::Full(weights)) => {
                    let state = states
                        .iter_mut()
                        .find(|s| s.kernel.info().capacity_rows >= rows)
                        .unwrap();
                    weights[layer].bind(&state.kernel, &mut state.slots)?;
                    state.slots[0] = wire.ptr;
                    state.slots[1] = ids.ptr;
                    state.slots[2] = routing.ptr;
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
                    state.kernel.launch(&args)?;
                    let layout = match state.kernel.output_kind() {
                        ds41rt_ffi::V41ExpertOutputKind::Bf16Routes => Tp2RoutedLayout::Bf16Routes,
                        ds41rt_ffi::V41ExpertOutputKind::Fp32Tokens => Tp2RoutedLayout::Fp32Tokens,
                        ds41rt_ffi::V41ExpertOutputKind::Fp32Routes => Tp2RoutedLayout::Fp32Routes,
                    };
                    let mut source = self.output.buffer;
                    source.ptr = state.slots[41];
                    source.bytes = rows as usize * 5120 * layout.element_bytes();
                    device.library.copy_d2d_async(
                        self.output.buffer,
                        source,
                        source.bytes,
                        self.stream.raw,
                    )?;
                    layout
                }
                _ => anyhow::bail!("TP2 execution/weight format mismatch"),
            };
            if let Some([_, end]) = &self.timing {
                device.library.cuda_event_record(end.raw, self.stream.raw)?;
            }
            Ok(token_sums)
        });
        if queued.is_err() {
            self.stream.drain()?;
        }
        queued
    }
}

pub(crate) struct PeerReduction<'a> {
    // Transfer drains before any referenced staging or output can be freed.
    transfer: PeerTransfer<'a>,
    local_ready: Event<'a>,
    staging: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
    capacity: u32,
}
impl<'a> PeerReduction<'a> {
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid TP2 reduction capacity"
        );
        Ok(capacity as usize * 5120 * (6 * 4 + 2))
    }
    pub fn new(remote: Device<'a>, local: Device<'a>, capacity: u32) -> Result<Self> {
        Self::device_bytes(capacity)?;
        ensure!(
            matches!((local.id, remote.id), (0, 1) | (1, 0)),
            "TP2 requires devices 0/1"
        );
        Ok(Self {
            transfer: PeerTransfer::new(remote, local)?,
            local_ready: Event::new(local)?,
            staging: Allocation::new(local, capacity as usize * 5120 * 6 * 4)?,
            output: Allocation::new(local, capacity as usize * 5120 * 2)?,
            reducer: local.library.v41_tp2_expert_reducer()?,
            capacity,
        })
    }
    /// # Safety
    /// Input writes are ordered on their producer streams. No conflicting
    /// aliases may touch either input until this call completes or cancellation
    /// drains the chain. The returned view lives until this owner is reused.
    pub async unsafe fn reduce(
        &mut self,
        local: &Allocation<'a>,
        remote: &Allocation<'a>,
        local_producer: &Stream<'a>,
        remote_producer: &Stream<'a>,
        rows: u32,
        layout: Tp2RoutedLayout,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "TP2 reduction exceeds capacity"
        );
        let token_sums = layout.token_sums();
        let bytes = rows as usize * 5120 * layout.element_bytes();
        ensure!(
            local.device.id == self.output.device.id
                && local.buffer.bytes >= bytes
                && std::ptr::eq(local.device.library, self.output.device.library),
            "local rank owner mismatch"
        );
        // Keep not-yet-ready transfers off the copy engine. This cooperative
        // wait belongs only to this operation's producer, never another lane.
        let started = tracing::enabled!(target: "ds41rt::timing", tracing::Level::DEBUG)
            .then(std::time::Instant::now);
        remote_producer.wait().await?;
        let producer_us = started.map(|s| s.elapsed().as_micros() as u64);
        self.local_ready.record(local_producer)?;
        let ready = &self.local_ready;
        let output = &mut self.output;
        let reducer = &self.reducer;
        unsafe {
            self.transfer
                .copy_then(
                    remote,
                    &mut self.staging,
                    remote_producer,
                    bytes,
                    |peer, stream| {
                        output
                            .device
                            .library
                            .cuda_stream_wait_event(stream, ready.raw)?;
                        let (rank0, rank1) = if local.device.id == 0 {
                            (local.buffer, peer)
                        } else {
                            (peer, local.buffer)
                        };
                        if layout == Tp2RoutedLayout::Bf16Routes {
                            reducer.reduce_bf16_routes(rank0, rank1, output.buffer, rows, stream)
                        } else {
                            reducer.reduce(rank0, rank1, output.buffer, rows, token_sums, stream)
                        }
                    },
                )
                .await?;
        }
        if let Some(started) = started {
            tracing::debug!(target: "ds41rt::timing", rows, token_sums, ?layout,
                producer_us=producer_us.unwrap(),
                copy_reduce_us=started.elapsed().as_micros() as u64-producer_us.unwrap(),
                "TP2 routed completion");
        }
        let mut result = self.output.buffer;
        result.bytes = rows as usize * 5120 * 2;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_ffi::NativeLibrary;

    #[test]
    fn routed_layout_sizes_keep_nvfp4_topk_axis() {
        assert_eq!(Tp2RoutedLayout::Fp32Routes.element_bytes(), 24);
        assert_eq!(Tp2RoutedLayout::Fp32Tokens.element_bytes(), 4);
        assert_eq!(Tp2RoutedLayout::Bf16Routes.element_bytes(), 12);
        assert!(!Tp2RoutedLayout::Bf16Routes.token_sums());
        for rows in [1, 16, 80, 4096] {
            assert!(PeerReduction::device_bytes(rows).unwrap() >=
                rows as usize * 5120 * (Tp2RoutedLayout::Bf16Routes.element_bytes() + 2));
        }
    }
    #[test]
    #[ignore = "requires EXL3 snapshot, TP2 package and CUDA"]
    fn exl3_single_rank_capacity_launch_probe() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let device = Device {
            library: &lib,
            id: 0,
        };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        let weights = Rc::new(RankWeights::load_exl3(
            device,
            &catalog,
            1,
            4_000_000_000,
            Path::new(&std::env::var("DS41RT_EXL3_AOT")?),
        )?);
        let inputs = [
            Allocation::new(device, 16 * 5280)?,
            Allocation::new(device, 16 * 24)?,
            Allocation::new(device, 16 * 24)?,
        ];
        let producer = Stream::new(device)?;
        let mut wave = RankWave::new(weights, 16)?;
        let mut wire = vec![0u8; 16 * 5280];
        for row in wire.chunks_exact_mut(5280) {
            row[..5120].fill(0x38);
            row[5120..].fill(127);
        }
        lib.copy_h2d(inputs[0].buffer, &wire)?;
        lib.copy_h2d(
            inputs[1].buffer,
            &(0..96i32)
                .flat_map(|i| (i % 6).to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        lib.copy_h2d(
            inputs[2].buffer,
            &(0..96)
                .flat_map(|_| (1f32 / 6.).to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        for rows in [16, 1, 3, 16] {
            unsafe {
                wave.enqueue(
                    0,
                    rows,
                    inputs[0].buffer,
                    inputs[1].buffer,
                    inputs[2].buffer,
                    &producer,
                )?;
            }
            wave.stream.drain()?;
            let mut bytes = vec![0; rows as usize * 5120 * 4];
            let output = Ds41rtDeviceBuffer {
                bytes: bytes.len(),
                ..wave.output.buffer
            };
            lib.copy_d2h(&mut bytes, output)?;
            let values: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.),
                "invalid rank output"
            );
            eprintln!("single rank rows={rows} passed finite/nonzero launch");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires official DS41RT_SNAPSHOT, TP2 DS41RT_NATIVE_LIB and two GPUs"]
    fn real_encoder_rank_waves_share_weights_across_independent_lanes() -> Result<()> {
        let capacity: u32 = std::env::var("DS41RT_TP2_TEST_CAPACITY")
            .unwrap_or_else(|_| "16".into())
            .parse()?;
        ensure!(
            [1, 16, 80, 256, 1024, 4096].contains(&capacity),
            "invalid fixture capacity"
        );
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let load = |rank: usize| -> Result<_> {
            if catalog.exl3().is_some() {
                RankWeights::load_exl3(
                    devices[rank],
                    &catalog,
                    1,
                    4_000_000_000,
                    Path::new(&std::env::var("DS41RT_EXL3_AOT")?),
                )
            } else {
                RankWeights::load(devices[rank], &catalog, 1, 4_000_000_000)
            }
        };
        let weights = [Rc::new(load(0)?), Rc::new(load(1)?)];
        if catalog.exl3().is_some() {
            for (rank, owner) in weights.iter().enumerate() {
                if let RankStorage::Exl3 { weights, .. } = &*owner.weights {
                    eprintln!(
                        "EXL3 rank {rank} resident bytes: {}",
                        weights
                            .iter()
                            .map(|w| w.budget.resident_bytes)
                            .sum::<usize>()
                    );
                }
            }
            let bytes = ExpertWave::exl3_device_bytes(
                Path::new(&std::env::var("DS41RT_EXL3_AOT")?),
                capacity,
            )?;
            eprintln!("EXL3 TP2 workspace per GPU per lane: {bytes}");
        }
        let mut first = ExpertWave::new(weights.clone(), capacity)?;
        let mut second = ExpertWave::new(weights, capacity)?;
        let mut inputs = Vec::new();
        for device in devices {
            inputs.push((
                Allocation::new(device, capacity as usize * 5280)?,
                Allocation::new(device, capacity as usize * 24)?,
                Allocation::new(device, capacity as usize * 24)?,
                Stream::new(device)?,
            ));
        }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, nonzero) in [
            (capacity, false),
            (1, false),
            (capacity, true),
            (capacity.min(3), true),
            (1, true),
            (capacity, false),
        ] {
            let mut wire = vec![0u8; capacity as usize * 5280];
            for row in wire.chunks_exact_mut(5280) {
                row[..5120].fill(if nonzero { 0x38 } else { 0 });
                row[5120..].fill(127);
            }
            let ids: Vec<u8> = (0..capacity as i32 * 6)
                .flat_map(|i| (i % 6).to_ne_bytes())
                .collect();
            let routing: Vec<u8> = (0..capacity * 6)
                .flat_map(|_| (1f32 / 6.).to_ne_bytes())
                .collect();
            for (rank, (w, i, r, _)) in inputs.iter().enumerate() {
                devices[rank].run(|| {
                    lib.copy_h2d(w.buffer, &wire)?;
                    lib.copy_h2d(i.buffer, &ids)?;
                    lib.copy_h2d(r.buffer, &routing)
                })?;
            }
            let make_inputs = || {
                std::array::from_fn(|rank| RankInputs {
                    wire: inputs[rank].0.buffer,
                    ids: inputs[rank].1.buffer,
                    routing: inputs[rank].2.buffer,
                    producer: &inputs[rank].3,
                })
            };
            let (x, y) = runtime.block_on(async {
                if std::env::var_os("DS41RT_TP2_TEST_SERIAL").is_some() {
                    let x = unsafe { first.execute(0, rows, 0, make_inputs()).await };
                    let y = unsafe { second.execute(0, rows, 1, make_inputs()).await };
                    (x, y)
                } else {
                    tokio::join!(
                        unsafe { first.execute(0, rows, 0, make_inputs()) },
                        unsafe { second.execute(0, rows, 1, make_inputs()) }
                    )
                }
            });
            eprintln!(
                "rows={rows} nonzero={nonzero} lane0={:?} lane1={:?}",
                x.as_ref().map(|_| ()),
                y.as_ref().map(|_| ())
            );
            let mut results = Vec::new();
            for (device, output) in [(devices[0], x?), (devices[1], y?)] {
                let mut host = vec![0; output.bytes];
                device.run(|| lib.copy_d2h(&mut host, output))?;
                assert!(host
                    .chunks_exact(2)
                    .all(|v| (u16::from_ne_bytes([v[0], v[1]]) & 0x7f80) != 0x7f80));
                assert_eq!(
                    host.chunks_exact(2)
                        .any(|v| u16::from_ne_bytes([v[0], v[1]]) & 0x7fff != 0),
                    nonzero
                );
                results.push(host);
            }
            let atomic = match &first.ranks[0].backend {
                RankBackend::Full { states, .. } => {
                    states
                        .iter()
                        .find(|state| state.kernel.info().capacity_rows >= rows)
                        .unwrap()
                        .kernel
                        .info()
                        .abi_version
                        == 3
                }
                RankBackend::Exl3(_) => false,
            };
            if atomic {
                // Independent atomic executions can straddle a BF16 rounding
                // boundary. Bound both each value and aggregate error; do not
                // dump multi-megabyte buffers when this fixture fails.
                let mut squared_error = 0f64;
                let mut squared_signal = 0f64;
                let mut changed = 0usize;
                for (a, b) in results[0].chunks_exact(2).zip(results[1].chunks_exact(2)) {
                    let a = u16::from_ne_bytes([a[0], a[1]]);
                    let b = u16::from_ne_bytes([b[0], b[1]]);
                    assert!(
                        a.abs_diff(b) <= 1,
                        "atomic output differs by more than one BF16 step: {a} vs {b}"
                    );
                    changed += usize::from(a != b);
                    let x = f32::from_bits((a as u32) << 16) as f64;
                    let y = f32::from_bits((b as u32) << 16) as f64;
                    squared_error += (x - y) * (x - y);
                    squared_signal += x * x;
                }
                let relative = (squared_error / squared_signal.max(f64::MIN_POSITIVE)).sqrt();
                assert!(relative < 1e-6, "atomic lane relative L2: {relative}");
                eprintln!("TP2 atomic rows={rows} changed={changed} relative_l2={relative}");
            } else {
                assert!(
                    results[0] == results[1],
                    "deterministic TP2 lane outputs differ at rows={rows}"
                );
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        // Releasing one lane must not unload AOT modules retained by the other.
        drop(second);
        let make_inputs = || {
            std::array::from_fn(|rank| RankInputs {
                wire: inputs[rank].0.buffer,
                ids: inputs[rank].1.buffer,
                routing: inputs[rank].2.buffer,
                producer: &inputs[rank].3,
            })
        };
        let output = runtime.block_on(unsafe { first.execute(0, capacity, 0, make_inputs()) })?;
        let mut host = vec![0; output.bytes];
        devices[0].run(|| lib.copy_d2h(&mut host, output))?;
        ensure!(
            host.chunks_exact(2)
                .all(|v| u16::from_ne_bytes([v[0], v[1]]) & 0x7fff == 0),
            "surviving lane lost its zero-input result"
        );
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("surviving lane after peer lane drop passed");
        Ok(())
    }

    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA devices"]
    fn opposite_lane_peer_reductions_preserve_results_and_device() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d0 = Device {
            library: &lib,
            id: 0,
        };
        let d1 = Device {
            library: &lib,
            id: 1,
        };
        let capacity = 16;
        let bytes = capacity * 5120 * 6 * 4;
        let a = Allocation::new(d0, bytes)?;
        let b = Allocation::new(d1, bytes)?;
        let p0 = Stream::new(d0)?;
        let p1 = Stream::new(d1)?;
        let mut lane0 = PeerReduction::new(d1, d0, capacity as u32)?;
        let mut lane1 = PeerReduction::new(d0, d1, capacity as u32)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, token_sums, va, vb) in [
            (1, false, 1f32, 2f32),
            (16, true, 3., -1.),
            (16, false, -2., 4.),
        ] {
            let host_a: Vec<u8> = va.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            let host_b: Vec<u8> = vb.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            d0.run(|| lib.copy_h2d(a.buffer, &host_a))?;
            d1.run(|| lib.copy_h2d(b.buffer, &host_b))?;
            let (x, y) = runtime.block_on(async {
                tokio::join!(
                    unsafe { lane0.reduce(&a, &b, &p0, &p1, rows, if token_sums { Tp2RoutedLayout::Fp32Tokens } else { Tp2RoutedLayout::Fp32Routes }) },
                    unsafe { lane1.reduce(&b, &a, &p1, &p0, rows, if token_sums { Tp2RoutedLayout::Fp32Tokens } else { Tp2RoutedLayout::Fp32Routes }) }
                )
            });
            let expected = (((va + vb) * if token_sums { 1. } else { 6. }).to_bits() >> 16) as u16;
            for (device, output) in [(d0, x?), (d1, y?)] {
                let mut host = vec![0; output.bytes];
                device.run(|| lib.copy_d2h(&mut host, output))?;
                assert!(host
                    .chunks_exact(2)
                    .all(|v| u16::from_ne_bytes([v[0], v[1]]) == expected));
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tp2/reference_tests.rs"]
mod reference_tests;
