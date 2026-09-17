//! Native draft expert halves, with device-scoped ownership and fixed scratch.
use crate::v41_experts::{ExpertLayer, ExpertWeights};
use crate::v41_memory::device::{Allocation, Device, DeviceOwner, Stream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41ExpertKernel, V41ExpertLaunchArgs};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, rc::Rc};

pub(super) struct Weights<'a> {
    pub device: Device<'a>,
    stages: DeviceOwner<'a, Vec<ExpertWeights<'a>>>,
}
impl<'a> Weights<'a> {
    /// Alternate rank loads within each stage to reuse checkpoint pages. Every
    /// partially loaded rank retains its device owner on an early return.
    pub fn load_pair(
        devices: [Device<'a>; 2],
        catalog: &OfficialV41Catalog,
        budgets: [usize; 2],
    ) -> Result<[Rc<Self>; 2]> {
        ensure!(
            devices[0].id == 0
                && devices[1].id == 1
                && std::ptr::eq(devices[0].library, devices[1].library),
            "invalid draft TP2 device pair"
        );
        let mut pair = [
            Self {
                device: devices[0],
                stages: devices[0].own(|| Ok(Vec::with_capacity(3)))?,
            },
            Self {
                device: devices[1],
                stages: devices[1].own(|| Ok(Vec::with_capacity(3)))?,
            },
        ];
        let mut remaining = budgets;
        for stage in 0..3 {
            for rank in 0..2 {
                devices[rank].run(|| {
                    let weight = ExpertWeights::load(
                        devices[rank].library,
                        catalog,
                        ExpertLayer::DsparkTp2 { stage, rank },
                        remaining[rank],
                    )?;
                    remaining[rank] = remaining[rank]
                        .checked_sub(weight.budget().resident_bytes)
                        .context("draft TP2 weight budget overflow")?;
                    pair[rank].stages.push(weight);
                    Ok(())
                })?;
            }
        }
        Ok(pair.map(Rc::new))
    }
    pub fn resident_bytes(&self) -> usize {
        self.stages.iter().map(|w| w.budget().resident_bytes).sum()
    }
}

struct State<'a> {
    kernel: V41ExpertKernel<'a>,
    slots: [*mut c_void; 44],
}
/// One lane's workspace for one rank. Enqueue borrows an external stream so the
/// containing draft chain can capture both ranks in its single CUDA graph.
/// That chain must drain its streams and destroy graphs before dropping this.
pub(super) struct Wave<'a> {
    states: Vec<State<'a>>,
    _scratch: Allocation<'a>,
    weights: Rc<Weights<'a>>,
    capacity: u32,
}
impl<'a> Wave<'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!((1..=4096).contains(&capacity), "invalid draft TP2 capacity");
        let maximum = [1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&v| v >= capacity)
            .unwrap();
        Ok([1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .filter(|&v| v <= maximum)
            .collect())
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::capacities(capacity)?
            .into_iter()
            .map(|c| {
                Ok(usize::try_from(
                    library.v41_dspark_tp2_expert_info(c)?.scratch_bytes,
                )?)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .context("missing draft TP2 capacity")
    }
    pub fn new(weights: Rc<Weights<'a>>, capacity: u32) -> Result<Self> {
        let device = weights.device;
        let bytes = Self::device_bytes(device.library, capacity)?;
        let scratch = Allocation::new(device, bytes)?;
        let stream = Stream::new(device)?;
        let mut states = Vec::new();
        for cap in Self::capacities(capacity)? {
            let kernel = device.run(|| device.library.v41_dspark_tp2_expert_kernel(cap))?;
            let mut slots = [std::ptr::null_mut(); 44];
            let initialized = device.run(|| unsafe {
                kernel.bind_scratch(scratch.buffer.ptr, bytes as u64, &mut slots)?;
                kernel.initialize_scratch(scratch.buffer.ptr, bytes as u64, stream.raw)
            });
            let drained = stream.drain();
            initialized.and(drained)?;
            states.push(State { kernel, slots });
        }
        Ok(Self {
            states,
            _scratch: scratch,
            weights,
            capacity,
        })
    }
    /// Inputs and stream belong to this rank and remain alive until completion.
    /// Output borrows shared scratch and must be consumed before the next stage.
    /// The caller drains external streams on cancellation or partial enqueue.
    pub unsafe fn enqueue(
        &mut self,
        stage: usize,
        rows: u32,
        inputs: [Ds41rtDeviceBuffer; 3],
        stream: *mut c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            stage < 3 && rows > 0 && rows <= self.capacity,
            "draft TP2 stage or rows out of range"
        );
        let device = self.weights.device;
        for (buffer, width) in inputs.iter().zip([5280, 12, 12]) {
            ensure!(
                buffer.device_id == device.id && buffer.bytes >= rows as usize * width,
                "draft TP2 input device or extent differs"
            );
        }
        device.run(|| unsafe {
            let state = self
                .states
                .iter_mut()
                .find(|s| s.kernel.info().capacity_rows >= rows)
                .unwrap();
            self.weights.stages[stage].bind(&state.kernel, &mut state.slots)?;
            for (slot, input) in state.slots[..3].iter_mut().zip(inputs) {
                *slot = input.ptr;
            }
            let info = state.kernel.info();
            state.kernel.launch(&V41ExpertLaunchArgs {
                tensors: state.slots,
                num_tokens: rows as i32,
                max_rows: info.max_rows,
                scatter_rows: rows as i32 * 3,
                rows_padded: info.rows_padded,
                max_tasks: info.max_tasks,
                max_phys_tiles: info.max_phys_tiles,
                max_active_clusters: info.max_active_clusters,
                stream,
            })?;
            let mut output = self._scratch.buffer;
            output.ptr = state.slots[41];
            output.bytes = rows as usize * 3 * 5120 * 4;
            Ok(output)
        })
    }
}
