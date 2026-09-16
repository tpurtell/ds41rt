//! Routed draft compute with fixed inputs/output and lane-owned graph storage.
use super::{DsparkRouter, DsparkSharedFfn, DsparkWeights};
use crate::v41_experts::exl3::execution::Exl3InputFormat;
use crate::v41_experts::{
    exl3::{execution::Exl3Execution, Exl3Weights},
    ExpertExecution,
};
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Bf16Add};
use std::{ffi::c_void, path::Path, rc::Rc};

pub(super) enum DraftExperts<'w, 'a> {
    Full(ExpertExecution<'w, 'a>),
    Exl3(CompressedDraftExperts<'w, 'a>),
}
impl DraftExperts<'_, '_> {
    pub fn stream(&self) -> *mut c_void {
        match self {
            Self::Full(v) => v.stream(),
            Self::Exl3(v) => v.stream.raw,
        }
    }
    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Full(v) => v.synchronize(),
            Self::Exl3(v) => unsafe { v.stream.library.cuda_stream_synchronize(v.stream.raw) },
        }
    }
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        match self {
            Self::Full(v) => v.inputs(),
            Self::Exl3(v) => v.inputs.each_ref().map(|b| b.buffer),
        }
    }
    pub fn output(&self) -> Option<Ds41rtDeviceBuffer> {
        match self {
            Self::Full(v) => v.output(),
            Self::Exl3(v) => Some(v.output.buffer),
        }
    }
    pub unsafe fn enqueue_draft_ffn_on(
        &mut self,
        router: &mut DsparkRouter<'_, '_>,
        shared: &mut DsparkSharedFfn<'_, '_>,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        match self {
            Self::Full(v) => unsafe { v.enqueue_draft_ffn_on(router, shared, rows, stream) },
            Self::Exl3(v) => unsafe { v.enqueue(router, shared, rows, stream) },
        }
    }
}

pub(super) struct CompressedDraftExperts<'w, 'a> {
    // Drain owned stream before any referenced stage workspace is released.
    stream: LoadStream<'a>,
    states: Vec<Exl3Execution<'a>>,
    inputs: [DeviceAllocation<'a>; 3],
    shared: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    add: V41Bf16Add<'a>,
    owner: &'w DsparkWeights<'a>,
    stage: usize,
    capacity: u32,
}
impl<'w, 'a> CompressedDraftExperts<'w, 'a> {
    fn capacities(capacity: u32) -> Result<Vec<u32>> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid EXL3 draft capacity"
        );
        let maximum = [1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .find(|&c| c >= capacity)
            .unwrap();
        Ok([1, 16, 80, 256, 1024, 4096]
            .into_iter()
            .filter(|&c| c <= maximum)
            .collect())
    }
    pub fn device_bytes(directory: &Path, capacity: u32) -> Result<usize> {
        Self::capacities(capacity)?.into_iter().try_fold(
            capacity as usize * (10240 * 3 + 12 * 2),
            |bytes, c| {
                bytes
                    .checked_add(Exl3Execution::plan(
                        &directory.join(format!("m{c}")),
                        Exl3InputFormat::Bf16,
                    )?)
                    .context("EXL3 draft workspace overflow")
            },
        )
    }
    pub unsafe fn new(
        owner: &'w DsparkWeights<'a>,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        stage: usize,
        capacity: u32,
    ) -> Result<Self> {
        ensure!(
            stage < 3 && weights.len() == 3,
            "EXL3 draft requires three stage weights"
        );
        for (index, weight) in weights.iter().enumerate() {
            ensure!(
                matches!(weight.layout.layer, ds41rt_loader::V41Exl3Layer::Dspark(s) if s == index)
                    && weight.layout.world == 1
                    && weight.layout.rank == 0,
                "EXL3 draft stage placement differs"
            );
        }
        let library = owner.library;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut states = Vec::new();
        for c in Self::capacities(capacity)? {
            let state = unsafe {
                Exl3Execution::new(library, weights.clone(), &directory.join(format!("m{c}")))?
            };
            ensure!(
                state.capacity() == c as usize && state.output_element_bytes() == 2,
                "EXL3 draft requires matching capacity and BF16 output"
            );
            states.push(state);
        }
        Ok(Self {
            stream,
            states,
            inputs: [
                DeviceAllocation::new(library, capacity as usize * 10240)?,
                DeviceAllocation::new(library, capacity as usize * 12)?,
                DeviceAllocation::new(library, capacity as usize * 12)?,
            ],
            shared: DeviceAllocation::new(library, capacity as usize * 10240)?,
            output: DeviceAllocation::new(library, capacity as usize * 10240)?,
            add: library.v41_bf16_add()?,
            owner,
            stage,
            capacity,
        })
    }
    // The containing FFN/stage owns and drains the supplied capture stream.
    unsafe fn enqueue(
        &mut self,
        router: &mut DsparkRouter<'_, '_>,
        shared: &mut DsparkSharedFfn<'_, '_>,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "EXL3 draft rows exceed capacity"
        );
        ensure!(
            router.matches_stage(self.owner, self.stage)
                && shared.matches_stage(self.owner, self.stage),
            "EXL3 draft FFN stage owners differ"
        );
        let inputs = self.inputs.each_ref().map(|b| b.buffer);
        unsafe {
            router.enqueue(inputs, rows as usize, stream)?;
            shared.enqueue(inputs[0], self.shared.buffer, rows, stream)?;
            let state = self
                .states
                .iter_mut()
                .find(|s| s.capacity() >= rows as usize)
                .unwrap();
            let routed = state.launch_layer(self.stage, inputs, rows as usize, stream)?;
            self.add.launch(
                routed,
                self.shared.buffer,
                self.output.buffer,
                rows as usize * 5120,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod reference_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_ffi::NativeLibrary;

    fn initialize(lib: &NativeLibrary, inputs: [Ds41rtDeviceBuffer; 2], seed: usize) -> Result<()> {
        let residual: Vec<u8> = (0..inputs[0].bytes / 2)
            .flat_map(|i| {
                let value = (((i * 7 + seed) % 31) as f32 - 15.) * 0.0625;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let pre: Vec<u8> = (0..inputs[1].bytes / 4)
            .flat_map(|_| 0.25f32.to_le_bytes())
            .collect();
        lib.copy_h2d(inputs[0], &residual)?;
        lib.copy_h2d(inputs[1], &pre)
    }
    fn download(lib: &NativeLibrary, outputs: [Ds41rtDeviceBuffer; 2]) -> Result<Vec<Vec<u8>>> {
        outputs
            .into_iter()
            .map(|buffer| {
                let mut bytes = vec![0; buffer.bytes];
                lib.copy_d2h(&mut bytes, buffer)?;
                Ok(bytes)
            })
            .collect()
    }

    #[test]
    #[ignore = "requires EXL3 snapshot, RTX native library and dSpark AOT package"]
    fn exl3_dspark_all_stage_ffns_replay_changed_inputs() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let device: i32 = std::env::var("DS41RT_DSPARK_TEST_DEVICE").unwrap_or_else(|_| "0".into()).parse()?;
        ensure!(matches!(device, 0 | 1), "invalid dSpark test device");
        lib.cuda_set_device(device)?;
        println!("dSpark test device={device}");
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        let directory = std::path::PathBuf::from(std::env::var("DS41RT_EXL3_AOT")?);
        let weights = DsparkWeights::load_with_width(
            &lib,
            &catalog,
            16,
            2,
            32usize << 30,
            16 << 20,
            5,
            Some(&directory),
        )?;
        let budget = weights.budget();
        println!(
            "EXL3 dSpark experts={} auxiliary={} execution_per_wave={}",
            budget.expert_resident_bytes,
            budget.auxiliary_resident_bytes,
            budget.execution_bytes_per_wave
        );
        for stage in 0..3 {
            let bytes = weights.ffn_bytes(16)?;
            let mut eager = weights.ffn(stage, 16, bytes)?;
            let mut graph = weights.ffn(stage, 16, bytes)?;
            initialize(&lib, graph.inputs(), 0)?;
            unsafe {
                graph.capture(3)?;
            }
            for seed in [1, 9, 2] {
                initialize(&lib, eager.inputs(), seed)?;
                initialize(&lib, graph.inputs(), seed)?;
                let expected = download(&lib, unsafe { eager.execute(3)? })?;
                let actual = download(&lib, unsafe { graph.replay(3)? })?;
                ensure!(
                    actual == expected,
                    "EXL3 dSpark stage {stage} seed {seed} eager/graph mismatch"
                );
                ensure!(
                    actual[0]
                        .chunks_exact(2)
                        .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80)
                        && actual[0].iter().any(|&v| v != 0),
                    "invalid dSpark residual"
                );
                ensure!(
                    actual[1]
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                    "invalid dSpark pre-mix"
                );
            }
            assert!(unsafe { graph.replay(1) }.is_err());
            assert!(graph.output().is_err());
            unsafe {
                graph.replay(3)?;
            }
            assert!(unsafe { eager.execute(0) }.is_err());
            assert!(eager.output().is_err());
            unsafe {
                eager.execute(1)?;
                eager.execute(16)?;
                eager.execute(3)?;
            }
            println!("stage={stage}: changed-input eager/graph parity, independent storage, 1/16/3 live rows and rejection recovery passed");
        }
        Ok(())
    }
}
