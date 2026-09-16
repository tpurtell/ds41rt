//! Lane-owned draft terminal: RTX1 normalization/sampling, TP2 shared head.
use super::*;
use crate::v41_memory::device::{Device, DeviceOwner};
use crate::v41_target_head::distributed::DistributedVocabularyWave;
use crate::v41_tensors::VocabularyShard;

#[derive(Clone, Copy)]
enum Phase { Normalize, Project, Assemble, Sample }

pub(crate) struct DistributedDsparkTerminal<'w, 'a> {
    terminal: DeviceOwner<'a, DsparkTerminal<'w, 'a>>,
    head: DistributedVocabularyWave<'w, 'a>,
    graphs: [[Option<*mut c_void>; 16]; 2],
    ready: Option<usize>,
    pending: Option<(usize, Phase)>,
}
impl<'w, 'a> DistributedDsparkTerminal<'w, 'a> {
    pub fn device_bytes(capacity: usize, split: usize) -> Result<[usize; 2]> {
        Self::device_bytes_with_width(capacity, split, 5)
    }
    pub fn device_bytes_with_width(capacity: usize, split: usize, width: usize) -> Result<[usize; 2]> {
        let terminal = DsparkTerminal::device_bytes_with_width(capacity, width)? - V41VocabularyProjection::WORKSPACE_BYTES;
        let mut bytes = DistributedVocabularyWave::device_bytes(capacity * width, split)?;
        bytes[1] += terminal;
        Ok(bytes)
    }
    pub fn new(devices: [Device<'a>; 2], weights: &'w DsparkWeights<'a>,
        shards: [&'w VocabularyShard<'a>; 2], capacity: usize, budgets: [usize; 2]) -> Result<Self> {
        let required = Self::device_bytes_with_width(capacity, shards[0].tokens().end, weights.draft_width)?;
        ensure!(required.iter().zip(budgets).all(|(need, budget)| *need <= budget),
            "distributed draft terminal exceeds budget");
        ensure!(weights.tensor("mtp.2.norm.weight")?.device_id == devices[1].id,
            "draft terminal weights must reside on rank 1");
        let head_bytes = DistributedVocabularyWave::device_bytes(capacity * weights.draft_width, shards[0].tokens().end)?;
        Ok(Self {
            terminal: devices[1].own(|| weights.terminal_storage(None, capacity, required[1] - head_bytes[1]))?,
            head: DistributedVocabularyWave::new(devices, shards, capacity * weights.draft_width, head_bytes)?,
            graphs: [[None; 16]; 2], ready: None, pending: None,
        })
    }
    /// GPU1 residual/pre-mix and anchor storage, with the ordinary terminal layout.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] { self.terminal.inputs() }
    pub fn validate_sampling(&self, requests: usize) -> Result<()> { self.terminal.validate_sampling(requests) }
    pub fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed terminal still pending");
        self.ready = None;
        self.terminal.stage_sampling(rngs, temperatures)
    }
    unsafe fn enqueue_stage(&self, requests: usize, sampling: bool) -> Result<()> {
        let device = self.terminal.device;
        let terminal = self.terminal.get();
        device.run(|| unsafe {
            if let Some(graph) = self.graphs[usize::from(sampling)][requests - 1] {
                device.library.cuda_graph_launch(graph, terminal.stream.raw)
            } else if sampling { terminal.enqueue_sampling_on(requests, terminal.stream.raw) }
            else { terminal.enqueue_normalize_on(requests, terminal.stream.raw) }
        })
    }
    unsafe fn capture_stage(&mut self, requests: usize, sampling: bool) -> Result<()> {
        let device = self.terminal.device;
        let terminal = self.terminal.get();
        let mode = usize::from(sampling);
        if self.graphs[mode][requests - 1].is_none() {
            let graph = device.run(|| unsafe {
                device.library.cuda_graph_begin_capture(terminal.stream.raw)?;
                let queued = if sampling { terminal.enqueue_sampling_on(requests, terminal.stream.raw) }
                    else { terminal.enqueue_normalize_on(requests, terminal.stream.raw) };
                let captured = device.library.cuda_graph_end_capture(terminal.stream.raw);
                match (queued, captured) {
                    (Ok(()), Ok(graph)) => Ok(graph),
                    (Err(error), Ok(graph)) => { device.library.cuda_graph_exec_destroy(graph)?; Err(error) }
                    (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
                }
            })?;
            self.graphs[mode][requests - 1] = Some(graph);
        }
        Ok(())
    }
    /// # Safety
    /// Inputs have the ordinary terminal layout and finite values, with anchors
    /// below 129280. GPU1 producers have completed. Inputs and borrowed weights
    /// stay immutable until return/cancellation. No external consumer races outputs.
    pub async unsafe fn execute(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        unsafe { self.begin(requests)?; }
        struct Cancel<'s, 'w, 'a> { terminal: &'s mut DistributedDsparkTerminal<'w, 'a>, armed: bool }
        impl Drop for Cancel<'_, '_, '_> {
            fn drop(&mut self) { if self.armed { self.terminal.cancel(); } }
        }
        let mut pending = Cancel { terminal: self, armed: true };
        while !pending.terminal.poll()? { tokio::task::yield_now().await; }
        pending.armed = false;
        pending.terminal.output()
    }
    /// # Safety
    /// Same inputs as `execute`, retained without conflicting access until
    /// successful polling or cancellation. No scheduler borrow survives this call.
    pub unsafe fn begin(&mut self, requests: usize) -> Result<()> {
        ensure!(self.pending.is_none(), "distributed terminal still pending");
        self.ready = None;
        self.terminal.validate_sampling(requests)?;
        self.pending = Some((requests, Phase::Normalize));
        let device = self.terminal.device;
        let queued = device.run(|| unsafe {
            let terminal = self.terminal.get_mut();
            terminal.upload_sampling_on(terminal.stream.raw)
        }).and_then(|_| unsafe { self.enqueue_stage(requests, false) });
        if queued.is_err() { self.cancel(); }
        queued
    }
    pub fn poll(&mut self) -> Result<bool> {
        ensure!(self.pending.is_some(), "distributed terminal not pending");
        let result = (|| -> Result<bool> {
            loop {
                let (requests, phase) = self.pending.unwrap();
                match phase {
                    Phase::Normalize | Phase::Sample => {
                        let device = self.terminal.device;
                        if !device.run(|| unsafe { device.library.cuda_stream_query(self.terminal.stream.raw) })? {
                            return Ok(false);
                        }
                        let sampling = matches!(phase, Phase::Sample);
                        unsafe { self.capture_stage(requests, sampling)?; }
                        if sampling {
                            self.pending = None;
                            self.ready = Some(requests);
                            return Ok(true);
                        }
                        unsafe { self.head.begin_logits(self.terminal.normalized.buffer, requests * self.terminal.weights.draft_width)?; }
                        self.pending = Some((requests, Phase::Project));
                    }
                    Phase::Project => {
                        if !self.head.poll_logits()? { return Ok(false); }
                        unsafe { self.head.begin_copy_logits(self.terminal.shared_logits.buffer)?; }
                        self.pending = Some((requests, Phase::Assemble));
                    }
                    Phase::Assemble => {
                        if !self.head.poll_copy_logits()? { return Ok(false); }
                        unsafe { self.enqueue_stage(requests, true)?; }
                        self.pending = Some((requests, Phase::Sample));
                    }
                }
            }
        })();
        if result.is_err() { self.cancel(); }
        result
    }
    pub fn cancel(&mut self) {
        if self.pending.take().is_some() {
            self.head.cancel_copy_logits();
            self.head.cancel_logits();
            if let Err(error) = self.terminal.device.run(|| self.terminal.synchronize()) {
                tracing::error!(%error, "draining cancelled distributed terminal");
            }
        }
        self.ready = None;
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.terminal.output_storage(self.ready.context("distributed draft terminal output unpublished")?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_dspark_terminal_matches_full_head() -> Result<()> {
        check_terminal_width(5)
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_dspark_k7_terminal_matches_full_head() -> Result<()> {
        check_terminal_width(7)
    }
    fn check_terminal_width(width: usize) -> Result<()> {
        let lib = unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let shards = [devices[0].own(|| VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
            devices[1].own(|| VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?];
        let full = devices[1].own(|| VocabularyHead::load(&lib, &catalog, 2 << 30, 16 << 20))?;
        let weights = devices[1].own(|| DsparkWeights::load_with_width(&lib, &catalog, if width == 7 { 256 } else { 80 }, 1, 32usize << 30, 16 << 20, width, None))?;
        let budgets = DistributedDsparkTerminal::device_bytes_with_width(16, 64640, width)?;
        let mut lanes = [DistributedDsparkTerminal::new(devices, &weights, [&shards[0], &shards[1]], 16, budgets)?,
            DistributedDsparkTerminal::new(devices, &weights, [&shards[0], &shards[1]], 16, budgets)?];
        let mut reference = devices[1].own(|| weights.terminal(&full, 16, DsparkTerminal::device_bytes_with_width(16, width)?))?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let download = |buffers: [Ds41rtDeviceBuffer; 3]| -> Result<Vec<Vec<u8>>> {
            buffers.into_iter().map(|buffer| {
                let mut bytes = vec![0; buffer.bytes];
                devices[1].run(|| lib.copy_d2h(&mut bytes, buffer))?;
                Ok(bytes)
            }).collect()
        };
        for (cycle, count) in [1, 3, 8, 16, 3].into_iter().enumerate() {
            let mut expected = Vec::new();
            for (lane, terminal) in lanes.iter_mut().enumerate() {
                let residual: Vec<u8> = (0..count * width * 4 * 5120).flat_map(|i| {
                    let value = ((i * (17 + lane * 2) + cycle * 7) % 127) as f32 / 64. - 1.;
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                }).collect();
                let pre: Vec<u8> = (0..count * width * 4).flat_map(|i| (0.1f32 * (1 + i % 4) as f32).to_ne_bytes()).collect();
                let anchors: Vec<u8> = (0..count).flat_map(|i| ((i * 31 + lane * 11 + cycle * 5) as u32).to_ne_bytes()).collect();
                for inputs in [terminal.inputs(), reference.inputs()] {
                    for (buffer, bytes) in inputs.into_iter().zip([&residual, &pre, &anchors]) {
                        devices[1].run(|| lib.copy_h2d(buffer, bytes))?;
                    }
                }
                let seed = (lane * 100 + cycle * 1000) as u64;
                let mut rngs: Vec<_> = (0..count).map(|i| DsparkRng::new(seed + i as u64)).collect();
                let mut reference_rngs: Vec<_> = (0..count).map(|i| DsparkRng::new(seed + i as u64)).collect();
                let temperatures: Vec<_> = (0..count).map(|i| if (i + cycle) % 2 == 0 { 0. } else { 0.7 }).collect();
                terminal.stage_sampling(&mut rngs.iter_mut().collect::<Vec<_>>(), &temperatures)?;
                devices[1].run(|| reference.prepare_sampling(&mut reference_rngs.iter_mut().collect::<Vec<_>>(), &temperatures))?;
                let output = devices[1].run(|| unsafe { reference.execute(count) })?;
                expected.push(download(output)?);
            }
            for replay in 0..2 {
                let [first, second] = &mut lanes;
                runtime.block_on(async {
                    let (a, b) = tokio::join!(unsafe { first.execute(count) }, unsafe { second.execute(count) });
                    a.and(b).map(|_| ())
                })?;
                for (lane, terminal) in lanes.iter().enumerate() {
                    let actual = download(terminal.output()?)?;
                    for part in 0..3 {
                        ensure!(actual[part] == expected[lane][part],
                            "distributed terminal differs: cycle={cycle} requests={count} lane={lane} replay={replay} part={part}");
                    }
                }
            }
            eprintln!("PASS distributed dSpark K{width} requests={count}: tokens, corrected logits and confidence exact; concurrent cold/replay, mixed temperatures");
        }
        use std::{future::Future, task::{Context, Poll, Waker}};
        unsafe { lanes[0].begin(3)?; lanes[1].begin(3)?; }
        assert!(unsafe { lanes[0].begin(3) }.is_err());
        assert!(lanes[0].stage_sampling(&mut [], &[]).is_err());
        while !lanes[1].poll()? { std::thread::yield_now(); }
        assert!(lanes[1].output().is_ok());
        assert!(lanes[0].output().is_err());
        assert!(matches!(lanes[0].pending, Some((3, Phase::Normalize))));
        lanes[0].cancel();
        assert!(lanes[0].output().is_err());
        eprintln!("PASS terminal begin/poll: one lane completes while its peer remains unpolled; pending reuse rejected");
        let mut pending = Box::pin(unsafe { lanes[0].execute(3) });
        assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(pending);
        assert!(lanes[0].output().is_err());
        runtime.block_on(unsafe { lanes[0].execute(3) })?;
        assert!(lanes[0].output().is_ok());
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS distributed dSpark cancellation, reuse and device restoration");
        Ok(())
    }
}
impl Drop for DistributedDsparkTerminal<'_, '_> {
    fn drop(&mut self) {
        self.cancel();
        let device = self.terminal.device;
        if let Err(error) = device.run(|| {
            self.terminal.synchronize()?;
            for graph in self.graphs.iter_mut().flatten().filter_map(Option::take) {
                unsafe { device.library.cuda_graph_exec_destroy(graph)?; }
            }
            Ok(())
        }) { tracing::error!(%error, "destroying distributed draft terminal graphs"); }
    }
}
