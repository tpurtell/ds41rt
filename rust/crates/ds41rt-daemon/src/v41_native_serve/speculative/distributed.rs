//! GPU1-owned request/cache lifecycle with independent distributed draft chains.
use super::*;
use crate::v41_experts::dspark::DistributedDsparkChain;
use crate::v41_memory::device::{Device, DeviceOwner};
use crate::v41_tensors::VocabularyShard;

impl<'w, 'a> DraftRuntime<'w, 'a, DistributedDsparkChain<'w, 'a>> {
    /// The returned owner scopes destruction to GPU1. Synchronous runtime methods
    /// select their execution device internally and restore it before returning;
    /// no device scope survives a scheduler yield.
    pub fn with_distributed_requests(devices: [Device<'a>; 2], weights: &'w DsparkWeights<'a>,
        table: &'w NativeRtxTensors<'a>, shards: [&'w VocabularyShard<'a>; 2],
        capacity: u32, requests: u32) -> Result<DeviceOwner<'a, Self>> {
        ensure!((1..=16).contains(&requests), "invalid draft request limit");
        devices[1].own(|| {
            let lib = devices[1].library;
            let lane_count = if requests == 1 { 1 } else { 2 };
            let lane_requests = requests.div_ceil(lane_count as u32);
            let lane_bytes = DistributedDsparkChain::device_bytes(weights, lane_requests, shards[0].tokens().end)?;
            let mut mains = vec![weights.main_context(capacity, DsparkMainContext::device_bytes(lib, capacity)?)?];
            if lane_count > 1 { mains.push(weights.main_context(80, DsparkMainContext::device_bytes(lib, 80)?)?); }
            let chains = (0..lane_count).map(|_| DistributedDsparkChain::new(devices, weights, table,
                shards, lane_requests, lane_bytes)).collect::<Result<Vec<_>>>()?;
            let window = || DsparkWindow::new(lib, requests as usize, capacity,
                DsparkWindow::device_bytes(requests as usize, capacity)?);
            tracing::info!(lanes=lane_count, lane_requests, draft_width=weights.draft_width(), gpu0_bytes_per_lane=lane_bytes[0],
                gpu1_bytes_per_lane=lane_bytes[1], "distributed lane-local dSpark draft workspaces");
            Ok(Self {
                mains, chains, windows: [window()?, window()?, window()?],
                pending_commit_ids: vec![Vec::new(); lane_count], pending_prefix_ids: [None, None],
                requests: Default::default(), pending: vec![None; lane_count],
                request_limit: requests as usize, draft_limit: 5, draft_width: weights.draft_width(),
                confidence_trace: Default::default(), adaptive: None,
                confidence_cutoff: None, reuse_floor: None,
                cost_model: None,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_dspark_cache::WindowChunk;
    fn append<'a, C: DraftChain<'a>>(runtime: &mut DraftRuntime<'_, 'a, C>, lib: &NativeLibrary, cycle: usize) -> Result<u64> {
        let position = if cycle == 0 { 0 } else { 4 + cycle as u64 };
        let tokens = if cycle == 0 { 5 } else { 1 };
        for stage in 0..3 {
            let values: Vec<u8> = (0..80 * 512).flat_map(|i| {
                let v = ((i * 7 + stage * 11 + cycle * 13) % 63) as f32 / 64. - 0.5;
                ((v.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            lib.copy_h2d(runtime.windows[stage].source(), &values)?;
            let chunks: Vec<_> = runtime.requests.values().map(|r| WindowChunk {
                lease: r.leases[stage], position, source_row: r.slot as u32 * tokens, tokens,
            }).collect();
            unsafe { runtime.windows[stage].write(&chunks)?; }
        }
        Ok(position + u64::from(tokens))
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
    fn distributed_draft_runtime_matches_single_and_restores_prefixes() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let names = ["embed.weight".to_string()];
        let embedding = devices[0].own(|| NativeRtxTensors::load(&lib, &catalog, &names,
            NativeRtxTensors::plan(&catalog, &names)?, 16 << 20))?;
        let shards = [devices[0].own(|| VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
            devices[1].own(|| VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?];
        let full = devices[1].own(|| VocabularyHead::load(&lib, &catalog, 2 << 30, 16 << 20))?;
        let exl3_directory = std::env::var_os("DS41RT_EXL3_AOT").map(std::path::PathBuf::from);
        let weights = devices[1].own(|| DsparkWeights::load_serving_with_width(&lib, &catalog,
            80, 16, 32usize << 30, 16 << 20, 5, exl3_directory.as_deref()))?;
        let mut actual = DraftRuntime::with_distributed_requests(devices, &weights, &embedding,
            [&shards[0], &shards[1]], 80, 16)?;
        let mut reference = devices[1].own(|| DraftRuntime::with_requests(&lib, &weights, &embedding, &full, 80, 16))?;
        devices[1].run(|| {
            actual.set_adaptive(true); reference.set_adaptive(true);
            actual.reserve_prefixes(4)?; reference.reserve_prefixes(4)?;
            for id in 1000..1016 { actual.admit(id)?; reference.admit(id)?; }
            Ok(())
        })?;
        for (cycle, count) in [1, 3, 8, 3].into_iter().enumerate() {
            let end = devices[1].run(|| {
                let end = append(actual.get_mut(), &lib, cycle)?;
                ensure!(end == append(reference.get_mut(), &lib, cycle)?, "fixture cache frontier differs");
                Ok(end)
            })?;
            let inputs: [Vec<_>; 2] = std::array::from_fn(|lane| (0..count).map(|row| {
                let id = 1000 + (lane * 8 + row) as u64;
                (id, (42 + lane * 101 + row * 31 + cycle * 17) as u32, end, 12)
            }).collect());
            let mut expected = [None, None];
            let mut observed = [None, None];
            // Admit both lanes, then finish lane 1 while lane 0 is left unpolled.
            for lane in 0..2 {
                observed[lane] = actual.poll_propose(lane, &inputs[lane])?;
                assert_eq!(lib.cuda_get_device()?, 0);
                expected[lane] = devices[1].run(|| reference.poll_propose(lane, &inputs[lane]))?;
            }
            for lane in [1, 0] {
                while observed[lane].is_none() || expected[lane].is_none() {
                    if observed[lane].is_none() { observed[lane] = actual.poll_propose(lane, &inputs[lane])?; }
                    assert_eq!(lib.cuda_get_device()?, 0);
                    if expected[lane].is_none() {
                        expected[lane] = devices[1].run(|| reference.poll_propose(lane, &inputs[lane]))?;
                    }
                    std::thread::yield_now();
                }
                ensure!(observed[lane].as_ref().unwrap().0 == expected[lane].as_ref().unwrap().0, "runtime proposal differs");
                for &(id, _, _, _) in &inputs[lane] {
                    ensure!(actual.confidence_trace(id) == reference.confidence_trace(id), "runtime confidence differs");
                }
                if lane == 1 { assert!(actual.pending[0].is_some()); }
            }
            eprintln!("PASS distributed runtime requests_per_lane={count}: identical proposals/confidence and independent polling");
        }
        actual.queue_prefix(0, 1000, 8)?;
        while !actual.prefix_ready(0, 1000)? { std::thread::yield_now(); }
        let saved = actual.finish_prefix(0, 1000)?;
        actual.release(1000)?;
        actual.admit(2000)?;
        actual.restore_prefix(2000, 8, &saved)?;
        actual.validate_position(2000, 8)?;
        for id in 1001..1016 { actual.release(id)?; }
        actual.release(2000)?;
        assert_eq!(lib.cuda_get_device()?, 0);
        drop(actual); // Retained snapshot storage outlives the producing runtime.
        assert_eq!(lib.cuda_get_device()?, 0);
        drop(saved); // The prefix carries GPU1 ownership even outside its runtime.
        assert_eq!(lib.cuda_get_device()?, 0);
        eprintln!("PASS distributed runtime queued prefix, release/re-admission, restore and device restoration");
        Ok(())
    }
}
