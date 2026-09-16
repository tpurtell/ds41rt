use super::*;
use crate::v41_dspark_cache::WindowChunk;

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
fn distributed_dspark_chain_matches_full_head() -> Result<()> {
    let lib = unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
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
    let width = std::env::var("DS41RT_DSPARK_TEST_WIDTH").unwrap_or_else(|_| "5".into()).parse::<usize>()?;
    ensure!([5, 7].contains(&width), "draft qualification requires width 5 or 7");
    let capacity = crate::v41_experts::dspark::DsparkAttentionWave::projection_capacity_with_width(16, width)?;
    let weights = devices[1].own(|| DsparkWeights::load_with_width(&lib, &catalog, capacity,
        1, 32usize << 30, 16 << 20, width, exl3_directory.as_deref()))?;
    let budgets = devices[1].run(|| DistributedDsparkChain::device_bytes(&weights, 16, 64640))?;
    let mut lanes = [DistributedDsparkChain::new(devices, &weights, &embedding, [&shards[0], &shards[1]], 16, budgets)?,
        DistributedDsparkChain::new(devices, &weights, &embedding, [&shards[0], &shards[1]], 16, budgets)?];
    let mut reference = devices[1].own(|| weights.draft(&embedding, &full, 16, weights.draft_bytes(16)?))?;
    let mut windows = devices[1].own(|| (0..3).map(|_| DsparkWindow::new(&lib, 16, 80,
        DsparkWindow::device_bytes(16, 80)?)).collect::<Result<Vec<_>>>())?;
    let leases = devices[1].run(|| windows.iter_mut().map(|window| (0..16).map(|slot|
        window.begin_request(slot, 1000 + slot as u64)).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>())?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let download = |buffers: [Ds41rtDeviceBuffer; 3]| -> Result<Vec<Vec<u8>>> {
        buffers.into_iter().map(|buffer| {
            let mut bytes = vec![0; buffer.bytes];
            devices[1].run(|| lib.copy_d2h(&mut bytes, buffer))?;
            Ok(bytes)
        }).collect()
    };
    for (cycle, count) in [1, 3, 8, 16, 3].into_iter().enumerate() {
        let position = if cycle == 0 { 0 } else { 4 + cycle as u64 };
        let tokens = if cycle == 0 { 5 } else { 1 };
        for stage in 0..3 {
            let source: Vec<u8> = (0..80 * 512).flat_map(|i| {
                let value = ((i * 7 + stage * 11 + cycle * 13) % 63) as f32 / 64. - 0.5;
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            devices[1].run(|| {
                lib.copy_h2d(windows[stage].source(), &source)?;
                let chunks: Vec<_> = (0..16).map(|slot| WindowChunk {
                    lease: leases[stage][slot], position, source_row: slot as u32 * tokens, tokens,
                }).collect();
                unsafe { windows[stage].write(&chunks) }
            })?;
        }
        let end = position + u64::from(tokens);
        let bindings: Vec<Vec<Vec<_>>> = (0..2).map(|lane| (0..3).map(|stage| (0..count).map(|row| {
            let slot = if lane == 0 { row } else { 15 - row };
            (leases[stage][slot], end)
        }).collect()).collect()).collect();
        let window_refs = [&windows[0], &windows[1], &windows[2]];
        let mut expected = Vec::new();
        for (lane, chain) in lanes.iter_mut().enumerate() {
            let seeds: Vec<_> = (0..count).map(|row| (42 + row * 31 + lane * 101 + cycle * 17) as i32).collect();
            let temperatures: Vec<_> = (0..count).map(|row| if (row + cycle) % 2 == 0 { 0. } else { 0.7 }).collect();
            let mut rngs: Vec<_> = (0..count).map(|row| DsparkRng::new((lane * 100 + cycle * 1000 + row) as u64)).collect();
            let mut reference_rngs: Vec<_> = (0..count).map(|row| DsparkRng::new((lane * 100 + cycle * 1000 + row) as u64)).collect();
            chain.stage_tokens(&seeds)?;
            chain.stage_sampling(&mut rngs.iter_mut().collect::<Vec<_>>(), &temperatures)?;
            devices[1].run(|| {
                reference.set_tokens(&seeds)?;
                reference.prepare_sampling(&mut reference_rngs.iter_mut().collect::<Vec<_>>(), &temperatures)?;
                unsafe { reference.execute(window_refs, [&bindings[lane][0], &bindings[lane][1], &bindings[lane][2]])?; }
                Ok(())
            })?;
            expected.push(download(reference.draft_output()?)?);
        }
        for replay in 0..2 {
            let [first, second] = &mut lanes;
            if replay == 0 { runtime.block_on(async {
                let (a, b) = tokio::join!(unsafe { first.execute(window_refs, [&bindings[0][0], &bindings[0][1], &bindings[0][2]]) },
                    unsafe { second.execute(window_refs, [&bindings[1][0], &bindings[1][1], &bindings[1][2]]) });
                a.and(b).map(|_| ())
            })?; } else {
                unsafe {
                    first.begin_replay(window_refs, [&bindings[0][0], &bindings[0][1], &bindings[0][2]])?;
                    second.begin_replay(window_refs, [&bindings[1][0], &bindings[1][1], &bindings[1][2]])?;
                }
                assert!(first.stage_tokens(&[42]).is_err());
                let mut compact = [None, None];
                while compact[1].is_none() { compact[1] = second.poll_replay()?; std::thread::yield_now(); }
                assert!(first.output().is_err());
                assert!(matches!(first.pending, Some(Phase::Transformer(_))));
                while compact[0].is_none() { compact[0] = first.poll_replay()?; std::thread::yield_now(); }
                for (lane, value) in compact.into_iter().enumerate() {
                    let (tokens, confidence) = value.unwrap();
                    let token_bytes: Vec<_> = tokens.into_iter().flat_map(u32::to_ne_bytes).collect();
                    let confidence_bytes: Vec<_> = confidence.into_iter().flat_map(f32::to_ne_bytes).collect();
                    ensure!(token_bytes == expected[lane][0] && confidence_bytes == expected[lane][2],
                        "compact polled draft output differs");
                }
            }
            for (lane, chain) in lanes.iter().enumerate() {
                let actual = download(chain.output()?)?;
                for part in 0..3 {
                    ensure!(actual[part] == expected[lane][part],
                        "distributed draft differs cycle={cycle} count={count} lane={lane} replay={replay} part={part}");
                    if part > 0 { ensure!(actual[part].chunks_exact(4).all(|v|
                        f32::from_ne_bytes(v.try_into().unwrap()).is_finite()), "nonfinite distributed draft output"); }
                }
            }
        }
        eprintln!("PASS complete distributed draft width={width} requests={count}: exact tokens/logits/confidence, peer embedding, two lanes, changed cache/seed/order, cold/replay");
    }
    use std::{future::Future, task::{Context, Poll, Waker}};
    let bindings: Vec<Vec<_>> = (0..3).map(|stage| (0..3).map(|row| (leases[stage][row], 9)).collect()).collect();
    let refs = [&bindings[0][..], &bindings[1][..], &bindings[2][..]];
    let mut pending = Box::pin(unsafe { lanes[0].execute([&windows[0], &windows[1], &windows[2]], refs) });
    assert!(matches!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
    drop(pending);
    assert!(lanes[0].output().is_err());
    runtime.block_on(unsafe { lanes[0].execute([&windows[0], &windows[1], &windows[2]], refs) })?;
    assert!(lanes[0].output().is_ok());
    devices[1].run(|| {
        for stage in 0..3 { for lease in &leases[stage] { windows[stage].release(*lease)?; } }
        Ok(())
    })?;
    assert_eq!(lib.cuda_get_device()?, 0);
    eprintln!("PASS complete distributed draft cancellation, reuse, cache lease release and device restoration");
    Ok(())
}
