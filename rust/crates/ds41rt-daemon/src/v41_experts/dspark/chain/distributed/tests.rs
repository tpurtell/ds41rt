use super::*;
use crate::v41_dspark_cache::WindowChunk;

#[test]
#[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT and two CUDA GPUs"]
fn distributed_dspark_chain_matches_full_head() -> Result<()> { check_chain(false,true) }
#[test]
#[ignore = "requires native checkpoint, TP2 draft kernels and two CUDA GPUs"]
fn distributed_dspark_tp2_chain_matches_full_experts() -> Result<()> { check_chain(true,true) }
#[test]
#[ignore = "requires native checkpoint, TP2 draft kernels and two CUDA GPUs"]
fn distributed_dspark_tp2_chain_independent_lanes() -> Result<()> { check_chain(true,false) }
fn check_chain(tp2:bool,full_reference:bool) -> Result<()> {
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
    let weights = devices[1].own(|| if tp2 {
        DsparkWeights::load_tp2_with_width(&lib,&catalog,capacity,2,[32usize<<30;2],16<<20,width)
    } else {DsparkWeights::load_with_width(&lib, &catalog, capacity,
        1, 32usize << 30, 16 << 20, width, exl3_directory.as_deref())})?;
    let full_experts=if tp2 && full_reference {Some(devices[1].own(||DsparkWeights::load_with_width(&lib,&catalog,capacity,
        1,32usize<<30,16<<20,width,None))?)} else {None};
    let reference_weights=full_experts.as_deref().unwrap_or(&weights);
    let compare=|actual:&[u8],expected:&[u8],part:usize|->bool {
        if !tp2 || !full_reference || part==0 {return actual==expected;}
        if actual.len()!=expected.len() {return false;}
        let mut squared=0f64;let mut norm=0f64;let mut maximum=(0f32,0f32,0f32,0usize);let mut failed=0;
        for (index,(a,b)) in actual.chunks_exact(4).zip(expected.chunks_exact(4)).enumerate() {
            let a=f32::from_ne_bytes(a.try_into().unwrap());let b=f32::from_ne_bytes(b.try_into().unwrap());
            if !a.is_finite()||!b.is_finite() {return false;}
            let delta=(a-b).abs();squared+=f64::from(delta).powi(2);norm+=f64::from(b).powi(2);
            if delta>maximum.0 {maximum=(delta,a,b,index);}
            // Logits near zero need an absolute bound: BF16 transformer
            // rounding is propagated through the FP32 vocabulary projection.
            let absolute=if part==1 {1e-3} else {1e-4};
            if delta>absolute+1e-3*b.abs() {failed+=1;}
        }
        eprintln!("TP2 draft comparison part={part} values={} relative_rms={} max_delta_actual_expected_index={maximum:?} outside_tolerance={failed}",actual.len()/4,(squared/norm.max(1e-30)).sqrt());
        if part==1 {
            // Draft quality depends on the distribution, not relative error of
            // individual near-zero logits. Require stable sampling probabilities
            // as well as an RMS guard; token IDs are checked exactly separately.
            let mut max_tv=0f64;let mut max_probability_delta=0f64;
            for (a,b) in actual.chunks_exact(129280*4).zip(expected.chunks_exact(129280*4)) {
                let decode=|v:&[u8]|v.chunks_exact(4).map(|x|f64::from(f32::from_ne_bytes(x.try_into().unwrap()))).collect::<Vec<_>>();
                let mut a=decode(a);let mut b=decode(b);
                for values in [&mut a,&mut b] {
                    let maximum=values.iter().copied().fold(f64::NEG_INFINITY,f64::max);
                    let mut sum=0.;for value in values.iter_mut() {*value=(*value-maximum).exp();sum+=*value;}
                    for value in values.iter_mut() {*value/=sum;}
                }
                let mut tv=0.;for (a,b) in a.into_iter().zip(b) {let delta=(a-b).abs();tv+=delta;max_probability_delta=max_probability_delta.max(delta);}
                max_tv=max_tv.max(tv*0.5);
            }
            eprintln!("TP2 draft distribution max_total_variation={max_tv} max_probability_delta={max_probability_delta}");
            max_tv<=1e-3 && max_probability_delta<=1e-3 && (squared/norm.max(1e-30)).sqrt()<1e-3
        } else { failed==0 && (squared/norm.max(1e-30)).sqrt()<1e-4 }
    };
    let budgets = devices[1].run(|| DistributedDsparkChain::device_bytes(&weights, 16, 64640))?;
    let mut lanes = [DistributedDsparkChain::new(devices, &weights, &embedding, [&shards[0], &shards[1]], 16, budgets)?,
        DistributedDsparkChain::new(devices, &weights, &embedding, [&shards[0], &shards[1]], 16, budgets)?];
    let mut reference = devices[1].own(|| reference_weights.draft(&embedding, &full, 16, reference_weights.draft_bytes(16)?))?;
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
                    ensure!(compare(&token_bytes,&expected[lane][0],0) && compare(&confidence_bytes,&expected[lane][2],2),
                        "compact polled draft output differs");
                }
            }
            for (lane, chain) in lanes.iter().enumerate() {
                let actual = download(chain.output()?)?;
                for part in 0..3 {
                    ensure!(compare(&actual[part],&expected[lane][part],part),
                        "distributed draft differs cycle={cycle} count={count} lane={lane} replay={replay} part={part}");
                    if part > 0 { ensure!(actual[part].chunks_exact(4).all(|v|
                        f32::from_ne_bytes(v.try_into().unwrap()).is_finite()), "nonfinite distributed draft output"); }
                }
            }
        }
        eprintln!("PASS complete distributed draft tp2={tp2} width={width} requests={count}: tokens/logits/confidence checked, peer embedding, two lanes, changed cache/seed/order, cold/replay");
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
