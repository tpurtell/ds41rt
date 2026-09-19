use super::*;
use ds41rt_ffi::Ds41rtDeviceBuffer;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use ds41rt_transport::v41_expert::V41Tp4Roce;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

fn encoder_bytes(lib: &NativeLibrary, lane: &BackboneLane<'_, '_>) -> Result<Vec<u8>> {
    let out = lane.output()?;
    let mut bytes = Vec::new();
    for buffer in [out.residual, out.pre] {
        let start = bytes.len(); bytes.resize(start + buffer.bytes, 0);
        lib.copy_d2h(&mut bytes[start..], buffer)?;
    }
    Ok(bytes)
}

fn production_bytes(lib: &NativeLibrary, bank: &BackboneCache<'_>, batch: &CacheBatch,
    execution: &BackboneExecution<'_, '_>, layer: usize) -> Result<Vec<Vec<u8>>> {
    let w = execution.windows[layer].output(bank.window(batch, layer)?)?;
    let mut buffers = vec![w.values, w.scales, w.projected, w.normalized, w.frequencies];
    if let Some(i) = SOURCES.iter().position(|&l| l == layer) {
        let s = execution.sources[i].output(bank.source(batch, layer)?)?;
        buffers.extend([s.buffer, s.frequencies, s.index_key, s.index_packed,
            s.index_scales, s.kv_values, s.kv_scales]);
    }
    buffers.into_iter().map(|buffer| {
        let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
    }).collect()
}
fn check_queued_production<'w, 'a>(lib: &NativeLibrary, runtime: &tokio::runtime::Runtime,
    bank: &BackboneCache<'a>, batch: &CacheBatch, lane: &BackboneLane<'w, 'a>,
    execution: &mut BackboneExecution<'w, 'a>, index: &mut IndexLane<'w, 'a>,
    reference_index: &mut IndexLane<'w, 'a>) -> Result<()> {
    let query = lane.query_output()?;
    let layer = query.layer;
    unsafe { bank.produce_window(batch, &query, &mut execution.windows[layer])?; }
    if let Some(i) = SOURCES.iter().position(|&l| l == layer) {
        unsafe { bank.produce_source(batch, &query, &mut execution.sources[i])?; }
    }
    let reference = production_bytes(lib, bank, batch, execution, layer)?;
    execution.windows[layer].clear_graph()?;
    if let Some(i) = SOURCES.iter().position(|&l| l == layer) { execution.sources[i].clear_graph()?; }
    // Unpolled cancellation must drain before staging and scratch are reused.
    drop(unsafe { execution.enqueue_production(bank, batch, lane)? });
    if layer == 2 {
        drop(unsafe { execution.enqueue_production_and_index(bank, batch, lane, index)? });
        index.restart()?;
    }
    for warm in [false, true] {
        runtime.block_on(async {
            let mut pending = unsafe {
                if warm { execution.enqueue_production_and_index(bank, batch, lane, index)? }
                else { execution.enqueue_production(bank, batch, lane)? }
            };
            while !unsafe { pending.poll(bank, batch)? } { tokio::task::yield_now().await; }
            Ok::<_, anyhow::Error>(())
        })?;
        assert_eq!(production_bytes(lib, bank, batch, execution, layer)?, reference,
            "queued producer differs at layer {layer}, warm={warm}");
        if !warm { execution.queued_production = None; }
    }
    if INDEX.contains(&layer) {
        let source = SOURCES.iter().rposition(|&l| l <= layer).map(|i| &execution.sources[i]);
        let cache = bank.attention(batch, layer, &execution.windows[layer], source)?;
        unsafe { reference_index.select(&query, &cache)?; }
        let expected = reference_index.output(layer, &cache)?;
        let actual = index.output(layer, &cache)?;
        let read = |buffer: Ds41rtDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        assert_eq!(read(expected.selected)?, read(actual.selected)?, "queued selections differ at layer {layer}");
        assert_eq!(expected.candidate_blocks().map(read).transpose()?, actual.candidate_blocks().map(read).transpose()?,
            "queued retained candidates differ at layer {layer}");
    }
    Ok(())
}

fn queued_engram_fixture(runtime: &tokio::runtime::Runtime, requests: &Requests<'_>,
    batch: &mut crate::v41_requests::RequestBatch, upload: &mut EngramDeviceRows<'_>,
    gate: &mut EngramGate<'_, '_>, lane: &mut BackboneLane<'_, '_>) -> Result<()> {
    runtime.block_on(async {
        let start = Instant::now();
        loop {
            match requests.poll_engram_gather(batch, lane)? {
                ds41rt_loader::EngramGatherPoll::Ready(lease) => {
                    upload.check_cooperative(&lease.view()?).await?;
                    let rows = upload.view()?;
                    unsafe { lane.check_queued_engram(gate, &rows).await?;
                        lane.apply_engram_cooperative(gate, &rows).await?; }
                    return Ok(());
                }
                ds41rt_loader::EngramGatherPoll::Cancelled => anyhow::bail!("fixture gather cancelled"),
                ds41rt_loader::EngramGatherPoll::Pending => {}
            }
            ensure!(start.elapsed() < Duration::from_secs(120), "fixture gather timeout");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
}

#[test]
fn real_layer_zero_executes_embedding_attention_tp4_and_mhc() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_LAYER0_LIBRARY") else {
        eprintln!("skip distributed layer test: DS41RT_LAYER0_LIBRARY unset");
        return Ok(());
    };
    let model = std::env::var_os("DS41RT_LAYER0_MODEL").context("DS41RT_LAYER0_MODEL required")?;
    let peers: [SocketAddr; 4] = std::env::var("DS41RT_LAYER0_PEERS")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("four test peers required"))?;
    let lib = unsafe { NativeLibrary::load(path)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&model),
    )?;
    let start = Instant::now();
    let weights = BackboneLaneWeights::load(
        &lib,
        &catalog,
        BackboneLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let producers = CacheProducerWeights::load(
        &lib,
        &catalog,
        CacheProducerWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let index_weights = IndexLaneWeights::load(
        &lib,
        &catalog,
        IndexLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let names = ["embed.weight".to_string()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 * 1024 * 1024,
    )?;
    eprintln!(
        "layer fixture official coordinator weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let mut embedding =
        TargetEmbeddingWave::new(&lib, &table, 80, TargetEmbeddingWave::device_bytes(80)?)?;
    let mut lane = BackboneLane::new(
        &weights,
        80,
        BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let mut index = IndexLane::new(
        &index_weights,
        80,
        IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let mut reference_index = IndexLane::new(&index_weights, 80,
        IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum())?;
    let mut execution = BackboneExecution::new(
        &producers,
        80,
        BackboneExecution::workspace_bytes(&lib, 80)?,
    )?;
    let map = ds41rt_loader::EngramTokenMap::from_file(
        &std::path::Path::new(&model).join("tokenizer.json"),
    )?;
    let pipeline =
        unsafe { ds41rt_loader::EngramPipeline::new(&catalog, map, 80, 2, 8 * 1024 * 1024)? };
    let mut requests = Requests::new(
        &lib,
        pipeline,
        16,
        [16; 4],
        BackboneCache::device_bytes(16, [16; 4])?,
    )?;
    let engram_weights =
        EngramLayerWeights::load(&lib, &catalog, 0, 256 * 1024 * 1024, 16 * 1024 * 1024)?;
    let engram14_weights = EngramLayerWeights::load(&lib, &catalog, 1, 256 * 1024 * 1024, 16 * 1024 * 1024)?;
    let mut gate = EngramGate::new(&engram_weights, 80, 128 * 1024 * 1024)?;
    let mut gate14 = EngramGate::new(&engram14_weights, 80, 128 * 1024 * 1024)?;
    let mut upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
    let roce = V41Tp4Roce::new(
        peers,
        [1, 2, 3, 4],
        80,
        TcpTransportConfig { timing: false,
            timeout: Duration::from_secs(120),
            max_frame_bytes: 2 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(80)?)?;
    use crate::v41_experts::{ExpertLayer, ExpertWeights, local::LocalExpertWave};
    let local_budget = ExpertWeights::plan(&lib, &catalog, ExpertLayer::BackboneFull { layer: 0 })?;
    let local_weights = std::rc::Rc::new(vec![ExpertWeights::load(&lib, &catalog,
        ExpertLayer::BackboneFull { layer: 0 }, local_budget.peak_device_bytes()?)?]);
    let mut local_fixture = LocalExpertWave::new(&lib, local_weights, 80, LocalExpertWave::device_bytes(&lib, 80)?)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut previous = None;
    let mut reference_encoder = Vec::new();
    for cycle in 0..2 {
        execution.restart();
        lane.restart()?;
        index.restart()?;
        let leases = (0..16)
            .map(|slot| requests.admit(slot, 1000 + slot as u64))
            .collect::<Result<Vec<_>>>()?;
        let token_count = if cycle == 0 { std::env::var("DS41RT_LAYER0_TOKENS_PER_REQUEST")
            .ok().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(5) } else { 5 };
        ensure!((1..=5).contains(&token_count), "fixture tokens per request exceed capacity");
        let tokens = (0..(16 * token_count) as u32)
            .map(|i| (i * 7919 + cycle * 113 + 17) % 129280)
            .collect::<Vec<_>>();
        let work = leases
            .iter()
            .zip(tokens.chunks_exact(token_count))
            .map(|(&lease, tokens)| RequestTokens {
                lease,
                tokens,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        if cycle == 1 {
            for &lease in &leases { requests.begin_encoder(lease, 10)?; }
            execution.restart_for(CacheStage::Encoder);
        }
        let mut batch = if cycle == 1 { requests.reserve_encoder(&work)? } else { requests.prepare(&work)? };
        let positions = batch.cache()?.positions();
        unsafe {
            requests.begin_input(&batch, &mut embedding, &mut lane)?;
        }
        runtime.block_on(unsafe { lane.check_queued_query() })?;
        // Query parity creates fresh bindings; restore the block's matching one.
        lane.restart()?;
        unsafe { requests.begin_input(&batch, &mut embedding, &mut lane)?; }
        let start = Instant::now();
        let mut successor = None;
        if cycle == 1 {
            let prepared = unsafe { requests.prepare_encoder_layer(&batch, &mut execution, &mut lane, &mut index)? };
            // The prepared FFN borrows the lane, but no request/cache owner.
            successor = Some(requests.reserve_encoder(&work)?);
            for &lease in &leases { assert_eq!(requests.cache().committed_end(lease)?, 0); }
            let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, batch.image_mask()) })?;
            unsafe { execution.complete_layer(batch.cache()?, &mut lane, completed)?; }
        } else {
            // Dropping queued attention must drain before the same producer,
            // cache batch and lane are reused, even without polling completion.
            let pending = unsafe { execution.prepare_layer_cooperative(requests.cache(),
                batch.cache()?, &mut lane, &mut index)? };
            drop(pending);
            execution.restart();
            lane.restart()?;
            index.restart()?;
            unsafe { requests.begin_input(&batch, &mut embedding, &mut lane)?; }
            check_queued_production(&lib, &runtime, requests.cache(), batch.cache()?, &lane, &mut execution, &mut index, &mut reference_index)?;
            let prepared = unsafe { execution.prepare_layer_cooperative(requests.cache(),
                batch.cache()?, &mut lane, &mut index)? };
            let prepared = runtime.block_on(unsafe { prepared.check_queued_ffn(batch.image_mask(), Some(&mut local_fixture)) })?;
            let completed = runtime.block_on(unsafe {
                prepared.execute(&mut transport, 0, batch.image_mask())
            })?;
            runtime.block_on(unsafe { lane.check_queued_finish(completed.result.binding(), completed.result.values) })?;
            runtime.block_on(unsafe { execution.complete_layer_cooperative(batch.cache()?, &mut lane, completed) })?;
        }
        let output = lane.output()?;
        assert_eq!(output.layer, 0);
        assert_eq!(output.tokens, positions);
        let mut residual = vec![0; output.residual.bytes];
        lib.copy_d2h(&mut residual, output.residual)?;
        let mut pre = vec![0; output.pre.bytes];
        lib.copy_d2h(&mut pre, output.pre)?;
        assert!(residual
            .chunks_exact(2)
            .all(|b| u16::from_ne_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80));
        assert!(pre
            .chunks_exact(4)
            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
        if let Some(old) = &previous {
            assert_ne!(
                old, &residual,
                "changed token input reused stale layer output"
            );
        }
        if let Some(dir) = std::env::var_os("DS41RT_LAYER0_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("layer0-c{cycle}-residual.bin")), &residual)?;
            std::fs::write(dir.join(format!("layer0-c{cycle}-pre.bin")), &pre)?;
        }
        previous = Some(residual);
        lane.advance()?;
        if cycle == 0 {
            queued_engram_fixture(&runtime, &requests, &mut batch, &mut upload, &mut gate, &mut lane)?;
        } else {
        let deadline = Instant::now() + Duration::from_secs(120);
        while !unsafe { requests.poll_engram(&mut batch, &mut upload, &mut gate, &mut lane)? } {
            ensure!(Instant::now() < deadline, "layer-1 engram timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
        }
        let prepared = lane.prepared_input()?;
        assert_eq!(prepared.layer, 1);
        assert_eq!(prepared.tokens, positions);
        let mut gated = vec![0; prepared.residual.bytes];
        lib.copy_d2h(&mut gated, prepared.residual)?;
        assert_ne!(
            gated,
            *previous.as_ref().unwrap(),
            "engram did not update real residual"
        );
        let query = unsafe { lane.begin_prepared()? };
        assert_eq!(query.layer, 1);
        assert_eq!(query.tokens()?, positions);
        let mut rotated = vec![0; query.rotated.bytes];
        lib.copy_d2h(&mut rotated, query.rotated)?;
        for bytes in [&gated, &rotated] {
            assert!(bytes
                .chunks_exact(2)
                .all(|b| u16::from_ne_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80));
        }
        if let Some(dir) = std::env::var_os("DS41RT_LAYER0_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::write(dir.join(format!("layer1-c{cycle}-gated.bin")), &gated)?;
            std::fs::write(dir.join(format!("layer1-c{cycle}-query.bin")), &rotated)?;
        }
        if cycle == 0 {
            for layer in 1..37 {
                if layer != 1 {
                    lane.advance()?;
                    if layer == 14 {
                        queued_engram_fixture(&runtime, &requests, &mut batch, &mut upload, &mut gate14, &mut lane)?;
                    }
                    unsafe { lane.begin_prepared()?; }
                }
                check_queued_production(&lib, &runtime, requests.cache(), batch.cache()?, &lane, &mut execution, &mut index, &mut reference_index)?;
                let prepared = unsafe { execution.prepare_layer_cooperative(requests.cache(),
                    batch.cache()?, &mut lane, &mut index)? };
                let prepared = runtime.block_on(unsafe { prepared.check_queued_ffn(batch.image_mask(), None) })?;
                let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, batch.image_mask()) })?;
                runtime.block_on(unsafe { lane.check_queued_finish(completed.result.binding(), completed.result.values) })?;
                runtime.block_on(unsafe { execution.complete_layer_cooperative(batch.cache()?, &mut lane, completed) })?;
            }
            eprintln!("PASS queued production/index: 37 windows, four compressed sources, eight index layers and retained source-20 candidates; cold/warm graphs, cancellation/reuse, exact direct-output parity");
        }
        if cycle == 1 {
            for layer in 1..20 {
                if layer != 1 {
                    lane.advance()?;
                    if layer == 14 {
                        let deadline = Instant::now() + Duration::from_secs(120);
                        while !unsafe { requests.poll_engram(&mut batch, &mut upload, &mut gate14, &mut lane)? } {
                            ensure!(Instant::now() < deadline, "encoder layer-14 engram timed out");
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    unsafe { lane.begin_prepared()?; }
                }
                let prepared = unsafe { requests.prepare_encoder_layer(&batch, &mut execution, &mut lane, &mut index)? };
                let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, batch.image_mask()) })?;
                unsafe { execution.complete_layer(batch.cache()?, &mut lane, completed)?; }
                assert_eq!(lane.output()?.layer, layer);
            }
            reference_encoder.push(encoder_bytes(&lib, &lane)?);
            lane.advance()?;
            unsafe {
                lane.begin_prepared()?;
                requests.publish_encoder_boundary(&batch, &mut execution, &lane)?;
            }
            requests.commit(&mut batch, &mut execution, &[token_count as u32; 16])?;
            let mut successor = successor.take().context("queued encoder successor missing")?;
            requests.validate(&successor)?;
            for &lease in &leases { assert_eq!(requests.cache().committed_end(lease)?, 5); }
            execution.restart_for(CacheStage::Encoder);
            lane.restart()?; index.restart()?;
            unsafe { requests.begin_input(&successor, &mut embedding, &mut lane)?; }
            for layer in 0..20 {
                if layer != 0 {
                    lane.advance()?;
                    if layer == 1 || layer == 14 {
                        let current_gate = if layer == 1 { &mut gate } else { &mut gate14 };
                        let deadline = Instant::now() + Duration::from_secs(120);
                        while !unsafe { requests.poll_engram(&mut successor, &mut upload, current_gate, &mut lane)? } {
                            ensure!(Instant::now() < deadline, "successor engram timed out");
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    unsafe { lane.begin_prepared()?; }
                }
                let prepared = unsafe { requests.prepare_encoder_layer(&successor, &mut execution, &mut lane, &mut index)? };
                let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, successor.image_mask()) })?;
                unsafe { execution.complete_layer(successor.cache()?, &mut lane, completed)?; }
            }
            reference_encoder.push(encoder_bytes(&lib, &lane)?);
            lane.advance()?;
            unsafe {
                lane.begin_prepared()?;
                requests.publish_encoder_boundary(&successor, &mut execution, &lane)?;
            }
            requests.commit(&mut successor, &mut execution, &[token_count as u32; 16])?;
            for &lease in &leases {
                assert_eq!(requests.cache().committed_end(lease)?, 10);
                assert_eq!(requests.begin_decoder_replay(lease)?, 0);
                requests.release(lease)?;
            }
            eprintln!("PASS full reserved encoder: 16 requests, two queued chunks, all 20 layers, both Engrams, odd compressor carry, source-20 boundary and decoder readiness");
            continue;

        }
        assert!(
            requests
                .commit(&mut batch, &mut execution, &[token_count as u32; 16])
                .is_err(),
            "partial model pass committed"
        );
        for &lease in &leases {
            assert!(requests.cache().committed_end(lease).is_err());
        }
        if let Some(successor) = successor { assert!(requests.validate(&successor).is_err()); }

        eprintln!("PASS cycle={cycle} rows={} requests=16 actual distributed layer0 -> mapped layer1 engram -> prepared query; incomplete commit revoked request histories; elapsed={:.3}s", 16 * token_count, start.elapsed().as_secs_f64());
    }
    assert_eq!(reference_encoder.len(), 2);
    let mut lane1 = BackboneLane::new(&weights, 80, BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum())?;
    let mut index1 = IndexLane::new(&index_weights, 80, IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum())?;
    let mut execution1 = BackboneExecution::new(&producers, 80, BackboneExecution::workspace_bytes(&lib, 80)?)?;
    let roce1 = V41Tp4Roce::new(peers, [1, 2, 3, 4], 80, TcpTransportConfig { timing: false,
        timeout: Duration::from_secs(120), max_frame_bytes: 2 * 1024 * 1024,
    })?;
    let mut transport1 = NativeTp4Wave::new(&lib, roce1, NativeTp4Wave::device_bytes(80)?)?;
    let leases = (0..16).map(|slot| requests.admit(slot, 3000 + slot as u64)).collect::<Result<Vec<_>>>()?;
    for &lease in &leases { requests.begin_encoder(lease, 10)?; }
    let tokens = (0..80u32).map(|i| (i * 7919 + 113 + 17) % 129280).collect::<Vec<_>>();
    let work = leases.iter().zip(tokens.chunks_exact(5)).map(|(&lease, tokens)| RequestTokens {
        lease, tokens, image_mask: None, kind: ExpertV2SourceKind::Prefill,
    }).collect::<Vec<_>>();
    let mut first = requests.reserve_encoder(&work)?;
    let mut second = requests.reserve_encoder(&work)?;
    execution.restart_for(CacheStage::Encoder); execution1.restart_for(CacheStage::Encoder);
    lane.restart()?; lane1.restart()?; index.restart()?; index1.restart()?;
    unsafe {
        requests.begin_input(&first, &mut embedding, &mut lane)?;
        requests.begin_input(&second, &mut embedding, &mut lane1)?;
    }
    for layer in 0..20 {
        if layer != 0 {
            for (batch, current_lane) in [(&mut first, &mut lane), (&mut second, &mut lane1)] {
                current_lane.advance()?;
                if layer == 1 || layer == 14 {
                    let current_gate = if layer == 1 { &mut gate } else { &mut gate14 };
                    let deadline = Instant::now() + Duration::from_secs(120);
                    while !unsafe { requests.poll_engram(batch, &mut upload, current_gate, current_lane)? } {
                        ensure!(Instant::now() < deadline, "paired encoder engram timed out");
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                unsafe { current_lane.begin_prepared()?; }
            }
        }
        runtime.block_on(unsafe { requests.execute_encoder_pair_layer(
            [&mut first, &mut second], [&mut execution, &mut execution1],
            [&mut lane, &mut lane1], [&mut index, &mut index1], [&mut transport, &mut transport1]) })?;
    }
    assert_eq!(encoder_bytes(&lib, &lane)?, reference_encoder[0], "first paired encoder differs from serial");
    assert_eq!(encoder_bytes(&lib, &lane1)?, reference_encoder[1], "second paired encoder differs from serial");
    for (batch, current_lane, current_execution) in [(&mut first, &mut lane, &mut execution),
        (&mut second, &mut lane1, &mut execution1)] {
        current_lane.advance()?;
        unsafe {
            current_lane.begin_prepared()?;
            requests.publish_encoder_boundary(batch, current_execution, current_lane)?;
        }
        requests.commit(batch, current_execution, &[5; 16])?;
    }
    for &lease in &leases {
        assert_eq!(requests.cache().committed_end(lease)?, 10);
        assert_eq!(requests.begin_decoder_replay(lease)?, 0);
        requests.release(lease)?;
    }
    eprintln!("PASS paired encoder: two independent lanes/QP sets, all 20 layers, 16 requests, both final residual/pre outputs byte-identical to sequential");
    Ok(())
}
