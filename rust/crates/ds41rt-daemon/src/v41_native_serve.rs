pub(crate) mod prefill_target;
use prefill_target::PrefillTarget;
use speculative::DraftChain;
pub(crate) mod speculative;
mod scheduler;
mod distributed;
mod scores;
mod constraints;
use scores::TokenScores;
mod prefix;
pub(crate) mod memory;
use crate::v41_backbone_cache::BackboneCache;
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_execution::CacheProducerWeights;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_experts::coordinator::NativeTp4Wave;
use crate::v41_index_lane::IndexLane;
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{TargetHeadWave, TargetHeadWeights};
use crate::v41_target_pass::TargetPass;
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::Context;
use anyhow::{ensure, Result};
use ds41rt_api::native_v41::{InferenceChunk, InferenceFinishReason, NativeRequest, PromptUsage};
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::v41_expert::V41Tp4Roce;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use speculative::DraftRuntime;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub(crate) async fn run(args: crate::cli::NativeServeArgs) -> Result<()> {
    ensure!(args.peers.len() == 4, "four Spark peers required");
    let listen = args.listen.clone();
    let limits = ds41rt_api::native_v41::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    let (send, receive) = mpsc::channel(args.concurrency as usize);
    let (ready, readiness) = oneshot::channel();
    let worker_thread = std::thread::Builder::new()
        .name("v41-target-cuda".into())
        .spawn(move || {
            let mut ready = Some(ready);
            let result = worker(args, receive, &mut ready);
            if let Some(ready) = ready.take() {
                let _ = ready.send(Err(result
                    .as_ref()
                    .err()
                    .map(|e| format!("{e:#}"))
                    .unwrap_or_else(|| "worker stopped during startup".into())));
            }
            if let Err(error) = result {
                tracing::error!(%error,"native target worker stopped");
            }
        })?;
    readiness
        .await
        .context("native target startup stopped")?
        .map_err(anyhow::Error::msg)?;
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!(%listen,"native V4.1 target API ready");
    axum::serve(listener, ds41rt_api::native_v41::router_with_limits(send, limits))
        .with_graceful_shutdown(async {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
        })
        .await?;
    tokio::task::spawn_blocking(move || worker_thread.join())
        .await?
        .map_err(|_| anyhow::anyhow!("native CUDA worker panicked during shutdown"))?;
    Ok(())
}
// Reserve a supported AOT capacity once; live prefill chunks retain the user's
// requested size. All backbone/draft workspaces and transport share this bound.
fn prefill_capacity(batch_tokens: u32) -> Result<u32> {
    anyhow::ensure!(
        (80..=4096).contains(&batch_tokens),
        "prefill batch must be in 80..=4096"
    );
    [80, 256, 1024, 4096]
        .into_iter()
        .find(|&capacity| capacity >= batch_tokens.max(128))
        .context("no prefill capacity covers the requested batch")
}

#[cfg(test)]
mod prefill_capacity_tests {
    use super::prefill_capacity;

    #[test]
    fn intermediate_batches_use_covering_preallocated_capacity() {
        for (batch, expected) in [
            (80, 256),
            (81, 256),
            (256, 256),
            (257, 1024),
            (1024, 1024),
            (1025, 4096),
            (2048, 4096),
            (4096, 4096),
        ] {
            assert_eq!(prefill_capacity(batch).unwrap(), expected);
        }
        for invalid in [0, 79, 4097, u32::MAX] {
            assert!(prefill_capacity(invalid).is_err());
        }
    }
}

fn worker(
    args: crate::cli::NativeServeArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<(), String>>>,
) -> Result<()> {
    if args.rtx_gpus == 2 { return distributed::worker(args, receive, ready); }
    let capacity = prefill_capacity(args.prefill_batch_tokens)?;
    let rows = capacity as usize;
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        &args.snapshot,
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
        "native target backbone/index/embedding weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let embedding =
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?;
    let lane = BackboneLane::new(
        &weights,
        capacity,
        BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let index = IndexLane::new(
        &index_weights,
        capacity,
        IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let execution = BackboneExecution::new(
        &producers,
        capacity,
        BackboneExecution::workspace_bytes(&lib, capacity)?,
    )?;
    let map = ds41rt_loader::EngramTokenMap::from_file(&&args.snapshot.join("tokenizer.json"))?;
    // Both Engram layers retain their staging leases until consumed. Reserve
    // both layers for both active lanes so the second lane can gather early.
    let gather_slots = 2 * ds41rt_core::ENGRAM_LAYERS.len();
    let pipeline =
        unsafe { ds41rt_loader::EngramPipeline::new(&catalog, map, rows, gather_slots, rows * 64 * 1024)? };
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let gates = [
        EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
        EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?,
    ];
    let upload = EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?;
    let vocabulary = VocabularyHead::load(
        &lib,
        &catalog,
        VocabularyHead::plan(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head_weights = TargetHeadWeights::load(
        &lib,
        &catalog,
        TargetHeadWeights::device_bytes(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head = head_weights.wave(&vocabulary, if args.dspark_draft_limit > 5 { 64 } else { 48 }, TargetHeadWave::device_bytes(if args.dspark_draft_limit > 5 { 64 } else { 48 })?)?;
    let mut pass = TargetPass::new(
        embedding,
        lane,
        index,
        execution,
        upload,
        gates,
        head,
        crate::v41_target_pass::TargetTapWave::new(
            &lib,
            rows,
            crate::v41_target_pass::TargetTapWave::device_bytes(rows)?,
        )?,
        Duration::from_secs(120),
    )?;
    let roce = V41Tp4Roce::new(
        args.peers
            .clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("four Spark peers required"))?,
        [1, 2, 3, 4],
        capacity,
        TcpTransportConfig {
            timeout: Duration::from_secs(120),
            max_frame_bytes: 64 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(capacity)?)?;
    let mut prefill_pass = TargetPass::new(
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?,
        BackboneLane::new(&weights, capacity, BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        IndexLane::new(&index_weights, capacity, IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        BackboneExecution::new(&producers, capacity, BackboneExecution::workspace_bytes(&lib, capacity)?)?,
        EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?,
        [EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
         EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?],
        head_weights.wave(&vocabulary, if args.dspark_draft_limit > 5 { 64 } else { 48 }, TargetHeadWave::device_bytes(if args.dspark_draft_limit > 5 { 64 } else { 48 })?)?,
        crate::v41_target_pass::TargetTapWave::new(&lib, rows, crate::v41_target_pass::TargetTapWave::device_bytes(rows)?)?,
        Duration::from_secs(120),
    )?;
    if args.dspark && args.dspark_draft_limit > 5 {
        pass.reserve_sparse_decode_rows(64)?;
        prefill_pass.reserve_sparse_decode_rows(64)?;
    }
    let prefill_roce = V41Tp4Roce::new(args.peers.clone().try_into()
        .map_err(|_| anyhow::anyhow!("four Spark peers required"))?, [1, 2, 3, 4], capacity,
        TcpTransportConfig { timeout: Duration::from_secs(120), max_frame_bytes: 64 * 1024 * 1024 })?;
    let mut prefill_transport = NativeTp4Wave::new(&lib, prefill_roce, NativeTp4Wave::device_bytes(capacity)?)?;
    let draft_weights = if args.dspark {
        Some(crate::v41_experts::dspark::DsparkWeights::load_serving_with_width(
            &lib,
            &catalog,
            capacity,
            args.concurrency,
            32 * 1024 * 1024 * 1024,
            16 * 1024 * 1024,
            if args.dspark_draft_limit > 5 { 7 } else { 5 },
            Some(&args.native_lib.parent().context("native library directory missing")?.join("exl3/dspark")),
        )?)
    } else {
        None
    };
    let mut draft = draft_weights
        .as_ref()
        .map(|weights| DraftRuntime::with_requests(&lib, weights, &table, &vocabulary, capacity, args.concurrency))
        .transpose()?;
    if let Some(draft) = &mut draft {
        draft.set_draft_limit(args.dspark_draft_limit)?;
        draft.set_adaptive(args.adaptive_dspark());
        draft.set_confidence_cutoff(args.dspark_confidence_cutoff);
        draft.set_reuse_floor(args.dspark_reuse_floor)?;
    }
    let mut vision = crate::v41_vision::VisionRuntime::new(&lib, &catalog, 9216,
        crate::v41_vision::VisionRuntime::device_bytes(&catalog, 9216)?)?;
    // Reserve both retention banks plus one in-flight snapshot per lane. These
    // allocations are counted before choosing KV capacity and local expert layers.
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else {
        (args.prefix_cache_entries as usize).checked_mul(2).and_then(|n| n.checked_add(2))
            .context("snapshot slot count overflow")?
    };
    let target_prefix_pool = (snapshot_slots > 0).then(|| crate::v41_memory::SnapshotPool::new(
        &lib, crate::v41_backbone_cache::BackbonePrefix::device_bytes(), snapshot_slots)).transpose()?;
    let draft_snapshot_bytes = draft.as_mut().map(|d| d.reserve_prefixes(snapshot_slots)).transpose()?.unwrap_or(0);
    let snapshot_bytes = target_prefix_pool.as_ref().map_or(0, crate::v41_memory::SnapshotPool::device_bytes) + draft_snapshot_bytes;
    tracing::info!(snapshot_slots, snapshot_bytes, "snapshot arenas reserved before serving");
    // Size after vision, both lanes, transports and optional draft allocations are live.
    let (free, total) = lib.cuda_memory_info()?;
    let pool = memory::PoolPlan::new(args.concurrency as usize, args.max_context_tokens as usize,
        args.prefix_cache_entries as usize, snapshot_bytes, args.kv_pool_size, args.memory_reservation, free, total)?;
    tracing::info!(retained_turn_limit=args.prefix_cache_entries, prompt_snapshot_limit=args.prefix_cache_entries, source_pages=?pool.pages, global_bytes=pool.global_bytes,
        cache_bytes=pool.cache_bytes, device_occupied_bytes=pool.occupied_before,
        reservation_bytes=pool.reservation_bytes, runtime_headroom_bytes=memory::RUNTIME_HEADROOM,
        "native KV pool reservation");
    let mut requests = Requests::new(&lib, pipeline, args.concurrency as usize, pool.pages, pool.cache_bytes)?;
    if let Some(pool) = target_prefix_pool { requests.install_prefix_pool(pool)?; }
    if args.rtx_expert_layers != memory::LocalLayers::Count(0) {
        use crate::v41_experts::{ExpertLayer, ExpertWeights, local::LocalExpertWave};
        let local_started = Instant::now();
        use crate::v41_experts::exl3::Exl3Weights;
        let exl3_directory = args.native_lib.parent().context("native library directory missing")?.join("exl3/rtx-tp1");
        let compressed = catalog.exl3().is_some();
        let per_lane = if compressed { LocalExpertWave::exl3_device_bytes(&exl3_directory, capacity)? }
            else { LocalExpertWave::device_bytes(&lib, capacity)? };
        let budgets = (0..40).map(|layer| {
            let selection = ExpertLayer::BackboneFull { layer };
            if compressed { Exl3Weights::plan(&catalog, selection) }
            else { ExpertWeights::plan(&lib, &catalog, selection) }
        }).collect::<Result<Vec<_>>>()?;
        let (free, total) = lib.cuda_memory_info()?;
        let plan = memory::LocalLayerPlan::new(args.rtx_expert_layers, &budgets,
            per_lane.checked_mul(2).context("local lane budget overflow")?, free, total, pool.reservation_bytes)?;
        tracing::info!(layers=plan.layers, resident_bytes=plan.resident_bytes,
            workspace_bytes=plan.workspace_bytes, peak_bytes=plan.peak_bytes,
            "bottom-up RTX expert placement");
        if plan.layers > 0 && compressed {
            let mut loaded = Vec::with_capacity(plan.layers);
            for layer in 0..plan.layers {
                loaded.push(Exl3Weights::load(&lib, &catalog, ExpertLayer::BackboneFull { layer },
                    budgets[layer].peak_device_bytes()?)?);
            }
            let weights = std::rc::Rc::new(loaded);
            transport.install_local(unsafe { LocalExpertWave::new_exl3(&lib, weights.clone(),
                &exl3_directory, capacity, per_lane)? })?;
            prefill_transport.install_local(unsafe { LocalExpertWave::new_exl3(&lib, weights,
                &exl3_directory, capacity, per_lane)? })?;
        } else if plan.layers > 0 {
            let mut loaded = Vec::with_capacity(plan.layers);
            for layer in 0..plan.layers {
                loaded.push(ExpertWeights::load(&lib, &catalog, ExpertLayer::BackboneFull { layer },
                    budgets[layer].peak_device_bytes()?)?);
            }
            let weights = std::rc::Rc::new(loaded);
            transport.install_local(LocalExpertWave::new(&lib, weights.clone(), capacity, per_lane)?)?;
            prefill_transport.install_local(LocalExpertWave::new(&lib, weights, capacity, per_lane)?)?;
        }
        tracing::info!(layers=plan.layers, elapsed_ms=local_started.elapsed().as_millis(),
            "local RTX experts ready");
    }
    if let Some(draft) = &mut draft { draft.configure_cost_model(&transport)?; }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    ready
        .take()
        .context("startup readiness missing")?
        .send(Ok(()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut prefill_pass,
        &mut requests, &mut transport, &mut prefill_transport, draft.as_mut(), &mut vision)
}

fn prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary,
    runtime: &tokio::runtime::Runtime,
    pass: &mut P,
    other: &mut P,
    requests: &mut Requests<'a>,
    transport: &mut P::Transport,
    other_transport: &mut P::Transport,
    lease: crate::v41_backbone_cache::CacheLease,
    tokens: &[u32],
    chunk_rows: usize,
    job: &NativeRequest,
    draft: Option<&mut DraftRuntime<'_, 'a, C>>,
) -> Result<TokenScores> {
    use crate::v41_backbone_cache::{CacheStage, CacheWork};
    let end = tokens.len() as u64;
    let cached = requests.cache().committed_end(lease)? as usize;
    let stage = requests.cache().stage(lease)?;
    let replay = stage == CacheStage::EncoderReplay;
    if cached > 0 && stage == CacheStage::Full {
        return prefill_continuation(
            lib,
            runtime,
            pass,
            requests,
            transport,
            lease,
            &tokens[cached..],
            chunk_rows,
            job,
            draft,
        );
    }
    let mut suffix = pass.new_suffix(lib, end)?;
    if replay {
        let start = requests.cache().history_end(lease)? as usize;
        ensure!(
            cached - start <= 128,
            "encoder prefix replay exceeds one window"
        );
        for chunk in tokens[start..cached].chunks(chunk_rows) {
            ensure!(!job.events.is_closed(), "client disconnected");
            let mut batch = requests.prepare(&[RequestTokens {
                lease,
                tokens: chunk,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let result = (|| -> Result<()> {
                runtime.block_on(unsafe {
                    pass.encoder_part(requests, &mut batch, transport, &mut suffix)
                })?;
                ensure!(!job.events.is_closed(), "client disconnected");
                runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, None, chunk.len() as u32))
            })();
            if result.is_err() {
                pass.discard(&mut batch)?;
            }
            result?;
        }
    } else if stage == CacheStage::Full {
        requests.begin_encoder(lease, end)?;
    }
    let mut chunks = tokens[cached..].chunks(chunk_rows);
    // Keep the ordinary path for short prompts; pair full chunks first otherwise.
    if chunks.len() == 1 {
        let chunk = chunks.next().expect("one chunk");
        ensure!(!job.events.is_closed(), "client disconnected");
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let started = Instant::now();
        let result = (|| -> Result<()> {
            runtime.block_on(unsafe { pass.encoder_part(requests, &mut batch, transport, &mut suffix) })?;
            ensure!(!job.events.is_closed(), "client disconnected");
            runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, None, chunk.len() as u32))
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        result?;
        tracing::debug!(target: "ds41rt::timing", rows=chunk.len(), total_us=started.elapsed().as_micros() as u64, "target encoder step");
    }
    if chunks.len() != 0 {
        let chunks: Vec<_> = chunks.collect();
        let started = Instant::now();
        runtime.block_on(unsafe { pass.encoder_stream(other, requests, lease, &chunks,
            [transport, other_transport], &mut suffix, &|| !job.events.is_closed()) })?;
        tracing::debug!(target: "ds41rt::timing", rows=tokens.len(),
            total_us=started.elapsed().as_micros() as u64, "target encoder stream");
    }
    ensure!(!job.events.is_closed(), "client disconnected");
    let start = requests.begin_decoder_replay(lease)?;
    let rows = (end - start) as u32;
    let mut batch = requests.prepare_replay(&[CacheWork { lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
    let started = Instant::now();
    let result = (|| -> Result<TokenScores> {
        let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch,
            transport, &[rows as usize - 1], Some(&suffix)) })?;
        let scores = TokenScores::new(bytes)?;
        ensure!(!job.events.is_closed(), "client disconnected");
        runtime.block_on(pass.commit_prefill(requests, &mut batch, draft, rows))?;
        Ok(scores)
    })();
    if result.is_err() { pass.discard(&mut batch)?; }
    tracing::debug!(target: "ds41rt::timing", rows, total_us=started.elapsed().as_micros() as u64, "target decoder replay");
    result
}

fn prefill_continuation<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    pass: &mut P, requests: &mut Requests<'a>, transport: &mut P::Transport,
    lease: crate::v41_backbone_cache::CacheLease, tokens: &[u32], chunk_rows: usize,
    job: &NativeRequest, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<TokenScores> {
    ensure!(!tokens.is_empty(), "prefix continuation has no uncached rows");
    let mut anchor = None;
    for chunk in tokens.chunks(chunk_rows) {
        ensure!(!job.events.is_closed(), "client disconnected");
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let result = (|| -> Result<TokenScores> {
            let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch, transport,
                &[chunk.len() - 1], None) })?;
            let scores = TokenScores::new(bytes)?;
            ensure!(!job.events.is_closed(), "client disconnected");
            runtime.block_on(pass.commit_prefill(requests, &mut batch, draft.as_deref_mut(), chunk.len() as u32))?;
            Ok(scores)
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        anchor = Some(result?);
    }
    anchor.context("prefix continuation produced no logits")
}
