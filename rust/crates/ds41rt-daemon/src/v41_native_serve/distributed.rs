//! Production owners for the 20/20 attention split and bottom-up TP2 experts.
use super::*;
use crate::v41_backbone_cache::CachePlacement;
use crate::v41_backbone_execution::DistributedExecution;
use crate::v41_backbone_shared::tp2::Weights as SharedWeights;
use crate::v41_engram::placement::{PlacedEngram, PlacedEngramWeights};
use crate::v41_experts::{tp2::RankWeights, tp2_ffn, ExpertLayer, ExpertWeights};
use crate::v41_memory::device::Device;
use crate::v41_target_head::distributed_target::DistributedTargetHead;
use crate::v41_target_pass::{DistributedTargetPass, TargetTapWave};
use std::rc::Rc;

pub(super) fn worker(args: crate::cli::NativeServeArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<(), String>>>) -> Result<()> {
    let expert_layers = match args.rtx_expert_layers {
        memory::LocalLayers::Auto => 20,
        memory::LocalLayers::Count(count) => {
            ensure!((20..=40).contains(&count), "dual RTX expert layers must be 20..=40");
            count as usize
        }
    };
    prefill_capacity(args.prefill_batch_tokens)?;
    // AOT kernels keep their full scratch; row storage follows live capacity.
    let capacity = args.prefill_batch_tokens.max(256);
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    lib.cuda_set_device(0)?;
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    for device in devices { device.run(|| lib.cuda_enable_peer(1 - device.id))?; }
    let memory_checkpoint = |stage: &str| -> Result<()> {
        let occupied = devices.map(|d| d.run(|| { let (free, total) = lib.cuda_memory_info()?; Ok(total - free) }))
            .into_iter().collect::<Result<Vec<_>>>()?;
        tracing::info!(stage, occupied_bytes=?occupied, "dual RTX startup memory");
        Ok(())
    };
    memory_checkpoint("CUDA contexts and peer access")?;
    let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &args.snapshot)?;
    let map = CachePlacement::encoder_decoder();
    let started = Instant::now();
    // Sum resident storage plus the largest transient loading excess.
    let compressed = catalog.exl3().is_some();
    let exl3_directory = args.native_lib.parent().context("native library directory missing")?.join("exl3/rtx-tp2");
    if compressed {
        let per_lane = crate::v41_experts::tp2::ExpertWave::exl3_device_bytes(&exl3_directory, capacity)?;
        tracing::info!(per_gpu_per_lane_bytes=per_lane, capacity, "EXL3 TP2 expert workspace plan");
    }
    let rank_budgets = [0usize, 1].map(|rank| -> Result<usize> {
        let budgets = (0..expert_layers).map(|layer| {
            let selection = ExpertLayer::BackboneTp2 { layer, rank };
            if compressed { crate::v41_experts::exl3::Exl3Weights::plan(&catalog, selection) }
            else { ExpertWeights::plan(&lib, &catalog, selection) }
        }).collect::<Result<Vec<_>>>()?;
        let resident = budgets.iter().try_fold(0usize, |sum, b| sum.checked_add(b.resident_bytes)
            .context("encoder resident budget overflow"))?;
        let transient = budgets.iter().map(|b| Ok(b.peak_device_bytes()? - b.resident_bytes))
            .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        resident.checked_add(transient).context("encoder load budget overflow")
    }).into_iter().collect::<Result<Vec<_>>>()?;
    let weights = BackboneLaneWeights::load_distributed(
        &lib,
        &catalog,
        map,
        BackboneLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("attention mHC router weights")?;
    let producers = CacheProducerWeights::load_distributed(
        &lib,
        &catalog,
        map,
        CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("cache producer weights")?;
    let iw = IndexLaneWeights::load_distributed(
        &lib,
        &catalog,
        map,
        IndexLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("index query weights")?;
    let ew = PlacedEngramWeights::load(
        &lib,
        &catalog,
        map,
        PlacedEngramWeights::device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    memory_checkpoint("Engram weights")?;
    let names = ["embed.weight".to_string()];
    let table = devices[0].own(|| {
        NativeRtxTensors::load(
            &lib,
            &catalog,
            &names,
            NativeRtxTensors::plan(&catalog, &names)?,
            16 << 20,
        )
    })?;
    memory_checkpoint("embedding weights")?;
    let vocab = [
        devices[0].own(|| crate::v41_tensors::VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
        devices[1].own(|| crate::v41_tensors::VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?,
    ];
    memory_checkpoint("vocabulary weights")?;
    let hw = devices[1].own(|| {
        TargetHeadWeights::load(
            &lib,
            &catalog,
            TargetHeadWeights::device_bytes(&catalog)?,
            16 << 20,
        )
    })?;
    memory_checkpoint("head normalization weights")?;
    eprintln!("loading bottom {expert_layers} expert layers as TP2");
    let load_rank = |rank: usize| -> Result<_> {
        let weights = if compressed {
            RankWeights::load_exl3(devices[rank], &catalog, expert_layers, rank_budgets[rank], &exl3_directory)?
        } else { RankWeights::load(devices[rank], &catalog, expert_layers, rank_budgets[rank])? };
        Ok(Rc::new(weights))
    };
    let routed = [load_rank(0)?, load_rank(1)?];
    let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
        .map(|r| {
            (0..40)
                .map(|l| {
                    SharedWeights::load(
                        devices[r],
                        &catalog,
                        l,
                        SharedWeights::load_peak_device_bytes(),
                    )
                })
                .collect::<Result<Vec<_>>>()
                .map(Rc::new)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .ok()
        .expect("two ranks");
    memory_checkpoint("weights")?;
    tracing::info!(capacity, backbone_lane_bytes=BackboneLane::placed_workspace_bytes(&lib, capacity)?,
        producer_bytes=?crate::v41_backbone_execution::PlacedProducerWaves::device_bytes(&lib, map, capacity)?,
        "dual RTX per-lane workspace plan");
    let make_pass = |decoder_capacity: u32| -> Result<DistributedTargetPass<'_, '_>> {
        let lanes = [
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes(&lib, capacity)?,
                0,
            )?,
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes(&lib, capacity)?,
                1,
            )?,
        ];
        let indices = [
            Some(IndexLane::new_on_device(
                &iw,
                capacity,
                IndexLane::placed_workspace_bytes(&lib, map, capacity, 0)?
                    .iter()
                    .sum(),
                0,
            )?),
            Some(IndexLane::new_on_device(
                &iw,
                decoder_capacity,
                IndexLane::placed_workspace_bytes(&lib, map, decoder_capacity, 1)?
                    .iter()
                    .sum(),
                1,
            )?),
        ];
        DistributedTargetPass::new(
            map,
            devices[0].own(|| {
                TargetEmbeddingWave::new(
                    &lib,
                    &table,
                    capacity as usize,
                    TargetEmbeddingWave::device_bytes(capacity as usize)?,
                )
            })?,
            lanes,
            indices,
            DistributedExecution::new(
                &producers,
                capacity,
                crate::v41_backbone_execution::PlacedProducerWaves::device_bytes(
                    &lib, map, capacity,
                )?,
            )?,
            PlacedEngram::new(&ew, capacity as usize, PlacedEngram::device_bytes(&lib, map, capacity as usize)?)?,
            DistributedTargetHead::new(devices, &hw, [&vocab[0], &vocab[1]], if args.dspark_draft_limit > 5 { 64 } else { 48 },
                DistributedTargetHead::device_bytes(if args.dspark_draft_limit > 5 { 64 } else { 48 }, 64640)?)?,
            devices[1]
                .own(|| TargetTapWave::new(&lib, decoder_capacity as usize, TargetTapWave::device_bytes(decoder_capacity as usize)?))?,
            Duration::from_secs(120),
        )
    };
    let mut pass = make_pass(capacity).context("constructing first distributed target lane")?;
    if args.dspark && args.dspark_draft_limit > 5 { pass.reserve_sparse_decode_rows(64)?; }
    memory_checkpoint("first target lane")?;
    // Both lanes also produce source 20 from every encoder row on GPU1, so
    // backbone/query and producer buffers retain full prefill capacity there.
    // Only the first pass handles decoder replay and full cached continuation;
    // lane 1's GPU1 index selection and taps serve at most 64 verify rows.
    // TP2 expert workspaces remain full capacity on both GPUs and lanes.
    let mut second = make_pass(80).context("constructing second distributed target lane")?;
    if args.dspark && args.dspark_draft_limit > 5 { second.reserve_sparse_decode_rows(64)?; }
    memory_checkpoint("second target lane")?;
    let make_transport = || {
        let mut transport = devices[1].own(|| NativeTp4Wave::new(&lib,
            V41Tp4Roce::new(args.peers.clone().try_into().map_err(|_| anyhow::anyhow!("four Spark peers required"))?,
                [1, 2, 3, 4], capacity, TcpTransportConfig { timeout: Duration::from_secs(120),
                    max_frame_bytes: 64 << 20 })?, NativeTp4Wave::device_bytes(capacity)?))?;
        transport.install_tp2(tp2_ffn::Wave::new(routed.clone(), shared.clone(), expert_layers, capacity)?)?;
        Ok::<_, anyhow::Error>(transport)
    };
    let mut transport = make_transport()?;
    let mut second_transport = make_transport()?;
    memory_checkpoint("TP2 transports")?;
    let draft_weights = if args.dspark {
        Some(devices[1].own(|| crate::v41_experts::dspark::DsparkWeights::load_serving_with_width(&lib, &catalog,
            capacity, args.concurrency, 32 << 30, 16 << 20, if args.dspark_draft_limit > 5 { 7 } else { 5 },
            Some(&args.native_lib.parent().context("native library directory missing")?.join("exl3/dspark"))))?)
    } else { None };
    memory_checkpoint("draft weights")?;
    let mut draft = draft_weights.as_ref().map(|weights| DraftRuntime::with_distributed_requests(
        devices, weights, &table, [&vocab[0], &vocab[1]], capacity, args.concurrency)).transpose()?;
    if let Some(draft) = &mut draft {
        draft.set_draft_limit(args.dspark_draft_limit)?;
        draft.set_adaptive(args.adaptive_dspark());
        draft.set_confidence_cutoff(args.dspark_confidence_cutoff);
        draft.set_reuse_floor(args.dspark_reuse_floor)?;
        draft.configure_cost_model(&transport)?;
    }
    memory_checkpoint("draft runtime")?;
    // Vision and target snapshot copies use GPU0. These allocations precede KV sizing.
    let mut vision = crate::v41_vision::VisionRuntime::new(&lib, &catalog, 9216,
        crate::v41_vision::VisionRuntime::device_bytes(&catalog, 9216)?)?;
    memory_checkpoint("vision")?;
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else { 2 * args.prefix_cache_entries as usize + 2 };
    let target_prefix_pool = (snapshot_slots > 0).then(|| crate::v41_memory::SnapshotPool::new(
        &lib, crate::v41_backbone_cache::BackbonePrefix::device_bytes(), snapshot_slots)).transpose()?;
    let draft_snapshot_bytes = draft.as_mut().map(|d| d.reserve_prefixes(snapshot_slots)).transpose()?.unwrap_or(0);
    let snapshot_bytes = target_prefix_pool.as_ref().map_or(0, crate::v41_memory::SnapshotPool::device_bytes) + draft_snapshot_bytes;
    memory_checkpoint("snapshot arenas")?;
    let memory = [devices[0].run(|| lib.cuda_memory_info())?, devices[1].run(|| lib.cuda_memory_info())?];
    let pool = memory::distributed::PoolPlan::new(map, args.concurrency as usize,
        args.max_context_tokens as usize, args.prefix_cache_entries as usize, snapshot_bytes,
        args.kv_pool_size, args.memory_reservation, memory)?;
    tracing::info!(source_pages=?pool.pages, global_bytes=pool.global_bytes, cache_bytes=?pool.cache_bytes,
        occupied_before=?pool.occupied_before, reservation_bytes=?pool.reservation_bytes,
        unused_bytes=?pool.unused_bytes, desired_groups=pool.desired_groups, snapshot_slots, snapshot_bytes,
        runtime_headroom_bytes=memory::distributed::RUNTIME_HEADROOM,
        rtx_expert_layers=expert_layers, "dual RTX cache reservation after fixed allocations");
    let token_map = ds41rt_loader::EngramTokenMap::from_file(&args.snapshot.join("tokenizer.json"))?;
    let rows = capacity as usize;
    let pipeline = unsafe { ds41rt_loader::EngramPipeline::new(&catalog, token_map, rows,
        2 * ds41rt_core::ENGRAM_LAYERS.len(), rows * 64 * 1024)? };
    let mut requests = Requests::new_distributed(&lib, pipeline, args.concurrency as usize,
        pool.pages, map, pool.cache_bytes)?;
    if let Some(pool) = target_prefix_pool { requests.install_prefix_pool(pool)?; }
    memory_checkpoint("allocated KV cache")?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    tracing::info!(elapsed_ms=started.elapsed().as_millis(), "dual RTX serving owners ready");
    ready.take().context("startup readiness missing")?.send(Ok(()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut second, &mut requests,
        &mut transport, &mut second_transport, draft.as_mut().map(|d| d.get_mut()), &mut vision)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, DS41RT_DUAL_PEERS, two GPUs and live Sparks"]
    fn distributed_worker_serves_text_requests() -> Result<()> {
        use clap::{Args, FromArgMatches};
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        let snapshot = std::env::var("DS41RT_SNAPSHOT")?;
        let native = std::env::var("DS41RT_NATIVE_LIB")?;
        let peers = std::env::var("DS41RT_DUAL_PEERS")?;
        let batch = std::env::var("DS41RT_WORKER_PREFILL").unwrap_or_else(|_| "80".into());
        let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("fixture"))
            .try_get_matches_from(["fixture", "--snapshot", &snapshot, "--native-lib", &native,
                "--peers", &peers, "--dspark", "--rtx-gpus", "2", "--prefill-batch-tokens", &batch])?;
        let args = crate::cli::NativeServeArgs::from_arg_matches(&matches)?;
        let (send, receive) = mpsc::channel(16);
        let mut outputs = Vec::new();
        let arithmetic = if std::env::var_os("DS41RT_WORKER_LONG").is_some() {
            format!("{}\nWhat is two plus two? Answer briefly.", "This is background context.\n".repeat(600))
        } else { "What is two plus two? Answer briefly.".into() };
        for text in [arithmetic.as_str(), "Write a Python function that adds two numbers.", arithmetic.as_str()] {
            let (events, output) = mpsc::channel(64);
            send.blocking_send(NativeRequest { prompt: format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>"),
                constraint: None, images: Vec::new(), max_tokens: 8, events })?;
            outputs.push(output);
        }
        drop(send);
        let (ready, mut readiness) = oneshot::channel();
        super::super::worker(args, receive, &mut Some(ready))?;
        readiness.try_recv()?.map_err(anyhow::Error::msg)?;
        for (i, mut output) in outputs.into_iter().enumerate() {
            let mut ready = 0;
            let mut finish = 0;
            let mut response = String::new();
            while let Ok(chunk) = output.try_recv() {
                match chunk {
                    Ok(InferenceChunk::Ready { prompt_usage, .. }) => {
                        ready += 1;
                        if i == 2 { ensure!(prompt_usage.prompt_cache_hit_tokens == prompt_usage.prompt_tokens,
                            "repeated worker prompt did not restore the exact prefix"); }
                    },
                    Ok(InferenceChunk::Text { content, .. }) => response.push_str(&content),
                    Ok(InferenceChunk::Finish { .. }) => finish += 1,
                    Err(error) => anyhow::bail!("worker request {i} failed: {error:?}"),
                    _ => (),
                }
            }
            ensure!(ready == 1 && finish == 1 && !response.is_empty(), "worker request {i} did not complete");
            eprintln!("PASS distributed worker request {i}: {response:?}");
        }
        Ok(())
    }
}
