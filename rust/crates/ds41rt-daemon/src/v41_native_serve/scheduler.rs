use super::speculative::DraftChain;
use super::*;
use crate::v41_target_pass::VerificationTarget;
use crate::v41_target_head::TargetSamplingRowRequest;
mod independent;
mod admission;
mod layout;
use layout::ServingTarget;
use super::scores::{BatchScores, VOCAB};
use crate::v41_backbone_cache::CacheLease;
use crate::v41_requests::RequestBatch;
use super::prefix::{ImageKeys, PrefixCache, SnapshotKind};

#[cfg(test)]
pub(crate) fn exercise_distributed_decode<'t, 'd, 'a: 'd>(lib: &'a NativeLibrary,
    runtime: &tokio::runtime::Runtime, snapshot: &std::path::Path,
    first: &mut crate::v41_target_pass::DistributedTargetPass<'t, 'a>,
    second: &mut crate::v41_target_pass::DistributedTargetPass<'t, 'a>,
    requests: &mut Requests<'a>,
    transports: [&mut crate::v41_memory::device::DeviceOwner<'a, NativeTp4Wave<'a>>; 2],
    lease: CacheLease, id: u64, tokens: &[u32], anchor: u32,
    draft: &mut DraftRuntime<'d, 'a, crate::v41_experts::dspark::DistributedDsparkChain<'d, 'a>>,
) -> Result<()> {
    let (events, mut output) = mpsc::channel(64);
    let (_submit, receive) = mpsc::channel(1);
    let mut prefixes = PrefixCache::new(2);
    let image_keys = prefixes.prepare_key(tokens, &[])?;
    let mut request = Active { constraint: None, id, lease,
        job: NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), max_tokens: 4, sampling: Default::default(), events },
        decoder: ds41rt_loader::streaming_token_decoder(snapshot, false)?, anchor,
        generated: 0, buffered: 0, lane: 0, finished: false, cacheable: false,
        tokens: tokens.to_vec(), image_keys, next_after_commit: None };
    request.emit(&[anchor])?;
    ensure!(!request.finished, "fixture requires a nonterminal continuation anchor");
    let mut active = [Some(request), None];
    let [first_transport, second_transport] = transports;
    type Pass<'w, 'a> = crate::v41_target_pass::DistributedTargetPass<'w, 'a>;
    Pass::begin_request(first_transport)?;
    Pass::begin_request(second_transport)?;
    Pass::decode_round(lib, runtime, first, second, requests, first_transport, second_transport,
        &mut active, &[vec![0], vec![]], Some(draft), &mut prefixes, &receive, admission::Wake::default())?;
    ensure!(active.iter().all(Option::is_none), "distributed serving round did not retire its request");
    ensure!(requests.cache().request_id(lease).is_err(), "retired target lease remains live");
    let mut finished = 0;
    while let Ok(chunk) = output.try_recv() {
        match chunk {
            Ok(InferenceChunk::Finish { .. }) => finished += 1,
            Err(error) => anyhow::bail!("distributed serving output failed: {error:?}"),
            _ => (),
        }
    }
    ensure!(finished == 1, "distributed serving did not emit exactly one finish");
    // Reusing the same ID verifies that retirement released the draft owner too.
    draft.admit(id)?;
    draft.release(id)?;
    Pass::reset_connections(first_transport)?;
    Pass::reset_connections(second_transport)?;
    Ok(())
}

pub(super) struct Active<'a> {
    constraint: Option<super::constraints::State<'a>>,
    id: u64,
    lease: CacheLease,
    job: NativeRequest,
    decoder: ds41rt_loader::StreamingTokenDecoder,
    anchor: u32,
    generated: usize,
    buffered: usize,
    lane: usize,
    finished: bool,
    cacheable: bool,
    tokens: Vec<u32>,
    image_keys: ImageKeys,
    next_after_commit: Option<TokenScores>,
}
impl Active<'_> {
    fn emit_one(&mut self, token: u32) -> Result<[Option<InferenceChunk>; 3]> {
        ensure!(!self.job.events.is_closed(), "client disconnected");
        if let Some(constraint) = &mut self.constraint { constraint.accept(token)?; }
        self.anchor = token;
        self.tokens.push(token);
        self.generated += 1;
        self.buffered += 1;
        let mut chunks = [None, None, None];
        let mut count = 0;
        let mut push = |chunk| { chunks[count] = Some(chunk); count += 1; };
        if token != 1 {
            if let Some(content) = self.decoder.step(token)? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
        }
        if token == 1 || self.generated == self.job.max_tokens {
            if let Some(content) = self.decoder.finish()? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
            if self.buffered > 0 {
                push(InferenceChunk::Text { content: String::new(), content_tokens: self.buffered });
                self.buffered = 0;
            }
            push(InferenceChunk::Finish { finish_reason: if token == 1 { InferenceFinishReason::Stop }
                else { InferenceFinishReason::Length } });
            self.finished = true;
            self.cacheable = true;
        }
        Ok(chunks)
    }
    fn emit(&mut self, tokens: &[u32]) -> Result<()> {
        for &token in tokens {
            for chunk in self.emit_one(token)?.into_iter().flatten() { self.job.events.blocking_send(Ok(chunk))?; }
            if self.finished { break; }
        }
        Ok(())
    }
}

fn retire_request<'a, C: DraftChain<'a>>(request: Active<'a>, requests: &mut Requests<'a>,
    prefixes: &mut PrefixCache<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
    // Chunk-4b: retention is consulted **before** the retained frontier is
    // required. With the turn bank disabled `retain` early-returns, so requiring
    // `next_after_commit` first would turn a cache-disabled deployment into a
    // spurious "finished request has no retained logits" warning -- and the
    // scheduler correctly skipped that transfer.
    if request.cacheable && prefixes.turn_bank_enabled()
        && requests.cache().request_id(request.lease).is_ok() {
        let retained = request.next_after_commit.as_ref().context("finished request has no retained logits")
            .and_then(|next| prefixes.retain(SnapshotKind::Turn, &request.tokens, &request.image_keys,
                next, request.id, request.lease, requests, draft.as_deref_mut()));
        if let Err(error) = retained { tracing::warn!(%error, "completed request prefix was not retained"); }
    }
    // Release both owners even if one cleanup reports an error.
    let target = requests.release_if_present(request.lease);
    let speculative = draft.map(|draft| draft.release(request.id)).transpose();
    target.and(speculative.map(|_| ()))
}

/// Allocate the optional pinned pool before signalling HTTP readiness. A bad
/// cache configuration or allocation failure must fail startup, not leave a
/// healthy-looking front door whose worker has already exited.
pub(super) fn prepare_prefix_cache<'a>(lib: &'a NativeLibrary, args: &crate::cli::NativeServeArgs,
    requests: &Requests<'a>) -> Result<PrefixCache<'a>> {
    let template = requests.cache().sources()[0].get().source_cache().page_segments(0)[0];
    let mut config = args.host_cache_config()?;
    if matches!(args.host_cache_bytes, super::memory::HostBudget::Auto) {
        // Logical source capacity, irrespective of which GPU owns each source.
        // Do not count replicated storage or private COW/tail pages as tokens.
        let raw_tokens = requests.cache().sources().iter().zip([2u64, 2, 2, 1])
            .map(|(source, ratio)| source.get().source_cache().capacity as u64 * ratio)
            .min().context("missing compressed cache sources")?;
        let spare_tokens = (u64::from(args.concurrency) + 2*u64::from(args.prefix_cache_entries))*512;
        let budget = ds41rt_hostcache::budget::plan(raw_tokens.saturating_sub(spare_tokens),
            args.prefix_cache_entries, args.max_context_tokens, config.chunk_bytes, args.dspark)?;
        config.bytes = budget.pinned_bytes;
        config.validate()?;
        tracing::info!(target: "ds41rt::host_cache", capacity=?budget,
            "automatic host cache budget (staging and snapshot overhead excluded from token capacity)");
    }
    let host_cache = super::prefix::HostCacheBinding::new(lib, config, template)?;
    Ok(PrefixCache::new(args.prefix_cache_entries as usize).with_host_cache(host_cache))
}

/// The per-second `stats` payload for the serving worker.
///
/// `target_sampling` (and therefore every `frontier_*` counter) is published
/// **unconditionally**. Before the chunk-4b review it was nested under
/// `if let Some(metrics) = prefixes.host_metrics()`, so the retention-gate
/// counters vanished in exactly the cache-disabled deployment the gate targets
/// and the phase-3 campaign could not measure the saving there (review FIX 1).
/// `host_cache`/`host_cache_config` are `null` without a cache, which preserves
/// the previous key shape when one is bound.
fn serving_stats(prefixes: &PrefixCache<'_>) -> serde_json::Value {
    // Deliberately exports the cache's whole effective `Config` under
    // `host_cache_config` (packet HC-9), not just `store_pace_ns`: fleet
    // operators tune several of these knobs, and one key keeps the export
    // forward-compatible as new knobs land.
    serde_json::json!({
        "host_cache": prefixes.host_metrics(),
        "host_cache_config": prefixes.host_config(),
        "target_sampling": sampling_stats::snapshot(),
    })
}

pub(super) fn serve<'w, 'a, P: ServingTarget<'w, 'a>>(lib: &'a NativeLibrary, args: &crate::cli::NativeServeArgs,
    runtime: &tokio::runtime::Runtime, receive: &mut mpsc::Receiver<NativeRequest>,
    first: &mut P, second: &mut P,
    requests: &mut Requests<'a>, first_transport: &mut P::Transport,
    second_transport: &mut P::Transport, mut draft: Option<&mut DraftRuntime<'w, 'a, P::Chain>>,
    vision: &mut crate::v41_vision::VisionRuntime<'a>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
    mut prefixes: PrefixCache<'a>,
) -> Result<()> {
    let mut active: Vec<Option<Active<'a>>> = (0..args.concurrency).map(|_| None).collect();
    let mut compiler = super::constraints::Compiler::new(lib, args.snapshot.join("tokenizer.json"));
    let mut id = 0u64;
    let mut closed = false;
    let mut pending: Option<admission::Pending> = None;
    let mut stats_published = Instant::now();
    let limits = ds41rt_api::native_v41::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    loop {
        prefixes.tick();
        if stats_published.elapsed() >= std::time::Duration::from_secs(1) {
            stats_published = Instant::now();
            if let Ok(mut slot) = stats.lock() {
                *slot = serving_stats(&prefixes);
            }
        }
        // This point is reached only after both complete stacks have drained and
        // committed. No cache owner is migrated or retired inside a layer stack.
        for entry in &mut active {
            if entry.as_ref().is_some_and(|r| r.finished || r.job.events.is_closed()) {
                retire_request(entry.take().unwrap(), requests, &mut prefixes, draft.as_deref_mut())?;
            }
        }
        let mut loads = [0usize; 2];
        for request in active.iter().flatten() { loads[request.lane] += 1; }
        while loads[0].abs_diff(loads[1]) > 1 {
            let heavy = usize::from(loads[1] > loads[0]);
            let request = active.iter_mut().flatten().find(|r| r.lane == heavy).unwrap();
            request.lane = 1 - heavy;
            loads[heavy] -= 1; loads[1 - heavy] += 1;
        }
        // Admit available work at a completed boundary. Prefill currently owns
        // both lanes; mixed prefill/decode interleaving is a subsequent policy.
        while let Some(slot) = active.iter().position(Option::is_none) {
            let active_count = active.iter().flatten().count();
            if pending.as_ref().is_some_and(|p| p.active_when_blocked == active_count
                && !p.prepared.job.events.is_closed()) { break; }
            let prepared = if let Some(pending) = pending.take() { pending.prepared } else {
                let job = if active_count == 0 && !closed {
                    match receive.blocking_recv() { Some(job) => job, None => { closed = true; break; } }
                } else {
                    match receive.try_recv() {
                        Ok(job) => job,
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; break; }
                    }
                };
                if job.events.is_closed() { continue; }
                let events = job.events.clone();
                match admission::Prepared::new(job, &args.snapshot, limits) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                            .cloned().unwrap_or_else(|| format!("{error:#}").into());
                        let _ = events.blocking_send(Err(failure));
                        continue;
                    }
                }
            };
            if prepared.job.events.is_closed() { continue; }
            id = id.checked_add(1).context("request ID exhausted")?;
            let lease = requests.admit(slot, id)?;
            let lane = usize::from(loads[1] < loads[0]);
            let events = prepared.job.events.clone();
            let admitted = (|| -> Result<_> {
                let admission::Prepared { job, prompt, images } = &prepared;
                if let Some(draft) = draft.as_deref_mut() { draft.admit(id)?; }
                let image_keys = prefixes.prepare_key(prompt, images)?;
                if !images.is_empty() {
                    requests.attach_images(lease, crate::v41_requests::RequestImages::new(images)?)?;
                }
                let hit = prefixes.restore(prompt, &image_keys, id, lease, requests, draft.as_deref_mut())?;
                // Check the declared lifetime budget of every active request,
                // including the new request, against the actual source pages.
                // This preserves prefix sharing and accounts for partial-page COW.
                let mut capacity = active.iter().flatten().map(|r| Ok((r.lease,
                    admission::remaining_budget(r.tokens.len(), r.job.max_tokens-r.generated,
                        requests.cache().committed_end(r.lease)?)?)))
                    .collect::<Result<Vec<_>>>()?;
                capacity.push((lease, admission::remaining_budget(prompt.len(), job.max_tokens,
                    requests.cache().committed_end(lease)?)?));
                prefixes.make_room(requests, &capacity)?;
                Ok((image_keys, hit))
            })();
            let (image_keys, hit) = match admitted {
                Ok(value) => value,
                Err(error) => {
                    requests.release_if_present(lease)?;
                    if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                    if error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>().is_some() {
                        if active_count > 0 {
                            tracing::debug!(request_id=id, active_count, "waiting for request KV token budget");
                            pending = Some(admission::Pending { prepared, active_when_blocked: active_count });
                            break;
                        }
                        let _ = events.blocking_send(Err(ds41rt_api::native_v41::NativeFailure::BadRequest(
                            "prompt plus max_tokens exceeds the GPU KV pool; reduce max_tokens or increase the pool".into())));
                    } else {
                        let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                            .cloned().unwrap_or_else(|| format!("{error:#}").into());
                        let _ = events.blocking_send(Err(failure));
                    }
                    continue;
                }
            };
            let admission::Prepared { job, prompt, images } = prepared;
            let result = (|| -> Result<Active<'a>> {
                let mut constraint = job.constraint.as_ref().map(|spec| compiler.matcher(spec)).transpose()?;
                ensure!(!job.events.is_closed(), "client disconnected");
                let decoder = ds41rt_loader::streaming_token_decoder(&args.snapshot, false)?;
                let cached = hit.as_ref().map_or(0, |(end, _)| *end);
                let source_end = requests.cache().committed_end(lease)? as usize;
                if !images.is_empty() {
                    // Deliberately unheld (packet HC-9): this is per-image pre-prefill
                    // preparation, not a batch-tokens chunk, so the store-pace pacing hold
                    // does not apply here; the hold covers prefill chunk dispatch only.
                    let start = if requests.cache().stage(lease)? == crate::v41_backbone_cache::CacheStage::EncoderReplay {
                        requests.cache().history_end(lease)? as usize
                    } else { source_end };
                    let needed = requests.images(lease)?.needed(start, prompt.len())?;
                    let started = Instant::now();
                    for &index in &needed {
                        ensure!(!job.events.is_closed(), "client disconnected");
                        let features = vision.encode(&images[index].image)?;
                        let mut bytes = vec![0; features.bytes];
                        lib.copy_d2h(&mut bytes, features)?;
                        requests.install_image_features(lease, index, bytes)?;
                    }
                    tracing::info!(request_id=id, images=images.len(), encoded_images=needed.len(),
                        encoder_ms=started.elapsed().as_secs_f64()*1000.0, "native vision preparation");
                }
                drop(images);
                job.events.blocking_send(Ok(InferenceChunk::Ready {
                    system_fingerprint: Some(if draft.is_some() { "ds41rt-native-fp4-kv-dspark" }
                        else { "ds41rt-native-fp4-kv" }.into()),
                    prompt_usage: PromptUsage { prompt_tokens: prompt.len(), prompt_cache_hit_tokens: cached },
                }))?;
                P::begin_request(first_transport)?; P::begin_request(second_transport)?;
                let scores = if cached == prompt.len() { hit.expect("complete prefix hit").1.context("exact prefix has no logits")? }
                else { prefill(lib, runtime, first, second, requests, first_transport,
                    second_transport, lease, &prompt, args.prefill_batch_tokens as usize, &job,
                    draft.as_deref_mut(), &mut || prefixes.prefill_hold())? };
                if cached != prompt.len() {
                  if let Err(error) = prefixes.retain(SnapshotKind::Prompt, &prompt, &image_keys, &scores, id, lease, requests, draft.as_deref_mut()) {
                    tracing::warn!(%error, "prompt prefix was not retained");
                  }
                }
                tracing::debug!(request_id=id, prompt_tokens=prompt.len(), cached_tokens=cached, "native prefix admission");
                let mask = constraint.as_mut().map(|state| state.mask()).transpose()?.flatten();
                // The first generated token is emitted-token index 0.
                let anchor = scores.sample(mask, job.sampling, 0)?;
                Ok(Active { constraint, id, lease, job, decoder, anchor, generated: 0, buffered: 0, lane,
                    finished: false, cacheable: false, tokens: prompt, image_keys, next_after_commit: Some(scores) })
            })();
            match result {
                Ok(mut request) => {
                    if let Err(error) = request.emit(&[request.anchor]) {
                        let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                        request.finished = true;
                    }
                    active[slot] = Some(request); loads[lane] += 1;
                }
                Err(error) => {
                    // Other completed requests retain their caches.
                    let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                        .cloned().unwrap_or_else(|| format!("{error:#}").into());
                    let _ = events.blocking_send(Err(failure));
                    tracing::warn!(%error, "native request admission failed");
                    requests.release_if_present(lease)?;
                    if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                    P::reset_connections(first_transport)?; P::reset_connections(second_transport)?;
                }
            }
        }
        if active.iter().all(Option::is_none) { if closed { break; } else { continue; } }
        let members: [Vec<usize>; 2] = std::array::from_fn(|lane| active.iter().enumerate()
            .filter_map(|(slot, request)| request.as_ref().filter(|r| r.lane == lane && !r.finished
                && !r.job.events.is_closed()).map(|_| slot)).collect());
        if members.iter().all(Vec::is_empty) { continue; }
        let capacity: Vec<_> = members.iter().flatten().map(|&slot| {
            let r = active[slot].as_ref().unwrap();
            (r.lease, (r.job.max_tokens - r.generated).min(draft.as_ref().map_or(1, |d| d.max_verify_rows())) as u32)
        }).collect();
        let room = prefixes.make_room(requests, &capacity);
        if let Err(error) = &room {
            if let Some(pressure) = error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>() {
                let slot = *members.iter().flatten().nth(pressure.work_index)
                    .context("pool pressure references an invalid append participant")?;
                let request = active[slot].as_mut().unwrap();
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
                // No layer stack started. The next completed-boundary cleanup
                // releases only this owner's pages, then retries remaining work.
                continue;
            }
        }
        let result = room.and_then(|_| P::decode_round(lib, runtime, first, second,
            requests, first_transport, second_transport, &mut active, &members,
            draft.as_deref_mut(), &mut prefixes, receive,
            admission::Wake { blocked_at: pending.as_ref().map(|p| p.active_when_blocked),
                pending: pending.as_ref().map(|p| &p.prepared.job) }));
        if let Err(error) = result {
            tracing::error!(error=%format!("{error:#}"), "native decode round failed");
            P::reset_connections(first_transport)?; P::reset_connections(second_transport)?;
            for request in active.iter_mut().flatten() {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
            }
        }
    }
    Ok(())
}

/// Process-wide device-terminal instrumentation.
///
/// Chunk 4a's fallback rule is "visible, never silent". The per-row fallback is
/// also logged at `WARN` on the `ds41rt::sampling` target, but a log line is
/// easy to miss and impossible to alert on, so every device round and every
/// fallback row is counted here and exported through the existing per-second
/// `stats` JSON under `target_sampling` (the `serve` loop is the only publisher).
///
/// Counters are monotonic since process start and are read with `Relaxed`
/// ordering: they are diagnostics, not a synchronization device.
pub(super) mod sampling_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Sampled rounds routed into the device terminal rather than the CPU path.
    pub(super) static DEVICE_ROUNDS: AtomicU64 = AtomicU64::new(0);
    /// Rounds whose every row was a fallback (today: every request stochastic
    /// with `top_k >= 257`). Those stay on the CPU path; the device is not
    /// launched for a round it would only serve as no-ops.
    pub(super) static CPU_ROUNDS: AtomicU64 = AtomicU64::new(0);
    /// Rows the device selected.
    pub(super) static DEVICE_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Rows the host routed to the CPU sampler before the launch.
    pub(super) static PLANNED_FALLBACK_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Rows the device launched but reported a non-OK, non-request-fault status
    /// for (`INTERNAL`), and which the host re-sampled on the CPU.
    pub(super) static REFUSED_FALLBACK_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Finishing frontier rows whose full row was downloaded from the device
    /// because retention needed it (prefix caching enabled and the row was not
    /// already on the host).
    pub(super) static FRONTIER_DOWNLOAD_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Finishing frontier rows the retention gate **skipped**: prefix caching is
    /// disabled and no frontier will be stored, or the finishing client has
    /// disconnected. Each skipped row is one `ROW_BYTES` full-row transfer the
    /// pre-chunk-4b code issued unconditionally (design §10.4).
    pub(super) static FRONTIER_GATED_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Bytes [`FRONTIER_GATED_ROWS`] would have cost. This is the **gate**
    /// removal only (serial and independent lane both paid it before 4b).
    pub(super) static FRONTIER_GATED_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Finishing frontier rows retained in place from bytes the round had already
    /// downloaded (a fallback row). On the **serial** lane the pre-4b code never
    /// transferred these (`retain_from_device` short-circuited on packed bytes);
    /// on the **independent** lane it did (`download_logits` downloaded every
    /// entry of `frontier_downloads`), so only the independent-lane rows are a
    /// byte saving. The two cases are therefore counted separately.
    pub(super) static FRONTIER_PACKED_ROWS: AtomicU64 = AtomicU64::new(0);
    /// Independent-lane packed-frontier transfers removed. Kept apart from
    /// [`FRONTIER_GATED_BYTES`] so no number is presented as the total saving
    /// (chunk-4b review FIX 2).
    pub(super) static FRONTIER_PACKED_SAVED_BYTES: AtomicU64 = AtomicU64::new(0);

    pub(super) fn record_frontier_download(rows: usize) {
        FRONTIER_DOWNLOAD_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// Record `rows` frontiers the gate skipped, and the bytes they would have cost.
    pub(super) fn record_frontier_gated(rows: usize, row_bytes: usize) {
        FRONTIER_GATED_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        FRONTIER_GATED_BYTES.fetch_add(rows as u64 * row_bytes as u64, Ordering::Relaxed);
    }

    /// Record `rows` packed frontier retentions. `transferred_before` is true only
    /// on the independent lane, which downloaded every frontier row before 4b.
    pub(super) fn record_frontier_packed(rows: usize, row_bytes: usize, transferred_before: bool) {
        FRONTIER_PACKED_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        if transferred_before {
            FRONTIER_PACKED_SAVED_BYTES.fetch_add(rows as u64 * row_bytes as u64, Ordering::Relaxed);
        }
    }

    pub(super) fn record_round(device_round: bool, device_rows: usize, planned: usize) {
        if device_round {
            DEVICE_ROUNDS.fetch_add(1, Ordering::Relaxed);
        } else {
            CPU_ROUNDS.fetch_add(1, Ordering::Relaxed);
        }
        DEVICE_ROWS.fetch_add(device_rows as u64, Ordering::Relaxed);
        PLANNED_FALLBACK_ROWS.fetch_add(planned as u64, Ordering::Relaxed);
    }

    pub(super) fn record_refused(rows: usize) {
        REFUSED_FALLBACK_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    }

    /// The exported snapshot: the device-terminal counters under the stats
    /// JSON's `target_sampling` key. A fallback is therefore visible to an
    /// operator, not only in the `WARN` log line.
    pub(super) fn snapshot() -> serde_json::Value {
        let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        serde_json::json!({
            "device_rounds": get(&DEVICE_ROUNDS),
            "cpu_fallback_rounds": get(&CPU_ROUNDS),
            "device_rows": get(&DEVICE_ROWS),
            "planned_fallback_rows": get(&PLANNED_FALLBACK_ROWS),
            "refused_fallback_rows": get(&REFUSED_FALLBACK_ROWS),
            "frontier_download_rows": get(&FRONTIER_DOWNLOAD_ROWS),
            "frontier_gated_rows": get(&FRONTIER_GATED_ROWS),
            "frontier_gated_bytes": get(&FRONTIER_GATED_BYTES),
            "frontier_packed_rows": get(&FRONTIER_PACKED_ROWS),
            "frontier_packed_saved_bytes": get(&FRONTIER_PACKED_SAVED_BYTES),
        })
    }
}

/// Which kernel route one row of a sampled round takes.
///
/// Chunk 4a is the first chunk that serves stochastic rows on the device, so a
/// round is no longer homogeneous. Every row is classified **independently** by
/// its own parameters, exactly the way the device kernels classify themselves
/// (`v41_sampling_gpu.cu`: K1's greedy branch, `k2_applicable`, the K3/K4
/// eligibility block, and K5's ordered-row class), and the classification is the
/// routing table of design §4.6:
///
/// | host route | condition | kernels |
/// | --- | --- | --- |
/// | [`SamplingRoute::DeviceGreedy`] | greedy (`temperature < 1e-5` or `top_k == 1`) | K1 |
/// | [`SamplingRoute::DeviceFastPath`] | `top_k == 0 && top_p >= 1.0` | K1 → K2 |
/// | [`SamplingRoute::DeviceOrdered`] | `top_k in 1..=256` (ordered path), or `top_k == 0 && top_p < 1.0` (K5 case 3) | K1 → K3 → K4 → K5 |
/// | [`SamplingRoute::CpuFallback`] | the device cannot serve this row (below) | CPU sampler on the downloaded row |
///
/// The all-greedy, unconstrained, untraced round never reaches the router: it is
/// the `compact` flag that selects the existing compact lane (K1 argmax + two
/// small D2H) without building a `SamplingPlan` at all, so there is no plan-level
/// route for it.
///
/// `DeviceOrdered` deliberately does **not** distinguish `top_k < survivor_count`
/// from `top_k >= survivor_count`: K3/K4 are a per-row no-op in the second case
/// and K5 then treats the retained set as every survivor (its cases 2 and 3), so
/// the kernel contract already covers it. What the host must exclude is a
/// `top_k` above K5's `kBlock` retained-table limit (256), which is
/// [`SamplingRoute::CpuFallback`] and is a **loud, counted** fallback rather than
/// a silent wrong token (design §20 item 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SamplingRoute {
    DeviceGreedy,
    DeviceFastPath,
    DeviceOrdered,
    CpuFallback,
}

impl SamplingRoute {
    /// Whether the row is selected by a device kernel. A `CpuFallback` row is
    /// not; every other route is.
    pub(crate) fn device_served(self) -> bool {
        !matches!(self, SamplingRoute::CpuFallback)
    }
    /// Whether the row needs the K3/K4 → K5 ordered tail.
    fn is_ordered(self) -> bool {
        matches!(self, SamplingRoute::DeviceOrdered)
    }
}

/// The host-side capability gate for one row, pure over the request parameters
/// and the row's K1 inputs.
///
/// This must mirror the device's own eligibility exactly; where it cannot (the
/// survivor count is a device reduction), it chooses the *conservative* side and
/// the row falls back to the CPU sampler, which is always correct and is counted
/// (see [`SAMPLING_FALLBACK_ROWS`]).
///
/// Fallback cases, and why each is a fallback rather than an error:
///
/// 1. **`top_k >= 257`.** K5's inclusive-prefix table is one `__shared__` f32
///    per rank and holds at most `kBlock == 256`; a wider retained list is
///    `INTERNAL` by contract (design §20 item 6). The CPU sampler serves it
///    exactly.
/// 2. **A parameter outside the validator's accepted range** (`temperature` not
///    finite or outside `[0, 2]`, `top_p` not finite or outside `(0, 1]`,
///    `min_p` not finite or outside `[0, 1]`, or an inconsistent `ln_min_p`).
///    The FFI validator rejects the **whole batch** for such a row, so it must
///    not reach the launch at all. The CPU sampler reports the same
///    `InvalidParameter` error the CPU path reports today.
/// 3. **`min_p > 1.0`.** With `min_p > 1` K1's survivor count is zero on every
///    row, which K1 reports as `INTERNAL` by contract; the CPU sampler's filter
///    chain keeps the best token alive (`target_sampling.rs:23-25`) and samples
///    it. Routing it to the device would turn a servable request into an error.
///
/// Everything else is servable. In particular a constrained (masked) stochastic
/// row is servable: the mask is applied by K1/K2/K5 first, exactly as the CPU
/// `select_verification_sampled` applies it.
fn sampling_route(params: ds41rt_core::TargetSamplingParams) -> SamplingRoute {
    if params.is_greedy() {
        return SamplingRoute::DeviceGreedy;
    }
    let top_k = params.top_k().map_or(0u32, |k| k as u32);
    let temperature = params.temperature();
    let top_p = params.top_p();
    let min_p = params.min_p();
    let finite_range = temperature.is_finite()
        && (0.0..=2.0).contains(&temperature)
        && top_p.is_finite()
        && top_p > 0.0
        && top_p <= 1.0
        && min_p.is_finite()
        && (0.0..=1.0).contains(&min_p);
    if !finite_range {
        return SamplingRoute::CpuFallback;
    }
    if min_p > 1.0 {
        // `min_p` is validated to `<= 1.0` by `TargetSamplingParams::new`; a
        // raw value above it would zero the survivor set.
        return SamplingRoute::CpuFallback;
    }
    // `ln_min_p` is always the host `f32::ln` the validator recomputes: the
    // plan builds both from the same `min_p` with `target_sampling_row`, so
    // there is no separate check to make here.
    if top_k > 0 {
        if top_k > crate::v41_target_head::DS41RT_V41_SAMPLING_MAX_RETAINED {
            return SamplingRoute::CpuFallback;
        }
        return SamplingRoute::DeviceOrdered;
    }
    if top_p < 1.0 {
        // K5 case 3: no retained list, the nucleus is drawn from every survivor.
        return SamplingRoute::DeviceOrdered;
    }
    SamplingRoute::DeviceFastPath
}

/// Per-row parameters and routing for one sampled round.
pub(crate) struct SamplingPlan {
    /// The device-facing parameter blocks, in batch-row order.
    pub(crate) rows: Vec<TargetSamplingRowRequest>,
    pub(crate) greedy: Vec<bool>,
    /// Per row, the route the row takes. Parallel to `rows`.
    pub(crate) route: Vec<SamplingRoute>,
    /// Per row, the row's grammar mask (or `None`). Parallel to `rows`, and the
    /// source of both the mask arena and the CPU fallback re-sample's mask.
    pub(crate) mask: Vec<Option<Vec<u32>>>,
    /// Per row, the **request's** sampling parameters, so a fallback row is
    /// re-sampled with the request's own values rather than the device-facing
    /// substitution.
    pub(crate) params: Vec<ds41rt_core::TargetSamplingParams>,
    /// Per row, the **absolute emitted-token index** the row's draw is keyed on,
    /// exactly as the CPU path keys it. Kept per row so a fallback draw cannot
    /// depend on the member layout.
    pub(crate) position: Vec<u64>,
}

impl SamplingPlan {
    /// Whether the row was a pre-launch fallback (the device was never asked to
    /// serve it). A post-launch device error is handled by the caller, which
    /// adds the row to the fallback set it passes to [`SamplingRound::fallback_rows`].
    pub(crate) fn planned_fallback(&self, row: usize) -> bool {
        !self.route[row].device_served()
    }
    /// How many rows the host routed to the CPU sampler before the launch.
    #[cfg(test)]
    pub(crate) fn planned_fallback_count(&self) -> usize {
        self.route.iter().filter(|route| !route.device_served()).count()
    }
    /// Whether any row needs the K3/K4 -> K5 ordered tail (the same predicate
    /// the wave launch takes).
    #[cfg(test)]
    pub(crate) fn routes_need_ordered_tail(&self) -> bool {
        self.route.iter().any(|route| route.is_ordered())
    }
    /// Whether the round has at least one device-servable row, i.e. whether
    /// launching the device terminal can select anything at all.
    #[cfg(test)]
    pub(crate) fn routes_need_device_terminal(&self) -> bool {
        self.route.iter().any(|route| route.device_served())
    }
}

/// One member's sampling inputs for a round, resolved from `Active` by the
/// caller so the planner itself is a pure function of its arguments.
#[derive(Clone)]
pub(crate) struct SamplingMember {
    pub(crate) params: ds41rt_core::TargetSamplingParams,
    pub(crate) base_position: u64,
    /// Per row: the grammar mask, or `None` when the grammar allows every
    /// token. `needs_mask` is a per-row property, so it decides that row's
    /// `NO_MASK` flag — a constrained request whose grammar is exhausted on a
    /// later row must not have the previous row's grammar applied to it.
    pub(crate) row_masks: Vec<Option<Vec<u32>>>,
}

/// Build the per-row parameter blocks and routes for one round.
///
/// `position(offset + index)` is the row's absolute emitted-token index, so a
/// row's draw and its greedy decision never depend on the batch layout. The
/// seed and position of a row are request-local by construction: they come from
/// the owning request's `TargetSamplingParams` and its own `generated` counter,
/// never from the row's place in the batch or from a peer.
///
/// `mask_row` is simply the row's ordinal: the mask arena is `rows x
/// ceil(vocab/32)` and indexed by row, with `NO_MASK` rows carrying the
/// `0xFFFFFFFF` sentinel. Keeping the two a single flat mapping means no
/// ordinal remapping can silently pair a row with another row's grammar; it also
/// satisfies K5's mandatory identity `output_row == block_row`, which
/// `TargetSamplingWave::upload` re-asserts.
///
/// A row that must fall back (`sampling_route` → `CpuFallback`) is given a
/// **device-neutral** parameter block: an unconstrained greedy row. That keeps
/// the whole batch acceptable to the FFI validator (which rejects the batch, not
/// the row, for an out-of-range parameter or an unmaterializable `top_k`) and
/// makes every downstream stage a per-row no-op for it, so the device can never
/// publish a token that the host then has to remember to ignore. The CPU
/// re-sample reads the *planned* parameters, which are kept separately, not
/// these.
pub(crate) fn build_target_sampling_plan(members: &[SamplingMember], inputs: &[Vec<u32>],
) -> Result<SamplingPlan> {
    let mut plan = SamplingPlan {
        rows: Vec::new(), greedy: Vec::new(), route: Vec::new(), mask: Vec::new(),
        params: Vec::new(), position: Vec::new(),
    };
    for (meta, input) in members.iter().zip(inputs) {
        let params = meta.params;
        ensure!(
            meta.row_masks.len() == input.len(),
            "sampling row masks differ from the member's rows"
        );
        for index in 0..input.len() {
            let greedy = params.is_greedy();
            let output_row = u32::try_from(plan.rows.len()).context("sampled row index overflow")?;
            let needs_mask = meta.row_masks[index].is_some();
            let route = sampling_route(params);
            // `STRICT_FINITE` is set for masked and greedy rows, which are
            // exactly the rows whose finiteness must be checked before the mask
            // test (`scores.rs::argmax`). A stochastic unmasked row stays
            // permissive, which is what the validator also requires.
            let mut flags = 0u32;
            if needs_mask || greedy {
                flags |= ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE;
            }
            if greedy {
                flags |= ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_GREEDY;
            }
            let mask_row = if needs_mask {
                output_row
            } else {
                flags |= ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK;
                ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW
            };
            let row = if route.device_served() {
                target_sampling_row(params, meta.base_position + index as u64, mask_row,
                    flags, output_row)
            } else {
                fallback_sampling_row(meta.base_position + index as u64, output_row)
            };
            plan.rows.push(TargetSamplingRowRequest { row, greedy });
            plan.greedy.push(greedy);
            plan.route.push(route);
            plan.mask.push(meta.row_masks[index].clone());
            plan.params.push(params);
            plan.position.push(meta.base_position + index as u64);
        }
    }
    Ok(plan)
}

/// The device-facing block for a row the device cannot serve.
///
/// It is a valid **unconstrained greedy** row: the FFI validator's range checks
/// pass, K1's argmax branch runs (writing an id the host ignores), and every
/// later stage is a per-row no-op (`k2_applicable` is false for a greedy row,
/// the K3/K4 eligibility block requires non-greedy, and K5's ordered row class
/// requires non-greedy). The seed/position are kept so the block stays a
/// faithful record of which row it is.
fn fallback_sampling_row(position: u64, output_row: u32) -> ds41rt_ffi::Ds41rtV41SamplerRow {
    ds41rt_ffi::Ds41rtV41SamplerRow {
        seed: 0,
        position,
        temperature: 0.0,
        top_p: 1.0,
        min_p: 0.0,
        top_k: 0,
        mask_row: ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW,
        flags: ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_GREEDY
            | ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE
            | ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK,
        output_row,
        ln_min_p: f32::NEG_INFINITY,
        ..ds41rt_ffi::Ds41rtV41SamplerRow::default()
    }
}

/// Resolve one row's parameter block from the request's sampling parameters.
///
/// The host does every disabled-encoding resolution here, so the kernel's
/// branch is a plain test (design §5.1):
/// * `top_k = None` -> `0` (disabled); `Some(k)` passes through, including
///   `k > vocab`, which the kernel treats as a no-op;
/// * `min_p = 0` -> `ln_min_p = -inf`, so the device's single add produces
///   `-inf` and the survivor predicate keeps the whole allowed row;
/// * otherwise `ln_min_p` is the **same** Rust `f32::ln` the CPU sampler
///   evaluates, so the threshold comparison is bit-identical (design §6.3a).
fn target_sampling_row(params: ds41rt_core::TargetSamplingParams, position: u64,
    mask_row: u32, flags: u32, output_row: u32,
) -> ds41rt_ffi::Ds41rtV41SamplerRow {
    ds41rt_ffi::Ds41rtV41SamplerRow {
        seed: params.seed(),
        position,
        temperature: params.temperature(),
        top_p: params.top_p(),
        min_p: params.min_p(),
        top_k: params.top_k().map_or(0, |k| k as u32),
        mask_row,
        flags,
        output_row,
        ln_min_p: if params.min_p() > 0.0 { params.min_p().ln() } else { f32::NEG_INFINITY },
        ..ds41rt_ffi::Ds41rtV41SamplerRow::default()
    }
}

/// Fill the host mask arena: exactly one `ceil(vocab/32)`-word row per batch
/// row, indexed by row ordinal, so `mask_row == row` is the only mapping.
///
/// `row_masks[member][index]` is that row's grammar mask, or `None` when the
/// grammar allows every token. A `None` row's arena slice stays zero and is
/// never read (the plan marks it `NO_MASK`), so it can never inherit a
/// neighbour's grammar.
///
/// The §5.3 remainder rule (clear every bit `>= vocab` of the final word) is
/// applied by [`TargetSamplingWave::upload`] immediately before the H2D copy.
pub(crate) fn build_sampling_masks(row_masks: &[Vec<Option<Vec<u32>>>], arena: &mut [u32]) -> Result<()> {
    let words_per_row = VOCAB.div_ceil(32);
    let rows: usize = row_masks.iter().map(Vec::len).sum();
    ensure!(arena.len() >= rows * words_per_row, "grammar mask arena is too small");
    let mut row = 0usize;
    for member in row_masks {
        for mask in member {
            if let Some(mask) = mask {
                ensure!(mask.len() == words_per_row, "grammar mask extent differs");
                let start = row * words_per_row;
                arena[start..start + words_per_row].copy_from_slice(mask);
            }
            row += 1;
        }
    }
    ensure!(row == rows, "grammar mask arena row count differs");
    Ok(())
}

/// Gate the device-selected terminal on both the round shape and the layout's
/// capability.
///
/// A layout whose [`VerificationTarget::supports_sampled_terminal`] is false
/// (chunk 1: `DistributedTargetPass`) keeps the CPU path even for an all-greedy
/// round, because routing it into the terminal would fail the whole lane on
/// `execute_shared_sampled`'s default "no device-selected sampling terminal".
/// That is the **round-level** fallback and it is deliberately not per row: the
/// terminal is absent for the whole layout, so there is no per-row device
/// substitute to fall back *to*.
fn use_sampled_terminal(supports_terminal: bool, routable: bool) -> bool {
    supports_terminal && routable
}

/// Whether this round can use the device terminal at all.
///
/// Chunk 4a: the terminal is used whenever the layout provides it and the round
/// has at least one row, because a stochastic row is now servable (K1 → K2, or
/// K1 → K3 → K4 → K5) and an unservable row is a counted per-row fallback inside
/// the device round rather than a reason to move the whole round to the CPU.
/// This is deliberately *not* "every row is servable": moving the whole round to
/// the CPU because one row has `top_k = 300` would force every greedy and
/// stochastic peer in the round to download full logits, which contract
/// §7.1.15 forbids.
fn round_has_device_rows<'a>(active: &[Option<Active<'a>>], members: &[usize]) -> bool {
    !members.is_empty()
        && members.iter().any(|&slot| {
            active[slot]
                .as_ref()
                .is_some_and(|request| sampling_route(request.job.sampling).device_served())
        })
}

/// Everything one sampled round needs, built once per lane round.
pub(crate) struct SamplingRound {
    pub(crate) plan: SamplingPlan,
    pub(crate) arena: Vec<u32>,
    /// Rows whose full logits must be downloaded for the CPU-side trace. The
    /// `ds41rt::logit_trace` target logs `top_two`, which only the raw row
    /// provides, so a traced round downloads every row exactly as the
    /// pre-device path did (design §8.5/R15).
    pub(crate) trace_rows: Vec<usize>,
}

impl SamplingRound {
    /// Whether any row needs the K3/K4 → K5 ordered tail (design §20 items 1, 2).
    fn ordered_rows(&self) -> bool {
        self.plan.route.iter().any(|route| route.is_ordered())
    }
    /// Rows that have to be downloaded and re-sampled on the CPU, ascending.
    ///
    /// This is the union of the pre-launch fallback set and the post-launch
    /// device errors the caller discovered; the caller passes the latter in.
    pub(crate) fn fallback_rows(&self, extra: &[usize]) -> Vec<usize> {
        let mut rows: Vec<usize> = (0..self.plan.rows.len())
            .filter(|&row| self.plan.planned_fallback(row))
            .chain(extra.iter().copied())
            .collect();
        rows.sort_unstable();
        rows.dedup();
        rows
    }
}

/// Build the per-row plan and the host mask arena for a device-selected round.
///
/// `trace` mirrors the commit-side `ds41rt::logit_trace` gate so both sides
/// agree on which rows are downloaded.
fn build_sampling_round<'a>(active: &[Option<Active<'a>>], members: &[usize],
    inputs: &[Vec<u32>], trace: bool,
) -> Result<SamplingRound> {
    let mut resolved: Vec<SamplingMember> = Vec::with_capacity(members.len());
    for (&slot, input) in members.iter().zip(inputs) {
        let request = active[slot].as_ref().unwrap();
        let row_masks = match request.constraint.as_ref() {
            Some(constraint) => constraint.prepare_verification_masks(input)?,
            None => vec![None; input.len()],
        };
        resolved.push(SamplingMember {
            params: request.job.sampling,
            base_position: request.generated as u64,
            row_masks,
        });
    }
    let plan = build_target_sampling_plan(&resolved, inputs)?;
    let rows: usize = inputs.iter().map(Vec::len).sum();
    let mut arena = vec![0u32; rows * VOCAB.div_ceil(32)];
    // Only the rows the device will actually read are staged. A fallback row is
    // already `NO_MASK` in the plan, so its arena slice stays zero and the
    // upload skips it; clearing it here as well keeps the mapping explicit.
    let row_masks: Vec<Vec<Option<Vec<u32>>>> = resolved.iter().enumerate()
        .map(|(member, meta)| {
            meta.row_masks.iter().enumerate().map(|(index, mask)| {
                let global = resolved[..member].iter().map(|m| m.row_masks.len()).sum::<usize>()
                    + index;
                if plan.planned_fallback(global) { None } else { mask.clone() }
            }).collect()
        })
        .collect();
    build_sampling_masks(&row_masks, &mut arena)?;
    let trace_rows = if trace { (0..rows).collect() } else { Vec::new() };
    Ok(SamplingRound { plan, arena, trace_rows })
}

/// Resolve the CPU-sampled token of every row the device could not serve.
///
/// This is the fallback half of chunk 4a's "fallback, never silent" rule, and it
/// runs **after** the device launch (the pre-launch fallback rows are known
/// before it; the device-error rows are known only from `SampledTargetRows`).
/// `next` must already carry the full logits of every row in `rows`; the caller
/// downloads them.
///
/// The values are the plan's own, not the device-facing substitution: the mask
/// is the row's prepared grammar mask and the parameters are the request's, at
/// the row's absolute position. That is exactly what the CPU path computes in
/// `sample_target_rows` / `State::select_verification_sampled`, so the fallback
/// is the CPU value rather than a look-alike.
///
/// **The result is stored into `next.best[row]`** (`BatchScores::store_sampled`).
/// That matters for a row the device refused *after* the launch: its plan route
/// is still `device_served()`, so the commit path consumes `next.best[row]`
/// directly, and a kernel that reported `INTERNAL` left that slot at the
/// caller's sentinel. Computing the fallback without storing it would commit the
/// stale device slot — a silent wrong token. Storing it also means the commit
/// path never needs to recompute a fallback row, so no row is sampled twice.
pub(crate) fn resolve_fallback_rows(next: &mut BatchScores, round: &SamplingRound,
    rows: &[usize],
) -> Result<()> {
    for &row in rows {
        let mask = round.plan.mask[row].as_deref();
        let params = round.plan.params[row];
        let position = round.plan.position[row];
        next.store_sampled(row, mask, params, position)?;
    }
    Ok(())
}

/// Run the device-selected terminal and assemble the selection batch.
///
/// Chunk 4a: this is entered for any round with at least one device-servable
/// row, so a row's parameters decide its kernels. The rows the device cannot
/// serve are excluded from the plan's device-facing blocks *before* the launch
/// and re-sampled on the CPU from their downloaded rows afterwards; every other
/// row downloads nothing beyond the small ids/scores/status vectors. A traced
/// round still downloads every row (`SamplingRound::trace_rows`) because
/// `ds41rt::logit_trace` logs `top_two`.
///
/// A **device** row whose status is not OK is itself a fallback (the device
/// refused a row the host believed servable), not a lane error: it is added to
/// the fallback set, counted, and re-sampled on the CPU. The two statuses that
/// are genuinely the request's fault and must stay hard errors —
/// `EMPTY_CANDIDATES` (contract §7.1.13) and `NONFINITE_LOGIT` — are propagated
/// by `check_status` exactly as the CPU path propagates them, because the CPU
/// re-sample would fail identically.
async fn execute_sampled_rows<'a>(pass: &mut TargetPass<'_, 'a>,
    requests: &Requests<'a>, batch: &mut RequestBatch, transport: &mut NativeTp4Wave<'a>,
    round: &SamplingRound,
) -> Result<BatchScores> {
    let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
    unsafe {
        pass.execute_sampled(requests, batch, transport, 0, &selected, &round.plan.rows,
            Some(&round.arena), VOCAB.div_ceil(32), round.ordered_rows()).await?;
    }
    let sampled = pass.sampled_rows()?;
    ensure!(sampled.rows() == round.plan.rows.len(), "sampled row count differs from the plan");
    let (device_rows, refused) = admit_device_rows(round, &sampled)?;
    sampled.check_status(&device_rows)?;
    let fallback = round.fallback_rows(&refused);
    let download: Vec<usize> = fallback.iter().copied().chain(round.trace_rows.iter().copied())
        .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let next = if download.is_empty() {
        BatchScores::from_sampled(&sampled, &round.plan.greedy)?
    } else {
        let bytes = pass.download_sampled_rows(&sampled, &download).await?;
        let mut next = sampled.with_full_logits(&download, bytes)?;
        resolve_fallback_rows(&mut next, round, &fallback)?;
        next
    };
    sampling_stats::record_refused(refused.len());
    sampling_stats::record_round(true, device_rows.len(),
        (0..round.plan.rows.len()).filter(|&row| round.plan.planned_fallback(row)).count());
    Ok(next)
}

/// Independent-lane twin of [`execute_sampled_rows`]: the lane already holds
/// the shared request bank behind a `RefCell`, so it uses the cooperative
/// terminal and only downloads the rows the CPU still needs.
async fn execute_shared_sampled_rows<'a, P: VerificationTarget<'a> + ?Sized>(
    pass: &mut P, requests: &std::cell::RefCell<&mut Requests<'a>>,
    batch: &mut RequestBatch, transport: &mut P::Transport, round: &SamplingRound,
) -> Result<BatchScores> {
    let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
    unsafe {
        pass.execute_shared_sampled(requests, batch, transport, 0, &selected, &round.plan.rows,
            Some(&round.arena), VOCAB.div_ceil(32), round.ordered_rows()).await?;
    }
    let sampled = pass.sampled_rows()?;
    ensure!(sampled.rows() == round.plan.rows.len(), "sampled row count differs from the plan");
    let (device_rows, refused) = admit_device_rows(round, &sampled)?;
    sampled.check_status(&device_rows)?;
    let fallback = round.fallback_rows(&refused);
    let download: Vec<usize> = fallback.iter().copied().chain(round.trace_rows.iter().copied())
        .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let next = if download.is_empty() {
        BatchScores::from_sampled(&sampled, &round.plan.greedy)?
    } else {
        let bytes = pass.download_sampled_rows(&sampled, &download).await?;
        let mut next = sampled.with_full_logits(&download, bytes)?;
        resolve_fallback_rows(&mut next, round, &fallback)?;
        next
    };
    sampling_stats::record_refused(refused.len());
    sampling_stats::record_round(true, device_rows.len(),
        (0..round.plan.rows.len()).filter(|&row| round.plan.planned_fallback(row)).count());
    Ok(next)
}

/// Admit the device rows of one launch and record the ones the device refused.
///
/// Returns `(rows to status-check, rows the device refused)`.
///
/// The first is the plan's device rows whose status is `OK` plus the two
/// request-fault statuses; the second is every other device row, which the
/// caller folds into its fallback set and counts, so an INTERNAL from K1, K3/K4
/// or K5 becomes a correct CPU token and a visible fallback rather than a silent
/// bad token.
///
/// A `CpuFallback` row is skipped entirely: its device-facing block is a
/// device-neutral greedy row by construction, so its status says nothing about
/// the request.
pub(crate) fn admit_device_rows(round: &SamplingRound,
    sampled: &crate::v41_target_head::SampledTargetRows,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    for row in 0..sampled.rows() {
        if !round.plan.route[row].device_served() {
            continue;
        }
        match sampled.status[row] {
            ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_OK => admitted.push(row),
            ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_EMPTY_CANDIDATES
            | ds41rt_ffi::DS41RT_V41_SAMPLER_STATUS_NONFINITE_LOGIT => {
                // The request's fault, not a capability miss: the CPU re-sample
                // fails identically, so let `check_status` render the CPU's own
                // message (design D1).
                admitted.push(row);
            }
            status => {
                // INTERNAL, or any future code: the device could not serve a row
                // the host believed servable. Fall back loudly and count it.
                tracing::warn!(target: "ds41rt::sampling", row, status,
                    "device sampler row fell back to the CPU sampler");
                refused.push(row);
            }
        }
    }
    Ok((admitted, refused))
}

async fn execute_logits<'a>(lib: &'a NativeLibrary, pass: &mut TargetPass<'_, 'a>,
    requests: &Requests<'a>, batch: &mut Option<RequestBatch>, transport: &mut NativeTp4Wave<'a>,
    capture_routes: bool, compact: bool,
) -> Result<BatchScores> {
    let Some(batch) = batch else { return BatchScores::new(Vec::new()); };
    pass.set_route_capture(capture_routes);
    let result = async {
        let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
        if compact {
            return BatchScores::from_greedy(unsafe { pass.execute_greedy(requests, batch, transport, 0, &selected).await? });
        }
        let logits = unsafe { pass.execute(requests, batch, transport, 0, &selected).await? };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        BatchScores::new(bytes)
    }.await;
    pass.set_route_capture(false);
    result
}

fn prepare_decode_lane<'a>(requests: &mut Requests<'a>, active: &[Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], speculative: bool) -> Result<RequestBatch> {
    let work: Vec<_> = members.iter().zip(inputs).map(|(&slot, tokens)| RequestTokens {
        lease: active[slot].as_ref().unwrap().lease, tokens, image_mask: None,
        kind: if speculative { ExpertV2SourceKind::MtpVerify } else { ExpertV2SourceKind::Decode },
    }).collect();
    requests.prepare(&work)
}

fn single_lane_round<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    lane: usize, pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    transport: &mut NativeTp4Wave<'a>, active: &mut [Option<Active<'a>>],
    members: &[usize], mut draft: Option<&mut DraftRuntime<'_, 'a>>, retain_enabled: bool,
) -> Result<()> {
    let started = Instant::now();
    ensure!(!members.is_empty() && members.len() <= 8, "invalid single decode lane");
    let speculative = draft.is_some();
    let capture_routes = draft.as_deref().is_some_and(DraftRuntime::capture_routes);
    let seeds = members.iter().map(|&slot| {
        let r = active[slot].as_ref().unwrap();
        Ok((r.id, r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens-r.generated))
    }).collect::<Result<Vec<_>>>()?;
    let draft_start = Instant::now();
    let mut inputs = if let Some(draft) = draft.as_deref_mut() { draft.propose(lib, &seeds)? }
        else { seeds.iter().map(|r| vec![r.1]).collect() };
    for (&slot, input) in members.iter().zip(&mut inputs) {
        let r = active[slot].as_ref().unwrap();
        if let Some(constraint) = &r.constraint { constraint.truncate_proposal(input)?; }
        else if let Some(draft) = draft.as_deref() {
            input.truncate(draft.confidence_prefix(r.id, input.len()-1)? + 1);
        }
    }
    if members.iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none()) {
        if let Some(draft) = draft.as_deref().filter(|d| d.reuse_enabled() || d.adaptive_enabled()) {
            let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                (active[slot].as_ref().unwrap().id, lane, input.len()-1)).collect();
            let lengths = if draft.adaptive_enabled() {
                draft.select_prefixes(&candidates, draft_start.elapsed().as_micros() as u64)?
            } else { draft.select_reuse_prefixes(&candidates)? };
            if let Some(lengths) = lengths {
                for (input, length) in inputs.iter_mut().zip(lengths) { input.truncate(length+1); }
            }
        }
    }
    let draft_us = draft_start.elapsed().as_micros() as u64;
    let prepare_start = Instant::now();
    let mut batch = Some(prepare_decode_lane(requests, active, members, &inputs, speculative)?);
    let prepare_us = prepare_start.elapsed().as_micros() as u64;
    let prepared_us = started.elapsed().as_micros() as u64;
    let compact = !tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG)
        && members.iter().all(|&slot| {
            let request = active[slot].as_ref().unwrap();
            request.constraint.is_none() && request.job.sampling.is_greedy()
        });
    let batch_id = batch.as_ref().unwrap().cache()?.identity();
    if tracing::enabled!(target: "ds41rt::cost_model", tracing::Level::DEBUG) {
        if let Some(draft) = draft.as_deref() {
            let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                (active[slot].as_ref().unwrap().id, lane, input.len()-1)).collect();
            draft.trace_cost_forecast(batch_id, &candidates);
        }
    }
    // Chunk-4a scope: a round is device-selected whenever the layout supports
    // the terminal and at least one row is device-servable. Each row's own
    // parameters then decide its kernels (K1, K1->K2 or K1->K3->K4->K5), and a
    // row the device cannot serve is re-sampled on the CPU inside the same
    // round. Only two shapes still take the whole-round CPU path: the `compact`
    // greedy lane above (which never reaches the sampler), and a round in which
    // **every** row is unservable — there the device launch would select nothing
    // and the rows' logits have to be downloaded anyway.
    //
    // A traced round still takes this path, but `SamplingRound::trace_rows` then
    // downloads every row for `top_two`.
    if !compact && round_has_device_rows(active, members) {
        let round = build_sampling_round(active, members, &inputs,
            tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG))?;
        let next = runtime.block_on(execute_sampled_rows(pass, requests,
            batch.as_mut().unwrap(), transport, &round));
        // Reuse the shared tail below by re-entering the same control flow.
        let next = match next {
            Ok(next) => next,
            Err(error) => {
                if let Some(batch) = &mut batch { pass.discard(batch)?; }
                return Err(error);
            }
        };
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests,
            active, members, &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes, 0,
            Some(&round), retain_enabled)?;
        for (&slot, tokens) in members.iter().zip(emissions) {
            let request = active[slot].as_mut().unwrap();
            if let Err(error) = request.emit(&tokens) {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
            }
        }
        tracing::debug!(target: "ds41rt::timing", speculative,
            requests=members.len(), lane0=if lane == 0 { members.len() } else { 0 },
            lane1=if lane == 1 { members.len() } else { 0 },
            proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
            draft_us, prepare_us, verify_us=started.elapsed().as_micros() as u64 - prepared_us,
            total_us=started.elapsed().as_micros() as u64, "native scheduler round");
        return Ok(());
    }
    // No device terminal ran on this branch, so **no** rows were device-selected:
    // the round is a CPU round with zero device rows, exactly what the counter
    // must say. Counting the rows that merely *would have been* servable would
    // make the export claim device work that never happened.
    sampling_stats::record_round(false, 0, 0);
    let next = runtime.block_on(execute_logits(lib, pass, requests, &mut batch, transport, capture_routes, compact));
    let executed_us = started.elapsed().as_micros() as u64;
    let result = (|| -> Result<()> {
        let next = next?;
        tracing::debug!(target: "ds41rt::cost_model", batch=batch_id, lane,
            requests=members.len(), rows=inputs.iter().map(Vec::len).sum::<usize>(),
            prepared_us, verify_us=executed_us-prepared_us, "verification round cost");
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests,
            active, members, &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes,
            executed_us-prepared_us, None, retain_enabled)?;
        for (&slot, tokens) in members.iter().zip(emissions) {
            let request = active[slot].as_mut().unwrap();
            if let Err(error) = request.emit(&tokens) {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
            }
        }
        tracing::debug!(target: "ds41rt::timing", speculative,
            requests=members.len(), lane0=if lane == 0 { members.len() } else { 0 },
            lane1=if lane == 1 { members.len() } else { 0 },
            proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
            draft_us, prepare_us, verify_us=executed_us-prepared_us,
            total_us=started.elapsed().as_micros() as u64, "native scheduler round");
        Ok(())
    })();
    if result.is_err() {
        // The sole execution future has returned before discarding private state.
        if let Some(batch) = &mut batch { pass.discard(batch)?; }
    }
    result
}

/// Sample one unconstrained stochastic request's verification rows.
///
/// `offset` is the request's anchor row in the flattened batch and
/// `base_position` is its absolute emitted-token index. Draws are keyed on
/// `base_position + index`, never on the batch row, so a member's position in a
/// multi-request round and any rejected draft rows cannot shift its stream.
fn sample_target_rows(
    next: &BatchScores,
    offset: usize,
    input: &[u32],
    params: ds41rt_core::TargetSamplingParams,
    base_position: u64,
) -> Result<Vec<u32>> {
    (0..input.len())
        .map(|index| next.sample(offset + index, None, params, base_position + index as u64))
        .collect()
}

/// How a finishing row's retained frontier is obtained.
///
/// A device-sampled **stochastic** row's `next.best[row]` is a *draw*, not an
/// argmax, so the retention cross-check in
/// [`BatchScores::retain_downloaded_with`] must not run for it: it validates a
/// greedy selection and would raise a spurious "GPU and retained CPU greedy
/// selection differ". The recorded id is the value the verification already
/// consumed, so the retained token is that id with the downloaded row for a
/// later grammar to re-select from. Every other row keeps the cross-check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FrontierRetain {
    Checked,
    RecordedSample,
}

/// How a finishing row's retained frontier is obtained.
///
/// Retention is the **only** consumer of a finishing row's full logits, so the
/// frontier transfer must be scheduled against the retention decision, not
/// against "this round did not download the whole batch" (design §10.4). Chunk
/// 4b additionally distinguishes a row that already carries its own packed
/// logits — a fallback row the CPU re-sampled, whose row was downloaded inside
/// the round — from a row that genuinely needs a transfer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FrontierDownload {
    /// The whole batch already holds its logits (the CPU path or a traced
    /// round): retain in place exactly as before.
    WholeBatch,
    /// This row's own logits are already on the host (a fallback row): retain in
    /// place; no second transfer.
    PackedRow,
    /// Retention is enabled and the row is not on the host: issue the one-row D2H.
    Device,
    /// No retention will consume this row: no transfer, and no retained frontier.
    Skip,
}

/// The retention gate for one finishing frontier row.
///
/// `retain_enabled` is [`PrefixCache::turn_bank_enabled`]: with prefix caching
/// disabled (`prefix_cache_entries == 0`) the finishing request's snapshot is
/// dropped by `retain`/`queue_retain` early-return, so its frontier row is never
/// read. Before chunk 4b that row was still downloaded (`finishing &&
/// !next.has_full_logits()`), a `517,120 B` D2H per finishing request that bought
/// nothing. `row_has_logits` is the **row's** packed state, not the batch's: a
/// finishing fallback row was downloaded so the CPU could re-sample it, and
/// re-downloading it on the independent lane was wasted.
///
/// `events_open` mirrors the one case in which `finishing` does not become
/// `cacheable`: `Active::emit_one` refuses a closed client before it can set the
/// flag, so a disconnected request must not pay a retention transfer either.
pub(crate) fn frontier_download(whole_batch_has_logits: bool, row_has_logits: bool,
    retain_enabled: bool, finishing: bool, events_open: bool,
) -> FrontierDownload {
    if !finishing || !events_open {
        return FrontierDownload::Skip;
    }
    if whole_batch_has_logits {
        return FrontierDownload::WholeBatch;
    }
    if row_has_logits {
        return FrontierDownload::PackedRow;
    }
    if retain_enabled {
        FrontierDownload::Device
    } else {
        FrontierDownload::Skip
    }
}

/// Retain a frontier row whose bytes are already packed into `next`, applying
/// the same [`FrontierRetain`] rule as the downloaded path.
///
/// A packed row in a **device** round is a fallback row the CPU re-sampled, so
/// its `best` is a draw (stochastic) or the masked CPU argmax (greedy) and the
/// cross-check is meaningful. This is the no-transfer twin of
/// [`resolve_frontier`], used by both the serial and the independent lane so the
/// two cannot drift. (`WholeBatch` keeps its own `next.retain` path: a fully
/// downloaded batch is CPU-selected, where the stored `best` is the unmasked
/// argmax and the mask cross-check would be spurious.)
pub(crate) fn retain_packed_frontier(next: &BatchScores, row: usize, retain: FrontierRetain,
    mask: Option<&[u32]>,
) -> Result<TokenScores> {
    match retain {
        FrontierRetain::Checked => next.retain_packed(row, mask),
        FrontierRetain::RecordedSample => next.retain(row),
    }
}

/// Which retention a finishing frontier row takes.
///
/// **In a device round, `best[row]` of a non-greedy row is always a draw**,
/// whichever producer selected it: K2 for a fast-path row, K5 for an ordered
/// row, or the CPU re-sample for a fallback row (planned, or refused after the
/// launch). It is therefore classified on the parameters alone — keying on
/// `route[frontier].device_served()` is wrong, because a planned-fallback row is
/// **not** device-served while its stored `best` is a draw, so the cross-check
/// would compare an argmax against a draw and fail the whole lane with "GPU and
/// retained CPU greedy selection differ".
///
/// A greedy row's `best` is an argmax whichever path produced it, so it keeps
/// [`FrontierRetain::Checked`] — including a greedy fallback row, whose stored
/// value is the argmax the CPU side computed. Outside a device round the
/// classifier is not reached: a whole-CPU round has every row's logits, so
/// `next.has_full_logits()` takes the trusting `retain` path instead.
pub(crate) fn frontier_retain(round: Option<&SamplingRound>,
    params: ds41rt_core::TargetSamplingParams,
) -> FrontierRetain {
    if round.is_some() && !params.is_greedy() {
        FrontierRetain::RecordedSample
    } else {
        FrontierRetain::Checked
    }
}

struct CommitDecision {
    accepted_drafts: u32,
    emitted: usize,
    accepted: Vec<u32>,
    emissions: Vec<Vec<u32>>,
    next_after_commit: Vec<Option<TokenScores>>,
    /// Finishing frontier rows that need the one-row device D2H: retention is
    /// enabled and the row is not already on the host.
    frontier_downloads: Vec<(usize, usize, Option<Vec<u32>>, FrontierRetain)>,
    /// Finishing frontier rows whose bytes are already packed into `next`
    /// (fallback rows in a device round): retained without a transfer.
    frontier_packed: Vec<(usize, usize, Option<Vec<u32>>, FrontierRetain)>,
}
/// The per-row target selection of a round.
///
/// `DeviceSelected` means K1 produced the id for every row (masked or not), so
/// the commit path must **consume** those ids: re-deriving them from logits
/// would need the full rows the device path deliberately did not download.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleSource {
    DeviceSelected,
    CpuRecomputed,
}

/// The selected token for one request's rows.
///
/// For `DeviceSelected` the ids are exactly `next.best[offset..offset+len]`,
/// which is the slice K1 filled: the device applies the grammar mask itself, so
/// a constrained row consumes its masked id instead of re-running the masked
/// argmax (which would fail, because no logits were downloaded).
fn selected_target_rows(next: &BatchScores, offset: usize, len: usize,
    source: SampleSource,
) -> Result<std::borrow::Cow<'_, [u32]>> {
    use std::borrow::Cow;
    ensure!(
        offset + len <= next.best.len(),
        "target selection rows are outside the batch"
    );
    Ok(match source {
        SampleSource::DeviceSelected => Cow::Borrowed(&next.best[offset..offset + len]),
        SampleSource::CpuRecomputed => Cow::Owned(next.best[offset..offset + len].to_vec()),
    })
}

/// Chunk-4a's per-row twin of [`select_target_row`].
///
/// A mixed round has no single source: a row the device served is consumed from
/// `next.best`, while a row the device could not serve carries its full logits
/// and is re-sampled from the request's own parameters. `round` decides that per
/// row (`planned_fallback(row)`), and `None` means the whole round was
/// CPU-selected. The closure is consulted only when the row has no stored
/// selection at all, so the device-selection property stays testable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_routed<'a, F>(next: &'a BatchScores, round: Option<&SamplingRound>,
    offset: usize, row: usize, len: usize, recompute: F,
) -> Result<std::borrow::Cow<'a, [u32]>>
where
    F: FnOnce() -> Result<Vec<u32>>,
{
    let stored = match round {
        // A row whose selection is already stored is consumed as-is, whichever
        // producer stored it: a device row's `best[row]` came from the kernels,
        // and a fallback row's was written by `BatchScores::store_sampled` after
        // the launch. Keying on "does this row carry its logits" rather than on
        // the plan route is what makes a row the device **refused after the
        // launch** commit the stored CPU token instead of the stale device slot
        // (a failed kernel writes no id), and it means a fallback row is never
        // sampled a second time by the closure below.
        Some(round) => round.plan.route[row].device_served() || next.has_row_logits(row),
        // No round: the whole batch was CPU-produced, so every row is stored and
        // the closure is dead. Kept for the existing chunk-1 tests, which call
        // `select_routed` with `None` to assert the device path cannot reach the
        // closure.
        None => false,
    };
    let source = if stored { SampleSource::DeviceSelected } else { SampleSource::CpuRecomputed };
    select_target_row(next, offset, len, source, recompute)
}

/// One request's target selection.
///
/// `recompute` is the CPU selection (masked argmax, target sample, or cached
/// id). It is consulted **only** for [`SampleSource::CpuRecomputed`]; a
/// device-selected round must never call it, because that path needs full
/// logits the device path deliberately did not download. Keeping the two
/// separate is what makes that property testable rather than incidental.
fn select_target_row<F>(next: &BatchScores, offset: usize, len: usize, source: SampleSource,
    recompute: F,
) -> Result<std::borrow::Cow<'_, [u32]>>
where
    F: FnOnce() -> Result<Vec<u32>>,
{
    match source {
        SampleSource::DeviceSelected => selected_target_rows(next, offset, len, source),
        SampleSource::CpuRecomputed => Ok(std::borrow::Cow::Owned(recompute()?)),
    }
}

fn prepare_commit_lane<'a, C: DraftChain<'a>>(lane: usize,
    requests: &Requests<'a>, active: &[Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    next: &BatchScores, draft: Option<&DraftRuntime<'_, 'a, C>>, verify_us: u64,
    round: Option<&SamplingRound>, retain_enabled: bool,
) -> Result<CommitDecision> {
    let mut accepted_drafts = 0u32;
    let mut emitted = 0usize;
    let mut offset = 0;
    let mut accepted = Vec::new();
    let mut emissions = Vec::new();
    let mut next_after_commit = Vec::new();
    let mut frontier_downloads = Vec::new();
    let mut frontier_packed = Vec::new();
    for (&slot, input) in members.iter().zip(inputs) {
        let request = active[slot].as_ref().unwrap();
        let params = request.job.sampling;
        let base_position = request.generated as u64;
        // Stochastic requests select a target sample per row. Reusing the same
        // greedy verifier turns "draft equals target argmax" into the exact
        // sample-and-match rule: a draft is accepted while it equals the target
        // sample, and the first mismatch emits the target sample. The emitted
        // token is always the target draw, so speculation cannot bias p.
        //
        // Chunk 4a: the source is per **row**, not per round. A device row (any
        // route, greedy or stochastic) consumes the id the device published; a
        // fallback row consumes the id `resolve_fallback_rows` re-sampled from
        // its downloaded logits. The closure below is consulted only for a row
        // with no device route at all (a whole-round CPU path), so the
        // device-selection property stays testable rather than incidental.
        let selected = select_routed(next, round, offset, offset, input.len(), || {
            let state = request.constraint.as_ref();
            if params.is_greedy() {
                match state {
                    Some(state) => state.select_verification(next, offset, input),
                    None => Ok(next.best[offset..offset + input.len()].to_vec()),
                }
            } else {
                match state {
                    Some(state) => {
                        state.select_verification_sampled(next, offset, input, params, base_position)
                    }
                    None => sample_target_rows(next, offset, input, params, base_position),
                }
            }
        })?;
        let selected: &[u32] = &selected;
        let decision = ds41rt_core::verify_dspark_greedy(input,
            selected, 1, request.job.max_tokens - request.generated)
            .map_err(anyhow::Error::msg)?;
        if tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG) {
            let top_two = (offset..offset + input.len())
                .map(|row| next.top_two(row)).collect::<Result<Vec<_>>>()?;
            tracing::debug!(target: "ds41rt::logit_trace",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                input=?input, selected=?selected, top_two=?top_two,
                accepted_inputs=decision.accepted_inputs,
                emitted=?decision.emitted,
                constrained=request.constraint.is_some(),
                "native verification logits");
        }
        if let Some(confidence) = draft
            .and_then(|draft| draft.confidence_trace(request.id)) {
            // Agreement after the first mismatch is conditional on a
            // rejected history and must not be treated as acceptance.
            let matched = input.iter().skip(1).zip(selected.iter())
                .take_while(|(proposal, target)| proposal == target).count();
            tracing::debug!(target: "ds41rt::draft_policy",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                verifier_rows=input.len(), lane_rows=inputs.iter().map(Vec::len).sum::<usize>(),
                constrained=request.constraint.is_some(), raw_confidence=?confidence,
                matched_prefix=matched, accepted_inputs=decision.accepted_inputs,
                eos=decision.eos, length_limit=decision.length_limit,
                verify_us,
                "native draft policy observation");
        }
        accepted_drafts += decision.accepted_inputs - 1;
        emitted += decision.emitted.len();
        let finishing = decision.emitted.contains(&1)
            || request.generated + decision.emitted.len() >= request.job.max_tokens;
        let frontier = offset + decision.accepted_inputs as usize - 1;
        // Chunk-4b retention gate. Retention is the only reader of a finishing
        // frontier row, so the one-row D2H is scheduled only when a snapshot will
        // actually store it. `whole_batch_has_logits` (a CPU round or a traced
        // round) keeps the pre-existing in-place retain exactly; a fallback row
        // whose bytes were already downloaded is retained from its own packed
        // slice; and a device row in a cache-disabled deployment is skipped
        // entirely instead of paying a 517,120 B transfer per finishing request.
        let action = frontier_download(next.has_full_logits(),
            next.has_row_logits(frontier), retain_enabled, finishing,
            !request.job.events.is_closed());
        let member = next_after_commit.len();
        match action {
            FrontierDownload::Skip => {
                if finishing && !next.has_full_logits() && !next.has_row_logits(frontier) {
                    // The pre-chunk-4b code downloaded this row here; count the
                    // transfer the gate removed.
                    sampling_stats::record_frontier_gated(1, crate::v41_native_serve::scores::ROW_BYTES);
                }
                next_after_commit.push(None);
            }
            FrontierDownload::WholeBatch => next_after_commit.push(Some(next.retain(frontier)?)),
            FrontierDownload::PackedRow | FrontierDownload::Device => {
                // The frontier belongs to this member's own hypothetical
                // prefix, so a constrained row retains against its grammar
                // mask rather than the plain argmax.
                let row_in_round = frontier - offset;
                let mask = match request.constraint.as_ref() {
                    Some(state) => state.prepare_verification_mask_row(input, row_in_round)?,
                    None => None,
                };
                // A non-greedy frontier in a device round is a *draw* from
                // whichever producer selected it, so its retention must not run
                // the greedy argmax cross-check; a greedy row keeps it. See
                // `frontier_retain` for why the route is deliberately not part of
                // the decision.
                let retain = frontier_retain(round, params);
                if action == FrontierDownload::Device {
                    sampling_stats::record_frontier_download(1);
                    frontier_downloads.push((member, frontier, mask, retain));
                } else {
                    frontier_packed.push((member, frontier, mask, retain));
                }
                next_after_commit.push(None);
            }
        }
        offset += input.len(); accepted.push(decision.accepted_inputs); emissions.push(decision.emitted);
    }
    Ok(CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit,
        frontier_downloads, frontier_packed })
}
fn publish_commit_lane<'a, C: DraftChain<'a>>(pass: &impl VerificationTarget<'a>, active: &mut [Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], owned_batch: &mut Option<RequestBatch>,
    mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, capture_routes: bool, decision: CommitDecision,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit, frontier_downloads, frontier_packed } = decision;
    ensure!(frontier_downloads.is_empty(), "retained frontier downloads are incomplete");
    ensure!(frontier_packed.is_empty(), "retained packed frontier is incomplete");
    if let Some(draft) = draft.as_deref_mut() {
        if capture_routes {
            let mut offset = 0;
            for ((&slot, input), &count) in members.iter().zip(inputs).zip(&accepted) {
                draft.observe_accepted_routes(active[slot].as_ref().unwrap().id, offset,
                    count as usize, pass.captured_routes())?;
                offset += input.len();
            }
        }
    }
    *owned_batch = None;
    for (&slot, next_token) in members.iter().zip(next_after_commit) {
        active[slot].as_mut().unwrap().next_after_commit = next_token;
    }
    Ok((accepted_drafts, emitted, emissions))
}
/// Resolve one finishing row's retained frontier from its downloaded bytes.
///
/// This is the single place that decides between the two retention shapes, so
/// the single-lane and independent-lane paths cannot drift:
///
/// * [`FrontierRetain::Checked`] — the recorded id is a greedy selection, so it
///   is validated against the row's own grammar mask by
///   [`BatchScores::retain_downloaded_with`] (the chunk-1 property: a device/CPU
///   greedy disagreement is an error, not a silent token).
/// * [`FrontierRetain::RecordedSample`] — the recorded id is a stochastic draw
///   produced by the same path, so the row is retained as-is
///   ([`BatchScores::retain`]); running the argmax cross-check would be checking
///   a draw for being an argmax.
pub(crate) fn resolve_frontier(retain: FrontierRetain, next: &BatchScores, row: usize,
    bytes: &[u8],
    mask: Option<&[u32]>,
) -> Result<TokenScores> {
    match retain {
        FrontierRetain::Checked => next.retain_downloaded_with(row, bytes, mask),
        FrontierRetain::RecordedSample => next.retain_from_bytes(row, bytes),
    }
}

#[allow(clippy::too_many_arguments)]
fn commit_lane<'w, 'a>(lib: &'a NativeLibrary, lane: usize,
    pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    active: &mut [Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    owned_batch: &mut Option<RequestBatch>, next: &BatchScores,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>, capture_routes: bool, verify_us: u64,
    round: Option<&SamplingRound>, retain_enabled: bool,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let Some(batch) = owned_batch else { return Ok((0, 0, Vec::new())); };
    let mut decision = prepare_commit_lane(lane, requests, active, members, inputs,
        next, draft.as_deref(), verify_us, round, retain_enabled)?;
    // A frontier row whose bytes are already packed (a fallback row the CPU
    // re-sampled) is retained in place: no second transfer.
    let packed: Vec<(usize, usize, Option<Vec<u32>>, FrontierRetain)> =
        std::mem::take(&mut decision.frontier_packed);
    // Serial lane: the pre-4b code reached these rows through
    // `retain_from_device`, which short-circuits on packed bytes, so no transfer
    // is saved here -- only the independent lane's `download_logits` paid one.
    sampling_stats::record_frontier_packed(packed.len(),
        crate::v41_native_serve::scores::ROW_BYTES, false);
    for (member, row, mask, retain) in packed {
        decision.next_after_commit[member] =
            Some(retain_packed_frontier(next, row, retain, mask.as_deref())?);
    }
    // Each remaining finishing frontier row is downloaded by itself through the
    // head's existing per-row D2H path (the same one the greedy device path used
    // in chunk 1), because a finishing row is rare and it is the only route that
    // reuses the batch's own logits view without a second head pass.
    let frontier: Vec<(usize, usize, Option<Vec<u32>>, FrontierRetain)> =
        std::mem::take(&mut decision.frontier_downloads);
    for (member, row, mask, retain) in frontier {
        let logits = pass.output(batch)?.logits;
        decision.next_after_commit[member] = Some(match retain {
            FrontierRetain::Checked =>
                next.retain_from_device(lib, logits, row, mask.as_deref())?,
            FrontierRetain::RecordedSample =>
                next.retain_recorded_from_device(lib, logits, row)?,
        });
    }
    if let Some(draft) = draft.as_deref_mut() { draft.commit_batch(pass, requests, batch, &decision.accepted)?; }
    else { pass.commit(requests, batch, &decision.accepted)?; }
    publish_commit_lane(pass, active, members, inputs, owned_batch, draft, capture_routes, decision)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    use crate::v41_native_serve::scores::{ROW_BYTES, VOCAB};

    /// Position-sensitive logits so different absolute indices and rows differ.
    fn rows(count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * ROW_BYTES);
        for row in 0..count {
            for token in 0..VOCAB {
                let value = (((row * 31 + token * 7) % 23) as f32) * 0.25;
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        bytes
    }

    /// Two members in one round: the anchor row offsets differ from the
    /// generated indices, and every draw must follow the absolute emitted
    /// position rather than the batch row.
    #[test]
    fn sample_target_rows_keys_draws_on_absolute_position() {
        let batch = BatchScores::new(rows(5)).unwrap();
        let params =
            ds41rt_core::TargetSamplingParams::new(0.9, 0.97, Some(8), 0.02, 4242).unwrap();
        assert!(!params.is_greedy());

        // Member A: anchor at batch row 0, generated index 0.
        let a = sample_target_rows(&batch, 0, &[0, 0], params, 0).unwrap();
        // Member B: anchor at batch row 2, generated index 100.
        let b = sample_target_rows(&batch, 2, &[0, 0, 0], params, 100).unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 3);
        assert_eq!(a[0], batch.sample(0, None, params, 0).unwrap());
        assert_eq!(a[1], batch.sample(1, None, params, 1).unwrap());
        assert_eq!(b[0], batch.sample(2, None, params, 100).unwrap());
        assert_eq!(b[1], batch.sample(3, None, params, 101).unwrap());
        assert_eq!(b[2], batch.sample(4, None, params, 102).unwrap());

        // Sensitivity: keying member B on the row index (the classic wiring
        // bug) must be distinguishable from keying it on the emitted index.
        let row_keyed: Vec<u32> = (0..3)
            .map(|index| batch.sample(2 + index, None, params, (2 + index) as u64).unwrap())
            .collect();
        assert_ne!(b, row_keyed, "offset-keyed draws must not match emitted-index draws");
    }

    /// P0-A seam: a device-selected round consumes K1's ids for **every** row,
    /// constrained or not, and never reaches the CPU recompute. Re-deriving a
    /// masked row needs logits the device path deliberately did not download, so
    /// this is what keeps a constrained greedy round from failing at commit.
    #[test]
    fn device_selected_rows_consume_ids_without_downloading_logits() {
        // Six greedy rows: rows 1 and 4 are constrained, the rest are not.
        let greedy = vec![true; 6];
        let ids: Vec<u32> = vec![5, 3, 7, 1, 4, 2];
        let scores: Vec<f32> = vec![1.0; 6];
        let logits = ds41rt_ffi::Ds41rtDeviceBuffer {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            ..Default::default()
        };
        let sampled = crate::v41_target_head::SampledTargetRows {
            ids: ids.clone(),
            scores,
            status: vec![0u32; 6],
            status_detail: vec![0u32; 6],
            logits,
        };
        let next = BatchScores::from_sampled(&sampled, &greedy).unwrap();

        // The device path returns exactly K1's ids and must not invoke the CPU
        // selection, whose failure is the "requires full logits" error below.
        let selected = select_target_row(&next, 0, 6, SampleSource::DeviceSelected, || {
            Err(anyhow::anyhow!("the device path must not recompute a row"))
        })
        .unwrap();
        assert_eq!(&selected[..], &ids[..]);
        // Borrowed, not copied: the ids already live in the batch.
        assert!(matches!(selected, std::borrow::Cow::Borrowed(_)));

        // A batch built from device ids has no row bytes, so the CPU recompute
        // the pre-fix commit code used is exactly what fails here.
        let err = next.select(1, Some(&[0xFFFF_FFFFu32])).unwrap_err();
        assert!(err.to_string().contains("requires full logits"), "unexpected error: {err}");

        // The same seam still serves the CPU path when it is the source.
        let cpu = select_target_row(&next, 0, 6, SampleSource::CpuRecomputed, || Ok(vec![42; 6]))
            .unwrap();
        assert_eq!(&cpu[..], &[42u32; 6]);

        // Sensitivity: the CPU closure must not be called for a device round.
        let mut called = false;
        let _ = select_target_row(&next, 0, 1, SampleSource::DeviceSelected, || {
            called = true;
            Ok(vec![0])
        });
        assert!(!called, "device-selected rows must not consult the CPU recompute");
    }

    /// The chunk-4a routing table: one row per kernel class, asserted by the
    /// production router rather than by prose. Each expected route names the
    /// kernels the row takes (design §4.6 and the kernel eligibility blocks in
    /// `v41_sampling_gpu.cu`).
    #[test]
    fn the_per_row_router_selects_the_documented_kernel_class() {
        use ds41rt_core::TargetSamplingParams;
        // The four required stochastic profiles, plus greedy, top_k == 1 and
        // the unservable band.
        let cases: Vec<(&str, TargetSamplingParams, SamplingRoute)> = vec![
            ("greedy (no temperature)", TargetSamplingParams::greedy(), SamplingRoute::DeviceGreedy),
            ("top_k == 1 is greedy", TargetSamplingParams::new(0.9, 1.0, Some(1), 0.0, 1).unwrap(),
                SamplingRoute::DeviceGreedy),
            ("temperature only", TargetSamplingParams::new(0.7, 1.0, None, 0.0, 1).unwrap(),
                SamplingRoute::DeviceFastPath),
            ("temperature 0.2 + top_p 0.95",
                TargetSamplingParams::new(0.2, 0.95, None, 0.0, 1).unwrap(),
                SamplingRoute::DeviceOrdered),
            ("temperature 0.7 + top_p 0.9",
                TargetSamplingParams::new(0.7, 0.9, None, 0.0, 1).unwrap(),
                SamplingRoute::DeviceOrdered),
            ("temperature 0.7 + min_p 0.05",
                TargetSamplingParams::new(0.7, 1.0, None, 0.05, 1).unwrap(),
                SamplingRoute::DeviceFastPath),
            ("temperature 0.7 + top_k 40",
                TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 1).unwrap(),
                SamplingRoute::DeviceOrdered),
            ("top_k at K5's kBlock limit",
                TargetSamplingParams::new(0.7, 1.0, Some(256), 0.0, 1).unwrap(),
                SamplingRoute::DeviceOrdered),
            ("top_k above K5's retained table",
                TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 1).unwrap(),
                SamplingRoute::CpuFallback),
            ("top_k above the retained table with top_p < 1",
                TargetSamplingParams::new(0.7, 0.9, Some(300), 0.0, 1).unwrap(),
                SamplingRoute::CpuFallback),
        ];
        for (name, params, expected) in &cases {
            assert_eq!(sampling_route(*params), *expected, "{name}");
        }
        // A row the router calls device-served must be one the FFI validator
        // would accept, because the whole batch is rejected otherwise: every
        // device route moves `top_k` into the retained-table range and leaves the
        // ranges alone.
        for (name, params, route) in &cases {
            if !route.device_served() {
                continue;
            }
            let top_k = params.top_k().map_or(0, |k| k as u32);
            assert!(top_k <= crate::v41_target_head::DS41RT_V41_SAMPLING_MAX_RETAINED,
                "{name} must not exceed K5's retained table");
            assert!(params.temperature().is_finite() && params.temperature() <= 2.0, "{name}");
            assert!(params.top_p().is_finite() && params.top_p() > 0.0 && params.top_p() <= 1.0,
                "{name}");
            assert!(params.min_p().is_finite() && params.min_p() <= 1.0, "{name}");
        }
    }

    /// A mixed round is routed per **row**, not per round: greedy, fast-path,
    /// ordered and fallback rows coexist, and the plan's device-facing blocks
    /// carry each row's own parameters.
    #[test]
    fn a_mixed_round_routes_every_row_by_its_own_parameters() {
        use ds41rt_core::TargetSamplingParams;
        let greedy = TargetSamplingParams::greedy().with_seed(1);
        let fast = TargetSamplingParams::new(0.7, 1.0, None, 0.05, 2).unwrap();
        let ordered = TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 3).unwrap();
        let fallback = TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 4).unwrap();
        let mut mask = vec![0u32; VOCAB.div_ceil(32)];
        mask[0] = 0b1111;
        let members = vec![
            SamplingMember { params: greedy, base_position: 100, row_masks: vec![None, None] },
            SamplingMember { params: fast, base_position: 0, row_masks: vec![None] },
            SamplingMember { params: ordered, base_position: 7, row_masks: vec![None, None, None] },
            SamplingMember { params: fallback, base_position: 50, row_masks: vec![None] },
        ];
        let inputs = vec![vec![0, 0], vec![0], vec![0, 0, 0], vec![0]];
        let plan = build_target_sampling_plan(&members, &inputs).unwrap();
        assert_eq!(plan.route, vec![
            SamplingRoute::DeviceGreedy, SamplingRoute::DeviceGreedy,
            SamplingRoute::DeviceFastPath,
            SamplingRoute::DeviceOrdered, SamplingRoute::DeviceOrdered, SamplingRoute::DeviceOrdered,
            SamplingRoute::CpuFallback,
        ]);
        // Per-row parameters travel with their own row, at their own absolute
        // position, and a fallback row's device-facing block is device-neutral.
        let positions: Vec<u64> = plan.rows.iter().map(|r| r.row.position).collect();
        assert_eq!(positions, vec![100, 101, 0, 7, 8, 9, 50]);
        assert_eq!(plan.rows[4].row.top_k, 40);
        assert_eq!(plan.rows[4].row.temperature, 0.7);
        assert_eq!(plan.rows[6].row.top_k, 0, "a fallback row's device block is a no-op");
        assert_eq!(plan.rows[6].row.temperature, 0.0);
        assert_eq!(plan.rows[6].row.mask_row, ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW);
        assert_ne!(plan.rows[6].row.flags & ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_GREEDY, 0,
            "the fallback block is a greedy no-op");
        // The request's own values survive in the plan for the CPU re-sample.
        assert_eq!(plan.params[6].top_k(), Some(300));
        assert_eq!(plan.position[6], 50);
        // The route decides the device requirement, and the fallback row's
        // route is the only one that is not device-served.
        assert_eq!(plan.route.iter().filter(|route| route.device_served()).count(), 6);
        assert!(plan.routes_need_ordered_tail());
    }

    /// A round where every row is unservable stays on the CPU path; a round with
    /// one servable row uses the device terminal and falls back only the
    /// unservable row. This is the per-row policy, stated as a property.
    #[test]
    fn a_single_unservable_row_falls_back_per_row_not_per_round() {
        use ds41rt_core::TargetSamplingParams;
        let unservable = TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 1).unwrap();
        let servable = TargetSamplingParams::new(0.7, 0.9, None, 0.0, 2).unwrap();
        let members = |params: ds41rt_core::TargetSamplingParams| SamplingMember {
            params, base_position: 0, row_masks: vec![None],
        };
        let all = build_target_sampling_plan(&[members(unservable)], &[vec![0]]).unwrap();
        assert!(!all.routes_need_device_terminal(),
            "an all-fallback round must keep the CPU path");
        let mixed = build_target_sampling_plan(
            &[members(unservable), members(servable)], &[vec![0], vec![0]]).unwrap();
        assert!(mixed.routes_need_device_terminal(),
            "one servable row is enough to use the device terminal");
        assert_eq!(mixed.planned_fallback_count(), 1);
    }

    /// The fallback row is re-sampled with the request's own parameters and the
    /// row's own mask and absolute position -- not with the device-facing
    /// substitution, and not at a batch-relative position.
    #[test]
    fn a_fallback_row_is_re_sampled_from_its_own_values() {
        use ds41rt_core::TargetSamplingParams;
        let params = TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 4242).unwrap();
        let member = SamplingMember { params, base_position: 900, row_masks: vec![None] };
        let plan = build_target_sampling_plan(&[member], &[vec![0]]).unwrap();
        assert!(plan.planned_fallback(0));
        assert_eq!(plan.params[0].top_k(), Some(300));
        assert_eq!(plan.params[0].seed(), 4242);
        assert_eq!(plan.position[0], 900);
        assert_eq!(plan.mask[0], None);

        // The CPU value the fallback must reproduce, over a real logit row.
        let mut logits = vec![-6.0f32; VOCAB];
        logits[5] = 4.0;
        logits[17] = 3.5;
        let expected = params.select_token(&logits, None, 900).unwrap() as u32;
        let batch = BatchScores::new(logits.iter().flat_map(|value| value.to_ne_bytes()).collect())
            .unwrap();
        assert_eq!(batch.sample(0, None, plan.params[0], plan.position[0]).unwrap(), expected);
        // The position decides the draw: the seeded uniform at the row's own
        // absolute position must differ from the batch-relative one, which is
        // the wiring this pins.
        assert_ne!(params.random_uniform(900).to_bits(), params.random_uniform(0).to_bits(),
            "the fallback draw must be keyed on the row's absolute position");
        assert_eq!(batch.sample(0, None, plan.params[0], 0).unwrap(),
            params.select_token(&logits, None, 0).unwrap() as u32,
            "a batch-relative draw is a different value");
    }

    /// The fallback counters are exported, monotonic and cumulative, so an
    /// operator can see a fallback that no log line happened to catch.
    #[test]
    fn fallback_instrumentation_is_exported() {
        let before = sampling_stats::snapshot();
        sampling_stats::record_round(true, 3, 1);
        sampling_stats::record_refused(2);
        let after = sampling_stats::snapshot();
        let field = |value: &serde_json::Value, name: &str| value[name].as_u64().unwrap();
        assert_eq!(field(&after, "device_rounds"), field(&before, "device_rounds") + 1);
        assert_eq!(field(&after, "device_rows"), field(&before, "device_rows") + 3);
        assert_eq!(field(&after, "planned_fallback_rows"),
            field(&before, "planned_fallback_rows") + 1);
        assert_eq!(field(&after, "refused_fallback_rows"),
            field(&before, "refused_fallback_rows") + 2);
        assert_eq!(field(&after, "cpu_fallback_rounds"), field(&before, "cpu_fallback_rounds"));
    }

    /// Chunk-4b retention gate: a finishing frontier is downloaded **only** when
    /// a snapshot will consume it, and a row already packed into the batch is
    /// never transferred twice.
    ///
    /// The table is the exact policy `prepare_commit_lane` applies; each row is a
    /// reachable production shape (whole-CPU round, cache-disabled device round,
    /// cache-enabled device round, fallback row with packed bytes, disconnected
    /// finishing client).
    #[test]
    fn frontier_transfer_is_gated_on_retention_and_per_row_logits() {
        // (whole_batch, row_packed, retain_enabled, finishing, events_open)
        let cases: [((bool, bool, bool, bool, bool), FrontierDownload); 8] = [
            // Whole-CPU or fully traced round: retain in place, no transfer.
            ((true, true, false, true, true), FrontierDownload::WholeBatch),
            ((true, true, true, true, true), FrontierDownload::WholeBatch),
            // Device round, cache enabled, ordinary finishing row: one-row D2H.
            ((false, false, true, true, true), FrontierDownload::Device),
            // Device round, cache **disabled**: the pre-4b unconditional transfer
            // is gone; `retain` would have early-returned.
            ((false, false, false, true, true), FrontierDownload::Skip),
            // A fallback row already packed by the round's own download: reused,
            // never transferred a second time (the independent lane's old waste).
            ((false, true, true, true, true), FrontierDownload::PackedRow),
            ((false, true, false, true, true), FrontierDownload::PackedRow),
            // Not finishing: nothing to retain.
            ((false, false, true, false, true), FrontierDownload::Skip),
            // A finishing client that has disconnected never becomes cacheable
            // (`Active::emit_one` refuses before it can set the flag).
            ((false, false, true, true, false), FrontierDownload::Skip),
        ];
        for ((whole, packed, enabled, finishing, open), expected) in cases {
            assert_eq!(frontier_download(whole, packed, enabled, finishing, open), expected,
                "whole={whole} packed={packed} enabled={enabled} finishing={finishing} open={open}");
        }
    }

    /// The turn-bank capability answers exactly what `retain`/`queue_retain` use
    /// (`bank.limit() > 0`), so a zero-entry cache disables the frontier transfer
    /// and a configured one keeps it.
    #[test]
    fn turn_bank_enabled_mirrors_the_retention_limit() {
        assert!(!PrefixCache::new(0).turn_bank_enabled(), "limit 0 disables retention");
        assert!(PrefixCache::new(1).turn_bank_enabled());
        assert!(PrefixCache::new(32).turn_bank_enabled());
    }

    /// The bytes the gate removes are exported, so the saving is observable and
    /// not just a code-path claim. The gate removal and the independent lane's
    /// packed-row removal are counted separately (review FIX 2).
    ///
    /// Fragility note (delta review): `sampling_stats` counters are process-global
    /// and this test uses before/after deltas, so it is deterministic only while
    /// no other test running in parallel touches the frontier counters. Today no
    /// CPU test calls `prepare_commit_lane`/`record_frontier_*`, so it is safe; a
    /// future parallel test that does would make it flaky.
    #[test]
    fn frontier_gate_instrumentation_is_exported_in_bytes() {
        let before = sampling_stats::snapshot();
        sampling_stats::record_frontier_gated(2, ROW_BYTES);
        sampling_stats::record_frontier_download(1);
        // Serial: a reuse only. Independent: the same row was a real transfer.
        sampling_stats::record_frontier_packed(1, ROW_BYTES, false);
        sampling_stats::record_frontier_packed(3, ROW_BYTES, true);
        let after = sampling_stats::snapshot();
        let field = |value: &serde_json::Value, name: &str| value[name].as_u64().unwrap();
        assert_eq!(field(&after, "frontier_gated_rows"),
            field(&before, "frontier_gated_rows") + 2);
        assert_eq!(field(&after, "frontier_gated_bytes"),
            field(&before, "frontier_gated_bytes") + 2 * ROW_BYTES as u64);
        assert_eq!(field(&after, "frontier_download_rows"),
            field(&before, "frontier_download_rows") + 1);
        assert_eq!(field(&after, "frontier_packed_rows"),
            field(&before, "frontier_packed_rows") + 4);
        assert_eq!(field(&after, "frontier_packed_saved_bytes"),
            field(&before, "frontier_packed_saved_bytes") + 3 * ROW_BYTES as u64,
            "only the independent lane's packed rows were transfers before 4b");
    }

    /// Review FIX 1: `target_sampling` must be published even with **no** host
    /// cache, because the frontier counters exist for exactly that deployment.
    ///
    /// Scope caveat (delta review): this pins the payload built **inside**
    /// [`serving_stats`], for both a disabled and a configured bank. It does not
    /// run the `serve` loop, so it cannot prove the call site at
    /// `scheduler.rs:206` will never be re-gated; that would need a running
    /// worker.
    #[test]
    fn serving_stats_publish_sampling_counters_without_a_host_cache() {
        const COUNTERS: [&str; 10] = ["device_rounds", "cpu_fallback_rounds", "device_rows",
            "planned_fallback_rows", "refused_fallback_rows", "frontier_download_rows",
            "frontier_gated_rows", "frontier_gated_bytes", "frontier_packed_rows",
            "frontier_packed_saved_bytes"];
        // `PrefixCache::new(0)` has no host cache and a disabled turn bank.
        let disabled = PrefixCache::new(0);
        assert!(disabled.host_metrics().is_none());
        assert!(!disabled.turn_bank_enabled());
        let snapshot = serving_stats(&disabled);
        assert!(snapshot["host_cache"].is_null(), "{snapshot}");
        assert!(snapshot["host_cache_config"].is_null(), "{snapshot}");
        for key in COUNTERS {
            assert!(snapshot["target_sampling"][key].is_u64(), "missing {key} in {snapshot}");
        }
        // A configured bank keeps the **full** key set (the host keys stay null
        // here, because this fixture has no host cache attached either).
        let configured = serving_stats(&PrefixCache::new(8));
        assert!(configured["host_cache"].is_null(), "{configured}");
        assert!(configured["host_cache_config"].is_null(), "{configured}");
        for key in COUNTERS {
            assert!(configured["target_sampling"][key].is_u64(),
                "missing {key} in {configured}");
        }
    }

    /// A packed frontier is retained under the same Checked/RecordedSample rule
    /// as a downloaded one, so the independent lane's no-transfer path cannot
    /// drift from `resolve_frontier`.
    #[test]
    fn packed_frontier_retention_matches_the_downloaded_rule() {
        let logits: Vec<f32> = (0..VOCAB).map(|token| if token == 5 { 4.0 } else { -1.0 }).collect();
        let bytes: Vec<u8> = logits.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let batch = BatchScores::new(bytes).unwrap();
        // Greedy (Checked): the cross-check passes and the id is the argmax.
        let checked = retain_packed_frontier(&batch, 0, FrontierRetain::Checked, None).unwrap();
        assert_eq!(checked.select(None).unwrap(), 5);
        // A packed row whose stored draw is not the argmax must be retained
        // as-is under RecordedSample, and rejected under Checked.
        let mut drawn = BatchScores::new(
            logits.iter().flat_map(|value| value.to_ne_bytes()).collect()).unwrap();
        drawn.best[0] = 7;
        assert_eq!(retain_packed_frontier(&drawn, 0, FrontierRetain::RecordedSample, None)
            .unwrap().select(None).unwrap(), 7);
        assert!(retain_packed_frontier(&drawn, 0, FrontierRetain::Checked, None).is_err(),
            "the Checked rule must still reject a draw");
    }

    /// A layout without the sampled terminal keeps the CPU path. Routing it into
    /// the terminal would fail the whole lane on `execute_shared_sampled`'s
    /// default bail, which is the regression the capability gate fixes; the gate
    /// is what decides the route. Chunk 4a keeps this as the one **round-level**
    /// fallback, because a missing terminal has no per-row device substitute.
    #[test]
    fn layouts_without_the_sampled_terminal_keep_the_cpu_path() {
        // Chunk 1's two layouts, as a property of the type.
        assert!(
            <crate::v41_target_pass::TargetPass<'static, 'static> as
                crate::v41_target_pass::VerificationTarget<'static>>::SUPPORTS_SAMPLED_TERMINAL,
            "TargetPass implements the sampled terminal"
        );
        assert!(
            !<crate::v41_target_pass::DistributedTargetPass<'static, 'static> as
                crate::v41_target_pass::VerificationTarget<'static>>::SUPPORTS_SAMPLED_TERMINAL,
            "the distributed layout has no sampled terminal"
        );
        // The gate is capability AND "some row is device-servable". A
        // stochastic round is no longer a reason to move the whole round to the
        // CPU -- only a round with no device-servable row is.
        assert!(use_sampled_terminal(true, true));
        assert!(!use_sampled_terminal(false, true), "no terminal -> CPU fallback");
        assert!(!use_sampled_terminal(true, false), "no device row -> CPU path");
        assert!(!use_sampled_terminal(false, false));
    }

    /// DEFECT-3 seam: `needs_mask` is per **row**, so a row whose grammar
    /// allows every token must be marked `NO_MASK` and get a zeroed arena row.
    /// An early masked row followed by an unmasked one (and the reverse) is the
    /// sequence that a reused mask buffer corrupts.
    #[test]
    fn per_row_needs_mask_marks_unmasked_rows_and_leaves_their_arena_zero() {
        use ds41rt_core::TargetSamplingParams;
        let words = VOCAB.div_ceil(32);
        let pattern = |tag: u32| {
            let mut mask = vec![0u32; words];
            mask[0] = tag;
            mask[words - 1] = tag ^ 0xFFFF_FFFF;
            mask
        };
        // Member 0: row 0 masked, rows 1-2 masked, row 3 needs no mask.
        // Member 1: row 0 needs no mask, row 1 masked.
        let row_masks: Vec<Vec<Option<Vec<u32>>>> = vec![
            vec![Some(pattern(0xA1)), Some(pattern(0xA2)), Some(pattern(0xA3)), None],
            vec![None, Some(pattern(0xB1))],
        ];
        let members: Vec<SamplingMember> = row_masks
            .iter()
            .map(|masks| SamplingMember {
                params: TargetSamplingParams::greedy(),
                base_position: 0,
                row_masks: masks.clone(),
            })
            .collect();
        let inputs: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3], vec![4, 5]];
        let plan = build_target_sampling_plan(&members, &inputs).unwrap();
        // `NO_MASK` is decided per row, not per member.
        let no_mask: Vec<bool> = plan
            .rows
            .iter()
            .map(|request| {
                request.row.flags & ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK != 0
            })
            .collect();
        assert_eq!(no_mask, vec![false, false, false, true, true, false]);
        for (row, request) in plan.rows.iter().enumerate() {
            if no_mask[row] {
                assert_eq!(request.row.mask_row, ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW);
            } else {
                assert_eq!(request.row.mask_row as usize, row);
            }
        }
        // The arena carries exactly the masked rows at their own ordinals, and a
        // zeroed row for every `None`, so no row can inherit a neighbour's bits.
        let mut arena = vec![0u32; 6 * words];
        build_sampling_masks(&row_masks, &mut arena).unwrap();
        for (row, expected) in [(0usize, 0xA1u32), (1, 0xA2), (2, 0xA3), (5, 0xB1)] {
            assert_eq!(arena[row * words], expected, "arena row {row}");
            assert_eq!(arena[row * words + words - 1], expected ^ 0xFFFF_FFFF, "arena row {row}");
        }
        for row in [3usize, 4] {
            assert!(
                arena[row * words..(row + 1) * words].iter().all(|word| *word == 0),
                "unmasked arena row {row} must stay zeroed"
            );
        }
    }

    /// P0-B seam: the mask arena is `rows x words`, indexed by row ordinal, so a
    /// round that mixes unconstrained and constrained members lands each mask on
    /// its own row. Packing only the constrained rows (the pre-fix staging)
    /// shifts every mask after the first unconstrained member.
    #[test]
    fn mask_arena_is_indexed_by_row_ordinal_for_mixed_rounds() {
        let words = VOCAB.div_ceil(32);
        // Two unconstrained members, then a constrained one, then two more
        // unconstrained ones, then a constrained one.
        let row_masks: Vec<Vec<Option<Vec<u32>>>> = vec![
            vec![None, None],
            vec![None],
            vec![Some(pattern(20)), Some(pattern(21)), Some(pattern(22))],
            vec![None],
            vec![None, None],
            vec![Some(pattern(50))],
        ];
        let mut arena = vec![0u32; 10 * words];
        build_sampling_masks(&row_masks, &mut arena).unwrap();
        // `pattern(tag)` puts `0xAA00 + tag` in word 0 and `0xBB00 + tag` in word 1.
        // Global row ordinals: member 2 starts at row 3, member 5 at row 9.
        for (ordinal, tag) in [(3usize, 20u32), (4, 21), (5, 22), (9, 50)] {
            assert_eq!(arena[ordinal * words], 0xAA00 + tag, "mask row {ordinal}");
            assert_eq!(arena[ordinal * words + 1], 0xBB00 + tag, "mask row {ordinal}");
        }
        // Sensitivity: packing only the constrained rows would place the fourth
        // constrained row (tag 50) at ordinal 3, so a shifted arena cannot pass.
        assert_eq!(arena[3 * words], 0xAA00 + 20, "mask row 3 must be member 2's first row");
        assert_ne!(arena[3 * words], 0xAA00 + 50, "ordinal packing must not be contiguous");
        // Every unconstrained ordinal is zeroed and therefore never read.
        for ordinal in [0usize, 1, 2, 6, 7, 8] {
            assert!(arena[ordinal * words..(ordinal + 1) * words].iter().all(|word| *word == 0),
                "NO_MASK row {ordinal} must stay zeroed");
        }
    }

    /// A two-word mask pattern that is recognisable per row.
    fn pattern(tag: u32) -> Vec<u32> {
        let mut mask = vec![0u32; VOCAB.div_ceil(32)];
        mask[0] = 0xAA00 + tag;
        mask[1] = 0xBB00 + tag;
        mask
    }

    /// P3: the plan is per row — absolute position, row-ordinal `mask_row`, the
    /// `NO_MASK` sentinel, and `STRICT_FINITE` on greedy/constrained rows only.
    #[test]
    fn plan_flags_and_positions_are_per_row_and_self_consistent() {
        use ds41rt_core::TargetSamplingParams;
        let greedy = TargetSamplingParams::greedy();
        let stochastic = TargetSamplingParams::new(0.7, 0.9, Some(40), 0.05, 11).unwrap();
        let members = vec![
            SamplingMember { params: stochastic, base_position: 100, row_masks: vec![None; 3] },
            SamplingMember {
                params: greedy,
                base_position: 7,
                row_masks: vec![Some(pattern(1)), Some(pattern(2))],
            },
        ];
        let inputs = vec![vec![0, 1, 2], vec![3, 4]];
        let plan = build_target_sampling_plan(&members, &inputs).unwrap();
        assert_eq!(plan.rows.len(), 5);
        assert_eq!(plan.greedy, vec![false, false, false, true, true]);
        // Positions are absolute per member and never reset by the batch layout.
        assert_eq!(plan.rows.iter().map(|r| r.row.position).collect::<Vec<_>>(),
            vec![100, 101, 102, 7, 8]);
        // Row ordinals are the mask row for constrained rows, the sentinel else.
        assert_eq!(plan.rows.iter().map(|r| r.row.mask_row).collect::<Vec<_>>(),
            vec![
                ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW,
                ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW,
                ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW,
                3,
                4,
            ]);
        assert_eq!(plan.rows.iter().map(|r| r.row.output_row).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]);
        // The unconstrained stochastic rows carry NO_MASK and stay permissive;
        // the constrained greedy rows carry STRICT_FINITE and GREEDY. No row has
        // both NO_MASK and a real mask row.
        for (index, request) in plan.rows.iter().enumerate() {
            let flags = request.row.flags;
            let no_mask = flags & ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK != 0;
            let strict = flags & ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE != 0;
            assert_eq!(no_mask, index < 3, "row {index} NO_MASK");
            assert_eq!(strict, index >= 3, "row {index} STRICT_FINITE");
        }
    }

    /// Host side of the device ABI: disabled encodings are resolved here, and
    /// `ln_min_p` is the exact host threshold the kernel adds.
    #[test]
    fn target_sampling_row_resolves_disabled_encodings_and_min_p_threshold() {
        use ds41rt_core::TargetSamplingParams;
        let greedy = TargetSamplingParams::greedy().with_seed(7);
        let row = target_sampling_row(
            greedy,
            0,
            ds41rt_ffi::DS41RT_V41_SAMPLER_NO_MASK_ROW,
            ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_NO_MASK,
            0,
        );
        assert_eq!(row.temperature, 0.0);
        assert_eq!(row.top_p, 1.0);
        assert_eq!(row.top_k, 0, "None must be encoded as the disabled zero");
        assert_eq!(row.min_p, 0.0);
        assert_eq!(row.seed, 7);
        assert_eq!(row.ln_min_p.to_bits(), f32::NEG_INFINITY.to_bits());

        let filtered = TargetSamplingParams::new(0.7, 0.9, Some(40), 0.05, 4242).unwrap();
        let row = target_sampling_row(
            filtered,
            100,
            0,
            ds41rt_ffi::DS41RT_V41_SAMPLER_FLAG_STRICT_FINITE,
            3,
        );
        assert_eq!(row.position, 100);
        assert_eq!(row.temperature, 0.7);
        assert_eq!(row.top_p, 0.9);
        assert_eq!(row.top_k, 40);
        assert_eq!(row.output_row, 3);
        // The device never calls logf, so this must be the host f32::ln bits.
        assert_eq!(row.ln_min_p.to_bits(), 0.05f32.ln().to_bits());
    }
}
