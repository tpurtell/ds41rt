use super::speculative::DraftChain;
use super::*;
use crate::v41_target_pass::VerificationTarget;
mod independent;
mod admission;
mod layout;
use layout::ServingTarget;
use super::scores::BatchScores;
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
        job: NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), max_tokens: 4, events },
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
    if request.cacheable && requests.cache().request_id(request.lease).is_ok() {
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

pub(super) fn serve<'w, 'a, P: ServingTarget<'w, 'a>>(lib: &'a NativeLibrary, args: &crate::cli::NativeServeArgs,
    runtime: &tokio::runtime::Runtime, receive: &mut mpsc::Receiver<NativeRequest>,
    first: &mut P, second: &mut P,
    requests: &mut Requests<'a>, first_transport: &mut P::Transport,
    second_transport: &mut P::Transport, mut draft: Option<&mut DraftRuntime<'w, 'a, P::Chain>>,
    vision: &mut crate::v41_vision::VisionRuntime<'a>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
) -> Result<()> {
    let mut active: Vec<Option<Active<'a>>> = (0..args.concurrency).map(|_| None).collect();
    let mut compiler = super::constraints::Compiler::new(lib, args.snapshot.join("tokenizer.json"));
    let mut id = 0u64;
    let mut closed = false;
    let mut pending: Option<admission::Pending> = None;
    let template = requests.cache().sources()[0].get().source_cache().page_segments(0)[0];
    let host_cache = super::prefix::HostCacheBinding::new(lib, args.host_cache_config()?, template)?;
    let mut prefixes = PrefixCache::new(args.prefix_cache_entries as usize).with_host_cache(host_cache);
    let mut stats_published = Instant::now();
    let limits = ds41rt_api::native_v41::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    loop {
        prefixes.tick();
        if stats_published.elapsed() >= std::time::Duration::from_secs(1) {
            stats_published = Instant::now();
            if let Some(metrics) = prefixes.host_metrics() {
                if let Ok(mut slot) = stats.lock() {
                    // Deliberately exports the cache's whole effective `Config` under
                    // `host_cache_config` (packet HC-9), not just `store_pace_ns`: fleet
                    // operators tune several of these knobs, and one key keeps the export
                    // forward-compatible as new knobs land.
                    *slot = serde_json::json!({
                        "host_cache": metrics,
                        "host_cache_config": prefixes.host_config(),
                    });
                }
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
                let anchor = scores.select(mask)?;
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
    members: &[usize], mut draft: Option<&mut DraftRuntime<'_, 'a>>,
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
        && members.iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none());
    let batch_id = batch.as_ref().unwrap().cache()?.identity();
    if tracing::enabled!(target: "ds41rt::cost_model", tracing::Level::DEBUG) {
        if let Some(draft) = draft.as_deref() {
            let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                (active[slot].as_ref().unwrap().id, lane, input.len()-1)).collect();
            draft.trace_cost_forecast(batch_id, &candidates);
        }
    }
    let next = runtime.block_on(execute_logits(lib, pass, requests, &mut batch, transport, capture_routes, compact));
    let executed_us = started.elapsed().as_micros() as u64;
    let result = (|| -> Result<()> {
        let next = next?;
        tracing::debug!(target: "ds41rt::cost_model", batch=batch_id, lane,
            requests=members.len(), rows=inputs.iter().map(Vec::len).sum::<usize>(),
            prepared_us, verify_us=executed_us-prepared_us, "verification round cost");
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests, active, members,
            &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes, executed_us-prepared_us)?;
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

struct CommitDecision {
    accepted_drafts: u32,
    emitted: usize,
    accepted: Vec<u32>,
    emissions: Vec<Vec<u32>>,
    next_after_commit: Vec<Option<TokenScores>>,
    frontier_downloads: Vec<(usize, usize)>,
}
fn prepare_commit_lane<'a, C: DraftChain<'a>>(lane: usize,
    requests: &Requests<'a>, active: &[Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    next: &BatchScores, draft: Option<&DraftRuntime<'_, 'a, C>>, verify_us: u64,
) -> Result<CommitDecision> {
    let mut accepted_drafts = 0u32;
    let mut emitted = 0usize;
    let mut offset = 0;
    let mut accepted = Vec::new();
    let mut emissions = Vec::new();
    let mut next_after_commit = Vec::new();
    let mut frontier_downloads = Vec::new();
    for (&slot, input) in members.iter().zip(inputs) {
        let request = active[slot].as_ref().unwrap();
        let constrained = request.constraint.as_ref().map(|state|
            state.select_verification(next, offset, input)).transpose()?;
        let selected = constrained.as_deref().unwrap_or(&next.best[offset..offset + input.len()]);
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
        next_after_commit.push(if finishing && next.has_full_logits() {
            Some(next.retain(frontier)?)
        } else {
            if finishing { frontier_downloads.push((next_after_commit.len(), frontier)); }
            None
        });
        offset += input.len(); accepted.push(decision.accepted_inputs); emissions.push(decision.emitted);
    }
    Ok(CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit, frontier_downloads })
}
fn publish_commit_lane<'a, C: DraftChain<'a>>(pass: &impl VerificationTarget<'a>, active: &mut [Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], owned_batch: &mut Option<RequestBatch>,
    mut draft: Option<&mut DraftRuntime<'_, 'a, C>>, capture_routes: bool, decision: CommitDecision,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit, frontier_downloads } = decision;
    ensure!(frontier_downloads.is_empty(), "retained frontier downloads are incomplete");
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
fn commit_lane<'w, 'a>(lib: &'a NativeLibrary, lane: usize, pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    active: &mut [Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    owned_batch: &mut Option<RequestBatch>, next: &BatchScores,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>, capture_routes: bool, verify_us: u64,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let Some(batch) = owned_batch else { return Ok((0, 0, Vec::new())); };
    let mut decision = prepare_commit_lane(lane, requests, active, members, inputs,
        next, draft.as_deref(), verify_us)?;
    for (member, row) in decision.frontier_downloads.drain(..) {
        decision.next_after_commit[member] = Some(next.retain_from_device(lib, pass.output(batch)?.logits, row)?);
    }
    if let Some(draft) = draft.as_deref_mut() { draft.commit_batch(pass, requests, batch, &decision.accepted)?; }
    else { pass.commit(requests, batch, &decision.accepted)?; }
    publish_commit_lane(pass, active, members, inputs, owned_batch, draft, capture_routes, decision)
}
