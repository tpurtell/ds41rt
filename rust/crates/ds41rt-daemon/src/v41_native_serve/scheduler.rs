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

/// Per-row parameters and greedy flags for one sampled round.
struct SamplingPlan {
    rows: Vec<TargetSamplingRowRequest>,
    greedy: Vec<bool>,
}

/// One member's sampling inputs for a round, resolved from `Active` by the
/// caller so the planner itself is a pure function of its arguments.
#[derive(Clone)]
struct SamplingMember {
    params: ds41rt_core::TargetSamplingParams,
    base_position: u64,
    /// Per row: the grammar mask, or `None` when the grammar allows every
    /// token. `needs_mask` is a per-row property, so it decides that row's
    /// `NO_MASK` flag — a constrained request whose grammar is exhausted on a
    /// later row must not have the previous row's grammar applied to it.
    row_masks: Vec<Option<Vec<u32>>>,
}

/// Build the per-row parameter blocks for one round.
///
/// `position(offset + index)` is the row's absolute emitted-token index, so a
/// row's draw and its greedy decision never depend on the batch layout.
///
/// `mask_row` is simply the row's ordinal: the mask arena is `rows x
/// ceil(vocab/32)` and indexed by row, with `NO_MASK` rows carrying the
/// `0xFFFFFFFF` sentinel. Keeping the two a single flat mapping means no
/// ordinal remapping can silently pair a row with another row's grammar.
fn build_target_sampling_plan(members: &[SamplingMember], inputs: &[Vec<u32>],
) -> Result<SamplingPlan> {
    let mut plan = SamplingPlan { rows: Vec::new(), greedy: Vec::new() };
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
            plan.rows.push(TargetSamplingRowRequest {
                row: target_sampling_row(params, meta.base_position + index as u64, mask_row,
                    flags, output_row),
                greedy,
            });
            plan.greedy.push(greedy);
        }
    }
    Ok(plan)
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
fn build_sampling_masks(row_masks: &[Vec<Option<Vec<u32>>>], arena: &mut [u32]) -> Result<()> {
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
fn use_sampled_terminal(supports_terminal: bool, fully_greedy: bool) -> bool {
    supports_terminal && fully_greedy
}

/// Whether this round can use the device terminal.
///
/// Chunk 1 routes **greedy rows only** through K1 and leaves stochastic rows on
/// the CPU path, so a round is device-selected exactly when every row is
/// greedy. A constrained greedy row is included: it is a masked argmax, which
/// is what K1 computes, and the commit path consumes the device's id.
fn round_is_fully_greedy<'a>(active: &[Option<Active<'a>>], members: &[usize]) -> bool {
    !members.is_empty()
        && members.iter().all(|&slot| {
            active[slot]
                .as_ref()
                .is_some_and(|request| request.job.sampling.is_greedy())
        })
}

/// Scope of the device terminal in chunk 1: **all-greedy rounds** (any mix of
/// unconstrained and constrained members) are device-selected. Any round that
/// contains a stochastic member stays entirely on the existing CPU path and
/// downloads its full rows for every member, greedy ones included. Removing
/// that qualifier is chunk 2's job, not this chunk's.
///
/// Everything one sampled round needs, built once per lane round.
struct SamplingRound {
    plan: SamplingPlan,
    arena: Vec<u32>,
    /// Rows whose full logits must be downloaded for the CPU-side trace. The
    /// `ds41rt::logit_trace` target logs `top_two`, which only the raw row
    /// provides, so a traced round downloads every row exactly as the
    /// pre-device path did (design §8.5/R15).
    trace_rows: Vec<usize>,
}

/// Build the per-row plan and the host mask arena for a fully greedy round.
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
    let row_masks: Vec<Vec<Option<Vec<u32>>>> =
        resolved.iter().map(|member| member.row_masks.clone()).collect();
    build_sampling_masks(&row_masks, &mut arena)?;
    let trace_rows = if trace { (0..rows).collect() } else { Vec::new() };
    Ok(SamplingRound { plan, arena, trace_rows })
}

/// Run the device-selected terminal and assemble the selection batch.
///
/// This is entered only for an **all-greedy** round, so no row needs the CPU
/// sampler. An untraced round therefore downloads nothing beyond the small
/// ids/scores/status vectors K1 produced, and its rows carry ids but no logits.
/// A traced round downloads every row (`SamplingRound::trace_rows`) because
/// `ds41rt::logit_trace` logs `top_two`.
///
/// A round containing any stochastic member never reaches here: it stays on the
/// CPU path and downloads every member's rows, greedy ones included. That
/// qualifier is exact for chunk 1 and is chunk 2's to remove.
async fn execute_sampled_rows<'a>(pass: &mut TargetPass<'_, 'a>,
    requests: &Requests<'a>, batch: &mut RequestBatch, transport: &mut NativeTp4Wave<'a>,
    round: &SamplingRound,
) -> Result<BatchScores> {
    let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
    unsafe {
        pass.execute_sampled(requests, batch, transport, 0, &selected, &round.plan.rows,
            Some(&round.arena), VOCAB.div_ceil(32)).await?;
    }
    let sampled = pass.sampled_rows()?;
    ensure!(sampled.rows() == round.plan.rows.len(), "sampled row count differs from the plan");
    sampled.check_status(&(0..sampled.rows()).collect::<Vec<_>>())?;
    // Untraced all-greedy rounds download nothing; a traced round downloads
    // only the rows the trace itself logs.
    if round.trace_rows.is_empty() {
        return BatchScores::from_sampled(&sampled, &round.plan.greedy);
    }
    let bytes = pass.download_sampled_rows(&sampled, &round.trace_rows).await?;
    sampled.with_full_logits(&round.trace_rows, bytes)
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
            Some(&round.arena), VOCAB.div_ceil(32)).await?;
    }
    let sampled = pass.sampled_rows()?;
    ensure!(sampled.rows() == round.plan.rows.len(), "sampled row count differs from the plan");
    sampled.check_status(&(0..sampled.rows()).collect::<Vec<_>>())?;
    if round.trace_rows.is_empty() {
        return BatchScores::from_sampled(&sampled, &round.plan.greedy);
    }
    let bytes = pass.download_sampled_rows(&sampled, &round.trace_rows).await?;
    sampled.with_full_logits(&round.trace_rows, bytes)
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
    // Chunk-1 scope: an **all-greedy** round (any mix of unconstrained and
    // constrained members) is selected on the device. A round containing any
    // stochastic member stays entirely on the existing CPU path below and
    // downloads full rows for every member, greedy ones included — that
    // qualifier is what makes the residency claim exact, and removing it is
    // chunk 2's job. A traced round still takes this path, but
    // `SamplingRound::trace_rows` then downloads every row for `top_two`.
    if !compact && round_is_fully_greedy(active, members) {
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
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests, active, members,
            &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes, 0,
            SampleSource::DeviceSelected)?;
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
    let next = runtime.block_on(execute_logits(lib, pass, requests, &mut batch, transport, capture_routes, compact));
    let executed_us = started.elapsed().as_micros() as u64;
    let result = (|| -> Result<()> {
        let next = next?;
        tracing::debug!(target: "ds41rt::cost_model", batch=batch_id, lane,
            requests=members.len(), rows=inputs.iter().map(Vec::len).sum::<usize>(),
            prepared_us, verify_us=executed_us-prepared_us, "verification round cost");
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests, active, members,
            &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes,
            executed_us-prepared_us, SampleSource::CpuRecomputed)?;
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

struct CommitDecision {
    accepted_drafts: u32,
    emitted: usize,
    accepted: Vec<u32>,
    emissions: Vec<Vec<u32>>,
    next_after_commit: Vec<Option<TokenScores>>,
    frontier_downloads: Vec<(usize, usize, Option<Vec<u32>>)>,
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
    source: SampleSource,
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
        let params = request.job.sampling;
        let base_position = request.generated as u64;
        // Stochastic requests select a target sample per row. Reusing the same
        // greedy verifier turns "draft equals target argmax" into the exact
        // sample-and-match rule: a draft is accepted while it equals the target
        // sample, and the first mismatch emits the target sample. The emitted
        // token is always the target draw, so speculation cannot bias p.
        // A device-selected round consumes K1's ids directly; every other round
        // keeps the existing per-row CPU selection, which is passed in as a
        // closure so the device path can never reach it.
        debug_assert!(
            source == SampleSource::CpuRecomputed || params.is_greedy(),
            "device selection is only claimed for greedy rounds"
        );
        let selected = select_target_row(next, offset, input.len(), source, || {
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
        next_after_commit.push(if finishing && next.has_full_logits() {
            Some(next.retain(frontier)?)
        } else {
            if finishing {
                // The frontier belongs to this member's own hypothetical
                // prefix, so a constrained row retains against its grammar
                // mask rather than the plain argmax.
                let row_in_round = frontier - offset;
                let mask = match request.constraint.as_ref() {
                    Some(state) => state.prepare_verification_mask_row(input, row_in_round)?,
                    None => None,
                };
                frontier_downloads.push((next_after_commit.len(), frontier, mask));
            }
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
    source: SampleSource,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let Some(batch) = owned_batch else { return Ok((0, 0, Vec::new())); };
    let mut decision = prepare_commit_lane(lane, requests, active, members, inputs,
        next, draft.as_deref(), verify_us, source)?;
    for (member, row, mask) in decision.frontier_downloads.drain(..) {
        let mask = mask.as_deref();
        decision.next_after_commit[member] =
            Some(next.retain_from_device(lib, pass.output(batch)?.logits, row, mask)?);
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

    /// A layout without the sampled terminal keeps the CPU path even for an
    /// all-greedy round. Routing it into the terminal would fail the whole lane
    /// on `execute_shared_sampled`'s default bail, which is the regression the
    /// capability gate fixes; the gate is what decides the route.
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
            "the distributed layout has no sampled terminal in chunk 1"
        );
        // The gate: capability AND round shape.
        assert!(use_sampled_terminal(true, true));
        assert!(!use_sampled_terminal(false, true), "no terminal -> CPU fallback");
        assert!(!use_sampled_terminal(true, false), "stochastic round -> CPU path");
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
