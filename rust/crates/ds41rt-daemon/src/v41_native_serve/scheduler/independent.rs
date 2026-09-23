//! Lane-local rounds; shared owners are borrowed only during synchronous work.
use super::*;
use std::cell::{Cell, RefCell};

pub(super) fn run<'a, P: VerificationTarget<'a>, C: DraftChain<'a>>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    first: &mut P, second: &mut P, requests: &mut Requests<'a>,
    first_transport: &mut P::Transport, second_transport: &mut P::Transport,
    active: &mut [Option<Active<'a>>], draft: Option<&mut DraftRuntime<'_, 'a, C>>,
    prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>,
) -> Result<()> {
    let requests = RefCell::new(requests);
    let active = RefCell::new(active);
    let draft = RefCell::new(draft);
    let prefixes = RefCell::new(prefixes);
    let drain = Cell::new(false);
    // Do not cancel the peer future on error: it may own queued CUDA/RDMA work.
    let results = runtime.block_on(async { tokio::join!(
        lane(0, lib, first, first_transport, &requests, &active, &draft, &prefixes, receive, &drain, wake),
        lane(1, lib, second, second_transport, &requests, &active, &draft, &prefixes, receive, &drain, wake),
    ) });
    results.0?; results.1?;
    Ok(())
}

async fn lane<'a, P: VerificationTarget<'a>, C: DraftChain<'a>>(lane: usize, lib: &'a NativeLibrary, pass: &mut P,
    transport: &mut P::Transport, requests: &RefCell<&mut Requests<'a>>,
    active: &RefCell<&mut [Option<Active<'a>>]>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a, C>>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, receive: &mpsc::Receiver<NativeRequest>, drain: &Cell<bool>, wake: admission::Wake<'_>,
) -> Result<()> {
    let result = async {
        let mut round_id = 0u64;
        loop {
            if drain.get() { return Ok(()); }
            // This lane has completed all of its own GPU/transport work. Retire
            // only its requests; the peer need not stop or migrate survivors.
            let retired: Vec<_> = active.borrow().iter().enumerate().filter_map(|(slot, entry)|
                entry.as_ref().filter(|r| r.lane == lane && (r.finished || r.job.events.is_closed()))
                    .map(|_| slot)).collect();
            for slot in retired {
                let request = active.borrow_mut()[slot].take().unwrap();
                let request_id = request.id;
                retire(lane, request, requests, prefixes, draft).await?;
                tracing::debug!(target: "ds41rt::lane_schedule", lane, request_id, round_id,
                    "independent lane request retired");
            }
            // Admission/prefill still uses both execution lanes.
            let members: Vec<_> = {
                let active = active.borrow();
                if wake.ready(active.iter().flatten().count(), active.len(), !receive.is_empty()) {
                    drain.set(true); return Ok(());
                }
                active.iter().enumerate().filter_map(|(slot, r)|
                    r.as_ref().filter(|r| r.lane == lane).map(|_| slot)).collect()
            };
            if members.is_empty() { return Ok(()); }
            ensure!(members.len() <= 8, "independent lane exceeds eight requests");
            let started = Instant::now();
            // Reserve target capacity and snapshot this lane's seeds, then release
            // all bank borrows before waiting on this lane's draft workspace.
            let seeds = {
                let active = active.borrow();
                let mut requests = requests.borrow_mut();
                let verify_rows = draft.borrow().as_ref().map_or(1, |d| d.max_verify_rows());
                let capacity: Vec<_> = members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    (r.lease, (r.job.max_tokens-r.generated).min(verify_rows) as u32)
                }).collect();
                prefixes.borrow_mut().make_room(&mut requests, &capacity)?;
                members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    Ok((r.id, r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens-r.generated))
                }).collect::<Result<Vec<_>>>()?
            };
            let (mut inputs, draft_us) = loop {
                let proposed = {
                    let mut draft = draft.borrow_mut();
                    if let Some(draft) = draft.as_deref_mut() { draft.poll_propose(lane, &seeds)? }
                    else { Some((seeds.iter().map(|r| vec![r.1]).collect(), 0)) }
                };
                if let Some(inputs) = proposed { break inputs; }
                // Even when a peer requests a cohort drain, finish our pending
                // draft and transaction before retirement can recycle its slots.
                tokio::task::yield_now().await;
            };
            let shared = active.borrow().iter().flatten().any(|r| r.lane != lane);
            let (inputs, mut batch, capture_routes) = {
                let active = active.borrow();
                let mut requests = requests.borrow_mut();
                let mut draft = draft.borrow_mut();
                for (&slot, input) in members.iter().zip(&mut inputs) {
                    let r = active[slot].as_ref().unwrap();
                    if let Some(constraint) = &r.constraint { constraint.truncate_proposal(input)?; }
                }
                // Lengths are chosen from this lane's proposals and routes only;
                // the peer lane's activity selects the shared-regime fit.
                if let Some(draft) = draft.as_deref_mut() {
                    let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                        (active[slot].as_ref().unwrap().id, input.len()-1)).collect();
                    if let Some(lengths) = draft.select_lengths(lane, &candidates, shared)? {
                        for (input, length) in inputs.iter_mut().zip(lengths) { input.truncate(length+1); }
                    }
                }
                let batch = prepare_decode_lane(&mut requests, &active, &members, &inputs, draft.is_some())?;
                (inputs, Some(batch), draft.as_deref().is_some_and(DraftRuntime::capture_routes))
            };
            let prepared_us = started.elapsed().as_micros() as u64;
            round_id += 1;
            tracing::debug!(target: "ds41rt::lane_schedule", lane, round_id, requests=members.len(),
                "independent verifier issued");
            let operation: Result<()> = async {
                pass.set_route_capture(capture_routes)?;
                let current = batch.as_mut().unwrap();
                let batch_id = current.cache()?.identity();
                let selected: Vec<_> = (0..current.cache()?.positions().len()).collect();
                let active_borrow = active.borrow();
                let compact = !tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG)
                    && members.iter().all(|&slot| {
                        let request = active_borrow[slot].as_ref().unwrap();
                        request.constraint.is_none() && request.job.sampling.is_greedy()
                    });
                drop(active_borrow);
                // Chunk 4a: a lane round is device-selected whenever the layout
                // supports the terminal and at least one row is servable; each
                // row's parameters then decide its kernels, and an unservable
                // row is re-sampled on the CPU inside the same round. A round
                // whose every row is unservable, or a layout without the
                // terminal, keeps the whole-round CPU path.
                let mut round = None;
                let next = if compact {
                    BatchScores::from_greedy(unsafe {
                        pass.execute_shared_greedy(requests, current, transport, 0, &selected).await?
                    })?
                } else if use_sampled_terminal(P::SUPPORTS_SAMPLED_TERMINAL,
                    round_has_device_rows(&active.borrow(), &members)) {
                    let built = build_sampling_round(&active.borrow(), &members, &inputs,
                        tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG))?;
                    let scored = execute_shared_sampled_rows(pass, requests, current,
                        transport, &built).await?;
                    round = Some(built);
                    scored
                } else {
                    // No device terminal ran, so no row was device-selected: the
                    // counter must not claim the servable rows as device work.
                    sampling_stats::record_round(false, 0, 0);
                    unsafe { pass.execute_shared(requests, current, transport, 0, &selected).await?; }
                    BatchScores::new(pass.download_logits(current, &selected).await?)?
                };
                let verify_us = started.elapsed().as_micros() as u64 - prepared_us;
                tracing::debug!(target: "ds41rt::cost_model", batch=batch_id, lane, round_id,
                    requests=members.len(), rows=selected.len(), prepared_us, verify_us,
                    "verification round cost");
                let retain_enabled = prefixes.borrow().turn_bank_enabled();
                let mut decision = prepare_commit_lane(lane, &requests.borrow(),
                    &active.borrow(), &members, &inputs, &next, draft.borrow().as_deref(),
                    verify_us, round.as_ref(), retain_enabled)?;
                // A frontier row whose bytes are already packed (a fallback row
                // the CPU re-sampled) is retained in place: the pre-chunk-4b
                // code downloaded it here a second time even though it was
                // already on the host. This lane paid that transfer, so the
                // saving is counted here and not on the serial lane.
                let packed = std::mem::take(&mut decision.frontier_packed);
                sampling_stats::record_frontier_packed(packed.len(),
                    scores::ROW_BYTES, true);
                for (member, row, mask, retain) in packed {
                    decision.next_after_commit[member] =
                        Some(retain_packed_frontier(&next, row, retain, mask.as_deref())?);
                }
                if !decision.frontier_downloads.is_empty() {
                    let rows: Vec<_> = decision.frontier_downloads.iter().map(|&(_, row, _, _)| row).collect();
                    let bytes = pass.download_logits(batch.as_ref().unwrap(), &rows).await?;
                    ensure!(bytes.len() == rows.len() * scores::ROW_BYTES, "retained frontier download extent differs");
                    for ((member, row, mask, retain), bytes) in decision.frontier_downloads.drain(..)
                        .zip(bytes.chunks_exact(scores::ROW_BYTES)) {
                        decision.next_after_commit[member] =
                            Some(resolve_frontier(retain, &next, row, bytes, mask.as_deref())?);
                    }
                }
                let committed: Result<()> = async {
                    if let Some(draft) = draft.borrow_mut().as_deref_mut() {
                        draft.begin_queued_commit(lane, pass, &requests.borrow(),
                            batch.as_ref().unwrap(), &decision.accepted)?;
                    }
                    pass.enqueue_cache_commit(&requests.borrow(), batch.as_ref().unwrap(), &decision.accepted)?;
                    loop {
                        let draft_ready = draft.borrow().as_deref().map(|draft| draft.poll_queued_commit(lane))
                            .transpose()?.unwrap_or(true);
                        if pass.poll_cache_commit()? && draft_ready { break; }
                        tokio::task::yield_now().await;
                    }
                    if let Some(draft) = draft.borrow_mut().as_deref_mut() {
                        draft.finish_queued_commit(lane, pass, &mut requests.borrow_mut(),
                            batch.as_mut().unwrap(), &decision.accepted)
                    } else {
                        pass.commit(&mut requests.borrow_mut(), batch.as_mut().unwrap(), &decision.accepted)
                    }
                }.await;
                if let Err(error) = committed {
                    if let Err(cleanup) = pass.abort_cache_commit(&mut requests.borrow_mut()) {
                        tracing::error!(%cleanup, "draining failed lane window commit");
                    }
                    if let Some(draft) = draft.borrow_mut().as_deref_mut() {
                        if let Err(cleanup) = draft.abort_queued_commit(lane, &mut requests.borrow_mut(), batch.as_mut().unwrap()) {
                            tracing::error!(%cleanup, "draining failed lane draft commit");
                        }
                    } else { requests.borrow_mut().revoke_batch(batch.as_mut().unwrap()); }
                    return Err(error);
                }
                let (accepted, emitted, emissions, accepted_inputs) = publish_commit_lane(&mut active.borrow_mut(),
                    &members, &mut batch, decision)?;
                tracing::debug!(target: "ds41rt::lane_schedule", lane, round_id,
                    "independent verifier committed");
                for (&slot, tokens) in members.iter().zip(emissions) {
                    let sender = active.borrow()[slot].as_ref().unwrap().job.events.clone();
                    let delivered: Result<()> = async {
                        for token in tokens {
                            let (chunks, finished) = {
                                let mut active = active.borrow_mut();
                                let request = active[slot].as_mut().unwrap();
                                (request.emit_one(token)?, request.finished)
                            };
                            for chunk in chunks.into_iter().flatten() { sender.send(Ok(chunk)).await?; }
                            if finished { break; }
                        }
                        Ok(())
                    }.await;
                    if let Err(error) = delivered {
                        active.borrow_mut()[slot].as_mut().unwrap().finished = true;
                        let _ = sender.send(Err(format!("{error:#}").into())).await;
                    }
                }
                observe_lane_round(draft.borrow_mut().as_deref_mut(), capture_routes, lane, shared,
                    pass.captured_routes(), pass.captured_layer_done(), &active.borrow(), &members, &inputs,
                    &accepted_inputs, started);
                tracing::debug!(target: "ds41rt::timing", lane, requests=members.len(),
                    proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
                    draft_us, prepared_us, verify_us, total_us=started.elapsed().as_micros() as u64,
                    "native independent lane round");
                Ok(())
            }.await;
            let stopped_capture = pass.set_route_capture(false);
            let discarded = if let Some(batch) = &mut batch {
                // Successful commits relinquish ownership. Failed execution or
                // commit must drain/discard this lane before the outer reset.
                pass.discard(batch)
            } else { Ok(()) };
            operation?;
            stopped_capture?;
            discarded?;
            // Give already-ready remote completions an opportunity to run before
            // queuing another draft on the shared RTX.
            tokio::task::yield_now().await;
        }
    }.await;
    if result.is_err() { drain.set(true); }
    result
}

async fn retire<'a, C: DraftChain<'a>>(lane: usize, request: Active<'a>, requests: &RefCell<&mut Requests<'a>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a, C>>>) -> Result<()> {
    // Chunk-4b: consult the turn bank before requiring the retained frontier, so
    // a cache-disabled deployment does not warn about a frontier it correctly
    // chose not to download.
    let cacheable = request.cacheable && prefixes.borrow().turn_bank_enabled()
        && requests.borrow().cache().request_id(request.lease).is_ok();
    if cacheable {
        let retained: Result<()> = async {
            let next = request.next_after_commit.as_ref().context("finished request has no retained logits")?;
            let queued = prefixes.borrow_mut().queue_retain(lane, SnapshotKind::Turn, &request.tokens,
                &request.image_keys, next, request.id, request.lease, &mut requests.borrow_mut(),
                draft.borrow_mut().as_deref_mut())?;
            if queued {
                tracing::debug!(target: "ds41rt::lane_schedule", lane, request_id=request.id,
                    "independent snapshot queued");
                loop {
                    let ready = prefixes.borrow_mut().poll_retain(lane, &mut requests.borrow_mut(),
                        draft.borrow_mut().as_deref_mut())?;
                    if ready { break; }
                    tokio::task::yield_now().await;
                }
                tracing::debug!(target: "ds41rt::lane_schedule", lane, request_id=request.id,
                    "independent snapshot published");
            }
            Ok(())
        }.await;
        if let Err(error) = retained {
            if let Err(cleanup) = prefixes.borrow_mut().abort_retain(lane, &mut requests.borrow_mut(),
                draft.borrow_mut().as_deref_mut()) {
                tracing::error!(%cleanup, "draining failed independent snapshot");
            }
            tracing::warn!(%error, "completed request prefix was not retained");
        }
    }
    let target = requests.borrow_mut().release_if_present(request.lease);
    let speculative = draft.borrow_mut().as_deref_mut().map(|d| d.release(request.id)).transpose();
    target.and(speculative.map(|_| ()))
}
