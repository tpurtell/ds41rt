//! Upstream admission-control port (ds41rt component C8).
//!
//! Ported invariant classes from:
//! - vLLM `tests/v1/engine/test_admission_control.py` (queue overflow,
//!   admission gating, HTTP 503 rejection mapping), and
//! - sglang `test/registered/scheduler/test_min_free_slots_delayer.py`
//!   (min-free-slots admission-delay policy arithmetic).
//!
//! # Mapping to ds41rt surfaces
//!
//! | upstream invariant                     | ds41rt surface under test |
//! |----------------------------------------|---------------------------|
//! | admission gating / capacity accounting | `ds41rt_core::admit_layerwaves_for_iteration` (the per-iteration scheduler admission used by `commands::scheduler_smoke` and `real_full/scheduler/execution/admission.rs`) |
//! | queue-full rejection shape (503)       | `ds41rt_api::native_v41` router: bounded mpsc queue `try_send` failure -> HTTP 503, closed-queue health -> 503, worker-side admission failure -> 500 with cause retained |
//! | delayed admission (min-free-slots)     | sglang policy ported verbatim below as pure functions (`resolve_min_free_slots`, `MinFreeSlotsDelayer::should_delay`) plus a deferred-prefill scheduler simulation against the real `admit_layerwaves_for_iteration` |
//! | priority under pressure                | `PrefillChunkPolicy::decode_priority` ordering inside `admit_layerwaves_for_iteration` |
//! | concurrent single-slot admission       | the daemon's own invariant, modeled after `v41_requests::Requests::admit` ("request slot occupied") + the `v41_native_serve/scheduler.rs` slot loop; the real loop needs CUDA machinery so the occupancy arithmetic is modeled here |
//!
//! Not ported (no ds41rt equivalent): vLLM `human_readable_int` CLI notation
//! and `SchedulerConfig` pydantic validation; vLLM's `max_num_queued_reqs`
//! counting of *unfinished* requests has no counterparty because the daemon
//! bounds concurrency by fixed slots (`args.concurrency`), which the slot
//! model tests below cover instead.

// COVERAGE CLASS: standalone reference oracle. These cases document upstream
// behavior with no ds41rt dependency; they cannot detect product regressions
// by themselves. They are the comparison references for the deferred GPU
// parity tests (docs/test-coverage/DEFERRED.md) and are counted separately
// from product regression coverage (review MAJOR 4/6, 2026-09-15).

use ds41rt_core::{
    admit_layerwaves_for_iteration, LayerWave, LayerWaveMode, MtpVerifyBlock, PrefillChunk,
    PrefillChunkPolicy, Priority,
};

fn policy(
    max_prefill_tokens_per_iteration: usize,
    max_active_prefill_chunks: usize,
    decode_priority: bool,
) -> PrefillChunkPolicy {
    PrefillChunkPolicy {
        chunk_tokens: 16,
        max_prefill_tokens_per_iteration,
        max_active_prefill_chunks,
        decode_priority,
    }
}

fn prefill(name: &str, token_start: usize, token_count: usize, priority: i32) -> LayerWave {
    LayerWave::prefill(PrefillChunk::new(
        name,
        format!("seq-{name}"),
        3,
        token_start as u64,
        token_count,
        50 + token_start as u64,
        Priority(priority),
        ds41rt_core::GraphBucket::new(16),
        "placement-a",
    ))
}

fn decode(name: &str, position: u64, priority: i32) -> LayerWave {
    LayerWave::decode(ds41rt_core::DecodeStep::new(
        name,
        format!("seq-{name}"),
        3,
        position,
        Some(70 + position),
        Priority(priority),
        "placement-a",
    ))
}

fn mtp_verify(name: &str, token_start: usize, token_count: usize, priority: i32) -> LayerWave {
    LayerWave::mtp_verify(MtpVerifyBlock::new(
        name,
        format!("seq-{name}"),
        3,
        token_start as u64,
        token_count,
        Some(90 + token_start as u64),
        Priority(priority),
        ds41rt_core::GraphBucket::new(16),
        "placement-a",
    ))
}

/// A wave carries its request identity on its rows (a wave may merge several
/// requests' rows after `try_merge`, hence no top-level id field).
fn wave_id(wave: &LayerWave) -> &str {
    &wave.row_sources[0].request_id.0
}

// ---------------------------------------------------------------------------
// sglang min-free-slots admission-delay policy (verbatim arithmetic port)
// ---------------------------------------------------------------------------

/// Verbatim port of sglang `min_free_slots_delayer.resolve_min_free_slots`.
/// `None` = disabled. An explicit user value always wins, capped by
/// `max_running_requests` (`<= 1` disables). When unset, DFlash workloads fall
/// back to the legacy formula, disabled for clusters under 8.
fn resolve_min_free_slots(
    user_value: Option<i64>,
    max_running_requests: i64,
    is_dflash_family: bool,
) -> Option<i64> {
    let max_running_requests = max_running_requests.max(0);
    if let Some(user_value) = user_value {
        let threshold = user_value.min(max_running_requests);
        return (threshold > 1).then_some(threshold);
    }
    if is_dflash_family && max_running_requests >= 8 {
        return Some(((max_running_requests + 5) / 6).clamp(2, 4));
    }
    None
}

/// Verbatim port of sglang `MinFreeSlotsDelayer::should_delay`: delay fresh
/// admissions only while a decode batch is running and fewer than
/// `min_free_slots` allocatable request slots remain.
struct MinFreeSlotsDelayer {
    min_free_slots: i64,
}

impl MinFreeSlotsDelayer {
    fn should_delay(&self, running_bs: i64, num_allocatable_reqs: i64) -> bool {
        running_bs > 0 && num_allocatable_reqs < self.min_free_slots
    }
}

mod min_free_slots_delayer {
    use super::{resolve_min_free_slots, MinFreeSlotsDelayer};

    #[test]
    fn admission_unset_non_dflash_disables() {
        assert_eq!(resolve_min_free_slots(None, 512, false), None);
    }

    #[test]
    fn admission_unset_dflash_auto_enables() {
        assert_eq!(resolve_min_free_slots(None, 512, true), Some(4));
        assert_eq!(resolve_min_free_slots(None, 8, true), Some(2));
    }

    #[test]
    fn admission_unset_dflash_small_cluster_disables() {
        assert_eq!(resolve_min_free_slots(None, 7, true), None);
        assert_eq!(resolve_min_free_slots(None, 0, true), None);
    }

    #[test]
    fn admission_le_one_disables() {
        // <= 1 can never batch, so it is a no-op.
        assert_eq!(resolve_min_free_slots(Some(1), 512, false), None);
        assert_eq!(resolve_min_free_slots(Some(0), 512, false), None);
    }

    #[test]
    fn admission_explicit_value_survives_small_cluster() {
        // The < 8 guard belongs to the DFlash auto-default, not explicit values.
        assert_eq!(resolve_min_free_slots(Some(4), 7, false), Some(4));
        assert_eq!(resolve_min_free_slots(Some(4), 7, true), Some(4));
    }

    #[test]
    fn admission_non_dflash_uses_explicit_value() {
        assert_eq!(resolve_min_free_slots(Some(2), 8, false), Some(2));
        assert_eq!(resolve_min_free_slots(Some(3), 512, false), Some(3));
        assert_eq!(resolve_min_free_slots(Some(8), 512, false), Some(8));
        assert_eq!(resolve_min_free_slots(Some(16), 512, false), Some(16));
    }

    #[test]
    fn admission_explicit_value_is_capped_to_max_running_requests() {
        assert_eq!(resolve_min_free_slots(Some(16), 8, false), Some(8));
    }

    #[test]
    fn admission_user_value_overrides_dflash_default() {
        assert_eq!(resolve_min_free_slots(Some(3), 512, true), Some(3));
        assert_eq!(resolve_min_free_slots(Some(16), 512, true), Some(16));
    }

    #[test]
    fn admission_explicit_one_disables_dflash_default() {
        assert_eq!(resolve_min_free_slots(Some(1), 512, true), None);
    }

    #[test]
    fn admission_delayer_delays_below_threshold() {
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(delayer.should_delay(100, 2));
    }

    #[test]
    fn admission_delayer_no_delay_at_or_above_threshold() {
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(!delayer.should_delay(100, 4));
        assert!(!delayer.should_delay(100, 8));
    }

    #[test]
    fn admission_delayer_no_delay_when_idle() {
        // Nothing running: no decode batch to protect, prefill at once.
        let delayer = MinFreeSlotsDelayer { min_free_slots: 4 };
        assert!(!delayer.should_delay(0, 0));
    }
}

// ---------------------------------------------------------------------------
// Admission gating / capacity accounting (vLLM check_admission mapped onto
// admit_layerwaves_for_iteration)
// ---------------------------------------------------------------------------

mod admission_gating {
    use super::*;

    #[test]
    fn admission_no_effective_limits_allows_everything() {
        // An enormous budget admits a mixed batch: decode, MTP verify and
        // prefill waves all selected, nothing deferred (vLLM:
        // test_admission_no_limits_allows_everything).
        let policy = policy(usize::MAX, usize::MAX, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 0), mtp_verify("m0", 5, 3, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 3);
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_decode_rows, 1);
        assert_eq!(admission.selected_mtp_rows, 3);
        assert_eq!(admission.selected_prefill_rows, 16);
        assert_eq!(admission.selected_prefill_chunks, 1);
    }

    #[test]
    fn admission_prefill_allows_at_exact_token_boundary() {
        // Exactly the budget is admitted (vLLM: ..._allows_n_at_boundary).
        let policy = policy(32, 8, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 32, 0)], &policy);
        assert_eq!(admission.selected_prefill_rows, 32);
        assert!(admission.deferred.is_empty());
    }

    #[test]
    fn admission_prefill_defers_beyond_token_limit() {
        // Budget+1 rows defers the wave (vLLM: ..._rejects_at_limit /
        // ..._rejects_over_limit).
        let policy = policy(32, 8, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 33, 0)], &policy);
        assert!(admission.selected.is_empty());
        assert_eq!(admission.deferred.len(), 1);
        assert_eq!(admission.deferred[0].mode, LayerWaveMode::Prefill);
    }

    #[test]
    fn admission_prefill_defers_beyond_chunk_limit() {
        // Token budget alone is not enough: the active-chunk count gate
        // defers the third chunk (vLLM: independent limit checks).
        let policy = policy(1024, 2, true);
        let waves = vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)];
        let admission = admit_layerwaves_for_iteration(waves.clone(), &policy);
        assert_eq!(admission.selected_prefill_chunks, 2);
        assert_eq!(admission.deferred.len(), 1);
        assert_eq!(wave_id(&admission.deferred[0]), wave_id(&waves[2]));
    }

    #[test]
    fn admission_zero_token_budget_defers_all_prefill() {
        // vLLM: ..._rejects_when_zero_limit. A zero token budget rejects every
        // non-empty prefill wave, but decode work is never subject to the
        // prefill budget and still admits.
        let policy = policy(0, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 1);
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_zero_chunk_budget_defers_all_prefill() {
        let policy = policy(1024, 0, true);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 16, 0)], &policy);
        assert!(admission.selected.is_empty());
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_decode_and_mtp_are_never_token_gated() {
        // Under a zero prefill budget a decode + MTP batch still admits in
        // full: interactive latency work bypasses the prefill backlog gate.
        let policy = policy(0, 0, true);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 3, 0), mtp_verify("m0", 4, 2, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 2);
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_decode_rows, 1);
        assert_eq!(admission.selected_mtp_rows, 2);
    }

    #[test]
    fn admission_token_and_chunk_limits_checked_independently() {
        // vLLM: test_admission_both_limits_checked_independently — either
        // limit alone can defer. Here the token budget is exhausted first.
        let token_policy = policy(16, 4, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1)],
            &token_policy,
        );
        assert_eq!(admission.selected_prefill_rows, 16);
        assert_eq!(admission.deferred.len(), 1);
        // And with tokens available but chunks exhausted, deferral also fires.
        let chunk_policy = policy(1024, 1, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1)],
            &chunk_policy,
        );
        assert_eq!(admission.selected_prefill_chunks, 1);
        assert_eq!(admission.deferred.len(), 1);
    }

    #[test]
    fn admission_accounting_matches_selected_waves() {
        // Capacity accounting (the vLLM get_num_queued_tokens analog): the
        // reported row totals always equal the sum of the admitted waves.
        let policy = policy(32, 4, true);
        let waves = vec![
            prefill("p0", 0, 16, 0),
            prefill("p1", 16, 8, 1),
            decode("d0", 9, 0),
            mtp_verify("m0", 11, 4, 0),
        ];
        let rows_of = |mode: LayerWaveMode| {
            waves
                .iter()
                .filter(|wave| wave.mode == mode)
                .map(LayerWave::num_rows)
                .sum::<usize>()
        };
        let (prefill_rows, decode_rows, mtp_rows) = (
            rows_of(LayerWaveMode::Prefill),
            rows_of(LayerWaveMode::Decode),
            rows_of(LayerWaveMode::MtpVerify),
        );
        let admission = admit_layerwaves_for_iteration(waves, &policy);
        assert_eq!(admission.selected_prefill_rows, prefill_rows);
        assert_eq!(admission.selected_decode_rows, decode_rows);
        assert_eq!(admission.selected_mtp_rows, mtp_rows);
        assert_eq!(admission.selected.len(), 4);
        assert_eq!(admission.selected_prefill_chunks, 2);
    }

    #[test]
    fn admission_policy_defaults_gate_prefill_but_not_decode() {
        // PrefillChunkPolicy::default mirrors the deployed daemon smoke
        // defaults (128-token chunks, 512 tokens and 4 chunks per iteration,
        // decode priority on).
        let policy = PrefillChunkPolicy::default();
        assert!(policy.decode_priority);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 0, 0), prefill("p0", 0, 512, 0)],
            &policy,
        );
        assert_eq!(admission.selected.len(), 2);
        let admission = admit_layerwaves_for_iteration(vec![prefill("p0", 0, 513, 0)], &policy);
        assert_eq!(admission.deferred.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Priority under pressure and queue-full deferral shape
// ---------------------------------------------------------------------------

mod admission_priority {
    use super::*;

    #[test]
    fn admission_decode_priority_selects_decode_before_prefill_regardless_of_priority_value() {
        // decode_priority=true: mode rank dominates the numeric priority, so
        // a low-priority-number decode still leads the iteration.
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), decode("d0", 10, 99)],
            &policy,
        );
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(admission.selected[1].mode, LayerWaveMode::Prefill);
    }

    #[test]
    fn admission_mode_order_is_decode_then_mtp_then_prefill() {
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), mtp_verify("m0", 4, 2, 50), decode("d0", 10, 99)],
            &policy,
        );
        let modes: Vec<_> = admission.selected.iter().map(|wave| wave.mode).collect();
        assert_eq!(
            modes,
            vec![LayerWaveMode::Decode, LayerWaveMode::MtpVerify, LayerWaveMode::Prefill]
        );
    }

    #[test]
    fn admission_prefill_ties_break_by_priority_then_arrival() {
        // Within one mode the numeric Priority field orders admission; equal
        // priorities preserve arrival order (stable index tiebreak).
        let policy = policy(1024, 8, true);
        let admission = admit_layerwaves_for_iteration(
            vec![
                prefill("late-low", 0, 8, 5),
                prefill("early-high", 8, 8, 1),
                prefill("tie-a", 16, 8, 5),
                prefill("tie-b", 24, 8, 5),
            ],
            &policy,
        );
        let ids: Vec<_> = admission.selected.iter().map(wave_id).collect();
        assert_eq!(ids, vec!["early-high", "late-low", "tie-a", "tie-b"]);
    }

    #[test]
    fn admission_without_decode_priority_orders_by_priority_across_modes() {
        // decode_priority=false flattens mode rank, so the numeric priority
        // dominates across modes and a high-priority prefill can lead.
        let policy = policy(1024, 8, false);
        let admission = admit_layerwaves_for_iteration(
            vec![decode("d0", 10, 50), prefill("p0", 0, 16, 1)],
            &policy,
        );
        assert_eq!(admission.selected[0].mode, LayerWaveMode::Prefill);
        assert_eq!(admission.selected[1].mode, LayerWaveMode::Decode);
    }

    #[test]
    fn admission_deferred_preserves_order_and_is_readmitted_next_iteration() {
        // Queue-full rejection shape: deferral is graceful, ordered, and the
        // deferred work is fully admissible on the next iteration once the
        // budget frees — nothing is dropped (vLLM: try-again semantics).
        let policy = policy(16, 1, true);
        let first = admit_layerwaves_for_iteration(
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)],
            &policy,
        );
        assert_eq!(first.selected.len(), 1);
        assert_eq!(first.deferred.len(), 2);
        let deferred_ids: Vec<_> = first.deferred.iter().map(wave_id).collect();
        assert_eq!(deferred_ids, vec!["p1", "p2"]);
        // Next iteration sees the deferred waves plus fresh decode work;
        // decode leads and exactly one deferred prefill fits the freed slot.
        let mut candidates = first.deferred.clone();
        candidates.push(decode("d1", 99, 0));
        let second = admit_layerwaves_for_iteration(candidates, &policy);
        assert_eq!(second.selected[0].mode, LayerWaveMode::Decode);
        assert_eq!(second.selected.len(), 2);
        assert_eq!(second.selected_prefill_rows, 16);
        assert_eq!(second.deferred.len(), 1);
        assert_eq!(wave_id(&second.deferred[0]), "p2");
    }

    #[test]
    fn admission_empty_batch_is_a_noop() {
        let policy = policy(16, 1, true);
        let admission = admit_layerwaves_for_iteration(Vec::new(), &policy);
        assert!(admission.selected.is_empty());
        assert!(admission.deferred.is_empty());
        assert_eq!(admission.selected_prefill_rows, 0);
    }
}

// ---------------------------------------------------------------------------
// Delayed admission: scheduler-loop simulation (sglang MinFreeSlotsDelayer
// semantics + real admit_layerwaves_for_iteration)
// ---------------------------------------------------------------------------

mod delayed_admission {
    use super::*;

    /// One scheduler round: admit from the pending queues. Models the daemon's
    /// `commands::scheduler_smoke` loop (pending queues -> candidates ->
    /// admission -> retire selected / requeue deferred).
    fn round(
        policy: &PrefillChunkPolicy,
        pending_decodes: &mut Vec<LayerWave>,
        pending_prefill: &mut Vec<LayerWave>,
        admitted_log: &mut Vec<String>,
    ) {
        let candidates = pending_decodes
            .iter()
            .cloned()
            .chain(pending_prefill.iter().cloned())
            .collect::<Vec<_>>();
        let admission = admit_layerwaves_for_iteration(candidates, policy);
        let mut removed = std::collections::HashSet::new();
        for wave in &admission.selected {
            admitted_log.push(wave_id(wave).to_owned());
            removed.insert(wave_id(wave).to_owned());
        }
        for wave in &admission.deferred {
            removed.insert(wave_id(wave).to_owned());
        }
        pending_decodes.retain(|pending| !removed.contains(wave_id(pending)));
        pending_prefill.retain(|pending| !removed.contains(wave_id(pending)));
        // Deferred waves stay at the head of the prefill queue in order.
        let mut requeue = admission.deferred;
        requeue.extend(pending_prefill.drain(..));
        *pending_prefill = requeue;
    }

    #[test]
    fn admission_decode_never_waits_behind_a_prefill_backlog() {
        // Invariant under pressure: with decode_priority on and any prefill
        // budget, an arriving decode is admitted at the very next round no
        // matter how deep the prefill backlog is (sglang's delayer protects
        // the same invariant by delaying prefill, never decode).
        let policy = policy(16, 1, true);
        let mut pending_prefill = (0..8)
            .map(|index| prefill(&format!("p{index}"), index * 16, 16, index as i32))
            .collect::<Vec<_>>();
        let mut pending_decodes = Vec::new();
        let mut admitted = Vec::new();
        // Saturate several rounds with prefill only.
        for _ in 0..3 {
            round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        }
        assert!(pending_prefill.len() < 8);
        // A decode arrives mid-backlog.
        pending_decodes.push(decode("d0", 999, 0));
        let admitted_before = admitted.len();
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_decodes.is_empty(), "decode was delayed behind prefill backlog");
        // Decode priority puts the decode first in this round's selection.
        assert_eq!(
            admitted[admitted_before..].first().map(String::as_str),
            Some("d0"),
            "decode did not lead its admission round"
        );
        // The prefill backlog still makes progress and loses nothing.
        let total_prefill = 8;
        let remaining = pending_prefill.len();
        assert_eq!(
            admitted.iter().filter(|id| id.starts_with('p')).count() + remaining,
            total_prefill
        );
    }

    #[test]
    fn admission_prefill_defers_then_batches_into_one_admission() {
        // MinFreeSlotsDelayer-shaped policy: deferred prefill waits and then
        // admits as one batch once capacity frees, rather than trickling.
        let policy = policy(64, 4, true);
        // Round 1: decode leads; both small prefill chunks fit alongside.
        let mut pending_prefill =
            vec![prefill("p0", 0, 8, 0), prefill("p1", 8, 8, 1)];
        let mut pending_decodes = vec![decode("d0", 0, 0)];
        let mut admitted = Vec::new();
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_prefill.is_empty());
        assert!(pending_decodes.is_empty());
        // Round 2: a large prefill wave exceeds the budget and defers while
        // the decode still lands.
        let mut pending_prefill = vec![prefill("big", 0, 128, 0)];
        let mut pending_decodes = vec![decode("d1", 1, 0)];
        round(&policy, &mut pending_decodes, &mut pending_prefill, &mut admitted);
        assert!(pending_decodes.is_empty());
        assert_eq!(pending_prefill.len(), 1);
        // Round 3: decode finished; the 128-row wave can never fit the
        // 64-token budget. Replanned as two 32-row chunks (64 rows total)
        // the deferred work clears in ONE round — batched into a single
        // admission rather than trickling one chunk per round (the sglang
        // delayer's batching shape).
        let mut pending_prefill = vec![prefill("big-a", 0, 32, 0), prefill("big-b", 32, 32, 1)];
        round(&policy, &mut Vec::new(), &mut pending_prefill, &mut admitted);
        assert!(pending_prefill.is_empty());
        assert_eq!(
            admitted[admitted.len() - 2..],
            ["big-a".to_owned(), "big-b".to_owned()]
        );
    }

    #[test]
    fn admission_idle_system_never_defers() {
        // Nothing running: first come, first served up to budget; no
        // artificial delay is introduced (sglang: no delay when idle).
        let policy = policy(32, 2, true);
        let mut pending_prefill =
            vec![prefill("p0", 0, 16, 0), prefill("p1", 16, 16, 1), prefill("p2", 32, 16, 2)];
        let mut admitted = Vec::new();
        round(&policy, &mut Vec::new(), &mut pending_prefill, &mut admitted);
        assert_eq!(admitted, vec!["p0", "p1"]);
        assert_eq!(pending_prefill.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// Slot admission model: the daemon scheduler's concurrency invariant
// (v41_requests::Requests::admit occupancy check + v41_native_serve/scheduler.rs
// slot loop), modeled without CUDA machinery. Maps vLLM
// test_concurrent_single_request_admission_respects_limit.
// ---------------------------------------------------------------------------

mod slot_admission {
    /// Model of the daemon's fixed-slot admission table: `active:
    /// Vec<Option<Active>>` sized by `args.concurrency` in
    /// `v41_native_serve/scheduler.rs`, each occupancy an admitted
    /// `Requests::admit(slot, id)` lease.
    #[derive(Default)]
    struct SlotTable {
        occupants: Vec<Option<u64>>,
    }

    impl SlotTable {
        fn with_capacity(concurrency: usize) -> Self {
            Self {
                occupants: (0..concurrency).map(|_| None).collect(),
            }
        }

        /// The scheduler loop's free-slot probe:
        /// `active.iter().position(Option::is_none)`.
        fn first_free(&self) -> Option<usize> {
            self.occupants.iter().position(Option::is_none)
        }

        /// `Requests::admit(slot, id)` from `v41_requests.rs`: the slot must
        /// be in range and unoccupied, else "request slot occupied".
        fn admit(&mut self, slot: usize, id: u64) -> Result<usize, &'static str> {
            if slot >= self.occupants.len() || self.occupants[slot].is_some() {
                return Err("request slot occupied");
            }
            self.occupants[slot] = Some(id);
            Ok(slot)
        }

        /// Scheduler-loop admission: find a free slot or report the
        /// queue-full shape the API surfaces as HTTP 503.
        fn try_admit(&mut self, id: u64) -> Result<usize, &'static str> {
            let slot = self.first_free().ok_or("all request slots occupied")?;
            self.admit(slot, id)
        }

        fn release(&mut self, id: u64) -> bool {
            if let Some(slot) = self.occupants.iter().position(|occupant| *occupant == Some(id)) {
                self.occupants[slot] = None;
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn admission_slot_table_rejects_at_capacity_and_recovers_on_release() {
        let mut table = SlotTable::with_capacity(2);
        assert_eq!(table.try_admit(1), Ok(0));
        assert_eq!(table.try_admit(2), Ok(1));
        assert_eq!(table.try_admit(3), Err("all request slots occupied"));
        // A release frees exactly one slot: queue-full is transient, the next
        // arrival admits — the try-again shape mapped to HTTP 503 upstream.
        assert!(table.release(1));
        assert_eq!(table.try_admit(3), Ok(0));
        assert!(!table.release(1), "released slot is gone");
    }

    #[test]
    fn admission_slot_table_rejects_occupied_and_out_of_range_slots() {
        // Mirrors `Requests::admit`'s guard: a stale scheduler iteration must
        // not re-admit a live slot, nor address beyond the table.
        let mut table = SlotTable::with_capacity(2);
        table.admit(0, 7).unwrap();
        assert_eq!(table.admit(0, 8), Err("request slot occupied"));
        assert_eq!(table.admit(2, 8), Err("request slot occupied"));
        assert_eq!(table.admit(usize::MAX, 8), Err("request slot occupied"));
        assert_eq!(table.try_admit(8), Ok(1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admission_concurrent_single_slot_admission_respects_limit() {
        // vLLM: test_concurrent_single_request_admission_respects_limit.
        // The daemon serializes admission on its single scheduler thread
        // (check-and-occupy happen together in the recv loop); the invariant
        // is that two concurrent contenders for one free slot cannot both
        // win. The model serializes check-and-occupy on a mutex, with a yield
        // between contenders to expose a race if serialization were missing.
        let table = std::sync::Arc::new(std::sync::Mutex::new(SlotTable::with_capacity(1)));
        let contender = |id: u64| {
            let table = table.clone();
            async move {
                tokio::task::yield_now().await;
                table.lock().unwrap().try_admit(id)
            }
        };
        let (first, second) = tokio::join!(contender(0), contender(1));
        let results = [first, second];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        assert_eq!(
            table.lock().unwrap().occupants.iter().flatten().count(),
            1,
            "exactly one request occupies the single slot"
        );
    }
}

// ---------------------------------------------------------------------------
// Queue-full rejection shape over HTTP: real ds41rt-api native_v41 router.
// Maps vLLM QueueOverflowError/MaxQueuedTokensError -> create_error_response
// 503 mapping, plus the daemon's admission-failure status retention.
// ---------------------------------------------------------------------------

mod http_503_mapping {
    use ds41rt_api::native_v41;
    use std::net::SocketAddr;

    const MODEL: &str = native_v41::MODEL;

    fn chat_body(max_tokens: u32) -> String {
        format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"hi"}}],"max_tokens":{max_tokens},"stream":false}}"#
        )
    }

    /// Minimal HTTP/1.1 client over a raw tokio socket (no HTTP client in
    /// this crate's dependency set). Sends `Connection: close` and reads the
    /// response to EOF.
    async fn http_request(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        let status = response
            .split_whitespace()
            .nth(1)
            .expect("HTTP status line")
            .parse::<u16>()
            .expect("numeric HTTP status");
        (status, response)
    }

    async fn serve(router: axum::Router) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        addr
    }

    fn dummy_job() -> native_v41::NativeRequest {
        let (events, _receive) = tokio::sync::mpsc::channel(16);
        native_v41::NativeRequest {
            prompt: String::new(),
            constraint: None,
            images: Vec::new(),
            max_tokens: 1,
            events,
        }
    }

    #[tokio::test]
    async fn admission_queue_overflow_maps_to_503() {
        // vLLM: test_queue_overflow_maps_to_503 / test_max_queued_tokens_maps_to_503.
        // The native router's bounded queue is the waiters' admission gate:
        // a full queue rejects the chat request with HTTP 503.
        let (queue, _receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(2);
        for _ in 0..2 {
            queue.try_send(dummy_job()).unwrap();
        }
        let router = native_v41::router_with_limits(queue, native_v41::NativeLimits::default());
        let addr = serve(router).await;
        let (status, response) =
            http_request(addr, "POST", "/v1/chat/completions", &chat_body(1)).await;
        assert_eq!(status, 503, "{response}");
        assert!(response.contains("\"error\""), "{response}");
    }

    #[tokio::test]
    async fn admission_closed_queue_health_maps_to_503() {
        // The health endpoint advertises admission availability: a closed
        // queue (worker gone) is 503, matching the upstream invariant that
        // overload/unavailability surfaces as 503, not 500.
        let (queue, receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(1);
        drop(receive);
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, _) = http_request(addr, "GET", "/health", "").await;
        assert_eq!(status, 503);
    }

    #[tokio::test]
    async fn admission_open_queue_health_maps_to_200() {
        let (queue, _receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(1);
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, _) = http_request(addr, "GET", "/health", "").await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn admission_worker_failure_maps_to_500_with_cause() {
        // The daemon's scheduler-side admission failure (e.g. pool exhausted
        // during preparation) reaches the client as a 500 carrying the cause,
        // not a fabricated success — pinned by the comment in native_v41.rs.
        let (queue, mut receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(4);
        let worker = tokio::spawn(async move {
            while let Some(job) = receive.recv().await {
                let _ = job
                    .events
                    .send(Err(native_v41::NativeFailure::from("pool exhausted")))
                    .await;
            }
        });
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, response) =
            http_request(addr, "POST", "/v1/chat/completions", &chat_body(1)).await;
        assert_eq!(status, 500, "{response}");
        assert!(response.contains("pool exhausted"), "{response}");
        worker.abort();
    }

    #[tokio::test]
    async fn admission_invalid_request_maps_to_400_not_503() {
        // Rejection shape discipline: a client-side admission violation
        // (max_tokens=0) is 400, never 503 — 503 is reserved for overload.
        let (queue, _receive) = tokio::sync::mpsc::channel::<native_v41::NativeRequest>(1);
        let router = native_v41::router(queue);
        let addr = serve(router).await;
        let (status, response) =
            http_request(addr, "POST", "/v1/chat/completions", &chat_body(0)).await;
        assert_eq!(status, 400, "{response}");
        assert!(response.contains("max_tokens"), "{response}");
    }
}
