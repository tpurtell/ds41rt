//! Exact full-vocabulary target sampling for the native serving path.
//!
//! The production `serve-native` engine selects one token per verified row from
//! the full 129,280-token vocabulary. Greedy selection stays on the device
//! (`cuda_logits_argmax_checked_f32_async`); every stochastic mode routes the
//! finished logit row through this module so the release semantics are defined
//! once, in one place, and can be exercised without a GPU.
//!
//! Filter order, confirmed against vLLM's `v1/sample/sampler.py`:
//!
//! 1. grammar / allowed-token mask (applied first, outside this module)
//! 2. temperature scaling `logit / temperature`
//! 3. `min_p`: keep tokens whose scaled logit is at least
//!    `max_scaled_logit + ln(min_p)`, i.e. probability at least
//!    `min_p * max_probability` (vLLM's argmax-invariant `MinPLogitsProcessor`
//!    runs after temperature and before top-k/top-p; `min_p = 0` is disabled)
//! 4. `top_k`: keep the `k` highest scaled logits (`None` disables the filter;
//!    `k >= vocabulary` is a no-op)
//! 5. `top_p`: keep the smallest descending prefix whose probability mass
//!    reaches `top_p`; `top_p = 1.0` is disabled and keeps everything
//!    (vLLM applies top-k and then top-p inside one passed sampler)
//!
//! The filter chain can never empty the candidate set: the best token survives
//! every stage, and a grammar that allows no token is a hard error rather than
//! a silent fallback to the unmasked distribution.
//!
//! Vocabulary width is discovered from the logit row; nothing here assumes the
//! official 129,280-token checkpoint, so a future vocabulary needs no change.
//!
//! Delta from vLLM on one edge: at a top-k boundary tie this keeps exactly `k`
//! candidates (lowest token id wins the tie), matching ds41rt's existing native
//! sampler. vLLM keeps every token equal to the k-th value. Real logits hit an
//! exact float tie with negligible probability.
//!
//! Determinism: the sampling draw is a pure function of `(seed, position)`, so
//! request compaction, batching and speculative row layout cannot change a
//! token. Callers pass the request's absolute decode position, never a
//! batch-local row index.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use thiserror::Error;

/// Upper bound accepted for `temperature` by the serving protocol.
pub const MAX_TARGET_TEMPERATURE: f32 = 2.0;

/// Temperatures below this are greedy, matching vLLM's `_SAMPLING_EPS`.
pub const GREEDY_TEMPERATURE_EPS: f32 = 1.0e-5;

/// Full-precision upper draw bound, matching the native and Triton samplers.
const MAX_UNIFORM: f32 = 0.999_999_94;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TargetSamplingError {
    #[error("target sampling requires a non-empty vocabulary")]
    EmptyVocabulary,
    #[error("grammar allows no target token")]
    EmptyCandidates,
    #[error("target logit at token {token} is not finite")]
    NonFiniteLogit { token: usize },
    #[error("grammar mask width {actual} does not match vocabulary {vocab}")]
    MaskWidth { actual: usize, vocab: usize },
    #[error("invalid target sampling parameter: {0}")]
    InvalidParameter(&'static str),
}

impl TargetSamplingError {
    /// True when the error is the caller's request rather than a model failure.
    pub fn is_bad_request(&self) -> bool {
        matches!(
            self,
            Self::MaskWidth { .. } | Self::InvalidParameter(_)
        )
    }
}

/// Validated sampling parameters carried by one served request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetSamplingParams {
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    min_p: f32,
    seed: u64,
}

impl Default for TargetSamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

impl TargetSamplingParams {
    /// Served default: greedy, byte-for-byte the pre-sampling behaviour.
    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            seed: 0,
        }
    }

    /// Build validated parameters. `top_k = None` disables the top-k filter;
    /// `min_p = 0` and `top_p = 1.0` are the disabled bounds.
    pub fn new(
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        min_p: f32,
        seed: u64,
    ) -> Result<Self, TargetSamplingError> {
        if !temperature.is_finite() || !(0.0..=MAX_TARGET_TEMPERATURE).contains(&temperature) {
            return Err(TargetSamplingError::InvalidParameter("temperature"));
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err(TargetSamplingError::InvalidParameter("top_p"));
        }
        if top_k == Some(0) {
            return Err(TargetSamplingError::InvalidParameter("top_k"));
        }
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(TargetSamplingError::InvalidParameter("min_p"));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
            min_p,
            seed,
        })
    }

    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    pub const fn top_k(self) -> Option<usize> {
        self.top_k
    }

    pub const fn min_p(self) -> f32 {
        self.min_p
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// ds41rt keeps the OpenAI signed seed convention: every `i64` maps to its
    /// two's-complement `u64` (so `-1` is a real seed, not "unseeded"). This is
    /// an intentional difference from vLLM, which treats only `-1` as unseeded
    /// but otherwise accepts signed seeds.
    pub const fn seed_from_i64(seed: i64) -> u64 {
        seed as u64
    }

    /// Greedy when temperature is effectively zero or top-k collapses to one
    /// token. Both cases are exactly the legacy argmax and must not consume a
    /// random draw. `temperature < 1e-5` mirrors vLLM's `_SAMPLING_EPS`.
    pub fn is_greedy(self) -> bool {
        self.temperature < GREEDY_TEMPERATURE_EPS || self.top_k == Some(1)
    }

    /// Any filter that narrows the distribution beyond raw temperature.
    pub fn has_filters(self) -> bool {
        self.top_k.is_some() || self.min_p > 0.0 || self.top_p < 1.0
    }

    /// Deterministic uniform in `[0, 1)` for a request-local decode position.
    /// SplitMix64 over `(domain, seed, position)`, where `domain` separates the
    /// target sample stream from every draft/proposal RNG: a proposal that
    /// happens to share an integer seed with the target cannot correlate with
    /// the acceptance draw. Draws depend only on the absolute emitted-token
    /// index, so rejected draft rows never shift a future draw.
    pub fn random_uniform(self, position: u64) -> f32 {
        const TARGET_SAMPLING_DOMAIN: u64 = 0x7f4a_7c15_9e37_79b9;
        let mut mixed = self
            .seed
            .wrapping_add(TARGET_SAMPLING_DOMAIN)
            .wrapping_add(position.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        let mantissa = (mixed >> 40) as u32;
        mantissa as f32 * (1.0 / 16_777_216.0)
    }

    /// Select the token for `position`, drawing its uniform deterministically.
    pub fn select_token(
        self,
        logits: &[f32],
        mask: Option<&[u32]>,
        position: u64,
    ) -> Result<usize, TargetSamplingError> {
        self.select_token_with_uniform(logits, mask, self.random_uniform(position))
    }

    /// Select with an explicit uniform. This is the pure entry point used by
    /// tests, the CPU oracle and any future device kernel equivalence check.
    pub fn select_token_with_uniform(
        self,
        logits: &[f32],
        mask: Option<&[u32]>,
        uniform: f32,
    ) -> Result<usize, TargetSamplingError> {
        if logits.is_empty() {
            return Err(TargetSamplingError::EmptyVocabulary);
        }
        if let Some(mask) = mask {
            let expected = logits.len().div_ceil(u32::BITS as usize);
            if mask.len() != expected {
                return Err(TargetSamplingError::MaskWidth {
                    actual: mask.len(),
                    vocab: logits.len(),
                });
            }
        }
        let allowed = |token: usize| -> bool {
            mask.is_none_or(|words| words[token / 32] & (1u32 << (token % 32)) != 0)
        };
        if self.is_greedy() {
            return argmax_allowed(logits, &allowed);
        }
        if !uniform.is_finite() {
            // A non-finite draw has no meaningful position on the CDF; refuse
            // rather than silently clamping it to a token.
            return Err(TargetSamplingError::InvalidParameter("uniform"));
        }
        sample_allowed(self, logits, &allowed, uniform)
    }
}

fn argmax_allowed(
    logits: &[f32],
    allowed: &impl Fn(usize) -> bool,
) -> Result<usize, TargetSamplingError> {
    let mut best = None;
    let mut maximum = f32::NEG_INFINITY;
    for (token, &logit) in logits.iter().enumerate() {
        if !allowed(token) {
            continue;
        }
        if !logit.is_finite() {
            return Err(TargetSamplingError::NonFiniteLogit { token });
        }
        if logit > maximum {
            maximum = logit;
            best = Some(token);
        }
    }
    best.ok_or(TargetSamplingError::EmptyCandidates)
}

/// Above this survivor count the ordered path uses a stable radix sort instead
/// of the comparison sort. Chosen so the comparison sort stays within the small
/// branch's cache-friendly range.
const ORDERED_SORT_CAP: usize = 8192;
const RADIX_BITS: u32 = 16;
const RADIX_BUCKETS: usize = 1 << RADIX_BITS;
const RADIX_MASK: u32 = RADIX_BUCKETS as u32 - 1;

/// One ranked survivor. The unique token id makes the ranking a total order, so
/// the sorted permutation is unique and every ordered branch must reproduce it.
#[derive(Clone, Copy, Debug)]
struct Ranked {
    scaled: f32,
    id: u32,
}

impl Ranked {
    /// Strictly better under the served comparator (scaled desc, id asc).
    fn better_than(self, other: Self) -> bool {
        match self.scaled.partial_cmp(&other.scaled) {
            Some(Ordering::Greater) => true,
            Some(Ordering::Equal) => self.id < other.id,
            _ => false,
        }
    }
}

/// Min-heap entry whose `Ord` puts the WORST element at the root, so a bounded
/// capacity-k heap keeps exactly the k best survivors.
#[derive(Clone, Copy, Debug)]
struct WorstFirst(Ranked);
impl PartialEq for WorstFirst {
    fn eq(&self, other: &Self) -> bool {
        self.0.scaled == other.0.scaled && self.0.id == other.0.id
    }
}
impl Eq for WorstFirst {}
impl PartialOrd for WorstFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WorstFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        // Greater == worse, so the max-heap root is the worst candidate.
        match self.0.scaled.partial_cmp(&other.0.scaled) {
            Some(Ordering::Less) => Ordering::Greater,
            Some(Ordering::Greater) => Ordering::Less,
            Some(Ordering::Equal) => self.0.id.cmp(&other.0.id),
            None => Ordering::Equal,
        }
    }
}

/// The served ranking comparator, byte-identical to the pre-optimization code:
/// `partial_cmp` (not `total_cmp`), descending, with an ascending id tie-break.
fn sort_ranked_descending(ranked: &mut [Ranked]) {
    ranked.sort_unstable_by(|a, b| {
        b.scaled
            .partial_cmp(&a.scaled)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
}

/// Descending monotone 32-bit key for the radix sort. `-0.0` is canonicalized to
/// `+0.0` (the comparator treats them equal), then the standard float-to-ordered
/// map is inverted so larger keys sort first.
fn descending_radix_key(scaled: f32) -> u32 {
    let value = if scaled == 0.0 { 0.0 } else { scaled };
    let bits = value.to_bits();
    let ascending = if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    };
    !ascending
}

#[derive(Default)]
struct RadixScratch {
    temp: Vec<Ranked>,
    counts: Vec<u32>,
}

// Per-thread radix scratch. High-water retention is bounded by the largest row
// sampled on the thread: `temp` holds one `Ranked` (8 B) per survivor (129,280
// tokens => ~1.03 MiB) and `counts` is 65,536 `u32` (256 KiB), ~1.29 MiB total.
// The `RefCell` borrow assumes the selector is not re-entered on the same
// thread; sampling is synchronous and never calls back into itself, so the
// borrow cannot overlap and the buffers are never observed half-written.
thread_local! {
    static RADIX_SCRATCH: RefCell<RadixScratch> = RefCell::new(RadixScratch::default());
}

/// Stable LSD radix sort over the descending 32-bit key. Input is in token-id
/// order, so equal keys keep ascending ids and the output equals the comparison
/// sort's unique permutation.
fn radix_sort_descending(records: &mut [Ranked]) {
    let len = records.len();
    if len < 2 {
        return;
    }
    RADIX_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.counts.len() < RADIX_BUCKETS {
            scratch.counts.resize(RADIX_BUCKETS, 0);
        }
        let mut temp = std::mem::take(&mut scratch.temp);
        temp.clear();
        temp.resize(len, Ranked { scaled: 0.0, id: 0 });
        radix_scatter(records, &mut temp, &mut scratch.counts, 0);
        radix_scatter(&temp, records, &mut scratch.counts, RADIX_BITS);
        scratch.temp = temp;
    });
}

fn radix_scatter(src: &[Ranked], dst: &mut [Ranked], counts: &mut [u32], shift: u32) {
    let buckets = &mut counts[..RADIX_BUCKETS];
    buckets.fill(0);
    for record in src {
        buckets[((descending_radix_key(record.scaled) >> shift) & RADIX_MASK) as usize] += 1;
    }
    let mut running = 0u32;
    for count in buckets.iter_mut() {
        let bucket = *count;
        *count = running;
        running += bucket;
    }
    for record in src {
        let bucket = ((descending_radix_key(record.scaled) >> shift) & RADIX_MASK) as usize;
        dst[buckets[bucket] as usize] = *record;
        buckets[bucket] += 1;
    }
}

struct RankedSample {
    id: usize,
    /// Internals compared by the exactness oracle; unread in production builds.
    #[cfg_attr(not(test), allow(dead_code))]
    total: f32,
    #[cfg_attr(not(test), allow(dead_code))]
    nucleus_count: usize,
}

fn sample_allowed(
    params: TargetSamplingParams,
    logits: &[f32],
    allowed: &impl Fn(usize) -> bool,
    uniform: f32,
) -> Result<usize, TargetSamplingError> {
    Ok(sample_allowed_internals(params, logits, allowed, uniform)?.id)
}

fn sample_allowed_internals(
    params: TargetSamplingParams,
    logits: &[f32],
    allowed: &impl Fn(usize) -> bool,
    uniform: f32,
) -> Result<RankedSample, TargetSamplingError> {
    let inv_temperature = 1.0 / params.temperature;
    let mut max_scaled = f32::NEG_INFINITY;
    let mut allowed_count = 0usize;
    for (token, &logit) in logits.iter().enumerate() {
        if !allowed(token) {
            continue;
        }
        if !logit.is_finite() {
            return Err(TargetSamplingError::NonFiniteLogit { token });
        }
        allowed_count += 1;
        max_scaled = max_scaled.max(logit * inv_temperature);
    }
    if allowed_count == 0 {
        // A grammar that allows no token is reported as such, not as a
        // temperature error from the -inf maximum.
        return Err(TargetSamplingError::EmptyCandidates);
    }
    if !max_scaled.is_finite() {
        // Only reachable for pathologically small positive temperatures.
        return Err(TargetSamplingError::InvalidParameter("temperature"));
    }
    // min_p: keep p >= min_p * max_p, equivalently scaled >= max_scaled + ln(min_p).
    let min_scaled = if params.min_p > 0.0 {
        max_scaled + params.min_p.ln()
    } else {
        f32::NEG_INFINITY
    };

    let survivors = |token: usize| -> bool {
        allowed(token) && logits[token] * inv_temperature >= min_scaled
    };
    let uniform = MAX_UNIFORM.min(uniform.max(0.0));

    // Fast exact path: with no top-k and a disabled top-p nucleus there is
    // nothing to order, so sample the surviving categorical directly.
    if params.top_k.is_none() && params.top_p >= 1.0 {
        let (id, total) = sample_categorical(logits, &survivors, inv_temperature, max_scaled, uniform)?;
        return Ok(RankedSample { id, total, nucleus_count: 0 });
    }

    let mut survivor_count = 0usize;
    for token in 0..logits.len() {
        if survivors(token) {
            survivor_count += 1;
        }
    }

    // Branch 1: bounded top-k selection. A capacity-k heap of the k best, then
    // the served comparator sorts only those k, which equals `full[..k]`.
    if let Some(top_k) = params.top_k {
        if top_k < survivor_count {
            let mut heap: BinaryHeap<WorstFirst> = BinaryHeap::with_capacity(top_k);
            for token in 0..logits.len() {
                if !survivors(token) {
                    continue;
                }
                let candidate = Ranked {
                    scaled: logits[token] * inv_temperature,
                    id: token as u32,
                };
                if heap.len() < top_k {
                    heap.push(WorstFirst(candidate));
                } else if candidate.better_than(heap.peek().unwrap().0) {
                    heap.pop();
                    heap.push(WorstFirst(candidate));
                }
            }
            let mut ranked: Vec<Ranked> = heap.into_iter().map(|entry| entry.0).collect();
            sort_ranked_descending(&mut ranked);
            return Ok(sample_from_ranked(&ranked, max_scaled, params.top_p, uniform));
        }
        // `top_k >= survivor_count` truncates nothing; fall through to the full
        // ordering branches.
    }

    let mut ranked: Vec<Ranked> = Vec::with_capacity(survivor_count.min(ORDERED_SORT_CAP));
    for token in 0..logits.len() {
        if survivors(token) {
            ranked.push(Ranked {
                scaled: logits[token] * inv_temperature,
                id: token as u32,
            });
        }
    }

    if survivor_count <= ORDERED_SORT_CAP {
        // Branch 2: small sort, unchanged comparator.
        sort_ranked_descending(&mut ranked);
    } else {
        // Branch 3: stable radix sort over the descending key.
        #[cfg(test)]
        note_branch3_call();
        radix_sort_descending(&mut ranked);
    }
    Ok(sample_from_ranked(&ranked, max_scaled, params.top_p, uniform))
}

/// Shared weights/total/nucleus/draw tail used by every ordered branch. The
/// operation order is byte-identical to the pre-optimization inline block.
fn sample_from_ranked(
    ranked: &[Ranked],
    max_scaled: f32,
    top_p: f32,
    uniform: f32,
) -> RankedSample {
    debug_assert!(!ranked.is_empty(), "best token always survives filters");
    let mut weights: Vec<f32> = ranked
        .iter()
        .map(|record| (record.scaled - max_scaled).exp())
        .collect();
    let total: f32 = weights.iter().copied().sum();
    let total = total.max(1.0e-20);
    for weight in &mut weights {
        *weight /= total;
    }

    // top_p nucleus over the (optionally top-k-truncated) ranked list.
    let top_p = top_p.clamp(1.0e-6, 1.0);
    let mut nucleus_mass = 0.0f32;
    let mut nucleus_count = 0usize;
    for &weight in &weights {
        nucleus_mass += weight;
        nucleus_count += 1;
        if nucleus_mass >= top_p {
            break;
        }
    }
    let nucleus_mass = nucleus_mass.max(1.0e-20);
    let target = uniform * nucleus_mass;
    let mut cumulative = 0.0f32;
    let mut selected = nucleus_count - 1;
    for (rank, &weight) in weights.iter().enumerate().take(nucleus_count) {
        cumulative += weight;
        if target <= cumulative {
            selected = rank;
            break;
        }
    }
    RankedSample {
        id: ranked[selected].id as usize,
        total,
        nucleus_count,
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only count of radix-branch entries on the current test thread.
    static BRANCH3_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn branch3_calls() -> usize {
    BRANCH3_CALLS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn note_branch3_call() {
    BRANCH3_CALLS.with(|calls| calls.set(calls.get() + 1));
}

/// Exact categorical draw over an arbitrary allowed subset in token order.
fn sample_categorical(
    logits: &[f32],
    allowed: &impl Fn(usize) -> bool,
    inv_temperature: f32,
    max_scaled: f32,
    uniform: f32,
) -> Result<(usize, f32), TargetSamplingError> {
    let mut total = 0.0f32;
    for (token, &logit) in logits.iter().enumerate() {
        if allowed(token) {
            total += (logit * inv_temperature - max_scaled).exp();
        }
    }
    let total = total.max(1.0e-20);
    let target = uniform * total;
    let mut cumulative = 0.0f32;
    let mut last = None;
    for (token, &logit) in logits.iter().enumerate() {
        if !allowed(token) {
            continue;
        }
        last = Some(token);
        cumulative += (logit * inv_temperature - max_scaled).exp();
        if target <= cumulative {
            return Ok((token, total));
        }
    }
    last.map(|token| (token, total))
        .ok_or(TargetSamplingError::EmptyCandidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 8;

    fn params(
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        min_p: f32,
    ) -> TargetSamplingParams {
        TargetSamplingParams::new(temperature, top_p, top_k, min_p, 0).unwrap()
    }

    fn mask_for(allowed: &[usize]) -> Vec<u32> {
        let mut mask = vec![0u32; VOCAB.div_ceil(32)];
        for &token in allowed {
            mask[token / 32] |= 1 << (token % 32);
        }
        mask
    }

    #[test]
    fn greedy_matches_argmax_and_lowest_tie() {
        let logits = [1.0, 5.0, 3.0, 5.0, 0.0, -1.0, 2.0, 0.5];
        assert_eq!(
            TargetSamplingParams::greedy()
                .select_token(&logits, None, 0)
                .unwrap(),
            1
        );
        // top_k = 1 is the same argmax and never draws.
        assert_eq!(
            params(1.0, 1.0, Some(1), 0.0)
                .select_token(&logits, None, 7)
                .unwrap(),
            1
        );
        // temperature = 0 is greedy even with filters set.
        assert_eq!(
            params(0.0, 0.01, Some(4), 0.9)
                .select_token(&logits, None, 7)
                .unwrap(),
            1
        );
    }

    #[test]
    fn temperature_below_vllm_epsilon_is_greedy() {
        let logits = [1.0, 5.0, 3.0, 5.0, 0.0, -1.0, 2.0, 0.5];
        // vLLM's _SAMPLING_EPS is 1e-5: anything below it collapses to argmax.
        let tiny = params(1.0e-6, 0.01, Some(3), 0.9);
        assert!(tiny.is_greedy());
        assert_eq!(tiny.select_token(&logits, None, 11).unwrap(), 1);
        // At or above the epsilon sampling is active.
        assert!(!params(GREEDY_TEMPERATURE_EPS, 1.0, None, 0.0).is_greedy());
    }

    #[test]
    fn top_k_at_or_above_vocabulary_is_a_no_op() {
        let logits = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sampler = params(1.0, 1.0, Some(VOCAB), 0.0);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen.len(), VOCAB, "k == vocab must keep the full support");
        // Larger than vocab is also accepted and remains a no-op.
        assert!(params(1.0, 1.0, Some(VOCAB * 4), 0.0)
            .select_token(&logits, None, 0)
            .is_ok());
    }

    #[test]
    fn signed_seeds_map_to_twos_complement_and_stay_deterministic() {
        assert_eq!(
            TargetSamplingParams::seed_from_i64(-1),
            u64::MAX
        );
        assert_eq!(TargetSamplingParams::seed_from_i64(0), 0);
        assert_eq!(TargetSamplingParams::seed_from_i64(i64::MIN), 1u64 << 63);
        // A negative request seed is a real, reproducible stream.
        let negative = TargetSamplingParams::new(
            1.0,
            1.0,
            None,
            0.0,
            TargetSamplingParams::seed_from_i64(-7),
        )
        .unwrap();
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        for position in 0..32 {
            assert_eq!(
                negative.select_token(&logits, None, position).unwrap(),
                negative.select_token(&logits, None, position).unwrap()
            );
        }
    }

    #[test]
    fn zero_seed_is_explicit_and_not_treated_as_unset() {
        // seed 0 and seed 1 are distinct streams; seed 0 must not be a sentinel
        // that the caller silently replaces.
        let zero = TargetSamplingParams::new(1.0, 1.0, None, 0.0, 0).unwrap();
        let one = zero.with_seed(1);
        assert_ne!(zero.seed(), one.seed());
        assert_eq!(zero.seed(), 0);
    }

    #[test]
    fn min_p_one_retains_every_tied_maximum() {
        // Three tokens share the exact maximum; min_p = 1 must keep all of
        // them, not collapse the tie to one token.
        let logits = [5.0f32, 5.0, 5.0, 0.0, -1.0, -2.0, -3.0, -4.0];
        let sampler = params(1.0, 1.0, None, 1.0);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen, [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn tiny_but_non_greedy_temperature_is_finite_and_confident() {
        // 1e-4 is above vLLM's 1e-5 greedy epsilon but still very peaked. The
        // scaled-logit path must stay finite (no inf/inf NaN).
        let logits = [12.5f32, -3.0, 7.25, 0.0, -8.0, 1.0, 2.0, 3.0];
        let sampler = params(1.0e-4, 1.0, None, 0.0);
        assert!(!sampler.is_greedy());
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            let token = sampler
                .select_token_with_uniform(&logits, None, uniform)
                .unwrap();
            assert_eq!(token, 0, "tiny temperature must be effectively one-hot");
        }
        // A pathologically large logit at a tiny temperature is reported as an
        // error, never as NaN or a panic.
        let extreme = [f32::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let error = sampler
            .select_token_with_uniform(&extreme, None, 0.5)
            .unwrap_err();
        assert_eq!(error, TargetSamplingError::InvalidParameter("temperature"));
    }

    #[test]
    fn non_finite_uniform_is_rejected_in_stochastic_mode_only() {
        let logits = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let sampler = params(1.0, 1.0, None, 0.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                sampler
                    .select_token_with_uniform(&logits, None, bad)
                    .unwrap_err(),
                TargetSamplingError::InvalidParameter("uniform")
            );
        }
        // Greedy never consumes the draw, so the same uniform is irrelevant.
        assert_eq!(
            TargetSamplingParams::greedy()
                .select_token_with_uniform(&logits, None, f32::NAN)
                .unwrap(),
            7
        );
    }

    #[test]
    fn greedy_respects_mask_and_errors_when_empty() {
        let logits = [1.0, 5.0, 3.0, 5.0, 0.0, -1.0, 2.0, 0.5];
        let mask = mask_for(&[2, 3]);
        assert_eq!(
            TargetSamplingParams::greedy()
                .select_token(&logits, Some(&mask), 0)
                .unwrap(),
            3
        );
        let empty = [0u32; 1];
        assert_eq!(
            TargetSamplingParams::greedy()
                .select_token(&logits, Some(&empty), 0)
                .unwrap_err(),
            TargetSamplingError::EmptyCandidates
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        for (temperature, top_p, top_k, min_p) in [
            (2.5, 1.0, None, 0.0),
            (-0.1, 1.0, None, 0.0),
            (1.0, 0.0, None, 0.0),
            (1.0, 1.5, None, 0.0),
            (1.0, 1.0, Some(0), 0.0),
            (1.0, 1.0, None, -0.1),
            (1.0, 1.0, None, 1.5),
        ] {
            assert!(
                TargetSamplingParams::new(temperature, top_p, top_k, min_p, 0).is_err(),
                "({temperature}, {top_p}, {top_k:?}, {min_p}) must be rejected"
            );
        }
        assert!(TargetSamplingParams::new(2.0, 1.0, Some(1), 1.0, 0).is_ok());
    }

    #[test]
    fn temperature_only_sampling_covers_the_whole_support() {
        let logits = [1.0, 5.0, 3.0, 5.0, 0.0, -1.0, 2.0, 0.5];
        let sampler = params(1.0, 1.0, None, 0.0);
        // A near-zero uniform lands on the first token in token order; a
        // near-one uniform lands on the last. No filter may remove any token.
        assert_eq!(sampler.select_token_with_uniform(&logits, None, 0.0).unwrap(), 0);
        assert_eq!(
            sampler
                .select_token_with_uniform(&logits, None, MAX_UNIFORM)
                .unwrap(),
            7
        );
        // Monotonic sweep: every token id is reachable.
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=10_000 {
            let uniform = step as f32 / 10_000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen.len(), VOCAB);
    }

    #[test]
    fn min_p_threshold_is_relative_to_the_best_scaled_logit() {
        // ln-scale probabilities 0.5, 0.25, 0.25.
        let logits = [0.0f32, -std::f32::consts::LN_2, -std::f32::consts::LN_2];
        // min_p = 0.6 keeps only the dominant token (others are 0.5 * max).
        let sampler = params(1.0, 1.0, None, 0.6);
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            assert_eq!(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
                0
            );
        }
        // min_p = 0.4 keeps all three (the boundary survives: p >= min_p * max).
        let sampler = params(1.0, 1.0, None, 0.4);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen, [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn top_k_restricts_support_to_the_highest_scaled_logits() {
        let logits = [0.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
        let sampler = params(1.0, 1.0, Some(2), 0.0);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen, [1, 2].into_iter().collect());
    }

    #[test]
    fn top_p_keeps_the_smallest_descending_nucleus() {
        // Softmax([0, ln .25, ln .25, ln .5]) = [.5, .125, .125, .25].
        let logits = [0.0f32, (0.25f32).ln(), (0.25f32).ln(), (0.5f32).ln()];
        // top_p = 0.5 is reached by token 0 alone (mass .5).
        let strict = params(1.0, 0.5, None, 0.0);
        let mut strict_seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            strict_seen.insert(
                strict
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(strict_seen, [0].into_iter().collect());
        // top_p = 0.7 needs {0, 3}: .5 + .25 = .75 >= .7, excluding .125 tokens.
        let sampler = params(1.0, 0.7, None, 0.0);
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            seen.insert(
                sampler
                    .select_token_with_uniform(&logits, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen, [0, 3].into_iter().collect());
    }

    #[test]
    fn mask_is_applied_before_sampling_filters() {
        let logits = [9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Only tokens 2 and 3 are legal; a naive filter-first sampler could
        // pick 0 or 1. The mask must win.
        let mask = mask_for(&[2, 3]);
        let sampler = params(1.0, 1.0, None, 0.0);
        for step in 0..=1000 {
            let uniform = step as f32 / 1000.0;
            let token = sampler
                .select_token_with_uniform(&logits, Some(&mask), uniform)
                .unwrap();
            assert!(token == 2 || token == 3, "picked illegal token {token}");
        }
    }

    #[test]
    fn filters_never_empty_the_candidate_set() {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        // min_p = 1 keeps only the best; an extremely small top_p still keeps it.
        for sampler in [
            params(1.0, 1.0, None, 1.0),
            params(1.0, 1.0e-6, None, 0.0),
            params(1.0, 1.0e-6, Some(1), 1.0),
        ] {
            for step in 0..=100 {
                let uniform = step as f32 / 100.0;
                assert_eq!(
                    sampler
                        .select_token_with_uniform(&logits, None, uniform)
                        .unwrap(),
                    7
                );
            }
        }
    }

    #[test]
    fn draws_are_deterministic_in_seed_and_position() {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let a = TargetSamplingParams::new(1.0, 1.0, None, 0.0, 1234).unwrap();
        let b = a.with_seed(1234);
        for position in 0..64 {
            assert_eq!(
                a.select_token(&logits, None, position).unwrap(),
                b.select_token(&logits, None, position).unwrap()
            );
        }
        // seed = 0 is a real seed, not "unset".
        let zero = TargetSamplingParams::new(1.0, 1.0, None, 0.0, 0).unwrap();
        let first = zero.select_token(&logits, None, 0).unwrap();
        assert_eq!(zero.select_token(&logits, None, 0).unwrap(), first);
        // A different seed changes the stream on at least one position.
        let other = zero.with_seed(1);
        assert!((0..64).any(|position| {
            zero.select_token(&logits, None, position).unwrap()
                != other.select_token(&logits, None, position).unwrap()
        }));
    }

    #[test]
    fn uniform_depends_on_position_not_on_batch_layout() {
        let sampler = TargetSamplingParams::new(1.0, 1.0, None, 0.0, 99).unwrap();
        // The draw at absolute position p is fixed regardless of which row it
        // occupies in a batch: sampling uses only (seed, position).
        for position in [0u64, 1, 2, 7, 31, 128] {
            let expected = sampler.random_uniform(position);
            assert_eq!(sampler.random_uniform(position), expected);
        }
    }

    #[test]
    fn non_finite_allowed_logits_are_rejected() {
        let mut logits = [0.0f32; VOCAB];
        logits[3] = f32::NAN;
        assert_eq!(
            params(1.0, 1.0, None, 0.0)
                .select_token(&logits, None, 0)
                .unwrap_err(),
            TargetSamplingError::NonFiniteLogit { token: 3 }
        );
        // A masked non-finite token is never read.
        let mask = mask_for(&[0, 1, 2]);
        assert!(params(1.0, 1.0, None, 0.0)
            .select_token(&logits, Some(&mask), 0)
            .is_ok());
    }

    #[test]
    fn empty_mask_is_reported_as_empty_candidates_in_every_mode() {
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let empty = [0u32; 1];
        // An all-zero mask leaves no candidate. The stochastic path must not
        // report this as a temperature error via the -inf maximum.
        for sampler in [
            params(1.0, 1.0, None, 0.0),
            params(0.7, 0.9, Some(3), 0.1),
            params(1.0, 1.0e-6, Some(2), 1.0),
        ] {
            assert_eq!(
                sampler.select_token(&logits, Some(&empty), 0).unwrap_err(),
                TargetSamplingError::EmptyCandidates
            );
            assert_eq!(
                sampler
                    .select_token_with_uniform(&logits, Some(&empty), 0.5)
                    .unwrap_err(),
                TargetSamplingError::EmptyCandidates
            );
        }
    }

    #[test]
    fn mask_width_must_match_vocabulary() {
        let logits = [0.0f32; VOCAB];
        let wide = vec![u32::MAX; 2];
        assert_eq!(
            params(1.0, 1.0, None, 0.0)
                .select_token(&logits, Some(&wide), 0)
                .unwrap_err(),
            TargetSamplingError::MaskWidth {
                actual: 2,
                vocab: VOCAB
            }
        );
    }

    #[test]
    fn stochastic_selection_matches_reference_distribution() {
        // Chi-square-free distribution check: a wide uniform sweep must track
        // the analytic softmax within a tight total-variation bound.
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let sampler = params(1.0, 1.0, None, 0.0);
        let draws = 100_000;
        let mut counts = [0usize; 4];
        for step in 0..draws {
            let uniform = (step as f32 + 0.5) / draws as f32;
            counts[sampler
                .select_token_with_uniform(&logits, None, uniform)
                .unwrap()] += 1;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let expected: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let total_variation: f32 = (0..4)
            .map(|token| ((counts[token] as f32 / draws as f32) - expected[token]).abs())
            .sum::<f32>()
            / 2.0;
        assert!(
            total_variation < 0.01,
            "empirical distribution drifted: tv={total_variation} counts={counts:?}"
        );
    }

    /// Pinned-logit regression: the production speculative path reuses
    /// `verify_dspark_greedy` with *sampled* target rows (sample-and-match). Over
    /// several rounds with a controllable mismatch, the emitted sequence must
    /// equal sequential autoregressive sampling at the same absolute positions.
    /// This pins: the all-match bonus, the first-mismatch emission, that rejected
    /// rows never shift a later draw, and that constrained rows use the
    /// hypothetical grammar mask.
    #[test]
    fn speculative_sample_match_equals_sequential_sampling() {
        use crate::verify_dspark_greedy;
        const SEQ_VOCAB: usize = 6;
        // Never selected below, so no early eos stop complicates the sequence.
        const EOS: u32 = 999;

        fn logits(position: usize) -> Vec<f32> {
            (0..SEQ_VOCAB)
                .map(|token| (((position * 7 + token * 13) % 11) as f32) * 0.5)
                .collect()
        }
        fn mask_for(allowed: &[usize]) -> Vec<u32> {
            let mut mask = vec![0u32; SEQ_VOCAB.div_ceil(32)];
            for &token in allowed {
                mask[token / 32] |= 1 << (token % 32);
            }
            mask
        }

        for (label, seed, constrained) in
            [("unconstrained", 11u64, false), ("constrained", 29u64, true)]
        {
            let params = TargetSamplingParams::new(0.9, 0.97, Some(4), 0.02, seed).unwrap();
            let mask_at = |position: usize| -> Option<Vec<u32>> {
                if constrained {
                    Some(mask_for(&[position % SEQ_VOCAB, (position + 1) % SEQ_VOCAB]))
                } else {
                    None
                }
            };
            let total = 24usize;

            // Sequential reference: one exact draw per absolute position.
            let sequential: Vec<u32> = (0..total)
                .map(|position| {
                    params
                        .select_token(
                            &logits(position),
                            mask_at(position).as_deref(),
                            position as u64,
                        )
                        .unwrap() as u32
                })
                .collect();

            let mut emitted = Vec::new();
            let mut position = 0usize;
            let mut rounds = 0usize;
            while position < total {
                let width = (total - position).min(4);
                let selected: Vec<u32> = (0..width)
                    .map(|index| {
                        let at = position + index;
                        params
                            .select_token(&logits(at), mask_at(at).as_deref(), at as u64)
                            .unwrap() as u32
                    })
                    .collect();
                // `verify_dspark_greedy` compares input[i+1] (the draft for the
                // token after row i) with target row i, so a perfect draft is
                // `input[0] = anchor`, `input[1..] = selected[..width-1]`.
                let mut input = vec![0u32; width];
                input[0] = if position == 0 { 0 } else { sequential[position - 1] };
                input[1..].copy_from_slice(&selected[..width - 1]);
                // Even rounds propose a perfect draft (all-match bonus); odd
                // rounds corrupt one later proposal to force a first mismatch.
                let mismatch = (rounds % 2 == 1)
                    .then(|| 1 + rounds % width.saturating_sub(1).max(1));
                if let Some(index) = mismatch.filter(|index| *index < width) {
                    input[index] = (input[index] + 1) % SEQ_VOCAB as u32;
                }
                let decision =
                    verify_dspark_greedy(&input, &selected, EOS, total - position).unwrap();
                let expected_len = mismatch.filter(|index| *index < width).unwrap_or(width);
                assert_eq!(
                    decision.emitted.len(),
                    expected_len,
                    "{label} round {rounds}: wrong emitted width"
                );
                assert_eq!(
                    decision.accepted_inputs as usize,
                    expected_len,
                    "{label} round {rounds}: wrong accepted prefix"
                );
                assert_eq!(
                    decision.emitted,
                    sequential[position..position + expected_len].to_vec(),
                    "{label} round {rounds}: speculative tokens diverged from sequential"
                );
                if constrained {
                    for (index, &token) in decision.emitted.iter().enumerate() {
                        let mask = mask_at(position + index).unwrap();
                        assert!(
                            mask[token as usize / 32] & (1 << (token % 32)) != 0,
                            "{label}: emitted token {token} violates the row grammar"
                        );
                    }
                }
                emitted.extend_from_slice(&decision.emitted);
                // Only emitted tokens advance the request; rejected rows must
                // leave the next absolute position unchanged.
                position += decision.emitted.len();
                rounds += 1;
                assert!(rounds <= total, "{label}: speculative loop did not progress");
            }
            assert_eq!(emitted, sequential, "{label}: full sequence diverged");
            assert!(rounds >= 6, "{label}: bonus rounds never exercised a multi-token commit");
        }
    }

    // -----------------------------------------------------------------
    // Ordered-path optimization: bit-identity against the pre-change oracle
    // -----------------------------------------------------------------

    /// Verbatim pre-optimization implementation, retained only as the exactness
    /// oracle for the optimized branches.
    struct Reference {
        token: usize,
        total: Option<f32>,
        nucleus_count: Option<usize>,
        /// Uniform thresholds: rank `j` is selected while `uniform <=
        /// boundaries[j]`.
        boundaries: Vec<f32>,
    }

    fn mask_of(vocab: usize, allowed: impl Fn(usize) -> bool) -> Vec<u32> {
        let mut mask = vec![0u32; vocab.div_ceil(32)];
        for token in 0..vocab {
            if allowed(token) {
                mask[token / 32] |= 1 << (token % 32);
            }
        }
        mask
    }

    fn reference_select(
        params: TargetSamplingParams,
        logits: &[f32],
        mask: Option<&[u32]>,
        uniform: f32,
    ) -> Result<Reference, TargetSamplingError> {
        let vocab = logits.len();
        if vocab == 0 {
            return Err(TargetSamplingError::EmptyVocabulary);
        }
        if let Some(mask) = mask {
            let expected = vocab.div_ceil(u32::BITS as usize);
            if mask.len() != expected {
                return Err(TargetSamplingError::MaskWidth {
                    actual: mask.len(),
                    vocab,
                });
            }
        }
        let allowed = |token: usize| {
            mask.is_none_or(|words| words[token / 32] & (1u32 << (token % 32)) != 0)
        };
        if params.is_greedy() {
            let mut best = None;
            let mut maximum = f32::NEG_INFINITY;
            for (token, &logit) in logits.iter().enumerate() {
                if !allowed(token) {
                    continue;
                }
                if !logit.is_finite() {
                    return Err(TargetSamplingError::NonFiniteLogit { token });
                }
                if logit > maximum {
                    maximum = logit;
                    best = Some(token);
                }
            }
            return best
                .map(|token| Reference {
                    token,
                    total: None,
                    nucleus_count: None,
                    boundaries: Vec::new(),
                })
                .ok_or(TargetSamplingError::EmptyCandidates);
        }
        let inv_temperature = 1.0 / params.temperature;
        let mut max_scaled = f32::NEG_INFINITY;
        let mut allowed_count = 0usize;
        for (token, &logit) in logits.iter().enumerate() {
            if !allowed(token) {
                continue;
            }
            if !logit.is_finite() {
                return Err(TargetSamplingError::NonFiniteLogit { token });
            }
            allowed_count += 1;
            max_scaled = max_scaled.max(logit * inv_temperature);
        }
        if allowed_count == 0 {
            return Err(TargetSamplingError::EmptyCandidates);
        }
        if !max_scaled.is_finite() {
            return Err(TargetSamplingError::InvalidParameter("temperature"));
        }
        let min_scaled = if params.min_p > 0.0 {
            max_scaled + params.min_p.ln()
        } else {
            f32::NEG_INFINITY
        };
        let survivors =
            |token: usize| allowed(token) && logits[token] * inv_temperature >= min_scaled;
        let uniform = MAX_UNIFORM.min(uniform.max(0.0));
        if params.top_k.is_none() && params.top_p >= 1.0 {
            let mut total = 0.0f32;
            for token in 0..vocab {
                if survivors(token) {
                    total += (logits[token] * inv_temperature - max_scaled).exp();
                }
            }
            let total = total.max(1.0e-20);
            let target = uniform * total;
            let mut cumulative = 0.0f32;
            let mut last = None;
            for token in 0..vocab {
                if !survivors(token) {
                    continue;
                }
                last = Some(token);
                cumulative += (logits[token] * inv_temperature - max_scaled).exp();
                if target <= cumulative {
                    return Ok(Reference {
                        token,
                        total: Some(total),
                        nucleus_count: None,
                        boundaries: Vec::new(),
                    });
                }
            }
            return last
                .map(|token| Reference {
                    token,
                    total: Some(total),
                    nucleus_count: None,
                    boundaries: Vec::new(),
                })
                .ok_or(TargetSamplingError::EmptyCandidates);
        }
        let mut ranked: Vec<(f32, u32)> = Vec::new();
        for token in 0..vocab {
            if survivors(token) {
                ranked.push((logits[token] * inv_temperature, token as u32));
            }
        }
        ranked.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        if let Some(top_k) = params.top_k {
            ranked.truncate(top_k.min(ranked.len()));
        }
        let mut weights: Vec<f32> = ranked
            .iter()
            .map(|(scaled, _)| (scaled - max_scaled).exp())
            .collect();
        let total: f32 = weights.iter().copied().sum();
        let total = total.max(1.0e-20);
        for weight in &mut weights {
            *weight /= total;
        }
        let top_p = params.top_p.clamp(1.0e-6, 1.0);
        let mut nucleus_mass = 0.0f32;
        let mut nucleus_count = 0usize;
        for &weight in &weights {
            nucleus_mass += weight;
            nucleus_count += 1;
            if nucleus_mass >= top_p {
                break;
            }
        }
        let nucleus_mass = nucleus_mass.max(1.0e-20);
        let mut cumulative = 0.0f32;
        let mut boundaries = Vec::with_capacity(nucleus_count);
        for rank in 0..nucleus_count {
            cumulative += weights[rank];
            boundaries.push(cumulative / nucleus_mass);
        }
        let target = uniform * nucleus_mass;
        let mut cumulative = 0.0f32;
        let mut selected = nucleus_count - 1;
        for rank in 0..nucleus_count {
            cumulative += weights[rank];
            if target <= cumulative {
                selected = rank;
                break;
            }
        }
        Ok(Reference {
            token: ranked[selected].1 as usize,
            total: Some(total),
            nucleus_count: Some(nucleus_count),
            boundaries,
        })
    }

    fn compare_to_reference(
        label: &str,
        params: TargetSamplingParams,
        logits: &[f32],
        mask: Option<&[u32]>,
        uniform: f32,
    ) {
        let actual = params.select_token_with_uniform(logits, mask, uniform);
        match reference_select(params, logits, mask, uniform) {
            Ok(reference) => {
                let token = actual.unwrap_or_else(|error| {
                    panic!("{label} params={params:?} uniform={uniform}: optimized {error}, reference {}", reference.token)
                });
                assert_eq!(
                    token, reference.token,
                    "{label} params={params:?} uniform={uniform} vocab={}",
                    logits.len()
                );
                // Compare production internals whenever the oracle exposes any
                // (ordered path: total + nucleus_count; fast path: total only).
                if reference.nucleus_count.is_some() || reference.total.is_some() {
                    let allowed = |token: usize| {
                        mask.is_none_or(|words: &[u32]| {
                            words[token / 32] & (1u32 << (token % 32)) != 0
                        })
                    };
                    let internals =
                        sample_allowed_internals(params, logits, &allowed, uniform)
                            .expect("optimized internals");
                    if let Some(nucleus_count) = reference.nucleus_count {
                        assert_eq!(
                            internals.nucleus_count, nucleus_count,
                            "{label} params={params:?} uniform={uniform} nucleus_count"
                        );
                    }
                    if let Some(total) = reference.total {
                        assert_eq!(
                            internals.total, total,
                            "{label} params={params:?} uniform={uniform} total"
                        );
                    }
                }
            }
            Err(reference_error) => assert_eq!(
                actual.expect_err("optimized must fail like the reference"),
                reference_error,
                "{label} params={params:?} uniform={uniform}"
            ),
        }
    }

    fn adversarial_rows(vocab: usize) -> Vec<(&'static str, Vec<f32>)> {
        let mut rows: Vec<(&'static str, Vec<f32>)> = Vec::new();
        rows.push(("all_equal", vec![0.0; vocab]));
        rows.push((
            "two_value",
            (0..vocab)
                .map(|index| if index % 2 == 0 { 0.0 } else { 5.0 })
                .collect(),
        ));
        rows.push((
            "geometric",
            (0..vocab).map(|index| -(index as f32) * 0.25).collect(),
        ));
        let mut huge_gap = vec![-1.0f32; vocab];
        huge_gap[0] = 60.0;
        rows.push(("huge_gap", huge_gap));
        let mut dominant = vec![-2.0f32; vocab];
        dominant[vocab / 2] = 40.0;
        rows.push(("single_dominant", dominant));
        let mut ties = vec![0.0f32; vocab];
        for token in 0..vocab.min(5) {
            ties[token] = 10.0;
        }
        rows.push(("ties_at_max", ties));
        let mut mixed = vec![0.0f32; vocab];
        for (index, value) in mixed.iter_mut().enumerate() {
            *value = match index % 6 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(1), // subnormal
                3 => -(index as f32),
                4 => (index as f32) * 0.5,
                _ => -1.0,
            };
        }
        rows.push(("negpos_zero_subnormal", mixed));
        rows.push((
            "negative_only",
            (0..vocab).map(|index| -(1.0 + index as f32 * 0.01)).collect(),
        ));
        rows.push(("flat_negative", vec![-3.0; vocab]));
        rows
    }

    fn param_grid(vocab: usize) -> Vec<TargetSamplingParams> {
        let temperatures = [1.0e-5f32, 1.0e-4, 0.2, 0.7, 2.0];
        let top_ks = [
            Some(1usize),
            Some(2),
            Some(40),
            vocab.checked_sub(1),
            Some(vocab),
            Some(vocab + 1),
            None,
        ];
        let top_ps = [1.0e-6f32, 0.5, 0.9, 0.95, 1.0 - 1.0e-7, 1.0];
        let min_ps = [0.0f32, 1.0e-6, 0.05, 0.5, 1.0];
        let mut grid = Vec::new();
        for temperature in temperatures {
            for top_k in top_ks {
                for top_p in top_ps {
                    for min_p in min_ps {
                        if let Ok(params) =
                            TargetSamplingParams::new(temperature, top_p, top_k, min_p, 0)
                        {
                            grid.push(params);
                        }
                    }
                }
            }
        }
        grid
    }

    #[test]
    fn optimized_paths_are_bit_identical_to_reference_on_adversarial_rows() {
        let vocab = 33usize;
        let rows = adversarial_rows(vocab);
        let grid = param_grid(vocab);
        let masks: Vec<Option<Vec<u32>>> = vec![
            None,
            Some(mask_of(vocab, |token| token < 5)),
            Some(mask_of(vocab, |token| token % 3 != 0)),
        ];
        let uniforms = [0.0f32, 0.25, 0.5, 0.75, MAX_UNIFORM, 1.0];
        let mut comparisons = 0usize;
        for (name, logits) in &rows {
            for params in &grid {
                for mask in &masks {
                    for &uniform in &uniforms {
                        compare_to_reference(name, *params, logits, mask.as_deref(), uniform);
                        comparisons += 1;
                    }
                    // Probe the exact reference cumulative boundaries.
                    if let Ok(reference) =
                        reference_select(*params, logits, mask.as_deref(), 0.0)
                    {
                        for &boundary in reference.boundaries.iter().take(12) {
                            for delta in [-1.0e-6f32, 0.0, 1.0e-6] {
                                let uniform = (boundary + delta).clamp(0.0, MAX_UNIFORM);
                                compare_to_reference(
                                    name,
                                    *params,
                                    logits,
                                    mask.as_deref(),
                                    uniform,
                                );
                                comparisons += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            comparisons > 50_000,
            "property sweep covered too few cases: {comparisons}"
        );
    }

    #[test]
    fn large_vocabulary_paths_match_reference_at_boundaries() {
        for vocab in [1usize, 32, 33, 100, 127, 129_280, 129_281] {
            let rows = adversarial_rows(vocab);
            // A reduced grid keeps the 129k-row reference calls tractable.
            let mut grid = vec![
                TargetSamplingParams::greedy(),
                TargetSamplingParams::new(1.0, 1.0, None, 0.0, 0).unwrap(),
                TargetSamplingParams::new(0.7, 0.9, None, 0.0, 0).unwrap(),
                TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 0).unwrap(),
                TargetSamplingParams::new(0.7, 0.9, None, 0.05, 0).unwrap(),
                TargetSamplingParams::new(2.0, 0.999_999_9, None, 0.5, 0).unwrap(),
            ];
            if let Some(k) = vocab.checked_sub(1).filter(|k| *k > 0) {
                grid.push(TargetSamplingParams::new(1.0, 0.95, Some(k), 0.0, 0).unwrap());
            }
            for (name, logits) in rows.iter().take(3) {
                for params in &grid {
                    for uniform in [0.0f32, 0.5, MAX_UNIFORM] {
                        compare_to_reference(name, *params, logits, None, uniform);
                    }
                }
            }
            // Masked variants at this vocabulary width.
            let mask = mask_of(vocab, |token| token % 2 == 0);
            for params in grid.iter().take(3) {
                for uniform in [0.25f32, 0.75] {
                    compare_to_reference("masked", *params, &rows[0].1, Some(&mask), uniform);
                }
            }
        }
    }

    #[test]
    fn seeded_profiles_replay_identically_to_reference() {
        let vocab = 256usize;
        let logits: Vec<f32> = (0..vocab)
            .map(|index| {
                let base = ((index * 37 % 101) as f32) * 0.1 - 5.0;
                if index < 3 {
                    base + 20.0
                } else {
                    base
                }
            })
            .collect();
        let profiles = [
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 1).unwrap(),
            TargetSamplingParams::new(0.7, 0.9, None, 0.0, 1).unwrap(),
            TargetSamplingParams::new(0.2, 0.95, None, 0.0, 1).unwrap(),
            TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 1).unwrap(),
            TargetSamplingParams::greedy(),
        ];
        for params in profiles {
            for position in 0..64u64 {
                let uniform = params.random_uniform(position);
                compare_to_reference("seeded", params, &logits, None, uniform);
            }
        }
    }

    #[test]
    fn large_ordered_path_is_reached_and_never_silently_capped() {
        let vocab = ORDERED_SORT_CAP * 2 + 7;
        let flat = vec![0.0f32; vocab];
        let ordered = TargetSamplingParams::new(1.0, 0.9, None, 0.0, 0).unwrap();
        let before = branch3_calls();
        let token = ordered.select_token(&flat, None, 0).unwrap();
        let after = branch3_calls();
        assert!(
            after > before,
            "a flat row above the cap must take the radix branch"
        );
        assert!(token < vocab);

        // top_p = 1 / no top_k uses the fast path, so the radix branch is not
        // entered and every token stays reachable.
        let full = TargetSamplingParams::new(1.0, 1.0, None, 0.0, 0).unwrap();
        let before = branch3_calls();
        let mut seen = std::collections::BTreeSet::new();
        for step in 0..=20_000 {
            let uniform = step as f32 / 20_000.0;
            seen.insert(
                full.select_token_with_uniform(&flat, None, uniform)
                    .unwrap(),
            );
        }
        assert_eq!(seen.len(), vocab, "flat top_p=1 must reach every token");
        assert_eq!(
            branch3_calls(),
            before,
            "top_p=1 / no top_k must stay on the fast path"
        );
    }

    #[test]
    fn mask_survivor_counts_around_the_cap_and_empty_mask() {
        let vocab = ORDERED_SORT_CAP * 2;
        let flat = vec![0.0f32; vocab];
        let params = TargetSamplingParams::new(1.0, 0.9, None, 0.0, 0).unwrap();
        for allowed in [
            ORDERED_SORT_CAP - 1,
            ORDERED_SORT_CAP,
            ORDERED_SORT_CAP + 1,
            vocab,
        ] {
            let mask = mask_of(vocab, |token| token < allowed);
            assert!(
                params.select_token(&flat, Some(&mask), 0).is_ok(),
                "mask with {allowed} survivors must sample"
            );
        }
        let empty = vec![0u32; vocab.div_ceil(32)];
        assert_eq!(
            params.select_token(&flat, Some(&empty), 0).unwrap_err(),
            TargetSamplingError::EmptyCandidates
        );
        // Branch choice must follow the survivor count, not the vocabulary.
        let before = branch3_calls();
        let small = mask_of(vocab, |token| token < 64);
        let _ = params.select_token(&flat, Some(&small), 0).unwrap();
        assert_eq!(
            branch3_calls(),
            before,
            "a small masked survivor set must not use the radix branch"
        );
    }

    /// The exact CAP-1 / CAP / CAP+1 handoff on a flat row, compared to the
    /// oracle on token + total + nucleus_count. This is the only place a
    /// branch-handoff permutation bug could hide.
    #[test]
    fn cap_boundary_oracle_covers_branch_handoff() {
        let vocab = ORDERED_SORT_CAP + 8;
        let flat = vec![0.0f32; vocab];
        let params = TargetSamplingParams::new(1.0, 0.9, None, 0.0, 0).unwrap();
        for survivors in [
            ORDERED_SORT_CAP - 1,
            ORDERED_SORT_CAP,
            ORDERED_SORT_CAP + 1,
        ] {
            let mask = mask_of(vocab, |token| token < survivors);
            let before = branch3_calls();
            for uniform in [0.0f32, 0.37, 0.5, 0.93, MAX_UNIFORM] {
                compare_to_reference("cap", params, &flat, Some(&mask), uniform);
            }
            assert_eq!(
                branch3_calls() > before,
                survivors > ORDERED_SORT_CAP,
                "survivors={survivors}: radix branch selection wrong"
            );
        }
    }

    /// CAP < S < V with an interleaved mask (S = 64,640 over the real
    /// vocabulary): the radix branch must still match the oracle exactly.
    #[test]
    fn mid_range_survivor_count_matches_reference() {
        let vocab = 129_280usize;
        let flat = vec![0.0f32; vocab];
        let mask = mask_of(vocab, |token| token % 2 == 0);
        let params = TargetSamplingParams::new(1.0, 0.9, None, 0.0, 0).unwrap();
        let before = branch3_calls();
        for uniform in [0.0f32, 0.41, 0.5, 0.87, MAX_UNIFORM] {
            compare_to_reference("mid", params, &flat, Some(&mask), uniform);
        }
        assert!(
            branch3_calls() > before,
            "S > CAP must take the radix branch"
        );
    }

    /// `logit = -f32::MAX` is finite but scales to `-inf` at temperature 1e-5;
    /// `min_p = 0` keeps it. Both the radix (no top_k) and heap (top_k=100)
    /// branches must match the oracle and not panic.
    #[test]
    fn negative_infinity_survivor_ordered_paths_match_reference() {
        let vocab = ORDERED_SORT_CAP * 2 + 1;
        let mut logits = vec![0.0f32; vocab];
        logits[vocab - 1] = -f32::MAX;
        let radix = TargetSamplingParams::new(1.0e-5, 0.9, None, 0.0, 0).unwrap();
        let heap = TargetSamplingParams::new(1.0e-5, 0.9, Some(100), 0.0, 0).unwrap();
        let before = branch3_calls();
        for uniform in [0.0f32, 0.5, MAX_UNIFORM] {
            compare_to_reference("-inf-radix", radix, &logits, None, uniform);
            compare_to_reference("-inf-heap", heap, &logits, None, uniform);
        }
        assert!(
            branch3_calls() > before,
            "the no-top_k -inf variant must enter the radix branch"
        );
    }
}
