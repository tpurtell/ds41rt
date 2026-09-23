//! Online dSpark verification-length policy.
//!
//! Each round, a lane verifies an anchor row plus a chosen number of draft
//! rows per request. The policy chooses those lengths to maximize expected
//! committed tokens per unit of predicted lane time. Work that each committed
//! token needs anyway (its own rows) is paid once whichever round verifies the
//! token, so the ratio's argmax equals minimizing fixed plus wasted resource
//! per committed token.
//!
//! Time is priced from the resources that are actually consumed:
//!
//! * per layer `l >= 1`, `alpha + beta * rows + (1 / bandwidth) * megabytes`,
//!   where megabytes is the routed-expert weight traffic of that layer. The
//!   traffic is `expert_bytes(l) * sum_e ceil(routes_e / 16)`: the grouped
//!   slice kernels read each expert's slice once per 16-row group. Local RTX
//!   layers and remote Spark layers are separate resource classes;
//! * per round, a residual `A + B * rows + C * requests` for everything else
//!   (draft, embedding and layer 0, vocabulary head, sampling, commit).
//!
//! Every coefficient is fitted online from the lane's own timings, with
//! exponential forgetting and Huber-weighted residuals, around weak physical
//! priors. Changing the quantization or tensor-parallel width changes the
//! known byte counts; the fitted bandwidths absorb clocks, thermal state and
//! kernel efficiency. No offline calibration is used.
//!
//! Expert traffic for unexecuted draft rows is forecast from each request's
//! own recent committed routes: draft row `k` stands in for the `k`-th most
//! recent committed token, averaged over several shifted windows. Cross-request
//! sharing within a lane falls out of the exact multiset union.
//!
//! This module performs no device work.
use std::collections::{BTreeMap, VecDeque};

pub const DSPARK_LAYERS: usize = 40;
const EXPERTS: usize = 384;
const TOPK: usize = 6;
const GROUP_ROWS: u8 = 16;
/// Committed tokens retained per request.
const HISTORY: usize = 24;
/// Shifted windows averaged by the forecast.
const WINDOWS: usize = 4;
/// Layer samples each resource class needs before the policy engages.
const WARM_LAYER_SAMPLES: u64 = 120;
/// Round samples needed before the policy engages.
const WARM_ROUND_SAMPLES: u64 = 12;
/// Per-sample forgetting of the calibration curvature: memory of roughly 500
/// reached samples per position, so the fit tracks drift.
const CALIBRATION_DECAY: f64 = 0.998;
/// Initial curvature (prior strength) of each position's calibration.
const CALIBRATION_PRIOR: f64 = 2.;
/// Per-round weight of the mean prediction-residual correction.
const BIAS_RATE: f64 = 0.02;

/// Execution resource of one layer's routed experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DsparkLayerClass {
    /// Coordinator RTX (TP1 or TP2).
    Local = 0,
    /// Spark expert ranks behind the network boundary.
    Remote = 1,
}

impl DsparkLayerClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }
}

/// Installed placement: resource class and per-device slice bytes of one
/// expert, for every layer.
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkPlacement {
    class: [DsparkLayerClass; DSPARK_LAYERS],
    expert_bytes: [f64; DSPARK_LAYERS],
}

impl DsparkPlacement {
    pub fn new(
        class: [DsparkLayerClass; DSPARK_LAYERS],
        expert_bytes: [f64; DSPARK_LAYERS],
    ) -> Result<Self, &'static str> {
        if expert_bytes.iter().any(|b| !b.is_finite() || *b <= 0.) {
            return Err("expert slice bytes must be finite and positive");
        }
        Ok(Self { class, expert_bytes })
    }
    pub fn class(&self, layer: usize) -> DsparkLayerClass {
        self.class[layer]
    }
    pub fn expert_bytes(&self, layer: usize) -> f64 {
        self.expert_bytes[layer]
    }
    fn layers_in(&self, class: DsparkLayerClass) -> usize {
        (1..DSPARK_LAYERS).filter(|&l| self.class[l] == class).count()
    }
}

/// Exponentially forgotten, prior-regularized, Huber-weighted least squares
/// with nonnegative coefficients.
#[derive(Clone, Debug)]
struct Estimator<const D: usize> {
    decay: f64,
    sxx: [[f64; D]; D],
    sxy: [f64; D],
    prior_mean: [f64; D],
    prior_precision: [f64; D],
    theta: [f64; D],
    scale: f64,
    samples: u64,
}

impl<const D: usize> Estimator<D> {
    fn new(decay: f64, prior_mean: [f64; D], prior_precision: [f64; D]) -> Self {
        Self {
            decay,
            sxx: [[0.; D]; D],
            sxy: [0.; D],
            prior_mean,
            prior_precision,
            theta: prior_mean,
            scale: 0.,
            samples: 0,
        }
    }
    fn predict(&self, x: &[f64; D]) -> f64 {
        x.iter().zip(&self.theta).map(|(x, t)| x * t).sum()
    }
    fn observe(&mut self, x: [f64; D], y: f64) {
        if !y.is_finite() || y < 0. || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        let residual = y - self.predict(&x);
        let magnitude = residual.abs();
        let settled = self.samples >= 8 && self.scale > 0.;
        // Graph capture, prefill interleave and scheduler stalls produce rare
        // large positive residuals; they must not drag the fit.
        let weight = if settled && magnitude > 3. * self.scale {
            3. * self.scale / magnitude
        } else {
            1.
        };
        self.scale = if self.samples == 0 {
            magnitude.max(1.)
        } else {
            let clipped = if settled { magnitude.min(6. * self.scale) } else { magnitude };
            (0.97 * self.scale + 0.03 * clipped).max(1e-3)
        };
        for i in 0..D {
            for j in 0..D {
                self.sxx[i][j] = self.decay * self.sxx[i][j] + weight * x[i] * x[j];
            }
            self.sxy[i] = self.decay * self.sxy[i] + weight * x[i] * y;
        }
        self.samples += 1;
        self.solve();
    }
    /// Solve the regularized normal equations, holding coefficients that would
    /// turn negative at zero (active set, at most D passes).
    fn solve(&mut self) {
        let mut fixed = [false; D];
        for _ in 0..D {
            let mut a = [[0.; D]; D];
            let mut b = [0.; D];
            for i in 0..D {
                if fixed[i] {
                    a[i][i] = 1.;
                    continue;
                }
                for j in 0..D {
                    if !fixed[j] {
                        a[i][j] = self.sxx[i][j];
                    }
                }
                a[i][i] += self.prior_precision[i];
                b[i] = self.sxy[i] + self.prior_precision[i] * self.prior_mean[i];
            }
            let Some(solution) = solve_linear(a, b) else { return };
            match (0..D).find(|&i| !fixed[i] && solution[i] < 0.) {
                Some(i) => fixed[i] = true,
                None => {
                    self.theta = solution;
                    return;
                }
            }
        }
    }
}

fn solve_linear<const D: usize>(mut a: [[f64; D]; D], mut b: [f64; D]) -> Option<[f64; D]> {
    for column in 0..D {
        let pivot = (column..D).max_by(|&x, &y| a[x][column].abs().total_cmp(&a[y][column].abs()))?;
        if a[pivot][column].abs() < 1e-300 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        for row in column + 1..D {
            let factor = a[row][column] / a[column][column];
            for k in column..D {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = [0.; D];
    for row in (0..D).rev() {
        let tail: f64 = (row + 1..D).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Online Platt scaling `logit' = a * logit + b` by an online Newton step on
/// the log-loss, with a decayed Fisher-information matrix: effectively a
/// running maximum-likelihood fit over the last few hundred samples.
#[derive(Clone, Copy, Debug)]
struct Platt {
    theta: [f64; 2],
    information: [[f64; 2]; 2],
}

impl Platt {
    fn new() -> Self {
        Self { theta: [1., 0.], information: [[CALIBRATION_PRIOR, 0.], [0., CALIBRATION_PRIOR]] }
    }
    fn logit(probability: f64) -> f64 {
        let p = probability.clamp(1e-6, 1. - 1e-6);
        (p / (1. - p)).ln()
    }
    fn apply(&self, probability: f64) -> f64 {
        let z = self.theta[0] * Self::logit(probability) + self.theta[1];
        1. / (1. + (-z).exp())
    }
    fn observe(&mut self, probability: f64, outcome: f64) {
        let x = [Self::logit(probability), 1.];
        let p = self.apply(probability);
        let curvature = (p * (1. - p)).max(0.02);
        for i in 0..2 {
            for j in 0..2 {
                self.information[i][j] = CALIBRATION_DECAY * self.information[i][j] + curvature * x[i] * x[j];
            }
            // Keep the matrix well conditioned when one direction is unexcited
            // (constant raw confidence).
            self.information[i][i] = self.information[i][i].max(1e-3);
        }
        let [[a, b], [c, d]] = self.information;
        let determinant = a * d - b * c;
        if !(determinant.is_finite() && determinant > 1e-12) { return; }
        let gradient = [(outcome - p) * x[0], outcome - p];
        let step = [(d * gradient[0] - b * gradient[1]) / determinant,
            (a * gradient[1] - c * gradient[0]) / determinant];
        // Slope in [0, 3]: calibration may flatten but never invert confidence.
        self.theta[0] = (self.theta[0] + step[0].clamp(-1., 1.)).clamp(0., 3.);
        self.theta[1] = (self.theta[1] + step[1].clamp(-2., 2.)).clamp(-8., 8.);
    }
}

/// Fitted coefficients for one regime, exported for monitoring.
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkCostSnapshot {
    /// Per class: intercept µs, µs per row, µs per MB, samples, residual scale µs.
    pub layers: [[f64; 5]; 2],
    /// Round residual: intercept µs, µs per row, µs per request, samples, scale.
    pub round: [f64; 5],
}

#[derive(Clone, Debug)]
struct CostModel {
    /// [regime][class], x = [1, rows, MB].
    layer: [[Estimator<3>; 2]; 2],
    /// [regime], x = [1, rows, requests].
    round: [Estimator<3>; 2],
    layers_in: [usize; 2],
}

impl CostModel {
    fn new(placement: &DsparkPlacement) -> Self {
        // Physical priors (RTX PRO 6000 local slices at ~1.2 TB/s effective,
        // GB10 Spark slices at ~175 GB/s effective) worth a hundredth of one
        // sample: they only resolve directions the data cannot, such as the
        // intercept/row split while every round has the same shape. Rows and
        // expert traffic are strongly correlated, so any stronger prior biases
        // the fitted bandwidth along that near-null direction.
        let n0 = 0.01;
        let local = Estimator::new(0.998, [700., 15., 0.8], [n0, n0 * 36., n0 * 300f64.powi(2)]);
        let remote = Estimator::new(0.998, [900., 15., 5.7], [n0, n0 * 36., n0 * 100f64.powi(2)]);
        let round = Estimator::new(0.98, [8000., 300., 200.], [n0, n0 * 36., n0]);
        Self {
            layer: [[local.clone(), remote.clone()], [local, remote]],
            round: [round.clone(), round],
            layers_in: [
                placement.layers_in(DsparkLayerClass::Local),
                placement.layers_in(DsparkLayerClass::Remote),
            ],
        }
    }
    fn warm(&self, regime: usize) -> bool {
        self.round[regime].samples >= WARM_ROUND_SAMPLES
            && (0..2).all(|c| self.layers_in[c] == 0 || self.layer[regime][c].samples >= WARM_LAYER_SAMPLES)
    }
    /// The regime's own fit once warm, else the other regime's warm fit.
    fn usable_regime(&self, shared: bool) -> Option<usize> {
        let own = usize::from(shared);
        if self.warm(own) { Some(own) } else if self.warm(1 - own) { Some(1 - own) } else { None }
    }
    fn predict(&self, regime: usize, rows: f64, requests: f64, megabytes: [f64; 2]) -> f64 {
        let mut total = self.round[regime].predict(&[1., rows, requests]);
        for class in 0..2 {
            let theta = &self.layer[regime][class].theta;
            let layers = self.layers_in[class] as f64;
            total += layers * (theta[0] + theta[1] * rows) + theta[2] * megabytes[class];
        }
        total
    }
    /// Marginal µs of one more row plus `megabytes` of new weight traffic.
    fn marginal(&self, regime: usize, megabytes: [f64; 2]) -> f64 {
        let mut total = self.round[regime].theta[1];
        for class in 0..2 {
            let theta = &self.layer[regime][class].theta;
            total += self.layers_in[class] as f64 * theta[1] + theta[2] * megabytes[class];
        }
        total
    }
    fn snapshot(&self, regime: usize) -> DsparkCostSnapshot {
        let pack = |e: &Estimator<3>| [e.theta[0], e.theta[1], e.theta[2], e.samples as f64, e.scale];
        DsparkCostSnapshot {
            layers: [pack(&self.layer[regime][0]), pack(&self.layer[regime][1])],
            round: pack(&self.round[regime]),
        }
    }
}

type TokenRoutes = [[u16; TOPK]; DSPARK_LAYERS];

/// One request's candidate: conditional acceptance probability of each
/// available draft position, in order (sigmoid of the draft confidence).
#[derive(Clone, Copy, Debug)]
pub struct DsparkCandidate<'a> {
    pub id: u64,
    pub confidence: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DsparkSelection {
    /// Draft rows retained per request, excluding the anchor.
    pub lengths: Vec<usize>,
    pub expected_tokens: f64,
    pub predicted_us: f64,
    /// Shapes evaluated along the forward trajectory.
    pub evaluated: usize,
}

/// One verified request of a completed round.
#[derive(Clone, Copy, Debug)]
pub struct DsparkObservedRequest<'a> {
    pub id: u64,
    /// Verifier rows of this request, including its anchor.
    pub rows: usize,
    /// Accepted inputs, including the anchor (at least one).
    pub accepted: usize,
    /// Conditional confidence of the verified draft positions, if known.
    pub confidence: Option<&'a [f64]>,
}

/// Timing and routes of one completed lane round.
#[derive(Clone, Copy, Debug)]
pub struct DsparkRoundObservation<'a> {
    /// Another lane had active requests during this round.
    pub shared: bool,
    pub requests: &'a [DsparkObservedRequest<'a>],
    /// Routes per layer, one entry per lane row in request order.
    pub routes: &'a [Vec<[u32; TOPK]>],
    /// Wall µs of layers 1..40 (index 0 unused), each from the previous
    /// layer's FFN completion to its own.
    pub layer_us: &'a [Option<f64>; DSPARK_LAYERS],
    /// Round wall µs from draft start to commit completion.
    pub total_us: f64,
    /// The policy's prediction for this round's shape, if it made one.
    pub predicted_us: Option<f64>,
}

/// Cumulative serving statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DsparkPolicyStats {
    pub rounds: u64,
    pub selected_rounds: u64,
    pub verified_rows: u64,
    pub verified_drafts: u64,
    pub accepted_drafts: u64,
    pub emitted_requests: u64,
    /// Requests verified with k draft rows, k = 0..=7.
    pub draft_rows: [u64; 8],
    /// Per draft position: times reached with all earlier drafts accepted,
    /// summed conditional confidence there, and acceptances there.
    pub position_reached: [u64; 7],
    /// Summed calibrated confidence where reached (what the policy used).
    pub position_confidence: [f64; 7],
    /// Summed raw drafter confidence where reached.
    pub position_raw_confidence: [f64; 7],
    pub position_accepted: [u64; 7],
    pub predicted_rounds: u64,
    pub prediction_error_us: f64,
    pub prediction_abs_error_us: f64,
    pub observed_us: f64,
}

/// Whole-policy state: route history, cost fit, and statistics.
#[derive(Clone, Debug)]
pub struct DsparkPolicy {
    placement: DsparkPlacement,
    fixed: bool,
    cost: CostModel,
    history: BTreeMap<u64, VecDeque<TokenRoutes>>,
    /// Expected new expert groups per layer contributed by a token whose routes
    /// are not yet known (short histories), from observed novelty.
    novelty: [f64; DSPARK_LAYERS],
    /// Per-position Platt calibration of the drafter's confidence. Confidence
    /// is conditional on earlier positions being accepted, and its reliability
    /// varies by position: past the drafter's native five-token block it
    /// saturates near 0.99 whatever the content, so a slope as well as an
    /// offset is needed. Each position is refitted online from the outcomes
    /// of the rounds that reached it.
    calibration: [Platt; crate::MAX_DSPARK_PROPOSALS],
    /// Mean residual of the predicted round time per regime. The robust fit
    /// tracks typical rounds, while throughput depends on mean time including
    /// stalls; this constant restores the mean without moving the marginals.
    bias: [f64; 2],
    stats: DsparkPolicyStats,
    counts: Vec<[[u8; EXPERTS]; DSPARK_LAYERS]>,
}

impl DsparkPolicy {
    /// `fixed` verifies every available draft but still fits and reports.
    pub fn new(placement: DsparkPlacement, fixed: bool) -> Self {
        Self {
            cost: CostModel::new(&placement),
            placement,
            fixed,
            history: BTreeMap::new(),
            novelty: [2.5; DSPARK_LAYERS],
            calibration: [Platt::new(); crate::MAX_DSPARK_PROPOSALS],
            bias: [0.; 2],
            stats: DsparkPolicyStats::default(),
            counts: vec![[[0; EXPERTS]; DSPARK_LAYERS]; WINDOWS],
        }
    }
    pub fn fixed(&self) -> bool {
        self.fixed
    }
    pub fn placement(&self) -> &DsparkPlacement {
        &self.placement
    }
    pub fn stats(&self) -> &DsparkPolicyStats {
        &self.stats
    }
    pub fn cost_snapshot(&self, shared: bool) -> DsparkCostSnapshot {
        self.cost.snapshot(usize::from(shared))
    }
    pub fn warm(&self, shared: bool) -> bool {
        self.cost.warm(usize::from(shared))
    }
    pub fn release(&mut self, id: u64) {
        self.history.remove(&id);
    }
    /// Learned (slope, offset) on the raw logit, per draft position.
    pub fn calibration(&self) -> [(f64, f64); crate::MAX_DSPARK_PROPOSALS] {
        self.calibration.map(|platt| (platt.theta[0], platt.theta[1]))
    }
    /// Mean prediction-residual correction per regime (solo, shared), µs.
    pub fn time_bias(&self) -> [f64; 2] {
        self.bias
    }
    fn calibrated(&self, position: usize, probability: f64) -> f64 {
        self.calibration[position].apply(probability)
    }

    /// Predicted lane µs for explicit lengths, if the fit is usable.
    pub fn predict(&mut self, shared: bool, ids: &[u64], lengths: &[usize]) -> Option<f64> {
        let regime = self.cost.usable_regime(shared)?;
        self.clear_counts();
        let mut megabytes = [0.; 2];
        for (index, (&id, &length)) in ids.iter().zip(lengths).enumerate() {
            for row in 0..=length {
                let delta = self.add_row(id, row, index);
                megabytes[0] += delta[0];
                megabytes[1] += delta[1];
            }
        }
        let rows = ids.len() + lengths.iter().sum::<usize>();
        Some(self.cost.predict(regime, rows as f64, ids.len() as f64, megabytes) + self.bias[regime])
    }

    /// Choose draft lengths for one lane. Returns `None` while the fit is still
    /// warming up or in fixed mode; the caller then verifies every draft.
    pub fn select(&mut self, shared: bool, candidates: &[DsparkCandidate<'_>])
        -> Result<Option<DsparkSelection>, &'static str> {
        if candidates.is_empty() || candidates.len() > 16 {
            return Err("policy requires one to sixteen requests");
        }
        for candidate in candidates {
            if candidate.confidence.len() > crate::MAX_DSPARK_PROPOSALS
                || candidate.confidence.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
                return Err("invalid conditional draft confidence");
            }
        }
        if self.fixed {
            return Ok(None);
        }
        let Some(regime) = self.cost.usable_regime(shared) else { return Ok(None) };
        // Survival of row k: the product of calibrated conditional acceptance
        // probabilities of positions 1..=k.
        let cumulative: Vec<Vec<f64>> = candidates.iter().map(|c| {
            let mut product = 1.;
            c.confidence.iter().enumerate()
                .map(|(position, &p)| { product *= self.calibrated(position, p); product }).collect()
        }).collect();
        self.clear_counts();
        let mut megabytes = [0.; 2];
        for (index, candidate) in candidates.iter().enumerate() {
            let delta = self.add_row(candidate.id, 0, index);
            megabytes[0] += delta[0];
            megabytes[1] += delta[1];
        }
        let requests = candidates.len() as f64;
        let mut rows = candidates.len();
        let mut lengths = vec![0usize; candidates.len()];
        let mut expected = requests;
        let mut time = self.cost.predict(regime, rows as f64, requests, megabytes) + self.bias[regime];
        let mut best = DsparkSelection {
            lengths: lengths.clone(), expected_tokens: expected, predicted_us: time, evaluated: 1,
        };
        let mut evaluated = 1;
        loop {
            // Forward growth: evaluate every request's next draft row against
            // the current joint shape and take the best resulting ratio. The
            // trajectory continues through temporary losses to the full shape,
            // and the best visited shape is returned.
            let mut choice: Option<(usize, f64, f64)> = None;
            for (index, candidate) in candidates.iter().enumerate() {
                let next = lengths[index] + 1;
                if next > candidate.confidence.len() { continue; }
                let delta = self.peek_row(candidate.id, next);
                let candidate_expected = expected + cumulative[index][next - 1];
                let candidate_time = time + self.cost.marginal(regime, delta);
                evaluated += 1;
                if choice.is_none_or(|(_, e, t)| candidate_expected / candidate_time > e / t) {
                    choice = Some((index, candidate_expected, candidate_time));
                }
            }
            let Some((index, candidate_expected, candidate_time)) = choice else { break };
            lengths[index] += 1;
            rows += 1;
            self.add_row(candidates[index].id, lengths[index], index);
            expected = candidate_expected;
            time = candidate_time;
            // Prefer the longer shape on an exact tie.
            if expected / time >= best.expected_tokens / best.predicted_us {
                best.lengths.clone_from(&lengths);
                best.expected_tokens = expected;
                best.predicted_us = time;
            }
        }
        debug_assert_eq!(rows, candidates.len() + lengths.iter().sum::<usize>());
        best.evaluated = evaluated;
        Ok(Some(best))
    }

    fn clear_counts(&mut self) {
        for window in &mut self.counts {
            for layer in window.iter_mut() { layer.fill(0); }
        }
    }

    /// Route set standing in for row `row` of request `id` in window `window`.
    fn stand_in(&self, id: u64, row: usize, window: usize) -> Option<&TokenRoutes> {
        self.history.get(&id)?.get(window + row)
    }

    /// Megabytes of new weight traffic per class if row `row` of `id` joined
    /// the current shape, averaged over the forecast windows.
    fn peek_row(&self, id: u64, row: usize) -> [f64; 2] {
        let mut total = [0.; 2];
        for window in 0..WINDOWS {
            let routes = self.stand_in(id, row, window);
            for layer in 1..DSPARK_LAYERS {
                let groups = match routes {
                    Some(routes) => routes[layer].iter()
                        .filter(|&&e| self.counts[window][layer][e as usize] % GROUP_ROWS == 0).count() as f64,
                    None if row == 0 => TOPK as f64,
                    None => self.novelty[layer],
                };
                total[self.placement.class[layer] as usize] += groups * self.placement.expert_bytes[layer];
            }
        }
        total.map(|bytes| bytes / WINDOWS as f64 / 1e6)
    }

    fn add_row(&mut self, id: u64, row: usize, _request: usize) -> [f64; 2] {
        let delta = self.peek_row(id, row);
        for window in 0..WINDOWS {
            let Some(routes) = self.history.get(&id).and_then(|h| h.get(window + row)).copied() else { continue };
            for layer in 1..DSPARK_LAYERS {
                for &expert in &routes[layer] {
                    let count = &mut self.counts[window][layer][expert as usize];
                    *count = count.saturating_add(1);
                }
            }
        }
        delta
    }

    /// Record a completed round: timings update the cost fit, accepted rows'
    /// routes extend each request's history, acceptance updates statistics.
    pub fn observe(&mut self, round: DsparkRoundObservation<'_>) -> Result<(), &'static str> {
        let rows: usize = round.requests.iter().map(|r| r.rows).sum();
        if round.routes.len() != DSPARK_LAYERS || round.routes.iter().any(|r| r.len() != rows) {
            return Err("route capture does not match the verified rows");
        }
        if round.requests.iter().any(|r| r.rows == 0 || r.accepted == 0 || r.accepted > r.rows) {
            return Err("invalid verified request extent");
        }
        let regime = usize::from(round.shared);
        let requests = round.requests.len() as f64;
        let mut layer_sum = 0.;
        let mut complete = true;
        for layer in 1..DSPARK_LAYERS {
            let Some(elapsed) = round.layer_us[layer] else { complete = false; continue };
            layer_sum += elapsed;
            let megabytes = route_groups(&round.routes[layer]) * self.placement.expert_bytes[layer] / 1e6;
            let class = self.placement.class[layer] as usize;
            self.cost.layer[regime][class].observe([1., rows as f64, megabytes], elapsed);
        }
        if complete && round.total_us.is_finite() {
            self.cost.round[regime].observe([1., rows as f64, requests], (round.total_us - layer_sum).max(0.));
        }
        let mut offset = 0;
        for request in round.requests {
            let history = self.history.entry(request.id).or_default();
            for row in offset..offset + request.accepted {
                let mut token = [[0u16; TOPK]; DSPARK_LAYERS];
                for layer in 0..DSPARK_LAYERS {
                    for (slot, &expert) in round.routes[layer][row].iter().enumerate() {
                        let expert = expert & 511;
                        if expert as usize >= EXPERTS { return Err("route expert exceeds the model"); }
                        token[layer][slot] = expert as u16;
                    }
                }
                if history.len() >= 3 {
                    for layer in 1..DSPARK_LAYERS {
                        let novel = token[layer].iter()
                            .filter(|e| !history.iter().take(3).any(|t| t[layer].contains(e))).count();
                        self.novelty[layer] = 0.995 * self.novelty[layer] + 0.005 * novel as f64;
                    }
                }
                history.push_front(token);
                history.truncate(HISTORY);
            }
            offset += request.rows;
            let drafts = request.rows - 1;
            let accepted = request.accepted - 1;
            self.stats.verified_drafts += drafts as u64;
            self.stats.accepted_drafts += accepted as u64;
            self.stats.emitted_requests += 1;
            self.stats.draft_rows[drafts.min(7)] += 1;
            if let Some(confidence) = request.confidence {
                // Only positions whose predecessors were all accepted carry
                // evidence about the conditional acceptance probability.
                for position in 0..drafts.min(confidence.len()).min(accepted + 1) {
                    let outcome = f64::from(u8::from(position < accepted));
                    let calibrated = self.calibrated(position, confidence[position]);
                    self.stats.position_reached[position] += 1;
                    self.stats.position_confidence[position] += calibrated;
                    self.stats.position_raw_confidence[position] += confidence[position];
                    self.stats.position_accepted[position] += u64::from(position < accepted);
                    self.calibration[position].observe(confidence[position], outcome);
                }
            }
        }
        self.stats.rounds += 1;
        self.stats.verified_rows += rows as u64;
        if let Some(predicted) = round.predicted_us {
            self.stats.selected_rounds += 1;
            if round.total_us.is_finite() {
                // Bounded step: one stalled round cannot swing the correction.
                let residual = (round.total_us - predicted).clamp(-0.5 * predicted.abs(), 0.5 * predicted.abs());
                self.bias[regime] += BIAS_RATE * residual;
                self.stats.predicted_rounds += 1;
                self.stats.prediction_error_us += predicted - round.total_us;
                self.stats.prediction_abs_error_us += (predicted - round.total_us).abs();
                self.stats.observed_us += round.total_us;
            }
        }
        Ok(())
    }
}

/// Weight-read groups of one layer: each expert's slice is read once per
/// sixteen routed rows.
fn route_groups(rows: &[[u32; TOPK]]) -> f64 {
    let mut counts = [0u16; EXPERTS];
    for route in rows {
        for &expert in route {
            if let Some(count) = counts.get_mut((expert & 511) as usize) { *count += 1; }
        }
    }
    counts.iter().map(|&c| c.div_ceil(u16::from(GROUP_ROWS))).sum::<u16>() as f64
}

/// Expected committed tokens for conditional confidences: the mandatory
/// anchor/bonus plus the cumulative acceptance probability of each draft.
pub fn dspark_expected_tokens(confidence: &[f64]) -> f64 {
    let mut product = 1.;
    1. + confidence.iter().map(|p| { product *= p; product }).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(local: usize) -> DsparkPlacement {
        let class = std::array::from_fn(|l| if l < local { DsparkLayerClass::Local } else { DsparkLayerClass::Remote });
        let bytes = std::array::from_fn(|l| if l < local { 18_800_640. } else { 4_700_160. });
        DsparkPlacement::new(class, bytes).unwrap()
    }

    /// Deterministic synthetic router: each token draws six experts from a
    /// per-layer pool so that neighboring tokens share some experts.
    fn token_routes(token: u64, pool: u32) -> Vec<[u32; 6]> {
        (0..40u64).map(|layer| {
            let mut out = [0u32; 6];
            let mut seed = token.wrapping_mul(0x9e3779b97f4a7c15) ^ layer.wrapping_mul(0xbf58476d1ce4e5b9);
            let mut used = 0;
            while used < 6 {
                seed = seed.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                let expert = (layer as u32 * 7 + (z % pool as u64) as u32) % 384;
                if !out[..used].contains(&expert) { out[used] = expert; used += 1; }
            }
            out
        }).collect()
    }

    /// Synthetic ground truth used to train the estimator.
    struct Truth { alpha: [f64; 2], beta: [f64; 2], us_per_mb: [f64; 2], round: [f64; 3] }

    fn run_round(policy: &mut DsparkPolicy, truth: &Truth, token: &mut u64, rows: usize, accepted: usize,
        predicted: Option<f64>) -> f64 {
        let token_rows: Vec<_> = (0..rows).map(|r| token_routes(*token + r as u64, 24)).collect();
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|l| token_rows.iter().map(|t| t[l]).collect()).collect();
        let mut layer_us = [None; 40];
        let mut total = truth.round[0] + truth.round[1] * rows as f64 + truth.round[2];
        for layer in 1..40 {
            let class = policy.placement.class(layer) as usize;
            let mb = route_groups(&routes[layer]) * policy.placement.expert_bytes(layer) / 1e6;
            let t = truth.alpha[class] + truth.beta[class] * rows as f64 + truth.us_per_mb[class] * mb;
            layer_us[layer] = Some(t);
            total += t;
        }
        let requests = [DsparkObservedRequest { id: 1, rows, accepted, confidence: None }];
        policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
            layer_us: &layer_us, total_us: total, predicted_us: predicted }).unwrap();
        *token += accepted as u64;
        total
    }

    fn truth() -> Truth {
        Truth { alpha: [600., 700.], beta: [8., 12.], us_per_mb: [0.9, 7.5], round: [9000., 250., 150.] }
    }

    #[test]
    fn estimator_recovers_coefficients_and_predicts_rounds() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let truth = truth();
        let mut token = 0;
        for round in 0..400 {
            let rows = 1 + round % 8;
            run_round(&mut policy, &truth, &mut token, rows, rows, None);
        }
        assert!(policy.warm(false));
        let snapshot = policy.cost_snapshot(false);
        let remote = snapshot.layers[1];
        assert!((remote[2] - 7.5).abs() < 0.3, "remote µs/MB {remote:?}");
        assert!((remote[1] - 12.).abs() < 3., "remote µs/row {remote:?}");
        // Prediction of an explicit shape against the same truth.
        let lengths = [4];
        let predicted = policy.predict(false, &[1], &lengths).unwrap();
        let actual = run_round(&mut policy, &truth, &mut token, 5, 5, Some(predicted));
        assert!((predicted - actual).abs() / actual < 0.05, "{predicted} vs {actual}");
    }

    #[test]
    fn noisy_layers_and_capture_outliers_do_not_bias_the_bandwidth() {
        let mut estimator: Estimator<3> = Estimator::new(0.998, [900., 15., 5.7], [0.01, 0.36, 100.]);
        let mut state = 12345u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        for sample in 0..20_000u64 {
            let rows = 1 + sample % 8;
            let groups = 6. + 3. * (rows - 1) as f64 + (uniform() * 6.).floor() - 3.;
            let megabytes = groups * 4.70016;
            let mut y = 700. + 12. * rows as f64 + 7.5 * megabytes;
            y *= 1. + 0.1 * (uniform() - 0.5);
            // One layer in fifty stalls on graph capture or a peer prefill.
            if uniform() < 0.02 { y += 20_000. * uniform(); }
            estimator.observe([1., rows as f64, megabytes], y);
        }
        let [alpha, beta, us_per_mb] = estimator.theta;
        assert!((us_per_mb - 7.5).abs() / 7.5 < 0.08, "{:?}", estimator.theta);
        let predicted = alpha + 6. * beta + 7.5 * 0. + us_per_mb * 21. * 4.70016;
        let truth = 700. + 72. + 7.5 * 21. * 4.70016;
        assert!((predicted - truth).abs() / truth < 0.05, "{predicted} vs {truth}");
    }

    #[test]
    fn certain_drafts_are_always_verified_and_hopeless_ones_trimmed() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        let certain = [1.; 5];
        let selection = policy.select(false, &[DsparkCandidate { id: 1, confidence: &certain }]).unwrap().unwrap();
        assert_eq!(selection.lengths, [5]);
        assert!((selection.expected_tokens - 6.).abs() < 1e-4);
        let hopeless = [0.02; 5];
        let selection = policy.select(false, &[DsparkCandidate { id: 1, confidence: &hopeless }]).unwrap().unwrap();
        assert_eq!(selection.lengths, [0]);
        // A falling confidence curve is cut where cumulative acceptance no longer
        // pays for the next row's new expert traffic.
        let falling = [0.95, 0.9, 0.5, 0.3, 0.2];
        let selection = policy.select(false, &[DsparkCandidate { id: 1, confidence: &falling }]).unwrap().unwrap();
        assert!((1..5).contains(&selection.lengths[0]), "{selection:?}");
    }

    #[test]
    fn selection_matches_exhaustive_search_for_one_request() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        for confidence in [[0.9, 0.8, 0.7, 0.6, 0.5], [0.6, 0.9, 0.9, 0.9, 0.9], [0.99, 0.2, 0.9, 0.9, 0.9]] {
            let selection = policy.select(false, &[DsparkCandidate { id: 1, confidence: &confidence }]).unwrap().unwrap();
            let best = (0..=5).max_by(|&a, &b| {
                let ratio = |k: usize| dspark_expected_tokens(&confidence[..k]) / policy.clone().predict(false, &[1], &[k]).unwrap();
                ratio(a).total_cmp(&ratio(b))
            }).unwrap();
            assert_eq!(selection.lengths, [best], "{confidence:?}");
        }
    }

    #[test]
    fn warmup_and_fixed_mode_verify_everything() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        assert_eq!(policy.select(false, &[DsparkCandidate { id: 1, confidence: &[0.1; 5] }]), Ok(None));
        let mut fixed = DsparkPolicy::new(placement(5), true);
        let truth = truth();
        let mut token = 0;
        for _ in 0..300 { run_round(&mut fixed, &truth, &mut token, 6, 3, None); }
        assert!(fixed.warm(false));
        assert_eq!(fixed.select(false, &[DsparkCandidate { id: 1, confidence: &[0.1; 5] }]), Ok(None));
        assert_eq!(fixed.stats().draft_rows[5], 300);
        assert_eq!(fixed.stats().accepted_drafts, 600);
    }

    #[test]
    fn shared_experts_across_requests_are_charged_once() {
        let mut policy = DsparkPolicy::new(placement(0), false);
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        let layer_us = [None; 40];
        for id in [1, 2] {
            for _ in 0..4 {
                let requests = [DsparkObservedRequest { id, rows: 1, accepted: 1, confidence: None }];
                policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
                    layer_us: &layer_us, total_us: f64::NAN, predicted_us: None }).unwrap();
            }
        }
        policy.clear_counts();
        let first = policy.add_row(1, 0, 0);
        let second = policy.add_row(2, 0, 1);
        assert!(first[1] > 0.);
        assert_eq!(second[1], 0.);
    }

    #[test]
    fn groups_follow_the_sixteen_row_kernel_tiles() {
        let rows = vec![[1, 2, 3, 4, 5, 6]; 17];
        assert_eq!(route_groups(&rows), 12.);
        assert_eq!(route_groups(&rows[..16]), 6.);
    }

    #[test]
    fn confidence_reliability_counts_only_reached_positions() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]; 5]).collect();
        let confidence = [0.9, 0.8, 0.7, 0.6];
        let requests = [DsparkObservedRequest { id: 1, rows: 5, accepted: 2, confidence: Some(&confidence) }];
        policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
            layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None }).unwrap();
        let stats = policy.stats();
        assert_eq!(stats.position_reached, [1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(stats.position_accepted, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(stats.draft_rows[4], 1);
    }

    #[test]
    fn overconfident_positions_are_recalibrated_from_reached_outcomes() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]; 8]).collect();
        // The drafter reports 0.95 at every position; position 1 is truly 0.95,
        // position 6 only 0.8.
        let confidence = [0.95; 7];
        let mut state = 99u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        for round in 0..20_000 {
            if round == 600 {
                // Position 6 has been reached ~460 times: already close.
                assert!(policy.stats().position_reached[5] < 520);
                assert!((policy.calibrated(5, 0.95) - 0.8).abs() < 0.05, "{:?}", policy.calibration());
            }
            let truth = [0.95, 0.95, 0.95, 0.95, 0.95, 0.8, 0.8];
            let accepted = truth.iter().take_while(|&&p| uniform() < p).count();
            let requests = [DsparkObservedRequest { id: 1, rows: 8, accepted: accepted + 1, confidence: Some(&confidence) }];
            policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
                layer_us: &[None; 40], total_us: f64::NAN, predicted_us: None }).unwrap();
        }
        assert!((policy.calibrated(0, 0.95) - 0.95).abs() < 0.02, "{:?}", policy.calibration());
        assert!((policy.calibrated(5, 0.95) - 0.8).abs() < 0.03, "{:?}", policy.calibration());
        // Reliability stats report the calibrated confidence the policy used.
        let stats = policy.stats();
        let mean = stats.position_confidence[5] / stats.position_reached[5] as f64;
        let observed = stats.position_accepted[5] as f64 / stats.position_reached[5] as f64;
        assert!((mean - observed).abs() < 0.03, "{mean} vs {observed}");
    }

    #[test]
    fn saturated_confidence_falls_back_to_its_base_rate() {
        let mut state = 7u64;
        let mut uniform = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 };
        // Uninformative: raw 0.95..0.999 while acceptance is always 0.82.
        let mut flat = Platt::new();
        // Informative: acceptance equals the raw confidence.
        let mut honest = Platt::new();
        for _ in 0..4000 {
            let raw = 0.95 + 0.049 * uniform();
            flat.observe(raw, f64::from(u8::from(uniform() < 0.82)));
            let raw = 0.3 + 0.69 * uniform();
            honest.observe(raw, f64::from(u8::from(uniform() < raw)));
        }
        for raw in [0.95, 0.98, 0.995] {
            assert!((flat.apply(raw) - 0.82).abs() < 0.04, "{raw} -> {} {:?}", flat.apply(raw), flat.theta);
        }
        for raw in [0.4, 0.7, 0.9] {
            assert!((honest.apply(raw) - raw).abs() < 0.05, "{raw} -> {} {:?}", honest.apply(raw), honest.theta);
        }
    }

    #[test]
    fn mean_time_bias_absorbs_skewed_stalls() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        let truth = truth();
        let mut token = 0;
        for round in 0..300 { run_round(&mut policy, &truth, &mut token, 1 + round % 6, 1 + round % 6, None); }
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        for round in 0..2000 {
            let predicted = policy.predict(false, &[1], &[0]).unwrap();
            // Every tenth round stalls by 10 ms: mean excess 1 ms.
            let total = predicted - policy.time_bias()[0] + if round % 10 == 0 { 10_000. } else { 0. };
            let requests = [DsparkObservedRequest { id: 1, rows: 1, accepted: 1, confidence: None }];
            policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
                layer_us: &[None; 40], total_us: total, predicted_us: Some(predicted) }).unwrap();
        }
        assert!((policy.time_bias()[0] - 1000.).abs() < 250., "{:?}", policy.time_bias());
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut policy = DsparkPolicy::new(placement(5), false);
        assert!(policy.select(false, &[]).is_err());
        assert!(policy.select(false, &[DsparkCandidate { id: 1, confidence: &[f64::NAN] }]).is_err());
        assert!(policy.select(false, &[DsparkCandidate { id: 1, confidence: &[1.1] }]).is_err());
        assert!(policy.select(false, &[DsparkCandidate { id: 1, confidence: &[0.5; 8] }]).is_err());
        let requests = [DsparkObservedRequest { id: 1, rows: 2, accepted: 1, confidence: None }];
        let routes: Vec<Vec<[u32; 6]>> = (0..40).map(|_| vec![[1, 2, 3, 4, 5, 6]]).collect();
        assert!(policy.observe(DsparkRoundObservation { shared: false, requests: &requests, routes: &routes,
            layer_us: &[None; 40], total_us: 1., predicted_us: None }).is_err());
        assert!(DsparkPlacement::new([DsparkLayerClass::Remote; 40], [0.; 40]).is_err());
    }
}
