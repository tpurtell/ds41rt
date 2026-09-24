//! Producer side of the live engine console (`GET /` on the API port).
//!
//! Cost contract for the CUDA worker:
//! - No viewer connected: every hook is one relaxed load of the hub's viewer
//!   count (through [`live`]) plus the always-on lifetime totals in [`totals`],
//!   which are a handful of relaxed atomic adds per round.
//! - Viewers connected: a hook builds one small owned event and `try_send`s it
//!   into a bounded channel. It never blocks, never serializes and never
//!   decodes text; a full channel drops the event.
//!
//! The console thread owns everything else: per-request state, token text
//! decoding, JSON encoding, the 50 ms frame cadence and the snapshot a newly
//! connected page starts from.
use crate::v41_backbone_lane::FfnSplit;
use ds41rt_api::native_v41::ConsoleHub;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{atomic::{AtomicU64, Ordering}, Arc, OnceLock};
use std::time::{Duration, Instant};

const FRAME: Duration = Duration::from_millis(50);
const GAUGES_MS: u64 = 250;
const CHANNEL: usize = 8192;
const RECENT: usize = 16;

struct Producer {
    hub: Arc<ConsoleHub>,
    events: SyncSender<Event>,
    base: Instant,
    /// Milliseconds since `base` of the last gauge event.
    gauges_at: AtomicU64,
}

static PRODUCER: OnceLock<Producer> = OnceLock::new();

/// A handle that exists only while at least one console viewer is connected.
#[derive(Clone, Copy)]
pub(crate) struct Live(&'static Producer);

/// The console handle if a viewer is connected; `None` costs one atomic load.
#[inline]
pub(crate) fn live() -> Option<Live> {
    let producer = PRODUCER.get()?;
    (producer.hub.viewers() > 0).then_some(Live(producer))
}

impl Live {
    /// Whether token ids should be captured for the text view.
    #[inline]
    pub fn text(self) -> bool { self.0.hub.text_enabled() }
    #[inline]
    pub fn push(self, event: Event) { let _ = self.0.events.try_send(event); }
    /// True at most every 250 ms; the caller then pushes a [`Gauges`] event.
    pub fn gauges_due(self) -> bool {
        let now = self.0.base.elapsed().as_millis() as u64;
        let last = self.0.gauges_at.load(Ordering::Relaxed);
        now.saturating_sub(last) >= GAUGES_MS
            && self.0.gauges_at.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
    }
}

/// Per-request lifecycle events are rare, so they are sent whenever the console
/// is installed. That keeps the snapshot's request list right for a page that
/// connects in the middle of a long generation.
pub(crate) fn lifecycle(event: Event) {
    if let Some(producer) = PRODUCER.get() { let _ = producer.events.try_send(event); }
}

pub(crate) enum Event {
    Round(Round),
    Prefill(Prefill),
    Admit { id: u64, at: Instant, prompt: u32, cached: u32, max: u32, lane: u8, grammar: bool, images: u16 },
    First { id: u64, at: Instant, token: u32 },
    Retire { id: u64, at: Instant, reason: &'static str, generated: u32 },
    Gauges(Gauges),
}

pub(crate) struct Round {
    pub lane: u8,
    pub shared: bool,
    pub started: Instant,
    pub finished: Instant,
    pub draft_us: u64,
    pub prepare_us: u64,
    pub verify_us: u64,
    /// Device time of layers 0..40 from the policy's CUDA events (NaN: none).
    pub layer_us: Vec<f32>,
    pub ffn: FfnSplit,
    pub requests: Vec<RoundRequest>,
}

pub(crate) struct RoundRequest {
    pub id: u64,
    /// Draft tokens proposed before grammar and length-policy truncation.
    pub drafted: u8,
    /// Draft rows the target verified.
    pub verified: u8,
    /// Draft rows accepted; the next emitted token is the target's own.
    pub accepted: u8,
    pub emitted: u8,
    pub masked: bool,
    pub finished: bool,
    /// Text view only: the proposal after the anchor and the emitted ids.
    pub proposal: Vec<u32>,
    pub emissions: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(crate) enum PrefillKind { Chunk, Single, Continuation, Replay, Restore }
impl PrefillKind {
    fn name(self) -> &'static str {
        match self {
            Self::Chunk => "chunk", Self::Single => "single", Self::Continuation => "continuation",
            Self::Replay => "replay", Self::Restore => "restore",
        }
    }
}

pub(crate) struct Prefill {
    pub kind: PrefillKind,
    pub lane: u8,
    pub index: u32,
    pub of: u32,
    pub rows: u32,
    pub started: Instant,
    pub finished: Instant,
}

impl Prefill {
    /// Record a prefill step that has just completed.
    pub fn done(kind: PrefillKind, lane: usize, index: usize, of: usize, rows: usize, started: Instant) {
        if let Some(live) = live() {
            live.push(Event::Prefill(Prefill { kind, lane: lane as u8, index: index as u32,
                of: of as u32, rows: rows as u32, started, finished: Instant::now() }));
        }
    }
}

#[derive(Default)]
pub(crate) struct Gauges {
    pub lanes: [u8; 2],
    pub queued: u32,
    pub pending: Option<bool>,
    /// Pages of the most utilized compressed-KV source: total, free, held by
    /// active requests, tokens per page row.
    pub kv: Option<[u64; 4]>,
    pub host: Option<Value>,
}

/// Lifetime serving counters. Always on; exported in `/v1/stats` and the console.
pub(crate) mod totals {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ADMITTED: AtomicU64 = AtomicU64::new(0);
    static RETIRED: AtomicU64 = AtomicU64::new(0);
    static INPUT: AtomicU64 = AtomicU64::new(0);
    static CACHED: AtomicU64 = AtomicU64::new(0);
    static PREFILL: AtomicU64 = AtomicU64::new(0);
    static OUTPUT: AtomicU64 = AtomicU64::new(0);
    static ROUNDS: AtomicU64 = AtomicU64::new(0);
    static DRAFTED: AtomicU64 = AtomicU64::new(0);
    static VERIFIED: AtomicU64 = AtomicU64::new(0);
    static ACCEPTED: AtomicU64 = AtomicU64::new(0);

    pub fn admitted(prompt: usize, cached: usize) {
        ADMITTED.fetch_add(1, Relaxed);
        INPUT.fetch_add(prompt as u64, Relaxed);
        CACHED.fetch_add(cached as u64, Relaxed);
    }
    pub fn retired() { RETIRED.fetch_add(1, Relaxed); }
    pub fn prefill(rows: usize) { PREFILL.fetch_add(rows as u64, Relaxed); }
    pub fn output(tokens: usize) { OUTPUT.fetch_add(tokens as u64, Relaxed); }
    pub fn round(drafted: u64, verified: u64, accepted: u64, emitted: u64) {
        ROUNDS.fetch_add(1, Relaxed);
        DRAFTED.fetch_add(drafted, Relaxed);
        VERIFIED.fetch_add(verified, Relaxed);
        ACCEPTED.fetch_add(accepted, Relaxed);
        OUTPUT.fetch_add(emitted, Relaxed);
    }
    pub fn snapshot() -> serde_json::Value {
        serde_json::json!({
            "requests_admitted": ADMITTED.load(Relaxed),
            "requests_retired": RETIRED.load(Relaxed),
            "input_tokens": INPUT.load(Relaxed),
            "cached_input_tokens": CACHED.load(Relaxed),
            "prefill_tokens": PREFILL.load(Relaxed),
            "output_tokens": OUTPUT.load(Relaxed),
            "verification_rounds": ROUNDS.load(Relaxed),
            "drafted_tokens": DRAFTED.load(Relaxed),
            "verified_drafts": VERIFIED.load(Relaxed),
            "accepted_drafts": ACCEPTED.load(Relaxed),
        })
    }
}

/// The per-round counts every path needs, captured before emission consumes
/// the token vectors. Updates the lifetime totals.
pub(crate) struct Tally {
    drafted: [u8; 8],
    verified: [u8; 8],
    accepted: [u8; 8],
    emitted: [u8; 8],
    proposal: Vec<Vec<u32>>,
    emissions: Vec<Vec<u32>>,
}

/// The proposal of each member before truncation, captured right after drafting.
pub(crate) struct Proposal { drafted: [u8; 8], tokens: Vec<Vec<u32>> }
impl Proposal {
    #[inline]
    pub fn capture(inputs: &[Vec<u32>], live: Option<Live>) -> Self {
        let mut drafted = [0u8; 8];
        for (slot, input) in drafted.iter_mut().zip(inputs) { *slot = input.len().saturating_sub(1) as u8; }
        let tokens = if live.is_some_and(Live::text) {
            inputs.iter().map(|input| input.get(1..).unwrap_or_default().to_vec()).collect()
        } else { Vec::new() };
        Self { drafted, tokens }
    }
}

impl Tally {
    pub fn new(proposal: Proposal, inputs: &[Vec<u32>], accepted_inputs: &[u32],
        emissions: &[Vec<u32>], live: Option<Live>) -> Self {
        let mut tally = Self { drafted: proposal.drafted, verified: [0; 8], accepted: [0; 8], emitted: [0; 8],
            proposal: proposal.tokens, emissions: Vec::new() };
        for (index, input) in inputs.iter().enumerate().take(8) {
            tally.verified[index] = input.len().saturating_sub(1) as u8;
            tally.accepted[index] = accepted_inputs.get(index).map_or(0, |&a| a.saturating_sub(1)) as u8;
            tally.emitted[index] = emissions.get(index).map_or(0, Vec::len) as u8;
        }
        let sum = |values: &[u8; 8]| values.iter().map(|&v| u64::from(v)).sum::<u64>();
        totals::round(sum(&tally.drafted), sum(&tally.verified), sum(&tally.accepted), sum(&tally.emitted));
        if live.is_some_and(Live::text) { tally.emissions = emissions.to_vec(); }
        tally
    }
    /// Build the round event; `members` gives each member's id, grammar mask
    /// state and whether it finished this round.
    #[allow(clippy::too_many_arguments)]
    pub fn round(mut self, lane: usize, shared: bool, started: Instant, draft_us: u64, prepare_us: u64,
        verify_us: u64, layer_us: &[Option<f64>], ffn: FfnSplit,
        members: impl Iterator<Item = (u64, bool, bool)>) -> Event {
        let requests = members.enumerate().map(|(index, (id, masked, finished))| RoundRequest {
            id, masked, finished,
            drafted: self.drafted[index], verified: self.verified[index],
            accepted: self.accepted[index], emitted: self.emitted[index],
            proposal: self.proposal.get_mut(index).map(std::mem::take).unwrap_or_default(),
            emissions: self.emissions.get_mut(index).map(std::mem::take).unwrap_or_default(),
        }).collect();
        Event::Round(Round {
            lane: lane as u8, shared, started, finished: Instant::now(), draft_us, prepare_us, verify_us,
            layer_us: layer_us.iter().map(|v| v.map_or(f32::NAN, |v| v as f32)).collect(),
            ffn, requests,
        })
    }
}

/// Static facts the page shows in its header.
pub(crate) struct Config {
    pub snapshot: PathBuf,
    pub info: Value,
}

/// Bind the console to its hub and start the console thread.
pub(crate) fn install(hub: Arc<ConsoleHub>, config: Config) -> anyhow::Result<()> {
    let (events, receive) = sync_channel(CHANNEL);
    let base = Instant::now();
    let producer = Producer { hub: hub.clone(), events, base, gauges_at: AtomicU64::new(0) };
    if PRODUCER.set(producer).is_err() {
        anyhow::bail!("console producer is already installed");
    }
    std::thread::Builder::new().name("ds41rt-console".into())
        .spawn(move || Worker::new(hub, config, base).run(receive))?;
    Ok(())
}

struct Request {
    lane: u8,
    prompt: u32,
    cached: u32,
    max: u32,
    grammar: bool,
    images: u16,
    admitted: f64,
    first: Option<f64>,
    generated: u32,
    decoder: Option<ds41rt_loader::StreamingTokenDecoder>,
}

struct Worker {
    hub: Arc<ConsoleHub>,
    config: Config,
    base: Instant,
    requests: BTreeMap<u64, Request>,
    prefilling: Option<u64>,
    prefill_done: u64,
    recent: VecDeque<Value>,
    gauges: Value,
    policy: Value,
    policy_at: Instant,
    events: Vec<Value>,
    pieces: HashMap<u32, String>,
    frame_at: Instant,
    snapshot_at: Instant,
}

impl Worker {
    fn new(hub: Arc<ConsoleHub>, config: Config, base: Instant) -> Self {
        let long_ago = base.checked_sub(Duration::from_secs(10)).unwrap_or(base);
        Self { hub, config, base, requests: BTreeMap::new(), prefilling: None, prefill_done: 0,
            recent: VecDeque::new(), gauges: Value::Null, policy: Value::Null, policy_at: long_ago,
            events: Vec::new(), pieces: HashMap::new(), frame_at: base, snapshot_at: long_ago }
    }

    fn ms(&self, at: Instant) -> f64 {
        (at.saturating_duration_since(self.base).as_micros() as f64) / 1000.0
    }

    fn run(mut self, receive: Receiver<Event>) {
        loop {
            match receive.recv_timeout(FRAME) {
                Ok(event) => {
                    self.handle(event);
                    while let Ok(event) = receive.try_recv() { self.handle(event); }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            let viewers = self.hub.viewers() > 0;
            if !viewers {
                // Keep request text state bounded while nobody is watching.
                self.events.clear();
                for request in self.requests.values_mut() { request.decoder = None; }
            }
            if self.policy_at.elapsed() >= Duration::from_secs(1) {
                self.policy_at = Instant::now();
                self.policy = super::speculative::policy_snapshot();
            }
            if viewers && self.frame_at.elapsed() >= FRAME {
                self.frame_at = Instant::now();
                let frame = json!({
                    "type": "frame", "now": self.ms(Instant::now()),
                    "ev": std::mem::take(&mut self.events),
                    "g": self.gauges(),
                });
                self.hub.publish(frame.to_string());
            }
            let period = if viewers { Duration::from_millis(250) } else { Duration::from_secs(2) };
            if self.snapshot_at.elapsed() >= period {
                self.snapshot_at = Instant::now();
                let snapshot = self.snapshot();
                self.hub.set_snapshot(snapshot.to_string());
            }
        }
    }

    fn gauges(&self) -> Value {
        let mut gauges = self.gauges.clone();
        if !gauges.is_object() { gauges = json!({}); }
        let object = gauges.as_object_mut().unwrap();
        object.insert("totals".into(), totals::snapshot());
        object.insert("active".into(), json!(self.requests.len()));
        object.insert("prefilling".into(), json!(self.prefilling.and_then(|id| self.requests.get(&id)
            .map(|request| json!({"id": id, "done": self.prefill_done, "prompt": request.prompt,
                "cached": request.cached})))));
        object.insert("dspark".into(), self.policy.clone());
        gauges
    }

    fn snapshot(&self) -> Value {
        let requests: Vec<_> = self.requests.iter().map(|(&id, r)| json!({
            "id": id, "lane": r.lane, "prompt": r.prompt, "cached": r.cached, "max": r.max,
            "gen": r.generated, "grammar": r.grammar, "images": r.images,
            "admitted": r.admitted, "first": r.first,
        })).collect();
        json!({
            "type": "snapshot", "now": self.ms(Instant::now()), "text": self.hub.text_enabled(),
            "config": self.config.info, "requests": requests, "recent": self.recent,
            "g": self.gauges(),
        })
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Admit { id, at, prompt, cached, max, lane, grammar, images } => {
                let admitted = self.ms(at);
                self.requests.insert(id, Request { lane, prompt, cached, max, grammar, images, admitted,
                    first: None, generated: 0, decoder: None });
                self.prefilling = Some(id);
                self.prefill_done = u64::from(cached);
                self.events.push(json!({"e": "admit", "id": id, "t": admitted, "prompt": prompt,
                    "cached": cached, "max": max, "lane": lane, "grammar": grammar, "images": images}));
            }
            Event::First { id, at, token } => {
                let t = self.ms(at);
                if self.prefilling == Some(id) { self.prefilling = None; }
                let text = self.hub.text_enabled() && self.hub.viewers() > 0;
                let mut piece = None;
                if let Some(request) = self.requests.get_mut(&id) {
                    request.first = Some(t);
                    request.generated = 1;
                    if text { piece = Some(Self::step(&self.config, request, token, false)); }
                }
                let ttft = self.requests.get(&id).map(|r| t - r.admitted);
                self.events.push(json!({"e": "first", "id": id, "t": t, "ttft": ttft,
                    "text": piece.map(|p| json!([["t", p]]))}));
            }
            Event::Retire { id, at, reason, generated } => {
                let t = self.ms(at);
                if self.prefilling == Some(id) { self.prefilling = None; }
                if let Some(request) = self.requests.remove(&id) {
                    let record = json!({"id": id, "t": t, "reason": reason, "prompt": request.prompt,
                        "cached": request.cached, "gen": generated, "admitted": request.admitted,
                        "first": request.first, "grammar": request.grammar});
                    self.recent.push_front(record);
                    self.recent.truncate(RECENT);
                }
                self.events.push(json!({"e": "retire", "id": id, "t": t, "reason": reason, "gen": generated}));
            }
            Event::Prefill(step) => {
                if matches!(step.kind, PrefillKind::Chunk | PrefillKind::Single | PrefillKind::Continuation) {
                    self.prefill_done += u64::from(step.rows);
                }
                self.events.push(json!({"e": "prefill", "id": self.prefilling, "k": step.kind.name(),
                    "lane": step.lane, "i": step.index, "n": step.of, "rows": step.rows,
                    "t0": self.ms(step.started), "t1": self.ms(step.finished)}));
            }
            Event::Gauges(gauges) => {
                let mut value = json!({"lanes": gauges.lanes, "queued": gauges.queued});
                let object = value.as_object_mut().unwrap();
                if let Some(pending) = gauges.pending { object.insert("pending".into(), json!(pending)); }
                else if let Some(pending) = self.gauges.get("pending") { object.insert("pending".into(), pending.clone()); }
                if let Some([total, free, active, ratio]) = gauges.kv {
                    object.insert("kv".into(), json!({"pages": total, "free": free, "active": active,
                        "tokens_per_page": 256 * ratio}));
                }
                if let Some(host) = gauges.host { object.insert("host".into(), host); }
                self.gauges = value;
            }
            Event::Round(round) => self.round(round),
        }
    }

    fn round(&mut self, round: Round) {
        let text = self.hub.text_enabled() && self.hub.viewers() > 0;
        let mut rows = Vec::with_capacity(round.requests.len());
        for member in &round.requests {
            let mut row = json!([member.id, member.drafted, member.verified, member.accepted,
                member.emitted, member.masked as u8, member.finished as u8]);
            let lane = round.lane;
            if let Some(request) = self.requests.get_mut(&member.id) {
                request.lane = lane;
                request.generated += u32::from(member.emitted);
                if text && !member.emissions.is_empty() {
                    let segments = Self::segments(&self.config, &mut self.pieces, request, member);
                    row.as_array_mut().unwrap().push(segments);
                }
            }
            rows.push(row);
        }
        let layers: Vec<Value> = round.layer_us.iter()
            .map(|v| if v.is_finite() { json!((*v * 10.0).round() / 10.0) } else { Value::Null }).collect();
        let ffn = &round.ffn;
        self.events.push(json!({
            "e": "round", "lane": round.lane, "shared": round.shared,
            "t0": self.ms(round.started), "t1": self.ms(round.finished),
            "draft": round.draft_us, "prepare": round.prepare_us, "verify": round.verify_us,
            "layers": layers,
            "ffn": [ffn.local_layers, ffn.local_routed_us, ffn.local_total_us, ffn.remote_layers,
                ffn.remote_routed_us, ffn.remote_dispatch_us, ffn.remote_shared_us, ffn.remote_collect_us],
            "req": rows,
        }));
    }

    /// Stream-decode one emitted token for its request.
    fn step(config: &Config, request: &mut Request, token: u32, finish: bool) -> String {
        let mut out = String::new();
        if request.decoder.is_none() {
            request.decoder = ds41rt_loader::streaming_token_decoder(&config.snapshot, false).ok();
        }
        let Some(decoder) = request.decoder.as_mut() else { return out };
        if token == 1 {
            out.push('␄');
        } else if let Ok(Some(text)) = decoder.step(token) {
            out.push_str(&text);
        }
        if finish || token == 1 {
            if let Ok(Some(text)) = decoder.finish() { out.push_str(&text); }
            request.decoder = None;
        }
        out
    }

    /// One draft token decoded on its own; partial UTF-8 shows as U+FFFD.
    fn piece(config: &Config, pieces: &mut HashMap<u32, String>, token: u32) -> String {
        if let Some(piece) = pieces.get(&token) { return piece.clone(); }
        let piece = ds41rt_loader::streaming_token_decoder(&config.snapshot, false).ok()
            .and_then(|mut decoder| {
                let first = decoder.step(token).ok().flatten().unwrap_or_default();
                let rest = decoder.finish().ok().flatten().unwrap_or_default();
                Some(first + &rest)
            }).unwrap_or_default();
        if pieces.len() < 65536 { pieces.insert(token, piece.clone()); }
        piece
    }

    /// Text segments of one request's round: `a` accepted drafts, `t` the
    /// target's token, `r` verified drafts that were rejected or discarded,
    /// `s` drafts that were never verified.
    fn segments(config: &Config, pieces: &mut HashMap<u32, String>, request: &mut Request,
        member: &RoundRequest) -> Value {
        let accepted = usize::from(member.accepted).min(member.emissions.len());
        let mut segments = Vec::new();
        let mut accepted_text = String::new();
        for (index, &token) in member.emissions.iter().enumerate() {
            let last = index + 1 == member.emissions.len();
            let piece = Self::step(config, request, token, last && member.finished);
            if index < accepted { accepted_text.push_str(&piece); }
            else {
                if !accepted_text.is_empty() { segments.push(json!(["a", std::mem::take(&mut accepted_text)])); }
                segments.push(json!(["t", piece]));
            }
        }
        if !accepted_text.is_empty() { segments.push(json!(["a", accepted_text])); }
        let verified = usize::from(member.verified).min(member.proposal.len());
        let rejected: String = member.proposal.get(accepted..verified).unwrap_or_default().iter()
            .map(|&token| Self::piece(config, pieces, token)).collect();
        if !rejected.is_empty() { segments.push(json!(["r", rejected])); }
        let skipped: String = member.proposal.get(verified..).unwrap_or_default().iter()
            .map(|&token| Self::piece(config, pieces, token)).collect();
        if !skipped.is_empty() { segments.push(json!(["s", skipped])); }
        Value::Array(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_derives_counts_from_round_shapes() {
        // Two requests: 5 drafted, 4 verified, 2 accepted; and an anchor-only row.
        let inputs = vec![vec![10, 11, 12, 13, 14], vec![20]];
        let proposal = Proposal { drafted: [5, 0, 0, 0, 0, 0, 0, 0], tokens: Vec::new() };
        let tally = Tally::new(proposal, &inputs, &[3, 1], &[vec![11, 12, 99], vec![21]], None);
        assert_eq!(&tally.verified[..2], &[4, 0]);
        assert_eq!(&tally.accepted[..2], &[2, 0]);
        assert_eq!(&tally.emitted[..2], &[3, 1]);
        let Event::Round(round) = tally.round(1, true, Instant::now(), 1, 2, 3, &[None, Some(5.0)],
            FfnSplit::default(), [(7, false, false), (8, true, true)].into_iter()) else { panic!() };
        assert_eq!(round.requests.len(), 2);
        assert_eq!((round.requests[0].id, round.requests[0].drafted, round.requests[0].accepted), (7, 5, 2));
        assert!(round.requests[1].masked && round.requests[1].finished);
        assert!(round.layer_us[0].is_nan() && round.layer_us[1] == 5.0);
    }
}
