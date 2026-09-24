# Live engine console at `/`

Status: implemented on `dev`, under interactive iteration before
any release. Written 2026-09-25 against `dev` at `e62143f`.

Implementation map:
- Page: `rust/crates/ds41rt-api/assets/console.html`, compiled in with `include_str!`.
- Routes and fan-out: `rust/crates/ds41rt-api/src/native_v41/console.rs`
  (`GET /`, `GET /v1/console` WebSocket, `GET /v1/console/snapshot`).
- Producer, lifetime totals and console thread:
  `rust/crates/ds41rt-daemon/src/v41_native_serve/console.rs`.
- Hooks: both round paths in `scheduler.rs` / `scheduler/independent.rs`,
  admission and retirement, prefill chunk sites, and the per-pass FFN stage
  split (`FfnSplit`) in `v41_backbone_lane.rs`.
- Text view opt-in: `--console-text` / `DS41RT_CONSOLE_TEXT=1` (forwarded by `run.sh`).
- Lifetime totals are also exported under `totals` in `/v1/stats`.
- Traffic generator for trying it out: `scripts/console-load.py`.

## Goal

Serve a real-time dashboard from the coordinator at `GET /` that shows what
the engine is doing right now: headline rates, capacity meters, a ticker of
every verification round on both execution lanes, prefill activity, and the
durations of the pipeline's micro-steps. It must not slow the engine. It is
for the operator watching one instance, not a metrics backend.

## Hard constraints

1. **Zero measurable cost to the decode cycle.** Nothing on the CUDA thread
   blocks, allocates in the hot loop, or syncs the device for the console.
   When nobody is connected the whole feature reduces to one relaxed atomic
   load per round.
2. **Only what is already measured.** Micro-step durations come from timers
   and CUDA events the engine records today for the dSpark policy and the
   `ds41rt::timing` traces. New events are opt-in and listed separately
   (phase 3) with their cost stated before they are added.
3. **No text on the wire by default.** The feed carries counts, ids, and
   durations. Completion token ids are added only under the server opt-in
   for the text ticker mode (see below); prompt text never leaves the
   process.
4. **No new deployment surface.** The page is one HTML file compiled into the
   `ds41rt` binary with `include_str!`; no Dockerfile change, no CDN, no
   external assets (the page must work on the fabric network without
   internet).
5. **Same code on 1x and 2x RTX.** Both layouts run `scheduler::serve`, so
   the hooks are shared; the feed carries the `shared` flag the policy
   already tracks.

## What the engine exposes today (baseline)

- Routes (`rust/crates/ds41rt-api/src/native_v41.rs:87-93`): `/health`,
  `/v1/models`, `/v1/stats`, `/v1/chat/completions`. Nothing at `/`.
- `/v1/stats` is a `Mutex<serde_json::Value>` refreshed at most once per
  second at the top of the scheduler loop (`scheduler.rs:200-209`). It goes
  stale while both lanes run independently and while the process is idle.
  It has `host_cache`, `target_sampling`, `dspark_policy` and HTTP queue
  counters. It has no token rates, no per-request state, no lane occupancy
  and no KV-pool usage.
- Per-round data already exists at the `ds41rt::timing` debug events
  (`scheduler.rs:1212, 1244`; `scheduler/independent.rs:234`):
  `proposed, accepted, emitted, draft_us, prepare_us, verify_us, total_us`.
  Enabling the tracing target at DEBUG is not acceptable in production
  (`logit_trace` shares the filter and forces full logit downloads).
- The scheduler already funnels every round through `observe_lane_round`
  (`scheduler.rs:1637`), which receives lane, `shared`, per-request proposal
  rows, per-request accepted counts, the round start `Instant`, `draft_us`,
  and the per-layer CUDA-event durations (`captured_layer_us`). That is the
  natural single emit point for round telemetry.
- Prefill runs synchronously inside admission and owns both lanes
  (`scheduler.rs:225-226`). Chunks alternate lanes by parity in
  `execute_encoder_stream_held`; there is a per-chunk hook (`before_chunk`)
  already used for the host-cache pacing hold.
- No token is ever injected into the output. Grammar constraints (tool-call
  DSML structural tags, JSON schema) only mask sampling and truncate
  proposals (`v41_native_serve/constraints.rs`). So "template output" is
  visualized as *spans decoded under an active grammar mask*, which the
  scheduler knows per request per round.

## Architecture

```
CUDA thread (scheduler)                     tokio HTTP runtime
─────────────────────────                   ─────────────────────────────
observe_lane_round ─┐                       GET /            → include_str! page
prefill chunk hook ─┤ push Event            GET /v1/console  → WebSocket upgrade
admit / retire     ─┤ (Vec, no lock)            │ subscribe broadcast
gauges every 50 ms ─┘                           │ send frames as they arrive
       │ flush ≤ every 50 ms                    │ first frame = full snapshot
       ▼                                        ▼
broadcast::Sender<Arc<Frame>>  ───────────►  page: ring buffers + canvas
(send is sync; skipped when receiver_count()==0)
```

### Telemetry producer (CUDA thread)

- A `Console` struct owned by the scheduler holds a `Vec<Event>` accumulator,
  the last flush `Instant`, and the `broadcast::Sender<Arc<Frame>>`.
- `Console::push` is called from: `observe_lane_round` (one `Round` per lane
  round with its per-request outcomes), the prefill chunk hook (`Prefill`),
  admission (`Admit`, `Pending`, `Queued`), retirement (`Retire`), and round
  failure (`Fault`).
- `Console::maybe_flush` runs after each push. If 50 ms have passed since
  the last flush it builds a `Frame { t, events, gauges }`, wraps it in
  `Arc`, and calls `broadcast::Sender::send`. `send` is synchronous and
  lock-free; it never awaits.
- **Gate:** every push first checks `sender.receiver_count() == 0` (a
  relaxed atomic load). With no page open the accumulator stays empty and
  nothing else runs. This is the "one atomic per round" cost.
- The lanes only return to the scheduler loop on a drain, so the flush lives
  inside the hook path, not at the loop top. This also fixes the `/v1/stats`
  staleness: the same flush republishes the stats snapshot.
- Frames are bounded: at ~150 rounds/s/lane the round events are ~80 bytes
  each in JSON, ~25 KB/s per client. The broadcast channel has capacity 64;
  a slow client sees `Lagged`, gets a fresh snapshot, and continues.
- Nothing is serialized on the CUDA thread. Serialization to JSON happens in
  the WebSocket task, once per frame, then the bytes are cloned per client.

### Transport

- `GET /v1/console` is an axum WebSocket route (enable axum's `ws` feature;
  it pulls `tokio-tungstenite`, already in the dependency graph of axum).
- The first message is a `snapshot`: static config (revision, image tag,
  layout, lane count, draft limit, prefill chunk rows, KV pool pages, host
  cache quota, concurrency limit), plus current gauges and the live request
  list, so a page opened mid-run is correct immediately.
- After that, frames as above. Text frames, JSON. No client to server
  messages except pings; the page cannot change engine state.
- `GET /v1/console/snapshot` returns the same snapshot as plain JSON so
  scripts (`bench-ds41-*`) can poll gauges without a socket.

### Wire schema

```jsonc
// snapshot (first message and on reconnect)
{"type":"snapshot","t":123456.7,"config":{...},"gauges":{...},"requests":[...]}

// frame (≤ 20 per second)
{"type":"frame","t":123506.7,"gauges":{...},"events":[
  {"e":"round","t0":123480.1,"t1":123487.9,"lane":0,"shared":true,
   "draft_us":1810,"prepare_us":210,"verify_us":5600,
   "layers":{"local_us":3300,"remote_us":2100},           // sum of captured_layer_us by class
   "req":[{"id":18960,"w":5,"v":4,"a":3,"g":false,"fin":false}, ...]},
  {"e":"prefill","t0":...,"t1":...,"id":18961,"lane":1,"rows":2048,"index":1,"of":3,"cached":false},
  {"e":"admit","t":...,"id":18961,"prompt":3120,"cached":1024,"max":1024,"lane":0,"grammar":true,"queue_ms":4},
  {"e":"pending","t":...,"id":18962,"needed":40,"available":12},
  {"e":"retire","t":...,"id":18960,"reason":"stop","generated":412,"elapsed_ms":6120,"ttft_ms":380}
]}
```

Per request per round, `w` is the drafted width (0 when the request got an
anchor only), `v` the rows verified after grammar and policy truncation, `a`
the accepted drafts, `g` whether a grammar mask was active. Emitted tokens
are always `a + 1` unless the request finished. The page derives per-position
outcomes: positions `1..=a` accepted, `a+1` the target's token (bonus when
`a == w`, correction otherwise), `a+2..=v` rejected and discarded,
`v+1..=w` drafted but never verified. None of that is text.

`gauges` (every frame): `active[2]` per lane, `pending`, `queued`,
`kv_pages{active,retained,total}`, `host{bytes_used,quota,resident}`,
cumulative counters `emitted, drafted, verified, accepted, prefill_rows,
input, cached_input, output`, `policy{mode,warm,width}`, `uptime_s`.
Rates are computed on the page from counter deltas, so the engine never
keeps a rate window.

`kv_pages.retained` is `total - free - active`; `free` is
`PagePool.free.len()`, `active` is the sum of active leases' page counts.
The lease page count needs a small accessor (verify in
`v41_backbone_cache.rs` / `source_cache/ownership.rs`).

### Page

One file, `rust/crates/ds41rt-api/assets/console.html`, vanilla JS and
Canvas 2D, no framework, no fonts fetched from the network (system sans and
monospace stacks). The mock in the proposal artifact uses web fonts for
presentation only.

## Layout, top to bottom

1. **Header strip.** Product name, serving state pill, git revision, image
   tag, layout (`1×RTX + 4 Spark` / `2×RTX`), policy state
   (`bandwidth · w5 · warm`), uptime, socket state.
2. **Headline tiles** (hero numbers, 60 s sparkline each, 1 s bins):
   - Output tokens/s (emitted to clients)
   - Accepted drafts/s, with acceptance rate `accepted/verified`
   - Proposed drafts/s, with `verified/proposed`
   - Prefill tokens/s
   - Concurrent requests (decoding per lane, plus prefill)
   - Total input tokens (with cached share)
   - Total output tokens (with accepted-draft share)
3. **Capacity meters** (two-tone tracks):
   - Concurrency: active + pending against the 16 limit; per-lane counts;
     last cycle time per lane.
   - KV cache: active pages + retained snapshot pages over total pool pages;
     tokens resident; snapshot count.
   - Host KV offload: bytes used over quota; lookups and hits; restore p50
     from the existing latency buckets.
4. **Execution lanes ticker** (the hero block). Details below.
5. **Pipeline micro-steps** (left) and **dSpark acceptance by position**
   plus **recent requests** (right).

## The lane ticker

The key problem is rate. A lane completes a round every 5–30 ms, so
anything that draws one glyph per round and scrolls at a fixed pixel rate is
either a blur or a crawl. The answer is to make **x a true time axis** and to
let **the round's own duration be its width**.

- Time flows right to left; the newest round touches the right edge. The
  scale is selectable (150, 300, 600, 1200 px/s), default 300 px/s, so the
  block shows about the last 4 s. At 300 px/s a 7 ms round is a 2 px column
  and a 25 ms round is 7 px. Column width therefore *is* cycle time, and the
  two lanes read against each other on the same axis (independent lanes
  drift, single-lane rounds line up, a drain shows as both lanes stopping).
- One row per active request, 18 px tall, grouped by lane with a thin rule
  and a `LANE n` label in the gutter. Rows appear on admission and disappear
  on retirement, so the block never reserves space per lane. An empty lane
  collapses to its rule and the word `idle`.
- Each round paints one column in each member row. Cells stack by draft
  position from the bottom, 2 px per position (8 slots fit width 7 plus the
  target token):
  - green = accepted draft
  - blue = the target's token (bonus after a full accept, correction after
    a mismatch)
  - red = rejected draft (verified and discarded)
  - gray = drafted but never verified (policy or grammar truncation)
  The blue cell splits every column: everything below it was useful,
  everything above it was wasted. That is the reading at a glance, and it
  makes the green/red pair safe for colorblind viewers because position
  encodes the same thing as hue. A request with no draft that round shows
  a lone blue cell.
- A 1 px amber line under a row marks spans decoded under a grammar mask
  (tool call, JSON schema). That is the "template output" indicator.
- Gutter per row: short request id, tokens generated, a 38 px progress bar
  against `max_tokens`.
- **Prefill row** at the top, above lane 0. Prefill owns both lanes, so
  while it runs the request rows are empty and the prefill row explains the
  gap. Each chunk is a violet segment whose width is its real duration,
  drawn in the upper or lower half of the row by the lane it ran on, so the
  parity pairing of the encoder stream is visible. A prefix restore is a
  lighter leading segment. Chunk size (80–4096 rows) shows as segment width
  at the row's constant tokens/s, which is why it is meaningful after all.
- Time axis with 0.5 s ticks (0.25 s above 600 px/s).
- A view switch on the block selects this time-axis view or the text
  ticker described in the next section.
- Hover freezes the stream and shows the round: lane, solo/shared,
  requests in the round, drafted/verified/accepted/emitted for that row,
  draft/local/remote/head microseconds. A pause button does the same.
- Rendering: one `requestAnimationFrame` loop, ring buffer of columns per
  lane trimmed at 40 s, draw only the visible window. Sixteen rows of a few
  hundred columns is trivial for Canvas 2D. Under
  `prefers-reduced-motion` the canvas redraws once per second instead.

## Text ticker mode

A second view of the same block, switched on the page. Every row keeps its
place, but instead of 2 px cells each round contributes the **text of its
tokens**, so the operator can almost read the outputs as they stream:

- The main line is what the client received: accepted drafts in green,
  the target's own token in blue (bonus or correction). Grammar-masked spans
  keep the amber underline.
- The drafter's wrong guesses float **above** the line in small red type,
  right-aligned to the blue correction they lost to; drafts that were never
  verified sit beside them in gray. The main line therefore stays readable
  and the ghosts show what the drafter would have said.
- Text enters from the right at the row's own token rate (a slide-in that
  decays over ~90 ms), so a fast request visibly outruns a slow one. There is
  no time axis in this mode; the prefill row becomes a progress bar
  (`done / prompt tokens`, restored prefix noted).
- Newlines and tabs render as `⏎` and `→` so code stays on one line.
- Prompt text is never shown. Prefill remains counts only.

### What it costs and where the text comes from

- **Hot path:** with text enabled, the round event carries the drafted ids
  (`w` u32 per request) and the emitted ids (`a + 1` u32). The scheduler
  already has both slices in `commit_lane` / `publish_commit_lane`
  (`inputs[i]` and `emissions[i]`), so it is a copy of about a dozen
  integers per request per round into the accumulator. No decoding on the
  CUDA thread.
- **HTTP side:** the WebSocket task owns an id-to-piece table built once
  from the tokenizer and, per active request, a `StreamingTokenDecoder`
  for the emitted stream (same type the API uses, so byte-level merges
  produce the same text the client sees). Rejected and unverified drafts
  are decoded lossily piece by piece; a partial UTF-8 piece shows as `�`.
  The frame then carries `text: [{s, k}]` per request per round with `k`
  in `a | t | r | s`.
- **Opt-in on the server:** `--console-text` (or `DS41RT_CONSOLE_TEXT=1`).
  The console has no authentication beyond network reach, the same as the
  API, but text mode exposes *other sessions'* completions to anyone who
  can open the page. Default off; the page shows the mode as unavailable
  when the snapshot says so.
- Frame size grows to roughly 100 bytes per request per round with text,
  still under 200 KB/s per viewer at full concurrency.

## Pipeline micro-steps block

A single column of thin horizontal bars, one per step, each showing the
**last round's duration** as a solid bar, the **60 s median** as a lighter
bar behind it, and the **60 s max** as a tick. Each group shares one scale
so bars within a group compare; groups are labeled with the clock they come
from (host `Instant` vs CUDA events). Remote (Spark) steps use the violet
accent so the RTX/Spark split reads at a glance.

Groups and sources, split into what is free today and what would need new
events. "Free" means the `Instant` or CUDA event is already taken on every
production round; only the destination changes (today most of these values
are formatted for a `tracing::debug!` that the filter drops).

### Free today, host clock (already computed every round)

| Step | Source | Notes |
| --- | --- | --- |
| Lane cycle `total_us` | `scheduler.rs:1140-1249`, `independent.rs:57-234` | Already feeds the policy round fit |
| `draft_us` (propose + length selection) | same | The 3 draft stages are one graph replay; no per-stage split exists |
| `prepare_us`, `verify_us` | same | Verify = the whole 39-layer pass plus head |
| Commit + emit | `total - (draft + prepare + verify)` | Remainder, not a separate timer |
| Per remote layer: `routed_us`, `dispatch_us`, `collect_us`, `shared_us` | `v41_backbone_lane.rs:317-381` | `routed_us` is real device progress (the router host-waits); the others are host enqueue spans |
| Reply collection: `upload_us`, `receive_us`, `reduce_us` | `v41_experts/coordinator.rs:682-715` | `receive_us` is the wait for all four Spark ranks: the Spark cycle as seen from the RTX |
| Prefill: encoder step, per chunk, decoder replay | `v41_native_serve.rs:661-701`, `encoder_stream.rs:89-114` | Host clock, on every prefill |
| Host cache store/restore latency | `ds41rt-hostcache/src/metrics.rs:80-88` | 8 buckets, 1 ms to 1 s, already in `/v1/stats` |
| Vision encode | `scheduler.rs:314-322` | `tracing::info!` today |

### Free today, CUDA events (the one device-accurate probe)

`captured_layer_us` (`v41_target_pass.rs:142-144`) gives FFN-finish to
FFN-finish for layers 1..39, recorded on the FFN stream and read with
`cudaEventElapsedTime` after the chain drains. It is on whenever dSpark is
loaded (`capture_routes` is `policy.is_some()`), 39 elapsed reads per round,
and the raw values are dropped right after the policy fits its regression.
On 2 RTX the first layer after the GPU handoff is timed from the entry
event (`distributed.rs:230-238`). Each layer's number is device wall time
for the whole layer including any wait for routes or Spark replies.

The console shows this as:
- RTX-resident layer (mean of the local class), Spark-routed layer (mean of
  the remote class), slowest layer this round.
- A **39-bar layer profile strip** of the last round, blue for local layers
  and violet for remote, so a slow Spark rank or a cold layer shows up as a
  spike at its index.

### Not measured today (phase 3, opt-in, sampled)

- Inside a layer: query, window KV, compressor, index query and selection,
  sparse attention, router staging, shared and local experts, remote
  reduction. The v14 chain's ordering events are created with
  `cudaEventDisableTiming` and there is a single head event per device that
  is re-recorded at every stage (`v41_memory/chain.rs:69-70, 174, 208`), so
  they cannot be reused for elapsed time. Timing needs new events at the
  `chain::finish` sites.
- Head and device sampler: no timers in production code.
- Spark worker side (`upload_us`, `kernel_us`, `compact_us`): measured only
  under `ds41rt::expert_timing` DEBUG and written to the worker log. Showing
  it on the coordinator needs a timing field in the reply header
  (`protocol_v2.rs:295-307`), a protocol change.

Cost control for phase 3: 39 layers × ~10 stages is ~400 event records and
~400 elapsed reads per round, easily 0.5 ms on a 7 ms cycle. So stage events
are **sampled**: one layer per round, rotating through 1..39, which keeps it
at ~10 events per round and still refreshes every layer about four times per
second at C1. Even so it ships off by default until the A/B shows no change.

## Phases

1. **Feed + page skeleton.** `Console` producer, gate, flush, WebSocket
   route, snapshot, `/` page with header, tiles, meters, lane ticker,
   prefill row. Hooks: `observe_lane_round`, prefill chunk hook, admission,
   retirement. New gauges: KV pool pages, active per lane.
2. **Micro-steps from existing measurements** and the acceptance-by-position
   panel (from the existing `dspark_policy` reliability table), recent
   requests, host-cache restore latency.
3. **Opt-in stage events** inside a layer (query, window KV, compressor,
   index, attention, router, experts, reduce), behind a flag, with an A/B
   showing no cycle-time change before it defaults on. Off by default until
   then.

## Verification plan

- A/B: identical-config decode battery with (a) no page open, (b) one page
  open at 300 px/s, (c) `DS41RT_CONSOLE=0` build flag removing the hooks.
  Accept only if (a) and (c) are within run-to-run variance on code C1–C8
  and (b) is within 1 %.
- Unit test for outcome derivation (`w, v, a` to per-position cells) and
  frame bounds.
- Smoke: `scripts/api-smoke.sh` extended with a socket client that reads
  one snapshot and one frame.

## Overhead measurement (2026-09-25)

2×RTX + 4 Spark, greedy, 600 output tokens per request, prompts cached after
warmup. Tokens per second, median of runs:

| Arm | C1 (6 runs) | C8 (4 runs) |
| --- | --- | --- |
| v14 release coordinator | 174.1 | 296.9 |
| Console build, no viewer | 173.8 | 293.4 |
| Console build, WebSocket viewer on the text feed | 175.3 | 288.5 |

The C8 arms overlap completely (v14 ranges 287–306). A separate same-binary
ABBA at C8 gave 295.0 without a viewer and 297.1 with one. Tools:
`.cache/console/ab.py` and `.cache/console/xab.sh` (not committed). With text
on, the feed is about 16 frames and roughly 105 KB per second per viewer at C8.

## Open decisions

1. **"Accept tps".** Proposed as accepted drafts per second, shown next to
   output tokens per second (emitted). Confirm or swap the headline.
2. **"Template output".** Proposed as grammar-masked spans. If reasoning vs.
   content phase is also wanted, the scheduler can flag the think-close
   token id at emit time (one integer compare per token); not included by
   default.
3. **Concurrency of pages.** Broadcast capacity 64 frames, any number of
   viewers; each viewer costs one JSON clone per frame on the HTTP runtime.
4. **Snapshot endpoint name** (`/v1/console/snapshot`) and whether
   `/v1/stats` should be folded into it or kept as is.
