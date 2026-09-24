# v14 verification cycle-time checklist (working file)

Working checklist for the v14 cycle-time program. It is deleted in the v14
release commit. Source of the items: the per-round cost audit of v13 (fitted
online cost model reconciled against measured C1 round time: about 39 ms per
code round on one RTX, 32 ms on two RTX).

Rules: official checkpoint only; no NVFP4 downgrades; target LM head stays
BF16; draft-side precision changes are allowed (output is verified). Each item
is measured on one RTX + four Sparks (ostrich, dodo, emu, kiwi) against the
previous step before commit, except two-RTX-only items. Commit messages carry
approximate before → after. Scratch files go under `.cache/`, never
`/mnt/scratch`.

Status: `[ ]` open, `[~]` in progress, `[x]` landed (commit), `[-]` dropped
(reason).

## 0. Baseline and instrumentation

- [x] Baseline of v13 at C1 code on one RTX (timing logs + Nsight trace):
  38.7 ms/round = draft 2.7 + verify 35.3 + tail 0.7. Remote layer: RTX
  pre-dispatch ~320 µs with ~200 µs GPU busy; Spark round trip ~560 µs
  (kernel 510 µs at 6 rows / 23 experts). GPU busy 37% of the round; ~6 ms of
  2–20 µs launch gaps.
- [x] The 3.4 ms "stall" is a fit artifact (unmeasured layer 0, head, draft
  sit in the round residual); round times are tight (median 39.1, p90 40.9).
- [x] `run.sh` forwards `RUST_LOG` to Spark workers (was silently `info`) and
  quotes remote arguments (empty values shifted later positions).
- [x] Deterministic oracle: `--dspark-fixed` greedy battery output hashes are
  identical across v13 launches; numerics-preserving changes must match.

## 1. Round structure (host scheduling)

- [x] Stage chain for verification passes (01591f0): per-device CUDA events
  order query, window KV, compressor, index, attention, router staging, shared
  and local experts, reduction, mHC finish, taps and head; the host waits only
  for routes. Prefill stays host-drained (chaining it diverged at the first
  token). Chained vs drained greedy outputs identical. Remote layer 298 → 252
  µs from first upload to routes.
- [x] dSpark policy fed device-timed layer costs (CUDA events at FFN finish);
  host stamps under the chain misled it (topic C2 −8% → +13% vs v13).
- [~] Two-RTX stage chain: per-device events; peer DMA (block handoff, TP2
  FFN, TP2 shared, result return) settles the chain first; head settles once.
- [x] Target greedy argmax: bitwise identical, 171 → 17 µs in the pass (e9ae29f).
- [x] Router: bitwise identical, 17.6 → 12.0 µs at 6 rows (e9ae29f).

- [x] Engram at layers 1 and 14: queued gather path on chained passes and a
  yield instead of the 1 ms retry sleep (01591f0).
- [ ] Launch the next draft as soon as the commit is queued; do emission,
  observe and logging while it runs. (Measured tail is ~0.2 ms; low value.)
- [x] Remote layers: the shared expert no longer blocks RDMA polling (chained).
- [x] mHC finish, next-input copies and taps stream-ordered (01591f0).
- [ ] Head: fuse final norm; remove sequential waits.

## 2. Remote expert boundary (coordinator + Spark worker)

- [ ] Router graph writes routes/rows straight into registered send slots;
  remove per-layer Vec clones, channel/oneshot allocation, repeated prefix
  encoding.
- [ ] Coordinator receive loop: tight poll without tokio yield per miss,
  in-place response handling, no pooled-frame copy.
- [x] Worker: hidden rows copied on-stream from the mapped request frame, route
  ids/weights from pinned staging; upload 17 → 4 µs per layer (827383a).
- [ ] Worker: fold compaction into the expert kernel epilogue or graph the
  two launches.
- [ ] GPUDirect (dmabuf) registration of RTX rank planes, removing the
  host→device plane uploads.

## 3. Expert kernels (b12x fork)

- [-] Double-buffered grouped slice kernel: measured on dodo the kernel is
  already DRAM-bound (~228 GB/s vs 235 GB/s torch max-read) until CTAs exceed
  the 96 resident slots; buffering would cost residency. Dropped for decode.
- [x] Single-CTA route plan for decode capacities: identical outputs,
  12.3 → 3.6 µs per call on GB10 (acb4dfb, b12x 7fcc094).
- [ ] Fuse slice reduce with the worker's compact route sum (~8 µs).
- [~] Worker L2 prefetch of predicted next-layer experts during idle gaps,
  behind `DS41RT_EXPERT_L2_PREFETCH=K`. Route traces: 2.3 of 3 predicted
  experts are used by the next request (code and topic).
  Dodo microbenchmark: 3 of 23 experts in the 18 MB persisting set-aside cut
  the kernel 15% (582 → 494 µs). Needs: hit rate from real route traces;
  gate to a single active lane (lanes are not layer-locked, so the next
  request is ambiguous at C2+); A/B at C1 and C2–C16.

- [x] Batched window KV commit: one upload + one launch for all 40 layers
  (d7d3b32).

## 4. Local (RTX-resident) layers

- [x] Local routed experts overlap the shared expert; only the final reduce
  joins it (01591f0).
- [ ] Graph the local expert pipeline and fuse the finish reduce.
- [ ] Shared expert: quantize the input once for gate and up.
- [ ] TP2 (two RTX): sum routes per token before the peer transfer (FP32
  per-token sums instead of FP32 route planes); drop the extra copy and waits.

## 5. Attention side

- [ ] Reduce host round trips per layer (query, window KV, attention graphs):
  event-chained stages, fewer polls.
- [ ] Remove redundant frequency computation and residual D2D copies.

## 6. dSpark draft

- [x] Draft attention: bitwise identical, 150 → 39 µs at a full window
  (e9ae29f).
- [~] Draft vocabulary head in FP8 behind `DS41RT_DRAFT_FP8_HEAD=1` (860 →
  440 µs at 5 rows, 2.6% rms logit error); needs an acceptance A/B.
- [ ] Two-RTX draft terminal as one graph (no host-driven phases); one-RTX
  propose with a single sync.
- [ ] Draft kv projection reuses the query-side quantized input.

## 7. Two-RTX sampled decode

- [ ] Device sampler for the two-RTX layout (no full-logit downloads).

## 8. Release v14

- [ ] Final images built from a frozen clean commit; one-RTX and two-RTX
  launches pass smoke checks.
- [ ] Interleaved A/B, v13 vs v14, on both layouts.
- [ ] README full decode metric set for one and two RTX; performance report
  and release notes.
- [ ] Publish images, promote configs, fast-forward main, `release/v14`, tag,
  GitHub release; delete this checklist.

## Measurements log

| Step | Commit | C1 code 1×RTX tok/s | Round ms | Notes |
|---|---|---:|---:|---|
| v13 baseline | 00b25e7 | 134 | 38.7 | timing logs, C1 code |
| kernels + worker + planner + chain | build c | 145 | 35.8 | timing logs |
| + device-timed policy costs | build d | 152 | — | lite A/B vs v13 137 (+10.5%) |
