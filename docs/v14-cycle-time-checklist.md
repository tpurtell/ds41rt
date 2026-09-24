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

- [ ] Baseline per-layer/per-phase timing of v13 at C1 on one RTX from existing
  `ds41rt::timing` logs.
- [ ] Round-phase stamps for the untimed gaps: draft phases, commit wait,
  emission, observe, embed + layer 0.
- [ ] Attribute the 3.4 ms mean stall beyond the robust fit (round-time
  histogram, per-layer outliers).

## 1. Round structure (host scheduling)

- [ ] Engram at layers 1 and 14: replace the 1 ms sleep with a prompt wake-up;
  start the gather as soon as the round's tokens are known.
- [ ] Launch the next draft as soon as the commit is queued; do emission,
  observe and logging while it runs.
- [ ] Remote layers: poll RDMA before waiting on the shared expert.
- [ ] Stream-order the mHC finish, next-input copies and taps waits (events,
  not host polls).
- [ ] Head: fuse final norm; remove sequential waits.

## 2. Remote expert boundary (coordinator + Spark worker)

- [ ] Router graph writes routes/rows straight into registered send slots;
  remove per-layer Vec clones, channel/oneshot allocation, repeated prefix
  encoding.
- [ ] Coordinator receive loop: tight poll without tokio yield per miss,
  in-place response handling, no pooled-frame copy.
- [ ] Worker: read inputs from the device-visible request slot (GB10 unified
  memory) instead of three synchronous uploads; drop the bind-layer and
  pre-upload syncs.
- [ ] Worker: fold compaction into the expert kernel epilogue or graph the
  two launches.
- [ ] GPUDirect (dmabuf) registration of RTX rank planes, removing the
  host→device plane uploads.

## 3. Expert kernels (b12x fork)

- [ ] Grouped slice kernel: multi-stage cp.async pipeline (load/compute
  overlap) to cut the per-CTA serial floor; qualify on RTX (TP1/TP2) and GB10
  (TP4).

## 4. Local (RTX-resident) layers

- [ ] Overlap the shared expert with the routed experts (second stream) and
  remove the router → shared → routed host waits.
- [ ] Graph the local expert pipeline and fuse the finish reduce.
- [ ] Shared expert: quantize the input once for gate and up.
- [ ] TP2 (two RTX): sum routes per token before the peer transfer (FP32
  per-token sums instead of FP32 route planes); drop the extra copy and waits.

## 5. Attention side

- [ ] Reduce host round trips per layer (query, window KV, attention graphs):
  event-chained stages, fewer polls.
- [ ] Remove redundant frequency computation and residual D2D copies.

## 6. dSpark draft

- [ ] Draft vocabulary head on tensor cores (BF16 in, FP32 accumulate), then
  evaluate an FP8 draft head against acceptance.
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
| v13 baseline | 00b25e7 | 134 | 39.3 | release campaign |
