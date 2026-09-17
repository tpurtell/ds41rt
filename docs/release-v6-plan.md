# V6: dual-RTX parallelism and RAM-backed KV

Status: implementation and review. Baseline: v5 (`2b71d90`).

Increase useful dual-RTX serving throughput while preserving the winning
configuration as the default. Evaluate attention, projection, and dSpark TP2
independently; shipping supported opt-in paths is acceptable when they do not win.

## Implementation sequence

- [ ] Review and port PR #4 onto current dev, preserving contributor attribution,
  current SparkInfer interfaces, and independent execution lanes.
- [ ] Resolve issue #2: token-aware admission, pressure waiting, and safe progress
  when decode needs additional pages. Distinguish retained-prefix offload from
  active-request parking; host snapshots alone do not guarantee active capacity.
- [ ] Resolve issue #3: bounded HTTP admission, default queue depth equal to
  concurrency, 25-second wait budget, 429 with Retry-After for overload and 503
  for shutdown. Bound waiting request memory and support cancellation.
- [ ] Map current retention/SWA controls and set the requested leading-edge
  default to 20. Compute logical device-plus-host capacity above
  slots * max_context_tokens, allowing for snapshot overhead and staging.
  Replicated device copies count once toward logical capacity.
- [ ] Replace the dual-RTX 20-layer minimum with budget-driven bottom-up placement;
  coordinate the chosen RTX/Spark boundary before loading and verify fast startup.
- [ ] Implement independently selectable TP2 projection and attention paths,
  considering head partitioning with replicated KV (DCP1-style). Account for
  reductions, replicated state, graph storage, scratch and host paging bandwidth.
- [ ] Evaluate TP2 dSpark independently, including its experts, attention,
  projections and shared embedding/head access. Preserve lane-owned workspaces.
- [ ] Measure combinations only after individual correctness and timing checks.
  Recalibrate adaptive drafting for any winning placement/timing configuration.

## Evidence and release

- Use the full checkpoint for performance. Prioritize weighted content, code,
  reasoning-code, topic and mixed concurrency; counting is a headline metric.
- Use three final samples; 400 W per RTX and standard memory speed. No cooldown
  requirement. Preserve failures and record source, model, layout and cache size.
- Validate cache restore, pressure progress, cancellation, tools and long-context
  needles for affected paths. EXL3 receives basic compatibility checks only.
- Change defaults only after end-to-end wins without material regressions;
  include startup and prefill alongside the decode-first assessment.
- At publication move historical EXL3 tables to a linked secondary page, retain
  clear full-checkpoint results in README, and document new memory allocations
  in bytes and logical tokens. Include a diagram below the goal introduction.
- Commit and push incremental work. Final publication includes notes, containers,
  and release/v6 at the exact final tagged commit after qualification.

## Initial findings

PR #4 targets release/v3 and introduces pinned-RAM retained snapshots despite
its disk-persistence title. It also includes protocol/loader fixes and tests.
Review its dual-device ownership and active-pressure behavior before enabling it.
The current dual worker explicitly requires 20–40 resident routed-expert layers;
the launcher also assumes Spark execution starts at layer 20. Both must agree
with flexible placement. Current documentation contains older single-RTX
ownership descriptions and must be updated from the implementation.

Sources: [PR #4](https://github.com/tpurtell/ds41rt/pull/4),
[issue #2](https://github.com/tpurtell/ds41rt/issues/2),
[issue #3](https://github.com/tpurtell/ds41rt/issues/3).

## Integration progress (September 17)

The `work/v6-hostcache` branch merges PR #4's original commits onto v5,
preserving attribution. Its host cache remains disabled by default pending
complete serving validation. The daemon compiles; the host-cache suite passes
214 tests (two soak tests remain ignored). Native HTTP tests pass 28 cases,
including bounded queue waiting, cancellation, overload status and Retry-After.
The recipe and binary retention default is now 20; the existing separate prompt
and completed-turn banks each use that limit, which the RAM planner must include.

Dual-device review found that host restore incorrectly labelled dSpark prefixes
as GPU0-owned. Restored rings now allocate on their runtime device, and snapshot
ownership derives from the actual buffers. Two hardware tests pass on both RTX
cards: mixed-device RAM round trips through both batch and fallback copies, and
GPU1 draft ownership with both pooled and directly allocated storage.

Token-aware admission now checks prompt plus output allowances for the whole
active cohort against actual source-page capacity, preserving prefix sharing
and partial-page copy accounting. A blocked request keeps only its prepared host
input and retries after retirement. The lane wake policy prevents a waiting
request or nonempty HTTP queue from repeatedly stopping decode while the same
requests still occupy the pool; cancellation wakes admission. An individually
oversized request receives a clear 400 response instead of an execution failure.
Three CPU policy tests and a native source-page pressure test pass. Core, loader
and transport library tests also pass (390 cases, seven hardware-dependent cases
ignored). The one-off serving pressure test below passes. Performance qualification
and the final default RAM configuration remain outstanding.

### One-off small-pool torture test

Run on September 17 against `5f487c1`, using the full native checkpoint and v5
native kernels: dual RTX, 20 TP2 routed layers, dSpark enabled, C4, two retained
entries per bank, 6 MiB requested global KV (13 groups = 6,656 logical tokens,
5,923,840 allocated global bytes), 64 MiB pinned RAM in 8 MiB chunks, and an HTTP
queue of four with a 500 ms queue-space wait. These are deliberately restrictive
test settings, not proposed release defaults. RTX power limits and memory speed
were unchanged. This is not a throughput benchmark.

The deterministic randomized pressure phase lasted 120.42 seconds. Including
warm-up and recovery, 1,085 requests yielded 48 HTTP 200 responses, 992 retryable
429 responses, two explicit 400 responses for requests exceeding the GPU KV pool,
and 43 cancellations before headers. There were 45 intentional cancellations in
total (two after HTTP 200), no unexpected 5xx responses, stream errors, transport
exceptions or 40-second request timeouts. All three final recovery requests
finished successfully in 0.296, 0.182 and 0.184 seconds.

Cache telemetry recorded 43 completed RAM restores, 116 device evictions and 49
host evictions. All eleven `restore_failures` were logged no-device-room skips;
there were no CUDA copy failures or restore timeouts. The current metric combines
capacity skips with copy failures, so interpret it alongside the logs. The final
stats snapshot had 74 completed stores of 75 issued (publication can lag while
the scheduler waits for the next request). Normal v5 serving was restored after
the test. The local evidence bundle is `~/.cache/ds41rt-v6-torture/`; its script is
deliberately not part of recurring release qualification.

This admission policy reserves future GPU capacity; it does not implement active
request parking in RAM. RAM sizing, complete cache qualification, flexible expert
placement, all TP2 experiments and v6 publication remain required.
