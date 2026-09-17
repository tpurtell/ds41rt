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
