# DS41RT v13 release notes

**Status: PUBLISHED.** The v13 image pair was built from immutable source
`00b25e7025410cdb1dadc7960b8e94ae9b89db48`, evaluated against v12 in
interleaved A/B sessions on one and two RTX, measured with the release decode
battery, pushed to GHCR, and verified by anonymous pulls on the matching
architectures. `latest` resolves to the same pair. The runtime default and the
example configurations name `:v13`.

## What changed

V13 replaces every adaptive dSpark length selector with one online
bandwidth-balance policy ([design](dspark-bandwidth-policy.md)). Each round it
chooses how many drafted tokens each request verifies by maximizing expected
committed tokens per unit of predicted round time. Time is priced from the
resources the round consumes: per layer, routed-expert weight traffic (known
slice bytes per 16-row group) over an effective bandwidth, separately for
RTX-local and Spark-remote layers, plus the draft pass and a per-round
residual. Every coefficient is fitted continuously from the lane's own layer
timings, so the policy needs no offline profile and adapts to quantization,
TP width, clocks and thermal state. Draft expert traffic is forecast from each
request's recent committed routes. The drafter's confidence is recalibrated
per position online (Platt scaling fitted by an online Newton step).

Removed: the joint suffix-removal selector, the confidence cutoff and reuse
floor, the placement cost profiles, `DS41RT_ADAPTIVE_COST_MODE` and
`DS41RT_ADAPTIVE_COST_PROFILE`, the `--dspark-confidence-cutoff`,
`--dspark-reuse-floor` and hidden `--dspark-adaptive` flags, and the offline
fit scripts. `--dspark-fixed` (`DSPARK_DRAFT_POLICY=full`) remains the
verify-everything control. `/v1/stats` exports the policy under
`dspark_policy`: fitted costs and implied bandwidth, prediction error, draft
length and width histograms, acceptance totals and per-position confidence
reliability. Route capture now also covers the single-lane sampled path.

dSpark now drafts its trained five-token block by default on both layouts
(two RTX previously drafted seven). `--dspark-draft-limit 7` loads the
extended block and chooses 5 or 7 per round; it recovers two-RTX counting
decode but measured 2–5% slower on code concurrency, so it is opt-in.

## Evaluation

Against v12, in four interleaved sessions per arm per layout
([report](release-v13-performance.md)): on two RTX, v13 is 2–5% faster on
per-case greedy and sampled C1 decode, topic C1, and code, topic and mixed
C2–C16, with the weighted batteries tied. On one RTX it is 2–3% faster on
per-case greedy and sampled C1 decode and ties elsewhere. Losses: topic C1 on
one RTX (−4.8%), code C1 on two RTX (−2.1%), and two-RTX counting (about 195
versus 221 tok/s, from the five-token default). No session had a quality
failure.

Release decode (official checkpoint, four Sparks, C1 dSpark):

| Measurement | 1 RTX | 2 RTX |
|---|---:|---:|
| Weighted decode | 91.10 | 108.91 |
| C1 code decode | 134.01 | 161.20 |
| Counting decode | 159.35 | 192.56 |

## Images

The checkpoint is unchanged:
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`.
The Spark image advertises `tp2;tp3;tp6` as well as the default TP4 shard.

| Role | Published tag | Architecture | Registry manifest digest |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v13` | amd64 | `sha256:f0a7c245be0224ee6d472bc57c6e42eb578ba6ba8b5b2195afe558937c5282de` |
| Spark expert | `ghcr.io/tpurtell/ds41rt-spark-expert:v13` | arm64 | `sha256:42d867515f16d37794127477d16d3d2de9cc40e0e6bac6cade4a784548a9ba86` |

The Spark local image ID is
`sha256:2ce1637492b251e9c3a33393bef16181f3d0d9448ee9e58a78a861b4db9b4f62` on
ostrich, dodo, emu and kiwi. Both labels name source `00b25e7` and SparkInfer
`4b0954148523b5a2e93813f963d483ffd350b9c9`. Before the push, `latest` named the
v12 pair (`ea80ce9f…` coordinator, `d28bd3fe…` Spark) and both `:v13` tags were
absent; after it, `latest` resolves to the v13 digests above.

The v13 Spark image was distributed only to the four default Spark hosts. The
six-Spark example configurations (`tp2ep3`, `tp3ep2`, `tp6ep1`) need
`docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v13` on rhea and moa before
their dry runs pass; every other example and both default layouts pass
`./run.sh --dry-run`. V12 remains available as the previous numbered pair.
