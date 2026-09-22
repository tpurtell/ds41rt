# DS41RT v11 performance

**STATUS: PUBLISHED.** This page carries two measured datasets and the
optimization and diagnostic evidence behind them, and it has been through the
campaign audit and the release review. The image pair is published, so the
registry manifest digests in the provenance table and §9 are the registry's own
values, each re-checked by an anonymous pull on the matching architecture.

The page is linked from [release-v11-notes.md](release-v11-notes.md). It replaces
the illustration of a single headline with two explicitly labelled datasets plus
the control and research records that explain them.

## Provenance: five distinct revisions

No two of these are the same commit, and the release record must not conflate
them. The tag annotation will name the immutable image build source; the checkout
at the tag is not claimed to be byte-identical to it.

| Role | Revision | Meaning |
| --- | --- | --- |
| Final image build source | `fb5115466a8c70c063280e25957577f284e903e3` | the sampler optimization; exports `org.opencontainers.image.revision` |
| Final image (coordinator) | `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89` | **local image id**, version `v11`, amd64 |
| Final image (spark expert) | `sha256:f1233987c3b9ba13d468d621c96945052db15e7ede9a8feaf101f506aee85516` | **local image id**, version `v11`, arm64, identical on all four required hosts |
| Baseline image build source | `9d9b3e0ce8bd85f6b8eced515d0ab00be015f2d0` | the pre-optimization candidate |
| Qualification host | `8a7da7da9bd30d4ec797926ee4c5c8c6cef734d7` | host HEAD that produced the live qualification |
| Benchmark host | `ec7cfb5679696aa44d81486cf36e26d25595cfbc` | host HEAD that produced the baseline campaign |

The two rows above are **local image ids** (`docker image inspect`), which is what
`identity.json`, the campaign reports and the identity chain reference. They are a
different kind of value from a registry manifest digest and must not be
substituted for one. The published pair's **registry manifest digests** are:

| Role | Registry manifest digest |
| --- | --- |
| Coordinator `:v11` (amd64) | `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89` |
| Spark expert `:v11` (arm64) | `sha256:fc50ac1cf7f309727d406962168fbe278807848566ae84abde0709b9ca5ba27a` |

For the coordinator the local id and the manifest digest coincide; for the Spark
expert they differ, as expected between an image config id and a manifest digest.
`latest` resolves to the same two manifest digests.

The engine baseline is `3f8ea80d8e728cf04832dd520c22a9bbf71f8827`; the final
image build source carries it.

## 1. Baseline dataset - attempt-02, image source `9d9b3e0`

One RTX coordinator plus four Spark workers, official native checkpoint, dSpark
on, five strict profiles, three repeats, natural per-category budgets.

| Profile | Weighted median (tok/s) |
| --- | ---: |
| greedy | 94.41 |
| temp0.2 + top_p0.95 | 47.35 |
| temp0.7 + top_p0.9 | 47.52 |
| temp0.7 + min_p0.05 | 88.26 |
| temp0.7 + top_k40 | 47.14 |

Validator: **PASS**, zero failures, 15/15 raw reports, cache gates clean,
identity deep-equal.

- `runs/sampling-1rtx/attempt-02/campaign-validation.json`
  `e513ae13d48e14bc52f7674d5db6320b5369bd7dc7888153fae9ce651d2e5a1b`
- `runs/sampling-1rtx/attempt-02/campaign-aggregate.json`
  `0dc1d8e33878af94077465d73fffd810953f2f980906fadd2d0b08108c607211`
- `runs/sampling-1rtx/attempt-02/identity.json`
  `a801fefb1b0a162985eff36ce093061763427ebbb92c99a0f26f9ad517a72ae1`
- `runs/sampling-1rtx/attempt-02/campaign-progress.log`
  `252fe2313bc7570a3e46d60e118c799bc1939638d8347ac257b4acba31cd9e32`
- Preserved README measured draft:
  `runs/sampling-1rtx/attempt-02/readme-measured-draft.diff`
  `7dee855331a47d9d82079523ccab30e6598fe2d9fb24b763ede72459d6174af5`

These baseline numbers are the **pre-optimization record**, not the release
table. They remain in this page so the optimization's effect is legible rather
than asserted.

A first attempt (`attempt-01-cache-collision`) was rejected before measurement:
every profile shared one nonce seed, so identical prompts hit the same prefix
cache and the run failed the cache gate. It is preserved as evidence and does not
count toward any number here.

## 2. Final dataset - attempt-03, image source `fb51154`

Same topology, checkpoint, profiles, repeats, budgets and nonce rules; only the
image build source differs.

| Profile | Baseline `9d9b3e0` | Final `fb51154` |
| --- | ---: | ---: |
| greedy | 94.41 | 95.46 |
| temp0.2 + top_p0.95 | 47.35 | 80.58 |
| temp0.7 + top_p0.9 | 47.52 | 83.29 |
| temp0.7 + min_p0.05 | 88.26 | 88.99 |
| temp0.7 + top_k40 | 47.14 | 87.74 |

Validator: **PASS**, zero failures, 15/15 raw reports, identity deep-equal,
cache gates clean.

- `runs/sampling-1rtx/attempt-03/campaign-validation.json`
  `476b233029ee01cb200a7632463dea2a1f0e509b8b256140a0bab79ebfc5ba5b`
- `runs/sampling-1rtx/attempt-03/campaign-aggregate.json`
  `cb1ca666b5cbfeeea3d39fd8d44619db093255248676836d3fe426d92e5745ff`
- `runs/sampling-1rtx/attempt-03/identity.json`
  `285544d4e350bacc4137031ac884aded92e73f6fc207ccaf44584e18bedc3653`

**Statistical presentation (final dataset).** Only **greedy** is clearly first:
95.46, a margin of **+6.47 to +14.88 TPS** over the other profiles, whose own
spreads are 0.97-5.73. The **four stochastic profiles form one tight cluster and
must not be ordered internally**, because each apparent ordering is within that
profile pair's noise:

| Profile | Repeats (tok/s) | Spread | Median |
| --- | --- | ---: | ---: |
| greedy | 92.9590 / 97.3036 / 95.4597 | 4.34 | 95.46 |
| temp0.7 + min_p0.05 | 88.2909 / 88.9857 / 89.2613 | 0.97 | 88.99 |
| temp0.7 + top_k40 | 88.2760 / 86.2790 / 87.7424 | 2.00 | 87.74 |
| temp0.7 + top_p0.9 | 84.1616 / 83.2861 / 82.5288 | 1.63 | 83.29 |
| temp0.2 + top_p0.95 | 78.6887 / 84.4165 / 80.5842 | 5.73 | 80.58 |

- temp0.7+min_p0.05 versus temp0.7+top_k40: margin **+1.24** against a 2.00
  spread - within noise.
- temp0.7+top_p0.9 versus temp0.2+top_p0.95: margin **+2.70** against a 5.73
  spread - within noise.

Greedy is presented as clearly faster and the four stochastic modes as one
cluster; no 2/3/4/5 ordering is claimed.

### Workload identity: the gain is elapsed time, not a workload artifact

The two campaigns measured the **same work**. Weighted token counts are exactly
equal between baseline and final for every profile - greedy 6608,
temp0.2+top_p0.95 7166, temp0.7+min_p0.05 7013, temp0.7+top_k40 6067,
temp0.7+top_p0.9 6870 - and the per-case median completion tokens are identical
across every profile and case (all nine weighted cases plus the counting
diagnostic).

The entire difference is therefore reduced elapsed time for the same output:

| Profile | Baseline elapsed (s) | Final elapsed (s) |
| --- | ---: | ---: |
| temp0.7 + top_k40 | 128.44 | 69.37 |
| temp0.7 + top_p0.9 | 144.33 | 82.47 |
| temp0.2 + top_p0.95 | 151.70 | 88.11 |
| greedy | 70.30 | 69.50 (0.989x) |
| temp0.7 + min_p0.05 | 79.82 | 78.89 (0.988x) |

Greedy and min_p were already near the sampler-path floor, which is why their
gain is about 1 percent, while the three ordered-filter profiles cut elapsed time
by roughly 42-46 percent (throughput +70 to +86 percent). Evidence:
`timed_tokens_total` / `timed_seconds_total` in the two campaign aggregates
listed above.

### What changed between the two datasets

`fb51154` replaces the ordered sampler path's per-row structured sort with
bit-identical heap/radix selection; there is no semantic change. The remaining
gap to greedy is the cost of applying the filter chain at all, plus a smaller
speculative benefit for stochastic profiles - see §4.

## 3. The optimization record

Commit `fb5115466a8c70c063280e25957577f284e903e3` ("Optimize ordered sampler path
with bit-identical heap/radix selection"), files:

- `rust/crates/ds41rt-core/src/target_sampling.rs`
  `e73824006806d84f15fcb6f7e49c6cee2f460f7c78724189f4f18da74ffb53a9`
- `rust/crates/ds41rt-core/tests/target_sampling_bench.rs`
  `529076e3976405b5a80bebc4ddb02296cf8d174b34d0073fe9bffb3a72e412eb`

CPU microbenchmark, ordered-path cost before -> after (us per row):

| Profile | Before | After |
| --- | ---: | ---: |
| top_k40 | 7,612 | 345 / 428 |
| top_p0.9 | 7,853 | 938 / 1,088 |
| top_p0.95 | 8,055 | 1,141 / 1,338 |
| min_p0.05 | 261 | 246 (unchanged) |
| greedy | 78 | 67 |

**Bit-identity verification.** Selection identity is checked against a verbatim
copy of the pre-change oracle: adversarial rows across the full parameter grid,
with masks and boundary uniforms, a seeded replay of all five profiles, and V
boundaries. The committed regression tests add cap-boundary oracle cases, the
`CAP < S < V` radix case, the `-inf`-scaled case and a fast-path total
comparison.

**Honest limits of that verification.** It does not diff a separately built
pre-change binary. Fully-tied rows take the radix branch, whose cost is
data-independent, so they can be up to about **1.5x slower** than the old
structured sort while still far below the previous baseline; the near-monotone
top-k heap worst case is about **1.8 ms**.

## 4. The `--no-dspark` control decomposition

Run on the **baseline** image (`9d9b3e0`) with the same images, config, profiles,
budgets, repeats and nonce seeds; only dSpark differs. This separates the host
sampler cost from speculative waste.

| Profile | With dSpark | No dSpark | dSpark speedup |
| --- | ---: | ---: | ---: |
| greedy | 94.41 | 47.72 | 1.98x |
| temp0.7 + min_p0.05 | 88.26 | 46.90 | 1.88x |
| temp0.2 + top_p0.95 | 47.35 | 34.17 | 1.39x |
| temp0.7 + top_p0.9 | 47.52 | 34.56 | 1.38x |
| temp0.7 + top_k40 | 47.14 | 34.88 | 1.35x |

**No profile is faster without dSpark**, so speculation is a net win at every
vector and no adaptive-gate tuning was applied. The control also shows an
intrinsic sampler-path cost for top_p/top_k (about 34.5 no-dSpark versus 47.7
greedy) and a smaller speculative speedup for them (about 1.37x versus 1.98x):
the with-dSpark gap combines host sampler cost and reduced speculative benefit,
it is not acceptance alone.

- `runs/sampling-1rtx/control-nodspark/decomposition.json`
  `53671bad7b5fe857bd87e2209c4a34d3c5cd83ea8c1f775a1d1462bc79825847`
- `runs/sampling-1rtx/control-nodspark/identity.json`
  `fabe1fab6d94563a3d94ee831729d63b2741d18293407c412d4d16c7a7c18644`
- aggregate `deed2424d64c64c8e56eeafee02837cde8547af6fa2b6034f1802cfe7cb4a306`,
  progress `ddcd6f3ad514be9b0c33cd661ba6c990e51884bfe3d21668881d2cedd54b7d74`
  (as recorded inside `decomposition.json`)

Adaptive gating follows the standard result that speculation pays when
`alpha > c`, with expected accepted tokens `(1 - alpha^(gamma+1)) / (1 - alpha)`
(Leviathan Thm 3.8). The control above is the evidence that kept the gate
unchanged for v11.

## 5. Rejection-sampling research decision

Our stochastic dSpark verifier is an **exact speculative-sampling rejection
correction with the draft as a point mass** (`q = delta`): exact for any draft
and temperature, with acceptance equal to `p(draft token)`. No scheme that
receives only a token id or scalar can improve on it.

**Considered and rejected for this release:** full-draft-distribution rejection
correction, which needs a full-vocabulary or candidate-set `q` plus a
normalizer. The gain does not exist at our benchmark temperatures. vLLM PR
#20459 measures accepted length **2.30** (point mass) versus **2.31** (with draft
probabilities) at `T = 0.7`; the gain appears at `T >= 1.0` and becomes large at
`T >= 1.3-2.0`. This is recorded as a **future item for high-temperature
workloads**, not as a defect.

**Correctness advantage to state:** vLLM returns HTTP 400 for `min_p` under
speculative decode because `min_p` is argmax-invariant and was silently dropped
on verified tokens (issues #42744, #42802, #31982). Our verifier applies the
**full filter chain** - mask -> temperature -> `min_p` -> `top_k` -> `top_p` - to
the verified target distribution, so `min_p` is honored under speculation.

**Not guaranteed:** bitwise parity between speculative and non-speculative token
streams. This is an explicit non-guarantee upstream as well (M=1 versus M=k+1
numerics, issue #54506). Our documented guarantee stays exactly: **same logits +
same execution path within this build**.

## 6. dSpark activity on the final image

Untimed instrumented proof on the final image (`fb51154`), one sequential short
request per profile (`hello`, 32-token natural budget) with the debug filter, to
show drafting is active:

| Profile | Rounds | Proposed | Accepted | Accepted fraction |
| --- | ---: | ---: | ---: | ---: |
| greedy | 10 | 29 | 21 | 0.724 |
| temp0.2 + top_p0.95 | 14 | 46 | 17 | 0.326 |
| temp0.7 + top_p0.9 | 6 | 24 | 11 | 0.450 |
| temp0.7 + min_p0.05 | 15 | 42 | 16 | 0.381 |
| temp0.7 + top_k40 | 12 | 33 | 19 | 0.548 |

- `runs/sampling-1rtx/attempt-03/dspark-debug/debug-manifest.json`
  `ecb3fc2609620e83c01337f1b583259f582cbbf66d1728ca1b7ec6ebc49296e7`
- `runs/sampling-1rtx/attempt-03/dspark-debug/parsed-counts.json`
  `7b67f74da76e7f13526ddd8b12661a1300d5d5fa2b7db1613f9cc870c8a9e652`

**Denominator clarification (as the auditor recorded it):** the `Proposed` and
`Accepted` columns above are the raw merged parsed totals, while the accepted
fractions come from the summary files' **filtered** denominators, which exclude
terminal or constrained observations. The two must not be mixed: do not derive a
fraction by dividing the raw columns in this table, and do not compare a raw
total against a filtered fraction. Each profile's own summary is the authority
for its fraction.

**Caveat, as recorded by the harness:** this is **one short request per profile**
and proves active drafting, not a serving acceptance rate. Do not read these
fractions as campaign acceptance; the campaign's own accounting governs.

## 7. What this page does not claim

- It is not a qualification of any topology the campaign did not measure.
- It does not rank the four stochastic profiles against each other; only greedy
  is clearly faster.
- It does not claim bitwise spec/non-spec parity, and it does not claim full
  bit-exact agreement with vLLM: this build resolves exact-`k` ties to the lowest
  token id deterministically, while vLLM may keep ties; the no-temperature
  default is greedy; a seed of `-1` is deterministic here and unseeded in vLLM.
- The greedy device route applies only when the whole batch is greedy and
  unconstrained; a mixed batch forces the full-logit path while its greedy rows
  still resolve by argmax.
- It does not claim the optimization is faster on every input: see the
  tied-row caveat in §3.

## 8. Evidence index

| Artifact | SHA-256 |
| --- | --- |
| `scripts/validate-sampling-campaign.py` | `c1e1f87a507a7cdc45310f05df3c732a5450f7160639509da8bb5fd778f82895` |
| `scripts/aggregate-sampling-decode.py` | `7181fc4772254d72d990dc184c02c1053e07f646e35000aa28bb74fa6f705ed1` |
| attempt-02 validation | `e513ae13d48e14bc52f7674d5db6320b5369bd7dc7888153fae9ce651d2e5a1b` |
| attempt-02 aggregate | `0dc1d8e33878af94077465d73fffd810953f2f980906fadd2d0b08108c607211` |
| attempt-02 identity | `a801fefb1b0a162985eff36ce093061763427ebbb92c99a0f26f9ad517a72ae1` |
| attempt-03 validation | `476b233029ee01cb200a7632463dea2a1f0e509b8b256140a0bab79ebfc5ba5b` |
| attempt-03 aggregate | `cb1ca666b5cbfeeea3d39fd8d44619db093255248676836d3fe426d92e5745ff` |
| attempt-03 identity | `285544d4e350bacc4137031ac884aded92e73f6fc207ccaf44584e18bedc3653` |
| control decomposition | `53671bad7b5fe857bd87e2209c4a34d3c5cd83ea8c1f775a1d1462bc79825847` |
| dspark debug manifest (final) | `ecb3fc2609620e83c01337f1b583259f582cbbf66d1728ca1b7ec6ebc49296e7` |
| dspark parsed counts (final) | `7b67f74da76e7f13526ddd8b12661a1300d5d5fa2b7db1613f9cc870c8a9e652` |
| optimization `target_sampling.rs` | `e73824006806d84f15fcb6f7e49c6cee2f460f7c78724189f4f18da74ffb53a9` |
| optimization `target_sampling_bench.rs` | `529076e3976405b5a80bebc4ddb02296cf8d174b34d0073fe9bffb3a72e412eb` |
| baseline README measured draft | `7dee855331a47d9d82079523ccab30e6598fe2d9fb24b763ede72459d6174af5` |
| re-qualification acceptance (final pair) | `a3ad9b13932769a835e35f0d8c89594641d2dbbe382ac5021d45afe0821cefc0` |
| re-qualification manifest (final pair) | `2a32bdad0a2948116eec0ab54f74870f472c711d5c80042b26c1d36032d01a7c` |
| re-qualification harness | `d83341f0a393c5c51ac2b8a06ffd510923262c523ece6835ccfa8d561a47b385` |

The validator and aggregator scripts carry no source hash inside the campaign
artifacts they produce, so their hashes are recorded here explicitly to bind
provenance.

## 9. Publication state

- **The final pair re-qualified.** Live qualification on image source `fb51154`
  passed **91 checks / 69 observations / 0 failures**, with all replay checks
  exact and an **empty check-outcome delta versus the baseline qualification** -
  the optimization changed no acceptance outcome. Evidence:
  `runs/sampling-requal/attempt-03/sampling-acceptance.json`
  `a3ad9b13932769a835e35f0d8c89594641d2dbbe382ac5021d45afe0821cefc0`, manifest
  `2a32bdad0a2948116eec0ab54f74870f472c711d5c80042b26c1d36032d01a7c`, harness
  `d83341f0a393c5c51ac2b8a06ffd510923262c523ece6835ccfa8d561a47b385`.
- The campaign audit independently recomputed every median and case cell from raw
  and returned **PUBLISHABLE**; its mandatory presentation rule is applied in §2.
- Published registry **manifest digests**: coordinator
  `sha256:e0e5d631a54e84f36cd1cd99e2a7cf4e2092c15919c8c79403007ba0852d0d89`,
  spark expert
  `sha256:fc50ac1cf7f309727d406962168fbe278807848566ae84abde0709b9ca5ba27a`;
  `latest` resolves to the same pair. Anonymous pulls verified on `raptor`
  (amd64) and `kiwi` (arm64).
- The `v11` tag was absent from both repositories before the push
  (`runs/v11-release/publication/v11-absence-recheck.txt`).
- The runtime default is promoted to `:v11`; rollback is the promotion change
  reverted plus the recorded `latest` re-point.
