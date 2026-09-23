# DS41RT v12 sampling performance

The v12 image source is `b23a13011020286d5e1f9d7477c6f6c71ab637cc`. This
release moves the served stochastic target sampler to CUDA for ordinary and
constrained decoding. The production host still routes top-k values above its
256-entry retained-list capacity to the CPU. The campaign below measures the
official DeepSeek V4.1 Flash checkpoint on one selected RTX PRO 6000 Blackwell
and four Spark workers with dSpark enabled; the second RTX is idle.

## Sampling contract and validation

The five measured vectors are greedy (`temperature=0`), low-temperature nucleus
(`temperature=0.2, top_p=0.95`), standard nucleus (`temperature=0.7,
top_p=0.9`), min-p (`temperature=0.7, min_p=0.05`), and top-k
(`temperature=0.7, top_k=40`). Every report explicitly sets the other filters
to their disabled values and supplies a seed. The qualifier exercises the
same controls on unconstrained, JSON-schema-constrained, and tool-call output.

The final CUDA source passed the native sampler selftest (**401 cases, 43,648
assertions**) on SM120 and compiled for SM121. The frozen seeded campaign
passed uniform-edge, replay, distribution, speculation, raw-API, mixed-row,
boundary-equality, and reference-precision checks. The retained ordered path
matched the CPU on **147,456/147,456 draws**. The full-survivor top-p path
selected a different token on **50,984/196,608 draws (25.9318%)**, while all
recorded tokens stayed in range and within the allowed set. This is a measured
CPU/GPU floating-point accumulation residual, not seed instability: repeated
device runs are deterministic, and the warp aggregation leaves the ordered draw
CSV byte-identical to the earlier radix implementation. The fast min-p/no-top-k
path differed on **119,988/983,040 draws (12.2058%)**. The 12 distribution
cells passed the top-64 analytic-noise verdict. Raw whole-vocabulary support
TV is too sparse at 8,192 draws to serve as distribution evidence. Exact mask,
greedy, tie, and boundary policies passed; the per-cell allowances are preserved
in [release-v12-sampler-residual-bounds.json](release-v12-sampler-residual-bounds.json)
(SHA-256 `a53f8ec77ece2f2ce63a579f5fb731acded841e72e856dbe6527546c3f6192a9`).

The live v12 native API smoke check passed. The broader sampler qualifier
passed **92 checks, 0 failures** across all five vectors, including seeded
replay, invalid controls, constrained JSON, and strict tool arguments. Its
low-temperature two-seed text probe happened to produce the same answer; the
synthetic distribution campaign above is the stochastic evidence for that
vector.

## Fixed-logit sampler latency

These are CUDA-event medians of 48 repetitions after five warmups, on the
129,280-token long-tail fixture. They measure the sampler call chain only, not
model execution, scheduling, Spark communication, or end-to-end decode. The
profiling library was built from the frozen source (`libds41rt_native.so`
SHA-256 `95ca9514661c8db2704eb3ff28ec4c01739eb5c115e0769a1b258c36774c4082`)
and the profiler executable's SHA-256 is
`3a97144f0d83db92d66dfbab856d7ceecf17f785cf9148806459c49488f40a9b`.
The standalone release image carries a separately built native library from
the same source; the image artifact manifest records its own hash.

| Mode | 1 row (ms) | 4 rows (ms) | 48 rows (ms) |
| --- | ---: | ---: | ---: |
| Greedy | 0.0317 | 0.0317 | 0.0318 |
| Temperature 0.2, top-p 0.95 | 0.8500 | 0.8571 | 0.8694 |
| Temperature 0.7, top-p 0.9 | 0.9482 | 0.9493 | 0.9627 |
| Temperature 0.7, min-p 0.05 | 0.1197 | 0.1527 | 0.1588 |
| Temperature 0.7, top-k 40 | 0.3267 | 0.3616 | 0.3698 |

The four-row nucleus call takes **1.35 ms** on an all-tied row and **1.67 ms**
on a near-uniform row at `temperature=0.7, top_p=0.9`. Those are deliberately
wide survivor sets. The fixed-point warp reduction preserves the exact integer
mass sum while limiting contended shared-memory atomics; the all-tied case was
27.39 ms before that reduction and 1.35 ms afterward on the same profiler.
The live campaign is the release throughput measure because these kernel times
alone cannot establish the effect on accepted dSpark tokens.

## Live decode campaign

The final report records five discarded profile warmups followed by three
interleaved repeats of nine weighted content cases and one weight-zero counting
diagnostic. The weight sum is 8.0: seven cases have weight 1 and natural JSON and
schema JSON each have weight 0.5. Each profile has a distinct prompt nonce,
fixed across repeats, so columns are not identical prompts. Each case uses its
natural output budget; decode tokens and elapsed time are retained in the raw
reports. The weighted figure is the median of each repeat's weighted
tokens/second ratio, not a mean of the case medians. The campaign validator
checks all 15 raw reports, quality and cache gates, the interleaved order,
and the captured image/hardware identity.

The campaign ran on 2026-09-23 with coordinator local image ID
`sha256:ea80ce9f313d9e708d34007cc21d4cc2c09b051936e94905fa6f7b7a19cd6814`
(amd64) and Spark local image ID
`sha256:3e989038ca1f88f6b3dcc96fdd1f2540a43708814fbf7a5db068762277fbbc7c`
(arm64, identical on ostrich, dodo, emu and kiwi). Both image labels name
`b23a13011020286d5e1f9d7477c6f6c71ab637cc`; the launch configuration's
SHA-256 is `30e174048ba1952a47bdfe1bfa3b3c16c60096c7a5e50c23bdece03826a8aba4`.
The selected RTX was `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`, at a 400 W
limit and standard 13,365 MHz loaded memory clock. The other card was idle
(12 MiB allocated, 0% utilization in the mid-campaign snapshot).

The validator passed **15/15 raw reports with zero failures**. Every weighted
case and the weight-zero counting diagnostic passed, and every weighted sample
had zero cached tokens. The weighted values below are medians of the three
per-repeat weighted decode ratios, in observed decode tokens per second:

| Sampling mode | Weighted median | Repeats in time order | Spread |
| --- | ---: | --- | ---: |
| Greedy | 94.17 | 92.09 / 96.04 / 94.17 | 3.94 |
| T0.2, top-p 0.95 | 89.65 | 87.57 / 93.59 / 89.65 | 6.02 |
| T0.7, top-p 0.9 | 89.73 | 91.86 / 89.73 / 89.03 | 2.84 |
| T0.7, min-p 0.05 | 90.91 | 89.82 / 90.91 / 91.24 | 1.42 |
| T0.7, top-k 40 | 89.64 | 92.07 / 87.37 / 89.64 | 4.70 |

Per-content-case medians of three observed decode rates, in tokens per second:

| Content case | Greedy | T0.2 / p0.95 | T0.7 / p0.9 | T0.7 / min-p0.05 | T0.7 / top-k40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 138.5 | 133.0 | 129.0 | 127.3 | 133.3 |
| Code with reasoning | 103.7 | 100.4 | 97.6 | 100.4 | 95.2 |
| Math | 134.1 | 131.6 | 140.7 | 126.5 | 132.7 |
| Fable | 61.0 | 57.3 | 54.6 | 57.6 | 57.6 |
| Hello | 65.3 | 72.2 | 65.1 | 65.5 | 75.2 |
| Topic | 74.3 | 69.4 | 73.0 | 72.5 | 75.0 |
| Natural JSON (0.5) | 98.6 | 96.6 | 95.4 | 85.2 | 103.5 |
| Schema JSON (0.5) | 104.3 | 99.7 | 93.6 | 97.0 | 112.1 |
| Multilingual | 76.8 | 76.6 | 70.4 | 74.6 | 71.1 |
| **Weighted (sum w = 8.0)** | **94.17** | **89.65** | **89.73** | **90.91** | **89.64** |
| Counting 1–200 (weight 0) | 166.4 | 162.6 | 162.4 | 166.2 | 164.9 |

The four stochastic weighted medians are within 1.27 tok/s of each other,
smaller than their observed repeat spreads; these data do not resolve an order
among them. Greedy is 3.26–4.53 tok/s higher than the stochastic medians, but
the low-temperature repeat spread is 6.02 tok/s, so even that separation is
not uniform across repeats. Individual content cases vary more widely.

## Paired v11 comparison

The v11 final campaign and v12 used the same natural-budget corpus, nonce and
sampling seeds. A comparison of all 150 `(profile, repeat, case)` pairs found
**identical request bodies, completion-token counts, and response-content
SHA-256s**. The measured output workload is therefore paired exactly, despite
the synthetic CPU/GPU residual above. It was run on different dates, so normal
deployment drift is still possible. The table compares weighted medians, not
the per-content medians:

| Mode | v11 CPU sampler | v12 GPU sampler | Change |
| --- | ---: | ---: | ---: |
| Greedy | 95.46 | 94.17 | −1.36% |
| T0.2, top-p 0.95 | 80.58 | 89.65 | +11.25% |
| T0.7, top-p 0.9 | 83.29 | 89.73 | +7.74% |
| T0.7, min-p 0.05 | 88.99 | 90.91 | +2.16% |
| T0.7, top-k 40 | 87.74 | 89.64 | +2.17% |

Both nucleus gains exceed either campaign's own repeat spread; the smaller
changes for greedy, min-p and top-k do not. The paired weighted token totals
are 6,608 / 7,166 / 6,870 / 7,013 / 6,067 in the table's mode order in both
releases. The paired record is `paired-v11-v12.json` in the release evidence
package; it names both source commits, verifies all 150 sample pairs and records
the aggregate hashes.

## Evidence identities

The release evidence package contains the 15 raw reports, five discarded
warmups, campaign validator and aggregate, the paired comparison, API
qualification, native selftest, fixed-logit profiler JSON, seeded GPU
validation, build and image verification, and publication records. These
SHA-256 values identify the key inputs before archive packaging:

| Artifact | SHA-256 |
| --- | --- |
| Campaign identity | `3111cef554c12ff5c6ccc06a84280ba918a522e2f1208c4ba8396e79476be04d` |
| Campaign validation | `5b21e51a4c15422866a9959c87043bd3949c71195c536049fc5bfd1a558a0687` |
| Campaign aggregate | `0d3b337141ddd3b9f782ebf74badffb5fbbc542459feea7ec068dc03243e18be` |
| Paired v11/v12 audit | `b04d2a173d76698a1f7adf99e15225ec8ff4970d38b86fc4914c15c37b3dc3f3` |
| Live sampler qualification | `86ca7a001a0bbd2f4195855d67cfe895a47197b9646e26987a05036b936a5801` |
| Final seeded validation summary | `d2c25b5357bbbc9d2c43d7d02299083630c5f9a445e6468e8f79127877e5648b` |
