# DS41RT v14 release notes

**Status: PUBLISHED.** The v14 image pair was built from source
`af9d96923f70af1cd385960e28450e5ec0653ff8`, evaluated against v13 in
interleaved A/B sessions on one and two RTX, measured with the release decode
battery, pushed to GHCR, and verified by anonymous pulls on the matching
architectures. `latest` resolves to the same pair. The runtime default and the
example configurations name `:v14`.

## What changed

V14 shortens the verification cycle: the time from one committed token block
to the next. The checkpoint, precision and output contract are unchanged. The
target keeps its BF16 vocabulary head and FP32 accumulation; nothing moves to
NVFP4. Every change below either leaves outputs bitwise identical or changes
only floating-point addition order inside FP32 sums.

**Device-ordered verification passes.** Each layer used to drain every stage
on the host before the next was submitted: mHC finish, query, window KV,
compressor, index, attention, router staging, shared and local experts, and
the reduction. The RTX idled while the host issued each launch. Verification
passes now order these stages with per-device CUDA events (the stage chain).
The host waits only where it must read results: routes before the Spark
dispatch and the head output. Prefill stays host-drained. On two RTX the chain
spans both GPUs; peer copies settle the chain first. `DS41RT_STAGE_CHAIN=0`
restores host-drained stages as a diagnostic.

**Overlapped cache producers.** The query stage marks a fork once the
normalized layer input is ready. The window-KV and compressor producers read
only that input, so they run alongside the query projections instead of after
them. On one RTX, the compressed-attention layers also stopped draining the
chain on the host (up to 324 µs per layer) and dropped their synchronous
copies.

**Device-timed policy costs.** The dSpark bandwidth policy now fits its layer
costs from CUDA events at each FFN finish instead of host timestamps, which
the chain made misleading. On two RTX the first layer after the GPU handoff is
timed from an entry event on the receiving GPU. Without it, the round cost
never fitted there and the policy stayed cold.

**Two-RTX device sampler.** Sampled rounds on two RTX assemble both vocabulary
halves on the head GPU and run the single-GPU target sampler there. They
download ids, scores and status instead of full logits.

**TP2 route sums.** On two RTX, the 20 RTX-resident layers run routed experts
as TP2 ranks. Each rank now sums its six FP32 route planes per token before
the peer transfer, so one FP32 row per token crosses GPUs instead of six.
`DS41RT_TP2_TOKEN_SUMS=0` keeps route planes.

**Kernels and transfers.**
- Target greedy argmax, the 384-expert router and draft attention use faster
  kernels with bitwise-identical results.
- A single-CTA route planner handles decode capacities (SparkInfer b12x
  `7fcc094`).
- Window KV commits all 40 layers with one upload and one launch.
- Spark workers copy hidden rows on their stream from the mapped request
  frame.

**Tooling.** `run.sh` forwards `RUST_LOG` to Spark workers, quotes remote
launch arguments, and forwards the diagnostic switches above when set.

Measured and not adopted: persisting-L2 prefetch of predicted Spark experts
(served hits, yet the set-aside slowed the expert kernel), an FP8 draft
vocabulary head (acceptance loss cancelled its speed), and several per-layer
upload and dispatch micro-optimizations with no measurable effect.

## Evaluation

Against v13, in four interleaved sessions per arm per layout
([report](release-v14-performance.md)), v14 is faster on every measured
family on both layouts, and every interval excludes 1:

| Family | 1 RTX | 2 RTX |
|---|---:|---:|
| Greedy C1, weighted battery | +16.6% | +18.2% |
| T0.7/top-p 0.9 C1, weighted | +16.4% | +37.7% |
| Code C1 | +12.3% | +13.3% |
| Topic C1 | +16.4% | +18.5% |
| Mixed C1 | +14.4% | +15.1% |
| Code C2–C16 | +7.7% | +7.2% |
| Topic C2–C16 | +10.5% | +17.2% |
| Mixed C2–C16 | +6.4% | +13.6% |

The only key that did not improve is two-RTX code at C16 (1,312 versus 1,299
tok/s, −1.0%). No session had a quality failure.

Release decode (official checkpoint, four Sparks, C1 dSpark):

| Measurement | 1 RTX | 2 RTX |
|---|---:|---:|
| Weighted decode | 104.69 | 129.67 |
| C1 code decode | 146.12 | 179.86 |
| Counting decode | 182.39 | 219.86 |
| Target-only weighted decode | 59.84 | 60.67 |

## Images

The checkpoint is unchanged:
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`.
The Spark image advertises `tp2;tp3;tp6` as well as the default TP4 shard.
Both images were built from `af9d969` with SparkInfer
`7fcc094edcc93af61fdfbe14300100e3204363ea`.

| Role | Published tag | Architecture | Registry manifest digest |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v14` | amd64 | `sha256:c49493737243adb693023d02d0dabbe0b64c56668ada7e55b3f70657811bf75e` |
| Spark expert | `ghcr.io/tpurtell/ds41rt-spark-expert:v14` | arm64 | `sha256:41d6cd3f66ac492a6e5d5318b9ae81c309ae16430a5ae5deba62f14a955f5b4c` |

The Spark local image ID is
`sha256:6720bdb57173e5f88185e1d8c3b41d7285bed255d2301256859fab8cf6301589` on
ostrich, dodo, emu and kiwi. Before the push, `latest` named the v13 pair
(`f0a7c245…` coordinator, `42d86751…` Spark) and both `:v14` tags were absent;
after it, `latest` resolves to the v14 digests above.

The v14 Spark image was distributed only to the four default Spark hosts. The
six-Spark example configurations (`tp2ep3`, `tp3ep2`, `tp6ep1`) need
`docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v14` on rhea and moa before
their dry runs pass; both default layouts and every other example pass
`./run.sh --dry-run` at their supported RTX counts. V13 remains available as
the previous numbered pair.

## Evidence

The [GitHub release](https://github.com/tpurtell/ds41rt/releases/tag/v14)
provides `v14-evidence.tar.gz` (44,511,038 bytes; SHA-256
`9303c2360dae7b2a6d838d13ac785fc2e1db989857b0da749a7ba3fe52f32273`), the payload
`SHA256SUMS` (443 files; SHA-256
`58c76f529a035b82d70c77797a972d8ad190ec92d41e27faad4a2e55952115f6`), and
`v14-release-assets.sha256`. The package holds every A/B session and analysis,
the full release campaign with its `/v1/stats` snapshots and derived tables,
the publication records, the final build log, and the cycle-time program's
measurement tooling and working checklist. It was scanned for token and
private-key patterns before upload.
