# DS41RT v12 release notes

**Status: PUBLISHED.** The v12 image pair was built from immutable source
`b23a13011020286d5e1f9d7477c6f6c71ab637cc`, qualified on one RTX plus
four Spark workers, measured with dSpark enabled, pushed to GHCR, and verified
by anonymous pulls on the matching architectures. `latest` resolves to the
same pair. The runtime default and all example release references name `:v12`.

V11 introduced the sampling controls using a host sampler. V12 moves the
served target-token sampler to CUDA for ordinary and constrained decoding.
It preserves `temperature`, `top_p`, `top_k`, `min_p`, and a deterministic seed
tied to absolute token position. Greedy and min-p/no-top-k use compact GPU
paths. Ordered top-k uses three count-radix passes. Full-survivor top-p uses
deterministic Q32 mass histograms with warp aggregation and separate value and
tie-ID radix selection. The `top_p=1` boundary retains a floating-point search
for the CPU's prefix-saturation behavior. The host still counts and routes
top-k values above its 256-entry retained-list capacity to the CPU.

The checkpoint is unchanged:
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`.
The Spark image advertises `tp2;tp3;tp6` as well as the default TP4 shard.

| Role | Published tag | Architecture | Registry manifest digest |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v12` | amd64 | `sha256:ea80ce9f313d9e708d34007cc21d4cc2c09b051936e94905fa6f7b7a19cd6814` |
| Spark expert | `ghcr.io/tpurtell/ds41rt-spark-expert:v12` | arm64 | `sha256:d28bd3fe888e691378e85a88a9b2af1292ec6c4b6b2c7cb3ca89d3d50e6bec71` |

The coordinator's local image ID is
`sha256:ea80ce9f313d9e708d34007cc21d4cc2c09b051936e94905fa6f7b7a19cd6814`;
the Spark local image ID is
`sha256:3e989038ca1f88f6b3dcc96fdd1f2540a43708814fbf7a5db068762277fbbc7c`
on all four required hosts. Local image IDs and registry manifest digests are
different identifiers. Both labels name source `b23a130` and SparkInfer
revision `4b0954148523b5a2e93813f963d483ffd350b9c9`. The annotated v12
tag points to the later release-documentation/config commit; its annotation
names `b23a130` as the image source rather than claiming byte identity. The registry's
post-push `latest` digests match the v12 pair. The pre-push `latest` digests
were the published v11 pair (`e0e5d631…` coordinator, `fc50ac1c…` Spark),
and both v12 tags returned HTTP 404 before publication.

The native sampler selftest passed 401 cases and 43,648 assertions. The seeded
GPU campaign passed replay, mask, tie, boundary, mixed-row, distribution and
speculation checks. Retained ordered sampling was CPU-token-exact on 147,456
draws. The survivor top-p CPU/GPU token mismatch rate was 25.9318% over
196,608 synthetic draws, and the fast min-p/no-top-k rate was 12.2058% over
983,040 draws; these bounded floating-point residuals are recorded per cell.
The live native API smoke check passed, and the five-vector sampling qualifier
passed **92 checks, 0 failures**, including constrained JSON and tool calls.

The five-mode, nine-content-case campaign passed its validator on 15 raw
reports with zero failures. Its v12 weighted medians of three interleaved
repeats, in observed decode tokens/s, were:

| Mode | v12 | v11 | Paired change |
| --- | ---: | ---: | ---: |
| Greedy | 94.17 | 95.46 | −1.36% |
| T0.2, top-p 0.95 | 89.65 | 80.58 | +11.25% |
| T0.7, top-p 0.9 | 89.73 | 83.29 | +7.74% |
| T0.7, min-p 0.05 | 90.91 | 88.99 | +2.16% |
| T0.7, top-k 40 | 89.64 | 87.74 | +2.17% |

All 150 v11/v12 request bodies, response hashes and completion-token counts
match, making the output workload exactly paired. The two nucleus gains exceed
either run's repeat spread; the smaller changes are within observed spread.
The four stochastic v12 medians are too close to rank among themselves. The
[v12 performance report](release-v12-performance.md) gives every content-type
result, repeat series, fixed-logit latency, identity, and evidence hashes.

The [GitHub release](https://github.com/tpurtell/ds41rt/releases/tag/v12)
provides `v12-evidence.tar.gz` (7,564,224 bytes; SHA-256
`ace298b38b684e84d332e0c29f5d85a02e1c72b745c16fb737d89b50e3b9c29b`),
the payload `SHA256SUMS` (10,803 bytes; SHA-256
`9ad4b8a6ed79c70feb45eb3e557333e8e5f16f28dc6f8eef8a48b9e8785e6230`),
and `v12-release-assets.sha256` (SHA-256
`c28214cdd391409bc413fe3a8d887fc7f152d3630ecd837f3dcb32962c3eeb12`).
The payload manifest verifies 104 files inside the archive; the separate
asset-level manifest names the tarball and payload manifest hashes. The
package was scanned for token and private-key patterns before upload.

`ds41rt.config` and the example configurations now select `:v12`.
`./build.sh --dry-run` and `./run.sh --dry-run` pass. The explicit build-target
file `ds41rt.build-v12.config` remains byte-for-byte frozen at its benchmarked
SHA-256, `30e174048ba1952a47bdfe1bfa3b3c16c60096c7a5e50c23bdece03826a8aba4`.
Its header describes the pre-promotion state; its operational values match
the promoted runtime default. The numbered v11 tags remain available for
rollback.

Seeded GPU sampling is reproducible for the same logits and execution, but is
not universally bit-identical to the CPU's floating-point accumulation. The
synthetic mismatch rates above are not observed end-to-end output differences
in the paired content campaign. Whole-vocabulary TV at 8,192 draws on very
wide support is underpowered; the distribution gate uses top-64 TV against
analytic noise. The live qualifier's two low-temperature seeds happened to
return the same answer on one text prompt, so the synthetic distribution
check is the evidence that the mode is stochastic. dSpark draft acceptance
counters are not exposed by `/v1/stats`; the campaign identity records dSpark
enabled, while the adaptive policy may suppress drafts on individual steps.
