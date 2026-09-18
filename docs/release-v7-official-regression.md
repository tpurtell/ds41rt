# Official checkpoint: v7 regression check

The v7 release images were run against the **official (Original) MXFP4
checkpoint** to confirm that the NVFP4 and EXL3 work did not regress the
default model. It did not: v7 measures marginally faster than the recorded
v6 campaign, and it reproduces the v6 deployment geometry exactly.

This check exists because the official columns in the [README](../README.md)
are the historical [v6 campaign](release-v6-performance.md), explicitly
**not re-campaigned for v7**. Nothing in the v7 reports proved the default
path was unchanged, so this is that proof.

## Method

- **Artifact under test**: the published images
  `ghcr.io/tpurtell/ds41rt-coordinator:v7` (`sha256:53af6703391f…`) and
  `ghcr.io/tpurtell/ds41rt-spark-expert:v7` (`sha256:81918fe41e2c…`) — the
  shipped containers, not a working-tree build.
- **Checkpoint**: `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`,
  the same revision the v6 campaign used.
- **Topology**: `1x` — one RTX PRO 6000 with four GB10 Sparks over RoCE
  (`--rtx-gpus 1`, `spark_world=4`), the v6 single-card configuration.
- **Workload**: `scripts/bench-ds41-release-decode.py`, three repeats,
  `--nonce-seed 61001`, `--include-counting`. This is the v6 command verbatim
  apart from the label, so the per-case nonce assignment and warmup order are
  identical. All 30 samples passed their objective checks.

### Conditions match the v6 campaign

| Control | v6 campaign | This run |
|---|---|---|
| Power limit | 400 W | 400 W |
| Memory clock | 14,001 MHz, no overclock | 14,001 MHz, no overclock |
| RTX driver | 595.91.07 | 595.91.07 |

### Deployment geometry is identical

Every placement and capacity figure the v7 coordinator logged matches the
recorded v6 deployment block:

| Field | v6 record | v7 observed |
|---|---:|---:|
| RTX expert layers | 5 | 5 |
| Remote dispatch (active Spark) layers | 35 | 35 |
| Global FP4 pool bytes | 16,700,672,000 | 16,700,672,000 |
| Runtime headroom bytes | 2,147,483,648 | 2,147,483,648 |
| Device tokens | 18,736,128 | 18,736,128 |
| Host cache tokens | 2,235,904 | 2,235,904 |
| Pinned host cache bytes | 3,489,660,928 | 3,489,660,928 |
| Source pages | 36,650 / 36,650 / 36,650 / 73,300 | 36,650 / 36,650 / 36,650 / 73,300 |

## Results

Tokens/s, median of three repeats.

| Measurement | v6 (recorded) | v7 (this check) | Change |
|---|---:|---:|---:|
| **C1 code decode** | 130.41 | **134.38** | +3.0% |
| Weighted nine-category decode | 92.00 | **93.57** | +1.7% |
| Counting (1–200) decode | 161.58 | **166.05** | +2.8% |

Per-content-type decode (v7):

| Content | Tokens/s |
|---|---:|
| Code | 134.38 |
| Code with reasoning | 102.35 |
| Math | 132.72 |
| Fable | 57.92 |
| Hello | 62.45 |
| Topic | 74.04 |
| Natural JSON | 105.51 |
| Schema JSON | 108.96 |
| Multilingual | 74.95 |
| Counting 1–200 | 166.05 |

## Reading

The three headline figures land within a few percent of the v6 record and
slightly above it. Two independent reviews of the v6→v7 diff reached the same
static conclusion before this measurement: every quantization-conditional in
the shared path resolves to the v6 accessor when the format is native, the
shared route reducer's official entry points (`reduce_compact<4>`,
`finish_local<float>`) are the v6 instantiations, and the new engine guards
(SM-count tolerance and the cooperative-grid clamp) only ever widen or reduce,
never alter, the 188-SM launch this host performs. The only behavioural change
on the official path is `[T; 4]` → `Vec` allocation sizing in the transport
and route-planning structures, which this measurement bounds at noise level.

## Provenance

Raw result: `single-decode.json`, SHA-256
`b8cc1f293d194723c83a6cdb6f66aa6234dfe031c95a26a0ff65a1e3f912e99e`,
in the v7 official-regression package. A first attempt was discarded unread
because the coordinator container lacked the RDMA device set; its file is
retained as `single-decode.misconfigured-transport.json`.

## Outstanding

- The **2x** official configuration (`--rtx-gpus 2`, Spark TP2) was not
  re-measured in this check; its v6 figures remain historical.
- Prefill, retained-context and concurrency sweeps for the official
  checkpoint were not re-campaigned for v7.
