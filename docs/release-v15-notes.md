# DS41RT v15 release notes

**Status: PUBLISHED.** The v15 image pair was built from source
`bd06bec42e219a65097235dd9092e4a0537d4b7a`, measured on the rows it changes, pushed to GHCR,
and verified by anonymous pulls on the matching architectures. `latest`
resolves to the same pair. The runtime default and the example configurations
name `:v15`.

## What changed

**Speculation for tool-enabled requests.** Any request that offered tools lost
dSpark speculation for all of its free text: prose, answers, and everything
around a tool call. Grammar requests trim draft proposals before verification,
and the trim treated XGrammar's `is_completed` as "stop drafting". That check
means the grammar's root rule *could* end here, and free text around tool-call
tags can always end, so every draft was discarded and each verification round
emitted one token. Drafts now stop only at the first token the grammar rejects,
or at a stop-token draft. A finished JSON value still admits only the stop
token, so schema-constrained output is unchanged. Agent clients that send their
tool list on every turn get the largest gain.

Tool-enabled decode, greedy, 400 output tokens, tokens/s. Before is the v14
code path on the same stack and session; after is the fix:

| Request | C1 before | C1 after | C4 before | C4 after |
|---|---:|---:|---:|---:|
| Tools offered, thinking off | 54–57 | 98–111 | 142 | 210 |
| Tools offered, thinking on | 60–63 | 101–109 | 154 | 220 |
| No tools (control) | 104–109 | 89–109 | 216 | 207 |
| JSON schema | 105–110 | 105–111 | 153 | 148 |

The first after-control request (89) ran while graphs were still being
captured after a restart. On the published v15 images:

| Request (C1 range of two, C4) | 1 RTX C1 | 1 RTX C4 | 2 RTX C1 | 2 RTX C4 |
|---|---:|---:|---:|---:|
| Tools offered, thinking off | 83.6–84.3 | 152.0 | 100.2–109.1 | 206.3 |
| Tools offered, thinking on | 83.2–89.5 | 160.1 | 102.5–108.7 | 215.4 |
| No tools (control) | 85.8–89.9 | 157.6 | 107.2–110.9 | 202.4 |
| JSON schema | 86.5–91.0 | 143.7 | 104.7–111.5 | 151.5 |

Tool-enabled requests now decode as fast as the same prompts without tools on
both layouts.

**Live engine console.** Opening the API port in a browser (`GET /`) shows a
real-time view of the engine:

- headline rates (output, accepted and proposed draft tokens, prefill), with
  60-second sparklines and peaks;
- concurrency per lane, KV pool pages (active, retained, total) and host KV
  offload;
- an execution-lane ticker with one row per request and one column per
  verification round, sized by the round's real duration and stacked by draft
  position (accepted, target token, rejected, not verified, grammar
  constrained), plus prefill chunks in the lane that ran them;
- an optional text view of the tokens each request is generating;
- pipeline micro-step timings (lane cycle, draft, verification, per-layer
  device time from the policy's CUDA events, Spark expert wait, prefill),
  a 39-layer profile, dSpark acceptance by draft position and recent requests.

The page is compiled into the binary and fed over a WebSocket
(`/v1/console`); `/v1/console/snapshot` returns the same state as JSON. The
CUDA worker does no console work unless a viewer is connected: one relaxed
atomic check per round plus always-on lifetime counters, which `/v1/stats`
now also exports as `totals`. With a viewer connected, rounds hand small events
to a bounded channel and a separate thread encodes and fans them out. The
micro-step timings reuse timers the engine already takes.

The text view is off by default because anyone who can reach the API port can
then read every session's output. Enable it with `--console-text` or
`DS41RT_CONSOLE_TEXT=1`, which `run.sh` forwards.

Overhead, two RTX and four Sparks, greedy, 600 output tokens, median tokens/s:

| Coordinator | C1 | C8 |
|---|---:|---:|
| v14 | 174.1 | 296.9 |
| v15 code, no viewer | 173.8 | 293.4 |
| v15 code, viewer on the text feed | 175.3 | 288.5 |

The C8 arms overlap completely (v14 alone ranged 287–306); a separate
same-binary ABBA gave 295.0 without and 297.1 with a viewer.

**Tooling.** `scripts/console-load.py` drives mixed traffic (prose, code,
reasoning, JSON schema, tool calls and long prompts) for watching the console.

## Evaluation

V15 changes decode only for grammar-constrained requests, so the v14 decode
campaign stands for every other row. The schema-JSON rows were remeasured on
the published v15 images with the v14 protocol (fresh launch, discarded warmup
battery, three repeats, nonce seeds 61001 for decode and 63001 for acceptance):

| Schema JSON | 1 RTX | 2 RTX |
|---|---:|---:|
| dSpark decode, tok/s (v14) | 138.74 (135.55) | 148.66 (159.10) |
| Acceptance, accepted/verified (emitted per round) | 89.62% (4.17) | 82.64% (4.03) |

The decode cells rerun the whole v14 nine-case battery, because a single-case
run assigns different nonces and so different prompts; only the schema row is
published. Its responses are token-for-token the same length as v14's (46, 46
and 46 tokens on one RTX; 41, 41 and 46 on two). The two-RTX median fell
because v14's third repeat (159.1) was a single high sample: a fresh v14 launch
on the same prompts, run as a control in the same session, measured 147.85
(147.3, 167.1, 147.9) against v15's 148.66 (147.9, 167.9, 148.7). The battery's
weighted scores were 110.04 (one RTX) and 131.03 (two RTX), against v14's
104.69 and 129.67; the README keeps v14's other rows.

Acceptance uses v14's schema prompts (nonce seed 63001) but counts drafts with
the exact lifetime totals the console exports. `/v1/stats` republishes at most
once per second and stops refreshing when the engine goes idle, so for runs of
three 41–46-token responses it misses part of the run: recomputed that way,
v15's own run reads 94.74% (4.86) on one RTX. V14's published schema acceptance,
96.39% (5.21) and 82.29% (4.29), came from `/v1/stats` and carries the same
error; emitted tokens per round here counts request rounds as output tokens
minus first tokens minus accepted drafts.

Target-only decode drafts nothing, so its schema column is unchanged from v14.

Tool calling (high-effort thinking, C16, 4096-token cap, one run) on the
published images: 155/176 (69 pass, 17 partial, 2 fail: TC-43 and TC-68). The three 2026-09-16 campaigns scored 155,
160 and 156 of 176; on the development build of the same code, one run scored
162/176 (75 pass, 12 partial, 1 fail). TC-43 failed in every campaign and
TC-68 in the third 2026-09-16 run.

## Images

The checkpoint is unchanged:
`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`.
The Spark image advertises `tp2;tp3;tp6` as well as the default TP4 shard.
Both images were built from `bd06bec` with SparkInfer
`7fcc094edcc93af61fdfbe14300100e3204363ea`; native code and dependency pins are
identical to v14.

| Role | Published tag | Architecture | Registry manifest digest |
| --- | --- | --- | --- |
| Coordinator | `ghcr.io/tpurtell/ds41rt-coordinator:v15` | amd64 | `sha256:edc93580ac95bf6c33f97f430fc442761a23fa7aeaca394515384552c620d62a` |
| Spark expert | `ghcr.io/tpurtell/ds41rt-spark-expert:v15` | arm64 | `sha256:92e8a70dca41787da332aa567078769f605220dadcff5bb1ff630494d5804988` |

The Spark local image ID is
`sha256:0a6c0faeea74c70980a2d10db8f72cd9e80adf836610075e2049e47ecb85cb9b` on
ostrich, dodo, emu and kiwi. Before the push, `latest` named the v14 pair
(`c4949373…` coordinator, `41d6cd3f…` Spark) and `:v15` was absent; after it,
`latest` resolves to the v15 digests above. Both roles were verified by
anonymous pull on their matching architectures.

The v15 Spark image was distributed only to the four default Spark hosts. The
six-Spark example configurations (`tp2ep3`, `tp3ep2`, `tp6ep1`) need
`docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v15` on rhea and moa before
their dry runs pass; both default layouts and every other example pass
`./run.sh --dry-run` at their supported RTX counts. V14 remains available as
the previous numbered pair.

## Evidence

The [GitHub release](https://github.com/tpurtell/ds41rt/releases/tag/v15)
provides `v15-evidence.tar.gz` (1,264,593 bytes; SHA-256
`6d36416f901197d6f42ea53e0450f038d8c0a882f1f446afb85ae0cd3705a107`), the payload
`SHA256SUMS` (93 files; SHA-256
`c76427a8133a751e0488b41045d2c8f1cb6d7a6cdcd755fe348831908097f1f2`), and
`v15-release-assets.sha256`. The package holds the focused campaign on the
final images (schema-JSON decode and acceptance, the v14 control, tool-enabled
decode and the tool evaluation), the publication records, the release build
log, and the development-session measurements cited above: the console
overhead A/B, tool-enabled decode before and after the fix, and the
development-build tool evaluation. It was scanned for token and private-key
patterns before upload.
