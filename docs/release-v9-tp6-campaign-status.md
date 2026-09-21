# v9 pure-TP6 campaign — status and pending-data index

**STATUS: RELEASE CANDIDATE. Final v9 images built and functionally validated; the performance numbers are validated warm-candidate measurements and registry publication is awaiting push.**

This is the index for the two reports (`docs/release-v9-tp6-1rtx-official.md`, `docs/release-v9-tp6-2rtx-official.md`). It records what is measured, what is retained as invalid/failed, and the exact data still missing. It is not a release note.

## Identity used by both arms

- Model: `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa45a94ad051997016db3960a90277` (official quant).
- Corpus `1972e572fbf9d468cd86c2db5d3d3de60641329edca152fe013a179eb55001de`, tokenizer `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`, nonce 61001.
- Candidate warm-build artifacts: coordinator lib `e9736c3e...`, Spark lib `c98b392d...`, coordinator daemon `a41a76ba...`, Spark daemon `884f4454...`. Warm build is not final container proof.
- Matched baselines measured for this campaign (published-v8 official images, NOT v9): 1x weighted 92.30 tok/s (spread 4.67%), 2x weighted 112.59 tok/s (spread 7.10%).
- Network epoch: A-only / single HCA during the runs; later Moa renumber to `10.55.0.6` and B `10.55.1.1-.6` are annotations. No dual-rail gain is claimed.

## Headline status

| Arm | Source | Weighted tok/s | Samples | Objective | Status |
|---|---|---:|---:|---|---|
| `official-v9-1x-tp6` (5 local / 35 remote) | `~/.cache/ds41rt-tp6-e2e/suite-tp6-1x35/headline-decode.json` | 96.83 | 30 | 15/15 | **VALID** |
| `official-v9-2x-tp6` (20 local / 20 remote) | `~/.cache/ds41rt-tp6-e2e/suite-tp6-2x20-fresh/headline-decode-confirmatory-fresh.json` | 110.19 | 30 | 15/15 | **VALID (fresh process)** |
| `official-v9-2x-tp6` old suite file | `~/.cache/ds41rt-tp6-e2e/suite-tp6-2x20/headline-decode.json` | 113.49 | 30 | 15/15 | **INVALID — prefix-cache contamination; never use** |

## Section completeness

| Section | 1x TP6 | 2x TP6 |
|---|---|---|
| Headline nine-category + counting | complete | complete (fresh) |
| Prefill matrix 30 cells | complete | complete |
| Retained 0/32K/64K/128K | complete (prime64) | complete only at prime16 |
| Retained 262144 | failed (strict reuse) | not run (one-repeat 2x diagnostic only) |
| Retained 2K control | complete | complete |
| Concurrency C1..C16 x3 (counting/code/topic) | complete | complete |
| Mixed traffic (real harness, C4 x3) | complete, median 175.84 | complete, median 188.47 |
| FFN/expert tiling | valid component coverage; no re-run planned | same |
| Runtime memory identity + resolved cost model | complete (existing log) | complete (existing log) |
| Controlled-cost side arm (TP4 legacy) | complete (88.5130) | not available (TP4 2x counts smoke only) |
| Final image identity | PUBLISHED (v9 + latest; anonymous-verified) | PUBLISHED (same) |

## Final clean build (v9)

Clean build rc0, engine revision `5d0d209509bf1f26731bc588dfe8dc72d1373ad0`, SparkInfer `4b0954148523b5a2e93813f963d483ffd350b9c9`, expert roles `tp2;tp3;tp6`. Coordinator `ghcr.io/tpurtell/ds41rt-coordinator:v9` amd64 index digest `sha256:786d1d67...` (linux/amd64 child `sha256:649e5c84...`); spark-expert `ghcr.io/tpurtell/ds41rt-spark-expert:v9` manifest digest `sha256:f0c67407...` (arm64; local config id `sha256:9e8c8248...`). Both `v9` and `latest` are **published and anonymous-verified**. Dist daemons (`8c14fd85...` coord, `c176f2c8...` spark) differ from the measured candidate daemons (`a41a76ba...`/`884f4454...`). Candidate performance is **validated warm-candidate performance** with precise provenance; the final images passed **bounded content-type functional checks for all four deployments** - functional checks, **not** a performance campaign. No final-image performance number is claimed.

## Mixed traffic

The real harness (`bench-real-full-mixed-concurrency.py`, scenario staggered-drain, concurrency 4, repeats 3, official model id passed explicitly) completed on both arms: TP6 1x median **175.84** tok/s (183.91/175.84/168.20); TP6 2x median **188.47** (203.88/179.87/188.47); every repeat `all_http_200=true` with 0 failed requests. Raw: `raw/tp6-1x-mixed.jsonl`, `raw/tp6-2x-mixed.jsonl`. The earlier HTTP 400/404 attempts are root-caused and superseded.

## Campaign-adjacent TP4 arms (context for the cost question)

- TP4 bridge 1x auto and 2x auto are **counts smoke only** (api-smoke rc=0, api-constrained rc=1 retained; counting C1 153.29 and 207.50 tok/s). They are **not** full content-type headlines.
- TP4 legacy 1x full headline (explicit `DS41RT_ADAPTIVE_COST_MODE=legacy`, fresh process, scored first): median weighted **88.51** tok/s (88.5130/88.4647/89.6720), 30 timed, 15/15 assessed, prefix all true. Raw: `raw/tp4-1x-legacy-decode.json`.
- Controlled-cost candidate comparison (both resolve `legacy-heuristic`): TP4 1x **88.5130** vs TP6 1x **96.8271**. This is a **new side arm** for attribution, not a replacement default baseline.
- The primary default comparison against published-v8 stays **confounded**: the v8 baseline's `builtin-calibration` resolution is **inferred from source (`profile=None`), not observed in a captured resolved log**, while the candidate launcher passes explicit `--spark-tp/--spark-ep` and resolves `legacy-heuristic`. The cost-policy finding that TP4 and TP6 already share a cost model holds only for the explicit candidate TP4 launcher; it does not remove the v8-baseline confound.

## Release status

The clean build ran rc0 after the Docker-wide local wipe, and the bounded **content-type functional checks** passed on all four production deployments. The images are **published** to `ghcr.io/tpurtell/ds41rt-{coordinator,spark-expert}` as `v9` and `latest` (same digest per role) and were **anonymous-pull verified** (fresh `DOCKER_CONFIG`, rc=0, matching digests). Coordinator index `sha256:786d1d67...` (linux/amd64 child `sha256:649e5c84...`); spark-expert manifest `sha256:f0c67407...`. Candidate performance is validated and labelled as candidate performance; the experimental canonical six-rank qualifier is out of scope and was not run. The GitHub release remains pending as a separate step.
