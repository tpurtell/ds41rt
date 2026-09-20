# Remaining kernel comparison — reviewed pre-execution plan

**Status: the 96-arm measurement and its independent audit are COMPLETE and a
retention decision is recorded** — keep the tested cap80/w192 and cap1/w64; no
extra AOT build; no default-serving change. Everything below is **historical
(pre-execution)**. Frozen audit `08a7d856…` (313 lines, no blocking items) after
review `c22a2673…` (earlier `c4154a1d…`); validator **`d9632e81…`** (96 ACCEPT,
errors [], 35 selftests, 97 hashes); results `results96-independent-analysis.json`
/ `results96-independent-tables.txt`, raw `results96/`. **Ledger closed:**
canonical `bce43420…` (executor `589f69b5…` superseded). **V2 caller repair
VERIFIED** (`1dc81573…`; current tool `ca8c9054…`, benchmark `ecf52e33…`;
focused 80 pass / 5 CUDA skips / 0 failed with CUDA hidden). **Fleet restored and
verified** against the original five
(`runs/tp-ep-preflight/FINAL-RESTORATION-REPORT.md`). **Scoped implementation,
measurement, selection and restoration are complete**; the frozen sources
`8255d1df…` and archived benchmark `2075cbf8…` stay separate from the current
tool. Optional future hypotheses (non-default widths, AOT, distributed/DRAM
measurement) are **not completion gates**. See `docs/tp-ep-results-summary.md` for
consolidated status. No default or AOT choice follows from this plan alone.

Reviewed and corrected against `runs/tp-ep-kernel/remaining-plan-review.md`. The
earlier draft's Part B count was double the accepted control (96 vs 48), its
per-group numbers would have been measured on the wrong workload, one command
could not be paired by the consolidator, and several labels were inaccurate. Those
are fixed below and in code.

## 0. What changed since the reviewed draft

| Review item | Fix |
| --- | --- |
| **B1** fixture never reached the per-group block | `_per_group_latency` now takes the arm's `live_ids`/`live_weights` and no longer regenerates a synthetic table. Regression-tested at source level. |
| **B2** multi-width command un-pairable | Harness **fails closed** on more than one width or more than one record; one width per invocation, one file per `(arm, width, pass, n)`. Consolidator `load_arm` now `raise`s instead of `assert`ing, so `-O` cannot silently reduce to `records[0]`. |
| **B3** Part B count | **48 invocations**, not 96. The "2 passes" is already inside the 4 files per (role, width, rows). |
| **C1** cross-harness/cross-session width delta | Added a **same-session control width** paired against 192 (see §1). |
| **C2** false "only variable against the sweep" | Claim removed. Width is the only controlled variable **within** Part B. |
| **C3** consolidator could mix provenance | `SIGNATURE_FIELDS` now includes `harness_sha256`, `benchmark_sha256`, `route_fixture_sha256`. |
| **C4** no width-order control in Part B | Widths **counterbalanced** in the explicit four-round schedule below; no extra sentinel invocation. |
| **H2/H3** dtype prose | Machine-readable `input_dtype` (source and wire forms), `output_dtype`, and per-record `oracle_dtype`. |
| **H4** field-level mislabel | Every provenance field is **provenance-level** (one per file); only `oracle_dtype` is per record. Stated in §4. |
| **H5** unreproducible test claim | Exact invocation and exact results in §5. |
| **H6** fixture is synthetic | Caveat carried in §1. |
| **H7** tautological dtype assert | Replaced with `assert tensor.dtype == torch.int32`. |

## 1. Scope and fixture honesty

The M64 fixture `scripts/fixtures/tp-ep-reuse-m64-e384.json` is
**distribution-representative, not expert-ID faithful**: its per-expert 25/24
split is **synthetic**, not a recovered multiset, and only the aggregate count and
route sum (7 experts, 173 of 384 routes = 45.05 %) are known. It is a
reuse-bearing stand-in, not a recorded production route table.

### Part A — width-192 production comparator and its same-session control

The production default is cap1/w64 and cap80/w192. 192 at cap80 is unmeasured.

| Cell | Roles | Invocations |
| --- | --- | ---: |
| w192 M8 cap80 | four-Spark pair × ABBA+BAAB | 8 |
| w192 M16 cap80 | four-Spark pair × ABBA+BAAB | 8 |
| w192 M8 cap80 | six-Spark pair × ABBA+BAAB | 8 |
| w192 M16 cap80 | six-Spark pair × ABBA+BAAB | 8 |
| **w64 control, four-Spark M8/M16** | four files per role per row count, paired with w192 | **16** |
| **Part A total** | | **48** |

The w64 control is **required by C1**: the archived w64/w128 arms ran on a
different harness revision (`452d6eb6`, **not retained on disk**) in a different
session, so a w192-versus-archive delta is not attributable to width. Measuring w64
and w192 in the same session, on the same harness, is what makes a width statement
possible at all.

### Part B — M64 reuse comparison, 48 invocations

4 roles × 1 row (M64) × 3 widths × 4 files = **48**. Capacity 80.

For EACH pair (A/B, then E/F), execute these rounds in order. At each listed
width execute the two listed roles in order, one invocation per role/width:

```text
round 1: widths 64,128,192; roles A,B  (or E,F)
round 2: widths 192,128,64; roles B,A  (or F,E)
round 3: widths 192,128,64; roles B,A  (or F,E)
round 4: widths 64,128,192; roles A,B  (or E,F)
```

This is 4 rounds × 3 widths × 2 roles × 2 pairs = **48**. Each width's
role order is ABBA then BAAB (or EFFE then FEEF), four samples per role.
There is no additional sentinel; report per-round variation from these samples.
This counterbalances order but cannot eliminate all temporal drift.

Total: **96 invocations** (Part A 48 + Part B 48).

## 2. Scheduling

One width per invocation (B2). Balanced ABBA then BAAB per (role, width, rows)
cell, which is 4 invocations per arm per cell:

```text
A(tp4 ep1) B(tp2 ep2) B A        then        B A A B
E(tp3 ep2) F(tp2 ep3) F E        then        F E E F
```

Part A concrete order: for each four-Spark row count (M8, then M16), use the
four-round Part B schedule with widths `[64,192]` in rounds 1/4 and `[192,64]`
in rounds 2/3, and roles A,B / B,A / B,A / A,B respectively. This is 16
invocations per row count (32 total). For each six-Spark row count, run only
w192 in role order E,F,F,E,F,E,E,F (8 each, 16 total). Part A therefore has
48 invocations. Cross-width comparisons at M8/M16 are supported only for the
four-Spark roles; the six-Spark w192 measurements compare topologies only.
Counterbalancing reduces ordering bias; it does not prove absence of clock drift.

## 3. Method (unchanged, and verified unchanged)

* `--lpt --weight-cost 1.0 --tile-cost 0 --tile-rows 16` — LPT only, explicit.
* `--samples 30 --replays 10 --warmup 5 --cold --flush-bytes 268435456`,
  `--amortised --max-group`, `--operands checkpoint`, layer 20, revision pinned.
* Warm / GPU-resident / cold reported separately. Cold is one timed replay after a
  256 MiB flush against the **measured** 24 MiB L2.
* **Post-timing oracle enforced per arm**: `post_timing_rel_l2 ==
  preflight.rel_l2`, plus a finite check, plus `compiled_callable_unchanged`.

### Timed functions unchanged — with the limit stated

The four timed functions (`warm_samples`, `cold_samples`, `amortised_samples`,
`_timed_condition`) are byte-identical between the active harness and the archived
historical `c741d34a…` source, and this is asserted by a test.

**Limit:** only `c741d34a` is retained on disk. The `452d6eb6` revision that
produced the archived 96 arms was **not archived**, so its timed functions cannot
be verified from source. Equivalence for the archived 96 is **not claimed**.

## 4. Provenance

Every future arm record carries, at **provenance level** (one per file):
`harness_path`, `harness_sha256`, `benchmark_sha256`, `route_fixture`,
`route_fixture_sha256`, `input_dtype`, `output_dtype`. Only `oracle_dtype` is
**per record**.

Dtypes are machine-readable, not prose:

* `input_dtype.source_activation` = `bfloat16` (pre-quantization)
* `input_dtype.kernel_wire_values` = `uint32` (FP8 E4M3 payload: 5120 bytes per row)
* `input_dtype.kernel_wire_scales` = `uint8` (UE8M0 K/32: 160 bytes per row;
  combined activation payload and scales are 5280 bytes)
* Weight wire documentation (not additional JSON fields): FP4 values packed in
  `uint32`, with E8M0 scales; this is distinct from the FP8 activation wire.
* `output_dtype.route_planes` = `float32`; `output_dtype.reduced_output` =
  `bfloat16`
* `oracle_dtype.oracle` = `float32`, `compared_actual` = `float32`

The archived 96 arms carry none of the revision fields. That historical
attestation gap remains recorded and is **not** retro-filled. Because the
signature now includes the revision hashes, old and new arms cannot be mixed into
one reported provenance set.

## 5. CPU validation — exact invocation and exact results

```sh
PYTHONDONTWRITEBYTECODE=1 \
DS41RT_SPARKINFER_SOURCE_DIR=$PWD/third_party/sparkinfer \
PYTHONPATH=$PWD/third_party/sparkinfer:$PWD/python/tools:$PWD/python/reference:$PWD/python \
.venv/bin/python -B -m pytest python/tests -q
```

Final independent review rerun: **573 passed, 3 failed, 2 skipped, 27 subtests passed** (the additional nine tests cover the group-route path).

The 3 failures are `test_tp_ep_real_smoke.py` (targets 2:0, 2:1, 4:0) with
`PermissionError` reading `/root/.cache/huggingface/.../model.safetensors.index.json`.
That is an **environment permission issue on this checkout, not a code defect**: the
module `skip`s when the snapshot is absent, but hits a hard permission error when
the path exists and is unreadable. It needs either a readable snapshot or a `skip`
on `PermissionError`; I have **not** changed it, since this plan does not use that
module.

Focused modules relevant to this plan: **38 passing**
(`test_tp_ep_route_fixture.py` 9, `test_tp_ep_group_route_path.py` 9,
`test_tp_ep_timing_gates.py` 20).

Note: plain `pytest python/tests/test_tp_ep_route_fixture.py` fails collection
without the `PYTHONPATH` above (`_pinned_sparkinfer` import).

## 6. Cost

Extrapolated from the measured 96-arm run (M1 ~9 s/arm; M8 ~30 s; M16 ~60 s at
48/96 active experts), not from speculation:

| Part | Invocations | Expected |
| --- | ---: | --- |
| A (w192 + four-replicate w64 control) | 48 | ~40-65 min |
| B (M64 fixture, counterbalanced) | 48 | ~50-90 min |
| **total** | **96** | **~1.5-2.6 h, provisional** |

The M64 per-arm cost is genuinely unknown — 48 active experts like w64 M8, but
reuse may change cache behaviour and no M64 arm has ever run. **The first few Part A
arms will re-derive the per-arm figure before Part B is allowed to run long.**

## 7. What this settles and what it does not

Settles: whether w192 at cap80 changes the four-Spark ranking **when measured in
the same session as its w64 control**, and how widths 64/128/192 compare under
reuse-bearing routes at the M64 shape.

Does not settle: any end-to-end serving effect; transport or inter-rank reduction
cost; **DRAM traffic, cache-hit rate or bandwidth** (timings are active-weight
footprint latency); M80 (diagnostic only, its route table names all 384 experts);
and no AOT recommendation on its own.

## 8. Gates (enforced in code, 20 negative tests)

Seven gates in `runs/tp-ep-kernel/timing/consolidate.py`: numerics, graph/preflight,
post-timing drift, compile-key, per-group maxima present and equal to the worst
group median, sequential-projection flag, and record-level capacity matching the
row bucket plus exact flush and sample counts. An arm failing any gate is rejected
and listed, never reported.

## 9. Ready commands (not run)

```sh
# One width per invocation. Part A: --widths 192 (and the w64 control).
# Part B adds: --route-fixture .../tp-ep-reuse-m64-e384.json --rows 64 --capacity 80
docker exec -e PYTHONDONTWRITEBYTECODE=1 \
  -e DS41RT_SPARKINFER_SOURCE_DIR=/workspace/ds41rt/third_party/sparkinfer \
  -e PYTHONPATH=/workspace/ds41rt/third_party/sparkinfer:/workspace/ds41rt/python/tools:/workspace/ds41rt/python/reference:/workspace/ds41rt/python \
  <container> bash -lc '
    set -o pipefail
    cd /workspace/ds41rt
    python python/tools/bench_tp_ep_kernel.py \
      --operands checkpoint \
      --snapshot /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
      --expect-revision dba1be0a40aa45a94ad051997016db3960a90277 --layer 20 \
      --topologies tp4 --ep-degree 1 --widths 192 --rows 8 --capacity 80 \
      --experts 384 --lpt --weight-cost 1.0 --tile-cost 0 --tile-rows 16 \
      --samples 30 --replays 10 --warmup 5 --cold --flush-bytes 268435456 \
      --amortised --max-group --output /scratch/ep-kernel/A_w192_m8_1_1.json \
      > /scratch/ep-kernel/A_w192_m8_1_1.log 2>&1
    rc=$?; echo "RUN_RC=$rc"; exit $rc'
```

## 10. Execution status (supersedes "not executed")

The 96-arm measurement was executed under the authorized lease and independently
audited (96/96 ACCEPT); the ledger is closed (`bce43420…`, validator
`d9632e81…`), the V2 caller repair is verified (`1dc81573…`, audit `08a7d856…`),
and the fleet is restored and verified. No MOA reacquisition is needed; the
decision recorded at the top (keep cap80/w192, cap1/w64; no extra AOT build; no
default change) is based on the actual results. **Scoped implementation,
measurement, selection and restoration are complete;** only optional future
hypotheses remain, and they are not completion gates.
