# TP/EP kernel timing milestone

Status: **complete for the 96-invocation sweep; lease released.** The width-192
production comparator and any AOT recommendation remain **open** and require a new
lease. Numbers come only from arms that passed every gate; the consolidation
script rejects any arm whose gates fail and records it.

**Representativeness limitation, stated up front.** This sweep uses the
benchmark's synthetic route table `((r + 1) * 6 + s) % 384`, which produces
**all-distinct experts with no reuse** at M8/M16.

Real traffic is different, and two distinct real-traffic figures must not be
conflated:

* **Observed live profile** (from the CPU representativeness handoff): TP4 at M8
  shows about **27 active experts over 48 routes**, and M64 about **48 active
  experts over 384 routes**, with routes concentrated in a high-count bin of
  **17 rows per expert bin** (the bin holds 17 rows, not 17 %), covering roughly
  **48 % of routes**.
* **Synthetic reuse fixture** (`scripts/fixtures/tp-ep-reuse-m64-e384.json`, for a
  future scoped check): its Tail bin holds **7 experts / 173 routes out of 384 =
  45.05 %**, `per_expert_counts [25,25,25,25,25,24,24]`, and is explicitly
  `synthetic_per_expert: true` with aggregate counts only.

So the fixture's 45.05 % is a **synthetic** stand-in near the observed ~48 %, not
a measurement of the same thing. This sweep has neither: it has no reuse at all.
So this sweep is an **all-distinct stress/comparison**, not a model of counting
traffic. It compares topologies under equal stress; it does not predict
production. A validated M64 reuse fixture exists at
`scripts/fixtures/tp-ep-reuse-m64-e384.json` for a future scoped check, and any
M64 comparison must pit candidate widths against 192 on that same fixture.

**What the timings are.** They are the **active weight footprint and its
resulting rank-local latency** — the bytes actually selected for the active
experts. They are **not** measured DRAM traffic, cache-hit rate, or TLB behaviour,
and no bandwidth conclusion is drawn from them.

## Scope and honesty limits

* **Four-Spark kernel selection** (TP4EP1 vs TP2EP2) and the **six-Spark choice**
  (TP3EP2 vs TP2EP3) are reported **separately**; they are different decisions and
  are not merged.
* Widths 64 and 128, and TP2EP3 versus TP3EP2, are **open measurements**. Nothing
  here selects a width or a topology in advance.
* Per-group maxima are **single-device sequential projections** of the critical
  group, **not** distributed concurrency: no transport, no inter-rank reducer and
  no group-sum cost is measured.
* Operands are **active-filled checkpoint banks** — real weights for the experts
  the route table names, unused slots zeroed. This is **not** a full real expert
  bank, and at M80 the route table names all 384 experts so the bank becomes a
  full bank (reason M80 stays diagnostic).
* Warm, GPU-resident (`--amortised`) and cold are reported **separately**. Cold is
  a 256 MiB flush then exactly one timed replay; the measured device L2 is
  25 165 824 bytes (24 MiB).
* One shared input and seed per cell; every arm in a cell receives the same active
  expert set, because the route table is deterministic and topology-independent.

## Provenance

| Item | Value |
| --- | --- |
| Host / container | MOA (172.22.2.6), `ds41rt-tpep-dev` CDI |
| Image | `sha256:dab2b979509231568ff375228fee9814d68cf5fdeb27edc77955cd35c1680234` |
| Device | NVIDIA GB10, SM 12.1, 48 SMs, L2 25 165 824 bytes |
| Snapshot | `dba1be0a40aa45a94ad051997016db3960a90277`, layer 20 |
| Index sha256 | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| Harness (active) | `bench_tp_ep_kernel.py` `452d6eb6003c27e9450d6e0659fa130da613cb38418e7b910edd1c63fafb124b` |
| Benchmark | `benchmark_v41_ep_groups.py` `2075cbf86fa462e0ec18508f2dec2a2c0f11fae5f8c4332532e5a1cbab5cdb04` |

### Harness revision ledger (authoritative JSON hash: `452d6eb6…`)

The harness went through three revisions during this work. Both changes were in
the **reporting path only**; the timed path is byte-identical throughout.

| Revision | Change | Failure it fixed |
| --- | --- | --- |
| `c741d34a…` | originally frozen at lease start | — |
| `aee81d6d…` | max-over-groups loop | iterated the summary keys alongside the group entries and raised `'str' object has no attribute 'get'` |
| **`452d6eb6…` (active)** | `_snapshot_arm_state` | restored by copying the **unpadded `rows * topk`** view into **capacity-sized** buffers, raising a size mismatch for every arm with rows < capacity (all M8/M16) |

Unchanged across all three: `warm_samples`, `cold_samples`, `amortised_samples`,
`_timed_condition`. The authoritative hash for any result in this document is the
active `452d6eb6…`; `c741d34a…` and `aee81d6d…` are historical.

## Matrix pass/fail

Full 96 timed invocations expected: 48 for the four-Spark pair (A/B) and 48 for
the six-Spark pair (E/F), each cell run as `A B B A` then `B A A B`.

| Group | Cells | Invocations | Passed | Failed |
| --- | ---: | ---: | ---: | ---: |
| four-Spark (A/B) | 6 | 48 | **48** | 0 |
| six-Spark (E/F) | 6 | 48 | **48** | 0 |
| **total** | 12 | **96** | **96** | **0** |

Every arm returned `RUN_RC=0`; all 96 passed every admissibility gate. Eight arms
from an aborted pre-fix run are **excluded** and none of their numbers is
reported. Provenance was identical across all 96 (one distinct provenance set).

Correctness gate inside the same window: **17 passed** (`PYTEST_RC=0`).

## Admissibility gates applied to every arm

**All seven are now enforced in `consolidate.py`** (gates 5 and 7 previously held
in the data but were not checked by the script; that gap is closed and each gate has
a negative unit test in `python/tests/test_tp_ep_timing_gates.py`, 20 tests). An arm
contributes a number only if all hold; otherwise it is listed as rejected with its
reason.

1. `preflight.rel_l2 < 0.01` and `preflight.cosine > 0.9999`
2. `graph_verified_not_noop` and `poisoned_only_in_preflight`
3. `post_timing_rel_l2 == preflight.rel_l2`
4. `compiled_callable_unchanged` (live row counts created no compile key)
5. `per_group.max_warm_median_us`, `max_amortised_median_us`,
   `max_cold_median_us` all present, not only group 0
6. `gpu_before`/`gpu_after` clocks recorded as observations only
7. `provenance.flush_bytes == 268435456`, `samples == 30`, and the **record-level**
   `records[0].capacity` matching the row bucket (1 for row 1, 80 for rows 8 and
   16). Note `provenance.capacity` is null in every arm and is therefore excluded
   from the provenance signature; the record field is authoritative.

## Results

96/96 accepted, 0 rejected. Full per-arm table and order effects in
`runs/tp-ep-kernel/timing/consolidation.md`; summary in `SUMMARY.txt`; raw arms
immutable in `runs/tp-ep-kernel/timing/arms/` with `ARCHIVE-MANIFEST.json`.

**Statistic and framing (explicit, per independent audit).** Each cell figure is
the **median of the four arm medians** for that role, not the median of the 120
underlying samples. Spread is the min-max of arm medians. **No confidence interval
is computed.** The ABBA/BAAB order is reported rather than averaged, and at w64 M1
the A-order pass-1 versus pass-2 difference (**+20.8 %**) is larger than the
topology difference there, so that cell is read as parity and the margin is not
quantified.

**Provenance attestation gap.** The arm JSONs record provenance fields but **no
harness, benchmark or snapshot-revision hash**, so the 96 results cannot be tied to
a harness hash from the data alone. The association lives in the run ledger
(`MOA-LEASE-RELEASED.json` and `ARCHIVE-MANIFEST.json`), which names the active
harness `452d6eb6…`. This gap is recorded, not retroactively filled, and the
archived `c741d34a…` file is labelled historical so it is not mistaken for the
sweep harness.

### Four-Spark kernel selection: TP4EP1 versus TP2EP2

| Cell | warm ratio | amortised ratio | cold ratio | reading |
| --- | ---: | ---: | ---: | --- |
| w64 M1 cap1 | 1.032 | 1.034 | 0.999 | parity within arm spread |
| w64 M8 cap80 | 1.006 | 1.005 | 1.003 | parity |
| w64 M16 cap80 | 1.001 | 1.000 | 0.998 | parity |
| w128 M1 cap1 | 1.122 | 1.124 | 1.043 | TP2EP2 faster warm; cold only mildly |
| w128 M8 cap80 | 1.100 | 1.103 | 1.101 | TP2EP2 ~10 % faster |
| w128 M16 cap80 | 1.117 | 1.119 | 1.115 | TP2EP2 ~11 % faster |

Ratio is TP4EP1 / TP2EP2, so **> 1 means TP2EP2 is faster**. Precisely:
- **TP2EP2 materially faster (> 1.04) in three cells** — w128 M1 (1.122), w128 M8
  (1.100), w128 M16 (1.117).
- **Near parity in the rest.** At w64 M8 and w64 M16 the ratios are 1.000-1.006,
  and the cold ratios there are 0.998-1.003, i.e. **TP4EP1 is marginally faster on
  some conditions**. A ratio of 0.998 or 0.999 is parity, not a TP4EP1 win.
- At w64 M1 the ratio is 1.032 warm but the four-arm spread is large (TP4EP1
  122.3-167.6 us), so that cell is parity, not a win either way.
This is **not** "TP2EP2 faster in every condition": it is faster in the w128 cells
and at parity elsewhere.

### Six-Spark choice: TP3EP2 versus TP2EP3

| Cell | warm ratio | amortised ratio | cold ratio |
| --- | ---: | ---: | ---: |
| w64 M1 cap1 | 1.135 | 1.137 | 0.997 |
| w64 M8 cap80 | 0.994 | 0.993 | 0.991 |
| w64 M16 cap80 | 1.003 | 1.003 | 1.005 |
| w128 M1 cap1 | 1.001 | 0.992 | 0.998 |
| w128 M8 cap80 | 1.007 | 1.006 | 1.003 |
| w128 M16 cap80 | 1.000 | 1.001 | 0.999 |

At M8/M16 the two are **at parity within 1 %** at both widths, so the six-Spark
choice is not settled by these numbers; the M1 w64 cell favours TP2EP3 by ~13 %.
Both six-Spark arms are far cheaper than either four-Spark arm at M8/M16, which is
expected because each rank owns fewer active experts (24 for TP3EP2, 16 for
TP2EP3) against 48 for the four-Spark arms.

### Width, descriptive only

At M1, w64 is faster than w128 for every topology (TP2EP2 155.9 vs 173.6 us; TP4EP1
160.8 vs 194.8 us). At capacity 80 the ordering is **topology-dependent**, so no
general width rule follows from this sweep:

| arm | M8 w64 | M8 w128 | M16 w64 | M16 w128 |
| --- | ---: | ---: | ---: | ---: |
| TP4EP1 | **1076.2** | 1117.8 | **2102.8** | 2241.8 |
| TP2EP2 | 1070.0 | **1016.3** | 2100.4 | **2007.7** |

TP4EP1 prefers w64 while TP2EP2 prefers w128 in the same cells. The production
default is **width 192 at capacity 80**, which this sweep does not measure, so no
production width is selected here.

### Order effect

The ABBA/BAAB schedule is reported rather than averaged: at w64 M1 the TP4EP1 arms
show a **+20.8 % pass-1 versus pass-2 warm difference**, far larger than any
topology difference in that cell. That is why the four-arm median, not a
two-arm comparison, is used, and why that cell is read as parity.

## Limits

* Single physical rank; parallel TP ranks are never summed. Per-group maxima are
  single-device sequential projections, not distributed concurrency.
* **Width 192 at M8/M16 — the production comparator — is NOT measured**, so no
  production-width recommendation and no AOT recommendation is made.
* **M64 is not measured.** A validated reuse fixture is ready at
  `scripts/fixtures/tp-ep-reuse-m64-e384.json`; a future scoped check must compare
  candidate widths against 192 on that same fixture.
* The sweep is **all-distinct with no expert reuse**, so it does not model
  counting traffic; read it as a stress comparison, not a production prediction.
* Timings are **active-weight-footprint latency**, not measured DRAM traffic,
  cache-hit rate or bandwidth.
* Operands are active-filled checkpoint banks — real weights for the active
  experts, unused slots zeroed — **not** a full real bank. The tensor allocation
  is fixed (6.724 GiB at full width 2304); the active useful footprint scales with
  the active count.
* No end-to-end serving effect, no production default, no native/kernel
  production change.
* LPT group loads are a **diagnostic projected cost model**, not the Rust
  scheduler; tie-breaking may differ from the Rust implementation.
* Two reporting-path harness bugs were found and fixed during the lease; the
  active harness hash is `452d6eb6…`, not the originally frozen `c741d34a…`.

## Release

MOA was released after the 96-invocation sweep completed: see
`runs/tp-ep-kernel/MOA-LEASE-RELEASED.json` for the run ID, UTC release time,
completion counts and process-check evidence. Width 192, M64 and any AOT
recommendation require a new lease.
