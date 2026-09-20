# TP/EP kernel qualification — readiness, provenance, and commands

Milestone owner: the Spark kernel/benchmark agent. This is the single readiness
and provenance report; results are recorded separately as they land.

## 1. CPU audit and provenance

### Frozen versus current benchmark source

| Revision | sha256 | Role |
| --- | --- | --- |
| pre-lease frozen | `7cc83b5afb2a33db97beb3ae853f01e6b5c7511e4a467dff0a06637a8263f098` | what the 17-test and first-smoke runs used |
| current | `2075cbf86fa462e0ec18508f2dec2a2c0f11fae5f8c4332532e5a1cbab5cdb04` | adds one loader-local change only |

The complete diff is three added comment lines plus making an existing default
explicit:

```diff
     weights, scales = {}, {}
     for name, shape in shapes.items():
-        weights[name] = torch_module.zeros(shape, dtype=torch_module.uint8)
+        weights[name] = torch_module.zeros(shape, dtype=torch_module.uint8,
+                                           device="cpu")
         scale_shape = (*shape[:-1], shape[-1] // 16)
-        scales[name] = torch_module.zeros(scale_shape, dtype=torch_module.uint8)
+        scales[name] = torch_module.zeros(scale_shape, dtype=torch_module.uint8,
+                                          device="cpu")
```

`torch.zeros` already defaults to CPU, so this is a no-op change in behaviour.
It affects only `checkpoint_selected_operands`, which is reachable only through
`--operands checkpoint`.

**Timing impact: none, and that is verified rather than argued.** All five
timing-path functions hash identically in both revisions:
`warm_samples`, `cold_samples`, `amortised_samples`, `_timed_condition`,
`measure_group`, and the checkpoint path itself is unchanged apart from the CPU
allocation described above. Note the harness was **not** byte-frozen at
`c741d34a…` for the whole lease: two reporting-path bugs were found and fixed
during execution (a max-over-groups loop that iterated summary keys, and a restore
helper that copied row-sized state into capacity-sized buffers). The active hash is
now `452d6eb6…`, listed below, with `c741d34a…` retained as historical. The timing
plan **does** use `--operands checkpoint` as its primary operand source.

### Frozen artifacts

| File | sha256 |
| --- | --- |
| `bench_tp_ep_kernel.py` (active) | `452d6eb6003c27e9450d6e0659fa130da613cb38418e7b910edd1c63fafb124b` |
| `bench_tp_ep_kernel.py` (historical, not active) | `c741d34a5e55af6eda1a79da728b5c71f33156f6da2ccd0584c849c1b16455e1` |
| `benchmark_v41_ep_groups.py` | `2075cbf86fa462e0ec18508f2dec2a2c0f11fae5f8c4332532e5a1cbab5cdb04` |
| `test_v41_ep_algebra.py` | `12033b089670abefd3db4a97bef6b776455313b9c507b00008abd913fff88643` |
| `test_v41_sentinel_masking.py` | `d7538df4b0085a56d631b751d0ac82013558aecfc4be5fe69dc8f197ffa9bb28` |
| `test_tp_ep_cost_model.py` | `935d5cf1cf4f3d9597f66b6f46e60349a0e2843d8c1ee8827b86e6178a839c18` |
| `test_tp_ep_real_operands.py` | `3919be3203b26662e82ecc844ccd4c968401ae8f56dd0944c46d1a24f94e1078` |
| `test_tp_ep_real_smoke.py` | `98634c712787e164928769bf7050217d1dc24e8850f4e1d516e84ca2994bd47d` |
| grid-run harness (frozen) | `0eb0fac467a86652de2110d777dceba3c4956932a8259e8745e5b8da99b3a055` |
| E384-failed harness (frozen) | `42307152eec2768a3eafd1307e9ed8f3312db2d529b13acf2c1b65958e0ecc4b` |

### CPU tests

20 pass (`test_tp_ep_cost_model.py` 9, `test_tp_ep_real_operands.py` 11), no GPU
required. They pin the cost-model arithmetic against `ds41rt-core`, the
active-filled operand semantics, and a duplicate-definition guard over every
harness file.

## 2. Correctness results to date, honestly stated

| Run | What | Outcome |
| --- | --- | --- |
| 17 tests, MOA | `test_v41_ep_algebra.py` + `test_v41_sentinel_masking.py` | **17 passed** — first GPU validation of the cleaned harness |
| real smoke, runs 1–6 | layer 20, TP2 r0/r1, TP4 r0 | **final two runs pass** (4 passed, 26.16 s and 63.78 s) |

Final passing numbers, layer 20, width 192, active ids `[1,4,7,11,19,23]`:

| Target | unmasked rel-L2 | cosine | masked rel-L2 | cosine | owned ids |
| --- | ---: | ---: | ---: | ---: | --- |
| TP2 rank0 | 0.0016072 | 0.99999869 | 0.0017096 | 0.99999851 | `[4]` |
| TP2 rank1 | 0.0016287 | 0.99999863 | 0.0016179 | 0.99999881 | `[1,7,11,19,23]` |
| TP4 rank0 | 0.0016622 | 0.99999857 | 0.0016709 | 0.99999851 | `[4]` |

All arms finite; all-unowned gate exact zero with oracle norm exactly 0.0. The
mask is a **fixture unit mask** (`expert % tp_degree`), not the production group
mapping and not a scheduler result.

### Failure history, not conflated

* **Runs 2–5** failed on **test-harness device-consistency bugs**: the route table
  is built on CPU while weights and outputs are CUDA, so tensors derived from it
  must be moved before combining. Four separate fixes. These are not kernel
  defects.
* **Run 1 produced `nan` and remains UNROOT-CAUSED.** I did not retain its masked
  diagnostics, and I could not reproduce the failure on CPU: with the run-1
  condition reproduced (unloaded expert slots zero, masked weights zero on
  unowned routes) the oracle is finite with a positive norm. The later failures
  are a **different class** and must not be described as explaining run 1. The
  honest statement is: the smoke was rebuilt to route only to loaded experts, and
  the rebuilt smoke passes.
* Run 1 also had a misleading `exit 0` because a pipe masked pytest's status.
  Every later run captures `$?` directly.

## 3. Timing plan

### Arms

| Arm | Topology | EP degree | Experts/rank | Groups | Widths | Rows | Capacity |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| A | TP4 | 1 | 6 | 1 | 64, 128 | 1 | 1 |
| B | TP2 | 2 | 3 | 2 | 64, 128 | 1 | 1 |
| C | TP4 | 1 | 6 | 1 | 64, 128 | 8, 16 | 80 |
| D | TP2 | 2 | 3 | 2 | 64, 128 | 8, 16 | 80 |
| E | TP3 | 2 | 3 | 2 | 64, 128 | 1, 8, 16 | 1 / 80 |
| **F** | **TP2** | **3** | **2** | **3** | **64, 128** | **1, 8, 16** | **1 / 80** |

Arms E and F are the user's primary six-Spark hypothesis: **TP3EP2 (3 experts
per rank, 2 groups, 3-Wide shards) versus TP2EP3 (2 experts per rank, 3 groups,
2-Wide shards)**. Both span six Sparks at the same full width, so the matched
comparison isolates the shard-width versus group-count tradeoff. An earlier draft
of this plan omitted TP2EP3; that omission is corrected here.

Capacity follows the authoritative dispatch: **rows 1 → capacity 1; rows 2..80 →
capacity 80**; there is no capacity-16 runtime path, so capacity 16 is not used.

### Operands: official checkpoint is primary

Timing runs with **`--operands checkpoint`** — active-filled real operands, which
the plan required earlier and which must not be quietly dropped. A synthetic run
is an explicit auxiliary cross-check only, and is labelled as such wherever it
appears.

* **Active-filled, not a full real bank.** Only the experts the live route table
  actually names are read; unused dense slots are zeros. The harness records
  `active_count`, `unused_slots_zeroed` and the label on every arm, and every
  record says so.
* **Same expert inputs and seed on every arm.** The route table is deterministic
  (`(row*6 + slot) % 384`), so the active set and the bank are identical across
  topologies for a given row count — which is what makes a cross-topology
  comparison legitimate. No arm gets a different expert population.
* **Hash and device load happen outside the timed region.** The operand digest is
  computed on CPU before any transfer, and the bank is loaded before warmup; the
  timed callable is unchanged from the frozen harness.

### Allocation versus active footprint (corrected)

Three different quantities were previously conflated. They are now separated, and
the numbers come from the actual records rather than from prose.

1. **Tensor allocation — FIXED, does not scale with active experts.** The loader
   allocates a zero-filled dense bank at the requested FULL width. For the
   benchmark geometry (`experts=384`, `full_width = intermediate * degree = 2304`)
   that is **7 219 445 760 bytes = 6.724 GiB**, and it is the *same* for every
   topology because the full width is the same. Confirmed by the smoke records,
   which report `full_dense_bank_bytes = 7 219 445 760` for TP2, TP3 and TP4
   alike.
2. **Selected checkpoint read bytes — scale with active experts.** Only the
   experts the live route table names are read from disk and hashed. This is the
   `hashed_active_bytes` field.
3. **Active useful weight footprint — scales with active experts**, including the
   packer's padding. This is what the earlier table reported, and labelling it
   "dense bank" was wrong.

| Active experts | Selected/read and useful footprint | Share of the fixed 6.724 GiB allocation |
| ---: | ---: | ---: |
| 6 (M1) | 112 803 840 B = 107.6 MiB | 1.6 % |
| 12 (M2) | 225 607 680 B = 215.2 MiB | 3.1 % |
| 48 (M8) | 902 430 720 B = 860.6 MiB | 12.5 % |
| 96 (M16) | 1 804 861 440 B = 1 721.2 MiB | 25.0 % |
| 384 (M80) | 7 219 445 760 B = 6 885.0 MiB | 100 % |

Per-expert packed bytes at this geometry are 18 800 640 B, identical across
degrees because the full width is identical.

### The two figures are different objects, verified against both sources

I checked this rather than asserting it. Both numbers are correct, for different
tensor layouts:

| Quantity | Per expert | All 384 experts | Source of truth |
| --- | ---: | ---: | --- |
| Native packed 4-bit, **Spark TP4 production rank** (576 padded to 640) | 5 222 400 B | **1.868 GiB** | shipped manifest + `ds41rt_v41_expert_packed_sizes` |
| Native packed 4-bit at full width 2304 | 18 800 640 B | **6.724 GiB** | same packer formula with `padded = 2304` |
| Benchmark loader bank, int8/logical at full width 2304 | 18 800 640 B | **6.724 GiB** | loader `full_dense_bank_bytes` in the records |

The native packer formula gives `w13 = padded*H`, `s13 = padded*H/16`,
`w2 = H*padded/2`, `s2 = H*padded/32`, totalling exactly 18 800 640 B per expert
when `padded = 2304` — **identical to the benchmark loader's per-expert total**.
So the loader figures are consistent with the native format at full width, and the
apparent 6.724 = 3.60 x 1.868 is a **layout difference** (full width 2304 versus
the 576-to-640 production slice), not a contradiction and not a measurement claim.

Production per-rank sizes for reference, all from the same packer formula:
TP4 (576->640) 1.868 GiB, TP3 (768) 2.241 GiB, TP2 (1152) 3.362 GiB, full (2304)
6.724 GiB.

### Method

* Balanced A/B/B/A: **separate invocations**, alternating order per width, because
  `--ep-degree` is one global scalar per process and `--topologies` selects from a
  fixed list rather than preserving argument order.
* LPT only, explicit: `--lpt --weight-cost 1.0 --tile-cost 0 --tile-rows 16`.
  Modulo masks are never a performance basis.
* `--samples 30 --replays 10`, warm and GPU-resident (`--amortised`) and cold.
* Cold: `--flush-bytes 268435456` (256 MiB), then exactly one timed replay per
  sample. **Measured L2 on the GB10 is 25 165 824 bytes (24 MiB)**, so a 256 MiB
  flush exceeds it by ~10x. The harness reads `L2_cache_size` at runtime and
  refuses a flush that does not exceed it.
* No timing until the correctness rerun passes in the same window.

### Commands

Exit status is captured truthfully: `rc=$?` immediately after the command, then
`echo` and `exit $rc`. **Do not** write `cmd | tee ...; echo EXIT=$?` — that
records the exit of the pipeline's last stage and masked a real pytest failure
earlier in this work. The runs write durable logs under the container's
`/scratch` (host `/home/tj/ds41rt-tpep`), which is verified host-visible.

```sh
# 0. correctness FIRST, in the same lease window. No timing until this passes.
docker exec -e PYTHONDONTWRITEBYTECODE=1 \
  -e DS41RT_SPARKINFER_SOURCE_DIR=/workspace/ds41rt/third_party/sparkinfer \
  -e PYTHONPATH=/workspace/ds41rt/third_party/sparkinfer:/workspace/ds41rt/python/tools:/workspace/ds41rt/python/reference:/workspace/ds41rt/python \
  <container> bash -lc '
    set -o pipefail
    cd /workspace/ds41rt
    python -m pytest python/tests/test_v41_ep_algebra.py \
      python/tests/test_v41_sentinel_masking.py -q 2>&1 \
      | tee /scratch/ep-kernel/timing-17tests.log
    rc=$?
    echo "PYTEST_RC=$rc"
    exit $rc'

# 1. one timed invocation, repeated per the A/B/B/A schedule below.
#    RUN is the repo runner variable for the full docker exec prefix.
run_arm() {   # $1=tp  $2=ep  $3=width  $4=rows  $5=capacity  $6=label
  docker exec -e PYTHONDONTWRITEBYTECODE=1 \
    -e DS41RT_SPARKINFER_SOURCE_DIR=/workspace/ds41rt/third_party/sparkinfer \
    -e PYTHONPATH=/workspace/ds41rt/third_party/sparkinfer:/workspace/ds41rt/python/tools:/workspace/ds41rt/python/reference:/workspace/ds41rt/python \
    <container> bash -lc "
      set -o pipefail
      cd /workspace/ds41rt
      python python/tools/bench_tp_ep_kernel.py \
        --operands checkpoint \
        --snapshot /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
        --expect-revision dba1be0a40aa45a94ad051997016db3960a90277 --layer 20 \
        --topologies $1 --ep-degree $2 --widths $3 --rows $4 --capacity $5 \
        --experts 384 \
        --lpt --weight-cost 1.0 --tile-cost 0 --tile-rows 16 \
        --samples 30 --replays 10 --warmup 5 --cold --flush-bytes 268435456 \
        --amortised --max-group \
        --output /scratch/ep-kernel/$6.json 2>&1 \
        | tee /scratch/ep-kernel/$6.log
      rc=\$?
      echo "RUN_RC=\$rc"
      exit \$rc"
}
```

### Execution order

Balanced **A/B/B/A**, four invocations per (width, rows) pair, then the reversed
pair for the mirrored arm — **eight timed invocations per width/rows cell**, not
four. `--ep-degree` is one global scalar per process, and `--topologies` selects
from a fixed list rather than preserving argument order, so the schedule is driven
by the caller:

```text
cell (width, rows):   A  B  B  A   then   B  A  A  B
```

For each width in {64, 128} and each rows in {1, 8, 16}:

```text
run_arm tp4 1  <w> <rows> <cap>  A_w<w>_m<rows>_1
run_arm tp2 2  <w> <rows> <cap>  B_w<w>_m<rows>_1
run_arm tp2 2  <w> <rows> <cap>  B_w<w>_m<rows>_2
run_arm tp4 1  <w> <rows> <cap>  A_w<w>_m<rows>_2
run_arm tp2 2  <w> <rows> <cap>  B_w<w>_m<rows>_3
run_arm tp4 1  <w> <rows> <cap>  A_w<w>_m<rows>_3
run_arm tp4 1  <w> <rows> <cap>  A_w<w>_m<rows>_4
run_arm tp2 2  <w> <rows> <cap>  B_w<w>_m<rows>_4
```

Six-Spark pair (arms E and F), same 8-invocation shape with `tp3 2` against
`tp2 3`. Capacity is 1 for rows 1 and 80 for rows 8 and 16.

### Gates before any number is quoted### Gates before any number is quoted

1. `preflight.rel_l2 < 0.01` and `preflight.cosine > 0.9999`.
2. `graph_verified_not_noop == true`, `poisoned_only_in_preflight == true`.
3. `post_timing_rel_l2 == preflight.rel_l2`.
4. `compiled_callable_unchanged == true`.
5. `per_group.max_warm_median_us`, `max_amortised_median_us`,
   `max_cold_median_us` all present, not just group 0.
6. `gpu_before`/`gpu_after` clocks recorded as observations only.
7. `provenance.flush_bytes == 268435456`, `samples == 30`, `capacity` matches the
   row bucket, and the harness sha256 recorded with the results.

### Scope discipline

Per-group maxima are **sequential projections** of the critical group on one
shard, not a concurrent distributed path. No transport, inter-rank reduction,
group-sum cost, end-to-end effect, or kernel-default change is claimed. Width 64
versus 128 is a measurement to be made, not a pre-judged selection.

## 4. Outstanding

* The GPU lease was granted; the 17-test gate passed in-window and the timing
  sweep ran. Width 192 at M8/M16 (the production comparator) remains unmeasured,
  so **no production-width recommendation is made**.
* Run 1's `nan`: unroot-caused, and explicitly not attributed to the kernel.
* TP3 arms for six-Spark support are planned but unmeasured.
