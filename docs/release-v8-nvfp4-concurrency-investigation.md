# Published v8 dual NVFP4 concurrency investigation

## Reproduction (2026-09-19)

Repository dev at c583e04; published coordinator engine 5a56f0eb9d8caea050f84f96332c66786e177607. No kernels changed. Existing untracked python/tools/gate_nvfp4_share_vs_perroute.py was left alone.

The already-running ghcr.io/tpurtell/ds41rt-coordinator:v8 was verified with docker inspect. It used two RTX PRO 6000 GPUs, four Spark peers, prefill capacity 2048, concurrency 16, dSpark, automatic host cache and default KV. Startup logged 20 TP2 expert layers. Code benchmark command:

```
./.venv/bin/python scripts/bench-ds41-concurrent-api.py --base-url http://127.0.0.1:8000 --output /tmp/astra-v8-concurrency-code.json --concurrency 1 2 4 8 16 --repeats 3 --label astra-v8-repro --case code
```

C1 repeats passed at 144.94, 147.18, 145.62 tok/s; C2 repeat 1 failed. Coordinator at 11:32:26.151333Z reported `native decode round failed: ds41rt_cuda_engram_dequant_bf16_async returned status 5: out of memory`. Postfailure nvidia-smi free memory was GPU0 2425 MiB / GPU1 4 MiB. Evidence: /tmp/astra-v8-concurrency-code.json, /tmp/astra-v8-concurrency-code.log, /tmp/astra-v8-coordinator.log.

Controlled restart used the same published image and original command plus `--kv-pool-size 2GiB`, keeping all four existing experts and the same 20-layer placement. This is a diagnostic intervention, NOT a fix or valid default-release result. C1/C2/C4/C8 all three repeats passed. C16 repeats 1 and 2 passed (553.82 and 607.38 aggregate tok/s); repeat 3 failed. At 11:38:58.620590Z coordinator reported `V4.1 compact reduction failed with CUDA status 2` (CUDA memory allocation error). Postfailure free memory: GPU0 2811 MiB / GPU1 4 MiB. Evidence: /tmp/astra-v8-kv2-code.json, /tmp/astra-v8-kv2-code.log, /tmp/astra-v8-kv2-coordinator.log. Additional memory delays failure but does not yet solve it.

Existing records were inspected, not overwritten: package code sweep currently fails C8 repeat 3, package topic fails warmup with incomplete SSE, package mixed has incomplete SSE rows, and /tmp/v7ab-concurrency-code.json fails C8 repeat 3. /tmp/v8-final-eval.log shows one partial run (25 passes/63 failures), then 88 failures and a third-run process error. These files reflect their current contents, which differ from some earlier campaign descriptions.

## Mechanism established and remaining attribution

The reproduced failure is CUDA memory exhaustion on the tighter second GPU, not evidence of a fixed stream ceiling or an RDMA timeout. First SSE chunk does not establish generation success. The Engram error site (native/cuda/kernels/engram.cu:99) merely launches a kernel and calls cudaGetLastError; it performs no cudaMalloc. It can report lazy module/launch resource exhaustion or prior CUDA error, and does not identify the allocation that consumed the reserve.

Startup has a definite accounting hole: rust/crates/ds41rt-daemon/src/v41_native_serve/distributed.rs:366 computes final KV pool before scheduler::prepare_prefix_cache at :389. The latter allocates hostcache streams and eager pinned slabs after KV. The claim at :300 that every other persistent owner is live is therefore not true for hostcache. memory/distributed.rs:7 reserves 800 MiB for runtime, and :123 subtracts fixed occupancy and that reserve before sizing KV.

Default startup measured cudaMemGetInfo occupied bytes after KV [98666020864,101136465920]. Hostcache then allocated 16374562816 bytes (61 x 256 MiB). Idle nvidia-smi showed 94790 / 97202 MiB. The apparent additional 694.75 / 750.75 MiB mixes CUDA and NVML metrics: it must NOT be presented as a measured host-mapping charge. Both same-metric checkpoints around hostcache and request-time graph/lazy module allocations remain needed. The smaller-KV run exhausted a further roughly GiB during the broader concurrency sweep; merely accounting one startup delta is not a validated fix.

Single most likely fault to address: insufficiently budgeted persistent plus request-time CUDA overhead, beginning at distributed.rs:366-389 and memory/distributed.rs:7, with graph residency/growth measured before choosing a reserve. Candidate structural fix: initialize hostcache before the final per-GPU memory snapshot and preserve runtime headroom after all persistent owners, then budget or bound graph/launch demand at configured concurrency. Auto hostcache depends on KV capacity (scheduler.rs:142-148), so this needs decoupled conservative planning or bounded coupled planning, not a blind reorder. Exact KV requests must remain strict. No speculative constant increase or weakened check has been committed.

## 90/30 allocation claim

Steady state is not exactly symmetric, but a persistent 90/30 GiB split is inconsistent with this successful 20-layer TP2 load:

- distributed.rs:341-344 loads the same expert_layers into both ranks; measured rank_peak_bytes are [76451266644,76451266644]. RankWeights::load (v41_experts/tp2.rs:163-191) loops all layers on its selected device.
- Crucially the NVFP4 pair is loaded SEQUENTIALLY: complete GPU0 rank before GPU1 rank. Pre-expert fixed occupancy is [17982291968,21000093696]. During GPU0 completion and early GPU1 loading, approximately 90/30 GiB is entirely plausible. This is a concrete startup path producing a large transient split, not proof of what the owner observed. A failed/interrupted second-rank load could also leave a transient observation, but is not a ready successful TP2 server.
- KV placement is source-sharded, not equal-byte: default measured cache bytes [2385404800,1597086336]. PoolPlan checks each card independently and uses the tighter card; no transfer of spare capacity between cards (memory/distributed.rs:131-167).
- dSpark default weights live on GPU1 (distributed.rs:268-277), approximately 7.98 GB in measured startup delta. Two lane-local draft workspaces charge per lane GPU0 19247552 / GPU1 418304888 bytes. This explains modest GPU1 heaviness, not GPU0 90/GPU1 30 after ready.
- Embedding, vision and some producer ownership favor GPU0; target lanes have deliberate nonuniform decoder capacities (distributed.rs:259-266).
- TP2 transport workspace is symmetric per rank; extra Spark transport exists on GPU1 (distributed.rs:311), measured totals [1794720344,2046378584].
- Hostcache uses CPU pinned memory, not a 16-18 GiB device slab: prefix/host_cache.rs:237 calls alloc_host_buffer, native/src/ds41rt_native.cc:1024 calls cudaHostAlloc(Portable|Mapped). CUDA mappings/bookkeeping can consume device overhead. Both CUDA contexts already exist; Device::run restores prior device (v41_memory/device.rs:42-60), so this is not accidental creation of a second context. Hostcache stream construction uses ambient GPU0 in this worker.

## Status

No code fix claimed; default and smaller-KV concurrency runs are both failures. No containers published, tags moved, or branches pushed. The diagnostic deployment was stopped with runs/v8/serve.sh nvfp4-2x stop after evidence capture.

Tests: distributed memory planner 9 passed. Daemon suite 810 passed, 7 failed, 101 ignored: one missing tests/fixtures/nvfp4/real_tensor_decode.json, six missing Python torch imports. Hostcache suite passed in full (59 unit tests plus integration suites; 2 ignored). Initial test attempts required selecting Python 3.12 instead of unsupported default 3.14, setting its shared-library path, and PYTHONHOME for embedding; final suite evidence is /tmp/astra-v8-suites-final.log, /tmp/astra-v8-memory-tests-final.log, /tmp/astra-v8-hostcache-tests.log. These are not green full-daemon results.

Precise unresolved blocker: no same-metric allocation trace yet attributes the additional GPU1 consumption to host mappings versus graph/lazy-module/launch residency across request shapes. The reduced-KV experiment still fails, so a reserve change alone is not established as sufficient. Shipping a fix or declaring concurrency/mixed/quality qualified would be unjustified. The first code instrumentation/change site is distributed.rs:389 (post-hostcache per-GPU cudaMemGetInfo and readiness headroom invariant), followed by accounting/bounding request-time growth rather than hiding its failure.
