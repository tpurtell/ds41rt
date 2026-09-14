# DeepSeek V4.1 EXL3 K3.25

Target: `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`.

This is the authoritative handoff for this quantization. A new assistant or
replicator should read this document, then inspect the repository, run records,
and live containers before acting. Recorded intentions are not completion
evidence. Do not restart a job because a chat ended or a tool observation timed
out. Keep this document current as implementation and execution advance.

## User requirements and operating contract

- Source model: `deepseek-ai/DeepSeek-V4.1-Flash`. Start from the latest remote
  `tpurtell/GPTQModel` main for a new run, not a sister project's older checkout;
  pin the selected commit and subsequent qualified changes for reproducibility.
- Vendor the fork under `third_party`, develop a separate V4.1 definition,
  commit incrementally, and push the fork and ds41rt changes to their `main`
  branches. This checkout currently uses local branch `dev`; publish its
  commits with an explicit `HEAD:main` fast-forward, not the stale local `main`.
- Build ds41rt's own documentation and scripts. `../glmrt` and `../ds4rt` are
  references for the established corpus, calibration policy, checkpointing,
  mixed-tier allocation, and distributed trellis machinery.
- Quantize only routed experts in BOTH the 40-block main model and three-block
  smaller dSpark. Use K3 candidates with error-based K4 upgrades at 3.25 routed
  payload bpw, allocating additional bits gate:up:down = 3:5:8. Do not silently
  omit dSpark or substitute a uniform tier. Preserve other source tensors.
- Reuse the latest calibration set's texts; identify and hash the exact corpus
  and render/tokenize for V4.1. Do not reuse another model's token IDs or silently
  replace the corpus. Verify the applicable established routing-coverage policy.
- Both RTX GPUs and all four Sparks are authorized. Activation generation and
  causal replay use only the two RTX GPUs. Independent trellis searches may
  run on the RTX GPUs and Sparks through the authenticated remote-worker path.
- Treat the checkpoint's own `inference/` code as the architecture oracle.
  A faster implementation is welcome after validation. Do not preserve wasteful
  conversions/allocations merely because the reference does them; distinguish
  numerical requirements from implementation inefficiencies and measure both.
- PLE stays memory mapped and reclaimable, with bounded gathers and prefetch
  as processing advances. Avoid hidden tensor references that retain mappings
  or activations. Inspect anonymous memory as well as RSS/file-backed pages;
  reclaimable PLE pages can obscure unrelated leaks.
- Export the two PLE tables into separate files from each other and all other
  tensors. Keep each table's necessary scales with that table, enabling later
  PLE quantization variants and hardlink composition.
- NVMe is approximately 14 GB/s and has limited free space. Use it for rolling
  wavefront state and recovery. Scratch at `/mnt/scratch` is approximately
  150 MB/s write, 500 MB/s read; use it if a full intermediate copy is necessary.
  Prefer direct final materialization in the standard local Hugging Face cache,
  with hardlinks to the published revision and no model redownload.
- Publish and verify `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`, and leave the
  matching complete revision in the user's standard Hugging Face cache.
- Launch the complete process once, detached from the assistant session. On
  failure preserve logs/checkpoints and stop; the user will request diagnosis
  and explicit resume, with code updates permitted during recovery.
- After several blocks are durably committed and memory is stable, check the
  live job only every 30 minutes to save tokens. Each check-in reports progress
  and final ETA. Use measured rates and state assumptions; until measurements
  exist say the ETA is unavailable rather than inventing one.

## Persistent run records and reconnection

Planned stable run root on NVMe:
`/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/`.
The following layout is a required implementation contract, not a claim that
these records or the launcher already exist:

| Path under run root | Purpose |
| --- | --- |
| `plan.json` | Immutable source/corpus/recipe/software identities and storage paths |
| `launch.json` | Coordinator and worker container IDs, image digests, host identities, launch command/time |
| `status.json` | Atomically updated phase, live handles, last committed main/dSpark block, error state and ETA |
| `logs/coordinator.log` | Persistent stdout/stderr, also available after container exit |
| `logs/workers/` | Worker logs copied or streamed to the coordinator run root |
| `events.jsonl` | Append-only timestamped phase/checkpoint/progress/failure events |
| `checkpoints/` | Authenticated durable projection and batch/block commit records |
| `reports/` | Preflight, correctness, memory, allocation, artifact and publication evidence |
| `recoveries/` | Each explicit resume's diagnosis and bound execution/code upgrade |

Do not put authentication tokens in logs, plans, commands recorded for public
use, or published model metadata. A status file is advisory: on reconnect,
read it and `launch.json`, then inspect those exact container IDs/processes.
For a live job, resume monitoring it. For a terminal job, inspect its exit code,
OOM status and failure log, then validate the last durable commit before an
explicit resume. Never infer that the next layer completed from a partial file.
Do not automatically restart a failed numerical job in a tight restart loop.
Recovery must bind code changes and state compatibility explicitly; a changed
numerical path can require invalidating affected downstream state.

The one-shot script must deploy workers, run calibration/quantization through
both namespaces, validate and export, and perform publication/cache steps
without needing another assistant turn to trigger a later phase. It must
preflight resources/identities and refuse to overwrite an existing run. Resume
must be a separate mode using the existing run root. Concrete launch, status,
log-tail and resume commands will be added here when that entry point exists.

Deployment requirement: one launch script must run the complete workflow in
detached containers, independent of this assistant session. Persist logs and
durable checkpoints on NVMe. Failure must retain evidence and stop; recovery
is an explicit resume after diagnosis, allowing reviewed implementation updates
without silently accepting incompatible old numerical state. The script must
deploy the four Spark trellis workers as well as the RTX coordinator. It must
not require an active tool session to supervise progress or trigger later
quantization phases/export. This deployment entry point is not implemented yet.

The GPTQModel starting revision is the remote fork's main HEAD observed on
2026-09-14: `339cccf0c4b62440b572440a7d06608ed9e4a165`. This is a new
submodule checkout, not the older GLMRT vendor pin. V4 support exists in this
fork; V4.1 support still needs implementation and numerical qualification.

The local complete source snapshot is
`dba1be0a40aa45a94ad051997016db3960a90277`. Its config has 40 main blocks
(384 experts) and three dSpark blocks (128 experts), hidden size 5120 and
expert intermediate size 2304. Quantize only routed projections. Preserve
vision, shared experts, attention, embeddings, heads, and other source tensors.

Allocate K4 upgrades from scored K3 candidates with gate:up:down ratio 3:5:8.
Per main block the quotas are 54/90/144; per dSpark block they are 18/30/48.
Replay each actual mixed block before calibrating its successor. Bind the
calibration corpus, source, implementation, recipe, and resume records by hash.
Reuse the latest calibration texts and attest tokenization for this model.

PLE tables at layers 1 and 14 must remain file-backed and reclaimable. Export
each table with its scales in its own shard, separated from other tensors.
The source shards currently also contain the small engram projection tensors,
so simply hardlinking those entire shards does not meet this separation.
Use bounded gathers and measured prefetch; do not allocate full tables as
Torch parameters or retain their pages through activation references.

Use NVMe for rolling activation/replay frontiers and durable projection
checkpoints. Aim to export directly into the standard Hugging Face cache,
then bind that same payload to the published commit without redownload.
Recheck peak space including K3 candidates, selected K4, export and both PLE
tables before launch; scratch is available if a full intermediate is necessary.

Current state: no quantizer running. The user stopped the ds41rt server and
released both GPUs on 2026-09-14; free memory was verified at approximately
95 GiB per GPU. Build this repository's own workflow and documentation;
the sister projects are implementation references, not this run's artifacts.
No calibrated block or final ETA exists yet. After several durable blocks and
stable memory measurements, monitor the live job every 30 minutes and report
completed blocks, memory, throughput, and estimated final completion time.

Completion requires full main+dSpark quantization, exact tier/payload audit,
quality and recovery verification, separate PLE files, successful Hub upload,
remote object verification, and the matching local standard-cache snapshot.

## Implementation evidence (2026-09-14)

The user authorized both RTX GPUs and all four Sparks. Activation generation
and causal replay stay on the RTX GPUs; distribute independent trellis searches
using the fork's authenticated `exl3_remote` scheduler. Remote-worker support
exists but needs qualification for this run before launch.

Transformers is pinned to the unmerged V4.1 text implementation from PR 48721,
revision `62d7ebd7de4938e072b7aaeb881593b79dc56835`, under
`third_party/transformers`. It reports `5.18.0.dev0` and imports with the
current GPTQModel fork after upgrading the quantization Python environment's
tokenizers to 0.23.1. It omits dSpark and vision execution. Preserve vision
source tensors; implement dSpark in our separate GPTQModel V4.1 definition.
No production auto-dispatch registration yet: stateful layerwise capture,
bounded source loading, and complete dSpark support are prerequisites.

The separate `deepseek_v41.py` currently provides unfused routed experts and
a parameter-free mapped PLE embedding. Tests pass for:

- exact five-layer FP32 logits before/after expert conversion, with eager
  expert dispatch on both sides (default grouped dispatch differs by 1.8e-7);
- exact expert outputs against the checkpoint's unchanged `Expert` body in
  FP32 and BF16 on CPU and each RTX GPU;
- exact mapped FP8+E8M0 embedding gathers against the Transformers lookup on
  CPU and both RTX GPUs, including chunking, repeats, and empty inputs;
- mapping lifetime, explicit prefetch/release, owned rows, and bounds.
- five-layer explicit replay with owned CPU boundaries, carrying both mHC
  pre-mix and CSA2 shared state, matching full-forward logits exactly and
  reproducing each layer when replayed twice from the same input state.

`gptqmodel.utils.v41_replay.V41ReplayBatch` now implements that replay contract
for full-prompt text batches. It drops consumed PLE gathers at their layer.
Durable serialization, source-layer loading, dSpark replay and integration
with the quantizer's layer loop still remain to be implemented and tested.
TileLang 0.1.14 was built in the development container for direct checkpoint
reference tests; the production dependency lock is still pending.

`gptqmodel.utils.v41_source.V41Source` now reads source tensors individually
and can assemble one decoded main block on a chosen device. Loading real
block 0 completed in 6.02 seconds: 1,174 tensors and 25.62 GiB allocated on
GPU 0. The process exited successfully and GPU allocation returned to idle.
This decoded representation is for diagnostics and trellis weights, not yet
the calibration forward baseline: native FP8 activation quantization still
needs integration. No full source rewrite was made.

The source header test validates every runtime parameter/buffer name and
logical shape for all 40 main blocks against the actual 48-shard checkpoint,
with exact coverage of all non-scale, non-PLE main-block tensors. The two
PLE full-read rejection cases also pass. dSpark source mapping/loading is
still pending and is not covered by this test.

Native source-forward work: `v41_native.V41NativeLinear` retains FP8/packed-FP4
weights and E8M0 scales and calls the checkpoint's TileLang kernels with
explicit BF16 output allocation (no process-global dtype dependency). Real
block 0 loads in 3.16 seconds and uses 6.91 GiB allocated GPU memory. Individual
FP8 and FP4 projections execute with finite BF16 output. This is not full-block
qualification.

`quantization/validate_native_block.py` compares real block 0 to the checkpoint
reference at 32 and 129 tokens. The official sparse attention's 64-head kernel
requires 141312 shared-memory bytes, exceeding SM120's limit; the harness calls
the unchanged kernel on independent 16-head groups. It also uses the official
runner's default CUDA device context for reference-generated indices. TileLang
emits a thread-sync warning for this kernel; do not ignore this when assessing
oracle reliability. The harness currently emits diagnostic error metrics, not
a passing acceptance result.

The initial block comparison differed by 2.16%/2.76% relative L2. A concrete
candidate discrepancy was found: Transformers RMSNorm rounded to BF16 before
the learned weight multiplication, while the checkpoint rounds after it.
The separate GPTQModel definition now uses the checkpoint order. Attention
input comparison is exactly zero after that change; block output still differs
by 1.33%/1.34%, and attention output differs by about 1%. Next investigation:
attention accumulation/rounding (candidate eager QK is BF16 whereas the reference
kernel accumulates FP32), then mHC and downstream route/output differences.
Do not begin production calibration until these numerical gates are resolved.

### Real block 0 parity resolved

The real block 0 native comparison now passes exactly at both 32 and 129 tokens:
zero output error, zero mHC pre-mix error, and zero error at every recorded
attention/FFN input and output. The harness now fails on a nonzero output/carry
difference. The final run exited 0; its persistent log is
`/home/tj/.cache/ds41rt/quantization/deepseek-v41-exl3-k325-v1/reports/native-block0-parity.log`.
All seven existing component/source tests also pass.

Additional discrepancies resolved in our separate GPTQModel implementation:

- Preserve FP32 rotary coefficients and complex multiplication through the
  final BF16 store; early rounding or separate real multiplies changed results.
- Use the reference's sparse-attention accumulation semantics. The candidate's
  eager BF16 QK and normalized-BF16 probability path was not equivalent.
- Apply mHC normalization after its projection, and preserve multiply/reduce
  order at residual expansion. Both affect low-precision downstream routing.

`V41NativeAttention` owns a full-prompt forward and uses shared compressed KV
directly, avoiding the candidate eager implementation's per-query duplicated
selected-KV bank. It rejects decode-cache input. Compressed source/consumer
branches are implemented but not yet numerically qualified; do not infer their
correctness from the sliding-only block-0 result. The complete calibration
pipeline, compressed/PLE blocks, dSpark, recovery and detached launcher remain
unfinished. Allocation/performance tuning follows correctness qualification;
the mHC multiply/reduce path is still a candidate for a measured fused kernel.

These are component gates, not full official-reference backbone parity or
quantization qualification. The checkpoint's `inference/model.py` remains the
architecture oracle. Preserve necessary arithmetic while measuring allocation
and conversion costs; reference implementation inefficiencies are not required.

Development container: `ds41rt-quant-dev`, based on local image
`sha256:6213ea40c79617373562d7f2d3cc5fa25ca9d03e27dd213361b216c3e315e9f4`.
It mounts this checkout at `/workspace`, source HF cache read-only at `/hf`,
and uses both vendored Python source trees through PYTHONPATH. It is a development
environment, not yet a reproducible production image. Run component gates with:

```bash
docker exec -w /workspace ds41rt-quant-dev python -m unittest discover -s quantization/tests -v
```

### Additional real-block gates (2026-09-14)

The native indexer now preserves reference BF16 score arithmetic, scale order,
and sorted selected positions while chunking query scoring. This resolved the
ratio-2 discrepancy; the earlier failed diagnostic log is retained as evidence.
The following persistent reports under the run's `reports/` directory show zero
hidden-state and mHC carry error:

- `native-block2-parity-r2.log`: ratio-2 source, 32 and 129 tokens.
- `native-block2-long-parity.log`: ratio-2 source, 2,049 tokens, exercising
  selection from more than 512 compressed positions.
- `native-block20-parity.log`: ratio-1 source, 32 and 129 tokens.
- `native-block1-ple-parity.log`: first PLE block, 32 and 129 tokens.
- `native-block14-ple-parity.log`: second PLE plus ratio-2 source, same lengths.

Block 2 and block 14 also match compressed KV, index keys and selected indices
exactly. PLE block comparisons gather actual checkpoint rows for random valid
hash indices and compare the complete Engram arithmetic; they do **not** prove
token-to-hash parity. All seven existing component/source tests pass again.
Cross-layer source/consumer and reindex/candidate reuse, tokenizer/hash parity,
dSpark and complete calibration/quantizer integration remain unqualified.
No production quantization job is running yet.

### Chained replay and durable batch frontiers

`validate_native_block.py --layer N --through-layer M` now tests consecutive
real blocks through `V41ReplayBatch.advance`, with independently owned CPU
boundaries for candidate and reference. Reports in `reports/`:

- `native-chain2-3-parity.log`: both blocks exactly match at 32, 129 and 2,049
  tokens, including compressed KV, index keys and selected positions.
- `native-chain20-24-parity.log`: all five blocks exactly match at 32 and 129
  tokens, including candidate masks and the reindexing consumer at block 24.

The latter lengths exercise candidate publication/reuse but do not exercise
actual candidate pruning: the source selects up to 2,048 blocks of 8 positions,
so a longer-than-16,384-token gate remains required. These are synthetic hidden
inputs to real block chains, not end-to-end tokenized-model qualification.

`gptqmodel.utils.v41_checkpoint` implements atomic safetensors replay snapshots:
owned CPU tensors, explicit typed metadata (including integer PLE layer keys),
file and directory fsync, SHA-256 verification, and exact caller-supplied
provenance matching before loading. Linux inode-pinned reads avoid a pathname
replacement between hashing and mapping. No pickle is used. The caller must
bind source/corpus/recipe/code identities and journal the returned checksum only
after save succeeds. This API does not itself commit quantized projections or
declare a block complete; the production transaction journal is still pending.

All ten component tests pass (`reports/component-tests-frontier.log`), including
corruption and identity rejection, simulated publication failure preserving the
old snapshot, independently owned loaded tensors, and five-layer logits exactly
matching full forward when every boundary is saved/reloaded. This is not yet a
full quantizer crash/resume test. dSpark, full-model input preparation, production
worker deployment, one-shot launcher and final export/upload remain unfinished.

### dSpark core loader and numerical gates

The checkpoint's `DSparkAttention` is not ordinary causal main-model attention:
prefill seeds a 128-position main-history ring without running the experts;
subsequent draft execution attends to that ring and all five draft positions.
Our `V41NativeDSparkAttention` reconstructs the ring from explicit projected main
history, preserving reference key order while projecting only the last window.
It carries no hidden mutable decode cache between quantization evaluations.

`V41Source.load_decoded_block(..., namespace="mtp", native_kernels=kernel)` now
loads the three dSpark core blocks with 128 experts and top-3 routing each,
without PLE or compressed attention. It excludes main_proj/main_norm and the
shared embedding/auxiliary heads; these still require explicit input preparation
and preservation in the exporter. Do not infer complete dSpark support from the
core loader or pass the whole mtp namespace through unquantized at export.

`quantization/validate_dspark.py` compares the real three-stage chain with the
checkpoint's original dSpark blocks, seeding the reference cache via its prefill
path and then executing a draft. Synthetic projected main features and synthetic
draft embeddings isolate core math. All stages have exactly zero hidden/carry
error at main lengths 2, 129 and 257 on RTX 0, and 1,025 on RTX 1, including
multiple window wraps. Persistent reports:

- `reports/dspark-core-parity.log`
- `reports/dspark-core-long-parity.log`
- `reports/component-tests-dspark.log`: all eleven tests pass; the new header
  gate covers every dSpark core name/shape and all 1,152 routed projections.

Still required: derive actual draft inputs from main-model target-layer **inputs**
(the reference collects stream means before blocks 37, 38 and 39, not their
outputs), qualify main_proj/main_norm and token/noise embeddings, integrate both
namespaces with calibration and quantization, and test durable end-to-end resume.
There is still no production quantization run or finished detached launcher.

### dSpark input adapter and target-feature capture

`V41ReplayBatch` now carries target-layer IDs and owned target features through
durable boundaries. A temporary attention-hyperconnection pre-hook captures the
unweighted BF16 stream mean after any PLE injection, before the block executes;
it is removed even on failure. The tiny five-layer replay test checks these
features survive every save/load boundary without numerical changes.

`V41Source.load_dspark_input` loads the real `mtp.0.main_proj`, `main_norm` and
shared BF16 token embedding (or reuses a supplied embedding). `V41DSparkInput`
builds the five-position noise draft and explicit main history. Teacher-forced
alignment is enforced by `prepare(target_features, token_ids, position=P)`:
main features cover only positions 0..P; the known first draft token is corpus
token P+1, replacing the main model's sampled next token; all remaining draft
positions are noise. Future main features are excluded before projection.
This is a calibration policy, not a claim that the reference itself teacher-forces.

`reports/dspark-input-core-parity.log` qualifies input projection/normalization,
real embedding gathers and the three-stage core chain against the checkpoint at
history lengths 2, 129 and 257, all exactly equal. Target features remain synthetic
in this diagnostic; full-main-to-draft corpus replay is still pending.
`reports/component-tests-dspark-input.log` records twelve passing tests, including
target ordering, P+1 token selection, noise placement, and future-feature exclusion
(future rows replaced with NaNs must not change the draft input).

Remaining production integration includes tokenizer/hash qualification, corpus
preparation, main-to-draft replay, EXL3 wavefront/worker scheduling, transaction
journaling and detached one-shot orchestration, isolated PLE export and upload.

### Corpus tokenizer and PLE hash attestation

The latest GLM-5.3 NEXT corpus is reused as unchanged raw prompt text from
`/home/tj/.cache/glmrt/calibration/glm-5.3-exl3-k4-next-v1/calibration.jsonl`.
Do not reuse its GLM token IDs/count as V4.1 identity. The original manifest is
`a4d51c98ca76f57d58d18b61779ba6ca79936f002f1812ca2ef4482d91824da5` (SHA-256).
`quantization/attest_inputs.py` records V4.1 tokenization and checks the checkpoint
hash implementation without loading model weights. Its contract is raw prompt
text, add_special_tokens=True, no chat rendering, padding, truncation or packing.

The completed attestation (`reports/input-attestation.log`) records:

- 1,441 prompts, 1,056,269 V4.1 tokens, longest prompt 1,074 tokens.
- Corpus SHA-256: `003686eedf3e533b016e939bb6bb462529b95322c2c44a222f0d8208c63f41ba`.
- Token-stream SHA-256: `5b8d681bead24b23f2ef97797c9b93b54f562cf3e6a73712ed5ce5dd66b6314a`.
- Tokenicer 0.0.14 agrees with direct loading on every corpus record and six
  boundary probes. Tokenicer supplies the existing EOS-as-padding default; the
  unpadded input IDs and masks agree. No normalization patch was necessary.
- All tokenizer entries map identically to the 99,092-entry compressed vocabulary;
  primes, offsets and multipliers match. Two 1,024-token sequences with and without
  DEAD masks produce exact reference hashes on CPU and both RTX GPUs.

The development environment has stale installed Transformers distribution
metadata (5.14.1) while PYTHONPATH loads vendored 5.18.0.dev0. The report explicitly
records both identities plus the imported modeling-file hash. The production
image must make the dependency installation reproducible, not trust distribution
metadata alone. This is an input/hash gate, not dense-model quality evaluation.
Full main-model corpus replay and quantization remain pending.

### Native main input adapter and executable mixed recipe

`V41MainInput` loads no decoder blocks or PLE tensor parameters: only the BF16
token embedding, attested small hash state, FP32 rotary coefficients, and mapped
PLE handles. It accepts joint equal-length unpadded independent text sequences.
PLE gathers move immediately into owned CPU replay storage. Native attention
constructs sliding indices, avoiding a quadratic causal mask in each checkpoint.
Padding/packing are not supported by this adapter; do not silently pad input
groups or split an already-issued joint batch when connecting the corpus loop.

`quantization/replay_source.py` is a diagnostic main-to-draft replay, **not** the
production one-shot launcher. It loads/releases one block at a time, preserves
main target features, and saves/reloads every boundary in a temporary directory.
Its temporary smoke snapshots are discarded on exit; persistent reports are in
the run's `reports/` directory. Production journal/checkpoint retention remains
separate work and must not copy this temporary-lifetime policy.

`quantization/mixed_recipe.py` supplies both `base` and `mtp` policies to the
fork's existing causal EXL3 mixed-tier selector. Its source-key validator demands
exact coverage of all 47,232 routed projections and per-block 54/90/144 main or
18/30/48 dSpark upgrades. This yields 11,808 K4 projections and exactly 3.25 bits
per routed weight, excluding metadata and preserved non-routed tensors. Selection
still uses the fork's measured K3 Hessian-relative error times natural gate-squared
mass; tests use synthetic tier maps only, not measured production selections.
All fifteen component tests pass (`reports/component-tests-main-input.log`),
including policy compatibility with the fork's selector for all 43 blocks.

The initial main-to-draft smoke completed successfully in 211.56 seconds:
two jointly executed copies of a 17-token real-tokenizer prompt, all 40 main
blocks and 3 dSpark blocks, finite outputs, and all 43 checkpoint round-trips.
Target captures accumulated at exactly 37, 38 and 39. Peak GPU allocation was
8,973,808,128 bytes; post-main-block allocation remained 1,358,417,408 bytes and
dropped to 34,590,208 after releasing the shared embedding for dSpark. Maximum
process RSS was 2,714,356 KiB. Evidence: `reports/native-main-draft-smoke.log`.
These are short-input native-source smoke figures, not production memory bounds,
full-reference end-to-end parity, calibrated quality or a quantization ETA.
No production quantization has started.

### Bounded natural-route Hessian capture

`gptqmodel.utils.v41_capture.V41Capture` captures selected experts only, sharing
one gate/up Hessian and a separate down Hessian per expert. It stores raw FP32
X.T@X sums with explicit row counts, chunked input conversions, and deterministic
CPU natural-route counts and gate-squared mass. Forward routing is never changed.
Down inputs include the model's activation/clamping and pre-down route weighting.
Capture hooks are removed on exit; failed captures cannot export partial evidence.
Zero-row experts retain explicit zero sums/counts, not fabricated calibration.
Low-coverage augmentation and normalization are separate pending integration.

Capture requires TF32 disabled at process startup before concurrent workers;
the initial development GPU probe correctly refused its enabled default. The
failure is retained in `reports/native-capture-parity.log`. With explicit FP32
startup, `reports/native-capture-parity-fp32.log` shows real block-0 output and
carry still exactly equal to the checkpoint at 129 tokens while four experts'
Hessians are captured, including one naturally zero-row expert. All seventeen
component tests pass, covering raw sums, shared gate/up input, route-weighted
down input, natural mass fractions, zero rows and hook cleanup.

`quantization/probe_exl3.py` is a diagnostic bridge from native MoE capture to the
actual K3 MCG trellis quantizer. It uses synthetic inputs and must never supply
production candidates or quality evidence. The production corpus loop, cold-route
policy, distributed scheduler and detached launcher remain unfinished.

The projection bridge probe succeeded (`reports/exl3-projection-probe.log`):
`layers.0.ffn.experts.3.w1.weight`, 16 naturally routed synthetic rows, physical
K3 MCG trellis shape [320,144,48] INT16 with FP16 suh/svh and MCG marker. Returned
reconstruction is finite, proxy error 0.0005012495. Search-call wall time was
16.06 seconds including ~15 seconds compiling the fork's EXL3 extension. No
production candidate was retained. This proves the capture/search API bridge,
not full-corpus quality, packed runtime replay correctness or throughput/ETA.

### Selected packed-weight replay

`v41_mixed_replay.install_projection` replaces only routed w1/w3/w2 linears,
validates K3/K4 MCG packed geometry and finite scales, and reconstructs BF16
weights from trellis/suh/svh via the fork's existing reconstruction helper.
The search function's raw `weight_q` must not be used for propagation: its
higher-precision scale/reconstruction path is not the serialized artifact.
This follows the fork's BF16 dense propagation contract, not a claim of exact
equivalence to every fused serving GEMM kernel. Packed payloads remain the
export artifact; the dense replay copy is only block-local working storage.

`reports/exl3-packed-replay-probe.log` records actual K3 capture/search, safetensors
save/load, installation and two identical finite mixed-MoE forwards. The raw
search weight differs from packed BF16 by up to 0.0006238967, confirming the need
for the explicit reconstruction boundary. Post-compilation probe duration was
1.39 seconds, but this sparse synthetic calibration is not production timing or
quality evidence. All nineteen component tests pass, including invalid packed
payload rejection without modifying the source module.

The equivalent K4 probe on RTX 1 also passed, including serialization and
repeatable mixed-MoE execution (`reports/exl3-k4-packed-replay-probe.log`), with
physical trellis shape [320,144,64]. Both tiers therefore cross the tested
capture/search/serialization/reconstruction boundary. Full selected-block
capture ordering, corpus execution, low-coverage handling and production
orchestration remain unfinished; these single-projection probes do not replace
those gates.

### Cold-expert recovery adapter

`V41Recovery` now uses the fork's `learned_router_ranked_choices` to verify exact
live top-k and retain adjacent-rank candidate inputs without modifying routing.
For this source that is main ranks 7–12 and dSpark ranks 4–6. Selection matches
the established V4 recovery implementation: rank-major, then corpus row order,
not score-gap sorting. It retains at most 1,024 owned CPU rows per selected
expert/rank; only the rows needed after the complete natural census are used.
Selected coordinates are hashed and observed/selected rank counts are recorded.

Recovery produces a new raw Hessian, preserving the original natural evidence.
It records natural, augmented, residual identity and effective row counts
separately. Remaining shortages add `missing * I` to raw sums before EXL3's
count normalization (equivalent to the fork's normalized-2I convention).
Recovery-only down inputs use unit route weight, matching the established
direct-expert policy, but preserve V4.1's FP32 SwiGLU/clamping then BF16 store.
The caller must install selected gate/up tiers before down-phase recovery.
Recovery does not contribute to natural gate-squared mass used for tier scores.

`reports/exl3-recovery-probe.log` passed the actual K3 search and serialized
mixed-MoE replay with 16 natural + 9 adjacent-rank + 999 identity rows. Those
numbers are from synthetic diagnostic input, not production corpus coverage.
Twenty component tests cover the adapters, including unchanged natural outputs,
candidate selection, unit-weight down construction and true-zero identity case.
Production still needs to bind stream identity and recovery parameters into the
durable journal, stage gate/up then down capture correctly, and deploy the full
corpus/distributed/one-shot workflow. No production quantization has started.

### Reusable routed FFN frontier and phase-specific capture

`V41RoutedBatch.from_replay` now runs the block's attention and router once,
capturing owned CPU FFN input/logits/weights/indices and stopping before any
expert GEMM. Expert subsets reuse this frontier instead of repeating complete
block execution. This is an input-capture artifact, not a propagated output.

`V41Capture(..., phase="gate_up" | "down")` allocates only that phase's Hessians.
`capture_routed` computes gate/up raw Hessians directly from routed inputs without
expert GEMMs. Down capture evaluates only selected experts' current gate/up
weights, with exact V4.1 clamping, FP32 SiLU, route multiplication and BF16 store;
it never executes their down projection. The block driver must install all
selected gate/up tiers before initiating down capture. `V41Recovery.observe_routed`
likewise reuses the same logits/inputs and evaluates rankings on the router's
device, without a second attention/FFN run.

The real block-0 diagnostic at 129 tokens matches full-forward hook capture
exactly for all three projections of experts 0–3, including a zero-row expert.
Block output/carry also remain exactly equal to the checkpoint reference:
`reports/native-direct-capture-parity.log`. Twenty-one component tests pass;
they include phase-specific allocation, no expert execution during frontier
capture, exact direct-vs-hook Hessians, and identical direct recovery evidence.
Durable routed-batch storage, the full block phase driver, corpus scheduling and
the detached distributed launcher are still pending production integration.
