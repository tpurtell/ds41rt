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

### Durable routed batches and artifact journal

The atomic safetensors checkpoint writer now distinguishes `replay` from
`routed` payloads. Routed batches validate tensor geometry/dtypes, finite values,
expert bounds and duplicate assignments before publication and after loading.
Legacy version-1 snapshots without a kind field remain replay-only. A routed
batch can never load as a completed replay state.

`quantization/run_store.py` provides a SQLite WAL/FULL-synchronous artifact
journal with immutable run identity and records. Payloads are fsynced before
record commitment; records bind kind, path, bytes, SHA-256 and hashes of committed
dependency records. Missing dependencies reject the transaction; orphan payloads
are retained but not counted as completed work. Reopening verifies identity,
and artifact reads verify payload/checksum and immediate dependency-record hashes.
Only payload paths inside the run root are accepted. This is not yet the phase
driver: the caller must freeze source/corpus/recipe/code identity, choose unique
artifact paths, define phase dependencies, and implement explicit recovery.

`reports/native-journaled-capture-parity.log` passes the real block-0 capture gate
after input/routed publication, journal commitment and routed reload: exact raw
Hessians, route evidence and checkpoint-reference block outputs. Twenty-four
component tests pass (`reports/component-tests-run-store.log`), including kind
separation, invalid routed publication preserving old bytes, journal reopen and
idempotence, missing dependencies, identity mismatch and payload corruption.
The diagnostic still discards its temporary files; production retention and
end-to-end crash/resume orchestration remain unfinished. No production quantizer
is running.

### Journaled block phase driver

`quantization/block_driver.py` now connects the validated components for one
block. It binds an ordered routed-batch inventory, captures phase-specific
Hessians in bounded expert subsets, journals recovered raw Hessians and K3
candidates, selects K4 by per-projection risk quotas, reconstructs selected
gate/up tensors, and only then captures/quantizes down. Completed phase markers
depend on their tier plan and every selected packed projection. Explicit resume
reloads completed candidates/phases rather than searching them again. Search
dispatch is injectable; the default currently uses the local RTX only.

The driver's tie-break is numeric expert index for equal risk, frozen in its
implementation identity. Checkpoint metadata now supports typed lists as well as
tuples to preserve the quantizer's nested metrics. All twenty-five component
tests pass, including a small-model injected search failure, no duplicate search
for committed candidates, phase ordering, exact upgrade counts and completed
phase reload. This test mocks search/installation; it is not real full-block
qualification.

A real full dSpark block-0 diagnostic was started with `probe_block.py` using
64 synthetic FFN rows, all 128 experts, and the exact 18/30/48 quotas. Its durable
root is `/home/tj/Developer/ds41rt/.ds41rt-cache/quantization-diagnostics/mtp0-phase-v1`
(container `/workspace/.ds41rt-cache/quantization-diagnostics/mtp0-phase-v1`).
Persistent log: `reports/full-mtp0-phase-probe.log` under the standard run report
root. Initial K3 gate/up candidates are committing successfully. Inspect the
live process/journal before any restart; completion has not yet been established.
These synthetic candidates must never enter the production artifact.

Still absent: full-corpus frontier creation, propagated mixed-block output
commitment, distributed worker dispatch/deployment, production recovery control,
detached end-to-end launcher, final export/upload. The block phase driver alone
does not declare a whole quantization block complete or satisfy deployment.

### Mixed-output wavefront commitment

`quantization/wavefront.py` now binds a block's ordered input inventory, captures
and journals routed FFN batches, invokes the phase driver, then propagates the
selected mixed block from each original input. Every output is checked for finite
hidden/carry state and the correct next-layer index, saved atomically, and bound
to its input plus the completed selected-down phase. Only after all outputs exist
does the block-complete record commit. The returned output keys/provenance become
the next layer's inputs; no native or capture-only output substitutes for them.

Explicit resume skips committed routed batches and outputs, reloading selected
weights through the phase driver when propagation remains. Completed block reload
verifies all output states. No retention/deletion or automatic restart is included.
An injected interruption after the first of two durable output batches resumes
without repeating it, produces exact selected-weight outputs, and rejects changed
input ordering. Twenty-six component tests pass (`reports/component-tests-wavefront.log`).
The unit test uses a mocked selected-weight phase; full checkpoint/corpus integration
is still required. The existing real mtp0 phase diagnostic remains live and is
committing candidates; it does not yet exercise this new propagation wrapper.

### Full dSpark phase qualification and attested corpus inputs

The previously live real-weight mtp0 diagnostic completed successfully in
576.27 seconds. Its persistent log contains exactly 384 K3 searches (128 each
for w1/w3/w2) and 96 K4 searches (18/30/48), followed by down-phase commitment
and finite, repeatable mixed-MLP output checks. These are synthetic 64-row
diagnostic candidates, not production calibration or final quality evidence.
The original run is terminal; an explicit same-identity completed-phase reload
was then run separately, with output in `reports/full-mtp0-phase-resume.log`:
passed in 1.81 seconds with no new candidate or capture events.

`quantization/corpus_inputs.py` now verifies unchanged corpus bytes and the
entire token stream against the passed input attestation. It rejects duplicate
record IDs, padding, invalid IDs, and changed counts/lengths/digests. Normal
tokenizer construction remains Tokenicer-owned; no tokenizer normalization
patch is introduced. The actual 1,441-record corpus passes this new path with
1,056,269 tokens, using the previously attested Tokenicer 0.0.14 environment.

Initial-frontier preparation freezes the ordered original records and attestation
in the journal, uses one unpadded original record per batch, and supports two
independently owned input adapters. The caller must construct these on the two
RTX devices; Sparks are not activation workers. Two-record windows bound pending
states and disk writers, with all SQLite writes on the coordinator thread.
Committed frontiers are checksum/provenance checked and skipped on explicit
resume; uncommitted orphan payloads are regenerated. Completion depends on every
input frontier. This is a callable pipeline component, not yet the detached
launcher or a launched full-corpus preparation run.

The interruption test stops after the first durable input, resumes only missing
records, verifies completed reload performs no preparation, and rejects reordered
inventory. The full component suite now has 27 passing tests in
`reports/component-tests-corpus-inputs.log`. The test uses lightweight adapters;
two-RTX integration of this new corpus coordinator remains to be exercised.

### Bounded distributed-search integration

The fork's existing `exl3_remote` implementation was inspected directly. It
already provides authenticated tensor envelopes, immutable slot assignments,
worker qualification, bounded two-request Spark staging, and worker-side
projection checkpoints. We reuse that protocol rather than build a second one.
`quantization/distributed_search.py` adapts our V4.1 weight/Hessian search calls:
two qualified coordinator slots plus four Spark endpoints are required, as is
a durable assignment store. Assignment identity includes run, projection and
K3/K4 tier. Remote requests bind decoded weights, raw recovered Hessians/count,
quantizer numerical contracts, run identity and execution identity. Results
retain execution/assignment evidence, and leases release on every exit path.
Only weights and Hessians are sent to Sparks; activation generation stays RTX-only.

`BlockDriver` now accepts explicitly concurrency-capable search backends, with
bounded windows (ten requests for this topology). Search runs in worker threads;
all journal loads/publications remain on the coordinator thread. K3 and K4
search batches finish before selected-weight installation and the next capture
phase. Serial search remains the default. This changes the driver code identity:
the old synthetic mtp0 diagnostic must not be silently resumed under this code.
Its successful completed-phase reload evidence above used the original identity.

All 29 component tests pass (`reports/component-tests-dispatch.log`). New tests
exercise simultaneous searches, journal-thread ownership, no repeat of committed
candidates, remote request validation, tier-distinct assignment IDs and lease
release after a network exception. Remote transport/search is mocked in that
test; no six-device numerical qualification is claimed. CPU decoding for remote
transfer exactly matches RTX decoding for all three real expert-0 projections
in main block 0 and mtp block 0 (`reports/remote-source-decode-parity.log`).

Read-only SSH checks confirm all four hosts (`ostrich,dodo,emu,kiwi`) are reachable,
each exposes a GB10, and none had a running Docker container at this check.
Worker image/preflight reconstruction, actual remote numerical qualification,
deployment and end-to-end one-shot orchestration remain required. No production
quantization job has been launched.

### Reconstructed Spark worker runtime

`quantization/exl3_worker.py` implements our worker endpoint using the inspected
ds4rt server as reference and the vendored authenticated protocol unchanged.
It bounds staged request bodies to two, serializes GPU execution, limits body
size/read time, closes connections after requests, signs responses, and retains
tracebacks in container logs. Worker checkpoint/resume uses the fork's existing
content-addressed store. Tests cover authentication, response signatures,
oversized-body rejection, and staging-capacity rejection/release. Thirty
component tests pass (`reports/component-tests-worker.log`).

`docker/Dockerfile.quant-worker` layers our current fork, including its sibling
`gptqmodel_ext` CUDA sources, onto the existing platform-local quantization base.
The inspected base on ostrich is
`sha256:a70e6af77cd323ae2fe507fbeb5f6353a6fdce3e9247d2d5b9632bf4c92a55ae`,
bound locally to `ds41rt-quant-base:a70e6af77cd3` for Docker builds. Deployment
must inspect/verify that binding; the tag alone is not sufficient identity.
Current worker image on ostrich:
`sha256:a555eaf016eeb9ddec333215a9dbc3c97b4e83733cf765cdfbd5340651a4319a`.
It imports GPTQModel 7.3.6 on Python 3.14.6 free-threaded, Torch 2.13.0+cu130,
Triton 3.7.1. Workers need no V4.1 tokenizer/model loader, so their inherited
Transformers 5.14.1 is not used for activation generation.

Identity-only execution on the actual GB10 passes; it records live GPU UUID,
runtime versions, image identity, worker hash and Python/CUDA source-tree hash.
TF32 is disabled. This runtime report is explicitly not numerical qualification.
Evidence: `reports/spark-worker-build-ext.log` and
`reports/spark-worker-identity-ext.log`.

The first standalone GPU search probe exposed missing sibling CUDA sources in
the initial image, before producing a candidate; its failure is preserved in
`reports/spark-worker-search-probe.log`. The corrected image's probe is recorded
separately in `reports/spark-worker-search-probe-ext.log`. It uses a synthetic
5120x2304 weight and raw diagonal Hessian at count 1024, not production weights.
The JIT cache uses the named Docker volume `ds41rt-quant-jit` on ostrich.
The corrected K3 probe completed successfully in 31.49 seconds including JIT
compilation, returning the expected 320x144x48 trellis and finite error metrics.
The probe container exited normally; no worker service is running yet. Authenticated
real-weight cross-device qualification and deployment on all four hosts remain
required, as does the complete detached coordinator workflow.

### Four live authenticated workers and real-weight execution qualification

All four hosts now run a detached container named `ds41rt-quant-worker`, serving
port 17841 with authentication and persistent `ds41rt-quant-worker-state` and
`ds41rt-quant-jit` Docker volumes. Restart policy is `no`: failure recovery remains
explicit. No production quantization coordinator is running. Container/image IDs
are saved in the run root's `workers.json`; inspect these containers and preserve
their state rather than starting duplicates. Their Docker logs remain on each
host, and startup logs are copied into `reports/<host>-worker-start.log`.

`quantization/qualify_worker.py` ran through the authenticated tensor protocol on
each host. All 48 cases pass: real checkpoint expert-0 weights from main block 0
and mtp block 0, each w1/w3/w2 at K3 and K4. Every packed tensor exactly matches
the RTX reference and an immediate repeat request is a worker checkpoint hit
with identical tensors. Evidence: `reports/<host>-worker-qualification.log`, each
ending in `qualification_passed` with 12 cases. The Hessian in this test is a
synthetic raw diagonal matrix at count 1024. This establishes sampled execution
equivalence, not full-corpus Hessian qualification or quantized-model quality.

All worker runtimes report the same combined Python/CUDA source hash
`0262b1e8593e25b06617e8ed36e5863f8e14a027e69749c69a2de6f487062a5c`.
Platform-local final image IDs differ because builds occurred on separate hosts;
each endpoint is bound to its own inspected image and live runtime identity.

`python quantization/deploy_workers.py --run-root <RUN_ROOT>` reconstructs the
worker deployment from our repository. It binds the inspected base image ID,
builds on all four hosts with bounded parallelism, creates/reuses a private token,
starts missing workers, and refuses mismatched or stopped existing containers.
It never removes containers or volumes. Running it against this deployment
successfully rebuilt from cache and reused all four original container IDs;
`reports/workers-deployment.log` records that check. It produces `workers.json`
atomically, but deliberately does not claim numerical qualification from running
state alone. The private `<RUN_ROOT>/worker-token` must not be committed or logged.

This deployment command is only a component of the required one-shot workflow.
Automatic identity collection/qualification gates, full-corpus two-RTX capture,
coordinator orchestration/retention, final export and upload still need integration.

### Journaled rolling-storage retirement

NVMe free space measured about 1007 GiB before this change. Retaining all old
frontiers and raw Hessians would consume that during the run, so
`quantization/retention.py` now provides an explicit completed-block cleanup
component. It verifies every replacement frontier and selected projection before
retiring the block's original inputs, routed batches, Hessians and unselected
candidate payloads. Selected weights, current outputs, phase/tier metadata and
all original artifact/dependency records remain intact.

`RunStore.retire_files` requires an explicit distinct key list and a committed
block barrier, checks transitive dependency-record hashes, rejects external or
symlink targets and aliased artifact paths, and verifies payload bytes before
deletion. Retirement intents commit durably before unlink, so an interrupted
cleanup can resume even after some files have disappeared. The same retirement
is idempotent. Normal verified reads reject retired payloads explicitly; metadata
reads remain available to validate descendants. Disk payloads are removed, not
archived, so regeneration would require replay/search from an earlier retained
boundary or source; the caller must preserve every still-required boundary.

This API requires exclusive coordinator ownership, not concurrent mutation of
the artifact tree. It is not called automatically yet. The forthcoming coordinator
must resume from its latest retained frontier rather than replaying completed
blocks whose old outputs have been intentionally retired. No existing diagnostic
or production payloads were removed while implementing this change; only test
temporary directories exercised deletion.

Tests cover interrupted unlink, immutable records after retirement, repeated
cleanup, unrelated-target rejection, and preservation of selected weights/current
frontiers. The wrapper test uses small synthetic metadata/frontiers; full-corpus
storage-retention integration remains required.

`Wavefront.latest(namespace)` supplies the resume boundary without reading older
retired payloads: completed block markers must form a contiguous prefix, and only
the newest output frontiers are loaded/verified. Corrupt newest outputs or a gap
in completion markers fail closed rather than silently selecting an older state.
All 32 component tests pass (`reports/component-tests-retention.log`).

### Namespace coordinator and frozen dSpark anchor policy

`quantization/coordinator.py` now connects the block driver, propagated output
wavefront, latest-frontier lookup and post-commit retirement. It binds each
namespace's original inputs, loads one block at a time, commits/cleans up before
advancing, and resumes cleanup after a block-commit interruption without executing
that block again. Namespace completion is separate from whole-model completion.
An exclusive run-directory lock is provided and must surround the entire eventual
coordinator lifecycle, including input creation. This is not yet a CLI launcher;
two-RTX execution placement, main-to-dSpark handoff and export remain to connect.

Re-reading the sister project's optimized dSpark procedure established the
pre-route anchor policy: exactly 327,680 stratified selections with seed 20260809,
one splitmix64 choice per equal-width global stratum, retaining original source
sequence groups and all five proposal rows jointly. Its old coordinates must not
be reused after V4.1 retokenization. `quantization/draft_anchors.py` reproduces
that selection algorithm with V4.1 eligibility (history ending at p >= 1 with
known token p+1), returns grouped coordinates and an explicit coordinate digest,
and refuses to silently resize a corpus too small for the requested sample.
The known-token choice remains the already documented teacher-forced corpus
policy, not a claim to reproduce the reference's sampled token generation.

The selection is fixed before dSpark routing, never enlarged after inspecting
coverage. The adapter must process all selected anchors of each source record
jointly, reuse projected main features, and preserve their individual causal
128-row histories. The existing single-anchor diagnostic adapter does not yet
satisfy that efficient joint production contract; integration must implement and
qualify it before a dSpark production run. Tests cover stratification, coordinate
bounds, stable hashing, full selection, lock exclusion, and recovery after block
commit/before retirement. Coordinator scheduling tests mock the block operation;
they are not full-corpus integration evidence.
All 35 component tests pass (`reports/component-tests-coordinator-anchors.log`).

### Joint dSpark prefix implementation — numerical qualification pending

The fork now has `V41DSparkInput.prepare_joint`: one source prompt, sorted
distinct selected positions, one projected main prefix, and joint five-row draft
groups. Native dSpark attention accepts explicit `anchor_positions`, projects
each shared main KV row once per block, and gathers per-anchor ring slots in
reference order. Early nonexistent slots are masked; RoPE uses each anchor's
absolute draft positions. The existing single-anchor path remains available.

All 37 component tests pass (`reports/component-tests-joint-dspark-ring.log`).
New tests verify prefix slicing, known/noise tokens, coordinate rejection, and
literal reference-style cache writes across the 128-row wrap boundary; changing
future main KV rows cannot affect earlier anchor windows.

`probe_joint_dspark.py` executes all three real-weight blocks with positions
1,7,127,128,256, retaining just one [1,257,5120] projected prefix for the five
anchors. Joint outputs are finite and repeatable, but **not yet qualified**:
even with TF32 disabled, joint versus separate-anchor hidden maximum differences
are 0.491943359375, 6.4677734375 and 35.0 across stages 0/1/2. Evidence is
`reports/joint-dspark-smoke-fp32.log`. Earlier failed config-key lookup and
TF32-enabled comparison logs are retained separately. These differences must
be localized against an independently adapted checkpoint-reference joint batch;
repeatability is not sufficient proof of correctness, and no production draft
Hessians should use this implementation yet. Batch-shape-dependent numerical
behavior is possible, but has not been established as the explanation.

This fork change affects native calibration modules, not the EXL3 search code.
The four running worker images still bind their previously qualified source
identities; no image was silently changed underneath a running worker.

### Joint dSpark checkpoint-reference gate passed

The joint-versus-single diagnostic above is now resolved for the tested cases.
`quantization/validate_joint_dspark.py` independently adapts only the checkpoint
attention's scalar-position/cache interface to per-anchor positions: main-cache
population uses literal sequential ring writes, rotary uses the checkpoint's
in-place complex routine per anchor, and projections, normalization, sparse
attention, output einsum, mHC and experts remain checkpoint-reference operations.
Main input projection and embeddings are also computed through the checkpoint's
`DSparkBlock.forward_embed`, not taken from the candidate as the expected answer.

Exact parity passes through all three real-weight stages for:
- five joint anchors at 1,7,127,128,256 on RTX0
  (`reports/joint-dspark-reference.log`, core gate before input-comparator addition);
- 231 consecutive anchors on RTX1, including exact reference input preparation
  (`reports/joint-dspark-reference-231.log`);
- anchors at 1,127,128,256,511,1024 on RTX0, including input preparation and
  multiple cache wraps (`reports/joint-dspark-reference-long.log`).

Every reported attention, hidden and pre-mix maximum difference is zero. No
candidate implementation change was required to pass these gates. This supports
using the fixed joint batching topology; separately issued anchors are not a
bitwise-equivalent substitute. The earlier divergent single-anchor results remain
recorded, not relabeled as parity. These are synthetic target features and real
weights, not final-corpus quality evidence. Full-corpus handoff and two-RTX
orchestration remain to connect before production starts.

### Durable main-to-dSpark handoff and combined namespace sequence

`quantization/draft_inputs.py` binds the completed mixed main-model frontier to
the original committed corpus and deterministic anchor selection. It creates one
joint draft input per selected source record, with two independently owned RTX
adapters and at most two preparations pending. The worker threads read verified
main checkpoints and save owned draft states; only the coordinator writes SQLite.
Each published draft input depends on its original main output and the frozen
handoff inventory. Partial handoffs resume only missing records; changed counts,
selection or main outputs reject recovery. No main-output payloads are retired by
this handoff, so the final target features remain available for validation.

`coordinator.quantize_namespaces` now connects initial main preparation, all main
blocks, the joint draft handoff and all draft blocks under one exclusive run lock.
Adapter factories are lazy, allowing completed-input resumes to skip allocation
and avoid attempting to read already-retired initial payloads. Main mapped-table
handles close after preparation. The final marker explicitly says
`namespaces-quantized-export-pending`; it is not a claim that export, upload or
the user's entire objective is complete.

Tests inject an interruption after one durable draft input, verify joint groups
and skip-on-resume behavior, reject changed selections, and confirm that a
completed-input resume never recreates adapters. These use small CPU adapters or
mocked namespace execution, not full-corpus integration. Production CLI/runtime
assembly, parallel two-RTX block activation execution, export/validation and the
detached one-shot end-to-end launcher remain required.
All 39 component tests pass (`reports/component-tests-coordinator-handoff.log`).

### Two-RTX activation replay and guarded native kernel dispatch

`Wavefront` now supports an independently loaded second block on the other RTX.
Routed activation preparation and output propagation use bounded two-batch
windows; immutable original ordinals determine device assignment even after a
partial resume. Journal reads/publications stay on the coordinator thread.
The activation-device tuple is part of the block input inventory, so changing
single/dual-device topology cannot silently reuse old block work. After search,
the replica loads every selected packed projection before output propagation.
`run_namespace(..., replica_device="cuda:1")` connects replica lifetime to each
block; both copies are released at the boundary. Hessian accumulation itself
still occurs on the driver's GPU over committed routed batches.

The first real dual-RTX probe failed in TileLang/TVM imported-module lookup
under simultaneous host dispatch (`reports/parallel-activation-probe.log`).
`quantization/native_kernels.py` guards kernel factory creation and callable
dispatch with a shared reentrant host lock. It does not synchronize CUDA devices;
GPU launches remain asynchronous. The coordinator supplies this guarded module
to both native block loads and input-adapter factories. Factories now accept the
kernel module as their argument, ensuring they do not accidentally retain an
unguarded reference. This serializes host registry access, not whole GPU forwards.

The guarded real-weight diagnostic passes on four joint dSpark batches across
both RTX devices: routed hidden/logits/weights/indices and propagated hidden/carry
exactly match serial RTX0 execution. Evidence:
`reports/parallel-activation-probe-guarded.log`; measured publication passes were
0.494s routed and 0.187s output for these small cached diagnostic batches, not a
production throughput estimate. The probe uses native weights; selected mixed
replica propagation and full-corpus performance remain integration gates.
No production quantization job has started.
All 39 component tests pass (`reports/component-tests-parallel-guarded.log`).

### Mixed-weight replica integration and full source hashing

The two-RTX probe now accepts `--mixed-fixture` to read the original synthetic
mtp0 phase-search results through a read-only SQLite connection. It verifies
each old packed payload/provenance, installs the 384 selected projections into
the primary block, then exercises the actual `Wavefront.process` replica restore,
parallel output propagation, block commitment and completed-block reload. The
fixture is never mutated or relabeled as calibration from the new probe inputs.

This integration passes on four joint draft input batches: mixed outputs on
both RTX GPUs exactly match serial replay using the selected primary weights,
and completed reload is unchanged. Evidence:
`reports/parallel-mixed-replica-probe.log`, event `mixed_replica_exact` with
384 selected projections. No new search was performed in this test; packed
candidates came from the earlier real quantizer diagnostic, not production data.

`quantization/attest_source.py` hashes all indexed checkpoint shards plus all
checkpoint-provided non-weight assets/reference files with four bounded readers.
It rejects unexpected weight shards, file mutation while hashing and mismatches
against 64-character Hugging Face blob addresses. The completed manifest is
published atomically outside the source checkpoint. Tests exercise full inventory,
blob mismatch, unexpected-shard rejection and output path safety. All 40 component
tests pass (`reports/component-tests-source-attestation.log`).

A full source hash run completed successfully in `reports/source-attestation.log`,
publishing `<RUN_ROOT>/source-attestation.json`: 48 shards, 96,085 indexed tensors,
88 files and 510,313,353,565 total bytes. Manifest SHA-256 is
`cde39e3b2392c5b16cd51f092b40f5081cc12287563ef15a4cd7270bb3706080`.
The hash process exited normally. Production
CLI/runtime assembly, final export/upload and the complete one-shot launcher are
still unfinished. No production quantization has started.

### Export layout and large-file integrity policy (user clarification)

The deliverable is a standard Hugging Face sharded safetensors checkpoint with
the normal `model.safetensors.index.json` tensor-to-file mapping, not a custom
PLE container. Keep each PLE's table and associated scale tensors in its own
dedicated group of shard files. Never mix either PLE group with ordinary model
weights or with the other PLE. A group may contain multiple complete safetensors
files; do not require one file per PLE. This permits materializing another model
by hard-linking unchanged PLE files as a group, without copying their payloads.
Hard-linked files must remain immutable; replacements use new files and an
updated index, never in-place writes to shared inodes.

Standard HF sharding assigns each complete tensor to one file; it does not
split a single tensor across files. Preserve tensor names and shapes and allow
an individual large tensor to exceed the usual target shard size. Do not invent
chunked tensor keys or a custom reconstruction format just to cap file sizes.
The model's V4.1/EXL3 loader compatibility remains a separate export validation
gate: a valid safetensors container alone does not establish model compatibility.

Avoid routine hashing of huge source, PLE, or final model shards. The full source
attestation above has already completed and must not be repeated on ordinary
startup/resume. Reuse that identity with file metadata, bounded header/index and
size checks; these are structural checks, not claims of cryptographic payload
verification. Reserve checksums for small manifests/recovery artifacts where
useful, or an explicit corruption investigation. Account separately for any
unavoidable transfer-client integrity requirements; do not add redundant passes.

`stream_shard.py` provides bounded raw-byte safetensors repacking without tensor
decoding, dtype conversion, whole-payload mapping or SHA calculation. It publishes
files atomically without replacement. This is an export primitive, not yet the
complete exporter or a completed HF model checkpoint.

`export_layout.py` now plans deterministic standard HF file inventories and an
index from final tensor descriptors, with independent filename numbering for
ordinary weights and each PLE. It requires both PLE table/scale pairs, rejects
unexpected PLE names, and keeps oversized tensors whole. A header-only dry run
over all 96,085 source tensors passes (`reports/export-layout-source-dry-run.json`).
At the 5 GB target, each PLE gets two files: its roughly 98.3 GB table and its
roughly 3.07 GB scale tensor. The dry-run ordinary-weight layout is not a final
quantized inventory and no large output files were created.

The fork's mapped PLE reader and V4.1 input adapter now accept independently
indexed weight/scale files, preserving the original shared-file path. Both
mappings support bounded owned gathers, prefetch and page reclamation, and are
closed together. Tests cover split-file gathers/repeats/release/close plus
standard safetensors shard round trips and PLE filename stability when ordinary
weights change. Final packed-weight export, model-loader integration, production
runtime assembly and the detached end-to-end launcher remain unfinished.

### Selected-payload export inventory

`export_inventory.py` now collects selected candidates only after all 43 block
completion records exist. It checks complete phase inventories, canonical
candidate paths, all 47,232 routed projections and the exact 11,808 K4 upgrades
with the prescribed per-block projection quotas. Source headers/index must agree;
the complete source routed inventory must consist exactly of those weight/scale
pairs. Only those pairs are replaced by source-native projection names with
`.trellis`, `.suh`, `.svh`, `.mcg` suffixes; other tensor names and raw payloads
remain unchanged. This source-native inventory still needs the final V4.1 model
loader/config contract and is not advertised as a loadable finished checkpoint.

Selected journal candidates are verified as recovery artifacts, then their typed
metadata is inspected for buffer offsets, geometry and provenance without loading
or reconstructing weight tensors. Huge source files are not hashed. All 45
component tests pass (`reports/component-tests-export-inventory.log`). A separate
read-only header check passes for all 480 actual K3/K4 candidate files from the
earlier mtp0 diagnostic (`reports/export-inventory-fixture-headers.json`); that
fixture is unchanged and is not production calibration. Final inventory writing,
loader/config integration and the production launcher remain pending.

### Resumable weight-shard writer

`write_export.py` connects the final tensor inventory to bounded repacking and
the standard HF index. It freezes inventory/layout/run identity plus source
device/inode/size/mtime metadata in a separate state directory before publishing
weights, under an exclusive export lock. Existing state requires explicit resume;
identity changes, unknown output files, structural corruption and truncation
stop without replacing published files. Shards are checked by complete tensor
names, dtypes, shapes, spans and file size; no huge-file payload hashes are added.
The index is published only after all shards pass and source metadata is unchanged.
Same-size payload corruption is explicitly outside this structural verification.

If a planned shard exactly matches a complete source file with unchanged tensor
names, the writer hard-links the immutable source inode (same filesystem); otherwise
it streams raw bytes, including an EXDEV fallback. This supports reusing isolated
PLE shards in later model materializations. It does not hard-link a mixed source
shard when only its PLE subset belongs in the output. Partial temporary files left
by an abrupt process death require inspection/recovery; they are never silently
accepted or deleted. Tests cover interruption after a shard, explicit resume,
no recopy on completed resume, changed identity rejection, truncation rejection,
ordinary safetensors reads via the generated index, and actual inode reuse.

The writer's terminal status is deliberately
`weights-index-complete-model-validation-pending`. It does not yet write the final
model config/tokenizer, establish V4.1/EXL3 model-loader compatibility, upload to
HF, or launch production. No production quantization or large export has started.

### Indexed EXL3 reference reload

The fork's `V41Source.packed_projection` now reads source-native routed EXL3
buffers from ordinary indexed safetensors shards. It validates main/draft
architecture bounds, exact four-buffer coverage (rejecting mixed source weight
or scale payloads), K3/K4 geometry/dtypes and finite scales. Routed buffer suffixes
are indexed once at source initialization rather than scanning the full checkpoint
index for every expert. `decoded` recognizes these projections and reconstructs
the serialized EXL3 weights in source `[out, in]` BF16 layout. Existing native
attention/shared paths and mapped PLE loading are unchanged. This enables the
existing block loader's dense reference replay, not yet public GPTQModel full-model
loading or a qualified fused serving path.

`probe_export_reload.py` reads the old mtp0 fixture journal in read-only mode,
verifies six small candidate recovery artifacts (w1/w3/w2 at K3 and K4), repacks
them into ordinary named/indexed safetensors files in temporary directories and
reloads through `V41Source`. Every packed tensor and BF16 reconstructed weight
matches exactly (`reports/export-reload-probe.log`). The temporary test exports
are cleaned up; the fixture is unchanged. No large model copy/hash pass was made.
Complete model config/tokenizer packaging, public loader/full-model validation,
production runtime and the end-to-end launcher remain unfinished.

### Export configuration assembly

`export_config.py` builds metadata using the fork's actual `EXL3Config`
serializer. The embedded discovery declaration contains only `quant_method`,
`format`, `checkpoint_format` and integer base `bits=3`; the routed average
`13/4` is explicit recipe metadata, not an invalid fractional per-matrix width.
The standalone `quantize_config.json` holds storage descriptors and actual K3/K4
widths for all 47,232 routed projections, checked against the exact recipe and
buffer geometry. The original source FP8/FP4 declaration is retained under
`meta.ds41rt.native_quantization_config`; it is not incorrectly left as the active
declaration for the newly EXL3 routed weights. Architecture fields remain copied
unchanged. Source-native naming and the requirement for loader validation are
explicit; this metadata does not itself confer compatibility on an unmodified
Transformers or GPTQModel loader.

The shard writer now freezes both generated config objects into its external
resume plan and publishes them without overwrite after the weights/index pass.
Its completion status remains model-validation-pending. Tests exercise complete
main+draft storage coverage, exact K4 count, source config preservation, wrong-tier
geometry rejection and metadata publication/resume. Tokenizer/assets packaging,
full loader validation, production runtime and detached launcher remain pending.
