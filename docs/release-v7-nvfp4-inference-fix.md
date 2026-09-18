# NVFP4 end-to-end inference fix — September 19, 2026

## Result

Verified NVIDIA ModelOpt NVFP4 W4A4 inference with the provided `v7q-a1`
harness: 20 backbone layers on two RTX PRO 6000 GPUs, layers 20–39 on
ostrich/dodo/emu/kiwi, with dSpark enabled. The original first-request CUDA
status 1 is resolved. The daemon and all four experts remain serving this
checkpoint after qualification.

Checkpoint: `nvidia/DeepSeek-V4.1-Flash-NVFP4`, snapshot
`3431dde3247c13b5957f682b1e3c6fcae2566079`.

## Root cause and changes

1. **Policy sentinel confused with compiled launch grid.** The exporter
   recorded `policy_max_active_clusters=-1` from the Python policy wrapper
   as the compiled entry's final scalar. The public runtime resolves that
   policy and passes positive `mac` from `_get_dynamic_kernel`; the kernel
   uses it directly as its cooperative grid dimension. Export the returned
   positive count (188 on RTX), not the sentinel. The shared native guard
   remains unchanged. The temporary diagnostic appeared after that guard,
   which explains why failing requests printed no launch diagnostic.
2. **Deterministic output is BF16 routes, not token sums.** Export output
   kind 2 (`BF16 [rows,6,5120]`); retain all six routes in TP2 buffers/copies;
   use dedicated FP32-accumulating BF16-route compaction and two-rank
   reduction. The previous TP2 code also passed two null planes into the
   existing four-plane reducer, which correctly rejected them.
3. **Input representation mismatches.** TP2 now supplies the already
   broadcast normalized BF16 values, not the native FP8 wire. Remote Spark
   requests similarly download BF16 values and use the BF16 protocol tag
   for NVFP4. Native and EXL3 retain their existing FP8 paths.
4. **Binding and lifetime defects.** Scratch binding preserves external
   request/weight slots while supplying placeholders only for unused
   W4A8-only slots 26–33. Spark small/decode arenas query the selected
   family. Native weight sizing queries native metadata. Each of the four
   asynchronous scalar uploads has a distinct immutable pinned source
   range until stream completion. Cross-format rebinding is rejected.
5. Removed temporary launch diagnostics; retained the pre-existing fix
   excluding duplicate native FP8-quantizer definitions from NVFP4 TUs.

The original Python probe's W4A8 failure was independent: it left slots
26–33 null, used BF16 instead of the FP8-K32 wire, and did not use native
packed extents. Corrected baseline probes showed both families launching
successfully with positive grids and rejecting -1/0 before dispatch.

## Builds and deployed identity

Two successful full invocations:

```sh
./wip.sh --slot v7q-a1 --role both
```

Services were stopped and GPUs confirmed idle before each build. The final
build includes the remote sender correction. Final slot fingerprints:

- coordinator: `b43700dd86f6c4a4c5068345823b44737e14e6876cedb90d052b9c2bbbe86cbe`
- Spark: `ac207663a5350361dc2c62255bc8f7f8826794271855b9e6fe2401ee3fe81bb4`

SHA256 verified in the deployed dual coordinator:

- `libds41rt_native.so`: `599f728fad28cd1550ed4dbcd876a497a53d5fa310a96280d1220dd3b2e7601e`
- `ds41rt`: `6025238584464f5ea8231c84ef5ef54b95e7597ffae80f60d2a46245b3350223`

The second build regenerated identical NVFP4 kernel `.o` and per-variant
headers. Its aggregate manifest hash for the copied variants header changed
because the exporter inventories the previous copied header before CMake
refreshes it; kernel payload hashes did not change.

## API evidence

Launch commands:

```sh
export MODEL_REL="hub/models--nvidia--DeepSeek-V4.1-Flash-NVFP4/snapshots/3431dde3247c13b5957f682b1e3c6fcae2566079"
runs/v7q-a1/serve-diffbot.sh start-experts 20
COORD_CONTAINER=ds41rt-coordinator-wip-dual runs/v7q-a1/serve-diffbot.sh start-coordinator 2 --dspark
```

Coordinator: `native V4.1 target API ready` at
`2026-09-18T16:32:42.856931Z`, with `rtx_expert_layers=20`. All four expert
logs report `first_layer=20`, `layers=20`, capacity 4096.

The exact requested smoke:

```sh
curl -sS -m 200 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}'
```

Response (successful inference; eight tokens exhausted in reasoning):

```json
{"id":"chatcmpl-28b36752-e2a1-4e9b-8621-eed4d7408496","object":"chat.completion","created":1789749168,"model":"deepseek-ai/DeepSeek-V4.1-Flash","system_fingerprint":"ds41rt-native-fp4-kv-dspark","choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"We need answer user asks \"Say OK"},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":33,"completion_tokens":8,"total_tokens":41,"prompt_tokens_details":{"cached_tokens":0},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":33}}
```

Same request with `max_tokens=128`:

```json
{"id":"chatcmpl-6d7b4257-b008-4b89-95de-a08272ceb425","object":"chat.completion","created":1789749183,"model":"deepseek-ai/DeepSeek-V4.1-Flash","system_fingerprint":"ds41rt-native-fp4-kv-dspark","choices":[{"index":0,"message":{"role":"assistant","content":"OK","reasoning_content":"We need answer user asks \"Say OK.\" We should just say OK. Need comply. Final only."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":33,"completion_tokens":24,"total_tokens":57,"prompt_tokens_details":{"cached_tokens":33},"prompt_cache_hit_tokens":33,"prompt_cache_miss_tokens":0}}
```

Three simultaneous fresh requests also completed with `finish_reason=stop`:

| Request | Prompt tokens | Exact final content |
|---|---:|---|
| Sum of 19 and 23 | 44 | `42` |
| Return exactly BLUE | 37 | `BLUE` |
| 180 repeated context sentences, then request READY | 1,838 | `READY` |

## Regression and numerical qualification

- `python3 -m unittest discover -s python/tests -p test_v41_expert_launch_contract.py -v`:
  host fixture compiles the actual shared engine and verifies guards for
  both native/NVFP4 families, including all 44 pointer slots.
- `cargo test -p ds41rt-ffi --lib`: **99 passed, 2 GPU tests ignored**.
- `cargo test -p ds41rt-daemon v41_experts --bin ds41rt`:
  **10 passed, 22 GPU tests ignored**.
- `cargo test -p ds41rt-daemon expert_request_wire_format_tracks_nvfp4_checkpoint`:
  passes. Host daemon builds use repository Python 3.12 (`PYO3_PYTHON` and
  its library directory in `LD_LIBRARY_PATH`); system Python 3.14 is newer
  than the pinned PyO3 supports. Container Python is 3.12.
- Real GPU test `bf16_route_reducers_match_exact_nonzero_oracle --ignored`:
  passes on RTX, exact BF16 compaction/two-rank reduction across rows
  1/16/80/3/1 and varied route/rank/column values.
- `native/tests/v41_nvfp4_launch_selftest.py`: NVFP4 RTX and Spark rows
  **1/16/80**, W4A8 RTX rows **1/16**, all pass. Covers positive launch
  counts, rejected sentinels and bad extents, every null slot, scratch
  preservation, output-kind/shape, poisoned outputs, and three graph replays.
- W4A8 pre-/post-first-build manifest is byte-identical; all **17 generated
  artifact hashes unchanged**.
- `native/tests/v41_nvfp4_numerics_selftest.py`: NVFP4 RTX and Spark rows
  **1/16**, identical nonzero packed weights and scales through native and
  public b12x prepare/bind/run paths. Initial and mutated input/routing
  outputs are **exactly equal, max_abs=0**. RTX absmax changes 2.671875 →
  5.46875; Spark 1.3359375 → 2.734375. Native reducer is bitwise equal to an
  ordered FP32 Torch sum of the BF16 routes. Three graph replays pass.
  The public m1 oracle uses grouped deterministic routing (public direct
  plus deterministic is not supported); native m1 uses exported direct.
  Eager parity printed exact equality; graph parity was checked with
  rtol=.01/atol=.002 during this run. Future script runs default to zero
  tolerance; that tightening was not separately rerun.

Example GPU test commands inside the build containers (GPUs must be idle):

```sh
python /wip/source/native/tests/v41_nvfp4_launch_selftest.py \
  /wip/slots/v7q-a1/coordinator/workspace/.ds41rt-wip/libds41rt_native.so \
  /wip/build/coordinator/native/v41_nvfp4_rtx_tp2/v41_nvfp4_experts.json \
  ds41rt_v41_nvfp4_tp2_expert --rows 16
python /wip/source/native/tests/v41_nvfp4_numerics_selftest.py \
  /wip/slots/v7q-a1/coordinator/workspace/.ds41rt-wip/libds41rt_native.so \
  /wip/build/coordinator/native/v41_nvfp4_rtx_tp2/v41_nvfp4_experts.json \
  --rows 16 --sparkinfer /wip/source/third_party/sparkinfer
```

For Spark use host ostrich/container `ds41rt-spark-expert-wip`, role directory
`spark-expert`, manifest `/wip/build/expert/native/v41_nvfp4_spark/v41_nvfp4_experts.json`,
and prefix `ds41rt_v41_nvfp4_expert`.

Local diagnostic logs: `/tmp/nvfp4-build-final.log`,
`/tmp/v41_contract_rebuilt.log`, `/tmp/v41_numerics_rtx_grouped.log`,
`/tmp/v41_validation_spark.log`, `/tmp/v41_gpu_validation_commands.txt`.

## Scope limits

No remaining blocker for the requested dual-RTX/four-Spark inference path.
This is correctness qualification, not a performance or model-quality
campaign. A full official-checkpoint/EXL3 daemon serve was not repeated;
W4A8 kernel/regression tests passed and existing FP32/EXL3 arithmetic is
unchanged. Single-RTX local NVFP4 remains explicitly unsupported as before.
