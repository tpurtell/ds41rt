# Single-card NVFP4 local expert ABI fix

## Root cause

The single-card placement used the W4A4 full-width kernel but retained two
W4A8 assumptions in `LocalExpertWave`:

1. Slot 0 received `RouterOutput.expert_input` (FP8 K32, 5,280 bytes/row), while
   NVFP4 requires `RouterOutput.input` (BF16, 10,240 bytes/row).
2. Slot 41 contains BF16 `[rows,6,5120]` routes for NVFP4, but the local reducer
   interpreted it as FP32 routes, reading twice the valid route extent.

These corrupt local FFN activations and therefore subsequent router logits and
weights. The protocol's finite-gate check exposes the corruption at remote
Spark dispatch. The router itself already consumes BF16 and writes IDs/weights;
neither its kernel nor the protocol validation needed modification. The working
TP2 path already handled BF16 routes explicitly.

## Change

Local activation selection and extent validation now use an explicit format,
checked against the selected native kernel metadata. NVFP4 uses BF16; MXFP4
and EXL3 retain FP8 K32. Mixed resident activation families are rejected.
Local reduction dispatches on the full output-kind enum rather than treating
all non-token outputs as FP32 routes.

The new fused BF16 local reducer sums six routes in FP32, rounds the routed sum
to BF16, adds optional BF16 shared output, then rounds to BF16. It adds no
scratch allocation or synchronization. Existing FP32 reduction and TP2 dispatch
remain unchanged. The new FFI symbol is optional for compatibility with older
libraries when NVFP4 local reduction is not requested.

## Verification

Branch: `work/v7-nvfp4-exl3`. Verified 2026-09-18 UTC.

- Stopped both coordinator containers and Spark services; verified idle RTX
  GPUs before `./wip.sh --slot v7q-a1 --role coordinator` (successful).
- `cargo test -p ds41rt-ffi`: 99 passed, 3 GPU/environment tests ignored.
- Focused daemon local activation test: passed. Host Python 3.14 needed
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for the existing PyO3 dependency.
- Container `ds41rt-ffi v41_experts::tests -- --include-ignored --test-threads=1`
  against rebuilt native library: all 5 passed, including existing BF16 compact
  and TP2 oracles and the new local BF16 oracle.
- New local GPU oracle under `compute-sanitizer --tool memcheck`: **0 errors**.
  Covers rows 1/3/16/80, nonuniform routes, FP32 accumulation, intermediate BF16
  rounding, optional shared input, exact shared/output alias, invalid rows,
  routed/output overlap rejection, output tail guards, and equivalence with the
  existing FP32-route reducer for representable inputs.

### End-to-end

Checkpoint:
`nvidia/DeepSeek-V4.1-Flash-NVFP4@3431dde3247c13b5957f682b1e3c6fcae2566079`.

Started coordinator with `start-coordinator 1 --dspark`, then Spark experts with
`start-experts 4`. Startup reported:

- RTX local layers: **4**; resident bytes: **30,576,500,736**.
- Layers 0–3: `rtx_local_shared1`.
- Layers 4–39: `spark_tp4_shared1`.
- Native V4.1 target API ready on port 8000.

Request: `What is 6*7? Reply with just the number.`, temperature 0, max_tokens 64.
Actual API response:

```json
{"id":"chatcmpl-0b0d77ce-77a9-4a7d-b36c-91cdc3893bc7","object":"chat.completion","created":1789754478,"model":"deepseek-ai/DeepSeek-V4.1-Flash","system_fingerprint":"ds41rt-native-fp4-kv-dspark","choices":[{"index":0,"message":{"role":"assistant","content":"42","reasoning_content":"We need answer only number. 6*7=42. Must reply just number."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":43,"completion_tokens":21,"total_tokens":64,"prompt_tokens_details":{"cached_tokens":0},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":43}}
```

Four additional requests at concurrency 2 returned `42` (43 cached prompt
tokens), `81`, `Paris`, and the exact sequence 1 through 20, all with stop finish
reason. This exercises actual local kernel input/output wiring, Spark dispatch,
prefill, speculative decode, and cache reuse, beyond the focused helper tests.

The 1x placement is ready to enter the campaign: this correctness blocker is
removed. This is not a completed throughput, long-context, or soak qualification.
MXFP4/EXL3 and full dual-NVFP4 API campaigns were not rerun; their input/output
branches are preserved and the existing dual BF16 GPU oracle passed.
