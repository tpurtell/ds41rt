# Upstream-derived test coverage

Ports of upstream unit-test coverage onto ds41rt harnesses, plus the small
product fixes those tests exposed. Upstream sources (immutable):

| Repo | Commit |
|---|---|
| vllm-project/vllm | `64856080b05054fc62754f37234f7b483b753445` |
| sgl-project/sglang | `832ec39cc0324cb0e7823dc8385e27a30c356bdd` |
| ggml-org/llama.cpp | `6011c34ce6099646ccdf0d39a61c6e681477c178` |

## What was ported

| Area | Harness | Upstream sources | Count |
|---|---|---|---|
| OpenAI protocol errors, streaming/SSE, stop strings, tool calls | `ds41rt-api` (legacy router) | vllm entrypoints suites, llama.cpp server patterns | 55 |
| Protocol/streaming invariants on the **production native router** | `ds41rt-api` `upstream_native_v41.rs` | vllm non-object-body, llama.cpp SSE contract | 5 |
| Sampler param validation + logits-processor reference | `ds41rt-api` | vllm sample/, logits_processors/ | 35 |
| Spec-decode acceptance math + bookkeeping | `ds41rt-core` | vllm spec_decode/, sglang spec/ | 25 |
| Hostcache eviction / radix invariants | `ds41rt-hostcache` | vllm core prefix-cache, sglang radix tree | 10 |
| Transport fault injection | `ds41rt-transport` | sgl-router failover/timeout/drain | 14 |
| Admission control | `ds41rt-daemon` | vllm engine admission, sglang scheduler | 39 |
| Container adversarial invariants | `ds41rt-loader` | llama.cpp gguf-py, test-gguf | 11 |
| Sampler reference oracle (llama.cpp exact vectors) | `python/tests` | llama.cpp test-sampling.cpp | 52 |
| FP4/E4M3/FP8 pack-math oracle | `python/tests` | vllm nvfp4/per-token-group quant | 51 |
| Quant-config validation | `python/tests` | vllm quantization config args | 39 |
| JSON-schema→grammar semantics | `python/tests` | llama.cpp test-json-schema* | 58 (+7 skip) |

## Running

```bash
# Rust (CPU-only; no GPU or weights required)
cargo test --workspace --no-fail-fast

# Python
cd python && pip install -e '.[test]'   # pytest, numpy, xgrammar==0.2.6
python -m pytest tests/
```

## Coverage classes

Two classes are labeled in each file header and counted separately:

1. **Product regression coverage** — exercises ds41rt code; failures mean
   product regressions.
2. **Standalone reference oracles** — document upstream behavior with no
   ds41rt dependency (e.g. llama.cpp's exact sampler vectors). They are the
   comparison references for the deferred GPU-parity tests and cannot
   detect product regressions alone.

## Skips and known failures

- `test_upstream_json_schema_grammar.py` skips 7 invariants xgrammar 0.2.6
  cannot enforce (`pattern`, `minLength`, `maxLength`, allOf merging,
  `uniqueItems`, `not`); re-check on a submodule bump. The module
  `pytest.importorskip`s xgrammar so a clean env skips rather than errors.
- Six pre-existing daemon tests fail on the v3 line: the Python planner
  expects a `b12x` API (`dsa_indexer.SOURCE_LAYOUT_PAGED`) the v3-pinned
  sparkinfer (`3882b935`) does not provide. Pre-dates this change; tracked
  separately.

## Validation ledger (2026-09-16, Romeo dev image, CUDA 13)

| Rev | Rust workspace passed | failed | ignored |
|---|---|---|---|
| `1b62a76` (v3 port tip, pre-test-coverage) | 1581 | 6 | 83 |
| `8754acca` (this change) | 1777 | 6 | 83 |

Delta: +196 tests, zero new failures, zero new skips. The 6 failures are the
identical pre-existing `b12x` planner/skew set (`dsa_indexer.SOURCE_LAYOUT_PAGED`
vs sparkinfer `3882b935`), unchanged in name and count on both revisions.
Python (pg, this change): 454 passed, 7 documented skips.

## Product fixes included (each verified by the ported tests)

1. `api`: 400 error bodies bounded (JsonRejection, native-router serde
   errors, and ApiError messages — validly typed huge fields included).
2. `api`: stop-string selection by earliest *completion* position with
   list-order ties (vllm `check_stop_strings` semantics).
3. `loader`: safetensors tensor rank bound (policy max 32).
4. `daemon`: dual-RTX serving passes the shared `/v1/stats` handle through
   (v3 merge repair had dropped it).

## Deferred

Fleet/GPU-dependent ports (kernel parity, sampler↔native parity,
FP4 wire-format parity against checkpoint tensors, hostcache serve-path
validation) are planned against a discrete-event census of the upstream
suites; not part of this change.
