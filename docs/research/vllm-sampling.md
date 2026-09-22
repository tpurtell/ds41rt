# vLLM Sampling-Parameter Semantics — Current `main`

**Scope / provenance.** All code claims were verified against the vLLM repository `main` branch, shallow-cloned at commit **`c723a831a81cb4ff89ea6d61b2d15109306ed1bc`** (2026-09-22). Where a claim comes from a specific file, the source is linked inline. Docs pages were fetched from `docs.vllm.ai/en/latest/` ("latest developer preview"). Inline links point at `main`; pin the commit for a frozen citation.

---

## 1. `SamplingParams` defaults

From the `SamplingParams` msgspec/pydantic dataclass in [`vllm/sampling_params.py`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) (lines 236–264); mirrored in the generated API reference [`docs.vllm.ai/en/latest/api/vllm/sampling_params/`](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/):

| Parameter | Default | Source docstring / note |
|---|---|---|
| `temperature` | **1.0** | "Controls the randomness… Zero means greedy sampling." |
| `top_p` | **1.0** | "Must be in (0, 1]. Set to 1 to consider all tokens." |
| `top_k` | **0** | "Set to 0 (or -1) to consider all tokens." |
| `min_p` | **0.0** | "Must be in [0, 1]. Set to 0 to disable this." |
| `seed` | **None** | "Random seed to use for the generation." |
| `repetition_penalty` | **1.0** | |
| `presence_penalty` | **0.0** | |
| `frequency_penalty` | **0.0** | |

Also relevant: module-level `_SAMPLING_EPS = 1e-5` and `_MAX_TEMP = 1e-2` ([`sampling_params.py` L27–28](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).

**`temperature=0` special-casing (greedy).** In `__post_init__`, if `self.temperature < _SAMPLING_EPS` (1e-5) vLLM force-resets the filters and marks the request greedy:

```python
if self.temperature < _SAMPLING_EPS:
    # Zero temperature means greedy sampling.
    self.top_p = 1.0
    self.top_k = 0
    self.min_p = 0.0
    self._verify_greedy_sampling()
```
([`sampling_params.py` L533–538](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)). `_verify_greedy_sampling` only enforces `n == 1`. `sampling_type` returns `SamplingType.GREEDY` iff `temperature < 1e-5` ([L781–787](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).

Separately, `0 < temperature < 1e-2` is clamped **up** to `1e-2` with a warning (to avoid NaNs) — [`sampling_params.py` L488–496](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).

**Important server caveat (not a `SamplingParams` default):** the OpenAI-compatible server applies the model repo's `generation_config.json` by default, which can override `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, and `max_new_tokens` (mapped to `max_tokens`); disable with `--generation-config vllm`. See the warning in [`docs/serving/online_serving/openai_compatible_server.md`](https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md) and `get_diff_sampling_param` in [`vllm/config/model.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py). The chat request's own fallback dict `_DEFAULT_SAMPLING_PARAMS` is `{repetition_penalty: 1.0, temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.0}` — [`vllm/entrypoints/openai/chat_completion/protocol.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py).

*Note:* `SamplingParams.max_tokens` defaults to 16 in the dataclass (not asked, but easy to trip on).

---

## 2. Accepted bounds / validation

All in `SamplingParams._verify_args` / `__post_init__` ([`sampling_params.py` L560–620](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)):

| Parameter | Accepted range | Behavior outside range |
|---|---|---|
| `temperature` | finite, `0 <= t <= 2` | raises `VLLMValidationError` ("temperature must be in [0, 2]", "must be a finite number", "must be non-negative") |
| `top_p` | `0.0 < top_p <= 1.0` | raises ("top_p must be in (0, 1]") |
| `top_k` | integer, `top_k >= -1` | raises if `top_k < -1` ("top_k must be 0 (disable), or at least 1"); raises if not an `int` |
| `min_p` | `0.0 <= min_p <= 1.0` | raises ("min_p must be in [0, 1]") |
| `repetition_penalty` | finite, `> 0.0` | raises ("must be a finite number" / "greater than zero") |
| `presence_penalty` | `-2.0 <= p <= 2.0` | raises |
| `frequency_penalty` | `-2.0 <= p <= 2.0` | raises |

On the OpenAI-compatible server these `VLLMValidationError`s surface as HTTP 400 client errors (the code is imported from `vllm.exceptions`).

**Non-finite values.** `temperature` and `repetition_penalty` are checked explicitly with `math.isfinite(...)`. For `top_p` and `min_p` there is no explicit `isfinite` call, but `NaN`/`±inf` are still rejected by the chained comparisons: `not 0.0 < nan <= 1.0` evaluates to `True` (all NaN comparisons are `False`), so the `VLLMValidationError` is raised. So non-finite `top_p`/`min_p` are rejected by the range check in code; this is a code-behavior inference, not a separately documented rule.

`min_p` is additionally **rejected with speculative decoding** and with **diffusion models** (`_validate_spec_decode`, `_validate_diffusion`, [`sampling_params.py`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).

---

## 3. `top_k` sentinels

- **Docstring:** "Controls the number of top tokens to consider. **Set to 0 (or -1) to consider all tokens.**" — [`sampling_params.py` L257–259](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).
- **Validation:** `# quietly accept -1 as disabled, but prefer 0`; `if self.top_k < -1: raise …` — [`sampling_params.py` L595–599](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py). So `0` and `-1` are both accepted and both mean "disabled"; anything `< -1` is an error.
- **Where the normalization actually happens (current V1 path):** `InputBatch.add_request` in [`vllm/v1/worker/gpu_input_batch.py` L409–414](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_input_batch.py):
  ```python
  top_k = sampling_params.top_k
  if 0 < top_k < self.vocab_size:
      self.top_k_reqs.add(req_id)
  else:
      top_k = self.vocab_size
  self.top_k_cpu[req_index] = top_k
  ```
  So `0`, `-1`, and any `top_k >= vocab_size` are rewritten to `vocab_size` and are **not** tracked as active top-k, which makes `no_top_k` true and passes `top_k=None` into the sampler (no filtering at all) — see `_make_sampling_metadata` in the same file.
- **`top_k = 1`:** the top-k mask keeps exactly the one highest-logit token — `apply_top_k_top_p_pytorch` in [`vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py):
  ```python
  top_k_mask = logits_sort.size(1) - k.to(torch.long)
  top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
  top_k_mask = logits_sort < top_k_mask
  logits_sort.masked_fill_(top_k_mask, -float("inf"))
  ```
  This yields a one-token (one-hot) distribution, i.e. effectively greedy. **However**, vLLM does not label `top_k=1` as "greedy": `sampling_type` is decided solely by `temperature` (`GREEDY` vs `RANDOM`/`RANDOM_SEED`), and `top_k=1` still travels the random-sampling path. The word "greedy" for `top_k=1` is **not stated in this source**; only the masking behavior is.

> ⚠️ **Premise correction.** There is **no `_get_top_k` function and no `-1 if top_k == 0` normalization anywhere in current vLLM `main`.** A whole-tree `git grep` at commit `c723a831` for `_get_top_k`, `top_k == 0`, and `-1 if top_k` returns no matches. The `-1` sentinel is now only *accepted* for backwards compatibility and the real normalization is the `vocab_size` rewrite shown above. Any plan that cites `_get_top_k` / `-1 if top_k == 0` would be citing a removed/older code path.

---

## 4. `min_p` semantics

- **Formula** — `MinPLogitsProcessor.apply` in [`vllm/v1/sample/logits_processor/builtin.py` L100–116](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/builtin.py):
  ```python
  probability_values = torch.nn.functional.softmax(logits, dim=-1)
  max_probabilities  = torch.amax(probability_values, dim=-1, keepdim=True)
  adjusted_min_p     = max_probabilities.mul_(self.min_p)
  invalid_token_mask = probability_values < adjusted_min_p
  logits.masked_fill_(invalid_token_mask, -float("inf"))
  ```
  So the threshold is **`min_p * max_probability`** (relative to the most likely token), and tokens whose probability is strictly below it are masked to `-inf`.
- **`min_p = 0.0`** → `adjusted_min_p = 0.0`, and since softmax probabilities are `> 0`, the strict `<` comparison masks nothing. Matches the docstring: "Set to 0 to disable this." ([`sampling_params.py` L260–263](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).
- **`min_p > 1`** → **rejected** by validation (`not 0.0 <= self.min_p <= 1.0` → `VLLMValidationError`), not silently clamped — [`sampling_params.py` L604–605](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).
- **Combining with top_p/top_k:** `min_p` is applied as an *argmax-invariant* logits processor, **before** top-k/top-p — see §5. `MinPLogitsProcessor.is_argmax_invariant()` returns `True` with the docstring **"Min-p never impacts greedy sampling."** ([`builtin.py` L47–49](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/builtin.py)). The default builtin processor list is `[MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor]` ([`vllm/v1/sample/logits_processor/__init__.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/__init__.py)).

---

## 5. Filter ordering

The authoritative ordering is the `Sampler` class docstring plus `Sampler.sample` in [`vllm/v1/sample/sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py):

```
7. Sample the next tokens. `sample` method performs the following steps:
   a) If not `all_random`, perform greedy sampling...
   b) Apply temperature.
   c) Apply logit processors which are argmax-invariant, by default the min_p processor.
   d) Apply top_k and/or top_p.
   e) Sample ...
```

Concretely:

1. deterministic/censoring processors first (non-argmax-invariant: min-tokens, logit-bias) — `apply_logits_processors`;
2. penalties (repetition/frequency/presence);
3. **temperature** (`apply_temperature`);
4. **argmax-invariant processors — `min_p`** (`for processor in sampling_metadata.logitsprocs.argmax_invariant`);
5. **`top_k`, then `top_p`** inside `TopKTopPSampler` → `apply_top_k_top_p` → `apply_top_k_top_p_pytorch`, where the top-k mask is applied first and the top-p mask second ([`topk_topp_sampler.py` L447–486](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py));
6. random draw.

**Always keeps at least one token / keeps the argmax:**
- top-p: after the cumulative mask, the code explicitly does `top_p_mask[:, -1] = False  # at least one` — because the sort is ascending, `[:, -1]` is the **highest-probability** token, so it is never masked ([`topk_topp_sampler.py` L481–485](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py)).
- top-k: keeps the top `k` tokens, so for `k >= 1` the argmax is retained.
- min-p: with `min_p <= 1`, the argmax token has `prob = max_prob >= max_prob * min_p`, so it is not masked (strict `<`); consistent with `is_argmax_invariant() == True`.

---

## 6. `seed`

- **Default when unset:** `seed = None` ([`sampling_params.py` L264](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)). `seed == -1` is normalized to `None` in `__post_init__` (`if self.seed == -1: self.seed = None`).
- **Per-request seeding:** a request is `SamplingType.RANDOM_SEED` iff `temperature >= 1e-5` **and** `seed is not None` ([`sampling_params.py` L781–787](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)). The per-request `torch.Generator` is created in [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py):
  ```python
  if (sampling_params and sampling_params.sampling_type == SamplingType.RANDOM_SEED):
      generator = torch.Generator(device=self.device)
      generator.manual_seed(sampling_params.seed)
  else:
      generator = None
  ```
  It is stored on `CachedRequestState.generator`, registered in `InputBatch.generators` (only for requests that have one), and consumed by `random_sample` in `topk_topp_sampler.py`, which resets the exponential noise for those rows with the per-request generator: "NOTE(woosuk): To batch-process the requests without their own seeds… Then, we overwrite the values for the requests that have their own seeds."
- **Determinism across runs/batching — vLLM's own answer:** *not guaranteed by default.* The Reproducibility doc states: "vLLM does not guarantee the reproducibility of the results by default, for the sake of performance." To get reproducible results you must either set `VLLM_ENABLE_V1_MULTIPROCESSING=0` (offline, deterministic scheduling) or enable **batch invariance** (outputs insensitive to scheduling/batch composition); **in online mode only batch invariance is available**, and reproducibility holds only "on the same hardware and the same vLLM version." Source: [`docs/usage/reproducibility.md`](https://github.com/vllm-project/vllm/blob/main/docs/usage/reproducibility.md) / [`docs.vllm.ai/en/latest/usage/reproducibility/`](https://docs.vllm.ai/en/latest/usage/reproducibility/), and [`docs/features/batch_invariance.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md) ("independent of the batch size or the order of requests").
- **Distinguish two different `seed`s.** The Reproducibility doc's global `seed` is the **engine/`LLM(seed=...)`** seed: "If a specific seed value is provided, the random states for `random`, `np.random`, and `torch.manual_seed` will be set accordingly," and "In V1, the `seed` parameter defaults to `0`." That is separate from the per-request `SamplingParams.seed` / OpenAI `seed` field above.
- **Backend caveat:** per-request generators are only honored on the native PyTorch sampling path. FlashInfer, ROCm/aiter, and the fused XPU sampler fall back to the native implementation when generators are present ("…does not support per-request generators. Falling back to PyTorch-native implementation.") — [`topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py). So even with a per-request seed, outputs need not match *across backends*.
- **OpenAI's own contract:** the official OpenAI OpenAPI spec describes `seed` as "a best effort to sample deterministically… **Determinism is not guaranteed**, and you should refer to the `system_fingerprint`…" — [`openai/openai-openapi` spec](https://github.com/openai/openai-openapi/blob/master/openapi.yaml). (`https://platform.openai.com/docs/api-reference/chat/create` returned HTTP 403 to this agent, so the spec is used as the primary OpenAI source.)

---

## 7. OpenAI-compatible server specifics

From [`vllm/entrypoints/openai/chat_completion/protocol.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py) and [`docs/serving/online_serving/openai_compatible_server.md`](https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md) (docs page: [`docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/`](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)):

- `ChatCompletionRequest` declares `top_k: int | None = None` and `min_p: float | None = None` inside the block marked `# --8<-- [start:chat-completion-sampling-params]`, i.e. the vLLM-specific sampling parameters that are **not** part of the OpenAI schema.
- The docs' tip: "vLLM supports some parameters that are not supported by OpenAI, `top_k` for example. You can pass these parameters to vLLM using the OpenAI client in the **`extra_body`** parameter of your requests, i.e. `extra_body={"top_k": 50}`." So on `/v1/chat/completions`, **`top_k` and `min_p` are passed via `extra_body`** (or merged directly into the raw JSON payload).
- **`min_p` is exposed** on the chat endpoint (`min_p: float | None = None`, consumed in `to_sampling_params`), alongside `top_k`; both are also in the Completions and Batch-Chat protocols' vLLM extension blocks.
- **`seed` is a first-class, standard OpenAI body field** here, not an `extra_body` extension: `seed: int | None = Field(None, ge=_INT64_MIN, le=_INT64_MAX)`, and `to_sampling_params` passes `seed=self.seed` straight through.
- **Server-side defaults:** request fields default to `None`, then fall back to `default_sampling_params` (the model's `generation_config.json` diff) and finally to `_DEFAULT_SAMPLING_PARAMS` (`temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0`). See `to_sampling_params` and `get_diff_sampling_param`.
- **Flags:** there is **no sampling-specific server flag** for `top_k`/`min_p`/`seed` — they work without any extra launch option. `--enable-auto-tool-choice` (with `--tool-call-parser` / `--chat-template`) is a **tool-calling** flag documented in [`docs/features/tool_calling.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md) ("mandatory Auto tool choice… enable the model to generate its own tool calls"); it has nothing to do with sampling parameters. The only launch flag that changes sampling defaults is `--generation-config vllm` (to ignore the model's `generation_config.json`).

---

## 8. Does `temperature=0` / greedy bypass `top_p`/`top_k`/`min_p`?

**Yes.**

1. At construction, greedy requests are normalized: `if self.temperature < _SAMPLING_EPS: self.top_p = 1.0; self.top_k = 0; self.min_p = 0.0` ([`sampling_params.py` L533–538](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).
2. In the sampler, if every request is greedy (`all_greedy`), it returns `greedy_sample(logits)` directly from step 7a — **before** temperature scaling, before the argmax-invariant (`min_p`) processors, and before `topk_topp_sampler` ([`vllm/v1/sample/sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py)):
   ```python
   if sampling_metadata.all_greedy:
       ...
       return greedy_sampled, processed_logprobs
   ```
3. In a **mixed** batch (some greedy, some random), the temperature/`min_p`/top-k/top-p path still executes, but the final selection is
   ```python
   sampled = torch.where(sampling_metadata.temperature < _SAMPLING_EPS,
                         greedy_sampled, random_sampled, out=greedy_sampled)
   ```
   so `top_p`/`top_k`/`min_p` cannot change the output for the temperature-0 rows. `apply_temperature` likewise guards `torch.where(temp < _SAMPLING_EPS, 1.0, temp)` to avoid divide-by-zero.

---

## Explicitly flagged / unresolved

1. **`_get_top_k` / `-1 if top_k == 0` do not exist in current vLLM `main`** (verified by whole-tree `git grep` at commit `c723a831`). The premise appears to reflect an older/pre-V1 path. Current normalization is `top_k → vocab_size` for `top_k == 0 || top_k == -1 || top_k >= vocab_size` in `gpu_input_batch.py`. Do not cite `_get_top_k` in the benchmark plan.
2. **`top_k = 1` = greedy** is *not stated in the vLLM sources*; it is a consequence of the top-k mask keeping exactly one token. vLLM classifies greedy solely by `temperature`.
3. **Non-finite `top_p`/`min_p` rejection** is code behavior from `not 0.0 < x <= 1.0` / `not 0.0 <= x <= 1.0` (NaN comparisons are `False`), not a separately documented statement.
4. **`https://platform.openai.com/docs/api-reference/chat/create` returned HTTP 403** to this agent; OpenAI's own `openai-openapi` spec was used instead for the `seed` contract.
5. The generated docs page [`docs.vllm.ai/en/latest/api/vllm/sampling_params/`](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/) was truncated during fetch; exact default values were taken from the raw source file.

---

## Sources (URLs actually fetched)

- [`vllm/sampling_params.py` (main)](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/sampling_params.py>
- [`docs.vllm.ai/en/latest/api/vllm/sampling_params/`](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
- [`vllm/v1/sample/sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/sampler.py>
- [`vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/ops/topk_topp_sampler.py>
- [`vllm/v1/sample/logits_processor/builtin.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/builtin.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/sample/logits_processor/builtin.py>
- [`vllm/v1/sample/logits_processor/__init__.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/logits_processor/__init__.py)
- [`vllm/v1/worker/gpu_input_batch.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_input_batch.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_input_batch.py>
- [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)
- [`vllm/v1/sample/rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)
- [`vllm/entrypoints/openai/chat_completion/protocol.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py) — raw: <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/openai/chat_completion/protocol.py>
- [`vllm/config/model.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)
- [`vllm/utils/torch_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/utils/torch_utils.py)
- [`docs/serving/online_serving/openai_compatible_server.md`](https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md) — docs: [`docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/`](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [`docs/usage/reproducibility.md`](https://github.com/vllm-project/vllm/blob/main/docs/usage/reproducibility.md) — docs: [`docs.vllm.ai/en/latest/usage/reproducibility/`](https://docs.vllm.ai/en/latest/usage/reproducibility/)
- [`docs/features/batch_invariance.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md)
- [`docs/features/tool_calling.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/tool_calling.md)
- [`openai/openai-openapi` spec (seed)](https://github.com/openai/openai-openapi/blob/master/openapi.yaml)
- [`docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/`](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/ops/topk_topp_sampler/) (fetched; page was nav-dominated)
- [`docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/sample/min_p/`](https://docs.vllm.ai/en/latest/api/vllm/v1/worker/gpu/sample/min_p/) and [`docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/`](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/) (surfaced in search; no unique claims used)
