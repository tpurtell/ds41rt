"""Deterministic, bounded model card included in the immutable export plan."""
import hashlib


def model_card_asset(source_revision):
    if len(source_revision) != 40 or any(char not in "0123456789abcdef" for char in source_revision):
        raise ValueError("model card requires an exact source commit")
    content = f"""---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - exl3
  - quantized
  - deepseek-v41
---

# DeepSeek-V4.1-EXL3-K3.25-v1

Routed-expert-only mixed EXL3 K3/K4 quantization of
`deepseek-ai/DeepSeek-V4.1-Flash`, source revision `{source_revision}`.
Built with the custom V4.1 support in [our GPTQModel fork](https://github.com/tpurtell/GPTQModel)
and the reproducible [ds41rt quantization workflow](https://github.com/tpurtell/ds41rt/tree/main/quantization).

## Quantization recipe

All routed gate, up and down projections are quantized: 40 main-model blocks
with 384 experts each, plus three dSpark blocks with 128 experts each.
The 47,232 projections comprise 35,424 K3 and 11,808 K4 projections.
K4 assignments rank base-K3 reconstruction error weighted by natural squared
router-gate mass, with gate:up:down allocation quotas of 3:5:8 per block.
The routed matrix average is 3.25 bits per weight; this is not the whole-model
storage rate and excludes scales and other packing overhead.

Calibration uses the unchanged GLM-5.3 EXL3 NEXT corpus: 1,441 original records,
1,056,269 tokens under this checkpoint's tokenizer. dSpark uses 327,680 fixed
stratified anchors (seed 20260809), grouped jointly by original prompt. Calibration
propagates the selected mixed weights layer by layer. Activation generation runs
on two RTX GPUs; trellis search uses those GPUs and four Spark workers.

## Storage and loading

Weights use standard safetensors files and a Hugging Face weight index, with
checkpoint-native tensor names. `quantize_config.json` records per-projection
EXL3 storage and preserves the source's native quantization configuration.
Non-routed tensors retain their source representation. Each PLE table and its
scales occupy an isolated shard group, separate from the other table and all
other tensors. A PLE group can span multiple files; tensors are not split.
These groups can be reused through hard links when constructing later variants.

Standard file formatting does **not** imply stock Transformers or an existing
inference engine can execute this mixed V4.1 checkpoint. Loading requires support
for V4.1, native non-routed weights, EXL3 routed projections, and dSpark as needed.
The bundled source reference inference code is preserved for architectural
reference and is not an EXL3 serving implementation.

## Validation and limitations

The publication workflow checks tensor inventories, packed-buffer geometry,
per-block bit quotas, shard/index/config consistency and PLE file isolation.
It does not perform a final full-model replay or retain calibration data for one.
Behavioral quality, normal prompts and tool calls remain unvalidated pending
inference-engine integration. No benchmark results or stock-loader compatibility
are claimed. Source-model capabilities described in `README.source.md` are not
validation results for this quantization. See `LICENSE` for the source license.
"""
    payload = content.encode("utf-8")
    return dict(content=content, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
