#!/usr/bin/env python3
"""Single source of truth for the EXL3 loader tier-family rules shared by tools.

Extracted verbatim from the accepted helpers in `qualify_v41_exl3_aot.py`
(`expected_decoder_family` / `checkpoint_global_family`) so that the numeric
oracle and the offline tile benchmark consult one copy of the rule instead of
paraphrasing it per tool.  The rule itself mirrors `decoder_family` in
`rust/crates/ds41rt-loader/src/v41_exl3.rs`, and
`python/tests/test_qualify_v41_exl3_aot_tiers.py` reads that Rust source and
fails if the loader rule moves underneath us.

Pure standard library: JSON + regex only, no tensors, no CUDA, no policy.
"""
from __future__ import annotations

import json
import re


def expected_decoder_family(observed):
    """The serving family a checkpoint's widths imply, mirroring the Rust loader.

    `decoder_family` in `rust/crates/ds41rt-loader/src/v41_exl3.rs` collects the
    distinct projection widths, requires every one of them to be in K2..K5, and for
    a single width retains an empty *adjacent* tier: `bit + 1`, except K5 which
    pairs down to K4. A uniform-K4 checkpoint therefore belongs to `[4, 5]`, not to
    `[3, 4]`; qualifying it as `[3, 4]` would compare the wrong quantization.
    """
    if any(isinstance(width, bool) or not isinstance(width, int) for width in observed):
        raise ValueError(f'trellis widths must be integers, got {sorted(map(repr, observed))}')
    bits = set(observed)
    if not bits:
        raise ValueError('checkpoint contributed no trellis widths to qualify')
    if invalid := sorted(bits - {2, 3, 4, 5}):
        raise ValueError(f'EXL3 decoder family must contain K2..K5 projections, got {invalid}')
    if len(bits) == 1:
        bit = next(iter(bits))
        bits.add(4 if bit == 5 else bit + 1)
    return sorted(bits)


def checkpoint_global_family(snapshot):
    """The serving family the loader would build for the WHOLE checkpoint.

    `decoder_family` in the Rust loader runs over every projection of every layer,
    target and mtp alike, reading `bits_per_weight` from this same manifest; the
    oracle otherwise only ever sees a handful of sampled experts, and a narrow
    sample can legitimise the wrong family. Pure JSON: no tensors, no CUDA.

    Returns `(family, projection_count, uniform, histogram)`. `projection_count` is 0
    and `histogram` empty for a raw single-width publication, which declares one
    `bits` value instead of a per-tensor table; `uniform` is the checkpoint-wide
    single-width state, which is what the coverage gate may lean on.
    """
    storage = _staged_tensor_storage(snapshot)
    if storage is not None:
        widths = [entry['bits_per_weight'] for entry in storage.values()]
        family = expected_decoder_family(widths)
        histogram = {str(width): widths.count(width) for width in sorted(set(widths))}
        return family, len(widths), len(set(widths)) == 1, histogram
    config = snapshot / 'config.json'
    quant = (json.loads(config.read_text()).get('quantization_config')
             if config.exists() else None)
    if not isinstance(quant, dict):
        raise ValueError(
            f'{snapshot} has neither tensor_storage nor a quantization_config object; '
            f'refusing to qualify a family inferred from sampled experts alone')
    # Same gates as parse_raw_publication in the Rust loader.
    for key, wanted in (('quant_method', 'exl3'), ('codebook', 'mcg'),
                        ('mtp_experts', 'source')):
        if quant.get(key) != wanted:
            raise ValueError(
                f'raw EXL3 publication requires {key}={wanted}, read {quant.get(key)!r} '
                f'from {config.name}')
    bits = quant.get('bits')
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4, 5):
        raise ValueError(
            f'raw EXL3 publication requires integer bits in 2..=5, read {bits!r}')
    return expected_decoder_family([bits]), 0, True, {}


_PROJECTION_NAME = re.compile(r'^(layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w3|w2)$')


def _staged_tensor_storage(snapshot):
    """Validate and return a staged publication's per-tensor table, else None.

    A staged publication IS its table. There is deliberately no fallback to a
    top-level `bits`: that value describes the packaging default, not the widths the
    loader reads per projection, so trusting it when the table is absent or holed
    would qualify a family the loader never sees.
    """
    manifest = snapshot / 'quantize_config.json'
    if not manifest.exists():
        return None
    storage = json.loads(manifest.read_text()).get('tensor_storage')
    if not isinstance(storage, dict) or not storage:
        raise ValueError(
            f'{manifest} must carry a nonempty tensor_storage; refusing to infer '
            f'the serving family from top-level bits')
    holes = [name for name, entry in storage.items()
             if not isinstance(entry, dict) or entry.get('bits_per_weight') is None]
    if holes:
        raise ValueError(
            f'{len(holes)} tensor_storage entries declare no bits_per_weight, '
            f'for example {holes[:3]}')
    illegal = sorted({repr(entry['bits_per_weight']) for entry in storage.values()
                      if isinstance(entry['bits_per_weight'], bool)
                      or not isinstance(entry['bits_per_weight'], int)})
    if illegal:
        raise ValueError(f'non-integer width type(s) in tensor_storage: {illegal[:4]}')
    return storage
