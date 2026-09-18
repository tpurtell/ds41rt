"""CPU oracle for ds41rt FP4/FP8 pack-and-scale reference math (component C10).

ds41rt's KV/weights use FP4 E2M1 values with group-16 E4M3 scales, and FP8
for local 128-token windows (see repo README, "Compressed global KV uses FP4
E2M1 values with group-16 E4M3 scales. The 128-token sliding windows remain
FP8"). This file is the executable numpy-only reference those formats are
validated against later. GPU parity is deferred; this oracle stands alone.

Reference math ported from upstream:

- vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  (``kE2M1ToFloat_handle``, ``break_fp4_bytes``, ``cast_to_fp4``,
  ``ref_nvfp4_quant``, ``ref_nvfp4_quant_dequant``, ``dequantize_to_dtype``),
  exercised upstream by
  vllm/tests/kernels/quantization/test_nvfp4_emulation.py
  (``test_triton_nvfp4_quant_dequant`` pins global_scale in {0.5, 1.0, 0.001}
  and block_size 16; ``test_triton_dequantize_nvfp4`` pins dequant semantics).
- vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py
  (``_per_token_group_quant_fp8`` triton reference math), exercised upstream
  by vllm/tests/kernels/quantization/test_per_token_group_quant.py
  (``test_per_token_group_quant_fp8`` pins the group 64/128 tail-group scale/
  value preservation contract; ``test_per_token_group_quant_fp8_packed_all_zero``
  pins UE8M0 byte 0x5E for all-zero input; the packed-scale layout formula is
  pinned in ``test_per_token_group_quant_fp8_packed``).
- vllm/tests/quantization/test_turboquant.py (``TestLloydMax`` /
  ``TestCentroids``): pure quant-math invariants for the Lloyd-Max KV
  quantizer (sorted centroids, midpoint boundaries, symmetric within ~4 sigma
  of N(0, 1/d)). Config/preset tables and custom-op calls deliberately not
  ported.
- sglang/python/sglang/test/quant_ref_utils.py and cpu_test_utils.py: oracle
  style/tolerances (hand-written references, ``fp8_max = 448``,
  ``float4_e2m1 max = 6.0``).

Upstream expectation notes (ds41rt-documented behavior asserted instead):

- E2M1 NaN handling: OCP E2M1 / MXFP4 gives the 4-bit payload NO NaN and
  NO infinity encoding — all 16 nibbles are finite (verified against the
  AMD Composable Kernel ocp_e2m1_mxfp4 reference, mxf4_utils.hpp: "no need
  to check for data as it does not have NaN representation"). vLLM and
  sglang both pin the full finite table [0, 0.5, 1, 1.5, 2, 3, 4, 6] with
  0b111 -> 6.0. Per the OCP saturating-convert convention (AMD CK
  sat_convert_to_type<f4_t>), NaN input saturates to the max-magnitude
  nibble (0x7 / 0xF); in MXFP4 containers NaN is instead signaled via the
  E8M0 scale byte, which ds41rt's group-16 E4M3 scale slots do not use.
- Tail groups: upstream ``per_token_group_quant_fp8`` asserts
  ``hidden_dim % group_size == 0`` and never sees a partial group. ds41rt's
  format implies windows/tensors whose length is not necessarily a multiple
  of the group size, so the oracle's per-token-group quant implements
  upstream's group math with a tail extension: the partial last group keeps
  its own scale computed over only the elements present (never reads past
  the end of the tensor).
COVERAGE CLASS: standalone reference oracle. These cases document upstream
behavior with no ds41rt dependency; they cannot detect product regressions by
themselves. They are the comparison references for the deferred GPU parity
tests (docs/test-coverage/DEFERRED.md), counted separately from product
regression coverage (review MAJOR 4/6, 2026-09-15).
"""


from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Format constants (pinned by upstream sources)
# ---------------------------------------------------------------------------

# vllm nvfp4_emulation_utils.py: kE2M1ToFloat_handle /
# sglang quant_ref_utils.py: kE2M1ToFloat — both pin this exact vector.
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# scalar_types.float4_e2m1f.max() / sglang FLOAT4_E2M1_MAX.
E2M1_MAX = 6.0

# fp8 e4m3fn limits (get_fp8_min_max / sglang FLOAT8_E4M3_MAX).
E4M3_MAX = 448.0
E4M3_MIN = -448.0

# NVFP4 group size (block_size=16 pinned throughout the upstream tests).
NVFP4_BLOCK_SIZE = 16

# eps pinned in upstream kernel wrappers and packed-kernel launch.
FP8_QUANT_EPS = 1e-10


# ---------------------------------------------------------------------------
# FP4 E2M1 encode/decode (OFP/OCP convention: 0bS111 is NaN)
# ---------------------------------------------------------------------------


def decode_e2m1(nibbles: np.ndarray) -> np.ndarray:
    """Decode FP4 E2M1 nibbles to float64.

    Sign bit 0x08; magnitude bits 0x07 index the E2M1 table. OCP E2M1 has
    no NaN/Inf payload encoding: all 16 nibbles are finite, 0b111 -> 6.0
    (pinned by both upstream tables). NaN handling lives in encode_e2m1.
    """
    nibbles = np.asarray(nibbles, dtype=np.uint8) & 0xF
    sign = (nibbles >> 3).astype(bool)
    mag = nibbles & 0x7
    values = E2M1_MAGNITUDES[mag].astype(np.float64)
    return np.where(sign, -values, values)


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    """Encode float values to the nearest E2M1 nibble (round-to-nearest).

    OCP E2M1 has no NaN encoding, so per the OCP saturating-convert
    convention (AMD CK sat_convert_to_type<f4_t>) non-finite inputs map to
    the max-magnitude nibble 0x7/0xF, which decodes back to +/-6.0. Finite
    magnitudes above E2M1_MAX saturate to 0b110 (6.0), mirroring upstream's
    clamp(-6, 6) before cast_to_fp4.
    """
    values = np.asarray(values, dtype=np.float64)
    sign_bit = np.where(np.signbit(values), np.uint8(0x8), np.uint8(0))
    ax = np.abs(values)
    finite = np.isfinite(ax)
    ax = np.where(finite, np.minimum(ax, E2M1_MAX), np.nan)
    # Round with the file's canonical upstream-pinned boundaries
    # (round_to_e2m1), then exact table lookup. argmin-on-distance picked the
    # lower magnitude on halfway ties (0.75 -> 0.5), contradicting that
    # convention (0.75 -> 1.0) — review finding 2026-09-15.
    rounded = round_to_e2m1(ax)
    idx = np.zeros(rounded.shape, dtype=np.int64)
    for i, magnitude in enumerate(E2M1_MAGNITUDES):
        idx = np.where(rounded == magnitude, i, idx)
    nibble = idx.astype(np.uint8) | sign_bit
    return np.where(finite, nibble, np.uint8(0x7) | sign_bit)


def pack_fp4_pairs(values: np.ndarray) -> np.ndarray:
    """Pack two FP4 values per uint8, low nibble first.

    Matches upstream ``break_fp4_bytes`` ordering exactly: for each packed
    byte, the low nibble decodes to element 2i and the high nibble to
    element 2i+1 (combined = stack((low, high)) in the upstream source).
    """
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError("pack_fp4_pairs requires an even element count")
    nib = encode_e2f1_bytesafe(flat)
    packed = (nib[0::2] & 0xF) | ((nib[1::2] & 0xF) << 4)
    return packed.astype(np.uint8).reshape(values.shape[:-1] + (values.shape[-1] // 2,))


def encode_e2f1_bytesafe(values: np.ndarray) -> np.ndarray:
    """Alias of encode_e2m1 kept for readability at pack sites."""
    return encode_e2m1(values)


def break_fp4_bytes(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 pairs into FP4 values, low nibble first.

    numpy port of vllm nvfp4_emulation_utils.py: break_fp4_bytes /
    sglang quant_ref_utils.py: break_fp4_bytes.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed & 0xF0) >> 4
    interleaved = np.stack((low, high), axis=-1)
    return decode_e2m1(interleaved.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,)))


# ---------------------------------------------------------------------------
# E4M3 (float8_e4m3fn) scale codec: 1-4-3, bias 7, no inf, max 448
# ---------------------------------------------------------------------------


def e4m3_encode(values: np.ndarray) -> np.ndarray:
    """Round float32 values to e4m3fn bit patterns (round-to-nearest-even).

    Values are clamped to +/-448 before rounding (matching upstream's
    torch.clamp(scale, -448, 448) before the .to(float8_e4m3fn) cast).
    NaN encodes to 0x7F/0xFF per the e4m3fn convention.
    """
    x = np.clip(np.asarray(values, dtype=np.float32), E4M3_MIN, E4M3_MAX)
    input_is_nan = np.isnan(x)
    # NaN sign comes from the INPUT (clip/where can canonicalize it away);
    # e4m3fn NaN bytes are 0x7F/0xFF — 0x07 is a finite value (review finding
    # 2026-09-15: NaN encoded as 0x07 decoded back to 0.013671875).
    sign = np.where(np.signbit(np.asarray(values, dtype=np.float32)), np.uint8(0x80), np.uint8(0))
    ax = np.minimum(np.abs(x), np.float32(E4M3_MAX))
    bits = ax.view(np.uint32)
    ieee_exp = ((bits >> 23) & 0xFF).astype(np.int64)
    ieee_man = (bits & 0x7FFFFF).astype(np.int64)

    is_zero = ax == 0
    is_nan = input_is_nan
    # Normal e4m3 range: ax >= 2^-6 (ieee biased exp >= 121).
    is_normal = (~is_zero) & (ieee_exp >= 121)
    # Round-to-nearest-even collapse of the 23-bit mantissa to 3 bits.
    man_rne = (ieee_man + 0x7FFFF + ((ieee_man >> 20) & 1)) >> 20
    exp = ieee_exp - 120
    exp = np.where(man_rne == 8, exp + 1, exp)
    man_rne = np.where(man_rne == 8, 0, man_rne)
    normal_byte = (np.clip(exp, 0, 15) << 3) | man_rne

    # Subnormal e4m3: ax < 2^-6, quantum 2^-9; below half a quantum -> 0.
    sub_quantum = np.rint(ax.astype(np.float64) * 2.0**9)
    sub_quantum = np.where(sub_quantum >= 8, 8, sub_quantum)
    sub_byte = sub_quantum.astype(np.int64)

    byte = np.where(is_normal, normal_byte, sub_byte)
    byte = np.where(is_zero | is_nan, np.where(is_nan, 0x7F, 0), byte).astype(np.uint8)
    return byte | sign


def e4m3_decode(byte: np.ndarray) -> np.ndarray:
    """Decode e4m3fn bit patterns to float64.

    0x7F/0xFF decode to NaN per the e4m3fn convention (upstream scales are
    clamped to +/-448 = 0x7E/0xFE, so NaN patterns never occur on the wire
    for scales).
    """
    byte = np.asarray(byte, dtype=np.uint8)
    sign = (byte >> 7).astype(bool)
    exp = ((byte >> 3) & 0xF).astype(np.int64)
    man = (byte & 0x7).astype(np.float64)
    val = np.where(exp == 0, man * 2.0**-9, (8.0 + man) * 2.0 ** (exp - 10))
    val = np.where((exp == 0xF) & (man == 0x7), np.nan, val)
    return np.where(sign, -val, val)


def e4m3_roundtrip(values: np.ndarray) -> np.ndarray:
    """float32 -> e4m3fn -> float32, mirroring upstream's
    ``.to(torch.float8_e4m3fn).to(torch.float32)``."""
    return e4m3_decode(e4m3_encode(values)).astype(np.float64)


# ---------------------------------------------------------------------------
# NVFP4 group-16 quantize/dequantize
# (port of ref_nvfp4_quant / ref_nvfp4_quant_dequant /
#  dequantize_to_dtype, CPU paths)
# ---------------------------------------------------------------------------


def round_to_e2m1(x: np.ndarray) -> np.ndarray:
    """Round to the nearest E2M1 value with upstream tie boundaries.

    numpy port of vllm nvfp4_emulation_utils.py: cast_to_fp4 (python
    reference) — threshold boundaries are pinned there and in the
    ``_round_to_fp4`` triton kernel.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(x < 0.0, -1.0, 1.0)
    ax = np.abs(x)
    r = np.where(ax > 5.0, 6.0, 0.0)
    r = np.where((ax >= 3.5) & (ax <= 5.0), 4.0, r)
    r = np.where((ax > 2.5) & (ax < 3.5), 3.0, r)
    r = np.where((ax >= 1.75) & (ax <= 2.5), 2.0, r)
    r = np.where((ax > 1.25) & (ax < 1.75), 1.5, r)
    r = np.where((ax >= 0.75) & (ax <= 1.25), 1.0, r)
    r = np.where((ax > 0.25) & (ax < 0.75), 0.5, r)
    return r * sign


def nvfp4_quant(x: np.ndarray, global_scale: float, block_size: int = NVFP4_BLOCK_SIZE):
    """NVFP4 quantize: per-group E2M1 values + per-group E4M3 block scales.

    numpy port of vllm nvfp4_emulation_utils.py: ref_nvfp4_quant (the CPU
    reference half exercised by upstream
    tests/kernels/quantization/test_nvfp4_emulation.py::
    test_triton_nvfp4_quant_dequant).
    """
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    assert n % block_size == 0, (n, block_size)
    xg = x.reshape(m, n // block_size, block_size)
    vec_max = np.max(np.abs(xg), axis=-1, keepdims=True)
    scale = global_scale * (vec_max / E2M1_MAX)
    scale = np.clip(scale, E4M3_MIN, E4M3_MAX)
    scale = e4m3_roundtrip(scale)

    # get_reciprocal guards: 0 if the value is 0.
    denom = scale * (0.0 if global_scale == 0 else 1.0 / global_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        output_scale = np.where(denom == 0.0, 0.0, 1.0 / denom)

    scaled_x = np.clip(xg * output_scale, -E2M1_MAX, E2M1_MAX)
    fp4_values = round_to_e2m1(scaled_x)
    return fp4_values.reshape(m, n), scale.reshape(m, n // block_size)


def nvfp4_quant_dequant(x: np.ndarray, global_scale: float, block_size: int = NVFP4_BLOCK_SIZE) -> np.ndarray:
    """NVFP4 quantize-dequantize.

    numpy port of ref_nvfp4_quant_dequant's CPU path: dequant = fp4 * (scale /
    global_scale), i.e. x_fp4 * x_blockscale with x_blockscale = scale / gs.
    """
    x = np.asarray(x, dtype=np.float64)
    m, n = x.shape
    fp4_values, scale = nvfp4_quant(x, global_scale, block_size)
    blockscale = scale.reshape(m, n // block_size, 1) / global_scale
    return (fp4_values.reshape(m, n // block_size, block_size) * blockscale).reshape(m, n)


def nvfp4_dequantize_packed(packed: np.ndarray, scales_e4m3: np.ndarray, global_scale: float, block_size: int = NVFP4_BLOCK_SIZE) -> np.ndarray:
    """Dequantize packed FP4 bytes + per-group E4M3 scales.

    Wire convention for THIS oracle: the encoded block scale is the complete
    multiplicative block scale (what ``nvfp4_quant_dequant`` multiplies), i.e.
    callers encode ``scales / global_scale`` — ``global_scale`` is accepted and
    asserted against for API symmetry but NOT applied again here.

    (Upstream checkpoint layout stores block scales EXCLUDING the global scale
    as separate ``weight_scale``/``weight_scale_2`` tensors multiplied at
    dequant — dequantize_to_dtype's ``sf * global_scale``. The deferred GPU
    parity test must map that split onto this oracle's folded convention.)
    """
    packed = np.asarray(packed, dtype=np.uint8)
    m, packed_k = packed.shape
    k = packed_k * 2
    assert k % block_size == 0
    values = break_fp4_bytes(packed).reshape(m, k // block_size, block_size)
    sf = e4m3_decode(np.asarray(scales_e4m3, dtype=np.uint8))
    return (values * sf.reshape(m, k // block_size, 1)).reshape(m, k)


# ---------------------------------------------------------------------------
# Per-token-group FP8 quantize (port of _per_token_group_quant_fp8 triton
# reference math) with ds41rt tail-group semantics
# ---------------------------------------------------------------------------


def per_token_group_quant_fp8(x: np.ndarray, group_size: int, eps: float = FP8_QUANT_EPS, use_ue8m0: bool = False):
    """Per-token-group FP8 (e4m3fn) quantize with tail-group semantics.

    Group math ported from vllm fp8_utils.py: _per_token_group_quant_fp8:
        absmax = max(max|group|, eps)
        scale_raw = max(absmax / 448, eps)   # eps floor: packed-kernel contract
        y_s = exp2(ceil(log2(scale_raw))) if ue8m0 else absmax / 448
        y_q = clamp(y / y_s, -448, 448) cast to e4m3fn

    (The eps floor on the ue8m0 path is pinned by upstream
    test_per_token_group_quant_fp8_packed_all_zero: all-zero input yields
    2^-33, UE8M0 byte 0x5E. The plain Triton reference omits the floor and
    would give 2^-42; the packed contract is what upstream pins.)

    Tail semantics (ds41rt-documented; upstream asserts divisibility and
    never sees a partial group — see module docstring): the partial last
    group keeps its own scale computed over only the elements present and
    never reads past the end of the row.

    Returns (y_q float64-of-e4m3fn-values, y_s float64 scales (tokens, G)).
    """
    x = np.asarray(x, dtype=np.float64)
    tokens, n = x.shape
    n_groups = (n + group_size - 1) // group_size
    y_q = np.zeros_like(x)
    y_s = np.zeros((tokens, n_groups))
    for g in range(n_groups):
        start = g * group_size
        group = x[:, start : start + group_size]  # never reads past n
        absmax = np.maximum(np.max(np.abs(group), axis=-1), eps)
        scale_raw = absmax / E4M3_MAX
        if use_ue8m0:
            # Packed-kernel contract pinned by upstream
            # test_per_token_group_quant_fp8_packed_all_zero: the raw scale
            # is floored at eps (fmax(y_s, 1e-10)) before the power-of-two
            # round-up, so an all-zero group yields 2^-33 (byte 0x5E), not
            # 2^-42. For any group with absmax >= eps * 448 the floor is
            # inert and this equals the plain Triton reference math.
            scale_raw = np.maximum(scale_raw, FP8_QUANT_EPS)
            with np.errstate(divide="ignore"):
                y_s[:, g] = np.exp2(np.ceil(np.log2(scale_raw)))
        else:
            y_s[:, g] = scale_raw
        y_q[:, start : start + group_size] = e4m3_roundtrip(
            np.clip(group / y_s[:, g : g + 1], E4M3_MIN, E4M3_MAX)
        )
    return y_q, y_s


def pack_ue8m0_scale_exponents(y_s: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack UE8M0 scale exponents into the upstream DeepGEMM int32 layout.

    Layout pinned by upstream tests/kernels/quantization/
    test_per_token_group_quant.py: test_per_token_group_quant_fp8_packed:
    each int32 holds 4 little-endian exponent bytes; group g of row `row`
    lands at index ``pack_col * tma_aligned_mn + row`` with
    ``pack_col = g // 4``, ``pos = g % 4``; mn is padded up to a multiple of
    4 (tma_aligned_mn). Padding slots are zero.
    """
    y_s = np.asarray(y_s, dtype=np.float64)
    mn, groups_per_row = y_s.shape
    tma_aligned_mn = ((mn + 3) // 4) * 4
    k_num_packed = (groups_per_row + 3) // 4
    num_scale_elems = mn + (k_num_packed - 1) * tma_aligned_mn
    packed = np.zeros(num_scale_elems, dtype=np.uint32)
    exponents = (np.float32(y_s).view(np.uint32) >> 23) & 0xFF
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            idx = pack_col * tma_aligned_mn + row
            packed[idx] |= np.uint32(exponents[row, g]) << (pos * 8)
    return packed, tma_aligned_mn


# ---------------------------------------------------------------------------
# Lloyd-Max quantizer reference (turboquant KV quant math; pure numpy port
# of the math exercised by vllm/tests/quantization/test_turboquant.py::
# TestLloydMax / TestCentroids)
# ---------------------------------------------------------------------------


def _gaussian_pdf(x: np.ndarray, sigma2: float) -> np.ndarray:
    return np.exp(-(x**2) / (2 * sigma2)) / math.sqrt(2 * math.pi * sigma2)


def solve_lloyd_max(d: int, bits: int, n_trapz: int = 200, iters: int = 200, tol: float = 1e-10):
    """Lloyd-Max quantizer for N(0, 1/d): centroids + decision boundaries.

    numpy port of the math behind turboquant's solve_lloyd_max (trapezoid
    integration instead of scipy quad, matching the upstream test's own
    _trapz(n=200) reference contract).
    """
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)
    n_levels = 2**bits
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = np.array([lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)])
    for _ in range(iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate([[lo * 3], boundaries, [hi * 3]])
        new_centroids = np.empty_like(centroids)
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            xs = np.linspace(a, b, n_trapz)
            pdf = _gaussian_pdf(xs, sigma2)
            num = np.trapezoid(xs * pdf, xs)
            den = np.trapezoid(pdf, xs)
            new_centroids[i] = num / den if den > 1e-15 else centroids[i]
        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    # Boundaries are midpoints of the *final* centroids (mirrors upstream
    # TestLloydMax::test_boundaries_are_midpoints).
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


# ===========================================================================
# Tests
# ===========================================================================


def test_e2m1_magnitude_table_matches_upstream_pinned_vector():
    # Pinned identically by vllm nvfp4_emulation_utils.py
    # (kE2M1ToFloat_handle, used by test_nvfp4_emulation.py) and sglang
    # quant_ref_utils.py (kE2M1ToFloat).
    np.testing.assert_array_equal(
        E2M1_MAGNITUDES, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    )
    assert E2M1_MAX == 6.0


def test_e2m1_decode_all_16_nibbles_ocp_convention():
    # OCP E2M1 / MXFP4 has NO NaN or Inf payload encoding: all 16 nibbles
    # are finite, 0b111 -> 6.0 (AMD CK ocp_e2m1_mxfp4 reference; both
    # upstream tables pin the same 8 magnitudes). ds41rt documents plain
    # "FP4 E2M1 values", so the oracle asserts the full finite decode.
    nibbles = np.arange(16, dtype=np.uint8)
    decoded = decode_e2m1(nibbles)
    assert decoded[0] == 0.0
    assert decoded[6] == 4.0
    assert decoded[7] == 6.0
    assert decoded[8] == 0.0
    assert decoded[14] == -4.0
    assert decoded[15] == -6.0
    # Every finite entry matches the upstream pinned magnitude table with
    # its sign — the exact vector pinned by vllm/sglang.
    for nib in range(16):
        expected = E2M1_MAGNITUDES[nib & 0x7] * (-1.0 if nib & 0x8 else 1.0)
        assert decoded[nib] == expected, (nib, decoded[nib], expected)


def test_e2m1_encode_roundtrip_finite_grid():
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    neg = np.array([0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF], dtype=np.uint8)
    expected = np.arange(8, dtype=np.uint8)
    np.testing.assert_array_equal(encode_e2m1(grid), expected)
    np.testing.assert_array_equal(encode_e2m1(-grid), neg)
    # Round-trip through the wire format.
    np.testing.assert_array_equal(decode_e2m1(encode_e2m1(grid)), grid)
    np.testing.assert_array_equal(decode_e2m1(encode_e2m1(-grid)), -grid)


def test_e2m1_encode_nan_saturates_max_nibble_ocp_convention():
    # OCP E2M1 has no NaN encoding; per the saturating-convert convention
    # (AMD CK sat_convert_to_type<f4_t>) NaN/Inf input saturates to the
    # max-magnitude nibble, which decodes back to +/-6.0.
    out = encode_e2m1(np.array([np.nan, -np.nan, np.inf, -np.inf]))
    assert out[0] == 0x7
    assert out[1] == 0xF
    assert out[2] == 0x7
    assert out[3] == 0xF
    np.testing.assert_array_equal(decode_e2m1(out), np.array([6.0, -6.0, 6.0, -6.0]))


def test_pack_unpack_low_nibble_first_matches_upstream_break_fp4_bytes():
    # Upstream break_fp4_bytes interleaves (low, high): byte b's low nibble
    # is element 2i, high nibble element 2i+1 (pinned by
    # test_triton_dequantize_nvfp4's triton-vs-reference equality).
    values = np.array([[0.5, -1.0, 3.0, 6.0, -4.0, 0.0, 1.5, -2.0]])
    packed = pack_fp4_pairs(values)
    assert packed.dtype == np.uint8
    assert packed.shape == (1, 4)
    assert packed[0, 0] == 0xA1  # low=0b0001 (0.5), high=0b1010 (-1.0)
    unpacked = break_fp4_bytes(packed)
    np.testing.assert_array_equal(unpacked, values)


def test_round_to_e2m1_thresholds_pinned_by_upstream_cast_to_fp4():
    # Boundary thresholds pinned by vllm nvfp4_emulation_utils.py:
    # cast_to_fp4 / _round_to_fp4 (0.25/0.75/1.25/1.75/2.5/3.5/5.0 cuts).
    cases = [
        (0.0, 0.0),
        (0.25, 0.0),  # (0, 0.25] -> 0
        (0.26, 0.5),
        (0.75, 1.0),  # [0.75, 1.25] -> 1
        (1.25, 1.0),
        (1.26, 1.5),
        (1.75, 2.0),  # [1.75, 2.5] -> 2
        (2.5, 2.0),
        (2.51, 3.0),
        (3.5, 4.0),  # [3.5, 5.0] -> 4
        (5.0, 4.0),
        (5.01, 6.0),  # > 5 -> 6
        (100.0, 6.0),
    ]
    for value, expected in cases:
        assert round_to_e2m1(np.array([value]))[0] == expected, (value, expected)
        assert round_to_e2m1(np.array([-value]))[0] == -expected, (-value,)


def test_e4m3_codec_pinned_values():
    # e4m3fn: bias 7, 3 mantissa bits, no inf, max 448 (= 0x7E).
    assert e4m3_encode(np.array([448.0]))[0] == 0x7E
    assert e4m3_encode(np.array([-448.0]))[0] == 0xFE
    assert e4m3_encode(np.array([1.0]))[0] == 0x38
    assert e4m3_encode(np.array([0.0]))[0] == 0x00
    assert e4m3_encode(np.array([2.0**-9]))[0] == 0x01  # min subnormal
    assert e4m3_encode(np.array([2.0**-6]))[0] == 0x08  # min normal
    assert np.isnan(e4m3_decode(np.array([0x7F]))[0])
    assert np.isnan(e4m3_decode(np.array([0xFF]))[0])
    # Round-trip preserves the e4m3fn value set exactly.
    rng = np.random.default_rng(0)
    samples = np.concatenate(
        [rng.uniform(-448, 448, 4096), rng.normal(0, 1e-3, 4096), [448.0, -448.0, 0.0]]
    )
    rt = e4m3_decode(e4m3_encode(samples))
    assert np.all(np.isfinite(rt))
    np.testing.assert_array_equal(e4m3_encode(rt), e4m3_encode(samples))


def test_nvfp4_quant_matches_hand_computed_reference():
    # Group of 16, global_scale 1.0: scale = e4m3(max|x|/6).
    x = np.zeros((1, 32))
    x[0, :16] = 6.0  # absmax 6 -> scale e4m3(1.0) = 1.0
    x[0, 16:] = 3.0  # absmax 3 -> scale e4m3(0.5) = 0.5
    fp4_values, scales = nvfp4_quant(x, 1.0)
    np.testing.assert_array_equal(scales[0], np.array([1.0, 0.5]))
    np.testing.assert_array_equal(fp4_values[0, :16], np.full(16, 6.0))
    np.testing.assert_array_equal(fp4_values[0, 16:], np.full(16, 6.0))
    # dequant = fp4 * scale/gs: first group 6*1=6, second 6*0.5=3.
    dq = nvfp4_quant_dequant(x, 1.0)
    np.testing.assert_array_equal(dq[0, :16], np.full(16, 6.0))
    np.testing.assert_array_equal(dq[0, 16:], np.full(16, 3.0))


@pytest.mark.parametrize("global_scale", [0.5, 1.0, 0.001])
def test_nvfp4_quant_dequant_roundtrip_error_bounds(global_scale):
    # global_scale values pinned by upstream
    # test_nvfp4_emulation.py::test_triton_nvfp4_quant_dequant.
    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, 64)) * 2.0
    dq = nvfp4_quant_dequant(x, global_scale)
    xg = x.reshape(8, 4, 16)
    absmax = np.max(np.abs(xg), axis=-1)
    err = np.abs(dq.reshape(8, 4, 16) - xg)

    if global_scale < 0.5:
        # gs=0.001 is pinned by upstream ONLY for ref-vs-kernel self-consistency
        # (their emulation test asserts kernel == reference, never absolute
        # error): with this data every group's raw scale (gs*absmax/6) falls
        # below the e4m3 denormal floor, so absolute error bounds are not a
        # contract of the format in this regime. Assert the actual contract:
        # finite output, and exact-zero inputs decode to exact zero via the
        # get_reciprocal 0-guard.
        assert np.all(np.isfinite(dq))
        assert np.all(dq[x == 0.0] == 0.0)
        return

    # Calibrated regime (gs >= 0.5): per group, dequant error <= half the
    # local grid gap * (scale/gs); worst grid gap is 2 (between 4 and 6), so
    # err <= scale/gs, and scale = e4m3(gs*absmax/6) <= gs*absmax/6 * (1 + 2^-4)
    # (RNE, 3-bit mantissa). Bound with headroom.
    bound = (absmax / E2M1_MAX) * 1.2
    assert np.all(err <= bound[..., None] + 1e-12)
    # Relative error (worst element in each group vs the group absmax) is
    # bounded for non-trivial magnitudes.
    rel_max = err.max(axis=-1) / np.maximum(absmax, 1e-12)
    assert np.all(rel_max[absmax > 1e-3] <= 0.2)


def test_nvfp4_zero_group_guard_matches_upstream_get_reciprocal():
    # All-zero group: scale quantizes to e4m3(0) = 0; upstream's
    # get_reciprocal zero-guard makes output_scale 0 (not inf), so the
    # dequantized result is exactly 0.
    x = np.zeros((1, 32))
    x[0, 16:] = 2.0
    dq = nvfp4_quant_dequant(x, 1.0)
    np.testing.assert_array_equal(dq[0, :16], np.zeros(16))


def test_nvfp4_packed_dequantize_matches_direct_quant_dequant():
    # Full wire-format loop: quant -> encode nibbles -> pack bytes ->
    # dequantize packed bytes + E4M3 scales, must equal quant_dequant.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 64))
    global_scale = 0.5
    fp4_values, scales = nvfp4_quant(x, global_scale)
    packed = pack_fp4_pairs(fp4_values)
    # Wire block scale = the complete multiplicative block scale (gs folded
    # out of nvfp4_quant's gs-inclusive scale) — see nvfp4_dequantize_packed.
    scale_bytes = e4m3_encode(scales / global_scale)
    dq_packed = nvfp4_dequantize_packed(packed, scale_bytes, global_scale)
    dq_direct = nvfp4_quant_dequant(x, global_scale)
    np.testing.assert_allclose(dq_packed, dq_direct, atol=0, rtol=0)


@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("shape", [(4, 7168), (3, 384), (32, 512), (16, 512)])
def test_per_token_group_quant_fp8_group64_128_contract(shape, group_size, use_ue8m0):
    # Shapes/group sizes pinned by upstream
    # test_per_token_group_quant.py::test_per_token_group_quant_fp8 (group
    # 64/128 contract; x scaled by 8 with seed 42 there).
    tokens, hidden = shape
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape) * 8.0
    y_q, y_s = per_token_group_quant_fp8(x, group_size, use_ue8m0=use_ue8m0)
    assert y_q.shape == x.shape
    assert y_s.shape == (tokens, hidden // group_size)
    # Scales: absmax/448, optionally rounded up to a power of two.
    absmax = np.max(np.abs(x.reshape(tokens, hidden // group_size, group_size)), axis=-1)
    expected_raw = absmax / E4M3_MAX
    if use_ue8m0:
        expected = np.exp2(np.ceil(np.log2(expected_raw)))
    else:
        expected = expected_raw
    np.testing.assert_allclose(y_s, expected, rtol=1e-6, atol=0)
    # Quantized values are exactly e4m3fn-representable and within range.
    np.testing.assert_array_equal(e4m3_encode(y_q), e4m3_encode(y_q.astype(np.float32)))
    assert np.all(np.abs(y_q) <= E4M3_MAX)
    # Dequantization error: y_q is round-to-nearest-even e4m3fn. In the
    # normal range the half-ULP is at most |v| * 2^-4 (3-bit mantissa);
    # below the subnormal threshold (2^-9) the absolute quantum is 2^-9,
    # adding a y_s * 2^-10 absolute floor term.
    dq = y_q * np.repeat(y_s, group_size, axis=1)
    err = np.abs(dq - x)
    bound = np.maximum(np.abs(x) * 2.0**-4, np.repeat(y_s, group_size, axis=1) * 2.0**-10)
    assert np.all(err <= bound * 1.001 + 1e-12)


@pytest.mark.parametrize("use_ue8m0", [False, True])
def test_per_token_group_quant_fp8_tail_group_semantics(use_ue8m0):
    # ds41rt tail-group contract (upstream asserts divisibility and never
    # sees a partial group; see module docstring): hidden=200 with
    # group_size=128 -> one full group + a 72-element tail with its own
    # scale, computed only over the elements present.
    x = np.arange(200, dtype=np.float64).reshape(1, 200) * 0.1 - 10.0
    y_q, y_s = per_token_group_quant_fp8(x, 128, use_ue8m0=use_ue8m0)
    assert y_s.shape == (1, 2)
    tail_absmax = np.max(np.abs(x[0, 128:]))
    tail_raw = tail_absmax / E4M3_MAX
    expected_tail = np.exp2(np.ceil(np.log2(tail_raw))) if use_ue8m0 else tail_raw
    assert y_s[0, 1] == pytest.approx(expected_tail, rel=1e-12)
    # Tail scale must be computed only from tail elements: dropping the last
    # element changes the tail absmax, hence the scale.
    # Tail scale must be computed only from tail elements. Truncate the tail
    # enough to cross a UE8M0 power-of-two boundary (dropping a single
    # element can legitimately leave the rounded scale unchanged): removing
    # the last 29 elements drops the tail absmax from 9.9 to 7.0, moving
    # the ue8m0 scale from 2^-5 to 2^-6, and moves the raw (non-ue8m0)
    # scale too.
    x_short = x[:, :-29]
    _, y_s_short = per_token_group_quant_fp8(x_short, 128, use_ue8m0=use_ue8m0)
    tail_absmax_short = np.max(np.abs(x_short[0, 128:]))
    assert tail_absmax_short == 7.0 != tail_absmax
    assert y_s_short[0, 1] != y_s[0, 1]
    # No element past the tensor is touched: quantizing a prefix of exactly
    # one full group yields an identical full-group scale.
    _, y_s_prefix = per_token_group_quant_fp8(x[:, :128], 128, use_ue8m0=use_ue8m0)
    assert y_s_prefix[0, 0] == pytest.approx(y_s[0, 0], rel=1e-12)


def test_per_token_group_quant_fp8_allzero_ue8m0_pinned_byte():
    # Pinned by upstream
    # test_per_token_group_quant_fp8_packed_all_zero: all-zero input yields
    # UE8M0 byte 0x5E (94). Kernel contract: y_s = eps/448, then
    # fmax(y_s, 1e-10) clamps back to 1e-10, then
    # exp2(ceil(log2(1e-10))) = 2^-33.
    # NOTE: the upstream *Triton reference* path instead computes
    # absmax = max(0, eps) = eps and scale_raw = eps/448 (~2^-42.1), which
    # would give 2^-42 — the packed kernel contract (2^-33, byte 0x5E) is
    # what upstream pins, so the oracle pins it here.
    x = np.zeros((4, 256))
    _, y_s = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    assert np.all(y_s == 2.0**-33)
    packed, _ = pack_ue8m0_scale_exponents(np.full((4, 4), 2.0**-33))
    assert np.all(packed == 0x5E5E5E5E)
    # Cross-check the float32 exponent-byte derivation pinned in the
    # upstream docstring: 1e-10 has biased exponent 0x5D with a non-zero
    # mantissa, and the ceil-round-up lands on 0x5E.
    assert ((np.float32(1e-10).view(np.uint32) >> 23) & 0xFF) == 0x5D
    assert ((np.float32(2.0**-33).view(np.uint32) >> 23) & 0xFF) == 0x5E


def test_ue8m0_scale_rounds_up_non_power_of_two():
    # Pinned by upstream
    # test_per_token_group_quant_fp8_packed_mantissa_rounds_up: absmax =
    # 1.5 * 448 = 672 forces the mantissa-round-up branch, scale = 2^1 = 2.
    x = np.full((4, 256), 672.0)
    _, y_s = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    assert np.all(y_s == 2.0)
    # Quantized values: 672/2 = 336, which sits exactly halfway between the
    # adjacent e4m3fn values 320 (mantissa 0b010) and 352 (mantissa 0b011);
    # round-to-nearest-even ties to the even mantissa, giving 320.
    y_q, _ = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    np.testing.assert_array_equal(y_q, np.full((4, 256), 320.0))


def test_packed_scale_layout_matches_upstream_formula():
    # Layout pinned by upstream
    # test_per_token_group_quant_fp8_packed (per-element expected loop):
    # idx = pack_col * tma_aligned_mn + row, byte pos = g % 4, mn padded to
    # a multiple of 4, padding slots zero.
    mn, groups_per_row = 3, 6  # mn=3 pads to 4, 6 groups -> 2 packed cols
    y_s = np.exp2(np.arange(mn * groups_per_row, dtype=np.float64).reshape(mn, groups_per_row) - 10)
    packed, tma_aligned_mn = pack_ue8m0_scale_exponents(y_s)
    assert tma_aligned_mn == 4
    exponents = (np.float32(y_s).view(np.uint32) >> 23) & 0xFF
    expected = np.zeros(mn + (2 - 1) * 4, dtype=np.uint32)
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col, pos = g // 4, g % 4
            expected[pack_col * 4 + row] |= np.uint32(exponents[row, g]) << (pos * 8)
    np.testing.assert_array_equal(packed, expected)
    # Padding slot: packed col 0 reserves rows 0..tma_aligned_mn-1, so the
    # padded row 3 (mn=3) lands at index 3 and must be zero.
    assert packed[3] == 0


@pytest.mark.parametrize("bits,expected_n", [(2, 4), (3, 8), (4, 16)])
def test_lloyd_max_shapes_and_sortedness(bits, expected_n):
    # Ported invariants from vllm/tests/quantization/test_turboquant.py::
    # TestLloydMax::test_solve_shapes / test_centroids_sorted /
    # test_boundaries_sorted (KV quant math).
    centroids, boundaries = solve_lloyd_max(128, bits)
    assert centroids.shape == (expected_n,)
    assert boundaries.shape == (expected_n - 1,)
    assert np.all(np.diff(centroids) > 0), "centroids not strictly sorted"
    assert np.all(np.diff(boundaries) > 0), "boundaries not strictly sorted"


@pytest.mark.parametrize("bits", [3, 4])
def test_lloyd_max_boundaries_are_midpoints_and_symmetric(bits):
    # TestLloydMax::test_boundaries_are_midpoints and
    # TestCentroids::test_centroids_symmetric_around_zero (tolerance 0.01
    # pinned there).
    centroids, boundaries = solve_lloyd_max(128, bits)
    np.testing.assert_allclose(boundaries, (centroids[:-1] + centroids[1:]) / 2.0, atol=1e-6)
    assert abs(centroids.mean()) < 0.01, "centroids not centered near 0"
    assert abs(centroids[0] + centroids[-1]) < 0.01
    # TestCentroids::test_centroids_within_4sigma: all centroids within
    # ~4 sigma of N(0, 1/d), sigma = sqrt(1/128).
    sigma = math.sqrt(1.0 / 128)
    assert np.all(np.abs(centroids) < 4 * sigma)


def test_lloyd_max_deterministic():
    # TestLloydMax::test_solve_deterministic.
    c1, b1 = solve_lloyd_max(128, 3)
    c2, b2 = solve_lloyd_max(128, 3)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Review findings 2026-09-15 (astra): halfway ties and non-finite encodes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (0.75, 1.0), (1.25, 1.0), (1.75, 2.0), (2.5, 2.0), (3.5, 4.0),
    (-0.75, -1.0), (-1.75, -2.0), (-3.5, -4.0),
])
def test_e2m1_encode_halfway_boundaries_match_round_to_e2m1(value, expected):
    # encode_e2m1 must agree with the file's canonical round_to_e2m1 at every
    # halfway boundary (argmin previously rounded all ties DOWN).
    nibble = encode_e2m1(np.array([value]))[0]
    assert decode_e2m1(np.array([nibble]))[0] == expected


def test_e2m1_encode_matches_round_to_e2m1_on_dense_grid():
    rng = np.random.default_rng(11)
    grid = np.concatenate([
        rng.uniform(-7, 7, 4000),
        np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -2.5]),
    ])
    decoded = decode_e2m1(encode_e2m1(grid))
    assert np.array_equal(decoded, round_to_e2m1(grid))


def test_e4m3_encode_nan_roundtrips_to_nan_with_sign():
    for value in (np.nan, -np.nan):
        byte = e4m3_encode(np.array([value]))[0]
        assert byte in (0x7F, 0xFF), f"NaN must encode to 0x7F/0xFF, got {byte:#04x}"
        decoded = e4m3_decode(np.array([byte]))[0]
        assert np.isnan(decoded), f"{byte:#04x} decoded to {decoded}, expected NaN"


def test_e4m3_encode_inf_saturates_signed_max():
    assert e4m3_decode(e4m3_encode(np.array([np.inf])))[0] == 448.0
    assert e4m3_decode(e4m3_encode(np.array([-np.inf])))[0] == -448.0
