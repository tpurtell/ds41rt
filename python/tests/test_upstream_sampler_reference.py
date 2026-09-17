"""CPU reference oracle for llama.cpp sampler semantics.

Ported from upstream llama.cpp ``tests/test-sampling.cpp`` (MIT) and
``src/llama-sampler.cpp`` (the implementation semantics the test pins down).
This is a numpy-only reference implementation of the temperature, top-k,
top-p, min-p and repetition-penalty samplers applied to llama.cpp's exact
test vectors, asserting the same expected probability outputs. It is the CPU
oracle that the deferred GPU parity test will later run the native ds41rt
sampler against; it has no dependency on any ds41rt module.

Semantics mirrored from llama-sampler.cpp (as of the upstream test):

- ``temp``: ``temp <= 0`` one-hots the argmax logit (others -> -inf);
  otherwise logits are divided by ``temp``.
- ``temp_ext`` with ``delta == 0`` reduces to ``temp``.
- ``top_k``: ``k <= 0`` keeps everything (no sort); otherwise keep the top-k
  logits, sorted descending.
- ``top_p``: ``p >= 1.0`` is a no-op (no sort); otherwise softmax, sort
  descending, keep the smallest prefix whose cumulative probability satisfies
  ``cum_sum >= p`` (always keeping at least ``min_keep`` tokens).
- ``min_p``: operates on **logits**, not probabilities:
  ``min_logit = max_logit + log(min_p)``; tokens with
  ``logit >= min_logit`` survive. Unsorted arrays filter in place; if that
  yields fewer than ``min_keep`` tokens, the sorted fallback keeps at least
  the top token.
- ``penalties``: for each token seen in the ``penalty_last_n`` window
  (here: exactly the accepted ``last_tokens``): if ``logit <= 0`` multiply by
  ``repeat_penalty`` else divide by it (guard against negative-logit
  inversion), then subtract ``count * alpha_frequency`` and
  ``alpha_presence`` if ``count > 0``. Probabilities are re-derived by a
  final softmax (the upstream test applies a ``dist`` sampler which
  renormalizes ``cur_p`` without changing its size; ``dist`` also *selects* a
  token via ``std::mt19937``, which only affects ``selected``, not ``p``, so
  the rng itself is not part of this oracle).
"""
"""
COVERAGE CLASS: standalone reference oracle. These cases document upstream
behavior with no ds41rt dependency; they cannot detect product regressions by
themselves. They are the comparison references for the deferred GPU parity
tests (docs/test-coverage/DEFERRED.md), counted separately from product
regression coverage (review MAJOR 4/6, 2026-09-15).
"""


from dataclasses import dataclass, field

import numpy as np
import pytest

NEG_INF = float("-inf")
# Same absolute tolerance as the upstream C++ check(): fabs(p - expected) < 1e-5
ATOL = 1e-5


@dataclass
class TokenData:
    """Mirror of llama_token_data (id, logit, p)."""

    id: int
    logit: float
    p: float = 0.0


@dataclass
class CurP:
    """Mirror of llama_token_data_array."""

    data: list
    sorted: bool = False
    selected: int = -1

    @property
    def size(self):
        return len(self.data)

    def ids(self):
        return [token.id for token in self.data]

    def probs(self):
        return np.array([token.p for token in self.data])


def cur_p_from_probs(probs):
    """sampler_tester(probs, probs_expected) constructor: logit = log(p)."""
    return CurP([TokenData(i, float(np.log(p)), float(p)) for i, p in enumerate(probs)])


def cur_p_from_vocab(n_vocab):
    """sampler_tester(n_vocab) constructor: logit = log(token_id) (id 0 -> -inf)."""
    return CurP(
        [TokenData(token_id, NEG_INF if token_id == 0 else float(np.log(token_id))) for token_id in range(n_vocab)]
    )


def softmax_inplace(cur: CurP, do_sort: bool) -> None:
    """llama_sampler_softmax_impl."""
    assert cur.size > 0
    if do_sort and not cur.sorted:
        cur.data.sort(key=lambda token: token.logit, reverse=True)
        cur.sorted = True
    max_l = cur.data[0].logit if cur.sorted else max(token.logit for token in cur.data)
    cum_sum = 0.0
    for token in cur.data:
        token.p = float(np.exp(token.logit - max_l))
        cum_sum += token.p
    for token in cur.data:
        token.p /= cum_sum


def sampler_temp_apply(cur: CurP, temp: float) -> None:
    """llama_sampler_temp_impl."""
    if cur.size == 0:
        return
    if temp <= 0.0:
        max_i = 0
        max_l = cur.data[0].logit
        for i in range(1, cur.size):
            if cur.data[i].logit > max_l:
                cur.data[max_i].logit = NEG_INF
                max_i = i
                max_l = cur.data[i].logit
            else:
                cur.data[i].logit = NEG_INF
        return
    for token in cur.data:
        token.logit /= temp


def sampler_top_k_apply(cur: CurP, k: int) -> None:
    """llama_sampler_top_k_impl."""
    if k <= 0:
        return
    k = min(k, cur.size)
    if not cur.sorted:
        cur.data.sort(key=lambda token: token.logit, reverse=True)
        cur.sorted = True
    del cur.data[k:]


def sampler_top_p_apply(cur: CurP, p: float, min_keep: int = 1) -> None:
    """llama_sampler_top_p_apply."""
    if p >= 1.0:
        return
    softmax_inplace(cur, do_sort=False)
    if not cur.sorted:
        cur.data.sort(key=lambda token: token.logit, reverse=True)
        cur.sorted = True
    cum_sum = 0.0
    last_idx = cur.size
    for i in range(cur.size):
        cum_sum += cur.data[i].p
        if cum_sum >= p and i + 1 >= min_keep:
            last_idx = i + 1
            break
    del cur.data[last_idx:]


def sampler_min_p_apply(cur: CurP, p: float, min_keep: int = 1) -> None:
    """llama_sampler_min_p_apply (logit-domain filtering)."""
    if p <= 0.0 or cur.size == 0:
        return
    min_p_applied = False
    if not cur.sorted:
        max_logit = max(token.logit for token in cur.data)
        min_logit = max_logit + float(np.log(p))
        filtered = [token for token in cur.data if token.logit >= min_logit]
        if filtered and len(filtered) >= min_keep:
            cur.data = filtered
            min_p_applied = True
    if not min_p_applied:
        if not cur.sorted:
            cur.data.sort(key=lambda token: token.logit, reverse=True)
            cur.sorted = True
        min_logit = cur.data[0].logit + float(np.log(p))
        i = 1  # first token always matches
        while i < cur.size:
            if cur.data[i].logit < min_logit and i >= min_keep:
                break
            i += 1
        del cur.data[i:]


def sampler_penalties_apply(cur: CurP, last_tokens, repeat: float, alpha_frequency: float, alpha_presence: float) -> None:
    """llama_sampler_penalties_apply with penalty_last_n == len(last_tokens)."""
    if repeat == 1.0 and alpha_frequency == 0.0 and alpha_presence == 0.0:
        return
    token_count = {}
    for token in last_tokens:
        token_count[token] = token_count.get(token, 0) + 1
    for token in cur.data:
        count = token_count.get(token.id)
        if not count:
            continue
        # Tokens with negative logits are multiplied, positive divided: the
        # common fix for the academic formulation dividing negatives into
        # higher likelihood.
        if token.logit <= 0:
            token.logit *= repeat
        else:
            token.logit /= repeat
        token.logit -= float(count) * alpha_frequency + (1.0 if count > 0 else 0.0) * alpha_presence
    cur.sorted = False


def sampler_dist_apply(cur: CurP, seed: int = 0) -> None:
    """llama_sampler_dist_apply: renormalize probabilities and select a token.

    The upstream ``check()`` only compares ``p`` values; selection uses
    ``std::mt19937`` which has no numpy equivalent, so we select with a
    numpy rng purely to assert a valid, deterministic index.
    """
    softmax_inplace(cur, do_sort=False)
    rng = np.random.default_rng(seed)
    cur.selected = int(rng.choice(cur.size, p=cur.probs()))
    assert 0 <= cur.selected < cur.size


def check_probs(cur: CurP, probs_expected) -> None:
    """sampler_tester::check()."""
    actual = cur.probs()
    expected = np.asarray(probs_expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"size {cur.size} != {len(probs_expected)}"
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# Exact-vector tests: test_temp / test_temp_ext / test_top_k / test_top_p /
# test_min_p / test_penalties from tests/test-sampling.cpp main().
# ---------------------------------------------------------------------------

TEMP_CASES = [
    ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 1.0),
    ([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0], 0.0),
]


@pytest.mark.parametrize("probs,expected,temp", TEMP_CASES)
def test_temp(probs, expected, temp):
    cur = cur_p_from_probs(probs)
    sampler_temp_apply(cur, temp)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


TEMP_EXT_CASES = [
    # delta=0 reduces temp_ext to temp
    ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 1.0, 0.0, 1.0),
    ([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0], 0.0, 0.0, 1.0),
]


@pytest.mark.parametrize("probs,expected,temp,delta,exponent", TEMP_EXT_CASES)
def test_temp_ext(probs, expected, temp, delta, exponent):
    assert delta == 0.0, "oracle only models the delta=0 reduction"
    cur = cur_p_from_probs(probs)
    sampler_temp_apply(cur, temp)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


TOP_K_CASES = [
    ([0.1, 0.2, 0.3, 0.4], [1.0], 1),
    ([0.1, 0.2, 0.3, 0.4], [0.44444, 0.33333, 0.22222], 3),
    ([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], 4),
    ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 0),
]


@pytest.mark.parametrize("probs,expected,k", TOP_K_CASES)
def test_top_k(probs, expected, k):
    cur = cur_p_from_probs(probs)
    sampler_top_k_apply(cur, k)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


TOP_P_CASES = [
    ([0.1, 0.2, 0.3, 0.4], [1.0], 0.0),
    ([0.1, 0.2, 0.3, 0.4], [0.571429, 0.428571], 0.7),
    ([0.1, 0.2, 0.3, 0.4], [0.44444, 0.33333, 0.22222], 0.8),
    ([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], 1.0),
]


@pytest.mark.parametrize("probs,expected,p", TOP_P_CASES)
def test_top_p(probs, expected, p):
    cur = cur_p_from_probs(probs)
    sampler_top_p_apply(cur, p, min_keep=0)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


MIN_P_CASES = [
    ([0.1, 0.2, 0.3, 0.4], [0.1 / 1.0, 0.2 / 1.0, 0.3 / 1.0, 0.4 / 1.0], 0.00),
    ([0.1, 0.2, 0.3, 0.4], [0.1 / 1.0, 0.2 / 1.0, 0.3 / 1.0, 0.4 / 1.0], 0.24),
    ([0.1, 0.2, 0.3, 0.4], [0.2 / 0.9, 0.3 / 0.9, 0.4 / 0.9], 0.26),
    ([0.1, 0.2, 0.3, 0.4], [0.2 / 0.9, 0.3 / 0.9, 0.4 / 0.9], 0.49),
    ([0.1, 0.2, 0.3, 0.4], [0.3 / 0.7, 0.4 / 0.7], 0.51),
    ([0.1, 0.2, 0.3, 0.4], [0.3 / 0.7, 0.4 / 0.7], 0.74),
    ([0.1, 0.2, 0.3, 0.4], [0.4 / 0.4], 0.76),
    ([0.1, 0.2, 0.3, 0.4], [0.4 / 0.4], 1.00),
    ([0.1, 0.2, 0.3, 0.4], [0.4 / 0.4], 1.05),
]


@pytest.mark.parametrize("probs,expected,p", MIN_P_CASES)
def test_min_p(probs, expected, p):
    cur = cur_p_from_probs(probs)
    sampler_min_p_apply(cur, p, min_keep=0)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


PENALTY_CASES = [
    # (probs, last_tokens, expected, repeat_penalty, alpha_frequency, alpha_presence)
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0, 0.25, 0.25, 0.25, 0.25], 50.0, 0.0, 0.0),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2], [0, 0, 0, 0.5, 0.5], 50.0, 0.0, 0.0),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2, 0, 0], [0, 0, 0, 0.5, 0.5], 50.0, 0.0, 0.0),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0], [0.000011, 0.249997, 0.249997, 0.249997, 0.249997], 1.0, 5.0, 5.0),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2], [0.000023, 0.000023, 0.000023, 0.499966, 0.499966], 1.0, 5.0, 5.0),
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0, 1, 2, 0, 0], [0.000000, 0.000023, 0.000023, 0.499977, 0.499977], 1.0, 5.0, 5.0),
]


@pytest.mark.parametrize("probs,last_tokens,expected,repeat,freq,present", PENALTY_CASES)
def test_penalties(probs, last_tokens, expected, repeat, freq, present):
    assert len(probs) == len(expected)
    cur = cur_p_from_probs(probs)
    # Upstream feeds each last token via llama_sampler_accept with
    # penalty_last_n == len(last_tokens), so the window covers all of them.
    sampler_penalties_apply(cur, last_tokens, repeat, freq, present)
    sampler_dist_apply(cur, seed=0)
    check_probs(cur, expected)


# ---------------------------------------------------------------------------
# Sampler-chain tests: test_sampler_queue(). n_vocab tokens with
# logit = log(token_id) (token 0 has p == 0), chains of k/p/m samplers.
# Expected sizes and id ranges replicate the C++ test's analytic oracle.
# ---------------------------------------------------------------------------


def _ceilf(value) -> int:
    return int(np.ceil(np.float32(value)))


def test_sampler_queue(n_vocab=10000, sequence="k", top_k=10000, top_p=1.0, min_p=1.0):
    cur = cur_p_from_vocab(n_vocab)
    max_token_id = n_vocab - 1
    min_token_id = 0

    for sampler in sequence:
        if sampler == "k":
            sampler_top_k_apply(cur, top_k)
        elif sampler == "p":
            sampler_top_p_apply(cur, top_p, min_keep=1)
        elif sampler == "m":
            sampler_min_p_apply(cur, min_p, min_keep=1)
        else:  # pragma: no cover - mirrors GGML_ABORT
            raise ValueError(f"unknown sampler {sampler}")

        # The C++ test applies llama_sampler_init_dist(0) after each step.
        sampler_dist_apply(cur, seed=0)

        size = cur.size
        if sampler == "k":
            expected_size = min(size, top_k)
            min_token_id = max(min_token_id, n_vocab - top_k)
            assert size == expected_size
            assert cur.data[0].id == max_token_id
            assert cur.data[expected_size - 1].id == min_token_id
        elif sampler == "p":
            # softmax over logits log(id) gives p proportional to id, so the
            # C++ oracle can use exact integer cumsums.
            softmax_divisor = n_vocab * (n_vocab - 1) // 2 - min_token_id * (min_token_id - 1) // 2
            softmax_numerator_target = _ceilf(np.float32(top_p) * softmax_divisor)
            min_token_id = n_vocab
            expected_size = 0
            cumsum = 0
            while True:  # do-while: always at least one token is sampled
                min_token_id -= 1
                expected_size += 1
                cumsum += min_token_id
                if not (cumsum < softmax_numerator_target):
                    break
            # token 0 has p == 0, need special consideration for cumsum
            # because top_p immediately returns
            if min_token_id == 1:
                min_token_id -= 1
                expected_size += 1
            assert size == expected_size
            if cur.sorted:
                assert cur.data[0].id == max_token_id
                assert cur.data[expected_size - 1].id == min_token_id
        elif sampler == "m":
            # Mirror C++ float32 arithmetic: ceilf((1.0f - min_p) * n_vocab)
            expected_size = _ceilf((np.float32(1.0) - np.float32(min_p)) * np.float32(n_vocab))
            expected_size = max(expected_size, 1)
            expected_size = min(expected_size, size)
            min_token_id = int(np.floor(np.float32(min_p) * np.float32(n_vocab)))
            min_token_id = max(min_token_id, 1)
            min_token_id = max(min_token_id, n_vocab - size)
            min_token_id = min(min_token_id, n_vocab - 1)
            assert size == expected_size
            if cur.sorted:
                assert cur.data[0].id == max_token_id
                assert cur.data[expected_size - 1].id == min_token_id


QUEUE_CASES = [
    (10000, "k", 10000, 1.0, 1.0),
    (10000, "k", 1, 1.0, 1.0),
    (10000, "p", 10000, 1.0, 1.0),
    (10000, "p", 10000, 0.0, 1.0),
    (10000, "m", 10000, 1.0, 1.0),
    (10000, "m", 10000, 1.0, 1e-12),
    (10000, "k", 100, 1.0, 1.0),
    (10000, "p", 10000, 0.0003, 1.0),
    (10000, "p", 10000, 0.8, 1.0),
    (10000, "m", 10000, 1.0, 9997.9 / 9999.0),
    (10000, "m", 10000, 1.0, 0.1),
    (10000, "kp", 100, 0.8, 0.1),
    (10000, "km", 100, 0.8, 0.1),
    (10000, "pk", 100, 0.8, 0.1),
    (10000, "pm", 100, 0.8, 0.1),
    (10000, "mk", 100, 0.8, 0.1),
    (10000, "mp", 100, 0.8, 9997.9 / 9999.0),
    (10000, "mp", 100, 0.8, 0.1),
    (10000, "kpm", 100, 0.8, 0.1),
    (10000, "kmp", 100, 0.8, 0.1),
    (10000, "pkm", 100, 0.8, 0.1),
    (10000, "pmk", 100, 0.8, 0.1),
    (10000, "mkp", 100, 0.8, 0.1),
    (10000, "mpk", 100, 0.8, 0.1),
]


@pytest.mark.parametrize(
    "n_vocab,sequence,top_k,top_p,min_p",
    QUEUE_CASES,
    ids=[f"{s}-k{k}-p{p}-m{m}" for _, s, k, p, m in QUEUE_CASES],
)
def test_sampler_queue_parametrized(n_vocab, sequence, top_k, top_p, min_p):
    test_sampler_queue(n_vocab, sequence, top_k, top_p, min_p)
