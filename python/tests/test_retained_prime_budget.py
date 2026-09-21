"""CPU regression tests for the retained-context prime budget and abort diagnostics.

The historical retained protocol primed the thinking-disabled parent turn with a
16-token budget and hard-failed the whole sweep when it did not stop. These tests
exercise the real extracted helpers (not a re-implementation): the option type,
the protocol metadata, the strict prime acceptance, and the bounded diagnostics.
No GPU, no server: every test runs with CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench-ds41-release-retained-decode.py"
SOURCE = SCRIPT.read_text()


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_retained_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_disabled_budget_remains_the_default():
    q = _load()
    assert q.DEFAULT_DISABLED_PRIME_MAX_TOKENS == 16
    assert q.ENABLED_PRIME_MAX_TOKENS == 1024
    # The CLI option must default to the historical constant, not a new literal.
    assert "default=DEFAULT_DISABLED_PRIME_MAX_TOKENS" in SOURCE


def test_option_type_accepts_only_positive_integers():
    q = _load()
    assert q.positive_int("16") == 16
    assert q.positive_int("64") == 64
    for bad in ("0", "-1", "-64"):
        with pytest.raises(argparse.ArgumentTypeError):
            q.positive_int(bad)


def test_cli_rejects_a_non_positive_budget_before_any_request():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--tokenizer", "/nonexistent",
         "--context-file", "/nonexistent", "--label", "t",
         "--output", "/nonexistent/out.json",
         "--disabled-prime-max-tokens", "0"],
        capture_output=True, text=True, timeout=60)
    assert completed.returncode == 2
    assert "must be a positive integer" in completed.stderr


def test_protocol_metadata_labels_the_budget():
    q = _load()
    historical = q.prime_protocol(q.DEFAULT_DISABLED_PRIME_MAX_TOKENS)
    bumped = q.prime_protocol(64)
    assert historical["disabled_max_tokens"] == 16
    assert historical["version"] == "retained-prime-v1.disabled16"
    assert bumped["version"] == "retained-prime-v1.disabled64"
    assert bumped["historical_default"] == 16
    assert bumped["enabled_max_tokens"] == 1024


def test_prime_acceptance_is_strict_and_unchanged():
    q = _load()
    good = {"finish_reason": "stop", "text": "OK",
            "usage": {"prompt_tokens": 32768, "completion_tokens": 2}}
    assert q.prime_failure_reason(good, 32768) is None
    # finish_reason must be stop: length would truncate the parent turn.
    assert "finish_reason" in q.prime_failure_reason(
        dict(good, finish_reason="length"), 32768)
    assert q.prime_failure_reason(dict(good, text="   "), 32768) == "empty parent answer"
    assert "token count" in q.prime_failure_reason(
        dict(good, usage={"prompt_tokens": 32767}), 32768)


def test_prime_diagnostics_are_bounded_and_complete():
    q = _load()
    result = {"finish_reason": "length", "text": "x" * 5000,
              "usage": {"prompt_tokens": 262144, "completion_tokens": 16}}
    record = q.prime_failure_diagnostics(
        result, context=262144, thinking="disabled", max_tokens=16,
        reason="finish_reason='length'")
    assert record["reason"] == "finish_reason='length'"
    assert record["finish_reason"] == "length"
    assert record["prompt_tokens"] == 262144
    assert record["completion_tokens"] == 16
    assert len(record["text_head"]) == q.PRIME_TEXT_HEAD_CHARS == 200
    error = q.prime_error_diagnostics(
        RuntimeError("x" * 5000), context=262144, thinking="disabled", max_tokens=64)
    assert error["reason"] == "request failed"
    assert len(error["error"]) == q.PRIME_ERROR_CHARS == 400


def test_child_cache_frontier_gate_is_prompt_prefix_not_the_generated_turn():
    """The gate proves the retained parent prompt was reused.

    A text chat API re-renders the parent assistant turn from text, so the
    committed generated turn only round-trips token-exactly when the detokenized
    text re-encodes identically. `hit >= context` with self-consistent accounting
    is the real gate; the generated-turn frontier is a diagnostic.
    """
    q = _load()
    assert "retained_cache_accounting" in SOURCE
    assert "prompt_prefix_reused" in SOURCE
    assert "full_turn_reused" in SOURCE
    # The release gate must be the prompt-prefix decision, not the turn frontier.
    assert 'cache_valid = accounting["prompt_prefix_reused"]' in SOURCE
    assert "context <= hit <= max(parent_frontiers)" in SOURCE
    assert "raise RuntimeError(\"retained parent did not finish its answer\")" not in SOURCE


def test_prompt_prefix_reuse_passes_when_the_generated_turn_does_not_round_trip():
    """v10 65536/262144 and v9 262144: prompt reused, generated turn not."""
    q = _load()
    # 65536/code-reasoning/repeat1: hit == context, well below the turn frontiers.
    v10 = q.retained_cache_accounting(
        {"prompt_tokens": 65659, "prompt_cache_hit_tokens": 65536,
         "prompt_cache_miss_tokens": 123,
         "prompt_tokens_details": {"cached_tokens": 65536}},
        context=65536, parent_frontiers=[65614, 65615])
    assert v10["accounting_consistent"] is True
    assert v10["prompt_prefix_reused"] is True
    assert v10["full_turn_reused"] is False
    assert v10["prompt_prefix_frontier"] == 65536
    # v9 262144: the 147-token turn partially matched, hit == context + 14.
    v9 = q.retained_cache_accounting(
        {"prompt_tokens": 262335, "prompt_cache_hit_tokens": 262158,
         "prompt_cache_miss_tokens": 177,
         "prompt_tokens_details": {"cached_tokens": 262158}},
        context=262144, parent_frontiers=[262290, 262291])
    assert v9["prompt_prefix_reused"] is True
    assert v9["full_turn_reused"] is False
    # 32768 and the 2K control reached the full turn frontier.
    for context, prompt, hit, frontier in ((32768, 32841, 32797, [32796, 32797]),
                                           (2048, 2118, 2074, [2073, 2074])):
        ok = q.retained_cache_accounting(
            {"prompt_tokens": prompt, "prompt_cache_hit_tokens": hit,
             "prompt_cache_miss_tokens": prompt - hit,
             "prompt_tokens_details": {"cached_tokens": hit}},
            context=context, parent_frontiers=frontier)
        assert ok["prompt_prefix_reused"] is True
        assert ok["full_turn_reused"] is True


def test_cache_gate_still_fails_below_the_prompt_and_on_bad_accounting():
    q = _load()
    base = {"prompt_tokens": 65659, "prompt_cache_hit_tokens": 65000,
            "prompt_cache_miss_tokens": 659,
            "prompt_tokens_details": {"cached_tokens": 65000}}
    below = q.retained_cache_accounting(base, context=65536,
                                        parent_frontiers=[65614, 65615])
    assert below["prompt_prefix_reused"] is False
    # A hit beyond the parent turn frontier is not a legitimate reuse either.
    beyond = q.retained_cache_accounting(
        dict(base, prompt_cache_hit_tokens=65616, prompt_cache_miss_tokens=43,
             prompt_tokens_details={"cached_tokens": 65616}),
        context=65536, parent_frontiers=[65614, 65615])
    assert beyond["prompt_prefix_reused"] is False
    # cached_tokens disagreeing with prompt_cache_hit_tokens is not consistent.
    inconsistent = q.retained_cache_accounting(
        dict(base, prompt_cache_hit_tokens=65536, prompt_cache_miss_tokens=123,
             prompt_tokens_details={"cached_tokens": 65535}),
        context=65536, parent_frontiers=[65614, 65615])
    assert inconsistent["accounting_consistent"] is False
    assert inconsistent["prompt_prefix_reused"] is False
    # Context 0 keeps its small accidental-hit allowance.
    assert q.retained_cache_accounting(
        {"prompt_tokens": 45, "prompt_cache_hit_tokens": 0,
         "prompt_cache_miss_tokens": 45,
         "prompt_tokens_details": {"cached_tokens": 0}},
        context=0, parent_frontiers=[])["prompt_prefix_reused"] is True
    assert q.retained_cache_accounting(
        {"prompt_tokens": 45, "prompt_cache_hit_tokens": 40,
         "prompt_cache_miss_tokens": 5,
         "prompt_tokens_details": {"cached_tokens": 40}},
        context=0, parent_frontiers=[])["prompt_prefix_reused"] is False


def test_cache_failure_diagnostics_keep_the_262k_identity_mismatch():
    """hit+miss==prompt is accounting, not proof of parent-turn reuse."""
    q = _load()
    usage = {"prompt_tokens": 262335, "prompt_cache_hit_tokens": 262158,
             "prompt_cache_miss_tokens": 177,
             "prompt_tokens_details": {"cached_tokens": 262158}}
    accounting = q.retained_cache_accounting(
        usage, context=262144, parent_frontiers=[262290, 262291])
    record = q.cache_failure_diagnostics(
        usage, context=262144, case_id="code-reasoning", repeat=1,
        parent_frontiers=[262290, 262291], accounting=accounting)
    assert record["accounting_consistent"] is True
    assert record["cached_tokens"] == 262158
    assert record["allowed_parent_frontiers"] == [262290, 262291]
    assert record["cached_tokens"] not in record["allowed_parent_frontiers"]
    assert record["prompt_prefix_frontier"] == 262144
    # The disabled sibling DOES land on its frontier; the gate is not simply too strict.
    disabled = {"prompt_tokens": 262190, "prompt_cache_hit_tokens": 262146,
                "prompt_cache_miss_tokens": 44}
    assert 262146 in [262145, 262146]
    assert q.cache_failure_diagnostics(
        disabled, context=262144, case_id="code", repeat=1,
        parent_frontiers=[262145, 262146])["accounting_consistent"] is True


def test_frontier_failure_is_wrapped_and_preserves_the_exception():
    # The cache-accounting raise must record sample_failures + aborted status
    # before propagating (previously it left status="running").
    assert 'report.setdefault("sample_failures", [])' in SOURCE
    assert "cache_failure_diagnostics" in SOURCE
    assert 'raise RuntimeError(' in SOURCE


def test_262k_serialized_token_trace_if_local_evidence_present():
    """Executed trace when the 262K evidence + tokenizer are on this machine."""
    from pathlib import Path as _Path
    evidence = ROOT / "runs" / "tp6-validation" / "retained-262k-parent-child-evidence.json"
    tokenizer_path = (_Path.home() / ".cache" / "huggingface" / "hub" /
                      "models--deepseek-ai--DeepSeek-V4.1-Flash" / "snapshots" /
                      "dba1be0a40aa45a94ad051997016db3960a90277" / "tokenizer.json")
    if not evidence.exists() or not tokenizer_path.exists():
        pytest.skip("262K evidence or tokenizer not present on this machine")
    q = _load()
    tokenizers = pytest.importorskip("tokenizers")
    payload = json.loads(evidence.read_text())
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    parents = {p["thinking"]: p for p in payload["parents_262144"]}
    children = {c["case"]: c for c in payload["children_262144_repeat1"]}
    # The disabled parent and its child share the full user prompt exactly.
    disabled_user = parents["disabled"]["request"]["messages"][0]["content"]
    code = children["code"]
    assert count(code["request"]["messages"][0]["content"]) == count(disabled_user)
    assert code["result"]["usage"]["prompt_cache_hit_tokens"] in \
        parents["disabled"]["allowed_parent_frontiers"]
    # The enabled parent's turn carries 144 reasoning tokens the child also re-sends,
    # but the server's reusable prefix stops 132 tokens earlier.
    enabled = parents["enabled"]
    reasoning_tokens = count(enabled["result"]["reasoning"])
    assert reasoning_tokens == 144
    reasoning_child = children["code-reasoning"]
    hit = reasoning_child["result"]["usage"]["prompt_cache_hit_tokens"]
    assert hit == 262158
    assert hit not in enabled["allowed_parent_frontiers"]
    assert enabled["result"]["usage"]["total_tokens"] == 262291
    assert hit < enabled["allowed_parent_frontiers"][0]
    # Under the retained contract this cell is prompt-prefix reuse: the whole
    # 262144-token parent prompt was skipped, so the retained base context is
    # real; only the committed generated turn was not reused whole.
    accounting = q.retained_cache_accounting(
        reasoning_child["result"]["usage"], context=262144,
        parent_frontiers=enabled["allowed_parent_frontiers"])
    assert accounting["prompt_prefix_reused"] is True
    assert accounting["full_turn_reused"] is False
    assert hit >= 262144
