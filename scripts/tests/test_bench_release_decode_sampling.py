"""CPU regression tests for the release decode sampling-comparison harness.

These lock the two things a future measured sampling campaign depends on and
that a manual CLI check cannot protect:

* the exact canonical request vectors the five named profiles send, so a profile
  can never silently inherit a server default or drift into an approximation; and
* the weighted decode-only arithmetic from the release corpus contract,
  ``sum(w * (completion_tokens - 1)) / sum(w * post_first_token_seconds)``.

``scripts/bench-ds41-release-decode.py`` is strict-only: there is deliberately
no approximation encoding, and ``--list-profiles`` documents the canonical
vectors. Nothing here contacts a server or a GPU.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "bench-ds41-release-decode.py"
CORPUS = REPO / "scripts" / "fixtures" / "release-semantic-corpus.json"

EXPECTED_PROFILES = {
    "greedy": dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0),
    "temp0.2-topp0.95": dict(temperature=0.2, top_p=0.95, top_k=0, min_p=0.0),
    "temp0.7-topp0.9": dict(temperature=0.7, top_p=0.9, top_k=0, min_p=0.0),
    "temp0.7-minp0.05": dict(temperature=0.7, top_p=1.0, top_k=0, min_p=0.05),
    "temp0.7-topk40": dict(temperature=0.7, top_p=1.0, top_k=40, min_p=0.0),
}

EXPECTED_STOCHASTIC_BODIES = {
    "temp0.2-topp0.95": {
        "temperature": 0.2, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
    },
    "temp0.7-topp0.9": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0,
    },
    "temp0.7-minp0.05": {
        "temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.05,
    },
    "temp0.7-topk40": {
        "temperature": 0.7, "top_p": 1.0, "top_k": 40, "min_p": 0.0,
    },
}


def _load_module():
    pytest.importorskip("tokenizers")
    spec = importlib.util.spec_from_file_location("ds41rt_bench_release_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Canonical profile vectors
# ---------------------------------------------------------------------------

def test_profiles_match_the_canonical_vectors() -> None:
    module = _load_module()
    assert module.SAMPLING_PROFILES == EXPECTED_PROFILES
    for name, vector in module.SAMPLING_PROFILES.items():
        assert set(vector) == {"temperature", "top_p", "top_k", "min_p"}, name


def test_default_profile_is_the_historical_greedy_control() -> None:
    module = _load_module()
    assert module.DEFAULT_SAMPLING_PROFILE == "greedy"


def test_greedy_sends_temperature_only() -> None:
    # vLLM forces top_p=1/top_k=0/min_p=0 when temperature < 1e-5, so the greedy
    # body must not carry the disabled filters: doing so would add request
    # surface that some engines reject without adding information.
    module = _load_module()
    vector, source = module.resolve_profile("greedy", {})
    assert module.sampling_request_fields(vector) == {"temperature": 0.0}
    assert source == "greedy"


@pytest.mark.parametrize("name", sorted(EXPECTED_STOCHASTIC_BODIES))
def test_stochastic_request_vectors_are_exact(name: str) -> None:
    module = _load_module()
    vector, _ = module.resolve_profile(name, {})
    assert module.sampling_request_fields(vector) == EXPECTED_STOCHASTIC_BODIES[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_STOCHASTIC_BODIES))
def test_stochastic_profiles_disable_unrelated_filters(name: str) -> None:
    """Exactly the tested filter is active; every other filter is disabled."""
    module = _load_module()
    vector = module.SAMPLING_PROFILES[name]
    active = []
    if vector["top_p"] != 1.0:
        active.append("top_p")
    if vector["top_k"] != 0:
        active.append("top_k")
    if vector["min_p"] != 0.0:
        active.append("min_p")
    assert len(active) == 1, f"{name} must exercise exactly one filter, got {active}"
    assert vector["temperature"] == 0.7 or name == "temp0.2-topp0.95"
    # And the transmitted body agrees with the canonical vector.
    assert module.sampling_request_fields(vector) == EXPECTED_STOCHASTIC_BODIES[name]


# ---------------------------------------------------------------------------
# Overrides and validation
# ---------------------------------------------------------------------------

def test_overrides_normalize_top_k_minus_one_to_disabled() -> None:
    module = _load_module()
    vector, source = module.resolve_profile("temp0.7-topk40", {"top_k": -1})
    assert vector["top_k"] == 0
    assert source == "custom:temp0.7-topk40"


def test_overrides_are_recorded_as_custom() -> None:
    module = _load_module()
    vector, source = module.resolve_profile("temp0.7-topp0.9", {"temperature": 0.5})
    assert vector["temperature"] == 0.5 and vector["top_p"] == 0.9
    assert source == "custom:temp0.7-topp0.9"


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": -2},
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"top_p": float("nan")},
        {"min_p": -0.1},
        {"min_p": 1.5},
        {"min_p": float("nan")},
    ],
)
def test_invalid_vectors_are_rejected(overrides) -> None:
    module = _load_module()
    with pytest.raises(SystemExit):
        module.resolve_profile("greedy", overrides)


def test_unknown_profile_is_rejected() -> None:
    module = _load_module()
    with pytest.raises(SystemExit):
        module.resolve_profile("no-such-profile", {})


def test_no_approximation_surface_remains() -> None:
    """Scope guard: the strict-only harness has no fallback encoding."""
    module = _load_module()
    assert not hasattr(module, "SAMPLING_ENCODINGS")
    assert not hasattr(module, "sampling_request_fields_approximate")
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--sampling-profile" in help_text
    assert "--sampling-encoding" not in help_text
    assert "--allow-approximation" not in help_text


def test_list_profiles_is_standalone_and_lists_five_profiles() -> None:
    """`--list-profiles` is docs surface; it must not require request args."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list-profiles"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == EXPECTED_PROFILES


# ---------------------------------------------------------------------------
# Weighted decode-only arithmetic
# ---------------------------------------------------------------------------

def _sample(weight, completion_tokens, finish, first_output, **extra):
    row = dict(weight=weight, usage=dict(completion_tokens=completion_tokens),
               finish_seconds=finish, first_output_seconds=first_output)
    row.update(extra)
    return row


def test_weighted_metric_matches_the_corpus_contract() -> None:
    module = _load_module()
    samples = [
        _sample(1.0, 11, 1.5, 0.5),   # 10 tokens over 1.0 s
        _sample(0.5, 21, 2.5, 0.5),   # 20 tokens over 2.0 s, weight 0.5
        _sample(1.0, 5, 1.0, 0.5),    # 4 tokens over 0.5 s
    ]
    metric = module.weighted_decode_metric(samples)
    assert metric["weighted_cases"] == 3
    assert metric["timed_tokens"] == pytest.approx(24.0)
    assert metric["timed_seconds"] == pytest.approx(2.5)
    assert metric["weighted_observed_decode_tokens_per_second"] == pytest.approx(9.6)


def test_weighted_metric_excludes_zero_weight_and_untimed_samples() -> None:
    module = _load_module()
    samples = [
        _sample(1.0, 11, 1.5, 0.5),
        # counting/orchid carry weight 0 and must never enter the weighted score
        _sample(0.0, 601, 3.0, 0.5),
        # a failed/untimed sample must not be scored or silently zero-timed
        dict(weight=1.0, usage=dict(completion_tokens=7), error="boom"),
    ]
    metric = module.weighted_decode_metric(samples)
    assert metric["weighted_cases"] == 1
    assert metric["weighted_observed_decode_tokens_per_second"] == pytest.approx(10.0)


def test_weighted_metric_returns_none_without_timed_seconds() -> None:
    module = _load_module()
    metric = module.weighted_decode_metric([_sample(1.0, 11, 0.5, 0.5)])
    assert metric["weighted_cases"] == 1
    assert metric["timed_seconds"] == 0.0
    assert metric["weighted_observed_decode_tokens_per_second"] is None


def test_metric_uses_completion_tokens_minus_one() -> None:
    """The first generated token is charged to prefill, not decode."""
    module = _load_module()
    one_token = module.weighted_decode_metric([_sample(1.0, 1, 1.0, 0.0)])
    assert one_token["timed_tokens"] == 0.0
    assert one_token["weighted_observed_decode_tokens_per_second"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Content categories and weights are preserved
# ---------------------------------------------------------------------------

def test_release_corpus_categories_and_weights_are_unchanged() -> None:
    corpus = json.loads(CORPUS.read_text())
    assert corpus["weighted_case_ids"] == [
        "code", "code-reasoning", "math", "fable", "hello", "topic",
        "structured-json", "structured-json-schema", "multilingual",
    ]
    weights = {case: corpus["cases"][case]["weight"] for case in corpus["weighted_case_ids"]}
    assert weights == {
        "code": 1.0, "code-reasoning": 1.0, "math": 1.0, "fable": 1.0,
        "hello": 1.0, "topic": 1.0, "structured-json": 0.5,
        "structured-json-schema": 0.5, "multilingual": 1.0,
    }
    # counting stays outside the weighted score.
    assert corpus["counting"]["weight"] == 0.0
    assert "count" in corpus["measurement_contract"]["excluded_from_weighted"]


def test_server_defaults_are_recorded_in_the_report_metadata() -> None:
    module = _load_module()
    for field in ("temperature", "top_p", "top_k", "min_p", "seed"):
        assert field in module.SERVER_DEFAULTS_IF_UNSET
    assert "top_k=0 or -1 disables top-k" in module.SAMPLING_SEMANTICS
    assert "temperature -> min_p -> top_k -> top_p" in module.SAMPLING_SEMANTICS


def test_dspark_evidence_limit_is_explicit() -> None:
    """/v1/stats and the usage block have no dSpark counters."""
    module = _load_module()
    evidence = module.DSPARK_EVIDENCE
    assert evidence["usage_block_counters"] is False
    assert evidence["stats_endpoint_counters"] is False
    assert evidence["requires_runtime_log_debug_pass"] is True
    assert evidence["debug_pass_perturbs_timing"] is True
    assert evidence["debug_pass_is_performance_evidence"] is False
    assert "suppressed-draft" in evidence["active_dspark_claim_requires"]


def test_provenance_separates_measurement_source_from_release_doc_commit() -> None:
    module = _load_module()
    identity = module.harness_identity(SCRIPT)
    assert identity["path"].endswith("bench-ds41-release-decode.py")
    assert len(identity["sha256"]) == 64
    revision = module.source_revision(REPO)
    assert revision is None or (
        len(revision) == 40 and all(c in "0123456789abcdef" for c in revision))
    assert "release_doc_commit" in module.PROVENANCE_NOTE


def test_campaign_cli_surface_exposes_interleaving_and_identity() -> None:
    """Interleaved rotation and hardware/image identity are first-class."""
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--repeat-index" in help_text
    assert "--identity-file" in help_text
    assert "hardware" in help_text.lower()


# ---------------------------------------------------------------------------
# Interleaved aggregation
# ---------------------------------------------------------------------------

AGGREGATOR = REPO / "scripts" / "aggregate-sampling-decode.py"


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("ds41rt_aggregate_sampling", AGGREGATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(profile, repeat, tps, start_ns, case="code", completion_tokens=11):
    return (
        {
            "sampling": {"profile": profile},
            "repeat_summaries": [{
                "repeat": repeat,
                "weighted_observed_decode_tokens_per_second": tps,
                "timed_tokens": (tps or 0) * 2,
                "timed_seconds": 2.0,
            }],
            "samples": [{
                "case": case,
                "started_ns": start_ns,
                "observed_decode_tokens_per_second": tps,
                "usage": {"completion_tokens": completion_tokens},
                "passed": True,
            }],
        },
        "0" * 64,
        f"{profile}-r{repeat}.json",
    )


def test_aggregate_detects_interleaved_rotation() -> None:
    module = _load_aggregator()
    documents = [
        _report("greedy", 1, 10.0, 100),
        _report("temp0.7-topk40", 1, 20.0, 200),
        _report("greedy", 2, 14.0, 300),
        _report("temp0.7-topk40", 2, 24.0, 400),
    ]
    result = module.aggregate(documents)
    assert result["profiles"]["greedy"]["weighted_observed_decode_tokens_per_second"] == [10.0, 14.0]
    assert result["profiles"]["greedy"]["median_weighted_observed_decode_tokens_per_second"] == 12.0
    assert result["profiles"]["temp0.7-topk40"]["median_weighted_observed_decode_tokens_per_second"] == 22.0
    assert [entry["profile"] for entry in result["execution_order"]] == [
        "greedy", "temp0.7-topk40", "greedy", "temp0.7-topk40"]
    assert result["interleaved"] is True
    assert result["profile_major"] is False
    assert result["weighted_ranking"] == ["temp0.7-topk40", "greedy"]


def test_aggregate_flags_profile_major_ordering() -> None:
    module = _load_aggregator()
    documents = [
        _report("greedy", 1, 10.0, 100),
        _report("greedy", 2, 14.0, 200),
        _report("temp0.7-topk40", 1, 20.0, 300),
        _report("temp0.7-topk40", 2, 24.0, 400),
    ]
    result = module.aggregate(documents)
    assert result["profile_major"] is True
    assert result["interleaved"] is False


def test_aggregate_rejects_non_sampling_reports() -> None:
    module = _load_aggregator()
    with pytest.raises(ValueError):
        module.aggregate([({"samples": []}, "0" * 64, "plain.json")])


def test_profile_sequence_helper() -> None:
    module = _load_aggregator()
    assert module.profile_sequence_is_major(["a", "a", "b", "b"]) is True
    assert module.profile_sequence_is_major(["a", "b", "a", "b"]) is False
    assert module.profile_sequence_is_major([]) is True


def test_aggregate_orders_multi_repeat_documents_by_repeat() -> None:
    """A report holding several repeats must not collapse to one timestamp."""
    module = _load_aggregator()

    def multi(profile, first_tps, second_tps, first_ns, second_ns):
        def row(repeat, tps, ns):
            return {"case": "code", "repeat": repeat, "started_ns": ns,
                    "observed_decode_tokens_per_second": tps,
                    "usage": {"completion_tokens": 11}, "passed": True}
        return ({
            "sampling": {"profile": profile},
            "repeat_summaries": [
                {"repeat": 1, "weighted_observed_decode_tokens_per_second": first_tps,
                 "timed_tokens": first_tps * 2, "timed_seconds": 2.0},
                {"repeat": 2, "weighted_observed_decode_tokens_per_second": second_tps,
                 "timed_tokens": second_tps * 2, "timed_seconds": 2.0},
            ],
            "samples": [row(1, first_tps, first_ns), row(2, second_tps, second_ns)],
        }, "0" * 64, f"{profile}.json")

    result = module.aggregate([
        multi("greedy", 10.0, 14.0, 100, 300),
        multi("temp0.7-topk40", 20.0, 24.0, 200, 400),
    ])
    assert [entry["profile"] for entry in result["execution_order"]] == [
        "greedy", "temp0.7-topk40", "greedy", "temp0.7-topk40"]
    assert [entry["repeat"] for entry in result["execution_order"]] == [1, 1, 2, 2]
    assert result["interleaved"] is True
    assert result["profile_major"] is False


def test_repeat_plan_validates_the_index() -> None:
    module = _load_module()
    assert module.repeat_plan(3, None) == [1, 2, 3]
    assert module.repeat_plan(3, 2) == [2]
    with pytest.raises(SystemExit):
        module.repeat_plan(3, 6)
    with pytest.raises(SystemExit):
        module.repeat_plan(3, 0)
    with pytest.raises(SystemExit):
        module.repeat_plan(0, None)


def test_greedy_effective_vector_matches_the_engine_contract() -> None:
    """requested, wire and applied must not be confusable under greedy."""
    module = _load_module()
    assert module.GREEDY_TEMPERATURE_EPSILON == 1e-5
    assert module.GREEDY_TOP_K == 1
    canonical, _ = module.resolve_profile("greedy", {})
    effective, non_applied = module.effective_sampling_vector(canonical)
    assert effective == {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0}
    assert non_applied == ["top_k"]          # requested 0 (disabled) vs applied 1
    assert module.sampling_request_fields(canonical) == {"temperature": 0}
    overridden, _ = module.resolve_profile("greedy", {"top_k": 40})
    effective, non_applied = module.effective_sampling_vector(overridden)
    assert effective == {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0}
    assert non_applied == ["top_k"]
    assert module.sampling_request_fields(overridden) == {"temperature": 0}


def test_greedy_epsilon_and_top_k_trigger_match_the_engine() -> None:
    module = _load_module()
    assert module.is_greedy_vector({"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0})
    assert module.is_greedy_vector({"temperature": 1e-6, "top_p": 1.0, "top_k": 0, "min_p": 0.0})
    assert module.is_greedy_vector({"temperature": 0.7, "top_p": 1.0, "top_k": 1, "min_p": 0.0})
    assert not module.is_greedy_vector({"temperature": 0.7, "top_p": 1.0, "top_k": 40, "min_p": 0.0})
    # temperature below epsilon: filters are inert and the body stays temperature-only
    assert module.sampling_request_fields(
        {"temperature": 1e-6, "top_p": 0.9, "top_k": 40, "min_p": 0.05}) == {"temperature": 1e-6}
    # top_k == 1 is the greedy trigger itself and MUST be transmitted
    assert module.sampling_request_fields(
        {"temperature": 0.7, "top_p": 0.9, "top_k": 1, "min_p": 0.0}) == {
        "temperature": 0.7, "top_p": 0.9, "top_k": 1, "min_p": 0.0}


def test_weighted_metric_ignores_partial_timing_rows() -> None:
    """A row with finish but no first-output timestamp is excluded, not fatal."""
    module = _load_module()
    samples = [
        _sample(1.0, 11, 1.5, 0.5),
        {"weight": 1.0, "usage": {"completion_tokens": 99}, "finish_seconds": 9.0},
    ]
    metric = module.weighted_decode_metric(samples)
    assert metric["weighted_cases"] == 1
    assert metric["weighted_observed_decode_tokens_per_second"] == pytest.approx(10.0)


def test_min_p_default_text_matches_the_strict_encoding() -> None:
    module = _load_module()
    text = module.SERVER_DEFAULTS_IF_UNSET["min_p"]
    assert "strict harness" in text
    assert "not a request field" not in text
    assert "does not yet accept" not in text


def test_fixed_length_applies_to_weighted_cases_only() -> None:
    """Counting keeps its corpus budget so count_to=200 is not truncated."""
    module = _load_module()
    assert module.case_decode_limits(256, 1.0, None, False) == (256, 256, True)
    assert module.case_decode_limits(256, 0.5, None, False) == (256, 256, True)
    # unweighted diagnostic: no fixed pin, no forced min_tokens, natural EOS
    assert module.case_decode_limits(256, 0.0, None, False) == (None, None, False)
    # explicit operator flags still apply everywhere
    assert module.case_decode_limits(256, 0.0, 100, False) == (None, 100, False)
    assert module.case_decode_limits(256, 0.0, None, True) == (None, None, True)
    assert module.case_decode_limits(None, 1.0, None, False) == (None, None, False)


def test_publishability_is_weighted_only_and_fail_closed() -> None:
    """A failing weight-0 diagnostic must not block the weighted comparison."""
    module = _load_module()

    def s(weight, passed):
        return {"weight": weight, "passed": passed}

    assert module.profile_publishable([s(1.0, True), s(0.5, True)]) is True
    assert module.profile_publishable([s(1.0, True), s(1.0, False)]) is False
    assert module.profile_publishable([]) is False
    assert module.profile_publishable([s(0.0, True)]) is False   # diagnostics only
    samples = [s(1.0, True), s(1.0, True), s(0.0, False)]
    assert module.profile_publishable(samples) is True
    assert module.diagnostics_passed(samples) is False
    assert module.all_samples_passed(samples) is False
    # fail-closed: only a literal True counts as a pass
    assert module.profile_publishable([s(1.0, "False")]) is False
    assert module.profile_publishable([s(1.0, 1)]) is False


def test_diagnostics_passed_defaults_true_without_diagnostics() -> None:
    module = _load_module()
    assert module.diagnostics_passed([{"weight": 1.0, "passed": True}]) is True
    assert module.diagnostics_passed([]) is True
    assert module.diagnostics_passed([{"weight": 0.0, "passed": True}]) is True
    assert module.diagnostics_passed([{"weight": 0.0, "passed": False}]) is False


def test_over_budget_cases_compares_effective_min_and_max() -> None:
    module = _load_module()
    assert module.over_budget_cases({}) == {}
    assert module.over_budget_cases({"a": (None, 50)}) == {}
    assert module.over_budget_cases({"a": (100, 100)}) == {}
    assert module.over_budget_cases({"a": (100, 101)}) == {}
    assert module.over_budget_cases({"a": (101, 100)}) == {"a": (101, 100)}
    assert module.over_budget_cases(
        {"code": (1000, 32), "counting": (None, 640)}) == {"code": (1000, 32)}


def test_effective_decode_limits_honours_the_fixed_override() -> None:
    """A fixed length overrides --min-tokens, so it is not a budget violation."""
    module = _load_module()
    corpus = {
        "weighted_case_ids": ["hello"],
        "cases": {"hello": {"weight": 1.0, "max_tokens": 32}},
        "counting": {"max_tokens": 640},
        "orchid": {"max_tokens": 1500},
    }
    # fixed 200 + operator min 300: effective min is the fixed 200, so (200, 200)
    assert module.effective_decode_limits(
        corpus, ["hello"], 200, 300, False, False, False) == {"hello": (200, 200)}
    assert module.over_budget_cases(
        module.effective_decode_limits(corpus, ["hello"], 200, 300, False, False, False)) == {}
    # no fixed: min 1000 exceeds the corpus 32 -> flagged
    assert module.effective_decode_limits(
        corpus, ["hello"], None, 1000, False, False, False) == {"hello": (1000, 32)}
    # diagnostics never take the fixed budget
    assert module.effective_decode_limits(
        corpus, ["hello"], 200, None, False, True, False) == {
        "hello": (200, 200), "counting": (None, 640)}
    # an operator min above a diagnostic budget is still caught
    assert module.over_budget_cases(module.effective_decode_limits(
        corpus, ["hello"], 200, 1000, False, True, False)) == {"counting": (1000, 640)}
