"""CPU tests for the additive sampling-campaign raw validator.

These lock the publication gates the campaign relies on and run against the REAL
frozen corpus fixture (weights, natural budgets and counting budget are read from
scripts/fixtures/release-semantic-corpus.json, never invented): completeness
(5x3, no duplicates), cross-file identity, mandatory pinned identity file with
schema (1 RTX + 4 Spark + dspark, immutable local image ids) and deep equality,
exact canonical per-profile vectors and effective/non-applied fields, per-sample
request sampling/seed/max_tokens for weighted AND counting, explicit stochastic
seed, natural budgets, exactly one weight-0 counting diagnostic, quality,
recomputed summary numbers and n=3 medians, true cyclic left rotation from raw
timestamps, mandatory corpus pinning, diagnostics disclosed separately, and
length spreads. No server, GPU or network.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "scripts" / "validate-sampling-campaign.py"
CORPUS_PATH = REPO / "scripts" / "fixtures" / "release-semantic-corpus.json"

CORPUS = json.loads(CORPUS_PATH.read_text())
CORPUS_SHA = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
PROFILES = ("greedy", "temp0.2-topp0.95", "temp0.7-topp0.9",
            "temp0.7-minp0.05", "temp0.7-topk40")
WEIGHTED = list(CORPUS["weighted_case_ids"])
WEIGHTS = {case: CORPUS["cases"][case]["weight"] for case in WEIGHTED}
BUDGETS = {case: CORPUS["cases"][case]["max_tokens"] for case in WEIGHTED}
COUNTING_BUDGET = CORPUS["counting"]["max_tokens"]
ORCHID_BUDGET = CORPUS["orchid"]["max_tokens"]
REQUEST_FIELDS = {
    "greedy": {"temperature": 0},
    "temp0.2-topp0.95": {"temperature": 0.2, "top_p": 0.95, "top_k": 0, "min_p": 0.0},
    "temp0.7-topp0.9": {"temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.0},
    "temp0.7-minp0.05": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.05},
    "temp0.7-topk40": {"temperature": 0.7, "top_p": 1.0, "top_k": 40, "min_p": 0.0},
}
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SEED = 20260922
NONCE_SEED = 79001
# Fixed per profile across its repeats, distinct across profiles (cache safety).
NONCE_BY_PROFILE = {profile: 79101 + index for index, profile in enumerate(PROFILES)}
SEGMENT_SECONDS = 0.9

IDENTITY = {
    "schema": "ds41rt.sampling-identity/1",
    "model": {"id": MODEL, "revision": "dba1be0a40aa45a94ad051997016db3960a90277"},
    "topology": {"rtx_coordinators": 1, "spark_workers": 4, "dspark": True},
    "coordinator": {
        "host": "raptor",
        "gpus": [{"index": 0, "name": "RTX PRO 6000 Blackwell", "uuid": "GPU-abc"}],
        "image": {"id": "sha256:coordinator-local", "source": "local-build",
                  "version": "v11-candidate"},
    },
    "spark_workers": [
        {"host": host, "image": {"id": f"sha256:spark-{host}", "source": "local-build",
                                 "version": "v11-candidate"}}
        for host in ("ostrich", "dodo", "emu", "kiwi")
    ],
}
IDENTITY_RAW = json.dumps(IDENTITY, indent=2).encode()
IDENTITY_SHA = hashlib.sha256(IDENTITY_RAW).hexdigest()


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_validate_campaign", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(documents, **kwargs):
    module = kwargs.pop("module", None) or _load()
    kwargs.setdefault("identity_document", IDENTITY)
    kwargs.setdefault("identity_sha256", IDENTITY_SHA)
    kwargs.setdefault("expect_identity_sha", IDENTITY_SHA)
    kwargs.setdefault("corpus", CORPUS)
    kwargs.setdefault("corpus_sha256", CORPUS_SHA)
    return module.validate(documents, **kwargs)


def _profile_request(profile, model, max_tokens):
    request = {"model": model, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    request.update(REQUEST_FIELDS[profile])
    if profile != "greedy":
        request["seed"] = SEED
    return request


def _report(profile, repeat, start_ns, completion_tokens=11, diagnostics_passed=True,
            weighted_passed=True, corpus_sha=CORPUS_SHA, identity=None, identity_sha=None,
            nonce_seed=None, repeats=3, encoding="strict", extra_diagnostics=(),
            model=MODEL):
    greedy = profile == "greedy"
    request_fields = REQUEST_FIELDS[profile]
    effective = ({"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0}
                 if greedy else dict(request_fields))
    identity = IDENTITY if identity is None else copy.deepcopy(identity)
    identity_sha = IDENTITY_SHA if identity_sha is None else identity_sha
    nonce_seed = NONCE_BY_PROFILE[profile] if nonce_seed is None else nonce_seed
    samples = []
    ns = start_ns
    for case in WEIGHTED:
        samples.append({
            "case": case, "weight": WEIGHTS[case], "started_ns": ns,
            "finish_seconds": ns / 1e9 + SEGMENT_SECONDS, "first_output_seconds": ns / 1e9,
            "usage": {"completion_tokens": completion_tokens}, "passed": weighted_passed,
            "request": _profile_request(profile, model, BUDGETS[case]),
            "requested_max_tokens": BUDGETS[case], "requested_min_tokens": None,
            "requested_ignore_eos": False, "requested_decode_tokens": None,
        })
        ns += 1000
    for extra in extra_diagnostics:
        budget = ORCHID_BUDGET if extra == "orchid" else BUDGETS.get(extra, COUNTING_BUDGET)
        samples.append({
            "case": extra, "weight": 0.0, "started_ns": ns,
            "finish_seconds": ns / 1e9 + SEGMENT_SECONDS, "first_output_seconds": ns / 1e9,
            "usage": {"completion_tokens": 201}, "passed": True,
            "request": _profile_request(profile, model, budget),
            "requested_max_tokens": budget, "requested_min_tokens": None,
            "requested_ignore_eos": False, "requested_decode_tokens": None,
        })
        ns += 1000
    samples.append({
        "case": "counting", "weight": 0.0, "started_ns": ns,
        "finish_seconds": ns / 1e9 + SEGMENT_SECONDS, "first_output_seconds": ns / 1e9,
        "usage": {"completion_tokens": 201}, "passed": diagnostics_passed,
        "request": _profile_request(profile, model, COUNTING_BUDGET),
        "requested_max_tokens": COUNTING_BUDGET, "requested_min_tokens": None,
        "requested_ignore_eos": False, "requested_decode_tokens": None,
    })
    metric = _metric(samples)
    return {
        "model": model,
        "base_url": "http://127.0.0.1:8000",
        "tokenizer_sha256": "t" * 64,
        "corpus_sha256": corpus_sha,
        "nonce_seed": nonce_seed,
        "repeats": repeats,
        "include_counting": True,
        "include_orchid": bool(extra_diagnostics),
        "selected_cases": list(WEIGHTED),
        "sampling": {
            "profile": profile,
            "requested_vector": ({"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
                                 if greedy else effective),
            "effective_vector": effective,
            "request_fields": request_fields,
            "non_applied_fields": ["top_k"] if greedy else [],
            "seed": None if greedy else SEED,
            "seed_source": "greedy-ignores-seed" if greedy else "explicit",
            "greedy": greedy,
            "encoding": encoding,
        },
        "decode_length_policy": {"fixed_decode_tokens": None},
        "provenance": {"source_revision": "s" * 40, "harness": {"sha256": "h" * 64},
                       "identity": identity, "identity_sha256": identity_sha},
        "samples": samples,
        "repeat_summaries": [{"repeat": repeat, **metric}],
        "weighted_passed": weighted_passed,
        "publishable": weighted_passed,
    }


def _metric(samples):
    weighted = [s for s in samples if s["weight"] > 0]
    timed_tokens = sum(s["weight"] * (s["usage"]["completion_tokens"] - 1) for s in weighted)
    timed_seconds = sum(s["weight"] * (s["finish_seconds"] - s["first_output_seconds"])
                        for s in weighted)
    return {"weighted_cases": len(weighted), "timed_tokens": timed_tokens,
            "timed_seconds": timed_seconds,
            "weighted_observed_decode_tokens_per_second": timed_tokens / timed_seconds}


def campaign(repeats=3, rotation=True, profile_major=False, **report_kwargs):
    counter = itertools.count(1_000_000)
    documents = []

    def add(profile, repeat):
        documents.append((_report(profile, repeat, next(counter),
                                  completion_tokens=10 + repeat, **report_kwargs),
                          "0" * 64, f"{profile}-r{repeat}.json"))

    if profile_major:
        for profile in PROFILES:
            for repeat in range(1, repeats + 1):
                add(profile, repeat)
        return documents
    for repeat in range(1, repeats + 1):
        order = list(PROFILES)
        if rotation:
            offset = repeat - 1
            order = order[offset:] + order[:offset]
        for profile in order:
            add(profile, repeat)
    return documents


def _failures(result):
    return " | ".join(result["failures"])


def test_real_corpus_budgets_are_the_ones_under_test() -> None:
    assert BUDGETS["code-reasoning"] == 4096
    assert BUDGETS["code"] == 320 and BUDGETS["hello"] == 32
    assert BUDGETS["topic"] == 384 and BUDGETS["structured-json"] == 128
    assert COUNTING_BUDGET == 640


def test_valid_rotated_campaign_passes_and_recomputes_the_median() -> None:
    result = _validate(campaign())
    assert result["passed"], _failures(result)
    assert set(result["medians"]) == set(PROFILES)
    # repeats use 11/12/13 completion tokens over 0.9 s -> 10/11/12 / 0.9
    assert result["medians"]["greedy"] == pytest.approx(11 / SEGMENT_SECONDS)
    assert result["repeat_series"]["greedy"] == pytest.approx(
        [10 / SEGMENT_SECONDS, 11 / SEGMENT_SECONDS, 12 / SEGMENT_SECONDS])
    assert result["files_seen"] == 15 and result["files_expected"] == 15
    assert result["rotation"]["rotated"] is True
    assert result["rotation"]["repeats_contiguous"] is True
    assert result["identity"]["deep_equal"] is True
    assert result["identity"]["schema_failures"] == []
    assert result["diagnostics"]["greedy"]["all_passed"] is True


def test_duplicate_report_fails() -> None:
    documents = campaign()
    documents.append(documents[0])
    result = _validate(documents)
    assert not result["passed"]
    assert "duplicate report" in _failures(result)
    assert "expected 15" in _failures(result)


def test_missing_repeat_fails_and_no_partial_median_is_published() -> None:
    documents = [d for d in campaign() if not d[2].endswith("-r3.json")]
    result = _validate(documents)
    assert not result["passed"]
    assert "missing" in _failures(result)
    assert result["medians"] == {}


def test_corpus_is_mandatory() -> None:
    result = _load().validate(campaign(), identity_document=IDENTITY,
                              identity_sha256=IDENTITY_SHA, expect_identity_sha=IDENTITY_SHA,
                              corpus=None, corpus_sha256=None)
    assert not result["passed"]
    assert "corpus fixture is required" in _failures(result)


def test_cross_file_identity_mismatch_fails() -> None:
    documents = campaign()
    documents[0][0]["tokenizer_sha256"] = "x" * 64
    result = _validate(documents)
    assert not result["passed"]
    assert "identity differs" in _failures(result)


def test_expected_identity_sha_mismatch_fails() -> None:
    result = _validate(campaign(), expect_identity_sha="z" * 64)
    assert not result["passed"]
    assert "expected" in _failures(result)


def test_identity_file_sha_must_match_every_report() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["provenance"]["identity_sha256"] = "0" * 64
    result = _validate(documents)
    assert not result["passed"]
    assert "identity_sha256" in _failures(result)


def test_identity_is_required() -> None:
    result = _load().validate(campaign(), identity_document=None, identity_sha256=None,
                              expect_identity_sha=None, corpus=CORPUS, corpus_sha256=CORPUS_SHA)
    assert not result["passed"]
    assert "identity snapshot file is required" in _failures(result)
    assert "--expect-identity-sha is required" in _failures(result)


def test_identity_model_id_must_match_the_report_model() -> None:
    other = copy.deepcopy(IDENTITY)
    other["model"]["id"] = "other-model"
    other_sha = hashlib.sha256(json.dumps(other).encode()).hexdigest()
    # reports keep the real model id while the identity names a different one
    documents = campaign(identity=other, identity_sha=other_sha)
    result = _validate(documents, identity_document=other, identity_sha256=other_sha,
                       expect_identity_sha=other_sha)
    assert not result["passed"]
    assert "identity.model.id" in _failures(result)


def test_host_source_revision_is_not_conflated_with_the_image_source() -> None:
    """Host HEAD and the baked image source are different facts.

    The reports' `provenance.source_revision` is the current host HEAD; the image
    source label (the frozen build, e.g. 9d9b3e0) lives in the identity. The
    validator must require only ONE shared source_revision across the reports and
    must never assume it equals the image source.
    """
    identity = copy.deepcopy(IDENTITY)
    identity["coordinator"]["image"]["source"] = "9d9b3e0"
    for worker in identity["spark_workers"]:
        worker["image"]["source"] = "9d9b3e0"
    sha = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    # reports keep their own host revision ("s" * 40), which is NOT 9d9b3e0
    documents = campaign(identity=identity, identity_sha=sha)
    result = _validate(documents, identity_document=identity, identity_sha256=sha,
                       expect_identity_sha=sha)
    assert result["passed"], _failures(result)


def test_source_revision_must_be_identical_across_reports() -> None:
    documents = campaign()
    documents[0][0]["provenance"]["source_revision"] = "a-different-host-head"
    result = _validate(documents)
    assert not result["passed"]
    assert "identity differs" in _failures(result)


def test_identity_schema_requires_topology_and_images() -> None:
    module = _load()
    missing_dspark = copy.deepcopy(IDENTITY)
    missing_dspark["topology"]["dspark"] = False
    assert any("dspark" in f for f in module.identity_schema_failures(missing_dspark))

    three_sparks = copy.deepcopy(IDENTITY)
    three_sparks["spark_workers"] = three_sparks["spark_workers"][:3]
    assert any("spark_workers" in f for f in module.identity_schema_failures(three_sparks))

    no_image_id = copy.deepcopy(IDENTITY)
    no_image_id["coordinator"]["image"]["id"] = ""
    assert any("image.id" in f for f in module.identity_schema_failures(no_image_id))

    no_labels = copy.deepcopy(IDENTITY)
    del no_labels["spark_workers"][0]["image"]["version"]
    assert any("version" in f for f in module.identity_schema_failures(no_labels))

    two_rtx = copy.deepcopy(IDENTITY)
    two_rtx["topology"]["rtx_coordinators"] = 2
    assert any("rtx_coordinators" in f for f in module.identity_schema_failures(two_rtx))

    assert module.identity_schema_failures(IDENTITY) == []
    assert module.identity_schema_failures({}) == ["identity snapshot is empty"]


def test_identity_deep_equality_survives_file_whitespace() -> None:
    # The pinned hash is the ORIGINAL file's raw hash; the embedded copy only has
    # to be deep-equal, so formatting never causes a false mismatch.
    compact = json.dumps(IDENTITY).encode()
    sha = hashlib.sha256(compact).hexdigest()
    documents = campaign(identity_sha=sha, identity=IDENTITY)
    result = _validate(documents, identity_document=IDENTITY, identity_sha256=sha,
                       expect_identity_sha=sha)
    assert result["passed"], _failures(result)


def test_embedded_identity_must_deep_equal_the_file() -> None:
    tampered = copy.deepcopy(IDENTITY)
    tampered["coordinator"]["host"] = "not-raptor"
    documents = campaign()
    for document, _, _ in documents:
        document["provenance"]["identity"] = copy.deepcopy(tampered)
    result = _validate(documents)
    assert not result["passed"]
    assert "deep-equal" in _failures(result)


def test_non_strict_encoding_fails() -> None:
    result = _validate(campaign(encoding="current-engine"))
    assert not result["passed"]
    assert "encoding" in _failures(result)


def test_canonical_vector_binding_is_exact_per_profile() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topk40-r2.json":
            document["sampling"]["request_fields"] = dict(REQUEST_FIELDS["temp0.7-minp0.05"])
    result = _validate(documents)
    assert not result["passed"]
    assert "canonical" in _failures(result) or "within-profile vector" in _failures(result)


def test_stochastic_effective_vector_and_non_applied_must_be_exact() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-minp0.05-r1.json":
            document["sampling"]["non_applied_fields"] = ["top_k"]
    result = _validate(documents)
    assert not result["passed"]
    assert "non_applied_fields" in _failures(result)


def test_greedy_effective_vector_and_non_applied_must_be_exact() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["sampling"]["non_applied_fields"] = []
    result = _validate(documents)
    assert not result["passed"]
    assert "non_applied_fields" in _failures(result)


def test_within_profile_vector_mismatch_fails_but_cross_profile_difference_is_fine() -> None:
    assert _validate(campaign())["passed"]
    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topk40-r2.json":
            document["sampling"]["request_fields"] = {
                "temperature": 0.7, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    result = _validate(documents)
    assert not result["passed"]
    assert "within-profile vector" in _failures(result)


def test_seed_rules() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topp0.9-r1.json":
            document["sampling"]["seed"] = 7
    assert "share one seed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, _ in documents:
        if document["sampling"]["greedy"]:
            continue
        document["sampling"]["seed"] = None
        document["sampling"]["seed_source"] = "unseeded-non-reproducible"
    assert "explicit non-null seed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["sampling"]["seed"] = 5
            document["sampling"]["seed_source"] = "explicit"
    assert "greedy profile must not carry a seed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topk40-r1.json":
            document["sampling"]["seed_source"] = "unseeded-non-reproducible"
    assert "stochastic seed_source should be 'explicit'" in _failures(_validate(documents))


def test_per_sample_request_seed_and_sampling_must_match_profile() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topk40-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code")["request"]["seed"] = 1
    assert "sample request seed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code")["request"]["seed"] = SEED
    assert "must omit the seed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "temp0.7-topk40-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code")["request"]["top_k"] = 0
    assert "sample request sampling" in _failures(_validate(documents))


def test_request_max_tokens_must_equal_metadata_and_corpus_for_weighted_and_counting() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code-reasoning")["request"]["max_tokens"] = 256
    assert "request.max_tokens" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code-reasoning")["requested_max_tokens"] = 256
    assert "requested_max_tokens" in _failures(_validate(documents))

    # the counting diagnostic is wired on the same natural policy
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "counting")["request"]["max_tokens"] = 256
    assert "request.max_tokens" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            counting = next(s for s in document["samples"] if s["case"] == "counting")
            counting["request"]["min_tokens"] = 10
    assert "min_tokens" in _failures(_validate(documents))


def test_nonce_seed_is_fixed_within_a_profile() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r2.json":
            document["nonce_seed"] = 12345
    failures = _failures(_validate(documents))
    assert "within-profile vector" in failures and "nonce_seed" in failures


def test_nonce_seed_must_be_distinct_across_profiles() -> None:
    documents = campaign()
    for document, _, _ in documents:
        document["nonce_seed"] = NONCE_SEED  # one seed shared by every profile
    result = _validate(documents)
    assert not result["passed"]
    assert "pairwise distinct" in _failures(result)


def _single_profile_documents():
    return [d for d in campaign() if d[2].startswith("greedy-")]


@pytest.mark.parametrize("bad_nonce", [None, [79101], {"seed": 79101}, True, "79101", 79101.0])
def test_single_profile_nonce_must_be_an_int(bad_nonce) -> None:
    """A lone profile must still fail on a non-int nonce (set() of one value passes)."""
    documents = _single_profile_documents()
    for document, _, _ in documents:
        document["nonce_seed"] = bad_nonce
    result = _validate(documents, profiles=("greedy",), repeats=3)
    assert not result["passed"]
    assert "nonce_seed must be an integer" in _failures(result)


def test_single_profile_missing_nonce_fails() -> None:
    documents = _single_profile_documents()
    for document, _, _ in documents:
        document.pop("nonce_seed")
    result = _validate(documents, profiles=("greedy",), repeats=3)
    assert not result["passed"]
    assert "nonce_seed must be an integer" in _failures(result)


def test_repeat_count_mismatch_fails() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r2.json":
            document["repeats"] = 5
    assert "identity differs" in _failures(_validate(documents))


def test_case_order_mismatch_fails() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            samples = document["samples"]
            samples[0], samples[1] = samples[1], samples[0]
    result = _validate(documents)
    assert not result["passed"]
    assert "case order" in _failures(result)


def test_recomputed_summary_numbers_are_gated() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["repeat_summaries"][0][
                "weighted_observed_decode_tokens_per_second"] = 999.0
    assert "recomputed" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["repeat_summaries"][0]["timed_tokens"] = 1.0
    assert "timed_tokens" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["repeat_summaries"][0]["weighted_cases"] = 8
    assert "weighted_cases" in _failures(_validate(documents))


def test_natural_budget_policy_must_be_null() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["decode_length_policy"]["fixed_decode_tokens"] = 256
    result = _validate(documents)
    assert not result["passed"]
    assert "fixed_decode_tokens" in _failures(result)


def test_weighted_sample_must_not_pin_length_or_ignore_eos() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code")["requested_decode_tokens"] = 256
    assert "pins a fixed decode length" in _failures(_validate(documents))

    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"]
                 if s["case"] == "code")["requested_ignore_eos"] = True
    assert "ignore_eos" in _failures(_validate(documents))


def test_exactly_one_weight_zero_diagnostic_required() -> None:
    result = _validate(campaign(extra_diagnostics=("orchid",)))
    assert not result["passed"]
    assert "exactly one weight-0 counting diagnostic" in _failures(result)

    documents = campaign()
    for document, _, _ in documents:
        document["include_counting"] = False
    assert "include_counting" in _failures(_validate(documents))


def test_missing_weighted_case_fails() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            document["samples"] = [s for s in document["samples"] if s["case"] != "math"]
    result = _validate(documents)
    assert not result["passed"]
    assert "weighted cases" in _failures(result)


def test_failed_weighted_sample_fails() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"] if s["case"] == "math")["passed"] = False
            document["weighted_passed"] = False
            document["publishable"] = False
    result = _validate(documents)
    assert not result["passed"]
    assert "not passed" in _failures(result) and "weighted_passed" in _failures(result)


def test_diagnostic_failure_is_disclosed_but_does_not_gate() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            next(s for s in document["samples"] if s["case"] == "counting")["passed"] = False
    result = _validate(documents)
    assert result["passed"], _failures(result)
    assert result["diagnostics"]["greedy"]["all_passed"] is False
    assert result["diagnostics"]["greedy"]["failed_cases"] == ["counting"]


def test_profile_major_ordering_fails() -> None:
    result = _validate(campaign(profile_major=True))
    assert not result["passed"]
    assert "profile-major" in _failures(result)


def test_repeats_must_follow_the_true_cyclic_rotation() -> None:
    result = _validate(campaign(rotation=False))
    assert not result["passed"]
    assert "cyclic rotation" in _failures(result)


def test_missing_started_ns_fails() -> None:
    documents = campaign()
    for document, _, path in documents:
        if path == "greedy-r1.json":
            for sample in document["samples"]:
                sample.pop("started_ns")
    result = _validate(documents)
    assert not result["passed"]
    assert "started_ns" in _failures(result)


def test_length_spreads_are_recorded() -> None:
    result = _validate(campaign())
    spread = result["length_spreads"]["temp0.7-topp0.9"]["code"]
    assert spread == {"samples": 3, "min": 11, "median": 12, "max": 13}


def test_corpus_fixture_pins_sha_weights_and_natural_budgets() -> None:
    documents = campaign()
    assert _validate(documents)["passed"]

    wrong_weights = copy.deepcopy(CORPUS)
    for case in WEIGHTED:
        wrong_weights["cases"][case]["weight"] = 1.0
    result = _validate(documents, corpus=wrong_weights, corpus_sha256=CORPUS_SHA)
    assert not result["passed"]
    assert "corpus fixture" in _failures(result)

    result = _validate(documents, corpus=CORPUS, corpus_sha256="0" * 64)
    assert not result["passed"]
    assert "corpus_sha256" in _failures(result)

    wrong_budget = copy.deepcopy(CORPUS)
    wrong_budget["cases"]["hello"]["max_tokens"] = 256
    result = _validate(documents, corpus=wrong_budget, corpus_sha256=CORPUS_SHA)
    assert not result["passed"]
    assert "corpus budget" in _failures(result)


def _write(tmp_path, documents):
    paths = []
    for document, _, name in documents:
        path = tmp_path / Path(name).name
        path.write_text(json.dumps(document))
        paths.append(str(path))
    return paths


def test_cli_pass_and_fail_exit_codes(tmp_path) -> None:
    identity_path = tmp_path / "identity.json"
    identity_path.write_bytes(IDENTITY_RAW)
    good = _write(tmp_path, campaign())
    output = tmp_path / "validation.json"
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--reports", *good,
         "--identity-file", str(identity_path), "--expect-identity-sha", IDENTITY_SHA,
         "--output", str(output)],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["passed"] is True

    documents = campaign()
    documents[0][0]["model"] = "someone-elses-model"
    bad = _write(tmp_path, documents)
    bad_output = tmp_path / "validation-bad.json"
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--reports", *bad,
         "--identity-file", str(identity_path), "--expect-identity-sha", IDENTITY_SHA,
         "--output", str(bad_output)],
        capture_output=True, text=True)
    assert completed.returncode == 1
    assert json.loads(bad_output.read_text())["passed"] is False


def test_cli_requires_identity_arguments(tmp_path) -> None:
    good = _write(tmp_path, campaign())
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--reports", *good,
         "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert "identity-file" in completed.stderr
