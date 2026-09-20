"""Adversarial CPU-only tests for the TP/EP E2E runner and corpus.

No GPU, service or network. These pin the review fixes: full-corpus/matrix
cardinality in both directions, structured matched identity, SSE chunk-latency
naming, and the rule that divergence is never auto-qualified by unrelated
component-level kernel evidence.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "qualify-ds41-tp-ep-e2e.py"
CORPUS = ROOT / "scripts" / "fixtures" / "tp-ep-e2e-corpus.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_tp_ep_e2e", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load()


def _row(index: int, text: str) -> dict:
    events = [dict(seconds=0.05 + i * 0.02, event=dict(choices=[dict(delta=dict(content="x"))])) for i in range(8)]
    return dict(index=index, passed=True, start=1000.0 + index, ttft_ms=50.0, first_output_ms=50.0,
                reasoning_ms=0.0, finish_ms=210.0, completion_tokens=8, prompt_tokens=16,
                observed_decode_tokens_per_second=46.9,
                inter_chunk=mod.inter_chunk_stats(dict(events=events)), text=text, reasoning="",
                content_check=dict(response_nonempty=True, objective_checks_passed=True))


def _identity(tp: int) -> dict:
    topology = ["--spark-tp", str(tp), "--spark-ep", str(4 // tp)]
    workers = [["expertd-native", "--rank", str(rank), "--world", "4", "--capacity", "4096",
                "--device-budget-bytes", "107374182400", "--first-layer", "11"] + topology
               for rank in range(4)]
    base = dict(source_commit="abc", source_artifact="manifest:1", model="deepseek-ai/DeepSeek-V4.1-Flash",
                revision="dba1be0a", rtx_gpus=2, requested_rtx_expert_layers="auto",
                resolved_rtx_expert_layers=11, spark_first_layer=11,
                spark_count=4, concurrency=16, max_context_tokens=1048576, max_output_tokens=393216,
                kv_pool="default", prefix_cache_entries=20, prefill_batch_tokens=2048,
                worker_capacity=4096,
                dspark="on", dspark_draft_limit=7, spark_device_budget_bytes=107374182400, power_caps_watts=400,
                source_snapshots={"daemon": "d1", "native": "n1"},
                expert_tp_manifests={"spark": {"tp_degree": 4, "intermediate": 576,
                                               "native_library_sha256": "h-spark"}},
                coordinator_argv=["serve-native", "--rtx-gpus", "2", "--prefill-batch-tokens", "2048",
                                  "--concurrency", "16", "--prefix-cache-entries", "20",
                                  "--max-context-tokens", "1048576", "--max-output-tokens", "393216", "--dspark"] + topology,
                worker_argv=workers)
    base.update(spark_tp=tp, spark_ep=4 // tp, spark_artifact_roles=["spark"] if tp == 4 else ["spark_tp2"],
                release_identity_full=f"fp-tp{tp}")
    if tp == 2:
        base["expert_tp_manifests"]["spark_tp2"] = {"tp_degree": 2, "intermediate": 1152,
                                                    "native_library_sha256": "h-tp2"}
    return base


def _arm(tp: int, text: str = "1,2,3,4") -> dict:
    matrix = dict(concurrency=[1, 2], repeats=3)
    return dict(arm=f"tp{tp}", corpus_sha256="deadbeef", matrix=matrix, startup_metadata=_identity(tp),
                probe=dict(passed=True, content="4", system_fingerprint="fp-engine"),
                cases=[dict(id="counting-1-64", category="counting", max_tokens=160, greedy_repeat_consistent=True,
                            records=[dict(concurrency=c, repeat=r + 1, warmup_passed=True,
                                          rows=[_row(i, text) for i in range(c)])
                                     for c in matrix["concurrency"] for r in range(matrix["repeats"])])],
                passed=True, failures=[])


def _add_second_case(arm: dict, text: str) -> None:
    case = copy.deepcopy(arm["cases"][0])
    case["id"] = "code-merge-intervals"
    for record in case["records"]:
        for row in record["rows"]:
            row["text"] = text
    arm["cases"].append(case)


def test_identity_fixture_supplies_every_required_field() -> None:
    identity = _identity(4)
    for field in mod.REQUIRED_IDENTITY:
        assert field in identity, field


def test_text_exact_is_labeled_text_exact_not_bit_exact() -> None:
    report = mod.compare_arms(_arm(4), _arm(2))
    assert report["gate_result"] == "text_exact"
    assert report["passed"] is True
    assert "bit_exact" not in json.dumps(report["gate_result"])


def test_empty_candidate_cases_cannot_pass_even_when_arm_passed_is_true() -> None:
    candidate = _arm(2)
    candidate["cases"] = []
    assert candidate["passed"] is True
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("no cases" in problem for problem in report["coverage"]["problems"])


def test_missing_case_in_candidate_fails() -> None:
    control, candidate = _arm(4), _arm(2)
    _add_second_case(control, "1,2,3,4")
    report = mod.compare_arms(control, candidate)
    assert report["gate_result"] == "FAIL"
    assert any("case sets differ" in problem for problem in report["coverage"]["problems"])


def test_extra_case_in_candidate_fails() -> None:
    control, candidate = _arm(4), _arm(2)
    _add_second_case(candidate, "1,2,3,4")
    report = mod.compare_arms(control, candidate)
    assert report["gate_result"] == "FAIL"
    assert any("case sets differ" in problem for problem in report["coverage"]["problems"])


def test_duplicate_matrix_record_fails() -> None:
    candidate = _arm(2)
    candidate["cases"][0]["records"].append(copy.deepcopy(candidate["cases"][0]["records"][0]))
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("duplicate" in problem for problem in report["coverage"]["problems"])


def test_index_hole_fails() -> None:
    candidate = _arm(2)
    candidate["cases"][0]["records"][0]["rows"] = []
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("index set" in problem for problem in report["coverage"]["problems"])


def test_missing_startup_metadata_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"] = None
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("startup_metadata missing" in problem for problem in report["identity"]["problems"])


def test_required_identity_mismatch_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["kv_pool"] = "other"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert "kv_pool" in report["identity"]["mismatched_fields"]


def test_allowed_topology_difference_is_recorded_not_mismatched() -> None:
    report = mod.compare_arms(_arm(4), _arm(2))
    assert report["identity"]["mismatched_fields"] == []
    allowed = report["identity"]["allowed_differences"]
    assert allowed["spark_tp"] == {"control": 4, "candidate": 2}
    assert allowed["spark_artifact_roles"]["candidate"] == ["spark_tp2"]


def test_irrelevant_kernel_evidence_cannot_qualify_divergence() -> None:
    report = mod.compare_arms(_arm(4), _arm(2, "1,2,3,5"),
                              kernel_evidence={"records": [dict(rel_l2=0.001, cosine=0.99999)]})
    assert report["gate_result"] == "text_divergent_needs_review"
    assert report["passed"] is False
    assert report["text"]["divergent_requests"] == 9


def test_specific_review_evidence_qualifies_divergence() -> None:
    reviews = {"reviews": [dict(case="counting-1-64", concurrency=c, repeat=r, request=i,
                                kind="matched_logits", passed=True, evidence="logits/out.json")
                           for c in (1, 2) for r in (1, 2, 3) for i in range(c)]}
    report = mod.compare_arms(_arm(4), _arm(2, "1,2,3,5"), review_evidence=reviews)
    assert report["gate_result"] == "text_divergent_review_qualified"
    assert report["passed"] is True
    assert report["review"]["uncovered"] == []


def test_partial_or_wrong_kind_review_stays_unqualified() -> None:
    partial = {"reviews": [dict(case="counting-1-64", concurrency=1, repeat=1, request=0,
                                kind="manual_review", passed=True, evidence="one output")]}
    report = mod.compare_arms(_arm(4), _arm(2, "1,2,3,5"), review_evidence=partial)
    assert report["gate_result"] == "text_divergent_needs_review"
    assert report["review"]["uncovered"]
    wrong_kind = {"reviews": [dict(case="counting-1-64", concurrency=c, repeat=r, request=i,
                                   kind="kernel_oracle", passed=True, evidence="x")
                              for c in (1, 2) for r in (1, 2, 3) for i in range(c)]}
    assert mod.compare_arms(_arm(4), _arm(2, "1,2,3,5"),
                            review_evidence=wrong_kind)["gate_result"] == "text_divergent_needs_review"


def test_latency_is_labeled_inter_chunk_not_inter_token() -> None:
    report = mod.compare_arms(_arm(4), _arm(2))
    row = report["metrics"]["candidate"]["counting-1-64|C1"]
    assert "inter_chunk_ms" in row and row["inter_chunk_ms"]["n"] >= 1
    assert "inter_token_ms" not in row
    assert "multiple tokens" in mod.inter_chunk_stats({"events": []})["definition"]
    assert _row(0, "x")["inter_chunk"]["samples"] == 7


def test_corpus_validates_and_rejects_adversarial_records() -> None:
    good = mod.load_corpus(CORPUS)
    assert mod.validate_corpus(good)["passed"]
    bad = list(good) + [(999, {"kind": "not-a-kind"})]
    assert not mod.validate_corpus(bad)["passed"]
    duplicate = list(good)
    duplicate.append((1000, copy.deepcopy(next(obj for _, obj in good if obj["kind"] == "case"))))
    assert any("duplicate" in e for e in mod.validate_corpus(duplicate)["errors"])
    mutated = []
    for lineno, obj in good:
        obj = copy.deepcopy(obj)
        if obj.get("kind") == "case":
            obj["prompt"] = ""
        mutated.append((lineno, obj))
    assert any("prompt required" in e for e in mod.validate_corpus(mutated)["errors"])
    no_matrix = [(lineno, obj) for lineno, obj in good if obj.get("kind") != "matrix"]
    assert any("missing matrix" in e for e in mod.validate_corpus(no_matrix)["errors"])


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([self._body])


def _mock_record(text: str, tokens: int = 6) -> dict:
    events = [dict(seconds=0.05 + i * 0.02, event=dict(choices=[dict(delta=dict(content="x"))]))
              for i in range(tokens)]
    return dict(text=text, reasoning="", first_output_seconds=0.05, first_content_seconds=0.05,
                finish_seconds=0.05 + 0.02 * tokens, finish_reason="stop",
                observed_decode_tokens_per_second=(tokens - 1) / (0.02 * tokens),
                usage=dict(prompt_tokens=10, completion_tokens=tokens, total_tokens=10 + tokens,
                           prompt_cache_hit_tokens=10),
                events=events)


@contextmanager
def _mock_api(monkeypatch, *, texts=("1,2,3,4",), fail_payload_at=None):
    """Replace the real client with deterministic CPU mocks and restore it."""
    original = dict(mod.API)
    state = dict(payload_calls=0, stream_calls=0)

    def payload(prompt, stream=False):
        state["payload_calls"] += 1
        if fail_payload_at is not None and state["payload_calls"] == fail_payload_at:
            raise RuntimeError("mock interruption")
        return dict(model="m", messages=[], temperature=0, max_tokens=96,
                    thinking=dict(type="disabled"), stream=stream)

    def open_request(base, body, api_key=None):
        return _FakeResponse(dict(choices=[dict(message=dict(content="4"), finish_reason="stop")],
                                  model="m", system_fingerprint="fp-mock"))

    def stream_case(base, body, cancel=False, api_key=None, on_first_content=None):
        index = state["stream_calls"]
        state["stream_calls"] += 1
        return _mock_record(texts[min(index, len(texts) - 1)])

    mod.API.update(payload=payload, open_request=open_request, stream_case=stream_case)
    monkeypatch.setattr(mod, "CHECK_OUTPUT", lambda case, text: dict(
        response_nonempty=bool(text.strip()), objective_checks=None, objective_checks_passed=True))
    try:
        yield state
    finally:
        mod.API.clear()
        mod.API.update(original)


def _run_args(tmp_path, output: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(corpus=CORPUS, arm="tp4ep1", base_url="http://mock", output=output,
                                 startup_metadata=None, case=["counting-1-64"],
                                 concurrency=[1], repeats=3)


def test_mock_run_retains_events_raw_usage_and_per_repeat_summary(monkeypatch, tmp_path) -> None:
    with _mock_api(monkeypatch, texts=("1,2,3,4",)):
        return_code = mod.cmd_run(_run_args(tmp_path, tmp_path / "arm.json"))
    report = json.loads((tmp_path / "arm.json").read_text())
    assert return_code == 1  # no --startup-metadata captured
    case = report["cases"][0]
    assert len(case["records"]) == 3
    for record in case["records"]:
        row = record["rows"][0]
        assert row["passed"] and row["events"] and row["usage"]["completion_tokens"] == 6
    # Per-repeat samples, then stats over the repeats (not one multi-period span).
    assert case["summary"][0]["aggregate_decode_tps"]["n"] == 3
    assert len(case["summary"][0]["per_repeat_aggregate_decode_tps"]) == 3
    assert case["greedy_repeat_consistent"] is True


def test_mock_run_retains_partial_case_on_interruption(monkeypatch, tmp_path) -> None:
    output = tmp_path / "arm.json"
    # payload call 1 is the probe, 2 the warmup, 3 repeat 1, 4 repeat 2 -> raise.
    with _mock_api(monkeypatch, texts=("1,2,3,4",), fail_payload_at=4):
        with pytest.raises(RuntimeError):
            mod.cmd_run(_run_args(tmp_path, output))
    report = json.loads(output.read_text())
    assert len(report["cases"]) == 1
    assert report["cases"][0]["records"], "the in-progress case must retain completed records"
    assert len(report["cases"][0]["records"][0]["rows"]) == 1


def test_mock_run_nondeterministic_repeats_fail(monkeypatch, tmp_path) -> None:
    with _mock_api(monkeypatch, texts=("1,2,3,4", "1,2,3,4", "1,2,3,5", "1,2,3,5")):
        mod.cmd_run(_run_args(tmp_path, tmp_path / "arm.json"))
    report = json.loads((tmp_path / "arm.json").read_text())
    assert report["passed"] is False
    assert any("changed across repeats" in failure for failure in report["failures"])
    assert report["cases"][0]["greedy_repeat_consistent"] is False


def test_single_request_aggregate_span_is_supported() -> None:
    assert mod.aggregate_tps([_row(0, "x")]) is not None
    assert mod.aggregate_tps([]) is None


def test_duplicate_case_id_in_arm_fails_compare() -> None:
    candidate = _arm(2)
    candidate["cases"].append(copy.deepcopy(candidate["cases"][0]))
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("duplicate case ids" in problem for problem in report["coverage"]["problems"])


def test_compare_recomputes_repeat_consistency_even_if_flag_lies() -> None:
    candidate = _arm(2)
    record = next(r for r in candidate["cases"][0]["records"] if r["concurrency"] == 1 and r["repeat"] == 2)
    record["rows"][0]["text"] = "1,2,3,9"
    assert candidate["cases"][0]["greedy_repeat_consistent"] is True
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("changed across repeats" in failure for failure in report["tier_a_failures"])


def test_candidate_expert_family_superset_is_allowed() -> None:
    report = mod.compare_arms(_arm(4), _arm(2))
    family = report["identity"]["family"]
    assert family["problems"] == []
    assert family["shared"] == ["spark"]
    assert family["added_in_candidate"] == ["spark_tp2"]
    assert report["passed"] is True


def test_shared_expert_family_hash_mismatch_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["expert_tp_manifests"]["spark"]["native_library_sha256"] = "other"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("spark" in problem and "native_library_sha256" in problem for problem in report["identity"]["problems"])


def test_missing_source_snapshots_fails() -> None:
    candidate = _arm(2)
    del candidate["startup_metadata"]["source_snapshots"]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("source_snapshots" in problem for problem in report["identity"]["problems"])


def test_declared_control_absent_from_argv_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["coordinator_argv"] = [
        flag for flag in candidate["startup_metadata"]["coordinator_argv"] if flag != "--prefill-batch-tokens"
    ]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--prefill-batch-tokens" in failure for failure in report["tier_a_failures"])


def test_coordinator_flag_value_mismatch_rejects() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["coordinator_argv"]
    argv[argv.index("--prefill-batch-tokens") + 1] = "4096"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--prefill-batch-tokens=4096" in failure for failure in report["tier_a_failures"])


def test_worker_rank_coverage_required() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["worker_argv"] = candidate["startup_metadata"]["worker_argv"][:-1]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("do not cover" in failure for failure in report["tier_a_failures"])


def test_worker_budget_value_mismatch_rejects() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["worker_argv"][0]
    argv[argv.index("--device-budget-bytes") + 1] = "1"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--device-budget-bytes=1" in failure for failure in report["tier_a_failures"])


def test_dspark_presence_must_match_metadata() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["coordinator_argv"] = [
        flag for flag in candidate["startup_metadata"]["coordinator_argv"] if flag != "--dspark"
    ]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--dspark presence" in failure for failure in report["tier_a_failures"])


def test_worker_argv_must_be_per_rank() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["worker_argv"] = candidate["startup_metadata"]["worker_argv"][0]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("per-rank list" in failure for failure in report["tier_a_failures"])


def test_mock_run_marks_rate_approximate_and_ttft_basis(monkeypatch, tmp_path) -> None:
    with _mock_api(monkeypatch, texts=("1,2,3,4",)):
        mod.cmd_run(_run_args(tmp_path, tmp_path / "arm.json"))
    row = json.loads((tmp_path / "arm.json").read_text())["cases"][0]["records"][0]["rows"][0]
    assert row["observed_decode_tokens_per_second_approximate"] is True
    assert "assumes one token in the first output chunk" in row["observed_decode_tokens_per_second_basis"]
    assert "OVERestimates" in row["observed_decode_tokens_per_second_basis"]
    assert "lower-bound" not in row["observed_decode_tokens_per_second_basis"]
    assert row["content_chunks"] == row["usage"]["completion_tokens"]
    assert row["ttft_excludes_reasoning"] is True


def test_requested_auto_resolved_actual_same_both_arms_pass() -> None:
    report = mod.compare_arms(_arm(4), _arm(2))
    assert report["gate_result"] == "text_exact"
    assert report["identity"]["mismatched_fields"] == []
    assert report["tier_a_failures"] == []


def test_resolved_actual_boundary_mismatch_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["resolved_rtx_expert_layers"] = 12
    candidate["startup_metadata"]["spark_first_layer"] = 12
    for argv in candidate["startup_metadata"]["worker_argv"]:
        argv[argv.index("--first-layer") + 1] = "12"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert "resolved_rtx_expert_layers" in report["identity"]["mismatched_fields"]


def test_missing_resolved_actual_fails() -> None:
    candidate = _arm(2)
    del candidate["startup_metadata"]["resolved_rtx_expert_layers"]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("resolved_rtx_expert_layers" in failure for failure in report["identity"]["missing_fields"])


def test_explicit_rtx_expert_layers_argument_must_match_resolved() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["coordinator_argv"] += ["--rtx-expert-layers", "12"]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--rtx-expert-layers=12" in failure for failure in report["tier_a_failures"])


def test_spark_first_layer_must_equal_min_resolved_39() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["spark_first_layer"] = 5
    for argv in candidate["startup_metadata"]["worker_argv"]:
        argv[argv.index("--first-layer") + 1] = "5"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("min(resolved_rtx_expert_layers, 39)" in failure for failure in report["tier_a_failures"])


def test_duplicate_rank_argv_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["worker_argv"].append(
        copy.deepcopy(candidate["startup_metadata"]["worker_argv"][0])
    )
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("rank argvs but world" in failure or "repeats a rank" in failure
               for failure in report["tier_a_failures"])


def test_capacity_below_prefill_bound_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["worker_capacity"] = 1024
    for argv in candidate["startup_metadata"]["worker_argv"]:
        argv[argv.index("--capacity") + 1] = "1024"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("below the 4096 required" in failure for failure in report["tier_a_failures"])


def test_unsupported_capacity_fails() -> None:
    candidate = _arm(2)
    candidate["startup_metadata"]["worker_capacity"] = 128
    for argv in candidate["startup_metadata"]["worker_argv"]:
        argv[argv.index("--capacity") + 1] = "128"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("not one of" in failure for failure in report["tier_a_failures"])


def test_capacity_flag_value_must_match_metadata() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["worker_argv"][0]
    argv[argv.index("--capacity") + 1] = "1024"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--capacity=1024" in failure for failure in report["tier_a_failures"])


CANONICAL_CODE = (
    "```python\n"
    "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
    '    """Merge overlapping closed integer intervals."""\n'
    "    if not intervals:\n"
    "        return []\n"
    "    ordered = sorted(intervals)\n"
    "    merged = [list(ordered[0])]\n"
    "    for start, end in ordered[1:]:\n"
    "        if start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return [tuple(pair) for pair in merged]\n"
    "\n"
    "assert merge_intervals([]) == []\n"
    "assert merge_intervals([(1, 3), (2, 6)]) == [(1, 6)]\n"
    "assert merge_intervals([(1, 2), (4, 5), (3, 4)]) == [(1, 2), (3, 5)]\n"
    "```"
)


def test_canonical_code_answer_passes_content_check() -> None:
    result = mod.content_check({"kind": "throughput_case", "case": "code"}, CANONICAL_CODE)
    assert result["objective_checks_passed"] is True


def test_complete_but_wrong_code_answer_fails() -> None:
    answer = "```python\ndef merge_intervals(intervals):\n    return intervals\n```"
    result = mod.content_check({"kind": "throughput_case", "case": "code"}, answer)
    assert result["objective_checks_passed"] is False


def test_code_prompt_states_the_check_requirements() -> None:
    cases = {obj["id"]: obj for _, obj in mod.load_corpus(CORPUS) if obj.get("kind") == "case"}
    prompt = cases["code-merge-intervals"]["prompt"].casefold()
    assert "type hint" in prompt and "docstring" in prompt and "assert" in prompt
    assert cases["code-merge-intervals"]["max_tokens"] >= 512
    assert cases["counting-1-64"]["max_tokens"] >= 256


def test_concurrency_value_mismatch_rejects() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["coordinator_argv"]
    argv[argv.index("--concurrency") + 1] = "8"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--concurrency=8" in failure for failure in report["tier_a_failures"])


def test_max_context_value_mismatch_rejects() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["coordinator_argv"]
    argv[argv.index("--max-context-tokens") + 1] = "524288"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--max-context-tokens=524288" in failure for failure in report["tier_a_failures"])


def test_max_output_value_mismatch_rejects() -> None:
    candidate = _arm(2)
    argv = candidate["startup_metadata"]["coordinator_argv"]
    argv[argv.index("--max-output-tokens") + 1] = "1024"
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("--max-output-tokens=1024" in failure for failure in report["tier_a_failures"])


def test_missing_concurrency_field_fails() -> None:
    candidate = _arm(2)
    del candidate["startup_metadata"]["concurrency"]
    report = mod.compare_arms(_arm(4), candidate)
    assert report["gate_result"] == "FAIL"
    assert any("concurrency" in failure for failure in report["identity"]["missing_fields"])
