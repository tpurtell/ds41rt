"""CPU-only tests for the isolated six-rank E2E wrapper.

No GPU, service, network or HTTP: the CLI run smoke stubs the frozen runner's
API dict on the wrapper's separately loaded copy, so the four-Spark runner in
another process is unaffected.
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
SIX = ROOT / "scripts" / "qualify-ds41-tp-ep-e2e-six.py"
V4 = ROOT / "scripts" / "fixtures" / "tp-ep-e2e-corpus-v4-six.jsonl"
V3 = ROOT / "scripts" / "fixtures" / "tp-ep-e2e-corpus.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_tp_ep_e2e_six", SIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


six = _load()


def _records():
    return six.load_jsonl(V4)


def _by(records, kind):
    return [record for record in records if record.get("kind") == kind]


def _configs(records=None):
    return {c["id"]: c for c in _by(records or _records(), "config")}


def _comparisons():
    return _by(_records(), "comparison")


def _forbidden():
    return _by(_records(), "forbidden_comparison")


def _errors(records):
    return six.validate_six(records, six.load_jsonl(V3))["errors"]


def _find(records, kind, **match):
    for record in _by(records, kind):
        if all(record.get(key) == value for key, value in match.items()):
            return record
    raise AssertionError(f"no {kind} matching {match}")


def _metadata(config_id: str) -> dict:
    config = _configs()[config_id]
    world = config["world"]
    topology = ["--spark-tp", str(config["spark_tp"]), "--spark-ep", str(config["spark_ep"])]
    requested = config["requested_rtx_expert_layers"]
    rtx_argv = ["--rtx-expert-layers", str(requested)]
    placement = ["--placement-directory", "/run/ds41rt-placement"] if config["require_placement_directory"] else []
    workers = [["expertd-native", "--rank", str(rank), "--world", str(world), "--capacity", "4096",
                "--device-budget-bytes", "109119320064", "--first-layer",
                str(config["expected_spark_first_layer"])] + topology
               for rank in range(world)]
    return dict(
        source_commit="abc", source_artifact="manifest:1", model="deepseek-ai/DeepSeek-V4.1-Flash",
        revision="dba1be0a", rtx_gpus=config["rtx_count"], requested_rtx_expert_layers=requested,
        resolved_rtx_expert_layers=config["expected_resolved_rtx_expert_layers"],
        spark_first_layer=config["expected_spark_first_layer"], spark_count=world,
        spark_tp=config["spark_tp"], spark_ep=config["spark_ep"],
        concurrency=16, max_context_tokens=1048576, max_output_tokens=393216,
        kv_pool="auto", prefix_cache_entries=20, prefill_batch_tokens=2048, worker_capacity=4096,
        dspark="on", dspark_draft_limit=config["dspark_draft_limit"],
        spark_device_budget_bytes=109119320064, power_caps_watts=400,
        source_snapshots={"daemon": "d1", "native": "n1"},
        expert_tp_manifests={
            "spark": {"tp_degree": 4, "intermediate": 576, "native_library_sha256": "h-spark"},
            "spark_tp2": {"tp_degree": 2, "intermediate": 1152, "native_library_sha256": "h-tp2"},
            "spark_tp3": {"tp_degree": 3, "intermediate": 768, "native_library_sha256": "h-tp3"},
        },
        coordinator_argv=["serve-native", "--rtx-gpus", str(config["rtx_count"]), "--prefill-batch-tokens", "2048",
                          "--concurrency", "16", "--prefix-cache-entries", "20",
                          "--max-context-tokens", "1048576", "--max-output-tokens", "393216", "--dspark"]
                         + rtx_argv + placement + topology,
        worker_argv=workers, active_topology=f"TP{config['spark_tp']}EP{config['spark_ep']}",
        release_identity_full=f"fp-{config_id}")


def _row(index, text):
    events = [dict(seconds=0.05 + i * 0.02, event=dict(choices=[dict(delta=dict(content="x"))])) for i in range(6)]
    return dict(index=index, passed=True, start=1000.0 + index, ttft_ms=50.0, first_output_ms=50.0,
                reasoning_ms=0.0, finish_ms=210.0, completion_tokens=6, prompt_tokens=10,
                observed_decode_tokens_per_second=47.6,
                inter_chunk=six.BASE["inter_chunk_stats"](dict(events=events)), text=text, reasoning="",
                content_check=dict(response_nonempty=True, objective_checks_passed=True))


def _full_matrix_arm(config_id: str, text: str = "1,2,3,4") -> dict:
    matrix = dict(concurrency=list(six.CANONICAL_CONCURRENCY), repeats=six.CANONICAL_REPEATS)
    cases = [dict(id=case["id"], category=case.get("category"), max_tokens=case.get("max_tokens"),
                  greedy_repeat_consistent=True,
                  records=[dict(concurrency=c, repeat=r, warmup_passed=True, rows=[_row(i, text) for i in range(c)])
                           for c in matrix["concurrency"] for r in range(1, matrix["repeats"] + 1)])
             for case in _by(_records(), "case")]
    return dict(arm=config_id, corpus_sha256=six.sha256(V4), matrix=matrix,
                startup_metadata=_metadata(config_id),
                probe=dict(passed=True, content="4", system_fingerprint="fp-six"),
                cases=cases, passed=True, failures=[], frozen_runner_sha256=six.FROZEN_RUNNER_SHA256)


def _valid_memory(config_id: str) -> dict:
    config = _configs()[config_id]
    reserve = config["spark_min_reserve_bytes"]
    budget = 130661208064 - reserve
    resident = config["expected_spark_resident_bytes"]
    hosts = [dict(rank=rank, mem_total_bytes=130661208064, device_budget_bytes=budget,
                  mem_available_bytes=reserve + (1 << 30), accounted_resident_bytes=resident,
                  accounted_workspace_bytes=budget - resident - (1 << 30), accounted_ring_bytes=1 << 30,
                  source_log=f"logs/rank{rank}.log") for rank in range(config["world"])]
    evidence = dict(spark=dict(reserve_bytes=reserve, kv_pool_class=config["kv_class"], hosts=hosts),
                    logs_independently_reviewed=True)
    if config["coordinator_runtime_headroom_bytes"] > 0:
        headroom = max(config["coordinator_runtime_headroom_bytes"], 2147483648)
        evidence["coordinator"] = dict(runtime_headroom_bytes=headroom,
                                       device_free_bytes=headroom + (1 << 30),
                                       source_log="logs/coordinator.log")
    return evidence


def _without_rtx_expert_layers(metadata: dict) -> dict:
    metadata["coordinator_argv"] = [token for token in metadata["coordinator_argv"]
                                    if token != "--rtx-expert-layers"
                                    and token != str(metadata["requested_rtx_expert_layers"])]
    return metadata


def _write(tmp_path, value) -> Path:
    path = tmp_path / f"arm-{value['arm']}.json"
    path.write_text(json.dumps(value))
    return path


# 1. Six physical ranks must be present exactly once.
def test_six_rank_world_accepts_six_worker_argvs() -> None:
    metadata = _metadata("2rtx6-tp2ep3")
    arm = {"startup_metadata": metadata}
    assert six.BASE["applied_control_report"](arm, arm) == []
    short = copy.deepcopy(metadata)
    short["worker_argv"] = short["worker_argv"][:5]
    problems = six.BASE["applied_control_report"]({"startup_metadata": short}, {"startup_metadata": short})
    assert any("rank argvs but world" in problem for problem in problems), problems
    duplicate = copy.deepcopy(metadata)
    duplicate["worker_argv"][5] = copy.deepcopy(duplicate["worker_argv"][0])
    problems = six.BASE["applied_control_report"]({"startup_metadata": duplicate}, {"startup_metadata": duplicate})
    assert any("repeats a rank" in problem for problem in problems), problems


# 2/3. Paired allowed; 1RTX vs 2RTX declared forbidden.
def test_paired_allowed_and_cross_hardware_forbidden() -> None:
    paired = six.comparison_report({"arm": "2rtx6-tp2ep3"}, {"arm": "2rtx6-tp3ep2"},
                                   _configs(), _comparisons(), _forbidden())
    assert paired["allowed"] and paired["passed"] and paired["class_"] == "paired", paired
    blocked = six.comparison_report({"arm": "1rtx6-tp3ep2"}, {"arm": "2rtx6-tp2ep3"},
                                    _configs(), _comparisons(), _forbidden())
    assert not blocked["allowed"] and blocked["class_"] == "forbidden"
    assert any("unequal rtx_count" in reason for reason in blocked["reasons"]), blocked


# 4. Same native library holds all TP families; the frozen family gate applies.
def test_family_manifests_identical_both_arms_pass() -> None:
    control, candidate = _metadata("2rtx6-tp2ep3"), _metadata("2rtx6-tp3ep2")
    report = six.BASE["family_report"](control, candidate)
    assert report["problems"] == [] and report["added_in_candidate"] == [], report
    broken = copy.deepcopy(candidate)
    broken["expert_tp_manifests"]["spark"]["native_library_sha256"] = "other"
    assert any("spark" in problem for problem in six.BASE["family_report"](control, broken)["problems"])


# 5. Cases are copied verbatim from the frozen four-Spark corpus.
def test_v4_cases_match_frozen_v3_verbatim() -> None:
    assert six.validate_six(_records(), six.load_jsonl(V3))["passed"]
    mutated = copy.deepcopy(_records())
    _find(mutated, "case", id="code-merge-intervals")["prompt"] = "shortcut"
    assert any("byte-for-byte" in error for error in _errors(mutated))


# 6. Only the three approved (rtx, spark, tp, ep) tuples are allowed.
def test_config_layouts_are_exactly_the_approved_tuples() -> None:
    mutated = copy.deepcopy(_records())
    _find(mutated, "config", id="2rtx6-tp3ep2")["spark_ep"] = 3
    assert any("spark_tp*spark_ep" in error for error in _errors(mutated))
    extra = copy.deepcopy(_records())
    extra.append(dict(_configs()["2rtx6-tp3ep2"], id="1rtx6-tp1ep6", spark_tp=1, spark_ep=6))
    assert any("unsupported layout" in error for error in _errors(extra))


# 7. Config field types must be real positive integers, not bools/zeros.
def test_config_field_types_reject_bool_and_zero() -> None:
    for mutate in (lambda c: c.update(rtx_count=True), lambda c: c.update(spark_count=0),
                   lambda c: c.update(remote_layers=False)):
        mutated = copy.deepcopy(_records())
        mutate(_find(mutated, "config", id="2rtx6-tp2ep3"))
        assert any("positive integer" in error for error in _errors(mutated)), _errors(mutated)


# 8. Exactly one header/controls/matrix; duplicate comparison ids fail.
def test_singletons_and_duplicate_ids_fail() -> None:
    doubled = copy.deepcopy(_records())
    doubled.append(copy.deepcopy(_find(doubled, "corpus")))
    doubled.append(copy.deepcopy(_find(doubled, "controls")))
    doubled.append(copy.deepcopy(_find(doubled, "matrix")))
    errors = _errors(doubled)
    assert any("exactly one header" in error for error in errors)
    assert any("exactly one controls" in error for error in errors)
    assert any("exactly one matrix" in error for error in errors)
    dup_cmp = copy.deepcopy(_records())
    dup_cmp.append(copy.deepcopy(_find(dup_cmp, "comparison", id="2rtx6-paired")))
    assert any("duplicate comparison id" in error for error in _errors(dup_cmp))


# 9. Duplicate case id plus a missing case must fail (exact set, not just count).
def test_duplicate_and_missing_case_fails() -> None:
    mutated = copy.deepcopy(_records())
    cases = _by(mutated, "case")
    mutated.remove(cases[-1])
    mutated.append(copy.deepcopy(cases[0]))
    errors = _errors(mutated)
    assert any("duplicate case ids" in error for error in errors), errors
    assert any("exactly once" in error for error in errors), errors


# 10. Provenance header hashes; validate reports the hashed ARG path.
def test_provenance_and_arg_path_sha(tmp_path) -> None:
    mutated = copy.deepcopy(_records())
    mutated[0]["frozen_runner_sha256"] = "0" * 64
    assert any("frozen_runner_sha256" in error for error in _errors(mutated))
    copy_path = tmp_path / "custom-six.jsonl"
    copy_path.write_text(V4.read_text() + "\n")
    report = six.validate_six(six.load_jsonl(copy_path), six.load_jsonl(V3), copy_path)
    assert report["passed"]
    assert report["corpus_sha256"] == six.sha256(copy_path) != six.sha256(V4)


# 11. Canonical scope: reduced matrix or missing case disqualifies even when the
#     arm's own matrix field agrees with its reduced records.
def test_canonical_scope_rejects_reduced_matrix_and_missing_case() -> None:
    reduced = _full_matrix_arm("2rtx6-tp2ep3")
    reduced["matrix"] = dict(concurrency=[1, 2, 4, 8], repeats=3)
    for case in reduced["cases"]:
        case["records"] = [record for record in case["records"] if record["concurrency"] != 16]
    assert any("!= canonical" in failure
               for failure in six.arm_scope_failures(_records(), "candidate", reduced, V4))
    missing = _full_matrix_arm("2rtx6-tp2ep3")
    missing["cases"] = missing["cases"][:-1]
    assert any("cases" in failure for failure in six.arm_scope_failures(_records(), "candidate", missing, V4))


# 12. 1RTX must pass --rtx-expert-layers 0 explicitly and omit --placement-directory.
def test_1rtx_requires_explicit_zero_and_no_placement() -> None:
    config = _configs()["1rtx6-tp3ep2"]
    auto = _without_rtx_expert_layers(dict(_metadata("1rtx6-tp3ep2"), requested_rtx_expert_layers="auto"))
    problems = six.verify_arm_metadata(_records(), config, auto)
    assert any("--rtx-expert-layers 0 explicitly" in problem for problem in problems), problems
    with_placement = _metadata("1rtx6-tp3ep2")
    with_placement["coordinator_argv"] += ["--placement-directory", "/run/x"]
    assert any("must omit --placement-directory" in problem
               for problem in six.verify_arm_metadata(_records(), config, with_placement))
    # 2RTX must also pass its explicit 20 to keep the KV from being reabsorbed.
    dual = _without_rtx_expert_layers(_metadata("2rtx6-tp2ep3"))
    assert any("--rtx-expert-layers 20 explicitly" in problem
               for problem in six.verify_arm_metadata(_records(), _configs()["2rtx6-tp2ep3"], dual))


# 13. Memory domains: Spark 20 GiB host reserve vs separate 2 GiB coordinator headroom,
#     actual arithmetic, and a review flag that cannot be skipped.
def test_memory_evidence_domains_are_separate_and_actual() -> None:
    config = _configs()["1rtx6-tp3ep2"]
    assert six.verify_memory_evidence({}, config)["ok"] is False
    good = {"memory_evidence": _valid_memory("1rtx6-tp3ep2")}
    assert six.verify_memory_evidence(good, config)["ok"] is True
    unreviewed = copy.deepcopy(good)
    unreviewed["memory_evidence"]["logs_independently_reviewed"] = False
    result = six.verify_memory_evidence(unreviewed, config)
    assert result["arithmetic_ok"] and not result["reviewed"] and not result["ok"]
    nineteen = _valid_memory("1rtx6-tp3ep2")
    nineteen["spark"]["reserve_bytes"] = 20401094656  # 19 GiB
    assert any("20 GiB" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": nineteen}, config)["reasons"])
    over_budget = _valid_memory("1rtx6-tp3ep2")
    host = over_budget["spark"]["hosts"][0]
    host["device_budget_bytes"] = host["mem_total_bytes"] - over_budget["spark"]["reserve_bytes"] + 1
    assert any("exceeds" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": over_budget}, config)["reasons"])
    # Each component is individually under budget but resident+workspace+ring is not.
    sum_over = _valid_memory("1rtx6-tp3ep2")
    host = sum_over["spark"]["hosts"][0]
    budget = host["device_budget_bytes"]
    host["accounted_resident_bytes"] = config["expected_spark_resident_bytes"]
    host["accounted_workspace_bytes"] = budget // 2
    host["accounted_ring_bytes"] = budget // 2
    assert any("resident+workspace+ring" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": sum_over}, config)["reasons"])
    for resident in (0, config["expected_spark_resident_bytes"] - 1):
        low = _valid_memory("1rtx6-tp3ep2")
        low["spark"]["hosts"][0]["accounted_resident_bytes"] = resident
        assert any("below the exact minimum" in reason
                   for reason in six.verify_memory_evidence({"memory_evidence": low}, config)["reasons"])
    over_avail = _valid_memory("1rtx6-tp3ep2")
    over_avail_host = over_avail["spark"]["hosts"][0]
    over_avail_host["mem_available_bytes"] = over_avail_host["mem_total_bytes"] + 1
    assert any("exceeds mem_total_bytes" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": over_avail}, config)["reasons"])
    missing_field = _valid_memory("1rtx6-tp3ep2")
    del missing_field["spark"]["hosts"][0]["accounted_ring_bytes"]
    assert any("accounted_resident/workspace/ring" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": missing_field}, config)["reasons"])
    short_hosts = _valid_memory("1rtx6-tp3ep2")
    short_hosts["spark"]["hosts"] = short_hosts["spark"]["hosts"][:5]
    assert any("must list all 6" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": short_hosts}, config)["reasons"])
    no_coordinator = _valid_memory("1rtx6-tp3ep2")
    del no_coordinator["coordinator"]
    assert any("coordinator" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": no_coordinator}, config)["reasons"])
    dual_pool = _valid_memory("1rtx6-tp3ep2")
    dual_pool["spark"]["kv_pool_class"] = "g3-paired"
    assert any("kv_pool_class" in reason
               for reason in six.verify_memory_evidence({"memory_evidence": dual_pool}, config)["reasons"])


# 14. Missing control on a non-standalone candidate fails structurally, no crash.
def test_compare_without_control_is_structured(tmp_path) -> None:
    candidate = _write(tmp_path, _full_matrix_arm("2rtx6-tp2ep3"))
    code = six.cmd_compare(types.SimpleNamespace(corpus=V4, control=None, candidate=candidate,
                                                 output=tmp_path / "no-control.json"))
    assert code == 1
    assert json.loads((tmp_path / "no-control.json").read_text())["gate_result"] == "comparison_not_allowed"


# 15. Paired full-matrix compare passes; standalone quality+memory qualifies.
def test_compare_paired_and_standalone(tmp_path) -> None:
    control = _write(tmp_path, _full_matrix_arm("2rtx6-tp2ep3"))
    candidate = _write(tmp_path, _full_matrix_arm("2rtx6-tp3ep2"))
    assert six.cmd_compare(types.SimpleNamespace(corpus=V4, control=control, candidate=candidate,
                                                 output=tmp_path / "paired.json")) == 0
    assert json.loads((tmp_path / "paired.json").read_text())["gate_result"] == "text_exact"

    standalone = _full_matrix_arm("1rtx6-tp3ep2")
    pending = _write(tmp_path, standalone)
    assert six.cmd_compare(types.SimpleNamespace(corpus=V4, control=None, candidate=pending,
                                                 output=tmp_path / "pending.json")) == 1
    pending_report = json.loads((tmp_path / "pending.json").read_text())
    assert pending_report["gate_result"] == "standalone_quality_pass_memory_review_required"
    assert pending_report["tier_a_failures"] == []
    standalone["startup_metadata"]["memory_evidence"] = _valid_memory("1rtx6-tp3ep2")
    qualified = _write(tmp_path, standalone)
    assert six.cmd_compare(types.SimpleNamespace(corpus=V4, control=None, candidate=qualified,
                                                 output=tmp_path / "qualified.json")) == 0
    assert json.loads((tmp_path / "qualified.json").read_text())["gate_result"] == "standalone_memory_qualified"


def test_cli_validate_smoke() -> None:
    assert six.cmd_validate(types.SimpleNamespace(corpus=V4)) == 0


# 16. run prevalidates config+metadata before any request (no HTTP reached).
def test_run_prevalidates_before_requests(tmp_path) -> None:
    bad = tmp_path / "bad-start.json"
    bad.write_text(json.dumps(_without_rtx_expert_layers(
        dict(_metadata("1rtx6-tp3ep2"), requested_rtx_expert_layers="auto"))))
    args = types.SimpleNamespace(arm="1rtx6-tp3ep2", corpus=V4, base_url="http://mock",
                                 output=tmp_path / "never.json", startup_metadata=bad,
                                 case=None, concurrency=None, repeats=None)
    with pytest.raises(SystemExit) as error:
        six.cmd_run(args)
    assert "--rtx-expert-layers 0 explicitly" in str(error.value)
    assert not (tmp_path / "never.json").exists()


# 17. Common corpus controls are enforced, so a self-consistent reduced run cannot qualify.
def test_common_controls_are_enforced(tmp_path) -> None:
    records = _records()
    config = _configs()["2rtx6-tp2ep3"]
    reduced = _metadata("2rtx6-tp2ep3")
    reduced["concurrency"] = 1
    reduced["coordinator_argv"] = ["1" if token == "16" else token for token in reduced["coordinator_argv"]]
    problems = six.verify_arm_metadata(records, config, reduced)
    assert any("concurrency=1" in problem and "common control 16" in problem for problem in problems), problems
    prefix = _metadata("2rtx6-tp2ep3")
    prefix["prefix_cache_entries"] = 1
    prefix["coordinator_argv"] = ["1" if token == "20" else token for token in prefix["coordinator_argv"]]
    assert any("prefix_cache_entries=1" in problem and "common control 20" in problem
               for problem in six.verify_arm_metadata(records, config, prefix))
    bad = tmp_path / "reduced-start.json"
    bad.write_text(json.dumps(reduced))
    args = types.SimpleNamespace(arm="2rtx6-tp2ep3", corpus=V4, base_url="http://mock",
                                 output=tmp_path / "never.json", startup_metadata=bad,
                                 case=None, concurrency=None, repeats=None)
    with pytest.raises(SystemExit) as error:
        six.cmd_run(args)
    assert "common control 16" in str(error.value)
    assert not (tmp_path / "never.json").exists()


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _stream_record(text: str, tokens: int = 6) -> dict:
    events = [dict(seconds=0.05 + i * 0.02, event=dict(choices=[dict(delta=dict(content="x"))])) for i in range(tokens)]
    return dict(text=text, reasoning="", first_output_seconds=0.05, first_content_seconds=0.05,
                finish_seconds=0.05 + 0.02 * tokens, finish_reason="stop",
                observed_decode_tokens_per_second=(tokens - 1) / (0.02 * tokens),
                usage=dict(prompt_tokens=10, completion_tokens=tokens, total_tokens=10 + tokens,
                           prompt_cache_hit_tokens=10),
                events=events)


@contextmanager
def _stub_api(text: str):
    original_api = dict(six.BASE["API"])
    original_check = six.BASE["CHECK_OUTPUT"]

    def payload(prompt, stream=False):
        return dict(model="m", messages=[], temperature=0, max_tokens=96,
                    thinking=dict(type="disabled"), stream=stream)

    def open_request(base, body, api_key=None):
        return _FakeResponse(dict(choices=[dict(message=dict(content="4"), finish_reason="stop")],
                                  model="m", system_fingerprint="fp-six"))

    def stream_case(base, body, cancel=False, api_key=None, on_first_content=None):
        return _stream_record(text)

    six.BASE["API"].update(payload=payload, open_request=open_request, stream_case=stream_case)
    six.BASE["CHECK_OUTPUT"] = lambda case, content: dict(
        response_nonempty=bool(content.strip()), objective_checks=None, objective_checks_passed=True)
    try:
        yield
    finally:
        six.BASE["API"].clear()
        six.BASE["API"].update(original_api)
        six.BASE["CHECK_OUTPUT"] = original_check


def test_cli_run_smoke_with_stub(tmp_path) -> None:
    text = ",".join(str(i) for i in range(1, 65))
    startup = tmp_path / "start.json"
    startup.write_text(json.dumps(_metadata("2rtx6-tp2ep3")))
    with _stub_api(text):
        code = six.cmd_run(types.SimpleNamespace(arm="2rtx6-tp2ep3", corpus=V4, base_url="http://mock",
                                                 output=tmp_path / "arm.json", startup_metadata=startup,
                                                 case=["counting-1-64"], concurrency=[1], repeats=1))
    assert code == 0
    arm = json.loads((tmp_path / "arm.json").read_text())
    assert arm["arm_config"]["spark_count"] == 6
    assert arm["frozen_runner_sha256"] == six.FROZEN_RUNNER_SHA256
    assert arm["corpus_sha256"] == six.sha256(V4)
    assert arm["cases"][0]["records"][0]["rows"][0]["events"]
