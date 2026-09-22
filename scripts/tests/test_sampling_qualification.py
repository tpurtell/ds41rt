#!/usr/bin/env python3
"""CPU-only regression tests for the production sampling acceptance harness.

``scripts/qualify-ds41-sampling-api.py`` is the native (production) sampling
qualification harness. Its own ``--self-test`` runs every check against an
in-process native-shaped fake server; these tests make that executable contract
part of the normal CPU test run, plus verify the evidence payload and the
missing-base-url failure mode. No GPU, service, container, Docker, SSH or network
access is required.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "qualify-ds41-sampling-api.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qualify_ds41_sampling_api", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve ``cls.__module__`` through ``sys.modules``, so the
    # dynamically loaded harness must be registered before execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_self_test_exercises_every_contract() -> None:
    """The harness's built-in self-test must pass on CPU with no live service."""
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--self-test"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "self-test ok" in completed.stdout


def test_self_test_fails_when_seeded_sampling_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampler that ignores the seed must not pass the stochastic cases.

    This pins the guard that stops replay-only checks from accepting a broken
    RNG: the fake server is forced to ignore ``seed`` while every HTTP
    convention stays valid.
    """
    original = MODULE.fake_answer

    def frozen(body):
        return original({**body, "seed": 0})

    monkeypatch.setattr(MODULE, "fake_answer", frozen)
    assert MODULE.self_test() == 1


def test_deployment_run_produces_complete_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """One deployment run records requests, checks, failures and log evidence."""
    original = urllib.request.urlopen
    stats_calls = {"count": 0}

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        if request.data is None:
            stats_calls["count"] += 1
            before = {"draft_proposals": 7, "http_queue_len": 0}
            after = {"draft_proposals": 12, "draft_accepted": 5, "http_queue_len": 0}
            return MODULE.FakeResponse(
                200, json.dumps(before if stats_calls["count"] == 1 else after).encode()
            )
        return MODULE.fake_server(json.loads(request.data.decode()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        result = MODULE.run_deployment(
            "stub", "http://stub", list(MODULE.VECTORS), api_key=None
        )
    finally:
        monkeypatch.setattr(urllib.request, "urlopen", original)

    assert not result.failures, result.failures
    payload = result.as_dict()
    assert payload["label"] == "stub"
    assert payload["vectors_served"] == {vector: True for vector in MODULE.VECTORS}
    assert set(payload["stats"]["draft_counter_keys"]) == {"draft_proposals", "draft_accepted"}
    assert payload["observations"], "no raw evidence recorded"
    for observation in payload["observations"]:
        assert "request" in observation and "status" in observation
        assert "output_sha256" in observation and "output_length" in observation
    # Every observation must be JSON-serializable as recorded.
    json.dumps(payload)


def test_log_scan_records_draft_evidence(tmp_path: Path) -> None:
    """Engine-log draft evidence is counted and sampled without a schema."""
    log = tmp_path / "engine.log"
    log.write_text(
        "startup complete\n"
        "dspark draft proposed 3 tokens\n"
        "target verify accepted 2 draft tokens\n"
        "unrelated line\n"
    )
    evidence = MODULE.scan_dspark_log(log)
    assert evidence is not None and evidence["available"] is True
    assert evidence["counts"]["draft"] == 2
    assert evidence["counts"]["dspark"] == 1
    assert len(evidence["sample_lines"]) == 2


def test_log_scan_records_missing_file(tmp_path: Path) -> None:
    evidence = MODULE.scan_dspark_log(tmp_path / "absent.log")
    assert evidence is not None and evidence["available"] is False


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ok": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "tags", "ok"],
    "additionalProperties": False,
}


def observation(text: str) -> "MODULE.Observation":
    return MODULE.Observation(name="schema-case", request={}, status=200, text=text)


def test_schema_conformance_accepts_a_conforming_payload() -> None:
    body = json.dumps({"answer": "4", "confidence": 100, "tags": ["math"], "ok": True})
    MODULE.check_schema_conformance(observation(body), ANSWER_SCHEMA)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"confidence": 100, "tags": [], "ok": True}, "missing required key"),
        (
            {"answer": "4", "confidence": 100, "tags": [], "ok": True, "extra": 1},
            "unexpected key",
        ),
        ({"answer": 4, "confidence": 100, "tags": [], "ok": True}, "expected string"),
        ({"answer": "4", "confidence": 100.5, "tags": [], "ok": True}, "expected integer"),
        ({"answer": "4", "confidence": 101, "tags": [], "ok": True}, "above maximum"),
        ({"answer": "4", "confidence": 100, "tags": [1], "ok": True}, "expected string"),
        ({"answer": "4", "confidence": 100, "tags": "math", "ok": True}, "expected array"),
        ({"answer": "4", "confidence": 100, "tags": [], "ok": "yes"}, "expected boolean"),
        (["not", "an", "object"], "expected object"),
    ],
)
def test_schema_conformance_rejects_wrong_type_and_extra_keys(payload, reason) -> None:
    with pytest.raises(MODULE.AcceptanceError) as error:
        MODULE.check_schema_conformance(observation(json.dumps(payload)), ANSWER_SCHEMA)
    assert reason in str(error.value)


def test_schema_conformance_rejects_non_json() -> None:
    with pytest.raises(MODULE.AcceptanceError) as error:
        MODULE.check_schema_conformance(observation("not json"), ANSWER_SCHEMA)
    assert "not JSON" in str(error.value)


TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def tool_observation(tool_calls, *, text: str = "", finish_reason: str = "tool_calls"):
    """Observation carrying protocol tool_calls, as the real response does."""
    return MODULE.Observation(
        name="tool-case",
        request={},
        status=200,
        text=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def test_tool_conformance_reads_message_tool_calls_with_null_content() -> None:
    """Regression: a real tool response has content null and message.tool_calls.

    The pre-fix harness parsed response *text* as JSON, so it failed every real
    tool-call response. Arguments here arrive as a JSON string, exactly as the
    protocol serializes them.
    """
    observation = tool_observation(
        [
            {
                "id": "call_abc",
                "type": "function",
                "name": "lookup",
                "arguments": "{\"city\": \"Taipei\"}",
            }
        ]
    )
    assert observation.output() == "", "content must be absent/null for a tool response"
    MODULE.tool_call_schema_check("lookup", TOOL_PARAMETERS)(observation)
    # Object-form arguments must also be accepted.
    MODULE.tool_call_schema_check("lookup", TOOL_PARAMETERS)(
        tool_observation([{"name": "lookup", "arguments": {"city": "Taipei"}}])
    )


def test_tool_replay_ignores_server_generated_call_ids() -> None:
    """Replay compares name + arguments; response ids differ between runs."""
    first = tool_observation(
        [{"id": "call_run_1", "type": "function", "name": "lookup", "arguments": '{"city": "Taipei"}'}]
    )
    second = tool_observation(
        [{"id": "call_run_2", "type": "function", "name": "lookup", "arguments": {"city": "Taipei"}}]
    )
    assert first.tool_replay() == second.tool_replay()
    # A genuine argument difference must still be visible.
    third = tool_observation([{"name": "lookup", "arguments": {"city": "Oslo"}}])
    assert first.tool_replay() != third.tool_replay()


def test_streamed_tool_deltas_assemble_like_the_protocol() -> None:
    """delta.tool_calls start + argument chunks must assemble into one call."""
    calls: dict = {}
    MODULE.assemble_stream_tool_calls(
        [
            {
                "index": 0,
                "id": "call_x",
                "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            }
        ],
        calls,
    )
    MODULE.assemble_stream_tool_calls([{"index": 0, "function": {"arguments": '{"city": '}}], calls)
    MODULE.assemble_stream_tool_calls([{"index": 0, "function": {"arguments": '"Taipei"}'}}], calls)
    assert calls[0]["name"] == "lookup"
    assembled = tool_observation([calls[0]])
    assert assembled.tool_replay() == [{"name": "lookup", "arguments": {"city": "Taipei"}}]


@pytest.mark.parametrize(
    "tool_calls,reason",
    [
        ([], "no tool_calls"),
        ([{"name": "other", "arguments": {"city": "x"}}], "expected tool 'lookup'"),
        ([{"name": "lookup", "arguments": {}}], "missing required key"),
        ([{"name": "lookup", "arguments": {"city": 5}}], "expected string"),
        ([{"name": "lookup", "arguments": {"city": "x", "z": 1}}], "unexpected key"),
        ([{"name": "lookup", "arguments": "not json"}], "not JSON"),
    ],
)
def test_tool_conformance_rejects_wrong_call_or_arguments(tool_calls, reason) -> None:
    with pytest.raises(MODULE.AcceptanceError) as error:
        MODULE.tool_call_schema_check("lookup", TOOL_PARAMETERS)(tool_observation(tool_calls))
    assert reason in str(error.value)


def test_tool_finish_reason_is_accepted_and_null_content_is_legal() -> None:
    MODULE.check_served(tool_observation([{"name": "lookup", "arguments": {"city": "x"}}]))
    with pytest.raises(MODULE.AcceptanceError):
        MODULE.check_served(
            MODULE.Observation(name="empty", request={}, status=200, finish_reason="stop")
        )


def test_fake_schema_preflight_matches_native_or_semantics() -> None:
    """Required-missing OR additionalProperties-not-false: either one rejects."""
    valid = {
        "type": "object",
        "json_schema": {
            "name": "ok",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }
    assert not MODULE.fake_rejects_schema({"response_format": valid})
    # required present but additionalProperties missing -> native rejects.
    missing_additional = {
        "type": "object",
        "json_schema": {
            "name": "bad",
            "strict": True,
            "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
        },
    }
    assert MODULE.fake_rejects_schema({"response_format": missing_additional})
    # additionalProperties false but required incomplete -> native rejects.
    incomplete_required = {
        "type": "object",
        "json_schema": {
            "name": "bad2",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }
    assert MODULE.fake_rejects_schema({"response_format": incomplete_required})
    # Non-strict schemas are not preflighted.
    non_strict = json.loads(json.dumps(missing_additional))
    non_strict["json_schema"]["strict"] = False
    assert not MODULE.fake_rejects_schema({"response_format": non_strict})


def test_legacy_style_tool_payload_in_content_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool response that hides tool_calls inside content must fail, not pass.

    This is the regression guard for the confirmed live bug: the harness must
    require protocol ``message.tool_calls`` (with ``content: null``), so an
    in-content encoding is correctly reported as a failure.
    """
    original = urllib.request.urlopen

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        if request.data is None:
            return MODULE.FakeResponse(200, b'{"http_queue_len":0}')
        body = json.loads(request.data.decode())
        if body.get("tools"):
            legacy = json.dumps(
                {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"city": "Taipei"}},
                        }
                    ]
                }
            )
            return MODULE.FakeResponse(
                200,
                json.dumps(
                    {
                        "choices": [
                            {"message": {"role": "assistant", "content": legacy}, "finish_reason": "stop"}
                        ]
                    }
                ).encode(),
            )
        return MODULE.fake_server(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        result = MODULE.run_deployment("legacy", "http://legacy", list(MODULE.VECTORS), api_key=None)
    finally:
        monkeypatch.setattr(urllib.request, "urlopen", original)
    tool_failures = [failure for failure in result.failures if "tool_calls" in failure]
    assert tool_failures, f"legacy in-content tool payload was not reported: {result.failures}"


def test_tool_replay_is_scoped_to_identical_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Greedy and stochastic tool runs may differ; only same-params runs replay.

    The fake returns a different tool argument for the greedy tool case, so this
    fails if the harness ever compares greedy and stochastic tool calls together.
    """
    original = urllib.request.urlopen

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        if request.data is None:
            return MODULE.FakeResponse(200, b'{"http_queue_len":0}')
        return MODULE.fake_server(json.loads(request.data.decode()))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        result = MODULE.run_deployment("scope", "http://scope", list(MODULE.VECTORS), api_key=None)
    finally:
        monkeypatch.setattr(urllib.request, "urlopen", original)

    assert not result.failures, result.failures
    assert "stochastic-tool-replay-identical" in result.checks
    assert "greedy-tool-validated-separately" in result.checks
    stochastic = result.by_name("constrained-tool")
    greedy = result.by_name("constrained-tool-greedy")
    assert stochastic is not None and greedy is not None
    # Different by design: the scoped check must not equate them.
    assert stochastic.tool_replay() != greedy.tool_replay()


def test_missing_base_url_is_a_clear_cli_error() -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "--base-url" in completed.stderr


def test_malformed_vector_is_rejected() -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--base-url", "http://127.0.0.1:1", "--vector", "nope"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
