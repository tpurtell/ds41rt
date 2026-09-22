#!/usr/bin/env python3
"""Production native-API sampling acceptance harness.

Targets the released ``serve-native`` path (``ds41rt-api`` ``native_v41``
router), not the legacy ``real_full`` sampler. It talks to one already-running
deployment over HTTP and never launches a model, container, Docker or SSH, so it
is safe to run before a cluster booking. The sampling matrix is *five sampling
vectors* (parameter combinations) exercised against *one deployment*: a vector
is a release behaviour, not a topology. Do not confuse the two when reporting.

Five sampling vectors (all against the same base URL):

1. ``v1-greedy``       temperature 0 (defaults; compatibility baseline)
2. ``v2-t02-p95``      temperature 0.2, top_p 0.95
3. ``v3-t07-p90``      temperature 0.7, top_p 0.9
4. ``v4-t07-minp005``  temperature 0.7, min_p 0.05
5. ``v5-t07-k40``      temperature 0.7, top_k 40

Per vector the harness checks: accepted by the API (not a 400), non-empty text,
exact replay with ``seed=0``, and bounded/absent error bodies. The stochastic
vectors (every non-greedy one, i.e. ``v2``-``v5``) must additionally reproduce
differently for two seeds on an unconstrained broad prompt. It separately checks
greedy knob-insensitivity, signed-seed replay (``seed=-2``), cross-request
isolation, invalid-parameter 400s, the designed acceptance of ``top_k`` above 64
and ``top_k >= vocabulary``, and constrained decoding.

Constrained decoding evidence is real conformance, not just parsing: the harness
validates each served JSON payload against the exact schema it sent (object
properties, ``required``, ``additionalProperties=false``, string/integer/boolean/
array types, integer bounds) and validates strict tool-call arguments the same
way. That checker is schema-specific by design and is not a generic JSON Schema
implementation; the engine's authoritative Draft 2020-12 validation remains
server-side. A non-conforming payload fails the run.

Token-level honoured-distribution checks (does min_p *really* cut the tail?) and
the dSpark sample-match equivalence belong to the core verifier tests, which can
use exact synthetic logits; HTTP text cannot establish a distribution. This
harness records missing draft evidence rather than requiring it. The native
worker exposes no draft/dSpark counters on ``/v1/stats`` today, so pass
``--log-file`` to scan engine logs for draft/spec/mtp/dspark lines; absence is
recorded, never failed.

Evidence rule (matches ``scripts/bench/README.md``): every reported cell comes
from a recorded observation in the output JSON; nothing is invented. The full
response text is retained for the replay-critical requests and for the rest only
its length and SHA-256 are kept, so the artifact stays bounded.

CPU-only dry run of every check against an in-process fake server::

    python3 scripts/qualify-ds41-sampling-api.py --self-test

Live run (five vectors on one deployment)::

    python3 scripts/qualify-ds41-sampling-api.py --base-url http://127.0.0.1:18041 \
        --output /tmp/sampling-acceptance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
ERROR_TYPE = "native_v41_error"
# Native serving defaults thinking ON; disable it so reasoning cannot consume the
# whole output budget before the sampling behaviour is observed.
THINKING = {"type": "disabled"}
# The five released sampling vectors, exactly as specified. ``None`` means the
# field is omitted from the request body.
VECTORS: dict[str, dict[str, Any]] = {
    "v1-greedy": {"temperature": 0.0},
    "v2-t02-p95": {"temperature": 0.2, "top_p": 0.95},
    "v3-t07-p90": {"temperature": 0.7, "top_p": 0.9},
    "v4-t07-minp005": {"temperature": 0.7, "min_p": 0.05},
    "v5-t07-k40": {"temperature": 0.7, "top_k": 40},
}
# A vector is stochastic when temperature is above the core's greedy epsilon
# (1e-5) and no filter collapses the distribution to one token.
GREEDY_VECTOR = "v1-greedy"


def stochastic_vectors() -> list[str]:
    return [
        name
        for name, params in VECTORS.items()
        if params.get("temperature", 0.0) >= 1.0e-5
        and params.get("top_k") != 1
        and params.get("min_p", 0.0) < 1.0
    ]


class AcceptanceError(AssertionError):
    """A pinned sampling contract was violated."""


@dataclass
class Case:
    """One request and the assertion that defines its acceptance."""

    name: str
    body: dict[str, Any]
    check: Callable[["Observation"], None]


@dataclass
class Observation:
    """Raw evidence for one request; every reported cell derives from here."""

    name: str
    request: dict[str, Any]
    status: int | None = None
    error: dict[str, Any] | None = None
    text: str = ""
    reasoning: str = ""
    # Tool calls are carried outside `content` (real protocol: message.tool_calls
    # with `content: null`). Kept separately so tool conformance never depends on
    # inventing JSON inside the text field.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    usage: dict[str, Any] | None = None
    stream_done: bool | None = None
    sse_frames: int | None = None
    elapsed_seconds: float | None = None
    transport_error: str | None = None

    def output(self) -> str:
        return self.text + self.reasoning

    def tool_replay(self) -> list[dict[str, Any]]:
        """Call identity for replay comparison: name + parsed arguments, never the id.

        Response call ids are server-generated and differ between otherwise
        identical runs, so they must not participate in replay equality. Streamed
        arguments arrive as an accumulated JSON string while non-streamed
        arguments arrive as an object, so both are parsed to the same value.
        """
        calls = []
        for call in self.tool_calls:
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"name": call.get("name"), "arguments": arguments})
        return calls

    def as_dict(self, retain_text: bool) -> dict[str, Any]:
        encoded = self.output().encode()
        tool_identity = json.dumps(self.tool_replay(), sort_keys=True, default=str)
        record: dict[str, Any] = {
            "name": self.name,
            "request": self.request,
            "status": self.status,
            "error": self.error,
            "output_length": len(self.output()),
            "output_sha256": hashlib.sha256(encoded).hexdigest(),
            "tool_call_count": len(self.tool_calls),
            "tool_calls": self.tool_calls,
            "tool_replay_sha256": hashlib.sha256(tool_identity.encode()).hexdigest(),
            "finish_reason": self.finish_reason,
            "system_fingerprint": self.system_fingerprint,
            "usage": self.usage,
            "stream_done": self.stream_done,
            "sse_frames": self.sse_frames,
            "elapsed_seconds": self.elapsed_seconds,
            "transport_error": self.transport_error,
            "text_retained": retain_text,
        }
        if retain_text:
            record["text"] = self.text
            record["reasoning"] = self.reasoning
        return record


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def payload(prompt: str, *, stream: bool = False, **sampling: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "thinking": dict(THINKING),
        "max_tokens": 96,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    for key, value in sampling.items():
        if value is not None:
            body[key] = value
    return body


def vector_body(name: str, prompt: str, *, stream: bool = False, **overrides: Any) -> dict[str, Any]:
    return payload(prompt, stream=stream, **{**VECTORS[name], **overrides})


def open_request(base_url: str, body: dict[str, Any], api_key: str | None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return urllib.request.urlopen(
        urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        ),
        timeout=180,
    )


def normalise_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Normalise one protocol tool call (arguments may be an object or a string)."""
    function = call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {
        "id": call.get("id"),
        "type": call.get("type"),
        "name": function.get("name"),
        "arguments": arguments,
    }


def assemble_stream_tool_calls(
    deltas: list[dict[str, Any]], calls: dict[int, dict[str, Any]]
) -> None:
    """Assemble streamed tool-call deltas by index (start carries name, then args)."""
    for delta in deltas:
        index = delta.get("index")
        if not isinstance(index, int):
            continue
        entry = calls.setdefault(
            index, {"id": None, "type": None, "name": None, "arguments": ""}
        )
        if delta.get("id") is not None:
            entry["id"] = delta.get("id")
        if delta.get("type") is not None:
            entry["type"] = delta.get("type")
        function = delta.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") is not None:
            entry["name"] = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            entry["arguments"] = arguments
        elif isinstance(arguments, str):
            existing = entry.get("arguments")
            entry["arguments"] = (
                existing + arguments if isinstance(existing, str) else arguments
            )


def run_case(base_url: str, case: Case, api_key: str | None) -> Observation:
    observation = Observation(name=case.name, request=case.body)
    started = time.perf_counter()
    stream = bool(case.body.get("stream"))
    streamed_calls: dict[int, dict[str, Any]] = {}
    try:
        with open_request(base_url, case.body, api_key) as response:
            observation.status = response.status
            if stream:
                observation.stream_done = False
                frames = 0
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        observation.stream_done = True
                        break
                    frames += 1
                    event = json.loads(data)
                    if event.get("error"):
                        observation.error = event["error"]
                        break
                    if event.get("usage"):
                        observation.usage = event["usage"]
                    if event.get("system_fingerprint"):
                        observation.system_fingerprint = event["system_fingerprint"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        observation.text += str(delta.get("content") or "")
                        observation.reasoning += str(delta.get("reasoning_content") or "")
                        assemble_stream_tool_calls(
                            delta.get("tool_calls") or [], streamed_calls
                        )
                        if choice.get("finish_reason"):
                            observation.finish_reason = choice["finish_reason"]
                observation.sse_frames = frames
                observation.tool_calls = [
                    streamed_calls[index] for index in sorted(streamed_calls)
                ]
            else:
                body = json.load(response)
                if body.get("error"):
                    observation.error = body["error"]
                observation.system_fingerprint = body.get("system_fingerprint")
                observation.usage = body.get("usage")
                for choice in body.get("choices") or []:
                    message = choice.get("message") or {}
                    observation.text += str(message.get("content") or "")
                    observation.reasoning += str(message.get("reasoning_content") or "")
                    for call in message.get("tool_calls") or []:
                        observation.tool_calls.append(normalise_tool_call(call))
                    observation.finish_reason = choice.get("finish_reason")
    except urllib.error.HTTPError as error:
        observation.status = error.code
        raw = error.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
            observation.error = parsed.get("error") or {"message": raw}
        except json.JSONDecodeError:
            observation.error = {"message": raw}
    except Exception as error:  # noqa: BLE001 - transport failures are evidence
        observation.transport_error = f"{type(error).__name__}: {error}"
    finally:
        observation.elapsed_seconds = time.perf_counter() - started
    return observation


def fetch_stats(base_url: str, api_key: str | None) -> dict[str, Any] | None:
    """Best-effort ``/v1/stats`` snapshot; absence is recorded, never failed."""
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base_url.rstrip("/") + "/v1/stats", headers=headers),
            timeout=30,
        ) as response:
            return json.load(response)
    except Exception:  # noqa: BLE001 - stats are diagnostic only
        return None


def draft_counter_keys(stats: dict[str, Any] | None) -> list[str]:
    """Stats key paths mentioning draft/spec/mtp/dspark (numeric leaves only)."""
    if not isinstance(stats, dict):
        return []
    markers = ("draft", "spec", "mtp", "dspark")
    keys: list[str] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, bool):
            return
        elif isinstance(value, int) and any(marker in prefix.lower() for marker in markers):
            keys.append(prefix)

    walk(stats, "")
    return sorted(keys)


def scan_dspark_log(log_path: Path | None) -> dict[str, Any] | None:
    """Count draft/dSpark mentions in an engine log; missing file is recorded.

    Performance-independent evidence source while ``/v1/stats`` exposes no draft
    counters. This does not parse a schema: it records counts and a bounded
    sample of matching lines so a reviewer can inspect the raw text.
    """
    if log_path is None:
        return None
    if not log_path.is_file():
        return {"path": str(log_path), "available": False, "note": "log file not found"}
    markers = ("draft", "dspark", "spec", "mtp")
    counts = {marker: 0 for marker in markers}
    samples: list[str] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            hits = [marker for marker in markers if marker in lowered]
            for marker in hits:
                counts[marker] += 1
            if hits and len(samples) < 20:
                samples.append(line.rstrip("\n")[:400])
    return {
        "path": str(log_path),
        "available": True,
        "counts": counts,
        "sample_lines": samples,
        "note": (
            "no draft/dspark mentions found in this log"
            if not any(counts.values())
            else "draft evidence found in engine log"
        ),
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def expect_ok(observation: Observation) -> None:
    if observation.transport_error:
        raise AcceptanceError(f"{observation.name}: transport error {observation.transport_error}")
    if observation.status != 200:
        raise AcceptanceError(
            f"{observation.name}: expected 200, got {observation.status} ({observation.error})"
        )


def expect_bad_request(observation: Observation) -> None:
    if observation.status != 400:
        raise AcceptanceError(
            f"{observation.name}: expected 400 for invalid sampling, got {observation.status}"
        )
    message = str((observation.error or {}).get("message") or "")
    if not message:
        raise AcceptanceError(f"{observation.name}: 400 body carried no error.message")
    if len(message) > 4000:
        raise AcceptanceError(
            f"{observation.name}: error message is not bounded ({len(message)} bytes)"
        )


def check_served(observation: Observation) -> None:
    """Request accepted, text produced, finish reason reported.

    A tool-call response is valid with ``content: null``; for those the payload
    is carried in ``tool_calls``, so either text or a tool call satisfies the
    content requirement. ``finish_reason`` may be ``stop``, ``length`` or
    ``tool_calls``.
    """
    if observation.status == 400:
        raise AcceptanceError(
            f"{observation.name}: sampling request rejected 400: {observation.error}"
        )
    expect_ok(observation)
    if not observation.output() and not observation.tool_calls:
        raise AcceptanceError(
            f"{observation.name}: response carried neither text nor tool_calls"
        )
    if observation.finish_reason not in {"stop", "length", "tool_calls"}:
        raise AcceptanceError(
            f"{observation.name}: missing finish_reason ({observation.finish_reason})"
        )


def check_stream_complete(observation: Observation) -> None:
    """SSE terminates with [DONE], carries usage, and reports a valid finish.

    Token-level streams may omit ``finish_reason`` on every frame, so a missing
    finish reason is accepted only when the stream actually produced content; a
    stream that produced neither text nor tool calls is a failure.
    """
    expect_ok(observation)
    if observation.stream_done is not True:
        raise AcceptanceError(f"{observation.name}: SSE stream did not terminate with [DONE]")
    if observation.usage is None:
        raise AcceptanceError(f"{observation.name}: stream carried no usage with include_usage")
    if observation.finish_reason is None:
        if not observation.output() and not observation.tool_calls:
            raise AcceptanceError(
                f"{observation.name}: stream produced no content and no finish_reason"
            )
        return
    if observation.finish_reason not in {"stop", "length", "tool_calls"}:
        raise AcceptanceError(
            f"{observation.name}: stream carried unknown finish_reason ({observation.finish_reason})"
        )


def check_parseable_json(observation: Observation) -> None:
    """Parse-only check, used where the harness does not own the schema."""
    expect_ok(observation)
    body = observation.text.strip()
    if not body:
        raise AcceptanceError(f"{observation.name}: constrained output is empty")
    try:
        json.loads(body)
    except json.JSONDecodeError as error:
        raise AcceptanceError(
            f"{observation.name}: constrained output is not JSON: {error}: {body!r}"
        )


def conforms_to_schema(value: Any, schema: dict[str, Any]) -> str | None:
    """Schema-specific conformance check for the harness-owned schemas.

    This is deliberately not a generic JSON Schema implementation: it covers the
    subset the harness itself sends (object, string, integer, boolean, array of
    strings, enum, required, additionalProperties=false, integer bounds). The
    engine performs authoritative Draft 2020-12 validation server-side; this
    check independently confirms the served payload on the client. Returns the
    failure detail, or None when the value conforms.
    """
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return f"expected object, got {type(value).__name__}"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"missing required key {key!r}"
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                return f"unexpected key(s) {extra} (additionalProperties=false)"
        for key, sub_schema in properties.items():
            if key in value:
                detail = conforms_to_schema(value[key], sub_schema)
                if detail is not None:
                    return f"{key}: {detail}"
        return None
    if expected == "array":
        if not isinstance(value, list):
            return f"expected array, got {type(value).__name__}"
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                detail = conforms_to_schema(item, item_schema)
                if detail is not None:
                    return f"[{index}]: {detail}"
        return None
    if expected == "string":
        return None if isinstance(value, str) else f"expected string, got {type(value).__name__}"
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if "minimum" in schema and value < schema["minimum"]:
            return f"{value} is below minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{value} is above maximum {schema['maximum']}"
        return None
    if expected == "boolean":
        return None if isinstance(value, bool) else f"expected boolean, got {type(value).__name__}"
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected number, got {type(value).__name__}"
        return None
    if "enum" in schema:
        return None if value in schema["enum"] else f"{value!r} is not one of {schema['enum']}"
    return None


def check_schema_conformance(observation: Observation, schema: dict[str, Any]) -> None:
    """Response must be JSON and conform to the schema the request sent."""
    check_parseable_json(observation)
    try:
        payload = json.loads(observation.text.strip())
    except json.JSONDecodeError:  # already reported by check_parseable_json
        return
    detail = conforms_to_schema(payload, schema)
    if detail is not None:
        raise AcceptanceError(
            f"{observation.name}: response does not conform to the request schema: {detail}"
        )


def schema_conformance_check(schema: dict[str, Any]) -> Callable[[Observation], None]:
    def check(observation: Observation) -> None:
        check_schema_conformance(observation, schema)

    return check


def tool_call_schema_check(expected_name: str, parameters: dict[str, Any]) -> Callable[[Observation], None]:
    """Tool output must carry a call to the requested function with conforming args.

    Reads the protocol ``tool_calls`` field (``message.tool_calls`` for
    non-stream, assembled ``delta.tool_calls`` for stream), never the text
    content: a real tool response carries ``content: null``. The engine validates
    tool calls server-side as well; this confirms the served payload client-side.
    """

    def check(observation: Observation) -> None:
        expect_ok(observation)
        if not observation.tool_calls:
            raise AcceptanceError(
                f"{observation.name}: response carried no tool_calls "
                f"(content was {observation.text!r})"
            )
        call = observation.tool_calls[0]
        if call.get("name") != expected_name:
            raise AcceptanceError(
                f"{observation.name}: expected tool {expected_name!r}, got {call.get('name')!r}"
            )
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise AcceptanceError(
                    f"{observation.name}: tool arguments are not JSON: {error}"
                ) from error
        if not isinstance(arguments, dict):
            raise AcceptanceError(
                f"{observation.name}: tool arguments must be an object, got {type(arguments).__name__}"
            )
        detail = conforms_to_schema(arguments, parameters)
        if detail is not None:
            raise AcceptanceError(
                f"{observation.name}: tool arguments do not conform to the strict schema: {detail}"
            )

    return check


def assert_replay(observations: list[Observation], label: str) -> None:
    for observation in observations:
        expect_ok(observation)
    outputs = [observation.output() for observation in observations]
    if len(set(outputs)) != 1:
        raise AcceptanceError(
            f"{label}: fixed-seed replay diverged across {len(outputs)} runs "
            f"(lengths {[len(value) for value in outputs]})"
        )


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

GREEDY_PROMPT = "What is 2 + 2? Answer with just the number."
BROAD_PROMPT = (
    "Write one sentence suggesting five different names for a pet cat. "
    "No two answers should be identical."
)


def vector_cases(vectors: list[str]) -> list[Case]:
    """Five-vector acceptance: served once, replayed with seed=0, streamed once."""
    cases: list[Case] = []
    for vector in vectors:
        if vector == GREEDY_VECTOR:
            body = vector_body(vector, GREEDY_PROMPT)
            cases.append(Case("greedy-served", body, check_served))
            cases.append(Case("greedy-replay-nonstream", dict(body), check_served))
            cases.append(
                Case("greedy-with-knobs", vector_body(vector, GREEDY_PROMPT, top_p=0.5, top_k=7, seed=0), check_served)
            )
            continue
        body = vector_body(vector, GREEDY_PROMPT, seed=0)
        cases.append(Case(f"{vector}-served", body, check_served))
        cases.append(Case(f"{vector}-replay-nonstream", dict(body), check_served))
        cases.append(
            Case(
                f"{vector}-replay-stream",
                vector_body(vector, GREEDY_PROMPT, stream=True, seed=0),
                check_stream_complete,
            )
        )
    return cases


def seed_cases() -> list[Case]:
    """Signed-seed replay and seed isolation.

    ``payload`` accepts negative seeds because the HTTP field is signed and the
    engine maps it to an unsigned stream by cast; ``-2`` is the pinned replay
    case for that mapping. ``seed=0`` must be a real, repeatable seed.
    """
    vector = "v3-t07-p90"
    cases = [
        Case("seed-zero-a", vector_body(vector, GREEDY_PROMPT, seed=0), check_served),
        Case("seed-zero-b", vector_body(vector, GREEDY_PROMPT, seed=0), check_served),
        Case("seed-negative-2-a", vector_body(vector, GREEDY_PROMPT, seed=-2), check_served),
        Case("seed-negative-2-b", vector_body(vector, GREEDY_PROMPT, seed=-2), check_served),
        Case("isolation-seed-11-first", vector_body(vector, BROAD_PROMPT, seed=11), check_served),
        Case("isolation-seed-22", vector_body(vector, BROAD_PROMPT, seed=22), check_served),
        Case("isolation-seed-11-again", vector_body(vector, BROAD_PROMPT, seed=11), check_served),
    ]
    return cases


def stochastic_cases(vectors: list[str]) -> list[Case]:
    """Seed diversity on unconstrained stochastic vectors with a broad prompt.

    Schema/regex/enum constraints can force a deterministic token sequence even
    for a stochastic sampler, so diversity is measured only here, with no
    ``response_format`` and no ``tools``. Vectors that collapse the distribution
    to one token (``top_k=1``, ``min_p=1``) are excluded by construction.
    """
    cases: list[Case] = []
    for vector in vectors:
        if vector not in stochastic_vectors():
            continue
        for seed in (11, 22):
            for repeat in range(2):
                cases.append(
                    Case(
                        f"stochastic-{vector}-seed-{seed}-run-{repeat}",
                        vector_body(vector, BROAD_PROMPT, seed=seed),
                        check_served,
                    )
                )
    return cases


def invalid_parameter_cases() -> list[Case]:
    return [
        Case("invalid-temperature-negative", payload("hello", temperature=-0.5), expect_bad_request),
        Case("invalid-temperature-above-range", payload("hello", temperature=2.5), expect_bad_request),
        Case("invalid-top-p-zero", payload("hello", temperature=1.0, top_p=0.0), expect_bad_request),
        Case("invalid-top-p-above-one", payload("hello", temperature=1.0, top_p=1.5), expect_bad_request),
        Case("invalid-top-k-below-minus-one", payload("hello", temperature=1.0, top_k=-2), expect_bad_request),
        Case("invalid-top-k-fractional", payload("hello", temperature=1.0, top_k=1.5), expect_bad_request),
        Case("invalid-min-p-above-one", payload("hello", temperature=1.0, min_p=1.5), expect_bad_request),
        Case("invalid-min-p-negative", payload("hello", temperature=1.0, min_p=-0.1), expect_bad_request),
        Case("invalid-seed-fractional", payload("hello", temperature=1.0, seed=1.5), expect_bad_request),
    ]


def boundary_cases() -> list[Case]:
    """Boundaries that must be accepted by design.

    ``top_k = 0`` and ``top_k = -1`` are disabled spellings that the protocol
    layer normalises to "no filter" (the core constructor rejects ``Some(0)``).
    ``top_k`` above the legacy 64 cap and ``top_k >= vocabulary`` are accepted:
    the full-logit sampler has no 64-wide limit, and ``k >= vocab`` is a no-op.
    These are asserted, not merely recorded.
    """
    base = "v3-t07-p90"
    cases = [
        Case("boundary-temperature-zero", payload(GREEDY_PROMPT, temperature=0), check_served),
        Case("boundary-temperature-two", vector_body(base, GREEDY_PROMPT, temperature=2.0, seed=0), check_served),
        Case("boundary-top-p-unity", vector_body(base, GREEDY_PROMPT, top_p=1.0, seed=0), check_served),
        Case("boundary-min-p-zero", vector_body(base, GREEDY_PROMPT, min_p=0.0, seed=0), check_served),
        Case("boundary-min-p-unity", vector_body(base, GREEDY_PROMPT, min_p=1.0, seed=0), check_served),
        Case("boundary-top-k-one", vector_body(base, GREEDY_PROMPT, top_k=1, seed=0), check_served),
        Case("boundary-top-k-64", vector_body(base, GREEDY_PROMPT, top_k=64, seed=0), check_served),
        Case("boundary-top-k-65", vector_body(base, GREEDY_PROMPT, top_k=65, seed=0), check_served),
        Case("boundary-top-k-1000", vector_body(base, GREEDY_PROMPT, top_k=1000, seed=0), check_served),
        Case("boundary-top-k-vocab", vector_body(base, GREEDY_PROMPT, top_k=129_280, seed=0), check_served),
        Case("boundary-top-k-above-vocab", vector_body(base, GREEDY_PROMPT, top_k=200_000, seed=0), check_served),
    ]
    for spelling in (0, -1):
        cases.append(
            Case(
                "boundary-top-k-disabled-"
                + str(spelling).replace("-", "negative"),
                vector_body(base, GREEDY_PROMPT, top_k=spelling, seed=0),
                check_served,
            )
        )
    return cases


def constrained_cases() -> list[Case]:
    # The served payload is checked against this exact schema on the client, so
    # the live run produces conformance evidence, not just "parses as JSON".
    answer_schema = {
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
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": answer_schema,
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    answer_prompt = (
        "Return JSON only: answer as the string four, confidence as the integer 100, "
        "tags as the array [\"math\"], and ok as the boolean true."
    )
    cases = [
        Case("constrained-json-object", payload("Return a JSON object with one key, ok, set to true.", response_format={"type": "json_object"}), check_parseable_json),
        Case("constrained-json-schema-greedy", payload(answer_prompt, response_format=schema), schema_conformance_check(answer_schema)),
        Case("constrained-json-schema-sampled", vector_body("v3-t07-p90", answer_prompt, seed=0, response_format=schema), schema_conformance_check(answer_schema)),
        Case("constrained-json-schema-sampled-replay", vector_body("v3-t07-p90", answer_prompt, seed=0, response_format=schema), schema_conformance_check(answer_schema)),
        Case("constrained-tool", vector_body("v3-t07-p90", "Look up the weather in Taipei.", seed=0, tools=tools, tool_choice="required"), tool_call_schema_check("lookup", tools[0]["function"]["parameters"])),
        Case("constrained-tool-replay", vector_body("v3-t07-p90", "Look up the weather in Taipei.", seed=0, tools=tools, tool_choice="required"), tool_call_schema_check("lookup", tools[0]["function"]["parameters"])),
        Case("constrained-tool-stream", vector_body("v3-t07-p90", "Look up the weather in Taipei.", seed=0, stream=True, tools=tools, tool_choice="required"), tool_call_schema_check("lookup", tools[0]["function"]["parameters"])),
        Case("constrained-tool-greedy", payload("Look up the weather in Taipei.", temperature=0, tools=tools, tool_choice="required"), tool_call_schema_check("lookup", tools[0]["function"]["parameters"])),
        Case(
            "constrained-invalid-schema",
            payload(
                "hello",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bad",
                        "strict": True,
                        "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
                    },
                },
            ),
            expect_bad_request,
        ),
    ]
    return cases


def all_cases(vectors: list[str]) -> list[Case]:
    return (
        vector_cases(vectors)
        + seed_cases()
        + stochastic_cases(vectors)
        + invalid_parameter_cases()
        + boundary_cases()
        + constrained_cases()
    )


# ---------------------------------------------------------------------------
# Deployment run
# ---------------------------------------------------------------------------


@dataclass
class DeploymentResult:
    label: str
    base_url: str
    observations: list[Observation] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    vectors_served: dict[str, bool] = field(default_factory=dict)
    stats_before: dict[str, Any] | None = None
    stats_after: dict[str, Any] | None = None
    log_evidence: dict[str, Any] | None = None

    def by_name(self, name: str) -> Observation | None:
        for observation in self.observations:
            if observation.name == name:
                return observation
        return None

    def as_dict(self) -> dict[str, Any]:
        retain = {
            "greedy-served",
            "greedy-replay-nonstream",
            "seed-zero-a",
            "seed-zero-b",
            "seed-negative-2-a",
            "seed-negative-2-b",
            "isolation-seed-11-first",
            "isolation-seed-11-again",
        }
        return {
            "label": self.label,
            "base_url": self.base_url,
            "vectors_served": self.vectors_served,
            "checks_passed": self.checks,
            "failures": self.failures,
            "notes": self.notes,
            "system_fingerprint": next(
                (o.system_fingerprint for o in self.observations if o.system_fingerprint), None
            ),
            "stats": {
                "available": self.stats_before is not None or self.stats_after is not None,
                "draft_counter_keys": draft_counter_keys(self.stats_after),
                "note": (
                    "no draft/speculative counters exposed by /v1/stats in this build"
                    if not draft_counter_keys(self.stats_after)
                    else "draft counters observed"
                ),
            },
            "log_evidence": self.log_evidence,
            "observations": [
                observation.as_dict(observation.name in retain) for observation in self.observations
            ],
        }


def run_deployment(
    label: str,
    base_url: str,
    vectors: list[str],
    api_key: str | None,
    log_path: Path | None = None,
) -> DeploymentResult:
    result = DeploymentResult(label=label, base_url=base_url)
    result.stats_before = fetch_stats(base_url, api_key)
    for case in all_cases(vectors):
        observation = run_case(base_url, case, api_key)
        result.observations.append(observation)
        try:
            case.check(observation)
            result.checks.append(case.name)
        except AcceptanceError as error:
            result.failures.append(str(error))
    result.stats_after = fetch_stats(base_url, api_key)
    result.log_evidence = scan_dspark_log(log_path)

    # --- per-vector replay and reachability -------------------------------
    for vector in vectors:
        served = result.by_name("greedy-served" if vector == GREEDY_VECTOR else f"{vector}-served")
        result.vectors_served[vector] = bool(served is not None and served.status == 200)
        if vector == GREEDY_VECTOR:
            try:
                assert_replay(
                    [o for o in (result.by_name("greedy-served"), result.by_name("greedy-replay-nonstream")) if o],
                    "greedy replay",
                )
                result.checks.append("greedy-replay-identical")
            except AcceptanceError as error:
                result.failures.append(str(error))
            continue
        replay = [result.by_name(f"{vector}-served"), result.by_name(f"{vector}-replay-nonstream")]
        try:
            assert_replay([o for o in replay if o], f"{vector} seed=0 non-stream replay")
            result.checks.append(f"{vector}-seed0-replay-identical")
        except AcceptanceError as error:
            result.failures.append(str(error))
        stream = result.by_name(f"{vector}-replay-stream")
        if stream is not None and stream.status == 200:
            try:
                assert_replay([o for o in replay + [stream] if o], f"{vector} stream/non-stream replay")
                result.checks.append(f"{vector}-stream-nonstream-identical")
            except AcceptanceError as error:
                result.failures.append(str(error))

    # Greedy must be insensitive to sampling knobs.
    greedy = result.by_name("greedy-served")
    knobs = result.by_name("greedy-with-knobs")
    if greedy is not None and knobs is not None and greedy.status == 200 and knobs.status == 200:
        if greedy.output() != knobs.output():
            result.failures.append("greedy output changed when top_p/top_k/seed were supplied")
        else:
            result.checks.append("greedy-knob-insensitive")

    top_k_status = {
        name: (result.by_name(name).status if result.by_name(name) else None)
        for name in (
            "boundary-top-k-disabled-0",
            "boundary-top-k-disabled-negative1",
            "boundary-top-k-65",
            "boundary-top-k-1000",
            "boundary-top-k-vocab",
            "boundary-top-k-above-vocab",
        )
    }
    result.checks.append("top-k-boundary-status:" + json.dumps(top_k_status, sort_keys=True))

    # Tool replay is scoped to identical sampling parameters: the three
    # stochastic v3-t07-p90 seed=0 runs (non-stream, replay, stream) must agree on
    # normalized calls (name + parsed arguments), never on server call ids. The
    # greedy tool run is sampled under different parameters, so it is validated
    # for conformance only and is deliberately NOT compared with the stochastic
    # runs; a different greedy argument set is legal.
    stochastic_tools = [
        result.by_name("constrained-tool"),
        result.by_name("constrained-tool-replay"),
        result.by_name("constrained-tool-stream"),
    ]
    if all(run is not None and run.status == 200 for run in stochastic_tools):
        present = [run for run in stochastic_tools if run is not None]
        identities = [json.dumps(run.tool_replay(), sort_keys=True, default=str) for run in present]
        if len(set(identities)) != 1:
            result.failures.append(
                "stochastic tool replay differs across transports at identical params/seed "
                "(name+arguments must match): "
                + json.dumps([run.tool_replay() for run in present])
            )
        else:
            result.checks.append("stochastic-tool-replay-identical")
    greedy_tool = result.by_name("constrained-tool-greedy")
    if greedy_tool is not None and greedy_tool.status == 200:
        # Conformance is asserted by its own case check; replay is not required
        # across different sampling parameters.
        result.checks.append("greedy-tool-validated-separately")

    # The schema-constrained sampled vector must also replay exactly.
    constrained_sampled = result.by_name("constrained-json-schema-sampled")
    constrained_replay = result.by_name("constrained-json-schema-sampled-replay")
    if constrained_sampled is not None and constrained_replay is not None:
        try:
            assert_replay(
                [constrained_sampled, constrained_replay],
                "constrained schema sampled seed=0 replay",
            )
            result.checks.append("constrained-schema-seed0-replay-identical")
        except AcceptanceError as error:
            result.failures.append(str(error))

    # Seed replay and isolation.
    for label_name, first_name, second_name in (
        ("seed-0", "seed-zero-a", "seed-zero-b"),
        ("seed--2", "seed-negative-2-a", "seed-negative-2-b"),
    ):
        first, second = result.by_name(first_name), result.by_name(second_name)
        if first is None or second is None:
            result.notes.append(f"{label_name} observations missing")
            continue
        try:
            assert_replay([first, second], f"signed {label_name} replay")
            result.checks.append(f"{label_name}-replay-identical")
        except AcceptanceError as error:
            result.failures.append(str(error))
    isolation_first = result.by_name("isolation-seed-11-first")
    isolation_again = result.by_name("isolation-seed-11-again")
    if isolation_first is not None and isolation_again is not None:
        if isolation_first.status == 200 and isolation_again.status == 200:
            if isolation_first.output() != isolation_again.output():
                result.failures.append(
                    "seed=11 output changed after an interleaved seed=22 request (RNG contamination)"
                )
            else:
                result.checks.append("request-isolation-seed-11-stable")

    # Stochastic diversity on unconstrained vectors. Every non-greedy vector must
    # show a seed-dependent difference; if none does, either every vector drew
    # the same tokens by chance or the RNG is not request-seeded. A single
    # divergent vector is reported as coverage but is not treated as proof that
    # every vector's RNG is live.
    vectors_with_diversity: list[str] = []
    vectors_checked: list[str] = []
    for vector in vectors:
        if vector not in stochastic_vectors():
            continue
        outputs: dict[int, str] = {}
        stable = True
        for seed in (11, 22):
            runs = [
                result.by_name(f"stochastic-{vector}-seed-{seed}-run-{repeat}") for repeat in range(2)
            ]
            if any(run is None or run.status != 200 for run in runs):
                continue
            values = [run.output() for run in runs if run is not None]
            if len(set(values)) != 1:
                result.failures.append(
                    f"{vector} seed={seed} replay diverged across runs "
                    f"(lengths {[len(value) for value in values]})"
                )
                stable = False
                continue
            outputs[seed] = values[0]
        if not stable or len(outputs) < 2:
            continue
        vectors_checked.append(vector)
        result.checks.append(
            "stochastic-replay-identical:" + vector + ":" + ",".join(str(seed) for seed in sorted(outputs))
        )
        if len(set(outputs.values())) > 1:
            vectors_with_diversity.append(vector)
    if vectors_checked and not vectors_with_diversity:
        result.failures.append(
            "seeded sampling never varied across seeds in any unconstrained stochastic vector "
            f"({vectors_checked}); check that the request seed reaches the sampler"
        )
    elif vectors_with_diversity:
        result.checks.append(
            "stochastic-seed-diversity-observed:" + ",".join(vectors_with_diversity)
        )
        missing = [vector for vector in vectors_checked if vector not in vectors_with_diversity]
        if missing:
            result.notes.append(
                "vectors whose two seeds happened to agree (weak coverage, not a failure): "
                + ",".join(missing)
            )

    if draft_counter_keys(result.stats_after):
        result.checks.append(
            "draft-counters-observed:" + ",".join(draft_counter_keys(result.stats_after))
        )
    elif result.log_evidence is None:
        result.notes.append(
            "no draft evidence: /v1/stats has no draft counters and no --log-file was provided"
        )
    elif not result.log_evidence.get("available"):
        result.notes.append(f"draft evidence unavailable: {result.log_evidence.get('note')}")
    elif not any(result.log_evidence.get("counts", {}).values()):
        result.notes.append("draft evidence: engine log contains no draft/spec/mtp/dspark mention")
    else:
        result.checks.append(
            "draft-log-evidence:" + json.dumps(result.log_evidence.get("counts", {}), sort_keys=True)
        )
    return result


# ---------------------------------------------------------------------------
# Self-test: exercise every check against an in-process fake server
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def __iter__(self):
        return iter(self._body.splitlines(keepends=True))


def fake_invalid(body: dict[str, Any]) -> bool:
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    min_p = body.get("min_p")
    seed = body.get("seed")
    if temperature is not None and (
        not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2
    ):
        return True
    if top_p is not None and (
        not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1
    ):
        return True
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < -1
    ):
        return True
    if min_p is not None and (
        not isinstance(min_p, (int, float)) or not math.isfinite(min_p) or not 0 <= min_p <= 1
    ):
        return True
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        return True
    return False


def fake_rejects_schema(body: dict[str, Any]) -> bool:
    """Mirror the native strict-subset preflight for the deliberately bad schema.

    ``validate_strict_json_schema`` rejects an object schema when
    ``additionalProperties`` is not exactly ``false`` OR when ``required`` does
    not list every property (both are checked independently, so the condition is
    an OR, not an AND). The harness schema is an object, so the preflight applies
    before the adapter sees it.
    """
    response_format = body.get("response_format") or {}
    definition = response_format.get("json_schema") or {}
    if not definition.get("strict"):
        return False
    schema = definition.get("schema") or {}
    if not isinstance(schema, dict) or "properties" not in schema:
        return False
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return True
    if schema.get("additionalProperties") is not False:
        return True
    required = schema.get("required")
    if not isinstance(required, list):
        return True
    return sorted(required) != sorted(properties)


def fake_answer(body: dict[str, Any]) -> str:
    """Seed- and vector-dependent but replayable text, so replay is separable."""
    temperature = body.get("temperature")
    signature = json.dumps(
        {
            "t": temperature,
            "p": body.get("top_p"),
            "k": body.get("top_k"),
            "m": body.get("min_p"),
        },
        sort_keys=True,
    )
    if temperature in (None, 0):
        suffix = "greedy"
    else:
        suffix = f"{body.get('seed')}:{signature}"
    if body.get("response_format"):
        # Conforming on purpose: the harness asserts the served payload against
        # the request schema, so the fake server must satisfy it.
        return json.dumps(
            {"answer": "4", "confidence": 100, "tags": ["math"], "ok": True}
        )
    return "ab" + suffix


def fake_tool_call(body: dict[str, Any]) -> dict[str, Any]:
    """One protocol tool call. The id is server-generated and varies per run.

    The argument value depends on the sampling parameters: greedy and stochastic
    requests are allowed to select different tokens, so the fake returns a
    different city for the greedy tool case. That makes any accidental
    greedy-vs-stochastic replay assertion fail loudly in the self-test.
    """
    temperature = body.get("temperature")
    city = "Taipei" if temperature in (None, 0) else f"Taipei-seed-{body.get('seed')}"
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"city": city})},
    }


def fake_nonstream_response(body: dict[str, Any]) -> dict[str, Any]:
    """Mirror the real non-stream payload, including tool-call responses."""
    if body.get("tools"):
        # Real tool responses carry `content: null` and finish_reason tool_calls.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [fake_tool_call(body)],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "system_fingerprint": "fake",
            "usage": {"completion_tokens": 8},
        }
    answer = fake_answer(body)
    return {
        "choices": [
            {"message": {"role": "assistant", "content": answer, "reasoning_content": ""}, "finish_reason": "stop"}
        ],
        "system_fingerprint": "fake",
        "usage": {"completion_tokens": len(answer)},
    }


def fake_stream_frames(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror the real streamed payload, including tool-call delta accumulation."""
    if body.get("tools"):
        call = fake_tool_call(body)
        # Start delta carries id/type/name, then arguments arrive in two pieces.
        arguments = call["function"]["arguments"]
        split = max(1, len(arguments) // 2)
        return [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {"name": call["function"]["name"], "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": arguments[:split]}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": arguments[split:]}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 8}},
        ]
    answer = fake_answer(body)
    return [
        {"choices": [{"delta": {"content": answer[:1]}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": answer[1:]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": len(answer)}},
    ]


def fake_server(body: dict[str, Any]) -> FakeResponse:
    if fake_invalid(body) or fake_rejects_schema(body):
        return FakeResponse(
            400, json.dumps({"error": {"message": "invalid request", "type": ERROR_TYPE}}).encode()
        )
    if body.get("stream"):
        payload = b"".join(
            b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in fake_stream_frames(body)
        ) + b"data: [DONE]\n\n"
        return FakeResponse(200, payload)
    return FakeResponse(200, json.dumps(fake_nonstream_response(body)).encode())


def patch_fake_server(stats_sequence: list[dict[str, Any]] | None = None):
    """Install the fake transport; returns the original urlopen for restoring."""
    original = urllib.request.urlopen
    calls = {"stats": 0}

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        if request.data is None:  # /v1/stats probe
            sequence = stats_sequence or [{"http_queue_len": 0}]
            index = min(calls["stats"], len(sequence) - 1)
            calls["stats"] += 1
            return FakeResponse(200, json.dumps(sequence[index]).encode())
        return fake_server(json.loads(request.data.decode()))

    urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
    return original


def self_test() -> int:
    original = patch_fake_server(
        [{"http_queue_len": 0, "draft_proposals": 7}, {"http_queue_len": 0, "draft_proposals": 12}]
    )
    try:
        result = run_deployment("fake", "http://fake", list(VECTORS), api_key=None)
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]
    if result.failures:
        print("self-test failures:", json.dumps(result.failures, indent=2))
        return 1
    required = {
        "greedy-replay-identical",
        "greedy-knob-insensitive",
        "seed-0-replay-identical",
        "seed--2-replay-identical",
        "request-isolation-seed-11-stable",
        "draft-counters-observed:draft_proposals",
        "boundary-top-k-65",
        "boundary-top-k-1000",
        "boundary-top-k-vocab",
        "boundary-top-k-disabled-0",
        "boundary-top-k-disabled-negative1",
        "constrained-json-schema-greedy",
        "constrained-json-schema-sampled",
        "constrained-json-schema-sampled-replay",
        "constrained-schema-seed0-replay-identical",
        "constrained-tool",
        "constrained-tool-replay",
        "constrained-tool-stream",
        "constrained-tool-greedy",
        "stochastic-tool-replay-identical",
        "greedy-tool-validated-separately",
    }
    required |= {f"{vector}-seed0-replay-identical" for vector in VECTORS if vector != GREEDY_VECTOR}
    required |= {
        f"stochastic-replay-identical:{vector}:11,22" for vector in stochastic_vectors()
    }
    missing = sorted(required - set(result.checks))
    if missing:
        print("self-test did not exercise:", missing)
        return 1
    if not any(check.startswith("stochastic-seed-diversity-observed") for check in result.checks):
        print("self-test did not observe stochastic seed diversity:", result.checks)
        return 1
    if not all(result.vectors_served.values()):
        print("self-test never served every vector:", result.vectors_served)
        return 1
    print(
        f"self-test ok ({len(result.checks)} checks, {len(result.observations)} observations, "
        f"{len(result.vectors_served)} vectors)"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", help="the single native deployment URL")
    parser.add_argument("--label", default="native", help="deployment label recorded in evidence")
    parser.add_argument("--vector", action="append", choices=sorted(VECTORS), help="restrict to these sampling vectors (default: all five)")
    parser.add_argument("--output", type=Path, help="write the full evidence JSON here")
    parser.add_argument("--api-key-env", help="environment variable holding the API key; never saved")
    parser.add_argument("--log-file", type=Path, help="engine log to scan for draft/dSpark evidence")
    parser.add_argument("--self-test", action="store_true", help="run every check against an in-process fake server")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.base_url:
        parser.error("--base-url is required unless --self-test is used")

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    vectors = args.vector or list(VECTORS)
    result = run_deployment(args.label, args.base_url, vectors, api_key, args.log_file)
    status = "PASS" if not result.failures else "FAIL"
    print(
        f"[{status}] {result.label} {result.base_url}: {len(result.checks)} checks, "
        f"{len(result.failures)} failures, {len(result.notes)} notes, "
        f"vectors={result.vectors_served}"
    )
    for failure in result.failures:
        print(f"    failure: {failure}")
    for note in result.notes:
        print(f"    note: {note}")

    verdict = {
        "scope": (
            "Production native V4.1 sampling acceptance on one deployment: five sampling vectors "
            "(greedy, temperature, top-p, min-p, mix) with seed=0 replay, signed-seed replay, "
            "request isolation, invalid-parameter rejection, constrained decoding, and recorded "
            "draft evidence. Token-level distribution and dSpark sample-match equivalence are core "
            "verifier tests, not this harness. Not a throughput or quality qualification."
        ),
        "model": MODEL,
        "vectors": vectors,
        "deployment": result.as_dict(),
        "passed": not result.failures,
    }
    if args.output:
        args.output.write_text(json.dumps(verdict, indent=2) + "\n")
        print(f"evidence: {args.output}")
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
