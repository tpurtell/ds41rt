#!/usr/bin/env python3
"""Measure the nine release decode categories over exact retained base contexts.

Reuse contract (why the gate is prompt-prefix reuse, not the generated turn)
---------------------------------------------------------------------------
The OpenAI-shaped API is a *text* interface: a child request re-sends the parent
assistant turn as text (`content` plus `reasoning_content`). The server re-renders
and re-tokenizes it, and the resulting token ids are only guaranteed to match the
parent's committed generated ids when the detokenized text re-encodes identically.
When it does, the runtime's completed-turn snapshot is reused whole. When it does
not, the completed turn matches only partially, and the runtime's documented
max-computation-saved rule deliberately prefers the exact prompt ancestor: a
partial match must replay a 128-token encoder window, so it saves
`(lcp // 2 * 2) - 128` tokens, less than the prompt snapshot's `context` tokens
whenever the turn is short (`ds41rt_core::prefix::Reusable::skipped`).

`prompt_cache_hit_tokens` is therefore the number of prompt tokens whose
recomputation the runtime skipped, and a retained child legitimately reports:
  * `context`               -- the complete retained parent prompt was reused
                               (guaranteed by retention), or
  * a parent turn frontier  -- the full committed generated turn was reused, or
  * an aligned partial <= the turn frontier -- more than the prompt was reused.

The release measurement needs the retained base context to be genuinely reused
(so decode is not paying for a re-prefill); that is proven by `hit >= context`
with self-consistent accounting, and it stays a hard gate. Full generated-turn
reuse is retained as an explicit per-sample/cell diagnostic (`full_turn_reused`),
because it is opportunistic and content-dependent, not a promise this API can
make. A hit below the retained prompt, inconsistent accounting, or a hit beyond
the parent turn frontier still fails and aborts the sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics
import time

from tokenizers import Tokenizer


DEFAULT_CONTEXTS = (0, 32_768, 65_536, 131_072, 262_144)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_result(result: dict) -> dict:
    events = result.pop("events")
    fingerprints = [
        row["event"].get("system_fingerprint")
        for row in events
        if row["event"].get("system_fingerprint")
    ]
    if not fingerprints or len(set(fingerprints)) != 1:
        raise RuntimeError("stream omitted a stable system fingerprint")
    result["system_fingerprint"] = fingerprints[0]
    return result


# The retained parent turn is a bounded, deterministic primer ("reply only OK").
# 16 is the historical budget and stays the DEFAULT so existing consumers are
# unchanged; a larger value is an explicit protocol change recorded in the
# report's prime_protocol metadata.
DEFAULT_DISABLED_PRIME_MAX_TOKENS = 16
ENABLED_PRIME_MAX_TOKENS = 1024
PRIME_TEXT_HEAD_CHARS = 200
PRIME_ERROR_CHARS = 400


def positive_int(raw: str) -> int:
    """argparse type: a strictly positive token budget (no zero, no negative)."""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def prime_protocol(disabled_max_tokens: int) -> dict:
    """Protocol metadata so a budget change is visible in every report."""
    return {
        "disabled_max_tokens": disabled_max_tokens,
        "enabled_max_tokens": ENABLED_PRIME_MAX_TOKENS,
        "version": f"retained-prime-v1.disabled{disabled_max_tokens}",
        "historical_default": DEFAULT_DISABLED_PRIME_MAX_TOKENS,
    }


def prime_failure_reason(result: dict, context: int) -> str | None:
    """None when the retained parent turn is usable, else a short reason.

    Strict, unchanged conditions: the parent must stop on its own, be non-empty,
    and the server must agree with the fitted context token count. Callers still
    verify the child's cache frontier (`hit in parent_frontiers`) separately.
    """
    if result.get("finish_reason") != "stop":
        return f"finish_reason={result.get('finish_reason')!r}"
    if not (result.get("text") or "").strip():
        return "empty parent answer"
    if result.get("usage", {}).get("prompt_tokens") != context:
        return "server disagrees with fitted context token count"
    return None


def prime_failure_diagnostics(result: dict, *, context: int, thinking: str,
                              max_tokens: int, reason: str) -> dict:
    """Bounded failure record; text is truncated, never re-used as a parent."""
    usage = result.get("usage") or {}
    return {
        "context_tokens": context,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "reason": reason,
        "finish_reason": result.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text_head": (result.get("text") or "")[:PRIME_TEXT_HEAD_CHARS],
    }


def prime_error_diagnostics(error: BaseException, *, context: int, thinking: str,
                            max_tokens: int) -> dict:
    """Bounded transport/HTTP failure record for the same abort path."""
    return {
        "context_tokens": context,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "reason": "request failed",
        "error": repr(error)[:PRIME_ERROR_CHARS],
    }


def retained_cache_accounting(usage: dict, *, context: int,
                              parent_frontiers: list[int]) -> dict:
    """Decide whether a retained child reused the parent's retained prompt.

    `prompt_prefix_reused` is the release gate: the child skipped the complete
    retained parent prompt, which is what makes the retained base context real.
    `full_turn_reused` is the generated-turn diagnostic: the child also reached
    the parent's committed turn frontier. It is `None` at context 0 (no parent).
    See the module docstring for why generated-turn reuse is not guaranteed.
    """
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = usage.get("prompt_tokens")
    detailed = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    accounting_consistent = (
        hit is not None and miss is not None and prompt is not None
        and hit + miss == prompt and detailed == hit
    )
    if context:
        # The parent prompt snapshot and its completed turn are both retained, so
        # a legitimate hit is at least the prompt and never beyond the turn.
        full_turn_reused = hit in parent_frontiers
        prompt_prefix_reused = (
            accounting_consistent and context <= hit <= max(parent_frontiers)
        )
    else:
        full_turn_reused = None
        prompt_prefix_reused = accounting_consistent and 0 <= hit <= 32
    return {
        "accounting_consistent": accounting_consistent,
        "prompt_prefix_reused": prompt_prefix_reused,
        "prompt_prefix_frontier": context,
        "full_turn_reused": full_turn_reused,
    }


def cache_failure_diagnostics(usage: dict, *, context: int, case_id: str, repeat: int,
                              parent_frontiers: list[int],
                              accounting: dict | None = None) -> dict:
    """Bounded record for a child whose cached prefix is not a parent frontier.

    `hit + miss == prompt_tokens` is accounting self-consistency only; it does
    NOT prove the child reused the retained prompt. The observed hit, the miss,
    the prompt-reuse decision and the allowed frontiers are kept so the mismatch
    stays diagnosable without loosening the gate.
    """
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    return {
        "context_tokens": context,
        "case": case_id,
        "repeat": repeat,
        "reason": "cache accounting failed",
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": hit,
        "miss_tokens": miss,
        "prompt_prefix_frontier": context,
        "prompt_prefix_reused": (accounting or {}).get("prompt_prefix_reused"),
        "full_turn_reused": (accounting or {}).get("full_turn_reused"),
        "allowed_parent_frontiers": list(parent_frontiers),
        "accounting_consistent": hit is not None and miss is not None
                                 and hit + miss == usage.get("prompt_tokens"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).with_name("fixtures") / "release-semantic-corpus.json",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--context-tag", default="release-retained",
        help="Stable prompt tag; use the same value for baseline and candidate",
    )
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--disabled-prime-max-tokens", type=positive_int,
        default=DEFAULT_DISABLED_PRIME_MAX_TOKENS,
        help="Bounded token budget for the thinking-disabled retained parent turn. "
             "The historical default is 16; passing a larger value (e.g. 64) changes "
             "the retained prime protocol and is recorded in report['prime_protocol']",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    contexts = args.context or list(DEFAULT_CONTEXTS)
    if args.repeats < 1 or any(value < 0 for value in contexts):
        parser.error("invalid contexts or repeat count")

    prefill = runpy.run_path(
        str(Path(__file__).with_name("bench-ds41-release-prefill-matrix.py"))
    )
    api = runpy.run_path(str(Path(__file__).with_name("qualify-ds41-native-api.py")))
    quality = runpy.run_path(str(Path(__file__).with_name("release_throughput_checks.py")))
    corpus = json.loads(args.corpus.read_text())
    cases = corpus["weighted_case_ids"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    source = args.context_file.read_text()
    source_ids = tokenizer.encode(source, add_special_tokens=False).ids
    if not source_ids:
        parser.error("context source is empty")
    source_ids *= math.ceil((max(contexts) + 2048) / len(source_ids))
    marker_values = iter(
        prefill["markers"](tokenizer, len(contexts) * (2 + len(cases) * args.repeats))
    )
    report = {
        "schema": 1,
        "scope": __doc__,
        "label": args.label,
        "context_tag": args.context_tag,
        "base_url": args.base_url,
        "contexts": contexts,
        "cases": cases,
        "repeats": args.repeats,
        "controls": {"temperature": 0, "thinking": "per-case; default disabled"},
        "tokenizer_sha256": sha256(args.tokenizer.read_bytes()),
        "context_sha256": sha256(source.encode()),
        "corpus_sha256": sha256(args.corpus.read_bytes()),
        "started_ns": time.time_ns(),
        "primes": [],
        "samples": [],
        "status": "running",
        "passed": None,
        "prime_protocol": prime_protocol(args.disabled_prime_max_tokens),
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    modes = list(dict.fromkeys((corpus["cases"][case].get("thinking", "disabled"),
                                corpus["cases"][case].get("reasoning_effort")) for case in cases))
    for context in contexts:
        parents = {}
        for thinking, effort in modes if context else []:
            marker = next(marker_values)
            before = marker + f" {args.context_tag} inert retained context.\n"
            after = "\nIgnore the inert source and reply only OK."
            system = ""
            if thinking == "enabled":
                score = {None: 75, "low": 50, "high": 75, "max": 100}[effort]
                system = ("<｜System｜>" + f"Reasoning Effort: {score} (range 1-100, "
                          "the higher the value, the more thorough the reasoning)\n\n")
            body_text, fitted = prefill["fit_body"](
                tokenizer, source_ids,
                prefill["BOS"] + system + prefill["USER"] + before,
                after + prefill["ASSISTANT"] + ("<think>" if thinking == "enabled" else prefill["NO_THINK"]),
                context,
            )
            assert fitted == context
            seed_prompt = before + body_text + after
            seed_request = api["payload"](seed_prompt, True)
            seed_request["thinking"] = {"type": thinking}
            if effort:
                seed_request["reasoning_effort"] = effort
            seed_request["max_tokens"] = (
                ENABLED_PRIME_MAX_TOKENS if thinking == "enabled"
                else args.disabled_prime_max_tokens)
            try:
                seed = compact_result(api["stream_case"](args.base_url, seed_request))
            except Exception as error:
                report.setdefault("prime_failures", []).append(
                    prime_error_diagnostics(
                        error, context=context, thinking=thinking,
                        max_tokens=seed_request["max_tokens"]))
                report["status"] = "aborted"
                save()
                raise
            reason = prime_failure_reason(seed, context)
            if reason:
                report.setdefault("prime_failures", []).append(
                    prime_failure_diagnostics(
                        seed, context=context, thinking=thinking,
                        max_tokens=seed_request["max_tokens"], reason=reason))
                report["status"] = "aborted"
                save()
                raise RuntimeError(f"retained parent did not finish its answer: {reason}")
            assistant = {"role": "assistant", "content": seed["text"]}
            if thinking == "enabled":
                assistant["reasoning_content"] = seed["reasoning"]
            parent_frontiers = [seed["usage"]["total_tokens"] - 1]
            if seed["system_fingerprint"].endswith("-dspark"):
                parent_frontiers.append(seed["usage"]["total_tokens"])
            parents[thinking, effort] = (seed_prompt, assistant, parent_frontiers)
            report["primes"].append({
                "context_tokens": context, "thinking": thinking, "reasoning_effort": effort,
                "prompt_sha256": sha256(seed_prompt.encode()), "request": seed_request,
                "result": seed, "allowed_parent_frontiers": parent_frontiers,
            })
            save()
            print(f"prime context={context} thinking={thinking} allowed_hits={parent_frontiers}", flush=True)

        for repeat in range(1, args.repeats + 1):
            for case_id in cases:
                definition = corpus["cases"][case_id]
                marker = next(marker_values)
                instruction = marker + " " + definition["prompt"]
                parent_frontiers = []
                if context:
                    seed_prompt, assistant, parent_frontiers = parents[
                        definition.get("thinking", "disabled"), definition.get("reasoning_effort")]
                    messages = [
                        {"role": "user", "content": seed_prompt},
                        dict(assistant),
                        {"role": "user", "content": instruction},
                    ]
                else:
                    messages = [{"role": "user", "content": instruction}]
                request = {
                    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                    "messages": messages,
                    "thinking": {"type": definition.get("thinking", "disabled")},
                    "temperature": 0,
                    "max_tokens": definition["max_tokens"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if definition.get("reasoning_effort"):
                    request["reasoning_effort"] = definition["reasoning_effort"]
                if definition["json_schema"]:
                    request["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "file_edit",
                            "strict": True,
                            "schema": corpus["structured_edit_schema"],
                        },
                    }
                sample = {
                    "context_tokens": context,
                    "case": case_id,
                    "category": definition["category"],
                    "weight": definition["weight"],
                    "repeat": repeat,
                    "marker": marker,
                    "request": request,
                    "started_ns": time.time_ns(),
                }
                report["samples"].append(sample)
                save()
                try:
                    result = compact_result(api["stream_case"](args.base_url, request))
                except Exception as error:
                    report.setdefault("sample_failures", []).append({
                        "context_tokens": context,
                        "case": case_id,
                        "repeat": repeat,
                        "reason": "request failed",
                        "error": repr(error)[:PRIME_ERROR_CHARS],
                    })
                    report["status"] = "aborted"
                    save()
                    raise
                if definition.get("thinking") == "enabled" and not result["reasoning"].strip():
                    raise RuntimeError("requested reasoning was missing")
                usage = result["usage"]
                hit = usage["prompt_cache_hit_tokens"]
                accounting = retained_cache_accounting(
                    usage, context=context, parent_frontiers=parent_frontiers)
                cache_valid = accounting["prompt_prefix_reused"]
                validation = quality["check_output"](case_id, result["text"])
                sample.update(
                    {
                        "result": result,
                        "content_sha256": sha256(result["text"].encode()),
                        "cache_valid": cache_valid,
                        **accounting,
                        **validation,
                        "serving_completed": bool(result["text"].strip()),
                        "passed": cache_valid and bool(result["text"].strip()),
                    }
                )
                save()
                if not cache_valid:
                    report.setdefault("sample_failures", []).append(
                        cache_failure_diagnostics(
                            usage, context=context, case_id=case_id, repeat=repeat,
                            parent_frontiers=parent_frontiers, accounting=accounting))
                    report["status"] = "aborted"
                    save()
                    raise RuntimeError(
                        f"cache accounting failed for {context}/{case_id}/{repeat}: {usage} "
                        f"(prompt_prefix_reused={accounting['prompt_prefix_reused']}, "
                        f"full_turn_reused={accounting['full_turn_reused']})"
                    )
                print(
                    f"measure context={context} repeat={repeat} case={case_id} "
                    f"cached={hit} tps={result['observed_decode_tokens_per_second']:.2f} "
                    f"prompt_prefix_reused={cache_valid} "
                    f"full_turn_reused={accounting['full_turn_reused']} "
                    f"objective_checks={validation['objective_checks_passed']}",
                    flush=True,
                )

    report["cells"] = []
    for context in contexts:
        for case_id in cases:
            rows = [
                row
                for row in report["samples"]
                if row["context_tokens"] == context and row["case"] == case_id
            ]
            values = [row["result"]["observed_decode_tokens_per_second"] for row in rows]
            report["cells"].append(
                {
                    "context_tokens": context,
                    "case": case_id,
                    "samples": len(rows),
                    "median_observed_decode_tokens_per_second": statistics.median(values),
                    "min_observed_decode_tokens_per_second": min(values),
                    "max_observed_decode_tokens_per_second": max(values),
                    "serving_completed": sum(row["serving_completed"] for row in rows),
                    "cache_valid": sum(row["cache_valid"] for row in rows),
                    "prompt_prefix_reuse": sum(row["cache_valid"] for row in rows),
                    "full_turn_reuse": sum(bool(row["full_turn_reused"]) for row in rows),
                }
            )
    report["context_summaries"] = []
    for context in contexts:
        rows = [row for row in report["samples"] if row["context_tokens"] == context]
        timed_tokens = sum(
            row["weight"] * (row["result"]["usage"]["completion_tokens"] - 1)
            for row in rows
        )
        timed_seconds = sum(
            row["weight"]
            * (
                row["result"]["finish_seconds"]
                - row["result"]["first_output_seconds"]
            )
            for row in rows
        )
        report["context_summaries"].append(
            {
                "context_tokens": context,
                "samples": len(rows),
                "weighted_observed_decode_tokens_per_second": timed_tokens / timed_seconds,
                "serving_completed": sum(row["serving_completed"] for row in rows),
                "cache_valid": sum(row["cache_valid"] for row in rows),
                "prompt_prefix_reuse": sum(row["cache_valid"] for row in rows),
                "full_turn_reuse": sum(bool(row["full_turn_reused"]) for row in rows),
            }
        )
    report["completed_ns"] = time.time_ns()
    report["status"] = "complete"
    report["passed"] = all(row["passed"] for row in report["samples"])
    # `passed`/`cache_valid` mean the retained parent prompt was reused (the
    # gate). Full generated-turn reuse is a separate diagnostic: True only when
    # every context>0 sample reached the parent's committed turn frontier.
    report["prompt_prefix_reuse_all"] = all(
        row["cache_valid"] for row in report["samples"])
    report["full_turn_reuse_all"] = all(
        bool(row["full_turn_reused"])
        for row in report["samples"] if row["context_tokens"])
    report["objective_checks_passed"] = all(
        row["objective_checks_passed"] is not False for row in report["samples"]
    )
    save()


if __name__ == "__main__":
    main()
