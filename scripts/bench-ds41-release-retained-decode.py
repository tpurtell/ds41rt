#!/usr/bin/env python3
"""Measure the nine release decode categories over exact retained base contexts."""

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
        "passed": False,
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for context in contexts:
        seed_prompt = None
        seed_content = None
        parent_frontiers: list[int] = []
        if context:
            marker = next(marker_values)
            before = marker + f" {args.context_tag} inert retained context.\n"
            after = "\nIgnore the inert source and reply only OK."
            body_text, fitted = prefill["fit_body"](
                tokenizer,
                source_ids,
                prefill["BOS"] + prefill["USER"] + before,
                after + prefill["ASSISTANT"] + prefill["NO_THINK"],
                context,
            )
            assert fitted == context
            seed_prompt = before + body_text + after
            seed_request = api["payload"](seed_prompt, True)
            seed_request["max_tokens"] = 16
            seed = compact_result(api["stream_case"](args.base_url, seed_request))
            if seed["usage"]["prompt_tokens"] != context:
                raise RuntimeError("server disagrees with fitted context token count")
            seed_content = seed["text"]
            parent_frontiers = [seed["usage"]["total_tokens"] - 1]
            if seed["system_fingerprint"].endswith("-dspark"):
                parent_frontiers.append(seed["usage"]["total_tokens"])
            report["primes"].append(
                {
                    "context_tokens": context,
                    "prompt_sha256": sha256(seed_prompt.encode()),
                    "request": seed_request,
                    "result": seed,
                    "allowed_parent_frontiers": parent_frontiers,
                }
            )
            save()
            print(
                f"prime context={context} allowed_hits={parent_frontiers}", flush=True
            )

        for repeat in range(1, args.repeats + 1):
            for case_id in cases:
                definition = corpus["cases"][case_id]
                marker = next(marker_values)
                instruction = marker + " " + definition["prompt"]
                if context:
                    messages = [
                        {"role": "user", "content": seed_prompt},
                        {"role": "assistant", "content": seed_content},
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
                result = compact_result(api["stream_case"](args.base_url, request))
                if definition.get("thinking") == "enabled" and not result["reasoning"].strip():
                    raise RuntimeError("requested reasoning was missing")
                usage = result["usage"]
                hit = usage["prompt_cache_hit_tokens"]
                miss = usage["prompt_cache_miss_tokens"]
                cache_valid = (
                    hit + miss == usage["prompt_tokens"]
                    and usage["prompt_tokens_details"]["cached_tokens"] == hit
                    and (
                        hit in parent_frontiers
                        if context
                        else 0 <= hit <= 32
                    )
                )
                validation = quality["check_output"](case_id, result["text"])
                sample.update(
                    {
                        "result": result,
                        "content_sha256": sha256(result["text"].encode()),
                        "cache_valid": cache_valid,
                        **validation,
                        "serving_completed": bool(result["text"].strip()),
                        "passed": cache_valid and bool(result["text"].strip()),
                    }
                )
                save()
                if not cache_valid:
                    raise RuntimeError(
                        f"cache accounting failed for {context}/{case_id}/{repeat}: {usage}"
                    )
                print(
                    f"measure context={context} repeat={repeat} case={case_id} "
                    f"cached={hit} tps={result['observed_decode_tokens_per_second']:.2f} "
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
            }
        )
    report["completed_ns"] = time.time_ns()
    report["passed"] = all(row["passed"] for row in report["samples"])
    report["objective_checks_passed"] = all(
        row["objective_checks_passed"] is not False for row in report["samples"]
    )
    save()


if __name__ == "__main__":
    main()
