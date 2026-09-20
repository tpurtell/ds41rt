#!/usr/bin/env python3
"""Corpus-driven TP4EP1 vs TP2EP2 end-to-end qualification runner.

Reuses the real client (``scripts/qualify-ds41-native-api.py``) and content
checks (``scripts/release_throughput_checks.py``); ``validate``/``compare`` are
CPU-only and ``run`` needs a live service. Adversarial CPU tests live in
``scripts/tests/test_tp_ep_e2e_protocol.py``. SSE gaps are ``inter_chunk`` (not
per-token); ``text_exact`` is text equality, not runtime bit-exactness, and
divergence needs per-output review evidence.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import runpy
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLIENT_PATH = REPO / "scripts" / "qualify-ds41-native-api.py"
CHECKS_PATH = REPO / "scripts" / "release_throughput_checks.py"
CORPUS_DEFAULT = REPO / "scripts" / "fixtures" / "tp-ep-e2e-corpus.jsonl"

API = runpy.run_path(str(CLIENT_PATH))
CHECK_OUTPUT = runpy.run_path(str(CHECKS_PATH))["check_output"]

CHECK_KINDS = ("throughput_case", "nonempty", "exact_normalized", "code_block_and_prose")
REVIEW_KINDS = ("matched_logits", "task_quality", "manual_review")
# Identity fields that must match across the two arms. Controls that are not
# applied on the release native path (for example SPARK_REDUCTION_MIN_ROWS, a
# run-wip-only key) are deliberately absent.
REQUIRED_IDENTITY = (
    "source_commit", "source_artifact", "model", "revision", "rtx_gpus",
    "resolved_rtx_expert_layers", "spark_first_layer", "spark_count", "concurrency",
    "max_context_tokens", "max_output_tokens", "kv_pool", "prefix_cache_entries",
    "prefill_batch_tokens", "worker_capacity", "dspark", "dspark_draft_limit",
    "spark_device_budget_bytes", "power_caps_watts",
)
ALLOWED_DIFFERENCES = ("spark_tp", "spark_ep", "spark_artifact_roles", "release_identity_full")  # families checked separately
# Structures with stricter rules than plain equality.
REQUIRED_IDENTITY_STRUCTURES = ("source_snapshots", "expert_tp_manifests")
# Flags run.sh actually passes for controls declared in startup_metadata. A
# declared control missing from the applied argv is a tier A failure: the value
# was recorded but never applied.
REQUIRED_COORDINATOR_FLAGS = ("--rtx-gpus", "--prefill-batch-tokens", "--concurrency",
                              "--prefix-cache-entries", "--max-context-tokens", "--max-output-tokens")
REQUIRED_WORKER_FLAGS = ("--rank", "--world", "--capacity", "--device-budget-bytes", "--first-layer")
TOPOLOGY_FLAGS = ("--spark-tp", "--spark-ep")
KNOWN_COORDINATOR_FLAGS = set(REQUIRED_COORDINATOR_FLAGS) | set(TOPOLOGY_FLAGS) | {
    "--kv-pool-size", "--rtx-expert-layers", "--dspark", "--dspark-draft-limit"}
KNOWN_WORKER_FLAGS = set(REQUIRED_WORKER_FLAGS) | {"--capacity"} | set(TOPOLOGY_FLAGS)
SUPPORTED_WORKER_CAPACITIES = (1, 16, 80, 256, 1024, 4096)
_CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n.*?\n```", re.DOTALL | re.IGNORECASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load_corpus(path: Path):
    records = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append((lineno, json.loads(line)))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {error}")
    return records


def validate_corpus(records) -> dict:
    errors, header, controls, matrix, gates = [], None, None, None, None
    cases, prefills = [], []
    for lineno, obj in records:
        kind = obj.get("kind")
        if kind == "corpus":
            header = obj
        elif kind == "controls":
            controls = obj
        elif kind == "matrix":
            matrix = obj
        elif kind == "case":
            cases.append(obj)
        elif kind == "prefill":
            prefills.append(obj)
        elif kind == "gates":
            gates = obj
        elif kind != "future_scope":
            errors.append(f"line {lineno}: unknown kind {kind!r}")
    for label, value in (("corpus header", header), ("controls", controls), ("gates", gates)):
        if value is None:
            errors.append(f"missing {label}")
    concurrency = repeats = None
    if matrix is None:
        errors.append("missing matrix")
    else:
        concurrency = matrix.get("concurrency")
        if not isinstance(concurrency, list) or not concurrency or any(
            not isinstance(c, int) or not 1 <= c <= 16 for c in concurrency
        ):
            errors.append("matrix.concurrency must be integers in 1..16")
        elif concurrency != sorted(set(concurrency)):
            errors.append("matrix.concurrency must be sorted and unique")
        repeats = matrix.get("repeats")
        if not isinstance(repeats, int) or repeats < 1:
            errors.append("matrix.repeats must be >= 1")
    ids = [c.get("id") for c in cases] + [p.get("id") for p in prefills]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case/prefill ids")
    if not cases:
        errors.append("no cases")
    for case in cases:
        label = case.get("id")
        if not isinstance(case.get("id"), str) or not case["id"]:
            errors.append("case without id")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            errors.append(f"case {label!r}: prompt required")
        if not isinstance(case.get("max_tokens"), int) or case["max_tokens"] < 1:
            errors.append(f"case {label!r}: max_tokens must be >= 1")
        check = case.get("check")
        if not isinstance(check, dict) or check.get("kind") not in CHECK_KINDS:
            errors.append(f"case {label!r}: unsupported check {check!r}")
        elif check["kind"] == "throughput_case" and not check.get("case"):
            errors.append(f"case {label!r}: throughput_case needs a case id")
        elif check["kind"] == "exact_normalized" and not check.get("count_to"):
            errors.append(f"case {label!r}: exact_normalized needs count_to")
    for prefill in prefills:
        if not isinstance(prefill.get("target_tokens"), int) or prefill["target_tokens"] < 1:
            errors.append(f"prefill {prefill.get('id')!r}: target_tokens must be >= 1")
    return dict(passed=not errors, errors=errors, cases=len(cases), prefills=len(prefills), concurrency=concurrency, repeats=repeats)


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(max(1, math.ceil(pct / 100.0 * len(ordered))), len(ordered)) - 1]


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=percentile(values, 95), min=min(values), max=max(values))


def ms(seconds):
    return None if seconds is None else seconds * 1000.0


def content_check(check: dict, text: str) -> dict:
    kind = check["kind"]
    stripped = text.strip()
    if kind == "throughput_case":
        result = CHECK_OUTPUT(check["case"], text)
        return dict(kind=kind, case=check["case"], response_nonempty=bool(result["response_nonempty"]),
                    objective_checks=result.get("objective_checks"),
                    objective_checks_passed=result.get("objective_checks_passed"))
    if kind == "nonempty":
        minimum = int(check.get("min_chars", 1))
        return dict(kind=kind, response_nonempty=bool(stripped), min_chars=minimum,
                    objective_checks_passed=bool(stripped) and len(stripped) >= minimum)
    if kind == "exact_normalized":
        expected = [str(i) for i in range(1, int(check["count_to"]) + 1)]
        observed = [p.strip() for p in stripped.split(check.get("separator", ","))] if stripped else []
        return dict(kind=kind, response_nonempty=bool(stripped), objective_checks_passed=observed == expected)
    if kind == "code_block_and_prose":
        block = _CODE_FENCE.search(text)
        prose = _CODE_FENCE.sub("", text).strip()
        return dict(kind=kind, response_nonempty=bool(stripped),
                    objective_checks_passed=bool(block) and bool(prose))
    raise SystemExit(f"unsupported check {kind!r}")


def chunk_seconds(record: dict):
    times = []
    for event in record.get("events", []):
        for choice in event.get("event", {}).get("choices", []):
            if choice.get("delta", {}).get("content"):
                times.append(event["seconds"])
    return times


def inter_chunk_stats(record: dict) -> dict:
    times = chunk_seconds(record)
    gaps = [ms(b - a) for a, b in zip(times, times[1:])]
    if not gaps:
        return dict(samples=0, mean_ms=None, median_ms=None, p95_ms=None,
                    definition="gap between SSE content events; an event may carry multiple tokens")
    return dict(samples=len(gaps), mean_ms=statistics.mean(gaps), median_ms=statistics.median(gaps),
                p95_ms=percentile(gaps, 95),
                definition="gap between SSE content events; an event may carry multiple tokens")


def run_one(case: dict, index: int, base_url: str, nonce: str) -> dict:
    prompt = f"tp-ep-e2e {nonce} {index}. {case['prompt']}"
    body = API["payload"](prompt, True)
    body["max_tokens"] = case["max_tokens"]
    body["thinking"] = {"type": case.get("thinking", "disabled")}
    started = time.perf_counter()
    try:
        record = API["stream_case"](base_url, body)
        usage = record["usage"]
        return dict(index=index, passed=True, start=started, ttft_ms=ms(record["first_content_seconds"]),
                    first_output_ms=ms(record["first_output_seconds"]),
                    reasoning_ms=ms(record["first_content_seconds"] - record["first_output_seconds"]),
                    finish_ms=ms(record["finish_seconds"]), completion_tokens=usage["completion_tokens"],
                    prompt_tokens=usage["prompt_tokens"],
                    prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
                    observed_decode_tokens_per_second=record["observed_decode_tokens_per_second"],
                    inter_chunk=inter_chunk_stats(record), text=record["text"], reasoning=record["reasoning"],
                    events=record["events"], usage=usage, content_chunks=len(chunk_seconds(record)),
                    observed_decode_tokens_per_second_approximate=True,
                    observed_decode_tokens_per_second_basis=(
                        "(completion_tokens-1)/(finish-first_output) assumes one token in the first output chunk; with "
                        "k>1 it OVERestimates the remaining rate (N-k)/span and no per-token timestamps exist"),
                    ttft_excludes_reasoning=True,
                    content_check=content_check(case["check"], record["text"]))
    except Exception as error:  # noqa: BLE001 - the record must retain the failure
        return dict(index=index, passed=False, start=started, error=repr(error),
                    partial=getattr(error, "record", None))


def aggregate_tps(rows):
    # A single request (C1) still has a valid span and denominator; do not
    # discard it. Only a degenerate span yields None.
    ok = [r for r in rows if r.get("passed") and r.get("finish_ms") is not None]
    if not ok:
        return None
    begin = min(r["start"] + ((r.get("first_output_ms") or r.get("ttft_ms") or 0.0) / 1000.0) for r in ok)
    end = max(r["start"] + (r["finish_ms"] / 1000.0) for r in ok)
    return None if end <= begin else sum(r["completion_tokens"] - 1 for r in ok) / (end - begin)


def per_concurrency_summary(case_records, concurrency):
    """Per-repeat aggregate throughput samples, then stats over them (one span
    per repeat; merging repeats would add idle gaps and bias the rate down)."""
    summary = []
    for c in concurrency:
        samples = [aggregate_tps(rec["rows"]) for rec in case_records if rec["concurrency"] == c]
        samples = [sample for sample in samples if sample is not None]
        summary.append(dict(concurrency=c, per_repeat_aggregate_decode_tps=samples,
                            aggregate_decode_tps=summarize(samples)))
    return summary


def cmd_validate(args) -> int:
    summary = validate_corpus(load_corpus(args.corpus))
    summary["corpus"] = str(args.corpus)
    summary["corpus_sha256"] = sha256(args.corpus)
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


def cmd_run(args) -> int:
    records = load_corpus(args.corpus)
    summary = validate_corpus(records)
    if not summary["passed"]:
        raise SystemExit("corpus invalid: " + "; ".join(summary["errors"]))
    by_kind = {}
    for _, obj in records:
        by_kind.setdefault(obj["kind"], []).append(obj)
    controls, matrix, cases = by_kind["controls"][0], by_kind["matrix"][0], by_kind["case"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
        if not cases:
            raise SystemExit("no corpus case matches --case")
    concurrency = args.concurrency or matrix["concurrency"]
    repeats = args.repeats or matrix["repeats"]
    startup = json.loads(args.startup_metadata.read_text()) if args.startup_metadata else None
    report = dict(arm=args.arm, base_url=args.base_url, corpus=str(args.corpus),
                  corpus_sha256=sha256(args.corpus), controls=controls,
                  matrix=dict(concurrency=concurrency, repeats=repeats), startup_metadata=startup,
                  probe=None, cases=[], failures=[], passed=False,
                  scope="One four-Spark TP4EP1/TP2EP2 arm: decode TTFT, inter-chunk latency and decode tokens/s only; prefill is separate.")
    try:
        with API["open_request"](args.base_url, API["payload"]("What is 2 + 2? Answer with just the number.")) as response:
            first = json.load(response)
        content = first["choices"][0]["message"]["content"].strip()
        report["probe"] = dict(passed=content == "4", content=content, model=first.get("model"),
                               system_fingerprint=first.get("system_fingerprint"))
    except Exception as error:  # noqa: BLE001
        report["probe"] = dict(passed=False, error=repr(error))
    write_json(args.output, report)
    failures = [] if report["probe"].get("passed") else ["readiness probe failed"]
    if startup is None:
        failures.append("no --startup-metadata captured")
    for case in cases:
        # Attach the case object before any request so an interruption keeps the
        # in-progress case, and update records/summary incrementally.
        case_entry = dict(id=case["id"], category=case["category"], max_tokens=case["max_tokens"],
                          summary=per_concurrency_summary([], concurrency),
                          greedy_repeat_consistent=None, records=[])
        report["cases"].append(case_entry)
        write_json(args.output, report)
        case_records = case_entry["records"]
        for c in concurrency:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    warmup = next(pool.map(lambda i: run_one(case, i, args.base_url, controls["nonce"]), range(1)))
            except Exception as error:  # noqa: BLE001
                warmup = dict(passed=False, error=repr(error))
            for repeat in range(repeats):
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    rows = list(pool.map(lambda i: run_one(case, i, args.base_url, controls["nonce"]), range(c)))
                case_records.append(dict(concurrency=c, repeat=repeat + 1,
                                         warmup_passed=bool(warmup.get("passed")), rows=rows))
                case_entry["summary"] = per_concurrency_summary(case_records, concurrency)
                write_json(args.output, report)
        consistent = []
        for c in concurrency:
            group = [r for r in case_records if r["concurrency"] == c]
            for index in range(c):
                texts = {r["rows"][index].get("text") for r in group if r["rows"][index].get("passed")}
                consistent.append(len(texts) <= 1)
        case_entry["greedy_repeat_consistent"] = all(consistent)
        if not case_entry["greedy_repeat_consistent"]:
            failures.append(f"{case['id']} greedy output changed across repeats")
        for rec in case_records:
            for row in rec["rows"]:
                where = f"{case['id']} C{rec['concurrency']} repeat {rec['repeat']} request {row['index']}"
                if not row.get("passed"):
                    failures.append(f"{where} failed")
                elif not (row.get("content_check") or {}).get("objective_checks_passed"):
                    failures.append(f"{where} content check")
        case_entry["summary"] = per_concurrency_summary(case_records, concurrency)
        write_json(args.output, report)
    report["failures"] = failures
    report["passed"] = not failures
    write_json(args.output, report)
    print(json.dumps(dict(arm=args.arm, passed=report["passed"], failures=len(failures), output=str(args.output)), indent=2))
    return 0 if report["passed"] else 1


def arm_metrics(arm: dict) -> dict:
    metrics = {}
    for case in arm.get("cases", []):
        for rec in case.get("records", []):
            key = f"{case['id']}|C{rec['concurrency']}"
            bucket = metrics.setdefault(key, dict(ttft_ms=[], inter_chunk_ms=[], per_stream_decode_tps=[], aggregate_decode_tps=[]))
            rows = [r for r in rec["rows"] if r.get("passed")]
            for row in rows:
                if row.get("ttft_ms") is not None:
                    bucket["ttft_ms"].append(row["ttft_ms"])
                chunk = (row.get("inter_chunk") or {}).get("median_ms")
                if chunk is not None:
                    bucket["inter_chunk_ms"].append(chunk)
                if row.get("observed_decode_tokens_per_second") is not None:
                    bucket["per_stream_decode_tps"].append(row["observed_decode_tokens_per_second"])
            aggregate = aggregate_tps(rows)
            if aggregate is not None:
                bucket["aggregate_decode_tps"].append(aggregate)
    return {key: dict(ttft_ms=summarize(v["ttft_ms"]), inter_chunk_ms=summarize(v["inter_chunk_ms"]),
                      per_stream_decode_tps=summarize(v["per_stream_decode_tps"]),
                      aggregate_decode_tps=summarize(v["aggregate_decode_tps"])) for key, v in metrics.items()}


def expected_keys(matrix: dict):
    return {(c, r) for c in matrix.get("concurrency", []) for r in range(1, int(matrix.get("repeats", 0)) + 1)}


def coverage_report(control: dict, candidate: dict) -> dict:
    problems = []
    matrix = control.get("matrix") or {}
    if matrix != candidate.get("matrix"):
        problems.append("matrix differs between arms")
    expected = expected_keys(matrix)
    # A dict comprehension would silently drop a duplicate case id, so reject
    # duplicates explicitly in both arms before building the maps.
    for name, arm in (("control", control), ("candidate", candidate)):
        ids = [case.get("id") for case in arm.get("cases", [])]
        if len(ids) != len(set(ids)):
            problems.append(f"{name} arm has duplicate case ids")
    ctl = {c["id"]: c for c in control.get("cases", [])}
    cand = {c["id"]: c for c in candidate.get("cases", [])}
    if not cand:
        problems.append("candidate arm has no cases")
    if set(ctl) != set(cand):
        problems.append(f"case sets differ missing={sorted(set(ctl) - set(cand))} extra={sorted(set(cand) - set(ctl))}")
    matched = 0
    for cid in sorted(set(ctl) & set(cand)):
        maps = {}
        for name, case in (("control", ctl[cid]), ("candidate", cand[cid])):
            keys = [(rec["concurrency"], rec["repeat"]) for rec in case.get("records", [])]
            if len(keys) != len(set(keys)):
                problems.append(f"{name} {cid}: duplicate (concurrency,repeat)")
            if set(keys) != expected:
                problems.append(f"{name} {cid}: matrix coverage {sorted(keys)} != {sorted(expected)}")
            for rec in case.get("records", []):
                indices = [row["index"] for row in rec.get("rows", [])]
                if sorted(indices) != list(range(rec["concurrency"])):
                    problems.append(f"{name} {cid} C{rec['concurrency']} repeat {rec['repeat']}: bad index set {sorted(indices)}")
            maps[name] = {(rec["concurrency"], rec["repeat"]): rec for rec in case.get("records", [])}
        for key in sorted(set(maps["control"]) | set(maps["candidate"])):
            left, right = maps["control"].get(key), maps["candidate"].get(key)
            if left is None or right is None:
                problems.append(f"{cid} {key}: record present in only one arm")
                continue
            left_ids = {row["index"] for row in left.get("rows", [])}
            right_ids = {row["index"] for row in right.get("rows", [])}
            if left_ids != right_ids:
                problems.append(f"{cid} {key}: index sets differ between arms")
            matched += len(left_ids & right_ids)
    expected_requests = sum(matrix.get("concurrency", [])) * int(matrix.get("repeats", 0)) * len(cand)
    if matched != expected_requests:
        problems.append(f"matched {matched} of {expected_requests} expected requests")
    return dict(problems=problems, matched_requests=matched, expected_requests=expected_requests, case_ids=sorted(cand), matrix=matrix)


def family_report(left: dict, right: dict) -> dict:
    """Baseline families must appear in the candidate with equal library hashes;
    the candidate may add the family its role selects."""
    problems = []
    left_families, right_families = left.get("expert_tp_manifests"), right.get("expert_tp_manifests")
    if not isinstance(left_families, dict) or not isinstance(right_families, dict):
        return dict(problems=["expert_tp_manifests missing in one or both arms"], shared=[], added_in_candidate=[])
    missing = sorted(set(left_families) - set(right_families))
    if missing:
        problems.append(f"candidate lacks baseline expert families {missing}")
    for family in sorted(set(left_families) & set(right_families)):
        left_hash = (left_families.get(family) or {}).get("native_library_sha256")
        right_hash = (right_families.get(family) or {}).get("native_library_sha256")
        if not left_hash or left_hash != right_hash:
            problems.append(f"expert family {family} native_library_sha256 differs or is missing")
    return dict(problems=problems, shared=sorted(set(left_families) & set(right_families)), added_in_candidate=sorted(set(right_families) - set(left_families)))


def parse_argv(argv):
    """Minimal known-flag parser: a flag's value is the next non-flag token."""
    values, duplicates, index = {}, [], 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if token in values:
                duplicates.append(token)
            following = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else True
            values[token] = following
            index += 2 if following is not True else 1
        else:
            index += 1
    return values, duplicates


def coordinator_applied(name, metadata, argv, problems):
    values, duplicates = parse_argv(argv)
    problems.extend(f"{name}: declared control {flag} is absent from the coordinator argv"
                    for flag in REQUIRED_COORDINATOR_FLAGS if flag not in values)
    for flag in ("--rtx-gpus", "--prefill-batch-tokens", "--prefix-cache-entries", "--concurrency",
                 "--max-context-tokens", "--max-output-tokens") + TOPOLOGY_FLAGS:
        key = flag.lstrip("-").replace("-", "_")
        if flag in values and str(values[flag]) != str(metadata.get(key)):
            problems.append(f"{name}: coordinator {flag}={values[flag]} but metadata {key}={metadata.get(key)}")
    if "--kv-pool-size" in values and str(values["--kv-pool-size"]) != str(metadata.get("kv_pool")):
        problems.append(f"{name}: coordinator --kv-pool-size={values['--kv-pool-size']} but metadata kv_pool={metadata.get('kv_pool')}")
    resolved, requested = metadata.get("resolved_rtx_expert_layers"), metadata.get("requested_rtx_expert_layers")
    if not (requested == "auto" or isinstance(requested, int) and not isinstance(requested, bool)):
        problems.append(f"{name}: requested_rtx_expert_layers must be 'auto' or an integer, got {requested!r}")
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        problems.append(f"{name}: resolved_rtx_expert_layers must be the actual integer plan boundary, got {resolved!r}")
    elif str(metadata.get("spark_first_layer")) != str(min(resolved, 39)):
        problems.append(f"{name}: spark_first_layer={metadata.get('spark_first_layer')} != min(resolved_rtx_expert_layers, 39)={min(resolved, 39)}")
    if "--rtx-expert-layers" in values:
        if str(values["--rtx-expert-layers"]) != str(resolved):
            problems.append(f"{name}: coordinator --rtx-expert-layers={values['--rtx-expert-layers']} but resolved actual={resolved}")
    elif requested != "auto":
        problems.append(f"{name}: no --rtx-expert-layers argument but requested_rtx_expert_layers={requested!r}")
    dspark_on = str(metadata.get("dspark")).lower() in ("on", "true", "1", "yes")
    if ("--dspark" in values) != dspark_on:
        problems.append(f"{name}: --dspark presence {('--dspark' in values)} does not match metadata dspark={metadata.get('dspark')}")
    draft = values.get("--dspark-draft-limit", "7" if str(metadata.get("rtx_gpus")) == "2" else "5")
    if str(draft) != str(metadata.get("dspark_draft_limit")):
        problems.append(f"{name}: effective dspark_draft_limit={draft} but metadata={metadata.get('dspark_draft_limit')}")
    return [flag for flag in duplicates if flag in KNOWN_COORDINATOR_FLAGS]


def worker_applied(name, metadata, per_rank, problems):
    world = int(metadata.get("spark_tp", 0)) * int(metadata.get("spark_ep", 0))
    if world and str(metadata.get("spark_count")) != str(world):
        problems.append(f"{name}: spark_count={metadata.get('spark_count')} != spark_tp*spark_ep={world}")
    capacity = metadata.get("worker_capacity")
    minimum = next((c for c in (80, 256, 1024, 4096) if int(metadata.get("prefill_batch_tokens", 0)) <= c), 4096)
    if capacity not in SUPPORTED_WORKER_CAPACITIES:
        problems.append(f"{name}: worker_capacity={capacity!r} is not one of {SUPPORTED_WORKER_CAPACITIES}")
    elif capacity < minimum:
        problems.append(f"{name}: worker_capacity={capacity} is below the {minimum} required for prefill_batch_tokens={metadata.get('prefill_batch_tokens')}")
    if len(per_rank) != world:
        problems.append(f"{name}: worker_argv has {len(per_rank)} rank argvs but world={world}")
    ranks = []
    for index, argv in enumerate(per_rank):
        values, duplicates = parse_argv(argv)
        problems.extend(f"{name}: declared worker control {flag} is missing from the rank-{index} argv"
                        for flag in REQUIRED_WORKER_FLAGS if flag not in values)
        rank = values.get("--rank")
        if rank is None or not str(rank).isdigit() or not 0 <= int(rank) < world:
            problems.append(f"{name}: worker rank-{index} --rank={rank} is not a valid rank in 0..{world - 1}")
        else:
            ranks.append(int(rank))
        for flag, expected in (("--world", world), ("--capacity", capacity),
                               ("--device-budget-bytes", metadata.get("spark_device_budget_bytes")),
                               ("--first-layer", metadata.get("spark_first_layer")),
                               ("--spark-tp", metadata.get("spark_tp")), ("--spark-ep", metadata.get("spark_ep"))):
            if flag in values and str(values[flag]) != str(expected):
                problems.append(f"{name}: worker rank-{index} {flag}={values[flag]} but metadata expects {expected}")
        problems.extend(f"{name}: worker rank-{index} duplicate flags {flag}"
                        for flag in duplicates if flag in KNOWN_WORKER_FLAGS)
    if len(ranks) != len(set(ranks)):
        problems.append(f"{name}: worker_argv repeats a rank {sorted(ranks)}")
    if world and set(ranks) != set(range(world)):
        problems.append(f"{name}: worker ranks {sorted(ranks)} do not cover 0..{world - 1}")


def applied_control_report(control: dict, candidate: dict) -> list:
    problems = []
    for name, arm in (("control", control), ("candidate", candidate)):
        metadata = arm.get("startup_metadata") or {}
        coordinator, workers = metadata.get("coordinator_argv"), metadata.get("worker_argv")
        if not isinstance(coordinator, list) or not coordinator:
            problems.append(f"{name} startup_metadata lacks coordinator_argv")
            continue
        if not isinstance(workers, list) or not workers or not all(isinstance(rank, list) for rank in workers):
            problems.append(f"{name} startup_metadata worker_argv must be a per-rank list of argv lists")
            continue
        problems.extend(f"{name}: duplicate coordinator flags {flag}"
                        for flag in coordinator_applied(name, metadata, coordinator, problems))
        worker_applied(name, metadata, workers, problems)
    return problems


def identity_report(control: dict, candidate: dict) -> dict:
    left, right = control.get("startup_metadata"), candidate.get("startup_metadata")
    problems = []
    if not isinstance(left, dict) or not isinstance(right, dict):
        return dict(problems=["startup_metadata missing in one or both arms"], missing_fields=[], mismatched_fields=[],
                    allowed_differences={}, required=list(REQUIRED_IDENTITY), family=dict(problems=[], shared=[], added_in_candidate=[]))
    missing = [f for f in REQUIRED_IDENTITY if f not in left or f not in right]
    missing += [f for f in REQUIRED_IDENTITY_STRUCTURES + ("requested_rtx_expert_layers",) if f not in left or f not in right]
    if missing:
        problems.append(f"missing identity fields {missing}")
    mismatched = [f for f in REQUIRED_IDENTITY if f in left and f in right and left[f] != right[f]]
    mismatched += [f for f in REQUIRED_IDENTITY_STRUCTURES
                   if f in left and f in right and f != "expert_tp_manifests" and left[f] != right[f]]
    if mismatched:
        problems.append(f"identity mismatch {mismatched}")
    family = family_report(left, right)
    problems.extend(family["problems"])
    probes = {control.get("probe", {}).get("system_fingerprint"), candidate.get("probe", {}).get("system_fingerprint")}
    if len(probes) != 1 or None in probes:
        problems.append(f"engine system_fingerprint differs or is missing: {sorted(map(str, probes))}")
    required = list(REQUIRED_IDENTITY) + list(REQUIRED_IDENTITY_STRUCTURES) + ["requested_rtx_expert_layers"]
    return dict(problems=problems, missing_fields=missing, mismatched_fields=mismatched, family=family, required=required,
                allowed_differences={f: {"control": left.get(f), "candidate": right.get(f)} for f in ALLOWED_DIFFERENCES})


def repeat_inconsistencies(arm: dict) -> list:
    """Recompute within-arm greedy repeat consistency from the row data, because
    compare must not trust an arm's own flag."""
    problems = []
    for case in arm.get("cases", []):
        by_concurrency = {}
        for rec in case.get("records", []):
            by_concurrency.setdefault(rec["concurrency"], []).append(rec["rows"])
        for concurrency, record_rows in by_concurrency.items():
            for index in range(concurrency):
                texts = set()
                for rows in record_rows:
                    row = next((r for r in rows if r.get("index") == index), None)
                    if row and row.get("passed"):
                        texts.add(row.get("text"))
                if len(texts) > 1:
                    problems.append(f"{case['id']} C{concurrency} request {index} greedy text changed across repeats")
        if case.get("greedy_repeat_consistent") is False:
            problems.append(f"{case['id']} greedy_repeat_consistent is false")
    return problems


def compare_arms(control: dict, candidate: dict, review_evidence: dict | None = None,
                 kernel_evidence: dict | None = None) -> dict:
    coverage = coverage_report(control, candidate)
    identity = identity_report(control, candidate)
    tier_a = list(coverage["problems"]) + list(identity["problems"]) + applied_control_report(control, candidate)
    for name, arm in (("control", control), ("candidate", candidate)):
        if not arm.get("passed"):
            tier_a.append(f"{name} arm did not pass: {arm.get('failures')}")
        if not arm.get("probe", {}).get("passed"):
            tier_a.append(f"{name} readiness probe failed")
        tier_a.extend(f"{name} arm: {problem}" for problem in repeat_inconsistencies(arm))
    ctl = {c["id"]: c for c in control.get("cases", [])}
    cand = {c["id"]: c for c in candidate.get("cases", [])}
    exact, divergent = 0, []
    for cid in sorted(set(ctl) & set(cand)):
        maps = {}
        for name, case in (("control", ctl[cid]), ("candidate", cand[cid])):
            maps[name] = {(rec["concurrency"], rec["repeat"]): rec for rec in case.get("records", [])}
        for key in sorted(set(maps["control"]) & set(maps["candidate"])):
            left = {row["index"]: row for row in maps["control"][key].get("rows", [])}
            right = {row["index"]: row for row in maps["candidate"][key].get("rows", [])}
            for index in sorted(set(left) & set(right)):
                a, b = left[index], right[index]
                where = f"{cid} C{key[0]} repeat {key[1]} request {index}"
                if not a.get("passed") or not b.get("passed"):
                    tier_a.append(f"{where} did not complete")
                    continue
                if not (a.get("content_check") or {}).get("objective_checks_passed"):
                    tier_a.append(f"{where} control content check failed")
                if not (b.get("content_check") or {}).get("objective_checks_passed"):
                    tier_a.append(f"{where} candidate content check failed")
                if a.get("text") == b.get("text"):
                    exact += 1
                else:
                    divergent.append(dict(case=cid, concurrency=key[0], repeat=key[1], request=index,
                                          control_chars=len(a.get("text") or ""), candidate_chars=len(b.get("text") or ""),
                                          control_check=(a.get("content_check") or {}).get("objective_checks_passed"),
                                          candidate_check=(b.get("content_check") or {}).get("objective_checks_passed")))
    entries = {}
    for entry in (review_evidence or {}).get("reviews", []):
        entries[(entry.get("case"), entry.get("concurrency"), entry.get("repeat"), entry.get("request"))] = entry
    uncovered = []
    for d in divergent:
        entry = entries.get((d["case"], d["concurrency"], d["repeat"], d["request"]))
        if not entry or entry.get("kind") not in REVIEW_KINDS or entry.get("passed") is not True or not str(entry.get("evidence", "")).strip():
            uncovered.append(dict(case=d["case"], concurrency=d["concurrency"], repeat=d["repeat"], request=d["request"]))
    if tier_a:
        gate, passed = "FAIL", False
    elif not divergent:
        gate, passed = "text_exact", True
    elif uncovered:
        gate, passed = "text_divergent_needs_review", False
    else:
        gate, passed = "text_divergent_review_qualified", True
    return dict(scope="Matched four-Spark TP4EP1 control vs TP2EP2 candidate; TTFT, inter-chunk latency and "
                       "decode tokens/s are separate, text equality is not runtime bit-exactness, and divergence "
                       "needs per-output review evidence.",
                gate_result=gate, passed=passed, tier_a_failures=tier_a, coverage=coverage, identity=identity,
                text=dict(exact_matches=exact, divergent_requests=len(divergent), divergences=divergent),
                review=dict(provided=review_evidence is not None, entries=len(entries), uncovered=uncovered,
                            kinds=list(REVIEW_KINDS)),
                kernel_evidence_context=(dict(sha256=sha256(Path(kernel_evidence["__path__"])),
                                              note="component-level context only; cannot qualify end-to-end equivalence")
                                         if kernel_evidence and kernel_evidence.get("__path__") else None),
                metrics=dict(control=arm_metrics(control), candidate=arm_metrics(candidate)))


def cmd_compare(args) -> int:
    control, candidate = json.loads(args.control.read_text()), json.loads(args.candidate.read_text())
    review = json.loads(args.review_evidence.read_text()) if args.review_evidence else None
    kernel = {"__path__": str(args.kernel_evidence)} if args.kernel_evidence else None
    report = compare_arms(control, candidate, review, kernel)
    report["control"], report["candidate"] = str(args.control), str(args.candidate)
    if args.review_evidence:
        report["review"]["sha256"] = sha256(args.review_evidence)
    if args.kernel_evidence:
        report["kernel_evidence_context"]["path"] = str(args.kernel_evidence)
    write_json(args.output, report)
    summary = {key: report[key] for key in ("gate_result", "passed", "tier_a_failures")}
    summary |= dict(exact_matches=report["text"]["exact_matches"], divergent_requests=report["text"]["divergent_requests"])
    print(json.dumps(summary, indent=2))
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="CPU-only corpus schema check")
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("run", help="drive one arm against a live service")
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    p.add_argument("--arm", required=True, choices=("tp4ep1", "tp2ep2"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--startup-metadata", type=Path, help="structured matched-control JSON captured from run.sh --dry-run")
    p.add_argument("--case", action="append")
    p.add_argument("--concurrency", type=int, nargs="+")
    p.add_argument("--repeats", type=int)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("compare", help="CPU-only matched-arm report and gates")
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--review-evidence", type=Path, help="per-output matched-logits/task-quality/manual review")
    p.add_argument("--kernel-evidence", type=Path, help="component-level context only; cannot qualify E2E")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
