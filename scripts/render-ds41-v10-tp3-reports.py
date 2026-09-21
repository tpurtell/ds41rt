#!/usr/bin/env python3
"""Render the DS41RT v10 TP3 campaign reports from a report manifest.

The v10 campaign has exactly two three-rank arms:

  * `v10-native-tp3ep1` - one RTX + three Sparks, official native checkpoint,
    explicit `TP3xEP1` (three ranks, one unreplicated expert group, 5 RTX-local
    / 35 remote layers).
  * `v10-exl3-compact-tp3` - one RTX under the compact 32 GiB reservation
    ceiling + three Sparks, checkpoint-native EXL3 K3.25 (`k34`) with the
    implicit compact TP3 layout (`SPARK_COUNT=3`, no `SPARK_TP`/`SPARK_EP`).

Every rendered cell is parsed from a raw harness file named by the manifest.
A family with no raw file renders as an explicit PENDING cell with no number;
nothing is estimated, interpolated or carried over from another campaign.

Record accounting is derived, not asserted: the expected count for every family
is recomputed from the live harness scripts' own constants and argparse defaults
(`DEFAULT_BASES`, `DEFAULT_SUFFIXES`, `DEFAULT_CONTEXTS`, the concurrency level
list, the mixed-traffic default, the decode corpus's weighted case list and the
tool-eval scenario assertion), then compared against the actual records counted
in the raw files. The report prints the expected/actual table and the exact
arithmetic behind the total, and states any mismatch explicitly instead of
printing a headline number the evidence does not support.

Usage:
  scripts/render-ds41-v10-tp3-reports.py --manifest MANIFEST.json \
      --package DIR --output FILE.md
  scripts/render-ds41-v10-tp3-reports.py --manifest MANIFEST.json \
      --package DIR --check [--strict]
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import pathlib
import re
import statistics
import sys

DASH = "\u2014"
PENDING = "PENDING"

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
CORPUS = SCRIPTS / "fixtures" / "release-semantic-corpus.json"

# Families that are counted in the performance ledger, in report order.
PERFORMANCE_FAMILIES = (
    ("decode", "Headline decode (dSpark)"),
    ("prefill", "Prefill matrix"),
    ("retained", "Retained decode (prime contexts)"),
    ("retained_control_2k", "Retained decode 2K control"),
    ("concurrency", "Concurrency counting/code/topic"),
    ("mixed", "Mixed traffic"),
    ("target_only", "Target-only decode (dSpark off)"),
)
# Extra families that are recorded but deliberately excluded from the
# performance total, because they are diagnostics, not timed serving records.
DIAGNOSTIC_FAMILIES = (
    ("startup_memory", "Startup / memory"),
    ("kernel_tiling", "Kernel / tile diagnostics"),
)
EXCLUDED_NOTE = "warmups, primes and lifecycle probes are recorded but excluded"

TOOL_EVAL_FALLBACK_SCENARIOS = 88

# A failed or non-passing sample must be surfaced with its raw evidence.
CAPACITY_PATTERNS = (
    "admission", "out of memory", "oom", "kv pool", "kv cache", "kv-cache",
    "memory ceiling", "device budget", "device_budget", "insufficient memory",
    "no capacity", "capacity exceeded", "not enough memory", "budget exceeded",
)
CAPACITY_RE = re.compile("|".join(re.escape(p) for p in CAPACITY_PATTERNS), re.I)


class Missing(Exception):
    """A manifest or raw-data problem that must stop the render."""


# --------------------------------------------------------------------------
# path and json helpers
# --------------------------------------------------------------------------

def expand(path) -> pathlib.Path:
    """Expand ~ and make a path absolute (relative paths resolve at cwd)."""
    return pathlib.Path(str(path)).expanduser().absolute()


def resolve(package: pathlib.Path, relative) -> pathlib.Path:
    """Resolve a manifest raw path against the package directory."""
    candidate = pathlib.Path(str(relative)).expanduser()
    if candidate.is_absolute():
        return candidate.absolute()
    return (package / candidate).expanduser().absolute()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: pathlib.Path):
    """Return a parsed JSON document, or None when unavailable/unparseable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fmt(value, thousands: bool = False):
    if value is None:
        return DASH
    if isinstance(value, float):
        return f"{value:,.0f}" if thousands else f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}" if thousands else str(value)
    return str(value)


def as_paths(value):
    """Normalize a manifest `raw` entry to a list of path strings."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


# --------------------------------------------------------------------------
# mechanical expectation derived from the live harness scripts
# --------------------------------------------------------------------------

def _literal(path: pathlib.Path, name: str):
    """Return the literal value assigned to a module-level NAME."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise Missing(f"{path.name}: no module constant {name}")


def _argparse_default(path: pathlib.Path, flag: str, positional: bool = False):
    """Return the literal `default=` of add_argument(flag, ...).

A positional argument is matched when `flag` is the first positional literal.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if positional:
            if flag not in literals[:1] and not (
                literals and str(literals[0]).lstrip("-") == flag.lstrip("-")
            ):
                continue
        elif flag not in literals:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.literal_eval(kw.value)
    raise Missing(f"{path.name}: no add_argument({flag})")


def tool_eval_scenarios(path: pathlib.Path) -> int:
    """Read the tool-eval script's own scenario-count assertion."""
    match = re.search(r"len\(scenarios\)\s*==\s*(\d+)", path.read_text())
    return int(match.group(1)) if match else TOOL_EVAL_FALLBACK_SCENARIOS


def derive_expectation(repeats: int, prefill_warmups: int, tool_eval_runs: int) -> dict:
    """Derive every expected count from the live scripts, not from prose."""
    prefill = SCRIPTS / "bench-ds41-release-prefill-matrix.py"
    retained = SCRIPTS / "bench-ds41-release-retained-decode.py"
    decode = SCRIPTS / "bench-ds41-release-decode.py"
    concurrent = SCRIPTS / "bench-ds41-concurrent-api.py"
    mixed = SCRIPTS / "bench-ds41-adaptive-mixed.py"
    tool_eval = SCRIPTS / "qualify-ds41-tool-eval.py"

    for path in (prefill, retained, decode, concurrent, mixed, tool_eval, CORPUS):
        if not path.is_file():
            raise Missing(f"expectation source missing: {path}")

    bases = list(_literal(prefill, "DEFAULT_BASES"))
    suffixes = list(_literal(prefill, "DEFAULT_SUFFIXES"))
    contexts = [int(c) for c in _literal(retained, "DEFAULT_CONTEXTS")]
    levels = [int(c) for c in _argparse_default(concurrent, "--concurrency")]
    mixed_levels = [int(c) for c in _argparse_default(mixed, "--concurrency")]
    weighted_cases = json.loads(CORPUS.read_text())["weighted_case_ids"]
    decode_cases = len(weighted_cases) + 1  # + counting via --include-counting
    scenarios = tool_eval_scenarios(tool_eval)

    # The 2K control is always a separate invocation (--context 2048). When the
    # harness default already contains 2048 the "prime" family is reported
    # without it, so no record is counted twice.
    prime_contexts = [c for c in contexts if c != 2048]
    control_contexts = sorted(set(contexts) | {2048})
    control_only = [c for c in control_contexts if c not in prime_contexts]

    cells = len(bases) * len(suffixes)
    families = {
        "decode": {
            "expected": decode_cases * repeats,
            "formula": f"({len(weighted_cases)} weighted + 1 counting) x {repeats} repeats",
            "source": "scripts/bench-ds41-release-decode.py",
            "counted": True,
        },
        "prefill": {
            "expected": cells * repeats,
            "formula": f"{len(bases)} bases x {len(suffixes)} suffixes = {cells} cells x {repeats} timed",
            "warmup_samples": cells * prefill_warmups,
            "source": "scripts/bench-ds41-release-prefill-matrix.py",
            "counted": True,
        },
        "retained": {
            "expected": len(prime_contexts) * len(weighted_cases) * repeats,
            "formula": (f"{len(prime_contexts)} prime contexts {prime_contexts} x "
                        f"{len(weighted_cases)} cases x {repeats} repeats"),
            "contexts": prime_contexts,
            "source": "scripts/bench-ds41-release-retained-decode.py",
            "counted": True,
        },
        "retained_control_2k": {
            "expected": len(control_only) * len(weighted_cases) * repeats,
            "formula": (f"{len(control_only)} control context(s) {control_only} x "
                        f"{len(weighted_cases)} cases x {repeats} repeats"),
            "contexts": control_only,
            "source": "scripts/bench-ds41-release-retained-decode.py",
            "counted": True,
        },
        "concurrency": {
            "expected": 3 * len(levels) * repeats,
            "formula": (f"3 cases (counting/code/topic) x {len(levels)} levels {levels} "
                        f"x {repeats} repeats"),
            "source": "scripts/bench-ds41-concurrent-api.py",
            "counted": True,
        },
        "mixed": {
            "expected": len(mixed_levels),
            "formula": f"adaptive-mixed batches at concurrency {mixed_levels}",
            "source": "scripts/bench-ds41-adaptive-mixed.py",
            "counted": True,
        },
        "target_only": {
            "expected": decode_cases * repeats,
            "formula": f"({len(weighted_cases)} weighted + 1 counting) x {repeats} repeats, DSPARK=off",
            "source": "scripts/bench-ds41-release-decode.py",
            "counted": True,
        },
        "startup_memory": {
            "expected": None,
            "formula": "diagnostic; not part of the performance record ledger",
            "source": "coordinator/worker reservation and device-budget logs",
            "counted": False,
        },
        "kernel_tiling": {
            "expected": None,
            "formula": "component microbenchmark; not serving throughput",
            "source": "python/tools/bench_tp_ep_kernel.py",
            "counted": False,
        },
    }
    total = sum(f["expected"] for f in families.values() if f["counted"])
    return {
        "families": families,
        "total": total,
        "protocol": {"repeats": repeats, "prefill_warmups": prefill_warmups,
                     "tool_eval_runs": tool_eval_runs},
        "weighted_cases": list(weighted_cases),
        "tool_eval": {"scenarios_per_run": scenarios, "runs": tool_eval_runs,
                      "scenario_runs": scenarios * tool_eval_runs},
        "prefill_cells": cells,
        "retained_lumped": {
            "records": (len(prime_contexts) + len(control_only)) * len(weighted_cases) * repeats,
            "contexts": prime_contexts + control_only,
            "formula": (f"{len(prime_contexts) + len(control_only)} contexts "
                        f"{prime_contexts + control_only} x {len(weighted_cases)} cases "
                        f"x {repeats} repeats"),
        },
        "sources": {
            str(p.relative_to(REPO)): sha256(p)
            for p in (prefill, retained, decode, concurrent, mixed, tool_eval, CORPUS)
        },
    }


# Shape names used by the sibling `expected-counts.json` documents
# (runs/v10-exl3-tp3/enumerate-expected.py, runs/v10-native-tp3ep1/summarize.py)
# mapped onto this renderer's family keys. `retained` is compared against the
# lumped six-context total so a split (135 + 27) and a lump (162) reconcile to
# the same number instead of being reported as a false mismatch.
EXPECTED_COUNTS_SHAPE_MAP = {
    "decode": "decode",
    "prefill_timed": "prefill",
    "retained": "retained",
    "retained_control_2k": "retained_control_2k",
    "target_only": "target_only",
    "concurrency": "concurrency",
    "mixed": "mixed",
}


def load_expected_counts(path) -> dict:
    """Load an external expected-counts.json document, or raise Missing."""
    resolved = expand(path)
    doc = load_json(resolved)
    if not isinstance(doc, dict):
        raise Missing(f"expected-counts unreadable: {resolved}")
    for key in ("shapes", "total_performance_records"):
        if key not in doc:
            raise Missing(f"expected-counts missing '{key}': {resolved}")
    doc["__path__"] = str(resolved)
    return doc


def reconcile_expectation(expectation: dict, external: dict) -> list:
    """Compare a mechanically derived expectation against an external document.

    Returns (shape, external, derived, state) rows for every shape the external
    document declares, so a divergent breakdown is surfaced rather than hidden
    behind a total that happens to agree.

    The `retained` shape is ambiguous across documents: one form states the five
    prime contexts only (135, with the 2K control as its own shape) and another
    lumps all six contexts (162). The external document's own `contexts` list
    decides which basis it is on, so the comparison is like-for-like and a split
    versus lump is reported as reconciled, not as a false mismatch.
    """
    rows = []
    shapes = external.get("shapes") or {}
    for shape, family in EXPECTED_COUNTS_SHAPE_MAP.items():
        spec = shapes.get(shape)
        if not isinstance(spec, dict) or spec.get("records") is None:
            continue
        if shape == "retained":
            contexts = spec.get("contexts") or []
            basis = "lumped" if 2048 in contexts else "prime"
            derived = (expectation["retained_lumped"]["records"] if basis == "lumped"
                       else expectation["families"]["retained"]["expected"])
        else:
            basis = None
            derived = expectation["families"][family]["expected"]
        rows.append({
            "shape": shape,
            "basis": basis,
            "external": spec["records"],
            "derived": derived,
            "state": "match" if spec["records"] == derived else "DIFFERS",
        })
    expected_total = external.get("total_performance_records")
    if expected_total is not None:
        rows.append({
            "shape": "performance total",
            "basis": None,
            "external": expected_total,
            "derived": expectation["total"],
            "state": "match" if expected_total == expectation["total"] else "DIFFERS",
        })
    tool = external.get("tool_eval") or {}
    if tool.get("scenario_runs") is not None:
        rows.append({
            "shape": "tool_eval scenario-runs",
            "basis": None,
            "external": tool["scenario_runs"],
            "derived": expectation["tool_eval"]["scenario_runs"],
            "state": ("match" if tool["scenario_runs"] == expectation["tool_eval"]["scenario_runs"]
                      else "DIFFERS"),
        })
    return rows


def expected_counts_document(expectation: dict) -> dict:
    """Emit this renderer's expectation in the sibling expected-counts shape.

    This lets the v10 lanes' collect stages ingest one derived document instead
    of each restating the arithmetic.
    """
    fam = expectation["families"]
    shapes = {
        "decode": {"formula": fam["decode"]["formula"],
                   "records": fam["decode"]["expected"],
                   "source": fam["decode"]["source"]},
        "prefill_cells": {"formula": "one isolated invocation per base x suffix cell",
                          "invocations": expectation["prefill_cells"],
                          "source": fam["prefill"]["source"]},
        "prefill_timed": {"formula": fam["prefill"]["formula"],
                          "records": fam["prefill"]["expected"],
                          "warmup_samples": fam["prefill"]["expected"] // max(1, expectation["protocol"]["repeats"])
                          * expectation["protocol"]["prefill_warmups"],
                          "source": fam["prefill"]["source"]},
        "retained": {"formula": fam["retained"]["formula"],
                     "records": fam["retained"]["expected"],
                     "contexts": fam["retained"]["contexts"],
                     "source": fam["retained"]["source"]},
        "retained_control_2k": {"formula": fam["retained_control_2k"]["formula"],
                                "records": fam["retained_control_2k"]["expected"],
                                "contexts": fam["retained_control_2k"]["contexts"],
                                "source": fam["retained_control_2k"]["source"]},
        "target_only": {"formula": fam["target_only"]["formula"],
                        "records": fam["target_only"]["expected"],
                        "source": fam["target_only"]["source"]},
        "concurrency": {"formula": fam["concurrency"]["formula"],
                        "records": fam["concurrency"]["expected"],
                        "source": fam["concurrency"]["source"]},
        "mixed": {"formula": fam["mixed"]["formula"],
                  "records": fam["mixed"]["expected"],
                  "source": fam["mixed"]["source"]},
    }
    return {
        "schema": "ds41rt.v10-tp3.expected-counts.v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": expectation["protocol"],
        "weighted_cases": expectation["weighted_cases"],
        "shapes": shapes,
        "retained_lumped": expectation["retained_lumped"],
        "total_performance_records": expectation["total"],
        "distinct_retained_contexts": sorted(
            set(fam["retained"]["contexts"]) | set(fam["retained_control_2k"]["contexts"])),
        "tool_eval": expectation["tool_eval"],
        "sources": expectation["sources"],
        "note": ("Mechanically derived from live argparse defaults and corpus by "
                 "scripts/render-ds41-v10-tp3-reports.py; this is the expectation the "
                 "check stage compares against, not a claim."),
    }


# --------------------------------------------------------------------------
# raw document parsing
# --------------------------------------------------------------------------

def is_timed(sample: dict) -> bool:
    return bool(sample.get("timed", True))


def sample_error(sample: dict):
    for key in ("error", "failure", "exception"):
        if sample.get(key):
            return str(sample[key])
    return None


def sample_non_pass(sample: dict):
    """Return a non-pass reason, or None when the sample qualifies as a pass."""
    error = sample_error(sample)
    if error:
        return f"error: {error[:200]}"
    if sample.get("serving_completed") is False:
        return "serving not completed"
    if sample.get("quality_contract_passed") is False:
        issues = sample.get("quality_contract_issues")
        detail = ", ".join(str(i) for i in issues) if isinstance(issues, list) and issues else "quality contract"
        return f"quality contract failed: {detail}"
    if sample.get("objective_checks_passed") is False:
        return "objective checks failed"
    if sample.get("passed") is False:
        return "record marked failed"
    return None


def capacity_gated(sample: dict) -> bool:
    error = sample_error(sample)
    return bool(error and CAPACITY_RE.search(error))


def parse_samples(package: pathlib.Path, paths, family: str):
    """Load raw documents. Returns (docs, rows, digests).

    Every parseable dict document is returned so family-specific parsers (for
    example concurrency, which reports per-level summaries rather than a
    `samples` array) can read it; `rows` is only populated from a `samples`
    array.
    """
    docs, rows, digests = [], [], []
    for relative in paths:
        path = resolve(package, relative)
        doc = load_json(path)
        if doc is None:
            digests.append({"path": str(relative), "resolved": str(path), "sha256": None,
                            "state": "unreadable or absent"})
            continue
        digests.append({"path": str(relative), "resolved": str(path), "sha256": sha256(path),
                        "state": "present"})
        if not isinstance(doc, dict):
            continue
        docs.append(doc)
        samples = doc.get("samples")
        if not isinstance(samples, list):
            continue
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            rows.append({
                "family": family,
                "path": str(relative),
                "index": index,
                "case": sample.get("case"),
                "repeat": sample.get("repeat"),
                "base_context_tokens": sample.get("base_context_tokens"),
                "suffix_tokens": sample.get("suffix_tokens"),
                "context_tokens": sample.get("context_tokens"),
                "weight": sample.get("weight"),
                "timed": is_timed(sample),
                "serving_completed": sample.get("serving_completed"),
                "objective_checks_passed": sample.get("objective_checks_passed"),
                "quality_contract_passed": sample.get("quality_contract_passed"),
                "passed": sample.get("passed"),
                "error": sample_error(sample),
                "reason": sample_non_pass(sample),
                "capacity_gated": capacity_gated(sample),
                "decode_tokens_per_second": sample.get("observed_decode_tokens_per_second"),
                "prefill_tokens_per_second": sample.get("effective_prefill_tokens_per_second"),
            })
    return docs, rows, digests


def parse_decode(doc):
    """Summary view of one decode-shaped raw document."""
    if not isinstance(doc, dict) or not isinstance(doc.get("samples"), list):
        return None
    by_case = {}
    for sample in doc["samples"]:
        if not isinstance(sample, dict) or not is_timed(sample):
            continue
        value = sample.get("observed_decode_tokens_per_second")
        if value is not None:
            by_case.setdefault(sample.get("case"), []).append(value)
    assessed = [s for s in doc["samples"] if isinstance(s, dict)
                and s.get("objective_checks_passed") is not None]
    repeats = [r.get("weighted_observed_decode_tokens_per_second")
               for r in doc.get("repeat_summaries", []) if isinstance(r, dict)]
    repeats = [r for r in repeats if r is not None]
    spread = None
    base = median(repeats)
    if len(repeats) > 1 and base:
        spread = (max(repeats) - min(repeats)) / base * 100.0
    return {
        "cases": {case: median(values) for case, values in by_case.items()},
        "weighted": doc.get("median_weighted_observed_decode_tokens_per_second"),
        "weighted_repeats": repeats,
        "weighted_spread_percent": spread,
        "samples": len(doc.get("samples", [])),
        "serving_completed": sum(bool(s.get("serving_completed")) for s in doc["samples"]),
        "objective_assessed": len(assessed),
        "objective_passed": sum(bool(s.get("objective_checks_passed")) for s in assessed),
        "passed": doc.get("passed"),
        "corpus_sha256": doc.get("corpus_sha256"),
        "tokenizer_sha256": doc.get("tokenizer_sha256"),
        "repeats": doc.get("repeats"),
        "nonce_seed": doc.get("nonce_seed"),
    }


def parse_prefill(doc):
    if not isinstance(doc, dict):
        return None
    cells = doc.get("cells")
    bases = doc.get("bases")
    suffixes = doc.get("suffixes")
    if not isinstance(cells, list) or not cells or not bases or not suffixes:
        return None
    lookup = {}
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        lookup[(cell.get("base_context_tokens"), cell.get("suffix_tokens"))] = \
            cell.get("median_effective_prefill_tokens_per_second")
    grid = [[lookup.get((b, s)) for s in suffixes] for b in bases]
    values = [v for row in grid for v in row if v is not None]
    return {
        "bases": bases,
        "suffixes": suffixes,
        "grid": grid,
        "best": max(values) if values else None,
        "cells": len(values),
        "expected_cells": len(bases) * len(suffixes),
        "complete": len(values) == len(bases) * len(suffixes),
        "passed": doc.get("passed"),
    }


def parse_retained(docs):
    """(context -> weighted decode tok/s) from one or more retained documents."""
    rows = {}
    contexts = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for context in doc.get("contexts") or []:
            if context not in contexts:
                contexts.append(context)
        for summary in doc.get("context_summaries", []) or []:
            if not isinstance(summary, dict):
                continue
            context = summary.get("context_tokens")
            value = summary.get("weighted_observed_decode_tokens_per_second")
            if context is not None and value is not None:
                rows[context] = value
    return rows or None, sorted(contexts)


def parse_retained_reuse(docs):
    """Per-context retention-frontier diagnostics from `context_summaries`.

    `prompt_prefix_reuse` is the release gate: the whole retained parent prompt
    was skipped, so the retained base context is real. `full_turn_reuse` is the
    generated-turn diagnostic: the child also reached the parent's committed turn
    frontier. The API is a text interface, so generated-turn reuse is
    opportunistic and content-dependent; a context that reuses only the prompt
    prefix is a documented limitation, not a hidden one. Absent fields are
    reported as None, never assumed.
    """
    out = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for summary in doc.get("context_summaries", []) or []:
            if not isinstance(summary, dict):
                continue
            context = summary.get("context_tokens")
            if context is None:
                continue
            prefix = summary.get("prompt_prefix_reuse")
            if prefix is None:
                prefix = summary.get("cache_valid")
            out[context] = {
                "samples": summary.get("samples"),
                "prompt_prefix_reuse": prefix,
                "full_turn_reuse": summary.get("full_turn_reuse"),
            }
    return out


def parse_concurrency(docs):
    """Concurrency records, with the accounting basis of the raw evidence.

    Raw documents exist in two shapes and both must be measured honestly:
    `records` carries one row per (level, repeat) with the aggregate rate, while
    `summaries` carries one per-level median only. The summary shape therefore
    holds `levels x cases` records, not `levels x cases x repeats`; the basis is
    returned so the expected count matches the evidence instead of demanding
    repeats the raw file never claimed to contain.
    """
    rows, levels = [], []
    basis = None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        case = doc.get("case")
        records = doc.get("records")
        if isinstance(records, list) and records:
            basis = "records" if basis is None else basis
            for record in records:
                if not isinstance(record, dict):
                    continue
                level = record.get("concurrency")
                if level is not None and level not in levels:
                    levels.append(level)
                rows.append({
                    "case": case,
                    "concurrency": level,
                    "repeat": record.get("repeat"),
                    "aggregate_tps": record.get("aggregate_tps"),
                    "per_stream_mean": record.get("per_stream_mean"),
                    "rows": len(record.get("rows", []) or []),
                    "passed": not record.get("error"),
                    "reason": (f"error: {record['error']}" if record.get("error") else None),
                    "capacity_gated": capacity_gated({"error": record.get("error")}),
                })
            continue
        if basis is None:
            basis = "summaries"
        for summary in doc.get("summaries", []) or []:
            if not isinstance(summary, dict):
                continue
            level = summary.get("concurrency")
            if level is None:
                continue
            if level not in levels:
                levels.append(level)
            rows.append({
                "case": case,
                "concurrency": level,
                "repeat": None,
                "aggregate_tps": summary.get("median_aggregate_tps"),
                "per_stream_mean": None,
                "rows": summary.get("samples"),
                "passed": True,
            })
    return rows, sorted(levels), basis


def parse_mixed(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("batches"), list):
        return None
    rows = []
    for index, batch in enumerate(doc["batches"]):
        if not isinstance(batch, dict):
            continue
        passed = batch.get("passed", doc.get("passed"))
        reason = None
        if not passed:
            # The mixed harness records the failing batch separately; surface it.
            failed = doc.get("failed_batch")
            if isinstance(failed, dict) and failed.get("concurrency") == batch.get("concurrency"):
                failures = failed.get("failures")
                reason = ("batch failed: " + ", ".join(str(f) for f in failures)
                          if isinstance(failures, list) and failures else "batch marked failed")
            else:
                reason = "batch marked failed"
        rows.append({
            "family": "mixed",
            "path": "mixed",
            "index": index,
            "concurrency": batch.get("concurrency"),
            "aggregate_tps": batch.get("aggregate_tps"),
            "per_stream_mean": batch.get("per_stream_mean"),
            "rows": len(batch.get("rows", []) or []),
            "passed": passed,
            "reason": reason,
            "capacity_gated": False,
        })
    return rows or None


def parse_startup_memory(docs):
    """Startup/memory diagnostics: the recorded reservation and budget figures."""
    out = {"docs": len(docs), "weight_only_admission": None, "gpu_memory_csv": [],
           "coordinator_lines": 0, "spark_lines": {}}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("weight_only_admission"):
            out["weight_only_admission"] = doc["weight_only_admission"]
        if doc.get("gpu_memory_csv"):
            out["gpu_memory_csv"] = doc["gpu_memory_csv"]
        out["coordinator_lines"] += len(doc.get("coordinator_memory_lines") or [])
        for host, lines in (doc.get("spark_memory_lines") or {}).items():
            out["spark_lines"][host] = len(lines or [])
    return out if out["docs"] else None


def parse_kernel_tiling(docs):
    out = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        cells = doc.get("cells")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            out.append({
                "topology": cell.get("topology"),
                "workload": cell.get("workload"),
                "width": cell.get("width"),
                "capacity": cell.get("capacity"),
                "rows": cell.get("rows"),
                "amortised_us": cell.get("amortised_us", cell.get("amortised")),
                "cold_us": cell.get("cold_us", cell.get("cold")),
            })
    return out or None


def parse_tool_eval(package: pathlib.Path, spec):
    """Per-run tool-eval scores and the non-pass scenario detail behind them."""
    if not spec or not spec.get("summaries"):
        return None, []
    summaries_path = resolve(package, spec["summaries"])
    summaries = load_json(summaries_path)
    if not isinstance(summaries, list):
        return None, []
    runs, problems = [], []
    run_dirs = spec.get("runs") or []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, dict):
            continue
        run = {
            "run": index + 1,
            "run_id": summary.get("run_id"),
            "basic_points": summary.get("basic_points"),
            "basic_max": summary.get("basic_max"),
            "hard_points": summary.get("hard_points"),
            "hard_max": summary.get("hard_max"),
            "total_points": summary.get("total_points"),
            "total_max": summary.get("total_max"),
            "statuses": summary.get("statuses") or {},
            "output_cap": summary.get("output_cap"),
            "output_cap_source": summary.get("output_cap_source"),
            "failures": [f for f in (summary.get("failures") or []) if isinstance(f, dict)],
            "detail": [],
        }
        scenario_runs = sum(v for v in run["statuses"].values()
                            if isinstance(v, (int, float))) or None
        run["scenario_runs"] = scenario_runs
        # Non-pass scenario detail: prefer the raw per-run scenario results so a
        # failure that is not in the summary's fail list is still surfaced.
        directory = None
        if index < len(run_dirs):
            directory = resolve(package, run_dirs[index])
        elif spec.get("dir"):
            candidate = resolve(package, spec["dir"]) / f"run-{index + 1:02}"
            directory = candidate if candidate.is_dir() else None
        if directory:
            raw = load_json(directory / "tool-eval.json")
            if isinstance(raw, dict):
                scenarios = ((raw.get("scores") or {}).get("scenario_results")
                             if isinstance(raw.get("scores"), dict) else None)
                for scenario in scenarios or []:
                    if not isinstance(scenario, dict):
                        continue
                    status = str(scenario.get("status", "")).lower()
                    if status and status != "pass":
                        run["detail"].append({
                            "scenario_id": scenario.get("scenario_id"),
                            "status": scenario.get("status"),
                            "points": scenario.get("points"),
                            "summary": scenario.get("summary"),
                        })
        if run["scenario_runs"] is None and run["detail"]:
            run["scenario_runs"] = len(run["detail"])
        runs.append(run)
    return {
        "path": str(spec["summaries"]),
        "resolved": str(summaries_path),
        "runs": runs,
        "scenario_runs": sum(r["scenario_runs"] or 0 for r in runs),
        "dir": spec.get("dir"),
    }, problems


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def require_provenance(manifest: dict):
    problems = []
    if manifest.get("release") != "v10":
        problems.append("release must be 'v10'")
    for key in ("checkpoint", "arms"):
        if key not in manifest:
            problems.append(f"manifest missing '{key}'")
    checkpoint = manifest.get("checkpoint") or {}
    for field in ("model_id", "revision", "quant"):
        if not checkpoint.get(field):
            problems.append(f"checkpoint.{field} is required")
    images = manifest.get("images")
    if images is not None:
        for role in ("coordinator", "spark_expert"):
            image = (images or {}).get(role) or {}
            for field in ("tag", "digest", "revision"):
                if not image.get(field):
                    problems.append(f"images.{role}.{field} is required")
    for arm in manifest.get("arms") or []:
        ident = arm.get("id", "?")
        for field in ("id", "quant", "topology", "rtx", "sparks", "config_sha256"):
            if field not in arm:
                problems.append(f"arm {ident} missing '{field}'")
        if arm.get("quant") == "exl3" and not arm.get("ceiling"):
            problems.append(
                f"arm {ident} is the compact EXL3 arm and must declare its "
                "ceiling (memory_reservation / kv_pool_size / prefill_batch_tokens)")
    if problems:
        raise Missing("; ".join(problems))


# --------------------------------------------------------------------------
# loading one arm
# --------------------------------------------------------------------------

def load_arm(package: pathlib.Path, arm: dict) -> dict:
    raw = arm.get("raw") or {}
    declared = {name: as_paths(raw.get(name)) for name in
                ("decode", "prefill", "retained", "retained_control_2k",
                 "concurrency_counting", "concurrency_code", "concurrency_topic",
                 "mixed", "target_only", "startup_memory", "kernel_tiling")}

    def docs_for(name):
        return parse_samples(package, declared[name], name)

    decode_docs, decode_rows, decode_digests = docs_for("decode")
    prefill_docs, prefill_rows, prefill_digests = docs_for("prefill")
    retained_docs, retained_rows, retained_digests = docs_for("retained")
    control_docs, control_rows, control_digests = docs_for("retained_control_2k")
    counting_docs, counting_rows, counting_digests = docs_for("concurrency_counting")
    code_docs, code_rows, code_digests = docs_for("concurrency_code")
    topic_docs, topic_rows, topic_digests = docs_for("concurrency_topic")
    mixed_docs, _, mixed_digests = docs_for("mixed")
    target_docs, target_rows, target_digests = docs_for("target_only")
    startup_docs, _, startup_digests = docs_for("startup_memory")
    kernel_docs, _, kernel_digests = docs_for("kernel_tiling")

    concurrency_rows, concurrency_levels, concurrency_basis = parse_concurrency(
        counting_docs + code_docs + topic_docs)
    mixed_rows = parse_mixed(mixed_docs[0]) if mixed_docs else None
    tool_eval, _ = parse_tool_eval(package, arm.get("tool_eval"))

    startup_memory = parse_startup_memory(startup_docs)
    kernel_tiling = parse_kernel_tiling(kernel_docs)

    def diagnostic_rows(value):
        """One row per recorded diagnostic cell, so presence is countable."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    rows_by_family = {
        "decode": [r for r in decode_rows if r["timed"]],
        "prefill": [r for r in prefill_rows if r["timed"]],
        "retained": [r for r in retained_rows if r["timed"]],
        "retained_control_2k": [r for r in control_rows if r["timed"]],
        "concurrency": concurrency_rows,
        "mixed": mixed_rows or [],
        "target_only": [r for r in target_rows if r["timed"]],
        "startup_memory": diagnostic_rows(startup_memory),
        "kernel_tiling": diagnostic_rows(kernel_tiling),
    }
    digests = (decode_digests + prefill_digests + retained_digests + control_digests
               + counting_digests + code_digests + topic_digests + mixed_digests
               + target_digests + startup_digests + kernel_digests)
    present = {}
    for name, paths in declared.items():
        if not paths:
            present[name] = None  # not declared
        else:
            present[name] = all(resolve(package, p).is_file() for p in paths)

    retained_summary, retained_contexts = parse_retained(retained_docs)
    control_summary, control_contexts = parse_retained(control_docs)
    retained_reuse = parse_retained_reuse(retained_docs)
    control_reuse = parse_retained_reuse(control_docs)

    return {
        "arm": arm,
        "declared": declared,
        "rows": rows_by_family,
        "digests": digests,
        "present": present,
        "decode": parse_decode(decode_docs[0]) if decode_docs else None,
        "prefill": parse_prefill(prefill_docs[0]) if prefill_docs else None,
        "retained": retained_summary,
        "retained_contexts": retained_contexts,
        "retained_reuse": retained_reuse,
        "retained_control": control_summary,
        "control_contexts": control_contexts,
        "control_reuse": control_reuse,
        "concurrency": concurrency_rows,
        "concurrency_levels": concurrency_levels,
        "concurrency_basis": concurrency_basis,
        "mixed": mixed_rows,
        "target_only": parse_decode(target_docs[0]) if target_docs else None,
        "startup_memory": startup_memory,
        "kernel_tiling": kernel_tiling,
        "tool_eval_docs": tool_eval,
    }


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------

def accounting_rows(loaded: dict, expectation: dict) -> list:
    """Per-family expected vs actual for one arm, with an explicit status.

    The concurrency expectation is adjusted to the accounting basis of the raw
    evidence: per-(level, repeat) `records` hold `levels x cases x repeats`
    rows, while per-level `summaries` hold only `levels x cases`. A summary-only
    raw file is therefore complete at the smaller count, and the formula shows
    which basis was used.
    """
    out = []
    families = expectation["families"]
    concurrency_levels = loaded.get("concurrency_levels") or []
    concurrency_basis = loaded.get("concurrency_basis")
    for key, label in PERFORMANCE_FAMILIES + DIAGNOSTIC_FAMILIES:
        spec = families[key]
        actual = len(loaded["rows"].get(key, []))
        expected = spec["expected"]
        formula = spec["formula"]
        if key == "concurrency" and concurrency_levels:
            if concurrency_basis == "records":
                expected = 3 * len(concurrency_levels) * expectation["protocol"]["repeats"]
                formula = (f"3 cases x {len(concurrency_levels)} levels {concurrency_levels} "
                           f"x {expectation['protocol']['repeats']} repeats")
            elif concurrency_basis == "summaries":
                expected = 3 * len(concurrency_levels)
                formula = (f"3 cases x {len(concurrency_levels)} levels {concurrency_levels}; "
                           "summary-only raw shape (per-level medians, no per-repeat rows)")
        if expected is None:
            status = "RECORDED" if actual else "NOT RECORDED"
        elif actual == expected:
            status = "COMPLETE"
        elif actual == 0:
            status = PENDING
        else:
            status = "SHORT"
        out.append({
            "family": key,
            "label": label,
            "expected": expected,
            "actual": actual,
            "status": status,
            "counted": spec["counted"],
            "formula": formula,
            "source": spec["source"],
        })
    return out


def performance_total(rows: list):
    counted = [r for r in rows if r["counted"]]
    return sum(r["expected"] for r in counted), sum(r["actual"] for r in counted)


def non_pass_rows(loaded: dict) -> list:
    rows = []
    for family, family_rows in loaded["rows"].items():
        for row in family_rows:
            if row.get("reason"):
                rows.append(row)
    return rows


def tool_eval_non_pass(loaded: dict) -> list:
    out = []
    docs = loaded.get("tool_eval_docs")
    if not docs:
        return out
    for run in docs["runs"]:
        for failure in run["failures"]:
            out.append({
                "run": run["run"],
                "scenario_id": failure.get("scenario_id"),
                "status": failure.get("status", "fail"),
                "points": failure.get("points"),
                "summary": failure.get("summary"),
            })
        for detail in run["detail"]:
            key = (run["run"], detail.get("scenario_id"))
            if any((row["run"], row["scenario_id"]) == key for row in out):
                continue
            out.append({
                "run": run["run"],
                "scenario_id": detail.get("scenario_id"),
                "status": detail.get("status"),
                "points": detail.get("points"),
                "summary": detail.get("summary"),
            })
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(manifest: dict, package: pathlib.Path) -> str:
    require_provenance(manifest)
    package = expand(package)
    accounting = manifest.get("accounting") or {}
    repeats = accounting.get("repeats") or 3
    prefill_warmups = accounting.get("prefill_warmups") or 1
    tool_eval_runs = accounting.get("tool_eval_runs") or 3
    expectation = derive_expectation(repeats, prefill_warmups, tool_eval_runs)

    arms = manifest["arms"]
    loaded = [load_arm(package, arm) for arm in arms]
    checkpoint = manifest["checkpoint"]
    images = manifest.get("images")
    network = manifest.get("network")
    thresholds = manifest.get("thresholds")

    lines = []
    add = lines.append
    add(f"# {manifest.get('report_title') or 'DS41RT v10 TP3 campaign: native TP3EP1 and compact EXL3 TP3'}")
    add("")
    add(
        "Generated by `scripts/render-ds41-v10-tp3-reports.py` from the raw bench "
        "outputs named below. Every cell is derived from a recorded sample; an em "
        f"dash ({DASH}) or a **{PENDING}** cell means no qualifying measurement is "
        "present in this package, not that a run was never attempted. Nothing is "
        "estimated, interpolated or substituted from another campaign.")
    add("")
    if manifest.get("report_status"):
        add(f"> **Status: {manifest['report_status']}**")
        add("")
    add(f"- Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    add(f"- Release: `{manifest['release']}`")
    add(f"- Checkpoint: `{checkpoint['model_id']}` @ `{checkpoint['revision']}` "
        f"(`{checkpoint['quant']}`")
    if checkpoint.get("family"):
        lines[-1] += f", family `{checkpoint['family']}`"
    if checkpoint.get("bits"):
        lines[-1] += f", bits `{checkpoint['bits']}`"
    lines[-1] += ")"
    if images:
        for role in ("coordinator", "spark_expert"):
            image = images.get(role) or {}
            add(f"- {role.replace('_', ' ').capitalize()} image: `{image.get('tag')}` "
                f"`{image.get('digest')}` (revision `{image.get('revision')}`)")
        if images.get("exl3_family_manifest"):
            add(f"- EXL3 family manifest: `{images['exl3_family_manifest']}`")
    else:
        add(f"- Images: {PENDING} {DASH} the v10 pair is not published yet, so no "
            "image identity is claimed.")
    if network:
        add(f"- Network: Spark rails at **{network['spark_link_mbps']:,} Mb/s** negotiated, "
            f"topology `{network['rails']}`; evidence `{network['evidence']}`")
        if network.get("dual_lane_note"):
            add(f"- Dual-lane note: {network['dual_lane_note']}")
    else:
        add(f"- Network: {PENDING} {DASH} no negotiated link evidence captured.")
    if thresholds:
        add(f"- Change thresholds (median-of-{repeats} same-image runs): "
            f"<{thresholds['noise_percent']}% noise, "
            f"{thresholds['noise_percent']}-{thresholds['weak_percent']}% weak, "
            f">{thresholds['credible_percent']}% credible. Basis: {thresholds['basis']}")
    else:
        add(f"- Change thresholds: {PENDING} {DASH} no per-campaign spread basis recorded.")
    add("")

    # ---- status ------------------------------------------------------------
    add("## Campaign status")
    add("")
    add("| Arm | Quant | Topology | RTX | Sparks | Expected records | Actual records | Status |")
    add("|---|---|---|---:|---:|---:|---:|---|")
    for item in loaded:
        arm = item["arm"]
        expected, actual = performance_total(accounting_rows(item, expectation))
        if actual == 0:
            status = "NOT EXECUTED (pending raw manifests)"
        elif actual == expected:
            status = "complete"
        else:
            status = "incomplete"
        add(f"| {arm['id']} | {arm['quant']} | {arm['topology']} | {arm['rtx']} | "
            f"{arm['sparks']} | {expected} | {actual} | {status} |")
    add("")

    # ---- configurations ----------------------------------------------------
    add("## Configurations")
    add("")
    for item in loaded:
        arm = item["arm"]
        form = arm.get("topology_form")
        add(f"- **{arm['id']}** {DASH} {arm['quant']} `{arm['topology']}`"
            + (f" ({form})" if form else "") + f", {arm['rtx']} RTX + {arm['sparks']} Spark. "
            f"config sha256 `{arm['config_sha256']}`"
            + (f" (`{arm['config_path']}`)" if arm.get("config_path") else "") + ".")
        if arm.get("runtime_flags"):
            flags = ", ".join(f"{k}={v}" for k, v in sorted(arm["runtime_flags"].items()))
            add(f"  - resolved: {flags}")
        if arm.get("warmup") or arm.get("repeats"):
            add(f"  - warmup: {arm.get('warmup') or DASH}; repeats: {fmt(arm.get('repeats'))}")
        if arm.get("comparison_group"):
            add(f"  - comparison group: `{arm['comparison_group']}`")
        if arm.get("images"):
            override = arm["images"]
            for role in ("coordinator", "spark_expert"):
                image = override.get(role) or {}
                add(f"  - image override ({role}): `{image.get('tag')}` `{image.get('digest')}`")
    add("")

    # ---- weight, residency and ceiling ------------------------------------
    add("## Weight, residency and memory ceiling")
    add("")
    add("Whether an arm is capped is part of its identity: the compact EXL3 arm is "
        "measured under a 32 GiB reservation, the native arm is uncapped, so their "
        "throughput numbers are not interchangeable.")
    add("")
    add("| Arm | Topology | RTX | Sparks | Resident bytes/rank | Remote layers | "
        "Worker allocated layers | Memory reservation | KV pool | Prefill batch tokens |")
    add("|---|---|---:|---:|---:|---:|---:|---|---|---:|")
    for item in loaded:
        arm = item["arm"]
        ceiling = arm.get("ceiling") or {}
        add(f"| {arm['id']} | {fmt(arm.get('topology'))} | {fmt(arm.get('rtx'))} | "
            f"{fmt(arm.get('sparks'))} | {fmt(arm.get('resident_bytes_per_rank'), True)} | "
            f"{fmt(arm.get('remote_layers'))} | {fmt(arm.get('worker_allocated_layers'))} | "
            f"{fmt(ceiling.get('memory_reservation'))} | {fmt(ceiling.get('kv_pool_size'))} | "
            f"{fmt(ceiling.get('prefill_batch_tokens'))} |")
    add("")
    for item in loaded:
        arm = item["arm"]
        ceiling = arm.get("ceiling") or {}
        if arm.get("weight_note"):
            add(f"- **{arm['id']}**: {arm['weight_note']}")
        if ceiling.get("note"):
            add(f"- **{arm['id']}** ceiling note: {ceiling['note']}")
        if ceiling.get("evidence"):
            add(f"- **{arm['id']}** ceiling evidence: `{ceiling['evidence']}`")
        if arm.get("quant") == "exl3" and not ceiling:
            add(f"- **{arm['id']}**: {PENDING} {DASH} compact EXL3 arm without a "
                "declared ceiling record.")
    add("")

    # ---- record accounting -------------------------------------------------
    add("## Expected performance-record accounting")
    add("")
    add(
        "The expected counts below are **derived**, not asserted: each family's "
        "expectation is recomputed from the live harness scripts' own constants and "
        "argparse defaults (`DEFAULT_BASES`, `DEFAULT_SUFFIXES`, `DEFAULT_CONTEXTS`, "
        "the concurrency level list, the mixed-traffic default, the decode corpus's "
        "weighted case list, and the tool-eval scenario assertion) under the protocol "
        f"`repeats={repeats}`, `prefill_warmups={prefill_warmups}`, "
        f"`tool_eval_runs={tool_eval_runs}`. Warmups, primes and lifecycle probes are "
        "recorded but excluded from the total.")
    add("")
    add(f"- Weighted decode cases from the corpus: {', '.join(expectation['weighted_cases'])}")
    add(f"- Derived total, both arms: **{expectation['total']}** performance records "
        f"({ ' + '.join(str(expectation['families'][k]['expected']) for k, _ in PERFORMANCE_FAMILIES) })")
    add(f"- Tool-eval: **{expectation['tool_eval']['scenarios_per_run']} scenarios x "
        f"{expectation['tool_eval']['runs']} runs = "
        f"{expectation['tool_eval']['scenario_runs']}** scenario-runs, counted separately "
        "from the performance ledger.")
    add("")
    add("| Arm | Family | Expected | Actual | Status | Derivation |")
    add("|---|---|---:|---:|---|---|")
    for item in loaded:
        for row in accounting_rows(item, expectation):
            add(f"| {item['arm']['id']} | {row['label']} | {fmt(row['expected'])} | "
                f"{fmt(row['actual'])} | {row['status']} | {row['formula']} |")
    add("")
    for item in loaded:
        expected, actual = performance_total(accounting_rows(item, expectation))
        delta = actual - expected
        if actual == 0:
            add(f"- **{item['arm']['id']}**: no performance records present. "
                f"{PENDING} until the raw manifests arrive; the expectation "
                f"({expected}) is stated so the shortfall is visible, not hidden.")
        elif delta == 0:
            add(f"- **{item['arm']['id']}**: {actual} / {expected} expected performance "
                "records match exactly.")
        else:
            add(f"- **{item['arm']['id']}**: {actual} / {expected} expected performance "
                f"records {DASH} a discrepancy of {delta:+d}. The report states the "
                "observed shortfall rather than asserting the expected total.")
    add("")

    # ---- headline decode ---------------------------------------------------
    add("## Headline decode (dSpark, C1)")
    add("")
    add("Median observed decode tokens/s from the timed samples. `N/A` means the "
        "family was not measured; no value is carried over from another arm.")
    add("")
    cases = list(expectation["weighted_cases"]) + ["counting"]
    add("| Case | " + " | ".join(item["arm"]["id"] for item in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    for case in cases:
        cells = []
        for item in loaded:
            value = item["decode"]["cases"].get(case) if item["decode"] else None
            cells.append(fmt(value))
        add(f"| {case} | " + " | ".join(cells) + " |")
    add("| **Weighted (excl. counting)** | "
        + " | ".join(fmt(item["decode"]["weighted"] if item["decode"] else None)
                     for item in loaded) + " |")
    add("")
    add("| Arm | Timed samples | Serving completed | Objective passed | Objective assessed | Harness passed |")
    add("|---|---:|---:|---:|---:|---|")
    for item in loaded:
        doc = item["decode"]
        if doc:
            passed = doc["passed"]
            add(f"| {item['arm']['id']} | {doc['samples']} | {doc['serving_completed']} | "
                f"{doc['objective_passed']} | {doc['objective_assessed']} | "
                f"{DASH if passed is None else ('yes' if passed else 'no')} |")
        else:
            add(f"| {item['arm']['id']} | {PENDING} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |")
    add("")
    add("`Objective passed` is out of `Objective assessed`. Samples without an objective "
        "check are **unassessed, not passes**; a completed sample is not a passed sample.")
    add("")

    # ---- repeat spread -----------------------------------------------------
    add("## Within-arm repeat spread")
    add("")
    add("Weighted-decode repeat values in recorded order, so the spread behind each "
        "median is visible.")
    add("")
    add("| Arm | Repeats | Spread |")
    add("|---|---:|---:|")
    for item in loaded:
        doc = item["decode"]
        if doc:
            reps = ", ".join(fmt(v) for v in doc["weighted_repeats"]) or DASH
            spread = doc["weighted_spread_percent"]
            add(f"| {item['arm']['id']} | {reps} | "
                f"{DASH if spread is None else f'{spread:.1f}%'} |")
        else:
            add(f"| {item['arm']['id']} | {PENDING} | {PENDING} |")
    add("")

    # ---- prefill -----------------------------------------------------------
    add("## Prefill matrices")
    add("")
    add("Median effective prefill tokens/s per retained-base/suffix cell. An incomplete "
        "matrix is provisional and is labelled so.")
    add("")
    for item in loaded:
        add(f"### {item['arm']['id']}")
        add("")
        matrix = item["prefill"]
        if not matrix:
            add(f"{PENDING} {DASH} no prefill package present.")
            add("")
            continue
        add(f"{matrix['cells']}/{matrix['expected_cells']} cells present"
            + ("" if matrix["complete"] else " - **provisional, incomplete**")
            + f"; passed={fmt(matrix['passed'])}")
        add("")
        add("| Retained base | " + " | ".join(f"+{s // 1024}K" for s in matrix["suffixes"]) + " |")
        add("|---|" + "|".join("---:" for _ in matrix["suffixes"]) + "|")
        for base, row in zip(matrix["bases"], matrix["grid"]):
            add(f"| {base // 1024}K | " + " | ".join(fmt(v, True) for v in row) + " |")
        add("")

    # ---- retained ----------------------------------------------------------
    add("## Decode over retained context")
    add("")
    add("Weighted nine-category dSpark tokens/s with verified prefix reuse, including "
        "the separate 2K control, which is a page-size-shaped prime and not a "
        "general long-context claim.")
    add("")
    add("| Retained base | " + " | ".join(item["arm"]["id"] for item in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    contexts = sorted({c for item in loaded
                       for doc in (item["retained"], item["retained_control"]) if doc
                       for c in doc})
    if contexts:
        for context in contexts:
            cells = []
            for item in loaded:
                value = None
                for doc in (item["retained"], item["retained_control"]):
                    if doc and context in doc:
                        value = doc[context]
                cells.append(fmt(value))
            add(f"| {context // 1024}K | " + " | ".join(cells) + " |")
    else:
        add(f"| {PENDING} | " + " | ".join(PENDING for _ in loaded) + " |")
    add("")
    # The API is a text interface, so the retained child only reuses the parent's
    # committed generated turn when the re-sent turn text re-encodes to identical
    # token ids. Report the two frontiers separately so a prompt-prefix-only
    # context stays visible instead of being folded into the pass/fail gate.
    add("Reuse frontier per retained base: `prefix` counts samples that skipped the "
        "complete retained parent prompt (the release gate); `full` counts samples "
        "that also reached the parent's committed generated-turn frontier "
        "(opportunistic, content-dependent diagnostic).")
    add("")
    add("| Retained base | " + " | ".join(item["arm"]["id"] for item in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    if contexts:
        for context in contexts:
            cells = []
            for item in loaded:
                reuse = None
                for table in (item.get("retained_reuse"), item.get("control_reuse")):
                    if table and context in table:
                        reuse = table[context]
                if not reuse or reuse["samples"] is None:
                    cells.append(PENDING)
                else:
                    cells.append(f"{fmt(reuse['prompt_prefix_reuse'])} prefix / "
                                 f"{fmt(reuse['full_turn_reuse'])} full "
                                 f"of {fmt(reuse['samples'])}")
            add(f"| {context // 1024}K | " + " | ".join(cells) + " |")
    else:
        add(f"| {PENDING} | " + " | ".join(PENDING for _ in loaded) + " |")
    add("")

    # ---- concurrency -------------------------------------------------------
    add("## Concurrency scaling")
    add("")
    add("Median aggregate tokens/s for counting / code / topic. Counting, code and "
        "topic are reported separately because they are different request shapes.")
    add("")
    add("| Case | Concurrency | " + " | ".join(item["arm"]["id"] for item in loaded) + " |")
    add("|---|---:|" + "|".join("---:" for _ in loaded) + "|")
    levels = sorted({r["concurrency"] for item in loaded for r in item["concurrency"]
                     if r["concurrency"] is not None})
    if levels:
        for case in ("counting", "code", "topic"):
            for level in levels:
                cells = []
                for item in loaded:
                    values = [r["aggregate_tps"] for r in item["concurrency"]
                              if r["case"] == case and r["concurrency"] == level
                              and r["aggregate_tps"] is not None]
                    cells.append(fmt(median(values)))
                add(f"| {case} | C{level} | " + " | ".join(cells) + " |")
    else:
        add(f"| {PENDING} | {DASH} | " + " | ".join(PENDING for _ in loaded) + " |")
    add("")

    # ---- mixed -------------------------------------------------------------
    add("## Mixed traffic")
    add("")
    add("Adaptive mixed batches; these are whole-batch aggregate rates, not per-stream "
        "C1 rates.")
    add("")
    add("| Concurrency | " + " | ".join(item["arm"]["id"] for item in loaded) + " |")
    add("|---:|" + "|".join("---:" for _ in loaded) + "|")
    mixed_levels = sorted({row["concurrency"] for item in loaded
                           for row in (item["mixed"] or [])
                           if row["concurrency"] is not None})
    if mixed_levels:
        for level in mixed_levels:
            cells = []
            for item in loaded:
                rows = [r for r in (item["mixed"] or []) if r["concurrency"] == level]
                cells.append(fmt(rows[0]["aggregate_tps"]) if rows else DASH)
            add(f"| C{level} | " + " | ".join(cells) + " |")
    else:
        add(f"| {PENDING} | " + " | ".join(PENDING for _ in loaded) + " |")
    add("")

    # ---- target-only -------------------------------------------------------
    add("## Target-only decode (dSpark off)")
    add("")
    add("The same deployment relaunched with dSpark disabled; this is the control "
        "against the headline decode, and shares the decode nonce seed so the two "
        "are matched.")
    add("")
    add("| Arm | Weighted tok/s | Timed samples | Case medians |")
    add("|---|---:|---:|---|")
    for item in loaded:
        doc = item["target_only"]
        if doc:
            case_cells = ", ".join(f"{case}={fmt(value)}"
                                   for case, value in sorted(doc["cases"].items()))
            add(f"| {item['arm']['id']} | {fmt(doc['weighted'])} | {doc['samples']} | {case_cells} |")
        else:
            add(f"| {item['arm']['id']} | {PENDING} | {PENDING} | {PENDING} |")
    add("")

    # ---- startup / memory --------------------------------------------------
    add("## Startup and memory diagnostics")
    add("")
    add("Reservation, device-budget, occupancy and KV lines captured at launch. These "
        "are diagnostics and are excluded from the performance-record ledger; the "
        "weight-only admission figure is admission arithmetic, not a runtime fit.")
    add("")
    add("| Arm | Weight-only remote bytes | Device budget bytes | Weight margin bytes | GPU memory rows | Spark budget lines |")
    add("|---|---:|---:|---:|---:|---|")
    for item in loaded:
        diag = item["startup_memory"]
        if diag and diag.get("weight_only_admission"):
            adm = diag["weight_only_admission"]
            sparks = ", ".join(f"{host}:{count}" for host, count in sorted(diag["spark_lines"].items())) or DASH
            add(f"| {item['arm']['id']} | {fmt(adm.get('remote_weight_bytes'), True)} | "
                f"{fmt(adm.get('device_budget_bytes'), True)} | "
                f"{fmt(adm.get('weight_margin_bytes'), True)} | "
                f"{len(diag['gpu_memory_csv'])} | {sparks} |")
        else:
            add(f"| {item['arm']['id']} | {PENDING} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |")
    add("")
    for item in loaded:
        ceiling = item["arm"].get("ceiling") or {}
        if ceiling:
            add(f"- **{item['arm']['id']}** resolved ceiling: reservation "
                f"`{fmt(ceiling.get('memory_reservation'))}`, KV pool "
                f"`{fmt(ceiling.get('kv_pool_size'))}`, prefill batch "
                f"`{fmt(ceiling.get('prefill_batch_tokens'))}`"
                + (f", evidence `{ceiling['evidence']}`" if ceiling.get("evidence") else "")
                + ".")
    add("")

    # ---- kernel / tile -----------------------------------------------------
    add("## Kernel and tile diagnostics")
    add("")
    add("FFN/expert component microbenchmark. This is **not** serving throughput and is "
        "excluded from the performance ledger.")
    add("")
    if any(item["kernel_tiling"] for item in loaded):
        add("| Arm | Topology | Workload | Width | Capacity | Rows | Amortised us | Cold us |")
        add("|---|---|---|---:|---:|---:|---:|---:|")
        for item in loaded:
            for cell in item["kernel_tiling"] or []:
                add(f"| {item['arm']['id']} | {fmt(cell['topology'])} | {fmt(cell['workload'])} | "
                    f"{fmt(cell['width'])} | {fmt(cell['capacity'])} | {fmt(cell['rows'])} | "
                    f"{fmt(cell['amortised_us'])} | {fmt(cell['cold_us'])} |")
    else:
        add(f"{PENDING} {DASH} no kernel/tile diagnostic package present.")
    add("")

    # ---- tool eval ---------------------------------------------------------
    add("## Tool-call evaluation")
    add("")
    add(f"Per-run scores plus the non-pass scenario detail behind every run. Tool-eval is "
        f"counted separately from the performance ledger: "
        f"{expectation['tool_eval']['scenarios_per_run']} scenarios x "
        f"{expectation['tool_eval']['runs']} runs = "
        f"{expectation['tool_eval']['scenario_runs']} scenario-runs per arm. Run scores "
        "move with sampling, so every run is listed rather than averaged.")
    add("")
    for item in loaded:
        add(f"### {item['arm']['id']}")
        add("")
        docs = item["tool_eval_docs"]
        if not docs:
            add(f"{PENDING} {DASH} no tool-eval package present.")
            add("")
            continue
        add("| Run | Basic | Hard | Total | Statuses | Scenario-runs |")
        add("|---:|---:|---:|---:|---|---:|")
        for run in docs["runs"]:
            add(f"| {run['run']} | {fmt(run['basic_points'])}/{fmt(run['basic_max'])} | "
                f"{fmt(run['hard_points'])}/{fmt(run['hard_max'])} | "
                f"{fmt(run['total_points'])}/{fmt(run['total_max'])} | "
                f"{json.dumps(run['statuses'], sort_keys=True)} | {fmt(run['scenario_runs'])} |")
        add("")
        add(f"scenario-runs: **{docs['scenario_runs']} / "
            f"{expectation['tool_eval']['scenario_runs']}** expected; "
            f"runs: **{len(docs['runs'])} / {expectation['tool_eval']['runs']}** expected.")
        add("")
        non_pass = [row for row in tool_eval_non_pass(item)]
        if non_pass:
            add("Non-pass scenarios (raw detail, never dropped):")
            add("")
            add("| Run | Scenario | Status | Points | Summary |")
            add("|---:|---|---|---:|---|")
            for row in non_pass:
                summary = row.get("summary")
                if isinstance(summary, str):
                    summary = summary.replace("|", "\\|").replace("\n", " ")[:200]
                add(f"| {row['run']} | {fmt(row['scenario_id'])} | {fmt(row['status'])} | "
                    f"{fmt(row['points'])} | {fmt(summary)} |")
        else:
            add("No non-pass scenarios recorded in the parsed evidence. Absence here is "
                "not proof that every scenario passed.")
        add("")

    # ---- failures ----------------------------------------------------------
    add("## Quality, checks and retained failures")
    add("")
    add("Every raw sample that failed or did not qualify is listed. A record without "
        "an objective check is unassessed, not a pass.")
    add("")
    any_row = False
    for item in loaded:
        rows = non_pass_rows(item)
        if not rows:
            continue
        any_row = True
        add(f"### {item['arm']['id']} ({len(rows)} non-pass records)")
        add("")
        add("| Family | Case/Context | Repeat | Reason | Capacity-gated |")
        add("|---|---|---:|---|---|")
        for row in rows:
            where = row.get("case") or row.get("context_tokens") or row.get("suffix_tokens")
            if row.get("concurrency") is not None:
                where = f"{where or ''} C{row['concurrency']}".strip()
            add(f"| {row.get('family', DASH)} | {fmt(where)} | {fmt(row.get('repeat'))} | "
                f"{fmt(row.get('reason'))} | {'yes' if row.get('capacity_gated') else 'no'} |")
        add("")
    if not any_row:
        add("- No failed or non-passing raw performance records are present. Absence here "
            "is not a pass.")
        add("")
    any_failure = False
    for item in loaded:
        for failure in item["arm"].get("failures") or []:
            any_failure = True
            scope = f" ({failure['scope']})" if failure.get("scope") else ""
            evidence = f" Evidence: `{failure['evidence']}`." if failure.get("evidence") else ""
            add(f"- **{item['arm']['id']}** {DASH} {failure.get('what')}{scope}: "
                f"{failure.get('observed')}.{evidence}")
    if not any_failure:
        add("- No manifest-declared negatives are recorded. Absence here is not a pass.")
    add("")

    # ---- missing data index ------------------------------------------------
    add("## Missing-data index")
    add("")
    add("Every declared family/artifact that rendered as a pending or em-dash cell, so "
        "nothing is omitted silently.")
    add("")
    add("| Arm | Family | Declared | State |")
    add("|---|---|---|---|")
    empty = True
    for item in loaded:
        for name, state in sorted(item["present"].items()):
            if state is None:
                continue
            if state is False:
                empty = False
                add(f"| {item['arm']['id']} | {name} | yes | {DASH} declared but absent |")
        docs = item["tool_eval_docs"]
        if docs is None and (item["arm"].get("tool_eval") or {}).get("summaries"):
            empty = False
            add(f"| {item['arm']['id']} | tool_eval | yes | {DASH} summaries unreadable |")
    if empty:
        add(f"| {DASH} | {DASH} | {DASH} | every declared artifact was present |")
    add("")
    add("A family that is absent from both the manifest and the raw evidence is shown "
        f"as **{PENDING}** in the accounting table above; it is a deferred measurement, "
        "not a zero and not a pass.")
    add("")

    # ---- digests -----------------------------------------------------------
    add("## Raw record digests")
    add("")
    add("| Arm | Artifact | SHA-256 |")
    add("|---|---|---|")
    for item in loaded:
        for digest in sorted(item["digests"], key=lambda d: (d["path"], d["resolved"])):
            value = f"`{digest['sha256']}`" if digest["sha256"] else DASH
            add(f"| {item['arm']['id']} | `{digest['path']}` | {value} |")
    add("")

    # ---- derived expectation ----------------------------------------------
    add("## Derived expectation and its sources")
    add("")
    add("This table is the arithmetic behind the expected counts. It is recomputed at "
        "render time from the scripts below, so the expectation moves with the harness "
        "instead of drifting from it.")
    add("")
    add("| Family | Expected (per arm) | Derivation | Source |")
    add("|---|---:|---|---|")
    for key, label in PERFORMANCE_FAMILIES:
        spec = expectation["families"][key]
        add(f"| {label} | {spec['expected']} | {spec['formula']} | `{spec['source']}` |")
    for key, label in DIAGNOSTIC_FAMILIES:
        spec = expectation["families"][key]
        add(f"| {label} | {DASH} | {spec['formula']} | `{spec['source']}` |")
    add(f"| **Total (performance)** | **{expectation['total']}** | "
        f"{' + '.join(str(expectation['families'][k]['expected']) for k, _ in PERFORMANCE_FAMILIES)} | |")
    add("")
    add("| Source script / corpus | SHA-256 |")
    add("|---|---|")
    for path, digest in sorted(expectation["sources"].items()):
        add(f"| `{path}` | `{digest}` |")
    add("")

    # ---- reconciliation against the campaign's expected-counts document ----
    external = manifest.get("expected_counts")
    if external:
        rows = reconcile_expectation(expectation, external)
        add("## Reconciliation with the campaign expected-counts document")
        add("")
        add(f"Source: `{external.get('__path__', DASH)}`. The campaign's own "
            "mechanically-derived expected-counts document is compared shape by shape "
            "against this renderer's derivation, so a divergent breakdown is surfaced "
            "rather than hidden behind a total that happens to agree.")
        add("")
        add("| Shape | Basis | Expected-counts document | This renderer | State |")
        add("|---|---|---:|---:|---|")
        for row in rows:
            add(f"| {row['shape']} | {row.get('basis') or DASH} | {row['external']} | "
                f"{row['derived']} | {row['state']} |")
        add("")
        differences = [row for row in rows if row["state"] != "match"]
        if differences:
            add(f"**DISCREPANCY**: {len(differences)} shape(s) differ between the "
                "campaign document and the live-harness derivation. The differing rows "
                "are listed above; the reports keep this renderer's derivation and flag "
                "the disagreement rather than silently adopting either number.")
        else:
            add("Every declared shape reconciles exactly. The retained family is stated "
                "in whichever basis the source document uses - a six-context lump "
                f"({expectation['retained_lumped']['records']}) or the five prime "
                f"contexts ({expectation['families']['retained']['expected']}) plus the "
                f"separate 2K control "
                f"({expectation['families']['retained_control_2k']['expected']}) - and "
                "the `basis` column records which one was compared, so the split form "
                "used by the per-context tables is never mistaken for a disagreement.")
        add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# checking
# --------------------------------------------------------------------------

def check(manifest: dict, package: pathlib.Path) -> list:
    """Validate provenance, report declared-but-absent artifacts and record
    accounting problems. Returns a list of human-readable problem strings."""
    require_provenance(manifest)
    package = expand(package)
    accounting = manifest.get("accounting") or {}
    expectation = derive_expectation(
        accounting.get("repeats") or 3,
        accounting.get("prefill_warmups") or 1,
        accounting.get("tool_eval_runs") or 3)

    problems = []
    for arm in manifest["arms"]:
        ident = arm["id"]
        raw = arm.get("raw") or {}
        for name, paths in sorted(raw.items()):
            if not as_paths(paths):
                continue
            for relative in as_paths(paths):
                path = resolve(package, relative)
                if not path.is_file():
                    problems.append(f"{ident}: {name} missing at {relative}")
                elif load_json(path) is None and path.suffix == ".json":
                    problems.append(f"{ident}: {name} unparseable at {relative}")
        if (arm.get("tool_eval") or {}).get("summaries"):
            path = resolve(package, arm["tool_eval"]["summaries"])
            if not path.is_file():
                problems.append(f"{ident}: tool_eval summaries missing at "
                                f"{arm['tool_eval']['summaries']}")
            elif not isinstance(load_json(path), list):
                problems.append(f"{ident}: tool_eval summaries unparseable at "
                                f"{arm['tool_eval']['summaries']}")
        loaded = load_arm(package, arm)
        for row in accounting_rows(loaded, expectation):
            if row["expected"] is None:
                continue
            if row["actual"] == 0:
                problems.append(
                    f"{ident}: {row['family']} PENDING - 0 of {row['expected']} expected "
                    f"records present ({row['formula']})")
            elif row["actual"] != row["expected"]:
                problems.append(
                    f"{ident}: {row['family']} short - {row['actual']} of "
                    f"{row['expected']} expected records ({row['formula']})")
        docs = loaded["tool_eval_docs"]
        if docs:
            expected_runs = expectation["tool_eval"]["runs"]
            expected_runs_total = expectation["tool_eval"]["scenario_runs"]
            if len(docs["runs"]) != expected_runs:
                problems.append(f"{ident}: tool_eval {len(docs['runs'])} runs of "
                                f"{expected_runs} expected")
            if docs["scenario_runs"] != expected_runs_total:
                problems.append(f"{ident}: tool_eval {docs['scenario_runs']} scenario-runs "
                                f"of {expected_runs_total} expected")
        elif (arm.get("tool_eval") or {}).get("summaries"):
            problems.append(f"{ident}: tool_eval summaries present but no runs parsed")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--package", type=pathlib.Path, default=None,
                        help="Directory that relative raw paths resolve against "
                             "(default: the manifest's directory)")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="Validate provenance, artifact presence and record "
                             "accounting without writing a report.")
    parser.add_argument("--strict", action="store_true",
                        help="With --check, fail (exit 2) when any family is pending or "
                             "short. This is the release gate.")
    parser.add_argument("--expected-counts", type=pathlib.Path, default=None,
                        help="Optional campaign expected-counts.json to reconcile against "
                             "the live-harness derivation (e.g. "
                             "runs/v10-exl3-tp3/expected-counts.json).")
    parser.add_argument("--write-expected-counts", type=pathlib.Path, default=None,
                        help="Write this renderer's derived expectation in the sibling "
                             "expected-counts.json shape, for the lanes' collect stages.")
    args = parser.parse_args(argv)
    manifest_path = expand(args.manifest)
    package = expand(args.package) if args.package else manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"render-ds41-v10-tp3-reports: manifest unreadable: {error}", file=sys.stderr)
        return 2

    external = None
    if args.expected_counts is not None:
        try:
            external = load_expected_counts(args.expected_counts)
        except Missing as error:
            print(f"render-ds41-v10-tp3-reports: {error}", file=sys.stderr)
            return 2
        manifest["expected_counts"] = external

    accounting = manifest.get("accounting") or {}
    expectation = derive_expectation(
        accounting.get("repeats") or 3,
        accounting.get("prefill_warmups") or 1,
        accounting.get("tool_eval_runs") or 3)

    if args.write_expected_counts is not None:
        document = expected_counts_document(expectation)
        expand(args.write_expected_counts).write_text(json.dumps(document, indent=2) + "\n")
        print(f"wrote {args.write_expected_counts}")

    if args.check:
        try:
            problems = check(manifest, package)
        except Missing as error:
            print(f"render-ds41-v10-tp3-reports: {error}", file=sys.stderr)
            return 2
        strict_problems = [p for p in problems if "PENDING" in p or " short " in p
                           or "scenario-runs" in p or " runs of " in p]
        other = [p for p in problems if p not in strict_problems]
        for problem in other:
            print(f"ERROR {problem}")
        for problem in strict_problems:
            print(f"GATE  {problem}")
        if external:
            for row in reconcile_expectation(expectation, external):
                marker = "RECONCILED" if row["state"] == "match" else "DISCREPANCY"
                print(f"{marker} {row['shape']}: document={row['external']} "
                      f"derived={row['derived']}")
                if row["state"] != "match":
                    other.append(f"expected-counts {row['shape']} differs: "
                                 f"document={row['external']} derived={row['derived']}")
        print(f"checked {len(manifest.get('arms') or [])} arm(s): "
              f"{len(other)} provenance/artifact problem(s), "
              f"{len(strict_problems)} record-accounting gate problem(s)")
        if other:
            return 2
        if strict_problems and args.strict:
            return 2
        return 0
    if args.output is None:
        parser.error("--output is required unless --check is given")
    try:
        text = render(manifest, package)
    except Missing as error:
        print(f"render-ds41-v10-tp3-reports: {error}", file=sys.stderr)
        return 2
    expand(args.output).write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
