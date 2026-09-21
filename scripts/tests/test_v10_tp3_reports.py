#!/usr/bin/env python3
"""CPU-only tests for the v10 TP3 campaign reporting tooling.

Synthetic fixtures only: no GPU, no Docker, no SSH, no network. The tests cover

  * a complete synthetic manifest for both arms, exercising every measured
    family (decode, prefill, retained prime/control including 2K, concurrency
    counting/code/topic, mixed, target-only, startup/memory and kernel/tile
    diagnostics) and the derived 359-record accounting;
  * missing-data behaviour: PENDING cells, never an invented number;
  * failure behaviour: every failed or non-passing raw sample is surfaced, the
    tool-eval section carries per-run scores plus non-pass scenario detail;
  * the compact EXL3 arm's ceiling fields;
  * expected-counts.json ingestion and reconciliation;
  * path handling: absolute and ~-expanded raw paths;
  * the --strict release gate and its exit codes.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RENDERER = REPO / "scripts" / "render-ds41-v10-tp3-reports.py"
SCHEMA = REPO / "scripts" / "bench" / "v10-tp3-report-manifest.schema.json"
CAMPAIGN = REPO / "scripts" / "bench" / "run-v10-tp3-campaign.sh"
EXPECTED_COUNTS = REPO / "runs" / "v10-exl3-tp3" / "expected-counts.json"
DASH = "\u2014"
PENDING = "PENDING"

WEIGHTED = ["code", "code-reasoning", "math", "fable", "hello", "topic",
            "structured-json", "structured-json-schema", "multilingual"]
BASES = [0, 32768, 65536, 131072, 262144]
SUFFIXES = [1024, 2048, 4096, 8192, 16384, 32768]
PRIME_CONTEXTS = [0, 32768, 65536, 131072, 262144]
LEVELS = [1, 2, 4, 8, 16]
REPEATS = 3


def load_module():
    spec = importlib.util.spec_from_file_location("v10_reports", RENDERER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def module():
    return load_module()


# ---------------------------------------------------------------- fixtures --

def write_decode(path: Path, label="fixture", failed_cases=(), non_pass_case=None):
    samples = []
    for repeat in range(1, REPEATS + 1):
        for case in WEIGHTED + ["counting"]:
            value = 100.0 + repeat
            sample = {
                "repeat": repeat,
                "case": case,
                "weight": 1.0,
                "timed": True,
                "observed_decode_tokens_per_second": value,
                "serving_completed": True,
                "objective_checks_passed": True,
                "passed": True,
            }
            if case in failed_cases:
                sample.update(serving_completed=False, passed=False,
                              objective_checks_passed=False,
                              error="HTTP 500 internal error")
            if case == non_pass_case:
                sample.update(objective_checks_passed=False, passed=False)
            samples.append(sample)
    doc = {
        "scope": "synthetic fixture",
        "label": label,
        "passed": not failed_cases and non_pass_case is None,
        "repeats": REPEATS,
        "nonce_seed": 79001,
        "corpus_sha256": "f" * 64,
        "tokenizer_sha256": "0" * 64,
        "samples": samples,
        "repeat_summaries": [
            {"weighted_observed_decode_tokens_per_second": 100.0 + repeat}
            for repeat in range(1, REPEATS + 1)
        ],
        "median_weighted_observed_decode_tokens_per_second": 101.0,
    }
    path.write_text(json.dumps(doc))
    return doc


def write_prefill(path: Path, missing_cell=None):
    cells, samples = [], []
    for base in BASES:
        for suffix in SUFFIXES:
            if (base, suffix) == missing_cell:
                continue
            value = 3000.0 + base / 1024 + suffix / 1024
            cells.append({
                "base_context_tokens": base,
                "suffix_tokens": suffix,
                "median_effective_prefill_tokens_per_second": value,
            })
            for _ in range(REPEATS):
                samples.append({
                    "base_context_tokens": base,
                    "suffix_tokens": suffix,
                    "timed": True,
                    "effective_prefill_tokens_per_second": value,
                    "error": None,
                })
    doc = {"bases": BASES, "suffixes": SUFFIXES, "cells": cells,
           "samples": samples, "passed": not missing_cell}
    path.write_text(json.dumps(doc))
    return doc


def write_retained(path: Path, contexts, failed_context=None):
    samples = []
    for context in contexts:
        for case in WEIGHTED:
            for repeat in range(1, REPEATS + 1):
                value = 80.0 + context / 1e6
                sample = {
                    "context_tokens": context,
                    "case": case,
                    "repeat": repeat,
                    "timed": True,
                    "observed_decode_tokens_per_second": value,
                    "serving_completed": True,
                    "objective_checks_passed": True,
                    "passed": True,
                }
                if context == failed_context:
                    sample.update(error="kv pool exhausted (capacity)", passed=False,
                                  capacity_gated=True)
                samples.append(sample)
    doc = {"contexts": list(contexts), "samples": samples,
           "passed": failed_context is None,
           "context_summaries": [
               {"context_tokens": context,
                "weighted_observed_decode_tokens_per_second": 80.0 + context / 1e6}
               for context in contexts
           ]}
    path.write_text(json.dumps(doc))
    return doc


def write_concurrency(path: Path, case: str, error_level=None):
    """Write the summary-only concurrency shape (per-level median_aggregate_tps)."""
    summaries = []
    for level in LEVELS:
        summaries.append({
            "concurrency": level,
            "samples": REPEATS,
            "median_aggregate_tps": 50.0 * level,
            "min_aggregate_tps": 50.0 * level - 1,
            "max_aggregate_tps": 50.0 * level + 1,
        })
    doc = {"case": case, "summaries": summaries, "passed": True,
           "repeats": REPEATS, "concurrency": LEVELS}
    path.write_text(json.dumps(doc))
    return doc


def write_concurrency_records(path: Path, case: str, error_level=None):
    """Write the per-(level, repeat) `records` shape and return its row count."""
    records = []
    for level in LEVELS:
        for repeat in range(1, REPEATS + 1):
            record = {
                "concurrency": level,
                "repeat": repeat,
                "aggregate_tps": 50.0 * level,
                "per_stream_mean": 50.0,
                "rows": [{"ok": True}] * level,
                "error": None,
            }
            if level == error_level:
                record.update(error="request failed: HTTP 503", rows=[])
            records.append(record)
    doc = {"case": case, "records": records, "passed": error_level is None,
           "summaries": [
               {"concurrency": level, "samples": REPEATS,
                "median_aggregate_tps": 50.0 * level}
               for level in LEVELS
           ]}
    path.write_text(json.dumps(doc))
    return doc


def write_mixed(path: Path, failed=False):
    batches = []
    for level in (4, 16):
        batches.append({
            "concurrency": level,
            "aggregate_tps": 40.0 * level,
            "rows": [{"ok": True}] * level,
            "passed": not failed,
        })
    doc = {"batches": batches, "passed": not failed,
           "failed_batch": ({"concurrency": 16, "failures": ["deadline"]} if failed else None)}
    path.write_text(json.dumps(doc))
    return doc


def write_memory(path: Path):
    doc = {
        "weight_only_admission": {
            "per_rank_layer_bytes": 2406481920,
            "remote_layers": 35,
            "remote_weight_bytes": 35 * 2406481920,
            "device_budget_bytes": 109119320064,
            "weight_margin_bytes": 109119320064 - 35 * 2406481920,
        },
        "coordinator_memory_lines": ["device_occupied=1 reservation=32GiB kv=2GiB"],
        "spark_memory_lines": {"ostrich": ["device_budget=109119320064"],
                               "dodo": ["device_budget=109119320064"],
                               "emu": ["device_budget=109119320064"]},
        "gpu_memory_csv": ["index, uuid, memory.used, memory.total", "0, GPU-a, 2, 97887"],
    }
    path.write_text(json.dumps(doc))
    return doc


def write_kernel(path: Path):
    cells = []
    for width in (64, 128, 192):
        for rows in (1, 64, 4096):
            cells.append({
                "topology": "tp3", "workload": "decode", "width": width,
                "capacity": 80, "rows": rows,
                "amortised_us": 10.0 + width / 10, "cold_us": 20.0 + width / 10,
            })
    doc = {"cells": cells}
    path.write_text(json.dumps(doc))
    return doc


def write_tool_eval(directory: Path, runs=3, non_pass=("scenario_007", "scenario_070")):
    """Write summaries.json plus per-run tool-eval.json scenario results."""
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(1, runs + 1):
        run_dir = directory / f"run-{index:02}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scenarios = []
        for scenario_index in range(1, 89):
            scenario_id = f"t{scenario_index:05d}"
            status = "pass"
            if f"scenario_{scenario_index:03d}" in non_pass:
                status = "fail"
            scenarios.append({
                "scenario_id": scenario_id,
                "status": status,
                "points": 0 if status != "pass" else 2,
                "summary": f"synthetic {scenario_id}",
            })
        (run_dir / "tool-eval.json").write_text(json.dumps({
            "run_id": f"run-{index}",
            "scores": {"scenario_results": scenarios, "total_points": 172,
                       "max_points": 176},
            "config": {"concurrency": 16,
                       "extra_params": {"thinking": {"type": "enabled"},
                                        "reasoning_effort": "high"}},
        }))
        statuses = {"pass": 86, "fail": 2}
        failures = [{"scenario_id": f"t{int(name.split('_')[1]):05d}",
                     "summary": f"synthetic failure {name}", "points": 0}
                    for name in non_pass]
        summary = {
            "run_id": f"run-{index}",
            "basic_points": 138, "basic_max": 138,
            "hard_points": 34, "hard_max": 38,
            "total_points": 172, "total_max": 176,
            "statuses": statuses,
            "failures": failures,
            "output_cap": 4096,
            "output_cap_source": "benchmark default",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        summaries.append(summary)
    (directory / "summaries.json").write_text(json.dumps(summaries))
    return summaries


# ------------------------------------------------------------------ manifest -

def base_manifest():
    return {
        "release": "v10",
        "checkpoint": {
            "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
            "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "quant": "official",
        },
        "accounting": {"repeats": REPEATS, "prefill_warmups": 1, "tool_eval_runs": 3},
        "images": {
            "coordinator": {"tag": "ghcr.io/tpurtell/ds41rt-coordinator:v10",
                            "digest": "sha256:" + "a" * 64, "revision": "b" * 40},
            "spark_expert": {"tag": "ghcr.io/tpurtell/ds41rt-spark-expert:v10",
                             "digest": "sha256:" + "c" * 64, "revision": "b" * 40},
        },
        "network": {"spark_link_mbps": 100000, "rails": "A-only",
                    "evidence": "logs/network.txt"},
        "thresholds": {"noise_percent": 5, "weak_percent": 10, "credible_percent": 10,
                       "basis": "observed historical repeat spread"},
        "arms": [],
    }


def native_arm(raw):
    return {
        "id": "v10-native-tp3ep1",
        "quant": "official",
        "topology": "TP3xEP1",
        "topology_form": "explicit",
        "rtx": 1,
        "sparks": 3,
        "config_path": "examples/configs/tp3ep1-native.config",
        "config_sha256": "e" * 64,
        "runtime_flags": {"rtx_expert_layers": 5, "remote_dispatch_layers": 35},
        "warmup": "shape warmup",
        "repeats": REPEATS,
        "raw": raw,
    }


def exl3_arm(raw):
    return {
        "id": "v10-exl3-compact-tp3",
        "quant": "exl3",
        "topology": "TP3",
        "topology_form": "implicit",
        "rtx": 1,
        "sparks": 3,
        "config_path": "examples/configs/exl3-compact-tp3.config",
        "config_sha256": "9" * 64,
        "ceiling": {
            "memory_reservation": "32GiB",
            "memory_reservation_bytes": 34359738368,
            "kv_pool_size": "2GiB",
            "kv_pool_size_bytes": 2147483648,
            "prefill_batch_tokens": 256,
            "evidence": "logs/coordinator-memory.txt",
            "note": "bounded starting prefill; not a proven fit",
        },
        "raw": raw,
    }


def full_raw(package: Path, tool_eval_dir=None):
    """Write a complete, passing raw package and return its manifest `raw` map.

    Concurrency uses the per-(level, repeat) `records` shape the real harness
    emits; the summary-only shape has its own test.
    """
    write_decode(package / "decode.json")
    write_prefill(package / "prefill.json")
    for context in PRIME_CONTEXTS:
        write_retained(package / f"retained-ctx-{context}.json", [context])
    write_retained(package / "retained-control-2k.json", [2048])
    for case in ("counting", "code", "topic"):
        write_concurrency_records(package / f"concurrency-{case}.json", case)
    write_mixed(package / "mixed.json")
    write_decode(package / "target-only.json", label="target")
    write_memory(package / "memory.json")
    write_kernel(package / "kernel-tp3.json")
    raw = {
        "decode": "decode.json",
        "prefill": "prefill.json",
        "retained": [f"retained-ctx-{c}.json" for c in PRIME_CONTEXTS],
        "retained_control_2k": "retained-control-2k.json",
        "concurrency_counting": "concurrency-counting.json",
        "concurrency_code": "concurrency-code.json",
        "concurrency_topic": "concurrency-topic.json",
        "mixed": "mixed.json",
        "target_only": "target-only.json",
        "startup_memory": "memory.json",
        "kernel_tiling": "kernel-tp3.json",
    }
    if tool_eval_dir is not None:
        write_tool_eval(package / tool_eval_dir)
        raw["tool_eval_marker"] = None  # not a schema field; tool_eval is separate
    return raw


def complete_manifest(package: Path, tool_eval=True):
    manifest = base_manifest()
    raw = full_raw(package)
    arm = native_arm(raw)
    if tool_eval:
        write_tool_eval(package / "tool-eval")
        arm["tool_eval"] = {
            "summaries": "tool-eval/summaries.json",
            "dir": "tool-eval",
            "runs": ["tool-eval/run-01", "tool-eval/run-02", "tool-eval/run-03"],
        }
    manifest["arms"] = [arm]
    return manifest


def render(module, manifest, package: Path) -> str:
    return module.render(manifest, package)


# ------------------------------------------------------------ complete data --

def test_complete_synthetic_manifest_derives_359(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    assert "Derived total, both arms: **359** performance records" in text
    assert "30 + 90 + 135 + 27 + 45 + 2 + 30" in text
    # Every performance family is COMPLETE and the arm is not pending.
    status = text.split("## Campaign status", 1)[1].split("##", 1)[0]
    assert "359 | 359 | complete" in status
    assert PENDING not in status


def test_complete_manifest_accounting_rows(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    loaded = module.load_arm(tmp_path, manifest["arms"][0])
    expectation = module.derive_expectation(REPEATS, 1, 3)
    rows = {row["family"]: row for row in module.accounting_rows(loaded, expectation)}
    assert rows["decode"]["actual"] == 30
    assert rows["prefill"]["actual"] == 90
    assert rows["retained"]["actual"] == 135
    assert rows["retained_control_2k"]["actual"] == 27
    assert rows["concurrency"]["actual"] == 45
    assert rows["mixed"]["actual"] == 2
    assert rows["target_only"]["actual"] == 30
    # Performance families are all complete; diagnostics are recorded separately.
    assert all(row["status"] == "COMPLETE"
               for row in rows.values() if row["expected"] is not None)
    assert rows["startup_memory"]["status"] == "RECORDED"
    assert rows["kernel_tiling"]["status"] == "RECORDED"
    expected, actual = module.performance_total(list(rows.values()))
    assert (expected, actual) == (359, 359)


def test_all_measured_families_render_values(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    for section in ("## Headline decode (dSpark, C1)", "## Prefill matrices",
                    "## Decode over retained context", "## Concurrency scaling",
                    "## Mixed traffic", "## Target-only decode (dSpark off)",
                    "## Startup and memory diagnostics",
                    "## Kernel and tile diagnostics", "## Tool-call evaluation"):
        assert section in text
    # 2K control is present in the retained table.
    retained = text.split("## Decode over retained context", 1)[1].split("##", 1)[0]
    assert "| 2K |" in retained
    # prefill grid shows all 30 cells
    assert "30/30 cells present" in text


def test_concurrency_counting_code_topic_are_separate(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    section = text.split("## Concurrency scaling", 1)[1].split("##", 1)[0]
    for case in ("counting", "code", "topic"):
        assert f"| {case} | C1 |" in section
        assert f"| {case} | C16 |" in section


def test_startup_memory_and_ceiling_render(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    arm = exl3_arm(manifest["arms"][0]["raw"])
    manifest["arms"] = [arm]
    text = render(module, manifest, tmp_path)
    assert "| Memory reservation | KV pool | Prefill batch tokens |" in text
    assert "32GiB" in text and "2GiB" in text and "| 256 |" in text
    assert "84,226,867,200" in text  # remote weight bytes
    assert "109,119,320,064" in text  # device budget


def test_concurrency_records_shape_is_the_complete_basis(module, tmp_path):
    """The per-(level, repeat) `records` shape counts 45 rows and is complete."""
    manifest = base_manifest()
    raw = {}
    for case in ("counting", "code", "topic"):
        write_concurrency_records(tmp_path / f"concurrency-{case}.json", case)
        raw[f"concurrency_{'counting' if case == 'counting' else case}"] = f"concurrency-{case}.json"
    manifest["arms"] = [native_arm(raw)]
    loaded = module.load_arm(tmp_path, manifest["arms"][0])
    assert len(loaded["rows"]["concurrency"]) == 45
    assert loaded["concurrency_basis"] == "records"
    expectation = module.derive_expectation(REPEATS, 1, 3)
    rows = {row["family"]: row for row in module.accounting_rows(loaded, expectation)}
    assert rows["concurrency"]["status"] == "COMPLETE"
    assert rows["concurrency"]["expected"] == 45


def test_concurrency_summary_only_shape_is_complete_on_its_own_basis(module, tmp_path):
    """A summary-only raw file holds 15 records and must not be called short."""
    manifest = base_manifest()
    raw = {}
    for case in ("counting", "code", "topic"):
        write_concurrency(tmp_path / f"concurrency-{case}.json", case)
        raw[f"concurrency_{'counting' if case == 'counting' else case}"] = f"concurrency-{case}.json"
    manifest["arms"] = [native_arm(raw)]
    loaded = module.load_arm(tmp_path, manifest["arms"][0])
    assert len(loaded["rows"]["concurrency"]) == 15
    assert loaded["concurrency_basis"] == "summaries"
    expectation = module.derive_expectation(REPEATS, 1, 3)
    rows = {row["family"]: row for row in module.accounting_rows(loaded, expectation)}
    assert rows["concurrency"]["status"] == "COMPLETE"
    assert rows["concurrency"]["expected"] == 15
    text = render(module, manifest, tmp_path)
    assert "summary-only raw shape" in text


def test_concurrency_error_record_is_surfaced(module, tmp_path):
    manifest = base_manifest()
    write_concurrency_records(tmp_path / "concurrency-code.json", "code", error_level=8)
    manifest["arms"] = [native_arm({"concurrency_code": "concurrency-code.json"})]
    text = render(module, manifest, tmp_path)
    assert "HTTP 503" in text


def test_mixed_failed_batch_detail_is_surfaced(module, tmp_path):
    manifest = base_manifest()
    write_mixed(tmp_path / "mixed.json", failed=True)
    manifest["arms"] = [native_arm({"mixed": "mixed.json"})]
    text = render(module, manifest, tmp_path)
    assert "deadline" in text


def test_kernel_tiling_is_excluded_from_the_ledger(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    expectation = module.derive_expectation(REPEATS, 1, 3)
    assert expectation["families"]["kernel_tiling"]["counted"] is False
    assert expectation["families"]["startup_memory"]["counted"] is False
    assert expectation["total"] == 359
    text = render(module, manifest, tmp_path)
    assert "excluded from the performance ledger" in text


# ------------------------------------------------------------ missing data --

def test_missing_raw_data_renders_pending_not_a_number(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({})]
    text = render(module, manifest, tmp_path)
    assert PENDING in text
    status = text.split("## Campaign status", 1)[1].split("##", 1)[0]
    assert "NOT EXECUTED (pending raw manifests)" in status
    assert "0" in status  # actual records
    # No decode median is invented.
    assert f"| code | {DASH} |" in text


def test_declared_but_absent_artifact_is_listed(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({"decode": "nope.json"})]
    text = render(module, manifest, tmp_path)
    assert "declared but absent" in text


def test_partial_family_is_short_not_complete(module, tmp_path):
    manifest = base_manifest()
    write_decode(tmp_path / "decode.json")
    manifest["arms"] = [native_arm({"decode": "decode.json"})]
    loaded = module.load_arm(tmp_path, manifest["arms"][0])
    expectation = module.derive_expectation(REPEATS, 1, 3)
    rows = {row["family"]: row for row in module.accounting_rows(loaded, expectation)}
    assert rows["decode"]["status"] == "COMPLETE"
    assert rows["prefill"]["status"] == PENDING
    text = render(module, manifest, tmp_path)
    assert "a discrepancy of" in text or "PENDING" in text


def test_incomplete_prefill_is_provisional(module, tmp_path):
    manifest = base_manifest()
    write_prefill(tmp_path / "prefill.json", missing_cell=(0, 1024))
    manifest["arms"] = [native_arm({"prefill": "prefill.json"})]
    text = render(module, manifest, tmp_path)
    assert "29/30 cells present" in text
    assert "provisional, incomplete" in text


# ----------------------------------------------------------------- failures --

def test_non_pass_decode_sample_is_surfaced(module, tmp_path):
    manifest = base_manifest()
    write_decode(tmp_path / "decode.json", non_pass_case="math")
    manifest["arms"] = [native_arm({"decode": "decode.json"})]
    text = render(module, manifest, tmp_path)
    section = text.split("## Quality, checks and retained failures", 1)[1]
    assert "non-pass records" in section
    assert "| decode | math |" in section
    assert "objective checks failed" in section


def test_failed_request_error_is_surfaced(module, tmp_path):
    manifest = base_manifest()
    write_decode(tmp_path / "decode.json", failed_cases=("code",))
    manifest["arms"] = [native_arm({"decode": "decode.json"})]
    text = render(module, manifest, tmp_path)
    assert "HTTP 500 internal error" in text
    # Every repeat of the failing case is retained, not collapsed to one row.
    assert "| decode | code | 1 | error: HTTP 500 internal error | no |" in text
    assert "| decode | code | 3 | error: HTTP 500 internal error | no |" in text


def test_serving_not_completed_is_surfaced_without_an_error_text(module, tmp_path):
    manifest = base_manifest()
    samples = [{
        "repeat": 1, "case": "code", "weight": 1.0, "timed": True,
        "observed_decode_tokens_per_second": None,
        "serving_completed": False, "passed": False,
    }]
    (tmp_path / "decode.json").write_text(json.dumps(
        {"samples": samples, "passed": False, "repeat_summaries": []}))
    manifest["arms"] = [native_arm({"decode": "decode.json"})]
    text = render(module, manifest, tmp_path)
    assert "serving not completed" in text


def test_capacity_gated_sample_is_marked_and_not_a_pass(module, tmp_path):
    manifest = base_manifest()
    write_retained(tmp_path / "retained.json", [262144], failed_context=262144)
    manifest["arms"] = [native_arm({"retained": "retained.json"})]
    loaded = module.load_arm(tmp_path, manifest["arms"][0])
    rows = module.non_pass_rows(loaded)
    assert rows and all(row["capacity_gated"] for row in rows)
    text = render(module, manifest, tmp_path)
    assert "kv pool exhausted" in text
    assert "| yes |" in text


def test_manifest_declared_failure_is_retained(module, tmp_path):
    manifest = base_manifest()
    arm = native_arm({})
    arm["failures"] = [{
        "what": "api-constrained-smoke",
        "observed": "RC=1 with empty content under default thinking",
        "scope": "96-token budget",
        "evidence": "logs/21-api-constrained.log",
    }]
    manifest["arms"] = [arm]
    text = render(module, manifest, tmp_path)
    assert "api-constrained-smoke" in text
    assert "RC=1 with empty content" in text


def test_absent_failures_is_not_a_pass(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    assert "Absence here is not a pass" in text


def test_tool_eval_per_run_scores(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    section = text.split("## Tool-call evaluation", 1)[1].split("\n## ", 1)[0]
    assert "| 1 | 138/138 | 34/38 | 172/176 |" in section
    assert "scenario-runs: **264 / 264** expected" in section
    assert "runs: **3 / 3** expected" in section


def test_tool_eval_non_pass_scenario_detail(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    text = render(module, manifest, tmp_path)
    section = text.split("## Tool-call evaluation", 1)[1].split("\n## ", 1)[0]
    assert "Non-pass scenarios" in section
    assert "| 1 | t00007 | fail | 0 |" in section
    assert "synthetic failure scenario_007" in section


def test_tool_eval_absent_is_pending(module, tmp_path):
    manifest = complete_manifest(tmp_path, tool_eval=False)
    text = render(module, manifest, tmp_path)
    section = text.split("## Tool-call evaluation", 1)[1].split("\n## ", 1)[0]
    assert PENDING in section


# --------------------------------------------------------------- provenance --

def test_provenance_is_mandatory(module, tmp_path):
    manifest = base_manifest()
    del manifest["checkpoint"]
    manifest["arms"] = [native_arm({})]
    with pytest.raises(module.Missing):
        render(module, manifest, tmp_path)


def test_wrong_release_is_rejected(module, tmp_path):
    manifest = base_manifest()
    manifest["release"] = "v9"
    manifest["arms"] = [native_arm({})]
    with pytest.raises(module.Missing):
        render(module, manifest, tmp_path)


def test_missing_image_field_is_rejected(module, tmp_path):
    manifest = base_manifest()
    del manifest["images"]["coordinator"]["digest"]
    manifest["arms"] = [native_arm({})]
    with pytest.raises(module.Missing) as error:
        render(module, manifest, tmp_path)
    assert "images.coordinator.digest" in str(error.value)


def test_exl3_arm_without_ceiling_is_rejected(module, tmp_path):
    manifest = base_manifest()
    arm = exl3_arm({})
    del arm["ceiling"]
    manifest["arms"] = [arm]
    with pytest.raises(module.Missing) as error:
        render(module, manifest, tmp_path)
    assert "ceiling" in str(error.value)


def test_images_may_be_absent_pending_publication(module, tmp_path):
    manifest = base_manifest()
    del manifest["images"]
    manifest["arms"] = [native_arm({})]
    text = render(module, manifest, tmp_path)
    assert "the v10 pair is not published yet" in text


# -------------------------------------------------------------- path handling -

def test_absolute_raw_paths_are_accepted(module, tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    evidence = tmp_path / "elsewhere"
    evidence.mkdir()
    write_decode(evidence / "decode.json")
    manifest = base_manifest()
    manifest["arms"] = [native_arm({"decode": str(evidence / "decode.json")})]
    loaded = module.load_arm(package, manifest["arms"][0])
    assert len(loaded["rows"]["decode"]) == 30
    text = render(module, manifest, package)
    assert "| decode | code |" not in text  # not a non-pass
    assert "decode.json" in text


def test_tilde_paths_are_expanded(module):
    home = Path.home()
    assert module.expand("~/x.json") == (home / "x.json").absolute()
    assert module.resolve(Path("/tmp"), "~/x.json") == (home / "x.json").absolute()


def test_relative_paths_resolve_against_package(module, tmp_path):
    (tmp_path / "raw").mkdir()
    write_decode(tmp_path / "raw" / "decode.json")
    resolved = module.resolve(tmp_path, "raw/decode.json")
    assert resolved == (tmp_path / "raw" / "decode.json").absolute()


# ------------------------------------------------------------------- check ---

def test_check_reports_missing_and_pending(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({"decode": "missing.json"})]
    problems = module.check(manifest, tmp_path)
    assert any("decode missing" in problem for problem in problems)
    assert any("PENDING" in problem for problem in problems)


def test_check_complete_package_has_no_record_problems(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    problems = module.check(manifest, tmp_path)
    assert problems == []


def test_check_flags_short_tool_eval(module, tmp_path):
    manifest = base_manifest()
    arm = native_arm({})
    write_tool_eval(tmp_path / "tool-eval", runs=2)
    arm["tool_eval"] = {"summaries": "tool-eval/summaries.json"}
    manifest["arms"] = [arm]
    problems = module.check(manifest, tmp_path)
    assert any("tool_eval 2 runs of 3 expected" in problem for problem in problems)


def test_check_cli_exit_codes(tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({})]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    loose = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check"],
        capture_output=True, text=True)
    assert loose.returncode == 0, loose.stdout + loose.stderr
    assert "GATE" in loose.stdout
    strict = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check", "--strict"],
        capture_output=True, text=True)
    assert strict.returncode == 2
    assert "PENDING" in strict.stdout


def test_strict_gate_passes_on_complete_package(tmp_path):
    manifest = complete_manifest(tmp_path)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check", "--strict"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_cli_rejects_declared_but_absent(tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({"decode": "missing.json"})]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check"],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "decode missing" in result.stdout


def test_cli_writes_report(tmp_path):
    manifest = complete_manifest(tmp_path)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "out.md"
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--output", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert "DS41RT v10 TP3 campaign" in out.read_text()


def test_cli_rejects_bad_manifest(tmp_path):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps({"release": "v10", "arms": []}))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--output", str(tmp_path / "out.md")],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "missing" in result.stderr


# ------------------------------------------------- expected-counts ingestion -

def test_reconciler_reads_the_campaign_document(module):
    if not EXPECTED_COUNTS.is_file():
        pytest.skip("campaign expected-counts.json not present")
    external = module.load_expected_counts(EXPECTED_COUNTS)
    expectation = module.derive_expectation(REPEATS, 1, 3)
    rows = {row["shape"]: row for row in module.reconcile_expectation(expectation, external)}
    assert rows["performance total"]["state"] == "match"
    assert rows["performance total"]["external"] == 359
    assert rows["tool_eval scenario-runs"]["state"] == "match"
    # The campaign document states retained on the five-prime-context basis.
    assert rows["retained"]["basis"] == "prime"
    assert rows["retained"]["state"] == "match"
    assert rows["retained"]["external"] == 135
    assert rows["retained_control_2k"]["state"] == "match"


def test_reconciler_detects_a_lumped_retained_basis(module):
    expectation = module.derive_expectation(REPEATS, 1, 3)
    external = {
        "shapes": {"retained": {"records": 162, "contexts": PRIME_CONTEXTS + [2048]}},
        "total_performance_records": 359,
    }
    rows = {row["shape"]: row for row in module.reconcile_expectation(expectation, external)}
    assert rows["retained"]["basis"] == "lumped"
    assert rows["retained"]["state"] == "match"


def test_reconciler_flags_a_real_mismatch(module):
    expectation = module.derive_expectation(REPEATS, 1, 3)
    external = {"shapes": {"decode": {"records": 29}}, "total_performance_records": 358}
    rows = {row["shape"]: row for row in module.reconcile_expectation(expectation, external)}
    assert rows["decode"]["state"] == "DIFFERS"
    assert rows["performance total"]["state"] == "DIFFERS"


def test_expected_counts_document_round_trips(module):
    expectation = module.derive_expectation(REPEATS, 1, 3)
    document = module.expected_counts_document(expectation)
    assert document["total_performance_records"] == 359
    assert document["shapes"]["retained"]["records"] == 135
    assert document["shapes"]["retained_control_2k"]["records"] == 27
    assert document["retained_lumped"]["records"] == 162
    assert document["tool_eval"]["scenario_runs"] == 264
    # The emitted document reconciles against a freshly derived expectation.
    rows = module.reconcile_expectation(expectation, document)
    assert all(row["state"] == "match" for row in rows)


def test_cli_expected_counts_reconciliation(tmp_path):
    if not EXPECTED_COUNTS.is_file():
        pytest.skip("campaign expected-counts.json not present")
    manifest = base_manifest()
    manifest["arms"] = [native_arm({})]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check",
         "--expected-counts", str(EXPECTED_COUNTS)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RECONCILED performance total: document=359 derived=359" in result.stdout
    assert "DISCREPANCY" not in result.stdout


def test_cli_write_expected_counts(tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [native_arm({})]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "expected-counts.json"
    subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check", "--write-expected-counts", str(out)],
        capture_output=True, text=True)
    document = json.loads(out.read_text())
    assert document["total_performance_records"] == 359
    assert document["shapes"]["decode"]["records"] == 30


# ------------------------------------------------------- expected arithmetic --

def test_expected_counts_are_derived_not_asserted(module):
    expectation = module.derive_expectation(REPEATS, 1, 3)
    fam = expectation["families"]
    assert fam["decode"]["expected"] == len(WEIGHTED + ["counting"]) * REPEATS
    assert fam["prefill"]["expected"] == len(BASES) * len(SUFFIXES) * REPEATS
    assert fam["retained"]["expected"] == len(PRIME_CONTEXTS) * len(WEIGHTED) * REPEATS
    assert fam["retained_control_2k"]["expected"] == 1 * len(WEIGHTED) * REPEATS
    assert fam["concurrency"]["expected"] == 3 * len(LEVELS) * REPEATS
    assert fam["mixed"]["expected"] == 2
    assert expectation["total"] == 359
    assert expectation["retained_lumped"]["records"] == 162
    assert expectation["tool_eval"]["scenario_runs"] == 264


def test_repeats_protocol_changes_the_expectation(module):
    two = module.derive_expectation(2, 1, 3)
    three = module.derive_expectation(3, 1, 3)
    assert two["families"]["decode"]["expected"] == 20
    assert three["families"]["decode"]["expected"] == 30
    assert two["total"] != three["total"]


# --------------------------------------------------------------- schema file -

def test_schema_is_valid_and_v10_specific():
    schema = json.loads(SCHEMA.read_text())
    assert schema["properties"]["release"]["const"] == "v10"
    assert set(schema["required"]) == {"release", "checkpoint", "arms"}
    arm = schema["$defs"]["arm"]
    assert "ceiling" in arm["properties"]
    assert set(arm["properties"]["ceiling"]["properties"]) >= {
        "memory_reservation", "kv_pool_size", "prefill_batch_tokens"}
    raw = schema["$defs"]["raw"]["properties"]
    for family in ("decode", "prefill", "retained", "retained_control_2k",
                   "concurrency_counting", "concurrency_code", "concurrency_topic",
                   "mixed", "target_only", "startup_memory", "kernel_tiling"):
        assert family in raw, family
    assert "tool_eval" in arm["properties"]
    # Provenance/limitation fields are schema-valid so a measured manifest can
    # carry its caveats without failing validation.
    assert "report_class" in schema["properties"]
    assert "harness_provenance" in schema["properties"]
    assert {"evidence_package", "measurement_provenance", "limitations",
            "bootstrap_only", "bootstrap_note"} <= set(arm["properties"])
    assert {"harness_provenance", "measurement_provenance", "limitation",
            "correction"} <= set(schema["$defs"])


def test_campaign_runner_plan_mentions_both_arms_and_gate():
    result = subprocess.run(["bash", str(CAMPAIGN), "plan"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "v10-native-tp3ep1" in result.stdout
    assert "v10-exl3-compact-tp3" in result.stdout
    assert "359" in result.stdout
    assert "264" in result.stdout
    assert "PENDING" in result.stdout


def test_campaign_runner_is_cpu_only_and_not_executing():
    text = CAMPAIGN.read_text()
    assert "does NOT execute" in text
    for forbidden in ("docker run", "ssh ", "nvidia-smi", "run.sh --config"):
        assert forbidden not in text


# ------------------------------------------------------- committed documents --

DOCS = (
    REPO / "docs" / "release-v10-tp3-official-1x-3spark.md",
    REPO / "docs" / "release-v10-tp3-exl3-compact-1x-3spark.md",
    REPO / "docs" / "release-v10-tp3-campaign-status.md",
)
GENERATOR = REPO / "runs" / "v10-tp3" / "generate-docs.py"
MANIFESTS = (
    REPO / "runs" / "v10-tp3" / "v10-native-tp3ep1-manifest.json",
    REPO / "runs" / "v10-tp3" / "v10-exl3-compact-tp3-manifest.json",
    REPO / "runs" / "v10-tp3" / "campaign-manifest.json",
)
NATIVE_DOC = DOCS[0]
EXL3_DOC = DOCS[1]
CAMPAIGN_DOC = DOCS[2]
NATIVE_MANIFEST = MANIFESTS[0]
EXL3_MANIFEST = MANIFESTS[1]
CAMPAIGN_MANIFEST = MANIFESTS[2]


def load_generator():
    spec = importlib.util.spec_from_file_location("v10_generate_docs", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module



def test_exl3_committed_doc_is_pending_not_fabricated():
    if not EXL3_DOC.is_file():
        pytest.skip(f"{EXL3_DOC.name} not present in this checkout")
    text = EXL3_DOC.read_text()
    assert "**Status: NOT EXECUTED" in text
    assert "PENDING" in text
    # No measured-looking numbers may be smuggled into a pending report: the
    # headline decode table must be all em dashes.
    headline = text.split("## Headline decode", 1)[1].split("##", 1)[0]
    assert "| code | " + DASH + " |" in headline


def test_native_committed_doc_is_informational_and_incomplete():
    if not NATIVE_DOC.is_file():
        pytest.skip("native report not present in this checkout")
    text = NATIVE_DOC.read_text()
    assert "**Status: INCOMPLETE / INFORMATIONAL" in text
    assert "Report class: `informational`" in text
    assert "not a formal release qualification" in text
    # The limitations travel with the report and name the post-run changes.
    assert "## Measurement provenance and limitations" in text
    assert "config-not-remeasured" in text
    assert "retained-harness-corrected-post-run" in text
    assert "KV_POOL_SIZE=12GiB" in text
    assert "a057296" in text
    assert "2f8208b" in text
    # Measured values are published unchanged, not rounded or promoted.
    assert "| **Weighted (excl. counting)** | 82.94 |" in text
    assert "309 / 359" in text
    assert "88 / 264" in text


def test_campaign_committed_doc_separates_measured_from_pending():
    if not CAMPAIGN_DOC.is_file():
        pytest.skip("campaign status report not present in this checkout")
    text = CAMPAIGN_DOC.read_text()
    assert "INCOMPLETE / INFORMATIONAL" in text
    assert ("| v10-native-tp3ep1 | official | TP3xEP1 | 1 | 3 | 359 | 309 | "
            "incomplete |") in text
    assert ("| v10-exl3-compact-tp3 | exl3 | TP3 | 1 | 3 | 359 | 0 | "
            "NOT EXECUTED (pending raw manifests) |") in text
    assert "config-not-remeasured" in text


def test_exl3_manifest_declares_no_raw_evidence():
    if not EXL3_MANIFEST.is_file():
        pytest.skip("EXL3 manifest not present in this checkout")
    document = json.loads(EXL3_MANIFEST.read_text())
    assert document["release"] == "v10"
    for arm in document["arms"]:
        assert arm["raw"] == {}
        assert "tool_eval" not in arm


def test_native_manifest_declares_measured_evidence_and_limitations():
    if not NATIVE_MANIFEST.is_file():
        pytest.skip("native manifest not present in this checkout")
    document = json.loads(NATIVE_MANIFEST.read_text())
    assert document["report_class"] == "informational"
    assert "INCOMPLETE / INFORMATIONAL" in document["report_status"]
    arm = document["arms"][0]
    assert arm["raw"], "the measured native arm must name its raw evidence"
    assert arm["tool_eval"]["summaries"]
    assert arm["evidence_package"] == "runs/v10-native-tp3ep1"
    # The measured config identity survives: this is the pre-KV-pin hash.
    assert arm["config_sha256"] == \
        "9e738e727a088a2d047402af51c4a8b7444bf12be6b44d87ea954de7c60de4eb"
    ids = {limitation["id"] for limitation in arm["limitations"]}
    assert {"config-not-remeasured", "retained-harness-corrected-post-run",
            "tool-eval-capacity-gated", "informational-not-qualification"} <= ids
    provenance = arm["measurement_provenance"]
    assert provenance["remeasured"] is False
    assert provenance["published_profile_kv_pool_size"] == "12GiB"
    assert provenance["pin_commit"].startswith("a057296")
    assert provenance["published_profile_config_sha256"] != \
        provenance["measured_config_sha256"]
    harness = document["harness_provenance"]
    assert harness["digests"]["scripts/bench-ds41-release-retained-decode.py"] \
        .startswith("180d0a6b")
    assert harness["corrections"][0]["commit"].startswith("2f8208b")


def test_campaign_manifest_keeps_exl3_pending_and_native_measured():
    if not CAMPAIGN_MANIFEST.is_file():
        pytest.skip("campaign manifest not present in this checkout")
    document = json.loads(CAMPAIGN_MANIFEST.read_text())
    assert document["report_class"] == "informational"
    assert "INCOMPLETE / INFORMATIONAL" in document["report_status"]
    arms = {arm["id"]: arm for arm in document["arms"]}
    assert arms["v10-native-tp3ep1"]["raw"]
    assert arms["v10-exl3-compact-tp3"]["raw"] == {}
    assert "tool_eval" not in arms["v10-exl3-compact-tp3"]


def test_limitations_render_without_changing_measured_values(module, tmp_path):
    manifest = complete_manifest(tmp_path)
    manifest["report_class"] = "informational"
    manifest["report_status"] = "INCOMPLETE / INFORMATIONAL - fixture"
    measured = "1" * 64
    manifest["harness_provenance"] = {
        "digests": {"scripts/bench-ds41-release-retained-decode.py": measured},
        "corrections": [{
            "path": "scripts/bench-ds41-release-retained-decode.py",
            "commit": "2f8208ba65ee1450d9e331ff52dd0b0fec01b33f",
            "measured_sha256": measured,
            "note": "fixture correction",
        }],
    }
    manifest["arms"][0]["measurement_provenance"] = {
        "measured_config_sha256": "2" * 64,
        "measured_kv_pool_size": "unset / daemon auto (~16.7 GB)",
        "published_profile_commit": "a057296d121651f80070d81b2d7c8e862146d9d1",
        "published_profile_config_sha256": "3" * 64,
        "published_profile_kv_pool_size": "12GiB",
        "remeasured": False,
    }
    manifest["arms"][0]["limitations"] = [{
        "id": "config-not-remeasured",
        "summary": "fixture summary",
        "detail": "fixture detail",
        "evidence": "fixture evidence",
    }]
    text = render(module, manifest, tmp_path)
    assert "Report class: `informational`" in text
    assert "## Measurement provenance and limitations" in text
    assert "**config-not-remeasured**" in text
    assert "Remeasured on the published profile: **no**" in text
    assert "fixture correction" in text
    # The caveats do not move a single measured value.
    assert "Derived total, both arms: **359** performance records" in text
    assert "359 | 359 | complete" in text


def test_generator_reports_committed_docs_as_current():
    if not GENERATOR.is_file() or not all(path.is_file() for path in DOCS):
        pytest.skip("v10-tp3 generator or docs not present in this checkout")
    result = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generator_check_never_modifies_a_committed_artifact():
    """A --check run must not reset a measured manifest or report."""
    if not GENERATOR.is_file():
        pytest.skip("v10-tp3 generator not present in this checkout")
    tracked = [path for path in
               (NATIVE_MANIFEST, EXL3_MANIFEST, CAMPAIGN_MANIFEST,
                NATIVE_DOC, EXL3_DOC, CAMPAIGN_DOC) if path.is_file()]
    before = {path: path.read_bytes() for path in tracked}
    subprocess.run([sys.executable, str(GENERATOR), "--check"],
                   capture_output=True, text=True)
    after = {path: path.read_bytes() for path in tracked}
    assert before == after


def test_generator_bootstrap_is_labelled_when_nothing_is_measured(monkeypatch, tmp_path):
    module = load_generator()
    # Pretend neither the lane manifest nor a tracked canonical manifest exists.
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module, "LANE_MANIFESTS", {})
    document, _package, origin = module.canonical_for(module.ARMS[0])
    assert origin == "bootstrap"
    arm = document["arms"][0]
    assert arm["bootstrap_only"] is True
    assert arm["raw"] == {}
    assert "PRE-MEASUREMENT BOOTSTRAP" in arm["bootstrap_note"]
    assert document["report_status"].startswith("NOT EXECUTED")


def test_generator_preserves_a_tracked_measured_manifest(monkeypatch):
    """With no lane manifest, the tracked measured copy is loaded, not reset."""
    if not NATIVE_MANIFEST.is_file():
        pytest.skip("native manifest not present in this checkout")
    module = load_generator()
    monkeypatch.setattr(module, "LANE_MANIFESTS", {})
    document, package, origin = module.canonical_for(module.ARMS[0])
    assert origin == "canonical"
    assert document["arms"][0]["raw"]
    assert package == "runs/v10-native-tp3ep1"
    assert module.evidence_available(document) is True


def test_generator_overlays_publication_metadata_onto_lane_manifest(monkeypatch, tmp_path):
    """A lane regeneration that drops the caveats cannot erase them."""
    if not NATIVE_MANIFEST.is_file():
        pytest.skip("native manifest not present in this checkout")
    module = load_generator()
    lane = tmp_path / "lane-manifest.json"
    lane.write_text(json.dumps({
        "release": "v10",
        "arms": [{"id": "v10-native-tp3ep1", "raw": {"decode": "raw/lane.json"}}],
    }))
    (tmp_path / "v10-native-tp3ep1-manifest.json").write_text(
        NATIVE_MANIFEST.read_text())
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module, "LANE_MANIFESTS",
                        {"v10-native-tp3ep1": (lane, "runs/lane")})
    document, _package, origin = module.canonical_for(module.ARMS[0])
    assert origin == "measured"
    # the lane's measured evidence wins ...
    assert document["arms"][0]["raw"] == {"decode": "raw/lane.json"}
    # ... while the publication caveats are restored from the canonical record.
    assert document["report_class"] == "informational"
    assert document["report_status"].startswith("INCOMPLETE / INFORMATIONAL")
    assert document["arms"][0]["limitations"]
    assert document["arms"][0]["measurement_provenance"]["remeasured"] is False




def test_exl3_manifest_declares_the_ceiling():
    path = REPO / "runs" / "v10-tp3" / "v10-exl3-compact-tp3-manifest.json"
    if not path.is_file():
        pytest.skip("v10-tp3 manifests not present in this checkout")
    arm = json.loads(path.read_text())["arms"][0]
    ceiling = arm["ceiling"]
    assert ceiling["memory_reservation"] == "32GiB"
    assert ceiling["kv_pool_size"] == "2GiB"
    assert ceiling["prefill_batch_tokens"] == 256
    assert ceiling["memory_reservation_bytes"] == 32 * 1024 ** 3
    assert ceiling["kv_pool_size_bytes"] == 2 * 1024 ** 3
