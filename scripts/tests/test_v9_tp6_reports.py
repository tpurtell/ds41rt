#!/usr/bin/env python3
"""CPU-only tests for the v9/TP6 report renderer.

These tests use synthetic fixtures only. They assert renderer behaviour:
missing data becomes an em dash, no delta column is emitted, provenance is
mandatory, negative results survive, and no value is invented.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RENDERER = REPO / "scripts" / "render-ds41-v9-tp6-reports.py"
SCHEMA = REPO / "scripts" / "bench" / "v9-report-manifest.schema.json"
DASH = "\u2014"


def load_module():
    spec = importlib.util.spec_from_file_location("v9_reports", RENDERER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def module():
    return load_module()


def base_manifest():
    return {
        "release": "v9",
        "checkpoint": {
            "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
            "revision": "d" * 40,
            "quant": "official",
        },
        "images": {
            "coordinator": {"tag": "ghcr.io/tpurtell/ds41rt-coordinator:v9",
                            "digest": "sha256:" + "a" * 64, "revision": "b" * 40},
            "spark_expert": {"tag": "ghcr.io/tpurtell/ds41rt-spark-expert:v9",
                             "digest": "sha256:" + "c" * 64, "revision": "b" * 40},
        },
        "network": {"spark_link_mbps": 100000, "rails": "A-only",
                    "evidence": "provenance/links.txt"},
        "thresholds": {"noise_percent": 5, "weak_percent": 10, "credible_percent": 10,
                       "basis": "observed historical repeat spread"},
        "arms": [],
    }


def arm(arm_id, raw):
    return {
        "id": arm_id,
        "quant": "official",
        "topology": "TP4xEP1",
        "rtx": 1,
        "sparks": 4,
        "config_sha256": "e" * 64,
        "runtime_flags": {"rtx_expert_layers": 5, "remote_dispatch_layers": 35},
        "warmup": "shape warmup",
        "repeats": 3,
        "raw": raw,
    }


def write_decode(path: Path, weighted=(100.0, 102.0, 101.0), cases=None):
    cases = cases or {"code": [120.0, 121.0, 119.0], "counting": [150.0, 151.0, 149.0]}
    samples = []
    for case, values in cases.items():
        for index, value in enumerate(values):
            samples.append({
                "case": case,
                "repeat": index + 1,
                "observed_decode_tokens_per_second": value,
                "serving_completed": True,
                "objective_checks_passed": True,
            })
    doc = {
        "scope": "synthetic fixture",
        "label": "fixture",
        "passed": True,
        "repeats": 3,
        "nonce_seed": 1,
        "corpus_sha256": "f" * 64,
        "tokenizer_sha256": "0" * 64,
        "samples": samples,
        "repeat_summaries": [
            {"weighted_observed_decode_tokens_per_second": value} for value in weighted
        ],
        "median_weighted_observed_decode_tokens_per_second": sorted(weighted)[1],
    }
    path.write_text(json.dumps(doc))


def render(module, manifest, package: Path) -> str:
    return module.render(manifest, package)


def test_missing_raw_data_renders_em_dash(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": None, "prefill": "nope.json"})]
    text = render(module, manifest, tmp_path)
    assert "| Best prefill | " + DASH + " |" in text
    assert "not present" in text


def test_no_delta_column_in_headlines(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    headline = text.split("## Headlines", 1)[1].split("##", 1)[0]
    assert "\u0394" not in headline
    assert "Change" not in headline
    assert "| Measurement | a1 |" in headline


def test_decode_summary_medians_and_spread(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    write_decode(tmp_path / "d.json", weighted=(100.0, 102.0, 101.0))
    text = render(module, manifest, tmp_path)
    assert "| C1 code decode | 120.00 |" in text
    assert "| Counting decode | 150.00 |" in text
    assert "100.00, 102.00, 101.00" in text  # recorded repeat order is preserved


def test_provenance_is_mandatory(module, tmp_path):
    manifest = base_manifest()
    del manifest["images"]
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    with pytest.raises(module.Missing):
        render(module, manifest, tmp_path)


def test_missing_image_field_is_rejected(module, tmp_path):
    manifest = base_manifest()
    del manifest["images"]["coordinator"]["digest"]
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    with pytest.raises(module.Missing) as error:
        render(module, manifest, tmp_path)
    assert "images.coordinator.digest" in str(error.value)


def test_failures_are_retained(module, tmp_path):
    manifest = base_manifest()
    entry = arm("a1", {"decode": "d.json"})
    entry["failures"] = [
        {"what": "schema smoke", "observed": "empty content, finish_reason=length",
         "scope": "96-token budget with default thinking", "evidence": "raw/x.json"}
    ]
    manifest["arms"] = [entry]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    assert "schema smoke" in text
    assert "finish_reason=length" in text


def test_absent_failures_are_not_a_pass(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    assert "Absence here is not a pass" in text


def test_incomplete_prefill_is_provisional(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"prefill": "p.json"})]
    doc = {
        "bases": [0, 32768],
        "suffixes": [1024, 2048],
        "cells": [
            {"base_context_tokens": 0, "suffix_tokens": 1024,
             "median_effective_prefill_tokens_per_second": 3000.0},
            {"base_context_tokens": 0, "suffix_tokens": 2048,
             "median_effective_prefill_tokens_per_second": 4000.0},
            {"base_context_tokens": 32768, "suffix_tokens": 1024,
             "median_effective_prefill_tokens_per_second": 2500.0},
        ],
    }
    (tmp_path / "p.json").write_text(json.dumps(doc))
    text = render(module, manifest, tmp_path)
    assert "3/4 cells present" in text
    assert "provisional, incomplete" in text
    assert "| Best prefill | 4,000 |" in text


def test_digests_are_listed(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    assert "## Raw record digests" in text
    assert "decode (`d.json`) | `" in text


def test_cli_returns_two_on_bad_manifest(tmp_path):
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps({"release": "v9", "arms": []}))
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--output", str(tmp_path / "out.md")],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "missing" in result.stderr


def test_cli_writes_report(tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    write_decode(tmp_path / "d.json")
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "out.md"
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--output", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert "DS41RT v9 performance" in out.read_text()


def test_weight_residency_and_comparison_group(module, tmp_path):
    manifest = base_manifest()
    a1 = arm("a1", {"decode": "d.json"})
    a1["comparison_group"] = "official-1rtx"
    a1["resident_bytes_per_rank"] = 2_005_401_600
    a1["remote_layers"] = 35
    a1["worker_allocated_layers"] = 40
    a2 = arm("a2", {"decode": None})
    a2["topology"] = "TP6xEP1"
    a2["sparks"] = 6
    a2["comparison_group"] = "official-1rtx"
    a2["images"] = {
        "coordinator": {"tag": "ghcr.io/tpurtell/ds41rt-coordinator:v9",
                        "digest": "sha256:" + "9" * 64, "revision": "8" * 40},
        "spark_expert": {"tag": "ghcr.io/tpurtell/ds41rt-spark-expert:v9",
                         "digest": "sha256:" + "7" * 64, "revision": "8" * 40},
    }
    manifest["arms"] = [a1, a2]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    assert "## Weight and residency" in text
    assert "2,005,401,600" in text
    assert "comparison group" in text
    assert "image override" in text
    # An arm that declares no residency fields still renders, as an em dash.
    assert "| a2 | TP6xEP1 | 1 | 6 | " + DASH + " | " + DASH + " | " + DASH + " |" in text


def test_cost_policy_section_and_unresolved_flag(module, tmp_path):
    manifest = base_manifest()
    entry = arm("a1", {"decode": "d.json"})
    entry["cost_policy"] = {
        "default_mode": None,
        "legacy_mode": "legacy",
        "adaptive_mode_env": "legacy",
        "evidence": "evidence-1/coordinator.log",
    }
    manifest["arms"] = [entry]
    write_decode(tmp_path / "d.json")
    text = render(module, manifest, tmp_path)
    assert "## Speculative cost policy" in text
    # An unproven default mode must render as an em dash, not be guessed.
    assert f"| a1 | {DASH} | legacy | legacy | evidence-1/coordinator.log |" in text


def test_quality_section_separates_completed_from_passed(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "d.json"})]
    doc = {
        "samples": [
            {"case": "code", "observed_decode_tokens_per_second": 1.0,
             "serving_completed": True, "objective_checks_passed": True},
            {"case": "code", "observed_decode_tokens_per_second": 2.0,
             "serving_completed": True, "objective_checks_passed": None},
        ],
        "passed": True,
        "repeat_summaries": [],
    }
    (tmp_path / "d.json").write_text(json.dumps(doc))
    text = render(module, manifest, tmp_path)
    assert "unassessed, not passes" in text
    assert "| a1 | 2 | 2 | 1 | 1 | yes |" in text


def test_check_reports_missing_artifacts(module, tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "missing.json", "prefill": None})]
    problems = module.check(manifest, tmp_path)
    assert any("decode missing" in problem for problem in problems)
    assert any("prefill not declared" in problem for problem in problems)


def test_check_cli_strict_exit_code(tmp_path):
    manifest = base_manifest()
    manifest["arms"] = [arm("a1", {"decode": "missing.json"})]
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(manifest))
    loose = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check"],
        capture_output=True, text=True)
    assert loose.returncode == 0
    strict = subprocess.run(
        [sys.executable, str(RENDERER), "--manifest", str(manifest_path),
         "--package", str(tmp_path), "--check", "--strict"],
        capture_output=True, text=True)
    assert strict.returncode == 2
    assert "decode missing" in strict.stdout


def test_schema_file_is_valid_json_and_lists_required_provenance():
    schema = json.loads(SCHEMA.read_text())
    assert schema["required"] == ["release", "checkpoint", "images", "network", "thresholds", "arms"]
    assert set(schema["$defs"]["image"]["required"]) == {"tag", "digest", "revision"}
    assert "kernel_tiling" in schema["$defs"]["arm"]["properties"]["raw"]["properties"]
    assert "cost_policy" in schema["$defs"]["arm"]["properties"]
