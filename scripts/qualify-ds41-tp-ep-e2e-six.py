#!/usr/bin/env python3
"""Six-rank (full six-Spark) E2E qualification wrapper over the frozen four-Spark runner.

Does NOT edit or copy the frozen runner. It loads
``scripts/qualify-ds41-tp-ep-e2e.py`` with runpy under a SHA-256 drift
assertion, reuses its content checks, per-request run loop (``cmd_run``) and
core ``compare_arms``/``coverage_report``/``identity_report``/
``applied_control_report``, and layers on:

* a versioned six-rank corpus schema (explicit ``configs`` and ``comparisons``,
  with the four decode cases copied verbatim from the frozen four-Spark corpus).
  v4 is the accepted three-arm revision; v5 adds the two pure unreplicated
  TP6EP1 arms and is a strict superset of v4. ``--corpus`` defaults to v4 for
  compatibility, so a TP6 arm must name
  ``scripts/fixtures/tp-ep-e2e-corpus-v5-six.jsonl``;
* an explicit canonical-scope gate: a qualifying six-rank arm must have run the
  exact C1/2/4/8/16 x3 matrix over all four frozen cases, regardless of what its
  own ``matrix`` field claims (CLI overrides are allowed for pilots but cannot
  qualify);
* an explicit comparison gate: paired 2RTX6 arms may be speed/quality compared,
  a 1RTX6 standalone arm needs quality *and* actual per-rank memory evidence,
  and a declared forbidden cross-hardware comparison is rejected with its reason;
* startup-metadata prevalidation before any request is sent.

No defaults are changed and nothing here starts a GPU, server or container.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "qualify-ds41-tp-ep-e2e.py"
EXPECTED_RUNNER_SHA256 = "e197146aa6c98de0c8c731a698de69cbb0f3dadf3edb6fe20e3690e82a6008c4"
V3_CORPUS = REPO / "scripts" / "fixtures" / "tp-ep-e2e-corpus.jsonl"
V4_CORPUS = REPO / "scripts" / "fixtures" / "tp-ep-e2e-corpus-v4-six.jsonl"
# v5 adds the two pure unreplicated TP6EP1 arms (2 RTX and 1 RTX) on top of the
# accepted v4 arms. The v4 corpus and its recorded sha256 are untouched: the
# already-accepted replicated-group results keep binding their own revision.
V5_CORPUS = REPO / "scripts" / "fixtures" / "tp-ep-e2e-corpus-v5-six.jsonl"
# sha256 of the accepted v4 six-rank corpus (the "99034171..." revision the
# completed replicated-group arms bind). The selftest asserts it is unchanged.
EXPECTED_V4_CORPUS_SHA256 = "99034171dba1fe8e6c64e41f9cbf07ab960d6c77932079f44a70c789cf836d85"
CANONICAL_CONCURRENCY = [1, 2, 4, 8, 16]
CANONICAL_REPEATS = 3
CONFIG_KEYS = ("id", "rtx_count", "spark_count", "spark_tp", "spark_ep", "world",
               "expected_resolved_rtx_expert_layers", "expected_spark_first_layer", "remote_layers",
               "requested_rtx_expert_layers", "kv_class", "memory_qualification",
               "require_explicit_rtx_expert_layers", "require_placement_directory",
               "spark_min_reserve_bytes", "coordinator_runtime_headroom_bytes",
               "spark_layer_bytes", "expected_spark_resident_bytes", "dspark_draft_limit")
# 20 GiB Spark host reserve, required for every config (actual MemTotal minus 20 GiB).
SPARK_MIN_RESERVE_BYTES = 21474836480
# 2 GiB coordinator/RTX planner runtime headroom: a SEPARATE domain, never the Spark reserve.
COORDINATOR_RUNTIME_HEADROOM_MIN_BYTES = 2147483648
# Per-TP-rank routed weight bytes for one 40-layer backbone layer (release-common values).
SPARK_LAYER_BYTES = {2: 3609722880, 3: 2406481920, 4: 2005401600, 6: 1203240960}
# The only approved six-rank layouts, keyed by id: (rtx_count, spark_count, spark_tp, spark_ep).
# `6x1` is the pure unreplicated TP6 layout (six disjoint intermediate slices of
# every expert), not an expert-parallel/duplicated group.
EXPECTED_CONFIG_TUPLES = {
    "2rtx6-tp2ep3": (2, 6, 2, 3),
    "2rtx6-tp3ep2": (2, 6, 3, 2),
    "1rtx6-tp3ep2": (1, 6, 3, 2),
    "2rtx6-tp6ep1": (2, 6, 6, 1),
    "1rtx6-tp6ep1": (1, 6, 6, 1),
}
# Corpus revisions are immutable once their arms are accepted: v4 covers the
# three replicated-group arms, v5 adds the two pure-TP6 arms. A newer corpus may
# add configs but may never drop an accepted one.
APPROVED_CONFIGS_BY_VERSION = {
    4: frozenset({"2rtx6-tp2ep3", "2rtx6-tp3ep2", "1rtx6-tp3ep2"}),
    5: frozenset(EXPECTED_CONFIG_TUPLES),
}
LATEST_CORPUS_VERSION = 5


def positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def non_negative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(allow_drift: bool = False):
    actual = sha256(RUNNER)
    if actual != EXPECTED_RUNNER_SHA256 and not allow_drift:
        raise SystemExit(
            f"frozen runner drift: {RUNNER} is {actual}, expected {EXPECTED_RUNNER_SHA256}; "
            "refusing to qualify against a moved helper"
        )
    return runpy.run_path(str(RUNNER)), actual


BASE, FROZEN_RUNNER_SHA256 = load_frozen()


def load_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def by_kind(records, kind):
    return [record for record in records if record.get("kind") == kind]


def case_identity(case: dict) -> dict:
    return {key: case.get(key) for key in ("id", "prompt", "max_tokens", "thinking", "check")}


def validate_six(records, frozen_records, corpus_path: Path = V4_CORPUS) -> dict:
    errors, configs, comparisons = [], {}, {}
    headers, controls_list, matrices = (by_kind(records, "corpus"), by_kind(records, "controls"),
                                        by_kind(records, "matrix"))
    if len(headers) != 1:
        errors.append("corpus must carry exactly one header")
    if len(controls_list) != 1:
        errors.append("corpus must carry exactly one controls record")
    if len(matrices) != 1:
        errors.append("corpus must carry exactly one matrix record")
    header = headers[0] if len(headers) == 1 else None
    controls = controls_list[0] if len(controls_list) == 1 else None
    matrix = matrices[0] if len(matrices) == 1 else None
    version = header.get("version") if header is not None else None
    approved = APPROVED_CONFIGS_BY_VERSION.get(version)
    if header is not None and approved is None:
        errors.append(f"corpus version {version!r} is not an approved revision "
                      f"{sorted(APPROVED_CONFIGS_BY_VERSION)}")
    if header is not None:
        if header.get("base_corpus_sha256") != sha256(V3_CORPUS):
            errors.append("header base_corpus_sha256 does not match the frozen four-Spark corpus")
        if header.get("frozen_runner_sha256") != FROZEN_RUNNER_SHA256:
            errors.append("header frozen_runner_sha256 does not match the loaded frozen runner")
        if header.get("base_corpus") != "scripts/fixtures/tp-ep-e2e-corpus.jsonl":
            errors.append("header base_corpus path is unexpected")
    if controls is None:
        errors.append("missing controls")
    else:
        for fake in ("spark_count", "spark_tp", "spark_ep", "world"):
            if fake in controls:
                errors.append(f"controls must not carry a fake {fake}; per-arm topology lives in configs")
        if controls.get("expert_format") != "native" or controls.get("native_only") is not True:
            errors.append("six-rank qualification is native-only (expert_format=native, native_only=true)")
    if matrix is None:
        errors.append("missing matrix")
    else:
        if matrix.get("concurrency") != CANONICAL_CONCURRENCY or matrix.get("repeats") != CANONICAL_REPEATS:
            errors.append(f"matrix must be exactly C{CANONICAL_CONCURRENCY} x {CANONICAL_REPEATS}")
    for config in by_kind(records, "config"):
        cid = config.get("id")
        if not isinstance(cid, str) or not cid or cid in configs:
            errors.append(f"config id missing or duplicated: {cid!r}")
            continue
        configs[cid] = config
        missing = [key for key in CONFIG_KEYS if key not in config]
        if missing:
            errors.append(f"config {cid}: missing {missing}")
            continue
        if cid not in EXPECTED_CONFIG_TUPLES:
            errors.append(f"config {cid}: unsupported layout; allowed are {sorted(EXPECTED_CONFIG_TUPLES)}")
        elif (config["rtx_count"], config["spark_count"], config["spark_tp"], config["spark_ep"]) != EXPECTED_CONFIG_TUPLES[cid]:
            errors.append(f"config {cid}: (rtx,spark,tp,ep) does not match the approved tuple {EXPECTED_CONFIG_TUPLES[cid]}")
        for key in ("rtx_count", "spark_count", "spark_tp", "spark_ep", "world", "remote_layers"):
            if not positive_int(config[key]):
                errors.append(f"config {cid}: {key} must be a positive integer (not a bool)")
        if not non_negative_int(config["expected_resolved_rtx_expert_layers"]) or config["expected_resolved_rtx_expert_layers"] > 40:
            errors.append(f"config {cid}: expected_resolved_rtx_expert_layers must be an integer in 0..40")
        if not non_negative_int(config["expected_spark_first_layer"]) or config["expected_spark_first_layer"] > 39:
            errors.append(f"config {cid}: expected_spark_first_layer must be an integer in 0..39")
        if config["spark_min_reserve_bytes"] != SPARK_MIN_RESERVE_BYTES:
            errors.append(f"config {cid}: spark_min_reserve_bytes must be {SPARK_MIN_RESERVE_BYTES} (20 GiB) for every config")
        if not non_negative_int(config["coordinator_runtime_headroom_bytes"]):
            errors.append(f"config {cid}: coordinator_runtime_headroom_bytes must be a non-negative integer "
                          "(a separate coordinator domain, not the Spark reserve)")
        expected_layer = SPARK_LAYER_BYTES.get(config["spark_tp"])
        if expected_layer is None or config["spark_layer_bytes"] != expected_layer:
            errors.append(f"config {cid}: spark_layer_bytes must be {expected_layer} for TP{config['spark_tp']}")
        elif config["expected_spark_resident_bytes"] != config["spark_layer_bytes"] * config["remote_layers"]:
            errors.append(f"config {cid}: expected_spark_resident_bytes != spark_layer_bytes * remote_layers")
        if not isinstance(config["memory_qualification"], bool):
            errors.append(f"config {cid}: memory_qualification must be a boolean")
        if not isinstance(config["require_explicit_rtx_expert_layers"], bool) or not isinstance(config["require_placement_directory"], bool):
            errors.append(f"config {cid}: require_explicit_rtx_expert_layers/require_placement_directory must be booleans")
        if not positive_int(config["dspark_draft_limit"]):
            errors.append(f"config {cid}: dspark_draft_limit must be a positive integer")
        if config["spark_tp"] * config["spark_ep"] != config["spark_count"]:
            errors.append(f"config {cid}: spark_tp*spark_ep != spark_count")
        if config["world"] != config["spark_count"]:
            errors.append(f"config {cid}: world != spark_count")
        if config["expected_spark_first_layer"] != min(config["expected_resolved_rtx_expert_layers"], 39):
            errors.append(f"config {cid}: expected_spark_first_layer != min(expected_resolved_rtx_expert_layers, 39)")
        if config["remote_layers"] != 40 - config["expected_resolved_rtx_expert_layers"]:
            errors.append(f"config {cid}: remote_layers != 40 - expected_resolved_rtx_expert_layers")
        if not non_negative_int(config["requested_rtx_expert_layers"]):
            errors.append(f"config {cid}: requested_rtx_expert_layers must be an explicit integer")
        if config["rtx_count"] == 2 and config["requested_rtx_expert_layers"] != 20:
            errors.append(f"config {cid}: a 2-RTX six-rank arm must request 20 rtx expert layers explicitly")
        if config["rtx_count"] == 1:
            # The current official 1-RTX placement keeps a small local expert set
            # (5 local / 35 remote). All-40-remote is a separate measured sidearm,
            # not the release-equivalent baseline, so either explicit value parses.
            if not non_negative_int(config["requested_rtx_expert_layers"]):
                errors.append(f"config {cid}: a 1-RTX arm must request an explicit rtx expert layer count")
            if not positive_int(config["coordinator_runtime_headroom_bytes"]):
                errors.append(f"config {cid}: a standalone arm must declare the coordinator runtime headroom separately")
            # A 1-RTX arm with local expert layers must hand them off, so it needs
            # a placement directory; the all-remote arm must not have one.
            has_local = config["expected_resolved_rtx_expert_layers"] > 0
            if has_local != bool(config["require_placement_directory"]):
                errors.append(f"config {cid}: require_placement_directory must be true exactly when a "
                              "1-RTX arm keeps local expert layers")
    if approved is not None and sorted(configs) != sorted(approved):
        errors.append(f"corpus v{version} must define exactly its approved configs {sorted(approved)}; got {sorted(configs)}")
    comparison_ids = []
    for comparison in by_kind(records, "comparison"):
        if comparison.get("id") in comparison_ids:
            errors.append(f"duplicate comparison id {comparison.get('id')!r}")
        comparison_ids.append(comparison.get("id"))
        if comparison.get("class") not in ("paired", "standalone"):
            errors.append(f"comparison {comparison.get('id')}: class must be paired or standalone")
        if comparison.get("class") == "paired":
            for side in ("control", "candidate"):
                if comparison.get(side) not in configs:
                    errors.append(f"comparison {comparison.get('id')}: unknown {side} config")
        elif comparison.get("candidate") not in configs:
            errors.append(f"comparison {comparison.get('id')}: unknown candidate config")
        comparisons[comparison.get("id")] = comparison
    for record in by_kind(records, "forbidden_comparison"):
        if record.get("control") not in configs or record.get("candidate") not in configs:
            errors.append(f"forbidden comparison {record.get('id')}: unknown config")
        if not str(record.get("reason", "")).strip():
            errors.append(f"forbidden comparison {record.get('id')}: reason required")
    frozen_cases = {case["id"]: case_identity(case) for case in by_kind(frozen_records, "case")}
    v4_case_ids = [case.get("id") for case in by_kind(records, "case")]
    if len(v4_case_ids) != len(set(v4_case_ids)):
        errors.append("duplicate case ids in v4")
    for case in by_kind(records, "case"):
        cid = case.get("id")
        if cid not in frozen_cases:
            errors.append(f"case {cid!r} is not part of the frozen four-Spark corpus")
        elif case_identity(case) != frozen_cases[cid]:
            errors.append(f"case {cid!r} does not match the frozen prompt/max_tokens/check/thinking byte-for-byte")
    if sorted(v4_case_ids) != sorted(frozen_cases):
        errors.append("v4 must carry every frozen decode case exactly once")
    return dict(passed=not errors, errors=errors, configs=sorted(configs), comparisons=sorted(comparisons),
                cases=len(v4_case_ids), corpus_sha256=sha256(corpus_path),
                base_corpus_sha256=sha256(V3_CORPUS), frozen_runner_sha256=FROZEN_RUNNER_SHA256)


def verify_memory_evidence(metadata: dict, config: dict) -> dict:
    """Two separate memory domains with actual arithmetic, machine-checked only.

    Spark (20 GiB host reserve on all configs): accounted resident + workspace +
    ring must not exceed the device budget, resident must reach the exact
    per-TP-rank layer bytes times remote layers, available memory must fit both
    MemTotal and the reserve, and every accounted field is explicit.
    Coordinator (2 GiB planner headroom) is separate and reports RTX device free
    bytes, never host free memory. A nonempty source is only presence: the gate
    stays review-required until ``logs_independently_reviewed`` is set.
    """
    problems = []
    evidence = metadata.get("memory_evidence")
    if not isinstance(evidence, dict):
        return dict(ok=False, arithmetic_ok=False, reviewed=False, evidence=None,
                    reasons=["memory_evidence missing; standalone cannot qualify on quality alone"])
    spark = evidence.get("spark")
    if not isinstance(spark, dict):
        problems.append("memory_evidence.spark missing; the Spark reserve is a separate 20 GiB domain")
    else:
        reserve = spark.get("reserve_bytes")
        minimum = config.get("spark_min_reserve_bytes", SPARK_MIN_RESERVE_BYTES)
        if not positive_int(reserve) or reserve < minimum:
            problems.append(f"memory_evidence.spark.reserve_bytes must be >= {minimum} (20 GiB); "
                            "the 2 GiB coordinator headroom is not a Spark reserve")
        if spark.get("kv_pool_class") != config.get("kv_class"):
            problems.append(f"memory_evidence.spark.kv_pool_class must be {config.get('kv_class')!r} "
                            "(single auto, never the fixed dual pool)")
        expected_resident = config.get("expected_spark_resident_bytes")
        if not positive_int(expected_resident):
            problems.append("config expected_spark_resident_bytes is not a positive integer")
        hosts = spark.get("hosts")
        if not isinstance(hosts, list) or len(hosts) != config["world"]:
            problems.append(f"memory_evidence.spark.hosts must list all {config['world']} Spark hosts")
        else:
            seen = set()
            for host in hosts:
                if not isinstance(host, dict):
                    problems.append("memory_evidence.spark.hosts entries must be objects")
                    continue
                rank = host.get("rank")
                if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < config["world"] or rank in seen:
                    problems.append(f"memory_evidence.spark.hosts has an invalid or duplicate rank {rank!r}")
                else:
                    seen.add(rank)
                mem_total, budget, available = (host.get("mem_total_bytes"), host.get("device_budget_bytes"),
                                                host.get("mem_available_bytes"))
                resident, workspace, ring = (host.get("accounted_resident_bytes"),
                                             host.get("accounted_workspace_bytes"), host.get("accounted_ring_bytes"))
                if not positive_int(mem_total):
                    problems.append(f"spark rank {rank!r} mem_total_bytes must be a positive integer")
                if not positive_int(budget):
                    problems.append(f"spark rank {rank!r} device_budget_bytes must be a positive integer")
                if positive_int(mem_total) and positive_int(reserve) and positive_int(budget) and budget > mem_total - reserve:
                    problems.append(f"spark rank {rank!r} device_budget_bytes {budget} exceeds "
                                    f"mem_total_bytes {mem_total} - reserve {reserve}")
                if not positive_int(available) or (positive_int(reserve) and available < reserve):
                    problems.append(f"spark rank {rank!r} mem_available_bytes must be a positive integer >= the 20 GiB reserve")
                if positive_int(available) and positive_int(mem_total) and available > mem_total:
                    problems.append(f"spark rank {rank!r} mem_available_bytes {available} exceeds mem_total_bytes {mem_total}")
                accounted = (resident, workspace, ring)
                if not all(non_negative_int(value) for value in accounted):
                    problems.append(f"spark rank {rank!r} accounted_resident/workspace/ring must be explicit non-negative integers")
                elif positive_int(budget) and resident + workspace + ring > budget:
                    problems.append(f"spark rank {rank!r} accounted resident+workspace+ring "
                                    f"{resident + workspace + ring} exceeds device_budget_bytes {budget}")
                if positive_int(expected_resident) and non_negative_int(resident) and resident < expected_resident:
                    problems.append(f"spark rank {rank!r} accounted_resident_bytes {resident} is below the exact minimum "
                                    f"{expected_resident} for this TP/remote-layer layout")
                if not str(host.get("source_log", "")).strip():
                    problems.append(f"spark rank {rank!r} needs a source_log (presence only; independent review required)")
            if seen != set(range(config["world"])):
                problems.append(f"memory_evidence.spark.hosts ranks {sorted(seen)} do not cover 0..{config['world'] - 1}")
    headroom_min = config.get("coordinator_runtime_headroom_bytes", 0)
    coordinator = evidence.get("coordinator")
    if headroom_min > 0:
        if not isinstance(coordinator, dict):
            problems.append("memory_evidence.coordinator missing; the planner runtime headroom is a separate domain")
        else:
            headroom, device_free = coordinator.get("runtime_headroom_bytes"), coordinator.get("device_free_bytes")
            if not positive_int(headroom) or headroom < max(headroom_min, COORDINATOR_RUNTIME_HEADROOM_MIN_BYTES):
                problems.append(f"memory_evidence.coordinator.runtime_headroom_bytes must be >= "
                                f"{max(headroom_min, COORDINATOR_RUNTIME_HEADROOM_MIN_BYTES)} (2 GiB), separate from the Spark reserve")
            if not positive_int(device_free) or (positive_int(headroom) and device_free < headroom):
                problems.append("memory_evidence.coordinator.device_free_bytes must be RTX device free bytes >= the runtime headroom")
            if not str(coordinator.get("source_log", "")).strip():
                problems.append("memory_evidence.coordinator needs a source_log (presence only; independent review required)")
    elif coordinator is not None and not isinstance(coordinator, dict):
        problems.append("memory_evidence.coordinator must be an object when present")
    reviewed = evidence.get("logs_independently_reviewed") is True
    reasons = problems if reviewed else problems + [
        "machine schema/arithmetic checks only; set logs_independently_reviewed after an independent per-host log review"]
    return dict(ok=not problems and reviewed, arithmetic_ok=not problems, reviewed=reviewed,
                reasons=reasons, evidence=evidence)


def arm_scope_failures(records, label: str, arm: dict, corpus_path: Path) -> list:
    """A qualifying arm must have the canonical matrix/cases and matching SHAs,
    regardless of what its own matrix field claims."""
    matrix = next((record for record in by_kind(records, "matrix")), {})
    canonical = dict(concurrency=list(matrix.get("concurrency", [])), repeats=matrix.get("repeats"))
    expected_cases = sorted(case["id"] for case in by_kind(records, "case"))
    actual_cases = sorted(case["id"] for case in arm.get("cases", []))
    failures = []
    if arm.get("matrix") != canonical:
        failures.append(f"{label}: arm matrix {arm.get('matrix')} != canonical {canonical}")
    if actual_cases != expected_cases:
        failures.append(f"{label}: arm cases {actual_cases} != canonical {expected_cases}")
    if arm.get("corpus_sha256") != sha256(corpus_path):
        failures.append(f"{label}: corpus_sha256 does not match the v4 corpus")
    if arm.get("frozen_runner_sha256") != FROZEN_RUNNER_SHA256:
        failures.append(f"{label}: frozen_runner_sha256 does not match the loaded runner")
    return failures


def comparison_report(left, right, configs: dict, comparisons, forbidden) -> dict:
    left_id = left.get("arm") if left else None
    right_id = right.get("arm")
    if right_id not in configs or (left_id is not None and left_id not in configs):
        return dict(allowed=False, passed=False, class_=None,
                    reasons=[f"unknown config for arm(s) {left_id!r}/{right_id!r}"])
    if left_id is not None:
        for record in forbidden:
            if {record.get("control"), record.get("candidate")} == {left_id, right_id}:
                return dict(allowed=False, passed=False, class_="forbidden",
                            reasons=[f"declared forbidden comparison: {record.get('reason')}"])
        paired = next((c for c in comparisons if c.get("class") == "paired"
                       and {c.get("control"), c.get("candidate")} == {left_id, right_id}), None)
        if paired is not None:
            left_config, right_config = configs[left_id], configs[right_id]
            reasons = []
            checks = {
                "rtx_count": (left_config["rtx_count"], right_config["rtx_count"]),
                "spark_count": (left_config["spark_count"], right_config["spark_count"]),
                "resolved_rtx_expert_layers": (left_config["expected_resolved_rtx_expert_layers"],
                                               right_config["expected_resolved_rtx_expert_layers"]),
                "spark_first_layer": (left_config["expected_spark_first_layer"],
                                      right_config["expected_spark_first_layer"]),
                "kv_class": (left_config["kv_class"], right_config["kv_class"]),
            }
            for name, (a, b) in checks.items():
                if a != b:
                    reasons.append(f"paired comparison requires equal {name}: {a!r} != {b!r}")
            return dict(allowed=True, passed=not reasons, class_="paired", comparison=paired.get("id"), reasons=reasons)
    standalone = next((c for c in comparisons if c.get("class") == "standalone"
                       and c.get("candidate") == right_id), None)
    if standalone is not None and (left_id is None or left_id == right_id):
        memory = verify_memory_evidence(right.get("startup_metadata") or {}, configs[right_id])
        return dict(allowed=True, passed=memory["ok"], class_="standalone", comparison=standalone.get("id"),
                    memory=memory, reasons=memory["reasons"])
    return dict(allowed=False, passed=False, class_=None,
                reasons=[f"no declared comparison covers {left_id!r} and {right_id!r}"])


def cmd_validate(args) -> int:
    report = validate_six(load_jsonl(args.corpus), load_jsonl(V3_CORPUS), args.corpus)
    report["corpus"] = str(args.corpus)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def run_corpus_path(output: Path, records) -> Path:
    """The frozen runner's validate_corpus does not know the v4-only kinds
    (config/comparison/forbidden_comparison). Write a temporary run corpus that
    carries the identical controls/matrix/case/prefill records so the frozen run
    loop is reused unchanged, and record both hashes."""
    compatible = [record for record in records if record.get("kind") in
                  ("corpus", "controls", "matrix", "case", "prefill", "gates", "future_scope")]
    path = output.with_name(output.name + ".run-corpus.jsonl")
    path.write_text("".join(json.dumps(record) + "\n" for record in compatible))
    return path


def verify_common_controls(records, config: dict, metadata: dict) -> list:
    """Enforce the corpus common controls against the actual metadata, so a
    self-consistent but reduced run (metadata concurrency 1 and argv
    --concurrency 1) cannot qualify when the corpus says 16."""
    controls = next(iter(by_kind(records, "controls")), {})
    problems = []
    mapping = (("concurrency", controls.get("concurrency")),
               ("prefill_batch_tokens", controls.get("prefill_batch_tokens")),
               ("worker_capacity", controls.get("worker_capacity")),
               ("prefix_cache_entries", controls.get("prefix_cache_entries")),
               ("max_context_tokens", controls.get("max_context_tokens")),
               ("max_output_tokens", controls.get("max_output_tokens")),
               ("model", controls.get("checkpoint")))
    for field, expected in mapping:
        if expected is None:
            continue
        if metadata.get(field) != expected:
            problems.append(f"startup_metadata {field}={metadata.get(field)!r} != common control {expected!r}")
    if "expert_format" in metadata and metadata["expert_format"] != controls.get("expert_format"):
        problems.append(f"startup_metadata expert_format={metadata.get('expert_format')!r} != common control {controls.get('expert_format')!r}")
    if positive_int(metadata.get("worker_capacity")) and positive_int(controls.get("concurrency")):
        if metadata["worker_capacity"] < controls["concurrency"]:
            problems.append(f"worker_capacity={metadata['worker_capacity']} is below the C{controls['concurrency']} admission need")
    return problems


def corpus_revision(records) -> dict:
    """The corpus header's own revision identity, never a hardcoded default.

    A v5 report must not claim to be v4: the schema string and name are read
    from the corpus that was actually validated.
    """
    header = next((record for record in by_kind(records, "corpus")), {})
    version = header.get("version")
    return dict(version=version,
                schema=header.get("name") or f"tp-ep-e2e-v{version}-six",
                name=header.get("name"))


def verify_arm_metadata(records, config: dict, metadata: dict) -> list:
    problems = []
    for field, expected in (("rtx_gpus", config["rtx_count"]), ("spark_tp", config["spark_tp"]),
                            ("spark_ep", config["spark_ep"]), ("spark_count", config["spark_count"]),
                            ("resolved_rtx_expert_layers", config["expected_resolved_rtx_expert_layers"]),
                            ("spark_first_layer", config["expected_spark_first_layer"]),
                            ("requested_rtx_expert_layers", config["requested_rtx_expert_layers"]),
                            ("dspark_draft_limit", config["dspark_draft_limit"])):
        if metadata.get(field) != expected:
            problems.append(f"startup_metadata {field}={metadata.get(field)!r} != config {expected!r}")
    argv = metadata.get("coordinator_argv")
    parsed = BASE["parse_argv"](argv)[0] if isinstance(argv, list) and argv else {}
    if config["require_explicit_rtx_expert_layers"]:
        requested = str(config["requested_rtx_expert_layers"])
        if str(parsed.get("--rtx-expert-layers")) != requested:
            problems.append(f"coordinator argv must pass --rtx-expert-layers {requested} explicitly; "
                            "an omitted flag defaults to auto and reabsorbs the KV/experts")
    if config["require_placement_directory"] and "--placement-directory" not in parsed:
        problems.append("2RTX six-rank requires --placement-directory")
    if not config["require_placement_directory"] and "--placement-directory" in parsed:
        problems.append("1RTX must omit --placement-directory")
    problems.extend(verify_common_controls(records, config, metadata))
    problems.extend(verify_worker_runtime_role(records, config, metadata))
    problems.extend(verify_cost_model_evidence(records, metadata))
    wrapped = {"startup_metadata": metadata}
    problems.extend(BASE["applied_control_report"](wrapped, wrapped))
    return problems


# Native role id per explicit TP degree: the worker's own readiness line must
# report the role it loaded, and the family's logical intermediate. An artifact
# manifest existing in an image is not proof that a running rank loaded it.
EXPLICIT_TP_ROLE_INTERMEDIATE = {2: (5, 1152), 3: (6, 768), 4: (1, 640), 6: (7, 384)}
# Corpus revision at which the per-rank runtime-role evidence became a required
# field. Newer revisions cannot omit it; older accepted results are untouched.
WORKER_RUNTIME_ROLE_SINCE = 5


# The daemon's structured startup line. Matching the exact message keeps a
# similarly-worded readiness echo or a partial line from being accepted.
WORKER_READY_MESSAGE = "native local RoCE expert worker ready"
# CSI escape sequences (the real tracing log inserts dim/italic codes between a
# field name and its `=`, so the raw line does not parse until they are removed).
ANSI_CSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def worker_role_in_log(text: str, rank: int):
    """Extract `(role, intermediate, world)` from a worker's structured startup line.

    Only a line that (a) contains the exact ready message, (b) names this rank on
    a word boundary (`rank=1` must not match `rank=10`), and (c) carries both
    `role` and `intermediate` is accepted. ANSI control sequences are stripped
    first. The last such line wins, so a restart in the same appended log still
    reports the final load.
    """
    best = None
    for raw in text.splitlines():
        line = ANSI_CSI.sub("", raw)
        if WORKER_READY_MESSAGE not in line:
            continue
        if not re.search(rf"(?:^|[^0-9])rank={rank}(?![0-9])", line):
            continue
        fields = dict(re.findall(r"([a-z_]+)=([0-9]+)", line))
        if "role" in fields and "intermediate" in fields and "world" in fields:
            best = (int(fields["role"]), int(fields["intermediate"]), int(fields["world"]))
    return best


# The resolved adaptive verification-cost model, as the daemon logs it at
# startup (`cost_model=<label>`). A report must state which model its numbers
# came from; a topology A/B under two different models is confounded.
# `explicit-profile-missing` is deliberately NOT an acceptable resolved model:
# the daemon refuses to serve with that configuration, so a report claiming it
# describes a run that could not have served. `auto` may legitimately resolve to
# `explicit-profile` when the operator supplied a readable
# DS41RT_ADAPTIVE_COST_PROFILE path, or to `legacy-heuristic` otherwise.
COST_MODEL_LABELS = (
    "legacy-heuristic",
    "builtin-calibration",
    "explicit-profile",
)


def cost_model_in_log(text: str):
    """The last `cost_model=` label in a coordinator log, or None."""
    found = None
    for line in text.splitlines():
        for label in COST_MODEL_LABELS:
            if f"cost_model={label}" in line:
                found = label
    return found


def verify_cost_model_evidence(records, metadata: dict) -> list:
    """Bind the declared cost model to the captured coordinator log (v5+).

    The knob an operator requested is recorded separately from the model the
    daemon actually resolved, because `auto` resolves differently per layout.
    """
    revision = corpus_revision(records)["version"]
    requested = metadata.get("requested_adaptive_cost_mode")
    resolved = metadata.get("resolved_cost_model")
    if revision < WORKER_RUNTIME_ROLE_SINCE and requested is None and resolved is None:
        return []
    problems = []
    if requested not in (None, "auto", "legacy", "builtin", "profile"):
        problems.append(f"requested_adaptive_cost_mode must be auto/legacy/builtin/profile, got {requested!r}")
    if resolved not in COST_MODEL_LABELS:
        problems.append(f"resolved_cost_model must be one of {list(COST_MODEL_LABELS)}, got {resolved!r}")
    source_log = str(metadata.get("cost_model_source_log", "") or "").strip()
    declared = metadata.get("cost_model_source_log_sha256")
    if not source_log:
        problems.append("cost_model_source_log is required so the resolved model is traceable to a capture")
        return problems
    path = Path(source_log)
    if not path.is_file():
        return problems + [f"cost_model_source_log does not exist: {source_log}"]
    if not isinstance(declared, str) or len(declared) != 64:
        problems.append("cost_model_source_log_sha256 (64 hex) is required for the captured log")
    else:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != declared:
            problems.append("cost_model_source_log_sha256 does not match the captured coordinator log")
    observed = cost_model_in_log(path.read_text(errors="replace"))
    if observed is None:
        problems.append("captured coordinator log has no cost_model= startup line")
    elif resolved in COST_MODEL_LABELS and observed != resolved:
        problems.append(f"resolved_cost_model={resolved!r} but the captured log reports {observed!r}")
    # A pinned request must be the one that actually resolved. `auto` is defined
    # by environment-dependent resolution: with a readable profile path it
    # resolves to the explicit profile, otherwise to the legacy heuristic.
    if requested in ("legacy", "builtin", "profile"):
        expected = {"legacy": "legacy-heuristic", "builtin": "builtin-calibration",
                    "profile": "explicit-profile"}[requested]
        if resolved != expected:
            problems.append(f"requested_adaptive_cost_mode={requested} must resolve to {expected}, got {resolved!r}")
    elif requested == "auto" and resolved not in ("legacy-heuristic", "explicit-profile"):
        problems.append(f"requested_adaptive_cost_mode=auto cannot resolve to {resolved!r}; "
                        "auto is either the legacy heuristic or a supplied explicit profile")
    return problems


def verify_worker_runtime_role(records, config: dict, metadata: dict) -> list:
    """Every physical rank must have reported its own loaded role/geometry.

    The other gates prove the argv and the artifact manifests. They do not prove
    a worker actually loaded the shard family its topology needs: a library that
    advertises `spark_tp6` in a manifest but never initializes role 7 would pass
    argv checks and then fail (or silently mis-serve) at the first request.
    `worker_runtime_role` records the readiness line per rank.

    Required for every corpus revision at or above `WORKER_RUNTIME_ROLE_SINCE`.
    A recorded field on an older revision is validated too; an older revision
    that never recorded it keeps its already-accepted result intact.
    """
    revision = corpus_revision(records)["version"]
    if "worker_runtime_role" not in metadata and (
        not isinstance(revision, int) or revision < WORKER_RUNTIME_ROLE_SINCE
    ):
        return []
    world = world_expected = config["world"]
    tp = config["spark_tp"]
    expected_role, expected_intermediate = EXPLICIT_TP_ROLE_INTERMEDIATE[tp]
    reported = metadata.get("worker_runtime_role")
    if not isinstance(reported, list) or len(reported) != world:
        return [f"worker_runtime_role must list every one of the {world} ranks, got "
                f"{reported if not isinstance(reported, list) else len(reported)}"]
    problems = []
    seen = set()
    for entry in reported:
        if not isinstance(entry, dict):
            problems.append("worker_runtime_role entries must be objects")
            continue
        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world or rank in seen:
            problems.append(f"worker_runtime_role has an invalid or duplicate rank {rank!r}")
            continue
        seen.add(rank)
        if entry.get("role") != expected_role:
            problems.append(f"worker rank {rank} reported role {entry.get('role')!r}; "
                            f"TP{tp}EP{config['spark_ep']} requires native role {expected_role}")
        if entry.get("intermediate") != expected_intermediate:
            problems.append(f"worker rank {rank} reported intermediate {entry.get('intermediate')!r}; "
                            f"TP{tp} requires {expected_intermediate}")
        if entry.get("world") != world:
            problems.append(f"worker rank {rank} reported world {entry.get('world')!r} != {world}")
        source_log = str(entry.get("source_log", "")).strip()
        if not source_log:
            problems.append(f"worker rank {rank} needs a source_log")
            continue
        # From v5 a nonempty path is not enough: the captured log must hash to the
        # recorded digest, so a hand-typed role/intermediate cannot claim a log it
        # never produced.
        if revision >= WORKER_RUNTIME_ROLE_SINCE:
            declared = entry.get("source_log_sha256")
            if not isinstance(declared, str) or len(declared) != 64:
                problems.append(f"worker rank {rank} needs source_log_sha256 (64 hex) for the captured log")
            else:
                path = Path(source_log)
                if not path.is_file():
                    problems.append(f"worker rank {rank} source_log does not exist: {source_log}")
                else:
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual != declared:
                        problems.append(f"worker rank {rank} source_log_sha256 does not match the "
                                        f"captured log (declared {declared[:12]}…, actual {actual[:12]}…)")
                    else:
                        text = path.read_text(errors="replace")
                        observed = worker_role_in_log(text, rank)
                        if observed is None:
                            problems.append(f"worker rank {rank} captured log has no role/intermediate "
                                            "startup line")
                        else:
                            role, intermediate, world = observed
                            if role != expected_role:
                                problems.append(f"worker rank {rank} log reports role {role}; TP{tp} requires {expected_role}")
                            if intermediate != expected_intermediate:
                                problems.append(f"worker rank {rank} log reports intermediate {intermediate}; requires {expected_intermediate}")
                            if world != world_expected:
                                problems.append(f"worker rank {rank} log reports world {world}; requires {world_expected}")
    if seen != set(range(world)):
        problems.append(f"worker_runtime_role ranks {sorted(seen)} do not cover 0..{world - 1}")
    return problems


def cmd_run(args) -> int:
    records = load_jsonl(args.corpus)
    schema = validate_six(records, load_jsonl(V3_CORPUS), args.corpus)
    if not schema["passed"]:
        raise SystemExit("six corpus invalid: " + "; ".join(schema["errors"]))
    config = next((c for c in records if c.get("kind") == "config" and c.get("id") == args.arm), None)
    if config is None:
        raise SystemExit(f"--arm {args.arm!r} is not a config in {args.corpus}")
    if args.startup_metadata is None:
        raise SystemExit("six-rank run requires --startup-metadata with actual values")
    metadata = json.loads(Path(args.startup_metadata).read_text())
    problems = verify_arm_metadata(records, config, metadata)
    if problems:
        raise SystemExit("startup metadata rejected before any request: " + "; ".join(problems))
    run_corpus = run_corpus_path(Path(args.output), records)
    namespace = SimpleNamespace(arm=args.arm, corpus=run_corpus, base_url=args.base_url, output=args.output,
                                startup_metadata=args.startup_metadata, case=args.case,
                                concurrency=args.concurrency, repeats=args.repeats)
    code = BASE["cmd_run"](namespace)
    report = json.loads(Path(args.output).read_text())
    revision = corpus_revision(records)
    report.update(arm_config=config, corpus=str(args.corpus), corpus_sha256=sha256(args.corpus),
                  run_corpus=str(run_corpus), run_corpus_sha256=sha256(run_corpus),
                  frozen_runner_sha256=FROZEN_RUNNER_SHA256, base_corpus_sha256=sha256(V3_CORPUS),
                  corpus_revision=revision, six_runner="scripts/qualify-ds41-tp-ep-e2e-six.py")
    BASE["write_json"](args.output, report)
    return code


def cmd_compare(args) -> int:
    right = json.loads(Path(args.candidate).read_text())
    left = json.loads(Path(args.control).read_text()) if args.control else None
    records = load_jsonl(args.corpus)
    schema = validate_six(records, load_jsonl(V3_CORPUS), args.corpus)
    configs = {c["id"]: c for c in by_kind(records, "config")}
    if not schema["passed"]:
        report = dict(gate_result="corpus_invalid", passed=False, tier_a_failures=schema["errors"],
                      coverage=dict(problems=[]), identity=dict(problems=[], mismatched_fields=[]),
                      text=dict(exact_matches=0, divergent_requests=0, divergences=[]))
        report["control"] = str(args.control) if args.control else None
        report["candidate"] = str(args.candidate)
        report["six"] = dict(schema=corpus_revision(records)["schema"],
                             corpus_revision=corpus_revision(records), corpus=str(args.corpus),
                             corpus_sha256=sha256(args.corpus), base_corpus_sha256=sha256(V3_CORPUS),
                             frozen_runner_sha256=FROZEN_RUNNER_SHA256, comparison=None)
        BASE["write_json"](args.output, report)
        print(json.dumps(dict(gate_result=report["gate_result"], passed=False,
                              tier_a_failures=schema["errors"]), indent=2))
        return 1
    comparison = comparison_report(left, right, configs, by_kind(records, "comparison"),
                                   by_kind(records, "forbidden_comparison"))
    scope = arm_scope_failures(records, "candidate", right, args.corpus)
    if left is not None:
        scope += arm_scope_failures(records, "control", left, args.corpus)
    for label, arm in (("candidate", right), ("control", left)):
        if arm is None:
            continue
        config = configs.get(arm.get("arm"))
        if config is None:
            scope.append(f"{label}: arm id {arm.get('arm')!r} is not a named config in {args.corpus}")
            continue
        scope += [f"{label}: {problem}"
                  for problem in verify_arm_metadata(records, config, arm.get("startup_metadata") or {})]
    if comparison.get("class_") == "standalone":
        self_report = BASE["compare_arms"](right, right)
        quality_failures = list(scope) + list(self_report.get("tier_a_failures", []))
        memory = comparison.get("memory") or dict(ok=False, arithmetic_ok=False, reviewed=False,
                                                  reasons=["memory evidence not evaluated"])
        if quality_failures:
            gate, passed = "FAIL", False
        elif not memory.get("arithmetic_ok"):
            gate, passed = "standalone_quality_pass_memory_review_required", False
        elif not memory.get("reviewed"):
            gate, passed = "standalone_memory_schema_ok_review_required", False
        else:
            gate, passed = "standalone_memory_qualified", True
        report = dict(gate_result=gate, passed=passed, tier_a_failures=quality_failures,
                      coverage=self_report.get("coverage"), identity=self_report.get("identity"),
                      memory=memory, text=self_report.get("text"))
    elif left is None or not comparison.get("allowed"):
        report = dict(gate_result="comparison_not_allowed", passed=False,
                      tier_a_failures=scope + list(comparison.get("reasons", [])),
                      coverage=dict(problems=[]), identity=dict(problems=[], mismatched_fields=[]),
                      text=dict(exact_matches=0, divergent_requests=0, divergences=[]))
    else:
        base_report = BASE["compare_arms"](left, right)
        failures = scope + list(comparison.get("reasons", []))
        report = dict(base_report)
        report["tier_a_failures"] = list(base_report.get("tier_a_failures", [])) + failures
        report["passed"] = bool(base_report.get("passed")) and bool(comparison["passed"]) and not failures
        if failures:
            report["gate_result"] = "FAIL"
    report["control"] = str(args.control) if args.control else None
    report["candidate"] = str(args.candidate)
    revision = corpus_revision(records)
    report["six"] = dict(schema=revision["schema"], corpus_revision=revision, corpus=str(args.corpus),
                         corpus_sha256=sha256(args.corpus),
                         base_corpus_sha256=sha256(V3_CORPUS), frozen_runner_sha256=FROZEN_RUNNER_SHA256,
                         comparison=comparison)
    BASE["write_json"](args.output, report)
    print(json.dumps(dict(gate_result=report["gate_result"], passed=report["passed"],
                          comparison=comparison.get("class_"), reasons=comparison.get("reasons"),
                          tier_a_failures=report.get("tier_a_failures")), indent=2))
    return 0 if report["passed"] else 1


def _valid_memory(config_id: str, configs: dict) -> dict:
    config = configs[config_id]
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
        headroom = max(config["coordinator_runtime_headroom_bytes"], COORDINATOR_RUNTIME_HEADROOM_MIN_BYTES)
        evidence["coordinator"] = dict(runtime_headroom_bytes=headroom,
                                       device_free_bytes=headroom + (1 << 30),
                                       source_log="logs/coordinator.log")
    return evidence


def cmd_selftest(args) -> int:
    records = load_jsonl(V4_CORPUS)
    frozen = load_jsonl(V3_CORPUS)
    report = validate_six(records, frozen)
    assert report["passed"], report["errors"]
    configs = {c["id"]: c for c in by_kind(records, "config")}
    comparisons = by_kind(records, "comparison")
    forbidden = by_kind(records, "forbidden_comparison")
    assert comparison_report({"arm": "2rtx6-tp2ep3"}, {"arm": "2rtx6-tp3ep2"},
                             configs, comparisons, forbidden)["class_"] == "paired"
    blocked = comparison_report({"arm": "1rtx6-tp3ep2"}, {"arm": "2rtx6-tp2ep3"}, configs, comparisons, forbidden)
    assert not blocked["allowed"] and "unequal rtx_count" in blocked["reasons"][0], blocked
    assert verify_memory_evidence({}, configs["1rtx6-tp3ep2"])["ok"] is False
    assert verify_memory_evidence({"memory_evidence": _valid_memory("1rtx6-tp3ep2", configs)},
                                  configs["1rtx6-tp3ep2"])["ok"] is True
    one_rtx = {"startup_metadata": {"resolved_rtx_expert_layers": 0, "spark_first_layer": 0, "spark_tp": 3,
                                    "spark_ep": 2, "spark_count": 6, "rtx_gpus": 1, "requested_rtx_expert_layers": "auto",
                                    "coordinator_argv": ["serve-native", "--rtx-gpus", "1"]}}
    assert any("--rtx-expert-layers 0 explicitly" in problem
               for problem in verify_arm_metadata(records, configs["1rtx6-tp3ep2"], one_rtx["startup_metadata"]))
    # v5 adds the pure unreplicated TP6EP1 arms without changing the accepted v4
    # revision or its recorded sha256.
    v5 = load_jsonl(V5_CORPUS)
    v5_report = validate_six(v5, frozen, V5_CORPUS)
    assert v5_report["passed"], v5_report["errors"]
    v5_configs = {c["id"]: c for c in by_kind(v5, "config")}
    assert set(v5_configs) == set(EXPECTED_CONFIG_TUPLES), sorted(v5_configs)
    for cid in ("2rtx6-tp6ep1", "1rtx6-tp6ep1"):
        config = v5_configs[cid]
        assert (config["spark_tp"], config["spark_ep"]) == (6, 1)
        assert config["spark_layer_bytes"] == SPARK_LAYER_BYTES[6] == 1203240960
        assert config["expected_spark_resident_bytes"] == 1203240960 * config["remote_layers"]
    assert sha256(V4_CORPUS) == EXPECTED_V4_CORPUS_SHA256, "the accepted v4 corpus must not change"
    v5_comparisons = by_kind(v5, "comparison")
    v5_forbidden = by_kind(v5, "forbidden_comparison")
    pure = comparison_report({"arm": "2rtx6-tp6ep1"}, {"arm": "2rtx6-tp3ep2"},
                             v5_configs, v5_comparisons, v5_forbidden)
    assert pure["class_"] == "paired" and pure["allowed"], pure
    cross = comparison_report({"arm": "1rtx6-tp6ep1"}, {"arm": "2rtx6-tp6ep1"},
                              v5_configs, v5_comparisons, v5_forbidden)
    assert not cross["allowed"] and "forbidden" in " ".join(cross["reasons"]).lower(), cross
    assert verify_memory_evidence({}, v5_configs["1rtx6-tp6ep1"])["ok"] is False
    print(json.dumps(dict(selftest="PASS", corpus=report["corpus_sha256"],
                          corpus_v5=v5_report["corpus_sha256"], runner=FROZEN_RUNNER_SHA256), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="CPU-only v4 six-rank corpus schema check")
    p.add_argument("--corpus", type=Path, default=V4_CORPUS)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("selftest", help="CPU-only schema + comparison-gate smoke")
    p.set_defaults(func=cmd_selftest)
    p = sub.add_parser("run", help="drive one six-rank arm (reuses the frozen runner loop)")
    p.add_argument("--corpus", type=Path, default=V4_CORPUS)
    p.add_argument("--arm", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--startup-metadata", type=Path)
    p.add_argument("--case", action="append")
    p.add_argument("--concurrency", type=int, nargs="+")
    p.add_argument("--repeats", type=int)
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("compare", help="CPU-only six-rank comparison gate")
    p.add_argument("--corpus", type=Path, default=V4_CORPUS)
    p.add_argument("--control", type=Path, help="paired control arm; omit for a standalone candidate")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
