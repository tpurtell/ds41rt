"""CPU-only readiness tests for the pure unreplicated TP6EP1 arms.

Extends the six-rank readiness checks to the v5 corpus and its two TP6 arms
(`2rtx6-tp6ep1`, `1rtx6-tp6ep1`) without touching the accepted v4 readiness
module or its recorded corpus sha256. Reuses the same frozen gates and the same
template/config fill helpers, so the TP6 arm contract is exercised by the exact
validators the other arms use.

Pure TP6EP1 is one unreplicated group of six ranks: each rank holds a disjoint
2304/6 = 384 intermediate slice of EVERY routed expert, so no expert is
duplicated and the compact reducer sums six physical rank planes.

No GPU, service, HTTP, remote access or actual execution.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIX = ROOT / "scripts" / "qualify-ds41-tp-ep-e2e-six.py"
FIX = ROOT / "scripts" / "fixtures" / "tp-ep-six"
ARMS = ("2rtx6-tp6ep1", "1rtx6-tp6ep1")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


six = _load("ds41rt_tp_ep_six_qualify_tp6", SIX)
_readiness = _load("ds41rt_tp_ep_six_readiness_tp6_base", ROOT / "scripts" / "tests" / "test_tp_ep_six_readiness.py")
RECORDS = six.load_jsonl(six.V5_CORPUS)
CONFIGS = {config["id"]: config for config in six.by_kind(RECORDS, "config")}
# The shared fixture fill covers the replicated arms; a TP6 standalone declares
# the same 20 GiB Spark reserve, which the shared map does not name.
FILL = dict(_readiness.FILL, REQUIRED_SPARK_RESERVE=21474836480,
            REQUIRED_WORKER_READY_LOG="logs/rank.log",
            REQUIRED_WORKER_READY_LOG_SHA256="0" * 64,
            REQUIRED_ADAPTIVE_COST_MODE="auto",
            REQUIRED_RESOLVED_COST_MODEL="legacy-heuristic",
            REQUIRED_COST_MODEL_LOG="logs/coordinator.log",
            REQUIRED_COST_MODEL_LOG_SHA256="0" * 64)
# The v5 corpus requires a per-rank runtime role record; each worker's readiness
# line must report the role and intermediate it actually loaded.
# Each rank points at a real captured log file whose bytes are hashed into the
# metadata, so a hand-typed role/intermediate cannot claim a log it never made.
def _worker_log(tmp: Path, rank: int, role: int, intermediate: int, world: int = 6) -> Path:
    path = tmp / f"rank-{rank}-r{role}.log"
    # The daemon inserts ANSI CSI codes between a field name and its '='; the
    # parser must strip them, so the fixture writes the real noisy shape.
    path.write_text(
        "INFO ds41rt: \x1b[2mnative local RoCE expert worker ready\x1b[0m "
        f"rank={rank} world={world} \x1b[3mrole\x1b[0m={role} "
        f"\x1b[3mintermediate\x1b[0m={intermediate} first_layer=5\n"
    )
    return path


# Native role and logical intermediate per explicit TP degree, matching the
# qualifier's own table so a fixture cannot drift from the gate.
ROLE_INTERMEDIATE = {3: (6, 768), 6: (7, 384)}


def _worker_runtime_role(tmp: Path, tp: int) -> list:
    import hashlib
    role, intermediate = ROLE_INTERMEDIATE[tp]
    entries = []
    for rank in range(6):
        path = _worker_log(tmp, rank, role, intermediate)
        entries.append({
            "rank": rank, "role": role, "intermediate": intermediate, "world": 6,
            "source_log": str(path),
            "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return entries


def _template(arm: str) -> dict:
    return json.loads((FIX / f"startup-{arm}.template.json").read_text())


def _fill(value, config: dict, as_token: bool = False):
    if isinstance(value, dict):
        return {key: _fill(item, config) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, config, as_token=True) for item in value]
    if isinstance(value, str):
        if value == "REQUIRED_RESIDENT":
            mapped = config["expected_spark_resident_bytes"]
        elif value == "REQUIRED_SOURCE_LOG":
            mapped = "logs/rank.log"
        else:
            mapped = FILL.get(value, value)
        return str(mapped) if as_token and not isinstance(mapped, str) else mapped
    return value


def _cost_model_evidence(tmp: Path, resolved: str = "legacy-heuristic") -> dict:
    import hashlib
    path = tmp / "coordinator.log"
    path.write_text(f"INFO ds41rt: adaptive verification costs disabled; cost_model={resolved} mode=Auto\n")
    return {
        "requested_adaptive_cost_mode": "auto",
        "resolved_cost_model": resolved,
        "cost_model_source_log": str(path),
        "cost_model_source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _filled(arm: str, tmp: Path | None = None) -> dict:
    filled = _fill(_template(arm), CONFIGS[arm])
    if "memory_evidence" in filled:
        filled["memory_evidence"]["logs_independently_reviewed"] = True
    # Every rank reports its own loaded role, bound to a hashed captured log; a
    # manifest or a self-declared number alone is not proof.
    if tmp is not None:
        filled["worker_runtime_role"] = _worker_runtime_role(tmp, CONFIGS[arm]["spark_tp"])
        filled.update(_cost_model_evidence(tmp))
    return filled


def test_the_v5_corpus_is_the_only_tp6_bearing_revision() -> None:
    v5 = six.validate_six(RECORDS, six.load_jsonl(six.V3_CORPUS), six.V5_CORPUS)
    assert v5["passed"], v5["errors"]
    v4 = six.validate_six(six.load_jsonl(six.V4_CORPUS), six.load_jsonl(six.V3_CORPUS), six.V4_CORPUS)
    assert v4["passed"], v4["errors"]
    assert six.sha256(six.V4_CORPUS) == six.EXPECTED_V4_CORPUS_SHA256
    assert set(ARMS).issubset(CONFIGS)
    # v5 is a strict superset of the accepted v4 arm set: it adds the TP6 arms
    # and removes nothing, so no accepted v4 result is invalidated.
    v4_configs = {c["id"] for c in six.by_kind(six.load_jsonl(six.V4_CORPUS), "config")}
    assert v4_configs < set(CONFIGS)
    assert set(CONFIGS) - v4_configs == set(ARMS)
    # The report identity comes from the corpus revision, never a hardcoded v4.
    assert six.corpus_revision(RECORDS)["version"] == 5
    assert six.corpus_revision(RECORDS)["schema"] == "tp-ep-e2e-v5-six"
    assert six.corpus_revision(six.load_jsonl(six.V4_CORPUS))["schema"] == "tp-ep-e2e-v4-six"
    # The v5 scope names the pure TP6 family explicitly.
    scope = next(r for r in RECORDS if r.get("kind") == "future_scope")
    assert "TP6EP1" in json.dumps(scope)


def test_a_worker_that_never_loaded_role_seven_is_rejected(tmp_path) -> None:
    """An artifact manifest is not proof that a running rank loaded the role."""
    for arm in ARMS:
        config = CONFIGS[arm]
        filled = _filled(arm, tmp_path)
        assert six.verify_worker_runtime_role(RECORDS, config, filled) == []
        # Missing evidence, wrong role, wrong intermediate and a missing rank all
        # fail closed.
        for mutation, needle in (
            (lambda meta: meta.pop("worker_runtime_role"), "worker_runtime_role must list"),
            (lambda meta: meta["worker_runtime_role"][0].update(role=1), "requires native role 7"),
            (lambda meta: meta["worker_runtime_role"][0].update(intermediate=576), "requires 384"),
            (lambda meta: meta["worker_runtime_role"].pop(), "must list every one of the 6 ranks"),
        ):
            broken = json.loads(json.dumps(filled))
            mutation(broken)
            problems = six.verify_worker_runtime_role(RECORDS, config, broken)
            assert any(needle in problem for problem in problems), (arm, needle, problems)


def test_cost_model_evidence_must_match_the_captured_log(tmp_path) -> None:
    """The declared model must be the one the log actually reports."""
    config = CONFIGS["2rtx6-tp6ep1"]
    filled = _filled("2rtx6-tp6ep1", tmp_path)
    assert six.verify_cost_model_evidence(RECORDS, filled) == []
    for mutation, needle in (
        (lambda m: m.update(resolved_cost_model="builtin-calibration"), "captured log reports"),
        (lambda m: m.pop("cost_model_source_log"), "cost_model_source_log is required"),
        (lambda m: m.update(cost_model_source_log_sha256="0" * 64), "does not match"),
        (lambda m: m.update(cost_model_source_log="/nonexistent/coordinator.log"), "does not exist"),
        (lambda m: m.update(requested_adaptive_cost_mode="builtin"), "must resolve to builtin-calibration"),
        (lambda m: m.update(resolved_cost_model="made-up"), "must be one of"),
    ):
        broken = json.loads(json.dumps(filled))
        mutation(broken)
        problems = six.verify_cost_model_evidence(RECORDS, broken)
        assert any(needle in problem for problem in problems), (needle, problems)
    # A pinned builtin request with the matching resolved label is accepted.
    builtin = _filled("2rtx6-tp6ep1", tmp_path)
    builtin.update(_cost_model_evidence(tmp_path, "builtin-calibration"))
    builtin["requested_adaptive_cost_mode"] = "builtin"
    assert six.verify_cost_model_evidence(RECORDS, builtin) == []
    assert config["spark_tp"] == 6


def test_v5_control_templates_carry_the_new_evidence(tmp_path) -> None:
    """The v5 revision also covers the replicated control arms.

    The v5-specific templates add the runtime-role and cost-model evidence; the
    original v4 templates are untouched so the accepted hybrid campaign record
    still reproduces.
    """
    for arm, tp in (("2rtx6-tp3ep2", 3), ("1rtx6-tp3ep2", 3)):
        config = CONFIGS[arm]
        v5 = json.loads((FIX / f"startup-{arm}-v5.template.json").read_text())
        v4 = json.loads((FIX / f"startup-{arm}.template.json").read_text())
        assert "worker_runtime_role" not in v4, "the accepted v4 template must not change"
        assert "worker_runtime_role" in v5
        filled = _fill(v5, config)
        filled["worker_runtime_role"] = _worker_runtime_role(tmp_path, tp)
        filled.update(_cost_model_evidence(tmp_path))
        if "memory_evidence" in filled:
            filled["memory_evidence"]["logs_independently_reviewed"] = True
        assert six.verify_arm_metadata(RECORDS, config, filled) == [], arm
        assert six.verify_cost_model_evidence(RECORDS, filled) == []


def test_worker_role_parser_rejects_rank_prefix_and_missing_message(tmp_path) -> None:
    """`rank=1` must not satisfy rank 10, and a non-ready line must not parse."""
    import hashlib
    path = tmp_path / "prefix.log"
    path.write_text(
        "INFO ds41rt: native local RoCE expert worker ready rank=10 world=6 role=7 intermediate=384\n"
    )
    assert six.worker_role_in_log(path.read_text(), 1) is None
    assert six.worker_role_in_log(path.read_text(), 10) == (7, 384, 6)
    # A readiness echo without the exact message, or without the geometry, is not
    # evidence.
    assert six.worker_role_in_log("worker ready rank=1 world=6 role=7 intermediate=384\n", 1) is None
    assert six.worker_role_in_log(
        "native local RoCE expert worker ready rank=1 world=6 first_layer=5\n", 1) is None
    config = CONFIGS["2rtx6-tp6ep1"]
    entry = _worker_runtime_role(tmp_path, 6)[0]
    entry["role"] = 1  # hand-typed number that disagrees with the captured line
    bad = dict(_filled("2rtx6-tp6ep1", tmp_path))
    bad["worker_runtime_role"] = [entry] + _worker_runtime_role(tmp_path, 6)[1:]
    problems = six.verify_worker_runtime_role(RECORDS, config, bad)
    assert any("requires native role 7" in problem for problem in problems), problems
    # The captured log still reports 7; it is the declaration that is rejected.
    assert six.worker_role_in_log(Path(entry["source_log"]).read_text(), 0) == (7, 384, 6)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_log_sha256"]


def test_v4_arms_without_runtime_role_evidence_still_validate() -> None:
    """The new gate is versioned: accepted v4 results are not retro-invalidated."""
    v4_records = six.load_jsonl(six.V4_CORPUS)
    v4_config = next(c for c in six.by_kind(v4_records, "config") if c["id"] == "2rtx6-tp3ep2")
    metadata = {"resolved_rtx_expert_layers": 20, "spark_first_layer": 20, "spark_tp": 3,
                "spark_ep": 2, "spark_count": 6, "rtx_gpus": 2,
                "requested_rtx_expert_layers": 20,
                "coordinator_argv": ["serve-native", "--rtx-gpus", "2"]}
    assert six.verify_worker_runtime_role(v4_records, v4_config, metadata) == []


def test_tp6_templates_are_explicitly_incomplete() -> None:
    for arm in ARMS:
        template = _template(arm)
        assert template["_arm"] == arm
        assert template["_status"].startswith("TEMPLATE")
        assert "REQUIRED_" in json.dumps(template), f"{arm} has no REQUIRED markers"
        config = CONFIGS[arm]
        if config["memory_qualification"]:
            assert six.verify_memory_evidence(template, config)["ok"] is False
        else:
            assert "memory_evidence" not in template


def test_filled_tp6_templates_pass_the_six_gates(tmp_path) -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        filled = _filled(arm, tmp_path)
        assert six.verify_arm_metadata(RECORDS, config, filled) == [], arm
        if config["memory_qualification"]:
            memory = six.verify_memory_evidence(filled, config)
            assert memory["ok"] is True, memory["reasons"]
        else:
            assert "memory_evidence" not in filled


def test_tp6_argv_dry_run_encodes_the_pure_topology(tmp_path) -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        filled = _filled(arm, tmp_path)
        parsed = six.BASE["parse_argv"](filled["coordinator_argv"])[0]
        assert parsed["--spark-tp"] == "6"
        assert parsed["--spark-ep"] == "1"
        assert parsed["--rtx-expert-layers"] == str(config["requested_rtx_expert_layers"])
        assert parsed["--rtx-gpus"] == str(config["rtx_count"])
        assert ("--placement-directory" in parsed) is bool(config["require_placement_directory"])
        peers = parsed["--peers"]
        assert peers.count(":29441") == 6, peers
        assert "10.55.0.6:29441" not in peers, "the sixth rank is addressed on fabric A"
        if config["kv_class"] == "single-auto":
            assert "--kv-pool-size" not in parsed
            assert filled["kv_pool"] == "auto"
        else:
            assert parsed["--kv-pool-size"] == "5037542400"
        workers = filled["worker_argv"]
        assert len(workers) == config["world"] == 6
        for argv in workers:
            parsed_worker = six.BASE["parse_argv"](argv)[0]
            assert parsed_worker["--world"] == "6"
            assert parsed_worker["--spark-tp"] == "6"
            assert parsed_worker["--spark-ep"] == "1"
            assert parsed_worker["--first-layer"] == str(config["expected_spark_first_layer"])
        ranks = sorted(six.BASE["parse_argv"](argv)[0]["--rank"] for argv in workers)
        assert ranks == [str(index) for index in range(6)]


def test_staged_tp6_configs_match_the_corpus() -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        values = {}
        for raw in (FIX / f"site-{arm}.config").read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        assert values["SPARK_COUNT"] == "6"
        assert values["SPARK_TP"] == "6"
        assert values["SPARK_EP"] == "1"
        assert values["RTX_GPUS"] == str(config["rtx_count"])
        assert values["RTX_EXPERT_LAYERS"] == str(config["requested_rtx_expert_layers"])
        assert values["DSPARK"] == "on"
        assert values["CONCURRENCY"] == "16"
        assert values["SPARK_DEVICE_BUDGET_BYTES"] == "109119320064"
        if config["kv_class"] == "single-auto":
            assert "KV_POOL_SIZE" not in values
        else:
            assert values["KV_POOL_SIZE"] == "5037542400"


def test_the_example_overlay_matches_the_pure_tp6_contract() -> None:
    text = (ROOT / "examples" / "configs" / "tp6ep1-native.config").read_text()
    assert "SPARK_TP=6" in text
    assert "SPARK_EP=1" in text
    assert "SPARK_COUNT=6" in text
    assert "NOT RELEASE-QUALIFIED" in text
    assert "No expert is duplicated" in text
    assert "1,203,240,960" in text
