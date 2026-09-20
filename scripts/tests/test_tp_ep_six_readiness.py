"""CPU-only readiness tests for the six-node execution package.

Parses the staged configs and startup-metadata templates and dry-runs them
through the frozen six-rank gates. No GPU, service, HTTP, remote access or
actual execution.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIX = ROOT / "scripts" / "qualify-ds41-tp-ep-e2e-six.py"
FIX = ROOT / "scripts" / "fixtures" / "tp-ep-six"
DOC = ROOT / "docs" / "tp-ep-six-readiness.md"
ARMS = ("1rtx6-tp3ep2", "2rtx6-tp2ep3", "2rtx6-tp3ep2")


def _load():
    spec = importlib.util.spec_from_file_location("ds41rt_tp_ep_six_readiness", SIX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


six = _load()
RECORDS = six.load_jsonl(six.V4_CORPUS)
CONFIGS = {config["id"]: config for config in six.by_kind(RECORDS, "config")}

FILL = {
    "REQUIRED_SOURCE_COMMIT": "0" * 40,
    "REQUIRED_SOURCE_ARTIFACT": "coordinator=sha256:aaaa;expert=sha256:bbbb",
    "REQUIRED_DAEMON_SNAPSHOT": "daemon-snapshot",
    "REQUIRED_NATIVE_SNAPSHOT": "native-snapshot",
    "REQUIRED_SPARKINFER_REVISION": "sparkinfer-rev",
    "REQUIRED_MANIFEST_SPARK": "h-spark",
    "REQUIRED_MANIFEST_SPARK_TP2": "h-tp2",
    "REQUIRED_MANIFEST_SPARK_TP3": "h-tp3",
    "REQUIRED_DEVICE_BUDGET": 109119320064,
    "REQUIRED_RELEASE_IDENTITY": "fp-six",
    "REQUIRED_MEM_TOTAL": 130661208064,
    "REQUIRED_MEM_AVAILABLE": 125000000000,
    "REQUIRED_WORKSPACE": 5000000000,
    "REQUIRED_RING": 4000000000,
    "REQUIRED_COORD_DEVICE_FREE": 3000000000,
    "REQUIRED_COORD_SOURCE_LOG": "logs/coordinator.log",
}


def _template(arm: str) -> dict:
    return json.loads((FIX / f"startup-{arm}.template.json").read_text())


def _config_values(arm: str) -> dict:
    values = {}
    for raw in (FIX / f"site-{arm}.config").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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


def _filled(arm: str) -> dict:
    filled = _fill(_template(arm), CONFIGS[arm])
    if "memory_evidence" in filled:
        filled["memory_evidence"]["logs_independently_reviewed"] = True
    return filled


# 1. Every template parses, names its arm, and is explicitly incomplete. The arm
#    gate alone cannot see placeholder strings (argv and metadata agree on them);
#    the standalone memory gate is what rejects unfilled values numerically.
def test_templates_are_explicitly_incomplete() -> None:
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


# 2. Filled templates pass the six-rank arm gates (standalone also needs memory).
def test_filled_templates_pass_the_six_gates() -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        filled = _filled(arm)
        assert six.verify_arm_metadata(RECORDS, config, filled) == [], arm
        if config["memory_qualification"]:
            memory = six.verify_memory_evidence(filled, config)
            assert memory["ok"] is True, memory["reasons"]
        else:
            assert "memory_evidence" not in filled


# 3. Dry-run the coordinator/worker argv: explicit layers, rail binding, KV.
def test_argv_dry_run_encodes_the_contract() -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        filled = _filled(arm)
        parsed = six.BASE["parse_argv"](filled["coordinator_argv"])[0]
        assert parsed["--rtx-expert-layers"] == str(config["requested_rtx_expert_layers"])
        assert parsed["--rtx-gpus"] == str(config["rtx_count"])
        assert ("--placement-directory" in parsed) is bool(config["require_placement_directory"])
        assert parsed["--concurrency"] == "16"
        assert parsed["--prefill-batch-tokens"] == "2048"
        peers = parsed["--peers"]
        assert "10.55.0.5:29441" in peers and "10.55.0.12:29441" in peers
        assert "10.55.0.6:29441" not in peers, "moa must not be addressed on its fabric-A lane"
        assert parsed["--listen"] == "127.0.0.1:18000", "isolated candidate coordinator port"
        if config["kv_class"] == "single-auto":
            assert "--kv-pool-size" not in parsed
            assert filled["kv_pool"] == "auto"
        else:
            assert parsed["--kv-pool-size"] == "5037542400"
            assert filled["kv_pool"] == "5037542400"
        workers = filled["worker_argv"]
        assert len(workers) == config["world"] == 6
        ranks = sorted(six.BASE["parse_argv"](argv)[0]["--rank"] for argv in workers)
        assert ranks == [str(index) for index in range(6)]
        for argv in workers:
            parsed_worker = six.BASE["parse_argv"](argv)[0]
            assert parsed_worker["--world"] == "6"
            assert parsed_worker["--capacity"] == "4096"
            assert parsed_worker["--spark-tp"] == str(config["spark_tp"])
            assert parsed_worker["--spark-ep"] == str(config["spark_ep"])
            assert parsed_worker["--first-layer"] == str(config["expected_spark_first_layer"])
            assert parsed_worker["--listen"] == "0.0.0.0:29441", "isolated candidate expert port"
            assert parsed_worker["--device-budget-bytes"] == "109119320064", "global minimum-of-six budget"


# 4. Staged configs encode the same arm contract as the corpus.
def test_staged_configs_match_the_corpus() -> None:
    for arm in ARMS:
        config = CONFIGS[arm]
        values = _config_values(arm)
        assert values["SPARK_COUNT"] == "6"
        assert values["SPARK_TP"] == str(config["spark_tp"])
        assert values["SPARK_EP"] == str(config["spark_ep"])
        assert values["RTX_GPUS"] == str(config["rtx_count"])
        assert values["RTX_EXPERT_LAYERS"] == str(config["requested_rtx_expert_layers"])
        assert values["SPARK_5_LANE_A"] == "10.55.0.12"
        assert values["SPARK_4_LANE_A"] == "10.55.0.5"
        assert values["DSPARK"] == "on"
        assert values["CONCURRENCY"] == "16"
        assert values["SPARK_DEVICE_BUDGET_BYTES"] == "109119320064"
        if config["kv_class"] == "single-auto":
            assert "KV_POOL_SIZE" not in values
        else:
            assert values["KV_POOL_SIZE"] == "5037542400"
        assert "REQUIRED" in values["COORDINATOR_DOCKER_INFERENCE"]
        assert "REQUIRED" in values["SPARK_EXPERT_DOCKER_INFERENCE"]


# 5. The runbook pins the frozen identity, the 372 scope, and the rail facts.
def test_runbook_pins_frozen_identity_and_hardware_facts() -> None:
    text = DOC.read_text()
    assert six.sha256(six.V4_CORPUS) in text
    assert six.FROZEN_RUNNER_SHA256 in text
    assert six.sha256(six.V3_CORPUS) in text
    for arm in ARMS:
        assert arm in text
    assert "372" in text
    assert "gid_index=5" in text
    assert "10.55.0.12" in text
    assert "20 GiB" in text
    assert "96,259,276,800" in text
    assert "REQUIRED" in text
    assert "PROVISIONAL" in text
    assert "109119320064" in text
    assert "dab2b979" in text
    assert "d453812c" in text
    assert "18000" in text and "29441" in text
    assert "800 MiB" in text
    assert "six-node-prep" in text
    assert "not accepted" in text
    assert "forbidden comparison" in text
