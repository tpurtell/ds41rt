#!/usr/bin/env python3
"""CPU-only validation of the v9 TP6xEP1 candidate configs.

Loads each candidate through the real `scripts/release-common.sh` config loader
(no Docker, SSH, GPU or network) and asserts the resolved geometry/dials match
the published-v8 official TP4 baseline, so the only intended variable is the
Spark topology.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMMON = REPO / "scripts" / "release-common.sh"
CONFIGS = {
    1: REPO / "scripts" / "bench" / "candidate-tp6-1x-official-match.config",
    2: REPO / "scripts" / "bench" / "candidate-tp6-2x-official-match.config",
}
BASE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
KEY = "SPARK_5_LANE_A"

PROBE = r"""
set -euo pipefail
source "$1"
release_load_config "$2"
release_validate_spark_topology
for v in MODEL_ID MODEL_REVISION EXPERT_FORMAT SPARKINFER_EXL3 RTX_GPUS RTX_EXPERT_LAYERS \
         DSPARK CONCURRENCY PREFILL_BATCH_TOKENS PREFIX_CACHE_ENTRIES KV_POOL_SIZE \
         SPARK_COUNT SPARK_TP SPARK_EP SPARK_DEVICE_BUDGET_BYTES HOST_CACHE_BYTES \
         MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS; do
  printf '%s=%s\n' "$v" "${!v}"
done
printf 'MOA_LANE=%s\n' "$SPARK_5_LANE_A"
printf 'MOA_HOST=%s\n' "$SPARK_5_HOST"
printf 'RANKMAP=%s\n' "$(release_spark_rank_map | paste -sd';' -)"
"""


def resolve(config: Path) -> dict:
    result = subprocess.run(
        ["bash", "-c", PROBE, "probe", str(COMMON), str(config)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    values = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        values[key] = value
    return values


@pytest.mark.parametrize("rtx,layers", [(1, "5"), (2, "20")])
def test_candidate_resolves_to_matched_tp6_geometry(rtx, layers):
    values = resolve(CONFIGS[rtx])
    assert values["SPARK_COUNT"] == "6"
    assert values["SPARK_TP"] == "6"
    assert values["SPARK_EP"] == "1"
    assert values["RTX_GPUS"] == str(rtx)
    assert values["RTX_EXPERT_LAYERS"] == layers
    assert values["MODEL_ID"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert values["MODEL_REVISION"] == BASE_REVISION
    assert values["EXPERT_FORMAT"] == "native"
    assert values["SPARKINFER_EXL3"] == "disable"
    # `auto` pool reproduces the baseline; an explicit pin would be a confound.
    assert values["KV_POOL_SIZE"] == ""
    assert values["HOST_CACHE_BYTES"] == "auto"
    # Six group-major ranks, one replicated group.
    assert values["RANKMAP"] == "0 0 0;1 0 1;2 0 2;3 0 3;4 0 4;5 0 5"


def test_candidate_controls_match_the_baseline_dials():
    dials = [
        "DSPARK", "CONCURRENCY", "PREFILL_BATCH_TOKENS", "PREFIX_CACHE_ENTRIES",
        "MAX_CONTEXT_TOKENS", "MAX_OUTPUT_TOKENS", "HOST_CACHE_BYTES",
        "SPARK_DEVICE_BUDGET_BYTES", "KV_POOL_SIZE", "MODEL_ID", "MODEL_REVISION",
        "EXPERT_FORMAT",
    ]
    one = resolve(CONFIGS[1])
    two = resolve(CONFIGS[2])
    for dial in dials:
        assert one[dial] == two[dial], dial
    assert one["DSPARK"] == "on"
    assert one["CONCURRENCY"] == "16"
    assert one["PREFILL_BATCH_TOKENS"] == "2048"
    assert one["PREFIX_CACHE_ENTRIES"] == "20"
    assert one["MAX_CONTEXT_TOKENS"] == "1048576"
    assert one["MAX_OUTPUT_TOKENS"] == "393216"


def test_moa_uses_unified_fabric_and_separate_management():
    for rtx in (1, 2):
        values = resolve(CONFIGS[rtx])
        assert values["MOA_LANE"] == "10.55.0.6"
        assert values["MOA_HOST"] == "172.22.2.6"
        assert "not a shipped default" in CONFIGS[rtx].read_text()


ACTIVE_CONFIGS = [
    REPO / "ds41rt.config",
    *sorted((REPO / "examples" / "configs").glob("*.config")),
    *sorted((REPO / "scripts" / "fixtures" / "tp-ep-six").glob("site-*.config")),
    *CONFIGS.values(),
]


@pytest.mark.parametrize("config", ACTIVE_CONFIGS, ids=lambda p: p.name)
def test_active_configs_use_matching_isolated_rails(config):
    values = dict(
        line.split("=", 1) for line in config.read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    ranks = range(6) if config.name == "ds41rt.config" else range(int(values["SPARK_COUNT"]))
    # `release-common.sh` requires the secondary rail all-or-none
    # (`secondary Spark rail must provide all N active LANE_B values or none`).
    # The v10 three-rank candidate profiles are documented single-rail A-only and
    # name no LANE_B, so an absent rail is valid; a present one must be complete
    # and address each rank on the matching isolated subnet.
    lane_b = [values.get(f"SPARK_{rank}_LANE_B", "") for rank in ranks]
    assert all(lane_b) or not any(lane_b), (config.name, lane_b)
    for rank in ranks:
        assert values[f"SPARK_{rank}_LANE_A"] == f"10.55.0.{rank + 1}"
        if any(lane_b):
            assert lane_b[rank] == f"10.55.1.{rank + 1}"
    # Exercise the real parser as well as the address inventory.
    resolve(config)
