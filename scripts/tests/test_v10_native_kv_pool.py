#!/usr/bin/env python3
"""CPU-only checks for the pinned v10 native TP3EP1 KV pool.

The native TP3EP1 profile used to leave `KV_POOL_SIZE` unset. The daemon's `auto`
policy then sizes the KV/index pool to fill the device down to its fixed 2 GiB
`RUNTIME_HEADROOM`, so the pool is not bounded by the serving workload. The
v10-native-tp3ep1 lane measured the result and then failed tool-eval admission
with explicit OOM under `--parallel 16`; the evidence is retained at
`runs/v10-native-tp3ep1/tool-eval-short-oom/README.md` and
`runs/v10-native-tp3ep1/logs/coordinator-tool-eval-oom.log`.

These tests pin the fix and the arithmetic that justifies it:

* the profile carries exactly one explicit `KV_POOL_SIZE`;
* the pinned value is a whole number of the daemon's page groups and clears the
  launcher's own minimum admission page count;
* relative to the measured `auto` pool it returns about 3.6 GiB to device
  headroom, while still holding many times the observed tool-eval working set;
* it does not reduce the tool-eval protocol's concurrency or any per-request
  context/output limit.

No Docker, SSH, GPU, CUDA or network is touched. The measured constants are the
numbers recorded in the lane evidence above; they are inputs to the arithmetic,
not claims about a new run.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "examples" / "configs" / "tp3ep1-native.config"
LANE_README = REPO / "runs" / "v10-native-tp3ep1" / "tool-eval-short-oom" / "README.md"
SELECTOR = REPO / "scripts" / "select-release-gpus.py"

# The explicit pin this profile is expected to carry. Change it here and in the
# config together; every dependent number below is re-derived.
EXPECTED_POOL = "12GiB"

# Measured by the v10-native-tp3ep1 lane (logs/coordinator-memory.txt, also
# copied into raw/memory.json). These are the pool figures the fix is relative to.
MEASURED_POOL = {
    "groups": 36_650,
    "global_bytes": 16_700_672_000,
    "cache_bytes": 16_745_176_576,
    "reservation_bytes": 101_973_491_712,
    "runtime_headroom_bytes": 2_147_483_648,
    "device_occupied_bytes": 99_091_742_720,
}
TOTAL_DEVICE_BYTES = 102_641_958_912  # 97,887 MiB, nvidia-smi total from the same package
MEASURED_FREE_BYTES = TOTAL_DEVICE_BYTES - MEASURED_POOL["device_occupied_bytes"]

MINIMUM_RETURNED_BYTES = 3_500 * 1024 ** 2  # the fix must clearly beat noise
MINIMUM_POOL_TOKENS = 1_048_576 + 393_216  # one max-context request plus its output


def selector_module():
    """Load the launcher's own pool arithmetic instead of restating constants."""
    name = "select_release_gpus_for_kv_pool_test"
    spec = importlib.util.spec_from_file_location(name, SELECTOR)
    module = importlib.util.module_from_spec(spec)
    # Dataclass field resolution looks the module up in sys.modules, so register
    # it for the duration of the load and leave the session's registry clean.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module


def load_config() -> dict[str, str]:
    """Load the profile through the real release loader (bash only, no hardware)."""
    probe = (
        'release_load_config "$1"; '
        'printf "%s\\n" "$KV_POOL_SIZE" "$CONCURRENCY" "$PREFIX_CACHE_ENTRIES" '
        '"$MAX_CONTEXT_TOKENS" "$MAX_OUTPUT_TOKENS" "$PREFILL_BATCH_TOKENS" '
        '"$RTX_EXPERT_LAYERS" "$MEMORY_RESERVATION" "$SPARK_COUNT" "$SPARK_TP" "$SPARK_EP"'
    )
    result = subprocess.run(
        ["bash", "-euc", "source scripts/release-common.sh; " + probe, "test", str(CONFIG)],
        cwd=REPO, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    keys = ("KV_POOL_SIZE", "CONCURRENCY", "PREFIX_CACHE_ENTRIES", "MAX_CONTEXT_TOKENS",
            "MAX_OUTPUT_TOKENS", "PREFILL_BATCH_TOKENS", "RTX_EXPERT_LAYERS",
            "MEMORY_RESERVATION", "SPARK_COUNT", "SPARK_TP", "SPARK_EP")
    return dict(zip(keys, result.stdout.splitlines()))


def booked_cache_bytes(selector, groups: int) -> int:
    """KV cache bytes for `groups`, calibrated from the measured pool itself.

    The daemon's `cache_bytes` is `groups * GROUP_BYTES` plus a small per-group
    bookkeeping term, and the measured pool pins that term exactly:

        36,650 * 455,680 + 36,650 * 1,214 + 2 == 16,745,176,576
    """
    per_group = (MEASURED_POOL["cache_bytes"] - MEASURED_POOL["groups"] * selector.GROUP_BYTES) \
        // MEASURED_POOL["groups"]
    return groups * selector.GROUP_BYTES + groups * per_group + 2


def planned_groups(selector, values: dict[str, str]) -> int:
    """Groups the daemon's explicit-pool fitter can book.

    Round the pinned size down to whole groups, then step back while the booked
    cache would exceed the device budget left after `auto`'s fixed runtime
    headroom is applied to the launcher ceiling (the same `available` value the
    measured launch used).
    """
    groups = selector.parse_bytes(values["KV_POOL_SIZE"]) // selector.GROUP_BYTES
    available = MEASURED_POOL["reservation_bytes"] - MEASURED_POOL["groups"] * selector.GROUP_BYTES \
        - MEASURED_POOL["runtime_headroom_bytes"]
    while groups > 0 and booked_cache_bytes(selector, groups) > available:
        groups -= 1
    return groups


def test_profile_pins_exactly_one_kv_pool_size():
    values = load_config()
    assert values["KV_POOL_SIZE"] == EXPECTED_POOL
    assert values["MEMORY_RESERVATION"] == ""
    lines = [line for line in CONFIG.read_text().splitlines()
             if line.startswith("KV_POOL_SIZE=")]
    assert lines == [f"KV_POOL_SIZE={EXPECTED_POOL}"], lines


def test_pinned_pool_is_a_whole_page_group_count_that_clears_minimum_admission():
    selector = selector_module()
    values = load_config()
    groups = planned_groups(selector, values)
    assert 0 < groups <= 131_072
    # The daemon allocates whole groups as [g, g, g, 2g]; the pool log line's
    # global_bytes is exactly groups * GROUP_BYTES. The bound is nominal: with
    # 12GiB pinned the launcher's own conversion (a plain round-down, which is
    # what run.sh forwards) yields 28,276 groups = 12,884,807,680 B, and the
    # daemon's cache fit may step back a few groups from there.
    global_bytes = groups * selector.GROUP_BYTES
    assert global_bytes == 12_884_807_680
    assert selector.parse_bytes(values["KV_POOL_SIZE"]) // selector.GROUP_BYTES == 28_276
    assert selector.parse_bytes(values["KV_POOL_SIZE"]) - global_bytes < selector.GROUP_BYTES
    # The launcher's own `desired_groups` rejects anything below this.
    minimum = 2 * int(values["CONCURRENCY"]) + 2 * int(values["PREFIX_CACHE_ENTRIES"])
    assert groups >= minimum, (groups, minimum)
    # One maximum-context request plus its full output allowance still fits.
    tokens = global_bytes // selector.GROUP_BYTES * 512
    assert tokens >= MINIMUM_POOL_TOKENS, tokens


def test_pinned_pool_returns_headroom_without_changing_the_serving_contract():
    selector = selector_module()
    values = load_config()
    groups = planned_groups(selector, values)
    returned = MEASURED_POOL["cache_bytes"] - booked_cache_bytes(selector, groups)
    assert returned >= MINIMUM_RETURNED_BYTES, returned
    # Bottom-up expert placement is requested at a fixed layer count, so the
    # released bytes are not re-spent on local layers; the boundary stays 5/35.
    assert values["RTX_EXPERT_LAYERS"] == "5"
    assert 40 - int(values["RTX_EXPERT_LAYERS"]) == 35
    # The protocol concurrency and the per-request limits are untouched.
    assert values["CONCURRENCY"] == "16"
    assert values["MAX_CONTEXT_TOKENS"] == "1048576"
    assert values["MAX_OUTPUT_TOKENS"] == "393216"
    assert values["PREFILL_BATCH_TOKENS"] == "2048"  # clamped to 256 by the native lane
    assert values["SPARK_COUNT"] == "3"
    assert (values["SPARK_TP"], values["SPARK_EP"]) == ("3", "1")


def test_measured_oom_evidence_still_supports_the_arithmetic():
    if not LANE_README.is_file():
        pytest.skip("v10-native-tp3ep1 tool-eval evidence not present in this checkout")
    text = LANE_README.read_text()
    # The evidence this change answers, verbatim, and the protocol that produced it.
    assert "--parallel 16" in text
    assert "cudaGraphInstantiate failed: out of memory" in text
    assert "cudaMalloc failed: out of memory" in text
    # The measured residency the fix is relative to.
    numbers = {key: int(value.replace(",", "")) for key, value in re.findall(
        r"(device_occupied_bytes|device_budget_bytes)=([0-9,]+)", text)}
    assert numbers["device_occupied_bytes"] == MEASURED_POOL["device_occupied_bytes"]
    assert numbers["device_budget_bytes"] == MEASURED_POOL["reservation_bytes"]
    # The starting point was materially short: under 3 GiB inside the budget, and
    # under 4 GiB of genuinely free device memory.
    assert MEASURED_POOL["reservation_bytes"] - MEASURED_POOL["device_occupied_bytes"] == 2_881_748_992
    assert MEASURED_FREE_BYTES == 3_550_216_192
