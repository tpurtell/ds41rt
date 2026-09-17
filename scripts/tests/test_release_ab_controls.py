from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_COMMON = ROOT / "scripts" / "release-common.sh"
FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"

BASE_CONFIG = """\
MODEL_ID=deepseek-ai/DeepSeek-V4.1-Flash
MODEL_VARIANT=flash
EXPERT_FORMAT=native
SPARKINFER_EXL3=disable
DSPARK=on
COORDINATOR_GPU=0
SPARK_0_HOST=ostrich
SPARK_1_HOST=dodo
SPARK_2_HOST=emu
SPARK_3_HOST=kiwi
SPARK_0_LANE_A=10.55.0.1
SPARK_1_LANE_A=10.55.0.2
SPARK_2_LANE_A=10.55.0.3
SPARK_3_LANE_A=10.55.0.4
"""


def load_config(
    tmp_path: Path,
    extra: str = "",
    *,
    config_text: str = BASE_CONFIG,
) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "ds41rt.config"
    config.write_text(config_text + extra, encoding="utf-8")
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; release_load_config "$2"; '
                'printf "%s\\n" "$MODEL_ID" "$MODEL_VARIANT" '
                '"$EXPERT_FORMAT" "$DSPARK" "$DSPARK_DRAFT_POLICY" '
                '"$CONCURRENCY" "$SPARK_REDUCTION_MIN_ROWS" '
                '"$COORDINATOR_GPU" "$COORDINATOR_GPU_UUID" '
                '"$COORDINATOR_GPU_PCI_BUS_ID" '
                '"$SPARKINFER_EXL3" "$RELEASE_MODEL_ID" '
                '"$RELEASE_MODEL_REVISION"'
            ),
            "bash",
            str(RELEASE_COMMON),
            str(config),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def model_list_matches(payload: dict, model_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; release_model_list_matches "$2"',
            "bash",
            str(RELEASE_COMMON),
            model_id,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        input=json.dumps(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def native_model_list_matches(
    payload: dict, model_id: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; release_native_model_list_matches "$2"',
            "bash",
            str(RELEASE_COMMON),
            model_id,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        input=json.dumps(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def resolve_gpu_identity(
    tmp_path: Path,
    nvidia_smi_row: str,
    *,
    extra: str = "",
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' "
        + repr(nvidia_smi_row)
        + "\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    config = tmp_path / "physical.config"
    config.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; release_load_config "$2"; '
                "release_resolve_coordinator_gpu_identity; "
                'printf "%s\\n" "$RELEASE_COORDINATOR_GPU_UUID" '
                '"$RELEASE_COORDINATOR_GPU_HOST_INDEX" '
                '"$RELEASE_COORDINATOR_GPU_PCI_BUS_ID"'
            ),
            "bash",
            str(RELEASE_COMMON),
            str(config),
        ],
        cwd=ROOT,
        env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_release_config_has_safe_flash_defaults(tmp_path: Path) -> None:
    result = load_config(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "flash",
        "native",
        "on",
        "adaptive",
        "16",
        "16",
        "0",
        "",
        "",
        "disable",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "dba1be0a40aa45a94ad051997016db3960a90277",
    ]


def test_release_config_defaults_to_official_v41_flash(tmp_path: Path) -> None:
    result = load_config(
        tmp_path,
        config_text="""\
SPARK_0_HOST=ostrich
SPARK_1_HOST=dodo
SPARK_2_HOST=emu
SPARK_3_HOST=kiwi
SPARK_0_LANE_A=10.55.0.1
SPARK_1_LANE_A=10.55.0.2
SPARK_2_LANE_A=10.55.0.3
SPARK_3_LANE_A=10.55.0.4
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "flash",
        "native",
        "on",
        "adaptive",
        "16",
        "16",
        "0",
        "",
        "",
        "disable",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "dba1be0a40aa45a94ad051997016db3960a90277",
    ]


def test_release_facing_helpers_default_to_official_flash() -> None:
    surfaces = (
        ROOT / "justfile",
        ROOT / "scripts/api-smoke.sh",
        ROOT / "scripts/api-constrained-smoke.sh",
        ROOT / "scripts/ask-file.sh",
        ROOT / "scripts/doctor.sh",
        ROOT / "scripts/ds41rt-dev.sh",
        ROOT / "scripts/phase0-spark-tcp-bench.sh",
        ROOT / "scripts/real-slice-tcp-smoke.sh",
        ROOT / "scripts/real-full-tcp-smoke.sh",
        ROOT / "scripts/real-full-tcp-stream-smoke.sh",
        ROOT / "scripts/real-full-tcp-live-smoke.sh",
        ROOT / "scripts/real-full-tcp-serve.sh",
    )
    for surface in surfaces:
        text = surface.read_text(encoding="utf-8")
        assert FLASH_MODEL_ID in text, surface
        assert "deepseek-ai/DeepSeek-V4-Flash-0731" not in text, surface

    just_dry_run = subprocess.run(
        ["just", "--dry-run", "api-smoke"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = just_dry_run.stdout + just_dry_run.stderr
    assert FLASH_MODEL_ID in rendered
    assert "{{model_id}}" not in rendered


def test_release_settings_resolution_has_selected_gpu_access() -> None:
    release = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "release_resolve_coordinator_gpu_identity" in release
    assert 'gpu_request="device=$gpu_uuid_csv"' in release
    assert '-e "CUDA_VISIBLE_DEVICES=$gpu_uuid_csv"' in release


def test_native_release_has_explicit_startup_timeout() -> None:
    release = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert "DS41RT_RELEASE_READY_TIMEOUT_SECONDS:-900" in release
    assert "native expert did not become ready" in release
    assert "native API did not become ready" in release


def test_spark_collective_receive_uses_active_deadline() -> None:
    reduction = (
        ROOT / "rust/crates/ds41rt-daemon/src/commands/real_full/rdma_reduction.rs"
    ).read_text(encoding="utf-8")
    transport = (
        ROOT / "rust/crates/ds41rt-transport/src/verbs.rs"
    ).read_text(encoding="utf-8")
    native = (ROOT / "native/src/ds41rt_native.cc").read_text(encoding="utf-8")

    assert "SPARK_RDMA_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(60)" in reduction
    assert "wait_recv_slot_with_timeout(SPARK_RDMA_EXCHANGE_TIMEOUT)" in reduction
    assert "wait_recv_slot_with_timeout_and_stats(SPARK_RDMA_EXCHANGE_TIMEOUT)" in reduction
    assert "wait_recv_slot_with_timeout_and_stats(Duration::from_secs(1))" in transport
    assert "poll_stats(0, 1, timeout)" in transport
    assert "idle_wait ?" not in native


def test_release_config_accepts_future_pro_exl3_identity(tmp_path: Path) -> None:
    revision = "a" * 40
    result = load_config(
        tmp_path,
        "MODEL_ID=future-org/deepseek-v4-pro\n"
        "MODEL_VARIANT=pro\n"
        f"MODEL_REVISION={revision}\n"
        "EXPERT_FORMAT=exl3\n"
        "DSPARK=off\n"
        "SPARKINFER_EXL3=force\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "future-org/deepseek-v4-pro",
        "pro",
        "exl3",
        "off",
        "adaptive",
        "16",
        "16",
        "0",
        "",
        "",
        "force",
        "future-org/deepseek-v4-pro",
        revision,
    ]


def test_release_config_accepts_dedicated_flash_k2_identity(tmp_path: Path) -> None:
    revision = "b" * 64
    model_id = "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4"
    result = load_config(
        tmp_path,
        f"MODEL_ID={model_id}\n"
        f"MODEL_REVISION={revision}\n"
        "EXPERT_FORMAT=exl3\n"
        "SPARKINFER_EXL3=force\n",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        model_id,
        "flash",
        "exl3",
        "on",
        "adaptive",
        "16",
        "16",
        "0",
        "",
        "",
        "force",
        model_id,
        revision,
    ]


def test_release_config_accepts_adaptive_dspark_policy(tmp_path: Path) -> None:
    result = load_config(tmp_path, "DSPARK_DRAFT_POLICY=adaptive\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[4] == "adaptive"


def test_release_config_rejects_unknown_dspark_policy(tmp_path: Path) -> None:
    result = load_config(tmp_path, "DSPARK_DRAFT_POLICY=guess\n")
    assert result.returncode != 0
    assert "DSPARK_DRAFT_POLICY must be full or adaptive" in result.stderr


def test_release_config_validates_spark_reduction_cutoff(tmp_path: Path) -> None:
    selected = load_config(tmp_path, "SPARK_REDUCTION_MIN_ROWS=6\n")
    assert selected.returncode == 0, selected.stderr
    assert selected.stdout.splitlines()[6] == "6"

    invalid = load_config(tmp_path, "SPARK_REDUCTION_MIN_ROWS=0\n")
    assert invalid.returncode != 0
    assert "SPARK_REDUCTION_MIN_ROWS must be a positive integer" in invalid.stderr


def test_release_readiness_requires_exact_configured_api_identity() -> None:
    model_id = "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4"
    exact = {
        "object": "list",
        "data": [
            {"id": model_id},
            {"id": f"{model_id}-full"},
            {"id": "ds41rt-tiny"},
        ],
    }
    assert model_list_matches(exact, model_id).returncode == 0

    stale_native = {
        "object": "list",
        "data": [
            {"id": "deepseek-ai/DeepSeek-V4-Flash-0731"},
            {"id": "deepseek-ai/DeepSeek-V4-Flash-0731-full"},
        ],
    }
    assert model_list_matches(stale_native, model_id).returncode != 0

    duplicate = {
        "object": "list",
        "data": [
            {"id": model_id},
            {"id": f"{model_id}-full"},
            {"id": f"{model_id}-full"},
        ],
    }
    assert model_list_matches(duplicate, model_id).returncode != 0

    native = {"object": "list", "data": [{"id": model_id}]}
    assert native_model_list_matches(native, model_id).returncode == 0
    assert native_model_list_matches(native, f"{model_id}-other").returncode != 0
    duplicate_native = {"object": "list", "data": [{"id": model_id}] * 2}
    assert native_model_list_matches(duplicate_native, model_id).returncode != 0


def test_release_config_rejects_native_pro_and_accepts_gpu1(tmp_path: Path) -> None:
    native_pro = load_config(tmp_path, "MODEL_VARIANT=pro\n")
    assert native_pro.returncode == 2
    assert "requires EXPERT_FORMAT=exl3" in native_pro.stderr

    gpu1 = load_config(tmp_path, "COORDINATOR_GPU=1\n")
    assert gpu1.returncode == 0, gpu1.stderr
    assert gpu1.stdout.splitlines()[7] == "1"

    malformed_uuid = load_config(tmp_path, "COORDINATOR_GPU_UUID=GPU-1\n")
    assert malformed_uuid.returncode == 2
    assert "physical NVIDIA GPU UUID" in malformed_uuid.stderr

    unpaired_uuid = load_config(
        tmp_path,
        "COORDINATOR_GPU_UUID=GPU-95f8f212-9131-df99-fd53-7535965197d7\n",
    )
    assert unpaired_uuid.returncode == 2
    assert "must be set together" in unpaired_uuid.stderr


def test_release_resolves_and_preserves_physical_coordinator_gpu(
    tmp_path: Path,
) -> None:
    uuid = "GPU-95f8f212-9131-df99-fd53-7535965197d7"
    result = resolve_gpu_identity(
        tmp_path,
        f"0, {uuid}, 00000000:11:00.0",
        extra=(
            f"COORDINATOR_GPU_UUID={uuid}\n"
            "COORDINATOR_GPU_PCI_BUS_ID=00000000:11:00.0\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [uuid, "0", "00000000:11:00.0"]


def test_release_freezes_unconfigured_gpu0_to_its_physical_identity(
    tmp_path: Path,
) -> None:
    uuid = "GPU-95f8f212-9131-df99-fd53-7535965197d7"
    result = resolve_gpu_identity(tmp_path, f"0, {uuid}, 00000000:11:00.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [uuid, "0", "00000000:11:00.0"]


def test_release_rejects_physical_gpu_identity_mismatch(tmp_path: Path) -> None:
    configured_uuid = "GPU-95f8f212-9131-df99-fd53-7535965197d7"
    observed_uuid = "GPU-fe5b6dd0-028b-9474-dead-271d57762e05"
    result = resolve_gpu_identity(
        tmp_path,
        f"1, {observed_uuid}, 00000000:e1:00.0",
        extra=(
            f"COORDINATOR_GPU_UUID={configured_uuid}\n"
            "COORDINATOR_GPU_PCI_BUS_ID=00000000:11:00.0\n"
        ),
    )
    assert result.returncode == 2
    assert "resolved to another physical device" in result.stderr


def test_release_config_allows_explicit_high_concurrency_without_dspark(
    tmp_path: Path,
) -> None:
    result = load_config(tmp_path, "CONCURRENCY=4\nDSPARK=off\n")
    assert result.returncode == 0, result.stderr
    values = result.stdout.splitlines()
    assert values[3:6] == ["off", "adaptive", "4"]


def test_release_config_rejects_old_glm_and_out_of_range_controls(
    tmp_path: Path,
) -> None:
    old = load_config(tmp_path, "SPARKINFER_GLM_H64_QUERY_PROJECTION=auto\n")
    assert old.returncode == 2
    assert "unknown configuration key" in old.stderr

    concurrency = load_config(tmp_path, "CONCURRENCY=17\n")
    assert concurrency.returncode == 2
    assert "CONCURRENCY must be in 1..16" in concurrency.stderr


def test_release_config_rejects_exl3_kernel_mode_format_mismatches(
    tmp_path: Path,
) -> None:
    forced_native = load_config(tmp_path, "SPARKINFER_EXL3=force\n")
    assert forced_native.returncode == 2
    assert "force requires EXPERT_FORMAT=exl3" in forced_native.stderr

    disabled_exl3 = load_config(
        tmp_path,
        "EXPERT_FORMAT=exl3\nSPARKINFER_EXL3=disable\n",
    )
    assert disabled_exl3.returncode == 2
    assert "disable requires EXPERT_FORMAT=native" in disabled_exl3.stderr


def test_launchers_fingerprint_and_export_model_controls() -> None:
    release = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "release_api_advertises_native_model" in release
    assert "release_resolve_coordinator_gpu_identity" in release
    assert '"$RELEASE_COORDINATOR_GPU_UUID"' in release
    assert '"$RELEASE_MODEL_ID"' in release
    assert '"$RELEASE_MODEL_REVISION"' in release
    assert '"$CONCURRENCY"' in release
    assert '"$KV_POOL_SIZE"' in release
    assert '"$MEMORY_RESERVATION"' in release
    assert '"$PREFIX_CACHE_ENTRIES"' in release
    assert "SPARKINFER_GLM_H64" not in release

    for launcher_name in ("scripts/run-wip.sh",):
        launcher = (ROOT / launcher_name).read_text(encoding="utf-8")
        assert "SPARKINFER_EXL3" in launcher
        assert "DS41RT_SPARKINFER_EXL3=$SPARKINFER_EXL3" in launcher
        assert "DS41RT_MODEL_REVISION=$RELEASE_MODEL_REVISION" in launcher
        assert "release_api_advertises_model" in launcher
        assert "release_validate_model_list_file" in launcher
        assert '"$MODEL_VARIANT"' in launcher
        assert '"$EXPERT_FORMAT"' in launcher
        assert '"$DSPARK"' in launcher
        assert "validate_ds4_staged_snapshot.py" in launcher
        assert "--startup-contract-only" in launcher
        assert "qualified immutable publication" in launcher
        assert "release_resolve_coordinator_gpu_identity" in launcher
        assert '"$RELEASE_COORDINATOR_GPU_UUID"' in launcher
        assert "SPARKINFER_GLM_H64" not in launcher

    wip = (ROOT / "wip.sh").read_text(encoding="utf-8")
    assert "release_resolve_coordinator_gpu_identity" in wip
    assert '--gpus device="$RELEASE_COORDINATOR_GPU_UUID"' in wip
    assert "is not bound to the configured physical GPU" in wip

    dev = (ROOT / "scripts" / "ds41rt-dev.sh").read_text(encoding="utf-8")
    assert "release_resolve_coordinator_gpu_identity" in dev
    assert '--gpus device="$RELEASE_COORDINATOR_GPU_UUID"' in dev

    phase0 = (ROOT / "scripts/phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    assert 'model_revision="${DS41RT_MODEL_REVISION:-}"' in phase0
    assert '-e DS41RT_MODEL_REVISION="$model_revision"' in phase0
    assert (
        'wip_allow_historical_exl3_control="${DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL:-0}"'
        in phase0
    )
    assert 'wip_allow_historical_exl3_control="${73:-0}"' in phase0
    assert (
        '-e DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL='
        '"$wip_allow_historical_exl3_control"'
        in phase0
    )
