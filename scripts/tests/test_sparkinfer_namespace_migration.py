from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).parents[2]
PYTHON_ROOTS = (ROOT / "benchmarks", ROOT / "python", ROOT / "scripts")
IGNORED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
B12X_REQUIREMENT = re.compile(
    r"(?i)(?<![\w.-])b12x(?:\[[^\]\r\n]*\])?\s*"
    r"(?:===|==|~=|!=|<=|>=|<|>|@)"
)
CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
QUALIFIED_CUTLASS_VERSION = "4.6.2"
METADATA_FREE_PYTHON_CACHE_MARKERS = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
)


def active_python_sources() -> list[Path]:
    return sorted(
        path
        for root in PYTHON_ROOTS
        for path in root.rglob("*.py")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def is_retired_sparkinfer_module(module: str | None) -> bool:
    return module == "sparkinfer" or bool(
        module and module.startswith("sparkinfer.")
    )


def test_active_python_does_not_import_retired_sparkinfer_package() -> None:
    violations: list[str] = []
    for path in active_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                if is_retired_sparkinfer_module(module):
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{node.lineno}: {module}")

    assert not violations, (
        "active Python sources must import b12x, not the retired sparkinfer "
        "package:\n" + "\n".join(violations)
    )


def test_embedded_python_bridge_requires_b12x_namespace() -> None:
    bridge = (
        ROOT / "rust/crates/ds41rt-daemon/src/python_graph_capture.rs"
    ).read_text(encoding="utf-8")
    modules = bridge.split(
        "const COORDINATOR_PYTHON_CAPTURE_MODULES", maxsplit=1
    )[1].split("];", maxsplit=1)[0]

    assert '"b12x",' in modules
    assert '"sparkinfer",' not in modules


def test_standalone_tools_bootstrap_pinned_source_before_b12x_imports() -> None:
    violations: list[str] = []
    explicit_source_tools = {
        "compare_v41_expert_upstream.py",
        "compare_v41_index_topk.py",
        "compare_v41_mhc_upstream.py",
        "compare_v41_narrow_projection.py",
        "compare_v41_upstream_attention.py",
        "compare_v41_vocab_upstream.py",
        "compare_v41_wo_b_fusion.py",
    }
    tools_root = ROOT / "python" / "tools"
    for path in sorted(tools_root.glob("*.py")):
        if path.name == "_pinned_sparkinfer.py":
            continue
        text = path.read_text(encoding="utf-8")
        if path.name in explicit_source_tools:
            assert "--b12x-root" in text or "PYTHONPATH pointing at" in text
            continue
        tree = ast.parse(text, filename=str(path))
        external_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "b12x"
                    or alias.name.startswith("b12x.")
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == "b12x"
                    or bool(node.module and node.module.startswith("b12x."))
                )
            )
        ]
        if not external_imports:
            continue
        bootstrap_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            and any(alias.name == "_pinned_sparkinfer" for alias in node.names)
        ]
        relative = path.relative_to(ROOT)
        if len(bootstrap_imports) != 1:
            violations.append(f"{relative}: expected one _pinned_sparkinfer import")
            continue
        bootstrap_line = bootstrap_imports[0].lineno
        first_external_line = min(node.lineno for node in external_imports)
        if bootstrap_line >= first_external_line:
            violations.append(
                f"{relative}:{bootstrap_line}: bootstrap follows SparkInfer "
                f"import at line {first_external_line}"
            )

    assert not violations, (
        "standalone tools must verify and prepend DS41RT's pinned b12x/SparkInfer "
        "tree before importing it:\n" + "\n".join(violations)
    )


def test_build_metadata_does_not_pin_retired_b12x_package() -> None:
    package_files = [
        ROOT / "python" / "pyproject.toml",
        ROOT / "python" / "uv.lock",
        *sorted((ROOT / "docker").glob("Dockerfile*")),
        *sorted(ROOT.glob("requirements*.txt")),
    ]
    violations: list[str] = []
    for path in package_files:
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if B12X_REQUIREMENT.search(line):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                )

    assert not violations, (
        "build metadata must not install or pin an external b12x package:\n"
        + "\n".join(violations)
    )


def test_images_validate_the_pinned_b12x_import_namespace() -> None:
    for relative in ("docker/Dockerfile.dev", "docker/Dockerfile.release"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "pathlib, b12x" in text
        assert 'importlib.metadata.version("b12x")' in text
        assert "pathlib, sparkinfer" not in text
        assert 'importlib.metadata.version("sparkinfer")' not in text


def test_metadata_free_release_copies_filter_and_reject_python_caches() -> None:
    for relative in ("build.sh", "scripts/build-release-artifacts.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        exclude_lines = [
            line for line in text.splitlines() if "--exclude" in line
        ]
        for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
            assert any(marker in line for line in exclude_lines), (
                f"{relative} must exclude {marker} from metadata-free "
                "SparkInfer source copies"
            )
        assert text.count("--require-no-python-cache") == 1, (
            f"{relative} must guard its copied SparkInfer source exactly once"
        )

    for relative in (
        "build.sh",
        "wip.sh",
        "scripts/build-release-artifacts.sh",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        exclude_lines = [line for line in text.splitlines() if "--exclude" in line]
        assert any(".venv*" in line for line in exclude_lines), (
            f"{relative} must exclude every local named Python environment"
        )

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
        assert marker in dockerignore, (
            f".dockerignore must exclude SparkInfer cache marker {marker}"
        )
    for relative in ("docker/Dockerfile.dev", "docker/Dockerfile.release"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ENV PYTHONDONTWRITEBYTECODE=1" in text
        assert "--require-no-python-cache" in text, (
            f"{relative} must reject cached Python artifacts after COPY"
        )


def test_release_and_wip_exclude_legacy_ds4_aot() -> None:
    release = (ROOT / "scripts/build-release-artifacts.sh").read_text(
        encoding="utf-8"
    )
    assert "-DDS41RT_ENABLE_DS4_FLASH_AOT=OFF" in release, (
        "native V4.1 release artifacts must exclude the legacy DS4 Flash/Pro "
        "AOT bridge"
    )

    # The legacy DS4 Flash/Pro mixed-kernel launcher ABI drifted from the
    # pinned SparkInfer and no longer compiles; the native serve path is the
    # only supported development loop, so WIP artifacts also exclude it.
    wip = (ROOT / "scripts/build-wip-artifacts.sh").read_text(encoding="utf-8")
    assert "-DDS41RT_ENABLE_DS4_FLASH_AOT=OFF" in wip, (
        "development artifacts must exclude the stale legacy DS4 Flash/Pro "
        "AOT bridge"
    )


def test_remote_dev_staging_reconciles_the_pinned_fork() -> None:
    for relative in (
        "scripts/phase0-spark-tcp-bench.sh",
        "scripts/bench-verbs-app-coordinator-links.sh",
        "scripts/bench-verbs-app-pair.sh",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--delete-excluded" in text
        assert "--require-no-python-cache" in text
        for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
            assert marker in text, (
                f"{relative} must exclude SparkInfer cache marker {marker}"
            )


def test_fork_and_images_share_qualified_cutlass_pin() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "dependency agreement"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    versions: dict[str, str] = {}
    for requirement in dependencies:
        for package in CUTLASS_PACKAGES:
            prefix = f"{package}=="
            if requirement.startswith(prefix):
                versions[package] = requirement.removeprefix(prefix)

    assert versions == {
        package: QUALIFIED_CUTLASS_VERSION for package in CUTLASS_PACKAGES
    }, (
        "the SparkInfer fork must pin every CUTLASS DSL package to the "
        f"qualified {QUALIFIED_CUTLASS_VERSION} set, found {versions}"
    )

    image_pin = re.compile(r'nvidia-cutlass-dsl\[cu13\]==([^"]+)"')
    sparkinfer_images = []
    for dockerfile in sorted((ROOT / "docker").glob("Dockerfile*")):
        content = dockerfile.read_text(encoding="utf-8")
        if "third_party/sparkinfer" not in content:
            continue
        sparkinfer_images.append(dockerfile.name)
        match = image_pin.search(content)
        assert match is not None, f"{dockerfile.relative_to(ROOT)} has no CUTLASS DSL pin"
        assert match.group(1) == QUALIFIED_CUTLASS_VERSION, (
            f"{dockerfile.relative_to(ROOT)} pins CUTLASS DSL {match.group(1)}, "
            f"expected {QUALIFIED_CUTLASS_VERSION} to match the fork"
        )
    assert sparkinfer_images == ["Dockerfile.dev", "Dockerfile.release"]


def test_fork_accepts_the_ngc_base_torch_prerelease() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "the image dependency contract"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    torch_requirements = [
        Requirement(requirement)
        for requirement in dependencies
        if Requirement(requirement).name == "torch"
    ]
    assert len(torch_requirements) == 1
    assert Version("2.12.0a0") in torch_requirements[0].specifier, (
        "nvcr.io/nvidia/pytorch:26.05-py3 contains torch 2.12.0a0; "
        f"the fork requirement {torch_requirements[0]} rejects that base and "
        "makes the image's `uv pip check` fail"
    )


def test_standalone_bootstrap_imports_verified_submodule() -> None:
    env = os.environ.copy()
    tools_path = os.fspath(ROOT / "python" / "tools")
    env["PYTHONPATH"] = tools_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import _pinned_sparkinfer as pinned; "
                "print(pinned.IMPORTED_MODULE); print(pinned.REVISION); "
                "print(pinned.VERSION)"
            ),
        ],
        check=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    imported_module, revision, version = result.stdout.strip().splitlines()
    assert Path(imported_module).resolve().is_relative_to(
        (ROOT / "third_party" / "sparkinfer").resolve()
    )
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert Version(version) == Version("1.3.0")


def test_live_launchers_override_stale_cmake_sparkinfer_cache_entries() -> None:
    coordinator = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    spark = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")

    assert (
        '-DDS41RT_SPARKINFER_SOURCE_DIR="$repo_root/third_party/sparkinfer"'
        in coordinator
    )
    assert (
        '-DDS41RT_SPARKINFER_LOCK_FILE="$repo_root/third_party/sparkinfer.lock.json"'
        in coordinator
    )
    assert '-DDS41RT_ENABLE_DS4_FLASH_AOT="$ds4_flash_spark_aot"' in coordinator
    assert '"-DDS41RT_NCCL_INCLUDE_DIR=$host_nccl_include_dir"' in coordinator
    assert '"-DDS41RT_NCCL_LIBRARY=$host_nccl_library"' in coordinator
    assert '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc' in coordinator
    assert "native CMake cache lost requested CUDA/RDMA/NCCL options" in coordinator
    assert "DS41RT_ENABLE_CUDA:BOOL=ON" in coordinator
    assert 'DS41RT_ENABLE_RDMA:BOOL=$native_rdma' in coordinator
    assert "DS41RT_ENABLE_NCCL:BOOL=ON" in coordinator
    assert "-DDS41RT_SPARKINFER_SOURCE_DIR=" in spark
    assert "-DDS41RT_SPARKINFER_LOCK_FILE=" in spark
    coordinator_recipe = justfile.split(
        "build-native-coordinator-test:", maxsplit=1
    )[1].split("\n\n", maxsplit=1)[0]
    assert "-U DS41RT_ENABLE_B12X_AOT" in coordinator_recipe
    assert "-U DS41RT_ENABLE_B12X_COORDINATOR_AOT" in coordinator_recipe
    assert "-DDS41RT_SPARKINFER_SOURCE_DIR=" in coordinator_recipe
    assert "-DDS41RT_SPARKINFER_LOCK_FILE=" in coordinator_recipe


def test_launchers_use_only_the_packed_spark_moe_layout() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    release = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert "DS41RT_SPARK_MOE_MODE" not in phase0
    assert "DS41RT_SPARKINFER_SOURCE_W4A16" not in phase0
    assert "DS41RT_SPARKINFER_HYBRID_W4A4_W4A16" not in phase0
    assert "SPARK_MOE_MODE" not in release
    assert "DS41RT_SPARK_PREBUILT" not in release
    assert "DS41RT_SPARK_SKIP_STAGE" not in release
    assert "expertd-native" in release
    assert "serve-native" in release

    env = os.environ.copy()
    env["DS41RT_SPARK_PREBUILT"] = "1"
    env["DS41RT_SPARK_MOE_MODE"] = "hybrid-w4a4-w4a16"
    env["DS41RT_SERVE_PROFILE"] = "balanced"
    result = subprocess.run(
        [ROOT / "scripts" / "start-spark-experts-tcp.sh", "--dry-run"],
        check=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "DS41RT_SPARK_MOE_MODE" not in result.stdout
    assert "DS41RT_SERVE_PROFILE" not in result.stdout


def test_phase0_image_only_staging_does_not_remove_live_experts() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    cleanup = phase0.split("cleanup() {", maxsplit=1)[1].split(
        "\n}\ntrap cleanup EXIT", maxsplit=1
    )[0]

    assert '[ "$image_only" = "1" ]' in cleanup
    assert "docker rm -f" in cleanup


def test_spark_launcher_defaults_to_runtime_tp4_placement() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'use_diagnostic_placement="${DS41RT_SPARK_USE_DIAGNOSTIC_PLACEMENT:-0}"'
        in phase0
    )
    assert 'if [ "$use_diagnostic_placement" = "1" ]; then' in phase0
    assert 'catalog=""' in phase0
    assert 'ds4_flash_spark_aot="$DS41RT_DS4_FLASH_SPARK_AOT"' in phase0
    assert 'ds4_flash_spark_aot="${71:-1}"' in phase0
    assert "DS41RT_SPARK_TRANSFORMER_TP" not in phase0
    assert "DS41RT_SPARK_LAYER_BLOCK" not in phase0
    assert '-DDS41RT_ENABLE_DS4_FLASH_AOT="$ds4_flash_aot"' in phase0
    real_source_launch = phase0.split(
        'if [ "$DS41RT_BENCH_MODE" = "real" ]; then', maxsplit=1
    )[1].split("\nfi\n", maxsplit=1)[0]
    assert '--model-id "$DS41RT_MODEL_ID"' in real_source_launch
    startup_loop = phase0.rsplit(
        'for host_index in "${!hosts[@]}"; do', maxsplit=1
    )[1].split("\ndone", maxsplit=1)[0]
    assert 'if [ "$use_diagnostic_placement" = "1" ]; then' in startup_loop
    assert 'expert_ids+=("${host}=${host_index}")' in startup_loop


def test_phase0_remote_expertd_argument_vector_is_contiguous() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    launch_start = phase0.rindex('ssh -o BatchMode=yes "$host" bash -s --')
    launch_end = phase0.index("<<'REMOTE'", launch_start)
    launch = phase0[launch_start:launch_end].replace("\\\n", " ")
    tokens = shlex.split(launch)
    payload = tokens[tokens.index("--") + 1 :]

    remote_start = phase0.index("set -euo pipefail", launch_end)
    remote_end = phase0.index("discover_rdma_device_map()", remote_start)
    remote = phase0[remote_start:remote_end]
    assigned_positions = {
        int(match.group(1))
        for match in re.finditer(
            r'^[a-z][a-z0-9_]*="\$(?:\{)?([0-9]+)', remote, re.MULTILINE
        )
    }

    assert len(payload) == 73
    assert assigned_positions == set(range(1, 74))
    assert payload[-2:] == ["$model_revision", "$wip_allow_historical_exl3_control"]
    assert 'ds4_flash_spark_aot="${71:-1}"' in remote
    assert 'model_revision="${72:-}"' in remote
    assert 'wip_allow_historical_exl3_control="${73:-0}"' in remote


def test_release_preflight_requires_matching_engine_revisions() -> None:
    release = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert (
        """engine_commit="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}'"""
        in release
    )
    assert "coordinator image has no engine revision" in release
    assert 'test "$(docker image inspect -f' in release
    assert '"$image")" = "$engine"' in release
    fingerprint = release.split('fingerprint="$(', maxsplit=1)[1].split(
        'coordinator="$RELEASE_COORDINATOR_CONTAINER_NAME"', maxsplit=1
    )[0]
    assert '"$engine_commit"' in fingerprint


def test_release_build_overrides_the_base_image_version_label() -> None:
    build = (ROOT / "build.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "Dockerfile.release").read_text(
        encoding="utf-8"
    )

    assert 'ARG DS41RT_RELEASE_VERSION=unknown' in dockerfile
    assert 'LABEL org.opencontainers.image.version=${DS41RT_RELEASE_VERSION}' in dockerfile
    assert 'spark_release_version="${SPARK_EXPERT_DOCKER_INFERENCE##*:}"' in build
    assert '[[ "$spark_release_version" == "$release_version" ]]' in build
    assert 'release_version="$6"' in build
    assert 'source_manifest_sha256="${8-}"' in build
    assert build.count('--build-arg DS41RT_RELEASE_VERSION="$release_version"') == 2
    assert build.count('org.opencontainers.image.version') == 2
    remote_revision_label = next(
        line
        for line in build.splitlines()
        if "org.opencontainers.image.revision" in line
        and "SPARK_EXPERT_DOCKER_INFERENCE" in line
    )
    remote_version_label = next(
        line
        for line in build.splitlines()
        if "org.opencontainers.image.version" in line
        and "SPARK_EXPERT_DOCKER_INFERENCE" in line
    )
    assert remote_version_label == remote_revision_label.replace("revision", "version")


def test_packed_expert_warmups_use_bf16_wire() -> None:
    for launcher_name in (
        "real-full-tcp-serve.sh",
        "real-full-tcp-live-smoke.sh",
    ):
        launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")
        warmup = launcher.split("warmup_protocol_v2_experts() {", 1)[1].split(
            "\n}\n", 1
        )[0]

        assert "DS41RT_SPARK_MOE_MODE" not in warmup
        assert "--nvfp4-fp8-roundtrip" not in warmup
        assert 'if [ "$warmup_layer_id" != "78" ]; then' not in warmup
        assert 'local wire_contract="bf16-in/bf16-out"' in warmup
        assert "local warmup_routes_per_row=1" not in warmup
        assert (
            'expert_ids="$(warmup_expert_ids_for_rank '
            '"$warmup_routes_per_row")"'
            in warmup
        )
        assert "Every strict-TP4 rank receives the same global routes" in warmup
        assert "never assigns ownership of an expert" in warmup
        assert '--hidden-dim "$warmup_hidden_dim"' in warmup
        assert '--expert-ids "$expert_ids"' in warmup
        assert '--routes-per-row "$warmup_routes_per_row"' in warmup
        assert '"${wire_args[@]}"' in warmup
        assert 'warmup_pids+=("$!")' in warmup
        assert 'wait "${warmup_pids[$warmup_index]}"' in warmup
        for argument in (
            '--warmup-rows "$warmup_rows"',
            '--roundtrip-rows "$warmup_roundtrip_rows"',
            '--prefill-roundtrip-rows "$warmup_prefill_roundtrip_rows"',
            '--prefill-chain-rows "$warmup_prefill_chain_rows"',
        ):
            assert argument in warmup
        for variable, default in (
            ("warmup_rows", "1"),
            ("warmup_roundtrip_rows", "1"),
            ("warmup_prefill_roundtrip_rows", "16,256,512"),
            ("warmup_prefill_chain_rows", "16,256,512"),
        ):
            assert re.search(
                rf'^{variable}="\$\{{[^}}]+:-{re.escape(default)}\}}"$',
                launcher,
                re.MULTILINE,
            )

        expert_ids = launcher.split("warmup_expert_ids_for_rank() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        assert ".layer_id == $layer" not in expert_ids
        assert 'expert_ids+=("$expert_id")' in expert_ids
        assert '(IFS=,; echo "${expert_ids[*]}")' in expert_ids

    coordinator = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    assert 'config["hidden_size"]' in coordinator
    assert 'config["num_experts_per_tok"]' in coordinator
    assert 'config["n_routed_experts"]' in coordinator
    assert "local_files_only=True" in coordinator

    smoke = (ROOT / "scripts" / "real-full-tcp-live-smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "warmup_hidden_dim:-$(jq -er '.facts.hidden_size'" in smoke
    assert "warmup_routes_per_row:-$(jq -er '.facts.top_k'" in smoke


def test_deepseek_full_launchers_do_not_supply_ep_loadplans() -> None:
    for launcher_name in (
        "real-full-tcp-serve.sh",
        "real-full-tcp-live-smoke.sh",
    ):
        launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")
        assert "--loadplan" not in launcher
        assert 'loadplan="${LOADPLAN' not in launcher
        assert 'if [ -n "${LOADPLAN:-}" ]; then' in launcher
        assert "LOADPLAN is not supported by strict DeepSeek V4 TP4 serving" in launcher


def test_deepseek_serving_has_no_legacy_cuda_reference_or_short_k_entry() -> None:
    coordinator = (
        ROOT / "rust" / "crates" / "ds41rt-daemon" / "src" / "commands" / "coordinator.rs"
    ).read_text(encoding="utf-8")
    assert '"cuda-reference" => ds41rt_api::ApiBackend::RealDs4Full' not in coordinator

    entry = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "entry.rs"
    ).read_text(encoding="utf-8")
    assert "real_full_nvfp4_short_k" not in entry
    assert "audit_real_full_nvfp4_short_k" not in entry
    assert 'args.backend == "real-ds4-full"' in entry
    assert (
        "target_device_storage: Arc<Mutex<DeepseekV4TargetDeviceStorage>>" in entry
    )
    assert (
        "target_device_storage: Option<Arc<Mutex<DeepseekV4TargetDeviceStorage>>>"
        not in entry
    )
    assert "target_device_identity: RealFullSchedulerNativeTargetIdentity" in entry
    assert "reset_glm_dsa_sparse_mla_transient_state" not in entry

    generic_kv = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "kv"
        / "device.rs"
    ).read_text(encoding="utf-8")
    assert "flashinfer_glm_dsa_sparse_mla_prefill_device_buffers" not in generic_kv
    assert "use_direct_glm_dsa_sparse_mla_prefill" not in generic_kv
    assert "dsa_index_k_cache_b12x" not in generic_kv
    assert "generic device KV attention no longer accepts inherited GLM DSA" in generic_kv
    assert "NativeTargetMappingOnly" in generic_kv
    assert "cuda-native-target-page-map" in generic_kv

    scheduler_execution = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "scheduler"
        / "execution.rs"
    ).read_text(encoding="utf-8")
    assert "new_native_target_mapping(device_kv_storage_config)" in scheduler_execution
    assert (
        "pub(in crate::commands::real_full) native_target: "
        "RealFullSchedulerNativeTargetContext"
        in scheduler_execution
    )
    assert (
        "pub(in crate::commands::real_full) native_target: "
        "Option<RealFullSchedulerNativeTargetContext>"
        not in scheduler_execution
    )
    assert "validate_live_native_scheduler_contract(&kv_config, catalog)?;" in scheduler_execution
    assert scheduler_execution.count(
        "native_target.validate_for_model(&catalog.facts)?;"
    ) == 2
    assert (
        "packed KV snapshots are unavailable for native DeepSeek target serving"
        in entry
    )
    assert "generic_payload_allocated_bytes=0" in entry
    assert "generic_payload_avoided_bytes={}" in entry
    assert (
        "strict DeepSeek V4 TP4 serving requires tcp, tcp-debug-json, or verbs-host"
        in entry
    )
    assert "strict DeepSeek V4 TP4 sparse dispatch requires exactly" in entry
    assert "real_full_scheduler_execution_for_shape_with_state" not in entry
    assert "real_full_scheduler_execution_for_shape_with_sparse_tcp(" not in entry
    assert (
        "sparse_tcp_targets: Vec<TcpProtocolV2HostBatchTarget>"
        in entry
    )
    assert (
        "sparse_tcp_dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>"
        in entry
    )
    assert (
        "sparse_tcp_dispatch_worker: Option<Arc<RealFullSchedulerSparseTcpDispatchWorker>>"
        not in entry
    )
    assert "DS41RT_REAL_FULL_SERVE_FAST_TOKEN" not in entry
    assert "serve-fast-token-embedding-lm-head" not in entry
    assert "fast_embedding_lm_head_token_info" not in entry

    serving_launcher = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    assert "DS41RT_REAL_FULL_SERVE_FAST_TOKEN" not in serving_launcher

    api_backend = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-api"
        / "src"
        / "backends"
        / "real_full.rs"
    ).read_text(encoding="utf-8")
    assert "serve-fast-token-embedding-lm-head" not in api_backend
    assert "|| !full.scheduler_full_context_device_attention_complete" in api_backend
    assert "|| !full.scheduler_terminal_lm_head_uses_final_decode_device_hidden" in api_backend
    assert "|| !full.scheduler_terminal_lm_head_covers_full_vocabulary" in api_backend

    scheduler_admission = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "scheduler"
        / "execution"
        / "admission.rs"
    ).read_text(encoding="utf-8")
    assert (
        "if !native_target && scheduler_device_kv_readback_validation_enabled()"
        in scheduler_admission
    )

    coordinator_attention = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "coordinator_kernels"
        / "attention.rs"
    ).read_text(encoding="utf-8")
    for retired_symbol in (
        "GlmDsaSparseMlaPrefill",
        "flashinfer_glm_dsa_sparse_mla_prefill",
        "GLM_DSA_PREFILL",
        "LayerGlmDsaSparseMlaPrefill",
    ):
        assert retired_symbol not in coordinator_attention

    coordinator_kernels = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "coordinator_kernels"
        / "mod.rs"
    ).read_text(encoding="utf-8")
    assert "LayerGlmDsaSparseMlaPrefill" not in coordinator_kernels

    mla_capture = (
        ROOT
        / "python"
        / "reference"
        / "ds41rt_reference"
        / "b12x_mla_capture.py"
    ).read_text(encoding="utf-8")
    assert "prepare_b12x_glm_dsa_indexer_prefill" not in mla_capture
    assert "capture_b12x_glm_dsa_indexer_prefill" not in mla_capture
    assert "_B12X_GLM_DSA_INDEXER_STATES" not in mla_capture

    retired_native_dsa = (
        ROOT / "rust" / "crates" / "ds41rt-ffi" / "src" / "lib.rs",
        ROOT / "native" / "include" / "ds41rt_native.h",
        ROOT / "native" / "cuda" / "kernels" / "mla_indexing.cu",
        ROOT / "native" / "src" / "ds41rt_native.cc",
    )
    assert not (ROOT / "native" / "cuda" / "kernels" / "dsa_indexer.cu").exists()
    for path in retired_native_dsa:
        source = path.read_text(encoding="utf-8")
        for retired_symbol in (
            "glm_dsa_query_prepare_b12x",
            "glm_dsa_prefill_metadata",
            "glm_dsa_sort_selected_indices",
            "glm_dsa_index_k_pack_b12x",
            "glm_dsa_page_table",
            "target_kv_page_table_expand_indices",
        ):
            assert retired_symbol not in source

    ffi_source = retired_native_dsa[0].read_text(encoding="utf-8")
    assert "DS41RT_CUDA_GENERIC_KV_PAGE_SIZE" in ffi_source
    assert "cuda_generic_kv_page_table_expand_indices_async" in ffi_source

    kv_snapshot = (
        ROOT
        / "rust"
        / "crates"
        / "ds41rt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "scheduler"
        / "execution"
        / "snapshot.rs"
    ).read_text(encoding="utf-8")
    assert 'REAL_FULL_KV_SNAPSHOT_FORMAT: &str = "ds41rt-kv-v3"' in kv_snapshot
    assert "dsa_index_file" not in kv_snapshot

    launcher = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    assert 'case "$kv_cache_dtype" in' in launcher
    assert (
        "DS41RT_REAL_FULL_SERVE_KV_CACHE_DTYPE must be fp8"
        in launcher
    )
