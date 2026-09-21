"""CPU-only tests for the six-rank candidate launch path.

Runs the real ``scripts/run-tp-ep-six-candidate.sh`` gate and the reused
``scripts/run-tp-ep-native-candidate.sh`` in ``plan`` mode and in mocked
``start`` mode. The fakes mirror the real emit ordering: the RDMA GID/device line
appears only after a client connection (initialized by the bounded first request)
AND when the transport timing diagnostic was forwarded, so a pre-connection
capture cannot pass against fabricated evidence.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "run-tp-ep-six-candidate.sh"
LAUNCHER = ROOT / "scripts" / "run-tp-ep-native-candidate.sh"
FIX = ROOT / "scripts" / "fixtures" / "tp-ep-six"
HOSTS = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
BUDGET = "109119320064"
DEVICE_MAP = "10.55.0.22=mlx5_0,10.55.0.5=rocep1s0f0,10.55.0.11=roceP2p1s0f0,10.55.0.6=rocep1s0f0,10.55.0.12=roceP2p1s0f0"

FAKE_DOCKER = """#!/usr/bin/env bash
log="${FAKE_LOG:?}"; { printf 'docker'; printf ' %q' "$@"; printf '\\n'; } >> "$log"
[[ "$*" == *"DS41RT_PROTOCOL_V2_TCP_TIMING="* ]] && touch "${log}.timing"
[[ "${1:-}" == "inspect" ]] && { echo true; exit 0; }
if [[ "${1:-}" == "exec" ]]; then
  args=("$@"); i=1
  while (( i < ${#args[@]} )); do
    case "${args[$i]}" in
      -d) i=$((i+1)) ;;
      -e) i=$((i+2)) ;;
      *) break ;;
    esac
  done
  container="${args[$i]:-}"; i=$((i+1))
  cmd=("${args[@]:i}")
  if [[ "${cmd[0]:-}" == "test" ]]; then
    [[ "${cmd[1]:-}" == "-e" ]] && exit 1   # stale placement dir: does not exist
    exit 0                                   # wip-process existence check: exists
  fi
  if [[ "${cmd[0]:-}" == "cat" ]]; then
    # The published plan must echo the launch's own layout; the harness writes
    # that JSON to FAKE_PLAN_FILE so this fake never has to build JSON in shell.
    cat "${FAKE_PLAN_FILE:?}"; exit 0
  fi
  for token in "${cmd[@]}"; do [[ "$token" == "status" ]] && { echo "running 4321"; exit 0; }; done
  exit 0
fi
exit 0
"""

FAKE_SSH = """#!/usr/bin/env bash
log="${FAKE_LOG:?}"; { printf 'ssh'; printf ' %q' "$@"; printf '\\n'; } >> "$log"
[[ "$*" == *"DS41RT_PROTOCOL_V2_TCP_TIMING="* ]] && touch "${log}.timing"
args=("$@"); i=0; host=""
while (( i < ${#args[@]} )); do
  case "${args[$i]}" in
    -o) i=$((i+2)) ;;
    -*) i=$((i+1)) ;;
    *) host="${args[$i]}"; i=$((i+1)); break ;;
  esac
done
remote="${args[*]:i}"
case "$remote" in
  *"docker inspect"*) echo true; exit 0 ;;
  *"if [ -e"*) echo 0; exit 0 ;;
  *status*) echo "running 4321"; exit 0 ;;
  *"tail -c"*)
    case "$host" in
      ostrich) rank=0 ;; dodo) rank=1 ;; emu) rank=2 ;;
      kiwi) rank=3 ;; rhea) rank=4 ;; moa) rank=5 ;; *) rank=9 ;;
    esac
    # Structured startup evidence, ANSI-decorated like the real log, plus the
    # stale-line variants that must NOT satisfy the readiness wait.
    printf 'INFO ds41rt: \033[2mnative local RoCE expert worker ready\033[0m rank=%s world=6 \033[3mrole\033[0m=%s \033[3mintermediate\033[0m=%s first_layer=%s\n' \
      "$rank" "${FAKE_WORKER_ROLE:-7}" "${FAKE_WORKER_INTERMEDIATE:-384}" "${FAKE_WORKER_LOG_FIRST:-20}"
    echo "native local RoCE expert worker ready rank=$rank world=6 first_layer=0"
    # Real behaviour: the endpoint (and its GID line) exists only after a client
    # connection, and the line is timing-gated by the forwarded diagnostic flag.
    if [[ -f "${log}.connected" && -f "${log}.timing" ]]; then
      echo "server_gid=00000000000000000000ffff0a370005 gid_index=5"
    fi
    exit 0 ;;
esac
exit 0
"""

FAKE_CURL = """#!/usr/bin/env bash
log="${FAKE_LOG:?}"; { printf 'curl'; printf ' %q' "$@"; printf '\\n'; } >> "$log"
case "$*" in
  *"/v1/chat/completions"*) touch "${log}.connected"; exit 0 ;;
  *"/v1/models"*) echo '{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash"}]}' ;;
esac
exit 0
"""


def _fake_bin(tmp_path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("ssh", FAKE_SSH), ("curl", FAKE_CURL)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _start_env(bin_dir: Path, log: Path, rtx_gpus: int = 2, layers: int = 20, first: int = 20) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_LOG"] = str(log)
    # The fake coordinator publishes a plan matching the launch under test.
    plan_file = log.parent / "fake-plan.json"
    plan_file.write_text(json.dumps(dict(version=1, rtx_gpus=rtx_gpus, nonce="fresh",
                                         rtx_expert_layers=layers, spark_first_layer=first)))
    env["FAKE_PLAN_FILE"] = str(plan_file)
    env["FAKE_WORKER_LOG_FIRST"] = str(first)
    env["DS41RT_TPEP_L3_GRANT"] = "1"
    env["DS41RT_TPEP_WORKER_READY_TIMEOUT_SECONDS"] = "20"
    env["DS41RT_SPARKINFER_SOURCE_DIR"] = "/workspace/ds41rt/third_party/sparkinfer"
    env["PYTHONPATH"] = "/workspace/ds41rt/third_party/sparkinfer"
    return env


def _plan(config: str, rtx_gpus: int) -> str:
    result = subprocess.run(
        ["bash", str(WRAPPER), "plan", "--config", str(FIX / config), "--rtx-gpus", str(rtx_gpus),
         "--host-artifact-root", str(ROOT / "runs" / "tp-ep-six")],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    return result.stdout


# 1. Plan renders the corrected ordered lifecycle and an honest prestage step.
def test_plan_renders_ordered_six_rank_lifecycle() -> None:
    dual = _plan("site-2rtx6-tp3ep2.config", 2)
    steps = [line for line in dual.splitlines() if line.startswith("STEP ")]
    assert steps == ["STEP prestage-prerequisite (NOT performed by this launcher; run before start)",
                     "STEP coordinator-start", "STEP plan-read"] + \
        [f"STEP expert-start rank={index}" for index in range(6)] + \
        ["STEP worker-ready", "STEP plan-ack", "STEP readiness-gate",
         "STEP first-request", "STEP gid-capture"], steps
    assert "dspark=on dspark_draft_limit=7 (coordinator only)" in dual
    assert "tcp_timing=DS41RT_PROTOCOL_V2_TCP_TIMING=unset" in dual
    assert "PREREQ stage-0:" in dual and "NOT performed" in dual
    experts = [line for line in dual.splitlines() if line.startswith("CMD expert-")]
    assert len(experts) == 6
    assert all(f"ssh -o BatchMode=yes {host} " in line for line, host in zip(experts, HOSTS))
    assert "10.55.0.6:29441" in dual and "10.55.0.12:29441" not in dual

    single = _plan("site-1rtx6-tp3ep2.config", 1)
    single_steps = [line for line in single.splitlines() if line.startswith("STEP ")]
    assert single_steps == ["STEP prestage-prerequisite (NOT performed by this launcher; run before start)"] + \
        [f"STEP expert-start rank={index}" for index in range(6)] + \
        ["STEP worker-ready", "STEP coordinator-start", "STEP readiness-gate",
         "STEP first-request", "STEP gid-capture"], single_steps
    assert "dspark=on dspark_draft_limit=5 (coordinator only)" in single


# 2. Wrapper guards: overflow-safe budget bound, exact named arms, missing key.
def test_six_wrapper_rejects_bad_fixture_inputs(tmp_path) -> None:
    bin_dir, log = _fake_bin(tmp_path), tmp_path / "calls.log"
    base = (FIX / "site-2rtx6-tp3ep2.config").read_text()
    cases = {
        "budget-over": (base.replace(f"SPARK_DEVICE_BUDGET_BYTES={BUDGET}", "SPARK_DEVICE_BUDGET_BYTES=109119320065"),
                        "exceeds the global minimum-of-six"),
        "budget-2pow64": (base.replace(f"SPARK_DEVICE_BUDGET_BYTES={BUDGET}",
                                       "SPARK_DEVICE_BUDGET_BYTES=18446744073709551616"),
                          "exceeds the global minimum-of-six"),
        "budget-zero": (base.replace(f"SPARK_DEVICE_BUDGET_BYTES={BUDGET}", "SPARK_DEVICE_BUDGET_BYTES=0"),
                        "must be positive"),
        "arm-1rtx-tp2ep3": (base.replace("SPARK_TP=3", "SPARK_TP=2").replace("SPARK_EP=2", "SPARK_EP=3")
                            .replace("RTX_GPUS=2", "RTX_GPUS=1").replace("RTX_EXPERT_LAYERS=20", "RTX_EXPERT_LAYERS=0"),
                            "approved six-rank arms"),
        # A two-shard or non-unity-group TP6 combination is not a supported arm.
        "arm-tp6ep2": (base.replace("SPARK_TP=3", "SPARK_TP=6"),
                       "must be 6"),
        "arm-tp6ep3": (base.replace("SPARK_TP=3", "SPARK_TP=6").replace("SPARK_EP=2", "SPARK_EP=3"),
                       "must be 6"),
        "missing-count": (base.replace("SPARK_COUNT=6\n", ""), "SPARK_COUNT must be 6"),
    }
    for name, (text, expected) in cases.items():
        config = tmp_path / f"{name}.config"
        config.write_text(text)
        result = subprocess.run(
            ["bash", str(WRAPPER), "start", "--config", str(config), "--rtx-gpus", "2",
             "--first-layer", "20", "--run-id", name, "--host-artifact-root", str(ROOT / "runs" / "tp-ep-six")],
            capture_output=True, text=True, cwd=ROOT, env=_start_env(bin_dir, log))
        assert result.returncode == 2 and expected in result.stderr, (name, result.stderr)
    assert not log.exists() or log.read_text() == "", "a rejected config must send no remote command"


def _mock_start(tmp_path, config: str, rtx_gpus: int, first_layer: int, run_id: str, tcp_timing: bool,
                plan_layers: int | None = None):
    bin_dir, log = _fake_bin(tmp_path), tmp_path / "calls.log"
    command = ["bash", str(WRAPPER), "start", "--config", str(FIX / config), "--rtx-gpus", str(rtx_gpus),
               "--first-layer", str(first_layer), "--run-id", run_id, "--host-device-map", DEVICE_MAP,
               "--host-artifact-root", str(ROOT / "runs" / "tp-ep-six")]
    if tcp_timing:
        command += ["--tcp-timing", "1"]
    layers = plan_layers if plan_layers is not None else first_layer
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                            env=_start_env(bin_dir, log, rtx_gpus, layers, first_layer))
    calls = [line.replace("\\", "") for line in log.read_text().splitlines()] if log.exists() else []
    launches = [line for line in calls if "exec -d" in line
                and ("candidate-coordinator-18000" in line or "candidate-expert-29441" in line)]
    return result, calls, launches


# 3. Mocked dual start: ACK → health → first request → GID, seven launches in order.
def test_mock_start_dual_orders_ack_health_request_gid(tmp_path) -> None:
    result, calls, launches = _mock_start(tmp_path, "site-2rtx6-tp3ep2.config", 2, 20, "mockdual", True)
    assert result.returncode == 0, result.stderr
    assert len(launches) == 7, launches
    coordinator, experts = launches[0], launches[1:]
    assert "--placement-directory" in coordinator and "--dspark-draft-limit 7" in coordinator
    assert "--kv-pool-size 5037542400" in coordinator and "--rtx-expert-layers 20" in coordinator
    for index, line in enumerate(experts):
        assert line.startswith("ssh ") and HOSTS[index] in line
        assert f"--rank {index} " in line and "--world 6" in line and "--first-layer 20" in line
        assert f"--device-budget-bytes {BUDGET}" in line
    for line in launches:
        assert f"DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP={DEVICE_MAP}" in line, line
        assert "DS41RT_PROTOCOL_V2_TCP_TIMING=1" in line, line
        assert "docker run" not in line and "build.sh" not in line
    ack = calls.index(next(line for line in calls if "ready.json" in line))
    health = calls.index(next(line for line in calls if "/health" in line))
    first_request = calls.index(next(line for line in calls if "/v1/chat/completions" in line))
    assert calls.index(experts[-1]) < ack < health < first_request
    binding = ROOT / "runs" / "tp-ep-six" / "gid-binding-mockdual.txt"
    assert binding.exists() and "gid_index=5" in binding.read_text()
    binding.unlink()
    (tmp_path / "calls.log.connected").unlink(missing_ok=True)
    (tmp_path / "calls.log.timing").unlink(missing_ok=True)


# 4. Pre-connection negative: without the timing diagnostic the GID capture fails closed.
def test_mock_start_without_timing_fails_closed_on_gid(tmp_path) -> None:
    result, calls, _ = _mock_start(tmp_path, "site-2rtx6-tp3ep2.config", 2, 20, "mocknotiming", False)
    assert result.returncode != 0
    assert "logged no RDMA GID/device binding" in result.stderr, result.stderr
    binding = ROOT / "runs" / "tp-ep-six" / "gid-binding-mocknotiming.txt"
    assert not binding.exists() or "gid_index" not in binding.read_text()
    binding.unlink(missing_ok=True)
    for path in (tmp_path / "calls.log.connected", tmp_path / "calls.log.timing"):
        path.unlink(missing_ok=True)


# 5. Mocked single-RTX start: six workers at layer 0 BEFORE the coordinator, no handoff.
def test_mock_start_single_arm_orders_workers_first(tmp_path) -> None:
    result, calls, launches = _mock_start(tmp_path, "site-1rtx6-tp3ep2.config", 1, 0, "mocksingle", True)
    assert result.returncode == 0, result.stderr
    assert len(launches) == 7, launches
    workers, coordinator = launches[:6], launches[6]
    assert "--rtx-gpus 1" in coordinator and "--dspark-draft-limit 5" in coordinator
    assert "--rtx-expert-layers 0" in coordinator
    assert "--placement-directory" not in coordinator and "--kv-pool-size" not in coordinator
    for index, line in enumerate(workers):
        assert line.startswith("ssh ") and HOSTS[index] in line
        assert f"--rank {index} " in line and "--world 6" in line and "--first-layer 0" in line
    first_request = calls.index(next(line for line in calls if "/v1/chat/completions" in line))
    assert calls.index(workers[-1]) < calls.index(coordinator) < first_request
    binding = ROOT / "runs" / "tp-ep-six" / "gid-binding-mocksingle.txt"
    assert binding.exists() and "gid_index=5" in binding.read_text()
    binding.unlink()
    for path in (tmp_path / "calls.log.connected", tmp_path / "calls.log.timing"):
        path.unlink(missing_ok=True)


# 6b. Single-RTX explicit-topology TP6 with local routed layers: the coordinator
#     must publish a plan, be started exactly ONCE, start workers at the published
#     boundary, and acknowledge exactly one plan. This is the case where gating the
#     ack on rtx_gpus == 2 double-started the coordinator and never acknowledged.
def test_mock_start_single_rtx_local_count_acknowledges_once(tmp_path) -> None:
    result, calls, launches = _mock_start(tmp_path, "site-1rtx6-tp6ep1.config", 1, 5, "mocktp6local", True)
    assert result.returncode == 0, result.stderr
    assert len(launches) == 7, launches
    # With a handoff the coordinator starts first (to publish the plan) and the
    # workers follow at the published boundary.
    coordinator = next(line for line in launches if "candidate-coordinator" in line)
    workers = [line for line in launches if "candidate-expert" in line]
    assert len(workers) == 6, launches
    assert launches.index(coordinator) < launches.index(workers[0]), launches
    assert "--rtx-gpus 1" in coordinator
    assert "--rtx-expert-layers 5" in coordinator
    assert "--placement-directory" in coordinator
    assert "--spark-tp 6" in coordinator and "--spark-ep 1" in coordinator
    for index, line in enumerate(workers):
        assert line.startswith("ssh ") and HOSTS[index] in line
        assert f"--rank {index} " in line and "--world 6" in line and "--first-layer 5" in line
        assert "--spark-tp 6" in line and "--spark-ep 1" in line
    # Exactly one coordinator start for the whole run.
    starts = [line for line in calls if "exec -d" in line and "candidate-coordinator-18000" in line]
    assert len(starts) == 1, starts
    # Exactly one plan acknowledgement, and it happens after the workers start.
    acks = [line for line in calls if "ready.json" in line and "plan.json" in line]
    assert len(acks) == 1, acks
    first_request = calls.index(next(line for line in calls if "/v1/chat/completions" in line))
    assert calls.index(workers[-1]) < calls.index(acks[0]) < first_request
    binding = ROOT / "runs" / "tp-ep-six" / "gid-binding-mocktp6local.txt"
    binding.unlink(missing_ok=True)
    for path in (tmp_path / "calls.log.connected", tmp_path / "calls.log.timing"):
        path.unlink(missing_ok=True)


# 6. Pure unreplicated TP6 (SPARK_TP=6 SPARK_EP=1) is an approved six-rank arm:
#    six disjoint intermediate slices of every expert, one unreplicated group.
def test_tp6_dual_arm_is_approved_and_forwards_the_pure_topology(tmp_path) -> None:
    result, calls, launches = _mock_start(tmp_path, "site-2rtx6-tp6ep1.config", 2, 20, "mocktp6", True)
    assert result.returncode == 0, result.stderr
    assert len(launches) == 7, launches
    coordinator, experts = launches[0], launches[1:]
    assert "--spark-tp 6" in coordinator and "--spark-ep 1" in coordinator
    assert "--rtx-expert-layers 20" in coordinator and "--world 6" not in coordinator
    for index, line in enumerate(experts):
        assert line.startswith("ssh ") and HOSTS[index] in line
        assert f"--rank {index} " in line and "--world 6" in line
        assert "--spark-tp 6" in line and "--spark-ep 1" in line
        assert "--first-layer 20" in line
    binding = ROOT / "runs" / "tp-ep-six" / "gid-binding-mocktp6.txt"
    binding.unlink(missing_ok=True)
    for path in (tmp_path / "calls.log.connected", tmp_path / "calls.log.timing"):
        path.unlink(missing_ok=True)
