#!/usr/bin/env bash
# Plan/command rendering for scripts/run-tp-ep-native-candidate.sh.
#
# Sourced after scripts/release-common.sh by the candidate launcher. The
# functions read the launcher's resolved variables and the argv builders, so
# they must be called only after configuration resolution.
#
# candidate_print_plan emits a deterministic, parseable plan with no Docker,
# SSH, GPU or network action.

candidate_print_plan() {
  echo "mode=plan"
  echo "run_id=$run_id"
  echo "topology tp=$spark_tp ep=$spark_ep world=$spark_world"
  echo "rtx_gpus=$rtx_gpus"
  echo "first_layer=${resolved_first_layer:-pending}"
  echo "budget_bytes=$SPARK_DEVICE_BUDGET_BYTES"
  echo "admission=$admission"
  echo "expert_capacity=$expert_capacity (from PREFILL_BATCH_TOKENS=$PREFILL_BATCH_TOKENS)"
  echo "coordinator_container=$coordinator_container"
  echo "spark_container=$spark_container"
  echo "api_port=$api_port expert_port=$expert_port"
  echo "rail=A-only (candidate CLI has no secondary-rail endpoints; single rail, no fake dual)"
  echo "daemon_bin=$daemon_bin"
  echo "snapshot=$snapshot"
  echo "coordinator_native_lib=$coordinator_native_lib"
  echo "spark_native_lib=$spark_native_lib"
  echo "spark_lib_source=$spark_lib_source"
  echo "native_lib_env=coordinator:DS41RT_NATIVE_LIB=$coordinator_native_lib spark:DS41RT_NATIVE_LIB=$spark_native_lib"
  echo "artifact_contract=per-role daemon=$daemon_bin coordinator_lib=$coordinator_native_lib spark_lib=$spark_native_lib (per-host existence+arch verified on start; uniform Spark staging step below)"
  echo "wip_runtime_root=$wip_runtime_root"
  echo "coordinator_cuda_visible_devices=${coordinator_cuda_visible_devices:-unset} (coordinator container only)"
  echo "placement_dir=$placement_dir"
  echo "expect_spark_lib_sha256=${expect_spark_lib_sha256:-unset}"
  echo "role_manifest=${role_manifest:-unset}"
  echo "qualified=$qualified require_arch=$require_arch"
  echo "dspark=$DSPARK dspark_draft_limit=$dspark_draft_limit (coordinator only)"
  echo "host_device_map=DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=${DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-unset} (required on both roles)"
  echo "tcp_timing=DS41RT_PROTOCOL_V2_TCP_TIMING=${DS41RT_PROTOCOL_V2_TCP_TIMING:-unset} (startup diagnostic; unset means the RC GID line is not emitted)"
  echo "sparkinfer_source=${DS41RT_SPARKINFER_SOURCE_DIR:-unset} pythonpath=${PYTHONPATH:-unset}"
  echo "ssh_bind=${ssh_bind_ip:-unset} ssh_targets=$(for i in "${!hosts[@]}"; do printf '%s=%s ' "$i" "${hosts[$i]}"; done)"
  echo "STEP prestage-prerequisite (NOT performed by this launcher; run before start)"
  for rank in "${!hosts[@]}"; do
    echo "PREREQ stage-$rank: host=${hosts[$rank]} install -D -m644 $spark_lib_source_host $spark_lib_stage_host && verify aarch64+sha; container reads $spark_native_lib (built source $spark_lib_source)"
  done
  local planned_first="${resolved_first_layer:-<from-plan>}"
  if [[ "$rtx_gpus" == 2 ]]; then
    echo "STEP coordinator-start"
    echo "CMD coordinator: $(coordinator_command_string)"
    echo "STEP plan-read"
    echo "CMD plan-read: docker exec $coordinator_container cat $placement_dir/plan.json"
  fi
  for rank in "${!hosts[@]}"; do
    echo "STEP expert-start rank=$rank"
    echo "CMD expert-$rank: $(expert_command_string "$rank" "${hosts[$rank]}" "$planned_first")"
  done
  echo "STEP worker-ready"
  for rank in "${!hosts[@]}"; do
    echo "CMD worker-ready-$rank: host=${hosts[$rank]} log=$wip_runtime_root/$expert_process.log offset=<pre-start byte size> match='native local RoCE expert worker ready ... rank=$rank world=$spark_world first_layer=${resolved_first_layer:-<from-plan>}'"
  done
  if [[ "$rtx_gpus" == 1 ]]; then
    echo "STEP coordinator-start"
    echo "CMD coordinator: $(coordinator_command_string)"
  fi
  if [[ "$rtx_gpus" == 2 ]]; then
    echo "STEP plan-ack"
    echo "CMD plan-ack: docker exec $coordinator_container sh -c 'cp $placement_dir/plan.json $placement_dir/.ready-pending && mv $placement_dir/.ready-pending $placement_dir/ready.json'"
  fi
  echo "STEP readiness-gate"
  local probe_host="$listen_host"; [[ "$probe_host" != "0.0.0.0" ]] || probe_host=127.0.0.1
  echo "CMD readiness-api: curl -fsS http://$probe_host:$api_port/health && curl -fsS http://$probe_host:$api_port/v1/models"
  echo "CMD readiness-workers: docker exec -e DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root $coordinator_container $wip_process status $coordinator_process; per Spark rank: -e DS41RT_WIP_RUNTIME_ROOT=$wip_runtime_root $spark_wip_process status $expert_process"
  echo "STEP first-request"
  echo "CMD first-request: curl -fsS -m 120 -H 'Content-Type: application/json' -d '<bounded max_tokens=1 payload>' http://$probe_host:$api_port/v1/chat/completions (initializes the lazy per-connection RDMA endpoints)"
  echo "STEP gid-capture"
  echo "CMD gid-capture: AFTER the first request, per rank grep the current process log slice for gid_index=/client_gid=/server_gid=; fail closed if absent; write $repo_root/runs/tp-ep-six/gid-binding-<run_id>.txt"
}
