#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/phase0-spark-tcp-bench.sh

Stages this repo on the Spark hosts, starts Spark-hosted binary ProtocolV2
ds41rt expertd containers, then runs benchmarks/phase0_bench.py with explicit
Spark targets and an expected ProtocolV2 executor id.

Environment:
  DS41RT_SPARK_HOSTS                 default: ostrich,dodo,emu,kiwi
  DS41RT_PHASE0_SPARK_EXPERT_MODE    real or synthetic; default: real
  DS41RT_SPARK_IMAGE                 default: ds41rt-spark-expert-dev
  DS41RT_SPARK_IMAGE_COPY_METHOD     spark-netcat, ssh-relay, or none; default: spark-netcat
  DS41RT_SPARK_IMAGE_SEED_HOST       default: first host with DS41RT_SPARK_IMAGE
  DS41RT_SPARK_IMAGE_LINK_SUFFIX     default: .200gb for spark-netcat data path
  DS41RT_SPARK_IMAGE_COPY_PORT       default: 29420
  DS41RT_SPARK_BUILD_IMAGE           set 1 to build remotely when no seed image exists
  DS41RT_SPARK_FORCE_BUILD_IMAGE     set 1 to rebuild remotely even when image exists
  DS41RT_SPARK_IMAGE_ONLY            set 1 to stage/ensure images and exit before starting experts
  DS41RT_SPARK_BUILD_PROFILE         debug or release; default: release
  DS41RT_SPARK_PREBUILT              use prebuilt artifacts instead of building
                                      mounted source
  DS41RT_SPARK_PREBUILT_BIN          default: /opt/ds41rt/bin/ds41rt
  DS41RT_SPARK_PREBUILT_NATIVE_LIB   default: /opt/ds41rt/lib/libds41rt_native.so
  DS41RT_SPARK_USE_DIAGNOSTIC_PLACEMENT
                                      set 1 to pass generated catalog/loadplan
                                      files; default: 0. Normal serving resolves
                                      the checkpoint directly and infers only
                                      logical spark-0..3 owner metadata.
  DS41RT_SPARK_EXISTING_CONTAINER    execute inside this persistent dev
                                      container instead of creating one
  DS41RT_SPARK_RUNTIME_CACHE_DIR     persistent host/container cache root;
                                      defaults to /wip/cache for WIP containers
                                      and ~/.cache/ds41rt for release containers
  DS41RT_SPARK_GPU_RUNTIME           nvidia or manual; default: nvidia
  DS41RT_SPARK_WORKDIR               default: $HOME/ds41rt-phase0-spark-bench
  DS41RT_SPARK_SKIP_STAGE            set 1 when the committed source is already staged remotely
  DS41RT_SPARK_KEEP_EXPERTS          set 1 to leave remote containers running
  DS41RT_SPARK_EXPERT_PORT           default: 9100
  DS41RT_SPARK_EXPERT_TRANSPORT      tcp or verbs-host; default: tcp
  DS41RT_VERBS_APP_IB_PORT_NUM       optional verbs-host IB port override
  DS41RT_PHASE0_TCP_LAYER_ID         default: 3
  DS41RT_SPARK_EXPERT_REAL_LAYER     layer id, all, none, or empty; default: DS41RT_PHASE0_TCP_LAYER_ID
  DS41RT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS
                                      default: 1 for all-layer real daemon startup, else 0
  DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW
                                      default: 1 for real daemon startup
  DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS
                                      forward to real daemon; default: 1 for real mode, else 0
  DS41RT_B12X_SPARK_AOT               build direct SparkInfer AOT kernels; default: 1 for real mode
  DS41RT_DS4_FLASH_SPARK_AOT          build DeepSeek V4 Flash TP4 AOT kernels;
                                      default: 1 for real mode
  DS41RT_B12X_SPARK_ROUTE_LANES       concurrent direct-SparkInfer lanes in 1..4; default: 4
  DS41RT_B12X_SPARK_GROUPED_DECODE    grouped TP4 M=1/topk=8 decode; default: 1
  DS41RT_B12X_SPARK_W4A16_DEVICE_WEIGHTS
                                      cudaMalloc weight/scale slabs; default: 1
  DS41RT_B12X_SPARK_W4A16_DECODE_GRID_X process-start decode kernel grid override in 1..96; default: 32
  DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM  atomic top-k accumulation for M=1; default: 1 in real mode
  DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE  split-m1, wide, or ordered for M=2..8; default: wide
  DS41RT_EXPERT_INTERMEDIATE_SHARDS   1 or 4; default: 4 in real mode, 1 synthetic
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid; default: spark-rdma for real verbs-host, coordinator otherwise
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE bf16, fp8, or nvfp4; default: fp8
  DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE
                                      small-row owner dtype; default: bf16
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION_ROOT default: first Spark on the image link
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION_PORT default: 9200
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS default: 16 (spark/spark-hybrid) or 1 (spark-owner)
  DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION
                                      partition rows across Spark ranks before reduction; default: 0
  DS41RT_EXPERT_FUSED_FP8_REDUCTION fuse full-row FP8 root combine; default: 1
  DS41RT_EXPERT_NCCL_BF16_REDUCE     experimental root-only BF16 reduce before FP8 response; default: 0
  DS41RT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS fused small-batch default: 8
  DS41RT_EXPERT_INTERMEDIATE_OWNER_PORT default: DS41RT_SPARK_EXPERT_PORT
  DS41RT_EXPERT_INTERMEDIATE_OWNER_PEERS rank-ordered host:port CSV; default: Spark image-link hosts
  DS41RT_EXPERT_INTERMEDIATE_RDMA_PEERS rank-ordered host CSV; default: Spark image-link hosts
  DS41RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS
                                      rank-ordered secondary-rail host/IP CSV; defaults to
                                      10.55.0.5,.6,.7,.8 for the four known Sparks
  DS41RT_EXPERT_INTERMEDIATE_RDMA_DEVICES
                                      local ibverbs devices by rail; default: rocep1s0f0,roceP2p1s0f0
  DS41RT_EXPERT_INTERMEDIATE_RDMA_PORT  base for pair/rail ports; default: 9400
  DS41RT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES mapped ring slot capacity; default: 4194304
  DS41RT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH mapped ring slots per peer; default: 4
  DS41RT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES
                                      minimum per-peer payload to split; default: 262144
  DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES
                                      concurrent RDMA/NCCL request lanes in 1..8; default: 4
  DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
                                      full-top8 requests bypassing the general CPU
                                      row/group planner, in 8..2048; default: 2048
  DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP
                                      generated per Spark from RDMA netdev IPs; maps each
                                      accepted control-plane destination IP to its ibverbs device
  DS41RT_SPARK_NCCL_SOCKET_IFNAME     default: enp1s0f0np0
  DS41RT_SPARK_NCCL_IB_HCA            default: =rocep1s0f0,roceP2p1s0f0
  DS41RT_SPARK_NCCL_CROSS_NIC         0, 1, or topology-aware 2; default: 2
  DS41RT_SPARK_NCCL_NETDEVS_POLICY    AUTO, ALL, or MAX:N; default: ALL
  DS41RT_SPARK_NCCL_IB_MERGE_NICS     0 or 1; default: 1
  DS41RT_SPARK_NCCL_P2P_NET_CHUNKSIZE grouped send/recv chunk bytes; default: 131072
  DS41RT_SPARK_NCCL_DEBUG             default: WARN
  DS41RT_SPARK_NCCL_LAUNCH_ORDER_IMPLICIT
                                      order concurrent communicators; default: 1
  DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING
                                      forward to real daemon; default: 0
  DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING
                                      forward to real daemon; default: 0
  DS41RT_REAL_FULL_CUDA_ROUTE_VALIDATE forward CPU route validation; default: 0
  DS41RT_SPARK_SYNC_MODEL_CACHE       set 1 to copy the selected local Hugging
                                      Face model cache to missing Spark hosts;
                                      default: 0
  DS41RT_SPARK_MODEL_SYNC_ONLY        set 1 to stop after model-cache sync
  DS41RT_PHASE0_SPARK_SKIP_BENCH     set 1 to start/wait for experts without running the benchmark
EOF
  exit 0
fi

hosts_csv="${DS41RT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
model_id="${DS41RT_MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
model_revision="${DS41RT_MODEL_REVISION:-}"
wip_allow_historical_exl3_control="${DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL:-0}"
sync_model_cache="${DS41RT_SPARK_SYNC_MODEL_CACHE:-0}"
model_sync_only="${DS41RT_SPARK_MODEL_SYNC_ONLY:-0}"
mode="${DS41RT_PHASE0_SPARK_EXPERT_MODE:-real}"
remote_dir="${DS41RT_SPARK_WORKDIR:-}"
skip_stage="${DS41RT_SPARK_SKIP_STAGE:-0}"
image="${DS41RT_SPARK_IMAGE:-ds41rt-spark-expert-dev}"
image_copy_method="${DS41RT_SPARK_IMAGE_COPY_METHOD:-spark-netcat}"
image_seed_host="${DS41RT_SPARK_IMAGE_SEED_HOST:-}"
image_link_suffix="${DS41RT_SPARK_IMAGE_LINK_SUFFIX:-.200gb}"
image_copy_port="${DS41RT_SPARK_IMAGE_COPY_PORT:-29420}"
build_missing_image="${DS41RT_SPARK_BUILD_IMAGE:-0}"
force_build_image="${DS41RT_SPARK_FORCE_BUILD_IMAGE:-0}"
image_only="${DS41RT_SPARK_IMAGE_ONLY:-0}"
build_profile="${DS41RT_SPARK_BUILD_PROFILE:-release}"
prebuilt="${DS41RT_SPARK_PREBUILT:-0}"
prebuilt_bin="${DS41RT_SPARK_PREBUILT_BIN:-/opt/ds41rt/bin/ds41rt}"
prebuilt_native_lib="${DS41RT_SPARK_PREBUILT_NATIVE_LIB:-/opt/ds41rt/lib/libds41rt_native.so}"
use_diagnostic_placement="${DS41RT_SPARK_USE_DIAGNOSTIC_PLACEMENT:-0}"
existing_container="${DS41RT_SPARK_EXISTING_CONTAINER:-}"
runtime_cache_dir="${DS41RT_SPARK_RUNTIME_CACHE_DIR:-}"
gpu_runtime="${DS41RT_SPARK_GPU_RUNTIME:-nvidia}"
port="${DS41RT_SPARK_EXPERT_PORT:-9100}"
expert_transport="${DS41RT_SPARK_EXPERT_TRANSPORT:-tcp}"
verbs_ib_port_num="${DS41RT_VERBS_APP_IB_PORT_NUM:-}"
if [ "$use_diagnostic_placement" = "1" ]; then
  catalog="${DS41RT_PHASE0_SPARK_CATALOG:-.ds41rt-cache/model-artifacts/diagnostic/model_catalog.json}"
else
  # Production resolves tensor metadata directly from the selected HF snapshot.
  # An explicit catalog/loadplan is reserved for diagnostic/repro runs because
  # its owner names may not match the runtime spark-0..3 control roles.
  catalog=""
fi
loadplan_dir="${DS41RT_PHASE0_SPARK_LOADPLAN_DIR:-.ds41rt-cache/model-artifacts/diagnostic}"
layer_id="${DS41RT_PHASE0_TCP_LAYER_ID:-3}"
expert_real_layer="${DS41RT_SPARK_EXPERT_REAL_LAYER:-$layer_id}"
keep_experts="${DS41RT_SPARK_KEEP_EXPERTS:-0}"
skip_bench="${DS41RT_PHASE0_SPARK_SKIP_BENCH:-0}"
if [ "${DS41RT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS+x}" ]; then
  managed_route_projections="$DS41RT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS"
else
  case "$expert_real_layer" in
    ""|all|none) managed_route_projections=1 ;;
    *) managed_route_projections=0 ;;
  esac
fi
if [ "${DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW+x}" ]; then
  grouped_multirow="$DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW"
elif [ "$mode" = "real" ]; then
  grouped_multirow=1
else
  grouped_multirow=0
fi
if [ "${DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS+x}" ]; then
  route_cuda_graphs="$DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS"
elif [ "$mode" = "real" ]; then
  route_cuda_graphs=1
else
  route_cuda_graphs=0
fi
if [ "${DS41RT_B12X_SPARK_AOT+x}" ]; then
  b12x_spark_aot="$DS41RT_B12X_SPARK_AOT"
elif [ "$mode" = "real" ]; then
  b12x_spark_aot=1
else
  b12x_spark_aot=0
fi
if [ "${DS41RT_DS4_FLASH_SPARK_AOT+x}" ]; then
  ds4_flash_spark_aot="$DS41RT_DS4_FLASH_SPARK_AOT"
elif [ "$mode" = "real" ]; then
  ds4_flash_spark_aot=1
else
  ds4_flash_spark_aot=0
fi
route_timing="${DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING:-0}"
route_cuda_event_timing="${DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING:-0}"
route_validate="${DS41RT_REAL_FULL_CUDA_ROUTE_VALIDATE:-0}"
protocol_v2_verbs_host_execution_lanes="${DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-4}"
protocol_v2_packed_direct_max_rows="${DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS:-2048}"
route_preload_io_workers="${DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS:-128}"
route_preload_cooperative="${DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE:-1}"
weight_preload_nccl_port="${DS41RT_EXPERT_WEIGHT_PRELOAD_NCCL_PORT:-9350}"
b12x_route_lanes="${DS41RT_B12X_SPARK_ROUTE_LANES:-4}"
b12x_grouped_decode="${DS41RT_B12X_SPARK_GROUPED_DECODE:-1}"
b12x_w4a16_device_weights="${DS41RT_B12X_SPARK_W4A16_DEVICE_WEIGHTS:-1}"
b12x_w4a16_decode_grid_x="${DS41RT_B12X_SPARK_W4A16_DECODE_GRID_X:-}"
b12x_w4a16_small_m_mode="${DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE:-wide}"
if [ "${DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM+x}" ]; then
  b12x_w4a16_m1_fused_sum="$DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM"
elif [ "$mode" = "real" ]; then
  b12x_w4a16_m1_fused_sum=1
else
  b12x_w4a16_m1_fused_sum=0
fi
if [ "${DS41RT_EXPERT_INTERMEDIATE_SHARDS+x}" ]; then
  intermediate_shards="$DS41RT_EXPERT_INTERMEDIATE_SHARDS"
elif [ "$mode" = "real" ]; then
  intermediate_shards=4
else
  intermediate_shards=1
fi
if [ "${DS41RT_EXPERT_INTERMEDIATE_REDUCTION+x}" ]; then
  intermediate_reduction="$DS41RT_EXPERT_INTERMEDIATE_REDUCTION"
elif [ "$mode" = "real" ] && [ "$expert_transport" = "verbs-host" ]; then
  intermediate_reduction=spark-rdma
else
  intermediate_reduction=coordinator
fi
intermediate_reduction_dtype="${DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE:-fp8}"
intermediate_owner_reduction_dtype="${DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE:-bf16}"
intermediate_reduction_port="${DS41RT_EXPERT_INTERMEDIATE_REDUCTION_PORT:-9200}"
if [ "${DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION+x}" ]; then
  intermediate_row_sharded_reduction="$DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"
else
  case "$intermediate_reduction" in
    spark-rdma|spark-rdma-hybrid) intermediate_row_sharded_reduction=1 ;;
    *) intermediate_row_sharded_reduction=0 ;;
  esac
fi
fused_fp8_reduction="${DS41RT_EXPERT_FUSED_FP8_REDUCTION:-1}"
nccl_bf16_reduce="${DS41RT_EXPERT_NCCL_BF16_REDUCE:-0}"
if [ "${DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS+x}" ]; then
  intermediate_reduction_min_rows="$DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS"
elif [ "$intermediate_reduction" = "spark-owner" ]; then
  intermediate_reduction_min_rows=1
else
  intermediate_reduction_min_rows=16
fi
intermediate_owner_max_rows="${DS41RT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS:-8}"
intermediate_owner_port="${DS41RT_EXPERT_INTERMEDIATE_OWNER_PORT:-$port}"
intermediate_rdma_port="${DS41RT_EXPERT_INTERMEDIATE_RDMA_PORT:-9400}"
intermediate_rdma_slot_bytes="${DS41RT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES:-4194304}"
intermediate_rdma_ring_depth="${DS41RT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH:-4}"
intermediate_rdma_additional_peers="${DS41RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS:-}"
intermediate_rdma_devices="${DS41RT_EXPERT_INTERMEDIATE_RDMA_DEVICES:-}"
intermediate_rdma_stripe_min_bytes="${DS41RT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES:-262144}"
nccl_socket_ifname="${DS41RT_SPARK_NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
nccl_ib_hca="${DS41RT_SPARK_NCCL_IB_HCA:-=rocep1s0f0,roceP2p1s0f0}"
nccl_cross_nic="${DS41RT_SPARK_NCCL_CROSS_NIC:-2}"
nccl_netdevs_policy="${DS41RT_SPARK_NCCL_NETDEVS_POLICY:-ALL}"
nccl_ib_merge_nics="${DS41RT_SPARK_NCCL_IB_MERGE_NICS:-1}"
nccl_p2p_net_chunksize="${DS41RT_SPARK_NCCL_P2P_NET_CHUNKSIZE:-131072}"
nccl_debug="${DS41RT_SPARK_NCCL_DEBUG:-WARN}"
nccl_launch_order_implicit="${DS41RT_SPARK_NCCL_LAUNCH_ORDER_IMPLICIT:-1}"
container_prefix="${DS41RT_SPARK_CONTAINER_PREFIX:-ds41rt-phase0-tcp-expertd}"

case "$mode" in
  real|synthetic) ;;
  *)
    echo "DS41RT_PHASE0_SPARK_EXPERT_MODE must be real or synthetic, got: $mode" >&2
    exit 2
    ;;
esac
case "$nccl_cross_nic" in
  0|1|2) ;;
  *)
    echo "DS41RT_SPARK_NCCL_CROSS_NIC must be 0, 1, or 2, got: $nccl_cross_nic" >&2
    exit 2
    ;;
esac
if ! [[ "$nccl_netdevs_policy" = "AUTO" || "$nccl_netdevs_policy" = "ALL" || "$nccl_netdevs_policy" =~ ^MAX:[1-9][0-9]*$ ]]; then
  echo "DS41RT_SPARK_NCCL_NETDEVS_POLICY must be AUTO, ALL, or MAX:N, got: $nccl_netdevs_policy" >&2
  exit 2
fi
case "$nccl_ib_merge_nics" in
  0|1) ;;
  *)
    echo "DS41RT_SPARK_NCCL_IB_MERGE_NICS must be 0 or 1, got: $nccl_ib_merge_nics" >&2
    exit 2
    ;;
esac
if ! [[ "$nccl_p2p_net_chunksize" =~ ^[1-9][0-9]*$ ]]; then
  echo "DS41RT_SPARK_NCCL_P2P_NET_CHUNKSIZE must be a positive integer, got: $nccl_p2p_net_chunksize" >&2
  exit 2
fi

if [ "$intermediate_shards" != "1" ] && [ "$intermediate_shards" != "4" ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_SHARDS must be 1 or 4, got: $intermediate_shards" >&2
  exit 2
fi
case "$intermediate_reduction" in
  coordinator) ;;
  spark|spark-owner|spark-hybrid|spark-rdma|spark-rdma-hybrid)
    if [ "$mode" != "real" ] || [ "$intermediate_shards" != "4" ]; then
      echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires real mode with four intermediate shards" >&2
      exit 2
    fi
    if { [ "$intermediate_reduction" = "spark-owner" ] || [ "$intermediate_reduction" = "spark-hybrid" ] || [ "$intermediate_reduction" = "spark-rdma" ] || [ "$intermediate_reduction" = "spark-rdma-hybrid" ]; } && [ "$expert_transport" != "verbs-host" ]; then
      echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires DS41RT_SPARK_EXPERT_TRANSPORT=verbs-host" >&2
      exit 2
    fi
    if { [ "$intermediate_reduction" = "spark-rdma" ] || [ "$intermediate_reduction" = "spark-rdma-hybrid" ]; } && [ "$intermediate_row_sharded_reduction" != "1" ]; then
      echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION=1" >&2
      exit 2
    fi
    ;;
  *)
    echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION must be coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid, got: $intermediate_reduction" >&2
    exit 2
    ;;
esac
case "$intermediate_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE must be bf16, fp8, or nvfp4, got: $intermediate_reduction_dtype" >&2
    exit 2
    ;;
esac
case "$intermediate_owner_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE must be bf16, fp8, or nvfp4, got: $intermediate_owner_reduction_dtype" >&2
    exit 2
    ;;
esac
if ! [[ "$intermediate_reduction_port" =~ ^[0-9]+$ ]] || [ "$intermediate_reduction_port" -lt 1 ] || [ "$intermediate_reduction_port" -gt 65535 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_reduction_min_rows" =~ ^[0-9]+$ ]] || [ "$intermediate_reduction_min_rows" -lt 1 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS must be positive" >&2
  exit 2
fi
if [ "$intermediate_row_sharded_reduction" != "0" ] && [ "$intermediate_row_sharded_reduction" != "1" ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$intermediate_owner_max_rows" =~ ^[0-9]+$ ]] || [ "$intermediate_owner_max_rows" -lt 1 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_owner_port" =~ ^[0-9]+$ ]] || [ "$intermediate_owner_port" -lt 1 ] || [ "$intermediate_owner_port" -gt 65535 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_OWNER_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_port" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_port" -lt 1 ] || [ "$intermediate_rdma_port" -gt 65535 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_RDMA_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_slot_bytes" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_slot_bytes" -lt 1 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_ring_depth" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_ring_depth" -lt 1 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH must be positive" >&2
  exit 2
fi
if ! [[ "$intermediate_rdma_stripe_min_bytes" =~ ^[0-9]+$ ]] || [ "$intermediate_rdma_stripe_min_bytes" -lt 1 ]; then
  echo "DS41RT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES must be positive" >&2
  exit 2
fi

case "$image_copy_method" in
  spark-netcat|ssh-relay|none) ;;
  *)
    echo "DS41RT_SPARK_IMAGE_COPY_METHOD must be spark-netcat, ssh-relay, or none, got: $image_copy_method" >&2
    exit 2
    ;;
esac

if ! [[ "$image_copy_port" =~ ^[0-9]+$ ]] || [ "$image_copy_port" -lt 1 ] || [ "$image_copy_port" -gt 65535 ]; then
  echo "DS41RT_SPARK_IMAGE_COPY_PORT must be an integer in 1..65535" >&2
  exit 2
fi

if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
  echo "DS41RT_SPARK_EXPERT_PORT must be an integer in 1..65535" >&2
  exit 2
fi

case "$skip_bench" in
  0|1) ;;
  *)
    echo "DS41RT_PHASE0_SPARK_SKIP_BENCH must be 0 or 1, got: $skip_bench" >&2
    exit 2
    ;;
esac
case "$force_build_image" in
  0|1) ;;
  *)
    echo "DS41RT_SPARK_FORCE_BUILD_IMAGE must be 0 or 1, got: $force_build_image" >&2
    exit 2
    ;;
esac
case "$image_only" in
  0|1) ;;
  *)
    echo "DS41RT_SPARK_IMAGE_ONLY must be 0 or 1, got: $image_only" >&2
    exit 2
    ;;
esac

case "$build_profile" in
  debug|release) ;;
  *)
    echo "DS41RT_SPARK_BUILD_PROFILE must be debug or release, got: $build_profile" >&2
    exit 2
    ;;
esac
case "$prebuilt" in
  0|1) ;;
  *)
    echo "DS41RT_SPARK_PREBUILT must be 0 or 1, got: $prebuilt" >&2
    exit 2
    ;;
esac
case "$use_diagnostic_placement" in
  0|1) ;;
  *)
    echo "DS41RT_SPARK_USE_DIAGNOSTIC_PLACEMENT must be 0 or 1, got: $use_diagnostic_placement" >&2
    exit 2
    ;;
esac
if [ -n "$existing_container" ]; then
  [ "$prebuilt" = "1" ] || {
    echo "DS41RT_SPARK_EXISTING_CONTAINER requires DS41RT_SPARK_PREBUILT=1" >&2
    exit 2
  }
  [[ "$existing_container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || {
    echo "invalid DS41RT_SPARK_EXISTING_CONTAINER: $existing_container" >&2
    exit 2
  }
  [[ "$prebuilt_bin" = /* && "$prebuilt_native_lib" = /* ]] || {
    echo "persistent-container prebuilt artifact paths must be absolute" >&2
    exit 2
  }
fi
case "$gpu_runtime" in
  nvidia|manual) ;;
  *)
    echo "DS41RT_SPARK_GPU_RUNTIME must be nvidia or manual, got: $gpu_runtime" >&2
    exit 2
    ;;
esac
case "$expert_transport" in
  tcp|verbs-host) ;;
  *)
    echo "DS41RT_SPARK_EXPERT_TRANSPORT must be tcp or verbs-host, got: $expert_transport" >&2
    exit 2
    ;;
esac

case "$expert_real_layer" in
  ""|all|none) ;;
  *)
    if ! [[ "$expert_real_layer" =~ ^[0-9]+$ ]]; then
      echo "DS41RT_SPARK_EXPERT_REAL_LAYER must be a non-negative integer, all, none, or empty; got: $expert_real_layer" >&2
      exit 2
    fi
    ;;
esac

case "${managed_route_projections,,}" in
  ""|0|1|true|false|yes|no|managed|uma) ;;
  *)
    echo "DS41RT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS must be boolean-like, managed, uma, or empty; got: $managed_route_projections" >&2
    exit 2
    ;;
esac
case "${grouped_multirow,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW must be boolean-like, got: $grouped_multirow" >&2
    exit 2
    ;;
esac
case "${route_cuda_graphs,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS must be boolean-like, got: $route_cuda_graphs" >&2
    exit 2
    ;;
esac
case "${b12x_spark_aot,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_B12X_SPARK_AOT must be boolean-like, got: $b12x_spark_aot" >&2
    exit 2
    ;;
esac
case "${ds4_flash_spark_aot,,}" in
  0|false|no|off) ;;
  1|true|yes|on) ;;
  *)
    echo "DS41RT_DS4_FLASH_SPARK_AOT must be boolean-like, got: $ds4_flash_spark_aot" >&2
    exit 2
    ;;
esac
if ! [[ "$b12x_route_lanes" =~ ^[1-4]$ ]]; then
  echo "DS41RT_B12X_SPARK_ROUTE_LANES must be an integer in 1..4, got: $b12x_route_lanes" >&2
  exit 2
fi
if ! [[ "$protocol_v2_verbs_host_execution_lanes" =~ ^[1-8]$ ]]; then
  echo "DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES must be an integer in 1..8, got: $protocol_v2_verbs_host_execution_lanes" >&2
  exit 2
fi
if ! [[ "$protocol_v2_packed_direct_max_rows" =~ ^[0-9]+$ ]] \
  || [ "$protocol_v2_packed_direct_max_rows" -lt 8 ] \
  || [ "$protocol_v2_packed_direct_max_rows" -gt 2048 ]; then
  echo "DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS must be an integer in 8..2048, got: $protocol_v2_packed_direct_max_rows" >&2
  exit 2
fi
rdma_last_port=$((intermediate_rdma_port + protocol_v2_verbs_host_execution_lanes * 6 - 1))
if [ "$rdma_last_port" -gt 65535 ]; then
  echo "Spark RDMA pair/lane ports end at $rdma_last_port, above 65535" >&2
  exit 2
fi
case "${b12x_grouped_decode,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_B12X_SPARK_GROUPED_DECODE must be boolean-like, got: $b12x_grouped_decode" >&2
    exit 2
    ;;
esac
case "${b12x_w4a16_device_weights,,}" in
  0|1|true|false|yes|no|on|off|device|cuda) ;;
  *)
    echo "DS41RT_B12X_SPARK_W4A16_DEVICE_WEIGHTS must be boolean-like, device, or cuda; got: $b12x_w4a16_device_weights" >&2
    exit 2
    ;;
esac
if [ -n "$b12x_w4a16_decode_grid_x" ] &&
  { ! [[ "$b12x_w4a16_decode_grid_x" =~ ^[0-9]+$ ]] ||
    [ "$b12x_w4a16_decode_grid_x" -lt 1 ] ||
    [ "$b12x_w4a16_decode_grid_x" -gt 96 ]; }; then
  echo "DS41RT_B12X_SPARK_W4A16_DECODE_GRID_X must be an integer in 1..96, got: $b12x_w4a16_decode_grid_x" >&2
  exit 2
fi
case "${b12x_w4a16_m1_fused_sum,,}" in
  0|false|no|off) b12x_w4a16_m1_fused_sum=0 ;;
  1|true|yes|on) b12x_w4a16_m1_fused_sum=1 ;;
  *)
    echo "DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM must be boolean-like, got: $b12x_w4a16_m1_fused_sum" >&2
    exit 2
    ;;
esac
case "${b12x_w4a16_small_m_mode,,}" in
  ordered|wide|wide-ordered|split-m1) ;;
  *)
    echo "DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE must be wide, ordered, or split-m1; got: $b12x_w4a16_small_m_mode" >&2
    exit 2
    ;;
esac
case "${route_timing,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING must be boolean-like, got: $route_timing" >&2
    exit 2
    ;;
esac
case "${route_validate,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_REAL_FULL_CUDA_ROUTE_VALIDATE must be boolean-like, got: $route_validate" >&2
    exit 2
    ;;
esac
case "${route_cuda_event_timing,,}" in
  0|1|true|false|yes|no|on|off) ;;
  *)
    echo "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING must be boolean-like, got: $route_cuda_event_timing" >&2
    exit 2
    ;;
esac

spark_secondary_rail_ip() {
  case "$1" in
    ostrich) echo "10.55.0.5" ;;
    dodo) echo "10.55.0.6" ;;
    emu) echo "10.55.0.7" ;;
    kiwi) echo "10.55.0.8" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a hosts <<< "$hosts_csv"
if [ "${#hosts[@]}" -eq 0 ]; then
  echo "DS41RT_SPARK_HOSTS did not contain any hosts" >&2
  exit 2
fi
if [ "$image_only" != "1" ] && [ "$intermediate_shards" = "4" ] && [ "${#hosts[@]}" -ne 4 ]; then
  echo "four-way intermediate sharding requires exactly four Spark hosts" >&2
  exit 2
fi
intermediate_reduction_root="${DS41RT_EXPERT_INTERMEDIATE_REDUCTION_ROOT:-${hosts[0]}${image_link_suffix}}"
intermediate_owner_peers="${DS41RT_EXPERT_INTERMEDIATE_OWNER_PEERS:-}"
if [ -z "$intermediate_owner_peers" ]; then
  for host in "${hosts[@]}"; do
    if [ -n "$intermediate_owner_peers" ]; then
      intermediate_owner_peers+=","
    fi
    intermediate_owner_peers+="${host}${image_link_suffix}:${intermediate_owner_port}"
  done
fi
intermediate_rdma_peers="${DS41RT_EXPERT_INTERMEDIATE_RDMA_PEERS:-}"
if [ -z "$intermediate_rdma_peers" ]; then
  for host in "${hosts[@]}"; do
    if [ -n "$intermediate_rdma_peers" ]; then
      intermediate_rdma_peers+=","
    fi
    intermediate_rdma_peers+="${host}${image_link_suffix}"
  done
fi
if [ -z "$intermediate_rdma_additional_peers" ]; then
  discovered_additional_peers=""
  all_secondary_rails_known=1
  for host in "${hosts[@]}"; do
    host="$(echo "$host" | xargs)"
    if ! secondary_ip="$(spark_secondary_rail_ip "$host")"; then
      all_secondary_rails_known=0
      break
    fi
    if [ -n "$discovered_additional_peers" ]; then
      discovered_additional_peers+=","
    fi
    discovered_additional_peers+="$secondary_ip"
  done
  if [ "$all_secondary_rails_known" = "1" ]; then
    intermediate_rdma_additional_peers="$discovered_additional_peers"
  fi
fi
if [ -n "$intermediate_rdma_additional_peers" ] && [ -z "$intermediate_rdma_devices" ]; then
  intermediate_rdma_devices="rocep1s0f0,roceP2p1s0f0"
fi

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 2
  fi
}

need rsync
need ssh
if [ -z "$remote_dir" ]; then
  remote_dir="$(
    ssh -o BatchMode=yes "${hosts[0]}" \
      'printf "%s/ds41rt-phase0-spark-bench" "$HOME"'
  )"
fi
if [ "$mode" = "real" ] && [ "$image_only" != "1" ] && [ "$use_diagnostic_placement" = "1" ]; then
  need jq
  test -f "$catalog" || {
    echo "catalog not found: $catalog" >&2
    exit 2
  }
fi

container_name_for_host() {
  local host="$1"
  if [ -n "$existing_container" ]; then
    echo "$existing_container"
    return
  fi
  echo "${container_prefix}-${host}-${port}"
}

cleanup() {
  # Cache-only staging must be observational with respect to already-running
  # expert containers. The generic benchmark cleanup otherwise removes the
  # production-named workers even though this invocation never started them.
  if [ "$keep_experts" = "1" ] \
    || [ "$model_sync_only" = "1" ] \
    || [ "$image_only" = "1" ]; then
    return
  fi
  for host in "${hosts[@]}"; do
    local container
    container="$(container_name_for_host "$host")"
    if [ -n "$existing_container" ]; then
      ssh -o BatchMode=yes "$host" \
        "docker exec '$container' '$remote_dir/scripts/wip-process.sh' stop 'expert-$port'" \
        >/dev/null 2>&1 || true
      continue
    fi
    ssh -o BatchMode=yes "$host" bash -s -- "$container" "$remote_dir" <<'REMOTE' >/dev/null 2>&1 || true
set -euo pipefail
container="$1"
remote_dir="$2"
log_dir="$remote_dir/.ds41rt-cache/model-artifacts/diagnostic/logs"
mkdir -p "$log_dir"
docker logs "$container" >"$log_dir/${container}.log" 2>&1 || true
docker rm -f "$container" >/dev/null 2>&1 || true
REMOTE
  done
}
trap cleanup EXIT

stage_repo() {
  local host="$1"
  echo "== staging repo on $host:$remote_dir =="
  ssh -o BatchMode=yes "$host" "mkdir -p '$remote_dir'"
  rsync -az --delete \
    --exclude '.git' \
    --exclude '.venv/' \
    --exclude '.ds41rt-cache/' \
    --exclude '.ds41rt-wip/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude 'rust/target/' \
    --exclude 'native/build/' \
    --exclude 'native/build-cuda/' \
    --exclude 'native/build*/' \
    --exclude 'reports/phase0_artifacts/benchmarks/' \
    --exclude 'reports/phase0_artifacts/logs/' \
    --exclude 'reports/phase0_artifacts/smoke/' \
    --exclude 'reports/phase0_artifacts/tests/' \
    --exclude '.ds41rt-cache/model-artifacts/diagnostic/benchmarks/' \
    --exclude '.ds41rt-cache/model-artifacts/diagnostic/logs/' \
    --exclude '.ds41rt-cache/model-artifacts/diagnostic/smoke/' \
    --exclude '.ds41rt-cache/model-artifacts/diagnostic/tests/' \
    "$repo_root"/ "$host:$remote_dir"/
  # Reconcile the vendored fork independently: exclusions on the broad sync
  # must not preserve bytecode or tool caches from an older staged revision.
  rsync -az --delete --delete-excluded \
    --exclude '.git' \
    --exclude '.venv/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    "$repo_root/third_party/sparkinfer/" \
    "$host:$remote_dir/third_party/sparkinfer/"
  ssh -o BatchMode=yes "$host" \
    "python3 '$remote_dir/scripts/verify-sparkinfer-source.py' \
      --source '$remote_dir/third_party/sparkinfer' \
      --lock '$remote_dir/third_party/sparkinfer.lock.json' \
      --require-no-python-cache"
  if [ "$use_diagnostic_placement" = "1" ] \
    && { [[ "$catalog" == .ds41rt-cache/* ]] || [[ "$loadplan_dir" == .ds41rt-cache/* ]]; }; then
    ssh -o BatchMode=yes "$host" \
      "mkdir -p '$remote_dir/$(dirname "$catalog")' '$remote_dir/$loadplan_dir'"
    rsync -az "$repo_root/$catalog" "$host:$remote_dir/$catalog"
    rsync -az "$repo_root/$loadplan_dir/" "$host:$remote_dir/$loadplan_dir/"
  fi
}

sync_model_cache_to_host() {
  local host="$1"
  local model_cache_key="models--${model_id//\//--}"
  local local_hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
  local local_model_root="$local_hf_home/hub/$model_cache_key"
  local revision
  local remote_hf_home

  test -f "$local_model_root/refs/main" || {
    echo "local model cache is missing refs/main: $local_model_root" >&2
    return 1
  }
  revision="$(<"$local_model_root/refs/main")"
  test -n "$revision" && test -d "$local_model_root/snapshots/$revision" || {
    echo "local model snapshot is incomplete: $local_model_root revision=$revision" >&2
    return 1
  }
  if find "$local_model_root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
    echo "local model snapshot contains unresolved blobs: $local_model_root revision=$revision" >&2
    return 1
  fi

  remote_hf_home="$(
    ssh -o BatchMode=yes "$host" \
      'printf "%s" "${HF_HOME:-$HOME/.cache/huggingface}"'
  )"
  if ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_hf_home/hub/$model_cache_key" "$revision" <<'REMOTE'
set -euo pipefail
model_root="$1"
revision="$2"
test -f "$model_root/refs/main"
test "$(<"$model_root/refs/main")" = "$revision"
test -d "$model_root/snapshots/$revision"
if find "$model_root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
  exit 1
fi
REMOTE
  then
    echo "== model cache already complete on $host: $model_id@$revision =="
    return 0
  fi

  echo "== syncing model cache to $host: $model_id@$revision =="
  ssh -o BatchMode=yes "$host" "mkdir -p '$remote_hf_home/hub/$model_cache_key'"
  rsync -a --partial \
    "$local_model_root/" "$host:$remote_hf_home/hub/$model_cache_key/"
}

image_exists() {
  local host="$1"
  ssh -o BatchMode=yes "$host" "docker image inspect '$image' >/dev/null 2>&1"
}

select_image_seed() {
  local candidate
  if [ -n "$image_seed_host" ]; then
    image_exists "$image_seed_host" || {
      echo "DS41RT_SPARK_IMAGE_SEED_HOST does not have image '$image': $image_seed_host" >&2
      return 1
    }
    echo "$image_seed_host"
    return 0
  fi
  for candidate in "${hosts[@]}"; do
    if image_exists "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

wait_for_remote_listen() {
  local host="$1"
  local listen_port="$2"
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ssh -o BatchMode=yes "$host" "ss -ltn sport = :$listen_port 2>/dev/null | tail -n +2 | grep -q ." >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "remote listener on ${host}:${listen_port} did not become ready" >&2
  return 1
}

copy_image_spark_netcat() {
  local src="$1"
  local dest="$2"
  local dest_link="${dest}${image_link_suffix}"
  local state_dir="/tmp/ds41rt-image-copy-${USER:-tj}-${dest}-${image_copy_port}"
  echo "== copying $image from $src to $dest over ${dest_link}:${image_copy_port} =="
  ssh -o BatchMode=yes "$dest" bash -s -- "$image_copy_port" "$state_dir" <<'REMOTE'
set -euo pipefail
listen_port="$1"
state_dir="$2"
rm -rf "$state_dir"
mkdir -p "$state_dir"
nohup bash -c '
set -o pipefail
listen_port="$1"
state_dir="$2"
if nc -l -p "$listen_port" | docker load >"$state_dir/docker-load.log" 2>&1; then
  touch "$state_dir/success"
else
  status=$?
  echo "$status" >"$state_dir/status"
  exit "$status"
fi
' bash "$listen_port" "$state_dir" >"$state_dir/listener.log" 2>&1 < /dev/null &
echo $! >"$state_dir/pid"
REMOTE
  wait_for_remote_listen "$dest" "$image_copy_port"
  if ! ssh -o BatchMode=yes "$src" bash -s -- "$image" "$dest_link" "$image_copy_port" <<'REMOTE'
set -euo pipefail
image="$1"
dest_link="$2"
dest_port="$3"
docker save "$image" | nc -N "$dest_link" "$dest_port"
REMOTE
  then
    ssh -o BatchMode=yes "$dest" "test -f '$state_dir/pid' && kill \"\$(cat '$state_dir/pid')\" >/dev/null 2>&1 || true" || true
    return 1
  fi
  ssh -o BatchMode=yes "$dest" bash -s -- "$state_dir" "$image" <<'REMOTE'
set -euo pipefail
state_dir="$1"
image="$2"
pid="$(cat "$state_dir/pid")"
deadline=$((SECONDS + 1800))
while kill -0 "$pid" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    kill "$pid" >/dev/null 2>&1 || true
    echo "timed out waiting for docker load" >&2
    exit 1
  fi
  sleep 1
done
if [ -f "$state_dir/success" ] && docker image inspect "$image" >/dev/null 2>&1; then
  exit 0
fi
cat "$state_dir/listener.log" >&2 || true
cat "$state_dir/docker-load.log" >&2 || true
echo "docker load did not produce image $image" >&2
exit 1
REMOTE
}

copy_image_ssh_relay() {
  local src="$1"
  local dest="$2"
  echo "== copying $image from $src to $dest through local SSH relay =="
  ssh -o BatchMode=yes "$src" "docker save '$image'" | ssh -o BatchMode=yes "$dest" "docker load"
}

copy_image() {
  local src="$1"
  local dest="$2"
  case "$image_copy_method" in
    spark-netcat)
      copy_image_spark_netcat "$src" "$dest"
      ;;
    ssh-relay)
      copy_image_ssh_relay "$src" "$dest"
      ;;
    none)
      return 1
      ;;
  esac
}

ensure_image() {
  local host="$1"
  if [ "$force_build_image" != "1" ] && image_exists "$host"; then
    if [ -z "$image_seed_host" ]; then
      image_seed_host="$host"
    fi
    return
  fi
  if [ "$force_build_image" != "1" ] && [ "$image_copy_method" != "none" ]; then
    if [ -z "$image_seed_host" ]; then
      if ! image_seed_host="$(select_image_seed)"; then
        if [ -n "${DS41RT_SPARK_IMAGE_SEED_HOST:-}" ]; then
          exit 2
        fi
        image_seed_host=""
      fi
    fi
    if [ -n "$image_seed_host" ] && [ "$image_seed_host" != "$host" ]; then
      copy_image "$image_seed_host" "$host"
      image_exists "$host" && return
      echo "copying Spark image '$image' to $host did not make the image available" >&2
      exit 1
    fi
  fi
  if [ "$force_build_image" != "1" ] && [ "$build_missing_image" != "1" ]; then
    cat >&2 <<EOF
Spark image '$image' is missing on $host.
Seed it from another Spark, rerun with DS41RT_SPARK_IMAGE_COPY_METHOD=ssh-relay,
or explicitly rerun with DS41RT_SPARK_IMAGE_COPY_METHOD=none DS41RT_SPARK_BUILD_IMAGE=1.
EOF
    exit 2
  fi
  echo "== building $image on $host =="
  ssh -o BatchMode=yes "$host" bash -s -- "$remote_dir" "$image" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
cd "$remote_dir"
docker build \
  --platform linux/arm64 \
  --build-arg DS41RT_ROLE=expert \
  --build-arg CUDA_ARCH=121 \
  --build-arg TARGET_PLATFORM=linux/arm64 \
  -f docker/Dockerfile.dev \
  -t "$image" .
REMOTE
  if [ -z "$image_seed_host" ]; then
    image_seed_host="$host"
  fi
}

loadplan_for_host() {
  local host="$1"
  echo "${loadplan_dir}/loadplan.${host}.json"
}

expert_id_for_host() {
  local host="$1"
  local loadplan
  loadplan="$(loadplan_for_host "$host")"
  test -f "$loadplan" || {
    echo "loadplan not found for $host: $loadplan" >&2
    exit 2
  }
  local expert_id
  expert_id="$(
    jq -r --arg host "$host" --argjson layer "$layer_id" '
      .assignments[]
      | select(.owner == $host and .layer_id == $layer and (.tensor_name | endswith(".gate_proj.weight")))
      | .expert_id
    ' "$loadplan" | head -1
  )"
  if [ -z "$expert_id" ] || [ "$expert_id" = "null" ]; then
    echo "no owned layer $layer_id gate projection expert found in $loadplan" >&2
    exit 2
  fi
  echo "$expert_id"
}

start_expertd() {
  local host="$1"
  local container="$2"
  local loadplan="$3"
  local intermediate_shard_rank="$4"
  local catalog_arg="${catalog:-__none__}"
  local loadplan_arg="${loadplan:-__none__}"
  local verbs_ib_port_num_arg="${verbs_ib_port_num:-__unset__}"
  local b12x_w4a16_decode_grid_x_arg="${b12x_w4a16_decode_grid_x:-__unset__}"
  local release_config_sha256_arg="${DS41RT_RELEASE_CONFIG_SHA256:-__unset__}"
  local intermediate_owner_peers_arg="${intermediate_owner_peers:-__unset__}"
  local intermediate_rdma_additional_peers_arg="${intermediate_rdma_additional_peers:-__unset__}"
  local intermediate_rdma_devices_arg="${intermediate_rdma_devices:-__unset__}"
  # ssh flattens its argument vector into a remote shell command. A quoted
  # empty argument is not preserved, so use sentinels for optional arguments
  # late in the vector or every following positional parameter shifts left.
  local existing_container_arg="${existing_container:-__unset__}"
  local runtime_cache_dir_arg="${runtime_cache_dir:-__unset__}"
  echo "== starting $mode ProtocolV2 expertd on $host:$port transport=$expert_transport real_layer=${expert_real_layer:-all} intermediate_shard=${intermediate_shard_rank}/${intermediate_shards} intermediate_reduction=$intermediate_reduction reduction_dtype=$intermediate_reduction_dtype owner_reduction_dtype=$intermediate_owner_reduction_dtype fused_fp8_reduction=$fused_fp8_reduction protocol_v2_execution_lanes=$protocol_v2_verbs_host_execution_lanes packed_direct_max_rows=$protocol_v2_packed_direct_max_rows managed_route_projections=${managed_route_projections:-0} grouped_multirow=${grouped_multirow:-0} route_cuda_graphs=${route_cuda_graphs:-0} b12x_spark_aot=${b12x_spark_aot:-0} ds4_flash_spark_aot=${ds4_flash_spark_aot:-0} b12x_route_lanes=$b12x_route_lanes b12x_grouped_decode=$b12x_grouped_decode b12x_w4a16_device_weights=$b12x_w4a16_device_weights b12x_w4a16_m1_fused_sum=$b12x_w4a16_m1_fused_sum b12x_w4a16_small_m_mode=$b12x_w4a16_small_m_mode nccl_ib_hca=$nccl_ib_hca nccl_cross_nic=$nccl_cross_nic nccl_netdevs_policy=$nccl_netdevs_policy nccl_ib_merge_nics=$nccl_ib_merge_nics nccl_p2p_net_chunksize=$nccl_p2p_net_chunksize nccl_launch_order_implicit=$nccl_launch_order_implicit route_cuda_event_timing=${route_cuda_event_timing:-0} route_timing=${route_timing:-0} build_profile=$build_profile gpu_runtime=$gpu_runtime =="
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$remote_dir" "$image" "$container" "$mode" "$port" "$catalog_arg" "$loadplan_arg" "$host" "$layer_id" "$expert_real_layer" "$managed_route_projections" "$build_profile" "$gpu_runtime" "${DS41RT_PROTOCOL_V2_TCP_TIMING:-0}" "${DS41RT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING:-0}" "$grouped_multirow" "$route_cuda_graphs" "$route_timing" "$expert_transport" "$verbs_ib_port_num_arg" "$route_cuda_event_timing" "$b12x_spark_aot" "$route_validate" "$b12x_route_lanes" "$intermediate_shards" "$intermediate_shard_rank" "$intermediate_reduction" "$intermediate_reduction_dtype" "$intermediate_reduction_root" "$intermediate_reduction_port" "$intermediate_reduction_min_rows" "$nccl_socket_ifname" "$nccl_ib_hca" "$nccl_debug" "$b12x_grouped_decode" "$intermediate_owner_max_rows" "$intermediate_owner_port" "$intermediate_owner_peers_arg" "$fused_fp8_reduction" "$nccl_bf16_reduce" "$b12x_w4a16_decode_grid_x_arg" "$b12x_w4a16_device_weights" "$intermediate_row_sharded_reduction" "$nccl_launch_order_implicit" "$protocol_v2_verbs_host_execution_lanes" "$intermediate_rdma_peers" "$intermediate_rdma_port" "$intermediate_rdma_slot_bytes" "$intermediate_rdma_ring_depth" "$intermediate_owner_reduction_dtype" "$nccl_cross_nic" "$intermediate_rdma_additional_peers_arg" "$intermediate_rdma_devices_arg" "$intermediate_rdma_stripe_min_bytes" "$nccl_netdevs_policy" "$nccl_ib_merge_nics" "$nccl_p2p_net_chunksize" "$protocol_v2_packed_direct_max_rows" "$b12x_w4a16_m1_fused_sum" "$b12x_w4a16_small_m_mode" "$model_id" "$release_config_sha256_arg" "$prebuilt" "$route_preload_io_workers" "$weight_preload_nccl_port" "$route_preload_cooperative" "$existing_container_arg" "$prebuilt_bin" "$prebuilt_native_lib" "$runtime_cache_dir_arg" "$ds4_flash_spark_aot" "$model_revision" "$wip_allow_historical_exl3_control" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
image="$2"
container="$3"
mode="$4"
port="$5"
catalog="$6"
if [ "$catalog" = "__none__" ]; then
  catalog=""
fi
loadplan="$7"
if [ "$loadplan" = "__none__" ]; then
  loadplan=""
fi
role_hostname="$8"
layer_id="$9"
expert_real_layer="${10:-}"
managed_route_projections="${11:-0}"
build_profile="${12:-release}"
gpu_runtime="${13:-nvidia}"
protocol_v2_tcp_timing="${14:-0}"
protocol_v2_executor_timing="${15:-0}"
grouped_multirow="${16:-0}"
route_cuda_graphs="${17:-0}"
route_timing="${18:-0}"
expert_transport="${19:-tcp}"
verbs_ib_port_num="${20:-}"
if [ "$verbs_ib_port_num" = "__unset__" ]; then
  verbs_ib_port_num=""
fi
route_cuda_event_timing="${21:-0}"
b12x_spark_aot="${22:-0}"
route_validate="${23:-0}"
b12x_route_lanes="${24:-4}"
intermediate_shards="${25:-1}"
intermediate_shard_rank="${26:-0}"
runtime_role="spark-${intermediate_shard_rank}"
intermediate_reduction="${27:-coordinator}"
intermediate_reduction_dtype="${28:-fp8}"
intermediate_reduction_root="${29:-ostrich.200gb}"
intermediate_reduction_port="${30:-9200}"
intermediate_reduction_min_rows="${31:-16}"
nccl_socket_ifname="${32:-enp1s0f0np0}"
nccl_ib_hca="${33:-=rocep1s0f0,roceP2p1s0f0}"
nccl_debug="${34:-WARN}"
b12x_grouped_decode="${35:-1}"
intermediate_owner_max_rows="${36:-8}"
intermediate_owner_port="${37:-9100}"
intermediate_owner_peers="${38:-}"
if [ "$intermediate_owner_peers" = "__unset__" ]; then
  intermediate_owner_peers=""
fi
fused_fp8_reduction="${39:-1}"
nccl_bf16_reduce="${40:-0}"
b12x_w4a16_decode_grid_x="${41:-}"
if [ "$b12x_w4a16_decode_grid_x" = "__unset__" ]; then
  b12x_w4a16_decode_grid_x=""
fi
b12x_w4a16_device_weights="${42:-1}"
intermediate_row_sharded_reduction="${43:-0}"
nccl_launch_order_implicit="${44:-1}"
protocol_v2_verbs_host_execution_lanes="${45:-4}"
intermediate_rdma_peers="${46:-}"
intermediate_rdma_port="${47:-9400}"
intermediate_rdma_slot_bytes="${48:-4194304}"
intermediate_rdma_ring_depth="${49:-4}"
intermediate_owner_reduction_dtype="${50:-bf16}"
nccl_cross_nic="${51:-2}"
intermediate_rdma_additional_peers="${52:-}"
intermediate_rdma_devices="${53:-}"
if [ "$intermediate_rdma_additional_peers" = "__unset__" ]; then
  intermediate_rdma_additional_peers=""
fi
if [ "$intermediate_rdma_devices" = "__unset__" ]; then
  intermediate_rdma_devices=""
fi
intermediate_rdma_stripe_min_bytes="${54:-262144}"
nccl_netdevs_policy="${55:-ALL}"
nccl_ib_merge_nics="${56:-1}"
nccl_p2p_net_chunksize="${57:-131072}"
protocol_v2_packed_direct_max_rows="${58:-2048}"
b12x_w4a16_m1_fused_sum="${59:-0}"
b12x_w4a16_small_m_mode="${60:-wide}"
model_id="${61:-deepseek-ai/DeepSeek-V4.1-Flash}"
release_config_sha256="${62:-}"
if [ "$release_config_sha256" = "__unset__" ]; then
  release_config_sha256=""
fi
prebuilt="${63:-0}"
route_preload_io_workers="${64:-128}"
weight_preload_nccl_port="${65:-9350}"
route_preload_cooperative="${66:-1}"
existing_container="${67:-}"
if [ "$existing_container" = "__unset__" ]; then
  existing_container=""
fi
prebuilt_bin="${68:-/opt/ds41rt/bin/ds41rt}"
prebuilt_native_lib="${69:-/opt/ds41rt/lib/libds41rt_native.so}"
runtime_cache_dir="${70:-}"
if [ "$runtime_cache_dir" = "__unset__" ]; then
  runtime_cache_dir=""
fi
ds4_flash_spark_aot="${71:-1}"
model_revision="${72:-}"
wip_allow_historical_exl3_control="${73:-0}"

discover_rdma_device_map() {
  local entries=()
  local rdma_path
  for rdma_path in /sys/class/infiniband/*; do
    [ -e "$rdma_path" ] || continue
    local rdma_device
    rdma_device="$(basename "$rdma_path")"
    local net_path
    for net_path in "$rdma_path"/device/net/*; do
      [ -e "$net_path" ] || continue
      local net_device
      net_device="$(basename "$net_path")"
      local cidr
      while read -r cidr; do
        [ -n "$cidr" ] || continue
        entries+=("${cidr%/*}=${rdma_device}")
      done < <(ip -4 -o addr show dev "$net_device" 2>/dev/null | awk '{print $4}')
    done
  done
  local IFS=,
  echo "${entries[*]}"
}

rdma_device_map=""
if [ "$expert_transport" = "verbs-host" ]; then
  rdma_device_map="$(discover_rdma_device_map)"
  if [ -z "$rdma_device_map" ]; then
    echo "unable to map Spark RDMA device IPs from /sys/class/infiniband" >&2
    exit 2
  fi
  echo "verbs-host RDMA device map: $rdma_device_map" >&2
fi

if [ -z "$existing_container" ]; then
  docker rm -f "$container" >/dev/null 2>&1 || true
fi
if [ -n "$existing_container" ]; then
  runtime_cache_dir="${runtime_cache_dir:-/wip/cache}"
  runtime_catalog_cache_dir="$runtime_cache_dir/catalogs"
else
  host_runtime_cache_dir="${runtime_cache_dir:-${XDG_CACHE_HOME:-$HOME/.cache}/ds41rt}"
  mkdir -p "$host_runtime_cache_dir/catalogs"
  runtime_catalog_cache_dir=/var/cache/ds41rt/catalogs
fi
docker_args=(
  run -d
  --name "$container"
  --net=host
  --ipc=host
  # Docker's default seccomp profile blocks io_uring_setup. The expert image
  # runs only trusted, prebuilt ds41rt code and needs io_uring for cold HF reads.
  --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  -v "$remote_dir:/workspace/ds41rt"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:${HF_HOME:-$HOME/.cache/huggingface}:ro"
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface:ro"
  -e HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e DS41RT_MODEL_ID="$model_id"
  -e DS41RT_MODEL_REVISION="$model_revision"
  -e DS41RT_WIP_ALLOW_HISTORICAL_EXL3_CONTROL="$wip_allow_historical_exl3_control"
  -e DS41RT_RELEASE_CONFIG_SHA256="$release_config_sha256"
  -e DS41RT_RUNTIME_CATALOG_CACHE_DIR="$runtime_catalog_cache_dir"
  -e DS41RT_BENCH_MODE="$mode"
  -e DS41RT_BENCH_PORT="$port"
  -e DS41RT_BENCH_CATALOG="$catalog"
  -e DS41RT_BENCH_LOADPLAN="$loadplan"
  -e DS41RT_BENCH_ROLE_HOSTNAME="$runtime_role"
  -e DS41RT_BENCH_LAYER_ID="$layer_id"
  -e DS41RT_BENCH_REAL_LAYER="$expert_real_layer"
  -e DS41RT_BENCH_TRANSPORT="$expert_transport"
  -e DS41RT_SPARK_BUILD_PROFILE="$build_profile"
  -e DS41RT_SPARK_PREBUILT="$prebuilt"
  -e DS41RT_SPARK_PREBUILT_BIN="$prebuilt_bin"
  -e DS41RT_SPARK_PREBUILT_NATIVE_LIB="$prebuilt_native_lib"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_MANAGED_PROJECTIONS="$managed_route_projections"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS="$route_preload_io_workers"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE="$route_preload_cooperative"
  -e DS41RT_EXPERT_WEIGHT_PRELOAD_NCCL_PORT="$weight_preload_nccl_port"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW="$grouped_multirow"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS="$route_cuda_graphs"
  -e DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES="$protocol_v2_verbs_host_execution_lanes"
  -e DS41RT_B12X_SPARK_AOT_BUILD="$b12x_spark_aot"
  -e DS41RT_DS4_FLASH_SPARK_AOT_BUILD="$ds4_flash_spark_aot"
  -e DS41RT_B12X_SPARK_ROUTE_LANES="$b12x_route_lanes"
  -e DS41RT_B12X_SPARK_GROUPED_DECODE="$b12x_grouped_decode"
  -e DS41RT_B12X_SPARK_W4A16_DEVICE_WEIGHTS="$b12x_w4a16_device_weights"
  -e DS41RT_B12X_SPARK_W4A16_DECODE_GRID_X="$b12x_w4a16_decode_grid_x"
  -e DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM="$b12x_w4a16_m1_fused_sum"
  -e DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE="$b12x_w4a16_small_m_mode"
  -e DS41RT_EXPERT_INTERMEDIATE_SHARDS="$intermediate_shards"
  -e DS41RT_EXPERT_INTERMEDIATE_SHARD_RANK="$intermediate_shard_rank"
  -e DS41RT_EXPERT_INTERMEDIATE_REDUCTION="$intermediate_reduction"
  -e DS41RT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE="$intermediate_reduction_dtype"
  -e DS41RT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE="$intermediate_owner_reduction_dtype"
  -e DS41RT_EXPERT_INTERMEDIATE_REDUCTION_ROOT="$intermediate_reduction_root"
  -e DS41RT_EXPERT_INTERMEDIATE_REDUCTION_PORT="$intermediate_reduction_port"
  -e DS41RT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS="$intermediate_reduction_min_rows"
  -e DS41RT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION="$intermediate_row_sharded_reduction"
  -e DS41RT_EXPERT_FUSED_FP8_REDUCTION="$fused_fp8_reduction"
  -e DS41RT_EXPERT_NCCL_BF16_REDUCE="$nccl_bf16_reduce"
  -e DS41RT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS="$intermediate_owner_max_rows"
  -e DS41RT_EXPERT_INTERMEDIATE_OWNER_PORT="$intermediate_owner_port"
  -e DS41RT_EXPERT_INTERMEDIATE_OWNER_PEERS="$intermediate_owner_peers"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_PEERS="$intermediate_rdma_peers"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$intermediate_rdma_additional_peers"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_DEVICES="$intermediate_rdma_devices"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_PORT="$intermediate_rdma_port"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_SLOT_BYTES="$intermediate_rdma_slot_bytes"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_RING_DEPTH="$intermediate_rdma_ring_depth"
  -e DS41RT_EXPERT_INTERMEDIATE_RDMA_STRIPE_MIN_BYTES="$intermediate_rdma_stripe_min_bytes"
  -e NCCL_SOCKET_IFNAME="$nccl_socket_ifname"
  -e NCCL_IB_HCA="$nccl_ib_hca"
  -e NCCL_CROSS_NIC="$nccl_cross_nic"
  -e NCCL_NETDEVS_POLICY="$nccl_netdevs_policy"
  -e NCCL_IB_MERGE_NICS="$nccl_ib_merge_nics"
  -e NCCL_P2P_NET_CHUNKSIZE="$nccl_p2p_net_chunksize"
  -e NCCL_DEBUG="$nccl_debug"
  -e NCCL_LAUNCH_ORDER_IMPLICIT="$nccl_launch_order_implicit"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING="$route_cuda_event_timing"
  -e DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING="$route_timing"
  -e DS41RT_REAL_FULL_CUDA_ROUTE_VALIDATE="$route_validate"
  -e DS41RT_PROTOCOL_V2_TCP_TIMING="$protocol_v2_tcp_timing"
  -e DS41RT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING="$protocol_v2_executor_timing"
  -e DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS="$protocol_v2_packed_direct_max_rows"
)
if [ -z "$existing_container" ]; then
  docker_args+=(-v "$host_runtime_cache_dir:/var/cache/ds41rt")
fi
if [ -n "$rdma_device_map" ]; then
  docker_args+=(-e DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP="$rdma_device_map")
fi
case "$gpu_runtime" in
  nvidia)
    docker_args+=(--gpus all)
    nvml_library="$(readlink -f /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 2>/dev/null || true)"
    if [ -n "$nvml_library" ] && [ -s "$nvml_library" ]; then
      docker_args+=(-v "$nvml_library:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
    fi
    ;;
  manual)
    for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
      if [ -e "$device" ]; then
        docker_args+=(--device="$device")
      fi
    done
    if [ -d /dev/nvidia-caps ]; then
      for device in /dev/nvidia-caps/*; do
        if [ -e "$device" ]; then
          docker_args+=(--device="$device")
        fi
      done
    fi
    nvml_library="$(readlink -f /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1 2>/dev/null || true)"
    if [ -n "$nvml_library" ] && [ -s "$nvml_library" ]; then
      docker_args+=(-v "$nvml_library:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
    fi
    ;;
  *)
    echo "unsupported DS41RT_SPARK_GPU_RUNTIME=$gpu_runtime" >&2
    exit 2
    ;;
esac
if [ -n "$verbs_ib_port_num" ]; then
  docker_args+=(-e DS41RT_VERBS_APP_IB_PORT_NUM="$verbs_ib_port_num")
fi
if [ -e /dev/infiniband ]; then
  docker_args+=(--device=/dev/infiniband)
fi

if [ -n "$existing_container" ]; then
  [ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" = true ] || {
    echo "persistent WIP container is not running: $container" >&2
    exit 2
  }
  docker exec "$container" test -x "$remote_dir/scripts/wip-process.sh"
  docker exec "$container" mkdir -p "$runtime_catalog_cache_dir"
  docker_args+=(
    -e "PYTHONPATH=$remote_dir/third_party/sparkinfer:$remote_dir/python/reference/ds41rt_reference:$remote_dir/python/reference:/opt/ds41rt/third_party/sparkinfer"
  )
  exec_env_args=()
  for ((arg_index = 0; arg_index < ${#docker_args[@]}; arg_index++)); do
    if [ "${docker_args[$arg_index]}" = -e ]; then
      ((arg_index += 1))
      exec_env_args+=(-e "${docker_args[$arg_index]}")
    fi
  done
  process_name="expert-$port"
  docker exec "$container" \
    "$remote_dir/scripts/wip-process.sh" stop "$process_name"
  docker exec -d \
    "${exec_env_args[@]}" \
    -w "$remote_dir" \
    "$container" \
    "$remote_dir/scripts/wip-process.sh" run "$process_name" bash -lc '
set -euo pipefail
cd "$PWD"
test -x "$DS41RT_SPARK_PREBUILT_BIN"
test -s "$DS41RT_SPARK_PREBUILT_NATIVE_LIB"
export DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1
export DS41RT_NATIVE_LIB="$DS41RT_SPARK_PREBUILT_NATIVE_LIB"
real_layer_args=()
case "${DS41RT_BENCH_REAL_LAYER:-}" in
  ""|all|none) ;;
  *) real_layer_args=(--real-layer "$DS41RT_BENCH_REAL_LAYER") ;;
esac
exec "$DS41RT_SPARK_PREBUILT_BIN" expertd \
  --transport "${DS41RT_BENCH_TRANSPORT:-tcp}" \
  --listen "0.0.0.0:${DS41RT_BENCH_PORT}" \
  --model-id "$DS41RT_MODEL_ID" \
  "${real_layer_args[@]}" \
  --role "$DS41RT_BENCH_ROLE_HOSTNAME"
'
  exit 0
fi

docker "${docker_args[@]}" "$image" bash -lc '
set -euo pipefail
cd /workspace/ds41rt
if [ "${DS41RT_SPARK_PREBUILT:-0}" = "1" ]; then
  test -x "$DS41RT_SPARK_PREBUILT_BIN"
  test -s "$DS41RT_SPARK_PREBUILT_NATIVE_LIB"
  export DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1
  export DS41RT_NATIVE_LIB="$DS41RT_SPARK_PREBUILT_NATIVE_LIB"
  real_layer_args=()
  case "${DS41RT_BENCH_REAL_LAYER:-}" in
    ""|all|none) ;;
    *) real_layer_args=(--real-layer "$DS41RT_BENCH_REAL_LAYER") ;;
  esac
  exec "$DS41RT_SPARK_PREBUILT_BIN" expertd \
    --transport "${DS41RT_BENCH_TRANSPORT:-tcp}" \
    --listen "0.0.0.0:${DS41RT_BENCH_PORT}" \
    --model-id "$DS41RT_MODEL_ID" \
    "${real_layer_args[@]}" \
    --role "$DS41RT_BENCH_ROLE_HOSTNAME"
fi
cargo_args=()
bin_profile=debug
if [ "${DS41RT_SPARK_BUILD_PROFILE:-release}" = "release" ]; then
  cargo_args+=(--release)
  bin_profile=release
fi
cargo build --manifest-path rust/Cargo.toml -p ds41rt-daemon "${cargo_args[@]}"
clean_stale_cmake_build_dir() {
  local build_dir="$1"
  local expected_source="$2"
  local cache="${build_dir}/CMakeCache.txt"
  local cached_source=""
  if [ ! -f "$cache" ]; then
    return
  fi
  cached_source="$(sed -n "s/^CMAKE_HOME_DIRECTORY:INTERNAL=//p" "$cache" | tail -n 1)"
  if [ -n "$cached_source" ] && [ "$cached_source" != "$expected_source" ]; then
    echo "removing stale CMake build dir ${build_dir} cached_source=${cached_source} expected_source=${expected_source}" >&2
    rm -rf "$build_dir"
  fi
}
if [ "${DS41RT_BENCH_TRANSPORT:-tcp}" = "verbs-host" ] && [ "$DS41RT_BENCH_MODE" != "real" ]; then
  clean_stale_cmake_build_dir native/build-rdma "$(pwd)/native"
  python3 python/tools/check_native_rdma_build.py \
    --build-dir native/build-rdma \
    --output ".ds41rt-cache/model-artifacts/diagnostic/benchmarks/native_rdma_build_${DS41RT_BENCH_ROLE_HOSTNAME}.json" \
    --require-pass
  export DS41RT_NATIVE_LIB=/workspace/ds41rt/native/build-rdma/libds41rt_native.so
fi
if [ "$DS41RT_BENCH_MODE" = "real" ]; then
  rdma_enabled=OFF
  native_build_dir=native/build-cuda
  if [ "${DS41RT_BENCH_TRANSPORT:-tcp}" = "verbs-host" ]; then
    rdma_enabled=ON
    native_build_dir=native/build-cuda-rdma
  fi
  b12x_aot=OFF
  case "${DS41RT_B12X_SPARK_AOT_BUILD:-0}" in
    1|true|yes|on) b12x_aot=ON ;;
  esac
  ds4_flash_aot=OFF
  case "${DS41RT_DS4_FLASH_SPARK_AOT_BUILD:-0}" in
    1|true|yes|on) ds4_flash_aot=ON ;;
  esac
  nccl_enabled=OFF
  case "${DS41RT_EXPERT_INTERMEDIATE_REDUCTION:-coordinator}" in
    spark|spark-hybrid|spark-rdma|spark-rdma-hybrid) nccl_enabled=ON ;;
  esac
  clean_stale_cmake_build_dir "$native_build_dir" "$(pwd)/native"
  cmake -S native -B "$native_build_dir" -G Ninja \
    -U DS41RT_ENABLE_B12X_AOT \
    -U DS41RT_ENABLE_B12X_COORDINATOR_AOT \
    -DDS41RT_ENABLE_CUDA=ON \
    -DDS41RT_ENABLE_RDMA="$rdma_enabled" \
    -DDS41RT_ENABLE_SPARKINFER_AOT="$b12x_aot" \
    -DDS41RT_ENABLE_DS4_FLASH_AOT="$ds4_flash_aot" \
    -DDS41RT_SPARKINFER_SOURCE_DIR="${DS41RT_SPARKINFER_SOURCE_DIR:-$(pwd)/third_party/sparkinfer}" \
    -DDS41RT_SPARKINFER_LOCK_FILE="${DS41RT_SPARKINFER_LOCK_FILE:-$(pwd)/third_party/sparkinfer.lock.json}" \
    -DDS41RT_ENABLE_NCCL="$nccl_enabled" \
    -DDS41RT_CUDA_ARCHITECTURES=121
  cmake --build "$native_build_dir"
  export DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS=1
  export DS41RT_NATIVE_LIB="/workspace/ds41rt/${native_build_dir}/libds41rt_native.so"
  real_layer_args=()
  case "${DS41RT_BENCH_REAL_LAYER:-}" in
    ""|all|none) ;;
    *) real_layer_args=(--real-layer "$DS41RT_BENCH_REAL_LAYER") ;;
  esac
  catalog_args=()
  loadplan_args=()
  [ -z "$DS41RT_BENCH_CATALOG" ] || catalog_args=(--catalog "$DS41RT_BENCH_CATALOG")
  [ -z "$DS41RT_BENCH_LOADPLAN" ] || loadplan_args=(--loadplan "$DS41RT_BENCH_LOADPLAN")
  exec "rust/target/${bin_profile}/ds41rt" expertd \
    --transport "${DS41RT_BENCH_TRANSPORT:-tcp}" \
    --listen "0.0.0.0:${DS41RT_BENCH_PORT}" \
    --model-id "$DS41RT_MODEL_ID" \
    "${catalog_args[@]}" \
    "${loadplan_args[@]}" \
    "${real_layer_args[@]}" \
    --role "$DS41RT_BENCH_ROLE_HOSTNAME"
fi
exec "rust/target/${bin_profile}/ds41rt" expertd \
  --synthetic-weights \
  --transport "${DS41RT_BENCH_TRANSPORT:-tcp}" \
  --listen "0.0.0.0:${DS41RT_BENCH_PORT}"
'
REMOTE
}

wait_for_port() {
  local host="$1"
  local check_host
  check_host="$(
    ssh -G "$host" 2>/dev/null \
      | awk '$1 == "hostname" { print $2; exit }'
  )"
  check_host="${check_host:-$host}"
  local timeout_s="${DS41RT_SPARK_EXPERT_READY_TIMEOUT:-}"
  if [ -z "$timeout_s" ]; then
    if [ "$mode" = "real" ]; then
      timeout_s=900
    else
      timeout_s=180
    fi
  fi
  local deadline=$((SECONDS + timeout_s))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if timeout 2 bash -c ":</dev/tcp/${check_host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "expert daemon on ${host}:${port} did not become ready within ${timeout_s}s" >&2
  if [ -n "$existing_container" ]; then
    ssh -o BatchMode=yes "$host" \
      "docker exec '$existing_container' '$remote_dir/scripts/wip-process.sh' log 'expert-$port' 120" \
      >&2 || true
  else
    ssh -o BatchMode=yes "$host" "docker logs '$(container_name_for_host "$host")' 2>&1 | tail -120" >&2 || true
  fi
  exit 1
}

targets=()
expert_ids=()
if [ "$skip_stage" != "1" ] && [ "$mode" = "real" ] && [ "$sync_model_cache" = "1" ]; then
  model_sync_pids=()
  for host in "${hosts[@]}"; do
    sync_model_cache_to_host "$host" &
    model_sync_pids+=("$!")
  done
  model_sync_failed=0
  for pid in "${model_sync_pids[@]}"; do
    wait "$pid" || model_sync_failed=1
  done
  if [ "$model_sync_failed" != "0" ]; then
    echo "one or more Spark model-cache syncs failed" >&2
    exit 1
  fi
fi
if [ "$model_sync_only" = "1" ]; then
  if [ "$sync_model_cache" != "1" ]; then
    echo "DS41RT_SPARK_MODEL_SYNC_ONLY=1 requires DS41RT_SPARK_SYNC_MODEL_CACHE=1" >&2
    exit 2
  fi
  echo "Spark model-cache sync complete: $model_id"
  exit 0
fi
start_hosts=()
start_pids=()
for host_index in "${!hosts[@]}"; do
  host="${hosts[$host_index]}"
  if [ "$skip_stage" != "1" ]; then
    stage_repo "$host"
  fi
  ensure_image "$host"
done

if [ "$image_only" = "1" ]; then
  echo "Spark image '$image' is ready on: ${hosts[*]}"
  exit 0
fi

for host_index in "${!hosts[@]}"; do
  host="${hosts[$host_index]}"
  loadplan=""
  if [ "$mode" = "real" ]; then
    if [ "$use_diagnostic_placement" = "1" ]; then
      loadplan="$(loadplan_for_host "$host")"
      expert_ids+=("${host}=$(expert_id_for_host "$host")")
    else
      expert_ids+=("${host}=${host_index}")
    fi
  fi
  targets+=("${host}:${port}")
  start_expertd "$host" "$(container_name_for_host "$host")" "$loadplan" "$host_index" &
  start_hosts+=("$host")
  start_pids+=("$!")
done
start_failed=0
for host_index in "${!start_pids[@]}"; do
  if ! wait "${start_pids[$host_index]}"; then
    echo "failed to start expertd on ${start_hosts[$host_index]}" >&2
    start_failed=1
  fi
done
if [ "$start_failed" != "0" ]; then
  exit 1
fi

for host in "${hosts[@]}"; do
  wait_for_port "$host"
done

if [ "$mode" = "real" ]; then
  expected_executor="protocol-v2-real-nvfp4-checkpoint-executor"
else
  expected_executor="protocol-v2-synthetic-route-dependent-executor"
fi

export DS41RT_PHASE0_TCP_EXPERT_ADDRS="${DS41RT_PHASE0_TCP_EXPERT_ADDRS:-$(IFS=,; echo "${targets[*]}")}"
export DS41RT_PHASE0_TCP_EXPECTED_EXECUTOR="${DS41RT_PHASE0_TCP_EXPECTED_EXECUTOR:-$expected_executor}"
export DS41RT_PHASE0_TCP_REQUIRE_EXPECTED_EXECUTOR=1
export DS41RT_PHASE0_TCP_LAYER_ID="$layer_id"
if [ "$mode" = "real" ]; then
  export DS41RT_PHASE0_TCP_EXPERT_IDS="${DS41RT_PHASE0_TCP_EXPERT_IDS:-$(IFS=,; echo "${expert_ids[*]}")}"
fi

echo "== running phase0 binary ProtocolV2 TCP benchmark =="
echo "targets=$DS41RT_PHASE0_TCP_EXPERT_ADDRS"
echo "expected_executor=$DS41RT_PHASE0_TCP_EXPECTED_EXECUTOR"
echo "measured_timeout_ms=${DS41RT_PHASE0_TCP_TIMEOUT_MS:-5000}"
if [ "$mode" = "real" ]; then
  echo "expert_real_layer=${expert_real_layer:-all}"
  echo "managed_route_projections=${managed_route_projections:-0}"
  echo "grouped_multirow=${grouped_multirow:-0}"
  echo "route_cuda_graphs=${route_cuda_graphs:-0}"
  echo "b12x_spark_aot=${b12x_spark_aot:-0}"
  echo "b12x_route_lanes=$b12x_route_lanes"
  echo "b12x_grouped_decode=$b12x_grouped_decode"
  echo "b12x_w4a16_device_weights=$b12x_w4a16_device_weights"
  echo "protocol_v2_packed_direct_max_rows=$protocol_v2_packed_direct_max_rows"
  echo "route_cuda_event_timing=${route_cuda_event_timing:-0}"
  echo "gpu_runtime=${gpu_runtime}"
  echo "real_precompile_warmup=${DS41RT_PHASE0_TCP_WARMUP:-1} warmup_timeout_ms=${DS41RT_PHASE0_TCP_WARMUP_TIMEOUT_MS:-120000}"
fi
if [ "${DS41RT_PHASE0_TCP_EXPERT_IDS:-}" ]; then
  echo "expert_ids=$DS41RT_PHASE0_TCP_EXPERT_IDS"
fi

if [ "$skip_bench" = "1" ]; then
  echo "Skipping phase0 benchmark because DS41RT_PHASE0_SPARK_SKIP_BENCH=1"
else
  python3 benchmarks/phase0_bench.py
fi

if [ "$keep_experts" = "1" ]; then
  echo "Spark expert containers left running because DS41RT_SPARK_KEEP_EXPERTS=1"
fi
