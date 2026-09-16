use clap::{Args, Parser, Subcommand};
use ds41rt_core::{DEFAULT_MODEL_ID, DS4_FLASH_HIDDEN_SIZE};
use std::path::PathBuf;

pub(crate) const DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS: usize = 128 * 1024;

#[derive(Debug, Parser)]
#[command(name = "ds41rt", about = "DS41RT phase0 runtime CLI")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    Doctor(DoctorArgs),
    InspectModel(InspectModelArgs),
    MakeLoadplan(MakeLoadPlanArgs),
    LoadTensors(LoadTensorsArgs),
    Tokenize(TokenizeArgs),
    Coordinator(CoordinatorArgs),
    Expertd(ExpertDaemonArgs),
    /// Serve official V4.1 native TP4 experts over RoCE.
    ExpertdNative(NativeExpertDaemonArgs),
    /// Serve the official V4.1 target text path.
    ServeNative(NativeServeArgs),
    BenchRdma(BenchRdmaArgs),
    BenchRdmaRing(BenchRdmaRingArgs),
    BenchCudaKernels(BenchCudaKernelsArgs),
    BenchProtocolV2Tcp(BenchProtocolV2TcpArgs),
    BenchExpertReductionReplay(BenchExpertReductionReplayArgs),
    TransportCapabilities(TransportCapabilitiesArgs),
    SchedulerSmoke(SchedulerSmokeArgs),
    SchedulerRowAudit(SchedulerRowAuditArgs),
}

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    #[arg(long, default_value = "coordinator")]
    pub(crate) role: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct InspectModelArgs {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) out: PathBuf,
    #[arg(long)]
    pub(crate) summary: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct MakeLoadPlanArgs {
    #[arg(long)]
    pub(crate) catalog: PathBuf,
    #[arg(long, default_value = "modulo")]
    pub(crate) policy: String,
    #[arg(long, default_value = "spark-0,spark-1,spark-2,spark-3")]
    pub(crate) hosts: String,
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct LoadTensorsArgs {
    #[arg(long)]
    pub(crate) catalog: PathBuf,
    #[arg(long)]
    pub(crate) summary: PathBuf,
    #[arg(long = "tensor")]
    pub(crate) tensors: Vec<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) verify_hashes: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TokenizeArgs {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long)]
    pub(crate) text: String,
    #[arg(long, default_value_t = false)]
    pub(crate) add_special_tokens: bool,
}

#[derive(Debug, Args)]
pub(crate) struct CoordinatorArgs {
    #[arg(long, default_value = "tiny")]
    pub(crate) backend: String,
    #[arg(long, default_value = "inproc")]
    pub(crate) transport: String,
    #[arg(long, default_value = "fp8")]
    pub(crate) kv_cache_dtype: String,
    #[arg(long, default_value_t = DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS)]
    pub(crate) max_context_tokens: usize,
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub(crate) listen: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long, default_value = "spark-0,spark-1,spark-2,spark-3")]
    pub(crate) expert_hosts: String,
    #[arg(long)]
    pub(crate) catalog: Option<PathBuf>,
    #[arg(
        long,
        help = "Legacy expert-owner placement map; rejected by strict DeepSeek V4 TP=4 serving"
    )]
    pub(crate) loadplan: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) preflight_only: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ExpertDaemonArgs {
    #[arg(long, default_value_t = false)]
    pub(crate) synthetic_weights: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) preflight_only: bool,
    #[arg(long, default_value = "tcp")]
    pub(crate) transport: String,
    #[arg(long, default_value = "0.0.0.0:9100")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) loadplan: Option<PathBuf>,
    #[arg(long)]
    pub(crate) catalog: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) real_layer: Option<u32>,
    #[arg(long = "role", visible_alias = "role-hostname")]
    pub(crate) role_hostname: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct NativeExpertDaemonArgs {
    /// First resident backbone layer; use 20 when both RTX GPUs host the encoder.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..40))]
    pub(crate) first_layer: u32,
    /// Official local snapshot directory, including all shard headers.
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    /// Spark-role native library built with V4.1 expert AOT kernels.
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    /// Override the native EXL3 rank directory containing m1, m16 and larger capacities.
    #[arg(long)]
    pub(crate) exl3_aot_dir: Option<PathBuf>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..4))]
    pub(crate) rank: u32,
    #[arg(long, default_value_t = 16)]
    pub(crate) capacity: u32,
    /// Total device bytes allowed for resident weights, loading and execution.
    #[arg(long)]
    pub(crate) device_budget_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub(crate) max_frame_bytes: usize,
    #[arg(long, default_value = "0.0.0.0:9100")]
    pub(crate) listen: String,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaArgs {
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long, default_value = "auto")]
    pub(crate) mode: String,
    #[arg(long, default_value_t = 18515)]
    pub(crate) port: u16,
    #[arg(long, default_value = "4096,8192,12288,16384,32768,65536")]
    pub(crate) payload_bytes: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) duration_secs: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaRingArgs {
    #[arg(long, default_value = "server")]
    pub(crate) mode: String,
    #[arg(long, default_value = "0.0.0.0:18525")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long)]
    pub(crate) peers: Option<String>,
    #[arg(long, default_value_t = 16 * 1024)]
    pub(crate) slot_bytes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) depth: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) window: usize,
    #[arg(long)]
    pub(crate) request_bytes: Option<usize>,
    #[arg(long)]
    pub(crate) response_bytes: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_delay_us: u64,
    #[arg(long, default_value = "unspecified")]
    pub(crate) network_label: String,
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_echo: bool,
    #[arg(long, default_value = "fp8")]
    pub(crate) wire_codec: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) rows: usize,
    /// Hidden-width partial carried by each TP rank in the reduction benchmark.
    #[arg(long, default_value_t = DS4_FLASH_HIDDEN_SIZE)]
    pub(crate) row_width: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) kernel_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) reduction_rank: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) reduction_world_size: usize,
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchCudaKernelsArgs {
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long = "kernel", value_delimiter = ',')]
    pub(crate) kernels: Vec<String>,
    #[arg(long, default_value_t = 16)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 1024)]
    pub(crate) hidden_dim: usize,
    #[arg(long, default_value_t = 2048)]
    pub(crate) intermediate_dim: usize,
    #[arg(long, default_value_t = 1024)]
    pub(crate) output_dim: usize,
    #[arg(long, default_value_t = 4096)]
    pub(crate) vocab: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) routes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) top_k: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 10)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) require_cuda: bool,
}

#[derive(Debug, Args, Clone)]
pub(crate) struct BenchProtocolV2TcpArgs {
    #[arg(long)]
    pub(crate) addr: String,
    #[arg(long, default_value = "tcp")]
    pub(crate) transport: String,
    #[arg(long, default_value = "spark-tcp-expert")]
    pub(crate) target: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) request_id_start: u64,
    #[arg(long, default_value_t = 75)]
    pub(crate) hops: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) large_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) warmup_rows: usize,
    #[arg(long)]
    pub(crate) warmup_timeout_ms: Option<u64>,
    #[arg(long, default_value_t = false)]
    pub(crate) warmup_only: bool,
    /// Stop after the independent round-trip measurements.
    #[arg(long, default_value_t = false)]
    pub(crate) roundtrip_only: bool,
    #[arg(long, default_value = "1,2,4,8,16,64,256,512")]
    pub(crate) roundtrip_rows: String,
    #[arg(long, default_value = "1,2,3,4,5,6,8")]
    pub(crate) mtp_chain_rows: String,
    #[arg(long, default_value = "16,32,64,128,256,512")]
    pub(crate) prefill_roundtrip_rows: String,
    #[arg(long, default_value = "16,32,256,512")]
    pub(crate) prefill_chain_rows: String,
    /// Routed layer exercised by the protocol benchmark. Flash is MoE from layer 0.
    #[arg(long, default_value_t = 0)]
    pub(crate) layer_id: u32,
    #[arg(long, default_value_t = 0)]
    pub(crate) expert_id: u32,
    /// Hidden width carried by every ProtocolV2 row.
    #[arg(long, default_value_t = DS4_FLASH_HIDDEN_SIZE)]
    pub(crate) hidden_dim: usize,
    /// Comma-separated expert ID pattern, cycled across all generated routes.
    #[arg(long)]
    pub(crate) expert_ids: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub(crate) routes_per_row: usize,
    /// Use the production low-precision TP4 partial-response reduction codec.
    #[arg(long, default_value_t = false)]
    pub(crate) spark_reduction: bool,
    /// Use production NVFP4 ingress and row-scaled FP8 responses without Spark reduction.
    #[arg(long, default_value_t = false)]
    pub(crate) nvfp4_fp8_roundtrip: bool,
    #[arg(long)]
    pub(crate) expected_executor: Option<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) require_expected_executor: bool,
    #[arg(long, default_value_t = 5000)]
    pub(crate) timeout_ms: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub(crate) max_frame_bytes: usize,
}

#[derive(Debug, Args)]
pub(crate) struct BenchExpertReductionReplayArgs {
    /// Plain JSONL replay plan produced by plan_expert_reduction_replay.py.
    #[arg(long)]
    pub(crate) plan: PathBuf,
    /// Destination JSONL. The benchmark refuses to overwrite an existing file.
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(
        long,
        default_value = "ostrich=10.55.0.1:9100,dodo=10.55.0.2:9100,emu=10.55.0.3:9100,kiwi=10.55.0.4:9100"
    )]
    pub(crate) expert_hosts: String,
    #[arg(long, default_value = "semantic")]
    pub(crate) cohort: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) warmup_chains_per_m: usize,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct TransportCapabilitiesArgs {
    #[arg(long)]
    pub(crate) benchmark_jsonl: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct SchedulerSmokeArgs {
    #[arg(long, default_value_t = 512)]
    pub(crate) prefill_tokens: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) chunk_tokens: usize,
    #[arg(long, default_value_t = 32)]
    pub(crate) decode_arrivals: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) decode_period_iterations: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) max_prefill_tokens_per_iteration: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) max_active_prefill_chunks: usize,
}

#[derive(Debug, Args)]
pub(crate) struct SchedulerRowAuditArgs {
    #[arg(long = "input")]
    pub(crate) inputs: Vec<PathBuf>,
    #[arg(long = "input-list")]
    pub(crate) input_lists: Vec<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub(crate) next_window_count: usize,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_limits_default_to_model_maximum_and_allow_smaller_launches() {
        use clap::Parser;
        let base = ["ds41rt", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        let super::Commands::ServeNative(args) = super::Cli::try_parse_from(base).unwrap().command else {
            panic!("expected native serving");
        };
        assert_eq!(args.max_context_tokens, 1_048_576);
        assert_eq!(args.max_output_tokens, 393_216);
        assert_eq!(args.concurrency, 16);
        assert_eq!(args.prefix_cache_entries, 24);
        assert_eq!(args.dspark_draft_limit, 5);
        assert!(!args.adaptive_dspark()); // Target-only remains target-only.
        for (flags, adaptive) in [
            (vec!["--dspark"], true),
            (vec!["--dspark", "--dspark-adaptive", "--independent-decode-lanes"], true),
            (vec!["--dspark", "--dspark-fixed"], false),
            (vec!["--dspark", "--dspark-confidence-cutoff", "0.5"], false),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.adaptive_dspark(), adaptive);
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-fixed"])).is_err());
        assert!(super::Cli::try_parse_from(base.into_iter().chain(
            ["--dspark", "--dspark-fixed", "--dspark-adaptive"])).is_err());
        assert!(super::Cli::try_parse_from(base.into_iter().chain(
            ["--dspark", "--dspark-fixed", "--dspark-confidence-cutoff", "0.5"])).is_err());
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-adaptive"])).is_err());
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark", "--dspark-adaptive"])).is_ok());
        for (flags, expected) in [
            (vec!["--dspark"], 5),
            (vec!["--dspark", "--rtx-gpus", "1"], 5),
            (vec!["--dspark", "--rtx-gpus", "2"], 5),
            (vec!["--dspark", "--dspark-draft-limit", "5"], 5),
            (vec!["--dspark", "--rtx-gpus", "2", "--dspark-draft-limit", "7"], 7),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.dspark_draft_limit, expected);
        }
        for limit in ["1", "2", "3", "4", "5", "6", "7"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_ok());
        }
        for limit in ["0", "8"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_err());
        }
        for entries in ["0", "2", "24", "128"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", entries])).is_ok());
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", "129"])).is_err());
        for concurrency in ["1", "2", "16"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency, "--kv-pool-size", "1.5GiB", "--memory-reservation", "87.5%"])).is_ok());
        }
        for concurrency in ["0", "17"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency])).is_err());
        }
        for (context, output, valid) in [("256", "128", true), ("0", "128", false),
            ("1048577", "128", false), ("256", "0", false), ("256", "393217", false)] {
            let command = base.into_iter().chain(["--max-context-tokens", context, "--max-output-tokens", output]);
            assert_eq!(super::Cli::try_parse_from(command).is_ok(), valid);
        }
    }
    use super::*;

    #[test]
    fn transport_benchmark_defaults_use_flash_geometry() {
        let cli = Cli::try_parse_from(["ds41rt", "bench-rdma-ring"]).unwrap();
        let Commands::BenchRdmaRing(args) = cli.command else {
            panic!("expected bench-rdma-ring command");
        };
        assert_eq!(args.row_width, DS4_FLASH_HIDDEN_SIZE);

        let cli =
            Cli::try_parse_from(["ds41rt", "bench-protocol-v2-tcp", "--addr", "127.0.0.1:9100"])
                .unwrap();
        let Commands::BenchProtocolV2Tcp(args) = cli.command else {
            panic!("expected bench-protocol-v2-tcp command");
        };
        assert_eq!(args.layer_id, 0);
        assert_eq!(args.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
        assert!(!args.spark_reduction);
    }

    #[test]
    fn protocol_benchmark_names_tp4_reduction_without_expert_ownership() {
        let cli = Cli::try_parse_from([
            "ds41rt",
            "bench-protocol-v2-tcp",
            "--addr",
            "127.0.0.1:9100",
            "--routes-per-row",
            "6",
            "--spark-reduction",
        ])
        .unwrap();
        let Commands::BenchProtocolV2Tcp(args) = cli.command else {
            panic!("expected bench-protocol-v2-tcp command");
        };
        assert_eq!(args.routes_per_row, 6);
        assert!(args.spark_reduction);

        assert!(Cli::try_parse_from([
            "ds41rt",
            "bench-protocol-v2-tcp",
            "--addr",
            "127.0.0.1:9100",
            "--spark-owner-decode",
        ])
        .is_err());
    }
}

#[derive(Debug, Args)]
pub(crate) struct NativeServeArgs {
    /// Force one RTX or the distributed two-RTX layout (automatic launcher selection is pending).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=2))]
    pub rtx_gpus: u32,

    /// Maximum tokens per prefill step. Storage rounds up to an AOT capacity
    /// (80, 256, 1024, or 4096); all expert peers must support that capacity.
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u32).range(80..=4096))]
    pub prefill_batch_tokens: u32,

    /// Total prompt plus generated tokens; compressed cache is reserved at startup.
    #[arg(long, default_value_t = ds41rt_api::native_v41::MAX_CONTEXT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=1048576))]
    pub max_context_tokens: u32,

    /// Default and maximum generated tokens, further bounded by remaining context.
    #[arg(long, default_value_t = ds41rt_api::native_v41::MAX_OUTPUT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=393216))]
    pub max_output_tokens: u32,

    /// Exact global KV/index byte budget (B/MB/GB/MiB/GiB), rounded down to page groups.
    #[arg(long)]
    pub kv_pool_size: Option<crate::v41_native_serve::memory::ByteSize>,

    /// Total device occupancy ceiling (% or B/MB/GB/MiB/GiB); sizes KV after fixed allocations.
    #[arg(long)]
    pub memory_reservation: Option<crate::v41_native_serve::memory::Reservation>,

    /// Complete bottom-up RTX routed layers: auto fills available memory, or 0..40.
    #[arg(long, default_value = "auto")]
    pub rtx_expert_layers: crate::v41_native_serve::memory::LocalLayers,


    /// Maximum active requests, shared by both execution lanes.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..=16))]
    pub concurrency: u32,

    /// Retained completed turns, plus a separate prompt-repeat bank of this size; zero disables reuse.
    #[arg(long, default_value_t = 24, value_parser = clap::value_parser!(u32).range(0..=128))]
    pub prefix_cache_entries: u32,

    /// Enable greedy RTX dSpark proposal generation and target verification.
    #[arg(long)] pub dspark: bool,
    /// Maximum draft tokens per request, for adaptive or fixed verification.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u8).range(1..=7))]
    pub dspark_draft_limit: u8,
    /// Compatibility spelling: dSpark uses lane-local adaptive selection by default.
    #[arg(long, requires = "dspark", hide = true)]
    pub dspark_adaptive: bool,
    /// Disable adaptive prefix selection and verify the configured fixed draft limit.
    #[arg(long, requires = "dspark", conflicts_with_all = ["dspark_adaptive", "dspark_confidence_cutoff"])]
    pub dspark_fixed: bool,
    /// Experimental independent cumulative confidence cutoff, between zero and one.
    #[arg(long, requires = "dspark", conflicts_with = "dspark_adaptive", value_parser = parse_dspark_confidence)]
    pub dspark_confidence_cutoff: Option<f64>,
    /// With a confidence cutoff, lower it toward this positive floor for predicted expert reuse.
    #[arg(long, requires = "dspark_confidence_cutoff", value_parser = parse_dspark_confidence)]
    pub dspark_reuse_floor: Option<f64>,
    /// Compatibility spelling: decode lanes always advance independently.
    #[arg(long, hide = true)]
    pub independent_decode_lanes: bool,

    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long)] pub native_lib: PathBuf,
    #[arg(long,value_delimiter=',',num_args=1..)] pub peers: Vec<std::net::SocketAddr>,
    #[arg(long,default_value="127.0.0.1:8000")] pub listen: String,
}

fn parse_dspark_confidence(value: &str) -> Result<f64, String> {
    let value: f64 = value.parse().map_err(|_| "expected a probability".to_string())?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("confidence cutoff must be finite and between zero and one".into());
    }
    Ok(value)
}

impl NativeServeArgs {
    pub fn adaptive_dspark(&self) -> bool {
        self.dspark && !self.dspark_fixed && self.dspark_confidence_cutoff.is_none()
    }
}
