use anyhow::Result;
use clap::Parser;
use serde::Serialize;
use std::process::Command;

mod cli;
mod commands;
mod python_graph_capture;
mod v41_compressor;
mod v41_vision;
mod v41_index_query;
mod v41_index_lane;
mod v41_index_selection;
mod v41_experts;
mod v41_memory;
mod v41_window;
mod v41_sparse_attention;
mod v41_attention_query;
mod v41_layer_graphs;
mod v41_attention_binding;
mod v41_hc;
mod v41_shared_ffn;
mod v41_backbone_shared;
mod v41_backbone_router;
mod v41_backbone_hc;
mod v41_block;
mod v41_backbone_lane;
mod v41_backbone_cache;
mod v41_backbone_execution;
mod v41_requests;
mod v41_target_head;
mod v41_target_pass;
mod v41_native_serve;
mod v41_target_embedding;
mod v41_attention_output;
mod v41_projection_tp2;
mod v41_dspark_cache;
mod v41_spark_topology;
mod v41_tensors;
mod v41_engram;

use cli::{Cli, Commands};
use commands::bench_cuda_kernels::run_bench_cuda_kernels;
use commands::bench_expert_reduction_replay::run_bench_expert_reduction_replay;
use commands::bench_protocol_v2_tcp::run_bench_protocol_v2_tcp;
use commands::bench_rdma::run_bench_rdma;
use commands::bench_rdma_ring::run_bench_rdma_ring;
use commands::coordinator::run_coordinator;
use commands::doctor::run_doctor;
use commands::expertd::run_expertd;
use commands::model_artifacts::{
    run_inspect_model, run_load_tensors, run_make_loadplan, run_tokenize,
};
use commands::scheduler_row_audit::run_scheduler_row_audit;
use commands::scheduler_smoke::run_scheduler_smoke;
use commands::transport_capabilities::run_transport_capabilities;

#[derive(Debug, Serialize)]
pub(crate) struct Probe {
    pub(crate) ok: bool,
    pub(crate) output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Doctor(args) => run_doctor(args),
        Commands::InspectModel(args) => run_inspect_model(args),
        Commands::MakeLoadplan(args) => run_make_loadplan(args),
        Commands::LoadTensors(args) => run_load_tensors(args),
        Commands::Tokenize(args) => run_tokenize(args),
        Commands::Coordinator(args) => run_coordinator(args).await,
        Commands::Expertd(args) => run_expertd(args).await,
        Commands::ExpertdNative(args) => v41_experts::service::run(args).await,
        Commands::ServeNative(args) => v41_native_serve::run(args).await,
        Commands::BenchRdma(args) => run_bench_rdma(args),
        Commands::BenchRdmaRing(args) => run_bench_rdma_ring(args),
        Commands::BenchCudaKernels(args) => run_bench_cuda_kernels(args),
        Commands::BenchProtocolV2Tcp(args) => run_bench_protocol_v2_tcp(args).await,
        Commands::BenchExpertReductionReplay(args) => run_bench_expert_reduction_replay(args).await,
        Commands::TransportCapabilities(args) => run_transport_capabilities(args),
        Commands::SchedulerSmoke(args) => run_scheduler_smoke(args),
        Commands::SchedulerRowAudit(args) => run_scheduler_row_audit(args),
    }
}

pub(crate) fn command_probe(program: &str, args: &[&str]) -> Probe {
    match Command::new(program).args(args).output() {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            Probe {
                ok: output.status.success(),
                output: text.trim().to_owned(),
            }
        }
        Err(err) => Probe {
            ok: false,
            output: err.to_string(),
        },
    }
}
