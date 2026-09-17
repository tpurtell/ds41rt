mod v41_candidate_blocks;
mod v41_device_ops;
pub use v41_device_ops::V41Bf16Add;
pub use v41_candidate_blocks::V41CandidateBlocks;
mod v41_index_topk;
pub use v41_index_topk::V41IndexTopK;
mod v41_index_scores;
mod v41_kv;
mod v41_sparse_attention;
pub use v41_sparse_attention::{V41SparseAttention, V41SparseBatch, V41SparseSource, V41SparseWindow};
pub use v41_index_scores::V41IndexScores;
pub use v41_kv::V41Kv;
mod v41_compressor;
mod v41_vision;
pub use v41_vision::V41VisionOps;
pub use v41_compressor::V41Compressor;
mod v41_dspark_attention;
pub use v41_dspark_attention::{V41DsparkAttention,V41AttentionWindow};
mod v41_grouped_output;
pub use v41_grouped_output::{V41GroupedOutput, V41_GROUPED_OUTPUT_WORKSPACE};
mod v41_attention_ops;
pub use v41_attention_ops::V41AttentionOps;
mod v41_dspark_cache;
pub use v41_dspark_cache::{V41DsparkCache, V41KvWrite};
mod v41_router;
pub use v41_router::V41Router;
mod v41_hc;
pub use v41_hc::V41Hc;
mod v41_dspark;
pub use v41_dspark::{V41DraftStep, V41DsparkConfidence, V41VocabularyProjection};
mod v41_fp8;
pub use v41_fp8::{V41Fp8Info, V41Fp8Kernel, V41SharedSwiGlu};
mod v41_fp8_plan;
pub use v41_fp8_plan::{V41Fp8Plan, V41Fp8PlanInfo};
mod v41_experts;
mod v41_exl3;
pub use v41_exl3::{V41Exl3Info, V41Exl3Kernel, V41Exl3Layout, V41Exl3Routes};
mod v41_exl3_wire;
pub use v41_exl3_wire::V41Exl3Wire;
pub use v41_experts::{
    V41ExpertInfo, V41ExpertInputQuantizer, V41ExpertKernel, V41ExpertLaunchArgs, V41ExpertPacker, V41ExpertPointer,
    V41CompactReducer, V41LocalExpertReducer, V41Tp2ExpertReducer, V41RouteReducer, V41_EXPERT_POINTER_COUNT,
};
mod cuda_runtime;
pub use cuda_runtime::{select_copy_mechanism, CopyMechanism, CudaRuntime};

use anyhow::{Context, Result};
use libloading::{Library, Symbol};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub type Ds41rtStatus = c_int;

pub const DS41RT_STATUS_OK: Ds41rtStatus = 0;
pub const DS41RT_STATUS_CUDA_UNAVAILABLE: Ds41rtStatus = 3;
pub const DS41RT_STATUS_RDMA_UNAVAILABLE: Ds41rtStatus = 7;
pub const DS41RT_STATUS_NCCL_UNAVAILABLE: Ds41rtStatus = 8;
pub const DS41RT_DEVICE_BUFFER_FLAG_HOST_FALLBACK: u64 = 1;
pub const DS41RT_DEVICE_BUFFER_FLAG_MANAGED: u64 = 2;
pub const DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST: u64 = 4;
pub const DS41RT_HOST_BUFFER_FLAG_NONE: u64 = 0;
pub const DS41RT_HOST_BUFFER_FLAG_PINNED: u64 = 1;
pub const DS41RT_HOST_BUFFER_FLAG_HOST_FALLBACK: u64 = 2;
pub const DS41RT_HOST_BUFFER_FLAG_MAPPED: u64 = 4;
pub const DS41RT_ROUTE_SHARD_WIRE_BF16: u32 = 1;
pub const DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED: u32 = 2;
pub const DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3: u32 = 3;
pub const DS41RT_ROUTE_SHARD_LOCAL_F32: u32 = 1;
pub const DS41RT_ROUTE_SHARD_LOCAL_BF16: u32 = 2;
pub const DS41RT_CUDA_ROUTER_TOPK_MAX_K: usize = 64;
pub const DS41RT_CUDA_SAMPLE_TOPK_MAX_K: usize = 64;
pub const DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES: usize = 512;
pub const DS41RT_CUDA_MLA_FP8_DS_ROPE_VALUES: usize = 64;
pub const DS41RT_CUDA_MLA_FP8_DS_PROJECTED_VALUES: usize =
    DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES + DS41RT_CUDA_MLA_FP8_DS_ROPE_VALUES;
pub const DS41RT_CUDA_MLA_FP8_DS_SCALE_BYTES: usize = 16;
pub const DS41RT_CUDA_MLA_FP8_DS_PACKED_BYTES: usize = DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES
    + DS41RT_CUDA_MLA_FP8_DS_SCALE_BYTES
    + DS41RT_CUDA_MLA_FP8_DS_ROPE_VALUES * std::mem::size_of::<u16>();
/// Generic transactional-KV page width; DeepSeek target attention separately
/// owns a 256-token physical-page ABI.
pub const DS41RT_CUDA_GENERIC_KV_PAGE_SIZE: usize = 64;
pub const DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES: usize = 512;
pub const DS41RT_CUDA_MLA_MXFP4_DS_ROPE_VALUES: usize = 64;
pub const DS41RT_CUDA_MLA_MXFP4_DS_PROJECTED_VALUES: usize =
    DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES + DS41RT_CUDA_MLA_MXFP4_DS_ROPE_VALUES;
pub const DS41RT_CUDA_MLA_MXFP4_DS_BLOCK_SIZE: usize = 16;
pub const DS41RT_CUDA_MLA_MXFP4_DS_CODE_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES / 2;
pub const DS41RT_CUDA_MLA_MXFP4_DS_SCALE_BYTES: usize =
    DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES / DS41RT_CUDA_MLA_MXFP4_DS_BLOCK_SIZE;
pub const DS41RT_CUDA_MLA_MXFP4_DS_PADDING_BYTES: usize = 16;
pub const DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_CODE_BYTES
    + DS41RT_CUDA_MLA_MXFP4_DS_SCALE_BYTES
    + DS41RT_CUDA_MLA_MXFP4_DS_PADDING_BYTES
    + DS41RT_CUDA_MLA_MXFP4_DS_ROPE_VALUES * std::mem::size_of::<u16>();

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtCudaDeviceInfo {
    pub device_id: c_int,
    pub cuda_available: c_int,
    pub compute_capability_major: c_int,
    pub compute_capability_minor: c_int,
    pub integrated: c_int,
    pub can_map_host_memory: c_int,
    pub unified_addressing: c_int,
    pub total_memory_bytes: u64,
    pub name: [c_char; 128],
    pub driver_version: [c_char; 64],
    pub runtime_version: [c_char; 64],
}

impl Default for Ds41rtCudaDeviceInfo {
    fn default() -> Self {
        Self {
            device_id: 0,
            cuda_available: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            integrated: 0,
            can_map_host_memory: 0,
            unified_addressing: 0,
            total_memory_bytes: 0,
            name: [0; 128],
            driver_version: [0; 64],
            runtime_version: [0; 64],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtDeviceBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub device_id: c_int,
    pub flags: u64,
}

impl Default for Ds41rtDeviceBuffer {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            device_id: -1,
            flags: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtRouteShardReductionBuffers {
    pub local: Ds41rtDeviceBuffer,
    pub peers: [Ds41rtDeviceBuffer; 3],
    pub output_f32: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtHostBuffer {
    pub ptr: *mut c_void,
    pub bytes: usize,
    pub flags: u64,
}

impl Default for Ds41rtHostBuffer {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            flags: DS41RT_HOST_BUFFER_FLAG_NONE,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtNvfp4RouteBatchedMetadata {
    pub gate_weight: usize,
    pub gate_scale: usize,
    pub up_weight: usize,
    pub up_scale: usize,
    pub down_weight: usize,
    pub down_scale: usize,
    pub intermediate: usize,
    pub down_weight_row_stride_bytes: usize,
    pub down_scale_row_stride_bytes: usize,
    pub gate_scale_2: f32,
    pub up_scale_2: f32,
    pub down_scale_2: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtB12xSparkW4a16MoeBuffers {
    pub input: Ds41rtDeviceBuffer,
    pub w13_weight: Ds41rtDeviceBuffer,
    pub w2_weight: Ds41rtDeviceBuffer,
    pub fc1_output: Ds41rtDeviceBuffer,
    pub activated: Ds41rtDeviceBuffer,
    pub output: Ds41rtDeviceBuffer,
    pub w13_scale: Ds41rtDeviceBuffer,
    pub w2_scale: Ds41rtDeviceBuffer,
    pub w13_global_scale: Ds41rtDeviceBuffer,
    pub w2_global_scale: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub topk_weights: Ds41rtDeviceBuffer,
    pub fc1_scratch: Ds41rtDeviceBuffer,
    pub fc2_scratch: Ds41rtDeviceBuffer,
    pub locks: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4FlashSparkW4a16MoeBuffers {
    pub input: Ds41rtDeviceBuffer,
    pub w13_weight: Ds41rtDeviceBuffer,
    pub w2_weight: Ds41rtDeviceBuffer,
    pub fc1_output: Ds41rtDeviceBuffer,
    pub activated: Ds41rtDeviceBuffer,
    pub routed_output: Ds41rtDeviceBuffer,
    pub output: Ds41rtDeviceBuffer,
    pub w13_scale: Ds41rtDeviceBuffer,
    pub w2_scale: Ds41rtDeviceBuffer,
    pub w13_global_scale: Ds41rtDeviceBuffer,
    pub w2_global_scale: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub topk_weights: Ds41rtDeviceBuffer,
    pub fc1_scratch: Ds41rtDeviceBuffer,
    pub fc2_scratch: Ds41rtDeviceBuffer,
    pub locks: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4FlashSparkExl3K2MoeBuffers {
    pub input: Ds41rtDeviceBuffer,
    pub w13_trellis: Ds41rtDeviceBuffer,
    pub w2_trellis: Ds41rtDeviceBuffer,
    pub gate_suh: Ds41rtDeviceBuffer,
    pub up_suh: Ds41rtDeviceBuffer,
    pub intermediate_rotations: Ds41rtDeviceBuffer,
    pub down_svh: Ds41rtDeviceBuffer,
    pub expert_map: Ds41rtDeviceBuffer,
    pub dummy_scale: Ds41rtDeviceBuffer,
    pub trellis_lut: Ds41rtDeviceBuffer,
    pub global_scale: Ds41rtDeviceBuffer,
    pub topk_ids: Ds41rtDeviceBuffer,
    pub topk_weights: Ds41rtDeviceBuffer,
    pub rotation_gate: Ds41rtDeviceBuffer,
    pub rotation_up: Ds41rtDeviceBuffer,
    pub fc1_output: Ds41rtDeviceBuffer,
    pub activated: Ds41rtDeviceBuffer,
    pub routed_output: Ds41rtDeviceBuffer,
    pub output_f32: Ds41rtDeviceBuffer,
    pub output_bf16: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub expert_counts: Ds41rtDeviceBuffer,
    pub expert_offsets: Ds41rtDeviceBuffer,
    pub fc1_scratch: Ds41rtDeviceBuffer,
    pub fc2_scratch: Ds41rtDeviceBuffer,
    pub workspace: Ds41rtDeviceBuffer,
}

// Flash and Pro use distinct native symbols and validation contracts but the
// caller-owned EXL3 buffer table has the same C layout for both profiles.
pub type Ds41rtDs4ProSparkExl3K2MoeBuffers = Ds41rtDs4FlashSparkExl3K2MoeBuffers;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4FlashSparkExl3MixedMoeBuffers {
    pub input: Ds41rtDeviceBuffer,
    pub tier0_w13_trellis: Ds41rtDeviceBuffer,
    pub tier0_w2_trellis: Ds41rtDeviceBuffer,
    pub tier1_w13_trellis: Ds41rtDeviceBuffer,
    pub tier1_w2_trellis: Ds41rtDeviceBuffer,
    pub dummy_scale: Ds41rtDeviceBuffer,
    pub tier0_global_scale: Ds41rtDeviceBuffer,
    pub tier1_global_scale: Ds41rtDeviceBuffer,
    pub gate_suh: Ds41rtDeviceBuffer,
    pub up_suh: Ds41rtDeviceBuffer,
    pub intermediate_rotations: Ds41rtDeviceBuffer,
    pub down_svh: Ds41rtDeviceBuffer,
    pub trellis_lut: Ds41rtDeviceBuffer,
    pub global_to_combined: Ds41rtDeviceBuffer,
    pub descriptor_map: Ds41rtDeviceBuffer,
    pub topk_ids: Ds41rtDeviceBuffer,
    pub mapped_topk_ids: Ds41rtDeviceBuffer,
    pub topk_weights: Ds41rtDeviceBuffer,
    pub rotation_gate: Ds41rtDeviceBuffer,
    pub rotation_up: Ds41rtDeviceBuffer,
    pub fc1_output: Ds41rtDeviceBuffer,
    pub activated: Ds41rtDeviceBuffer,
    pub routed_output: Ds41rtDeviceBuffer,
    pub output_f32: Ds41rtDeviceBuffer,
    pub output_bf16: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub expert_counts: Ds41rtDeviceBuffer,
    pub expert_offsets: Ds41rtDeviceBuffer,
    pub fc1_scratch: Ds41rtDeviceBuffer,
    pub fc2_scratch: Ds41rtDeviceBuffer,
    pub workspace: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4FlashRoutePackBuffers {
    pub topk_ids: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub expert_counts: Ds41rtDeviceBuffer,
    pub expert_offsets: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4FlashSharedExpertFp8Buffers {
    pub input: Ds41rtDeviceBuffer,
    pub w1_weight: Ds41rtDeviceBuffer,
    pub w1_scale_mma: Ds41rtDeviceBuffer,
    pub w3_weight: Ds41rtDeviceBuffer,
    pub w3_scale_mma: Ds41rtDeviceBuffer,
    pub w2_weight: Ds41rtDeviceBuffer,
    pub w2_scale_mma: Ds41rtDeviceBuffer,
    pub gate: Ds41rtDeviceBuffer,
    pub up: Ds41rtDeviceBuffer,
    pub activated: Ds41rtDeviceBuffer,
    pub output: Ds41rtDeviceBuffer,
    pub alpha: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtDs4ProSharedExpertFp8PrefillBuffers {
    pub base: Ds41rtDs4FlashSharedExpertFp8Buffers,
    pub input_q_values: Ds41rtDeviceBuffer,
    pub input_q_scale_rows: Ds41rtDeviceBuffer,
    pub input_q_scale_mma: Ds41rtDeviceBuffer,
    pub activated_q_values: Ds41rtDeviceBuffer,
    pub activated_q_scale_rows: Ds41rtDeviceBuffer,
    pub activated_q_scale_mma: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtB12xCoordinatorW4a16Buffers {
    pub input: Ds41rtDeviceBuffer,
    pub weight: Ds41rtDeviceBuffer,
    pub output: Ds41rtDeviceBuffer,
    pub scale: Ds41rtDeviceBuffer,
    pub global_scale: Ds41rtDeviceBuffer,
    pub packed_route_indices: Ds41rtDeviceBuffer,
    pub block_expert_ids: Ds41rtDeviceBuffer,
    pub packed_route_count: Ds41rtDeviceBuffer,
    pub topk_weights: Ds41rtDeviceBuffer,
    pub c_tmp: Ds41rtDeviceBuffer,
    pub locks: Ds41rtDeviceBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtCudaGraphCaptureInfo {
    pub graph: *mut c_void,
    pub graph_exec: *mut c_void,
    pub node_count: usize,
    pub kernel_node_count: usize,
    pub memcpy_node_count: usize,
    pub memset_node_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ds41rtBf16Summary {
    pub checksum: f64,
    pub values: u64,
    pub finite_values: u64,
    pub nonzero_values: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaDeviceInfo {
    pub rdma_enabled: c_int,
    pub device_count: c_int,
    pub first_device_openable: c_int,
    pub first_device_guid: u64,
    pub first_device_name: [c_char; 128],
    pub first_device_transport: [c_char; 64],
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaDeviceInfo {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            device_count: 0,
            first_device_openable: 0,
            first_device_guid: 0,
            first_device_name: [0; 128],
            first_device_transport: [0; 64],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaHostBufferPlan {
    pub original_addr: usize,
    pub original_bytes: usize,
    pub alignment: usize,
    pub registered_addr: usize,
    pub prefix_bytes: usize,
    pub registered_span_bytes: usize,
    pub span_aligned: c_int,
    pub rdma_enabled: c_int,
}

impl Default for Ds41rtRdmaHostBufferPlan {
    fn default() -> Self {
        Self {
            original_addr: 0,
            original_bytes: 0,
            alignment: 0,
            registered_addr: 0,
            prefix_bytes: 0,
            registered_span_bytes: 0,
            span_aligned: 0,
            rdma_enabled: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRegisterProbe {
    pub bytes: usize,
    pub registered: c_int,
    pub lkey: u32,
    pub rkey: u32,
    pub device_name: [c_char; 128],
}

impl Default for Ds41rtRdmaRegisterProbe {
    fn default() -> Self {
        Self {
            bytes: 0,
            registered: 0,
            lkey: 0,
            rkey: 0,
            device_name: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcQpProbe {
    pub rdma_enabled: c_int,
    pub created: c_int,
    pub port_num: u32,
    pub qp_num: u32,
    pub lid: u32,
    pub active_mtu: u32,
    pub requested_send_wr: u32,
    pub requested_recv_wr: u32,
    pub requested_max_sge: u32,
    pub actual_max_send_wr: u32,
    pub actual_max_recv_wr: u32,
    pub actual_max_send_sge: u32,
    pub actual_max_recv_sge: u32,
    pub actual_max_inline_data: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaRcQpProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            created: 0,
            port_num: 0,
            qp_num: 0,
            lid: 0,
            active_mtu: 0,
            requested_send_wr: 0,
            requested_recv_wr: 0,
            requested_max_sge: 0,
            actual_max_send_wr: 0,
            actual_max_recv_wr: 0,
            actual_max_send_sge: 0,
            actual_max_recv_sge: 0,
            actual_max_inline_data: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcSendRecvProbe {
    pub rdma_enabled: c_int,
    pub completed: c_int,
    pub payload_matches: c_int,
    pub port_num: u32,
    pub bytes: usize,
    pub sender_qp_num: u32,
    pub receiver_qp_num: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaRcSendRecvProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            completed: 0,
            payload_matches: 0,
            port_num: 0,
            bytes: 0,
            sender_qp_num: 0,
            receiver_qp_num: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcProtocolV2LoopbackProbe {
    pub rdma_enabled: c_int,
    pub completed: c_int,
    pub request_payload_matches: c_int,
    pub response_payload_matches: c_int,
    pub port_num: u32,
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub client_qp_num: u32,
    pub server_qp_num: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaRcProtocolV2LoopbackProbe {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            completed: 0,
            request_payload_matches: 0,
            response_payload_matches: 0,
            port_num: 0,
            request_bytes: 0,
            response_bytes: 0,
            client_qp_num: 0,
            server_qp_num: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcEndpointInfo {
    pub rdma_enabled: c_int,
    pub handle: *mut c_void,
    pub port_num: u32,
    pub qp_num: u32,
    pub psn: u32,
    pub lid: u32,
    pub active_mtu: u32,
    pub send_frame_bytes: usize,
    pub recv_frame_bytes: usize,
    pub send_registered_span_bytes: usize,
    pub recv_registered_span_bytes: usize,
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_sge: u32,
    pub gid_hex: [c_char; 33],
    pub device_name: [c_char; 128],
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaRcEndpointInfo {
    fn default() -> Self {
        Self {
            rdma_enabled: 0,
            handle: std::ptr::null_mut(),
            port_num: 0,
            qp_num: 0,
            psn: 0,
            lid: 0,
            active_mtu: 0,
            send_frame_bytes: 0,
            recv_frame_bytes: 0,
            send_registered_span_bytes: 0,
            recv_registered_span_bytes: 0,
            max_send_wr: 0,
            max_recv_wr: 0,
            max_sge: 0,
            gid_hex: [0; 33],
            device_name: [0; 128],
            status: [0; 128],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcEndpointBufferView {
    pub host_ptr: *mut c_void,
    pub device_ptr: *mut c_void,
    pub bytes: usize,
    pub device_id: c_int,
    pub host_flags: u64,
}

impl Default for Ds41rtRdmaRcEndpointBufferView {
    fn default() -> Self {
        Self {
            host_ptr: std::ptr::null_mut(),
            device_ptr: std::ptr::null_mut(),
            bytes: 0,
            device_id: -1,
            host_flags: DS41RT_HOST_BUFFER_FLAG_NONE,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ds41rtRdmaRcCompletionStats {
    pub expected_send_completions: u32,
    pub expected_recv_completions: u32,
    pub send_completions: u32,
    pub recv_completions: u32,
    pub poll_iterations: u32,
    pub status: [c_char; 128],
}

impl Default for Ds41rtRdmaRcCompletionStats {
    fn default() -> Self {
        Self {
            expected_send_completions: 0,
            expected_recv_completions: 0,
            send_completions: 0,
            recv_completions: 0,
            poll_iterations: 0,
            status: [0; 128],
        }
    }
}

type VersionFn = unsafe extern "C" fn(out: *mut c_char, out_len: usize) -> Ds41rtStatus;
type DeviceInfoFn =
    unsafe extern "C" fn(device_id: c_int, out: *mut Ds41rtCudaDeviceInfo) -> Ds41rtStatus;
type AllocHostBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut Ds41rtHostBuffer) -> Ds41rtStatus;
type CudaHostBufferDeviceAliasFn =
    unsafe extern "C" fn(host: Ds41rtHostBuffer, out: *mut Ds41rtDeviceBuffer) -> Ds41rtStatus;
type FreeHostBufferFn = unsafe extern "C" fn(buf: *mut Ds41rtHostBuffer) -> Ds41rtStatus;
type AllocDeviceBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut Ds41rtDeviceBuffer) -> Ds41rtStatus;
type AllocManagedDeviceBufferFn =
    unsafe extern "C" fn(bytes: usize, out: *mut Ds41rtDeviceBuffer) -> Ds41rtStatus;
type FreeDeviceBufferFn = unsafe extern "C" fn(buf: *mut Ds41rtDeviceBuffer) -> Ds41rtStatus;
type CudaStreamCreateFn = unsafe extern "C" fn(out: *mut *mut c_void) -> Ds41rtStatus;
type CudaStreamDestroyFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaStreamSynchronizeFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaStreamQueryFn = unsafe extern "C" fn(cuda_stream: *mut c_void, ready: *mut i32) -> Ds41rtStatus;
type CudaStreamWaitEventFn =
    unsafe extern "C" fn(cuda_stream: *mut c_void, cuda_event: *mut c_void) -> Ds41rtStatus;
type CudaEventCreateFn = unsafe extern "C" fn(out: *mut *mut c_void) -> Ds41rtStatus;
type CudaEventDestroyFn = unsafe extern "C" fn(cuda_event: *mut c_void) -> Ds41rtStatus;
type CudaEventRecordFn =
    unsafe extern "C" fn(cuda_event: *mut c_void, cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaEventSynchronizeFn = unsafe extern "C" fn(cuda_event: *mut c_void) -> Ds41rtStatus;
type CudaEventElapsedMsFn = unsafe extern "C" fn(
    start_event: *mut c_void,
    end_event: *mut c_void,
    out_ms: *mut f32,
) -> Ds41rtStatus;
type CudaGraphBeginCaptureFn = unsafe extern "C" fn(cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaGraphEndCaptureFn = unsafe extern "C" fn(
    cuda_stream: *mut c_void,
    out_cuda_graph_exec: *mut *mut c_void,
) -> Ds41rtStatus;
type CudaGraphEndCaptureRetainedFn = unsafe extern "C" fn(
    cuda_stream: *mut c_void,
    out: *mut Ds41rtCudaGraphCaptureInfo,
) -> Ds41rtStatus;
type CudaGraphLaunchFn =
    unsafe extern "C" fn(cuda_graph_exec: *mut c_void, cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaGraphExecUpdateFn =
    unsafe extern "C" fn(cuda_graph_exec: *mut c_void, cuda_graph: *mut c_void) -> Ds41rtStatus;
type CudaGraphDestroyFn = unsafe extern "C" fn(cuda_graph: *mut c_void) -> Ds41rtStatus;
type CudaGraphExecDestroyFn = unsafe extern "C" fn(cuda_graph_exec: *mut c_void) -> Ds41rtStatus;
type CudaGraphUpdateRmsNormBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    x: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateLayerNormAffineF32Bf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    x: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    bias: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateLayerNormAffineBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    x: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    bias: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateLinearBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    input: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    bias: *const Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateEmbeddingLookupBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    embedding: Ds41rtDeviceBuffer,
    token_ids: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateLmHeadArgmaxBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateLmHeadSampleTopKToppBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateRouterTopKBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Ds41rtDeviceBuffer,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateSiluGatedMlpRowsBf16DownStrideNodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    x: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateResidualAddBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    residual: Ds41rtDeviceBuffer,
    delta: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    count: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateResidualAddF32DeltaBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    residual: Ds41rtDeviceBuffer,
    delta_f32: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    count: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateResidualAddSharedF32DeltaBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    residual: Ds41rtDeviceBuffer,
    shared_delta: Ds41rtDeviceBuffer,
    routed_delta_f32: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    count: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateCausalAttentionBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    q: Ds41rtDeviceBuffer,
    k: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateRopeBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    input: Ds41rtDeviceBuffer,
    positions: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateMlaRopeAttentionBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    q_nope: Ds41rtDeviceBuffer,
    q_rope: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateMlaRopeAttentionBf16SuffixNodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    q_nope: Ds41rtDeviceBuffer,
    q_rope: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaGraphUpdateMlaKvCacheUnpackBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    payload: Ds41rtDeviceBuffer,
    kv_latent: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    dsa_key: Ds41rtDeviceBuffer,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateMlaKvProjectedSplitBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    projected: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateF32ToBf16NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    src: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    count: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateScatterAddRowsBf16ToF32NodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    src: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaGraphUpdateKvCacheWriteBytesNodeFn = unsafe extern "C" fn(
    cuda_graph: *mut c_void,
    cuda_graph_exec: *mut c_void,
    kernel_node_index: usize,
    src: Ds41rtDeviceBuffer,
    cache: Ds41rtDeviceBuffer,
    cache_offset_bytes: usize,
    bytes: usize,
) -> Ds41rtStatus;
type CopyH2DFn =
    unsafe extern "C" fn(dst: Ds41rtDeviceBuffer, src: *const c_void, bytes: usize) -> Ds41rtStatus;
type CopyD2HFn =
    unsafe extern "C" fn(dst: *mut c_void, src: Ds41rtDeviceBuffer, bytes: usize) -> Ds41rtStatus;
type CopyD2DFn = unsafe extern "C" fn(
    dst: Ds41rtDeviceBuffer,
    src: Ds41rtDeviceBuffer,
    bytes: usize,
) -> Ds41rtStatus;
type CopyH2DAsyncFn = unsafe extern "C" fn(
    dst: Ds41rtDeviceBuffer,
    src: *const c_void,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CopyH2DBatchAsyncFn = unsafe extern "C" fn(
    dsts: *const Ds41rtDeviceBuffer,
    srcs: *const *const c_void,
    bytes: *const usize,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CopyH2D2DAsyncFn = unsafe extern "C" fn(
    dst: Ds41rtDeviceBuffer,
    dst_pitch_bytes: usize,
    src: *const c_void,
    src_pitch_bytes: usize,
    width_bytes: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CopyD2HAsyncFn = unsafe extern "C" fn(
    dst: *mut c_void,
    src: Ds41rtDeviceBuffer,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CopyD2DAsyncFn = unsafe extern "C" fn(
    dst: Ds41rtDeviceBuffer,
    src: Ds41rtDeviceBuffer,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CopyD2D2DAsyncFn = unsafe extern "C" fn(
    dst: Ds41rtDeviceBuffer,
    dst_pitch_bytes: usize,
    src: Ds41rtDeviceBuffer,
    src_pitch_bytes: usize,
    width_bytes: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type LastErrorFn = unsafe extern "C" fn(out: *mut c_char, out_len: usize) -> Ds41rtStatus;
type NcclUniqueIdBytesFn = unsafe extern "C" fn(out_bytes: *mut usize) -> Ds41rtStatus;
type NcclGetUniqueIdFn = unsafe extern "C" fn(out: *mut c_void, out_bytes: usize) -> Ds41rtStatus;
type NcclCommInitRankFn = unsafe extern "C" fn(
    unique_id: *const c_void,
    unique_id_bytes: usize,
    world_size: c_int,
    rank: c_int,
    out_handle: *mut *mut c_void,
) -> Ds41rtStatus;
type NcclGatherU8AsyncFn = unsafe extern "C" fn(
    handle: *mut c_void,
    send: Ds41rtDeviceBuffer,
    recv: Ds41rtDeviceBuffer,
    bytes: usize,
    root: c_int,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type NcclRowAllToAllU8AsyncFn = unsafe extern "C" fn(
    handle: *mut c_void,
    send: Ds41rtDeviceBuffer,
    recv: Ds41rtDeviceBuffer,
    rows: usize,
    row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type NcclAllReduceBf16AsyncFn = unsafe extern "C" fn(
    handle: *mut c_void,
    send: Ds41rtDeviceBuffer,
    recv: Ds41rtDeviceBuffer,
    values: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type NcclReduceBf16AsyncFn = unsafe extern "C" fn(
    handle: *mut c_void,
    send: Ds41rtDeviceBuffer,
    recv: Ds41rtDeviceBuffer,
    values: usize,
    root: c_int,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type NcclCommDestroyFn = unsafe extern "C" fn(handle: *mut c_void) -> Ds41rtStatus;
type CudaRmsNormF32Fn = unsafe extern "C" fn(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaRmsNormF32AsyncFn = unsafe extern "C" fn(
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    rows: c_int,
    hidden: c_int,
    eps: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRmsNormBf16Fn = unsafe extern "C" fn(
    x: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaRmsNormBf16AsyncFn = unsafe extern "C" fn(
    x: *const u16,
    weight: *const u16,
    out: *mut u16,
    rows: c_int,
    hidden: c_int,
    eps: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashRmsNormBf16AsyncFn = CudaRmsNormBf16AsyncFn;
type CudaDs4RmsNormBf16RneAsyncFn = CudaRmsNormBf16AsyncFn;
type CudaLayerNormAffineF32Bf16Fn = unsafe extern "C" fn(
    x: *const f32,
    weight: *const u16,
    bias: *const u16,
    out: *mut f32,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaLayerNormAffineF32Bf16AsyncFn = unsafe extern "C" fn(
    x: *const f32,
    weight: *const u16,
    bias: *const u16,
    out: *mut f32,
    rows: c_int,
    hidden: c_int,
    eps: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLayerNormAffineBf16Fn = unsafe extern "C" fn(
    x: *const u16,
    weight: *const u16,
    bias: *const u16,
    out: *mut u16,
    rows: c_int,
    hidden: c_int,
    eps: f32,
) -> Ds41rtStatus;
type CudaLayerNormAffineBf16AsyncFn = unsafe extern "C" fn(
    x: *const u16,
    weight: *const u16,
    bias: *const u16,
    out: *mut u16,
    rows: c_int,
    hidden: c_int,
    eps: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSiluGatedMlpF32Fn = unsafe extern "C" fn(
    x: *const f32,
    gate_weight: *const f32,
    up_weight: *const f32,
    down_weight: *const f32,
    out: *mut f32,
    hidden: c_int,
    intermediate: c_int,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsF32Fn = unsafe extern "C" fn(
    x: *const f32,
    gate_weight: *const f32,
    up_weight: *const f32,
    down_weight: *const f32,
    out: *mut f32,
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsF32AsyncFn = unsafe extern "C" fn(
    x: *const f32,
    gate_weight: *const f32,
    up_weight: *const f32,
    down_weight: *const f32,
    out: *mut f32,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16Fn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16AsyncFn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSiluMulBf16AsyncFn = unsafe extern "C" fn(
    gate_up: *const u16,
    out: *mut u16,
    rows: usize,
    intermediate: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16DownStrideFn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16DownStrideAsyncFn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16DownStrideStagedFn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    activation_workspace: *mut f32,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Ds41rtStatus;
type CudaSiluGatedMlpRowsBf16DownStrideStagedAsyncFn = unsafe extern "C" fn(
    x: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    activation_workspace: *mut f32,
    out: *mut u16,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16GroupedStagedAccumulateF32Fn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        gate_weight: *const u8,
        gate_scale: *const u8,
        up_weight: *const u8,
        up_scale: *const u8,
        down_weight: *const u8,
        down_scale: *const u8,
        activation_workspace: *mut f32,
        accumulator: *mut f32,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        intermediate: usize,
        output_dim: usize,
        down_weight_row_stride_bytes: usize,
        down_scale_row_stride_bytes: usize,
        gate_scale_2: f32,
        up_scale_2: f32,
        down_scale_2: f32,
    ) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16GroupedStagedAccumulateF32AsyncFn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        gate_weight: *const u8,
        gate_scale: *const u8,
        up_weight: *const u8,
        up_scale: *const u8,
        down_weight: *const u8,
        down_scale: *const u8,
        activation_workspace: *mut f32,
        accumulator: *mut f32,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        intermediate: usize,
        output_dim: usize,
        down_weight_row_stride_bytes: usize,
        down_scale_row_stride_bytes: usize,
        gate_scale_2: f32,
        up_scale_2: f32,
        down_scale_2: f32,
        cuda_stream: *mut c_void,
    ) -> Ds41rtStatus;
type CudaB12xSparkAotAvailableFn = unsafe extern "C" fn(out_available: *mut c_int) -> Ds41rtStatus;
type CudaB12xSparkAotInitFn = unsafe extern "C" fn() -> Ds41rtStatus;
type CudaDs4FlashSparkAotAvailableFn =
    unsafe extern "C" fn(out_available: *mut c_int) -> Ds41rtStatus;
type CudaDs4FlashSparkAotInitFn = unsafe extern "C" fn() -> Ds41rtStatus;
type CudaDs4ProSparkAotAvailableFn = CudaDs4FlashSparkAotAvailableFn;
type CudaDs4ProSparkAotInitFn = CudaDs4FlashSparkAotInitFn;
type CudaB12xQuantizeBf16Nvfp4RowPayloadAsyncFn = unsafe extern "C" fn(
    input: Ds41rtDeviceBuffer,
    payload: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xW4a16PackWeightAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    size_n: usize,
    row_rotation: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xW4a16PackWeightStridedAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    source_size_k: usize,
    source_start_k: usize,
    size_n: usize,
    row_rotation: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xW4a16PackScaleAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    size_n: usize,
    row_rotation: usize,
    scale_factor: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xW4a16PackScaleStridedAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    source_size_k: usize,
    source_start_k: usize,
    size_n: usize,
    row_rotation: usize,
    scale_factor: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashW4a16PackWeightAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    size_n: usize,
    row_rotation: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashW4a16PackWeightStridedAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_k: usize,
    source_size_k: usize,
    source_start_k: usize,
    size_n: usize,
    row_rotation: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashW4a16PackE8m0ScaleAsyncFn = CudaDs4FlashW4a16PackWeightAsyncFn;
type CudaDs4FlashW4a16PackE8m0ScaleStridedAsyncFn = CudaDs4FlashW4a16PackWeightStridedAsyncFn;
type CudaDs4FlashFp8PackBlockScaleMmaAsyncFn = unsafe extern "C" fn(
    source: Ds41rtDeviceBuffer,
    destination: Ds41rtDeviceBuffer,
    size_n: usize,
    size_k: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSharedExpertFp8Bf16AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSharedExpertFp8Buffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4ProSharedExpertFp8Bf16AsyncFn = CudaDs4FlashSharedExpertFp8Bf16AsyncFn;
type CudaDs4ProSharedExpertFp8PrefillBf16AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4ProSharedExpertFp8PrefillBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSparkW4a16DecodeM1Bf16AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSparkW4a16MoeBuffers,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSparkW4a16PrefillTopk6Bf16AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSparkW4a16MoeBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSparkExl3K2DecodeM1AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSparkExl3K2MoeBuffers,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSparkExl3K2PrefillTopk6AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSparkExl3K2MoeBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashSparkExl3K3DecodeM1AsyncFn = CudaDs4FlashSparkExl3K2DecodeM1AsyncFn;
type CudaDs4FlashSparkExl3K3PrefillTopk6AsyncFn = CudaDs4FlashSparkExl3K2PrefillTopk6AsyncFn;
type CudaDs4FlashSparkExl3MixedK2K3AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashSparkExl3MixedMoeBuffers,
    tier0_slots: usize,
    tier1_slots: usize,
    tier0_gate_experts: usize,
    tier1_gate_experts: usize,
    tier0_up_experts: usize,
    tier1_up_experts: usize,
    tier0_down_experts: usize,
    tier1_down_experts: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4ProSparkExl3K2DecodeM1AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4ProSparkExl3K2MoeBuffers,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4ProSparkExl3K2PrefillTopk6AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4ProSparkExl3K2MoeBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashPackTopk6RoutesAsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtDs4FlashRoutePackBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaQuantizeBf16WeightNvfp4AsyncFn = unsafe extern "C" fn(
    input: Ds41rtDeviceBuffer,
    packed: Ds41rtDeviceBuffer,
    scales: Ds41rtDeviceBuffer,
    rows: usize,
    cols: usize,
    global_scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xGatherNvfp4RowsBf16AsyncFn = unsafe extern "C" fn(
    payload: Ds41rtDeviceBuffer,
    source_rows: usize,
    source_row_stride_bytes: usize,
    row_indices: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xDequantizeNvfp4RowsBf16AsyncFn = unsafe extern "C" fn(
    payload: Ds41rtDeviceBuffer,
    row_stride_bytes: usize,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16DecodeM1Nvfp4AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    topk_ids: Ds41rtDeviceBuffer,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16M1ParityM2To8Nvfp4AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    topk_ids: Ds41rtDeviceBuffer,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16M1ParityGroupedM2To8Nvfp4AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16PrefillTopk8Nvfp4AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16PrefillTopk8Nvfp4Fp8AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    rows: usize,
    output_fp8: Ds41rtDeviceBuffer,
    output_fp8_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xSparkW4a16Top1AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xSparkW4a16MoeBuffers,
    rows: usize,
    capacity_rows: usize,
    expert_id: u32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xCoordinatorAotAvailableFn =
    unsafe extern "C" fn(out_available: *mut c_int) -> Ds41rtStatus;
type CudaB12xCoordinatorAotInitFn = unsafe extern "C" fn() -> Ds41rtStatus;
type CudaB12xCoordinatorW4a16QuantizePackWeightAsyncFn = unsafe extern "C" fn(
    input_bf16: Ds41rtDeviceBuffer,
    payload_scratch: Ds41rtDeviceBuffer,
    packed_weight: Ds41rtDeviceBuffer,
    packed_scale: Ds41rtDeviceBuffer,
    global_scale: Ds41rtDeviceBuffer,
    size_k: usize,
    size_n: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xCoordinatorW4a16BuffersAsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xCoordinatorW4a16Buffers,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaB12xCoordinatorW4a16BuffersRowsAsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xCoordinatorW4a16Buffers,
    active_rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaW8a16PackedOInitializeAsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtB12xCoordinatorW4a16Buffers,
    rows: usize,
    block_m: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedAccumulateF32Fn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        route_metadata: *const Ds41rtNvfp4RouteBatchedMetadata,
        activation_workspace: *mut f32,
        accumulator: *mut f32,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
    ) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedAccumulateF32AsyncFn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        route_metadata: *const Ds41rtNvfp4RouteBatchedMetadata,
        activation_workspace: *mut f32,
        accumulator: *mut f32,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedSingleRowBf16Fn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        route_metadata: *const Ds41rtNvfp4RouteBatchedMetadata,
        activation_workspace: *mut f32,
        out: *mut u16,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
    ) -> Ds41rtStatus;
type CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedSingleRowBf16AsyncFn =
    unsafe extern "C" fn(
        hidden: *const u16,
        row_indices: *const u32,
        route_weights: *const f32,
        route_metadata: *const Ds41rtNvfp4RouteBatchedMetadata,
        activation_workspace: *mut f32,
        out: *mut u16,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Ds41rtStatus;
type CudaResidualAddF32Fn = unsafe extern "C" fn(
    residual: *const f32,
    delta: *const f32,
    out: *mut f32,
    count: usize,
) -> Ds41rtStatus;
type CudaResidualAddF32AsyncFn = unsafe extern "C" fn(
    residual: *const f32,
    delta: *const f32,
    out: *mut f32,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaResidualAddBf16Fn = unsafe extern "C" fn(
    residual: *const u16,
    delta: *const u16,
    out: *mut u16,
    count: usize,
) -> Ds41rtStatus;
type CudaResidualAddBf16AsyncFn = unsafe extern "C" fn(
    residual: *const u16,
    delta: *const u16,
    out: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaResidualAddF32DeltaBf16Fn = unsafe extern "C" fn(
    residual: *const u16,
    delta_f32: *const f32,
    out: *mut u16,
    count: usize,
) -> Ds41rtStatus;
type CudaResidualAddF32DeltaBf16AsyncFn = unsafe extern "C" fn(
    residual: *const u16,
    delta_f32: *const f32,
    out: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaResidualAddSharedF32DeltaBf16Fn = unsafe extern "C" fn(
    residual: *const u16,
    shared_delta: *const u16,
    routed_delta_f32: *const f32,
    out: *mut u16,
    count: usize,
) -> Ds41rtStatus;
type CudaResidualAddSharedF32DeltaBf16AsyncFn = unsafe extern "C" fn(
    residual: *const u16,
    shared_delta: *const u16,
    routed_delta_f32: *const f32,
    out: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaResidualAddSharedFp8E4m3RowScaledBf16AsyncFn = unsafe extern "C" fn(
    residual: *const u16,
    shared_delta: *const u16,
    routed_delta_fp8: *const u8,
    out: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaFp8DecodeCombineResidualAsyncFn = unsafe extern "C" fn(
    residual: *const u16,
    shared_delta: *const u16,
    partials: *const u8,
    partial_row_stride_bytes: usize,
    output: *mut u16,
    partial_rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSchedulerMlpDeltaBf16Fn = unsafe extern "C" fn(
    hidden: *const u16,
    gate_weight: *const u16,
    up_weight: *const u16,
    down_weight: *const u16,
    out: *mut u16,
    rows: usize,
    hidden_dim: usize,
) -> Ds41rtStatus;
type CudaSummarizeBf16Fn = unsafe extern "C" fn(
    input: *const u16,
    count: usize,
    out: *mut Ds41rtBf16Summary,
) -> Ds41rtStatus;
type CudaSummarizeBf16AsyncFn = unsafe extern "C" fn(
    input: *const u16,
    count: usize,
    out_device: *mut Ds41rtBf16Summary,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaZeroF32Fn = unsafe extern "C" fn(dst: *mut f32, count: usize) -> Ds41rtStatus;
type CudaZeroF32AsyncFn =
    unsafe extern "C" fn(dst: *mut f32, count: usize, cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaAccumulateBf16ToF32Fn =
    unsafe extern "C" fn(src: *const u16, dst: *mut f32, count: usize) -> Ds41rtStatus;
type CudaAccumulateBf16ToF32AsyncFn = unsafe extern "C" fn(
    src: *const u16,
    dst: *mut f32,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaZeroBytesFn = unsafe extern "C" fn(dst: *mut c_void, bytes: usize) -> Ds41rtStatus;
type CudaZeroBytesAsyncFn =
    unsafe extern "C" fn(dst: *mut c_void, bytes: usize, cuda_stream: *mut c_void) -> Ds41rtStatus;
type CudaF32ToBf16Fn =
    unsafe extern "C" fn(src: *const f32, dst: *mut u16, count: usize) -> Ds41rtStatus;
type CudaF32ToBf16AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    dst: *mut u16,
    count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGatherRowsF32Fn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaGatherRowsF32AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGatherRowsF32ToFp8E4m3RowScaledFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaGatherRowsF32ToFp8E4m3RowScaledAsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaBf16RowsToFp8E4m3RowScaledAsyncFn = unsafe extern "C" fn(
    src: *const u16,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaCombineFp8E4m3RowScaledToFp8AsyncFn = unsafe extern "C" fn(
    local: *const f32,
    peers: *const u8,
    peer_payload_stride_bytes: usize,
    peer_count: usize,
    peer_row_stride_bytes: usize,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaCombineBf16Fp8E4m3RowScaledToFp8AsyncFn = unsafe extern "C" fn(
    local: *const u16,
    peers: *const u8,
    peer_payload_stride_bytes: usize,
    peer_count: usize,
    peer_row_stride_bytes: usize,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3Fn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut u8,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGatherRowsBf16Fn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    dst: *mut u16,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaGatherRowsBf16AsyncFn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    dst: *mut u16,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaCopyRowPrefixBf16Fn = unsafe extern "C" fn(
    src: *const u16,
    dst: *mut u16,
    rows: usize,
    src_row_width: usize,
    dst_row_width: usize,
    prefix_width: usize,
    src_row_offset: usize,
) -> Ds41rtStatus;
type CudaCopyRowPrefixBf16AsyncFn = unsafe extern "C" fn(
    src: *const u16,
    dst: *mut u16,
    rows: usize,
    src_row_width: usize,
    dst_row_width: usize,
    prefix_width: usize,
    src_row_offset: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaScatterAddRowsF32Fn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaScatterAddRowsF32AsyncFn = unsafe extern "C" fn(
    src: *const f32,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaScatterAddRowsBf16ToF32Fn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaScatterAddRowsBf16ToF32AsyncFn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaScatterAddRowsFp8E4m3RowScaledToF32Fn = unsafe extern "C" fn(
    src: *const u8,
    src_row_stride_bytes: usize,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaScatterAddRowsFp8E4m3RowScaledToF32AsyncFn = unsafe extern "C" fn(
    src: *const u8,
    src_row_stride_bytes: usize,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaScatterAddRowsNvfp4E2m1Fp8E4m3ToF32Fn = unsafe extern "C" fn(
    src: *const u8,
    src_row_stride_bytes: usize,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaScatterAddRowsNvfp4E2m1Fp8E4m3ToF32AsyncFn = unsafe extern "C" fn(
    src: *const u8,
    src_row_stride_bytes: usize,
    row_indices: *const u32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaReduceRouteShardsToF32Fn = unsafe extern "C" fn(
    buffers: *const Ds41rtRouteShardReductionBuffers,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    local_dtype: u32,
    peer_dtype: u32,
    peer_count: u32,
) -> Ds41rtStatus;
type CudaReduceRouteShardsToF32AsyncFn = unsafe extern "C" fn(
    buffers: *const Ds41rtRouteShardReductionBuffers,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    local_dtype: u32,
    peer_dtype: u32,
    peer_count: u32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaScatterAddRowsBf16WeightedToF32Fn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    row_weights: *const f32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
) -> Ds41rtStatus;
type CudaScatterAddRowsBf16WeightedToF32AsyncFn = unsafe extern "C" fn(
    src: *const u16,
    row_indices: *const u32,
    row_weights: *const f32,
    dst: *mut f32,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRouterTopKF32Fn = unsafe extern "C" fn(
    hidden: *const f32,
    router_weight: *const f32,
    correction_bias: *const f32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Ds41rtStatus;
type CudaRouterTopKF32AsyncFn = unsafe extern "C" fn(
    hidden: *const f32,
    router_weight: *const f32,
    correction_bias: *const f32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRouterTopKBf16Fn = unsafe extern "C" fn(
    hidden: *const u16,
    router_weight: *const u16,
    correction_bias: *const f32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Ds41rtStatus;
type CudaRouterTopKBf16AsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    router_weight: *const u16,
    correction_bias: *const f32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashRouterTopKBf16AsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    router_weight: *const u16,
    correction_bias: *const f32,
    token_to_experts: *const i64,
    token_ids: *const i64,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    hash_routing: c_int,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4FlashRouterRefineTopKBf16AsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    router_weight: *const u16,
    correction_bias: *const f32,
    approximate_logits: *mut f32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRouterTopKBf16CubAsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    router_weight: *const u16,
    correction_bias: *const f32,
    corrected_scores: *mut f32,
    sorted_corrected_scores: *mut f32,
    unsorted_indices: *mut u32,
    sorted_indices: *mut u32,
    segment_offsets: *mut i32,
    topk_indices: *mut u32,
    topk_scores: *mut f32,
    topk_weights: *mut f32,
    cub_temp_storage: *mut c_void,
    cub_temp_storage_bytes: usize,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearF32Fn = unsafe extern "C" fn(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    output: *mut f32,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Ds41rtStatus;
type CudaLinearF32AsyncFn = unsafe extern "C" fn(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    output: *mut f32,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearBf16Fn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    bias: *const u16,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Ds41rtStatus;
type CudaLinearBf16AsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    bias: *const u16,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearBf16CublasFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    bias: *const u16,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Ds41rtStatus;
type CudaLinearBf16CublasAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    bias: *const u16,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearBf16F32CublasAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    output: *mut f32,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearBf16M1ParityBatchedCublasLtAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaQuantizeBf16W8a16Group256AsyncFn = unsafe extern "C" fn(
    source: *const u16,
    weight: *mut i8,
    scales: *mut f32,
    input_dim: usize,
    output_dim: usize,
    k_major: i32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaQuantizeBf16W8a16Group256PackedAsyncFn = unsafe extern "C" fn(
    source: *const u16,
    weight: *mut i8,
    scales: *mut f32,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearW8a16Group256M1SimtAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const i8,
    scales: *const f32,
    output: *mut u16,
    input_dim: usize,
    output_dim: usize,
    variant: i32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearW8a16Group256M1WarpPackedAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const i8,
    scales: *const f32,
    output: *mut u16,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearW8a16Group256M1WarpPackedParityBatchedAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const i8,
    scales: *const f32,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearW8a16Group256M1ParityBatchedAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const i8,
    scales: *const f32,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaPreloadW8a16Group256AotFn =
    unsafe extern "C" fn(input_dim: usize, output_dim: usize) -> Ds41rtStatus;
type CudaLinearW8a16Group256AotAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const i8,
    scales: *const f32,
    output: *mut u16,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLinearBf16StridedBatchedCublasFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    output: *mut u16,
    batch_count: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_batch_stride: usize,
    weight_batch_stride: usize,
    output_batch_stride: usize,
) -> Ds41rtStatus;
type CudaLinearBf16StridedBatchedCublasAsyncFn = unsafe extern "C" fn(
    input: *const u16,
    weight: *const u16,
    output: *mut u16,
    batch_count: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_batch_stride: usize,
    weight_batch_stride: usize,
    output_batch_stride: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMatmulBf16StridedBatchedCublasAsyncFn = CudaLinearBf16StridedBatchedCublasAsyncFn;
type CudaCausalAttentionF32Fn = unsafe extern "C" fn(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaCausalAttentionF32AsyncFn = unsafe extern "C" fn(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaCausalAttentionBf16Fn = unsafe extern "C" fn(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaCausalAttentionBf16AsyncFn = unsafe extern "C" fn(
    q: *const u16,
    k: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRopeF32Fn = unsafe extern "C" fn(
    input: *const f32,
    positions: *const u32,
    out: *mut f32,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Ds41rtStatus;
type CudaRopeF32AsyncFn = unsafe extern "C" fn(
    input: *const f32,
    positions: *const u32,
    out: *mut f32,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaRopeBf16Fn = unsafe extern "C" fn(
    input: *const u16,
    positions: *const u32,
    out: *mut u16,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Ds41rtStatus;
type CudaRopeBf16AsyncFn = unsafe extern "C" fn(
    input: *const u16,
    positions: *const u32,
    out: *mut u16,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4DsparkPromptMetadataAsyncFn = unsafe extern "C" fn(
    positions: *mut u32,
    main_slots: *mut u32,
    cos_sin_cache: *mut f32,
    absolute_position_start: usize,
    request_slot: usize,
    rows: usize,
    theta: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDs4DsparkPrepareProposalAsyncFn = unsafe extern "C" fn(
    embedding: *const u16,
    draft_token_ids: *mut u32,
    proposal_residual: *mut u16,
    positions: *mut u32,
    main_slots: *mut u32,
    cos_sin_cache: *mut f32,
    selected_indices: *mut i32,
    selected_lengths: *mut i32,
    anchor_token_id: usize,
    noise_token_id: usize,
    request_slot: usize,
    cache_window_start: usize,
    main_context_end: usize,
    vocab_size: usize,
    hidden_size: usize,
    theta: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaRopeAttentionBf16Fn = unsafe extern "C" fn(
    q_nope: *const u16,
    q_rope: *const u16,
    k_nope: *const u16,
    k_rope: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaMlaRopeAttentionBf16AsyncFn = unsafe extern "C" fn(
    q_nope: *const u16,
    q_rope: *const u16,
    k_nope: *const u16,
    k_rope: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaRopeAttentionBf16SuffixFn = unsafe extern "C" fn(
    q_nope: *const u16,
    q_rope: *const u16,
    k_nope: *const u16,
    k_rope: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Ds41rtStatus;
type CudaMlaRopeAttentionBf16SuffixAsyncFn = unsafe extern "C" fn(
    q_nope: *const u16,
    q_rope: *const u16,
    k_nope: *const u16,
    k_rope: *const u16,
    v: *const u16,
    out: *mut u16,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaMergeStateBf16AsyncFn = unsafe extern "C" fn(
    accumulator: *mut u16,
    accumulator_lse: *mut f32,
    partial: *const u16,
    partial_lse: *const f32,
    heads: usize,
    kv_lora_rank: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSparseMlaNvfp4AsyncFn = unsafe extern "C" fn(
    query: *const u16,
    kv_payload: *const u8,
    selected_indices: *const i32,
    topk_lengths: *const i32,
    partial: *mut u16,
    partial_lse: *mut f32,
    output: *mut u16,
    output_lse: *mut f32,
    query_rows: usize,
    heads: usize,
    topk: usize,
    kv_row_stride_bytes: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSparseMlaBf16AsyncFn = unsafe extern "C" fn(
    query: *const u16,
    kv_payload: *const u8,
    selected_indices: *const i32,
    topk_lengths: *const i32,
    partial: *mut u16,
    partial_lse: *mut f32,
    output: *mut u16,
    output_lse: *mut f32,
    query_rows: usize,
    heads: usize,
    topk: usize,
    kv_row_stride_bytes: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSparseMlaBf16GatherKvAsyncFn = unsafe extern "C" fn(
    kv_payload: *const u8,
    selected_indices: *const i32,
    topk_lengths: *const i32,
    gathered_k: *mut u16,
    gathered_v: *mut u16,
    query_rows: usize,
    topk: usize,
    kv_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSparseMlaBf16SoftmaxAsyncFn = unsafe extern "C" fn(
    scores: *mut u16,
    topk_lengths: *const i32,
    output_lse: *mut f32,
    query_rows: usize,
    heads: usize,
    topk: usize,
    scale: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaSparseMlaNvfp4GatherFp8AsyncFn = unsafe extern "C" fn(
    nvfp4_kv: *const u8,
    selected_indices: *const i32,
    topk_lengths: *const i32,
    fp8_kv: *mut u8,
    fp8_indices: *mut i32,
    query_rows: usize,
    selected_index_stride: usize,
    staged_topk: usize,
    nvfp4_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaNvfp4ExpandFp8PagedAsyncFn = unsafe extern "C" fn(
    nvfp4_kv: *const u8,
    physical_pages: *const u32,
    active_rows: *const i32,
    fp8_kv: *mut u8,
    max_tokens: usize,
    page_size: usize,
    nvfp4_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaEmbeddingLookupF32Fn = unsafe extern "C" fn(
    embedding: *const f32,
    token_ids: *const u32,
    out: *mut f32,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Ds41rtStatus;
type CudaEmbeddingLookupF32AsyncFn = unsafe extern "C" fn(
    embedding: *const f32,
    token_ids: *const u32,
    out: *mut f32,
    rows: usize,
    vocab: usize,
    hidden: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaEmbeddingLookupBf16Fn = unsafe extern "C" fn(
    embedding: *const u16,
    token_ids: *const u32,
    out: *mut u16,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Ds41rtStatus;
type CudaEmbeddingLookupBf16AsyncFn = unsafe extern "C" fn(
    embedding: *const u16,
    token_ids: *const u32,
    out: *mut u16,
    rows: usize,
    vocab: usize,
    hidden: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLmHeadArgmaxBf16Fn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Ds41rtStatus;
type CudaLmHeadArgmaxBf16AsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaDsparkTerminalGreedyBf16AsyncFn = unsafe extern "C" fn(
    normalized_hidden_by_slot: *const u16,
    collapsed_hidden_by_slot: *const u16,
    shared_head: *const u16,
    markov_w1: *const u16,
    markov_w2: *const u16,
    confidence_weight: *const u16,
    active_slot_ids: *const u32,
    anchor_token_ids: *const u32,
    compact_normalized_hidden: *mut u16,
    shared_logits: *mut f32,
    markov_embeddings: *mut u16,
    markov_logits: *mut f32,
    output_token_ids: *mut u32,
    conditional_confidence: *mut f32,
    active_requests: usize,
    max_slots: usize,
    hidden_dim: usize,
    vocab: usize,
    markov_rank: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLmHeadSampleTopKToppBf16Fn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaLmHeadSampleTopKToppBf16AsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLmHeadArgmaxSampleTopKToppBf16StagedFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    out_argmax_indices: *mut u32,
    out_argmax_scores: *mut f32,
    out_sample_indices: *mut u32,
    out_sample_scores: *mut f32,
    logits_workspace: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaLmHeadArgmaxSampleTopKToppBf16StagedAsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    out_argmax_indices: *mut u32,
    out_argmax_scores: *mut f32,
    out_sample_indices: *mut u32,
    out_sample_scores: *mut f32,
    logits_workspace: *mut f32,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLmHeadSampleTopKToppBf16CubFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    logits_workspace: *mut f32,
    sorted_logits: *mut f32,
    unsorted_indices: *mut u32,
    sorted_indices: *mut u32,
    segment_offsets: *mut i32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    cub_temp_storage: *mut c_void,
    cub_temp_storage_bytes: usize,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaLmHeadSampleTopKToppBf16CubAsyncFn = unsafe extern "C" fn(
    hidden: *const u16,
    lm_head: *const u16,
    random_uniforms: *const f32,
    logits_workspace: *mut f32,
    sorted_logits: *mut f32,
    unsorted_indices: *mut u32,
    sorted_indices: *mut u32,
    segment_offsets: *mut i32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    cub_temp_storage: *mut c_void,
    cub_temp_storage_bytes: usize,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLogitsArgmaxF32Fn = unsafe extern "C" fn(
    logits: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    vocab: usize,
) -> Ds41rtStatus;
type CudaLogitsArgmaxF32AsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    vocab: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLogitsSampleTopKToppF32Fn = unsafe extern "C" fn(
    logits: *const f32,
    random_uniforms: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaLogitsSampleTopKToppF32AsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    random_uniforms: *const f32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaLogitsSampleTopKToppF32CubFn = unsafe extern "C" fn(
    logits: *const f32,
    random_uniforms: *const f32,
    sorted_logits: *mut f32,
    unsorted_indices: *mut u32,
    sorted_indices: *mut u32,
    segment_offsets: *mut i32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    cub_temp_storage: *mut c_void,
    cub_temp_storage_bytes: usize,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Ds41rtStatus;
type CudaLogitsSampleTopKToppF32CubAsyncFn = unsafe extern "C" fn(
    logits: *const f32,
    random_uniforms: *const f32,
    sorted_logits: *mut f32,
    unsorted_indices: *mut u32,
    sorted_indices: *mut u32,
    segment_offsets: *mut i32,
    out_indices: *mut u32,
    out_scores: *mut f32,
    cub_temp_storage: *mut c_void,
    cub_temp_storage_bytes: usize,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaKvCacheWriteBytesFn = unsafe extern "C" fn(
    src: *const u8,
    cache: *mut u8,
    cache_offset_bytes: usize,
    bytes: usize,
) -> Ds41rtStatus;
type CudaKvCacheWriteBytesAsyncFn = unsafe extern "C" fn(
    src: *const u8,
    cache: *mut u8,
    cache_offset_bytes: usize,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaKvCacheReadBytesFn = unsafe extern "C" fn(
    cache: *const u8,
    dst: *mut u8,
    cache_offset_bytes: usize,
    bytes: usize,
) -> Ds41rtStatus;
type CudaKvCacheReadBytesAsyncFn = unsafe extern "C" fn(
    cache: *const u8,
    dst: *mut u8,
    cache_offset_bytes: usize,
    bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaKvCacheWriteBlocksFn = unsafe extern "C" fn(
    src: *const u8,
    cache: *mut u8,
    src_offsets: *const u64,
    cache_offsets: *const u64,
    block_bytes: *const u64,
    block_count: usize,
) -> Ds41rtStatus;
type CudaKvCacheWriteBlocksAsyncFn = unsafe extern "C" fn(
    src: *const u8,
    cache: *mut u8,
    src_offsets: *const u64,
    cache_offsets: *const u64,
    block_bytes: *const u64,
    block_count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaKvCacheReadBlocksFn = unsafe extern "C" fn(
    cache: *const u8,
    dst: *mut u8,
    cache_offsets: *const u64,
    dst_offsets: *const u64,
    block_bytes: *const u64,
    block_count: usize,
) -> Ds41rtStatus;
type CudaKvCacheReadBlocksAsyncFn = unsafe extern "C" fn(
    cache: *const u8,
    dst: *mut u8,
    cache_offsets: *const u64,
    dst_offsets: *const u64,
    block_bytes: *const u64,
    block_count: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvCacheUnpackBf16Fn = unsafe extern "C" fn(
    payload: *const u8,
    kv_latent: *mut u16,
    k_rope: *mut u16,
    dsa_key: *mut u16,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaMlaKvCacheUnpackBf16AsyncFn = unsafe extern "C" fn(
    payload: *const u8,
    kv_latent: *mut u16,
    k_rope: *mut u16,
    dsa_key: *mut u16,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvProjectedSplitBf16Fn = unsafe extern "C" fn(
    projected: *const u16,
    k_nope: *mut u16,
    v: *mut u16,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
) -> Ds41rtStatus;
type CudaMlaKvProjectedSplitBf16AsyncFn = unsafe extern "C" fn(
    projected: *const u16,
    k_nope: *mut u16,
    v: *mut u16,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvPrepareBf16Fn = unsafe extern "C" fn(
    projected: *const u16,
    positions: *const u32,
    norm_weight: *const u16,
    prepared: *mut u16,
    rows: usize,
    projected_stride_bytes: usize,
    prepared_stride_bytes: usize,
    eps: f32,
    theta: f32,
) -> Ds41rtStatus;
type CudaMlaKvPrepareBf16AsyncFn = unsafe extern "C" fn(
    projected: *const u16,
    positions: *const u32,
    norm_weight: *const u16,
    prepared: *mut u16,
    rows: usize,
    projected_stride_bytes: usize,
    prepared_stride_bytes: usize,
    eps: f32,
    theta: f32,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaTransposeRowsHeadsBf16Fn = unsafe extern "C" fn(
    input: *const u16,
    output: *mut u16,
    rows: usize,
    heads: usize,
    width: usize,
) -> Ds41rtStatus;
type CudaTransposeRowsHeadsBf16AsyncFn = unsafe extern "C" fn(
    input: *const u16,
    output: *mut u16,
    rows: usize,
    heads: usize,
    width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaComposeAbsorbedQueryBf16Fn = unsafe extern "C" fn(
    latent_heads_rows: *const u16,
    rope_rows_heads: *const u16,
    output_rows_heads: *mut u16,
    rows: usize,
    heads: usize,
    latent_width: usize,
    rope_width: usize,
) -> Ds41rtStatus;
type CudaMlaComposeAbsorbedQueryBf16AsyncFn = unsafe extern "C" fn(
    latent_heads_rows: *const u16,
    rope_rows_heads: *const u16,
    output_rows_heads: *mut u16,
    rows: usize,
    heads: usize,
    latent_width: usize,
    rope_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGenericKvPageTableInitFn = unsafe extern "C" fn(
    page_table: *mut i32,
    query_rows: usize,
    page_table_width: usize,
) -> Ds41rtStatus;
type CudaGenericKvPageTableInitAsyncFn = unsafe extern "C" fn(
    page_table: *mut i32,
    query_rows: usize,
    page_table_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGenericKvPageTableInitBaseFn = unsafe extern "C" fn(
    page_table: *mut i32,
    query_rows: usize,
    page_table_width: usize,
    base_offset: usize,
) -> Ds41rtStatus;
type CudaGenericKvPageTableInitBaseAsyncFn = unsafe extern "C" fn(
    page_table: *mut i32,
    query_rows: usize,
    page_table_width: usize,
    base_offset: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGenericKvPageTableInitOffsetsAsyncFn = unsafe extern "C" fn(
    page_table: *mut i32,
    row_offsets: *const i32,
    query_rows: usize,
    page_table_width: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaGenericKvPageTableExpandIndicesFn = unsafe extern "C" fn(
    output_indices: *mut i32,
    physical_pages: *const u32,
    query_rows: usize,
    output_width: usize,
    active_tokens: usize,
) -> Ds41rtStatus;
type CudaGenericKvPageTableExpandIndicesAsyncFn = unsafe extern "C" fn(
    output_indices: *mut i32,
    physical_pages: *const u32,
    query_rows: usize,
    output_width: usize,
    active_tokens: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvPackFp8DsMlaFn = unsafe extern "C" fn(
    projected: *const u16,
    packed: *mut u8,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaMlaKvPackFp8DsMlaAsyncFn = unsafe extern "C" fn(
    projected: *const u16,
    packed: *mut u8,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvUnpackFp8DsMlaFn = unsafe extern "C" fn(
    packed: *const u8,
    projected: *mut u16,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaMlaKvUnpackFp8DsMlaAsyncFn = unsafe extern "C" fn(
    packed: *const u8,
    projected: *mut u16,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvPackMxfp4DsMlaFn = unsafe extern "C" fn(
    projected: *const u16,
    packed: *mut u8,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaMlaKvPackMxfp4DsMlaAsyncFn = unsafe extern "C" fn(
    projected: *const u16,
    packed: *mut u8,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaMlaKvUnpackMxfp4DsMlaFn = unsafe extern "C" fn(
    packed: *const u8,
    projected: *mut u16,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
) -> Ds41rtStatus;
type CudaMlaKvUnpackMxfp4DsMlaAsyncFn = unsafe extern "C" fn(
    packed: *const u8,
    projected: *mut u16,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Ds41rtStatus;
type CudaPackNibblesFn =
    unsafe extern "C" fn(codes: *const u8, packed: *mut u8, count: usize) -> Ds41rtStatus;
type CudaUnpackNibblesFn =
    unsafe extern "C" fn(packed: *const u8, codes: *mut u8, count: usize) -> Ds41rtStatus;
type RdmaDeviceInfoFn = unsafe extern "C" fn(out: *mut Ds41rtRdmaDeviceInfo) -> Ds41rtStatus;
type RdmaPlanHostBufferRegistrationFn = unsafe extern "C" fn(
    ptr: *const c_void,
    bytes: usize,
    alignment: usize,
    out: *mut Ds41rtRdmaHostBufferPlan,
) -> Ds41rtStatus;
type RdmaRegisterHostBufferProbeFn = unsafe extern "C" fn(
    ptr: *mut c_void,
    bytes: usize,
    out: *mut Ds41rtRdmaRegisterProbe,
) -> Ds41rtStatus;
type RdmaCreateRcQpProbeFn = unsafe extern "C" fn(
    port_num: u32,
    send_wr: u32,
    recv_wr: u32,
    max_sge: u32,
    out: *mut Ds41rtRdmaRcQpProbe,
) -> Ds41rtStatus;
type RdmaRcSendRecvLoopbackProbeFn = unsafe extern "C" fn(
    port_num: u32,
    bytes: usize,
    out: *mut Ds41rtRdmaRcSendRecvProbe,
) -> Ds41rtStatus;
type RdmaRcProtocolV2LoopbackProbeFn = unsafe extern "C" fn(
    port_num: u32,
    request_frame: *const c_void,
    request_bytes: usize,
    response_frame: *const c_void,
    response_bytes: usize,
    out: *mut Ds41rtRdmaRcProtocolV2LoopbackProbe,
) -> Ds41rtStatus;
type RdmaRcEndpointCreateFn = unsafe extern "C" fn(
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    out: *mut Ds41rtRdmaRcEndpointInfo,
) -> Ds41rtStatus;
type RdmaRcEndpointCreateWithBufferFlagsFn = unsafe extern "C" fn(
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    host_buffer_flags: u64,
    out: *mut Ds41rtRdmaRcEndpointInfo,
) -> Ds41rtStatus;
type RdmaRcEndpointCreateOnDeviceWithBufferFlagsFn = unsafe extern "C" fn(
    device_name: *const c_char,
    port_num: u32,
    local_psn: u32,
    send_frame_bytes: usize,
    recv_frame_bytes: usize,
    send_registered_span_bytes: usize,
    recv_registered_span_bytes: usize,
    max_send_wr: u32,
    max_recv_wr: u32,
    max_sge: u32,
    host_buffer_flags: u64,
    out: *mut Ds41rtRdmaRcEndpointInfo,
) -> Ds41rtStatus;
type RdmaRcEndpointBufferViewFn = unsafe extern "C" fn(
    handle: *mut c_void,
    receive_buffer: c_int,
    out: *mut Ds41rtRdmaRcEndpointBufferView,
) -> Ds41rtStatus;
type RdmaRcEndpointConnectFn = unsafe extern "C" fn(
    handle: *mut c_void,
    remote_qp_num: u32,
    remote_psn: u32,
    remote_lid: u32,
    remote_gid_hex: *const c_char,
) -> Ds41rtStatus;
type RdmaRcEndpointPostRecvFn =
    unsafe extern "C" fn(handle: *mut c_void, bytes: usize, wr_id: u64) -> Ds41rtStatus;
type RdmaRcEndpointPostRecvAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> Ds41rtStatus;
type RdmaRcEndpointPostSendAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> Ds41rtStatus;
type RdmaRcEndpointSendFn = unsafe extern "C" fn(
    handle: *mut c_void,
    frame: *const c_void,
    bytes: usize,
    wr_id: u64,
) -> Ds41rtStatus;
type RdmaRcEndpointSendAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    frame: *const c_void,
    offset_bytes: usize,
    bytes: usize,
    wr_id: u64,
) -> Ds41rtStatus;
type RdmaRcEndpointSendPartsAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    prefix: *const c_void,
    prefix_bytes: usize,
    payload: *const c_void,
    payload_bytes: usize,
    offset_bytes: usize,
    wr_id: u64,
) -> Ds41rtStatus;
type RdmaRcEndpointPollFn = unsafe extern "C" fn(
    handle: *mut c_void,
    expected_send_completions: u32,
    expected_recv_completions: u32,
    max_poll_iterations: u32,
    out: *mut Ds41rtRdmaRcCompletionStats,
) -> Ds41rtStatus;
type RdmaRcEndpointPollWithTimeoutFn = unsafe extern "C" fn(
    handle: *mut c_void,
    expected_send_completions: u32,
    expected_recv_completions: u32,
    max_poll_iterations: u32,
    active_event_poll_timeout_ms: u32,
    out: *mut Ds41rtRdmaRcCompletionStats,
) -> Ds41rtStatus;
type RdmaRcEndpointTryPollFn = unsafe extern "C" fn(
    handle: *mut c_void,
    max_send_completions: u32,
    max_recv_completions: u32,
    out: *mut Ds41rtRdmaRcCompletionStats,
) -> Ds41rtStatus;
type RdmaRcEndpointCopyRecvFn = unsafe extern "C" fn(
    handle: *mut c_void,
    out: *mut c_void,
    out_bytes: usize,
    bytes: usize,
) -> Ds41rtStatus;
type RdmaRcEndpointCopyRecvAtFn = unsafe extern "C" fn(
    handle: *mut c_void,
    out: *mut c_void,
    out_bytes: usize,
    offset_bytes: usize,
    bytes: usize,
) -> Ds41rtStatus;
type RdmaRcEndpointDestroyFn = unsafe extern "C" fn(handle: *mut c_void) -> Ds41rtStatus;

type XGrammarCompilerCreateFn = unsafe extern "C" fn(
    tokenizer_json_path: *const c_char,
    vocab_size: usize,
    stop_token_ids: *const i32,
    stop_token_count: usize,
    out_compiler: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarCompilerDestroyFn = unsafe extern "C" fn(compiler: *mut c_void) -> Ds41rtStatus;
type XGrammarCompileFn = unsafe extern "C" fn(
    compiler: *mut c_void,
    kind: c_int,
    grammar_json: *const c_char,
    strict: c_int,
    out_grammar: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarGrammarDestroyFn = unsafe extern "C" fn(grammar: *mut c_void) -> Ds41rtStatus;
type XGrammarMatcherCreateFn = unsafe extern "C" fn(
    grammar: *const c_void,
    out_matcher: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarMatcherForkFn = unsafe extern "C" fn(
    matcher: *const c_void,
    out_matcher: *mut *mut c_void,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarMatcherDestroyFn = unsafe extern "C" fn(matcher: *mut c_void) -> Ds41rtStatus;
type XGrammarMatcherFillBitmaskFn = unsafe extern "C" fn(
    matcher: *mut c_void,
    bitmask: *mut u32,
    bitmask_words: usize,
    out_needs_mask: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarMatcherAcceptTokenFn = unsafe extern "C" fn(
    matcher: *mut c_void,
    token_id: u32,
    out_accepted: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;
type XGrammarMatcherIsCompletedFn = unsafe extern "C" fn(
    matcher: *const c_void,
    out_completed: *mut c_int,
    error: *mut c_char,
    error_bytes: usize,
) -> Ds41rtStatus;

pub struct NativeLibrary {
    lib: Library,
    sync_h2d_staging: Mutex<SyncH2DStagingBuffer>,
    rdma_rc_endpoint_try_poll_fn: RdmaRcEndpointTryPollFn,
}

pub const DS41RT_XGRAMMAR_JSON_OBJECT: c_int = 1;
pub const DS41RT_XGRAMMAR_JSON_SCHEMA: c_int = 2;
pub const DS41RT_XGRAMMAR_STRUCTURAL_TAG: c_int = 3;

pub struct Ds41rtXGrammarCompiler<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

pub struct Ds41rtXGrammarGrammar<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

pub struct Ds41rtXGrammarMatcher<'a> {
    handle: *mut c_void,
    library: &'a NativeLibrary,
}

unsafe impl Send for Ds41rtXGrammarCompiler<'_> {}
unsafe impl Sync for Ds41rtXGrammarCompiler<'_> {}
unsafe impl Send for Ds41rtXGrammarGrammar<'_> {}
unsafe impl Sync for Ds41rtXGrammarGrammar<'_> {}
unsafe impl Send for Ds41rtXGrammarMatcher<'_> {}

const XGRAMMAR_ERROR_BYTES: usize = 2_048;

fn xgrammar_status(
    context: &str,
    status: Ds41rtStatus,
    error: &[c_char; XGRAMMAR_ERROR_BYTES],
) -> Result<()> {
    if status == DS41RT_STATUS_OK {
        return Ok(());
    }
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    if detail.is_empty() {
        anyhow::bail!("{context} returned status {status}");
    }
    anyhow::bail!("{context} returned status {status}: {detail}")
}

impl NativeLibrary {
    pub fn xgrammar_compiler<'a>(
        &'a self,
        tokenizer_json_path: &Path,
        vocab_size: usize,
        stop_token_ids: &[i32],
    ) -> Result<Ds41rtXGrammarCompiler<'a>> {
        let path = CString::new(tokenizer_json_path.to_string_lossy().as_bytes())
            .context("XGrammar tokenizer path contains a NUL byte")?;
        let create: Symbol<XGrammarCompilerCreateFn> =
            unsafe { self.lib.get(b"ds41rt_xgrammar_compiler_create")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            create(
                path.as_ptr(),
                vocab_size,
                if stop_token_ids.is_empty() {
                    std::ptr::null()
                } else {
                    stop_token_ids.as_ptr()
                },
                stop_token_ids.len(),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("ds41rt_xgrammar_compiler_create", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar compiler handle is null");
        Ok(Ds41rtXGrammarCompiler {
            handle,
            library: self,
        })
    }
}

impl<'a> Ds41rtXGrammarCompiler<'a> {
    pub fn compile(
        &self,
        kind: c_int,
        grammar_json: Option<&str>,
        strict: bool,
    ) -> Result<Ds41rtXGrammarGrammar<'a>> {
        let grammar_json = grammar_json
            .map(|value| {
                CString::new(value).context("XGrammar source contains an embedded NUL byte")
            })
            .transpose()?;
        let compile: Symbol<XGrammarCompileFn> =
            unsafe { self.library.lib.get(b"ds41rt_xgrammar_compile")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            compile(
                self.handle,
                kind,
                grammar_json
                    .as_ref()
                    .map_or(std::ptr::null(), |value| value.as_ptr()),
                c_int::from(strict),
                &mut handle,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("ds41rt_xgrammar_compile", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar grammar handle is null");
        Ok(Ds41rtXGrammarGrammar {
            handle,
            library: self.library,
        })
    }
}

impl<'a> Ds41rtXGrammarGrammar<'a> {
    pub fn matcher(&self) -> Result<Ds41rtXGrammarMatcher<'a>> {
        let create: Symbol<XGrammarMatcherCreateFn> =
            unsafe { self.library.lib.get(b"ds41rt_xgrammar_matcher_create")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe { create(self.handle, &mut handle, error.as_mut_ptr(), error.len()) };
        xgrammar_status("ds41rt_xgrammar_matcher_create", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "native XGrammar matcher handle is null");
        Ok(Ds41rtXGrammarMatcher {
            handle,
            library: self.library,
        })
    }
}

impl<'a> Ds41rtXGrammarMatcher<'a> {
    pub fn fork(&self) -> Result<Self> {
        let fork: Symbol<XGrammarMatcherForkFn> =
            unsafe { self.library.lib.get(b"ds41rt_xgrammar_matcher_fork")? };
        let mut handle = std::ptr::null_mut();
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe { fork(self.handle, &mut handle, error.as_mut_ptr(), error.len()) };
        xgrammar_status("ds41rt_xgrammar_matcher_fork", status, &error)?;
        anyhow::ensure!(!handle.is_null(), "forked native XGrammar matcher is null");
        Ok(Self {
            handle,
            library: self.library,
        })
    }

    pub fn fill_bitmask(&mut self, bitmask: &mut [u32]) -> Result<bool> {
        anyhow::ensure!(!bitmask.is_empty(), "XGrammar bitmask is empty");
        let fill: Symbol<XGrammarMatcherFillBitmaskFn> = unsafe {
            self.library
                .lib
                .get(b"ds41rt_xgrammar_matcher_fill_bitmask")?
        };
        let mut needs_mask = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            fill(
                self.handle,
                bitmask.as_mut_ptr(),
                bitmask.len(),
                &mut needs_mask,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("ds41rt_xgrammar_matcher_fill_bitmask", status, &error)?;
        Ok(needs_mask != 0)
    }

    pub fn accept_token(&mut self, token_id: u32) -> Result<bool> {
        let accept: Symbol<XGrammarMatcherAcceptTokenFn> = unsafe {
            self.library
                .lib
                .get(b"ds41rt_xgrammar_matcher_accept_token")?
        };
        let mut accepted = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            accept(
                self.handle,
                token_id,
                &mut accepted,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("ds41rt_xgrammar_matcher_accept_token", status, &error)?;
        Ok(accepted != 0)
    }

    pub fn is_completed(&self) -> Result<bool> {
        let completed: Symbol<XGrammarMatcherIsCompletedFn> = unsafe {
            self.library
                .lib
                .get(b"ds41rt_xgrammar_matcher_is_completed")?
        };
        let mut is_completed = 0;
        let mut error = [0 as c_char; XGRAMMAR_ERROR_BYTES];
        let status = unsafe {
            completed(
                self.handle,
                &mut is_completed,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        xgrammar_status("ds41rt_xgrammar_matcher_is_completed", status, &error)?;
        Ok(is_completed != 0)
    }
}

impl Drop for Ds41rtXGrammarCompiler<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarCompilerDestroyFn>(b"ds41rt_xgrammar_compiler_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for Ds41rtXGrammarGrammar<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarGrammarDestroyFn>(b"ds41rt_xgrammar_grammar_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

impl Drop for Ds41rtXGrammarMatcher<'_> {
    fn drop(&mut self) {
        if let Ok(destroy) = unsafe {
            self.library
                .lib
                .get::<XGrammarMatcherDestroyFn>(b"ds41rt_xgrammar_matcher_destroy")
        } {
            let _ = unsafe { destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

pub struct Ds41rtNcclComm {
    handle: *mut c_void,
    library: Arc<NativeLibrary>,
    world_size: usize,
    rank: usize,
}

unsafe impl Send for Ds41rtNcclComm {}

fn row_partition(rows: usize, world_size: usize, rank: usize) -> Result<(usize, usize)> {
    anyhow::ensure!(
        world_size > 0 && rank < world_size,
        "invalid row partition rank"
    );
    let base_rows = rows / world_size;
    let extra_rows = rows % world_size;
    let row_count = base_rows + usize::from(rank < extra_rows);
    let row_start = rank
        .checked_mul(base_rows)
        .and_then(|start| start.checked_add(rank.min(extra_rows)))
        .context("row partition start overflow")?;
    Ok((row_start, row_count))
}

impl Ds41rtNcclComm {
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub unsafe fn gather_u8_async(
        &self,
        send: Ds41rtDeviceBuffer,
        recv: Ds41rtDeviceBuffer,
        bytes: usize,
        root: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(bytes > 0, "NCCL gather byte count must be positive");
        anyhow::ensure!(
            root < self.world_size,
            "NCCL gather root {root} exceeds world size {}",
            self.world_size
        );
        anyhow::ensure!(
            !send.ptr.is_null() && send.bytes >= bytes,
            "NCCL gather send buffer has {} bytes for {bytes}",
            send.bytes
        );
        if self.rank == root {
            let required = bytes
                .checked_mul(self.world_size - 1)
                .context("NCCL gather receive byte count overflow")?;
            anyhow::ensure!(
                !recv.ptr.is_null() && recv.bytes >= required,
                "NCCL gather receive buffer has {} bytes for {required}",
                recv.bytes
            );
        }
        let gather_fn: Symbol<NcclGatherU8AsyncFn> =
            unsafe { self.library.lib.get(b"ds41rt_nccl_gather_u8_async")? };
        let status =
            unsafe { gather_fn(self.handle, send, recv, bytes, root as c_int, cuda_stream) };
        self.library
            .status_to_result("ds41rt_nccl_gather_u8_async", status)
    }

    pub unsafe fn row_all_to_all_u8_async(
        &self,
        send: Ds41rtDeviceBuffer,
        recv: Ds41rtDeviceBuffer,
        rows: usize,
        row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            rows >= self.world_size && row_stride_bytes > 0,
            "NCCL row all-to-all requires at least one row per rank"
        );
        let send_bytes = rows
            .checked_mul(row_stride_bytes)
            .context("NCCL row all-to-all send byte count overflow")?;
        let local_rows = row_partition(rows, self.world_size, self.rank)?.1;
        let recv_bytes = local_rows
            .checked_mul(row_stride_bytes)
            .and_then(|bytes| bytes.checked_mul(self.world_size - 1))
            .context("NCCL row all-to-all receive byte count overflow")?;
        anyhow::ensure!(
            !send.ptr.is_null() && send.bytes >= send_bytes,
            "NCCL row all-to-all send buffer has {} bytes for {send_bytes}",
            send.bytes
        );
        anyhow::ensure!(
            !recv.ptr.is_null() && recv.bytes >= recv_bytes,
            "NCCL row all-to-all receive buffer has {} bytes for {recv_bytes}",
            recv.bytes
        );
        let all_to_all_fn: Symbol<NcclRowAllToAllU8AsyncFn> = unsafe {
            self.library
                .lib
                .get(b"ds41rt_nccl_row_all_to_all_u8_async")?
        };
        let status =
            unsafe { all_to_all_fn(self.handle, send, recv, rows, row_stride_bytes, cuda_stream) };
        self.library
            .status_to_result("ds41rt_nccl_row_all_to_all_u8_async", status)
    }

    pub unsafe fn all_reduce_bf16_async(
        &self,
        send: Ds41rtDeviceBuffer,
        recv: Ds41rtDeviceBuffer,
        values: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let bytes = values
            .checked_mul(std::mem::size_of::<u16>())
            .context("NCCL BF16 all-reduce byte count overflow")?;
        anyhow::ensure!(
            values > 0,
            "NCCL BF16 all-reduce value count must be positive"
        );
        anyhow::ensure!(
            !send.ptr.is_null() && send.bytes >= bytes,
            "NCCL BF16 all-reduce send buffer has {} bytes for {bytes}",
            send.bytes
        );
        anyhow::ensure!(
            !recv.ptr.is_null() && recv.bytes >= bytes,
            "NCCL BF16 all-reduce receive buffer has {} bytes for {bytes}",
            recv.bytes
        );
        let all_reduce_fn: Symbol<NcclAllReduceBf16AsyncFn> =
            unsafe { self.library.lib.get(b"ds41rt_nccl_all_reduce_bf16_async")? };
        let status = unsafe { all_reduce_fn(self.handle, send, recv, values, cuda_stream) };
        self.library
            .status_to_result("ds41rt_nccl_all_reduce_bf16_async", status)
    }

    pub unsafe fn reduce_bf16_async(
        &self,
        send: Ds41rtDeviceBuffer,
        recv: Ds41rtDeviceBuffer,
        values: usize,
        root: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let bytes = values
            .checked_mul(std::mem::size_of::<u16>())
            .context("NCCL BF16 reduce byte count overflow")?;
        anyhow::ensure!(values > 0, "NCCL BF16 reduce value count must be positive");
        anyhow::ensure!(
            root < self.world_size,
            "NCCL BF16 reduce root {root} exceeds world size {}",
            self.world_size
        );
        anyhow::ensure!(
            !send.ptr.is_null() && send.bytes >= bytes,
            "NCCL BF16 reduce send buffer has {} bytes for {bytes}",
            send.bytes
        );
        if self.rank == root {
            anyhow::ensure!(
                !recv.ptr.is_null() && recv.bytes >= bytes,
                "NCCL BF16 reduce receive buffer has {} bytes for {bytes}",
                recv.bytes
            );
        }
        let reduce_fn: Symbol<NcclReduceBf16AsyncFn> =
            unsafe { self.library.lib.get(b"ds41rt_nccl_reduce_bf16_async")? };
        let status =
            unsafe { reduce_fn(self.handle, send, recv, values, root as c_int, cuda_stream) };
        self.library
            .status_to_result("ds41rt_nccl_reduce_bf16_async", status)
    }
}

impl Drop for Ds41rtNcclComm {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(destroy_fn) = unsafe {
            self.library
                .lib
                .get::<NcclCommDestroyFn>(b"ds41rt_nccl_comm_destroy")
        } {
            let _ = unsafe { destroy_fn(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

struct SyncH2DStagingBuffer {
    buffer: Ds41rtHostBuffer,
}

impl Default for SyncH2DStagingBuffer {
    fn default() -> Self {
        Self {
            buffer: Ds41rtHostBuffer::default(),
        }
    }
}

unsafe impl Send for SyncH2DStagingBuffer {}

impl SyncH2DStagingBuffer {
    fn ensure(&mut self, library: &NativeLibrary, bytes: usize) -> Result<Ds41rtHostBuffer> {
        if self.buffer.ptr.is_null() || self.buffer.bytes < bytes {
            if !self.buffer.ptr.is_null() {
                library
                    .free_host_buffer(&mut self.buffer)
                    .context("freeing undersized synchronous H2D pinned staging buffer")?;
                self.buffer = Ds41rtHostBuffer::default();
            }
            self.buffer = library
                .alloc_host_buffer(bytes)
                .context("allocating reusable synchronous H2D pinned staging buffer")?;
            if self.buffer.ptr.is_null() {
                anyhow::bail!("reusable synchronous H2D pinned staging buffer is null");
            }
            if self.buffer.bytes < bytes {
                let allocated_bytes = self.buffer.bytes;
                library
                    .free_host_buffer(&mut self.buffer)
                    .context("freeing undersized reusable synchronous H2D pinned staging buffer")?;
                self.buffer = Ds41rtHostBuffer::default();
                anyhow::bail!(
                    "reusable synchronous H2D pinned staging buffer bytes {} is smaller than source bytes {bytes}",
                    allocated_bytes
                );
            }
        }
        Ok(self.buffer)
    }

    fn release_with_library(&mut self, lib: &Library) {
        if self.buffer.ptr.is_null() {
            return;
        }
        if let Ok(free_fn) = unsafe { lib.get::<FreeHostBufferFn>(b"ds41rt_free_host_buffer") } {
            let _ = unsafe { free_fn(&mut self.buffer) };
        }
        self.buffer = Ds41rtHostBuffer::default();
    }
}

impl Drop for NativeLibrary {
    fn drop(&mut self) {
        if let Ok(staging) = self.sync_h2d_staging.get_mut() {
            staging.release_with_library(&self.lib);
        }
    }
}

impl NativeLibrary {
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self> {
        let lib = unsafe { Library::new(path.as_ref()) }
            .with_context(|| format!("loading native library {}", path.as_ref().display()))?;
        let rdma_rc_endpoint_try_poll_fn =
            unsafe { *lib.get::<RdmaRcEndpointTryPollFn>(b"ds41rt_rdma_rc_endpoint_try_poll")? };
        Ok(Self {
            lib,
            sync_h2d_staging: Mutex::new(SyncH2DStagingBuffer::default()),
            rdma_rc_endpoint_try_poll_fn,
        })
    }

    pub fn version(&self) -> Result<String> {
        let version_fn: Symbol<VersionFn> = unsafe { self.lib.get(b"ds41rt_native_version")? };
        let mut buf = vec![0 as c_char; 128];
        let status = unsafe { version_fn(buf.as_mut_ptr(), buf.len()) };
        self.status_to_result("ds41rt_native_version", status)?;
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    pub fn cuda_device_info(&self, device_id: i32) -> Result<Ds41rtCudaDeviceInfo> {
        let info_fn: Symbol<DeviceInfoFn> = unsafe { self.lib.get(b"ds41rt_cuda_device_info")? };
        let mut info = Ds41rtCudaDeviceInfo::default();
        let status = unsafe { info_fn(device_id, &mut info) };
        self.status_to_result("ds41rt_cuda_device_info", status)?;
        Ok(info)
    }

    pub fn nccl_unique_id_bytes(&self) -> Result<usize> {
        let bytes_fn: Symbol<NcclUniqueIdBytesFn> =
            unsafe { self.lib.get(b"ds41rt_nccl_unique_id_bytes")? };
        let mut bytes = 0_usize;
        let status = unsafe { bytes_fn(&mut bytes) };
        self.status_to_result("ds41rt_nccl_unique_id_bytes", status)?;
        anyhow::ensure!(bytes > 0, "native NCCL unique ID has zero bytes");
        Ok(bytes)
    }

    pub fn nccl_get_unique_id(&self) -> Result<Vec<u8>> {
        let bytes = self.nccl_unique_id_bytes()?;
        let mut unique_id = vec![0_u8; bytes];
        let get_fn: Symbol<NcclGetUniqueIdFn> =
            unsafe { self.lib.get(b"ds41rt_nccl_get_unique_id")? };
        let status = unsafe { get_fn(unique_id.as_mut_ptr().cast(), unique_id.len()) };
        self.status_to_result("ds41rt_nccl_get_unique_id", status)?;
        Ok(unique_id)
    }

    pub fn nccl_comm_init_rank(
        self: &Arc<Self>,
        unique_id: &[u8],
        world_size: usize,
        rank: usize,
    ) -> Result<Ds41rtNcclComm> {
        anyhow::ensure!(world_size > 1, "NCCL world size must exceed one");
        anyhow::ensure!(
            rank < world_size,
            "NCCL rank {rank} exceeds world size {world_size}"
        );
        anyhow::ensure!(
            unique_id.len() == self.nccl_unique_id_bytes()?,
            "NCCL unique ID has an unexpected byte size"
        );
        let world_size = c_int::try_from(world_size).context("NCCL world size exceeds c_int")?;
        let rank_c = c_int::try_from(rank).context("NCCL rank exceeds c_int")?;
        let init_fn: Symbol<NcclCommInitRankFn> =
            unsafe { self.lib.get(b"ds41rt_nccl_comm_init_rank")? };
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            init_fn(
                unique_id.as_ptr().cast(),
                unique_id.len(),
                world_size,
                rank_c,
                &mut handle,
            )
        };
        self.status_to_result("ds41rt_nccl_comm_init_rank", status)?;
        anyhow::ensure!(!handle.is_null(), "native NCCL communicator handle is null");
        Ok(Ds41rtNcclComm {
            handle,
            library: Arc::clone(self),
            world_size: world_size as usize,
            rank,
        })
    }

    pub fn alloc_host_buffer(&self, bytes: usize) -> Result<Ds41rtHostBuffer> {
        let alloc_fn: Symbol<AllocHostBufferFn> =
            unsafe { self.lib.get(b"ds41rt_alloc_host_buffer")? };
        let mut buffer = Ds41rtHostBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("ds41rt_alloc_host_buffer", status)?;
        Ok(buffer)
    }

    pub fn cuda_host_buffer_device_alias(
        &self,
        host: Ds41rtHostBuffer,
    ) -> Result<Ds41rtDeviceBuffer> {
        if host.ptr.is_null() || host.bytes == 0 {
            anyhow::bail!("mapped host buffer is empty");
        }
        let alias_fn: Symbol<CudaHostBufferDeviceAliasFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_host_buffer_device_alias")? };
        let mut alias = Ds41rtDeviceBuffer::default();
        let status = unsafe { alias_fn(host, &mut alias) };
        self.status_to_result("ds41rt_cuda_host_buffer_device_alias", status)?;
        Ok(alias)
    }

    pub fn free_host_buffer(&self, buffer: &mut Ds41rtHostBuffer) -> Result<()> {
        let free_fn: Symbol<FreeHostBufferFn> = unsafe { self.lib.get(b"ds41rt_free_host_buffer")? };
        let status = unsafe { free_fn(buffer) };
        self.status_to_result("ds41rt_free_host_buffer", status)
    }

    /// Free and total bytes on the current CUDA device, including other processes.
    pub fn cuda_memory_info(&self) -> Result<(usize, usize)> {
        let info: Symbol<unsafe extern "C" fn(*mut usize, *mut usize) -> Ds41rtStatus> =
            unsafe { self.lib.get(b"ds41rt_cuda_memory_info")? };
        let (mut free, mut total) = (0, 0);
        self.status_to_result("ds41rt_cuda_memory_info", unsafe { info(&mut free, &mut total) })?;
        anyhow::ensure!(free <= total && total > 0, "invalid CUDA memory information");
        Ok((free, total))
    }

    pub fn cuda_get_device(&self) -> Result<i32> {
        let call: Symbol<unsafe extern "C" fn(*mut i32) -> Ds41rtStatus> =
            unsafe { self.lib.get(b"ds41rt_cuda_get_device")? };
        let mut device = -1;
        self.status_to_result("ds41rt_cuda_get_device", unsafe { call(&mut device) })?;
        Ok(device)
    }

    /// Host-thread-local selection. Restore the previous device before yielding.
    pub fn cuda_set_device(&self, device: i32) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(i32) -> Ds41rtStatus> =
            unsafe { self.lib.get(b"ds41rt_cuda_set_device")? };
        self.status_to_result("ds41rt_cuda_set_device", unsafe { call(device) })
    }

    pub fn cuda_enable_peer(&self, peer: i32) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(i32) -> Ds41rtStatus> =
            unsafe { self.lib.get(b"ds41rt_cuda_enable_peer")? };
        self.status_to_result("ds41rt_cuda_enable_peer", unsafe { call(peer) })
    }

    /// Enqueue a cross-device transfer without joining other streams.
    ///
    /// # Safety
    /// `stream` must be live on the current, destination device. Source writes
    /// must precede the copy (use a stream event dependency). Both allocations
    /// must remain live, with no conflicting access, until the copy completes.
    pub unsafe fn copy_peer_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: Ds41rtDeviceBuffer,
        bytes: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(
            Ds41rtDeviceBuffer, Ds41rtDeviceBuffer, usize, *mut c_void,
        ) -> Ds41rtStatus> = unsafe { self.lib.get(b"ds41rt_copy_peer_async")? };
        self.status_to_result("ds41rt_copy_peer_async", unsafe { call(dst, src, bytes, stream) })
    }

    /// Copy pitched byte rows locally or between peer GPUs without synchronization.
    ///
    /// # Safety
    /// The stream belongs to the current destination device. Source writes are
    /// complete or ordered before this copy. Buffers remain live and disjoint,
    /// without conflicting access, until stream completion.
    pub unsafe fn copy_device_rows_async(
        &self, dst: Ds41rtDeviceBuffer, src: Ds41rtDeviceBuffer,
        width: usize, rows: usize, dst_pitch: usize, src_pitch: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let call: Symbol<unsafe extern "C" fn(
            Ds41rtDeviceBuffer, Ds41rtDeviceBuffer, usize, usize, usize, usize, *mut c_void,
        ) -> Ds41rtStatus> = unsafe { self.lib.get(b"ds41rt_copy_device_rows_async")? };
        self.status_to_result("ds41rt_copy_device_rows_async", unsafe {
            call(dst, src, width, rows, dst_pitch, src_pitch, stream)
        })
    }

    pub fn alloc_device_buffer(&self, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
        let alloc_fn: Symbol<AllocDeviceBufferFn> =
            unsafe { self.lib.get(b"ds41rt_alloc_device_buffer")? };
        let mut buffer = Ds41rtDeviceBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("ds41rt_alloc_device_buffer", status)?;
        Ok(buffer)
    }

    pub fn alloc_managed_device_buffer(&self, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
        let alloc_fn: Symbol<AllocManagedDeviceBufferFn> =
            unsafe { self.lib.get(b"ds41rt_alloc_managed_device_buffer")? };
        let mut buffer = Ds41rtDeviceBuffer::default();
        let status = unsafe { alloc_fn(bytes, &mut buffer) };
        self.status_to_result("ds41rt_alloc_managed_device_buffer", status)?;
        Ok(buffer)
    }

    pub fn free_device_buffer(&self, buffer: &mut Ds41rtDeviceBuffer) -> Result<()> {
        let free_fn: Symbol<FreeDeviceBufferFn> =
            unsafe { self.lib.get(b"ds41rt_free_device_buffer")? };
        let status = unsafe { free_fn(buffer) };
        self.status_to_result("ds41rt_free_device_buffer", status)
    }

    pub fn cuda_stream_create(&self) -> Result<*mut c_void> {
        let create_fn: Symbol<CudaStreamCreateFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_stream_create")? };
        let mut cuda_stream = std::ptr::null_mut();
        let status = unsafe { create_fn(&mut cuda_stream) };
        self.status_to_result("ds41rt_cuda_stream_create", status)?;
        Ok(cuda_stream)
    }

    pub unsafe fn cuda_stream_destroy(&self, cuda_stream: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaStreamDestroyFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_stream_destroy")? };
        let status = unsafe { destroy_fn(cuda_stream) };
        self.status_to_result("ds41rt_cuda_stream_destroy", status)
    }

    pub unsafe fn cuda_stream_synchronize(&self, cuda_stream: *mut c_void) -> Result<()> {
        let synchronize_fn: Symbol<CudaStreamSynchronizeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_stream_synchronize")? };
        let status = unsafe { synchronize_fn(cuda_stream) };
        self.status_to_result("ds41rt_cuda_stream_synchronize", status)
    }

    /// Poll completion without blocking the CUDA owner thread.
    /// The stream must remain live on the current CUDA device.
    pub unsafe fn cuda_stream_query(&self, cuda_stream: *mut c_void) -> Result<bool> {
        let query_fn: Symbol<CudaStreamQueryFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_stream_query")? };
        let mut ready = 0;
        let status = unsafe { query_fn(cuda_stream, &mut ready) };
        self.status_to_result("ds41rt_cuda_stream_query", status)?;
        anyhow::ensure!(ready == 0 || ready == 1, "invalid CUDA stream query result");
        Ok(ready == 1)
    }

    pub unsafe fn cuda_stream_wait_event(
        &self,
        cuda_stream: *mut c_void,
        cuda_event: *mut c_void,
    ) -> Result<()> {
        let wait_fn: Symbol<CudaStreamWaitEventFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_stream_wait_event")? };
        let status = unsafe { wait_fn(cuda_stream, cuda_event) };
        self.status_to_result("ds41rt_cuda_stream_wait_event", status)
    }

    pub fn cuda_event_create(&self) -> Result<*mut c_void> {
        let create_fn: Symbol<CudaEventCreateFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_event_create")? };
        let mut cuda_event = std::ptr::null_mut();
        let status = unsafe { create_fn(&mut cuda_event) };
        self.status_to_result("ds41rt_cuda_event_create", status)?;
        Ok(cuda_event)
    }

    pub unsafe fn cuda_event_destroy(&self, cuda_event: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaEventDestroyFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_event_destroy")? };
        let status = unsafe { destroy_fn(cuda_event) };
        self.status_to_result("ds41rt_cuda_event_destroy", status)
    }

    pub unsafe fn cuda_event_record(
        &self,
        cuda_event: *mut c_void,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let record_fn: Symbol<CudaEventRecordFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_event_record")? };
        let status = unsafe { record_fn(cuda_event, cuda_stream) };
        self.status_to_result("ds41rt_cuda_event_record", status)
    }

    pub unsafe fn cuda_event_synchronize(&self, cuda_event: *mut c_void) -> Result<()> {
        let synchronize_fn: Symbol<CudaEventSynchronizeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_event_synchronize")? };
        let status = unsafe { synchronize_fn(cuda_event) };
        self.status_to_result("ds41rt_cuda_event_synchronize", status)
    }

    pub unsafe fn cuda_event_elapsed_ms(
        &self,
        start_event: *mut c_void,
        end_event: *mut c_void,
    ) -> Result<f32> {
        let elapsed_fn: Symbol<CudaEventElapsedMsFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_event_elapsed_ms")? };
        let mut out_ms = 0.0_f32;
        let status = unsafe { elapsed_fn(start_event, end_event, &mut out_ms) };
        self.status_to_result("ds41rt_cuda_event_elapsed_ms", status)?;
        Ok(out_ms)
    }

    pub unsafe fn cuda_graph_begin_capture(&self, cuda_stream: *mut c_void) -> Result<()> {
        let begin_capture_fn: Symbol<CudaGraphBeginCaptureFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_begin_capture")? };
        let status = unsafe { begin_capture_fn(cuda_stream) };
        self.status_to_result("ds41rt_cuda_graph_begin_capture", status)
    }

    pub unsafe fn cuda_graph_end_capture(&self, cuda_stream: *mut c_void) -> Result<*mut c_void> {
        let end_capture_fn: Symbol<CudaGraphEndCaptureFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_end_capture")? };
        let mut cuda_graph_exec = std::ptr::null_mut();
        let status = unsafe { end_capture_fn(cuda_stream, &mut cuda_graph_exec) };
        self.status_to_result("ds41rt_cuda_graph_end_capture", status)?;
        Ok(cuda_graph_exec)
    }

    pub unsafe fn cuda_graph_end_capture_retained(
        &self,
        cuda_stream: *mut c_void,
    ) -> Result<Ds41rtCudaGraphCaptureInfo> {
        let end_capture_fn: Symbol<CudaGraphEndCaptureRetainedFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_end_capture_retained")? };
        let mut capture = Ds41rtCudaGraphCaptureInfo::default();
        let status = unsafe { end_capture_fn(cuda_stream, &mut capture) };
        self.status_to_result("ds41rt_cuda_graph_end_capture_retained", status)?;
        Ok(capture)
    }

    pub unsafe fn cuda_graph_launch(
        &self,
        cuda_graph_exec: *mut c_void,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let launch_fn: Symbol<CudaGraphLaunchFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_launch")? };
        let status = unsafe { launch_fn(cuda_graph_exec, cuda_stream) };
        self.status_to_result("ds41rt_cuda_graph_launch", status)
    }

    pub unsafe fn cuda_graph_exec_update(
        &self,
        cuda_graph_exec: *mut c_void,
        cuda_graph: *mut c_void,
    ) -> Result<()> {
        let update_fn: Symbol<CudaGraphExecUpdateFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_exec_update")? };
        let status = unsafe { update_fn(cuda_graph_exec, cuda_graph) };
        self.status_to_result("ds41rt_cuda_graph_exec_update", status)
    }

    pub unsafe fn cuda_graph_destroy(&self, cuda_graph: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaGraphDestroyFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_destroy")? };
        let status = unsafe { destroy_fn(cuda_graph) };
        self.status_to_result("ds41rt_cuda_graph_destroy", status)
    }

    pub unsafe fn cuda_graph_exec_destroy(&self, cuda_graph_exec: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<CudaGraphExecDestroyFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_exec_destroy")? };
        let status = unsafe { destroy_fn(cuda_graph_exec) };
        self.status_to_result("ds41rt_cuda_graph_exec_destroy", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_rmsnorm_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_rmsnorm_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_rmsnorm_bf16_node graph exec is null");
        }
        validate_f32_rows("ds41rt_cuda_graph_update_rmsnorm_bf16_node", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_graph_update_rmsnorm_bf16_node x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_graph_update_rmsnorm_bf16_node x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_rmsnorm_bf16_node weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_rmsnorm_bf16_node out",
            out,
            row_values,
        )?;

        let update_fn: Symbol<CudaGraphUpdateRmsNormBf16NodeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_update_rmsnorm_bf16_node")? };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                x,
                weight,
                out,
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_rmsnorm_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_layernorm_affine_f32_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node graph exec is null"
            );
        }
        validate_f32_rows(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node",
            rows,
            hidden,
        )?;
        let row_values = checked_row_values(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node x",
            rows as usize,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node x",
            x,
            row_values * std::mem::size_of::<f32>(),
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node bias",
            bias,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node out",
            out,
            row_values * std::mem::size_of::<f32>(),
        )?;

        let update_fn: Symbol<CudaGraphUpdateLayerNormAffineF32Bf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                x,
                weight,
                bias,
                out,
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_layernorm_affine_f32_bf16_node",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_layernorm_affine_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_layernorm_affine_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_layernorm_affine_bf16_node graph exec is null");
        }
        validate_f32_rows(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node",
            rows,
            hidden,
        )?;
        let row_values = checked_row_values(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node x",
            x,
            row_values,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node bias",
            bias,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_layernorm_affine_bf16_node out",
            out,
            row_values,
        )?;

        let update_fn: Symbol<CudaGraphUpdateLayerNormAffineBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_layernorm_affine_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                x,
                weight,
                bias,
                out,
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_layernorm_affine_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_linear_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_linear_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_linear_bf16_node graph exec is null");
        }
        validate_linear_bf16_buffers(
            "ds41rt_cuda_graph_update_linear_bf16_node",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let update_fn: Symbol<CudaGraphUpdateLinearBf16NodeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_update_linear_bf16_node")? };
        let bias_storage = bias;
        let bias_ptr = bias_storage
            .as_ref()
            .map(|buffer| buffer as *const Ds41rtDeviceBuffer)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                input,
                weight,
                bias_ptr,
                output,
                rows,
                input_dim,
                output_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_linear_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_embedding_lookup_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        embedding: Ds41rtDeviceBuffer,
        token_ids: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_embedding_lookup_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_embedding_lookup_bf16_node graph exec is null");
        }
        validate_embedding_lookup_bf16_buffers(
            "ds41rt_cuda_graph_update_embedding_lookup_bf16_node",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let update_fn: Symbol<CudaGraphUpdateEmbeddingLookupBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_embedding_lookup_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                embedding,
                token_ids,
                out,
                rows,
                vocab,
                hidden,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_embedding_lookup_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_lm_head_argmax_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_lm_head_argmax_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_lm_head_argmax_bf16_node graph exec is null");
        }
        validate_lm_head_argmax_bf16_buffers(
            "ds41rt_cuda_graph_update_lm_head_argmax_bf16_node",
            hidden,
            lm_head,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
        )?;

        let update_fn: Symbol<CudaGraphUpdateLmHeadArgmaxBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_lm_head_argmax_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                hidden,
                lm_head,
                out_indices,
                out_scores,
                rows,
                hidden_dim,
                vocab,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_lm_head_argmax_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_lm_head_sample_topk_topp_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node graph is null"
            );
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node graph exec is null"
            );
        }
        validate_lm_head_sample_topk_topp_bf16_buffers(
            "ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node",
            hidden,
            lm_head,
            random_uniforms,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let update_fn: Symbol<CudaGraphUpdateLmHeadSampleTopKToppBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                hidden,
                lm_head,
                random_uniforms,
                out_indices,
                out_scores,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_router_topk_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_router_topk_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_router_topk_bf16_node graph exec is null");
        }
        validate_router_topk_bf16_buffers(
            "ds41rt_cuda_graph_update_router_topk_bf16_node",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;

        let update_fn: Symbol<CudaGraphUpdateRouterTopKBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_router_topk_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                hidden,
                router_weight,
                correction_bias,
                topk_indices,
                topk_scores,
                topk_weights,
                rows,
                hidden_dim,
                experts,
                top_k,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_router_topk_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node graph is null"
            );
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node graph exec is null"
            );
        }
        validate_mlp_rows_bf16_down_stride_buffers(
            "ds41rt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
            down_stride,
        )?;

        let update_fn: Symbol<CudaGraphUpdateSiluGatedMlpRowsBf16DownStrideNodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                x,
                gate_weight,
                up_weight,
                down_weight,
                out,
                rows,
                hidden,
                intermediate,
                down_stride,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node",
            status,
        )
    }

    pub unsafe fn cuda_graph_update_residual_add_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        residual: Ds41rtDeviceBuffer,
        delta: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_residual_add_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_residual_add_bf16_node graph exec is null");
        }
        if count == 0 {
            anyhow::bail!("ds41rt_cuda_graph_update_residual_add_bf16_node count must be positive");
        }
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_bf16_node residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_bf16_node delta",
            delta,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_bf16_node out",
            out,
            count,
        )?;

        let update_fn: Symbol<CudaGraphUpdateResidualAddBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_residual_add_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                residual,
                delta,
                out,
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_residual_add_bf16_node", status)
    }

    pub unsafe fn cuda_graph_update_residual_add_f32_delta_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        residual: Ds41rtDeviceBuffer,
        delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node graph exec is null"
            );
        }
        if count == 0 {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node count must be positive"
            );
        }
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node residual",
            residual,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node delta_f32",
            delta_f32,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node out",
            out,
            count,
        )?;

        let update_fn: Symbol<CudaGraphUpdateResidualAddF32DeltaBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                residual,
                delta_f32,
                out,
                count,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_residual_add_f32_delta_bf16_node",
            status,
        )
    }

    pub unsafe fn cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        residual: Ds41rtDeviceBuffer,
        shared_delta: Ds41rtDeviceBuffer,
        routed_delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node graph is null"
            );
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node graph exec is null"
            );
        }
        if count == 0 {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node count must be positive"
            );
        }
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node shared_delta",
            shared_delta,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node routed_delta_f32",
            routed_delta_f32,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node out",
            out,
            count,
        )?;

        let update_fn: Symbol<CudaGraphUpdateResidualAddSharedF32DeltaBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                residual,
                shared_delta,
                routed_delta_f32,
                out,
                count,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node",
            status,
        )
    }

    pub unsafe fn cuda_graph_update_f32_to_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_f32_to_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_f32_to_bf16_node graph exec is null");
        }
        if count == 0 {
            anyhow::bail!("ds41rt_cuda_graph_update_f32_to_bf16_node count must be positive");
        }
        validate_f32_to_bf16_buffers("ds41rt_cuda_graph_update_f32_to_bf16_node", src, dst, count)?;

        let update_fn: Symbol<CudaGraphUpdateF32ToBf16NodeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_update_f32_to_bf16_node")? };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                src,
                dst,
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_f32_to_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node graph is null"
            );
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node graph exec is null"
            );
        }
        validate_row_scatter_bf16_to_f32_buffers(
            "ds41rt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node",
            src,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let update_fn: Symbol<CudaGraphUpdateScatterAddRowsBf16ToF32NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                src,
                row_indices,
                dst,
                dst_rows,
                rows,
                row_width,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node",
            status,
        )
    }

    pub unsafe fn cuda_graph_update_kv_cache_write_bytes_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        src: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_offset_bytes: usize,
        bytes: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_kv_cache_write_bytes_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_kv_cache_write_bytes_node graph exec is null");
        }
        if bytes == 0 {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_kv_cache_write_bytes_node bytes must be positive"
            );
        }
        validate_kv_cache_write_buffers(
            "ds41rt_cuda_graph_update_kv_cache_write_bytes_node",
            src,
            cache,
            cache_offset_bytes,
            bytes,
        )?;

        let update_fn: Symbol<CudaGraphUpdateKvCacheWriteBytesNodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_kv_cache_write_bytes_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                src,
                cache,
                cache_offset_bytes,
                bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_kv_cache_write_bytes_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_causal_attention_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        q: Ds41rtDeviceBuffer,
        k: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_causal_attention_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_causal_attention_bf16_node graph exec is null");
        }
        validate_causal_attention_bf16_buffers(
            "ds41rt_cuda_graph_update_causal_attention_bf16_node",
            q,
            k,
            v,
            out,
            rows,
            heads,
            qk_dim,
            v_dim,
        )?;

        let update_fn: Symbol<CudaGraphUpdateCausalAttentionBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_causal_attention_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                q,
                k,
                v,
                out,
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_causal_attention_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_rope_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        input: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_rope_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_rope_bf16_node graph exec is null");
        }
        validate_rope_bf16_buffers(
            "ds41rt_cuda_graph_update_rope_bf16_node",
            input,
            positions,
            out,
            rows,
            heads,
            rotary_dim,
            theta,
        )?;

        let update_fn: Symbol<CudaGraphUpdateRopeBf16NodeFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_graph_update_rope_bf16_node")? };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                input,
                positions,
                out,
                rows,
                heads,
                rotary_dim,
                theta,
            )
        };
        self.status_to_result("ds41rt_cuda_graph_update_rope_bf16_node", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_mla_rope_attention_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_mla_rope_attention_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_mla_rope_attention_bf16_node graph exec is null"
            );
        }
        validate_mla_rope_attention_bf16_buffers(
            "ds41rt_cuda_graph_update_mla_rope_attention_bf16_node",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let update_fn: Symbol<CudaGraphUpdateMlaRopeAttentionBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_mla_rope_attention_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                v,
                out,
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_mla_rope_attention_bf16_node",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_mla_rope_attention_bf16_suffix_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        query_row_offset: usize,
        query_rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_mla_rope_attention_bf16_suffix_node graph is null"
            );
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_mla_rope_attention_bf16_suffix_node graph exec is null"
            );
        }
        validate_mla_rope_attention_bf16_suffix_buffers(
            "ds41rt_cuda_graph_update_mla_rope_attention_bf16_suffix_node",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            query_row_offset,
            query_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let update_fn: Symbol<CudaGraphUpdateMlaRopeAttentionBf16SuffixNodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_mla_rope_attention_bf16_suffix_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                v,
                out,
                rows,
                query_row_offset,
                query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_mla_rope_attention_bf16_suffix_node",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_mla_kv_cache_unpack_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        payload: Ds41rtDeviceBuffer,
        kv_latent: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        dsa_key: Option<Ds41rtDeviceBuffer>,
        rows: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        dsa_dim: usize,
        payload_stride_bytes: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_mla_kv_cache_unpack_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_mla_kv_cache_unpack_bf16_node graph exec is null"
            );
        }
        validate_mla_kv_cache_unpack_bf16_buffers(
            "ds41rt_cuda_graph_update_mla_kv_cache_unpack_bf16_node",
            payload,
            kv_latent,
            k_rope,
            dsa_key,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
        )?;

        let update_fn: Symbol<CudaGraphUpdateMlaKvCacheUnpackBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_mla_kv_cache_unpack_bf16_node")?
        };
        let dsa_key_buffer = dsa_key.unwrap_or_default();
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                payload,
                kv_latent,
                k_rope,
                dsa_key_buffer,
                rows,
                kv_lora_rank,
                rope_dim,
                dsa_dim,
                payload_stride_bytes,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_mla_kv_cache_unpack_bf16_node",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_graph_update_mla_kv_projected_split_bf16_node(
        &self,
        cuda_graph: *mut c_void,
        cuda_graph_exec: *mut c_void,
        kernel_node_index: usize,
        projected: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
    ) -> Result<()> {
        if cuda_graph.is_null() {
            anyhow::bail!("ds41rt_cuda_graph_update_mla_kv_projected_split_bf16_node graph is null");
        }
        if cuda_graph_exec.is_null() {
            anyhow::bail!(
                "ds41rt_cuda_graph_update_mla_kv_projected_split_bf16_node graph exec is null"
            );
        }
        validate_mla_kv_projected_split_bf16_buffers(
            "ds41rt_cuda_graph_update_mla_kv_projected_split_bf16_node",
            projected,
            k_nope,
            v,
            rows,
            heads,
            nope_dim,
            v_dim,
        )?;

        let update_fn: Symbol<CudaGraphUpdateMlaKvProjectedSplitBf16NodeFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_graph_update_mla_kv_projected_split_bf16_node")?
        };
        let status = unsafe {
            update_fn(
                cuda_graph,
                cuda_graph_exec,
                kernel_node_index,
                projected,
                k_nope,
                v,
                rows,
                heads,
                nope_dim,
                v_dim,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_graph_update_mla_kv_projected_split_bf16_node",
            status,
        )
    }

    pub fn copy_h2d(&self, dst: Ds41rtDeviceBuffer, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if src.len() > dst.bytes {
            anyhow::bail!(
                "ds41rt_copy_h2d staged source byte count {} exceeds destination device buffer bytes {}",
                src.len(),
                dst.bytes
            );
        }
        let mut staging = self
            .sync_h2d_staging
            .lock()
            .map_err(|_| anyhow::anyhow!("synchronous H2D pinned staging lock is poisoned"))?;
        let staging = staging.ensure(self, src.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), staging.ptr.cast::<u8>(), src.len());
        }
        unsafe { self.copy_h2d_raw_ptr(dst, staging.ptr as *const c_void, src.len()) }
    }

    pub fn copy_host_buffer_h2d(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: Ds41rtHostBuffer,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if src.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_h2d pinned source host buffer is null");
        }
        if dst.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_h2d destination device buffer is null");
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "ds41rt_copy_h2d pinned source byte count {bytes} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "ds41rt_copy_h2d pinned source byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        unsafe { self.copy_h2d_raw_ptr(dst, src.ptr as *const c_void, bytes) }
    }

    unsafe fn copy_h2d_raw_ptr(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyH2DFn> = unsafe { self.lib.get(b"ds41rt_copy_h2d")? };
        let status = unsafe { copy_fn(dst, src, bytes) };
        self.status_to_result("ds41rt_copy_h2d", status)
    }

    pub unsafe fn copy_host_buffer_h2d_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: Ds41rtHostBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if src.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_h2d_async pinned source host buffer is null");
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "ds41rt_copy_h2d_async pinned source byte count {bytes} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyH2DAsyncFn> = unsafe { self.lib.get(b"ds41rt_copy_h2d_async")? };
        let status = unsafe { copy_fn(dst, src.ptr as *const c_void, bytes, cuda_stream) };
        self.status_to_result("ds41rt_copy_h2d_async", status)
    }

    pub unsafe fn copy_host_buffers_h2d_batch_async(
        &self,
        dsts: &[Ds41rtDeviceBuffer],
        srcs: &[Ds41rtHostBuffer],
        bytes: &[usize],
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_copy_h2d_batch_async";
        if dsts.is_empty() {
            return Ok(());
        }
        if dsts.len() != srcs.len() || dsts.len() != bytes.len() {
            anyhow::bail!(
                "{context} array lengths differ: dsts={} srcs={} bytes={}",
                dsts.len(),
                srcs.len(),
                bytes.len()
            );
        }
        if cuda_stream.is_null() {
            anyhow::bail!("{context} requires a non-default CUDA stream");
        }
        for (index, ((dst, src), &entry_bytes)) in dsts.iter().zip(srcs).zip(bytes).enumerate() {
            if entry_bytes == 0 {
                anyhow::bail!("{context} entry {index} is empty");
            }
            if src.ptr.is_null() {
                anyhow::bail!("{context} entry {index} source host buffer is null");
            }
            if entry_bytes > src.bytes {
                anyhow::bail!(
                    "{context} entry {index} byte count {entry_bytes} exceeds host buffer bytes {}",
                    src.bytes
                );
            }
            validate_device_buffer_bytes(
                &format!("{context} entry {index} destination"),
                *dst,
                entry_bytes,
            )?;
        }
        let source_ptrs = srcs
            .iter()
            .map(|source| source.ptr.cast_const())
            .collect::<Vec<_>>();
        let copy_fn: Symbol<CopyH2DBatchAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_copy_h2d_batch_async")? };
        let status = unsafe {
            copy_fn(
                dsts.as_ptr(),
                source_ptrs.as_ptr(),
                bytes.as_ptr(),
                dsts.len(),
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub unsafe fn copy_host_buffer_h2d_2d_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        dst_pitch_bytes: usize,
        src: Ds41rtHostBuffer,
        src_pitch_bytes: usize,
        width_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_copy_h2d_2d_async";
        if width_bytes == 0 || rows == 0 {
            return Ok(());
        }
        if src.ptr.is_null() {
            anyhow::bail!("{context} pinned source host buffer is null");
        }
        if dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes {
            anyhow::bail!("{context} pitch must be at least the row width");
        }
        let dst_required = (rows - 1)
            .checked_mul(dst_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D H2D destination byte span overflow")?;
        let src_required = (rows - 1)
            .checked_mul(src_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D H2D source byte span overflow")?;
        validate_device_buffer_bytes(&format!("{context} destination"), dst, dst_required)?;
        if src_required > src.bytes {
            anyhow::bail!(
                "{context} source byte span {src_required} exceeds host buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyH2D2DAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_copy_h2d_2d_async")? };
        let status = unsafe {
            copy_fn(
                dst,
                dst_pitch_bytes,
                src.ptr as *const c_void,
                src_pitch_bytes,
                width_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn copy_d2h(&self, dst: &mut [u8], src: Ds41rtDeviceBuffer) -> Result<()> {
        let copy_fn: Symbol<CopyD2HFn> = unsafe { self.lib.get(b"ds41rt_copy_d2h")? };
        let status = unsafe { copy_fn(dst.as_mut_ptr().cast(), src, dst.len()) };
        self.status_to_result("ds41rt_copy_d2h", status)
    }

    #[cfg(test)]
    fn sync_h2d_staging_snapshot(&self) -> Option<(usize, usize)> {
        let staging = self.sync_h2d_staging.lock().ok()?;
        (!staging.buffer.ptr.is_null())
            .then_some((staging.buffer.ptr as usize, staging.buffer.bytes))
    }

    /// Copy initialized device storage and wait for the copy to complete.
    /// Producers must have completed before this call; subsequent consumers
    /// may use any stream. Use `copy_d2d_async` for explicitly ordered work.
    pub fn copy_d2d(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: Ds41rtDeviceBuffer,
        bytes: usize,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if dst.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_d2d destination device buffer is null");
        }
        if src.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_d2d source device buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "ds41rt_copy_d2d byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "ds41rt_copy_d2d byte count {bytes} exceeds source device buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyD2DFn> = unsafe { self.lib.get(b"ds41rt_copy_d2d")? };
        let status = unsafe { copy_fn(dst, src, bytes) };
        self.status_to_result("ds41rt_copy_d2d", status)
    }

    pub unsafe fn copy_h2d_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: &[u8],
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyH2DAsyncFn> = unsafe { self.lib.get(b"ds41rt_copy_h2d_async")? };
        let status = unsafe { copy_fn(dst, src.as_ptr().cast(), src.len(), cuda_stream) };
        self.status_to_result("ds41rt_copy_h2d_async", status)
    }

    pub unsafe fn copy_d2h_async(
        &self,
        dst: &mut [u8],
        src: Ds41rtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let copy_fn: Symbol<CopyD2HAsyncFn> = unsafe { self.lib.get(b"ds41rt_copy_d2h_async")? };
        let status = unsafe { copy_fn(dst.as_mut_ptr().cast(), src, dst.len(), cuda_stream) };
        self.status_to_result("ds41rt_copy_d2h_async", status)
    }

    pub unsafe fn copy_d2h_host_buffer_async(
        &self,
        dst: Ds41rtHostBuffer,
        src: Ds41rtDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if dst.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_d2h_async pinned destination host buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "ds41rt_copy_d2h_async pinned destination byte count {bytes} exceeds host buffer bytes {}",
                dst.bytes
            );
        }
        let copy_fn: Symbol<CopyD2HAsyncFn> = unsafe { self.lib.get(b"ds41rt_copy_d2h_async")? };
        let status = unsafe { copy_fn(dst.ptr, src, bytes, cuda_stream) };
        self.status_to_result("ds41rt_copy_d2h_async", status)
    }

    pub unsafe fn copy_d2d_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        src: Ds41rtDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if dst.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_d2d_async destination device buffer is null");
        }
        if src.ptr.is_null() {
            anyhow::bail!("ds41rt_copy_d2d_async source device buffer is null");
        }
        if bytes > dst.bytes {
            anyhow::bail!(
                "ds41rt_copy_d2d_async byte count {bytes} exceeds destination device buffer bytes {}",
                dst.bytes
            );
        }
        if bytes > src.bytes {
            anyhow::bail!(
                "ds41rt_copy_d2d_async byte count {bytes} exceeds source device buffer bytes {}",
                src.bytes
            );
        }
        let copy_fn: Symbol<CopyD2DAsyncFn> = unsafe { self.lib.get(b"ds41rt_copy_d2d_async")? };
        let status = unsafe { copy_fn(dst, src, bytes, cuda_stream) };
        self.status_to_result("ds41rt_copy_d2d_async", status)
    }

    pub unsafe fn copy_d2d_2d_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        dst_pitch_bytes: usize,
        src: Ds41rtDeviceBuffer,
        src_pitch_bytes: usize,
        width_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_copy_d2d_2d_async";
        if width_bytes == 0 || rows == 0 {
            return Ok(());
        }
        if dst_pitch_bytes < width_bytes || src_pitch_bytes < width_bytes {
            anyhow::bail!("{context} pitch must be at least the row width");
        }
        let dst_required = (rows - 1)
            .checked_mul(dst_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D D2D destination byte span overflow")?;
        let src_required = (rows - 1)
            .checked_mul(src_pitch_bytes)
            .and_then(|bytes| bytes.checked_add(width_bytes))
            .context("2D D2D source byte span overflow")?;
        validate_device_buffer_bytes(&format!("{context} destination"), dst, dst_required)?;
        validate_device_buffer_bytes(&format!("{context} source"), src, src_required)?;
        if dst.device_id != src.device_id {
            anyhow::bail!("{context} buffers must reside on the same CUDA device");
        }
        let copy_fn: Symbol<CopyD2D2DAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_copy_d2d_2d_async")? };
        let status = unsafe {
            copy_fn(
                dst,
                dst_pitch_bytes,
                src,
                src_pitch_bytes,
                width_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn cuda_rmsnorm_f32(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_rmsnorm_f32", rows, hidden)?;
        let row_values = rows as usize * hidden as usize;
        validate_device_buffer_bytes("ds41rt_cuda_rmsnorm_f32 x", x, row_values * 4)?;
        validate_device_buffer_bytes("ds41rt_cuda_rmsnorm_f32 weight", weight, hidden as usize * 4)?;
        validate_device_buffer_bytes("ds41rt_cuda_rmsnorm_f32 out", out, row_values * 4)?;

        let kernel_fn: Symbol<CudaRmsNormF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rmsnorm_f32")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_rmsnorm_f32", status)
    }

    pub unsafe fn cuda_rmsnorm_f32_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_rmsnorm_f32_async", rows, hidden)?;
        let row_values = rows as usize * hidden as usize;
        validate_device_buffer_bytes("ds41rt_cuda_rmsnorm_f32_async x", x, row_values * 4)?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_rmsnorm_f32_async weight",
            weight,
            hidden as usize * 4,
        )?;
        validate_device_buffer_bytes("ds41rt_cuda_rmsnorm_f32_async out", out, row_values * 4)?;

        let kernel_fn: Symbol<CudaRmsNormF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rmsnorm_f32_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_rmsnorm_f32_async", status)
    }

    pub fn cuda_rmsnorm_bf16(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_rmsnorm_bf16", rows, hidden)?;
        let row_values =
            checked_row_values("ds41rt_cuda_rmsnorm_bf16 x", rows as usize, hidden as usize)?;
        validate_u16_buffer_values("ds41rt_cuda_rmsnorm_bf16 x", x, row_values)?;
        validate_u16_buffer_values("ds41rt_cuda_rmsnorm_bf16 weight", weight, hidden as usize)?;
        validate_u16_buffer_values("ds41rt_cuda_rmsnorm_bf16 out", out, row_values)?;

        let kernel_fn: Symbol<CudaRmsNormBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rmsnorm_bf16")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_rmsnorm_bf16", status)
    }

    /// Dequantize gathered engram FP8 rows and UE8M0 scales into a preallocated BF16 buffer.
    ///
    /// # Safety
    /// Buffers must be nonoverlapping, device-resident on the stream's device,
    /// and remain valid until the stream completes.
    pub unsafe fn cuda_engram_dequant_bf16_async(
        &self, weights: Ds41rtDeviceBuffer, scales: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer, hash_rows: i32, cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "ds41rt_cuda_engram_dequant_bf16_async";
        validate_f32_rows(NAME, hash_rows, 256)?;
        let values = checked_row_values(NAME, hash_rows as usize, 256)?;
        validate_device_buffer_bytes(NAME, weights, values)?;
        validate_device_buffer_bytes(NAME, scales, values / 32)?;
        validate_u16_buffer_values(NAME, out, values)?;
        type Kernel = unsafe extern "C" fn(*const u8, *const u8, *mut u16, i32, *mut c_void) -> Ds41rtStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"ds41rt_cuda_engram_dequant_bf16_async")? };
        let status = unsafe { kernel(weights.ptr.cast(), scales.ptr.cast(), out.ptr.cast(), hash_rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }

    /// Decode packed NVFP4 PLE rows directly to BF16 without another quantization.
    /// # Safety
    /// Input/output regions must not overlap and must remain live on the stream's device.
    pub unsafe fn cuda_engram_nvfp4_dequant_bf16_async(
        &self, weights: Ds41rtDeviceBuffer, scales: Ds41rtDeviceBuffer, global_scale: f32,
        out: Ds41rtDeviceBuffer, hash_rows: i32, cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "ds41rt_cuda_engram_nvfp4_dequant_bf16_async";
        anyhow::ensure!(global_scale.is_finite() && global_scale > 0.0, "invalid NVFP4 PLE global scale");
        validate_f32_rows(NAME, hash_rows, 256)?;
        let values = checked_row_values(NAME, hash_rows as usize, 256)?;
        validate_device_buffer_bytes(NAME, weights, values / 2)?;
        validate_device_buffer_bytes(NAME, scales, values / 16)?;
        validate_u16_buffer_values(NAME, out, values)?;
        type Kernel = unsafe extern "C" fn(*const u8, *const u8, f32, *mut u16, i32, *mut c_void) -> Ds41rtStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"ds41rt_cuda_engram_nvfp4_dequant_bf16_async")? };
        let status = unsafe { kernel(weights.ptr.cast(), scales.ptr.cast(), global_scale,
            out.ptr.cast(), hash_rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }

    /// Fused official V4.1 engram gate; buffers must remain live on the stream.
    ///
    /// # Safety
    /// All buffers must be device-resident on the stream's device; output may
    /// equal x but must not overlap projected_kv, weights, or the text mask.
    pub unsafe fn cuda_engram_gate_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        projected_kv: Ds41rtDeviceBuffer,
        q_weight: Ds41rtDeviceBuffer,
        k_weight: Ds41rtDeviceBuffer,
        text_mask: Option<Ds41rtDeviceBuffer>,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NAME: &str = "ds41rt_cuda_engram_gate_bf16_async";
        validate_f32_rows(NAME, rows, 5120)?;
        let residual_values = checked_row_values(NAME, rows as usize, 4 * 5120)?;
        let kv_values = checked_row_values(NAME, rows as usize, 5 * 5120)?;
        validate_u16_buffer_values(NAME, x, residual_values)?;
        validate_u16_buffer_values(NAME, out, residual_values)?;
        validate_u16_buffer_values(NAME, projected_kv, kv_values)?;
        validate_u16_buffer_values(NAME, q_weight, 4 * 5120)?;
        validate_u16_buffer_values(NAME, k_weight, 4 * 5120)?;
        if let Some(mask) = text_mask { validate_device_buffer_bytes(NAME, mask, rows as usize)?; }
        type Kernel = unsafe extern "C" fn(*const u16, *const u16, *const u16, *const u16,
            *const u8, *mut u16, i32, *mut c_void) -> Ds41rtStatus;
        let kernel: Symbol<Kernel> = unsafe { self.lib.get(b"ds41rt_cuda_engram_gate_bf16_async")? };
        let status = unsafe { kernel(x.ptr.cast(), projected_kv.ptr.cast(), q_weight.ptr.cast(),
            k_weight.ptr.cast(), text_mask.map_or(std::ptr::null(), |mask| mask.ptr.cast()),
            out.ptr.cast(), rows, cuda_stream) };
        self.status_to_result(NAME, status)
    }

    pub unsafe fn cuda_rmsnorm_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_rmsnorm_bf16_async", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_rmsnorm_bf16_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_rmsnorm_bf16_async x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_rmsnorm_bf16_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_rmsnorm_bf16_async out", out, row_values)?;

        let kernel_fn: Symbol<CudaRmsNormBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rmsnorm_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_rmsnorm_bf16_async", status)
    }

    pub unsafe fn cuda_ds4_flash_rmsnorm_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_ds4_flash_rmsnorm_bf16_async", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_ds4_flash_rmsnorm_bf16_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_ds4_flash_rmsnorm_bf16_async x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_ds4_flash_rmsnorm_bf16_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_ds4_flash_rmsnorm_bf16_async out",
            out,
            row_values,
        )?;

        let kernel_fn: Symbol<CudaDs4FlashRmsNormBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_flash_rmsnorm_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_rmsnorm_bf16_async", status)
    }

    pub unsafe fn cuda_ds4_rmsnorm_bf16_rne_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_ds4_rmsnorm_bf16_rne_async", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_ds4_rmsnorm_bf16_rne_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_ds4_rmsnorm_bf16_rne_async x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_ds4_rmsnorm_bf16_rne_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_ds4_rmsnorm_bf16_rne_async out", out, row_values)?;

        let kernel_fn: Symbol<CudaDs4RmsNormBf16RneAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_rmsnorm_bf16_rne_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_rmsnorm_bf16_rne_async", status)
    }

    pub fn cuda_layernorm_affine_f32_bf16(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_layernorm_affine_f32_bf16", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16 x",
            rows as usize,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_layernorm_affine_f32_bf16 x",
            x,
            row_values * std::mem::size_of::<f32>(),
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16 weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16 bias",
            bias,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_layernorm_affine_f32_bf16 out",
            out,
            row_values * std::mem::size_of::<f32>(),
        )?;

        let kernel_fn: Symbol<CudaLayerNormAffineF32Bf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_layernorm_affine_f32_bf16")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<u16>() as *const u16,
                bias.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_layernorm_affine_f32_bf16", status)
    }

    pub unsafe fn cuda_layernorm_affine_f32_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_layernorm_affine_f32_bf16_async", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_layernorm_affine_f32_bf16_async x",
            x,
            row_values * std::mem::size_of::<f32>(),
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_f32_bf16_async bias",
            bias,
            hidden as usize,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_layernorm_affine_f32_bf16_async out",
            out,
            row_values * std::mem::size_of::<f32>(),
        )?;

        let kernel_fn: Symbol<CudaLayerNormAffineF32Bf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_layernorm_affine_f32_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<u16>() as *const u16,
                bias.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_layernorm_affine_f32_bf16_async", status)
    }

    pub fn cuda_layernorm_affine_bf16(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_layernorm_affine_bf16", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_layernorm_affine_bf16 x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_layernorm_affine_bf16 x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_bf16 weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_bf16 bias",
            bias,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_layernorm_affine_bf16 out", out, row_values)?;

        let kernel_fn: Symbol<CudaLayerNormAffineBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_layernorm_affine_bf16")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
            )
        };
        self.status_to_result("ds41rt_cuda_layernorm_affine_bf16", status)
    }

    pub unsafe fn cuda_layernorm_affine_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: i32,
        hidden: i32,
        eps: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_rows("ds41rt_cuda_layernorm_affine_bf16_async", rows, hidden)?;
        let row_values = checked_row_values(
            "ds41rt_cuda_layernorm_affine_bf16_async x",
            rows as usize,
            hidden as usize,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_layernorm_affine_bf16_async x", x, row_values)?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_bf16_async weight",
            weight,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_bf16_async bias",
            bias,
            hidden as usize,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_layernorm_affine_bf16_async out",
            out,
            row_values,
        )?;

        let kernel_fn: Symbol<CudaLayerNormAffineBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_layernorm_affine_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                eps,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_layernorm_affine_bf16_async", status)
    }

    pub fn cuda_silu_gated_mlp_f32(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        hidden: i32,
        intermediate: i32,
    ) -> Result<()> {
        validate_positive_dim("ds41rt_cuda_silu_gated_mlp_f32 hidden", hidden)?;
        validate_positive_dim("ds41rt_cuda_silu_gated_mlp_f32 intermediate", intermediate)?;
        let hidden = hidden as usize;
        let intermediate = intermediate as usize;
        validate_device_buffer_bytes("ds41rt_cuda_silu_gated_mlp_f32 x", x, hidden * 4)?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_silu_gated_mlp_f32 gate_weight",
            gate_weight,
            intermediate * hidden * 4,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_silu_gated_mlp_f32 up_weight",
            up_weight,
            intermediate * hidden * 4,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_silu_gated_mlp_f32 down_weight",
            down_weight,
            hidden * intermediate * 4,
        )?;
        validate_device_buffer_bytes("ds41rt_cuda_silu_gated_mlp_f32 out", out, hidden * 4)?;

        let kernel_fn: Symbol<CudaSiluGatedMlpF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_gated_mlp_f32")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                gate_weight.ptr.cast::<f32>() as *const f32,
                up_weight.ptr.cast::<f32>() as *const f32,
                down_weight.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                hidden as i32,
                intermediate as i32,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_f32", status)
    }

    pub fn cuda_silu_gated_mlp_rows_f32(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        validate_mlp_rows_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_f32",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_gated_mlp_rows_f32")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                gate_weight.ptr.cast::<f32>() as *const f32,
                up_weight.ptr.cast::<f32>() as *const f32,
                down_weight.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                intermediate,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_rows_f32", status)
    }

    pub unsafe fn cuda_silu_gated_mlp_rows_f32_async(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mlp_rows_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_f32_async",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_gated_mlp_rows_f32_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<f32>() as *const f32,
                gate_weight.ptr.cast::<f32>() as *const f32,
                up_weight.ptr.cast::<f32>() as *const f32,
                down_weight.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                hidden,
                intermediate,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_rows_f32_async", status)
    }

    pub fn cuda_silu_gated_mlp_rows_bf16(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        validate_mlp_rows_bf16_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_rows_bf16", status)
    }

    pub unsafe fn cuda_silu_gated_mlp_rows_bf16_async(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mlp_rows_bf16_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_async",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_rows_bf16_async", status)
    }

    pub unsafe fn cuda_silu_mul_bf16_async(
        &self,
        gate_up: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        intermediate: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let gate_up_bytes = rows
            .checked_mul(intermediate)
            .and_then(|values| values.checked_mul(2))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("BF16 SiLU-mul input byte count overflow")?;
        let output_bytes = rows
            .checked_mul(intermediate)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("BF16 SiLU-mul output byte count overflow")?;
        anyhow::ensure!(
            rows > 0
                && intermediate > 0
                && !gate_up.ptr.is_null()
                && gate_up.bytes >= gate_up_bytes
                && !out.ptr.is_null()
                && out.bytes >= output_bytes,
            "BF16 SiLU-mul buffers are invalid: input={} (need {gate_up_bytes}) output={} (need {output_bytes}) rows={rows} intermediate={intermediate}",
            gate_up.bytes,
            out.bytes,
        );
        let kernel_fn: Symbol<CudaSiluMulBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_silu_mul_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                gate_up.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                intermediate,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_mul_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_silu_gated_mlp_rows_bf16_down_stride(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
    ) -> Result<()> {
        validate_mlp_rows_bf16_down_stride_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
            down_stride,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16DownStrideFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride")?
        };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
                down_stride,
            )
        };
        self.status_to_result("ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_silu_gated_mlp_rows_bf16_down_stride_async(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mlp_rows_bf16_down_stride_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_async",
            x,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden,
            intermediate,
            down_stride,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16DownStrideAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_async")?
        };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
                down_stride,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
    ) -> Result<()> {
        validate_mlp_rows_bf16_down_stride_staged_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged",
            x,
            gate_weight,
            up_weight,
            down_weight,
            activation_workspace,
            out,
            rows,
            hidden,
            intermediate,
            down_stride,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16DownStrideStagedFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged")?
        };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                activation_workspace.ptr.cast::<f32>(),
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
                down_stride,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
        &self,
        x: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mlp_rows_bf16_down_stride_staged_buffers(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async",
            x,
            gate_weight,
            up_weight,
            down_weight,
            activation_workspace,
            out,
            rows,
            hidden,
            intermediate,
            down_stride,
        )?;

        let kernel_fn: Symbol<CudaSiluGatedMlpRowsBf16DownStrideStagedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async")?
        };
        let status = unsafe {
            kernel_fn(
                x.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                activation_workspace.ptr.cast::<f32>(),
                out.ptr.cast::<u16>(),
                rows,
                hidden,
                intermediate,
                down_stride,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async",
            status,
        )
    }

    pub fn cuda_b12x_spark_aot_available(&self) -> Result<bool> {
        let available_fn: Symbol<CudaB12xSparkAotAvailableFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_spark_aot_available")? };
        let mut available = 0;
        let status = unsafe { available_fn(&mut available) };
        self.status_to_result("ds41rt_cuda_b12x_spark_aot_available", status)?;
        Ok(available != 0)
    }

    pub fn cuda_b12x_spark_aot_init(&self) -> Result<()> {
        let init_fn: Symbol<CudaB12xSparkAotInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_spark_aot_init")? };
        let status = unsafe { init_fn() };
        self.status_to_result("ds41rt_cuda_b12x_spark_aot_init", status)
    }

    pub fn cuda_ds4_flash_spark_aot_available(&self) -> Result<bool> {
        let available_fn: Symbol<CudaDs4FlashSparkAotAvailableFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_flash_spark_aot_available")? };
        let mut available = 0;
        let status = unsafe { available_fn(&mut available) };
        self.status_to_result("ds41rt_cuda_ds4_flash_spark_aot_available", status)?;
        Ok(available != 0)
    }

    pub fn cuda_ds4_flash_spark_aot_init(&self) -> Result<()> {
        let init_fn: Symbol<CudaDs4FlashSparkAotInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_flash_spark_aot_init")? };
        let status = unsafe { init_fn() };
        self.status_to_result("ds41rt_cuda_ds4_flash_spark_aot_init", status)
    }

    pub fn cuda_ds4_pro_spark_aot_available(&self) -> Result<bool> {
        let available_fn: Symbol<CudaDs4ProSparkAotAvailableFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_pro_spark_aot_available")? };
        let mut available = 0;
        let status = unsafe { available_fn(&mut available) };
        self.status_to_result("ds41rt_cuda_ds4_pro_spark_aot_available", status)?;
        Ok(available != 0)
    }

    pub fn cuda_ds4_pro_spark_aot_init(&self) -> Result<()> {
        let init_fn: Symbol<CudaDs4ProSparkAotInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_pro_spark_aot_init")? };
        let status = unsafe { init_fn() };
        self.status_to_result("ds41rt_cuda_ds4_pro_spark_aot_init", status)
    }

    pub unsafe fn cuda_ds4_flash_w4a16_pack_weight_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashW4a16PackWeightAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_w4a16_pack_weight_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_w4a16_pack_weight_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_flash_w4a16_pack_weight_strided_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        source_size_k: usize,
        source_start_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashW4a16PackWeightStridedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_w4a16_pack_weight_strided_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                source_size_k,
                source_start_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_w4a16_pack_weight_strided_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashW4a16PackE8m0ScaleAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        source_size_k: usize,
        source_start_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashW4a16PackE8m0ScaleStridedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                source_size_k,
                source_start_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_fp8_pack_block_scale_mma_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_n: usize,
        size_k: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashFp8PackBlockScaleMmaAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async")?
        };
        let status = unsafe { kernel_fn(source, destination, size_n, size_k, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_shared_expert_fp8_bf16_async(
        &self,
        buffers: &Ds41rtDs4FlashSharedExpertFp8Buffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSharedExpertFp8Bf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async", status)
    }

    pub unsafe fn cuda_ds4_pro_shared_expert_fp8_bf16_async(
        &self,
        buffers: &Ds41rtDs4FlashSharedExpertFp8Buffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4ProSharedExpertFp8Bf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_pro_shared_expert_fp8_bf16_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_pro_shared_expert_fp8_bf16_async", status)
    }

    pub unsafe fn cuda_ds4_pro_shared_expert_fp8_prefill_bf16_async(
        &self,
        buffers: &Ds41rtDs4ProSharedExpertFp8PrefillBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4ProSharedExpertFp8PrefillBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_pro_shared_expert_fp8_prefill_bf16_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_pro_shared_expert_fp8_prefill_bf16_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkW4a16MoeBuffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkW4a16DecodeM1Bf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkW4a16MoeBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkW4a16PrefillTopk6Bf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_spark_exl3_k2_decode_m1_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkExl3K2MoeBuffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkExl3K2DecodeM1AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_exl3_k2_decode_m1_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_flash_spark_exl3_k2_decode_m1_async", status)
    }

    pub unsafe fn cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkExl3K2MoeBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkExl3K2PrefillTopk6AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_spark_exl3_k3_decode_m1_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkExl3K2MoeBuffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkExl3K3DecodeM1AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_exl3_k3_decode_m1_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_flash_spark_exl3_k3_decode_m1_async", status)
    }

    pub unsafe fn cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkExl3K2MoeBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkExl3K3PrefillTopk6AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_spark_exl3_mixed_k2_k3_async(
        &self,
        buffers: &Ds41rtDs4FlashSparkExl3MixedMoeBuffers,
        tier0_slots: usize,
        tier1_slots: usize,
        tier0_gate_experts: usize,
        tier1_gate_experts: usize,
        tier0_up_experts: usize,
        tier1_up_experts: usize,
        tier0_down_experts: usize,
        tier1_down_experts: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashSparkExl3MixedK2K3AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_spark_exl3_mixed_k2_k3_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                tier0_slots,
                tier1_slots,
                tier0_gate_experts,
                tier1_gate_experts,
                tier0_up_experts,
                tier1_up_experts,
                tier0_down_experts,
                tier1_down_experts,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_spark_exl3_mixed_k2_k3_async", status)
    }

    pub unsafe fn cuda_ds4_pro_spark_exl3_k2_decode_m1_async(
        &self,
        buffers: &Ds41rtDs4ProSparkExl3K2MoeBuffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4ProSparkExl3K2DecodeM1AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_pro_spark_exl3_k2_decode_m1_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_pro_spark_exl3_k2_decode_m1_async", status)
    }

    pub unsafe fn cuda_ds4_pro_spark_exl3_k2_prefill_topk6_async(
        &self,
        buffers: &Ds41rtDs4ProSparkExl3K2MoeBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4ProSparkExl3K2PrefillTopk6AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_pro_spark_exl3_k2_prefill_topk6_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_ds4_pro_spark_exl3_k2_prefill_topk6_async",
            status,
        )
    }

    pub unsafe fn cuda_ds4_flash_pack_topk6_routes_async(
        &self,
        buffers: &Ds41rtDs4FlashRoutePackBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaDs4FlashPackTopk6RoutesAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_pack_topk6_routes_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result("ds41rt_cuda_ds4_flash_pack_topk6_routes_async", status)
    }

    pub unsafe fn cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
        &self,
        input: Ds41rtDeviceBuffer,
        payload: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xQuantizeBf16Nvfp4RowPayloadAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async")?
        };
        let status = unsafe { kernel_fn(input, payload, rows, hidden_dim, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_w4a16_pack_weight_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xW4a16PackWeightAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_w4a16_pack_weight_async")? };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_w4a16_pack_weight_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_b12x_w4a16_pack_weight_strided_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        source_size_k: usize,
        source_start_k: usize,
        size_n: usize,
        row_rotation: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xW4a16PackWeightStridedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_w4a16_pack_weight_strided_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                source_size_k,
                source_start_k,
                size_n,
                row_rotation,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_w4a16_pack_weight_strided_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_b12x_w4a16_pack_scale_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        size_n: usize,
        row_rotation: usize,
        scale_factor: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xW4a16PackScaleAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_w4a16_pack_scale_async")? };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                size_n,
                row_rotation,
                scale_factor,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_w4a16_pack_scale_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_b12x_w4a16_pack_scale_strided_async(
        &self,
        source: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        size_k: usize,
        source_size_k: usize,
        source_start_k: usize,
        size_n: usize,
        row_rotation: usize,
        scale_factor: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xW4a16PackScaleStridedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_w4a16_pack_scale_strided_async")?
        };
        let status = unsafe {
            kernel_fn(
                source,
                destination,
                size_k,
                source_size_k,
                source_start_k,
                size_n,
                row_rotation,
                scale_factor,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_w4a16_pack_scale_strided_async", status)
    }

    pub unsafe fn cuda_quantize_bf16_weight_nvfp4_async(
        &self,
        input: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        rows: usize,
        cols: usize,
        global_scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let input_bytes = rows
            .checked_mul(cols)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("BF16 weight quantization input byte count overflow")?;
        let packed_bytes = rows
            .checked_mul(cols / 2)
            .context("BF16 weight quantization packed byte count overflow")?;
        let scale_bytes = rows
            .checked_mul(cols / 16)
            .context("BF16 weight quantization scale byte count overflow")?;
        anyhow::ensure!(
            rows > 0
                && cols > 0
                && cols % 16 == 0
                && global_scale.is_finite()
                && global_scale > 0.0
                && input.bytes >= input_bytes
                && packed.bytes >= packed_bytes
                && scales.bytes >= scale_bytes,
            "invalid BF16 weight NVFP4 quantization buffers or geometry"
        );
        let kernel_fn: Symbol<CudaQuantizeBf16WeightNvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_quantize_bf16_weight_nvfp4_async")?
        };
        let status =
            unsafe { kernel_fn(input, packed, scales, rows, cols, global_scale, cuda_stream) };
        self.status_to_result("ds41rt_cuda_quantize_bf16_weight_nvfp4_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_b12x_gather_nvfp4_rows_bf16_async(
        &self,
        payload: Ds41rtDeviceBuffer,
        source_rows: usize,
        source_row_stride_bytes: usize,
        row_indices: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xGatherNvfp4RowsBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_gather_nvfp4_rows_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                payload,
                source_rows,
                source_row_stride_bytes,
                row_indices,
                output,
                rows,
                hidden_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_gather_nvfp4_rows_bf16_async", status)
    }

    pub unsafe fn cuda_b12x_dequantize_nvfp4_rows_bf16_async(
        &self,
        payload: Ds41rtDeviceBuffer,
        row_stride_bytes: usize,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xDequantizeNvfp4RowsBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_dequantize_nvfp4_rows_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                payload,
                row_stride_bytes,
                output,
                rows,
                hidden_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_dequantize_nvfp4_rows_bf16_async", status)
    }

    pub unsafe fn cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        topk_ids: Ds41rtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16DecodeM1Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                topk_ids,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async", status)
    }

    pub unsafe fn cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        topk_ids: Ds41rtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16DecodeM1Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                topk_ids,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        topk_ids: Ds41rtDeviceBuffer,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16M1ParityM2To8Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                topk_ids,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16M1ParityGroupedM2To8Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16M1ParityGroupedM2To8Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16PrefillTopk8Nvfp4AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                rows,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        input_payload: Ds41rtDeviceBuffer,
        input_payload_stride_bytes: usize,
        rows: usize,
        output_fp8: Ds41rtDeviceBuffer,
        output_fp8_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16PrefillTopk8Nvfp4Fp8AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                input_payload,
                input_payload_stride_bytes,
                rows,
                output_fp8,
                output_fp8_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_spark_w4a16_top1_async(
        &self,
        buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
        rows: usize,
        capacity_rows: usize,
        expert_id: u32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xSparkW4a16Top1AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_spark_w4a16_top1_async")? };
        let status = unsafe { kernel_fn(buffers, rows, capacity_rows, expert_id, cuda_stream) };
        self.status_to_result("ds41rt_cuda_b12x_spark_w4a16_top1_async", status)
    }

    pub fn cuda_b12x_coordinator_aot_available(&self) -> Result<bool> {
        let available_fn: Symbol<CudaB12xCoordinatorAotAvailableFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_coordinator_aot_available")? };
        let mut available = 0;
        let status = unsafe { available_fn(&mut available) };
        self.status_to_result("ds41rt_cuda_b12x_coordinator_aot_available", status)?;
        Ok(available != 0)
    }

    pub fn cuda_b12x_coordinator_aot_init(&self) -> Result<()> {
        let init_fn: Symbol<CudaB12xCoordinatorAotInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_b12x_coordinator_aot_init")? };
        let status = unsafe { init_fn() };
        self.status_to_result("ds41rt_cuda_b12x_coordinator_aot_init", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
        &self,
        input_bf16: Ds41rtDeviceBuffer,
        payload_scratch: Ds41rtDeviceBuffer,
        packed_weight: Ds41rtDeviceBuffer,
        packed_scale: Ds41rtDeviceBuffer,
        global_scale: Ds41rtDeviceBuffer,
        size_k: usize,
        size_n: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16QuantizePackWeightAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async")?
        };
        let status = unsafe {
            kernel_fn(
                input_bf16,
                payload_scratch,
                packed_weight,
                packed_scale,
                global_scale,
                size_k,
                size_n,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async",
            status,
        )
    }

    pub unsafe fn cuda_b12x_coordinator_w4a16_q_b_m1_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_q_b_m1_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result("ds41rt_cuda_b12x_coordinator_w4a16_q_b_m1_async", status)
    }

    pub unsafe fn cuda_b12x_coordinator_w4a16_q_b_m8_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        active_rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersRowsAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_q_b_m8_async")?
        };
        let status = unsafe { kernel_fn(buffers, active_rows, cuda_stream) };
        self.status_to_result("ds41rt_cuda_b12x_coordinator_w4a16_q_b_m8_async", status)
    }

    pub unsafe fn cuda_b12x_coordinator_w4a16_o_proj_m1_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_o_proj_m1_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result("ds41rt_cuda_b12x_coordinator_w4a16_o_proj_m1_async", status)
    }

    pub unsafe fn cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_async")?
        };
        let status = unsafe { kernel_fn(buffers, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        gate_scale: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        up_scale: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        down_scale: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        accumulator: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        intermediate: usize,
        output_dim: usize,
        down_weight_row_stride_bytes: usize,
        down_scale_row_stride_bytes: usize,
        gate_scale_2: f32,
        up_scale_2: f32,
        down_scale_2: f32,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_grouped_staged_accumulate_f32_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32",
            hidden,
            row_indices,
            route_weights,
            gate_weight,
            gate_scale,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            activation_workspace,
            accumulator,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            intermediate,
            output_dim,
            down_weight_row_stride_bytes,
            down_scale_row_stride_bytes,
            gate_scale_2,
            up_scale_2,
            down_scale_2,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16GroupedStagedAccumulateF32Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                gate_weight.ptr.cast::<u8>() as *const u8,
                gate_scale.ptr.cast::<u8>() as *const u8,
                up_weight.ptr.cast::<u8>() as *const u8,
                up_scale.ptr.cast::<u8>() as *const u8,
                down_weight.ptr.cast::<u8>() as *const u8,
                down_scale.ptr.cast::<u8>() as *const u8,
                activation_workspace.ptr.cast::<f32>(),
                accumulator.ptr.cast::<f32>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                intermediate,
                output_dim,
                down_weight_row_stride_bytes,
                down_scale_row_stride_bytes,
                gate_scale_2,
                up_scale_2,
                down_scale_2,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        gate_scale: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        up_scale: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        down_scale: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        accumulator: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        intermediate: usize,
        output_dim: usize,
        down_weight_row_stride_bytes: usize,
        down_scale_row_stride_bytes: usize,
        gate_scale_2: f32,
        up_scale_2: f32,
        down_scale_2: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_grouped_staged_accumulate_f32_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async",
            hidden,
            row_indices,
            route_weights,
            gate_weight,
            gate_scale,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            activation_workspace,
            accumulator,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            intermediate,
            output_dim,
            down_weight_row_stride_bytes,
            down_scale_row_stride_bytes,
            gate_scale_2,
            up_scale_2,
            down_scale_2,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16GroupedStagedAccumulateF32AsyncFn> = unsafe {
            self.lib.get(
                b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async",
            )?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                gate_weight.ptr.cast::<u8>() as *const u8,
                gate_scale.ptr.cast::<u8>() as *const u8,
                up_weight.ptr.cast::<u8>() as *const u8,
                up_scale.ptr.cast::<u8>() as *const u8,
                down_weight.ptr.cast::<u8>() as *const u8,
                down_scale.ptr.cast::<u8>() as *const u8,
                activation_workspace.ptr.cast::<f32>(),
                accumulator.ptr.cast::<f32>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                intermediate,
                output_dim,
                down_weight_row_stride_bytes,
                down_scale_row_stride_bytes,
                gate_scale_2,
                up_scale_2,
                down_scale_2,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        route_metadata: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        accumulator: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_batched_staged_accumulate_f32_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32",
            hidden,
            row_indices,
            route_weights,
            route_metadata,
            activation_workspace,
            accumulator,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            max_intermediate,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedAccumulateF32Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                route_metadata.ptr.cast::<Ds41rtNvfp4RouteBatchedMetadata>()
                    as *const Ds41rtNvfp4RouteBatchedMetadata,
                activation_workspace.ptr.cast::<f32>(),
                accumulator.ptr.cast::<f32>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                max_intermediate,
                output_dim,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        route_metadata: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        accumulator: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_batched_staged_accumulate_f32_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async",
            hidden,
            row_indices,
            route_weights,
            route_metadata,
            activation_workspace,
            accumulator,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            max_intermediate,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedAccumulateF32AsyncFn> = unsafe {
            self.lib.get(
                b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async",
            )?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                route_metadata.ptr.cast::<Ds41rtNvfp4RouteBatchedMetadata>()
                    as *const Ds41rtNvfp4RouteBatchedMetadata,
                activation_workspace.ptr.cast::<f32>(),
                accumulator.ptr.cast::<f32>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                max_intermediate,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        route_metadata: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_batched_staged_single_row_bf16_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16",
            hidden,
            row_indices,
            route_weights,
            route_metadata,
            activation_workspace,
            out,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            max_intermediate,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedSingleRowBf16Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                route_metadata.ptr.cast::<Ds41rtNvfp4RouteBatchedMetadata>()
                    as *const Ds41rtNvfp4RouteBatchedMetadata,
                activation_workspace.ptr.cast::<f32>(),
                out.ptr.cast::<u16>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                max_intermediate,
                output_dim,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        route_weights: Ds41rtDeviceBuffer,
        route_metadata: Ds41rtDeviceBuffer,
        activation_workspace: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        routes: usize,
        hidden_dim: usize,
        hidden_row_stride: usize,
        max_intermediate: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_nvfp4_route_bf16_batched_staged_single_row_bf16_buffers(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async",
            hidden,
            row_indices,
            route_weights,
            route_metadata,
            activation_workspace,
            out,
            rows,
            routes,
            hidden_dim,
            hidden_row_stride,
            max_intermediate,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaNvfp4SiluGatedMlpRouteBf16BatchedStagedSingleRowBf16AsyncFn> = unsafe {
            self.lib.get(
                b"ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async",
            )?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                route_weights.ptr.cast::<f32>() as *const f32,
                route_metadata.ptr.cast::<Ds41rtNvfp4RouteBatchedMetadata>()
                    as *const Ds41rtNvfp4RouteBatchedMetadata,
                activation_workspace.ptr.cast::<f32>(),
                out.ptr.cast::<u16>(),
                rows,
                routes,
                hidden_dim,
                hidden_row_stride,
                max_intermediate,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async",
            status,
        )
    }

    pub fn cuda_residual_add_f32(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_f32_buffer_values("ds41rt_cuda_residual_add_f32 residual", residual, count)?;
        validate_f32_buffer_values("ds41rt_cuda_residual_add_f32 delta", delta, count)?;
        validate_f32_buffer_values("ds41rt_cuda_residual_add_f32 out", out, count)?;

        let kernel_fn: Symbol<CudaResidualAddF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_residual_add_f32")? };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<f32>() as *const f32,
                delta.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_f32", status)
    }

    pub unsafe fn cuda_residual_add_f32_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_buffer_values(
            "ds41rt_cuda_residual_add_f32_async residual",
            residual,
            count,
        )?;
        validate_f32_buffer_values("ds41rt_cuda_residual_add_f32_async delta", delta, count)?;
        validate_f32_buffer_values("ds41rt_cuda_residual_add_f32_async out", out, count)?;

        let kernel_fn: Symbol<CudaResidualAddF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_residual_add_f32_async")? };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<f32>() as *const f32,
                delta.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_f32_async", status)
    }

    pub fn cuda_residual_add_bf16(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_u16_buffer_values("ds41rt_cuda_residual_add_bf16 residual", residual, count)?;
        validate_u16_buffer_values("ds41rt_cuda_residual_add_bf16 delta", delta, count)?;
        validate_u16_buffer_values("ds41rt_cuda_residual_add_bf16 out", out, count)?;

        let kernel_fn: Symbol<CudaResidualAddBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_residual_add_bf16")? };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                delta.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_bf16", status)
    }

    pub unsafe fn cuda_residual_add_bf16_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_bf16_async residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_residual_add_bf16_async delta", delta, count)?;
        validate_u16_buffer_values("ds41rt_cuda_residual_add_bf16_async out", out, count)?;

        let kernel_fn: Symbol<CudaResidualAddBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_residual_add_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                delta.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_bf16_async", status)
    }

    pub fn cuda_residual_add_f32_delta_bf16(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_f32_delta_bf16 residual",
            residual,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_residual_add_f32_delta_bf16 delta_f32",
            delta_f32,
            count,
        )?;
        validate_u16_buffer_values("ds41rt_cuda_residual_add_f32_delta_bf16 out", out, count)?;

        let kernel_fn: Symbol<CudaResidualAddF32DeltaBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_residual_add_f32_delta_bf16")? };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                delta_f32.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<u16>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_f32_delta_bf16", status)
    }

    pub unsafe fn cuda_residual_add_f32_delta_bf16_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_f32_delta_bf16_async residual",
            residual,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_residual_add_f32_delta_bf16_async delta_f32",
            delta_f32,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_f32_delta_bf16_async out",
            out,
            count,
        )?;

        let kernel_fn: Symbol<CudaResidualAddF32DeltaBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_residual_add_f32_delta_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                delta_f32.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_f32_delta_bf16_async", status)
    }

    pub fn cuda_residual_add_shared_f32_delta_bf16(
        &self,
        residual: Ds41rtDeviceBuffer,
        shared_delta: Ds41rtDeviceBuffer,
        routed_delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16 residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16 shared_delta",
            shared_delta,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16 routed_delta_f32",
            routed_delta_f32,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16 out",
            out,
            count,
        )?;

        let kernel_fn: Symbol<CudaResidualAddSharedF32DeltaBf16Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_residual_add_shared_f32_delta_bf16")?
        };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                shared_delta.ptr.cast::<u16>() as *const u16,
                routed_delta_f32.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<u16>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_residual_add_shared_f32_delta_bf16", status)
    }

    pub unsafe fn cuda_residual_add_shared_f32_delta_bf16_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        shared_delta: Ds41rtDeviceBuffer,
        routed_delta_f32: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16_async residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16_async shared_delta",
            shared_delta,
            count,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16_async routed_delta_f32",
            routed_delta_f32,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16_async out",
            out,
            count,
        )?;

        let kernel_fn: Symbol<CudaResidualAddSharedF32DeltaBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_residual_add_shared_f32_delta_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                shared_delta.ptr.cast::<u16>() as *const u16,
                routed_delta_f32.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_residual_add_shared_f32_delta_bf16_async",
            status,
        )
    }

    pub unsafe fn cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        shared_delta: Ds41rtDeviceBuffer,
        routed_delta_fp8: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async residual",
            residual,
            count,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async shared_delta",
            shared_delta,
            count,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async routed_delta_fp8",
            routed_delta_fp8,
            count
                .checked_add(std::mem::size_of::<f32>())
                .context(
                    "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async routed row byte count overflows usize",
                )?,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async out",
            out,
            count,
        )?;

        let kernel_fn: Symbol<CudaResidualAddSharedFp8E4m3RowScaledBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                shared_delta.ptr.cast::<u16>() as *const u16,
                routed_delta_fp8.ptr.cast::<u8>() as *const u8,
                out.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_fp8_decode_combine_residual_async(
        &self,
        residual: Ds41rtDeviceBuffer,
        shared_delta: Ds41rtDeviceBuffer,
        partials: Ds41rtDeviceBuffer,
        partial_row_stride_bytes: usize,
        output: Ds41rtDeviceBuffer,
        partial_rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            (1..=4).contains(&partial_rows),
            "ds41rt_cuda_fp8_decode_combine_residual_async partial rows {partial_rows} are outside 1..=4"
        );
        let minimum_row_stride = row_width
            .checked_add(std::mem::size_of::<f32>())
            .context("ds41rt_cuda_fp8_decode_combine_residual_async row stride overflows usize")?;
        anyhow::ensure!(
            partial_row_stride_bytes >= minimum_row_stride,
            "ds41rt_cuda_fp8_decode_combine_residual_async row stride {partial_row_stride_bytes} is smaller than {minimum_row_stride}"
        );
        validate_u16_buffer_values(
            "ds41rt_cuda_fp8_decode_combine_residual_async residual",
            residual,
            row_width,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_fp8_decode_combine_residual_async shared_delta",
            shared_delta,
            row_width,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_fp8_decode_combine_residual_async partials",
            partials,
            partial_rows.checked_mul(partial_row_stride_bytes).context(
                "ds41rt_cuda_fp8_decode_combine_residual_async partial bytes overflow usize",
            )?,
        )?;
        validate_u16_buffer_values(
            "ds41rt_cuda_fp8_decode_combine_residual_async output",
            output,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaFp8DecodeCombineResidualAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_fp8_decode_combine_residual_async")?
        };
        let status = unsafe {
            kernel_fn(
                residual.ptr.cast::<u16>() as *const u16,
                shared_delta.ptr.cast::<u16>() as *const u16,
                partials.ptr.cast::<u8>() as *const u8,
                partial_row_stride_bytes,
                output.ptr.cast::<u16>(),
                partial_rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_fp8_decode_combine_residual_async", status)
    }

    pub fn cuda_scheduler_mlp_delta_bf16(
        &self,
        hidden: Ds41rtDeviceBuffer,
        gate_weight: Ds41rtDeviceBuffer,
        up_weight: Ds41rtDeviceBuffer,
        down_weight: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        validate_scheduler_mlp_delta_bf16_buffers(
            "ds41rt_cuda_scheduler_mlp_delta_bf16",
            hidden,
            gate_weight,
            up_weight,
            down_weight,
            out,
            rows,
            hidden_dim,
        )?;

        let kernel_fn: Symbol<CudaSchedulerMlpDeltaBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_scheduler_mlp_delta_bf16")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                gate_weight.ptr.cast::<u16>() as *const u16,
                up_weight.ptr.cast::<u16>() as *const u16,
                down_weight.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                hidden_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_scheduler_mlp_delta_bf16", status)
    }

    pub fn cuda_summarize_bf16(
        &self,
        input: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<Ds41rtBf16Summary> {
        validate_u16_buffer_values("ds41rt_cuda_summarize_bf16 input", input, count)?;

        let kernel_fn: Symbol<CudaSummarizeBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_summarize_bf16")? };
        let mut summary = Ds41rtBf16Summary::default();
        let status =
            unsafe { kernel_fn(input.ptr.cast::<u16>() as *const u16, count, &mut summary) };
        self.status_to_result("ds41rt_cuda_summarize_bf16", status)?;
        Ok(summary)
    }

    pub unsafe fn cuda_summarize_bf16_async(
        &self,
        input: Ds41rtDeviceBuffer,
        count: usize,
        out_device: Ds41rtDeviceBuffer,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_u16_buffer_values("ds41rt_cuda_summarize_bf16_async input", input, count)?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_summarize_bf16_async out_device",
            out_device,
            std::mem::size_of::<Ds41rtBf16Summary>(),
        )?;

        let kernel_fn: Symbol<CudaSummarizeBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_summarize_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                count,
                out_device.ptr.cast::<Ds41rtBf16Summary>(),
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_summarize_bf16_async", status)
    }

    pub fn cuda_zero_f32(&self, dst: Ds41rtDeviceBuffer, count: usize) -> Result<()> {
        validate_f32_buffer_values("ds41rt_cuda_zero_f32 dst", dst, count)?;

        let kernel_fn: Symbol<CudaZeroF32Fn> = unsafe { self.lib.get(b"ds41rt_cuda_zero_f32")? };
        let status = unsafe { kernel_fn(dst.ptr.cast::<f32>(), count) };
        self.status_to_result("ds41rt_cuda_zero_f32", status)
    }

    pub unsafe fn cuda_zero_f32_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_buffer_values("ds41rt_cuda_zero_f32_async dst", dst, count)?;

        let kernel_fn: Symbol<CudaZeroF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_zero_f32_async")? };
        let status = unsafe { kernel_fn(dst.ptr.cast::<f32>(), count, cuda_stream) };
        self.status_to_result("ds41rt_cuda_zero_f32_async", status)
    }

    pub fn cuda_accumulate_bf16_to_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_bf16_to_f32_accumulation_buffers(
            "ds41rt_cuda_accumulate_bf16_to_f32",
            src,
            dst,
            count,
        )?;
        let kernel_fn: Symbol<CudaAccumulateBf16ToF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_accumulate_bf16_to_f32")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                dst.ptr.cast::<f32>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_accumulate_bf16_to_f32", status)
    }

    pub unsafe fn cuda_accumulate_bf16_to_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_bf16_to_f32_accumulation_buffers(
            "ds41rt_cuda_accumulate_bf16_to_f32_async",
            src,
            dst,
            count,
        )?;
        let kernel_fn: Symbol<CudaAccumulateBf16ToF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_accumulate_bf16_to_f32_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                dst.ptr.cast::<f32>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_accumulate_bf16_to_f32_async", status)
    }

    pub fn cuda_zero_bytes(&self, dst: Ds41rtDeviceBuffer, bytes: usize) -> Result<()> {
        validate_device_buffer_bytes("ds41rt_cuda_zero_bytes dst", dst, bytes)?;

        let kernel_fn: Symbol<CudaZeroBytesFn> = unsafe { self.lib.get(b"ds41rt_cuda_zero_bytes")? };
        let status = unsafe { kernel_fn(dst.ptr, bytes) };
        self.status_to_result("ds41rt_cuda_zero_bytes", status)
    }

    pub unsafe fn cuda_zero_bytes_async(
        &self,
        dst: Ds41rtDeviceBuffer,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_device_buffer_bytes("ds41rt_cuda_zero_bytes_async dst", dst, bytes)?;

        let kernel_fn: Symbol<CudaZeroBytesAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_zero_bytes_async")? };
        let status = unsafe { kernel_fn(dst.ptr, bytes, cuda_stream) };
        self.status_to_result("ds41rt_cuda_zero_bytes_async", status)
    }

    pub fn cuda_f32_to_bf16(
        &self,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        validate_f32_to_bf16_buffers("ds41rt_cuda_f32_to_bf16", src, dst, count)?;

        let kernel_fn: Symbol<CudaF32ToBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_f32_to_bf16")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<u16>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_f32_to_bf16", status)
    }

    pub unsafe fn cuda_f32_to_bf16_async(
        &self,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_f32_to_bf16_buffers("ds41rt_cuda_f32_to_bf16_async", src, dst, count)?;

        let kernel_fn: Symbol<CudaF32ToBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_f32_to_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<u16>(),
                count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_f32_to_bf16_async", status)
    }

    pub fn cuda_gather_rows_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_gather_buffers(
            "ds41rt_cuda_gather_rows_f32",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaGatherRowsF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_gather_rows_f32")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_f32", status)
    }

    pub unsafe fn cuda_gather_rows_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_buffers(
            "ds41rt_cuda_gather_rows_f32_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaGatherRowsF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_gather_rows_f32_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_f32_async", status)
    }

    pub fn cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
    ) -> Result<()> {
        validate_row_gather_f32_to_fp8_e4m3_row_scaled_buffers(
            "ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToFp8E4m3RowScaledFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_f32_to_fp8_e4m3_row_scaled_buffers(
            "ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToFp8E4m3RowScaledAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
        &self,
        src: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_bf16_rows_to_fp8_e4m3_row_scaled_buffers(
            "ds41rt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async",
            src,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaBf16RowsToFp8E4m3RowScaledAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_combine_fp8_e4m3_row_scaled_to_fp8_async(
        &self,
        local: Ds41rtDeviceBuffer,
        peers: Ds41rtDeviceBuffer,
        peer_payload_stride_bytes: usize,
        peer_count: usize,
        peer_row_stride_bytes: usize,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_combine_fp8_e4m3_row_scaled_buffers(
            "ds41rt_cuda_combine_fp8_e4m3_row_scaled_to_fp8_async",
            local,
            peers,
            peer_payload_stride_bytes,
            peer_count,
            peer_row_stride_bytes,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaCombineFp8E4m3RowScaledToFp8AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_combine_fp8_e4m3_row_scaled_to_fp8_async")?
        };
        let status = unsafe {
            kernel_fn(
                local.ptr.cast::<f32>() as *const f32,
                peers.ptr.cast::<u8>() as *const u8,
                peer_payload_stride_bytes,
                peer_count,
                peer_row_stride_bytes,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_combine_fp8_e4m3_row_scaled_to_fp8_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async(
        &self,
        local: Ds41rtDeviceBuffer,
        peers: Ds41rtDeviceBuffer,
        peer_payload_stride_bytes: usize,
        peer_count: usize,
        peer_row_stride_bytes: usize,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_combine_bf16_fp8_e4m3_row_scaled_buffers(
            "ds41rt_cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async",
            local,
            peers,
            peer_payload_stride_bytes,
            peer_count,
            peer_row_stride_bytes,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaCombineBf16Fp8E4m3RowScaledToFp8AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async")?
        };
        let status = unsafe {
            kernel_fn(
                local.ptr.cast::<u16>() as *const u16,
                peers.ptr.cast::<u8>() as *const u8,
                peer_payload_stride_bytes,
                peer_count,
                peer_row_stride_bytes,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
    ) -> Result<()> {
        validate_row_gather_f32_to_nvfp4_e2m1_fp8_e4m3_buffers(
            "ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        dst_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_f32_to_nvfp4_e2m1_fp8_e4m3_buffers(
            "ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
            dst_row_stride_bytes,
        )?;
        let kernel_fn: Symbol<CudaGatherRowsF32ToNvfp4E2m1Fp8E4m3AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u8>(),
                rows,
                row_width,
                dst_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async",
            status,
        )
    }

    pub fn cuda_gather_rows_bf16(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_gather_bf16_buffers(
            "ds41rt_cuda_gather_rows_bf16",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaGatherRowsBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_gather_rows_bf16")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u16>(),
                rows,
                row_width,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_bf16", status)
    }

    pub unsafe fn cuda_gather_rows_bf16_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_gather_bf16_buffers(
            "ds41rt_cuda_gather_rows_bf16_async",
            src,
            src_rows,
            row_indices,
            dst,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaGatherRowsBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_gather_rows_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<u16>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_gather_rows_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_copy_row_prefix_bf16(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        src_row_width: usize,
        dst_row_width: usize,
        prefix_width: usize,
        src_row_offset: usize,
    ) -> Result<()> {
        validate_row_prefix_copy_bf16_buffers(
            "ds41rt_cuda_copy_row_prefix_bf16",
            src,
            src_rows,
            dst,
            rows,
            src_row_width,
            dst_row_width,
            prefix_width,
            src_row_offset,
        )?;

        let kernel_fn: Symbol<CudaCopyRowPrefixBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_copy_row_prefix_bf16")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                dst.ptr.cast::<u16>(),
                rows,
                src_row_width,
                dst_row_width,
                prefix_width,
                src_row_offset,
            )
        };
        self.status_to_result("ds41rt_cuda_copy_row_prefix_bf16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_copy_row_prefix_bf16_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_rows: usize,
        dst: Ds41rtDeviceBuffer,
        rows: usize,
        src_row_width: usize,
        dst_row_width: usize,
        prefix_width: usize,
        src_row_offset: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_prefix_copy_bf16_buffers(
            "ds41rt_cuda_copy_row_prefix_bf16_async",
            src,
            src_rows,
            dst,
            rows,
            src_row_width,
            dst_row_width,
            prefix_width,
            src_row_offset,
        )?;

        let kernel_fn: Symbol<CudaCopyRowPrefixBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_copy_row_prefix_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                dst.ptr.cast::<u16>(),
                rows,
                src_row_width,
                dst_row_width,
                prefix_width,
                src_row_offset,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_copy_row_prefix_bf16_async", status)
    }

    pub fn cuda_scatter_add_rows_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_scatter_buffers(
            "ds41rt_cuda_scatter_add_rows_f32",
            src,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_scatter_add_rows_f32")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result("ds41rt_cuda_scatter_add_rows_f32", status)
    }

    pub unsafe fn cuda_scatter_add_rows_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_scatter_buffers(
            "ds41rt_cuda_scatter_add_rows_f32_async",
            src,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_scatter_add_rows_f32_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<f32>() as *const f32,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_scatter_add_rows_f32_async", status)
    }

    pub fn cuda_scatter_add_rows_bf16_to_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_scatter_bf16_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_bf16_to_f32",
            src,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsBf16ToF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_scatter_add_rows_bf16_to_f32")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result("ds41rt_cuda_scatter_add_rows_bf16_to_f32", status)
    }

    pub unsafe fn cuda_scatter_add_rows_bf16_to_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_scatter_bf16_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_bf16_to_f32_async",
            src,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsBf16ToF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_bf16_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_scatter_add_rows_bf16_to_f32_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        src_row_stride_bytes: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_scatter_fp8_e4m3_row_scaled_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32",
            src,
            src_row_stride_bytes,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;
        let kernel_fn: Symbol<CudaScatterAddRowsFp8E4m3RowScaledToF32Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                src_row_stride_bytes,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_row_stride_bytes: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_scatter_fp8_e4m3_row_scaled_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async",
            src,
            src_row_stride_bytes,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;
        let kernel_fn: Symbol<CudaScatterAddRowsFp8E4m3RowScaledToF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                src_row_stride_bytes,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        src_row_stride_bytes: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_scatter_nvfp4_e2m1_fp8_e4m3_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32",
            src,
            src_row_stride_bytes,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;
        let kernel_fn: Symbol<CudaScatterAddRowsNvfp4E2m1Fp8E4m3ToF32Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                src_row_stride_bytes,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        src_row_stride_bytes: usize,
        row_indices: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_scatter_nvfp4_e2m1_fp8_e4m3_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async",
            src,
            src_row_stride_bytes,
            row_indices,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;
        let kernel_fn: Symbol<CudaScatterAddRowsNvfp4E2m1Fp8E4m3ToF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                src_row_stride_bytes,
                row_indices.ptr.cast::<u32>() as *const u32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async",
            status,
        )
    }

    pub fn cuda_reduce_route_shards_to_f32(
        &self,
        buffers: &Ds41rtRouteShardReductionBuffers,
        rows: usize,
        row_width: usize,
        peer_row_stride_bytes: usize,
        local_dtype: u32,
        peer_dtype: u32,
        peer_count: usize,
    ) -> Result<()> {
        validate_route_shard_reduction_buffers(
            "ds41rt_cuda_reduce_route_shards_to_f32",
            buffers,
            rows,
            row_width,
            peer_row_stride_bytes,
            local_dtype,
            peer_dtype,
            peer_count,
        )?;
        let peer_count = u32::try_from(peer_count).context("route shard peer count exceeds u32")?;
        let kernel_fn: Symbol<CudaReduceRouteShardsToF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_reduce_route_shards_to_f32")? };
        let status = unsafe {
            kernel_fn(
                buffers,
                rows,
                row_width,
                peer_row_stride_bytes,
                local_dtype,
                peer_dtype,
                peer_count,
            )
        };
        self.status_to_result("ds41rt_cuda_reduce_route_shards_to_f32", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_reduce_route_shards_to_f32_async(
        &self,
        buffers: &Ds41rtRouteShardReductionBuffers,
        rows: usize,
        row_width: usize,
        peer_row_stride_bytes: usize,
        local_dtype: u32,
        peer_dtype: u32,
        peer_count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_route_shard_reduction_buffers(
            "ds41rt_cuda_reduce_route_shards_to_f32_async",
            buffers,
            rows,
            row_width,
            peer_row_stride_bytes,
            local_dtype,
            peer_dtype,
            peer_count,
        )?;
        let peer_count = u32::try_from(peer_count).context("route shard peer count exceeds u32")?;
        let kernel_fn: Symbol<CudaReduceRouteShardsToF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_reduce_route_shards_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                buffers,
                rows,
                row_width,
                peer_row_stride_bytes,
                local_dtype,
                peer_dtype,
                peer_count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_reduce_route_shards_to_f32_async", status)
    }

    pub fn cuda_scatter_add_rows_bf16_weighted_to_f32(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        row_weights: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
    ) -> Result<()> {
        validate_row_scatter_bf16_weighted_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32",
            src,
            row_indices,
            row_weights,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsBf16WeightedToF32Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                row_weights.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
            )
        };
        self.status_to_result("ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32", status)
    }

    pub unsafe fn cuda_scatter_add_rows_bf16_weighted_to_f32_async(
        &self,
        src: Ds41rtDeviceBuffer,
        row_indices: Ds41rtDeviceBuffer,
        row_weights: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        dst_rows: usize,
        rows: usize,
        row_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_row_scatter_bf16_weighted_to_f32_buffers(
            "ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32_async",
            src,
            row_indices,
            row_weights,
            dst,
            dst_rows,
            rows,
            row_width,
        )?;

        let kernel_fn: Symbol<CudaScatterAddRowsBf16WeightedToF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u16>() as *const u16,
                row_indices.ptr.cast::<u32>() as *const u32,
                row_weights.ptr.cast::<f32>() as *const f32,
                dst.ptr.cast::<f32>(),
                rows,
                row_width,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_scatter_add_rows_bf16_weighted_to_f32_async",
            status,
        )
    }

    pub fn cuda_kv_cache_write_bytes(
        &self,
        src: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_offset_bytes: usize,
        bytes: usize,
    ) -> Result<()> {
        validate_kv_cache_write_buffers(
            "ds41rt_cuda_kv_cache_write_bytes",
            src,
            cache,
            cache_offset_bytes,
            bytes,
        )?;

        let kernel_fn: Symbol<CudaKvCacheWriteBytesFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_write_bytes")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                cache.ptr.cast::<u8>(),
                cache_offset_bytes,
                bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_write_bytes", status)
    }

    pub unsafe fn cuda_kv_cache_write_bytes_async(
        &self,
        src: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_offset_bytes: usize,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_kv_cache_write_buffers(
            "ds41rt_cuda_kv_cache_write_bytes_async",
            src,
            cache,
            cache_offset_bytes,
            bytes,
        )?;

        let kernel_fn: Symbol<CudaKvCacheWriteBytesAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_write_bytes_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                cache.ptr.cast::<u8>(),
                cache_offset_bytes,
                bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_write_bytes_async", status)
    }

    pub fn cuda_kv_cache_read_bytes(
        &self,
        cache: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        cache_offset_bytes: usize,
        bytes: usize,
    ) -> Result<()> {
        validate_kv_cache_read_buffers(
            "ds41rt_cuda_kv_cache_read_bytes",
            cache,
            dst,
            cache_offset_bytes,
            bytes,
        )?;

        let kernel_fn: Symbol<CudaKvCacheReadBytesFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_read_bytes")? };
        let status = unsafe {
            kernel_fn(
                cache.ptr.cast::<u8>() as *const u8,
                dst.ptr.cast::<u8>(),
                cache_offset_bytes,
                bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_read_bytes", status)
    }

    pub unsafe fn cuda_kv_cache_read_bytes_async(
        &self,
        cache: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        cache_offset_bytes: usize,
        bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_kv_cache_read_buffers(
            "ds41rt_cuda_kv_cache_read_bytes_async",
            cache,
            dst,
            cache_offset_bytes,
            bytes,
        )?;

        let kernel_fn: Symbol<CudaKvCacheReadBytesAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_read_bytes_async")? };
        let status = unsafe {
            kernel_fn(
                cache.ptr.cast::<u8>() as *const u8,
                dst.ptr.cast::<u8>(),
                cache_offset_bytes,
                bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_read_bytes_async", status)
    }

    pub unsafe fn cuda_kv_cache_write_blocks(
        &self,
        src: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        src_offsets: Ds41rtDeviceBuffer,
        cache_offsets: Ds41rtDeviceBuffer,
        block_bytes: Ds41rtDeviceBuffer,
        block_count: usize,
    ) -> Result<()> {
        validate_kv_cache_write_block_buffers(
            "ds41rt_cuda_kv_cache_write_blocks",
            src,
            cache,
            src_offsets,
            cache_offsets,
            block_bytes,
            block_count,
        )?;

        let kernel_fn: Symbol<CudaKvCacheWriteBlocksFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_write_blocks")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                cache.ptr.cast::<u8>(),
                src_offsets.ptr.cast::<u64>() as *const u64,
                cache_offsets.ptr.cast::<u64>() as *const u64,
                block_bytes.ptr.cast::<u64>() as *const u64,
                block_count,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_write_blocks", status)
    }

    pub unsafe fn cuda_kv_cache_write_blocks_async(
        &self,
        src: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        src_offsets: Ds41rtDeviceBuffer,
        cache_offsets: Ds41rtDeviceBuffer,
        block_bytes: Ds41rtDeviceBuffer,
        block_count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_kv_cache_write_block_buffers(
            "ds41rt_cuda_kv_cache_write_blocks_async",
            src,
            cache,
            src_offsets,
            cache_offsets,
            block_bytes,
            block_count,
        )?;

        let kernel_fn: Symbol<CudaKvCacheWriteBlocksAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_write_blocks_async")? };
        let status = unsafe {
            kernel_fn(
                src.ptr.cast::<u8>() as *const u8,
                cache.ptr.cast::<u8>(),
                src_offsets.ptr.cast::<u64>() as *const u64,
                cache_offsets.ptr.cast::<u64>() as *const u64,
                block_bytes.ptr.cast::<u64>() as *const u64,
                block_count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_write_blocks_async", status)
    }

    pub unsafe fn cuda_kv_cache_read_blocks(
        &self,
        cache: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        cache_offsets: Ds41rtDeviceBuffer,
        dst_offsets: Ds41rtDeviceBuffer,
        block_bytes: Ds41rtDeviceBuffer,
        block_count: usize,
    ) -> Result<()> {
        validate_kv_cache_read_block_buffers(
            "ds41rt_cuda_kv_cache_read_blocks",
            cache,
            dst,
            cache_offsets,
            dst_offsets,
            block_bytes,
            block_count,
        )?;

        let kernel_fn: Symbol<CudaKvCacheReadBlocksFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_read_blocks")? };
        let status = unsafe {
            kernel_fn(
                cache.ptr.cast::<u8>() as *const u8,
                dst.ptr.cast::<u8>(),
                cache_offsets.ptr.cast::<u64>() as *const u64,
                dst_offsets.ptr.cast::<u64>() as *const u64,
                block_bytes.ptr.cast::<u64>() as *const u64,
                block_count,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_read_blocks", status)
    }

    pub unsafe fn cuda_kv_cache_read_blocks_async(
        &self,
        cache: Ds41rtDeviceBuffer,
        dst: Ds41rtDeviceBuffer,
        cache_offsets: Ds41rtDeviceBuffer,
        dst_offsets: Ds41rtDeviceBuffer,
        block_bytes: Ds41rtDeviceBuffer,
        block_count: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_kv_cache_read_block_buffers(
            "ds41rt_cuda_kv_cache_read_blocks_async",
            cache,
            dst,
            cache_offsets,
            dst_offsets,
            block_bytes,
            block_count,
        )?;

        let kernel_fn: Symbol<CudaKvCacheReadBlocksAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_kv_cache_read_blocks_async")? };
        let status = unsafe {
            kernel_fn(
                cache.ptr.cast::<u8>() as *const u8,
                dst.ptr.cast::<u8>(),
                cache_offsets.ptr.cast::<u64>() as *const u64,
                dst_offsets.ptr.cast::<u64>() as *const u64,
                block_bytes.ptr.cast::<u64>() as *const u64,
                block_count,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_kv_cache_read_blocks_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_mla_kv_cache_unpack_bf16(
        &self,
        payload: Ds41rtDeviceBuffer,
        kv_latent: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        dsa_key: Option<Ds41rtDeviceBuffer>,
        rows: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        dsa_dim: usize,
        payload_stride_bytes: usize,
    ) -> Result<()> {
        validate_mla_kv_cache_unpack_bf16_buffers(
            "ds41rt_cuda_mla_kv_cache_unpack_bf16",
            payload,
            kv_latent,
            k_rope,
            dsa_key,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvCacheUnpackBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_cache_unpack_bf16")? };
        let dsa_key_ptr = dsa_key
            .map(|buffer| buffer.ptr.cast::<u16>())
            .unwrap_or(std::ptr::null_mut());
        let status = unsafe {
            kernel_fn(
                payload.ptr.cast::<u8>() as *const u8,
                kv_latent.ptr.cast::<u16>(),
                k_rope.ptr.cast::<u16>(),
                dsa_key_ptr,
                rows,
                kv_lora_rank,
                rope_dim,
                dsa_dim,
                payload_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_cache_unpack_bf16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_mla_kv_cache_unpack_bf16_async(
        &self,
        payload: Ds41rtDeviceBuffer,
        kv_latent: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        dsa_key: Option<Ds41rtDeviceBuffer>,
        rows: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        dsa_dim: usize,
        payload_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_cache_unpack_bf16_buffers(
            "ds41rt_cuda_mla_kv_cache_unpack_bf16_async",
            payload,
            kv_latent,
            k_rope,
            dsa_key,
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvCacheUnpackBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_cache_unpack_bf16_async")? };
        let dsa_key_ptr = dsa_key
            .map(|buffer| buffer.ptr.cast::<u16>())
            .unwrap_or(std::ptr::null_mut());
        let status = unsafe {
            kernel_fn(
                payload.ptr.cast::<u8>() as *const u8,
                kv_latent.ptr.cast::<u16>(),
                k_rope.ptr.cast::<u16>(),
                dsa_key_ptr,
                rows,
                kv_lora_rank,
                rope_dim,
                dsa_dim,
                payload_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_cache_unpack_bf16_async", status)
    }

    pub fn cuda_mla_kv_projected_split_bf16(
        &self,
        projected: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
    ) -> Result<()> {
        validate_mla_kv_projected_split_bf16_buffers(
            "ds41rt_cuda_mla_kv_projected_split_bf16",
            projected,
            k_nope,
            v,
            rows,
            heads,
            nope_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaMlaKvProjectedSplitBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_projected_split_bf16")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>(),
                v.ptr.cast::<u16>(),
                rows,
                heads,
                nope_dim,
                v_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_projected_split_bf16", status)
    }

    pub unsafe fn cuda_mla_kv_projected_split_bf16_async(
        &self,
        projected: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        v_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_projected_split_bf16_buffers(
            "ds41rt_cuda_mla_kv_projected_split_bf16_async",
            projected,
            k_nope,
            v,
            rows,
            heads,
            nope_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaMlaKvProjectedSplitBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_kv_projected_split_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>(),
                v.ptr.cast::<u16>(),
                rows,
                heads,
                nope_dim,
                v_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_projected_split_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_mla_kv_prepare_bf16(
        &self,
        projected: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        norm_weight: Ds41rtDeviceBuffer,
        prepared: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        prepared_stride_bytes: usize,
        eps: f32,
        theta: f32,
    ) -> Result<()> {
        validate_mla_kv_prepare_bf16_buffers(
            "ds41rt_cuda_mla_kv_prepare_bf16",
            projected,
            positions,
            norm_weight,
            prepared,
            rows,
            projected_stride_bytes,
            prepared_stride_bytes,
            eps,
            theta,
        )?;
        let kernel_fn: Symbol<CudaMlaKvPrepareBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_prepare_bf16")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                positions.ptr.cast::<u32>() as *const u32,
                norm_weight.ptr.cast::<u16>() as *const u16,
                prepared.ptr.cast::<u16>(),
                rows,
                projected_stride_bytes,
                prepared_stride_bytes,
                eps,
                theta,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_prepare_bf16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_mla_kv_prepare_bf16_async(
        &self,
        projected: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        norm_weight: Ds41rtDeviceBuffer,
        prepared: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        prepared_stride_bytes: usize,
        eps: f32,
        theta: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_prepare_bf16_buffers(
            "ds41rt_cuda_mla_kv_prepare_bf16_async",
            projected,
            positions,
            norm_weight,
            prepared,
            rows,
            projected_stride_bytes,
            prepared_stride_bytes,
            eps,
            theta,
        )?;
        let kernel_fn: Symbol<CudaMlaKvPrepareBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_prepare_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                positions.ptr.cast::<u32>() as *const u32,
                norm_weight.ptr.cast::<u16>() as *const u16,
                prepared.ptr.cast::<u16>(),
                rows,
                projected_stride_bytes,
                prepared_stride_bytes,
                eps,
                theta,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_prepare_bf16_async", status)
    }

    pub fn cuda_transpose_rows_heads_bf16(
        &self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<()> {
        validate_transpose_rows_heads_bf16_buffers(
            "ds41rt_cuda_transpose_rows_heads_bf16",
            input,
            output,
            rows,
            heads,
            width,
        )?;
        let kernel_fn: Symbol<CudaTransposeRowsHeadsBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_transpose_rows_heads_bf16")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                rows,
                heads,
                width,
            )
        };
        self.status_to_result("ds41rt_cuda_transpose_rows_heads_bf16", status)
    }

    pub unsafe fn cuda_transpose_rows_heads_bf16_async(
        &self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_transpose_rows_heads_bf16_buffers(
            "ds41rt_cuda_transpose_rows_heads_bf16_async",
            input,
            output,
            rows,
            heads,
            width,
        )?;
        let kernel_fn: Symbol<CudaTransposeRowsHeadsBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_transpose_rows_heads_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                rows,
                heads,
                width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_transpose_rows_heads_bf16_async", status)
    }

    pub fn cuda_transpose_heads_rows_bf16(
        &self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<()> {
        validate_transpose_rows_heads_bf16_buffers(
            "ds41rt_cuda_transpose_heads_rows_bf16",
            input,
            output,
            rows,
            heads,
            width,
        )?;
        let kernel_fn: Symbol<CudaTransposeRowsHeadsBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_transpose_heads_rows_bf16")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                rows,
                heads,
                width,
            )
        };
        self.status_to_result("ds41rt_cuda_transpose_heads_rows_bf16", status)
    }

    pub unsafe fn cuda_transpose_heads_rows_bf16_async(
        &self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_transpose_rows_heads_bf16_buffers(
            "ds41rt_cuda_transpose_heads_rows_bf16_async",
            input,
            output,
            rows,
            heads,
            width,
        )?;
        let kernel_fn: Symbol<CudaTransposeRowsHeadsBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_transpose_heads_rows_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                rows,
                heads,
                width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_transpose_heads_rows_bf16_async", status)
    }

    pub fn cuda_mla_compose_absorbed_query_bf16(
        &self,
        latent_heads_rows: Ds41rtDeviceBuffer,
        rope_rows_heads: Ds41rtDeviceBuffer,
        output_rows_heads: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        latent_width: usize,
        rope_width: usize,
    ) -> Result<()> {
        validate_mla_compose_absorbed_query_bf16_buffers(
            "ds41rt_cuda_mla_compose_absorbed_query_bf16",
            latent_heads_rows,
            rope_rows_heads,
            output_rows_heads,
            rows,
            heads,
            latent_width,
            rope_width,
        )?;
        let kernel_fn: Symbol<CudaMlaComposeAbsorbedQueryBf16Fn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_compose_absorbed_query_bf16")?
        };
        let status = unsafe {
            kernel_fn(
                latent_heads_rows.ptr.cast::<u16>() as *const u16,
                rope_rows_heads.ptr.cast::<u16>() as *const u16,
                output_rows_heads.ptr.cast::<u16>(),
                rows,
                heads,
                latent_width,
                rope_width,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_compose_absorbed_query_bf16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_mla_compose_absorbed_query_bf16_async(
        &self,
        latent_heads_rows: Ds41rtDeviceBuffer,
        rope_rows_heads: Ds41rtDeviceBuffer,
        output_rows_heads: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        latent_width: usize,
        rope_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_compose_absorbed_query_bf16_buffers(
            "ds41rt_cuda_mla_compose_absorbed_query_bf16_async",
            latent_heads_rows,
            rope_rows_heads,
            output_rows_heads,
            rows,
            heads,
            latent_width,
            rope_width,
        )?;
        let kernel_fn: Symbol<CudaMlaComposeAbsorbedQueryBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_compose_absorbed_query_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                latent_heads_rows.ptr.cast::<u16>() as *const u16,
                rope_rows_heads.ptr.cast::<u16>() as *const u16,
                output_rows_heads.ptr.cast::<u16>(),
                rows,
                heads,
                latent_width,
                rope_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_compose_absorbed_query_bf16_async", status)
    }

    pub fn cuda_generic_kv_page_table_init(
        &self,
        page_table: Ds41rtDeviceBuffer,
        query_rows: usize,
        page_table_width: usize,
    ) -> Result<()> {
        validate_generic_kv_page_table_buffer(
            "ds41rt_cuda_generic_kv_page_table_init",
            page_table,
            query_rows,
            page_table_width,
        )?;
        let kernel_fn: Symbol<CudaGenericKvPageTableInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_generic_kv_page_table_init")? };
        let status =
            unsafe { kernel_fn(page_table.ptr.cast::<i32>(), query_rows, page_table_width) };
        self.status_to_result("ds41rt_cuda_generic_kv_page_table_init", status)
    }

    pub unsafe fn cuda_generic_kv_page_table_init_async(
        &self,
        page_table: Ds41rtDeviceBuffer,
        query_rows: usize,
        page_table_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_generic_kv_page_table_buffer(
            "ds41rt_cuda_generic_kv_page_table_init_async",
            page_table,
            query_rows,
            page_table_width,
        )?;
        let kernel_fn: Symbol<CudaGenericKvPageTableInitAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_generic_kv_page_table_init_async")?
        };
        let status = unsafe {
            kernel_fn(
                page_table.ptr.cast::<i32>(),
                query_rows,
                page_table_width,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_generic_kv_page_table_init_async", status)
    }

    pub fn cuda_generic_kv_page_table_init_base(
        &self,
        page_table: Ds41rtDeviceBuffer,
        query_rows: usize,
        page_table_width: usize,
        base_offset: usize,
    ) -> Result<()> {
        let context = "ds41rt_cuda_generic_kv_page_table_init_base";
        validate_generic_kv_page_table_buffer(context, page_table, query_rows, page_table_width)?;
        let kernel_fn: Symbol<CudaGenericKvPageTableInitBaseFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_generic_kv_page_table_init_base")?
        };
        let status = unsafe {
            kernel_fn(
                page_table.ptr.cast::<i32>(),
                query_rows,
                page_table_width,
                base_offset,
            )
        };
        self.status_to_result(context, status)
    }

    pub unsafe fn cuda_generic_kv_page_table_init_base_async(
        &self,
        page_table: Ds41rtDeviceBuffer,
        query_rows: usize,
        page_table_width: usize,
        base_offset: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_generic_kv_page_table_init_base_async";
        validate_generic_kv_page_table_buffer(context, page_table, query_rows, page_table_width)?;
        let kernel_fn: Symbol<CudaGenericKvPageTableInitBaseAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_generic_kv_page_table_init_base_async")?
        };
        let status = unsafe {
            kernel_fn(
                page_table.ptr.cast::<i32>(),
                query_rows,
                page_table_width,
                base_offset,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub unsafe fn cuda_generic_kv_page_table_init_offsets_async(
        &self,
        page_table: Ds41rtDeviceBuffer,
        row_offsets: Ds41rtDeviceBuffer,
        query_rows: usize,
        page_table_width: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_generic_kv_page_table_init_offsets_async";
        validate_generic_kv_page_table_buffer(context, page_table, query_rows, page_table_width)?;
        validate_device_buffer_bytes(
            &format!("{context} row_offsets"),
            row_offsets,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("generic KV page-table row-offset bytes overflow")?,
        )?;
        if row_offsets.device_id != page_table.device_id {
            anyhow::bail!("{context} buffers must be on one CUDA device");
        }
        let kernel_fn: Symbol<CudaGenericKvPageTableInitOffsetsAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_generic_kv_page_table_init_offsets_async")?
        };
        let status = unsafe {
            kernel_fn(
                page_table.ptr.cast::<i32>(),
                row_offsets.ptr.cast::<i32>() as *const i32,
                query_rows,
                page_table_width,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn cuda_generic_kv_page_table_expand_indices(
        &self,
        output_indices: Ds41rtDeviceBuffer,
        physical_pages: Ds41rtDeviceBuffer,
        query_rows: usize,
        output_width: usize,
        active_tokens: usize,
    ) -> Result<()> {
        let context = "ds41rt_cuda_generic_kv_page_table_expand_indices";
        validate_generic_kv_page_table_expand_indices_buffers(
            context,
            output_indices,
            physical_pages,
            query_rows,
            output_width,
            active_tokens,
        )?;
        let kernel_fn: Symbol<CudaGenericKvPageTableExpandIndicesFn> =
            unsafe { self.lib.get(context.as_bytes())? };
        let status = unsafe {
            kernel_fn(
                output_indices.ptr.cast::<i32>(),
                physical_pages.ptr.cast::<u32>() as *const u32,
                query_rows,
                output_width,
                active_tokens,
            )
        };
        self.status_to_result(context, status)
    }

    pub unsafe fn cuda_generic_kv_page_table_expand_indices_async(
        &self,
        output_indices: Ds41rtDeviceBuffer,
        physical_pages: Ds41rtDeviceBuffer,
        query_rows: usize,
        output_width: usize,
        active_tokens: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_generic_kv_page_table_expand_indices_async";
        validate_generic_kv_page_table_expand_indices_buffers(
            context,
            output_indices,
            physical_pages,
            query_rows,
            output_width,
            active_tokens,
        )?;
        let kernel_fn: Symbol<CudaGenericKvPageTableExpandIndicesAsyncFn> =
            unsafe { self.lib.get(context.as_bytes())? };
        let status = unsafe {
            kernel_fn(
                output_indices.ptr.cast::<i32>(),
                physical_pages.ptr.cast::<u32>() as *const u32,
                query_rows,
                output_width,
                active_tokens,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn cuda_mla_kv_pack_fp8_ds_mla(
        &self,
        projected: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
    ) -> Result<()> {
        validate_mla_kv_fp8_ds_mla_pack_buffers(
            "ds41rt_cuda_mla_kv_pack_fp8_ds_mla",
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvPackFp8DsMlaFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_pack_fp8_ds_mla")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                packed.ptr.cast::<u8>(),
                rows,
                projected_stride_bytes,
                packed_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_pack_fp8_ds_mla", status)
    }

    pub unsafe fn cuda_mla_kv_pack_fp8_ds_mla_async(
        &self,
        projected: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_fp8_ds_mla_pack_buffers(
            "ds41rt_cuda_mla_kv_pack_fp8_ds_mla_async",
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvPackFp8DsMlaAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_pack_fp8_ds_mla_async")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                packed.ptr.cast::<u8>(),
                rows,
                projected_stride_bytes,
                packed_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_pack_fp8_ds_mla_async", status)
    }

    pub fn cuda_mla_kv_unpack_fp8_ds_mla(
        &self,
        packed: Ds41rtDeviceBuffer,
        projected: Ds41rtDeviceBuffer,
        rows: usize,
        packed_stride_bytes: usize,
        projected_stride_bytes: usize,
    ) -> Result<()> {
        validate_mla_kv_fp8_ds_mla_unpack_buffers(
            "ds41rt_cuda_mla_kv_unpack_fp8_ds_mla",
            packed,
            projected,
            rows,
            packed_stride_bytes,
            projected_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvUnpackFp8DsMlaFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_unpack_fp8_ds_mla")? };
        let status = unsafe {
            kernel_fn(
                packed.ptr.cast::<u8>() as *const u8,
                projected.ptr.cast::<u16>(),
                rows,
                packed_stride_bytes,
                projected_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_unpack_fp8_ds_mla", status)
    }

    pub unsafe fn cuda_mla_kv_unpack_fp8_ds_mla_async(
        &self,
        packed: Ds41rtDeviceBuffer,
        projected: Ds41rtDeviceBuffer,
        rows: usize,
        packed_stride_bytes: usize,
        projected_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_fp8_ds_mla_unpack_buffers(
            "ds41rt_cuda_mla_kv_unpack_fp8_ds_mla_async",
            packed,
            projected,
            rows,
            packed_stride_bytes,
            projected_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvUnpackFp8DsMlaAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_unpack_fp8_ds_mla_async")? };
        let status = unsafe {
            kernel_fn(
                packed.ptr.cast::<u8>() as *const u8,
                projected.ptr.cast::<u16>(),
                rows,
                packed_stride_bytes,
                projected_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_unpack_fp8_ds_mla_async", status)
    }

    pub fn cuda_mla_kv_pack_mxfp4_ds_mla(
        &self,
        projected: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
    ) -> Result<()> {
        validate_mla_kv_mxfp4_ds_mla_pack_buffers(
            "ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla",
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvPackMxfp4DsMlaFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                packed.ptr.cast::<u8>(),
                rows,
                projected_stride_bytes,
                packed_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla", status)
    }

    pub unsafe fn cuda_mla_kv_pack_mxfp4_ds_mla_async(
        &self,
        projected: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        rows: usize,
        projected_stride_bytes: usize,
        packed_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_mxfp4_ds_mla_pack_buffers(
            "ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla_async",
            projected,
            packed,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvPackMxfp4DsMlaAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla_async")? };
        let status = unsafe {
            kernel_fn(
                projected.ptr.cast::<u16>() as *const u16,
                packed.ptr.cast::<u8>(),
                rows,
                projected_stride_bytes,
                packed_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_pack_mxfp4_ds_mla_async", status)
    }

    pub fn cuda_mla_kv_unpack_mxfp4_ds_mla(
        &self,
        packed: Ds41rtDeviceBuffer,
        projected: Ds41rtDeviceBuffer,
        rows: usize,
        packed_stride_bytes: usize,
        projected_stride_bytes: usize,
    ) -> Result<()> {
        validate_mla_kv_mxfp4_ds_mla_unpack_buffers(
            "ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla",
            packed,
            projected,
            rows,
            packed_stride_bytes,
            projected_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvUnpackMxfp4DsMlaFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla")? };
        let status = unsafe {
            kernel_fn(
                packed.ptr.cast::<u8>() as *const u8,
                projected.ptr.cast::<u16>(),
                rows,
                packed_stride_bytes,
                projected_stride_bytes,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla", status)
    }

    pub unsafe fn cuda_mla_kv_unpack_mxfp4_ds_mla_async(
        &self,
        packed: Ds41rtDeviceBuffer,
        projected: Ds41rtDeviceBuffer,
        rows: usize,
        packed_stride_bytes: usize,
        projected_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_kv_mxfp4_ds_mla_unpack_buffers(
            "ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla_async",
            packed,
            projected,
            rows,
            packed_stride_bytes,
            projected_stride_bytes,
        )?;

        let kernel_fn: Symbol<CudaMlaKvUnpackMxfp4DsMlaAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla_async")?
        };
        let status = unsafe {
            kernel_fn(
                packed.ptr.cast::<u8>() as *const u8,
                projected.ptr.cast::<u16>(),
                rows,
                packed_stride_bytes,
                projected_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_kv_unpack_mxfp4_ds_mla_async", status)
    }

    pub fn cuda_router_topk_f32(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<()> {
        validate_router_topk_buffers(
            "ds41rt_cuda_router_topk_f32",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;

        let kernel_fn: Symbol<CudaRouterTopKF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_router_topk_f32")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<f32>() as *const f32,
                router_weight.ptr.cast::<f32>() as *const f32,
                correction_bias.ptr.cast::<f32>() as *const f32,
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                experts,
                top_k,
            )
        };
        self.status_to_result("ds41rt_cuda_router_topk_f32", status)
    }

    pub unsafe fn cuda_router_topk_f32_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_router_topk_buffers(
            "ds41rt_cuda_router_topk_f32_async",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;

        let kernel_fn: Symbol<CudaRouterTopKF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_router_topk_f32_async")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<f32>() as *const f32,
                router_weight.ptr.cast::<f32>() as *const f32,
                correction_bias.ptr.cast::<f32>() as *const f32,
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                experts,
                top_k,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_router_topk_f32_async", status)
    }

    pub fn cuda_router_topk_bf16(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
    ) -> Result<()> {
        validate_router_topk_bf16_buffers(
            "ds41rt_cuda_router_topk_bf16",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;

        let kernel_fn: Symbol<CudaRouterTopKBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_router_topk_bf16")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias.ptr.cast::<f32>() as *const f32,
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                experts,
                top_k,
            )
        };
        self.status_to_result("ds41rt_cuda_router_topk_bf16", status)
    }

    pub unsafe fn cuda_router_topk_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_router_topk_bf16_buffers(
            "ds41rt_cuda_router_topk_bf16_async",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;

        let kernel_fn: Symbol<CudaRouterTopKBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_router_topk_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias.ptr.cast::<f32>() as *const f32,
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                experts,
                top_k,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_router_topk_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_flash_router_topk_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Option<Ds41rtDeviceBuffer>,
        token_to_experts: Option<Ds41rtDeviceBuffer>,
        token_ids: Option<Ds41rtDeviceBuffer>,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hash_routing: bool,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_ds4_flash_router_topk_bf16_buffers(
            "ds41rt_cuda_ds4_flash_router_topk_bf16_async",
            hidden,
            router_weight,
            correction_bias,
            token_to_experts,
            token_ids,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hash_routing,
        )?;

        let kernel_fn: Symbol<CudaDs4FlashRouterTopKBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_router_topk_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias
                    .map(|buffer| buffer.ptr.cast::<f32>() as *const f32)
                    .unwrap_or(std::ptr::null()),
                token_to_experts
                    .map(|buffer| buffer.ptr.cast::<i64>() as *const i64)
                    .unwrap_or(std::ptr::null()),
                token_ids
                    .map(|buffer| buffer.ptr.cast::<i64>() as *const i64)
                    .unwrap_or(std::ptr::null()),
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                c_int::from(hash_routing),
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_router_topk_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_pro_router_topk_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Option<Ds41rtDeviceBuffer>,
        token_to_experts: Option<Ds41rtDeviceBuffer>,
        token_ids: Option<Ds41rtDeviceBuffer>,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        hash_routing: bool,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_ds4_pro_router_topk_bf16_buffers(
            "ds41rt_cuda_ds4_pro_router_topk_bf16_async",
            hidden,
            router_weight,
            correction_bias,
            token_to_experts,
            token_ids,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hash_routing,
        )?;

        let kernel_fn: Symbol<CudaDs4FlashRouterTopKBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_ds4_pro_router_topk_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias
                    .map(|buffer| buffer.ptr.cast::<f32>() as *const f32)
                    .unwrap_or(std::ptr::null()),
                token_to_experts
                    .map(|buffer| buffer.ptr.cast::<i64>() as *const i64)
                    .unwrap_or(std::ptr::null()),
                token_ids
                    .map(|buffer| buffer.ptr.cast::<i64>() as *const i64)
                    .unwrap_or(std::ptr::null()),
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                c_int::from(hash_routing),
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_pro_router_topk_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_flash_router_refine_topk_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        approximate_logits: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_ds4_flash_router_refine_topk_bf16_buffers(
            "ds41rt_cuda_ds4_flash_router_refine_topk_bf16_async",
            hidden,
            router_weight,
            correction_bias,
            approximate_logits,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
        )?;
        let kernel_fn: Symbol<CudaDs4FlashRouterRefineTopKBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_flash_router_refine_topk_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias.ptr.cast::<f32>() as *const f32,
                approximate_logits.ptr.cast::<f32>(),
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                rows,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_ds4_flash_router_refine_topk_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_router_topk_bf16_cub_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        router_weight: Ds41rtDeviceBuffer,
        correction_bias: Ds41rtDeviceBuffer,
        corrected_scores: Ds41rtDeviceBuffer,
        sorted_corrected_scores: Ds41rtDeviceBuffer,
        unsorted_indices: Ds41rtDeviceBuffer,
        sorted_indices: Ds41rtDeviceBuffer,
        segment_offsets: Ds41rtDeviceBuffer,
        topk_indices: Ds41rtDeviceBuffer,
        topk_scores: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        cub_temp_storage: Ds41rtDeviceBuffer,
        cub_temp_storage_bytes: usize,
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_router_topk_bf16_buffers(
            "ds41rt_cuda_router_topk_bf16_cub_async",
            hidden,
            router_weight,
            correction_bias,
            topk_indices,
            topk_scores,
            topk_weights,
            rows,
            hidden_dim,
            experts,
            top_k,
        )?;
        let score_values = checked_row_values(
            "ds41rt_cuda_router_topk_bf16_cub_async score workspace",
            rows,
            experts,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_router_topk_bf16_cub_async corrected_scores",
            corrected_scores,
            score_values,
        )?;
        validate_f32_buffer_values(
            "ds41rt_cuda_router_topk_bf16_cub_async sorted_corrected_scores",
            sorted_corrected_scores,
            score_values,
        )?;
        validate_u32_buffer_values(
            "ds41rt_cuda_router_topk_bf16_cub_async unsorted_indices",
            unsorted_indices,
            score_values,
        )?;
        validate_u32_buffer_values(
            "ds41rt_cuda_router_topk_bf16_cub_async sorted_indices",
            sorted_indices,
            score_values,
        )?;
        let offset_values = rows.checked_add(1).context(
            "ds41rt_cuda_router_topk_bf16_cub_async segment offset count overflows usize",
        )?;
        let offset_bytes = offset_values
            .checked_mul(std::mem::size_of::<i32>())
            .context("ds41rt_cuda_router_topk_bf16_cub_async segment offset bytes overflow usize")?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_router_topk_bf16_cub_async segment_offsets",
            segment_offsets,
            offset_bytes,
        )?;
        validate_device_buffer_bytes(
            "ds41rt_cuda_router_topk_bf16_cub_async cub_temp_storage",
            cub_temp_storage,
            cub_temp_storage_bytes,
        )?;

        let kernel_fn: Symbol<CudaRouterTopKBf16CubAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_router_topk_bf16_cub_async")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                router_weight.ptr.cast::<u16>() as *const u16,
                correction_bias.ptr.cast::<f32>() as *const f32,
                corrected_scores.ptr.cast::<f32>(),
                sorted_corrected_scores.ptr.cast::<f32>(),
                unsorted_indices.ptr.cast::<u32>(),
                sorted_indices.ptr.cast::<u32>(),
                segment_offsets.ptr.cast::<i32>(),
                topk_indices.ptr.cast::<u32>(),
                topk_scores.ptr.cast::<f32>(),
                topk_weights.ptr.cast::<f32>(),
                cub_temp_storage.ptr,
                cub_temp_storage_bytes,
                rows,
                hidden_dim,
                experts,
                top_k,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_router_topk_bf16_cub_async", status)
    }

    pub fn cuda_linear_f32(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        validate_linear_buffers(
            "ds41rt_cuda_linear_f32",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearF32Fn> = unsafe { self.lib.get(b"ds41rt_cuda_linear_f32")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<f32>() as *const f32,
                bias_ptr,
                output.ptr.cast::<f32>(),
                rows,
                input_dim,
                output_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_f32", status)
    }

    pub unsafe fn cuda_linear_f32_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_buffers(
            "ds41rt_cuda_linear_f32_async",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_f32_async")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<f32>() as *const f32,
                weight.ptr.cast::<f32>() as *const f32,
                bias_ptr,
                output.ptr.cast::<f32>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_f32_async", status)
    }

    pub fn cuda_linear_bf16(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        validate_linear_bf16_buffers(
            "ds41rt_cuda_linear_bf16",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_bf16")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<u16>() as *const u16)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias_ptr,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16", status)
    }

    pub unsafe fn cuda_linear_bf16_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_bf16_buffers(
            "ds41rt_cuda_linear_bf16_async",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_bf16_async")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<u16>() as *const u16)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias_ptr,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16_async", status)
    }

    pub fn cuda_linear_bf16_cublas(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        validate_linear_bf16_buffers(
            "ds41rt_cuda_linear_bf16_cublas",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearBf16CublasFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_bf16_cublas")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<u16>() as *const u16)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias_ptr,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16_cublas", status)
    }

    pub unsafe fn cuda_linear_bf16_cublas_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_bf16_buffers(
            "ds41rt_cuda_linear_bf16_cublas_async",
            input,
            weight,
            bias,
            output,
            rows,
            input_dim,
            output_dim,
        )?;

        let kernel_fn: Symbol<CudaLinearBf16CublasAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_bf16_cublas_async")? };
        let bias_ptr = bias
            .map(|buffer| buffer.ptr.cast::<u16>() as *const u16)
            .unwrap_or(std::ptr::null());
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                bias_ptr,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16_cublas_async", status)
    }

    pub unsafe fn cuda_linear_bf16_f32_cublas_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_bf16_f32_buffers(
            "ds41rt_cuda_linear_bf16_f32_cublas_async",
            input,
            weight,
            output,
            rows,
            input_dim,
            output_dim,
        )?;
        let kernel_fn: Symbol<CudaLinearBf16F32CublasAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_linear_bf16_f32_cublas_async")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<f32>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16_f32_cublas_async", status)
    }

    pub unsafe fn cuda_linear_bf16_m1_parity_batched_cublaslt_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=16).contains(&rows),
            "BF16 M1-parity batched projection requires 2..=16 rows"
        );
        validate_linear_bf16_buffers(
            "ds41rt_cuda_linear_bf16_m1_parity_batched_cublaslt_async",
            input,
            weight,
            None,
            output,
            rows,
            input_dim,
            output_dim,
        )?;
        let kernel_fn: Symbol<CudaLinearBf16M1ParityBatchedCublasLtAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_bf16_m1_parity_batched_cublaslt_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_linear_bf16_m1_parity_batched_cublaslt_async",
            status,
        )
    }

    pub unsafe fn cuda_quantize_bf16_w8a16_group256_async(
        &self,
        source: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        input_dim: usize,
        output_dim: usize,
        k_major: bool,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let source_bytes = output_dim
            .checked_mul(input_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("W8A16 quantizer source bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("W8A16 quantizer weight bytes overflow")?;
        anyhow::ensure!(
            input_dim > 0 && input_dim % 256 == 0 && output_dim > 0,
            "W8A16 quantizer requires positive dimensions and K divisible by 256"
        );
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("W8A16 quantizer scale bytes overflow")?;
        for (label, buffer, expected) in [
            ("source", source, source_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "W8A16 quantizer {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
        }
        let kernel_fn: Symbol<CudaQuantizeBf16W8a16Group256AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_quantize_bf16_w8a16_group256_async")?
        };
        let status = unsafe {
            kernel_fn(
                source.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>(),
                scales.ptr.cast::<f32>(),
                input_dim,
                output_dim,
                i32::from(k_major),
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_quantize_bf16_w8a16_group256_async", status)
    }

    pub unsafe fn cuda_quantize_bf16_w8a16_group256_packed_async(
        &self,
        source: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let source_bytes = output_dim
            .checked_mul(input_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("packed W8A16 quantizer source bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("packed W8A16 quantizer weight bytes overflow")?;
        anyhow::ensure!(
            input_dim > 0
                && input_dim % 256 == 0
                && output_dim > 0
                && output_dim % 64 == 0,
            "packed W8A16 quantizer requires positive dimensions, K divisible by 256, and N divisible by 64"
        );
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("packed W8A16 quantizer scale bytes overflow")?;
        for (label, buffer, expected) in [
            ("source", source, source_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "packed W8A16 quantizer {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
        }
        let kernel_fn: Symbol<CudaQuantizeBf16W8a16Group256PackedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_quantize_bf16_w8a16_group256_packed_async")?
        };
        let status = unsafe {
            kernel_fn(
                source.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>(),
                scales.ptr.cast::<f32>(),
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_quantize_bf16_w8a16_group256_packed_async",
            status,
        )
    }

    pub unsafe fn cuda_w8a16_packed_o_aot_init(&self) -> Result<()> {
        let init_fn: Symbol<CudaB12xCoordinatorAotInitFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_w8a16_packed_o_aot_init")? };
        let status = unsafe { init_fn() };
        self.status_to_result("ds41rt_cuda_w8a16_packed_o_aot_init", status)
    }

    pub unsafe fn cuda_w8a16_packed_o_initialize_launch_buffers_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        rows: usize,
        block_m: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaW8a16PackedOInitializeAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_w8a16_packed_o_initialize_launch_buffers_async")?
        };
        let status = unsafe { kernel_fn(buffers, rows, block_m, cuda_stream) };
        self.status_to_result(
            "ds41rt_cuda_w8a16_packed_o_initialize_launch_buffers_async",
            status,
        )
    }

    pub unsafe fn cuda_w8a16_packed_o_async(
        &self,
        buffers: &Ds41rtB12xCoordinatorW4a16Buffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let kernel_fn: Symbol<CudaB12xCoordinatorW4a16BuffersRowsAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_w8a16_packed_o_async")? };
        let status = unsafe { kernel_fn(buffers, rows, cuda_stream) };
        self.status_to_result("ds41rt_cuda_w8a16_packed_o_async", status)
    }

    pub unsafe fn cuda_linear_w8a16_group256_m1_simt_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        input_dim: usize,
        output_dim: usize,
        variant: i32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            input_dim > 0 && input_dim % 256 == 0 && output_dim > 0,
            "W8A16 M=1 projection requires positive dimensions and K divisible by 256"
        );
        let input_bytes = input_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("W8A16 M=1 input bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("W8A16 M=1 weight bytes overflow")?;
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("W8A16 M=1 scale bytes overflow")?;
        let output_bytes = output_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("W8A16 M=1 output bytes overflow")?;
        for (label, buffer, expected) in [
            ("input", input, input_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
            ("output", output, output_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "W8A16 M=1 {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
        }
        let kernel_fn: Symbol<CudaLinearW8a16Group256M1SimtAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_w8a16_group256_m1_simt_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>() as *const i8,
                scales.ptr.cast::<f32>() as *const f32,
                output.ptr.cast::<u16>(),
                input_dim,
                output_dim,
                variant,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_w8a16_group256_m1_simt_async", status)
    }

    pub unsafe fn cuda_linear_w8a16_group256_m1_warp_packed_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            input_dim > 0
                && input_dim % (256 * 8) == 0
                && output_dim > 0
                && output_dim % 64 == 0,
            "packed W8A16 M=1 projection requires positive dimensions, K divisible by 2048, and N divisible by 64"
        );
        let input_bytes = input_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("packed W8A16 M=1 input bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("packed W8A16 M=1 weight bytes overflow")?;
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("packed W8A16 M=1 scale bytes overflow")?;
        let output_bytes = output_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("packed W8A16 M=1 output bytes overflow")?;
        for (label, buffer, expected) in [
            ("input", input, input_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
            ("output", output, output_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "packed W8A16 M=1 {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
        }
        let kernel_fn: Symbol<CudaLinearW8a16Group256M1WarpPackedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_w8a16_group256_m1_warp_packed_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>() as *const i8,
                scales.ptr.cast::<f32>() as *const f32,
                output.ptr.cast::<u16>(),
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_linear_w8a16_group256_m1_warp_packed_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=16).contains(&rows)
                && input_dim > 0
                && input_dim % (256 * 8) == 0
                && output_dim > 0
                && output_dim % 64 == 0,
            "packed W8A16 parity-batched projection requires 2..=16 rows, K divisible by 2048, and N divisible by 64"
        );
        let input_bytes = rows
            .checked_mul(input_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("packed W8A16 parity-batched input bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("packed W8A16 parity-batched weight bytes overflow")?;
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("packed W8A16 parity-batched scale bytes overflow")?;
        let output_bytes = rows
            .checked_mul(output_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("packed W8A16 parity-batched output bytes overflow")?;
        for (label, buffer, expected) in [
            ("input", input, input_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
            ("output", output, output_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "packed W8A16 parity-batched {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == input.device_id,
                "packed W8A16 parity-batched {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                input.device_id
            );
        }
        let kernel_fn: Symbol<CudaLinearW8a16Group256M1WarpPackedParityBatchedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>() as *const i8,
                scales.ptr.cast::<f32>() as *const f32,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_linear_w8a16_group256_m1_parity_batched_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=16).contains(&rows)
                && input_dim > 0
                && input_dim % 256 == 0
                && output_dim > 0,
            "W8A16 parity-batched projection requires 2..=16 rows, positive dimensions, and K divisible by 256"
        );
        let input_bytes = rows
            .checked_mul(input_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("W8A16 parity-batched input bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("W8A16 parity-batched weight bytes overflow")?;
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("W8A16 parity-batched scale bytes overflow")?;
        let output_bytes = rows
            .checked_mul(output_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("W8A16 parity-batched output bytes overflow")?;
        for (label, buffer, expected) in [
            ("input", input, input_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
            ("output", output, output_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "W8A16 parity-batched {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == input.device_id,
                "W8A16 parity-batched {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                input.device_id
            );
        }
        let kernel_fn: Symbol<CudaLinearW8a16Group256M1ParityBatchedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_w8a16_group256_m1_parity_batched_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>() as *const i8,
                scales.ptr.cast::<f32>() as *const f32,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_linear_w8a16_group256_m1_parity_batched_async",
            status,
        )
    }

    pub unsafe fn cuda_preload_w8a16_group256_aot(
        &self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            input_dim > 0 && input_dim % 256 == 0 && output_dim > 0,
            "W8A16 AOT preload requires positive dimensions and K divisible by 256"
        );
        let preload_fn: Symbol<CudaPreloadW8a16Group256AotFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_preload_w8a16_group256_aot")? };
        let status = unsafe { preload_fn(input_dim, output_dim) };
        self.status_to_result("ds41rt_cuda_preload_w8a16_group256_aot", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_linear_w8a16_group256_aot_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            (2..=2048).contains(&rows)
                && input_dim > 0
                && input_dim % 256 == 0
                && output_dim > 0,
            "W8A16 AOT projection requires 2..=2048 rows, positive dimensions, and K divisible by 256"
        );
        let input_bytes = rows
            .checked_mul(input_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("W8A16 AOT input bytes overflow")?;
        let weight_bytes = output_dim
            .checked_mul(input_dim)
            .context("W8A16 AOT weight bytes overflow")?;
        let scale_bytes = output_dim
            .checked_mul(input_dim / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("W8A16 AOT scale bytes overflow")?;
        let output_bytes = rows
            .checked_mul(output_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("W8A16 AOT output bytes overflow")?;
        for (label, buffer, expected) in [
            ("input", input, input_bytes),
            ("weight", weight, weight_bytes),
            ("scales", scales, scale_bytes),
            ("output", output, output_bytes),
        ] {
            anyhow::ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= expected,
                "W8A16 AOT {label} buffer has {} bytes, expected at least {expected}",
                buffer.bytes
            );
            anyhow::ensure!(
                buffer.device_id == input.device_id,
                "W8A16 AOT {label} buffer is on CUDA device {}, expected {}",
                buffer.device_id,
                input.device_id
            );
        }
        let kernel_fn: Symbol<CudaLinearW8a16Group256AotAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_w8a16_group256_aot_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<i8>() as *const i8,
                scales.ptr.cast::<f32>() as *const f32,
                output.ptr.cast::<u16>(),
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_w8a16_group256_aot_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_linear_bf16_strided_batched_cublas(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        batch_count: usize,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input_batch_stride: usize,
        weight_batch_stride: usize,
        output_batch_stride: usize,
    ) -> Result<()> {
        validate_linear_bf16_strided_batched_buffers(
            "ds41rt_cuda_linear_bf16_strided_batched_cublas",
            input,
            weight,
            output,
            batch_count,
            rows,
            input_dim,
            output_dim,
            input_batch_stride,
            weight_batch_stride,
            output_batch_stride,
        )?;
        let kernel_fn: Symbol<CudaLinearBf16StridedBatchedCublasFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_bf16_strided_batched_cublas")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                batch_count,
                rows,
                input_dim,
                output_dim,
                input_batch_stride,
                weight_batch_stride,
                output_batch_stride,
            )
        };
        self.status_to_result("ds41rt_cuda_linear_bf16_strided_batched_cublas", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_linear_bf16_strided_batched_cublas_async(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        batch_count: usize,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input_batch_stride: usize,
        weight_batch_stride: usize,
        output_batch_stride: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_bf16_strided_batched_buffers(
            "ds41rt_cuda_linear_bf16_strided_batched_cublas_async",
            input,
            weight,
            output,
            batch_count,
            rows,
            input_dim,
            output_dim,
            input_batch_stride,
            weight_batch_stride,
            output_batch_stride,
        )?;
        let kernel_fn: Symbol<CudaLinearBf16StridedBatchedCublasAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_linear_bf16_strided_batched_cublas_async")?
        };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                weight.ptr.cast::<u16>() as *const u16,
                output.ptr.cast::<u16>(),
                batch_count,
                rows,
                input_dim,
                output_dim,
                input_batch_stride,
                weight_batch_stride,
                output_batch_stride,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_linear_bf16_strided_batched_cublas_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_matmul_bf16_strided_batched_cublas_async(
        &self,
        input: Ds41rtDeviceBuffer,
        right: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        batch_count: usize,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        input_batch_stride: usize,
        right_batch_stride: usize,
        output_batch_stride: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_linear_bf16_strided_batched_buffers(
            "ds41rt_cuda_matmul_bf16_strided_batched_cublas_async",
            input,
            right,
            output,
            batch_count,
            rows,
            input_dim,
            output_dim,
            input_batch_stride,
            right_batch_stride,
            output_batch_stride,
        )?;
        let kernel_fn: Symbol<CudaMatmulBf16StridedBatchedCublasAsyncFn> = self
            .lib
            .get(b"ds41rt_cuda_matmul_bf16_strided_batched_cublas_async")?;
        let status = kernel_fn(
            input.ptr.cast::<u16>() as *const u16,
            right.ptr.cast::<u16>() as *const u16,
            output.ptr.cast::<u16>(),
            batch_count,
            rows,
            input_dim,
            output_dim,
            input_batch_stride,
            right_batch_stride,
            output_batch_stride,
            cuda_stream,
        );
        self.status_to_result(
            "ds41rt_cuda_matmul_bf16_strided_batched_cublas_async",
            status,
        )
    }

    pub fn cuda_causal_attention_f32(
        &self,
        q: Ds41rtDeviceBuffer,
        k: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        validate_causal_attention_buffers(
            "ds41rt_cuda_causal_attention_f32",
            q,
            k,
            v,
            out,
            rows,
            heads,
            qk_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaCausalAttentionF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_causal_attention_f32")? };
        let status = unsafe {
            kernel_fn(
                q.ptr.cast::<f32>() as *const f32,
                k.ptr.cast::<f32>() as *const f32,
                v.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result("ds41rt_cuda_causal_attention_f32", status)
    }

    pub unsafe fn cuda_causal_attention_f32_async(
        &self,
        q: Ds41rtDeviceBuffer,
        k: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_causal_attention_buffers(
            "ds41rt_cuda_causal_attention_f32_async",
            q,
            k,
            v,
            out,
            rows,
            heads,
            qk_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaCausalAttentionF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_causal_attention_f32_async")? };
        let status = unsafe {
            kernel_fn(
                q.ptr.cast::<f32>() as *const f32,
                k.ptr.cast::<f32>() as *const f32,
                v.ptr.cast::<f32>() as *const f32,
                out.ptr.cast::<f32>(),
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_causal_attention_f32_async", status)
    }

    pub fn cuda_causal_attention_bf16(
        &self,
        q: Ds41rtDeviceBuffer,
        k: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        validate_causal_attention_bf16_buffers(
            "ds41rt_cuda_causal_attention_bf16",
            q,
            k,
            v,
            out,
            rows,
            heads,
            qk_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaCausalAttentionBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_causal_attention_bf16")? };
        let status = unsafe {
            kernel_fn(
                q.ptr.cast::<u16>() as *const u16,
                k.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result("ds41rt_cuda_causal_attention_bf16", status)
    }

    pub unsafe fn cuda_causal_attention_bf16_async(
        &self,
        q: Ds41rtDeviceBuffer,
        k: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_causal_attention_bf16_buffers(
            "ds41rt_cuda_causal_attention_bf16_async",
            q,
            k,
            v,
            out,
            rows,
            heads,
            qk_dim,
            v_dim,
        )?;

        let kernel_fn: Symbol<CudaCausalAttentionBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_causal_attention_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                q.ptr.cast::<u16>() as *const u16,
                k.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_causal_attention_bf16_async", status)
    }

    pub unsafe fn cuda_ds4_dspark_prompt_metadata_async(
        &self,
        positions: Ds41rtDeviceBuffer,
        main_slots: Ds41rtDeviceBuffer,
        cos_sin_cache: Ds41rtDeviceBuffer,
        absolute_position_start: usize,
        request_slot: usize,
        rows: usize,
        theta: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_ds4_dspark_prompt_metadata_async";
        if rows == 0 || !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("{context} requires positive rows and finite positive theta");
        }
        validate_u32_buffer_values(&format!("{context} positions"), positions, rows)?;
        validate_u32_buffer_values(&format!("{context} main_slots"), main_slots, rows)?;
        let cos_sin_values = rows
            .checked_mul(64)
            .context("dSpark prompt cos/sin value count overflow")?;
        validate_f32_buffer_values(
            &format!("{context} cos_sin_cache"),
            cos_sin_cache,
            cos_sin_values,
        )?;
        let kernel_fn: Symbol<CudaDs4DsparkPromptMetadataAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_dspark_prompt_metadata_async")?
        };
        let status = unsafe {
            kernel_fn(
                positions.ptr.cast::<u32>(),
                main_slots.ptr.cast::<u32>(),
                cos_sin_cache.ptr.cast::<f32>(),
                absolute_position_start,
                request_slot,
                rows,
                theta,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_ds4_dspark_prepare_proposal_async(
        &self,
        embedding: Ds41rtDeviceBuffer,
        draft_token_ids: Ds41rtDeviceBuffer,
        proposal_residual: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        main_slots: Ds41rtDeviceBuffer,
        cos_sin_cache: Ds41rtDeviceBuffer,
        selected_indices: Ds41rtDeviceBuffer,
        selected_lengths: Ds41rtDeviceBuffer,
        anchor_token_id: usize,
        noise_token_id: usize,
        request_slot: usize,
        cache_window_start: usize,
        main_context_end: usize,
        vocab_size: usize,
        hidden_size: usize,
        theta: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const PROPOSAL_ROWS: usize = 5;
        const HC_MULT: usize = 4;
        const ROTARY_VALUES: usize = 64;
        const SELECTION_WIDTH: usize = 133;
        let context = "ds41rt_cuda_ds4_dspark_prepare_proposal_async";
        if vocab_size == 0 || hidden_size == 0 {
            anyhow::bail!("{context} requires positive vocabulary and hidden sizes");
        }
        if anchor_token_id >= vocab_size || noise_token_id >= vocab_size {
            anyhow::bail!(
                "{context} token ids {anchor_token_id}/{noise_token_id} exceed vocabulary {vocab_size}"
            );
        }
        if cache_window_start >= main_context_end || main_context_end - cache_window_start > 128 {
            anyhow::bail!(
                "{context} requires a non-empty target window of at most 128 tokens, got {cache_window_start}..{main_context_end}"
            );
        }
        if !theta.is_finite() || theta <= 0.0 {
            anyhow::bail!("{context} requires finite positive theta");
        }
        let embedding_values =
            checked_row_values(&format!("{context} embedding"), vocab_size, hidden_size)?;
        let residual_values = checked_row_values(
            &format!("{context} proposal_residual"),
            PROPOSAL_ROWS * HC_MULT,
            hidden_size,
        )?;
        validate_u16_buffer_values(&format!("{context} embedding"), embedding, embedding_values)?;
        validate_u32_buffer_values(
            &format!("{context} draft_token_ids"),
            draft_token_ids,
            PROPOSAL_ROWS,
        )?;
        validate_u16_buffer_values(
            &format!("{context} proposal_residual"),
            proposal_residual,
            residual_values,
        )?;
        validate_u32_buffer_values(&format!("{context} positions"), positions, PROPOSAL_ROWS)?;
        validate_u32_buffer_values(&format!("{context} main_slots"), main_slots, PROPOSAL_ROWS)?;
        validate_f32_buffer_values(
            &format!("{context} cos_sin_cache"),
            cos_sin_cache,
            PROPOSAL_ROWS * ROTARY_VALUES,
        )?;
        validate_i32_buffer_values(
            &format!("{context} selected_indices"),
            selected_indices,
            PROPOSAL_ROWS * SELECTION_WIDTH,
        )?;
        validate_i32_buffer_values(
            &format!("{context} selected_lengths"),
            selected_lengths,
            PROPOSAL_ROWS,
        )?;
        let kernel_fn: Symbol<CudaDs4DsparkPrepareProposalAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_ds4_dspark_prepare_proposal_async")?
        };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<u16>() as *const u16,
                draft_token_ids.ptr.cast::<u32>(),
                proposal_residual.ptr.cast::<u16>(),
                positions.ptr.cast::<u32>(),
                main_slots.ptr.cast::<u32>(),
                cos_sin_cache.ptr.cast::<f32>(),
                selected_indices.ptr.cast::<i32>(),
                selected_lengths.ptr.cast::<i32>(),
                anchor_token_id,
                noise_token_id,
                request_slot,
                cache_window_start,
                main_context_end,
                vocab_size,
                hidden_size,
                theta,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn cuda_rope_f32(
        &self,
        input: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<()> {
        validate_rope_buffers(
            "ds41rt_cuda_rope_f32",
            input,
            positions,
            out,
            rows,
            heads,
            rotary_dim,
            theta,
        )?;

        let kernel_fn: Symbol<CudaRopeF32Fn> = unsafe { self.lib.get(b"ds41rt_cuda_rope_f32")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<f32>() as *const f32,
                positions.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<f32>(),
                rows,
                heads,
                rotary_dim,
                theta,
            )
        };
        self.status_to_result("ds41rt_cuda_rope_f32", status)
    }

    pub unsafe fn cuda_rope_f32_async(
        &self,
        input: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_rope_buffers(
            "ds41rt_cuda_rope_f32_async",
            input,
            positions,
            out,
            rows,
            heads,
            rotary_dim,
            theta,
        )?;

        let kernel_fn: Symbol<CudaRopeF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rope_f32_async")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<f32>() as *const f32,
                positions.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<f32>(),
                rows,
                heads,
                rotary_dim,
                theta,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_rope_f32_async", status)
    }

    pub fn cuda_rope_bf16(
        &self,
        input: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<()> {
        validate_rope_bf16_buffers(
            "ds41rt_cuda_rope_bf16",
            input,
            positions,
            out,
            rows,
            heads,
            rotary_dim,
            theta,
        )?;

        let kernel_fn: Symbol<CudaRopeBf16Fn> = unsafe { self.lib.get(b"ds41rt_cuda_rope_bf16")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                positions.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                rotary_dim,
                theta,
            )
        };
        self.status_to_result("ds41rt_cuda_rope_bf16", status)
    }

    pub unsafe fn cuda_rope_bf16_async(
        &self,
        input: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_rope_bf16_buffers(
            "ds41rt_cuda_rope_bf16_async",
            input,
            positions,
            out,
            rows,
            heads,
            rotary_dim,
            theta,
        )?;

        let kernel_fn: Symbol<CudaRopeBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_rope_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                input.ptr.cast::<u16>() as *const u16,
                positions.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                rotary_dim,
                theta,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_rope_bf16_async", status)
    }

    pub fn cuda_mla_rope_attention_bf16(
        &self,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        validate_mla_rope_attention_bf16_buffers(
            "ds41rt_cuda_mla_rope_attention_bf16",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let kernel_fn: Symbol<CudaMlaRopeAttentionBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_rope_attention_bf16")? };
        let status = unsafe {
            kernel_fn(
                q_nope.ptr.cast::<u16>() as *const u16,
                q_rope.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>() as *const u16,
                k_rope.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_rope_attention_bf16", status)
    }

    pub unsafe fn cuda_mla_rope_attention_bf16_async(
        &self,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_rope_attention_bf16_buffers(
            "ds41rt_cuda_mla_rope_attention_bf16_async",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let kernel_fn: Symbol<CudaMlaRopeAttentionBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_rope_attention_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                q_nope.ptr.cast::<u16>() as *const u16,
                q_rope.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>() as *const u16,
                k_rope.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_rope_attention_bf16_async", status)
    }

    pub fn cuda_mla_rope_attention_bf16_suffix(
        &self,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        query_row_offset: usize,
        query_rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Result<()> {
        validate_mla_rope_attention_bf16_suffix_buffers(
            "ds41rt_cuda_mla_rope_attention_bf16_suffix",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            query_row_offset,
            query_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let kernel_fn: Symbol<CudaMlaRopeAttentionBf16SuffixFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_rope_attention_bf16_suffix")? };
        let status = unsafe {
            kernel_fn(
                q_nope.ptr.cast::<u16>() as *const u16,
                q_rope.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>() as *const u16,
                k_rope.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                query_row_offset,
                query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_rope_attention_bf16_suffix", status)
    }

    pub unsafe fn cuda_mla_rope_attention_bf16_suffix_async(
        &self,
        q_nope: Ds41rtDeviceBuffer,
        q_rope: Ds41rtDeviceBuffer,
        k_nope: Ds41rtDeviceBuffer,
        k_rope: Ds41rtDeviceBuffer,
        v: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        query_row_offset: usize,
        query_rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_mla_rope_attention_bf16_suffix_buffers(
            "ds41rt_cuda_mla_rope_attention_bf16_suffix_async",
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            v,
            out,
            rows,
            query_row_offset,
            query_rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        )?;

        let kernel_fn: Symbol<CudaMlaRopeAttentionBf16SuffixAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_rope_attention_bf16_suffix_async")?
        };
        let status = unsafe {
            kernel_fn(
                q_nope.ptr.cast::<u16>() as *const u16,
                q_rope.ptr.cast::<u16>() as *const u16,
                k_nope.ptr.cast::<u16>() as *const u16,
                k_rope.ptr.cast::<u16>() as *const u16,
                v.ptr.cast::<u16>() as *const u16,
                out.ptr.cast::<u16>(),
                rows,
                query_row_offset,
                query_rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_mla_rope_attention_bf16_suffix_async", status)
    }

    pub unsafe fn cuda_mla_merge_state_bf16_async(
        &self,
        accumulator: Ds41rtDeviceBuffer,
        accumulator_lse: Ds41rtDeviceBuffer,
        partial: Ds41rtDeviceBuffer,
        partial_lse: Ds41rtDeviceBuffer,
        heads: usize,
        kv_lora_rank: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_mla_merge_state_bf16_async";
        if heads == 0 || kv_lora_rank != 512 {
            anyhow::bail!("{context} requires heads>0 and kv_lora_rank=512");
        }
        let values = heads
            .checked_mul(kv_lora_rank)
            .context("MLA merge state value count overflow")?;
        validate_u16_buffer_values(&format!("{context} accumulator"), accumulator, values)?;
        validate_u16_buffer_values(&format!("{context} partial"), partial, values)?;
        let lse_bytes = heads
            .checked_mul(std::mem::size_of::<f32>())
            .context("MLA merge state LSE byte count overflow")?;
        validate_device_buffer_bytes(
            &format!("{context} accumulator_lse"),
            accumulator_lse,
            lse_bytes,
        )?;
        validate_device_buffer_bytes(&format!("{context} partial_lse"), partial_lse, lse_bytes)?;
        if accumulator.device_id != accumulator_lse.device_id
            || accumulator.device_id != partial.device_id
            || accumulator.device_id != partial_lse.device_id
        {
            anyhow::bail!("{context} buffers must reside on the same CUDA device");
        }
        let kernel_fn: Symbol<CudaMlaMergeStateBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_mla_merge_state_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                accumulator.ptr.cast::<u16>(),
                accumulator_lse.ptr.cast::<f32>(),
                partial.ptr.cast::<u16>() as *const u16,
                partial_lse.ptr.cast::<f32>() as *const f32,
                heads,
                kv_lora_rank,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_sparse_mla_nvfp4_async(
        &self,
        query: Ds41rtDeviceBuffer,
        kv_payload: Ds41rtDeviceBuffer,
        selected_indices: Ds41rtDeviceBuffer,
        topk_lengths: Ds41rtDeviceBuffer,
        partial: Ds41rtDeviceBuffer,
        partial_lse: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        output_lse: Ds41rtDeviceBuffer,
        query_rows: usize,
        heads: usize,
        topk: usize,
        kv_row_stride_bytes: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const RANK: usize = 512;
        const ROPE_DIM: usize = 64;
        const SPLITS: usize = 32;
        const SPLIT_QUERY_LIMIT: usize = 64;
        const NVFP4_ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        let context = "ds41rt_cuda_sparse_mla_nvfp4_async";
        if query_rows == 0
            || heads != 64
            || topk != 2048
            || kv_row_stride_bytes < NVFP4_ROW_BYTES
            || !scale.is_finite()
            || scale <= 0.0
        {
            anyhow::bail!(
                "{context} requires query_rows>0 heads=64 topk=2048 row_stride>=432 and positive finite scale"
            );
        }
        validate_u16_buffer_values(
            &format!("{context} query"),
            query,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(RANK + ROPE_DIM))
                .context("sparse NVFP4 MLA query values overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} selected_indices"),
            selected_indices,
            query_rows
                .checked_mul(topk)
                .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
                .context("sparse NVFP4 MLA index bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} topk_lengths"),
            topk_lengths,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse NVFP4 MLA length bytes overflow")?,
        )?;
        validate_u16_buffer_values(
            &format!("{context} output"),
            output,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(RANK))
                .context("sparse NVFP4 MLA output values overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} output_lse"),
            output_lse,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("sparse NVFP4 MLA output-LSE bytes overflow")?,
        )?;
        if query_rows <= SPLIT_QUERY_LIMIT {
            validate_u16_buffer_values(
                &format!("{context} partial"),
                partial,
                query_rows
                    .checked_mul(heads)
                    .and_then(|values| values.checked_mul(SPLITS))
                    .and_then(|values| values.checked_mul(RANK))
                    .context("sparse NVFP4 MLA partial values overflow")?,
            )?;
            validate_device_buffer_bytes(
                &format!("{context} partial_lse"),
                partial_lse,
                query_rows
                    .checked_mul(heads)
                    .and_then(|values| values.checked_mul(SPLITS))
                    .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                    .context("sparse NVFP4 MLA partial-LSE bytes overflow")?,
            )?;
        }
        if kv_payload.ptr.is_null() || kv_payload.bytes < kv_row_stride_bytes {
            anyhow::bail!("{context} KV payload is empty or smaller than one row");
        }
        for buffer in [
            kv_payload,
            selected_indices,
            topk_lengths,
            partial,
            partial_lse,
            output,
            output_lse,
        ] {
            if buffer.device_id != query.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaSparseMlaNvfp4AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_sparse_mla_nvfp4_async")? };
        let status = unsafe {
            kernel_fn(
                query.ptr.cast::<u16>() as *const u16,
                kv_payload.ptr.cast::<u8>() as *const u8,
                selected_indices.ptr.cast::<i32>() as *const i32,
                topk_lengths.ptr.cast::<i32>() as *const i32,
                partial.ptr.cast::<u16>(),
                partial_lse.ptr.cast::<f32>(),
                output.ptr.cast::<u16>(),
                output_lse.ptr.cast::<f32>(),
                query_rows,
                heads,
                topk,
                kv_row_stride_bytes,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_sparse_mla_bf16_async(
        &self,
        query: Ds41rtDeviceBuffer,
        kv_payload: Ds41rtDeviceBuffer,
        selected_indices: Ds41rtDeviceBuffer,
        topk_lengths: Ds41rtDeviceBuffer,
        partial: Ds41rtDeviceBuffer,
        partial_lse: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        output_lse: Ds41rtDeviceBuffer,
        query_rows: usize,
        heads: usize,
        topk: usize,
        kv_row_stride_bytes: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const RANK: usize = 512;
        const ROPE_DIM: usize = 64;
        const SPLITS: usize = 32;
        const SPLIT_QUERY_LIMIT: usize = 64;
        const BF16_ROW_BYTES: usize = (RANK + ROPE_DIM) * std::mem::size_of::<u16>();
        let context = "ds41rt_cuda_sparse_mla_bf16_async";
        if query_rows == 0
            || heads != 64
            || topk != 2048
            || kv_row_stride_bytes < BF16_ROW_BYTES
            || !scale.is_finite()
            || scale <= 0.0
        {
            anyhow::bail!(
                "{context} requires query_rows>0 heads=64 topk=2048 row_stride>=1152 and positive finite scale"
            );
        }
        validate_u16_buffer_values(
            &format!("{context} query"),
            query,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(RANK + ROPE_DIM))
                .context("sparse BF16 MLA query values overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} selected_indices"),
            selected_indices,
            query_rows
                .checked_mul(topk)
                .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
                .context("sparse BF16 MLA index bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} topk_lengths"),
            topk_lengths,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse BF16 MLA length bytes overflow")?,
        )?;
        validate_u16_buffer_values(
            &format!("{context} output"),
            output,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(RANK))
                .context("sparse BF16 MLA output values overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} output_lse"),
            output_lse,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("sparse BF16 MLA output-LSE bytes overflow")?,
        )?;
        if query_rows <= SPLIT_QUERY_LIMIT {
            validate_u16_buffer_values(
                &format!("{context} partial"),
                partial,
                query_rows
                    .checked_mul(heads)
                    .and_then(|values| values.checked_mul(SPLITS))
                    .and_then(|values| values.checked_mul(RANK))
                    .context("sparse BF16 MLA partial values overflow")?,
            )?;
            validate_device_buffer_bytes(
                &format!("{context} partial_lse"),
                partial_lse,
                query_rows
                    .checked_mul(heads)
                    .and_then(|values| values.checked_mul(SPLITS))
                    .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                    .context("sparse BF16 MLA partial-LSE bytes overflow")?,
            )?;
        }
        if kv_payload.ptr.is_null() || kv_payload.bytes < kv_row_stride_bytes {
            anyhow::bail!("{context} KV payload is empty or smaller than one row");
        }
        for buffer in [
            kv_payload,
            selected_indices,
            topk_lengths,
            partial,
            partial_lse,
            output,
            output_lse,
        ] {
            if buffer.device_id != query.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaSparseMlaBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_sparse_mla_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                query.ptr.cast::<u16>() as *const u16,
                kv_payload.ptr.cast::<u8>() as *const u8,
                selected_indices.ptr.cast::<i32>() as *const i32,
                topk_lengths.ptr.cast::<i32>() as *const i32,
                partial.ptr.cast::<u16>(),
                partial_lse.ptr.cast::<f32>(),
                output.ptr.cast::<u16>(),
                output_lse.ptr.cast::<f32>(),
                query_rows,
                heads,
                topk,
                kv_row_stride_bytes,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_sparse_mla_bf16_gather_kv_async(
        &self,
        kv_payload: Ds41rtDeviceBuffer,
        selected_indices: Ds41rtDeviceBuffer,
        topk_lengths: Ds41rtDeviceBuffer,
        gathered_k: Ds41rtDeviceBuffer,
        gathered_v: Ds41rtDeviceBuffer,
        query_rows: usize,
        topk: usize,
        kv_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const RANK: usize = 512;
        const HEAD_DIM: usize = 576;
        const BF16_ROW_BYTES: usize = HEAD_DIM * std::mem::size_of::<u16>();
        let context = "ds41rt_cuda_sparse_mla_bf16_gather_kv_async";
        if query_rows == 0 || topk != 2048 || kv_row_stride_bytes < BF16_ROW_BYTES {
            anyhow::bail!("{context} requires query_rows>0 topk=2048 and row_stride>=1152");
        }
        let selected_rows = query_rows
            .checked_mul(topk)
            .context("sparse BF16 MLA gathered row count overflow")?;
        validate_device_buffer_bytes(
            &format!("{context} selected_indices"),
            selected_indices,
            selected_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse BF16 MLA gathered-index bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} topk_lengths"),
            topk_lengths,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse BF16 MLA gathered-length bytes overflow")?,
        )?;
        validate_u16_buffer_values(
            &format!("{context} gathered_k"),
            gathered_k,
            selected_rows
                .checked_mul(HEAD_DIM)
                .context("sparse BF16 MLA gathered-K values overflow")?,
        )?;
        validate_u16_buffer_values(
            &format!("{context} gathered_v"),
            gathered_v,
            selected_rows
                .checked_mul(RANK)
                .context("sparse BF16 MLA gathered-V values overflow")?,
        )?;
        if kv_payload.ptr.is_null() || kv_payload.bytes < kv_row_stride_bytes {
            anyhow::bail!("{context} KV payload is empty or smaller than one row");
        }
        for buffer in [selected_indices, topk_lengths, gathered_k, gathered_v] {
            if buffer.device_id != kv_payload.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaSparseMlaBf16GatherKvAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_sparse_mla_bf16_gather_kv_async")?
        };
        let status = unsafe {
            kernel_fn(
                kv_payload.ptr.cast::<u8>() as *const u8,
                selected_indices.ptr.cast::<i32>() as *const i32,
                topk_lengths.ptr.cast::<i32>() as *const i32,
                gathered_k.ptr.cast::<u16>(),
                gathered_v.ptr.cast::<u16>(),
                query_rows,
                topk,
                kv_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_sparse_mla_bf16_softmax_async(
        &self,
        scores: Ds41rtDeviceBuffer,
        topk_lengths: Ds41rtDeviceBuffer,
        output_lse: Ds41rtDeviceBuffer,
        query_rows: usize,
        heads: usize,
        topk: usize,
        scale: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let context = "ds41rt_cuda_sparse_mla_bf16_softmax_async";
        if query_rows == 0 || heads != 64 || topk != 2048 || !scale.is_finite() || scale <= 0.0 {
            anyhow::bail!(
                "{context} requires query_rows>0 heads=64 topk=2048 and positive finite scale"
            );
        }
        validate_u16_buffer_values(
            &format!("{context} scores"),
            scores,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(topk))
                .context("sparse BF16 MLA score values overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} topk_lengths"),
            topk_lengths,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse BF16 MLA softmax-length bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} output_lse"),
            output_lse,
            query_rows
                .checked_mul(heads)
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("sparse BF16 MLA softmax-LSE bytes overflow")?,
        )?;
        for buffer in [topk_lengths, output_lse] {
            if buffer.device_id != scores.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaSparseMlaBf16SoftmaxAsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_sparse_mla_bf16_softmax_async")? };
        let status = unsafe {
            kernel_fn(
                scores.ptr.cast::<u16>(),
                topk_lengths.ptr.cast::<i32>() as *const i32,
                output_lse.ptr.cast::<f32>(),
                query_rows,
                heads,
                topk,
                scale,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_sparse_mla_nvfp4_gather_fp8_async(
        &self,
        nvfp4_kv: Ds41rtDeviceBuffer,
        selected_indices: Ds41rtDeviceBuffer,
        topk_lengths: Ds41rtDeviceBuffer,
        fp8_kv: Ds41rtDeviceBuffer,
        fp8_indices: Ds41rtDeviceBuffer,
        query_rows: usize,
        selected_index_stride: usize,
        staged_topk: usize,
        nvfp4_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NVFP4_ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        const FP8_ROW_BYTES: usize = 656;
        const MAX_QUERY_ROWS: usize = 64;
        let context = "ds41rt_cuda_sparse_mla_nvfp4_gather_fp8_async";
        if query_rows == 0
            || query_rows > MAX_QUERY_ROWS
            || staged_topk == 0
            || staged_topk > 2048
            || !staged_topk.is_multiple_of(64)
            || selected_index_stride < staged_topk
            || selected_index_stride > 2048
            || nvfp4_row_stride_bytes < NVFP4_ROW_BYTES
        {
            anyhow::bail!(
                "{context} requires query_rows in 1..=64, staged_topk in 64..=2048 \
                 divisible by 64, staged_topk<=selected_index_stride<=2048, and \
                 NVFP4 row stride>=432"
            );
        }
        let selected_rows = query_rows
            .checked_mul(selected_index_stride)
            .context("sparse NVFP4-to-FP8 selected rows overflow")?;
        let staged_rows = query_rows
            .checked_mul(staged_topk)
            .context("sparse NVFP4-to-FP8 staged rows overflow")?;
        validate_device_buffer_bytes(
            &format!("{context} selected_indices"),
            selected_indices,
            selected_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse NVFP4-to-FP8 index bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} topk_lengths"),
            topk_lengths,
            query_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse NVFP4-to-FP8 length bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} fp8_kv"),
            fp8_kv,
            staged_rows
                .checked_mul(FP8_ROW_BYTES)
                .context("sparse NVFP4-to-FP8 output bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} fp8_indices"),
            fp8_indices,
            staged_rows
                .checked_mul(std::mem::size_of::<i32>())
                .context("sparse NVFP4-to-FP8 output-index bytes overflow")?,
        )?;
        if nvfp4_kv.ptr.is_null() || nvfp4_kv.bytes < nvfp4_row_stride_bytes {
            anyhow::bail!("{context} NVFP4 KV payload is empty or smaller than one row");
        }
        for buffer in [selected_indices, topk_lengths, fp8_kv, fp8_indices] {
            if buffer.device_id != nvfp4_kv.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaSparseMlaNvfp4GatherFp8AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_sparse_mla_nvfp4_gather_fp8_async")?
        };
        let status = unsafe {
            kernel_fn(
                nvfp4_kv.ptr.cast::<u8>() as *const u8,
                selected_indices.ptr.cast::<i32>() as *const i32,
                topk_lengths.ptr.cast::<i32>() as *const i32,
                fp8_kv.ptr.cast::<u8>(),
                fp8_indices.ptr.cast::<i32>(),
                query_rows,
                selected_index_stride,
                staged_topk,
                nvfp4_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_mla_nvfp4_expand_fp8_paged_async(
        &self,
        nvfp4_kv: Ds41rtDeviceBuffer,
        physical_pages: Ds41rtDeviceBuffer,
        active_rows: Ds41rtDeviceBuffer,
        fp8_kv: Ds41rtDeviceBuffer,
        max_tokens: usize,
        page_size: usize,
        nvfp4_row_stride_bytes: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        const NVFP4_ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        const FP8_ROW_BYTES: usize = 656;
        let context = "ds41rt_cuda_mla_nvfp4_expand_fp8_paged_async";
        if max_tokens == 0
            || page_size == 0
            || max_tokens % page_size != 0
            || nvfp4_row_stride_bytes < NVFP4_ROW_BYTES
        {
            anyhow::bail!(
                "{context} requires positive page-aligned max_tokens and NVFP4 row stride>=432"
            );
        }
        validate_device_buffer_bytes(
            &format!("{context} nvfp4_kv"),
            nvfp4_kv,
            max_tokens
                .checked_mul(nvfp4_row_stride_bytes)
                .context("paged NVFP4 input bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} physical_pages"),
            physical_pages,
            (max_tokens / page_size)
                .checked_mul(std::mem::size_of::<u32>())
                .context("paged NVFP4 page-table bytes overflow")?,
        )?;
        validate_device_buffer_bytes(
            &format!("{context} active_rows"),
            active_rows,
            std::mem::size_of::<i32>(),
        )?;
        validate_device_buffer_bytes(
            &format!("{context} fp8_kv"),
            fp8_kv,
            max_tokens
                .checked_mul(FP8_ROW_BYTES)
                .context("paged FP8 output bytes overflow")?,
        )?;
        for buffer in [physical_pages, active_rows, fp8_kv] {
            if buffer.device_id != nvfp4_kv.device_id {
                anyhow::bail!("{context} buffers must reside on the same CUDA device");
            }
        }
        let kernel_fn: Symbol<CudaMlaNvfp4ExpandFp8PagedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_mla_nvfp4_expand_fp8_paged_async")?
        };
        let status = unsafe {
            kernel_fn(
                nvfp4_kv.ptr.cast::<u8>() as *const u8,
                physical_pages.ptr.cast::<u32>() as *const u32,
                active_rows.ptr.cast::<i32>() as *const i32,
                fp8_kv.ptr.cast::<u8>(),
                max_tokens,
                page_size,
                nvfp4_row_stride_bytes,
                cuda_stream,
            )
        };
        self.status_to_result(context, status)
    }

    pub fn cuda_embedding_lookup_f32(
        &self,
        embedding: Ds41rtDeviceBuffer,
        token_ids: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
    ) -> Result<()> {
        validate_embedding_lookup_buffers(
            "ds41rt_cuda_embedding_lookup_f32",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let kernel_fn: Symbol<CudaEmbeddingLookupF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_embedding_lookup_f32")? };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<f32>() as *const f32,
                token_ids.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<f32>(),
                rows,
                vocab,
                hidden,
            )
        };
        self.status_to_result("ds41rt_cuda_embedding_lookup_f32", status)
    }

    pub unsafe fn cuda_embedding_lookup_f32_async(
        &self,
        embedding: Ds41rtDeviceBuffer,
        token_ids: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_embedding_lookup_buffers(
            "ds41rt_cuda_embedding_lookup_f32_async",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let kernel_fn: Symbol<CudaEmbeddingLookupF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_embedding_lookup_f32_async")? };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<f32>() as *const f32,
                token_ids.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<f32>(),
                rows,
                vocab,
                hidden,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_embedding_lookup_f32_async", status)
    }

    pub fn cuda_embedding_lookup_bf16(
        &self,
        embedding: Ds41rtDeviceBuffer,
        token_ids: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
    ) -> Result<()> {
        validate_embedding_lookup_bf16_buffers(
            "ds41rt_cuda_embedding_lookup_bf16",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let kernel_fn: Symbol<CudaEmbeddingLookupBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_embedding_lookup_bf16")? };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<u16>() as *const u16,
                token_ids.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<u16>(),
                rows,
                vocab,
                hidden,
            )
        };
        self.status_to_result("ds41rt_cuda_embedding_lookup_bf16", status)
    }

    pub unsafe fn cuda_embedding_lookup_bf16_async(
        &self,
        embedding: Ds41rtDeviceBuffer,
        token_ids: Ds41rtDeviceBuffer,
        out: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        hidden: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_embedding_lookup_bf16_buffers(
            "ds41rt_cuda_embedding_lookup_bf16_async",
            embedding,
            token_ids,
            out,
            rows,
            vocab,
            hidden,
        )?;

        let kernel_fn: Symbol<CudaEmbeddingLookupBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_embedding_lookup_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                embedding.ptr.cast::<u16>() as *const u16,
                token_ids.ptr.cast::<u32>() as *const u32,
                out.ptr.cast::<u16>(),
                rows,
                vocab,
                hidden,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_embedding_lookup_bf16_async", status)
    }

    pub fn cuda_lm_head_argmax_bf16(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
    ) -> Result<()> {
        validate_lm_head_argmax_bf16_buffers(
            "ds41rt_cuda_lm_head_argmax_bf16",
            hidden,
            lm_head,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLmHeadArgmaxBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_lm_head_argmax_bf16")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_argmax_bf16", status)
    }

    pub unsafe fn cuda_lm_head_argmax_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_lm_head_argmax_bf16_buffers(
            "ds41rt_cuda_lm_head_argmax_bf16_async",
            hidden,
            lm_head,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLmHeadArgmaxBf16AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_lm_head_argmax_bf16_async")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_argmax_bf16_async", status)
    }

    /// Run the integrated dSpark terminal with one jointly issued request
    /// batch. Compact scratch and outputs are position-major; hidden storage
    /// remains indexed by stable request slot.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_dspark_terminal_greedy_bf16_async(
        &self,
        normalized_hidden_by_slot: Ds41rtDeviceBuffer,
        collapsed_hidden_by_slot: Ds41rtDeviceBuffer,
        shared_head: Ds41rtDeviceBuffer,
        markov_w1: Ds41rtDeviceBuffer,
        markov_w2: Ds41rtDeviceBuffer,
        confidence_weight: Ds41rtDeviceBuffer,
        active_slot_ids: Ds41rtDeviceBuffer,
        anchor_token_ids: Ds41rtDeviceBuffer,
        compact_normalized_hidden: Ds41rtDeviceBuffer,
        shared_logits: Ds41rtDeviceBuffer,
        markov_embeddings: Ds41rtDeviceBuffer,
        markov_logits: Ds41rtDeviceBuffer,
        output_token_ids: Ds41rtDeviceBuffer,
        conditional_confidence: Ds41rtDeviceBuffer,
        active_requests: usize,
        max_slots: usize,
        hidden_dim: usize,
        vocab: usize,
        markov_rank: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_dspark_terminal_greedy_bf16_buffers(
            "ds41rt_cuda_dspark_terminal_greedy_bf16_async",
            normalized_hidden_by_slot,
            collapsed_hidden_by_slot,
            shared_head,
            markov_w1,
            markov_w2,
            confidence_weight,
            active_slot_ids,
            anchor_token_ids,
            compact_normalized_hidden,
            shared_logits,
            markov_embeddings,
            markov_logits,
            output_token_ids,
            conditional_confidence,
            active_requests,
            max_slots,
            hidden_dim,
            vocab,
            markov_rank,
        )?;
        let kernel_fn: Symbol<CudaDsparkTerminalGreedyBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_dspark_terminal_greedy_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                normalized_hidden_by_slot.ptr.cast::<u16>() as *const u16,
                collapsed_hidden_by_slot.ptr.cast::<u16>() as *const u16,
                shared_head.ptr.cast::<u16>() as *const u16,
                markov_w1.ptr.cast::<u16>() as *const u16,
                markov_w2.ptr.cast::<u16>() as *const u16,
                confidence_weight.ptr.cast::<u16>() as *const u16,
                active_slot_ids.ptr.cast::<u32>() as *const u32,
                anchor_token_ids.ptr.cast::<u32>() as *const u32,
                compact_normalized_hidden.ptr.cast::<u16>(),
                shared_logits.ptr.cast::<f32>(),
                markov_embeddings.ptr.cast::<u16>(),
                markov_logits.ptr.cast::<f32>(),
                output_token_ids.ptr.cast::<u32>(),
                conditional_confidence.ptr.cast::<f32>(),
                active_requests,
                max_slots,
                hidden_dim,
                vocab,
                markov_rank,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_dspark_terminal_greedy_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_lm_head_sample_topk_topp_bf16(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        validate_lm_head_sample_topk_topp_bf16_buffers(
            "ds41rt_cuda_lm_head_sample_topk_topp_bf16",
            hidden,
            lm_head,
            random_uniforms,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadSampleTopKToppBf16Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_lm_head_sample_topk_topp_bf16")? };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_sample_topk_topp_bf16", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_lm_head_sample_topk_topp_bf16_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_lm_head_sample_topk_topp_bf16_buffers(
            "ds41rt_cuda_lm_head_sample_topk_topp_bf16_async",
            hidden,
            lm_head,
            random_uniforms,
            out_indices,
            out_scores,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadSampleTopKToppBf16AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_lm_head_sample_topk_topp_bf16_async")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_sample_topk_topp_bf16_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_argmax_indices: Ds41rtDeviceBuffer,
        out_argmax_scores: Ds41rtDeviceBuffer,
        out_sample_indices: Ds41rtDeviceBuffer,
        out_sample_scores: Ds41rtDeviceBuffer,
        logits_workspace: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        validate_lm_head_argmax_sample_topk_topp_bf16_staged_buffers(
            "ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged",
            hidden,
            lm_head,
            random_uniforms,
            out_argmax_indices,
            out_argmax_scores,
            out_sample_indices,
            out_sample_scores,
            logits_workspace,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadArgmaxSampleTopKToppBf16StagedFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_argmax_indices.ptr.cast::<u32>(),
                out_argmax_scores.ptr.cast::<f32>(),
                out_sample_indices.ptr.cast::<u32>(),
                out_sample_scores.ptr.cast::<f32>(),
                logits_workspace.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_argmax_indices: Ds41rtDeviceBuffer,
        out_argmax_scores: Ds41rtDeviceBuffer,
        out_sample_indices: Ds41rtDeviceBuffer,
        out_sample_scores: Ds41rtDeviceBuffer,
        logits_workspace: Ds41rtDeviceBuffer,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_lm_head_argmax_sample_topk_topp_bf16_staged_buffers(
            "ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async",
            hidden,
            lm_head,
            random_uniforms,
            out_argmax_indices,
            out_argmax_scores,
            out_sample_indices,
            out_sample_scores,
            logits_workspace,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadArgmaxSampleTopKToppBf16StagedAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_argmax_indices.ptr.cast::<u32>(),
                out_argmax_scores.ptr.cast::<f32>(),
                out_sample_indices.ptr.cast::<u32>(),
                out_sample_scores.ptr.cast::<f32>(),
                logits_workspace.ptr.cast::<f32>(),
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                cuda_stream,
            )
        };
        self.status_to_result(
            "ds41rt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async",
            status,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_lm_head_sample_topk_topp_bf16_cub(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        logits_workspace: Ds41rtDeviceBuffer,
        sorted_logits: Ds41rtDeviceBuffer,
        unsorted_indices: Ds41rtDeviceBuffer,
        sorted_indices: Ds41rtDeviceBuffer,
        segment_offsets: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        cub_temp_storage: Ds41rtDeviceBuffer,
        cub_temp_storage_bytes: usize,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        validate_lm_head_sample_topk_topp_bf16_cub_buffers(
            "ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub",
            hidden,
            lm_head,
            random_uniforms,
            logits_workspace,
            sorted_logits,
            unsorted_indices,
            sorted_indices,
            segment_offsets,
            out_indices,
            out_scores,
            cub_temp_storage,
            cub_temp_storage_bytes,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadSampleTopKToppBf16CubFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                logits_workspace.ptr.cast::<f32>(),
                sorted_logits.ptr.cast::<f32>(),
                unsorted_indices.ptr.cast::<u32>(),
                sorted_indices.ptr.cast::<u32>(),
                segment_offsets.ptr.cast::<i32>(),
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                cub_temp_storage.ptr,
                cub_temp_storage_bytes,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_lm_head_sample_topk_topp_bf16_cub_async(
        &self,
        hidden: Ds41rtDeviceBuffer,
        lm_head: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        logits_workspace: Ds41rtDeviceBuffer,
        sorted_logits: Ds41rtDeviceBuffer,
        unsorted_indices: Ds41rtDeviceBuffer,
        sorted_indices: Ds41rtDeviceBuffer,
        segment_offsets: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        cub_temp_storage: Ds41rtDeviceBuffer,
        cub_temp_storage_bytes: usize,
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_lm_head_sample_topk_topp_bf16_cub_buffers(
            "ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub_async",
            hidden,
            lm_head,
            random_uniforms,
            logits_workspace,
            sorted_logits,
            unsorted_indices,
            sorted_indices,
            segment_offsets,
            out_indices,
            out_scores,
            cub_temp_storage,
            cub_temp_storage_bytes,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLmHeadSampleTopKToppBf16CubAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub_async")?
        };
        let status = unsafe {
            kernel_fn(
                hidden.ptr.cast::<u16>() as *const u16,
                lm_head.ptr.cast::<u16>() as *const u16,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                logits_workspace.ptr.cast::<f32>(),
                sorted_logits.ptr.cast::<f32>(),
                unsorted_indices.ptr.cast::<u32>(),
                sorted_indices.ptr.cast::<u32>(),
                segment_offsets.ptr.cast::<i32>(),
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                cub_temp_storage.ptr,
                cub_temp_storage_bytes,
                rows,
                hidden_dim,
                vocab,
                temperature,
                top_k,
                top_p,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_lm_head_sample_topk_topp_bf16_cub_async", status)
    }

    pub fn cuda_logits_argmax_f32(
        &self,
        logits: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
    ) -> Result<()> {
        validate_logits_argmax_buffers(
            "ds41rt_cuda_logits_argmax_f32",
            logits,
            out_indices,
            out_scores,
            rows,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLogitsArgmaxF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_logits_argmax_f32")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_argmax_f32", status)
    }

    pub unsafe fn cuda_logits_argmax_f32_async(
        &self,
        logits: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_logits_argmax_buffers(
            "ds41rt_cuda_logits_argmax_f32_async",
            logits,
            out_indices,
            out_scores,
            rows,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLogitsArgmaxF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_logits_argmax_f32_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_argmax_f32_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_logits_argmax_checked_f32_async(
        &self,
        logits: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_logits_argmax_buffers(
            "ds41rt_cuda_logits_argmax_checked_f32_async",
            logits,
            out_indices,
            out_scores,
            rows,
            vocab,
        )?;

        let kernel_fn: Symbol<CudaLogitsArgmaxF32AsyncFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_logits_argmax_checked_f32_async")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_argmax_checked_f32_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_logits_sample_topk_topp_f32(
        &self,
        logits: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        validate_logits_sample_topk_topp_buffers(
            "ds41rt_cuda_logits_sample_topk_topp_f32",
            logits,
            random_uniforms,
            out_indices,
            out_scores,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLogitsSampleTopKToppF32Fn> =
            unsafe { self.lib.get(b"ds41rt_cuda_logits_sample_topk_topp_f32")? };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_sample_topk_topp_f32", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn cuda_logits_sample_topk_topp_f32_cub(
        &self,
        logits: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        sorted_logits: Ds41rtDeviceBuffer,
        unsorted_indices: Ds41rtDeviceBuffer,
        sorted_indices: Ds41rtDeviceBuffer,
        segment_offsets: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        cub_temp_storage: Ds41rtDeviceBuffer,
        cub_temp_storage_bytes: usize,
        rows: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<()> {
        validate_logits_sample_topk_topp_cub_buffers(
            "ds41rt_cuda_logits_sample_topk_topp_f32_cub",
            logits,
            random_uniforms,
            sorted_logits,
            unsorted_indices,
            sorted_indices,
            segment_offsets,
            out_indices,
            out_scores,
            cub_temp_storage,
            cub_temp_storage_bytes,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLogitsSampleTopKToppF32CubFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_logits_sample_topk_topp_f32_cub")?
        };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                sorted_logits.ptr.cast::<f32>(),
                unsorted_indices.ptr.cast::<u32>(),
                sorted_indices.ptr.cast::<u32>(),
                segment_offsets.ptr.cast::<i32>(),
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                cub_temp_storage.ptr,
                cub_temp_storage_bytes,
                rows,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_sample_topk_topp_f32_cub", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_logits_sample_topk_topp_f32_cub_async(
        &self,
        logits: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        sorted_logits: Ds41rtDeviceBuffer,
        unsorted_indices: Ds41rtDeviceBuffer,
        sorted_indices: Ds41rtDeviceBuffer,
        segment_offsets: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        cub_temp_storage: Ds41rtDeviceBuffer,
        cub_temp_storage_bytes: usize,
        rows: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_logits_sample_topk_topp_cub_buffers(
            "ds41rt_cuda_logits_sample_topk_topp_f32_cub_async",
            logits,
            random_uniforms,
            sorted_logits,
            unsorted_indices,
            sorted_indices,
            segment_offsets,
            out_indices,
            out_scores,
            cub_temp_storage,
            cub_temp_storage_bytes,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLogitsSampleTopKToppF32CubAsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_logits_sample_topk_topp_f32_cub_async")?
        };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                sorted_logits.ptr.cast::<f32>(),
                unsorted_indices.ptr.cast::<u32>(),
                sorted_indices.ptr.cast::<u32>(),
                segment_offsets.ptr.cast::<i32>(),
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                cub_temp_storage.ptr,
                cub_temp_storage_bytes,
                rows,
                vocab,
                temperature,
                top_k,
                top_p,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_sample_topk_topp_f32_cub_async", status)
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_logits_sample_topk_topp_f32_async(
        &self,
        logits: Ds41rtDeviceBuffer,
        random_uniforms: Ds41rtDeviceBuffer,
        out_indices: Ds41rtDeviceBuffer,
        out_scores: Ds41rtDeviceBuffer,
        rows: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        validate_logits_sample_topk_topp_buffers(
            "ds41rt_cuda_logits_sample_topk_topp_f32_async",
            logits,
            random_uniforms,
            out_indices,
            out_scores,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )?;

        let kernel_fn: Symbol<CudaLogitsSampleTopKToppF32AsyncFn> = unsafe {
            self.lib
                .get(b"ds41rt_cuda_logits_sample_topk_topp_f32_async")?
        };
        let status = unsafe {
            kernel_fn(
                logits.ptr.cast::<f32>() as *const f32,
                random_uniforms.ptr.cast::<f32>() as *const f32,
                out_indices.ptr.cast::<u32>(),
                out_scores.ptr.cast::<f32>(),
                rows,
                vocab,
                temperature,
                top_k,
                top_p,
                cuda_stream,
            )
        };
        self.status_to_result("ds41rt_cuda_logits_sample_topk_topp_f32_async", status)
    }

    pub fn cuda_pack_nibbles(
        &self,
        codes: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        let packed_count = count.div_ceil(2);
        if count > 0 {
            validate_device_buffer_bytes("ds41rt_cuda_pack_nibbles codes", codes, count)?;
            validate_device_buffer_bytes("ds41rt_cuda_pack_nibbles packed", packed, packed_count)?;
        }
        let pack_fn: Symbol<CudaPackNibblesFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_pack_nibbles")? };
        let status = unsafe {
            pack_fn(
                codes.ptr.cast::<u8>() as *const u8,
                packed.ptr.cast::<u8>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_pack_nibbles", status)
    }

    pub fn cuda_unpack_nibbles(
        &self,
        packed: Ds41rtDeviceBuffer,
        codes: Ds41rtDeviceBuffer,
        count: usize,
    ) -> Result<()> {
        let packed_count = count.div_ceil(2);
        if count > 0 {
            validate_device_buffer_bytes("ds41rt_cuda_unpack_nibbles packed", packed, packed_count)?;
            validate_device_buffer_bytes("ds41rt_cuda_unpack_nibbles codes", codes, count)?;
        }
        let unpack_fn: Symbol<CudaUnpackNibblesFn> =
            unsafe { self.lib.get(b"ds41rt_cuda_unpack_nibbles")? };
        let status = unsafe {
            unpack_fn(
                packed.ptr.cast::<u8>() as *const u8,
                codes.ptr.cast::<u8>(),
                count,
            )
        };
        self.status_to_result("ds41rt_cuda_unpack_nibbles", status)
    }

    pub fn rdma_device_info(&self) -> Result<Ds41rtRdmaDeviceInfo> {
        let info_fn: Symbol<RdmaDeviceInfoFn> = unsafe { self.lib.get(b"ds41rt_rdma_device_info")? };
        let mut info = Ds41rtRdmaDeviceInfo::default();
        let status = unsafe { info_fn(&mut info) };
        self.status_to_result("ds41rt_rdma_device_info", status)?;
        Ok(info)
    }

    pub fn rdma_plan_host_buffer_registration(
        &self,
        ptr: *const c_void,
        bytes: usize,
        alignment: usize,
    ) -> Result<Ds41rtRdmaHostBufferPlan> {
        let plan_fn: Symbol<RdmaPlanHostBufferRegistrationFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_plan_host_buffer_registration")? };
        let mut plan = Ds41rtRdmaHostBufferPlan::default();
        let status = unsafe { plan_fn(ptr, bytes, alignment, &mut plan) };
        self.status_to_result("ds41rt_rdma_plan_host_buffer_registration", status)?;
        Ok(plan)
    }

    pub fn rdma_register_host_buffer_probe(
        &self,
        buffer: &mut [u8],
    ) -> Result<Ds41rtRdmaRegisterProbe> {
        let register_fn: Symbol<RdmaRegisterHostBufferProbeFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_register_host_buffer_probe")? };
        let mut probe = Ds41rtRdmaRegisterProbe::default();
        let status = unsafe { register_fn(buffer.as_mut_ptr().cast(), buffer.len(), &mut probe) };
        self.status_to_result("ds41rt_rdma_register_host_buffer_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_create_rc_qp_probe(
        &self,
        port_num: u32,
        send_wr: u32,
        recv_wr: u32,
        max_sge: u32,
    ) -> Result<Ds41rtRdmaRcQpProbe> {
        let qp_fn: Symbol<RdmaCreateRcQpProbeFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_create_rc_qp_probe")? };
        let mut probe = Ds41rtRdmaRcQpProbe::default();
        let status = unsafe { qp_fn(port_num, send_wr, recv_wr, max_sge, &mut probe) };
        self.status_to_result("ds41rt_rdma_create_rc_qp_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_send_recv_loopback_probe(
        &self,
        port_num: u32,
        bytes: usize,
    ) -> Result<Ds41rtRdmaRcSendRecvProbe> {
        let probe_fn: Symbol<RdmaRcSendRecvLoopbackProbeFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_send_recv_loopback_probe")? };
        let mut probe = Ds41rtRdmaRcSendRecvProbe::default();
        let status = unsafe { probe_fn(port_num, bytes, &mut probe) };
        self.status_to_result("ds41rt_rdma_rc_send_recv_loopback_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_protocol_v2_loopback_probe(
        &self,
        port_num: u32,
        request_frame: &[u8],
        response_frame: &[u8],
    ) -> Result<Ds41rtRdmaRcProtocolV2LoopbackProbe> {
        let probe_fn: Symbol<RdmaRcProtocolV2LoopbackProbeFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_protocol_v2_loopback_probe")? };
        let mut probe = Ds41rtRdmaRcProtocolV2LoopbackProbe::default();
        let status = unsafe {
            probe_fn(
                port_num,
                request_frame.as_ptr().cast(),
                request_frame.len(),
                response_frame.as_ptr().cast(),
                response_frame.len(),
                &mut probe,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_protocol_v2_loopback_probe", status)?;
        Ok(probe)
    }

    pub fn rdma_rc_endpoint_create(
        &self,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
    ) -> Result<Ds41rtRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_create")? };
        let mut info = Ds41rtRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                &mut info,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_create", status)?;
        Ok(info)
    }

    pub fn rdma_rc_endpoint_create_with_buffer_flags(
        &self,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
        host_buffer_flags: u64,
    ) -> Result<Ds41rtRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateWithBufferFlagsFn> = unsafe {
            self.lib
                .get(b"ds41rt_rdma_rc_endpoint_create_with_buffer_flags")?
        };
        let mut info = Ds41rtRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                host_buffer_flags,
                &mut info,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_create_with_buffer_flags", status)?;
        Ok(info)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rdma_rc_endpoint_create_on_device_with_buffer_flags(
        &self,
        device_name: &str,
        port_num: u32,
        local_psn: u32,
        send_frame_bytes: usize,
        recv_frame_bytes: usize,
        send_registered_span_bytes: usize,
        recv_registered_span_bytes: usize,
        max_send_wr: u32,
        max_recv_wr: u32,
        max_sge: u32,
        host_buffer_flags: u64,
    ) -> Result<Ds41rtRdmaRcEndpointInfo> {
        let create_fn: Symbol<RdmaRcEndpointCreateOnDeviceWithBufferFlagsFn> = unsafe {
            self.lib
                .get(b"ds41rt_rdma_rc_endpoint_create_on_device_with_buffer_flags")?
        };
        let device_name = CString::new(device_name).context("RDMA device name contains NUL")?;
        let mut info = Ds41rtRdmaRcEndpointInfo::default();
        let status = unsafe {
            create_fn(
                device_name.as_ptr(),
                port_num,
                local_psn,
                send_frame_bytes,
                recv_frame_bytes,
                send_registered_span_bytes,
                recv_registered_span_bytes,
                max_send_wr,
                max_recv_wr,
                max_sge,
                host_buffer_flags,
                &mut info,
            )
        };
        self.status_to_result(
            "ds41rt_rdma_rc_endpoint_create_on_device_with_buffer_flags",
            status,
        )?;
        Ok(info)
    }

    pub fn rdma_rc_endpoint_buffer_view(
        &self,
        handle: *mut c_void,
        receive_buffer: bool,
    ) -> Result<Ds41rtRdmaRcEndpointBufferView> {
        let view_fn: Symbol<RdmaRcEndpointBufferViewFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_buffer_view")? };
        let mut view = Ds41rtRdmaRcEndpointBufferView::default();
        let status = unsafe { view_fn(handle, c_int::from(receive_buffer), &mut view) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_buffer_view", status)?;
        Ok(view)
    }

    pub fn rdma_rc_endpoint_connect(
        &self,
        handle: *mut c_void,
        remote_qp_num: u32,
        remote_psn: u32,
        remote_lid: u32,
        remote_gid_hex: &str,
    ) -> Result<()> {
        let connect_fn: Symbol<RdmaRcEndpointConnectFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_connect")? };
        let remote_gid_hex =
            CString::new(remote_gid_hex).context("RDMA remote GID contains nul byte")?;
        let status = unsafe {
            connect_fn(
                handle,
                remote_qp_num,
                remote_psn,
                remote_lid,
                remote_gid_hex.as_ptr(),
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_connect", status)
    }

    pub fn rdma_rc_endpoint_post_recv(
        &self,
        handle: *mut c_void,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let recv_fn: Symbol<RdmaRcEndpointPostRecvFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_post_recv")? };
        let status = unsafe { recv_fn(handle, bytes, wr_id) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_post_recv", status)
    }

    pub fn rdma_rc_endpoint_post_recv_at(
        &self,
        handle: *mut c_void,
        offset_bytes: usize,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let recv_fn: Symbol<RdmaRcEndpointPostRecvAtFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_post_recv_at")? };
        let status = unsafe { recv_fn(handle, offset_bytes, bytes, wr_id) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_post_recv_at", status)
    }

    pub fn rdma_rc_endpoint_post_send_at(
        &self,
        handle: *mut c_void,
        offset_bytes: usize,
        bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointPostSendAtFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_post_send_at")? };
        let status = unsafe { send_fn(handle, offset_bytes, bytes, wr_id) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_post_send_at", status)
    }

    pub fn rdma_rc_endpoint_send(
        &self,
        handle: *mut c_void,
        frame: &[u8],
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_send")? };
        let status = unsafe { send_fn(handle, frame.as_ptr().cast(), frame.len(), wr_id) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_send", status)
    }

    pub fn rdma_rc_endpoint_send_at(
        &self,
        handle: *mut c_void,
        frame: &[u8],
        offset_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendAtFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_send_at")? };
        let status = unsafe {
            send_fn(
                handle,
                frame.as_ptr().cast(),
                offset_bytes,
                frame.len(),
                wr_id,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_send_at", status)
    }

    pub fn rdma_rc_endpoint_send_parts_at(
        &self,
        handle: *mut c_void,
        prefix: &[u8],
        payload: &[u8],
        offset_bytes: usize,
        wr_id: u64,
    ) -> Result<()> {
        let send_fn: Symbol<RdmaRcEndpointSendPartsAtFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_send_parts_at")? };
        let status = unsafe {
            send_fn(
                handle,
                prefix.as_ptr().cast(),
                prefix.len(),
                payload.as_ptr().cast(),
                payload.len(),
                offset_bytes,
                wr_id,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_send_parts_at", status)
    }

    pub fn rdma_rc_endpoint_poll(
        &self,
        handle: *mut c_void,
        expected_send_completions: u32,
        expected_recv_completions: u32,
        max_poll_iterations: u32,
        active_event_poll_timeout_ms: u32,
    ) -> Result<Ds41rtRdmaRcCompletionStats> {
        let mut stats = Ds41rtRdmaRcCompletionStats::default();
        if let Ok(poll_fn) = unsafe {
            self.lib
                .get::<RdmaRcEndpointPollWithTimeoutFn>(b"ds41rt_rdma_rc_endpoint_poll_with_timeout")
        } {
            let status = unsafe {
                poll_fn(
                    handle,
                    expected_send_completions,
                    expected_recv_completions,
                    max_poll_iterations,
                    active_event_poll_timeout_ms,
                    &mut stats,
                )
            };
            self.status_to_result("ds41rt_rdma_rc_endpoint_poll_with_timeout", status)?;
        } else {
            let poll_fn: Symbol<RdmaRcEndpointPollFn> =
                unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_poll")? };
            let status = unsafe {
                poll_fn(
                    handle,
                    expected_send_completions,
                    expected_recv_completions,
                    max_poll_iterations,
                    &mut stats,
                )
            };
            self.status_to_result("ds41rt_rdma_rc_endpoint_poll", status)?;
        }
        Ok(stats)
    }

    pub fn rdma_rc_endpoint_try_poll(
        &self,
        handle: *mut c_void,
        max_send_completions: u32,
        max_recv_completions: u32,
    ) -> Result<Ds41rtRdmaRcCompletionStats> {
        let mut stats = Ds41rtRdmaRcCompletionStats::default();
        let status = unsafe {
            (self.rdma_rc_endpoint_try_poll_fn)(
                handle,
                max_send_completions,
                max_recv_completions,
                &mut stats,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_try_poll", status)?;
        Ok(stats)
    }

    pub fn rdma_rc_endpoint_copy_recv(
        &self,
        handle: *mut c_void,
        out: &mut [u8],
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<RdmaRcEndpointCopyRecvFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_copy_recv")? };
        let status = unsafe { copy_fn(handle, out.as_mut_ptr().cast(), out.len(), bytes) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_copy_recv", status)
    }

    pub fn rdma_rc_endpoint_copy_recv_at(
        &self,
        handle: *mut c_void,
        out: &mut [u8],
        offset_bytes: usize,
        bytes: usize,
    ) -> Result<()> {
        let copy_fn: Symbol<RdmaRcEndpointCopyRecvAtFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_copy_recv_at")? };
        let status = unsafe {
            copy_fn(
                handle,
                out.as_mut_ptr().cast(),
                out.len(),
                offset_bytes,
                bytes,
            )
        };
        self.status_to_result("ds41rt_rdma_rc_endpoint_copy_recv_at", status)
    }

    pub fn rdma_rc_endpoint_destroy(&self, handle: *mut c_void) -> Result<()> {
        let destroy_fn: Symbol<RdmaRcEndpointDestroyFn> =
            unsafe { self.lib.get(b"ds41rt_rdma_rc_endpoint_destroy")? };
        let status = unsafe { destroy_fn(handle) };
        self.status_to_result("ds41rt_rdma_rc_endpoint_destroy", status)
    }

    pub fn last_error(&self) -> Result<String> {
        let last_error_fn: Symbol<LastErrorFn> = unsafe { self.lib.get(b"ds41rt_last_error")? };
        let mut buf = vec![0 as c_char; 512];
        let status = unsafe { last_error_fn(buf.as_mut_ptr(), buf.len()) };
        if status != DS41RT_STATUS_OK {
            anyhow::bail!("ds41rt_last_error returned status {status}");
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    fn status_to_result(&self, context: &str, status: Ds41rtStatus) -> Result<()> {
        if status == DS41RT_STATUS_OK {
            return Ok(());
        }
        let last_error = self
            .last_error()
            .unwrap_or_else(|err| format!("last error unavailable: {err}"));
        anyhow::bail!("{context} returned status {status}: {last_error}");
    }
}

pub fn c_char_array_to_string(value: &[c_char]) -> String {
    let nul = value.iter().position(|ch| *ch == 0).unwrap_or(value.len());
    let bytes = value[..nul].iter().map(|ch| *ch as u8).collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn validate_positive_dim(context: &str, dim: i32) -> Result<()> {
    if dim <= 0 {
        anyhow::bail!("{context} must be positive, got {dim}");
    }
    Ok(())
}

fn validate_f32_rows(context: &str, rows: i32, hidden: i32) -> Result<()> {
    validate_positive_dim(&format!("{context} rows"), rows)?;
    validate_positive_dim(&format!("{context} hidden"), hidden)?;
    Ok(())
}

fn validate_f32_buffer_values(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(context, buffer, checked_f32_bytes(context, value_count)?)
}

fn validate_u32_buffer_values(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u32>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn validate_i32_buffer_values(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<i32>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn validate_u64_buffer_values(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u64>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn validate_u16_buffer_values(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    value_count: usize,
) -> Result<()> {
    validate_device_buffer_bytes(
        context,
        buffer,
        value_count
            .checked_mul(std::mem::size_of::<u16>())
            .with_context(|| format!("{context} byte count overflows usize"))?,
    )
}

fn checked_row_values(context: &str, rows: usize, row_width: usize) -> Result<usize> {
    rows.checked_mul(row_width)
        .with_context(|| format!("{context} row value count overflows usize"))
}

fn checked_f32_bytes(context: &str, value_count: usize) -> Result<usize> {
    value_count
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} byte count overflows usize"))
}

fn validate_row_gather_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_rows: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    let src_values = checked_row_values(&format!("{context} src"), src_rows, row_width)?;
    let dst_values = checked_row_values(&format!("{context} dst"), rows, row_width)?;
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    if src_rows == 0 {
        anyhow::bail!("{context} src_rows must be positive when rows and row_width are nonzero");
    }
    validate_f32_buffer_values(&format!("{context} src"), src, src_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} dst"), dst, dst_values)
}

fn validate_row_gather_bf16_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_rows: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    let src_values = checked_row_values(&format!("{context} src"), src_rows, row_width)?;
    let dst_values = checked_row_values(&format!("{context} dst"), rows, row_width)?;
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    if src_rows == 0 {
        anyhow::bail!("{context} src_rows must be positive when rows and row_width are nonzero");
    }
    validate_u16_buffer_values(&format!("{context} src"), src, src_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_u16_buffer_values(&format!("{context} dst"), dst, dst_values)
}

#[allow(clippy::too_many_arguments)]
fn validate_row_gather_f32_to_fp8_e4m3_row_scaled_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_rows: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 || src_rows == 0 {
        anyhow::bail!("{context} rows, row_width, and src_rows must be positive");
    }
    let minimum_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    if dst_row_stride_bytes < minimum_stride
        || dst_row_stride_bytes % std::mem::align_of::<f32>() != 0
    {
        anyhow::bail!(
            "{context} destination row stride {dst_row_stride_bytes} must be FP32-aligned and at least {minimum_stride} bytes"
        );
    }
    validate_f32_buffer_values(
        &format!("{context} src"),
        src,
        checked_row_values(&format!("{context} src"), src_rows, row_width)?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}

fn validate_bf16_rows_to_fp8_e4m3_row_scaled_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 {
        anyhow::bail!("{context} rows and row_width must be positive");
    }
    let minimum_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    if dst_row_stride_bytes < minimum_stride
        || dst_row_stride_bytes % std::mem::align_of::<f32>() != 0
    {
        anyhow::bail!(
            "{context} destination row stride {dst_row_stride_bytes} must be FP32-aligned and at least {minimum_stride} bytes"
        );
    }
    validate_u16_buffer_values(
        &format!("{context} src"),
        src,
        checked_row_values(&format!("{context} src"), rows, row_width)?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_combine_fp8_e4m3_row_scaled_buffers(
    context: &str,
    local: Ds41rtDeviceBuffer,
    peers: Ds41rtDeviceBuffer,
    peer_payload_stride_bytes: usize,
    peer_count: usize,
    peer_row_stride_bytes: usize,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 || peer_count == 0 {
        anyhow::bail!("{context} rows, row_width, and peer_count must be positive");
    }
    let minimum_row_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    let minimum_peer_payload = rows
        .checked_mul(peer_row_stride_bytes)
        .with_context(|| format!("{context} peer payload byte count overflows usize"))?;
    if peer_row_stride_bytes < minimum_row_stride
        || peer_row_stride_bytes % std::mem::align_of::<f32>() != 0
        || dst_row_stride_bytes < minimum_row_stride
        || dst_row_stride_bytes % std::mem::align_of::<f32>() != 0
        || peer_payload_stride_bytes < minimum_peer_payload
    {
        anyhow::bail!("{context} row or peer payload stride is invalid");
    }
    validate_f32_buffer_values(
        &format!("{context} local"),
        local,
        checked_row_values(&format!("{context} local"), rows, row_width)?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} peers"),
        peers,
        peer_count
            .checked_mul(peer_payload_stride_bytes)
            .with_context(|| format!("{context} peers byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_combine_bf16_fp8_e4m3_row_scaled_buffers(
    context: &str,
    local: Ds41rtDeviceBuffer,
    peers: Ds41rtDeviceBuffer,
    peer_payload_stride_bytes: usize,
    peer_count: usize,
    peer_row_stride_bytes: usize,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 || peer_count == 0 {
        anyhow::bail!("{context} rows, row_width, and peer_count must be positive");
    }
    let minimum_row_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    let minimum_peer_payload = rows
        .checked_mul(peer_row_stride_bytes)
        .with_context(|| format!("{context} peer payload byte count overflows usize"))?;
    if peer_row_stride_bytes < minimum_row_stride
        || peer_row_stride_bytes % std::mem::align_of::<f32>() != 0
        || dst_row_stride_bytes < minimum_row_stride
        || dst_row_stride_bytes % std::mem::align_of::<f32>() != 0
        || peer_payload_stride_bytes < minimum_peer_payload
    {
        anyhow::bail!("{context} row or peer payload stride is invalid");
    }
    validate_u16_buffer_values(
        &format!("{context} local"),
        local,
        checked_row_values(&format!("{context} local"), rows, row_width)?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} peers"),
        peers,
        peer_count
            .checked_mul(peer_payload_stride_bytes)
            .with_context(|| format!("{context} peers byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_row_gather_f32_to_nvfp4_e2m1_fp8_e4m3_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_rows: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    dst_row_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 || src_rows == 0 {
        anyhow::bail!("{context} rows and src_rows must be positive");
    }
    let minimum_stride = checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context, row_width)?;
    if dst_row_stride_bytes < minimum_stride {
        anyhow::bail!(
            "{context} destination row stride {dst_row_stride_bytes} must be at least {minimum_stride} bytes"
        );
    }
    validate_f32_buffer_values(
        &format!("{context} src"),
        src,
        checked_row_values(&format!("{context} src"), src_rows, row_width)?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_device_buffer_bytes(
        &format!("{context} dst"),
        dst,
        rows.checked_mul(dst_row_stride_bytes)
            .with_context(|| format!("{context} destination byte count overflows usize"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_row_prefix_copy_bf16_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_rows: usize,
    dst: Ds41rtDeviceBuffer,
    rows: usize,
    src_row_width: usize,
    dst_row_width: usize,
    prefix_width: usize,
    src_row_offset: usize,
) -> Result<()> {
    if rows == 0 || prefix_width == 0 {
        return Ok(());
    }
    if src_rows == 0 {
        anyhow::bail!("{context} src_rows must be positive when rows are nonzero");
    }
    if src_row_width == 0 {
        anyhow::bail!("{context} src_row_width must be positive");
    }
    if dst_row_width == 0 {
        anyhow::bail!("{context} dst_row_width must be positive");
    }
    if prefix_width > src_row_width {
        anyhow::bail!(
            "{context} prefix_width {prefix_width} exceeds src_row_width {src_row_width}"
        );
    }
    if prefix_width > dst_row_width {
        anyhow::bail!(
            "{context} prefix_width {prefix_width} exceeds dst_row_width {dst_row_width}"
        );
    }
    let end_src_row = src_row_offset
        .checked_add(rows)
        .with_context(|| format!("{context} source row range overflows usize"))?;
    if end_src_row > src_rows {
        anyhow::bail!(
            "{context} source rows {src_row_offset}..{end_src_row} exceed src_rows {src_rows}"
        );
    }
    let src_values = checked_row_values(&format!("{context} src"), src_rows, src_row_width)?;
    let dst_values = checked_row_values(&format!("{context} dst"), rows, dst_row_width)?;
    validate_u16_buffer_values(&format!("{context} src"), src, src_values)?;
    validate_u16_buffer_values(&format!("{context} dst"), dst, dst_values)
}

fn validate_f32_to_bf16_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    validate_f32_buffer_values(&format!("{context} src"), src, count)?;
    validate_u16_buffer_values(&format!("{context} dst"), dst, count)
}

fn validate_bf16_to_f32_accumulation_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    validate_u16_buffer_values(&format!("{context} src"), src, count)?;
    validate_f32_buffer_values(&format!("{context} dst"), dst, count)
}

fn validate_row_scatter_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    let src_values = checked_row_values(&format!("{context} src"), rows, row_width)?;
    let dst_values = checked_row_values(&format!("{context} dst"), dst_rows, row_width)?;
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    if dst_rows == 0 {
        anyhow::bail!("{context} dst_rows must be positive when rows and row_width are nonzero");
    }
    validate_f32_buffer_values(&format!("{context} src"), src, src_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} dst"), dst, dst_values)
}

fn validate_row_scatter_bf16_to_f32_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    let src_values = checked_row_values(&format!("{context} src"), rows, row_width)?;
    let dst_values = checked_row_values(&format!("{context} dst"), dst_rows, row_width)?;
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    if dst_rows == 0 {
        anyhow::bail!("{context} dst_rows must be positive when rows and row_width are nonzero");
    }
    validate_u16_buffer_values(&format!("{context} src"), src, src_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} dst"), dst, dst_values)
}

#[allow(clippy::too_many_arguments)]
fn validate_row_scatter_fp8_e4m3_row_scaled_to_f32_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_row_stride_bytes: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 || dst_rows == 0 {
        anyhow::bail!("{context} rows, row_width, and dst_rows must be positive");
    }
    let minimum_stride = row_width
        .checked_add(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} minimum row stride overflows usize"))?;
    if src_row_stride_bytes < minimum_stride
        || src_row_stride_bytes % std::mem::align_of::<f32>() != 0
    {
        anyhow::bail!(
            "{context} source row stride {src_row_stride_bytes} must be FP32-aligned and at least {minimum_stride} bytes"
        );
    }
    validate_device_buffer_bytes(
        &format!("{context} src"),
        src,
        rows.checked_mul(src_row_stride_bytes)
            .with_context(|| format!("{context} source byte count overflows usize"))?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_f32_buffer_values(
        &format!("{context} dst"),
        dst,
        checked_row_values(&format!("{context} dst"), dst_rows, row_width)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_row_scatter_nvfp4_e2m1_fp8_e4m3_to_f32_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    src_row_stride_bytes: usize,
    row_indices: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    if rows == 0 || dst_rows == 0 {
        anyhow::bail!("{context} rows and dst_rows must be positive");
    }
    let minimum_stride = checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context, row_width)?;
    if src_row_stride_bytes < minimum_stride {
        anyhow::bail!(
            "{context} source row stride {src_row_stride_bytes} must be at least {minimum_stride} bytes"
        );
    }
    validate_device_buffer_bytes(
        &format!("{context} src"),
        src,
        rows.checked_mul(src_row_stride_bytes)
            .with_context(|| format!("{context} source byte count overflows usize"))?,
    )?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, rows)?;
    validate_f32_buffer_values(
        &format!("{context} dst"),
        dst,
        checked_row_values(&format!("{context} dst"), dst_rows, row_width)?,
    )
}

fn checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context: &str, row_width: usize) -> Result<usize> {
    if row_width == 0 || row_width % 16 != 0 {
        anyhow::bail!("{context} row_width must be a positive multiple of 16, got {row_width}");
    }
    row_width
        .checked_div(2)
        .and_then(|packed| packed.checked_add(row_width / 16))
        .with_context(|| format!("{context} NVFP4 row byte count overflows usize"))
}

#[allow(clippy::too_many_arguments)]
fn validate_route_shard_reduction_buffers(
    context: &str,
    buffers: &Ds41rtRouteShardReductionBuffers,
    rows: usize,
    row_width: usize,
    peer_row_stride_bytes: usize,
    local_dtype: u32,
    peer_dtype: u32,
    peer_count: usize,
) -> Result<()> {
    if rows == 0 || row_width == 0 {
        anyhow::bail!("{context} rows and row_width must be positive");
    }
    if !(1..=buffers.peers.len()).contains(&peer_count) {
        anyhow::bail!(
            "{context} peer count {peer_count} must be in 1..={}",
            buffers.peers.len()
        );
    }
    let values = checked_row_values(context, rows, row_width)?;
    match local_dtype {
        DS41RT_ROUTE_SHARD_LOCAL_F32 => {
            validate_f32_buffer_values(&format!("{context} local_f32"), buffers.local, values)?;
        }
        DS41RT_ROUTE_SHARD_LOCAL_BF16 => {
            validate_u16_buffer_values(&format!("{context} local_bf16"), buffers.local, values)?;
        }
        other => anyhow::bail!("{context} unsupported local dtype {other}"),
    }
    validate_f32_buffer_values(&format!("{context} output_f32"), buffers.output_f32, values)?;
    let minimum_peer_stride = match peer_dtype {
        DS41RT_ROUTE_SHARD_WIRE_BF16 => row_width
            .checked_mul(std::mem::size_of::<u16>())
            .with_context(|| format!("{context} BF16 row byte count overflows usize"))?,
        DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED => {
            if peer_row_stride_bytes % std::mem::align_of::<f32>() != 0 {
                anyhow::bail!("{context} FP8 peer row stride must be FP32-aligned");
            }
            row_width
                .checked_add(std::mem::size_of::<f32>())
                .with_context(|| format!("{context} FP8 row byte count overflows usize"))?
        }
        DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3 => {
            checked_nvfp4_e2m1_fp8_e4m3_row_bytes(context, row_width)?
        }
        other => anyhow::bail!("{context} unsupported peer dtype {other}"),
    };
    if peer_row_stride_bytes < minimum_peer_stride {
        anyhow::bail!(
            "{context} peer row stride {peer_row_stride_bytes} is below {minimum_peer_stride}"
        );
    }
    let peer_bytes = rows
        .checked_mul(peer_row_stride_bytes)
        .with_context(|| format!("{context} peer byte count overflows usize"))?;
    for (index, peer) in buffers.peers[..peer_count].iter().copied().enumerate() {
        validate_device_buffer_bytes(&format!("{context} peer {index}"), peer, peer_bytes)?;
    }
    Ok(())
}

fn validate_row_scatter_bf16_weighted_to_f32_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    row_weights: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    dst_rows: usize,
    rows: usize,
    row_width: usize,
) -> Result<()> {
    validate_row_scatter_bf16_to_f32_buffers(
        context,
        src,
        row_indices,
        dst,
        dst_rows,
        rows,
        row_width,
    )?;
    if rows == 0 || row_width == 0 {
        return Ok(());
    }
    validate_f32_buffer_values(&format!("{context} row_weights"), row_weights, rows)
}

fn validate_kv_cache_write_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    cache: Ds41rtDeviceBuffer,
    cache_offset_bytes: usize,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    validate_device_buffer_bytes(&format!("{context} src"), src, bytes)?;
    validate_device_buffer_range(
        &format!("{context} cache"),
        cache,
        cache_offset_bytes,
        bytes,
    )
}

fn validate_kv_cache_read_buffers(
    context: &str,
    cache: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    cache_offset_bytes: usize,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    validate_device_buffer_range(
        &format!("{context} cache"),
        cache,
        cache_offset_bytes,
        bytes,
    )?;
    validate_device_buffer_bytes(&format!("{context} dst"), dst, bytes)
}

fn validate_kv_cache_write_block_buffers(
    context: &str,
    src: Ds41rtDeviceBuffer,
    cache: Ds41rtDeviceBuffer,
    src_offsets: Ds41rtDeviceBuffer,
    cache_offsets: Ds41rtDeviceBuffer,
    block_bytes: Ds41rtDeviceBuffer,
    block_count: usize,
) -> Result<()> {
    if block_count == 0 {
        return Ok(());
    }
    validate_device_buffer_present(&format!("{context} src"), src)?;
    validate_device_buffer_present(&format!("{context} cache"), cache)?;
    validate_u64_buffer_values(&format!("{context} src_offsets"), src_offsets, block_count)?;
    validate_u64_buffer_values(
        &format!("{context} cache_offsets"),
        cache_offsets,
        block_count,
    )?;
    validate_u64_buffer_values(&format!("{context} block_bytes"), block_bytes, block_count)
}

fn validate_kv_cache_read_block_buffers(
    context: &str,
    cache: Ds41rtDeviceBuffer,
    dst: Ds41rtDeviceBuffer,
    cache_offsets: Ds41rtDeviceBuffer,
    dst_offsets: Ds41rtDeviceBuffer,
    block_bytes: Ds41rtDeviceBuffer,
    block_count: usize,
) -> Result<()> {
    if block_count == 0 {
        return Ok(());
    }
    validate_device_buffer_present(&format!("{context} cache"), cache)?;
    validate_device_buffer_present(&format!("{context} dst"), dst)?;
    validate_u64_buffer_values(
        &format!("{context} cache_offsets"),
        cache_offsets,
        block_count,
    )?;
    validate_u64_buffer_values(&format!("{context} dst_offsets"), dst_offsets, block_count)?;
    validate_u64_buffer_values(&format!("{context} block_bytes"), block_bytes, block_count)
}

#[allow(clippy::too_many_arguments)]
fn validate_mla_kv_cache_unpack_bf16_buffers(
    context: &str,
    payload: Ds41rtDeviceBuffer,
    kv_latent: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    dsa_key: Option<Ds41rtDeviceBuffer>,
    rows: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    dsa_dim: usize,
    payload_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if kv_lora_rank == 0 {
        anyhow::bail!("{context} kv_lora_rank must be positive");
    }
    if rope_dim == 0 {
        anyhow::bail!("{context} rope_dim must be positive");
    }
    if payload_stride_bytes == 0 || payload_stride_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!("{context} payload_stride_bytes must be positive and BF16-aligned");
    }
    let packed_width = kv_lora_rank
        .checked_add(rope_dim)
        .and_then(|width| width.checked_add(dsa_dim))
        .with_context(|| format!("{context} packed KV width overflows usize"))?;
    let payload_stride_values = payload_stride_bytes / std::mem::size_of::<u16>();
    if payload_stride_values < packed_width {
        anyhow::bail!(
            "{context} payload stride is too small: stride_values={payload_stride_values} packed_width={packed_width}"
        );
    }
    let payload_bytes = rows
        .checked_mul(payload_stride_bytes)
        .with_context(|| format!("{context} payload byte count overflows usize"))?;
    validate_device_buffer_bytes(&format!("{context} payload"), payload, payload_bytes)?;
    validate_u16_buffer_values(
        &format!("{context} kv_latent"),
        kv_latent,
        checked_row_values(&format!("{context} kv_latent"), rows, kv_lora_rank)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} k_rope"),
        k_rope,
        checked_row_values(&format!("{context} k_rope"), rows, rope_dim)?,
    )?;
    match (dsa_dim, dsa_key) {
        (0, _) => Ok(()),
        (_, Some(buffer)) => validate_u16_buffer_values(
            &format!("{context} dsa_key"),
            buffer,
            checked_row_values(&format!("{context} dsa_key"), rows, dsa_dim)?,
        ),
        (_, None) => anyhow::bail!("{context} dsa_key buffer is required when dsa_dim > 0"),
    }
}

fn validate_mla_kv_projected_split_bf16_buffers(
    context: &str,
    projected: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    v_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if nope_dim == 0 {
        anyhow::bail!("{context} nope_dim must be positive");
    }
    if v_dim == 0 {
        anyhow::bail!("{context} v_dim must be positive");
    }
    let row_heads = rows
        .checked_mul(heads)
        .with_context(|| format!("{context} row-head count overflows usize"))?;
    let projected_width = nope_dim
        .checked_add(v_dim)
        .with_context(|| format!("{context} projected head width overflows usize"))?;
    validate_u16_buffer_values(
        &format!("{context} projected"),
        projected,
        checked_row_values(&format!("{context} projected"), row_heads, projected_width)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} k_nope"),
        k_nope,
        checked_row_values(&format!("{context} k_nope"), row_heads, nope_dim)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} v"),
        v,
        checked_row_values(&format!("{context} v"), row_heads, v_dim)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_mla_kv_prepare_bf16_buffers(
    context: &str,
    projected: Ds41rtDeviceBuffer,
    positions: Ds41rtDeviceBuffer,
    norm_weight: Ds41rtDeviceBuffer,
    prepared: Ds41rtDeviceBuffer,
    rows: usize,
    projected_stride_bytes: usize,
    prepared_stride_bytes: usize,
    eps: f32,
    theta: f32,
) -> Result<()> {
    const KV_LORA_RANK: usize = 512;
    const ROPE_DIM: usize = 64;
    let minimum_stride_bytes = (KV_LORA_RANK + ROPE_DIM)
        .checked_mul(std::mem::size_of::<u16>())
        .context("MLA KV prepare minimum stride overflows usize")?;
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    for (label, stride) in [
        ("projected", projected_stride_bytes),
        ("prepared", prepared_stride_bytes),
    ] {
        if stride < minimum_stride_bytes || stride % std::mem::size_of::<u16>() != 0 {
            anyhow::bail!(
                "{context} {label} stride must be BF16-aligned and at least {minimum_stride_bytes} bytes"
            );
        }
    }
    if !eps.is_finite() || eps <= 0.0 {
        anyhow::bail!("{context} eps must be finite and positive");
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("{context} theta must be finite and positive");
    }
    validate_device_buffer_bytes(
        &format!("{context} projected"),
        projected,
        rows.checked_mul(projected_stride_bytes)
            .with_context(|| format!("{context} projected bytes overflow usize"))?,
    )?;
    validate_u32_buffer_values(&format!("{context} positions"), positions, rows)?;
    validate_u16_buffer_values(&format!("{context} norm_weight"), norm_weight, KV_LORA_RANK)?;
    validate_device_buffer_bytes(
        &format!("{context} prepared"),
        prepared,
        rows.checked_mul(prepared_stride_bytes)
            .with_context(|| format!("{context} prepared bytes overflow usize"))?,
    )
}

fn validate_transpose_rows_heads_bf16_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    width: usize,
) -> Result<()> {
    const VECTOR_VALUES: usize = 16 / std::mem::size_of::<u16>();
    if rows == 0 || heads == 0 || width == 0 || width % VECTOR_VALUES != 0 {
        anyhow::bail!(
            "{context} requires positive rows/heads/width and a width divisible by {VECTOR_VALUES}"
        );
    }
    let bytes = rows
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(width))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} bytes overflow usize"))?;
    validate_device_buffer_bytes(&format!("{context} input"), input, bytes)?;
    validate_device_buffer_bytes(&format!("{context} output"), output, bytes)?;
    if input.device_id != output.device_id {
        anyhow::bail!(
            "{context} output is on CUDA device {}, expected {}",
            output.device_id,
            input.device_id
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_mla_compose_absorbed_query_bf16_buffers(
    context: &str,
    latent_heads_rows: Ds41rtDeviceBuffer,
    rope_rows_heads: Ds41rtDeviceBuffer,
    output_rows_heads: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    latent_width: usize,
    rope_width: usize,
) -> Result<()> {
    const VECTOR_VALUES: usize = 16 / std::mem::size_of::<u16>();
    if rows == 0
        || heads == 0
        || latent_width == 0
        || rope_width == 0
        || latent_width % VECTOR_VALUES != 0
        || rope_width % VECTOR_VALUES != 0
    {
        anyhow::bail!(
            "{context} requires positive dimensions and latent/rope widths divisible by {VECTOR_VALUES}"
        );
    }
    let shape_bytes = |width: usize, label: &str| {
        rows.checked_mul(heads)
            .and_then(|values| values.checked_mul(width))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{context} {label} bytes overflow usize"))
    };
    validate_device_buffer_bytes(
        &format!("{context} latent_heads_rows"),
        latent_heads_rows,
        shape_bytes(latent_width, "latent")?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} rope_rows_heads"),
        rope_rows_heads,
        shape_bytes(rope_width, "rope")?,
    )?;
    let output_width = latent_width
        .checked_add(rope_width)
        .with_context(|| format!("{context} output width overflow usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} output_rows_heads"),
        output_rows_heads,
        shape_bytes(output_width, "output")?,
    )?;
    for (label, buffer) in [
        ("rope_rows_heads", rope_rows_heads),
        ("output_rows_heads", output_rows_heads),
    ] {
        if buffer.device_id != latent_heads_rows.device_id {
            anyhow::bail!(
                "{context} {label} is on CUDA device {}, expected {}",
                buffer.device_id,
                latent_heads_rows.device_id
            );
        }
    }
    Ok(())
}

fn validate_generic_kv_page_table_buffer(
    context: &str,
    page_table: Ds41rtDeviceBuffer,
    query_rows: usize,
    page_table_width: usize,
) -> Result<()> {
    if query_rows == 0 || page_table_width == 0 || page_table_width > i32::MAX as usize {
        anyhow::bail!("{context} query_rows/page_table_width are invalid");
    }
    let entries = query_rows
        .checked_mul(page_table_width)
        .with_context(|| format!("{context} entry count overflow usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} page_table"),
        page_table,
        entries
            .checked_mul(std::mem::size_of::<i32>())
            .with_context(|| format!("{context} bytes overflow usize"))?,
    )
}

fn validate_generic_kv_page_table_expand_indices_buffers(
    context: &str,
    output_indices: Ds41rtDeviceBuffer,
    physical_pages: Ds41rtDeviceBuffer,
    query_rows: usize,
    output_width: usize,
    active_tokens: usize,
) -> Result<()> {
    if query_rows == 0
        || output_width == 0
        || active_tokens == 0
        || active_tokens > output_width
        || active_tokens > i32::MAX as usize
    {
        anyhow::bail!("{context} geometry is invalid");
    }
    let output_entries = query_rows
        .checked_mul(output_width)
        .with_context(|| format!("{context} output entry count overflow usize"))?;
    let physical_page_count = active_tokens
        .checked_add(DS41RT_CUDA_GENERIC_KV_PAGE_SIZE - 1)
        .with_context(|| format!("{context} physical page count overflow usize"))?
        / DS41RT_CUDA_GENERIC_KV_PAGE_SIZE;
    validate_device_buffer_bytes(
        &format!("{context} output_indices"),
        output_indices,
        output_entries
            .checked_mul(std::mem::size_of::<i32>())
            .with_context(|| format!("{context} output bytes overflow usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} physical_pages"),
        physical_pages,
        physical_page_count
            .checked_mul(std::mem::size_of::<u32>())
            .with_context(|| format!("{context} physical-page bytes overflow usize"))?,
    )?;
    if output_indices.device_id != physical_pages.device_id {
        anyhow::bail!("{context} buffers must be on one CUDA device");
    }
    Ok(())
}

fn validate_mla_kv_fp8_ds_mla_strides(
    context: &str,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if projected_stride_bytes == 0 || projected_stride_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!("{context} projected_stride_bytes must be positive and BF16-aligned");
    }
    if packed_stride_bytes < DS41RT_CUDA_MLA_FP8_DS_PACKED_BYTES {
        anyhow::bail!(
            "{context} packed_stride_bytes must be at least {}, got {packed_stride_bytes}",
            DS41RT_CUDA_MLA_FP8_DS_PACKED_BYTES
        );
    }
    let projected_stride_values = projected_stride_bytes / std::mem::size_of::<u16>();
    if projected_stride_values < DS41RT_CUDA_MLA_FP8_DS_PROJECTED_VALUES {
        anyhow::bail!(
            "{context} projected_stride_bytes must hold at least {} BF16 values, got {projected_stride_values}",
            DS41RT_CUDA_MLA_FP8_DS_PROJECTED_VALUES
        );
    }
    Ok(())
}

fn validate_mla_kv_fp8_ds_mla_pack_buffers(
    context: &str,
    projected: Ds41rtDeviceBuffer,
    packed: Ds41rtDeviceBuffer,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Result<()> {
    validate_mla_kv_fp8_ds_mla_strides(context, rows, projected_stride_bytes, packed_stride_bytes)?;
    validate_device_buffer_bytes(
        &format!("{context} projected"),
        projected,
        rows.checked_mul(projected_stride_bytes)
            .with_context(|| format!("{context} projected byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} packed"),
        packed,
        rows.checked_mul(packed_stride_bytes)
            .with_context(|| format!("{context} packed byte count overflows usize"))?,
    )
}

fn validate_mla_kv_fp8_ds_mla_unpack_buffers(
    context: &str,
    packed: Ds41rtDeviceBuffer,
    projected: Ds41rtDeviceBuffer,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
) -> Result<()> {
    validate_mla_kv_fp8_ds_mla_strides(context, rows, projected_stride_bytes, packed_stride_bytes)?;
    validate_device_buffer_bytes(
        &format!("{context} packed"),
        packed,
        rows.checked_mul(packed_stride_bytes)
            .with_context(|| format!("{context} packed byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} projected"),
        projected,
        rows.checked_mul(projected_stride_bytes)
            .with_context(|| format!("{context} projected byte count overflows usize"))?,
    )
}

fn validate_mla_kv_mxfp4_ds_mla_strides(
    context: &str,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if projected_stride_bytes == 0 || projected_stride_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!("{context} projected_stride_bytes must be positive and BF16-aligned");
    }
    if packed_stride_bytes < DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES {
        anyhow::bail!(
            "{context} packed_stride_bytes must be at least {}, got {packed_stride_bytes}",
            DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES
        );
    }
    let projected_stride_values = projected_stride_bytes / std::mem::size_of::<u16>();
    if projected_stride_values < DS41RT_CUDA_MLA_MXFP4_DS_PROJECTED_VALUES {
        anyhow::bail!(
            "{context} projected_stride_bytes must hold at least {} BF16 values, got {projected_stride_values}",
            DS41RT_CUDA_MLA_MXFP4_DS_PROJECTED_VALUES
        );
    }
    Ok(())
}

fn validate_mla_kv_mxfp4_ds_mla_pack_buffers(
    context: &str,
    projected: Ds41rtDeviceBuffer,
    packed: Ds41rtDeviceBuffer,
    rows: usize,
    projected_stride_bytes: usize,
    packed_stride_bytes: usize,
) -> Result<()> {
    validate_mla_kv_mxfp4_ds_mla_strides(
        context,
        rows,
        projected_stride_bytes,
        packed_stride_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} projected"),
        projected,
        rows.checked_mul(projected_stride_bytes)
            .with_context(|| format!("{context} projected byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} packed"),
        packed,
        rows.checked_mul(packed_stride_bytes)
            .with_context(|| format!("{context} packed byte count overflows usize"))?,
    )
}

fn validate_mla_kv_mxfp4_ds_mla_unpack_buffers(
    context: &str,
    packed: Ds41rtDeviceBuffer,
    projected: Ds41rtDeviceBuffer,
    rows: usize,
    packed_stride_bytes: usize,
    projected_stride_bytes: usize,
) -> Result<()> {
    validate_mla_kv_mxfp4_ds_mla_strides(
        context,
        rows,
        projected_stride_bytes,
        packed_stride_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} packed"),
        packed,
        rows.checked_mul(packed_stride_bytes)
            .with_context(|| format!("{context} packed byte count overflows usize"))?,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} projected"),
        projected,
        rows.checked_mul(projected_stride_bytes)
            .with_context(|| format!("{context} projected byte count overflows usize"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_router_topk_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Ds41rtDeviceBuffer,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if experts == 0 {
        anyhow::bail!("{context} experts must be positive");
    }
    if top_k == 0 || top_k > experts || top_k > DS41RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "{context} top_k must be in 1..=min(experts, {DS41RT_CUDA_ROUTER_TOPK_MAX_K}), got top_k={top_k} experts={experts}"
        );
    }
    let hidden_values = checked_row_values(&format!("{context} hidden"), rows, hidden_dim)?;
    let weight_values =
        checked_row_values(&format!("{context} router_weight"), experts, hidden_dim)?;
    let topk_values = checked_row_values(&format!("{context} topk"), rows, top_k)?;
    validate_f32_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_f32_buffer_values(
        &format!("{context} router_weight"),
        router_weight,
        weight_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} correction_bias"),
        correction_bias,
        experts,
    )?;
    validate_u32_buffer_values(
        &format!("{context} topk_indices"),
        topk_indices,
        topk_values,
    )?;
    validate_f32_buffer_values(&format!("{context} topk_scores"), topk_scores, topk_values)?;
    validate_f32_buffer_values(
        &format!("{context} topk_weights"),
        topk_weights,
        topk_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_router_topk_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Ds41rtDeviceBuffer,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if experts == 0 {
        anyhow::bail!("{context} experts must be positive");
    }
    if top_k == 0 || top_k > experts || top_k > DS41RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "{context} top_k must be in 1..=min(experts, {DS41RT_CUDA_ROUTER_TOPK_MAX_K}), got top_k={top_k} experts={experts}"
        );
    }
    let hidden_values = checked_row_values(&format!("{context} hidden"), rows, hidden_dim)?;
    let weight_values =
        checked_row_values(&format!("{context} router_weight"), experts, hidden_dim)?;
    let topk_values = checked_row_values(&format!("{context} topk"), rows, top_k)?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u16_buffer_values(
        &format!("{context} router_weight"),
        router_weight,
        weight_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} correction_bias"),
        correction_bias,
        experts,
    )?;
    validate_u32_buffer_values(
        &format!("{context} topk_indices"),
        topk_indices,
        topk_values,
    )?;
    validate_f32_buffer_values(&format!("{context} topk_scores"), topk_scores, topk_values)?;
    validate_f32_buffer_values(
        &format!("{context} topk_weights"),
        topk_weights,
        topk_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_ds4_flash_router_topk_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Option<Ds41rtDeviceBuffer>,
    token_to_experts: Option<Ds41rtDeviceBuffer>,
    token_ids: Option<Ds41rtDeviceBuffer>,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
    hash_routing: bool,
) -> Result<()> {
    const HIDDEN: usize = 4096;
    const EXPERTS: usize = 256;
    const TOP_K: usize = 6;
    const VOCAB: usize = 129_280;
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    validate_u16_buffer_values(
        &format!("{context} hidden"),
        hidden,
        checked_row_values(&format!("{context} hidden"), rows, HIDDEN)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} router_weight"),
        router_weight,
        EXPERTS * HIDDEN,
    )?;
    if hash_routing {
        validate_u64_buffer_values(
            &format!("{context} token_to_experts"),
            token_to_experts.context("Flash hash router requires token_to_experts")?,
            VOCAB * TOP_K,
        )?;
        validate_u64_buffer_values(
            &format!("{context} token_ids"),
            token_ids.context("Flash hash router requires token_ids")?,
            rows,
        )?;
    } else {
        validate_f32_buffer_values(
            &format!("{context} correction_bias"),
            correction_bias.context("Flash learned router requires correction_bias")?,
            EXPERTS,
        )?;
    }
    let topk_values = checked_row_values(&format!("{context} topk"), rows, TOP_K)?;
    let score_values = checked_row_values(
        &format!("{context} score workspace"),
        rows,
        TOP_K + if hash_routing { 0 } else { EXPERTS },
    )?;
    validate_u32_buffer_values(
        &format!("{context} topk_indices"),
        topk_indices,
        topk_values,
    )?;
    validate_f32_buffer_values(&format!("{context} topk_scores"), topk_scores, score_values)?;
    validate_f32_buffer_values(
        &format!("{context} topk_weights"),
        topk_weights,
        topk_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_ds4_flash_router_refine_topk_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Ds41rtDeviceBuffer,
    approximate_logits: Ds41rtDeviceBuffer,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
) -> Result<()> {
    const HIDDEN: usize = 4_096;
    const EXPERTS: usize = 256;
    const TOP_K: usize = 6;
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    validate_u16_buffer_values(
        &format!("{context} hidden"),
        hidden,
        checked_row_values(&format!("{context} hidden"), rows, HIDDEN)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} router_weight"),
        router_weight,
        EXPERTS * HIDDEN,
    )?;
    validate_f32_buffer_values(
        &format!("{context} correction_bias"),
        correction_bias,
        EXPERTS,
    )?;
    validate_f32_buffer_values(
        &format!("{context} approximate_logits"),
        approximate_logits,
        checked_row_values(&format!("{context} approximate logits"), rows, EXPERTS)?,
    )?;
    let topk_values = checked_row_values(&format!("{context} topk"), rows, TOP_K)?;
    validate_u32_buffer_values(
        &format!("{context} topk_indices"),
        topk_indices,
        topk_values,
    )?;
    validate_f32_buffer_values(&format!("{context} topk_scores"), topk_scores, topk_values)?;
    validate_f32_buffer_values(
        &format!("{context} topk_weights"),
        topk_weights,
        topk_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_ds4_pro_router_topk_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    router_weight: Ds41rtDeviceBuffer,
    correction_bias: Option<Ds41rtDeviceBuffer>,
    token_to_experts: Option<Ds41rtDeviceBuffer>,
    token_ids: Option<Ds41rtDeviceBuffer>,
    topk_indices: Ds41rtDeviceBuffer,
    topk_scores: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
    rows: usize,
    hash_routing: bool,
) -> Result<()> {
    const HIDDEN: usize = 7_168;
    const EXPERTS: usize = 384;
    const TOP_K: usize = 6;
    const VOCAB: usize = 129_280;
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    validate_u16_buffer_values(
        &format!("{context} hidden"),
        hidden,
        checked_row_values(&format!("{context} hidden"), rows, HIDDEN)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} router_weight"),
        router_weight,
        EXPERTS * HIDDEN,
    )?;
    if hash_routing {
        validate_u64_buffer_values(
            &format!("{context} token_to_experts"),
            token_to_experts.context("Pro hash router requires token_to_experts")?,
            VOCAB * TOP_K,
        )?;
        validate_u64_buffer_values(
            &format!("{context} token_ids"),
            token_ids.context("Pro hash router requires token_ids")?,
            rows,
        )?;
    } else {
        validate_f32_buffer_values(
            &format!("{context} correction_bias"),
            correction_bias.context("Pro learned router requires correction_bias")?,
            EXPERTS,
        )?;
    }
    let topk_values = checked_row_values(&format!("{context} topk"), rows, TOP_K)?;
    let score_values = checked_row_values(
        &format!("{context} score workspace"),
        rows,
        TOP_K + if hash_routing { 0 } else { EXPERTS },
    )?;
    validate_u32_buffer_values(
        &format!("{context} topk_indices"),
        topk_indices,
        topk_values,
    )?;
    validate_f32_buffer_values(&format!("{context} topk_scores"), topk_scores, score_values)?;
    validate_f32_buffer_values(
        &format!("{context} topk_weights"),
        topk_weights,
        topk_values,
    )
}

fn validate_mlp_rows_buffers(
    context: &str,
    x: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    if intermediate == 0 {
        anyhow::bail!("{context} intermediate must be positive");
    }
    let row_values = checked_row_values(&format!("{context} rows"), rows, hidden)?;
    let gate_values = checked_row_values(&format!("{context} gate_weight"), intermediate, hidden)?;
    let down_values = checked_row_values(&format!("{context} down_weight"), hidden, intermediate)?;
    validate_f32_buffer_values(&format!("{context} x"), x, row_values)?;
    validate_f32_buffer_values(&format!("{context} gate_weight"), gate_weight, gate_values)?;
    validate_f32_buffer_values(&format!("{context} up_weight"), up_weight, gate_values)?;
    validate_f32_buffer_values(&format!("{context} down_weight"), down_weight, down_values)?;
    validate_f32_buffer_values(&format!("{context} out"), out, row_values)
}

fn validate_mlp_rows_bf16_buffers(
    context: &str,
    x: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    if intermediate == 0 {
        anyhow::bail!("{context} intermediate must be positive");
    }
    let row_values = checked_row_values(&format!("{context} rows"), rows, hidden)?;
    let gate_values = checked_row_values(&format!("{context} gate_weight"), intermediate, hidden)?;
    let down_values = checked_row_values(&format!("{context} down_weight"), hidden, intermediate)?;
    validate_u16_buffer_values(&format!("{context} x"), x, row_values)?;
    validate_u16_buffer_values(&format!("{context} gate_weight"), gate_weight, gate_values)?;
    validate_u16_buffer_values(&format!("{context} up_weight"), up_weight, gate_values)?;
    validate_u16_buffer_values(&format!("{context} down_weight"), down_weight, down_values)?;
    validate_u16_buffer_values(&format!("{context} out"), out, row_values)
}

fn validate_scheduler_mlp_delta_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    let values = checked_row_values(context, rows, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, values)?;
    validate_u16_buffer_values(&format!("{context} gate_weight"), gate_weight, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} up_weight"), up_weight, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} down_weight"), down_weight, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} out"), out, values)
}

#[allow(clippy::too_many_arguments)]
fn validate_mlp_rows_bf16_down_stride_buffers(
    context: &str,
    x: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Result<()> {
    if down_stride < intermediate {
        anyhow::bail!(
            "{context} down_stride {down_stride} must be at least intermediate {intermediate}"
        );
    }
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    if intermediate == 0 {
        anyhow::bail!("{context} intermediate must be positive");
    }
    let row_values = checked_row_values(&format!("{context} rows"), rows, hidden)?;
    let gate_values = checked_row_values(&format!("{context} gate_weight"), intermediate, hidden)?;
    let down_values = checked_row_values(&format!("{context} down_weight"), hidden, down_stride)?;
    validate_u16_buffer_values(&format!("{context} x"), x, row_values)?;
    validate_u16_buffer_values(&format!("{context} gate_weight"), gate_weight, gate_values)?;
    validate_u16_buffer_values(&format!("{context} up_weight"), up_weight, gate_values)?;
    validate_u16_buffer_values(&format!("{context} down_weight"), down_weight, down_values)?;
    validate_u16_buffer_values(&format!("{context} out"), out, row_values)
}

#[allow(clippy::too_many_arguments)]
fn validate_mlp_rows_bf16_down_stride_staged_buffers(
    context: &str,
    x: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    activation_workspace: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    down_stride: usize,
) -> Result<()> {
    validate_mlp_rows_bf16_down_stride_buffers(
        context,
        x,
        gate_weight,
        up_weight,
        down_weight,
        out,
        rows,
        hidden,
        intermediate,
        down_stride,
    )?;
    let activation_values = checked_row_values(
        &format!("{context} activation_workspace"),
        rows,
        intermediate,
    )?;
    validate_f32_buffer_values(
        &format!("{context} activation_workspace"),
        activation_workspace,
        activation_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_nvfp4_route_bf16_grouped_accumulate_f32_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    route_weights: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    gate_scale: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    up_scale: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    down_scale: Ds41rtDeviceBuffer,
    accumulator: Ds41rtDeviceBuffer,
    rows: usize,
    routes: usize,
    hidden_dim: usize,
    hidden_row_stride: usize,
    intermediate: usize,
    output_dim: usize,
    down_weight_row_stride_bytes: usize,
    down_scale_row_stride_bytes: usize,
    gate_scale_2: f32,
    up_scale_2: f32,
    down_scale_2: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if routes == 0 {
        anyhow::bail!("{context} routes must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if hidden_row_stride < hidden_dim {
        anyhow::bail!(
            "{context} hidden_row_stride {hidden_row_stride} is smaller than hidden_dim {hidden_dim}"
        );
    }
    if intermediate == 0 {
        anyhow::bail!("{context} intermediate must be positive");
    }
    if output_dim == 0 {
        anyhow::bail!("{context} output_dim must be positive");
    }
    for (label, value) in [
        ("gate_scale_2", gate_scale_2),
        ("up_scale_2", up_scale_2),
        ("down_scale_2", down_scale_2),
    ] {
        if !value.is_finite() {
            anyhow::bail!("{context} {label} must be finite");
        }
    }

    let packed_hidden_bytes = hidden_dim
        .checked_add(1)
        .with_context(|| format!("{context} hidden_dim overflows packed byte calculation"))?
        / 2;
    let hidden_scale_bytes = hidden_dim
        .checked_add(15)
        .with_context(|| format!("{context} hidden_dim overflows scale byte calculation"))?
        / 16;
    let packed_intermediate_bytes = intermediate
        .checked_add(1)
        .with_context(|| format!("{context} intermediate overflows packed byte calculation"))?
        / 2;
    let intermediate_scale_bytes = intermediate
        .checked_add(15)
        .with_context(|| format!("{context} intermediate overflows scale byte calculation"))?
        / 16;
    if down_weight_row_stride_bytes < packed_intermediate_bytes {
        anyhow::bail!(
            "{context} down_weight row stride {down_weight_row_stride_bytes} is smaller than packed prefix {packed_intermediate_bytes}"
        );
    }
    if down_scale_row_stride_bytes < intermediate_scale_bytes {
        anyhow::bail!(
            "{context} down_scale row stride {down_scale_row_stride_bytes} is smaller than scale prefix {intermediate_scale_bytes}"
        );
    }
    let gate_weight_bytes = intermediate
        .checked_mul(packed_hidden_bytes)
        .with_context(|| format!("{context} gate/up weight byte count overflows usize"))?;
    let gate_scale_bytes = intermediate
        .checked_mul(hidden_scale_bytes)
        .with_context(|| format!("{context} gate/up scale byte count overflows usize"))?;
    let down_weight_bytes = output_dim
        .checked_mul(down_weight_row_stride_bytes)
        .with_context(|| format!("{context} down weight byte count overflows usize"))?;
    let down_scale_bytes = output_dim
        .checked_mul(down_scale_row_stride_bytes)
        .with_context(|| format!("{context} down scale byte count overflows usize"))?;
    let hidden_values = rows
        .checked_mul(hidden_row_stride)
        .with_context(|| format!("{context} hidden value count overflows usize"))?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, routes)?;
    validate_f32_buffer_values(&format!("{context} route_weights"), route_weights, routes)?;
    validate_device_buffer_bytes(
        &format!("{context} gate_weight"),
        gate_weight,
        gate_weight_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} gate_scale"),
        gate_scale,
        gate_scale_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} up_weight"),
        up_weight,
        gate_weight_bytes,
    )?;
    validate_device_buffer_bytes(&format!("{context} up_scale"), up_scale, gate_scale_bytes)?;
    validate_device_buffer_bytes(
        &format!("{context} down_weight"),
        down_weight,
        down_weight_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} down_scale"),
        down_scale,
        down_scale_bytes,
    )?;
    let accumulator_values = rows
        .checked_mul(output_dim)
        .with_context(|| format!("{context} accumulator value count overflows usize"))?;
    validate_f32_buffer_values(
        &format!("{context} accumulator"),
        accumulator,
        accumulator_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_nvfp4_route_bf16_grouped_staged_accumulate_f32_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    route_weights: Ds41rtDeviceBuffer,
    gate_weight: Ds41rtDeviceBuffer,
    gate_scale: Ds41rtDeviceBuffer,
    up_weight: Ds41rtDeviceBuffer,
    up_scale: Ds41rtDeviceBuffer,
    down_weight: Ds41rtDeviceBuffer,
    down_scale: Ds41rtDeviceBuffer,
    activation_workspace: Ds41rtDeviceBuffer,
    accumulator: Ds41rtDeviceBuffer,
    rows: usize,
    routes: usize,
    hidden_dim: usize,
    hidden_row_stride: usize,
    intermediate: usize,
    output_dim: usize,
    down_weight_row_stride_bytes: usize,
    down_scale_row_stride_bytes: usize,
    gate_scale_2: f32,
    up_scale_2: f32,
    down_scale_2: f32,
) -> Result<()> {
    validate_nvfp4_route_bf16_grouped_accumulate_f32_buffers(
        context,
        hidden,
        row_indices,
        route_weights,
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        down_weight,
        down_scale,
        accumulator,
        rows,
        routes,
        hidden_dim,
        hidden_row_stride,
        intermediate,
        output_dim,
        down_weight_row_stride_bytes,
        down_scale_row_stride_bytes,
        gate_scale_2,
        up_scale_2,
        down_scale_2,
    )?;
    let activation_values = routes
        .checked_mul(intermediate)
        .with_context(|| format!("{context} activation workspace value count overflows usize"))?;
    validate_f32_buffer_values(
        &format!("{context} activation_workspace"),
        activation_workspace,
        activation_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_nvfp4_route_bf16_batched_staged_accumulate_f32_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    route_weights: Ds41rtDeviceBuffer,
    route_metadata: Ds41rtDeviceBuffer,
    activation_workspace: Ds41rtDeviceBuffer,
    accumulator: Ds41rtDeviceBuffer,
    rows: usize,
    routes: usize,
    hidden_dim: usize,
    hidden_row_stride: usize,
    max_intermediate: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if routes == 0 {
        anyhow::bail!("{context} routes must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if hidden_row_stride < hidden_dim {
        anyhow::bail!(
            "{context} hidden_row_stride {hidden_row_stride} is smaller than hidden_dim {hidden_dim}"
        );
    }
    if max_intermediate == 0 {
        anyhow::bail!("{context} max_intermediate must be positive");
    }
    if output_dim == 0 {
        anyhow::bail!("{context} output_dim must be positive");
    }
    let hidden_values = rows
        .checked_mul(hidden_row_stride)
        .with_context(|| format!("{context} hidden value count overflows usize"))?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, routes)?;
    validate_f32_buffer_values(&format!("{context} route_weights"), route_weights, routes)?;
    let route_metadata_bytes = routes
        .checked_mul(std::mem::size_of::<Ds41rtNvfp4RouteBatchedMetadata>())
        .with_context(|| format!("{context} route metadata byte count overflows usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} route_metadata"),
        route_metadata,
        route_metadata_bytes,
    )?;
    let activation_values = routes
        .checked_mul(max_intermediate)
        .with_context(|| format!("{context} activation workspace value count overflows usize"))?;
    validate_f32_buffer_values(
        &format!("{context} activation_workspace"),
        activation_workspace,
        activation_values,
    )?;
    let accumulator_values = rows
        .checked_mul(output_dim)
        .with_context(|| format!("{context} accumulator value count overflows usize"))?;
    validate_f32_buffer_values(
        &format!("{context} accumulator"),
        accumulator,
        accumulator_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_nvfp4_route_bf16_batched_staged_single_row_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    row_indices: Ds41rtDeviceBuffer,
    route_weights: Ds41rtDeviceBuffer,
    route_metadata: Ds41rtDeviceBuffer,
    activation_workspace: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    routes: usize,
    hidden_dim: usize,
    hidden_row_stride: usize,
    max_intermediate: usize,
    output_dim: usize,
) -> Result<()> {
    if rows != 1 {
        anyhow::bail!("{context} rows must be 1 for single-row BF16 output");
    }
    if routes == 0 {
        anyhow::bail!("{context} routes must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if hidden_row_stride < hidden_dim {
        anyhow::bail!(
            "{context} hidden_row_stride {hidden_row_stride} is smaller than hidden_dim {hidden_dim}"
        );
    }
    if max_intermediate == 0 {
        anyhow::bail!("{context} max_intermediate must be positive");
    }
    if output_dim == 0 {
        anyhow::bail!("{context} output_dim must be positive");
    }
    let hidden_values = rows
        .checked_mul(hidden_row_stride)
        .with_context(|| format!("{context} hidden value count overflows usize"))?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u32_buffer_values(&format!("{context} row_indices"), row_indices, routes)?;
    validate_f32_buffer_values(&format!("{context} route_weights"), route_weights, routes)?;
    let route_metadata_bytes = routes
        .checked_mul(std::mem::size_of::<Ds41rtNvfp4RouteBatchedMetadata>())
        .with_context(|| format!("{context} route metadata byte count overflows usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} route_metadata"),
        route_metadata,
        route_metadata_bytes,
    )?;
    let activation_values = routes
        .checked_mul(max_intermediate)
        .with_context(|| format!("{context} activation workspace value count overflows usize"))?;
    validate_f32_buffer_values(
        &format!("{context} activation_workspace"),
        activation_workspace,
        activation_values,
    )?;
    validate_u16_buffer_values(&format!("{context} out"), out, output_dim)
}

fn validate_linear_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    bias: Option<Ds41rtDeviceBuffer>,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if input_dim == 0 {
        anyhow::bail!("{context} input_dim must be positive");
    }
    if output_dim == 0 {
        anyhow::bail!("{context} output_dim must be positive");
    }
    let input_values = checked_row_values(&format!("{context} input"), rows, input_dim)?;
    let weight_values = checked_row_values(&format!("{context} weight"), output_dim, input_dim)?;
    let output_values = checked_row_values(&format!("{context} output"), rows, output_dim)?;
    validate_f32_buffer_values(&format!("{context} input"), input, input_values)?;
    validate_f32_buffer_values(&format!("{context} weight"), weight, weight_values)?;
    if let Some(bias) = bias {
        validate_f32_buffer_values(&format!("{context} bias"), bias, output_dim)?;
    }
    validate_f32_buffer_values(&format!("{context} output"), output, output_values)
}

fn validate_linear_bf16_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    bias: Option<Ds41rtDeviceBuffer>,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if input_dim == 0 {
        anyhow::bail!("{context} input_dim must be positive");
    }
    if output_dim == 0 {
        anyhow::bail!("{context} output_dim must be positive");
    }
    let input_values = checked_row_values(&format!("{context} input"), rows, input_dim)?;
    let weight_values = checked_row_values(&format!("{context} weight"), output_dim, input_dim)?;
    let output_values = checked_row_values(&format!("{context} output"), rows, output_dim)?;
    validate_u16_buffer_values(&format!("{context} input"), input, input_values)?;
    validate_u16_buffer_values(&format!("{context} weight"), weight, weight_values)?;
    if let Some(bias) = bias {
        validate_u16_buffer_values(&format!("{context} bias"), bias, output_dim)?;
    }
    validate_u16_buffer_values(&format!("{context} output"), output, output_values)
}

fn validate_linear_bf16_f32_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if rows == 0 || input_dim == 0 || output_dim == 0 {
        anyhow::bail!("{context} dimensions must be positive");
    }
    validate_u16_buffer_values(
        &format!("{context} input"),
        input,
        checked_row_values(&format!("{context} input"), rows, input_dim)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} weight"),
        weight,
        checked_row_values(&format!("{context} weight"), output_dim, input_dim)?,
    )?;
    validate_f32_buffer_values(
        &format!("{context} output"),
        output,
        checked_row_values(&format!("{context} output"), rows, output_dim)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_linear_bf16_strided_batched_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    weight: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    batch_count: usize,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    input_batch_stride: usize,
    weight_batch_stride: usize,
    output_batch_stride: usize,
) -> Result<()> {
    if batch_count == 0 || rows == 0 || input_dim == 0 || output_dim == 0 {
        anyhow::bail!("{context} dimensions must be positive");
    }
    let strided_values = |label: &str, stride: usize, matrix_values: usize| {
        let batch_offset = (batch_count - 1)
            .checked_mul(stride)
            .with_context(|| format!("{context} {label} batch offset overflows usize"))?;
        batch_offset
            .checked_add(matrix_values)
            .with_context(|| format!("{context} {label} value count overflows usize"))
    };
    let input_matrix = checked_row_values(&format!("{context} input"), rows, input_dim)?;
    let weight_matrix = checked_row_values(&format!("{context} weight"), output_dim, input_dim)?;
    let output_matrix = checked_row_values(&format!("{context} output"), rows, output_dim)?;
    validate_u16_buffer_values(
        &format!("{context} input"),
        input,
        strided_values("input", input_batch_stride, input_matrix)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} weight"),
        weight,
        strided_values("weight", weight_batch_stride, weight_matrix)?,
    )?;
    validate_u16_buffer_values(
        &format!("{context} output"),
        output,
        strided_values("output", output_batch_stride, output_matrix)?,
    )
}

fn checked_3d_values(context: &str, a: usize, b: usize, c: usize) -> Result<usize> {
    let ab = a
        .checked_mul(b)
        .with_context(|| format!("{context} value count overflows usize"))?;
    ab.checked_mul(c)
        .with_context(|| format!("{context} value count overflows usize"))
}

fn validate_causal_attention_buffers(
    context: &str,
    q: Ds41rtDeviceBuffer,
    k: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if qk_dim == 0 {
        anyhow::bail!("{context} qk_dim must be positive");
    }
    if v_dim == 0 {
        anyhow::bail!("{context} v_dim must be positive");
    }
    let qk_values = checked_3d_values(&format!("{context} qk"), rows, heads, qk_dim)?;
    let v_values = checked_3d_values(&format!("{context} v"), rows, heads, v_dim)?;
    validate_f32_buffer_values(&format!("{context} q"), q, qk_values)?;
    validate_f32_buffer_values(&format!("{context} k"), k, qk_values)?;
    validate_f32_buffer_values(&format!("{context} v"), v, v_values)?;
    validate_f32_buffer_values(&format!("{context} out"), out, v_values)
}

fn validate_causal_attention_bf16_buffers(
    context: &str,
    q: Ds41rtDeviceBuffer,
    k: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    qk_dim: usize,
    v_dim: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if qk_dim == 0 {
        anyhow::bail!("{context} qk_dim must be positive");
    }
    if v_dim == 0 {
        anyhow::bail!("{context} v_dim must be positive");
    }
    let qk_values = checked_3d_values(&format!("{context} qk"), rows, heads, qk_dim)?;
    let v_values = checked_3d_values(&format!("{context} v"), rows, heads, v_dim)?;
    validate_u16_buffer_values(&format!("{context} q"), q, qk_values)?;
    validate_u16_buffer_values(&format!("{context} k"), k, qk_values)?;
    validate_u16_buffer_values(&format!("{context} v"), v, v_values)?;
    validate_u16_buffer_values(&format!("{context} out"), out, v_values)
}

fn validate_rope_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    positions: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        anyhow::bail!("{context} rotary_dim must be positive and even");
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("{context} theta must be finite and positive");
    }
    let values = checked_3d_values(&format!("{context} input"), rows, heads, rotary_dim)?;
    validate_f32_buffer_values(&format!("{context} input"), input, values)?;
    validate_u32_buffer_values(&format!("{context} positions"), positions, rows)?;
    validate_f32_buffer_values(&format!("{context} out"), out, values)
}

fn validate_rope_bf16_buffers(
    context: &str,
    input: Ds41rtDeviceBuffer,
    positions: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    rotary_dim: usize,
    theta: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        anyhow::bail!("{context} rotary_dim must be positive and even");
    }
    if !theta.is_finite() || theta <= 0.0 {
        anyhow::bail!("{context} theta must be finite and positive");
    }
    let values = checked_3d_values(&format!("{context} input"), rows, heads, rotary_dim)?;
    validate_u16_buffer_values(&format!("{context} input"), input, values)?;
    validate_u32_buffer_values(&format!("{context} positions"), positions, rows)?;
    validate_u16_buffer_values(&format!("{context} out"), out, values)
}

fn validate_mla_rope_attention_bf16_buffers(
    context: &str,
    q_nope: Ds41rtDeviceBuffer,
    q_rope: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if nope_dim == 0 {
        anyhow::bail!("{context} nope_dim must be positive");
    }
    if rope_dim == 0 || rope_dim % 2 != 0 {
        anyhow::bail!("{context} rope_dim must be positive and even");
    }
    if v_dim == 0 {
        anyhow::bail!("{context} v_dim must be positive");
    }
    if !scale.is_finite() {
        anyhow::bail!("{context} scale must be finite");
    }
    let nope_values = checked_3d_values(&format!("{context} q_nope"), rows, heads, nope_dim)?;
    let rope_values = checked_3d_values(&format!("{context} q_rope"), rows, heads, rope_dim)?;
    let k_rope_values = checked_row_values(&format!("{context} k_rope"), rows, rope_dim)?;
    let v_values = checked_3d_values(&format!("{context} v"), rows, heads, v_dim)?;
    validate_u16_buffer_values(&format!("{context} q_nope"), q_nope, nope_values)?;
    validate_u16_buffer_values(&format!("{context} q_rope"), q_rope, rope_values)?;
    validate_u16_buffer_values(&format!("{context} k_nope"), k_nope, nope_values)?;
    validate_u16_buffer_values(&format!("{context} k_rope"), k_rope, k_rope_values)?;
    validate_u16_buffer_values(&format!("{context} v"), v, v_values)?;
    validate_u16_buffer_values(&format!("{context} out"), out, v_values)
}

fn validate_mla_rope_attention_bf16_suffix_buffers(
    context: &str,
    q_nope: Ds41rtDeviceBuffer,
    q_rope: Ds41rtDeviceBuffer,
    k_nope: Ds41rtDeviceBuffer,
    k_rope: Ds41rtDeviceBuffer,
    v: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    query_row_offset: usize,
    query_rows: usize,
    heads: usize,
    nope_dim: usize,
    rope_dim: usize,
    v_dim: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if heads == 0 {
        anyhow::bail!("{context} heads must be positive");
    }
    if nope_dim == 0 {
        anyhow::bail!("{context} nope_dim must be positive");
    }
    if rope_dim == 0 || rope_dim % 2 != 0 {
        anyhow::bail!("{context} rope_dim must be positive and even");
    }
    if v_dim == 0 {
        anyhow::bail!("{context} v_dim must be positive");
    }
    if !scale.is_finite() {
        anyhow::bail!("{context} scale must be finite");
    }
    if query_rows == 0 {
        anyhow::bail!("{context} query_rows must be positive");
    }
    if query_row_offset > rows || query_rows > rows - query_row_offset {
        anyhow::bail!(
            "{context} query rows {}..{} exceed rows {rows}",
            query_row_offset,
            query_row_offset.saturating_add(query_rows)
        );
    }
    let nope_values = checked_3d_values(&format!("{context} q_nope"), rows, heads, nope_dim)?;
    let rope_values = checked_3d_values(&format!("{context} q_rope"), rows, heads, rope_dim)?;
    let k_rope_values = checked_row_values(&format!("{context} k_rope"), rows, rope_dim)?;
    let v_values = checked_3d_values(&format!("{context} v"), rows, heads, v_dim)?;
    let out_values = checked_3d_values(&format!("{context} out"), query_rows, heads, v_dim)?;
    validate_u16_buffer_values(&format!("{context} q_nope"), q_nope, nope_values)?;
    validate_u16_buffer_values(&format!("{context} q_rope"), q_rope, rope_values)?;
    validate_u16_buffer_values(&format!("{context} k_nope"), k_nope, nope_values)?;
    validate_u16_buffer_values(&format!("{context} k_rope"), k_rope, k_rope_values)?;
    validate_u16_buffer_values(&format!("{context} v"), v, v_values)?;
    validate_u16_buffer_values(&format!("{context} out"), out, out_values)
}

fn validate_embedding_lookup_buffers(
    context: &str,
    embedding: Ds41rtDeviceBuffer,
    token_ids: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    let embedding_values = checked_row_values(&format!("{context} embedding"), vocab, hidden)?;
    let out_values = checked_row_values(&format!("{context} out"), rows, hidden)?;
    validate_f32_buffer_values(&format!("{context} embedding"), embedding, embedding_values)?;
    validate_u32_buffer_values(&format!("{context} token_ids"), token_ids, rows)?;
    validate_f32_buffer_values(&format!("{context} out"), out, out_values)
}

fn validate_embedding_lookup_bf16_buffers(
    context: &str,
    embedding: Ds41rtDeviceBuffer,
    token_ids: Ds41rtDeviceBuffer,
    out: Ds41rtDeviceBuffer,
    rows: usize,
    vocab: usize,
    hidden: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if hidden == 0 {
        anyhow::bail!("{context} hidden must be positive");
    }
    let embedding_values = checked_row_values(&format!("{context} embedding"), vocab, hidden)?;
    let out_values = checked_row_values(&format!("{context} out"), rows, hidden)?;
    validate_u16_buffer_values(&format!("{context} embedding"), embedding, embedding_values)?;
    validate_u32_buffer_values(&format!("{context} token_ids"), token_ids, rows)?;
    validate_u16_buffer_values(&format!("{context} out"), out, out_values)
}

fn validate_lm_head_argmax_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    let hidden_values = checked_row_values(&format!("{context} hidden"), rows, hidden_dim)?;
    let lm_head_values = checked_row_values(&format!("{context} lm_head"), vocab, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u16_buffer_values(&format!("{context} lm_head"), lm_head, lm_head_values)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)
}

#[allow(clippy::too_many_arguments)]
fn validate_dspark_terminal_greedy_bf16_buffers(
    context: &str,
    normalized_hidden_by_slot: Ds41rtDeviceBuffer,
    collapsed_hidden_by_slot: Ds41rtDeviceBuffer,
    shared_head: Ds41rtDeviceBuffer,
    markov_w1: Ds41rtDeviceBuffer,
    markov_w2: Ds41rtDeviceBuffer,
    confidence_weight: Ds41rtDeviceBuffer,
    active_slot_ids: Ds41rtDeviceBuffer,
    anchor_token_ids: Ds41rtDeviceBuffer,
    compact_normalized_hidden: Ds41rtDeviceBuffer,
    shared_logits: Ds41rtDeviceBuffer,
    markov_embeddings: Ds41rtDeviceBuffer,
    markov_logits: Ds41rtDeviceBuffer,
    output_token_ids: Ds41rtDeviceBuffer,
    conditional_confidence: Ds41rtDeviceBuffer,
    active_requests: usize,
    max_slots: usize,
    hidden_dim: usize,
    vocab: usize,
    markov_rank: usize,
) -> Result<()> {
    const PROPOSAL_TOKENS: usize = 5;
    if active_requests == 0 || active_requests > max_slots {
        anyhow::bail!("{context} active requests {active_requests} must be in 1..={max_slots}");
    }
    if hidden_dim == 0 || vocab == 0 || markov_rank == 0 {
        anyhow::bail!("{context} hidden, vocab, and Markov rank must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output token IDs");
    }
    let slot_rows = max_slots
        .checked_mul(PROPOSAL_TOKENS)
        .with_context(|| format!("{context} slot rows overflow"))?;
    let compact_rows = active_requests
        .checked_mul(PROPOSAL_TOKENS)
        .with_context(|| format!("{context} compact rows overflow"))?;
    let slot_hidden_values =
        checked_row_values(&format!("{context} slot hidden"), slot_rows, hidden_dim)?;
    let compact_hidden_values = checked_row_values(
        &format!("{context} compact hidden"),
        compact_rows,
        hidden_dim,
    )?;
    let head_values = checked_row_values(&format!("{context} shared head"), vocab, hidden_dim)?;
    let markov_weight_values =
        checked_row_values(&format!("{context} Markov weight"), vocab, markov_rank)?;
    let confidence_weight_values = hidden_dim
        .checked_add(markov_rank)
        .with_context(|| format!("{context} confidence width overflow"))?;
    let shared_logits_values =
        checked_row_values(&format!("{context} shared logits"), compact_rows, vocab)?;
    let markov_embedding_values = checked_row_values(
        &format!("{context} Markov embeddings"),
        compact_rows,
        markov_rank,
    )?;
    let markov_logits_values =
        checked_row_values(&format!("{context} Markov logits"), active_requests, vocab)?;
    let output_token_values = active_requests
        .checked_mul(PROPOSAL_TOKENS + 1)
        .with_context(|| format!("{context} output token count overflow"))?;
    validate_u16_buffer_values(
        &format!("{context} normalized hidden by slot"),
        normalized_hidden_by_slot,
        slot_hidden_values,
    )?;
    validate_u16_buffer_values(
        &format!("{context} collapsed hidden by slot"),
        collapsed_hidden_by_slot,
        slot_hidden_values,
    )?;
    validate_u16_buffer_values(&format!("{context} shared head"), shared_head, head_values)?;
    validate_u16_buffer_values(
        &format!("{context} Markov w1"),
        markov_w1,
        markov_weight_values,
    )?;
    validate_u16_buffer_values(
        &format!("{context} Markov w2"),
        markov_w2,
        markov_weight_values,
    )?;
    validate_u16_buffer_values(
        &format!("{context} confidence weight"),
        confidence_weight,
        confidence_weight_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} active slot IDs"),
        active_slot_ids,
        active_requests,
    )?;
    validate_u32_buffer_values(
        &format!("{context} anchor token IDs"),
        anchor_token_ids,
        active_requests,
    )?;
    validate_u16_buffer_values(
        &format!("{context} compact normalized hidden"),
        compact_normalized_hidden,
        compact_hidden_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} shared logits"),
        shared_logits,
        shared_logits_values,
    )?;
    validate_u16_buffer_values(
        &format!("{context} Markov embeddings"),
        markov_embeddings,
        markov_embedding_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} Markov logits"),
        markov_logits,
        markov_logits_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} output token IDs"),
        output_token_ids,
        output_token_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} conditional confidence"),
        conditional_confidence,
        compact_rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_lm_head_sample_topk_topp_bf16_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if hidden_dim == 0 {
        anyhow::bail!("{context} hidden_dim must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    if top_k == 0 {
        anyhow::bail!("{context} top_k must be positive");
    }
    if top_k > vocab {
        anyhow::bail!("{context} top_k must not exceed vocab");
    }
    if top_k > 64 {
        anyhow::bail!("{context} top_k must not exceed 64");
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        anyhow::bail!("{context} temperature must be finite and positive");
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        anyhow::bail!("{context} top_p must be finite and in (0, 1]");
    }
    let hidden_values = checked_row_values(&format!("{context} hidden"), rows, hidden_dim)?;
    let lm_head_values = checked_row_values(&format!("{context} lm_head"), vocab, hidden_dim)?;
    validate_u16_buffer_values(&format!("{context} hidden"), hidden, hidden_values)?;
    validate_u16_buffer_values(&format!("{context} lm_head"), lm_head, lm_head_values)?;
    validate_f32_buffer_values(&format!("{context} random_uniforms"), random_uniforms, rows)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)
}

#[allow(clippy::too_many_arguments)]
fn validate_lm_head_argmax_sample_topk_topp_bf16_staged_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    out_argmax_indices: Ds41rtDeviceBuffer,
    out_argmax_scores: Ds41rtDeviceBuffer,
    out_sample_indices: Ds41rtDeviceBuffer,
    out_sample_scores: Ds41rtDeviceBuffer,
    logits_workspace: Ds41rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    validate_lm_head_sample_topk_topp_bf16_buffers(
        context,
        hidden,
        lm_head,
        random_uniforms,
        out_sample_indices,
        out_sample_scores,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    validate_u32_buffer_values(
        &format!("{context} out_argmax_indices"),
        out_argmax_indices,
        rows,
    )?;
    validate_f32_buffer_values(
        &format!("{context} out_argmax_scores"),
        out_argmax_scores,
        rows,
    )?;
    let logits_values = checked_row_values(&format!("{context} logits_workspace"), rows, vocab)?;
    validate_f32_buffer_values(
        &format!("{context} logits_workspace"),
        logits_workspace,
        logits_values,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_lm_head_sample_topk_topp_bf16_cub_buffers(
    context: &str,
    hidden: Ds41rtDeviceBuffer,
    lm_head: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    logits_workspace: Ds41rtDeviceBuffer,
    sorted_logits: Ds41rtDeviceBuffer,
    unsorted_indices: Ds41rtDeviceBuffer,
    sorted_indices: Ds41rtDeviceBuffer,
    segment_offsets: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    cub_temp_storage: Ds41rtDeviceBuffer,
    cub_temp_storage_bytes: usize,
    rows: usize,
    hidden_dim: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    validate_lm_head_sample_topk_topp_bf16_buffers(
        context,
        hidden,
        lm_head,
        random_uniforms,
        out_indices,
        out_scores,
        rows,
        hidden_dim,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    let logits_values = checked_row_values(&format!("{context} logits_workspace"), rows, vocab)?;
    validate_f32_buffer_values(
        &format!("{context} logits_workspace"),
        logits_workspace,
        logits_values,
    )?;
    validate_f32_buffer_values(
        &format!("{context} sorted_logits"),
        sorted_logits,
        logits_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} unsorted_indices"),
        unsorted_indices,
        logits_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} sorted_indices"),
        sorted_indices,
        logits_values,
    )?;
    let offset_values = rows
        .checked_add(1)
        .with_context(|| format!("{context} segment offset count overflows usize"))?;
    let offset_bytes = offset_values
        .checked_mul(std::mem::size_of::<i32>())
        .with_context(|| format!("{context} segment offset bytes overflow usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} segment_offsets"),
        segment_offsets,
        offset_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} cub_temp_storage"),
        cub_temp_storage,
        cub_temp_storage_bytes,
    )
}

fn validate_logits_argmax_buffers(
    context: &str,
    logits: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    vocab: usize,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    let logits_values = checked_row_values(&format!("{context} logits"), rows, vocab)?;
    validate_f32_buffer_values(&format!("{context} logits"), logits, logits_values)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)
}

#[allow(clippy::too_many_arguments)]
fn validate_logits_sample_topk_topp_buffers(
    context: &str,
    logits: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    if rows == 0 {
        anyhow::bail!("{context} rows must be positive");
    }
    if vocab == 0 {
        anyhow::bail!("{context} vocab must be positive");
    }
    if vocab > u32::MAX as usize {
        anyhow::bail!("{context} vocab must fit in u32 output indices");
    }
    if top_k == 0 || top_k > vocab || top_k > DS41RT_CUDA_SAMPLE_TOPK_MAX_K {
        anyhow::bail!(
            "{context} top_k must be in 1..=min(vocab, {DS41RT_CUDA_SAMPLE_TOPK_MAX_K}), got top_k={top_k} vocab={vocab}"
        );
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        anyhow::bail!("{context} temperature must be finite and positive");
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        anyhow::bail!("{context} top_p must be finite and in (0, 1]");
    }
    let logits_values = checked_row_values(&format!("{context} logits"), rows, vocab)?;
    validate_f32_buffer_values(&format!("{context} logits"), logits, logits_values)?;
    validate_f32_buffer_values(&format!("{context} random_uniforms"), random_uniforms, rows)?;
    validate_u32_buffer_values(&format!("{context} out_indices"), out_indices, rows)?;
    validate_f32_buffer_values(&format!("{context} out_scores"), out_scores, rows)
}

#[allow(clippy::too_many_arguments)]
fn validate_logits_sample_topk_topp_cub_buffers(
    context: &str,
    logits: Ds41rtDeviceBuffer,
    random_uniforms: Ds41rtDeviceBuffer,
    sorted_logits: Ds41rtDeviceBuffer,
    unsorted_indices: Ds41rtDeviceBuffer,
    sorted_indices: Ds41rtDeviceBuffer,
    segment_offsets: Ds41rtDeviceBuffer,
    out_indices: Ds41rtDeviceBuffer,
    out_scores: Ds41rtDeviceBuffer,
    cub_temp_storage: Ds41rtDeviceBuffer,
    cub_temp_storage_bytes: usize,
    rows: usize,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<()> {
    validate_logits_sample_topk_topp_buffers(
        context,
        logits,
        random_uniforms,
        out_indices,
        out_scores,
        rows,
        vocab,
        temperature,
        top_k,
        top_p,
    )?;
    let logits_values = checked_row_values(&format!("{context} logits workspace"), rows, vocab)?;
    validate_f32_buffer_values(
        &format!("{context} sorted_logits"),
        sorted_logits,
        logits_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} unsorted_indices"),
        unsorted_indices,
        logits_values,
    )?;
    validate_u32_buffer_values(
        &format!("{context} sorted_indices"),
        sorted_indices,
        logits_values,
    )?;
    let offset_values = rows
        .checked_add(1)
        .with_context(|| format!("{context} segment offset count overflows usize"))?;
    let offset_bytes = offset_values
        .checked_mul(std::mem::size_of::<i32>())
        .with_context(|| format!("{context} segment offset bytes overflow usize"))?;
    validate_device_buffer_bytes(
        &format!("{context} segment_offsets"),
        segment_offsets,
        offset_bytes,
    )?;
    validate_device_buffer_bytes(
        &format!("{context} cub_temp_storage"),
        cub_temp_storage,
        cub_temp_storage_bytes,
    )
}

fn validate_device_buffer_bytes(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    required_bytes: usize,
) -> Result<()> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} buffer pointer is null");
    }
    if buffer.bytes < required_bytes {
        anyhow::bail!(
            "{context} buffer is too small: has {} bytes, needs {required_bytes}",
            buffer.bytes
        );
    }
    Ok(())
}

fn validate_device_buffer_present(context: &str, buffer: Ds41rtDeviceBuffer) -> Result<()> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} buffer pointer is null");
    }
    if buffer.bytes == 0 {
        anyhow::bail!("{context} buffer is empty");
    }
    Ok(())
}

fn validate_device_buffer_range(
    context: &str,
    buffer: Ds41rtDeviceBuffer,
    offset_bytes: usize,
    required_bytes: usize,
) -> Result<()> {
    if buffer.ptr.is_null() {
        anyhow::bail!("{context} buffer pointer is null");
    }
    let end = offset_bytes
        .checked_add(required_bytes)
        .with_context(|| format!("{context} offset plus byte count overflows usize"))?;
    if buffer.bytes < end {
        anyhow::bail!(
            "{context} buffer range is too small: has {} bytes, needs end offset {end}",
            buffer.bytes
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use std::slice;

    fn native_library_path() -> Option<PathBuf> {
        if let Ok(path) = env::var("DS41RT_NATIVE_LIB") {
            return Some(PathBuf::from(path));
        }
        if env::var_os("DS41RT_DISABLE_NATIVE_AUTO_DISCOVERY").is_some() {
            return None;
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join("native/build/libds41rt_native.so");
        path.exists().then_some(path)
    }

    fn load_test_library() -> Result<Option<NativeLibrary>> {
        let Some(path) = native_library_path() else {
            eprintln!("skipping native FFI test because native/build/libds41rt_native.so is absent");
            return Ok(None);
        };
        let library = unsafe { NativeLibrary::load(path)? };
        Ok(Some(library))
    }

    fn synthetic_device_buffer(address: usize, bytes: usize) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: address as *mut c_void,
            bytes,
            device_id: 0,
            flags: 0,
        }
    }

    #[test]
    fn pro_router_ffi_requires_exact_production_geometry() {
        let rows = 5_usize;
        let hidden = synthetic_device_buffer(0x10_0000, rows * 7_168 * 2);
        let weight = synthetic_device_buffer(0x20_0000, 384 * 7_168 * 2);
        let bias = synthetic_device_buffer(0x30_0000, 384 * 4);
        let route_bytes = rows * 6 * 4;
        let indices = synthetic_device_buffer(0x40_0000, route_bytes);
        let score_workspace_bytes = rows * (6 + 384) * 4;
        let scores = synthetic_device_buffer(0x41_0000, score_workspace_bytes);
        let weights = synthetic_device_buffer(0x42_0000, route_bytes);
        validate_ds4_pro_router_topk_bf16_buffers(
            "test Pro learned router",
            hidden,
            weight,
            Some(bias),
            None,
            None,
            indices,
            scores,
            weights,
            rows,
            false,
        )
        .expect("validating exact Pro learned router buffers");

        let flash_width_hidden = synthetic_device_buffer(0x10_0000, rows * 4_096 * 2);
        let error = validate_ds4_pro_router_topk_bf16_buffers(
            "test Pro learned router",
            flash_width_hidden,
            weight,
            Some(bias),
            None,
            None,
            indices,
            scores,
            weights,
            rows,
            false,
        )
        .expect_err("rejecting Flash-width input for the Pro router");
        assert!(error.to_string().contains("hidden buffer is too small"));

        let token_to_experts = synthetic_device_buffer(0x50_0000, 129_280 * 6 * 8);
        let token_ids = synthetic_device_buffer(0x60_0000, rows * 8);
        validate_ds4_pro_router_topk_bf16_buffers(
            "test Pro hash router",
            hidden,
            weight,
            None,
            Some(token_to_experts),
            Some(token_ids),
            indices,
            scores,
            weights,
            rows,
            true,
        )
        .expect("validating exact Pro hash router buffers");
    }

    fn protocol_v2_frame(kind: u16, payload_bytes: usize) -> Vec<u8> {
        const HEADER_BYTES: usize = 96;
        let mut frame = vec![0_u8; HEADER_BYTES + payload_bytes];
        frame[..8].copy_from_slice(b"DS41RTE3");
        frame[8..10].copy_from_slice(&3_u16.to_le_bytes());
        frame[10..12].copy_from_slice(&kind.to_le_bytes());
        frame[12..16].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        let frame_len = frame.len() as u64;
        let wire_bytes_offset = if kind == 1 { 76 } else { 60 };
        frame[wire_bytes_offset..wire_bytes_offset + 8].copy_from_slice(&frame_len.to_le_bytes());
        for (idx, byte) in frame[HEADER_BYTES..].iter_mut().enumerate() {
            *byte = ((idx * 17 + kind as usize) & 0xff) as u8;
        }
        frame
    }

    fn f32_bytes(values: &[f32]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn u32_bytes(values: &[u32]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn i32_bytes(values: &[i32]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn u64_bytes(values: &[u64]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn u16_bytes(values: &[u16]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn nvfp4_route_metadata_bytes(values: &[Ds41rtNvfp4RouteBatchedMetadata]) -> &[u8] {
        unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % std::mem::size_of::<f32>(), 0);
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
        assert_eq!(bytes.len() % std::mem::size_of::<u32>(), 0);
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn bytes_to_i32_vec(bytes: &[u8]) -> Vec<i32> {
        assert_eq!(bytes.len() % std::mem::size_of::<i32>(), 0);
        bytes
            .chunks_exact(std::mem::size_of::<i32>())
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn bytes_to_u16_vec(bytes: &[u8]) -> Vec<u16> {
        assert_eq!(bytes.len() % std::mem::size_of::<u16>(), 0);
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn bytes_to_bf16_summary(bytes: &[u8]) -> Ds41rtBf16Summary {
        assert_eq!(bytes.len(), std::mem::size_of::<Ds41rtBf16Summary>());
        Ds41rtBf16Summary {
            checksum: f64::from_ne_bytes(bytes[0..8].try_into().unwrap()),
            values: u64::from_ne_bytes(bytes[8..16].try_into().unwrap()),
            finite_values: u64::from_ne_bytes(bytes[16..24].try_into().unwrap()),
            nonzero_values: u64::from_ne_bytes(bytes[24..32].try_into().unwrap()),
        }
    }

    fn assert_u16_device_buffer_eq(
        library: &NativeLibrary,
        buffer: Ds41rtDeviceBuffer,
        expected: &[u16],
    ) -> Result<()> {
        let mut bytes = vec![0_u8; std::mem::size_of_val(expected)];
        library.copy_d2h(&mut bytes, buffer)?;
        assert_eq!(bytes_to_u16_vec(&bytes), expected);
        Ok(())
    }

    fn bytes_to_bf16_f32_vec(bytes: &[u8]) -> Vec<f32> {
        assert_eq!(bytes.len() % std::mem::size_of::<u16>(), 0);
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| {
                let bits = u16::from_ne_bytes(chunk.try_into().unwrap());
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    fn f32_to_bf16(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    fn f32_to_bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits((value as u32) << 16)
    }

    fn bf16_values(values: &[f32]) -> Vec<u16> {
        values.iter().map(|value| f32_to_bf16(*value)).collect()
    }

    fn residual_add_f32_delta_bf16_expected(residual: &[u16], delta_f32: &[f32]) -> Vec<f32> {
        residual
            .iter()
            .zip(delta_f32.iter())
            .map(|(residual, delta)| {
                let rounded_delta = bf16_to_f32(f32_to_bf16(*delta));
                bf16_to_f32(f32_to_bf16(bf16_to_f32(*residual) + rounded_delta))
            })
            .collect()
    }

    fn residual_add_shared_f32_delta_bf16_expected(
        residual: &[u16],
        shared_delta: &[u16],
        routed_delta_f32: &[f32],
    ) -> Vec<f32> {
        residual
            .iter()
            .zip(shared_delta.iter())
            .zip(routed_delta_f32.iter())
            .map(|((residual, shared_delta), routed_delta)| {
                let rounded_routed = bf16_to_f32(f32_to_bf16(*routed_delta));
                let mlp_delta =
                    bf16_to_f32(f32_to_bf16(bf16_to_f32(*shared_delta) + rounded_routed));
                bf16_to_f32(f32_to_bf16(bf16_to_f32(*residual) + mlp_delta))
            })
            .collect()
    }

    #[derive(Debug)]
    struct RouterTopKExpected {
        indices: Vec<u32>,
        scores: Vec<f32>,
        weights: Vec<f32>,
    }

    const GLM52_ROUTED_SCALING_FACTOR: f32 = 2.5;

    fn router_sigmoid(value: f32) -> f32 {
        if value >= 0.0 {
            let exp_neg = (-value).exp();
            1.0 / (1.0 + exp_neg)
        } else {
            let exp_pos = value.exp();
            exp_pos / (1.0 + exp_pos)
        }
    }

    fn router_topk_expected(
        hidden: &[f32],
        router_weight: &[f32],
        correction_bias: &[f32],
        rows: usize,
        hidden_dim: usize,
        experts: usize,
        top_k: usize,
    ) -> RouterTopKExpected {
        let mut indices = vec![0_u32; rows * top_k];
        let mut scores = vec![0.0_f32; rows * top_k];
        let mut weights = vec![0.0_f32; rows * top_k];
        for row in 0..rows {
            let mut best_indices = vec![0_u32; top_k];
            let mut best_scores = vec![0.0_f32; top_k];
            let mut best_corrected = vec![f32::NEG_INFINITY; top_k];
            for expert in 0..experts {
                let mut logit = 0.0_f32;
                for col in 0..hidden_dim {
                    logit +=
                        hidden[row * hidden_dim + col] * router_weight[expert * hidden_dim + col];
                }
                let score = router_sigmoid(logit);
                let corrected = score + correction_bias[expert];
                for rank in 0..top_k {
                    if corrected > best_corrected[rank] {
                        for shift in (rank + 1..top_k).rev() {
                            best_corrected[shift] = best_corrected[shift - 1];
                            best_scores[shift] = best_scores[shift - 1];
                            best_indices[shift] = best_indices[shift - 1];
                        }
                        best_corrected[rank] = corrected;
                        best_scores[rank] = score;
                        best_indices[rank] = expert as u32;
                        break;
                    }
                }
            }
            let score_sum = best_scores.iter().sum::<f32>().max(1.0e-12);
            for rank in 0..top_k {
                indices[row * top_k + rank] = best_indices[rank];
                scores[row * top_k + rank] = best_scores[rank];
                weights[row * top_k + rank] =
                    best_scores[rank] / score_sum * GLM52_ROUTED_SCALING_FACTOR;
            }
        }
        RouterTopKExpected {
            indices,
            scores,
            weights,
        }
    }

    fn linear_expected(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows * output_dim];
        for row in 0..rows {
            for out_col in 0..output_dim {
                let mut acc = bias.map(|values| values[out_col]).unwrap_or(0.0);
                for col in 0..input_dim {
                    acc += input[row * input_dim + col] * weight[out_col * input_dim + col];
                }
                output[row * output_dim + out_col] = acc;
            }
        }
        output
    }

    fn embedding_lookup_expected(
        embedding: &[f32],
        token_ids: &[u32],
        vocab: usize,
        hidden: usize,
    ) -> Vec<f32> {
        assert_eq!(embedding.len(), vocab * hidden);
        let mut output = vec![0.0_f32; token_ids.len() * hidden];
        for (row, token_id) in token_ids.iter().enumerate() {
            let token_id = *token_id as usize;
            assert!(token_id < vocab);
            for col in 0..hidden {
                output[row * hidden + col] = embedding[token_id * hidden + col];
            }
        }
        output
    }

    #[derive(Debug)]
    struct LogitsArgmaxExpected {
        indices: Vec<u32>,
        scores: Vec<f32>,
    }

    fn logits_argmax_expected(logits: &[f32], rows: usize, vocab: usize) -> LogitsArgmaxExpected {
        assert_eq!(logits.len(), rows * vocab);
        let mut indices = vec![0_u32; rows];
        let mut scores = vec![f32::NEG_INFINITY; rows];
        for row in 0..rows {
            let mut best_index = 0_u32;
            let mut best_score = f32::NEG_INFINITY;
            for col in 0..vocab {
                let score = logits[row * vocab + col];
                let token_id = col as u32;
                if score > best_score || (score == best_score && token_id < best_index) {
                    best_score = score;
                    best_index = token_id;
                }
            }
            indices[row] = best_index;
            scores[row] = best_score;
        }
        LogitsArgmaxExpected { indices, scores }
    }

    fn lm_head_argmax_bf16_expected(
        hidden: &[u16],
        lm_head: &[u16],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
    ) -> LogitsArgmaxExpected {
        assert_eq!(hidden.len(), rows * hidden_dim);
        assert_eq!(lm_head.len(), vocab * hidden_dim);
        let mut indices = vec![0_u32; rows];
        let mut scores = vec![f32::NEG_INFINITY; rows];
        for row in 0..rows {
            let mut best_index = 0_u32;
            let mut best_score = f32::NEG_INFINITY;
            for token in 0..vocab {
                let mut score = 0.0_f32;
                for col in 0..hidden_dim {
                    score += bf16_to_f32(hidden[row * hidden_dim + col])
                        * bf16_to_f32(lm_head[token * hidden_dim + col]);
                }
                let token_id = token as u32;
                if score > best_score || (score == best_score && token_id < best_index) {
                    best_score = score;
                    best_index = token_id;
                }
            }
            indices[row] = best_index;
            scores[row] = best_score;
        }
        LogitsArgmaxExpected { indices, scores }
    }

    #[derive(Debug)]
    struct DsparkTerminalExpected {
        output_token_ids: Vec<u32>,
        shared_logits: Vec<f32>,
        markov_embeddings: Vec<u16>,
        conditional_confidence: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn dspark_terminal_greedy_bf16_expected(
        normalized_hidden_by_slot: &[u16],
        collapsed_hidden_by_slot: &[u16],
        shared_head: &[u16],
        markov_w1: &[u16],
        markov_w2: &[u16],
        confidence_weight: &[u16],
        active_slot_ids: &[u32],
        anchor_token_ids: &[u32],
        max_slots: usize,
        hidden_dim: usize,
        vocab: usize,
        markov_rank: usize,
    ) -> DsparkTerminalExpected {
        const PROPOSALS: usize = 5;
        let active = active_slot_ids.len();
        assert_eq!(anchor_token_ids.len(), active);
        assert_eq!(
            normalized_hidden_by_slot.len(),
            max_slots * PROPOSALS * hidden_dim
        );
        assert_eq!(
            collapsed_hidden_by_slot.len(),
            max_slots * PROPOSALS * hidden_dim
        );
        let rows = active * PROPOSALS;
        let mut shared_logits = vec![0.0_f32; rows * vocab];
        let mut markov_embeddings = vec![0_u16; rows * markov_rank];
        let mut output_token_ids = vec![0_u32; active * (PROPOSALS + 1)];
        output_token_ids[..active].copy_from_slice(anchor_token_ids);
        for position in 0..PROPOSALS {
            for request in 0..active {
                let slot = active_slot_ids[request] as usize;
                let compact_row = position * active + request;
                let slot_row = slot * PROPOSALS + position;
                for token in 0..vocab {
                    let mut score = 0.0_f32;
                    for col in 0..hidden_dim {
                        score +=
                            bf16_to_f32(normalized_hidden_by_slot[slot_row * hidden_dim + col])
                                * bf16_to_f32(shared_head[token * hidden_dim + col]);
                    }
                    shared_logits[compact_row * vocab + token] = score;
                }
            }
        }
        for position in 0..PROPOSALS {
            for request in 0..active {
                let compact_row = position * active + request;
                let previous = output_token_ids[position * active + request] as usize;
                let markov_row = compact_row * markov_rank;
                markov_embeddings[markov_row..markov_row + markov_rank].copy_from_slice(
                    &markov_w1[previous * markov_rank..(previous + 1) * markov_rank],
                );
                let mut best_score = f32::NEG_INFINITY;
                let mut best_token = 0_u32;
                for token in 0..vocab {
                    let mut bias = 0.0_f32;
                    for col in 0..markov_rank {
                        bias += bf16_to_f32(markov_embeddings[markov_row + col])
                            * bf16_to_f32(markov_w2[token * markov_rank + col]);
                    }
                    let logit = &mut shared_logits[compact_row * vocab + token];
                    *logit += bias;
                    let token_id = token as u32;
                    if *logit > best_score || (*logit == best_score && token_id < best_token) {
                        best_score = *logit;
                        best_token = token_id;
                    }
                }
                output_token_ids[(position + 1) * active + request] = best_token;
            }
        }
        let mut conditional_confidence = vec![0.0_f32; rows];
        for position in 0..PROPOSALS {
            for request in 0..active {
                let compact_row = position * active + request;
                let slot_row = active_slot_ids[request] as usize * PROPOSALS + position;
                let mut score = 0.0_f32;
                for col in 0..hidden_dim {
                    score += bf16_to_f32(collapsed_hidden_by_slot[slot_row * hidden_dim + col])
                        * bf16_to_f32(confidence_weight[col]);
                }
                for col in 0..markov_rank {
                    score += bf16_to_f32(markov_embeddings[compact_row * markov_rank + col])
                        * bf16_to_f32(confidence_weight[hidden_dim + col]);
                }
                conditional_confidence[compact_row] = 1.0 / (1.0 + (-score).exp());
            }
        }
        DsparkTerminalExpected {
            output_token_ids,
            shared_logits,
            markov_embeddings,
            conditional_confidence,
        }
    }

    fn lm_head_sample_topk_topp_bf16_expected(
        hidden: &[u16],
        lm_head: &[u16],
        random_uniforms: &[f32],
        rows: usize,
        hidden_dim: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> LogitsArgmaxExpected {
        assert_eq!(hidden.len(), rows * hidden_dim);
        assert_eq!(lm_head.len(), vocab * hidden_dim);
        let mut logits = vec![0.0_f32; rows * vocab];
        for row in 0..rows {
            for token in 0..vocab {
                let mut score = 0.0_f32;
                for col in 0..hidden_dim {
                    score += bf16_to_f32(hidden[row * hidden_dim + col])
                        * bf16_to_f32(lm_head[token * hidden_dim + col]);
                }
                logits[row * vocab + token] = score;
            }
        }
        logits_sample_topk_topp_expected(
            &logits,
            random_uniforms,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        )
    }

    fn logits_sample_topk_topp_expected(
        logits: &[f32],
        random_uniforms: &[f32],
        rows: usize,
        vocab: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> LogitsArgmaxExpected {
        assert_eq!(logits.len(), rows * vocab);
        assert_eq!(random_uniforms.len(), rows);
        let mut indices = vec![0_u32; rows];
        let mut scores = vec![0.0_f32; rows];
        for row in 0..rows {
            let mut best_logits = vec![f32::NEG_INFINITY; top_k];
            let mut best_indices = vec![0_u32; top_k];
            for col in 0..vocab {
                let logit = logits[row * vocab + col];
                let token_id = col as u32;
                for rank in 0..top_k {
                    if logit > best_logits[rank]
                        || (logit == best_logits[rank] && token_id < best_indices[rank])
                    {
                        for shift in (rank + 1..top_k).rev() {
                            best_logits[shift] = best_logits[shift - 1];
                            best_indices[shift] = best_indices[shift - 1];
                        }
                        best_logits[rank] = logit;
                        best_indices[rank] = token_id;
                        break;
                    }
                }
            }

            let mut scaled = vec![0.0_f32; top_k];
            let mut max_scaled = f32::NEG_INFINITY;
            for rank in 0..top_k {
                scaled[rank] = best_logits[rank] / temperature;
                max_scaled = max_scaled.max(scaled[rank]);
            }
            let mut probs = vec![0.0_f32; top_k];
            let mut total = 0.0_f32;
            for rank in 0..top_k {
                probs[rank] = (scaled[rank] - max_scaled).exp();
                total += probs[rank];
            }
            total = total.max(1.0e-20);
            for prob in probs.iter_mut() {
                *prob /= total;
            }

            let top_p_clamped = top_p.clamp(1.0e-6, 1.0);
            let mut nucleus_mass = 0.0_f32;
            let mut nucleus_count = 0;
            for (rank, prob) in probs.iter().enumerate() {
                nucleus_mass += *prob;
                nucleus_count = rank + 1;
                if nucleus_mass >= top_p_clamped {
                    break;
                }
            }
            nucleus_mass = nucleus_mass.max(1.0e-20);

            let target = random_uniforms[row].clamp(0.0, 0.99999994) * nucleus_mass;
            let mut cumulative = 0.0_f32;
            let mut selected_rank = nucleus_count - 1;
            for (rank, prob) in probs.iter().enumerate().take(nucleus_count) {
                cumulative += *prob;
                if target <= cumulative {
                    selected_rank = rank;
                    break;
                }
            }
            indices[row] = best_indices[selected_rank];
            scores[row] = probs[selected_rank] / nucleus_mass;
        }
        LogitsArgmaxExpected { indices, scores }
    }

    fn silu_gated_mlp_rows_expected(
        input: &[f32],
        gate_weight: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows * hidden];
        for row in 0..rows {
            for out_col in 0..hidden {
                let mut acc = 0.0_f32;
                for mid in 0..intermediate {
                    let mut gate = 0.0_f32;
                    let mut up = 0.0_f32;
                    for col in 0..hidden {
                        gate += input[row * hidden + col] * gate_weight[mid * hidden + col];
                        up += input[row * hidden + col] * up_weight[mid * hidden + col];
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    acc += silu * up * down_weight[out_col * intermediate + mid];
                }
                output[row * hidden + out_col] = acc;
            }
        }
        output
    }

    fn silu_gated_mlp_rows_bf16_down_stride_expected(
        input: &[u16],
        gate_weight: &[u16],
        up_weight: &[u16],
        down_weight: &[u16],
        rows: usize,
        hidden: usize,
        intermediate: usize,
        down_stride: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), rows * hidden);
        assert_eq!(gate_weight.len(), intermediate * hidden);
        assert_eq!(up_weight.len(), intermediate * hidden);
        assert_eq!(down_weight.len(), hidden * down_stride);
        let mut output = vec![0.0_f32; rows * hidden];
        for row in 0..rows {
            for out_col in 0..hidden {
                let mut acc = 0.0_f32;
                for mid in 0..intermediate {
                    let mut gate = 0.0_f32;
                    let mut up = 0.0_f32;
                    for col in 0..hidden {
                        gate += bf16_to_f32(input[row * hidden + col])
                            * bf16_to_f32(gate_weight[mid * hidden + col]);
                        up += bf16_to_f32(input[row * hidden + col])
                            * bf16_to_f32(up_weight[mid * hidden + col]);
                    }
                    let silu = gate / (1.0 + (-gate).exp());
                    acc += silu * up * bf16_to_f32(down_weight[out_col * down_stride + mid]);
                }
                output[row * hidden + out_col] = acc;
            }
        }
        output
    }

    fn nvfp4_code_value(code: u8) -> f32 {
        const CODEBOOK: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        CODEBOOK[(code & 0x0f) as usize]
    }

    fn f8e4m3_to_f32(byte: u8) -> f32 {
        if byte == 0 || byte == 0x80 {
            return 0.0;
        }
        let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = ((byte >> 3) & 0x0f) as i32;
        let mantissa = (byte & 0x07) as f32;
        let significand = if exponent == 0 {
            mantissa / 8.0
        } else {
            1.0 + mantissa / 8.0
        };
        let exponent_power = if exponent == 0 { -6 } else { exponent - 7 };
        sign * significand * 2.0_f32.powi(exponent_power)
    }

    fn packed_nvfp4_value(
        packed_row: &[u8],
        scale_row: &[u8],
        value_idx: usize,
        scale_2: f32,
    ) -> f32 {
        let packed = packed_row[value_idx / 2];
        let code = if value_idx % 2 == 0 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        nvfp4_code_value(code) * f8e4m3_to_f32(scale_row[value_idx / 16]) * scale_2
    }

    fn dot_packed_nvfp4(input: &[f32], packed_row: &[u8], scale_row: &[u8], scale_2: f32) -> f32 {
        input
            .iter()
            .enumerate()
            .map(|(idx, value)| value * packed_nvfp4_value(packed_row, scale_row, idx, scale_2))
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn nvfp4_route_expected(
        hidden: &[f32],
        gate_weight: &[u8],
        gate_scale: &[u8],
        up_weight: &[u8],
        up_scale: &[u8],
        down_weight: &[u8],
        down_scale: &[u8],
        hidden_dim: usize,
        intermediate: usize,
        output_dim: usize,
        gate_scale_2: f32,
        up_scale_2: f32,
        down_scale_2: f32,
        route_weight: f32,
    ) -> Vec<f32> {
        let packed_hidden_bytes = hidden_dim.div_ceil(2);
        let hidden_scale_bytes = hidden_dim.div_ceil(16);
        let packed_intermediate_bytes = intermediate.div_ceil(2);
        let intermediate_scale_bytes = intermediate.div_ceil(16);
        let mut activations = Vec::with_capacity(intermediate);
        for mid in 0..intermediate {
            let gate_start = mid * packed_hidden_bytes;
            let gate_scale_start = mid * hidden_scale_bytes;
            let up_start = mid * packed_hidden_bytes;
            let up_scale_start = mid * hidden_scale_bytes;
            let gate = dot_packed_nvfp4(
                hidden,
                &gate_weight[gate_start..gate_start + packed_hidden_bytes],
                &gate_scale[gate_scale_start..gate_scale_start + hidden_scale_bytes],
                gate_scale_2,
            );
            let up = dot_packed_nvfp4(
                hidden,
                &up_weight[up_start..up_start + packed_hidden_bytes],
                &up_scale[up_scale_start..up_scale_start + hidden_scale_bytes],
                up_scale_2,
            );
            activations.push(gate / (1.0 + (-gate).exp()) * up);
        }
        let mut outputs = vec![0.0_f32; output_dim];
        for out_col in 0..output_dim {
            let down_start = out_col * packed_intermediate_bytes;
            let down_scale_start = out_col * intermediate_scale_bytes;
            let mut acc = 0.0_f32;
            for mid in 0..intermediate {
                acc += activations[mid]
                    * packed_nvfp4_value(
                        &down_weight[down_start..down_start + packed_intermediate_bytes],
                        &down_scale[down_scale_start..down_scale_start + intermediate_scale_bytes],
                        mid,
                        down_scale_2,
                    );
            }
            outputs[out_col] = route_weight * acc;
        }
        outputs
    }

    fn causal_attention_expected(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rows: usize,
        heads: usize,
        qk_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows * heads * v_dim];
        for row in 0..rows {
            for head in 0..heads {
                let q_base = (row * heads + head) * qk_dim;
                let q_vec = &q[q_base..q_base + qk_dim];
                let mut max_score = f32::NEG_INFINITY;
                for key_row in 0..=row {
                    let k_base = (key_row * heads + head) * qk_dim;
                    let mut dot = 0.0_f32;
                    for col in 0..qk_dim {
                        dot += q_vec[col] * k[k_base + col];
                    }
                    max_score = max_score.max(dot * scale);
                }
                for v_col in 0..v_dim {
                    let mut denom = 0.0_f32;
                    let mut acc = 0.0_f32;
                    for key_row in 0..=row {
                        let k_base = (key_row * heads + head) * qk_dim;
                        let mut dot = 0.0_f32;
                        for col in 0..qk_dim {
                            dot += q_vec[col] * k[k_base + col];
                        }
                        let weight = (dot * scale - max_score).exp();
                        denom += weight;
                        acc += weight * v[(key_row * heads + head) * v_dim + v_col];
                    }
                    output[(row * heads + head) * v_dim + v_col] = acc / denom.max(1.0e-12);
                }
            }
        }
        output
    }

    fn rope_expected(
        input: &[f32],
        positions: &[u32],
        rows: usize,
        heads: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        assert_eq!(input.len(), rows * heads * rotary_dim);
        assert_eq!(positions.len(), rows);
        let mut output = vec![0.0_f32; input.len()];
        for row in 0..rows {
            for head in 0..heads {
                for pair in 0..rotary_dim / 2 {
                    let offset = (row * heads + head) * rotary_dim + pair * 2;
                    let angle =
                        positions[row] as f32 * theta.powf(-2.0 * pair as f32 / rotary_dim as f32);
                    let cos_value = angle.cos();
                    let sin_value = angle.sin();
                    let even = input[offset];
                    let odd = input[offset + 1];
                    output[offset] = even * cos_value - odd * sin_value;
                    output[offset + 1] = even * sin_value + odd * cos_value;
                }
            }
        }
        output
    }

    fn mla_rope_attention_expected(
        q_nope: &[f32],
        q_rope: &[f32],
        k_nope: &[f32],
        k_rope: &[f32],
        v: &[f32],
        rows: usize,
        heads: usize,
        nope_dim: usize,
        rope_dim: usize,
        v_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows * heads * v_dim];
        for row in 0..rows {
            for head in 0..heads {
                let q_nope_base = (row * heads + head) * nope_dim;
                let q_rope_base = (row * heads + head) * rope_dim;
                let q_nope_vec = &q_nope[q_nope_base..q_nope_base + nope_dim];
                let q_rope_vec = &q_rope[q_rope_base..q_rope_base + rope_dim];
                let mut max_score = f32::NEG_INFINITY;
                for key_row in 0..=row {
                    let k_nope_base = (key_row * heads + head) * nope_dim;
                    let k_rope_base = key_row * rope_dim;
                    let mut nope_dot = 0.0_f32;
                    for col in 0..nope_dim {
                        nope_dot += q_nope_vec[col] * k_nope[k_nope_base + col];
                    }
                    let mut rope_dot = 0.0_f32;
                    for col in 0..rope_dim {
                        rope_dot += q_rope_vec[col] * k_rope[k_rope_base + col];
                    }
                    max_score = max_score.max((nope_dot + rope_dot) * scale);
                }
                for v_col in 0..v_dim {
                    let mut denom = 0.0_f32;
                    let mut acc = 0.0_f32;
                    for key_row in 0..=row {
                        let k_nope_base = (key_row * heads + head) * nope_dim;
                        let k_rope_base = key_row * rope_dim;
                        let mut nope_dot = 0.0_f32;
                        for col in 0..nope_dim {
                            nope_dot += q_nope_vec[col] * k_nope[k_nope_base + col];
                        }
                        let mut rope_dot = 0.0_f32;
                        for col in 0..rope_dim {
                            rope_dot += q_rope_vec[col] * k_rope[k_rope_base + col];
                        }
                        let weight = ((nope_dot + rope_dot) * scale - max_score).exp();
                        denom += weight;
                        acc += weight * v[(key_row * heads + head) * v_dim + v_col];
                    }
                    output[(row * heads + head) * v_dim + v_col] = acc / denom.max(1.0e-12);
                }
            }
        }
        output
    }

    fn assert_cuda_unavailable(err: anyhow::Error) {
        let err = err.to_string();
        assert!(
            err.contains(&format!("status {DS41RT_STATUS_CUDA_UNAVAILABLE}")),
            "{err}"
        );
        assert!(err.contains("CUDA") || err.contains("cuda"), "{err}");
    }

    #[test]
    fn native_version_call() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let version = library.version()?;
        assert!(version.contains("ds41rt_native"));
        Ok(())
    }

    #[test]
    fn cuda_device_info_call() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        assert_eq!(info.device_id, 0);
        assert!(!c_char_array_to_string(&info.name).is_empty());
        Ok(())
    }

    #[test]
    fn allocate_copy_free_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"ds41rt native ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut buffer = library.alloc_device_buffer(input.len())?;

        library.copy_h2d(buffer, &input)?;
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut buffer)?;
        assert!(buffer.ptr.is_null());
        assert_eq!(buffer.bytes, 0);
        Ok(())
    }

    #[test]
    fn managed_device_buffer_copy_free_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"ds41rt managed native ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut buffer = library.alloc_managed_device_buffer(input.len())?;

        assert!(!buffer.ptr.is_null());
        assert!(buffer.bytes >= input.len());
        assert_ne!(buffer.flags & DS41RT_DEVICE_BUFFER_FLAG_MANAGED, 0);
        library.copy_h2d(buffer, &input)?;
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut buffer)?;
        assert!(buffer.ptr.is_null());
        assert_eq!(buffer.bytes, 0);
        Ok(())
    }

    #[test]
    fn copy_h2d_reuses_synchronous_pinned_staging() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let first = b"ds41rt native ffi reusable sync h2d staging".to_vec();
        let second = b"shorter sync h2d payload".to_vec();
        let mut output = vec![0_u8; first.len()];
        let mut buffer = library.alloc_device_buffer(first.len())?;

        library.copy_h2d(buffer, &first)?;
        let first_staging = library
            .sync_h2d_staging_snapshot()
            .expect("copy_h2d allocates reusable staging");
        library.copy_d2h(&mut output, buffer)?;
        assert_eq!(output, first);

        library.copy_h2d(buffer, &second)?;
        let second_staging = library
            .sync_h2d_staging_snapshot()
            .expect("copy_h2d keeps reusable staging");
        output[..second.len()].fill(0);
        library.copy_d2h(&mut output[..second.len()], buffer)?;
        assert_eq!(&output[..second.len()], second.as_slice());
        assert_eq!(second_staging.0, first_staging.0);
        assert_eq!(second_staging.1, first_staging.1);
        assert!(second_staging.1 >= first.len());

        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn device_to_device_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"ds41rt native ffi d2d roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut src = library.alloc_device_buffer(input.len())?;
        let mut dst = library.alloc_device_buffer(input.len())?;

        library.copy_h2d(src, &input)?;
        library.copy_d2d(dst, src, input.len())?;
        library.copy_d2h(&mut output, dst)?;
        assert_eq!(output, input);

        library.free_device_buffer(&mut src)?;
        library.free_device_buffer(&mut dst)?;
        Ok(())
    }

    #[test]
    fn pinned_host_buffer_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let input = b"ds41rt pinned staging ffi roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut host = library.alloc_host_buffer(input.len())?;
        assert!(!host.ptr.is_null());
        assert_eq!(host.bytes, input.len());
        assert_ne!(host.flags, DS41RT_HOST_BUFFER_FLAG_NONE);
        assert_ne!(host.flags & DS41RT_HOST_BUFFER_FLAG_PINNED, 0);
        assert_ne!(host.flags & DS41RT_HOST_BUFFER_FLAG_MAPPED, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), host.ptr.cast::<u8>(), input.len());
        }

        let mut device = library.alloc_device_buffer(input.len())?;
        library.copy_host_buffer_h2d(device, host, input.len())?;
        library.copy_d2h(&mut output, device)?;
        assert_eq!(output, input);

        let alias = library.cuda_host_buffer_device_alias(host)?;
        assert_eq!(alias.bytes, host.bytes);
        assert_ne!(alias.flags & DS41RT_DEVICE_BUFFER_FLAG_MAPPED_HOST, 0);
        unsafe {
            library.cuda_zero_bytes_async(alias, input.len(), std::ptr::null_mut())?;
            library.cuda_stream_synchronize(std::ptr::null_mut())?;
        }
        assert!(
            unsafe { std::slice::from_raw_parts(host.ptr.cast::<u8>(), input.len()) }
                .iter()
                .all(|byte| *byte == 0)
        );

        output.fill(0);
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), host.ptr.cast::<u8>(), input.len());
        }
        unsafe {
            library.copy_host_buffer_h2d_async(device, host, input.len(), std::ptr::null_mut())?;
        }
        library.copy_d2h(&mut output, device)?;
        assert_eq!(output, input);

        let pitched_source = [1_u8, 2, 3, 90, 91, 4, 5, 6, 92, 93, 7, 8, 9, 94, 95];
        let pitched_expected = [1_u8, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
        let mut pitched_host = library.alloc_host_buffer(pitched_source.len())?;
        let mut pitched_device = library.alloc_device_buffer(pitched_expected.len())?;
        library.copy_h2d(pitched_device, &[0_u8; 12])?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                pitched_source.as_ptr(),
                pitched_host.ptr.cast::<u8>(),
                pitched_source.len(),
            );
            library.copy_host_buffer_h2d_2d_async(
                pitched_device,
                4,
                pitched_host,
                5,
                3,
                3,
                std::ptr::null_mut(),
            )?;
        }
        let mut pitched_output = [0_u8; 12];
        library.copy_d2h(&mut pitched_output, pitched_device)?;
        assert_eq!(pitched_output, pitched_expected);
        library.free_device_buffer(&mut pitched_device)?;
        library.free_host_buffer(&mut pitched_host)?;

        library.free_device_buffer(&mut device)?;
        library.free_host_buffer(&mut host)?;
        assert!(host.ptr.is_null());
        assert_eq!(host.bytes, 0);
        assert_eq!(host.flags, DS41RT_HOST_BUFFER_FLAG_NONE);
        Ok(())
    }

    #[test]
    fn batched_pinned_host_buffer_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        if library.cuda_device_info(0)?.cuda_available == 0 {
            return Ok(());
        }
        let inputs = [
            b"batched-alpha".as_slice(),
            b"batched-metadata-beta".as_slice(),
            b"gamma".as_slice(),
        ];
        let mut hosts = [
            library.alloc_host_buffer(inputs[0].len())?,
            library.alloc_host_buffer(inputs[1].len())?,
            library.alloc_host_buffer(inputs[2].len())?,
        ];
        let mut devices = [
            library.alloc_device_buffer(inputs[0].len())?,
            library.alloc_device_buffer(inputs[1].len())?,
            library.alloc_device_buffer(inputs[2].len())?,
        ];
        for (input, host) in inputs.iter().zip(&hosts) {
            unsafe {
                std::ptr::copy_nonoverlapping(input.as_ptr(), host.ptr.cast::<u8>(), input.len());
            }
        }
        let sizes = inputs.map(<[u8]>::len);
        let stream = library.cuda_stream_create()?;
        unsafe {
            library.copy_host_buffers_h2d_batch_async(&devices, &hosts, &sizes, stream)?;
            library.cuda_stream_synchronize(stream)?;
            library.cuda_stream_destroy(stream)?;
        }
        for ((input, device), output) in inputs
            .iter()
            .zip(&devices)
            .zip(inputs.map(|input| vec![0_u8; input.len()]))
        {
            let mut output = output;
            library.copy_d2h(&mut output, *device)?;
            assert_eq!(output.as_slice(), *input);
        }
        for device in &mut devices {
            library.free_device_buffer(device)?;
        }
        for host in &mut hosts {
            library.free_host_buffer(host)?;
        }
        Ok(())
    }

    #[test]
    fn async_device_to_device_copy_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                std::ptr::null_mut()
            }
        };
        let input = b"ds41rt native ffi async d2d roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut src = library.alloc_device_buffer(input.len())?;
        let mut dst = library.alloc_device_buffer(input.len())?;

        let result = unsafe {
            library.copy_h2d_async(src, &input, cuda_stream)?;
            library.copy_d2d_async(dst, src, input.len(), cuda_stream)?;
            library.copy_d2h_async(&mut output, dst, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            Ok::<(), anyhow::Error>(())
        };
        let destroy_result = unsafe { library.cuda_stream_destroy(cuda_stream) };
        library.free_device_buffer(&mut src)?;
        library.free_device_buffer(&mut dst)?;
        result?;
        destroy_result?;
        assert_eq!(output, input);
        Ok(())
    }

    #[test]
    fn stream_and_async_copy_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let input = b"ds41rt native ffi async copy roundtrip".to_vec();
        let mut output = vec![0_u8; input.len()];
        let mut buffer = library.alloc_device_buffer(input.len())?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                std::ptr::null_mut()
            }
        };

        unsafe {
            library.copy_h2d_async(buffer, &input, cuda_stream)?;
            library.copy_d2h_async(&mut output, buffer, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_stream_destroy(cuda_stream)?;
            library.cuda_stream_destroy(std::ptr::null_mut())?;
        }
        assert_eq!(output, input);

        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_retained_capture_ffi_binding_reports_node_counts() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let x = [1.0_f32, -2.0, 0.5];
        let weight = [1.0_f32, 0.5, 1.5];
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut weight_buffer = library.alloc_device_buffer(std::mem::size_of_val(&weight))?;
        let mut out_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;

        unsafe {
            library.copy_h2d_async(x_buffer, f32_bytes(&x), cuda_stream)?;
            library.copy_h2d_async(weight_buffer, f32_bytes(&weight), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_rmsnorm_f32_async(
                x_buffer,
                weight_buffer,
                out_buffer,
                1,
                3,
                1.0e-5,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert!(!capture.graph.is_null());
            assert!(!capture.graph_exec.is_null());
            assert_eq!(capture.node_count, 1);
            assert_eq!(capture.kernel_node_count, 1);
            assert_eq!(capture.memcpy_node_count, 0);
            assert_eq!(capture.memset_node_count, 0);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_graph_exec_destroy(std::ptr::null_mut())?;
            library.cuda_graph_destroy(std::ptr::null_mut())?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_rmsnorm_bf16_node_swaps_weight_pointer() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let x = bf16_values(&[1.0_f32, 2.0, -1.0, -2.0]);
        let weight_a = bf16_values(&[1.0_f32, 1.0]);
        let weight_b = bf16_values(&[0.5_f32, 2.0]);
        let rows = 2_i32;
        let hidden = 2_i32;
        let eps = 1.0e-5_f32;
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut weight_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_a.as_slice()))?;
        let mut weight_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_b.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut out_bytes = vec![0_u8; std::mem::size_of_val(x.as_slice())];

        unsafe {
            library.copy_h2d_async(x_buffer, u16_bytes(&x), cuda_stream)?;
            library.copy_h2d_async(weight_a_buffer, u16_bytes(&weight_a), cuda_stream)?;
            library.copy_h2d_async(weight_b_buffer, u16_bytes(&weight_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_rmsnorm_bf16_async(
                x_buffer,
                weight_a_buffer,
                out_buffer,
                rows,
                hidden,
                eps,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_bytes);

            library.cuda_graph_update_rmsnorm_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                x_buffer,
                weight_b_buffer,
                out_buffer,
                rows,
                hidden,
                eps,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_bytes);

            assert!(out_a[0] > out_b[0]);
            assert!(out_a[1] < out_b[1]);
            assert!(out_a[2] < out_b[2]);
            assert!(out_a[3] > out_b[3]);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut weight_b_buffer)?;
        library.free_device_buffer(&mut weight_a_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_layernorm_affine_f32_bf16_node_swaps_weight_bias_and_output_pointers(
    ) -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let x = [1.0_f32, -1.0, 0.5, 2.0];
        let weight_a = bf16_values(&[1.0_f32, 1.0]);
        let bias_a = bf16_values(&[0.0_f32, 0.0]);
        let weight_b = bf16_values(&[0.5_f32, 2.0]);
        let bias_b = bf16_values(&[0.25_f32, -0.5]);
        let rows = 2_i32;
        let hidden = 2_i32;
        let eps = 1.0e-5_f32;
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut weight_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_a.as_slice()))?;
        let mut bias_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_a.as_slice()))?;
        let mut weight_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_b.as_slice()))?;
        let mut bias_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_b.as_slice()))?;
        let mut out_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut out_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut out_bytes = vec![0_u8; std::mem::size_of_val(&x)];

        unsafe {
            library.copy_h2d_async(x_buffer, f32_bytes(&x), cuda_stream)?;
            library.copy_h2d_async(weight_a_buffer, u16_bytes(&weight_a), cuda_stream)?;
            library.copy_h2d_async(bias_a_buffer, u16_bytes(&bias_a), cuda_stream)?;
            library.copy_h2d_async(weight_b_buffer, u16_bytes(&weight_b), cuda_stream)?;
            library.copy_h2d_async(bias_b_buffer, u16_bytes(&bias_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_layernorm_affine_f32_bf16_async(
                x_buffer,
                weight_a_buffer,
                bias_a_buffer,
                out_a_buffer,
                rows,
                hidden,
                eps,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_a_buffer)?;
            let out_a = bytes_to_f32_vec(&out_bytes);

            library.cuda_graph_update_layernorm_affine_f32_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                x_buffer,
                weight_b_buffer,
                bias_b_buffer,
                out_b_buffer,
                rows,
                hidden,
                eps,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_b_buffer)?;
            let out_b = bytes_to_f32_vec(&out_bytes);

            assert!(out_a[0] > out_b[0]);
            assert!(out_a[1] > out_b[1]);
            assert!(out_a[2] < out_b[2]);
            assert!(out_a[3] < out_b[3]);
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut bias_b_buffer)?;
        library.free_device_buffer(&mut weight_b_buffer)?;
        library.free_device_buffer(&mut bias_a_buffer)?;
        library.free_device_buffer(&mut weight_a_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_layernorm_affine_bf16_node_swaps_weight_bias_and_output_pointers(
    ) -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let x = bf16_values(&[1.0_f32, -1.0, 0.5, 2.0]);
        let weight_a = bf16_values(&[1.0_f32, 1.0]);
        let bias_a = bf16_values(&[0.0_f32, 0.0]);
        let weight_b = bf16_values(&[0.5_f32, 2.0]);
        let bias_b = bf16_values(&[0.25_f32, -0.5]);
        let rows = 2_i32;
        let hidden = 2_i32;
        let eps = 1.0e-5_f32;
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut weight_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_a.as_slice()))?;
        let mut bias_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_a.as_slice()))?;
        let mut weight_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_b.as_slice()))?;
        let mut bias_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_b.as_slice()))?;
        let mut out_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut out_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut out_bytes = vec![0_u8; std::mem::size_of_val(x.as_slice())];

        unsafe {
            library.copy_h2d_async(x_buffer, u16_bytes(&x), cuda_stream)?;
            library.copy_h2d_async(weight_a_buffer, u16_bytes(&weight_a), cuda_stream)?;
            library.copy_h2d_async(bias_a_buffer, u16_bytes(&bias_a), cuda_stream)?;
            library.copy_h2d_async(weight_b_buffer, u16_bytes(&weight_b), cuda_stream)?;
            library.copy_h2d_async(bias_b_buffer, u16_bytes(&bias_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_layernorm_affine_bf16_async(
                x_buffer,
                weight_a_buffer,
                bias_a_buffer,
                out_a_buffer,
                rows,
                hidden,
                eps,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_bytes);

            library.cuda_graph_update_layernorm_affine_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                x_buffer,
                weight_b_buffer,
                bias_b_buffer,
                out_b_buffer,
                rows,
                hidden,
                eps,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_bytes);

            assert!(out_a[0] > out_b[0]);
            assert!(out_a[1] > out_b[1]);
            assert!(out_a[2] < out_b[2]);
            assert!(out_a[3] < out_b[3]);
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut bias_b_buffer)?;
        library.free_device_buffer(&mut weight_b_buffer)?;
        library.free_device_buffer(&mut bias_a_buffer)?;
        library.free_device_buffer(&mut weight_a_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_linear_bf16_node_swaps_weight_and_bias_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let input = bf16_values(&[1.0_f32, 2.0, -1.0, 0.5]);
        let weight_a = bf16_values(&[1.0_f32, 0.0, 0.0, 1.0]);
        let bias_a = bf16_values(&[0.0_f32, 0.0]);
        let weight_b = bf16_values(&[2.0_f32, 0.0, 0.0, -1.0]);
        let bias_b = bf16_values(&[0.5_f32, 1.0]);
        let rows = 2_usize;
        let input_dim = 2_usize;
        let output_dim = 2_usize;
        let mut input_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input.as_slice()))?;
        let mut weight_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_a.as_slice()))?;
        let mut bias_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_a.as_slice()))?;
        let mut weight_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight_b.as_slice()))?;
        let mut bias_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias_b.as_slice()))?;
        let mut out_buffer =
            library.alloc_device_buffer(rows * output_dim * std::mem::size_of::<u16>())?;
        let mut out_bytes = vec![0_u8; rows * output_dim * std::mem::size_of::<u16>()];

        unsafe {
            library.copy_h2d_async(input_buffer, u16_bytes(&input), cuda_stream)?;
            library.copy_h2d_async(weight_a_buffer, u16_bytes(&weight_a), cuda_stream)?;
            library.copy_h2d_async(bias_a_buffer, u16_bytes(&bias_a), cuda_stream)?;
            library.copy_h2d_async(weight_b_buffer, u16_bytes(&weight_b), cuda_stream)?;
            library.copy_h2d_async(bias_b_buffer, u16_bytes(&bias_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_linear_bf16_async(
                input_buffer,
                weight_a_buffer,
                Some(bias_a_buffer),
                out_buffer,
                rows,
                input_dim,
                output_dim,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_bytes);
            assert!((out_a[0] - 1.0).abs() < 1.0e-3);
            assert!((out_a[1] - 2.0).abs() < 1.0e-3);
            assert!((out_a[2] + 1.0).abs() < 1.0e-3);
            assert!((out_a[3] - 0.5).abs() < 1.0e-3);

            library.cuda_graph_update_linear_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                input_buffer,
                weight_b_buffer,
                Some(bias_b_buffer),
                out_buffer,
                rows,
                input_dim,
                output_dim,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_bytes);
            assert!((out_b[0] - 2.5).abs() < 1.0e-3);
            assert!((out_b[1] + 1.0).abs() < 1.0e-3);
            assert!((out_b[2] + 1.5).abs() < 1.0e-3);
            assert!((out_b[3] - 0.5).abs() < 1.0e-3);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut bias_b_buffer)?;
        library.free_device_buffer(&mut weight_b_buffer)?;
        library.free_device_buffer(&mut bias_a_buffer)?;
        library.free_device_buffer(&mut weight_a_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_router_topk_bf16_node_swaps_weight_and_bias_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let rows = 2_usize;
        let hidden_dim = 3_usize;
        let experts = 4_usize;
        let top_k = 2_usize;
        let hidden_f32 = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let router_weight_a_f32 = [
            3.0_f32, 0.0, 0.0, 0.0, 3.0, 0.0, -3.0, 0.0, 0.0, 0.0, -3.0, 0.0,
        ];
        let router_weight_b_f32 = [
            -3.0_f32, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0,
        ];
        let correction_bias_a = [0.0_f32, 0.0, 0.0, 0.0];
        let correction_bias_b = [0.0_f32, 0.0, 0.25, 0.25];
        let hidden = bf16_values(&hidden_f32);
        let router_weight_a = bf16_values(&router_weight_a_f32);
        let router_weight_b = bf16_values(&router_weight_b_f32);
        let hidden_expected: Vec<f32> = hidden.iter().copied().map(bf16_to_f32).collect();
        let router_weight_a_expected: Vec<f32> =
            router_weight_a.iter().copied().map(bf16_to_f32).collect();
        let router_weight_b_expected: Vec<f32> =
            router_weight_b.iter().copied().map(bf16_to_f32).collect();
        let expected_a = router_topk_expected(
            &hidden_expected,
            &router_weight_a_expected,
            &correction_bias_a,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
        let expected_b = router_topk_expected(
            &hidden_expected,
            &router_weight_b_expected,
            &correction_bias_b,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
        let output_values = rows * top_k;
        let index_bytes_len = output_values * std::mem::size_of::<u32>();
        let score_bytes_len = output_values * std::mem::size_of::<f32>();
        let mut index_bytes = vec![0_u8; index_bytes_len];
        let mut score_bytes = vec![0_u8; score_bytes_len];
        let mut weight_bytes = vec![0_u8; score_bytes_len];

        let mut hidden_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(hidden.as_slice()))?;
        let mut router_weight_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(router_weight_a.as_slice()))?;
        let mut correction_bias_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&correction_bias_a))?;
        let mut router_weight_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(router_weight_b.as_slice()))?;
        let mut correction_bias_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&correction_bias_b))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes_len)?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes_len)?;
        let mut weight_buffer = library.alloc_device_buffer(score_bytes_len)?;

        unsafe {
            library.copy_h2d_async(hidden_buffer, u16_bytes(&hidden), cuda_stream)?;
            library.copy_h2d_async(
                router_weight_a_buffer,
                u16_bytes(&router_weight_a),
                cuda_stream,
            )?;
            library.copy_h2d_async(
                correction_bias_a_buffer,
                f32_bytes(&correction_bias_a),
                cuda_stream,
            )?;
            library.copy_h2d_async(
                router_weight_b_buffer,
                u16_bytes(&router_weight_b),
                cuda_stream,
            )?;
            library.copy_h2d_async(
                correction_bias_b_buffer,
                f32_bytes(&correction_bias_b),
                cuda_stream,
            )?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_router_topk_bf16_async(
                hidden_buffer,
                router_weight_a_buffer,
                correction_bias_a_buffer,
                index_buffer,
                score_buffer,
                weight_buffer,
                rows,
                hidden_dim,
                experts,
                top_k,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 3);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut index_bytes, index_buffer)?;
            library.copy_d2h(&mut score_bytes, score_buffer)?;
            library.copy_d2h(&mut weight_bytes, weight_buffer)?;
            assert_eq!(bytes_to_u32_vec(&index_bytes), expected_a.indices);
            for (actual, expected) in bytes_to_f32_vec(&score_bytes)
                .iter()
                .zip(expected_a.scores.iter())
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            for (actual, expected) in bytes_to_f32_vec(&weight_bytes)
                .iter()
                .zip(expected_a.weights.iter())
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }

            library.cuda_graph_update_router_topk_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                hidden_buffer,
                router_weight_b_buffer,
                correction_bias_b_buffer,
                index_buffer,
                score_buffer,
                weight_buffer,
                rows,
                hidden_dim,
                experts,
                top_k,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut index_bytes, index_buffer)?;
            library.copy_d2h(&mut score_bytes, score_buffer)?;
            library.copy_d2h(&mut weight_bytes, weight_buffer)?;
            assert_eq!(bytes_to_u32_vec(&index_bytes), expected_b.indices);
            for (actual, expected) in bytes_to_f32_vec(&score_bytes)
                .iter()
                .zip(expected_b.scores.iter())
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            for (actual, expected) in bytes_to_f32_vec(&weight_bytes)
                .iter()
                .zip(expected_b.weights.iter())
            {
                assert!((actual - expected).abs() < 1.0e-6);
            }

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut correction_bias_b_buffer)?;
        library.free_device_buffer(&mut router_weight_b_buffer)?;
        library.free_device_buffer(&mut correction_bias_a_buffer)?;
        library.free_device_buffer(&mut router_weight_a_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node_swaps_weight_pointers(
    ) -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let rows = 2_usize;
        let hidden = 2_usize;
        let intermediate = 2_usize;
        let down_stride = 3_usize;
        let input = bf16_values(&[0.25_f32, -0.5, 1.0, 0.75]);
        let gate_a = bf16_values(&[0.2_f32, -0.1, 0.4, 0.3]);
        let up_a = bf16_values(&[0.5_f32, 0.25, -0.3, 0.2]);
        let down_a = bf16_values(&[0.3_f32, -0.2, 99.0, 0.1, 0.4, -99.0]);
        let gate_b = bf16_values(&[-0.3_f32, 0.2, 0.1, -0.5]);
        let up_b = bf16_values(&[0.2_f32, -0.4, 0.6, 0.1]);
        let down_b = bf16_values(&[-0.5_f32, 0.3, 77.0, 0.25, -0.15, -77.0]);
        let expected_a = silu_gated_mlp_rows_bf16_down_stride_expected(
            &input,
            &gate_a,
            &up_a,
            &down_a,
            rows,
            hidden,
            intermediate,
            down_stride,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let expected_b = silu_gated_mlp_rows_bf16_down_stride_expected(
            &input,
            &gate_b,
            &up_b,
            &down_b,
            rows,
            hidden,
            intermediate,
            down_stride,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; rows * hidden * std::mem::size_of::<u16>()];

        let mut input_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input.as_slice()))?;
        let mut gate_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(gate_a.as_slice()))?;
        let mut up_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(up_a.as_slice()))?;
        let mut down_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(down_a.as_slice()))?;
        let mut gate_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(gate_b.as_slice()))?;
        let mut up_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(up_b.as_slice()))?;
        let mut down_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(down_b.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        unsafe {
            library.copy_h2d_async(input_buffer, u16_bytes(&input), cuda_stream)?;
            library.copy_h2d_async(gate_a_buffer, u16_bytes(&gate_a), cuda_stream)?;
            library.copy_h2d_async(up_a_buffer, u16_bytes(&up_a), cuda_stream)?;
            library.copy_h2d_async(down_a_buffer, u16_bytes(&down_a), cuda_stream)?;
            library.copy_h2d_async(gate_b_buffer, u16_bytes(&gate_b), cuda_stream)?;
            library.copy_h2d_async(up_b_buffer, u16_bytes(&up_b), cuda_stream)?;
            library.copy_h2d_async(down_b_buffer, u16_bytes(&down_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
                input_buffer,
                gate_a_buffer,
                up_a_buffer,
                down_a_buffer,
                out_buffer,
                rows,
                hidden,
                intermediate,
                down_stride,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_bytes);
            for (actual, expected) in out_a.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            library.cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
                capture.graph,
                capture.graph_exec,
                0,
                input_buffer,
                gate_b_buffer,
                up_b_buffer,
                down_b_buffer,
                out_buffer,
                rows,
                hidden,
                intermediate,
                down_stride,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_bytes, out_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_bytes);
            for (actual, expected) in out_b.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut down_b_buffer)?;
        library.free_device_buffer(&mut up_b_buffer)?;
        library.free_device_buffer(&mut gate_b_buffer)?;
        library.free_device_buffer(&mut down_a_buffer)?;
        library.free_device_buffer(&mut up_a_buffer)?;
        library.free_device_buffer(&mut gate_a_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_residual_add_bf16_node_swaps_buffer_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let residual_a = bf16_values(&[1.0_f32, -2.0, 0.5, 4.0]);
        let delta_a = bf16_values(&[0.25_f32, 3.0, -0.75, 1.0]);
        let residual_b = bf16_values(&[2.0_f32, 8.0, -1.0, 0.125]);
        let delta_b = bf16_values(&[-0.5_f32, -0.25, 0.5, 0.125]);
        let count = residual_a.len();
        let mut out_a_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];
        let mut out_b_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];

        let mut residual_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_a.as_slice()))?;
        let mut delta_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(delta_a.as_slice()))?;
        let mut residual_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_b.as_slice()))?;
        let mut delta_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(delta_b.as_slice()))?;
        let mut out_a_buffer = library.alloc_device_buffer(out_a_bytes.len())?;
        let mut out_b_buffer = library.alloc_device_buffer(out_b_bytes.len())?;

        unsafe {
            library.copy_h2d_async(residual_a_buffer, u16_bytes(&residual_a), cuda_stream)?;
            library.copy_h2d_async(delta_a_buffer, u16_bytes(&delta_a), cuda_stream)?;
            library.copy_h2d_async(residual_b_buffer, u16_bytes(&residual_b), cuda_stream)?;
            library.copy_h2d_async(delta_b_buffer, u16_bytes(&delta_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_residual_add_bf16_async(
                residual_a_buffer,
                delta_a_buffer,
                out_a_buffer,
                count,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            assert_eq!(out_a, vec![1.25, 1.0, -0.25, 5.0]);

            library.cuda_graph_update_residual_add_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                residual_b_buffer,
                delta_b_buffer,
                out_b_buffer,
                count,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            assert_eq!(out_b, vec![1.5, 7.75, -0.5, 0.25]);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut delta_b_buffer)?;
        library.free_device_buffer(&mut residual_b_buffer)?;
        library.free_device_buffer(&mut delta_a_buffer)?;
        library.free_device_buffer(&mut residual_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_residual_add_f32_delta_bf16_node_swaps_buffer_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let residual_a = bf16_values(&[1.0_f32, -2.0, 0.5, 4.0]);
        let delta_a = [0.251_f32, 3.01, -0.751, 1.01];
        let residual_b = bf16_values(&[2.0_f32, 8.0, -1.0, 0.125]);
        let delta_b = [-0.501_f32, -0.251, 0.501, 0.126];
        let count = residual_a.len();
        let expected_a = residual_add_f32_delta_bf16_expected(&residual_a, &delta_a);
        let expected_b = residual_add_f32_delta_bf16_expected(&residual_b, &delta_b);
        let mut out_a_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];
        let mut out_b_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];

        let mut residual_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_a.as_slice()))?;
        let mut delta_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(&delta_a))?;
        let mut residual_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_b.as_slice()))?;
        let mut delta_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(&delta_b))?;
        let mut out_a_buffer = library.alloc_device_buffer(out_a_bytes.len())?;
        let mut out_b_buffer = library.alloc_device_buffer(out_b_bytes.len())?;

        unsafe {
            library.copy_h2d_async(residual_a_buffer, u16_bytes(&residual_a), cuda_stream)?;
            library.copy_h2d_async(delta_a_buffer, f32_bytes(&delta_a), cuda_stream)?;
            library.copy_h2d_async(residual_b_buffer, u16_bytes(&residual_b), cuda_stream)?;
            library.copy_h2d_async(delta_b_buffer, f32_bytes(&delta_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_residual_add_f32_delta_bf16_async(
                residual_a_buffer,
                delta_a_buffer,
                out_a_buffer,
                count,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            assert_eq!(out_a, expected_a);

            library.cuda_graph_update_residual_add_f32_delta_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                residual_b_buffer,
                delta_b_buffer,
                out_b_buffer,
                count,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            assert_eq!(out_b, expected_b);
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut delta_b_buffer)?;
        library.free_device_buffer(&mut residual_b_buffer)?;
        library.free_device_buffer(&mut delta_a_buffer)?;
        library.free_device_buffer(&mut residual_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_residual_add_shared_f32_delta_bf16_node_swaps_buffer_pointers(
    ) -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let residual_a = bf16_values(&[1.0_f32, -2.0, 0.5, 4.0]);
        let shared_a = bf16_values(&[0.25_f32, -0.5, 0.125, 2.0]);
        let routed_a = [0.251_f32, 3.01, -0.751, 1.01];
        let residual_b = bf16_values(&[2.0_f32, 8.0, -1.0, 0.125]);
        let shared_b = bf16_values(&[-0.5_f32, 0.75, 0.5, -0.25]);
        let routed_b = [-0.501_f32, -0.251, 0.501, 0.126];
        let count = residual_a.len();
        let expected_a =
            residual_add_shared_f32_delta_bf16_expected(&residual_a, &shared_a, &routed_a);
        let expected_b =
            residual_add_shared_f32_delta_bf16_expected(&residual_b, &shared_b, &routed_b);
        let mut out_a_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];
        let mut out_b_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];

        let mut residual_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_a.as_slice()))?;
        let mut shared_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(shared_a.as_slice()))?;
        let mut routed_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(&routed_a))?;
        let mut residual_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual_b.as_slice()))?;
        let mut shared_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(shared_b.as_slice()))?;
        let mut routed_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(&routed_b))?;
        let mut out_a_buffer = library.alloc_device_buffer(out_a_bytes.len())?;
        let mut out_b_buffer = library.alloc_device_buffer(out_b_bytes.len())?;

        unsafe {
            library.copy_h2d_async(residual_a_buffer, u16_bytes(&residual_a), cuda_stream)?;
            library.copy_h2d_async(shared_a_buffer, u16_bytes(&shared_a), cuda_stream)?;
            library.copy_h2d_async(routed_a_buffer, f32_bytes(&routed_a), cuda_stream)?;
            library.copy_h2d_async(residual_b_buffer, u16_bytes(&residual_b), cuda_stream)?;
            library.copy_h2d_async(shared_b_buffer, u16_bytes(&shared_b), cuda_stream)?;
            library.copy_h2d_async(routed_b_buffer, f32_bytes(&routed_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_residual_add_shared_f32_delta_bf16_async(
                residual_a_buffer,
                shared_a_buffer,
                routed_a_buffer,
                out_a_buffer,
                count,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            assert_eq!(out_a, expected_a);

            library.cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                residual_b_buffer,
                shared_b_buffer,
                routed_b_buffer,
                out_b_buffer,
                count,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            assert_eq!(out_b, expected_b);
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut routed_b_buffer)?;
        library.free_device_buffer(&mut shared_b_buffer)?;
        library.free_device_buffer(&mut residual_b_buffer)?;
        library.free_device_buffer(&mut routed_a_buffer)?;
        library.free_device_buffer(&mut shared_a_buffer)?;
        library.free_device_buffer(&mut residual_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_f32_to_bf16_node_swaps_input_output_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let src_a = [1.0_f32, -2.5, 0.125, 8.0];
        let src_b = [-4.0_f32, 0.75, 3.5, -0.0625];
        let count = src_a.len();
        let expected_a = src_a
            .iter()
            .map(|value| bf16_to_f32(f32_to_bf16(*value)))
            .collect::<Vec<_>>();
        let expected_b = src_b
            .iter()
            .map(|value| bf16_to_f32(f32_to_bf16(*value)))
            .collect::<Vec<_>>();
        let mut out_a_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];
        let mut out_b_bytes = vec![0_u8; count * std::mem::size_of::<u16>()];

        let mut src_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(&src_a))?;
        let mut src_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(&src_b))?;
        let mut out_a_buffer = library.alloc_device_buffer(out_a_bytes.len())?;
        let mut out_b_buffer = library.alloc_device_buffer(out_b_bytes.len())?;

        unsafe {
            library.copy_h2d_async(src_a_buffer, f32_bytes(&src_a), cuda_stream)?;
            library.copy_h2d_async(src_b_buffer, f32_bytes(&src_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_f32_to_bf16_async(src_a_buffer, out_a_buffer, count, cuda_stream)?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            assert_eq!(out_a, expected_a);

            library.cuda_graph_update_f32_to_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                src_b_buffer,
                out_b_buffer,
                count,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            assert_eq!(out_b, expected_b);
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut src_b_buffer)?;
        library.free_device_buffer(&mut src_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_scatter_add_rows_bf16_to_f32_node_swaps_buffer_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let dst_rows = 3_usize;
        let rows = 3_usize;
        let row_width = 2_usize;
        let src_a = bf16_values(&[1.0_f32, 2.0, 3.0, 4.0, -0.5, 0.25]);
        let indices_a = [2_u32, 0, 2];
        let dst_a_initial = [10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let expected_a = [13.0_f32, 24.0, 30.0, 40.0, 50.5, 62.25];
        let src_b = bf16_values(&[-1.0_f32, 0.5, 2.0, -0.25, 0.75, 1.25]);
        let indices_b = [1_u32, 1, 0];
        let dst_b_initial = [0.5_f32, -0.5, 1.0, 2.0, -3.0, 4.0];
        let expected_b = [1.25_f32, 0.75, 2.0, 2.25, -3.0, 4.0];
        let mut out_a_bytes = vec![0_u8; dst_a_initial.len() * std::mem::size_of::<f32>()];
        let mut out_b_bytes = vec![0_u8; dst_b_initial.len() * std::mem::size_of::<f32>()];

        let mut src_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(src_a.as_slice()))?;
        let mut indices_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices_a))?;
        let mut dst_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&dst_a_initial))?;
        let mut src_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(src_b.as_slice()))?;
        let mut indices_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices_b))?;
        let mut dst_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&dst_b_initial))?;

        unsafe {
            library.copy_h2d_async(src_a_buffer, u16_bytes(&src_a), cuda_stream)?;
            library.copy_h2d_async(indices_a_buffer, u32_bytes(&indices_a), cuda_stream)?;
            library.copy_h2d_async(dst_a_buffer, f32_bytes(&dst_a_initial), cuda_stream)?;
            library.copy_h2d_async(src_b_buffer, u16_bytes(&src_b), cuda_stream)?;
            library.copy_h2d_async(indices_b_buffer, u32_bytes(&indices_b), cuda_stream)?;
            library.copy_h2d_async(dst_b_buffer, f32_bytes(&dst_b_initial), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_scatter_add_rows_bf16_to_f32_async(
                src_a_buffer,
                indices_a_buffer,
                dst_a_buffer,
                dst_rows,
                rows,
                row_width,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, dst_a_buffer)?;
            let out_a = bytes_to_f32_vec(&out_a_bytes);
            for (actual, expected) in out_a.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-6);
            }

            library.cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
                capture.graph,
                capture.graph_exec,
                0,
                src_b_buffer,
                indices_b_buffer,
                dst_b_buffer,
                dst_rows,
                rows,
                row_width,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, dst_b_buffer)?;
            let out_b = bytes_to_f32_vec(&out_b_bytes);
            for (actual, expected) in out_b.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut dst_b_buffer)?;
        library.free_device_buffer(&mut indices_b_buffer)?;
        library.free_device_buffer(&mut src_b_buffer)?;
        library.free_device_buffer(&mut dst_a_buffer)?;
        library.free_device_buffer(&mut indices_a_buffer)?;
        library.free_device_buffer(&mut src_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_kv_cache_write_bytes_node_swaps_buffer_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let src_a = [1_u8, 2, 3, 4];
        let src_b = [9_u8, 8, 7, 6];
        let cache_a_initial = [0_u8; 8];
        let cache_b_initial = [0x55_u8; 10];
        let mut cache_a_out = vec![0_u8; cache_a_initial.len()];
        let mut cache_b_out = vec![0_u8; cache_b_initial.len()];

        let mut src_a_buffer = library.alloc_device_buffer(src_a.len())?;
        let mut src_b_buffer = library.alloc_device_buffer(src_b.len())?;
        let mut cache_a_buffer = library.alloc_device_buffer(cache_a_initial.len())?;
        let mut cache_b_buffer = library.alloc_device_buffer(cache_b_initial.len())?;

        unsafe {
            library.copy_h2d_async(src_a_buffer, &src_a, cuda_stream)?;
            library.copy_h2d_async(src_b_buffer, &src_b, cuda_stream)?;
            library.copy_h2d_async(cache_a_buffer, &cache_a_initial, cuda_stream)?;
            library.copy_h2d_async(cache_b_buffer, &cache_b_initial, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_kv_cache_write_bytes_async(
                src_a_buffer,
                cache_a_buffer,
                2,
                src_a.len(),
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut cache_a_out, cache_a_buffer)?;
            assert_eq!(cache_a_out, vec![0, 0, 1, 2, 3, 4, 0, 0]);

            library.cuda_graph_update_kv_cache_write_bytes_node(
                capture.graph,
                capture.graph_exec,
                0,
                src_b_buffer,
                cache_b_buffer,
                3,
                src_b.len(),
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut cache_b_out, cache_b_buffer)?;
            assert_eq!(
                cache_b_out,
                vec![0x55, 0x55, 0x55, 9, 8, 7, 6, 0x55, 0x55, 0x55]
            );

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut cache_b_buffer)?;
        library.free_device_buffer(&mut cache_a_buffer)?;
        library.free_device_buffer(&mut src_b_buffer)?;
        library.free_device_buffer(&mut src_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_causal_attention_bf16_node_swaps_input_output_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let rows = 2_usize;
        let heads = 1_usize;
        let qk_dim = 2_usize;
        let v_dim = 2_usize;
        let scale = 0.5_f32;
        let q_a = bf16_values(&[1.0_f32, 0.0, 0.25, -0.5]);
        let k_a = bf16_values(&[1.0_f32, 0.0, 0.5, 0.25]);
        let v_a = bf16_values(&[2.0_f32, -1.0, 0.5, 1.5]);
        let q_b = bf16_values(&[-0.75_f32, 0.5, 0.25, 1.0]);
        let k_b = bf16_values(&[-0.5_f32, 0.25, 0.75, -0.25]);
        let v_b = bf16_values(&[-1.5_f32, 0.25, 2.0, -0.5]);
        let expected_a = causal_attention_expected(
            &q_a.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &k_a.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &v_a.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let expected_b = causal_attention_expected(
            &q_b.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &k_b.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &v_b.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let output_bytes_len = v_a.len() * std::mem::size_of::<u16>();
        let mut out_a_bytes = vec![0_u8; output_bytes_len];
        let mut out_b_bytes = vec![0_u8; output_bytes_len];

        let mut q_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(q_a.as_slice()))?;
        let mut k_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(k_a.as_slice()))?;
        let mut v_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(v_a.as_slice()))?;
        let mut q_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(q_b.as_slice()))?;
        let mut k_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(k_b.as_slice()))?;
        let mut v_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(v_b.as_slice()))?;
        let mut out_a_buffer = library.alloc_device_buffer(output_bytes_len)?;
        let mut out_b_buffer = library.alloc_device_buffer(output_bytes_len)?;

        unsafe {
            library.copy_h2d_async(q_a_buffer, u16_bytes(&q_a), cuda_stream)?;
            library.copy_h2d_async(k_a_buffer, u16_bytes(&k_a), cuda_stream)?;
            library.copy_h2d_async(v_a_buffer, u16_bytes(&v_a), cuda_stream)?;
            library.copy_h2d_async(q_b_buffer, u16_bytes(&q_b), cuda_stream)?;
            library.copy_h2d_async(k_b_buffer, u16_bytes(&k_b), cuda_stream)?;
            library.copy_h2d_async(v_b_buffer, u16_bytes(&v_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_causal_attention_bf16_async(
                q_a_buffer,
                k_a_buffer,
                v_a_buffer,
                out_a_buffer,
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            for (actual, expected) in out_a.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            library.cuda_graph_update_causal_attention_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                q_b_buffer,
                k_b_buffer,
                v_b_buffer,
                out_b_buffer,
                rows,
                heads,
                qk_dim,
                v_dim,
                scale,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            for (actual, expected) in out_b.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut v_b_buffer)?;
        library.free_device_buffer(&mut k_b_buffer)?;
        library.free_device_buffer(&mut q_b_buffer)?;
        library.free_device_buffer(&mut v_a_buffer)?;
        library.free_device_buffer(&mut k_a_buffer)?;
        library.free_device_buffer(&mut q_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_rope_bf16_node_swaps_input_position_output_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let rows = 2_usize;
        let heads = 1_usize;
        let rotary_dim = 4_usize;
        let theta = 10_000.0_f32;
        let input_a = bf16_values(&[1.0_f32, 0.0, 0.0, 1.0, 0.5, -0.5, 2.0, 0.0]);
        let input_b = bf16_values(&[-0.25_f32, 0.75, 1.5, -0.5, 0.0, 1.0, -1.0, 0.5]);
        let positions_a = [0_u32, 1];
        let positions_b = [2_u32, 3];
        let expected_a = rope_expected(
            &input_a.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &positions_a,
            rows,
            heads,
            rotary_dim,
            theta,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let expected_b = rope_expected(
            &input_b.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &positions_b,
            rows,
            heads,
            rotary_dim,
            theta,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let output_bytes_len = input_a.len() * std::mem::size_of::<u16>();
        let mut out_a_bytes = vec![0_u8; output_bytes_len];
        let mut out_b_bytes = vec![0_u8; output_bytes_len];

        let mut input_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input_a.as_slice()))?;
        let mut positions_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&positions_a))?;
        let mut input_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input_b.as_slice()))?;
        let mut positions_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&positions_b))?;
        let mut out_a_buffer = library.alloc_device_buffer(output_bytes_len)?;
        let mut out_b_buffer = library.alloc_device_buffer(output_bytes_len)?;

        unsafe {
            library.copy_h2d_async(input_a_buffer, u16_bytes(&input_a), cuda_stream)?;
            library.copy_h2d_async(positions_a_buffer, u32_bytes(&positions_a), cuda_stream)?;
            library.copy_h2d_async(input_b_buffer, u16_bytes(&input_b), cuda_stream)?;
            library.copy_h2d_async(positions_b_buffer, u32_bytes(&positions_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_rope_bf16_async(
                input_a_buffer,
                positions_a_buffer,
                out_a_buffer,
                rows,
                heads,
                rotary_dim,
                theta,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            for (actual, expected) in out_a.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            library.cuda_graph_update_rope_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                input_b_buffer,
                positions_b_buffer,
                out_b_buffer,
                rows,
                heads,
                rotary_dim,
                theta,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            for (actual, expected) in out_b.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut positions_b_buffer)?;
        library.free_device_buffer(&mut input_b_buffer)?;
        library.free_device_buffer(&mut positions_a_buffer)?;
        library.free_device_buffer(&mut input_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_graph_update_mla_rope_attention_bf16_node_swaps_input_output_pointers() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let stream_result = library.cuda_stream_create();
        let cuda_stream = match stream_result {
            Ok(cuda_stream) => {
                assert_eq!(info.cuda_available, 1);
                assert!(!cuda_stream.is_null());
                cuda_stream
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                return Ok(());
            }
        };

        let rows = 2_usize;
        let heads = 1_usize;
        let nope_dim = 2_usize;
        let rope_dim = 2_usize;
        let v_dim = 2_usize;
        let scale = 0.5_f32;
        let q_nope_a = bf16_values(&[1.0_f32, 0.0, 0.25, -0.5]);
        let q_rope_a = bf16_values(&[0.0_f32, 1.0, -0.25, 0.5]);
        let k_nope_a = bf16_values(&[1.0_f32, 0.0, 0.5, 0.25]);
        let k_rope_a = bf16_values(&[0.0_f32, 1.0, 0.25, -0.25]);
        let v_a = bf16_values(&[2.0_f32, -1.0, 0.5, 1.5]);
        let q_nope_b = bf16_values(&[-0.75_f32, 0.5, 0.25, 1.0]);
        let q_rope_b = bf16_values(&[0.5_f32, -1.0, 1.25, 0.0]);
        let k_nope_b = bf16_values(&[-0.5_f32, 0.25, 0.75, -0.25]);
        let k_rope_b = bf16_values(&[0.25_f32, -0.5, 0.0, 1.0]);
        let v_b = bf16_values(&[-1.5_f32, 0.25, 2.0, -0.5]);
        let expected_bf16 = |q_nope: &[u16],
                             q_rope: &[u16],
                             k_nope: &[u16],
                             k_rope: &[u16],
                             v: &[u16]|
         -> Vec<f32> {
            let q_nope_f32: Vec<f32> = q_nope.iter().copied().map(bf16_to_f32).collect();
            let q_rope_f32: Vec<f32> = q_rope.iter().copied().map(bf16_to_f32).collect();
            let k_nope_f32: Vec<f32> = k_nope.iter().copied().map(bf16_to_f32).collect();
            let k_rope_f32: Vec<f32> = k_rope.iter().copied().map(bf16_to_f32).collect();
            let v_f32: Vec<f32> = v.iter().copied().map(bf16_to_f32).collect();
            mla_rope_attention_expected(
                &q_nope_f32,
                &q_rope_f32,
                &k_nope_f32,
                &k_rope_f32,
                &v_f32,
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )
            .into_iter()
            .map(|value| bf16_to_f32(f32_to_bf16(value)))
            .collect()
        };
        let expected_a = expected_bf16(&q_nope_a, &q_rope_a, &k_nope_a, &k_rope_a, &v_a);
        let expected_b = expected_bf16(&q_nope_b, &q_rope_b, &k_nope_b, &k_rope_b, &v_b);
        let output_bytes_len = rows * heads * v_dim * std::mem::size_of::<u16>();
        let mut out_a_bytes = vec![0_u8; output_bytes_len];
        let mut out_b_bytes = vec![0_u8; output_bytes_len];

        let mut q_nope_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_nope_a.as_slice()))?;
        let mut q_rope_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_rope_a.as_slice()))?;
        let mut k_nope_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_nope_a.as_slice()))?;
        let mut k_rope_a_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_rope_a.as_slice()))?;
        let mut v_a_buffer = library.alloc_device_buffer(std::mem::size_of_val(v_a.as_slice()))?;
        let mut q_nope_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_nope_b.as_slice()))?;
        let mut q_rope_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_rope_b.as_slice()))?;
        let mut k_nope_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_nope_b.as_slice()))?;
        let mut k_rope_b_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_rope_b.as_slice()))?;
        let mut v_b_buffer = library.alloc_device_buffer(std::mem::size_of_val(v_b.as_slice()))?;
        let mut out_a_buffer = library.alloc_device_buffer(output_bytes_len)?;
        let mut out_b_buffer = library.alloc_device_buffer(output_bytes_len)?;

        unsafe {
            library.copy_h2d_async(q_nope_a_buffer, u16_bytes(&q_nope_a), cuda_stream)?;
            library.copy_h2d_async(q_rope_a_buffer, u16_bytes(&q_rope_a), cuda_stream)?;
            library.copy_h2d_async(k_nope_a_buffer, u16_bytes(&k_nope_a), cuda_stream)?;
            library.copy_h2d_async(k_rope_a_buffer, u16_bytes(&k_rope_a), cuda_stream)?;
            library.copy_h2d_async(v_a_buffer, u16_bytes(&v_a), cuda_stream)?;
            library.copy_h2d_async(q_nope_b_buffer, u16_bytes(&q_nope_b), cuda_stream)?;
            library.copy_h2d_async(q_rope_b_buffer, u16_bytes(&q_rope_b), cuda_stream)?;
            library.copy_h2d_async(k_nope_b_buffer, u16_bytes(&k_nope_b), cuda_stream)?;
            library.copy_h2d_async(k_rope_b_buffer, u16_bytes(&k_rope_b), cuda_stream)?;
            library.copy_h2d_async(v_b_buffer, u16_bytes(&v_b), cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_graph_begin_capture(cuda_stream)?;
            library.cuda_mla_rope_attention_bf16_async(
                q_nope_a_buffer,
                q_rope_a_buffer,
                k_nope_a_buffer,
                k_rope_a_buffer,
                v_a_buffer,
                out_a_buffer,
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
                cuda_stream,
            )?;
            let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
            assert_eq!(capture.kernel_node_count, 1);
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_a_bytes, out_a_buffer)?;
            let out_a = bytes_to_bf16_f32_vec(&out_a_bytes);
            for (actual, expected) in out_a.iter().zip(expected_a.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }

            library.cuda_graph_update_mla_rope_attention_bf16_node(
                capture.graph,
                capture.graph_exec,
                0,
                q_nope_b_buffer,
                q_rope_b_buffer,
                k_nope_b_buffer,
                k_rope_b_buffer,
                v_b_buffer,
                out_b_buffer,
                rows,
                heads,
                nope_dim,
                rope_dim,
                v_dim,
                scale,
            )?;
            library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.copy_d2h(&mut out_b_bytes, out_b_buffer)?;
            let out_b = bytes_to_bf16_f32_vec(&out_b_bytes);
            for (actual, expected) in out_b.iter().zip(expected_b.iter()) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
            assert_ne!(out_a, out_b);

            library.cuda_graph_exec_destroy(capture.graph_exec)?;
            library.cuda_graph_destroy(capture.graph)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }

        library.free_device_buffer(&mut out_b_buffer)?;
        library.free_device_buffer(&mut out_a_buffer)?;
        library.free_device_buffer(&mut v_b_buffer)?;
        library.free_device_buffer(&mut k_rope_b_buffer)?;
        library.free_device_buffer(&mut k_nope_b_buffer)?;
        library.free_device_buffer(&mut q_rope_b_buffer)?;
        library.free_device_buffer(&mut q_nope_b_buffer)?;
        library.free_device_buffer(&mut v_a_buffer)?;
        library.free_device_buffer(&mut k_rope_a_buffer)?;
        library.free_device_buffer(&mut k_nope_a_buffer)?;
        library.free_device_buffer(&mut q_rope_a_buffer)?;
        library.free_device_buffer(&mut q_nope_a_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_rmsnorm_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let x = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
        let weight = [1.0_f32, 0.5, 1.5];
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<f32>()];
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut weight_buffer = library.alloc_device_buffer(std::mem::size_of_val(&weight))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, f32_bytes(&x))?;
        library.copy_h2d(weight_buffer, f32_bytes(&weight))?;
        match library.cuda_rmsnorm_f32(x_buffer, weight_buffer, out_buffer, 2, 3, 1.0e-5) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                for row in 0..2 {
                    let row_values = &x[row * 3..row * 3 + 3];
                    let mean_square =
                        row_values.iter().map(|value| value * value).sum::<f32>() / 3.0;
                    let inv = (mean_square + 1.0e-5).sqrt().recip();
                    for col in 0..3 {
                        let expected = row_values[col] * inv * weight[col];
                        assert!((out[row * 3 + col] - expected).abs() < 1.0e-5);
                    }
                }
                unsafe {
                    library.cuda_rmsnorm_f32_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let async_out = bytes_to_f32_vec(&out_bytes);
                assert_eq!(async_out.len(), out.len());
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_rmsnorm_f32_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_rmsnorm_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let x_f32 = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
        let weight_f32 = [1.0_f32, 0.5, 1.5];
        let x = bf16_values(&x_f32);
        let weight = bf16_values(&weight_f32);
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<u16>()];
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, u16_bytes(&x))?;
        library.copy_h2d(weight_buffer, u16_bytes(&weight))?;
        match library.cuda_rmsnorm_bf16(x_buffer, weight_buffer, out_buffer, 2, 3, 1.0e-5) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for row in 0..2 {
                    let row_values = &x[row * 3..row * 3 + 3];
                    let mean_square = row_values
                        .iter()
                        .map(|value| {
                            let value = bf16_to_f32(*value);
                            value * value
                        })
                        .sum::<f32>()
                        / 3.0;
                    let inv = (mean_square + 1.0e-5).sqrt().recip();
                    for col in 0..3 {
                        let expected = bf16_to_f32(f32_to_bf16(
                            bf16_to_f32(row_values[col]) * inv * bf16_to_f32(weight[col]),
                        ));
                        assert!((out[row * 3 + col] - expected).abs() < 1.0e-5);
                    }
                }
                unsafe {
                    library.cuda_rmsnorm_bf16_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let async_out = bytes_to_bf16_f32_vec(&out_bytes);
                assert_eq!(async_out.len(), out.len());
                unsafe {
                    library.cuda_ds4_rmsnorm_bf16_rne_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let rne_out = bytes_to_bf16_f32_vec(&out_bytes);
                for row in 0..2 {
                    let row_values = &x[row * 3..row * 3 + 3];
                    let mean_square = row_values
                        .iter()
                        .map(|value| {
                            let value = bf16_to_f32(*value);
                            value * value
                        })
                        .sum::<f32>()
                        / 3.0;
                    let inv = (mean_square + 1.0e-5).sqrt().recip();
                    for col in 0..3 {
                        let expected = bf16_to_f32(f32_to_bf16_rne(
                            bf16_to_f32(row_values[col]) * inv * bf16_to_f32(weight[col]),
                        ));
                        assert!((rne_out[row * 3 + col] - expected).abs() < 1.0e-5);
                    }
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_rmsnorm_bf16_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
                let rne_err = unsafe {
                    library.cuda_ds4_rmsnorm_bf16_rne_async(
                        x_buffer,
                        weight_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(rne_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_layernorm_affine_f32_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let x = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
        let weight_f32 = [1.0_f32, 0.5, 1.5];
        let bias_f32 = [0.25_f32, -0.5, 0.125];
        let weight = bf16_values(&weight_f32);
        let bias = bf16_values(&bias_f32);
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<f32>()];
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight.as_slice()))?;
        let mut bias_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, f32_bytes(&x))?;
        library.copy_h2d(weight_buffer, u16_bytes(&weight))?;
        library.copy_h2d(bias_buffer, u16_bytes(&bias))?;
        match library.cuda_layernorm_affine_f32_bf16(
            x_buffer,
            weight_buffer,
            bias_buffer,
            out_buffer,
            2,
            3,
            1.0e-5,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                for row in 0..2 {
                    let row_values = &x[row * 3..row * 3 + 3];
                    let mean = row_values.iter().sum::<f32>() / 3.0;
                    let variance = row_values
                        .iter()
                        .map(|value| {
                            let centered = value - mean;
                            centered * centered
                        })
                        .sum::<f32>()
                        / 3.0;
                    let inv = (variance + 1.0e-5).sqrt().recip();
                    for col in 0..3 {
                        let expected = (row_values[col] - mean) * inv * bf16_to_f32(weight[col])
                            + bf16_to_f32(bias[col]);
                        assert!((out[row * 3 + col] - expected).abs() < 1.0e-5);
                    }
                }
                unsafe {
                    library.cuda_layernorm_affine_f32_bf16_async(
                        x_buffer,
                        weight_buffer,
                        bias_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let async_out = bytes_to_f32_vec(&out_bytes);
                assert_eq!(async_out.len(), out.len());
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_layernorm_affine_f32_bf16_async(
                        x_buffer,
                        weight_buffer,
                        bias_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut bias_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_layernorm_affine_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let x_f32 = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 3.0];
        let weight_f32 = [1.0_f32, 0.5, 1.5];
        let bias_f32 = [0.25_f32, -0.5, 0.125];
        let x = bf16_values(&x_f32);
        let weight = bf16_values(&weight_f32);
        let bias = bf16_values(&bias_f32);
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<u16>()];
        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight.as_slice()))?;
        let mut bias_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, u16_bytes(&x))?;
        library.copy_h2d(weight_buffer, u16_bytes(&weight))?;
        library.copy_h2d(bias_buffer, u16_bytes(&bias))?;
        match library.cuda_layernorm_affine_bf16(
            x_buffer,
            weight_buffer,
            bias_buffer,
            out_buffer,
            2,
            3,
            1.0e-5,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for row in 0..2 {
                    let row_values = &x[row * 3..row * 3 + 3];
                    let mean = row_values
                        .iter()
                        .map(|value| bf16_to_f32(*value))
                        .sum::<f32>()
                        / 3.0;
                    let variance = row_values
                        .iter()
                        .map(|value| {
                            let centered = bf16_to_f32(*value) - mean;
                            centered * centered
                        })
                        .sum::<f32>()
                        / 3.0;
                    let inv = (variance + 1.0e-5).sqrt().recip();
                    for col in 0..3 {
                        let expected = bf16_to_f32(f32_to_bf16(
                            (bf16_to_f32(row_values[col]) - mean) * inv * bf16_to_f32(weight[col])
                                + bf16_to_f32(bias[col]),
                        ));
                        assert!((out[row * 3 + col] - expected).abs() < 1.0e-5);
                    }
                }
                unsafe {
                    library.cuda_layernorm_affine_bf16_async(
                        x_buffer,
                        weight_buffer,
                        bias_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let async_out = bytes_to_bf16_f32_vec(&out_bytes);
                assert_eq!(async_out.len(), out.len());
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_layernorm_affine_bf16_async(
                        x_buffer,
                        weight_buffer,
                        bias_buffer,
                        out_buffer,
                        2,
                        3,
                        1.0e-5,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut bias_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_nibble_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let codes = [0_u8, 1, 2, 15, 7, 8, 3];
        let packed_len = codes.len().div_ceil(2);
        let mut packed = vec![0_u8; packed_len];
        let mut unpacked = vec![0_u8; codes.len()];
        let mut codes_buffer = library.alloc_device_buffer(codes.len())?;
        let mut packed_buffer = library.alloc_device_buffer(packed.len())?;
        let mut unpacked_buffer = library.alloc_device_buffer(unpacked.len())?;

        library.copy_h2d(codes_buffer, &codes)?;
        match library.cuda_pack_nibbles(codes_buffer, packed_buffer, codes.len()) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut packed, packed_buffer)?;
                assert_eq!(packed, vec![0x10, 0xf2, 0x87, 0x03]);
                library.cuda_unpack_nibbles(packed_buffer, unpacked_buffer, codes.len())?;
                library.copy_d2h(&mut unpacked, unpacked_buffer)?;
                assert_eq!(unpacked, codes);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
            }
        }

        library.free_device_buffer(&mut unpacked_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut codes_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_silu_gated_mlp_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let hidden = 2;
        let intermediate = 3;
        let x = [0.25_f32, -0.75];
        let gate_weight = [0.1_f32, -0.2, 0.3, 0.4, -0.5, 0.6];
        let up_weight = [-0.2_f32, 0.5, 0.7, -0.1, 0.2, 0.3];
        let down_weight = [0.4_f32, -0.6, 0.8, -0.3, 0.9, 0.1];
        let mut out_bytes = vec![0_u8; hidden * std::mem::size_of::<f32>()];

        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut gate_buffer = library.alloc_device_buffer(std::mem::size_of_val(&gate_weight))?;
        let mut up_buffer = library.alloc_device_buffer(std::mem::size_of_val(&up_weight))?;
        let mut down_buffer = library.alloc_device_buffer(std::mem::size_of_val(&down_weight))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, f32_bytes(&x))?;
        library.copy_h2d(gate_buffer, f32_bytes(&gate_weight))?;
        library.copy_h2d(up_buffer, f32_bytes(&up_weight))?;
        library.copy_h2d(down_buffer, f32_bytes(&down_weight))?;

        match library.cuda_silu_gated_mlp_f32(
            x_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            out_buffer,
            hidden as i32,
            intermediate as i32,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                for out_col in 0..hidden {
                    let mut expected = 0.0_f32;
                    for mid in 0..intermediate {
                        let mut gate = 0.0_f32;
                        let mut up = 0.0_f32;
                        for col in 0..hidden {
                            gate += x[col] * gate_weight[mid * hidden + col];
                            up += x[col] * up_weight[mid * hidden + col];
                        }
                        let silu = gate / (1.0 + (-gate).exp());
                        expected += silu * up * down_weight[out_col * intermediate + mid];
                    }
                    assert!((out[out_col] - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut down_buffer)?;
        library.free_device_buffer(&mut up_buffer)?;
        library.free_device_buffer(&mut gate_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_silu_gated_mlp_rows_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden = 3;
        let intermediate = 2;
        let x = [0.25_f32, -0.5, 0.75, -1.0, 0.5, 0.125];
        let gate_weight = [0.1_f32, 0.2, -0.1, -0.4, 0.5, 0.2];
        let up_weight = [-0.2_f32, 0.1, 0.3, 0.5, -0.3, 0.2];
        let down_weight = [0.2_f32, -0.1, 0.4, 0.1, -0.3, 0.5];
        let expected = silu_gated_mlp_rows_expected(
            &x,
            &gate_weight,
            &up_weight,
            &down_weight,
            rows,
            hidden,
            intermediate,
        );
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<f32>()];

        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(&x))?;
        let mut gate_buffer = library.alloc_device_buffer(std::mem::size_of_val(&gate_weight))?;
        let mut up_buffer = library.alloc_device_buffer(std::mem::size_of_val(&up_weight))?;
        let mut down_buffer = library.alloc_device_buffer(std::mem::size_of_val(&down_weight))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, f32_bytes(&x))?;
        library.copy_h2d(gate_buffer, f32_bytes(&gate_weight))?;
        library.copy_h2d(up_buffer, f32_bytes(&up_weight))?;
        library.copy_h2d(down_buffer, f32_bytes(&down_weight))?;

        match library.cuda_silu_gated_mlp_rows_f32(
            x_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            out_buffer,
            rows,
            hidden,
            intermediate,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_silu_gated_mlp_rows_f32_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_silu_gated_mlp_rows_f32_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut down_buffer)?;
        library.free_device_buffer(&mut up_buffer)?;
        library.free_device_buffer(&mut gate_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_silu_gated_mlp_rows_bf16_down_stride_kernel_ffi_binding_reports_or_executes(
    ) -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden = 3;
        let intermediate = 2;
        let down_stride = 4;
        let x = bf16_values(&[0.25_f32, -0.5, 0.75, -1.0, 0.5, 0.125]);
        let gate_weight = bf16_values(&[0.1_f32, 0.2, -0.1, -0.4, 0.5, 0.2]);
        let up_weight = bf16_values(&[-0.2_f32, 0.1, 0.3, 0.5, -0.3, 0.2]);
        let down_weight = bf16_values(&[
            0.2_f32, -0.1, 99.0, -99.0, 0.4, 0.1, 88.0, -88.0, -0.3, 0.5, 77.0, -77.0,
        ]);
        let expected = silu_gated_mlp_rows_bf16_down_stride_expected(
            &x,
            &gate_weight,
            &up_weight,
            &down_weight,
            rows,
            hidden,
            intermediate,
            down_stride,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; x.len() * std::mem::size_of::<u16>()];

        let mut x_buffer = library.alloc_device_buffer(std::mem::size_of_val(x.as_slice()))?;
        let mut gate_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(gate_weight.as_slice()))?;
        let mut up_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(up_weight.as_slice()))?;
        let mut down_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(down_weight.as_slice()))?;
        let mut activation_buffer =
            library.alloc_device_buffer(rows * intermediate * std::mem::size_of::<f32>())?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(x_buffer, u16_bytes(&x))?;
        library.copy_h2d(gate_buffer, u16_bytes(&gate_weight))?;
        library.copy_h2d(up_buffer, u16_bytes(&up_weight))?;
        library.copy_h2d(down_buffer, u16_bytes(&down_weight))?;

        match library.cuda_silu_gated_mlp_rows_bf16_down_stride(
            x_buffer,
            gate_buffer,
            up_buffer,
            down_buffer,
            out_buffer,
            rows,
            hidden,
            intermediate,
            down_stride,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                library.cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
                    x_buffer,
                    gate_buffer,
                    up_buffer,
                    down_buffer,
                    activation_buffer,
                    out_buffer,
                    rows,
                    hidden,
                    intermediate,
                    down_stride,
                )?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        activation_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_silu_gated_mlp_rows_bf16_down_stride_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
                let staged_err = library
                    .cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        activation_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(staged_err);
                let staged_async_err = unsafe {
                    library.cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
                        x_buffer,
                        gate_buffer,
                        up_buffer,
                        down_buffer,
                        activation_buffer,
                        out_buffer,
                        rows,
                        hidden,
                        intermediate,
                        down_stride,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(staged_async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut activation_buffer)?;
        library.free_device_buffer(&mut down_buffer)?;
        library.free_device_buffer(&mut up_buffer)?;
        library.free_device_buffer(&mut gate_buffer)?;
        library.free_device_buffer(&mut x_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_nvfp4_route_bf16_staged_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let hidden_dim = 3;
        let intermediate = 2;
        let output_dim = 2;
        let hidden_f32 = [1.0_f32, -0.5, 0.25];
        let hidden_bf16 = bf16_values(&hidden_f32);
        let hidden_for_expected = hidden_bf16
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let gate_weight = [0x9a_u8, 0x08, 0xab, 0x09];
        let up_weight = [0xc9_u8, 0x08, 0xba, 0x09];
        let down_weight = [0xa9_u8, 0xcb];
        let gate_scale = [0x38_u8, 0x38];
        let up_scale = [0x38_u8, 0x38];
        let down_scale = [0x38_u8, 0x38];
        let route_weight = 0.75_f32;
        let grouped_row_indices = [0_u32];
        let grouped_route_weights = [route_weight];
        let mut route_metadata = [Ds41rtNvfp4RouteBatchedMetadata::default()];
        let expected = nvfp4_route_expected(
            &hidden_for_expected,
            &gate_weight,
            &gate_scale,
            &up_weight,
            &up_scale,
            &down_weight,
            &down_scale,
            hidden_dim,
            intermediate,
            output_dim,
            1.0,
            1.0,
            1.0,
            route_weight,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; output_dim * std::mem::size_of::<u16>()];

        let mut hidden_buffer = library.alloc_device_buffer(std::mem::size_of_val(&hidden_bf16))?;
        let mut gate_weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gate_weight))?;
        let mut gate_scale_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gate_scale))?;
        let mut up_weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&up_weight))?;
        let mut up_scale_buffer = library.alloc_device_buffer(std::mem::size_of_val(&up_scale))?;
        let mut down_weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&down_weight))?;
        let mut down_scale_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&down_scale))?;
        let mut row_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&grouped_row_indices))?;
        let mut route_weights_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&grouped_route_weights))?;
        let mut route_metadata_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&route_metadata))?;
        let mut activation_buffer =
            library.alloc_device_buffer(intermediate * std::mem::size_of::<f32>())?;
        let mut accumulator_buffer =
            library.alloc_device_buffer(output_dim * std::mem::size_of::<f32>())?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(hidden_buffer, u16_bytes(&hidden_bf16))?;
        library.copy_h2d(row_indices_buffer, u32_bytes(&grouped_row_indices))?;
        library.copy_h2d(route_weights_buffer, f32_bytes(&grouped_route_weights))?;
        library.copy_h2d(gate_weight_buffer, &gate_weight)?;
        library.copy_h2d(gate_scale_buffer, &gate_scale)?;
        library.copy_h2d(up_weight_buffer, &up_weight)?;
        library.copy_h2d(up_scale_buffer, &up_scale)?;
        library.copy_h2d(down_weight_buffer, &down_weight)?;
        library.copy_h2d(down_scale_buffer, &down_scale)?;
        route_metadata[0] = Ds41rtNvfp4RouteBatchedMetadata {
            gate_weight: gate_weight_buffer.ptr as usize,
            gate_scale: gate_scale_buffer.ptr as usize,
            up_weight: up_weight_buffer.ptr as usize,
            up_scale: up_scale_buffer.ptr as usize,
            down_weight: down_weight_buffer.ptr as usize,
            down_scale: down_scale_buffer.ptr as usize,
            intermediate,
            down_weight_row_stride_bytes: 1,
            down_scale_row_stride_bytes: 1,
            gate_scale_2: 1.0,
            up_scale_2: 1.0,
            down_scale_2: 1.0,
        };
        library.copy_h2d(
            route_metadata_buffer,
            nvfp4_route_metadata_bytes(&route_metadata),
        )?;

        let first_result = library
            .cuda_zero_f32(accumulator_buffer, output_dim)
            .and_then(|_| {
                library.cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
                    hidden_buffer,
                    row_indices_buffer,
                    route_weights_buffer,
                    gate_weight_buffer,
                    gate_scale_buffer,
                    up_weight_buffer,
                    up_scale_buffer,
                    down_weight_buffer,
                    down_scale_buffer,
                    activation_buffer,
                    accumulator_buffer,
                    1,
                    grouped_row_indices.len(),
                    hidden_dim,
                    hidden_dim,
                    intermediate,
                    output_dim,
                    1,
                    1,
                    1.0,
                    1.0,
                    1.0,
                )
            });
        match first_result {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.cuda_f32_to_bf16(accumulator_buffer, out_buffer, output_dim)?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                library.cuda_zero_f32(accumulator_buffer, output_dim)?;
                unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            gate_weight_buffer,
                            gate_scale_buffer,
                            up_weight_buffer,
                            up_scale_buffer,
                            down_weight_buffer,
                            down_scale_buffer,
                            activation_buffer,
                            accumulator_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            1,
                            1,
                            1.0,
                            1.0,
                            1.0,
                            std::ptr::null_mut(),
                        )?;
                }
                library.cuda_f32_to_bf16(accumulator_buffer, out_buffer, output_dim)?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                library.cuda_zero_f32(accumulator_buffer, output_dim)?;
                library.cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
                    hidden_buffer,
                    row_indices_buffer,
                    route_weights_buffer,
                    route_metadata_buffer,
                    activation_buffer,
                    accumulator_buffer,
                    1,
                    grouped_row_indices.len(),
                    hidden_dim,
                    hidden_dim,
                    intermediate,
                    output_dim,
                )?;
                library.cuda_f32_to_bf16(accumulator_buffer, out_buffer, output_dim)?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                library.cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
                    hidden_buffer,
                    row_indices_buffer,
                    route_weights_buffer,
                    route_metadata_buffer,
                    activation_buffer,
                    out_buffer,
                    1,
                    grouped_row_indices.len(),
                    hidden_dim,
                    hidden_dim,
                    intermediate,
                    output_dim,
                )?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                library.cuda_zero_f32(accumulator_buffer, output_dim)?;
                unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            route_metadata_buffer,
                            activation_buffer,
                            accumulator_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            std::ptr::null_mut(),
                        )?;
                }
                library.cuda_f32_to_bf16(accumulator_buffer, out_buffer, output_dim)?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            route_metadata_buffer,
                            activation_buffer,
                            out_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            std::ptr::null_mut(),
                        )?;
                }
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let staged_err = library
                    .cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
                        hidden_buffer,
                        row_indices_buffer,
                        route_weights_buffer,
                        gate_weight_buffer,
                        gate_scale_buffer,
                        up_weight_buffer,
                        up_scale_buffer,
                        down_weight_buffer,
                        down_scale_buffer,
                        activation_buffer,
                        accumulator_buffer,
                        1,
                        grouped_row_indices.len(),
                        hidden_dim,
                        hidden_dim,
                        intermediate,
                        output_dim,
                        1,
                        1,
                        1.0,
                        1.0,
                        1.0,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(staged_err);
                let staged_async_err = unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            gate_weight_buffer,
                            gate_scale_buffer,
                            up_weight_buffer,
                            up_scale_buffer,
                            down_weight_buffer,
                            down_scale_buffer,
                            activation_buffer,
                            accumulator_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            1,
                            1,
                            1.0,
                            1.0,
                            1.0,
                            std::ptr::null_mut(),
                        )
                }
                .unwrap_err();
                assert_cuda_unavailable(staged_async_err);
                let batched_staged_err = library
                    .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
                        hidden_buffer,
                        row_indices_buffer,
                        route_weights_buffer,
                        route_metadata_buffer,
                        activation_buffer,
                        accumulator_buffer,
                        1,
                        grouped_row_indices.len(),
                        hidden_dim,
                        hidden_dim,
                        intermediate,
                        output_dim,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(batched_staged_err);
                let batched_staged_async_err = unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            route_metadata_buffer,
                            activation_buffer,
                            accumulator_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            std::ptr::null_mut(),
                        )
                }
                .unwrap_err();
                assert_cuda_unavailable(batched_staged_async_err);
                let batched_staged_single_row_err = library
                    .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
                        hidden_buffer,
                        row_indices_buffer,
                        route_weights_buffer,
                        route_metadata_buffer,
                        activation_buffer,
                        out_buffer,
                        1,
                        grouped_row_indices.len(),
                        hidden_dim,
                        hidden_dim,
                        intermediate,
                        output_dim,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(batched_staged_single_row_err);
                let batched_staged_single_row_async_err = unsafe {
                    library
                        .cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
                            hidden_buffer,
                            row_indices_buffer,
                            route_weights_buffer,
                            route_metadata_buffer,
                            activation_buffer,
                            out_buffer,
                            1,
                            grouped_row_indices.len(),
                            hidden_dim,
                            hidden_dim,
                            intermediate,
                            output_dim,
                            std::ptr::null_mut(),
                        )
                }
                .unwrap_err();
                assert_cuda_unavailable(batched_staged_single_row_async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut accumulator_buffer)?;
        library.free_device_buffer(&mut activation_buffer)?;
        library.free_device_buffer(&mut route_metadata_buffer)?;
        library.free_device_buffer(&mut route_weights_buffer)?;
        library.free_device_buffer(&mut row_indices_buffer)?;
        library.free_device_buffer(&mut down_scale_buffer)?;
        library.free_device_buffer(&mut down_weight_buffer)?;
        library.free_device_buffer(&mut up_scale_buffer)?;
        library.free_device_buffer(&mut up_weight_buffer)?;
        library.free_device_buffer(&mut gate_scale_buffer)?;
        library.free_device_buffer(&mut gate_weight_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_residual_add_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let residual = [0.25_f32, -0.5, 1.5, 2.0, -3.0, 4.5, 0.0, -0.125];
        let delta = [-0.5_f32, 0.25, 0.5, -1.0, 3.5, -2.0, 0.125, 0.25];
        let expected = residual
            .iter()
            .zip(delta.iter())
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; residual.len() * std::mem::size_of::<f32>()];
        let mut residual_buffer = library.alloc_device_buffer(std::mem::size_of_val(&residual))?;
        let mut delta_buffer = library.alloc_device_buffer(std::mem::size_of_val(&delta))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(residual_buffer, f32_bytes(&residual))?;
        library.copy_h2d(delta_buffer, f32_bytes(&delta))?;
        match library.cuda_residual_add_f32(
            residual_buffer,
            delta_buffer,
            out_buffer,
            residual.len(),
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                assert_eq!(out.len(), expected.len());
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
                unsafe {
                    library.cuda_residual_add_f32_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )?;
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_residual_add_f32_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut delta_buffer)?;
        library.free_device_buffer(&mut residual_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_residual_add_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let residual_f32 = [0.25_f32, -0.5, 1.5, 2.0, -3.0, 4.5, 0.0, -0.125];
        let delta_f32 = [-0.5_f32, 0.25, 0.5, -1.0, 3.5, -2.0, 0.125, 0.25];
        let residual = bf16_values(&residual_f32);
        let delta = bf16_values(&delta_f32);
        let expected = residual
            .iter()
            .zip(delta.iter())
            .map(|(left, right)| bf16_to_f32(f32_to_bf16(bf16_to_f32(*left) + bf16_to_f32(*right))))
            .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; residual.len() * std::mem::size_of::<u16>()];
        let mut residual_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual.as_slice()))?;
        let mut delta_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(delta.as_slice()))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(residual_buffer, u16_bytes(&residual))?;
        library.copy_h2d(delta_buffer, u16_bytes(&delta))?;
        match library.cuda_residual_add_bf16(
            residual_buffer,
            delta_buffer,
            out_buffer,
            residual.len(),
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                assert_eq!(out.len(), expected.len());
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
                unsafe {
                    library.cuda_residual_add_bf16_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )?;
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_residual_add_bf16_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut delta_buffer)?;
        library.free_device_buffer(&mut residual_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_residual_add_f32_delta_bf16_kernels_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let residual_f32 = [0.25_f32, -0.5, 1.5, 2.0, -3.0, 4.5, 0.0, -0.125];
        let shared_f32 = [-0.25_f32, 0.75, 0.125, -1.5, 2.0, -0.5, 0.25, 0.5];
        let delta_f32 = [-0.51_f32, 0.26, 0.501, -1.001, 3.51, -2.01, 0.126, 0.251];
        let residual = bf16_values(&residual_f32);
        let shared_delta = bf16_values(&shared_f32);
        let expected_delta = residual_add_f32_delta_bf16_expected(&residual, &delta_f32);
        let expected_shared =
            residual_add_shared_f32_delta_bf16_expected(&residual, &shared_delta, &delta_f32);
        let mut out_bytes = vec![0_u8; residual.len() * std::mem::size_of::<u16>()];

        let mut residual_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual.as_slice()))?;
        let mut shared_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(shared_delta.as_slice()))?;
        let mut delta_buffer = library.alloc_device_buffer(std::mem::size_of_val(&delta_f32))?;
        let mut out_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(residual_buffer, u16_bytes(&residual))?;
        library.copy_h2d(shared_buffer, u16_bytes(&shared_delta))?;
        library.copy_h2d(delta_buffer, f32_bytes(&delta_f32))?;
        match library.cuda_residual_add_f32_delta_bf16(
            residual_buffer,
            delta_buffer,
            out_buffer,
            residual.len(),
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                assert_eq!(bytes_to_bf16_f32_vec(&out_bytes), expected_delta);
                library.cuda_residual_add_shared_f32_delta_bf16(
                    residual_buffer,
                    shared_buffer,
                    delta_buffer,
                    out_buffer,
                    residual.len(),
                )?;
                library.copy_d2h(&mut out_bytes, out_buffer)?;
                assert_eq!(bytes_to_bf16_f32_vec(&out_bytes), expected_shared);
                unsafe {
                    library.cuda_residual_add_f32_delta_bf16_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_residual_add_shared_f32_delta_bf16_async(
                        residual_buffer,
                        shared_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )?;
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let shared_err = library
                    .cuda_residual_add_shared_f32_delta_bf16(
                        residual_buffer,
                        shared_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                    )
                    .unwrap_err();
                assert_cuda_unavailable(shared_err);
                let async_err = unsafe {
                    library.cuda_residual_add_f32_delta_bf16_async(
                        residual_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
                let shared_async_err = unsafe {
                    library.cuda_residual_add_shared_f32_delta_bf16_async(
                        residual_buffer,
                        shared_buffer,
                        delta_buffer,
                        out_buffer,
                        residual.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(shared_async_err);
            }
        }

        library.free_device_buffer(&mut out_buffer)?;
        library.free_device_buffer(&mut delta_buffer)?;
        library.free_device_buffer(&mut shared_buffer)?;
        library.free_device_buffer(&mut residual_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_accumulate_bf16_to_f32_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let first = bf16_values(&[0.25_f32, -0.5, 1.5, 2.0]);
        let second = bf16_values(&[-0.5_f32, 0.25, 0.5, -1.0]);
        let expected = first
            .iter()
            .zip(&second)
            .map(|(left, right)| bf16_to_f32(*left) + bf16_to_f32(*right))
            .collect::<Vec<_>>();
        let mut first_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(first.as_slice()))?;
        let mut second_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(second.as_slice()))?;
        let mut accumulator_buffer =
            library.alloc_device_buffer(expected.len() * std::mem::size_of::<f32>())?;
        library.copy_h2d(first_buffer, u16_bytes(&first))?;
        library.copy_h2d(second_buffer, u16_bytes(&second))?;

        match library.cuda_zero_f32(accumulator_buffer, expected.len()) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.cuda_accumulate_bf16_to_f32(
                    first_buffer,
                    accumulator_buffer,
                    expected.len(),
                )?;
                unsafe {
                    library.cuda_accumulate_bf16_to_f32_async(
                        second_buffer,
                        accumulator_buffer,
                        expected.len(),
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_stream_synchronize(std::ptr::null_mut())?;
                }
                let mut bytes = vec![0_u8; std::mem::size_of_val(expected.as_slice())];
                library.copy_d2h(&mut bytes, accumulator_buffer)?;
                let actual = bytes
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
            Err(error) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(error);
                let error = library
                    .cuda_accumulate_bf16_to_f32(first_buffer, accumulator_buffer, expected.len())
                    .unwrap_err();
                assert_cuda_unavailable(error);
                let error = unsafe {
                    library.cuda_accumulate_bf16_to_f32_async(
                        second_buffer,
                        accumulator_buffer,
                        expected.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(error);
            }
        }

        library.free_device_buffer(&mut accumulator_buffer)?;
        library.free_device_buffer(&mut second_buffer)?;
        library.free_device_buffer(&mut first_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_summarize_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let values_f32 = [0.25_f32, -0.5, 0.0, 1.5];
        let values = bf16_values(&values_f32);
        let expected_checksum = values
            .iter()
            .map(|value| bf16_to_f32(*value) as f64)
            .sum::<f64>();
        let mut input_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(values.as_slice()))?;
        let mut summary_buffer =
            library.alloc_device_buffer(std::mem::size_of::<Ds41rtBf16Summary>())?;

        library.copy_h2d(input_buffer, u16_bytes(&values))?;
        match library.cuda_summarize_bf16(input_buffer, values.len()) {
            Ok(summary) => {
                assert_eq!(info.cuda_available, 1);
                assert_eq!(summary.values, values.len() as u64);
                assert_eq!(summary.finite_values, values.len() as u64);
                assert_eq!(summary.nonzero_values, 3);
                assert!((summary.checksum - expected_checksum).abs() < 1.0e-6);

                unsafe {
                    library.cuda_summarize_bf16_async(
                        input_buffer,
                        values.len(),
                        summary_buffer,
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_stream_synchronize(std::ptr::null_mut())?;
                }
                let mut summary_bytes = vec![0_u8; std::mem::size_of::<Ds41rtBf16Summary>()];
                library.copy_d2h(&mut summary_bytes, summary_buffer)?;
                let async_summary = bytes_to_bf16_summary(&summary_bytes);
                assert_eq!(async_summary.values, values.len() as u64);
                assert_eq!(async_summary.finite_values, values.len() as u64);
                assert_eq!(async_summary.nonzero_values, 3);
                assert!((async_summary.checksum - expected_checksum).abs() < 1.0e-6);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_summarize_bf16_async(
                        input_buffer,
                        values.len(),
                        summary_buffer,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut summary_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_f32_to_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let values = [0.25_f32, -0.5, 1.5, 2.0, -3.0, 4.5, 0.0, -0.125];
        let expected = values
            .iter()
            .map(|value| bf16_to_f32(f32_to_bf16(*value)))
            .collect::<Vec<_>>();
        let mut out_bytes = vec![0_u8; values.len() * std::mem::size_of::<u16>()];
        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(&values))?;
        let mut dst_buffer = library.alloc_device_buffer(out_bytes.len())?;

        library.copy_h2d(src_buffer, f32_bytes(&values))?;
        match library.cuda_f32_to_bf16(src_buffer, dst_buffer, values.len()) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, dst_buffer)?;
                let out = bytes_to_bf16_f32_vec(&out_bytes);
                assert_eq!(out.len(), expected.len());
                for (actual, expected) in out.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
                unsafe {
                    library.cuda_f32_to_bf16_async(
                        src_buffer,
                        dst_buffer,
                        values.len(),
                        std::ptr::null_mut(),
                    )?;
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_f32_to_bf16_async(
                        src_buffer,
                        dst_buffer,
                        values.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut dst_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_zero_f32_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let values = [1.0_f32, -2.0, 3.5, -4.25, 5.0, -6.0, 7.0, -8.0];
        let mut out_bytes = vec![0_u8; std::mem::size_of_val(&values)];
        let mut buffer = library.alloc_device_buffer(std::mem::size_of_val(&values))?;

        library.copy_h2d(buffer, f32_bytes(&values))?;
        match library.cuda_zero_f32(buffer, values.len()) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                assert_eq!(out, vec![0.0_f32; values.len()]);
                library.copy_h2d(buffer, f32_bytes(&values))?;
                unsafe {
                    library.cuda_zero_f32_async(buffer, values.len(), std::ptr::null_mut())?;
                }
                library.copy_d2h(&mut out_bytes, buffer)?;
                let out = bytes_to_f32_vec(&out_bytes);
                assert_eq!(out, vec![0.0_f32; values.len()]);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_zero_f32_async(buffer, values.len(), std::ptr::null_mut())
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_zero_bytes_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let values = [1_u8, 2, 3, 4, 5, 6, 7];
        let mut out_bytes = vec![0_u8; values.len()];
        let mut buffer = library.alloc_device_buffer(values.len())?;

        library.copy_h2d(buffer, &values)?;
        match library.cuda_zero_bytes(buffer, values.len()) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut out_bytes, buffer)?;
                assert_eq!(out_bytes, vec![0_u8; values.len()]);
                library.copy_h2d(buffer, &values)?;
                unsafe {
                    library.cuda_zero_bytes_async(buffer, values.len(), std::ptr::null_mut())?;
                }
                library.copy_d2h(&mut out_bytes, buffer)?;
                assert_eq!(out_bytes, vec![0_u8; values.len()]);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_zero_bytes_async(buffer, values.len(), std::ptr::null_mut())
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_row_gather_scatter_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let row_width = 3;
        let source_rows = 4;
        let src = [
            0.0_f32, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, -3.0, -3.1, -3.2,
        ];
        let gather_indices = [2_u32, 0, 3];
        let expected_gathered = [2.0_f32, 2.1, 2.2, 0.0, 0.1, 0.2, -3.0, -3.1, -3.2];
        let partials = [0.25_f32, 0.5, 0.75, 1.0, 1.25, 1.5, -0.25, -0.5, -0.75];
        let scatter_indices = [1_u32, 3, 1];
        let expected_scattered = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.25, 1.5,
        ];
        let mut gathered_bytes = vec![0_u8; expected_gathered.len() * std::mem::size_of::<f32>()];
        let mut scattered_bytes = vec![0_u8; source_rows * row_width * std::mem::size_of::<f32>()];

        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(&src))?;
        let mut gather_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gather_indices))?;
        let mut gathered_buffer = library.alloc_device_buffer(gathered_bytes.len())?;
        let mut partials_buffer = library.alloc_device_buffer(std::mem::size_of_val(&partials))?;
        let mut scatter_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&scatter_indices))?;
        let mut scattered_buffer = library.alloc_device_buffer(scattered_bytes.len())?;

        library.copy_h2d(src_buffer, f32_bytes(&src))?;
        library.copy_h2d(gather_indices_buffer, u32_bytes(&gather_indices))?;
        library.copy_h2d(partials_buffer, f32_bytes(&partials))?;
        library.copy_h2d(scatter_indices_buffer, u32_bytes(&scatter_indices))?;
        library.copy_h2d(scattered_buffer, &scattered_bytes)?;

        match library.cuda_gather_rows_f32(
            src_buffer,
            source_rows,
            gather_indices_buffer,
            gathered_buffer,
            gather_indices.len(),
            row_width,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut gathered_bytes, gathered_buffer)?;
                let gathered = bytes_to_f32_vec(&gathered_bytes);
                assert_eq!(gathered.len(), expected_gathered.len());
                for (actual, expected) in gathered.iter().zip(expected_gathered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                library.cuda_scatter_add_rows_f32(
                    partials_buffer,
                    scatter_indices_buffer,
                    scattered_buffer,
                    source_rows,
                    scatter_indices.len(),
                    row_width,
                )?;
                library.copy_d2h(&mut scattered_bytes, scattered_buffer)?;
                let scattered = bytes_to_f32_vec(&scattered_bytes);
                assert_eq!(scattered.len(), expected_scattered.len());
                for (actual, expected) in scattered.iter().zip(expected_scattered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_gather_rows_f32_async(
                        src_buffer,
                        source_rows,
                        gather_indices_buffer,
                        gathered_buffer,
                        gather_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut gathered_bytes, gathered_buffer)?;
                let gathered = bytes_to_f32_vec(&gathered_bytes);
                for (actual, expected) in gathered.iter().zip(expected_gathered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                scattered_bytes.fill(0);
                library.copy_h2d(scattered_buffer, &scattered_bytes)?;
                unsafe {
                    library.cuda_scatter_add_rows_f32_async(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut scattered_bytes, scattered_buffer)?;
                let scattered = bytes_to_f32_vec(&scattered_bytes);
                for (actual, expected) in scattered.iter().zip(expected_scattered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_gather_err = unsafe {
                    library.cuda_gather_rows_f32_async(
                        src_buffer,
                        source_rows,
                        gather_indices_buffer,
                        gathered_buffer,
                        gather_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_gather_err);
                let scatter_err = library
                    .cuda_scatter_add_rows_f32(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(scatter_err);
                let async_scatter_err = unsafe {
                    library.cuda_scatter_add_rows_f32_async(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_scatter_err);
            }
        }

        library.free_device_buffer(&mut scattered_buffer)?;
        library.free_device_buffer(&mut scatter_indices_buffer)?;
        library.free_device_buffer(&mut partials_buffer)?;
        library.free_device_buffer(&mut gathered_buffer)?;
        library.free_device_buffer(&mut gather_indices_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_row_gather_scatter_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let row_width = 3;
        let source_rows = 4;
        let src = bf16_values(&[
            0.0_f32, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, -3.0, -3.1, -3.2,
        ]);
        let gather_indices = [2_u32, 0, 3];
        let mut expected_gathered = Vec::with_capacity(gather_indices.len() * row_width);
        for row in gather_indices {
            let start = row as usize * row_width;
            expected_gathered.extend_from_slice(&src[start..start + row_width]);
        }
        let partials = bf16_values(&[0.25_f32, 0.5, 0.75, 1.0, 1.25, 1.5, -0.25, -0.5, -0.75]);
        let scatter_indices = [1_u32, 3, 1];
        let expected_scattered = [
            0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.25, 1.5,
        ];
        let mut gathered_bytes = vec![0_u8; expected_gathered.len() * std::mem::size_of::<u16>()];
        let mut scattered_bytes = vec![0_u8; source_rows * row_width * std::mem::size_of::<f32>()];

        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(src.as_slice()))?;
        let mut gather_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gather_indices))?;
        let mut gathered_buffer = library.alloc_device_buffer(gathered_bytes.len())?;
        let mut partials_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(partials.as_slice()))?;
        let mut scatter_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&scatter_indices))?;
        let mut scattered_buffer = library.alloc_device_buffer(scattered_bytes.len())?;

        library.copy_h2d(src_buffer, u16_bytes(&src))?;
        library.copy_h2d(gather_indices_buffer, u32_bytes(&gather_indices))?;
        library.copy_h2d(partials_buffer, u16_bytes(&partials))?;
        library.copy_h2d(scatter_indices_buffer, u32_bytes(&scatter_indices))?;
        library.copy_h2d(scattered_buffer, &scattered_bytes)?;

        match library.cuda_gather_rows_bf16(
            src_buffer,
            source_rows,
            gather_indices_buffer,
            gathered_buffer,
            gather_indices.len(),
            row_width,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut gathered_bytes, gathered_buffer)?;
                let gathered = gathered_bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(gathered, expected_gathered);

                library.cuda_scatter_add_rows_bf16_to_f32(
                    partials_buffer,
                    scatter_indices_buffer,
                    scattered_buffer,
                    source_rows,
                    scatter_indices.len(),
                    row_width,
                )?;
                library.copy_d2h(&mut scattered_bytes, scattered_buffer)?;
                let scattered = bytes_to_f32_vec(&scattered_bytes);
                for (actual, expected) in scattered.iter().zip(expected_scattered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_gather_rows_bf16_async(
                        src_buffer,
                        source_rows,
                        gather_indices_buffer,
                        gathered_buffer,
                        gather_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut gathered_bytes, gathered_buffer)?;
                let gathered = gathered_bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(gathered, expected_gathered);

                scattered_bytes.fill(0);
                library.copy_h2d(scattered_buffer, &scattered_bytes)?;
                unsafe {
                    library.cuda_scatter_add_rows_bf16_to_f32_async(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut scattered_bytes, scattered_buffer)?;
                let scattered = bytes_to_f32_vec(&scattered_bytes);
                for (actual, expected) in scattered.iter().zip(expected_scattered.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_gather_err = unsafe {
                    library.cuda_gather_rows_bf16_async(
                        src_buffer,
                        source_rows,
                        gather_indices_buffer,
                        gathered_buffer,
                        gather_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_gather_err);
                let scatter_err = library
                    .cuda_scatter_add_rows_bf16_to_f32(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(scatter_err);
                let async_scatter_err = unsafe {
                    library.cuda_scatter_add_rows_bf16_to_f32_async(
                        partials_buffer,
                        scatter_indices_buffer,
                        scattered_buffer,
                        source_rows,
                        scatter_indices.len(),
                        row_width,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_scatter_err);
            }
        }

        library.free_device_buffer(&mut scattered_buffer)?;
        library.free_device_buffer(&mut scatter_indices_buffer)?;
        library.free_device_buffer(&mut partials_buffer)?;
        library.free_device_buffer(&mut gathered_buffer)?;
        library.free_device_buffer(&mut gather_indices_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_row_scaled_fp8_gather_scatter_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let row_width = 8_usize;
        let source_rows = 3_usize;
        let src = [
            0.0_f32, 0.125, -0.25, 0.5, 1.0, -2.0, 4.0, -8.0, 0.03, -0.07, 0.2, -0.4, 0.8, -1.6,
            3.2, -6.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let gather_indices = [2_u32, 0];
        let scatter_indices = [1_u32, 1];
        let row_stride_bytes = row_width + std::mem::size_of::<f32>();
        let mut packed = vec![0_u8; gather_indices.len() * row_stride_bytes];
        let mut scattered = vec![0_u8; source_rows * row_width * std::mem::size_of::<f32>()];

        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(&src))?;
        let mut gather_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gather_indices))?;
        let mut packed_buffer = library.alloc_device_buffer(packed.len())?;
        let mut scatter_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&scatter_indices))?;
        let mut scattered_buffer = library.alloc_device_buffer(scattered.len())?;
        library.copy_h2d(src_buffer, f32_bytes(&src))?;
        library.copy_h2d(gather_indices_buffer, u32_bytes(&gather_indices))?;
        library.copy_h2d(scatter_indices_buffer, u32_bytes(&scatter_indices))?;
        library.copy_h2d(scattered_buffer, &scattered)?;

        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
            src_buffer,
            source_rows,
            gather_indices_buffer,
            packed_buffer,
            gather_indices.len(),
            row_width,
            row_stride_bytes,
        )?;
        library.copy_d2h(&mut packed, packed_buffer)?;
        for row in 0..gather_indices.len() {
            let scale_offset = row * row_stride_bytes + row_width;
            let scale = f32::from_ne_bytes(
                packed[scale_offset..scale_offset + std::mem::size_of::<f32>()]
                    .try_into()
                    .unwrap(),
            );
            assert!(scale.is_finite() && scale > 0.0);
        }

        library.cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
            packed_buffer,
            row_stride_bytes,
            scatter_indices_buffer,
            scattered_buffer,
            source_rows,
            scatter_indices.len(),
            row_width,
        )?;
        library.copy_d2h(&mut scattered, scattered_buffer)?;
        let actual = bytes_to_f32_vec(&scattered);
        let expected_row = [1.0_f32, 2.125, 2.75, 4.5, 6.0, 4.0, 11.0, 0.0];
        for row in 0..source_rows {
            for col in 0..row_width {
                let expected = if row == 1 { expected_row[col] } else { 0.0 };
                let tolerance = 0.08 * expected.abs().max(1.0);
                assert!(
                    (actual[row * row_width + col] - expected).abs() <= tolerance,
                    "row-scaled FP8 mismatch row={row} col={col} actual={} expected={expected} tolerance={tolerance}",
                    actual[row * row_width + col]
                );
            }
        }

        library.free_device_buffer(&mut scattered_buffer)?;
        library.free_device_buffer(&mut scatter_indices_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut gather_indices_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_row_scaled_fp8_fused_residual_matches_scatter_path() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let row_width = 8_usize;
        let routed = [0.031_f32, -0.127, 0.499, -1.003, 2.51, -4.02, 7.125, -9.75];
        let residual = bf16_values(&[0.25, -0.5, 1.5, 2.0, -3.0, 4.5, 0.0, -0.125]);
        let shared = bf16_values(&[-0.25, 0.75, 0.125, -1.5, 2.0, -0.5, 0.25, 0.5]);
        let row_index = [0_u32];
        let row_stride_bytes = row_width + std::mem::size_of::<f32>();
        let output_bytes = row_width * std::mem::size_of::<u16>();

        let mut routed_buffer = library.alloc_device_buffer(std::mem::size_of_val(&routed))?;
        let mut row_index_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&row_index))?;
        let mut packed_buffer = library.alloc_device_buffer(row_stride_bytes)?;
        let mut accumulator_buffer =
            library.alloc_device_buffer(row_width * std::mem::size_of::<f32>())?;
        let mut residual_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(residual.as_slice()))?;
        let mut shared_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(shared.as_slice()))?;
        let mut scatter_output_buffer = library.alloc_device_buffer(output_bytes)?;
        let mut fused_output_buffer = library.alloc_device_buffer(output_bytes)?;

        library.copy_h2d(routed_buffer, f32_bytes(&routed))?;
        library.copy_h2d(row_index_buffer, u32_bytes(&row_index))?;
        library.copy_h2d(residual_buffer, u16_bytes(&residual))?;
        library.copy_h2d(shared_buffer, u16_bytes(&shared))?;
        library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
            routed_buffer,
            1,
            row_index_buffer,
            packed_buffer,
            1,
            row_width,
            row_stride_bytes,
        )?;

        library.cuda_zero_f32(accumulator_buffer, row_width)?;
        library.cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
            packed_buffer,
            row_stride_bytes,
            row_index_buffer,
            accumulator_buffer,
            1,
            1,
            row_width,
        )?;
        library.cuda_residual_add_shared_f32_delta_bf16(
            residual_buffer,
            shared_buffer,
            accumulator_buffer,
            scatter_output_buffer,
            row_width,
        )?;
        unsafe {
            library.cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async(
                residual_buffer,
                shared_buffer,
                packed_buffer,
                fused_output_buffer,
                row_width,
                std::ptr::null_mut(),
            )?;
            library.cuda_stream_synchronize(std::ptr::null_mut())?;
        }

        let mut scatter_output = vec![0_u8; output_bytes];
        let mut fused_output = vec![0_u8; output_bytes];
        library.copy_d2h(&mut scatter_output, scatter_output_buffer)?;
        library.copy_d2h(&mut fused_output, fused_output_buffer)?;
        assert_eq!(fused_output, scatter_output);

        library.free_device_buffer(&mut fused_output_buffer)?;
        library.free_device_buffer(&mut scatter_output_buffer)?;
        library.free_device_buffer(&mut shared_buffer)?;
        library.free_device_buffer(&mut residual_buffer)?;
        library.free_device_buffer(&mut accumulator_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut row_index_buffer)?;
        library.free_device_buffer(&mut routed_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_route_shard_reduction_supports_all_wire_codecs_in_place() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let rows = 2_usize;
        let row_width = 32_usize;
        let values = rows * row_width;
        let local_values = vec![1.0_f32; values];
        let peer_values = vec![2.0_f32; values];
        let row_indices = [0_u32, 1_u32];
        let f32_bytes_len = values * std::mem::size_of::<f32>();
        let bf16_stride = row_width * std::mem::size_of::<u16>();
        let fp8_stride = row_width + std::mem::size_of::<f32>();
        let nvfp4_stride = row_width / 2 + row_width / 16;
        let peer_capacity = rows * bf16_stride;

        let mut local_f32 = library.alloc_device_buffer(f32_bytes_len)?;
        let mut local_bf16 = library.alloc_device_buffer(values * std::mem::size_of::<u16>())?;
        let mut output_f32 = library.alloc_device_buffer(f32_bytes_len)?;
        let mut peer_source = library.alloc_device_buffer(f32_bytes_len)?;
        let mut indices = library.alloc_device_buffer(std::mem::size_of_val(&row_indices))?;
        let mut peer = library.alloc_device_buffer(peer_capacity)?;
        library.copy_h2d(peer_source, f32_bytes(&peer_values))?;
        library.copy_h2d(indices, u32_bytes(&row_indices))?;

        for (dtype, stride, tolerance) in [
            (DS41RT_ROUTE_SHARD_WIRE_BF16, bf16_stride, 1.0e-6_f32),
            (
                DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
                fp8_stride,
                5.0e-2_f32,
            ),
            (
                DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
                nvfp4_stride,
                2.5e-1_f32,
            ),
        ] {
            match dtype {
                DS41RT_ROUTE_SHARD_WIRE_BF16 => {
                    library.cuda_f32_to_bf16(peer_source, peer, values)?;
                }
                DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED => {
                    library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
                        peer_source,
                        rows,
                        indices,
                        peer,
                        rows,
                        row_width,
                        stride,
                    )?;
                }
                DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3 => {
                    library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
                        peer_source,
                        rows,
                        indices,
                        peer,
                        rows,
                        row_width,
                        stride,
                    )?;
                }
                _ => unreachable!(),
            }
            for local_dtype in [DS41RT_ROUTE_SHARD_LOCAL_F32, DS41RT_ROUTE_SHARD_LOCAL_BF16] {
                library.copy_h2d(local_f32, f32_bytes(&local_values))?;
                let (local, output) = if local_dtype == DS41RT_ROUTE_SHARD_LOCAL_F32 {
                    (local_f32, local_f32)
                } else {
                    library.cuda_f32_to_bf16(local_f32, local_bf16, values)?;
                    (local_bf16, output_f32)
                };
                library.cuda_reduce_route_shards_to_f32(
                    &Ds41rtRouteShardReductionBuffers {
                        local,
                        peers: [peer; 3],
                        output_f32: output,
                    },
                    rows,
                    row_width,
                    stride,
                    local_dtype,
                    dtype,
                    3,
                )?;
                let mut output_bytes = vec![0_u8; f32_bytes_len];
                library.copy_d2h(&mut output_bytes, output)?;
                for actual in bytes_to_f32_vec(&output_bytes) {
                    assert!(
                        (actual - 7.0).abs() <= tolerance,
                        "local_dtype={local_dtype} peer_dtype={dtype} actual={actual}"
                    );
                }
            }
        }

        library.free_device_buffer(&mut peer)?;
        library.free_device_buffer(&mut indices)?;
        library.free_device_buffer(&mut peer_source)?;
        library.free_device_buffer(&mut output_f32)?;
        library.free_device_buffer(&mut local_bf16)?;
        library.free_device_buffer(&mut local_f32)?;
        Ok(())
    }

    #[test]
    fn cuda_nvfp4_gather_scatter_roundtrip() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let row_width = 16_usize;
        let source_rows = 2_usize;
        let row0 = [
            0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            0.0,
        ];
        let row1 = [
            0.0_f32, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0,
            -3.0, 0.0,
        ];
        let src = [row0, row1].concat();
        let gather_indices = [1_u32, 0];
        let scatter_indices = [1_u32, 1];
        let row_stride_bytes = row_width / 2 + row_width / 16;
        let mut packed = vec![0_u8; gather_indices.len() * row_stride_bytes];
        let mut scattered = vec![0_u8; source_rows * row_width * std::mem::size_of::<f32>()];

        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(src.as_slice()))?;
        let mut gather_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&gather_indices))?;
        let mut packed_buffer = library.alloc_device_buffer(packed.len())?;
        let mut scatter_indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&scatter_indices))?;
        let mut scattered_buffer = library.alloc_device_buffer(scattered.len())?;
        library.copy_h2d(src_buffer, f32_bytes(&src))?;
        library.copy_h2d(gather_indices_buffer, u32_bytes(&gather_indices))?;
        library.copy_h2d(scatter_indices_buffer, u32_bytes(&scatter_indices))?;
        library.copy_h2d(scattered_buffer, &scattered)?;

        library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
            src_buffer,
            source_rows,
            gather_indices_buffer,
            packed_buffer,
            gather_indices.len(),
            row_width,
            row_stride_bytes,
        )?;
        library.copy_d2h(&mut packed, packed_buffer)?;
        assert_ne!(packed, vec![0_u8; packed.len()]);

        library.cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32(
            packed_buffer,
            row_stride_bytes,
            scatter_indices_buffer,
            scattered_buffer,
            source_rows,
            scatter_indices.len(),
            row_width,
        )?;
        library.copy_d2h(&mut scattered, scattered_buffer)?;
        let actual = bytes_to_f32_vec(&scattered);
        for row in 0..source_rows {
            for col in 0..row_width {
                let expected = if row == 1 { row0[col] + row1[col] } else { 0.0 };
                assert!(
                    (actual[row * row_width + col] - expected).abs() <= 0.01,
                    "NVFP4 mismatch row={row} col={col} actual={} expected={expected}",
                    actual[row * row_width + col]
                );
            }
        }

        library.free_device_buffer(&mut scattered_buffer)?;
        library.free_device_buffer(&mut scatter_indices_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut gather_indices_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_copy_row_prefix_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let src_rows = 4;
        let src_row_width = 5;
        let rows = 2;
        let dst_row_width = 6;
        let prefix_width = 3;
        let src_row_offset = 1;
        let src = bf16_values(&[
            0.0_f32, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 2.1, 2.2, 2.3, 2.4, -3.0,
            -3.1, -3.2, -3.3, -3.4,
        ]);
        let initial_dst = bf16_values(&[
            9.0_f32, 9.1, 9.2, 9.3, 9.4, 9.5, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5,
        ]);
        let mut expected = initial_dst.clone();
        for row in 0..rows {
            let src_start = (src_row_offset + row) * src_row_width;
            let dst_start = row * dst_row_width;
            expected[dst_start..dst_start + prefix_width]
                .copy_from_slice(&src[src_start..src_start + prefix_width]);
        }
        let mut dst_bytes = vec![0_u8; initial_dst.len() * std::mem::size_of::<u16>()];

        let mut src_buffer = library.alloc_device_buffer(std::mem::size_of_val(src.as_slice()))?;
        let mut dst_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(initial_dst.as_slice()))?;

        library.copy_h2d(src_buffer, u16_bytes(&src))?;
        library.copy_h2d(dst_buffer, u16_bytes(&initial_dst))?;

        match library.cuda_copy_row_prefix_bf16(
            src_buffer,
            src_rows,
            dst_buffer,
            rows,
            src_row_width,
            dst_row_width,
            prefix_width,
            src_row_offset,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut dst_bytes, dst_buffer)?;
                let out = dst_bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(out, expected);

                library.copy_h2d(dst_buffer, u16_bytes(&initial_dst))?;
                unsafe {
                    library.cuda_copy_row_prefix_bf16_async(
                        src_buffer,
                        src_rows,
                        dst_buffer,
                        rows,
                        src_row_width,
                        dst_row_width,
                        prefix_width,
                        src_row_offset,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut dst_bytes, dst_buffer)?;
                let out = dst_bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(out, expected);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_copy_row_prefix_bf16_async(
                        src_buffer,
                        src_rows,
                        dst_buffer,
                        rows,
                        src_row_width,
                        dst_row_width,
                        prefix_width,
                        src_row_offset,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut dst_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_kv_cache_block_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let src_payload = [
            0x10_u8, 0x11, 0x12, 0x13, 0x14, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x30, 0x31, 0x32,
            0x33, 0x34,
        ];
        let src_offsets = [0_u64, 5, 11];
        let cache_offsets = [3_u64, 16, 25];
        let block_bytes = [5_u64, 6, 5];
        let mut cache_bytes = vec![0_u8; 32];
        let mut dst_bytes = vec![0_u8; src_payload.len()];

        let mut src_buffer = library.alloc_device_buffer(src_payload.len())?;
        let mut cache_buffer = library.alloc_device_buffer(cache_bytes.len())?;
        let mut dst_buffer = library.alloc_device_buffer(dst_bytes.len())?;
        let mut src_offsets_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&src_offsets))?;
        let mut cache_offsets_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&cache_offsets))?;
        let mut dst_offsets_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&src_offsets))?;
        let mut block_bytes_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&block_bytes))?;

        library.copy_h2d(src_buffer, &src_payload)?;
        library.copy_h2d(cache_buffer, &cache_bytes)?;
        library.copy_h2d(dst_buffer, &dst_bytes)?;
        library.copy_h2d(src_offsets_buffer, u64_bytes(&src_offsets))?;
        library.copy_h2d(cache_offsets_buffer, u64_bytes(&cache_offsets))?;
        library.copy_h2d(dst_offsets_buffer, u64_bytes(&src_offsets))?;
        library.copy_h2d(block_bytes_buffer, u64_bytes(&block_bytes))?;

        let write_result = unsafe {
            library.cuda_kv_cache_write_blocks(
                src_buffer,
                cache_buffer,
                src_offsets_buffer,
                cache_offsets_buffer,
                block_bytes_buffer,
                block_bytes.len(),
            )
        };
        match write_result {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                unsafe {
                    library.cuda_kv_cache_read_blocks(
                        cache_buffer,
                        dst_buffer,
                        cache_offsets_buffer,
                        dst_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                    )?;
                }
                library.copy_d2h(&mut dst_bytes, dst_buffer)?;
                assert_eq!(dst_bytes, src_payload);

                unsafe {
                    library.cuda_kv_cache_write_blocks_async(
                        src_buffer,
                        cache_buffer,
                        src_offsets_buffer,
                        cache_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_kv_cache_read_blocks_async(
                        cache_buffer,
                        dst_buffer,
                        cache_offsets_buffer,
                        dst_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut dst_bytes, dst_buffer)?;
                assert_eq!(dst_bytes, src_payload);
                library.copy_d2h(&mut cache_bytes, cache_buffer)?;
                assert_eq!(&cache_bytes[3..8], &src_payload[0..5]);
                assert_eq!(&cache_bytes[16..22], &src_payload[5..11]);
                assert_eq!(&cache_bytes[25..30], &src_payload[11..16]);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let read_err = unsafe {
                    library.cuda_kv_cache_read_blocks(
                        cache_buffer,
                        dst_buffer,
                        cache_offsets_buffer,
                        dst_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(read_err);
                let async_write_err = unsafe {
                    library.cuda_kv_cache_write_blocks_async(
                        src_buffer,
                        cache_buffer,
                        src_offsets_buffer,
                        cache_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_write_err);
                let async_read_err = unsafe {
                    library.cuda_kv_cache_read_blocks_async(
                        cache_buffer,
                        dst_buffer,
                        cache_offsets_buffer,
                        dst_offsets_buffer,
                        block_bytes_buffer,
                        block_bytes.len(),
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_read_err);
            }
        }

        library.free_device_buffer(&mut block_bytes_buffer)?;
        library.free_device_buffer(&mut dst_offsets_buffer)?;
        library.free_device_buffer(&mut cache_offsets_buffer)?;
        library.free_device_buffer(&mut src_offsets_buffer)?;
        library.free_device_buffer(&mut dst_buffer)?;
        library.free_device_buffer(&mut cache_buffer)?;
        library.free_device_buffer(&mut src_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_mla_kv_cache_unpack_and_split_bf16_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let kv_lora_rank = 3;
        let rope_dim = 2;
        let dsa_dim = 1;
        let payload_stride_values = kv_lora_rank + rope_dim + dsa_dim;
        let payload_stride_bytes = payload_stride_values * std::mem::size_of::<u16>();
        let payload = [
            101_u16, 102, 103, 201, 202, 301, 111, 112, 113, 211, 212, 311,
        ];
        let expected_latent = [101_u16, 102, 103, 111, 112, 113];
        let expected_rope = [201_u16, 202, 211, 212];
        let expected_dsa = [301_u16, 311];

        let heads = 2;
        let nope_dim = 2;
        let v_dim = 3;
        let projected = [
            10_u16, 11, 20, 21, 22, 12, 13, 23, 24, 25, 30, 31, 40, 41, 42, 32, 33, 43, 44, 45,
        ];
        let expected_k_nope = [10_u16, 11, 12, 13, 30, 31, 32, 33];
        let expected_v = [20_u16, 21, 22, 23, 24, 25, 40, 41, 42, 43, 44, 45];

        let zero_latent = vec![0_u8; std::mem::size_of_val(expected_latent.as_slice())];
        let zero_rope = vec![0_u8; std::mem::size_of_val(expected_rope.as_slice())];
        let zero_dsa = vec![0_u8; std::mem::size_of_val(expected_dsa.as_slice())];
        let zero_k_nope = vec![0_u8; std::mem::size_of_val(expected_k_nope.as_slice())];
        let zero_v = vec![0_u8; std::mem::size_of_val(expected_v.as_slice())];

        let mut payload_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(payload.as_slice()))?;
        let mut latent_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(expected_latent.as_slice()))?;
        let mut rope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(expected_rope.as_slice()))?;
        let mut dsa_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(expected_dsa.as_slice()))?;
        let mut projected_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(projected.as_slice()))?;
        let mut k_nope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(expected_k_nope.as_slice()))?;
        let mut v_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(expected_v.as_slice()))?;

        library.copy_h2d(payload_buffer, u16_bytes(&payload))?;
        library.copy_h2d(projected_buffer, u16_bytes(&projected))?;
        library.copy_h2d(latent_buffer, &zero_latent)?;
        library.copy_h2d(rope_buffer, &zero_rope)?;
        library.copy_h2d(dsa_buffer, &zero_dsa)?;
        library.copy_h2d(k_nope_buffer, &zero_k_nope)?;
        library.copy_h2d(v_buffer, &zero_v)?;

        let unpack_result = library.cuda_mla_kv_cache_unpack_bf16(
            payload_buffer,
            latent_buffer,
            rope_buffer,
            Some(dsa_buffer),
            rows,
            kv_lora_rank,
            rope_dim,
            dsa_dim,
            payload_stride_bytes,
        );
        match unpack_result {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                assert_u16_device_buffer_eq(&library, latent_buffer, &expected_latent)?;
                assert_u16_device_buffer_eq(&library, rope_buffer, &expected_rope)?;
                assert_u16_device_buffer_eq(&library, dsa_buffer, &expected_dsa)?;

                library.copy_h2d(latent_buffer, &zero_latent)?;
                library.copy_h2d(rope_buffer, &zero_rope)?;
                library.copy_h2d(dsa_buffer, &zero_dsa)?;
                unsafe {
                    library.cuda_mla_kv_cache_unpack_bf16_async(
                        payload_buffer,
                        latent_buffer,
                        rope_buffer,
                        Some(dsa_buffer),
                        rows,
                        kv_lora_rank,
                        rope_dim,
                        dsa_dim,
                        payload_stride_bytes,
                        std::ptr::null_mut(),
                    )?;
                }
                assert_u16_device_buffer_eq(&library, latent_buffer, &expected_latent)?;
                assert_u16_device_buffer_eq(&library, rope_buffer, &expected_rope)?;
                assert_u16_device_buffer_eq(&library, dsa_buffer, &expected_dsa)?;

                library.cuda_mla_kv_projected_split_bf16(
                    projected_buffer,
                    k_nope_buffer,
                    v_buffer,
                    rows,
                    heads,
                    nope_dim,
                    v_dim,
                )?;
                assert_u16_device_buffer_eq(&library, k_nope_buffer, &expected_k_nope)?;
                assert_u16_device_buffer_eq(&library, v_buffer, &expected_v)?;

                library.copy_h2d(k_nope_buffer, &zero_k_nope)?;
                library.copy_h2d(v_buffer, &zero_v)?;
                unsafe {
                    library.cuda_mla_kv_projected_split_bf16_async(
                        projected_buffer,
                        k_nope_buffer,
                        v_buffer,
                        rows,
                        heads,
                        nope_dim,
                        v_dim,
                        std::ptr::null_mut(),
                    )?;
                }
                assert_u16_device_buffer_eq(&library, k_nope_buffer, &expected_k_nope)?;
                assert_u16_device_buffer_eq(&library, v_buffer, &expected_v)?;
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_unpack_err = unsafe {
                    library.cuda_mla_kv_cache_unpack_bf16_async(
                        payload_buffer,
                        latent_buffer,
                        rope_buffer,
                        Some(dsa_buffer),
                        rows,
                        kv_lora_rank,
                        rope_dim,
                        dsa_dim,
                        payload_stride_bytes,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_unpack_err);
                let split_err = library
                    .cuda_mla_kv_projected_split_bf16(
                        projected_buffer,
                        k_nope_buffer,
                        v_buffer,
                        rows,
                        heads,
                        nope_dim,
                        v_dim,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(split_err);
                let async_split_err = unsafe {
                    library.cuda_mla_kv_projected_split_bf16_async(
                        projected_buffer,
                        k_nope_buffer,
                        v_buffer,
                        rows,
                        heads,
                        nope_dim,
                        v_dim,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_split_err);
            }
        }

        library.free_device_buffer(&mut v_buffer)?;
        library.free_device_buffer(&mut k_nope_buffer)?;
        library.free_device_buffer(&mut projected_buffer)?;
        library.free_device_buffer(&mut dsa_buffer)?;
        library.free_device_buffer(&mut rope_buffer)?;
        library.free_device_buffer(&mut latent_buffer)?;
        library.free_device_buffer(&mut payload_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_sparse_mla_layout_and_page_table_kernels_match_reference() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3_usize;
        let heads = 2_usize;
        let width = 8_usize;
        let input = (0..rows * heads * width)
            .map(|value| value as u16)
            .collect::<Vec<_>>();
        let mut expected_transposed = vec![0_u16; input.len()];
        for row in 0..rows {
            for head in 0..heads {
                let src = (row * heads + head) * width;
                let dst = (head * rows + row) * width;
                expected_transposed[dst..dst + width].copy_from_slice(&input[src..src + width]);
            }
        }
        let mut input_buffer = library.alloc_device_buffer(std::mem::size_of_val(&input[..]))?;
        let mut transposed_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&input[..]))?;
        let mut restored_buffer = library.alloc_device_buffer(std::mem::size_of_val(&input[..]))?;
        library.copy_h2d(input_buffer, u16_bytes(&input))?;

        let transpose_result = library.cuda_transpose_rows_heads_bf16(
            input_buffer,
            transposed_buffer,
            rows,
            heads,
            width,
        );
        if info.cuda_available == 0 {
            assert_cuda_unavailable(transpose_result.unwrap_err());
            library.free_device_buffer(&mut restored_buffer)?;
            library.free_device_buffer(&mut transposed_buffer)?;
            library.free_device_buffer(&mut input_buffer)?;
            return Ok(());
        }
        transpose_result?;
        let mut transposed_bytes = vec![0_u8; std::mem::size_of_val(&input[..])];
        library.copy_d2h(&mut transposed_bytes, transposed_buffer)?;
        assert_eq!(bytes_to_u16_vec(&transposed_bytes), expected_transposed);
        library.cuda_transpose_heads_rows_bf16(
            transposed_buffer,
            restored_buffer,
            rows,
            heads,
            width,
        )?;
        let mut restored_bytes = vec![0_u8; std::mem::size_of_val(&input[..])];
        library.copy_d2h(&mut restored_bytes, restored_buffer)?;
        assert_eq!(bytes_to_u16_vec(&restored_bytes), input);

        let latent_width = 8_usize;
        let rope_width = 8_usize;
        let latent = expected_transposed.clone();
        let rope = (0..rows * heads * rope_width)
            .map(|value| 1_000_u16 + value as u16)
            .collect::<Vec<_>>();
        let mut expected_composed = Vec::with_capacity(rows * heads * (latent_width + rope_width));
        for row in 0..rows {
            for head in 0..heads {
                let latent_start = (head * rows + row) * latent_width;
                let rope_start = (row * heads + head) * rope_width;
                expected_composed
                    .extend_from_slice(&latent[latent_start..latent_start + latent_width]);
                expected_composed.extend_from_slice(&rope[rope_start..rope_start + rope_width]);
            }
        }
        let mut rope_buffer = library.alloc_device_buffer(std::mem::size_of_val(&rope[..]))?;
        let mut composed_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&expected_composed[..]))?;
        library.copy_h2d(rope_buffer, u16_bytes(&rope))?;
        library.cuda_mla_compose_absorbed_query_bf16(
            transposed_buffer,
            rope_buffer,
            composed_buffer,
            rows,
            heads,
            latent_width,
            rope_width,
        )?;
        let mut composed_bytes = vec![0_u8; std::mem::size_of_val(&expected_composed[..])];
        library.copy_d2h(&mut composed_bytes, composed_buffer)?;
        assert_eq!(bytes_to_u16_vec(&composed_bytes), expected_composed);

        let page_query_rows = 3_usize;
        let page_table_width = 5_usize;
        let page_table_entries = page_query_rows * page_table_width;
        let mut page_table_buffer =
            library.alloc_device_buffer(page_table_entries * std::mem::size_of::<i32>())?;
        library.cuda_generic_kv_page_table_init(
            page_table_buffer,
            page_query_rows,
            page_table_width,
        )?;
        let mut page_table_bytes = vec![0_u8; page_table_entries * std::mem::size_of::<i32>()];
        library.copy_d2h(&mut page_table_bytes, page_table_buffer)?;
        let page_table = bytes_to_i32_vec(&page_table_bytes);
        assert_eq!(page_table, [0_i32, 1, 2, 3, 4].repeat(page_query_rows));

        library.cuda_generic_kv_page_table_init_base(
            page_table_buffer,
            page_query_rows,
            page_table_width,
            11,
        )?;
        library.copy_d2h(&mut page_table_bytes, page_table_buffer)?;
        assert_eq!(
            bytes_to_i32_vec(&page_table_bytes),
            [11_i32, 12, 13, 14, 15].repeat(page_query_rows)
        );

        let row_offsets = [7_i32, 23, 41];
        let mut row_offsets_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&row_offsets))?;
        library.copy_h2d(row_offsets_buffer, i32_bytes(&row_offsets))?;
        let cuda_stream = library.cuda_stream_create()?;
        unsafe {
            library.cuda_generic_kv_page_table_init_offsets_async(
                page_table_buffer,
                row_offsets_buffer,
                page_query_rows,
                page_table_width,
                cuda_stream,
            )?;
            library.cuda_stream_synchronize(cuda_stream)?;
            library.cuda_stream_destroy(cuda_stream)?;
        }
        library.copy_d2h(&mut page_table_bytes, page_table_buffer)?;
        let expected_offset_table = row_offsets
            .iter()
            .flat_map(|offset| (0..page_table_width).map(move |column| offset + column as i32))
            .collect::<Vec<_>>();
        assert_eq!(bytes_to_i32_vec(&page_table_bytes), expected_offset_table);

        let physical_pages = [3_u32, 1_u32];
        let expanded_query_rows = 2_usize;
        let expanded_width = 80_usize;
        let expanded_active_tokens = 70_usize;
        let mut physical_pages_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&physical_pages))?;
        let mut expanded_indices_buffer = library.alloc_device_buffer(
            expanded_query_rows * expanded_width * std::mem::size_of::<i32>(),
        )?;
        library.copy_h2d(physical_pages_buffer, u32_bytes(&physical_pages))?;
        library.cuda_generic_kv_page_table_expand_indices(
            expanded_indices_buffer,
            physical_pages_buffer,
            expanded_query_rows,
            expanded_width,
            expanded_active_tokens,
        )?;
        let mut expanded_indices_bytes =
            vec![0_u8; expanded_query_rows * expanded_width * std::mem::size_of::<i32>()];
        library.copy_d2h(&mut expanded_indices_bytes, expanded_indices_buffer)?;
        let expected_expanded_row = (0..expanded_width)
            .map(|logical_token| {
                if logical_token >= expanded_active_tokens {
                    0
                } else {
                    let page = physical_pages[logical_token / DS41RT_CUDA_GENERIC_KV_PAGE_SIZE];
                    (page as usize * DS41RT_CUDA_GENERIC_KV_PAGE_SIZE
                        + logical_token % DS41RT_CUDA_GENERIC_KV_PAGE_SIZE)
                        as i32
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            bytes_to_i32_vec(&expanded_indices_bytes),
            expected_expanded_row.repeat(expanded_query_rows)
        );

        library.free_device_buffer(&mut expanded_indices_buffer)?;
        library.free_device_buffer(&mut physical_pages_buffer)?;
        library.free_device_buffer(&mut row_offsets_buffer)?;
        library.free_device_buffer(&mut page_table_buffer)?;
        library.free_device_buffer(&mut composed_buffer)?;
        library.free_device_buffer(&mut rope_buffer)?;
        library.free_device_buffer(&mut restored_buffer)?;
        library.free_device_buffer(&mut transposed_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_sparse_mla_nvfp4_reads_selected_compressed_rows_directly() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        if info.cuda_available == 0 {
            return Ok(());
        }
        const QUERY_ROWS: usize = 1;
        const HEADS: usize = 64;
        const RANK: usize = 512;
        const ROPE: usize = 64;
        const TOPK: usize = 2048;
        const SPLITS: usize = 32;
        const PHYSICAL_ROWS: usize = 3;
        const ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;

        let query = vec![0_u16; QUERY_ROWS * HEADS * (RANK + ROPE)];
        let mut kv = vec![0_u8; PHYSICAL_ROWS * ROW_BYTES];
        for row in kv.chunks_exact_mut(ROW_BYTES) {
            row[..RANK / 2].fill(0x22); // two +1 E2M1 codes per byte
            row[RANK / 2..RANK / 2 + RANK / 16].fill(0x38); // E4M3 scale 1
        }
        let mut indices = vec![0_i32; QUERY_ROWS * TOPK];
        indices[0] = 2;
        indices[1] = 0;
        indices[2] = 1;
        let lengths = [3_i32];

        let mut query_buffer = library.alloc_device_buffer(std::mem::size_of_val(&query[..]))?;
        let mut kv_buffer = library.alloc_device_buffer(kv.len())?;
        let mut indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices[..]))?;
        let mut lengths_buffer = library.alloc_device_buffer(std::mem::size_of_val(&lengths))?;
        let mut partial_buffer = library
            .alloc_device_buffer(QUERY_ROWS * HEADS * SPLITS * RANK * std::mem::size_of::<u16>())?;
        let mut partial_lse_buffer = library
            .alloc_device_buffer(QUERY_ROWS * HEADS * SPLITS * std::mem::size_of::<f32>())?;
        let output_bytes = QUERY_ROWS * HEADS * RANK * std::mem::size_of::<u16>();
        let mut output_buffer = library.alloc_device_buffer(output_bytes)?;
        let mut output_lse_buffer =
            library.alloc_device_buffer(QUERY_ROWS * HEADS * std::mem::size_of::<f32>())?;
        library.copy_h2d(query_buffer, u16_bytes(&query))?;
        library.copy_h2d(kv_buffer, &kv)?;
        library.copy_h2d(indices_buffer, i32_bytes(&indices))?;
        library.copy_h2d(lengths_buffer, i32_bytes(&lengths))?;
        unsafe {
            library.cuda_sparse_mla_nvfp4_async(
                query_buffer,
                kv_buffer,
                indices_buffer,
                lengths_buffer,
                partial_buffer,
                partial_lse_buffer,
                output_buffer,
                output_lse_buffer,
                QUERY_ROWS,
                HEADS,
                TOPK,
                ROW_BYTES,
                1.0,
                std::ptr::null_mut(),
            )?;
        }
        let mut output = vec![0_u8; output_bytes];
        library.copy_d2h(&mut output, output_buffer)?;
        for (index, bits) in bytes_to_u16_vec(&output).into_iter().enumerate() {
            let actual = bf16_to_f32(bits);
            assert!(
                (actual - 1.0).abs() <= 0.01,
                "direct sparse NVFP4 MLA output[{index}]={actual}, expected 1"
            );
        }

        library.free_device_buffer(&mut output_lse_buffer)?;
        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut partial_lse_buffer)?;
        library.free_device_buffer(&mut partial_buffer)?;
        library.free_device_buffer(&mut lengths_buffer)?;
        library.free_device_buffer(&mut indices_buffer)?;
        library.free_device_buffer(&mut kv_buffer)?;
        library.free_device_buffer(&mut query_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_sparse_mla_bf16_reads_selected_compressed_rows_directly() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        if info.cuda_available == 0 {
            return Ok(());
        }
        const QUERY_ROWS: usize = 1;
        const HEADS: usize = 64;
        const RANK: usize = 512;
        const ROPE: usize = 64;
        const TOPK: usize = 2048;
        const SPLITS: usize = 32;
        const PHYSICAL_ROWS: usize = 3;
        const ROW_VALUES: usize = RANK + ROPE;
        const ROW_BYTES: usize = ROW_VALUES * std::mem::size_of::<u16>();

        let mut query = vec![0_u16; QUERY_ROWS * HEADS * ROW_VALUES];
        for head in query.chunks_exact_mut(ROW_VALUES) {
            head[..RANK].fill(f32_to_bf16(1.0));
        }
        let mut kv = vec![0_u16; PHYSICAL_ROWS * ROW_VALUES];
        for (physical_row, row) in kv.chunks_exact_mut(ROW_VALUES).enumerate() {
            row[..RANK].fill(f32_to_bf16((physical_row + 1) as f32));
        }
        let mut indices = vec![0_i32; QUERY_ROWS * TOPK];
        indices[..2].copy_from_slice(&[2, 0]);
        let lengths = [2_i32];

        let mut query_buffer = library.alloc_device_buffer(std::mem::size_of_val(&query[..]))?;
        let mut kv_buffer = library.alloc_device_buffer(std::mem::size_of_val(&kv[..]))?;
        let mut indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices[..]))?;
        let mut lengths_buffer = library.alloc_device_buffer(std::mem::size_of_val(&lengths))?;
        let mut partial_buffer = library
            .alloc_device_buffer(QUERY_ROWS * HEADS * SPLITS * RANK * std::mem::size_of::<u16>())?;
        let mut partial_lse_buffer = library
            .alloc_device_buffer(QUERY_ROWS * HEADS * SPLITS * std::mem::size_of::<f32>())?;
        let output_bytes = QUERY_ROWS * HEADS * RANK * std::mem::size_of::<u16>();
        let mut output_buffer = library.alloc_device_buffer(output_bytes)?;
        let mut output_lse_buffer =
            library.alloc_device_buffer(QUERY_ROWS * HEADS * std::mem::size_of::<f32>())?;
        library.copy_h2d(query_buffer, u16_bytes(&query))?;
        library.copy_h2d(kv_buffer, u16_bytes(&kv))?;
        library.copy_h2d(indices_buffer, i32_bytes(&indices))?;
        library.copy_h2d(lengths_buffer, i32_bytes(&lengths))?;
        unsafe {
            library.cuda_sparse_mla_bf16_async(
                query_buffer,
                kv_buffer,
                indices_buffer,
                lengths_buffer,
                partial_buffer,
                partial_lse_buffer,
                output_buffer,
                output_lse_buffer,
                QUERY_ROWS,
                HEADS,
                TOPK,
                ROW_BYTES,
                1.0,
                std::ptr::null_mut(),
            )?;
        }
        let mut output = vec![0_u8; output_bytes];
        library.copy_d2h(&mut output, output_buffer)?;
        for (index, bits) in bytes_to_u16_vec(&output).into_iter().enumerate() {
            let actual = bf16_to_f32(bits);
            assert!(
                (actual - 3.0).abs() <= 0.01,
                "direct sparse BF16 MLA output[{index}]={actual}, expected selected high-score row 3"
            );
        }

        library.free_device_buffer(&mut output_lse_buffer)?;
        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut partial_lse_buffer)?;
        library.free_device_buffer(&mut partial_buffer)?;
        library.free_device_buffer(&mut lengths_buffer)?;
        library.free_device_buffer(&mut indices_buffer)?;
        library.free_device_buffer(&mut kv_buffer)?;
        library.free_device_buffer(&mut query_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_sparse_mla_bf16_gather_gemm_path_shares_selected_rows_across_heads() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        if info.cuda_available == 0 {
            return Ok(());
        }
        const QUERY_ROWS: usize = 2;
        const HEADS: usize = 64;
        const RANK: usize = 512;
        const ROPE: usize = 64;
        const HEAD_DIM: usize = RANK + ROPE;
        const TOPK: usize = 2048;
        const PHYSICAL_ROWS: usize = 3;
        const ROW_BYTES: usize = HEAD_DIM * std::mem::size_of::<u16>();

        let mut query = vec![0_u16; QUERY_ROWS * HEADS * HEAD_DIM];
        for head in query.chunks_exact_mut(HEAD_DIM) {
            head[..RANK].fill(f32_to_bf16(1.0));
        }
        let mut kv = vec![0_u16; PHYSICAL_ROWS * HEAD_DIM];
        for (physical_row, row) in kv.chunks_exact_mut(HEAD_DIM).enumerate() {
            row[..RANK].fill(f32_to_bf16((physical_row + 1) as f32));
        }
        let mut indices = vec![0_i32; QUERY_ROWS * TOPK];
        for row in indices.chunks_exact_mut(TOPK) {
            row[..2].copy_from_slice(&[2, 0]);
        }
        let lengths = [2_i32; QUERY_ROWS];

        let mut query_buffer = library.alloc_device_buffer(std::mem::size_of_val(&query[..]))?;
        let mut kv_buffer = library.alloc_device_buffer(std::mem::size_of_val(&kv[..]))?;
        let mut indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices[..]))?;
        let mut lengths_buffer = library.alloc_device_buffer(std::mem::size_of_val(&lengths))?;
        let mut gathered_k = library
            .alloc_device_buffer(QUERY_ROWS * TOPK * HEAD_DIM * std::mem::size_of::<u16>())?;
        let mut gathered_v =
            library.alloc_device_buffer(QUERY_ROWS * TOPK * RANK * std::mem::size_of::<u16>())?;
        let mut scores =
            library.alloc_device_buffer(QUERY_ROWS * HEADS * TOPK * std::mem::size_of::<u16>())?;
        let output_bytes = QUERY_ROWS * HEADS * RANK * std::mem::size_of::<u16>();
        let mut output = library.alloc_device_buffer(output_bytes)?;
        let mut output_lse =
            library.alloc_device_buffer(QUERY_ROWS * HEADS * std::mem::size_of::<f32>())?;
        library.copy_h2d(query_buffer, u16_bytes(&query))?;
        library.copy_h2d(kv_buffer, u16_bytes(&kv))?;
        library.copy_h2d(indices_buffer, i32_bytes(&indices))?;
        library.copy_h2d(lengths_buffer, i32_bytes(&lengths))?;
        unsafe {
            library.cuda_sparse_mla_bf16_gather_kv_async(
                kv_buffer,
                indices_buffer,
                lengths_buffer,
                gathered_k,
                gathered_v,
                QUERY_ROWS,
                TOPK,
                ROW_BYTES,
                std::ptr::null_mut(),
            )?;
            library.cuda_linear_bf16_strided_batched_cublas_async(
                query_buffer,
                gathered_k,
                scores,
                QUERY_ROWS,
                HEADS,
                HEAD_DIM,
                TOPK,
                HEADS * HEAD_DIM,
                TOPK * HEAD_DIM,
                HEADS * TOPK,
                std::ptr::null_mut(),
            )?;
            library.cuda_sparse_mla_bf16_softmax_async(
                scores,
                lengths_buffer,
                output_lse,
                QUERY_ROWS,
                HEADS,
                TOPK,
                1.0,
                std::ptr::null_mut(),
            )?;
            library.cuda_matmul_bf16_strided_batched_cublas_async(
                scores,
                gathered_v,
                output,
                QUERY_ROWS,
                HEADS,
                TOPK,
                RANK,
                HEADS * TOPK,
                TOPK * RANK,
                HEADS * RANK,
                std::ptr::null_mut(),
            )?;
        }
        let mut output_host = vec![0_u8; output_bytes];
        library.copy_d2h(&mut output_host, output)?;
        for (index, bits) in bytes_to_u16_vec(&output_host).into_iter().enumerate() {
            let actual = bf16_to_f32(bits);
            assert!(
                (actual - 3.0).abs() <= 0.01,
                "gather/GEMM sparse BF16 MLA output[{index}]={actual}, expected selected high-score row 3"
            );
        }

        library.free_device_buffer(&mut output_lse)?;
        library.free_device_buffer(&mut output)?;
        library.free_device_buffer(&mut scores)?;
        library.free_device_buffer(&mut gathered_v)?;
        library.free_device_buffer(&mut gathered_k)?;
        library.free_device_buffer(&mut lengths_buffer)?;
        library.free_device_buffer(&mut indices_buffer)?;
        library.free_device_buffer(&mut kv_buffer)?;
        library.free_device_buffer(&mut query_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_sparse_mla_nvfp4_gather_fp8_compacts_selected_rows() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        if info.cuda_available == 0 {
            return Ok(());
        }
        const QUERY_ROWS: usize = 1;
        const SELECTED_INDEX_STRIDE: usize = 2048;
        const STAGED_TOPK: usize = 128;
        const PHYSICAL_ROWS: usize = 3;
        const NVFP4_ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        const FP8_ROW_BYTES: usize = 656;
        const RANK: usize = 512;
        const ROPE_OFFSET: usize = RANK / 2 + RANK / 16 + DS41RT_CUDA_MLA_MXFP4_DS_PADDING_BYTES;
        const FP8_SCALE_OFFSET: usize = RANK;
        const FP8_ROPE_OFFSET: usize = RANK + 4 * std::mem::size_of::<f32>();

        let mut kv = vec![0_u8; PHYSICAL_ROWS * NVFP4_ROW_BYTES];
        for (physical_row, row) in kv.chunks_exact_mut(NVFP4_ROW_BYTES).enumerate() {
            row[..RANK / 2].fill(0x22); // two +1 E2M1 codes per byte
            row[RANK / 2..RANK / 2 + RANK / 16].fill(0x38); // E4M3 scale 1
            for rope_col in 0..64 {
                let bits = f32_to_bf16((physical_row * 100 + rope_col) as f32);
                let byte = ROPE_OFFSET + rope_col * std::mem::size_of::<u16>();
                row[byte..byte + 2].copy_from_slice(&bits.to_ne_bytes());
            }
        }
        let mut indices = vec![0_i32; QUERY_ROWS * SELECTED_INDEX_STRIDE];
        indices[..3].copy_from_slice(&[2, 0, 1]);
        let lengths = [3_i32];

        let mut kv_buffer = library.alloc_device_buffer(kv.len())?;
        let mut indices_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&indices[..]))?;
        let mut lengths_buffer = library.alloc_device_buffer(std::mem::size_of_val(&lengths))?;
        let mut fp8_buffer =
            library.alloc_device_buffer(QUERY_ROWS * STAGED_TOPK * FP8_ROW_BYTES)?;
        let mut fp8_indices_buffer =
            library.alloc_device_buffer(QUERY_ROWS * STAGED_TOPK * std::mem::size_of::<i32>())?;
        library.copy_h2d(kv_buffer, &kv)?;
        library.copy_h2d(indices_buffer, i32_bytes(&indices))?;
        library.copy_h2d(lengths_buffer, i32_bytes(&lengths))?;
        unsafe {
            library.cuda_sparse_mla_nvfp4_gather_fp8_async(
                kv_buffer,
                indices_buffer,
                lengths_buffer,
                fp8_buffer,
                fp8_indices_buffer,
                QUERY_ROWS,
                SELECTED_INDEX_STRIDE,
                STAGED_TOPK,
                NVFP4_ROW_BYTES,
                std::ptr::null_mut(),
            )?;
        }

        let mut fp8 = vec![0_u8; 3 * FP8_ROW_BYTES];
        let fp8_view = Ds41rtDeviceBuffer {
            bytes: fp8.len(),
            ..fp8_buffer
        };
        library.copy_d2h(&mut fp8, fp8_view)?;
        let mut staged_indices = vec![0_u8; 3 * std::mem::size_of::<i32>()];
        let fp8_indices_view = Ds41rtDeviceBuffer {
            bytes: staged_indices.len(),
            ..fp8_indices_buffer
        };
        library.copy_d2h(&mut staged_indices, fp8_indices_view)?;
        assert_eq!(bytes_to_i32_vec(&staged_indices), [0, 1, 2]);

        for (staged_row, physical_row) in [2_usize, 0, 1].into_iter().enumerate() {
            let row = &fp8[staged_row * FP8_ROW_BYTES..(staged_row + 1) * FP8_ROW_BYTES];
            for group in 0..4 {
                let byte = FP8_SCALE_OFFSET + group * std::mem::size_of::<f32>();
                let scale = f32::from_ne_bytes(row[byte..byte + 4].try_into().unwrap());
                let reconstructed = f8e4m3_to_f32(row[group * 128]) * scale;
                assert!(
                    (reconstructed - 1.0).abs() <= 0.01,
                    "staged row {staged_row} group {group} reconstructed {reconstructed}"
                );
            }
            for rope_col in 0..64 {
                let byte = FP8_ROPE_OFFSET + rope_col * std::mem::size_of::<u16>();
                let actual = u16::from_ne_bytes(row[byte..byte + 2].try_into().unwrap());
                let expected = f32_to_bf16((physical_row * 100 + rope_col) as f32);
                assert_eq!(actual, expected);
            }
        }

        library.free_device_buffer(&mut fp8_indices_buffer)?;
        library.free_device_buffer(&mut fp8_buffer)?;
        library.free_device_buffer(&mut lengths_buffer)?;
        library.free_device_buffer(&mut indices_buffer)?;
        library.free_device_buffer(&mut kv_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_mla_nvfp4_expand_fp8_paged_preserves_physical_slots() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        if info.cuda_available == 0 {
            return Ok(());
        }
        const PAGE_SIZE: usize = 64;
        const MAX_TOKENS: usize = 128;
        const ACTIVE_ROWS: i32 = 65;
        const NVFP4_ROW_BYTES: usize = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        const FP8_ROW_BYTES: usize = 656;
        const RANK: usize = 512;
        const ROPE_OFFSET: usize = RANK / 2 + RANK / 16 + DS41RT_CUDA_MLA_MXFP4_DS_PADDING_BYTES;
        const FP8_SCALE_OFFSET: usize = RANK;
        const FP8_ROPE_OFFSET: usize = RANK + 4 * std::mem::size_of::<f32>();

        let mut kv = vec![0_u8; MAX_TOKENS * NVFP4_ROW_BYTES];
        for (physical_row, row) in kv.chunks_exact_mut(NVFP4_ROW_BYTES).enumerate() {
            row[..RANK / 2].fill(0x22);
            row[RANK / 2..RANK / 2 + RANK / 16].fill(0x38);
            for rope_col in 0..64 {
                let bits = f32_to_bf16((physical_row * 100 + rope_col) as f32);
                let byte = ROPE_OFFSET + rope_col * std::mem::size_of::<u16>();
                row[byte..byte + 2].copy_from_slice(&bits.to_ne_bytes());
            }
        }
        let physical_pages = [1_u32, 0_u32];
        let active_rows = [ACTIVE_ROWS];
        let mut kv_buffer = library.alloc_device_buffer(kv.len())?;
        let mut pages_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&physical_pages))?;
        let mut active_buffer = library.alloc_device_buffer(std::mem::size_of_val(&active_rows))?;
        let mut fp8_buffer = library.alloc_device_buffer(MAX_TOKENS * FP8_ROW_BYTES)?;
        library.copy_h2d(kv_buffer, &kv)?;
        library.copy_h2d(pages_buffer, u32_bytes(&physical_pages))?;
        library.copy_h2d(active_buffer, i32_bytes(&active_rows))?;
        library.cuda_zero_bytes(fp8_buffer, fp8_buffer.bytes)?;
        unsafe {
            library.cuda_mla_nvfp4_expand_fp8_paged_async(
                kv_buffer,
                pages_buffer,
                active_buffer,
                fp8_buffer,
                MAX_TOKENS,
                PAGE_SIZE,
                NVFP4_ROW_BYTES,
                std::ptr::null_mut(),
            )?;
        }

        let mut fp8 = vec![0_u8; MAX_TOKENS * FP8_ROW_BYTES];
        library.copy_d2h(&mut fp8, fp8_buffer)?;
        for physical_row in [64_usize, 0] {
            let row = &fp8[physical_row * FP8_ROW_BYTES..(physical_row + 1) * FP8_ROW_BYTES];
            for group in 0..4 {
                let byte = FP8_SCALE_OFFSET + group * std::mem::size_of::<f32>();
                let scale = f32::from_ne_bytes(row[byte..byte + 4].try_into().unwrap());
                let reconstructed = f8e4m3_to_f32(row[group * 128]) * scale;
                assert!(
                    (reconstructed - 1.0).abs() <= 0.01,
                    "expanded physical row {physical_row} group {group} reconstructed {reconstructed}"
                );
            }
            for rope_col in 0..64 {
                let byte = FP8_ROPE_OFFSET + rope_col * std::mem::size_of::<u16>();
                let actual = u16::from_ne_bytes(row[byte..byte + 2].try_into().unwrap());
                let expected = f32_to_bf16((physical_row * 100 + rope_col) as f32);
                assert_eq!(actual, expected);
            }
        }
        assert!(
            fp8[FP8_ROW_BYTES..2 * FP8_ROW_BYTES]
                .iter()
                .all(|value| *value == 0),
            "inactive physical row was unexpectedly expanded"
        );

        library.free_device_buffer(&mut fp8_buffer)?;
        library.free_device_buffer(&mut active_buffer)?;
        library.free_device_buffer(&mut pages_buffer)?;
        library.free_device_buffer(&mut kv_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_mla_kv_fp8_ds_pack_roundtrip_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let projected_stride_values = DS41RT_CUDA_MLA_FP8_DS_PROJECTED_VALUES;
        let projected_stride_bytes = projected_stride_values * std::mem::size_of::<u16>();
        let packed_stride_bytes = DS41RT_CUDA_MLA_FP8_DS_PACKED_BYTES;
        let mut projected_f32 = Vec::with_capacity(rows * projected_stride_values);
        for row in 0..rows {
            projected_f32.extend(
                (0..DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES)
                    .map(|index| ((row * 19 + index % 97) as f32 - 48.0) / 128.0),
            );
            projected_f32.extend(
                (0..DS41RT_CUDA_MLA_FP8_DS_ROPE_VALUES)
                    .map(|index| ((row * 13 + index % 29) as f32 - 14.0) / 64.0),
            );
        }
        let projected = bf16_values(&projected_f32);
        let mut packed_bytes = vec![0_u8; rows * packed_stride_bytes];
        let mut unpacked_bytes = vec![0_u8; rows * projected_stride_bytes];
        let zero_packed = packed_bytes.clone();
        let zero_unpacked = unpacked_bytes.clone();

        let mut projected_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(projected.as_slice()))?;
        let mut packed_buffer = library.alloc_device_buffer(packed_bytes.len())?;
        let mut unpacked_buffer = library.alloc_device_buffer(unpacked_bytes.len())?;
        library.copy_h2d(projected_buffer, u16_bytes(&projected))?;
        library.copy_h2d(packed_buffer, &zero_packed)?;
        library.copy_h2d(unpacked_buffer, &zero_unpacked)?;

        let assert_roundtrip = |packed_bytes: &[u8], unpacked_bytes: &[u8]| {
            let unpacked = bytes_to_u16_vec(unpacked_bytes);
            let scale_offset = DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES;
            let rope_offset = scale_offset + DS41RT_CUDA_MLA_FP8_DS_SCALE_BYTES;
            for row in 0..rows {
                let projected_row = row * projected_stride_values;
                let packed_row = row * packed_stride_bytes;
                assert!(packed_bytes[packed_row + scale_offset
                    ..packed_row + scale_offset + DS41RT_CUDA_MLA_FP8_DS_SCALE_BYTES]
                    .iter()
                    .any(|byte| *byte != 0));
                assert_eq!(
                    &packed_bytes[packed_row + rope_offset
                        ..packed_row + rope_offset + DS41RT_CUDA_MLA_FP8_DS_ROPE_VALUES * 2],
                    u16_bytes(
                        &projected[projected_row + DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES
                            ..projected_row + projected_stride_values]
                    )
                );
                assert_eq!(
                    &unpacked[projected_row + DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES
                        ..projected_row + projected_stride_values],
                    &projected[projected_row + DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES
                        ..projected_row + projected_stride_values]
                );
                for col in 0..DS41RT_CUDA_MLA_FP8_DS_NOPE_VALUES {
                    let expected = bf16_to_f32(projected[projected_row + col]);
                    let actual = bf16_to_f32(unpacked[projected_row + col]);
                    assert!(
                        (actual - expected).abs() <= 0.025,
                        "row={row} col={col} expected={expected} actual={actual}"
                    );
                }
            }
        };

        let pack_result = library.cuda_mla_kv_pack_fp8_ds_mla(
            projected_buffer,
            packed_buffer,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        );
        match pack_result {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.cuda_mla_kv_unpack_fp8_ds_mla(
                    packed_buffer,
                    unpacked_buffer,
                    rows,
                    packed_stride_bytes,
                    projected_stride_bytes,
                )?;
                library.copy_d2h(&mut packed_bytes, packed_buffer)?;
                library.copy_d2h(&mut unpacked_bytes, unpacked_buffer)?;
                assert_roundtrip(&packed_bytes, &unpacked_bytes);

                library.copy_h2d(packed_buffer, &zero_packed)?;
                library.copy_h2d(unpacked_buffer, &zero_unpacked)?;
                unsafe {
                    library.cuda_mla_kv_pack_fp8_ds_mla_async(
                        projected_buffer,
                        packed_buffer,
                        rows,
                        projected_stride_bytes,
                        packed_stride_bytes,
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_mla_kv_unpack_fp8_ds_mla_async(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut packed_bytes, packed_buffer)?;
                library.copy_d2h(&mut unpacked_bytes, unpacked_buffer)?;
                assert_roundtrip(&packed_bytes, &unpacked_bytes);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let unpack_err = library
                    .cuda_mla_kv_unpack_fp8_ds_mla(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(unpack_err);
                let async_pack_err = unsafe {
                    library.cuda_mla_kv_pack_fp8_ds_mla_async(
                        projected_buffer,
                        packed_buffer,
                        rows,
                        projected_stride_bytes,
                        packed_stride_bytes,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_pack_err);
                let async_unpack_err = unsafe {
                    library.cuda_mla_kv_unpack_fp8_ds_mla_async(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_unpack_err);
            }
        }

        library.free_device_buffer(&mut unpacked_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut projected_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_mla_kv_mxfp4_ds_pack_roundtrip_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let projected_stride_values = DS41RT_CUDA_MLA_MXFP4_DS_PROJECTED_VALUES;
        let projected_stride_bytes = projected_stride_values * std::mem::size_of::<u16>();
        let packed_stride_bytes = DS41RT_CUDA_MLA_MXFP4_DS_PACKED_BYTES;
        let codebook = [
            0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        let mut projected_f32 = Vec::with_capacity(rows * projected_stride_values);
        for row in 0..rows {
            projected_f32.extend(
                (0..DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES)
                    .map(|index| codebook[(row + index) % codebook.len()]),
            );
            projected_f32.extend(
                (0..DS41RT_CUDA_MLA_MXFP4_DS_ROPE_VALUES)
                    .map(|index| ((row * 13 + index % 29) as f32 - 14.0) / 64.0),
            );
        }
        let projected = bf16_values(&projected_f32);
        let mut packed_bytes = vec![0_u8; rows * packed_stride_bytes];
        let mut unpacked_bytes = vec![0_u8; rows * projected_stride_bytes];
        let zero_packed = packed_bytes.clone();
        let zero_unpacked = unpacked_bytes.clone();

        let mut projected_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(projected.as_slice()))?;
        let mut packed_buffer = library.alloc_device_buffer(packed_bytes.len())?;
        let mut unpacked_buffer = library.alloc_device_buffer(unpacked_bytes.len())?;
        library.copy_h2d(projected_buffer, u16_bytes(&projected))?;
        library.copy_h2d(packed_buffer, &zero_packed)?;
        library.copy_h2d(unpacked_buffer, &zero_unpacked)?;

        let assert_roundtrip = |packed_bytes: &[u8], unpacked_bytes: &[u8]| {
            let unpacked = bytes_to_u16_vec(unpacked_bytes);
            let scale_offset = DS41RT_CUDA_MLA_MXFP4_DS_CODE_BYTES;
            let padding_offset = scale_offset + DS41RT_CUDA_MLA_MXFP4_DS_SCALE_BYTES;
            let rope_offset = padding_offset + DS41RT_CUDA_MLA_MXFP4_DS_PADDING_BYTES;
            for row in 0..rows {
                let projected_row = row * projected_stride_values;
                let packed_row = row * packed_stride_bytes;
                assert!(
                    packed_bytes[packed_row..packed_row + DS41RT_CUDA_MLA_MXFP4_DS_CODE_BYTES]
                        .iter()
                        .any(|byte| *byte != 0)
                );
                assert!(
                    packed_bytes[packed_row + scale_offset..packed_row + padding_offset]
                        .iter()
                        .all(|byte| *byte == 0x38)
                );
                assert!(
                    packed_bytes[packed_row + padding_offset..packed_row + rope_offset]
                        .iter()
                        .all(|byte| *byte == 0)
                );
                assert_eq!(
                    &packed_bytes[packed_row + rope_offset
                        ..packed_row + rope_offset + DS41RT_CUDA_MLA_MXFP4_DS_ROPE_VALUES * 2],
                    u16_bytes(
                        &projected[projected_row + DS41RT_CUDA_MLA_MXFP4_DS_NOPE_VALUES
                            ..projected_row + projected_stride_values]
                    )
                );
                assert_eq!(
                    &unpacked[projected_row..projected_row + projected_stride_values],
                    &projected[projected_row..projected_row + projected_stride_values]
                );
            }
        };

        let pack_result = library.cuda_mla_kv_pack_mxfp4_ds_mla(
            projected_buffer,
            packed_buffer,
            rows,
            projected_stride_bytes,
            packed_stride_bytes,
        );
        match pack_result {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.cuda_mla_kv_unpack_mxfp4_ds_mla(
                    packed_buffer,
                    unpacked_buffer,
                    rows,
                    packed_stride_bytes,
                    projected_stride_bytes,
                )?;
                library.copy_d2h(&mut packed_bytes, packed_buffer)?;
                library.copy_d2h(&mut unpacked_bytes, unpacked_buffer)?;
                assert_roundtrip(&packed_bytes, &unpacked_bytes);

                library.copy_h2d(packed_buffer, &zero_packed)?;
                library.copy_h2d(unpacked_buffer, &zero_unpacked)?;
                unsafe {
                    library.cuda_mla_kv_pack_mxfp4_ds_mla_async(
                        projected_buffer,
                        packed_buffer,
                        rows,
                        projected_stride_bytes,
                        packed_stride_bytes,
                        std::ptr::null_mut(),
                    )?;
                    library.cuda_mla_kv_unpack_mxfp4_ds_mla_async(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut packed_bytes, packed_buffer)?;
                library.copy_d2h(&mut unpacked_bytes, unpacked_buffer)?;
                assert_roundtrip(&packed_bytes, &unpacked_bytes);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let unpack_err = library
                    .cuda_mla_kv_unpack_mxfp4_ds_mla(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(unpack_err);
                let async_pack_err = unsafe {
                    library.cuda_mla_kv_pack_mxfp4_ds_mla_async(
                        projected_buffer,
                        packed_buffer,
                        rows,
                        projected_stride_bytes,
                        packed_stride_bytes,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_pack_err);
                let async_unpack_err = unsafe {
                    library.cuda_mla_kv_unpack_mxfp4_ds_mla_async(
                        packed_buffer,
                        unpacked_buffer,
                        rows,
                        packed_stride_bytes,
                        projected_stride_bytes,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_unpack_err);
            }
        }

        library.free_device_buffer(&mut unpacked_buffer)?;
        library.free_device_buffer(&mut packed_buffer)?;
        library.free_device_buffer(&mut projected_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_router_topk_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden_dim = 3;
        let experts = 4;
        let top_k = 2;
        let hidden = [1.0_f32, -0.5, 0.25, -0.25, 0.75, 1.0];
        let router_weight = [
            0.2_f32, -0.1, 0.5, 0.0, 0.3, -0.4, 0.6, -0.2, 0.1, -0.3, 0.4, 0.2,
        ];
        let correction_bias = [0.01_f32, -0.02, 0.03, 0.0];
        let expected = router_topk_expected(
            &hidden,
            &router_weight,
            &correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
        let mut index_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<f32>()];
        let mut weight_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<f32>()];

        let mut hidden_buffer = library.alloc_device_buffer(std::mem::size_of_val(&hidden))?;
        let mut router_weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&router_weight))?;
        let mut correction_bias_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&correction_bias))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;
        let mut weight_buffer = library.alloc_device_buffer(weight_bytes.len())?;

        library.copy_h2d(hidden_buffer, f32_bytes(&hidden))?;
        library.copy_h2d(router_weight_buffer, f32_bytes(&router_weight))?;
        library.copy_h2d(correction_bias_buffer, f32_bytes(&correction_bias))?;

        match library.cuda_router_topk_f32(
            hidden_buffer,
            router_weight_buffer,
            correction_bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer,
            rows,
            hidden_dim,
            experts,
            top_k,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                library.copy_d2h(&mut weight_bytes, weight_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                let scores = bytes_to_f32_vec(&score_bytes);
                let weights = bytes_to_f32_vec(&weight_bytes);
                for (actual, expected) in scores.iter().zip(expected.scores.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
                for (actual, expected) in weights.iter().zip(expected.weights.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_router_topk_f32_async(
                        hidden_buffer,
                        router_weight_buffer,
                        correction_bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_router_topk_f32_async(
                        hidden_buffer,
                        router_weight_buffer,
                        correction_bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut correction_bias_buffer)?;
        library.free_device_buffer(&mut router_weight_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_router_topk_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden_dim = 3;
        let experts = 4;
        let top_k = 2;
        let hidden_f32 = [1.0_f32, -0.5, 0.25, -0.25, 0.75, 1.0];
        let router_weight_f32 = [
            0.2_f32, -0.1, 0.5, 0.0, 0.3, -0.4, 0.6, -0.2, 0.1, -0.3, 0.4, 0.2,
        ];
        let correction_bias = [0.01_f32, -0.02, 0.03, 0.0];
        let hidden = bf16_values(&hidden_f32);
        let router_weight = bf16_values(&router_weight_f32);
        let hidden_expected: Vec<f32> = hidden.iter().copied().map(bf16_to_f32).collect();
        let router_weight_expected: Vec<f32> =
            router_weight.iter().copied().map(bf16_to_f32).collect();
        let expected = router_topk_expected(
            &hidden_expected,
            &router_weight_expected,
            &correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
        let mut index_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<f32>()];
        let mut weight_bytes = vec![0_u8; rows * top_k * std::mem::size_of::<f32>()];

        let mut hidden_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(hidden.as_slice()))?;
        let mut router_weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(router_weight.as_slice()))?;
        let mut correction_bias_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&correction_bias))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;
        let mut weight_buffer = library.alloc_device_buffer(weight_bytes.len())?;

        library.copy_h2d(hidden_buffer, u16_bytes(&hidden))?;
        library.copy_h2d(router_weight_buffer, u16_bytes(&router_weight))?;
        library.copy_h2d(correction_bias_buffer, f32_bytes(&correction_bias))?;

        match library.cuda_router_topk_bf16(
            hidden_buffer,
            router_weight_buffer,
            correction_bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer,
            rows,
            hidden_dim,
            experts,
            top_k,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                library.copy_d2h(&mut weight_bytes, weight_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                let scores = bytes_to_f32_vec(&score_bytes);
                let weights = bytes_to_f32_vec(&weight_bytes);
                for (actual, expected) in scores.iter().zip(expected.scores.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
                for (actual, expected) in weights.iter().zip(expected.weights.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_router_topk_bf16_async(
                        hidden_buffer,
                        router_weight_buffer,
                        correction_bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_router_topk_bf16_async(
                        hidden_buffer,
                        router_weight_buffer,
                        correction_bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut correction_bias_buffer)?;
        library.free_device_buffer(&mut router_weight_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_linear_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let input_dim = 3;
        let output_dim = 4;
        let input = [0.5_f32, -1.0, 2.0, -0.25, 0.75, 1.5];
        let weight = [
            0.2_f32, -0.1, 0.5, 0.0, 0.3, -0.4, 0.6, -0.2, 0.1, -0.3, 0.4, 0.2,
        ];
        let bias = [0.05_f32, -0.10, 0.15, 0.20];
        let expected_bias =
            linear_expected(&input, &weight, Some(&bias), rows, input_dim, output_dim);
        let expected_no_bias = linear_expected(&input, &weight, None, rows, input_dim, output_dim);
        let mut output_bytes = vec![0_u8; rows * output_dim * std::mem::size_of::<f32>()];

        let mut input_buffer = library.alloc_device_buffer(std::mem::size_of_val(&input))?;
        let mut weight_buffer = library.alloc_device_buffer(std::mem::size_of_val(&weight))?;
        let mut bias_buffer = library.alloc_device_buffer(std::mem::size_of_val(&bias))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(input_buffer, f32_bytes(&input))?;
        library.copy_h2d(weight_buffer, f32_bytes(&weight))?;
        library.copy_h2d(bias_buffer, f32_bytes(&bias))?;

        match library.cuda_linear_f32(
            input_buffer,
            weight_buffer,
            Some(bias_buffer),
            output_buffer,
            rows,
            input_dim,
            output_dim,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                library.cuda_linear_f32(
                    input_buffer,
                    weight_buffer,
                    None,
                    output_buffer,
                    rows,
                    input_dim,
                    output_dim,
                )?;
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_no_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_linear_f32_async(
                        input_buffer,
                        weight_buffer,
                        Some(bias_buffer),
                        output_buffer,
                        rows,
                        input_dim,
                        output_dim,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_linear_f32_async(
                        input_buffer,
                        weight_buffer,
                        Some(bias_buffer),
                        output_buffer,
                        rows,
                        input_dim,
                        output_dim,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut bias_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_linear_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let input_dim = 3;
        let output_dim = 4;
        let input_f32 = [0.5_f32, -1.0, 2.0, -0.25, 0.75, 1.5];
        let weight_f32 = [
            0.2_f32, -0.1, 0.5, 0.0, 0.3, -0.4, 0.6, -0.2, 0.1, -0.3, 0.4, 0.2,
        ];
        let bias_f32 = [0.05_f32, -0.10, 0.15, 0.20];
        let input = bf16_values(&input_f32);
        let weight = bf16_values(&weight_f32);
        let bias = bf16_values(&bias_f32);
        let input_expected = input
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let weight_expected = weight
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let bias_expected = bias
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let expected_bias = linear_expected(
            &input_expected,
            &weight_expected,
            Some(&bias_expected),
            rows,
            input_dim,
            output_dim,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let expected_no_bias = linear_expected(
            &input_expected,
            &weight_expected,
            None,
            rows,
            input_dim,
            output_dim,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let mut output_bytes = vec![0_u8; rows * output_dim * std::mem::size_of::<u16>()];

        let mut input_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input.as_slice()))?;
        let mut weight_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(weight.as_slice()))?;
        let mut bias_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(bias.as_slice()))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(input_buffer, u16_bytes(&input))?;
        library.copy_h2d(weight_buffer, u16_bytes(&weight))?;
        library.copy_h2d(bias_buffer, u16_bytes(&bias))?;

        match library.cuda_linear_bf16(
            input_buffer,
            weight_buffer,
            Some(bias_buffer),
            output_buffer,
            rows,
            input_dim,
            output_dim,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                library.cuda_linear_bf16(
                    input_buffer,
                    weight_buffer,
                    None,
                    output_buffer,
                    rows,
                    input_dim,
                    output_dim,
                )?;
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_no_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_linear_bf16_async(
                        input_buffer,
                        weight_buffer,
                        Some(bias_buffer),
                        output_buffer,
                        rows,
                        input_dim,
                        output_dim,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected_bias.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_linear_bf16_async(
                        input_buffer,
                        weight_buffer,
                        Some(bias_buffer),
                        output_buffer,
                        rows,
                        input_dim,
                        output_dim,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut bias_buffer)?;
        library.free_device_buffer(&mut weight_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_causal_attention_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let heads = 2;
        let qk_dim = 2;
        let v_dim = 3;
        let scale = 0.5_f32;
        let q = [
            0.1_f32, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.2, 0.9, 0.3, -0.4,
        ];
        let k = [
            0.2_f32, -0.1, 0.4, 0.3, -0.5, 0.6, 0.7, -0.8, 0.1, 0.5, -0.2, 0.9,
        ];
        let v = [
            0.1_f32, 0.2, 0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 1.0, -1.1, 1.2, -0.2, 0.4, -0.6,
            0.3, -0.7, 0.5,
        ];
        let expected = causal_attention_expected(&q, &k, &v, rows, heads, qk_dim, v_dim, scale);
        let mut output_bytes = vec![0_u8; rows * heads * v_dim * std::mem::size_of::<f32>()];

        let mut q_buffer = library.alloc_device_buffer(std::mem::size_of_val(&q))?;
        let mut k_buffer = library.alloc_device_buffer(std::mem::size_of_val(&k))?;
        let mut v_buffer = library.alloc_device_buffer(std::mem::size_of_val(&v))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(q_buffer, f32_bytes(&q))?;
        library.copy_h2d(k_buffer, f32_bytes(&k))?;
        library.copy_h2d(v_buffer, f32_bytes(&v))?;

        match library.cuda_causal_attention_f32(
            q_buffer,
            k_buffer,
            v_buffer,
            output_buffer,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }

                unsafe {
                    library.cuda_causal_attention_f32_async(
                        q_buffer,
                        k_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        qk_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-6);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_causal_attention_f32_async(
                        q_buffer,
                        k_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        qk_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut v_buffer)?;
        library.free_device_buffer(&mut k_buffer)?;
        library.free_device_buffer(&mut q_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_causal_attention_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let heads = 2;
        let qk_dim = 2;
        let v_dim = 3;
        let scale = 0.5_f32;
        let q_f32 = [
            0.1_f32, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.2, 0.9, 0.3, -0.4,
        ];
        let k_f32 = [
            0.2_f32, -0.1, 0.4, 0.3, -0.5, 0.6, 0.7, -0.8, 0.1, 0.5, -0.2, 0.9,
        ];
        let v_f32 = [
            0.1_f32, 0.2, 0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 1.0, -1.1, 1.2, -0.2, 0.4, -0.6,
            0.3, -0.7, 0.5,
        ];
        let q = bf16_values(&q_f32);
        let k = bf16_values(&k_f32);
        let v = bf16_values(&v_f32);
        let q_expected = q
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let k_expected = k
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let v_expected = v
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let expected = causal_attention_expected(
            &q_expected,
            &k_expected,
            &v_expected,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        )
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16(value)))
        .collect::<Vec<_>>();
        let mut output_bytes = vec![0_u8; rows * heads * v_dim * std::mem::size_of::<u16>()];

        let mut q_buffer = library.alloc_device_buffer(std::mem::size_of_val(q.as_slice()))?;
        let mut k_buffer = library.alloc_device_buffer(std::mem::size_of_val(k.as_slice()))?;
        let mut v_buffer = library.alloc_device_buffer(std::mem::size_of_val(v.as_slice()))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(q_buffer, u16_bytes(&q))?;
        library.copy_h2d(k_buffer, u16_bytes(&k))?;
        library.copy_h2d(v_buffer, u16_bytes(&v))?;

        match library.cuda_causal_attention_bf16(
            q_buffer,
            k_buffer,
            v_buffer,
            output_buffer,
            rows,
            heads,
            qk_dim,
            v_dim,
            scale,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_causal_attention_bf16_async(
                        q_buffer,
                        k_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        qk_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_causal_attention_bf16_async(
                        q_buffer,
                        k_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        qk_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut v_buffer)?;
        library.free_device_buffer(&mut k_buffer)?;
        library.free_device_buffer(&mut q_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_rope_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let heads = 2;
        let rotary_dim = 4;
        let theta = 10_000.0_f32;
        let input = [
            -0.20_f32, 0.45, 0.70, -0.10, 0.15, -0.35, 0.25, 0.05, 0.30, -0.15, 0.40, 0.25, -0.55,
            0.20, -0.10, 0.65, 0.55, 0.20, -0.35, 0.60, -0.40, 0.30, 0.20, -0.55,
        ];
        let positions = [0_u32, 1, 2];
        let expected = rope_expected(&input, &positions, rows, heads, rotary_dim, theta);
        let mut output_bytes = vec![0_u8; input.len() * std::mem::size_of::<f32>()];

        let mut input_buffer = library.alloc_device_buffer(std::mem::size_of_val(&input))?;
        let mut position_buffer = library.alloc_device_buffer(std::mem::size_of_val(&positions))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(input_buffer, f32_bytes(&input))?;
        library.copy_h2d(position_buffer, u32_bytes(&positions))?;

        match library.cuda_rope_f32(
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_rope_f32_async(
                        input_buffer,
                        position_buffer,
                        output_buffer,
                        rows,
                        heads,
                        rotary_dim,
                        theta,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_rope_f32_async(
                        input_buffer,
                        position_buffer,
                        output_buffer,
                        rows,
                        heads,
                        rotary_dim,
                        theta,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut position_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_dspark_prompt_metadata_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3usize;
        let absolute_position_start = 127usize;
        let request_slot = 2usize;
        let theta = 10_000.0_f32;
        let mut position_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut slot_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut cos_sin_bytes = vec![0_u8; rows * 64 * std::mem::size_of::<f32>()];
        let mut position_buffer = library.alloc_device_buffer(position_bytes.len())?;
        let mut slot_buffer = library.alloc_device_buffer(slot_bytes.len())?;
        let mut cos_sin_buffer = library.alloc_device_buffer(cos_sin_bytes.len())?;

        let launched = unsafe {
            library.cuda_ds4_dspark_prompt_metadata_async(
                position_buffer,
                slot_buffer,
                cos_sin_buffer,
                absolute_position_start,
                request_slot,
                rows,
                theta,
                std::ptr::null_mut(),
            )
        };
        match launched {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut position_bytes, position_buffer)?;
                library.copy_d2h(&mut slot_bytes, slot_buffer)?;
                library.copy_d2h(&mut cos_sin_bytes, cos_sin_buffer)?;
                assert_eq!(bytes_to_u32_vec(&position_bytes), vec![0, 1, 2]);
                assert_eq!(bytes_to_u32_vec(&slot_bytes), vec![639, 512, 513]);
                let cos_sin = bytes_to_f32_vec(&cos_sin_bytes);
                for row in 0..rows {
                    for pair in 0..32 {
                        let angle = (absolute_position_start + row) as f32
                            * theta.powf(-2.0 * pair as f32 / 64.0);
                        assert!((cos_sin[row * 64 + pair] - angle.cos()).abs() < 2.0e-5);
                        assert!((cos_sin[row * 64 + 32 + pair] - angle.sin()).abs() < 2.0e-5);
                    }
                }
            }
            Err(error) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(error);
            }
        }

        library.free_device_buffer(&mut cos_sin_buffer)?;
        library.free_device_buffer(&mut slot_buffer)?;
        library.free_device_buffer(&mut position_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_dspark_proposal_preparation_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        const PROPOSAL_ROWS: usize = 5;
        const HC_MULT: usize = 4;
        const SELECTION_WIDTH: usize = 133;
        let vocab_size = 6usize;
        let hidden_size = 4usize;
        let anchor_token_id = 2usize;
        let noise_token_id = 4usize;
        let request_slot = 2usize;
        let cache_window_start = 2usize;
        let main_context_end = 130usize;
        let theta = 10_000.0_f32;
        let embedding_f32 = (0..vocab_size * hidden_size)
            .map(|index| index as f32 / 8.0 - 1.0)
            .collect::<Vec<_>>();
        let embedding = bf16_values(&embedding_f32);
        let mut token_bytes = vec![0_u8; PROPOSAL_ROWS * std::mem::size_of::<u32>()];
        let mut residual_bytes =
            vec![0_u8; PROPOSAL_ROWS * HC_MULT * hidden_size * std::mem::size_of::<u16>()];
        let mut position_bytes = vec![0_u8; PROPOSAL_ROWS * std::mem::size_of::<u32>()];
        let mut slot_bytes = vec![0_u8; PROPOSAL_ROWS * std::mem::size_of::<u32>()];
        let mut cos_sin_bytes = vec![0_u8; PROPOSAL_ROWS * 64 * std::mem::size_of::<f32>()];
        let mut selection_bytes =
            vec![0_u8; PROPOSAL_ROWS * SELECTION_WIDTH * std::mem::size_of::<i32>()];
        let mut length_bytes = vec![0_u8; PROPOSAL_ROWS * std::mem::size_of::<i32>()];

        let mut embedding_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(embedding.as_slice()))?;
        let mut token_buffer = library.alloc_device_buffer(token_bytes.len())?;
        let mut residual_buffer = library.alloc_device_buffer(residual_bytes.len())?;
        let mut position_buffer = library.alloc_device_buffer(position_bytes.len())?;
        let mut slot_buffer = library.alloc_device_buffer(slot_bytes.len())?;
        let mut cos_sin_buffer = library.alloc_device_buffer(cos_sin_bytes.len())?;
        let mut selection_buffer = library.alloc_device_buffer(selection_bytes.len())?;
        let mut length_buffer = library.alloc_device_buffer(length_bytes.len())?;
        library.copy_h2d(embedding_buffer, u16_bytes(&embedding))?;

        let launched = unsafe {
            library.cuda_ds4_dspark_prepare_proposal_async(
                embedding_buffer,
                token_buffer,
                residual_buffer,
                position_buffer,
                slot_buffer,
                cos_sin_buffer,
                selection_buffer,
                length_buffer,
                anchor_token_id,
                noise_token_id,
                request_slot,
                cache_window_start,
                main_context_end,
                vocab_size,
                hidden_size,
                theta,
                std::ptr::null_mut(),
            )
        };
        match launched {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut token_bytes, token_buffer)?;
                library.copy_d2h(&mut residual_bytes, residual_buffer)?;
                library.copy_d2h(&mut position_bytes, position_buffer)?;
                library.copy_d2h(&mut slot_bytes, slot_buffer)?;
                library.copy_d2h(&mut cos_sin_bytes, cos_sin_buffer)?;
                library.copy_d2h(&mut selection_bytes, selection_buffer)?;
                library.copy_d2h(&mut length_bytes, length_buffer)?;

                assert_eq!(
                    bytes_to_u32_vec(&token_bytes),
                    vec![
                        anchor_token_id as u32,
                        noise_token_id as u32,
                        noise_token_id as u32,
                        noise_token_id as u32,
                        noise_token_id as u32,
                    ]
                );
                assert_eq!(bytes_to_u32_vec(&position_bytes), vec![0, 1, 2, 3, 4]);
                assert_eq!(bytes_to_u32_vec(&slot_bytes), vec![640, 641, 642, 643, 644]);
                let residual = bytes_to_u16_vec(&residual_bytes);
                for row in 0..PROPOSAL_ROWS {
                    let token = if row == 0 {
                        anchor_token_id
                    } else {
                        noise_token_id
                    };
                    for lane in 0..HC_MULT {
                        let start = (row * HC_MULT + lane) * hidden_size;
                        assert_eq!(
                            &residual[start..start + hidden_size],
                            &embedding[token * hidden_size..(token + 1) * hidden_size]
                        );
                    }
                }
                let selection = bytes_to_i32_vec(&selection_bytes);
                let expected = (512_i32..645).collect::<Vec<_>>();
                for row in 0..PROPOSAL_ROWS {
                    let start = row * SELECTION_WIDTH;
                    assert_eq!(&selection[start..start + SELECTION_WIDTH], &expected);
                }
                assert_eq!(bytes_to_i32_vec(&length_bytes), vec![133; PROPOSAL_ROWS]);
                let cos_sin = bytes_to_f32_vec(&cos_sin_bytes);
                for row in 0..PROPOSAL_ROWS {
                    for pair in 0..32 {
                        let angle =
                            (main_context_end + row) as f32 * theta.powf(-2.0 * pair as f32 / 64.0);
                        assert!((cos_sin[row * 64 + pair] - angle.cos()).abs() < 2.0e-5);
                        assert!((cos_sin[row * 64 + 32 + pair] - angle.sin()).abs() < 2.0e-5);
                    }
                }

                unsafe {
                    library.cuda_ds4_dspark_prepare_proposal_async(
                        embedding_buffer,
                        token_buffer,
                        residual_buffer,
                        position_buffer,
                        slot_buffer,
                        cos_sin_buffer,
                        selection_buffer,
                        length_buffer,
                        anchor_token_id,
                        noise_token_id,
                        request_slot,
                        120,
                        140,
                        vocab_size,
                        hidden_size,
                        theta,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut selection_bytes, selection_buffer)?;
                library.copy_d2h(&mut length_bytes, length_buffer)?;
                let selection = bytes_to_i32_vec(&selection_bytes);
                let expected_wrapped = (512_i32..524).chain(632_i32..645).collect::<Vec<_>>();
                assert_eq!(expected_wrapped.len(), 25);
                for row in 0..PROPOSAL_ROWS {
                    let start = row * SELECTION_WIDTH;
                    assert_eq!(&selection[start..start + 25], &expected_wrapped);
                    assert!(selection[start + 25..start + SELECTION_WIDTH]
                        .iter()
                        .all(|slot| *slot == -1));
                }
                assert_eq!(bytes_to_i32_vec(&length_bytes), vec![25; PROPOSAL_ROWS]);
            }
            Err(error) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(error);
            }
        }

        library.free_device_buffer(&mut length_buffer)?;
        library.free_device_buffer(&mut selection_buffer)?;
        library.free_device_buffer(&mut cos_sin_buffer)?;
        library.free_device_buffer(&mut slot_buffer)?;
        library.free_device_buffer(&mut position_buffer)?;
        library.free_device_buffer(&mut residual_buffer)?;
        library.free_device_buffer(&mut token_buffer)?;
        library.free_device_buffer(&mut embedding_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_rope_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let heads = 2;
        let rotary_dim = 4;
        let theta = 10_000.0_f32;
        let input_f32 = [
            -0.20_f32, 0.45, 0.70, -0.10, 0.15, -0.35, 0.25, 0.05, 0.30, -0.15, 0.40, 0.25, -0.55,
            0.20, -0.10, 0.65, 0.55, 0.20, -0.35, 0.60, -0.40, 0.30, 0.20, -0.55,
        ];
        let input = bf16_values(&input_f32);
        let input_expected = input
            .iter()
            .map(|value| bf16_to_f32(*value))
            .collect::<Vec<_>>();
        let positions = [0_u32, 1, 2];
        let expected = rope_expected(&input_expected, &positions, rows, heads, rotary_dim, theta)
            .into_iter()
            .map(|value| bf16_to_f32(f32_to_bf16(value)))
            .collect::<Vec<_>>();
        let mut output_bytes = vec![0_u8; input.len() * std::mem::size_of::<u16>()];

        let mut input_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(input.as_slice()))?;
        let mut position_buffer = library.alloc_device_buffer(std::mem::size_of_val(&positions))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(input_buffer, u16_bytes(&input))?;
        library.copy_h2d(position_buffer, u32_bytes(&positions))?;

        match library.cuda_rope_bf16(
            input_buffer,
            position_buffer,
            output_buffer,
            rows,
            heads,
            rotary_dim,
            theta,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_rope_bf16_async(
                        input_buffer,
                        position_buffer,
                        output_buffer,
                        rows,
                        heads,
                        rotary_dim,
                        theta,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_rope_bf16_async(
                        input_buffer,
                        position_buffer,
                        output_buffer,
                        rows,
                        heads,
                        rotary_dim,
                        theta,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut position_buffer)?;
        library.free_device_buffer(&mut input_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_mla_rope_attention_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let heads = 2;
        let nope_dim = 2;
        let rope_dim = 4;
        let v_dim = 3;
        let theta = 10_000.0_f32;
        let scale = 1.0_f32 / ((nope_dim + rope_dim) as f32).sqrt();
        let positions = [0_u32, 1, 2];
        let q_nope_f32 = [
            0.10_f32, -0.20, 0.30, 0.05, -0.40, 0.25, 0.15, -0.35, 0.50, 0.10, -0.25, 0.45,
        ];
        let q_rope_f32 = [
            -0.20_f32, 0.45, 0.70, -0.10, 0.15, -0.35, 0.25, 0.05, 0.30, -0.15, 0.40, 0.25, -0.55,
            0.20, -0.10, 0.65, 0.55, 0.20, -0.35, 0.60, -0.40, 0.30, 0.20, -0.55,
        ];
        let k_nope_f32 = [
            0.25_f32, 0.15, -0.10, 0.40, 0.35, -0.45, 0.60, 0.20, -0.30, 0.50, 0.45, -0.15,
        ];
        let k_rope_f32 = [
            0.10_f32, 0.50, -0.20, 0.30, 0.35, -0.25, 0.45, 0.15, -0.40, 0.30, 0.20, -0.55,
        ];
        let v_f32 = [
            0.10_f32, 0.20, 0.30, -0.40, -0.50, -0.60, 0.70, 0.80, 0.90, 1.00, -1.10, 1.20, -0.20,
            0.40, -0.60, 0.30, -0.70, 0.50,
        ];

        let q_nope = bf16_values(&q_nope_f32);
        let q_rope_unrotated = bf16_values(&q_rope_f32);
        let k_nope = bf16_values(&k_nope_f32);
        let k_rope_unrotated = bf16_values(&k_rope_f32);
        let v = bf16_values(&v_f32);
        let q_rope_quantized: Vec<f32> =
            q_rope_unrotated.iter().copied().map(bf16_to_f32).collect();
        let k_rope_quantized: Vec<f32> =
            k_rope_unrotated.iter().copied().map(bf16_to_f32).collect();
        let q_rope_rotated_f32 =
            rope_expected(&q_rope_quantized, &positions, rows, heads, rope_dim, theta);
        let k_rope_rotated_f32 =
            rope_expected(&k_rope_quantized, &positions, rows, 1, rope_dim, theta);
        let q_rope = bf16_values(&q_rope_rotated_f32);
        let k_rope = bf16_values(&k_rope_rotated_f32);
        let expected_raw = mla_rope_attention_expected(
            &q_nope.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &q_rope.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &k_nope.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &k_rope.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            &v.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(),
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        );
        let expected: Vec<f32> = expected_raw
            .into_iter()
            .map(|value| bf16_to_f32(f32_to_bf16(value)))
            .collect();
        let mut output_bytes = vec![0_u8; rows * heads * v_dim * std::mem::size_of::<u16>()];

        let mut q_nope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_nope.as_slice()))?;
        let mut q_rope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(q_rope.as_slice()))?;
        let mut k_nope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_nope.as_slice()))?;
        let mut k_rope_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(k_rope.as_slice()))?;
        let mut v_buffer = library.alloc_device_buffer(std::mem::size_of_val(v.as_slice()))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(q_nope_buffer, u16_bytes(&q_nope))?;
        library.copy_h2d(q_rope_buffer, u16_bytes(&q_rope))?;
        library.copy_h2d(k_nope_buffer, u16_bytes(&k_nope))?;
        library.copy_h2d(k_rope_buffer, u16_bytes(&k_rope))?;
        library.copy_h2d(v_buffer, u16_bytes(&v))?;

        match library.cuda_mla_rope_attention_bf16(
            q_nope_buffer,
            q_rope_buffer,
            k_nope_buffer,
            k_rope_buffer,
            v_buffer,
            output_buffer,
            rows,
            heads,
            nope_dim,
            rope_dim,
            v_dim,
            scale,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_mla_rope_attention_bf16_async(
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                let output = bytes_to_bf16_f32_vec(&output_bytes);
                for (actual, expected) in output.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_mla_rope_attention_bf16_async(
                        q_nope_buffer,
                        q_rope_buffer,
                        k_nope_buffer,
                        k_rope_buffer,
                        v_buffer,
                        output_buffer,
                        rows,
                        heads,
                        nope_dim,
                        rope_dim,
                        v_dim,
                        scale,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut v_buffer)?;
        library.free_device_buffer(&mut k_rope_buffer)?;
        library.free_device_buffer(&mut k_nope_buffer)?;
        library.free_device_buffer(&mut q_rope_buffer)?;
        library.free_device_buffer(&mut q_nope_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_embedding_lookup_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let vocab = 5;
        let hidden = 4;
        let token_ids = [3_u32, 0, 4, 2];
        let embedding = [
            0.0_f32, 0.1, 0.2, 0.3, 1.0, -1.1, 1.2, -1.3, 2.0, 2.1, -2.2, -2.3, 3.0, -3.1, 3.2,
            -3.3, -4.0, 4.1, -4.2, 4.3,
        ];
        let expected = embedding_lookup_expected(&embedding, &token_ids, vocab, hidden);
        let rows = token_ids.len();
        let mut output_bytes = vec![0_u8; rows * hidden * std::mem::size_of::<f32>()];

        let mut embedding_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&embedding))?;
        let mut token_ids_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&token_ids))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(embedding_buffer, f32_bytes(&embedding))?;
        library.copy_h2d(token_ids_buffer, u32_bytes(&token_ids))?;

        match library.cuda_embedding_lookup_f32(
            embedding_buffer,
            token_ids_buffer,
            output_buffer,
            rows,
            vocab,
            hidden,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                assert_eq!(bytes_to_f32_vec(&output_bytes), expected);

                unsafe {
                    library.cuda_embedding_lookup_f32_async(
                        embedding_buffer,
                        token_ids_buffer,
                        output_buffer,
                        rows,
                        vocab,
                        hidden,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                assert_eq!(bytes_to_f32_vec(&output_bytes), expected);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_embedding_lookup_f32_async(
                        embedding_buffer,
                        token_ids_buffer,
                        output_buffer,
                        rows,
                        vocab,
                        hidden,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut token_ids_buffer)?;
        library.free_device_buffer(&mut embedding_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_embedding_lookup_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let vocab = 5;
        let hidden = 4;
        let token_ids = [3_u32, 0, 4, 2];
        let embedding_f32 = [
            0.0_f32, 0.1, 0.2, 0.3, 1.0, -1.1, 1.2, -1.3, 2.0, 2.1, -2.2, -2.3, 3.0, -3.1, 3.2,
            -3.3, -4.0, 4.1, -4.2, 4.3,
        ];
        let embedding = bf16_values(&embedding_f32);
        let mut expected = Vec::with_capacity(token_ids.len() * hidden);
        for token_id in token_ids {
            let start = token_id as usize * hidden;
            expected.extend_from_slice(&embedding[start..start + hidden]);
        }
        let rows = token_ids.len();
        let mut output_bytes = vec![0_u8; rows * hidden * std::mem::size_of::<u16>()];

        let mut embedding_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(embedding.as_slice()))?;
        let mut token_ids_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&token_ids))?;
        let mut output_buffer = library.alloc_device_buffer(output_bytes.len())?;

        library.copy_h2d(embedding_buffer, u16_bytes(&embedding))?;
        library.copy_h2d(token_ids_buffer, u32_bytes(&token_ids))?;

        match library.cuda_embedding_lookup_bf16(
            embedding_buffer,
            token_ids_buffer,
            output_buffer,
            rows,
            vocab,
            hidden,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                assert_eq!(
                    output_bytes
                        .chunks_exact(std::mem::size_of::<u16>())
                        .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                    expected
                );

                unsafe {
                    library.cuda_embedding_lookup_bf16_async(
                        embedding_buffer,
                        token_ids_buffer,
                        output_buffer,
                        rows,
                        vocab,
                        hidden,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut output_bytes, output_buffer)?;
                assert_eq!(
                    output_bytes
                        .chunks_exact(std::mem::size_of::<u16>())
                        .map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                    expected
                );
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_embedding_lookup_bf16_async(
                        embedding_buffer,
                        token_ids_buffer,
                        output_buffer,
                        rows,
                        vocab,
                        hidden,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut output_buffer)?;
        library.free_device_buffer(&mut token_ids_buffer)?;
        library.free_device_buffer(&mut embedding_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_logits_argmax_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let vocab = 6;
        let logits = [
            -0.5_f32, 0.1, 0.8, 0.0, 0.8, -0.2, -1.0, -0.7, -0.9, -0.8, -0.6, -0.4, 1.25, 1.0, 0.5,
            1.25, -2.0, 0.0,
        ];
        let expected = logits_argmax_expected(&logits, rows, vocab);
        let mut index_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * std::mem::size_of::<f32>()];

        let mut logits_buffer = library.alloc_device_buffer(std::mem::size_of_val(&logits))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;

        library.copy_h2d(logits_buffer, f32_bytes(&logits))?;

        match library.cuda_logits_argmax_f32(logits_buffer, index_buffer, score_buffer, rows, vocab)
        {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);

                unsafe {
                    library.cuda_logits_argmax_f32_async(
                        logits_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        vocab,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_logits_argmax_f32_async(
                        logits_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        vocab,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut logits_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_lm_head_argmax_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden_dim = 3;
        let vocab = 5;
        let hidden = bf16_values(&[0.5_f32, -1.0, 0.25, -0.5, 0.75, 1.25]);
        let lm_head = bf16_values(&[
            0.25_f32, -0.5, 0.75, -0.25, 0.5, 0.125, 1.0, 0.0, -0.5, 0.25, -0.5, 0.75, -1.0, 0.25,
            0.5,
        ]);
        let expected = lm_head_argmax_bf16_expected(&hidden, &lm_head, rows, hidden_dim, vocab);
        let mut index_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * std::mem::size_of::<f32>()];

        let mut hidden_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(hidden.as_slice()))?;
        let mut lm_head_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(lm_head.as_slice()))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;

        library.copy_h2d(hidden_buffer, u16_bytes(&hidden))?;
        library.copy_h2d(lm_head_buffer, u16_bytes(&lm_head))?;

        match library.cuda_lm_head_argmax_bf16(
            hidden_buffer,
            lm_head_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);

                unsafe {
                    library.cuda_lm_head_argmax_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_lm_head_argmax_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut lm_head_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_dspark_terminal_greedy_joint_batch_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        const PROPOSALS: usize = 5;
        let active = 2usize;
        let max_slots = 3usize;
        let hidden = 4usize;
        let vocab = 7usize;
        let rank = 3usize;
        let normalized_f32 = (0..max_slots * PROPOSALS * hidden)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let collapsed_f32 = (0..max_slots * PROPOSALS * hidden)
            .map(|index| ((index * 7 % 29) as f32 - 14.0) / 16.0)
            .collect::<Vec<_>>();
        let shared_head_f32 = (0..vocab * hidden)
            .map(|index| ((index * 5 % 23) as f32 - 11.0) / 16.0)
            .collect::<Vec<_>>();
        let markov_w1_f32 = (0..vocab * rank)
            .map(|index| ((index * 13 % 19) as f32 - 9.0) / 16.0)
            .collect::<Vec<_>>();
        let markov_w2_f32 = (0..vocab * rank)
            .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let confidence_f32 = (0..hidden + rank)
            .map(|index| ((index * 5 % 13) as f32 - 6.0) / 32.0)
            .collect::<Vec<_>>();
        let normalized = bf16_values(&normalized_f32);
        let collapsed = bf16_values(&collapsed_f32);
        let shared_head = bf16_values(&shared_head_f32);
        let markov_w1 = bf16_values(&markov_w1_f32);
        let markov_w2 = bf16_values(&markov_w2_f32);
        let confidence_weight = bf16_values(&confidence_f32);
        // Deliberately non-contiguous and reversed: the native ABI must batch
        // logical requests without assuming live request slots are compact.
        let active_slot_ids = [2_u32, 0];
        let anchor_token_ids = [2_u32, 5];
        let expected = dspark_terminal_greedy_bf16_expected(
            &normalized,
            &collapsed,
            &shared_head,
            &markov_w1,
            &markov_w2,
            &confidence_weight,
            &active_slot_ids,
            &anchor_token_ids,
            max_slots,
            hidden,
            vocab,
            rank,
        );
        let rows = active * PROPOSALS;
        let mut normalized_buffer = library.alloc_device_buffer(normalized.len() * 2)?;
        let mut collapsed_buffer = library.alloc_device_buffer(collapsed.len() * 2)?;
        let mut shared_head_buffer = library.alloc_device_buffer(shared_head.len() * 2)?;
        let mut markov_w1_buffer = library.alloc_device_buffer(markov_w1.len() * 2)?;
        let mut markov_w2_buffer = library.alloc_device_buffer(markov_w2.len() * 2)?;
        let mut confidence_weight_buffer =
            library.alloc_device_buffer(confidence_weight.len() * 2)?;
        let mut active_slots_buffer = library.alloc_device_buffer(active * 4)?;
        let mut anchors_buffer = library.alloc_device_buffer(active * 4)?;
        let mut compact_hidden_buffer = library.alloc_device_buffer(rows * hidden * 2)?;
        let mut shared_logits_buffer = library.alloc_device_buffer(rows * vocab * 4)?;
        let mut markov_embeddings_buffer = library.alloc_device_buffer(rows * rank * 2)?;
        let mut markov_logits_buffer = library.alloc_device_buffer(active * vocab * 4)?;
        let mut output_tokens_buffer = library.alloc_device_buffer(active * (PROPOSALS + 1) * 4)?;
        let mut confidence_buffer = library.alloc_device_buffer(rows * 4)?;
        library.copy_h2d(normalized_buffer, u16_bytes(&normalized))?;
        library.copy_h2d(collapsed_buffer, u16_bytes(&collapsed))?;
        library.copy_h2d(shared_head_buffer, u16_bytes(&shared_head))?;
        library.copy_h2d(markov_w1_buffer, u16_bytes(&markov_w1))?;
        library.copy_h2d(markov_w2_buffer, u16_bytes(&markov_w2))?;
        library.copy_h2d(confidence_weight_buffer, u16_bytes(&confidence_weight))?;
        library.copy_h2d(active_slots_buffer, u32_bytes(&active_slot_ids))?;
        library.copy_h2d(anchors_buffer, u32_bytes(&anchor_token_ids))?;

        let launched = unsafe {
            library.cuda_dspark_terminal_greedy_bf16_async(
                normalized_buffer,
                collapsed_buffer,
                shared_head_buffer,
                markov_w1_buffer,
                markov_w2_buffer,
                confidence_weight_buffer,
                active_slots_buffer,
                anchors_buffer,
                compact_hidden_buffer,
                shared_logits_buffer,
                markov_embeddings_buffer,
                markov_logits_buffer,
                output_tokens_buffer,
                confidence_buffer,
                active,
                max_slots,
                hidden,
                vocab,
                rank,
                std::ptr::null_mut(),
            )
        };
        match launched {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                let mut output_bytes = vec![0_u8; active * (PROPOSALS + 1) * 4];
                let mut logits_bytes = vec![0_u8; rows * vocab * 4];
                let mut markov_bytes = vec![0_u8; rows * rank * 2];
                let mut confidence_bytes = vec![0_u8; rows * 4];
                library.copy_d2h(&mut output_bytes, output_tokens_buffer)?;
                library.copy_d2h(&mut logits_bytes, shared_logits_buffer)?;
                library.copy_d2h(&mut markov_bytes, markov_embeddings_buffer)?;
                library.copy_d2h(&mut confidence_bytes, confidence_buffer)?;
                assert_eq!(bytes_to_u32_vec(&output_bytes), expected.output_token_ids);
                assert_eq!(
                    markov_bytes
                        .chunks_exact(2)
                        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                    expected.markov_embeddings
                );
                for (actual, expected) in bytes_to_f32_vec(&logits_bytes)
                    .into_iter()
                    .zip(expected.shared_logits)
                {
                    assert!(
                        (actual - expected).abs() <= 1.0e-4,
                        "{actual} != {expected}"
                    );
                }
                for (actual, expected) in bytes_to_f32_vec(&confidence_bytes)
                    .into_iter()
                    .zip(expected.conditional_confidence)
                {
                    assert!(
                        (actual - expected).abs() <= 1.0e-5,
                        "{actual} != {expected}"
                    );
                }
                let cuda_stream = library.cuda_stream_create()?;
                unsafe {
                    library.cuda_graph_begin_capture(cuda_stream)?;
                    library.cuda_dspark_terminal_greedy_bf16_async(
                        normalized_buffer,
                        collapsed_buffer,
                        shared_head_buffer,
                        markov_w1_buffer,
                        markov_w2_buffer,
                        confidence_weight_buffer,
                        active_slots_buffer,
                        anchors_buffer,
                        compact_hidden_buffer,
                        shared_logits_buffer,
                        markov_embeddings_buffer,
                        markov_logits_buffer,
                        output_tokens_buffer,
                        confidence_buffer,
                        active,
                        max_slots,
                        hidden,
                        vocab,
                        rank,
                        cuda_stream,
                    )?;
                    let capture = library.cuda_graph_end_capture_retained(cuda_stream)?;
                    assert!(capture.kernel_node_count >= 18);
                    assert_eq!(capture.memcpy_node_count, 0);
                    assert_eq!(capture.memset_node_count, 0);
                    library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
                    library.cuda_graph_launch(capture.graph_exec, cuda_stream)?;
                    library.cuda_stream_synchronize(cuda_stream)?;
                    library.cuda_graph_exec_destroy(capture.graph_exec)?;
                    library.cuda_graph_destroy(capture.graph)?;
                    library.cuda_stream_destroy(cuda_stream)?;
                }
            }
            Err(error) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(error);
            }
        }

        library.free_device_buffer(&mut confidence_buffer)?;
        library.free_device_buffer(&mut output_tokens_buffer)?;
        library.free_device_buffer(&mut markov_logits_buffer)?;
        library.free_device_buffer(&mut markov_embeddings_buffer)?;
        library.free_device_buffer(&mut shared_logits_buffer)?;
        library.free_device_buffer(&mut compact_hidden_buffer)?;
        library.free_device_buffer(&mut anchors_buffer)?;
        library.free_device_buffer(&mut active_slots_buffer)?;
        library.free_device_buffer(&mut confidence_weight_buffer)?;
        library.free_device_buffer(&mut markov_w2_buffer)?;
        library.free_device_buffer(&mut markov_w1_buffer)?;
        library.free_device_buffer(&mut shared_head_buffer)?;
        library.free_device_buffer(&mut collapsed_buffer)?;
        library.free_device_buffer(&mut normalized_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_lm_head_sample_topk_topp_bf16_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 2;
        let hidden_dim = 3;
        let vocab = 5;
        let temperature = 0.7_f32;
        let top_k = 4;
        let top_p = 0.82_f32;
        let hidden = bf16_values(&[0.5_f32, -1.0, 0.25, -0.5, 0.75, 1.25]);
        let lm_head = bf16_values(&[
            0.25_f32, -0.5, 0.75, -0.25, 0.5, 0.125, 1.0, 0.0, -0.5, 0.25, -0.5, 0.75, -1.0, 0.25,
            0.5,
        ]);
        let random_uniforms = [0.0_f32, 0.99];
        let expected = lm_head_sample_topk_topp_bf16_expected(
            &hidden,
            &lm_head,
            &random_uniforms,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        );
        let expected_argmax =
            lm_head_argmax_bf16_expected(&hidden, &lm_head, rows, hidden_dim, vocab);
        let mut index_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * std::mem::size_of::<f32>()];
        let mut argmax_index_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut argmax_score_bytes = vec![0_u8; rows * std::mem::size_of::<f32>()];
        let logits_bytes = rows * vocab * std::mem::size_of::<f32>();

        let mut hidden_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(hidden.as_slice()))?;
        let mut lm_head_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(lm_head.as_slice()))?;
        let mut random_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(random_uniforms.as_slice()))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;
        let mut argmax_index_buffer = library.alloc_device_buffer(argmax_index_bytes.len())?;
        let mut argmax_score_buffer = library.alloc_device_buffer(argmax_score_bytes.len())?;
        let mut logits_buffer = library.alloc_device_buffer(logits_bytes)?;

        library.copy_h2d(hidden_buffer, u16_bytes(&hidden))?;
        library.copy_h2d(lm_head_buffer, u16_bytes(&lm_head))?;
        library.copy_h2d(random_buffer, f32_bytes(&random_uniforms))?;

        match library.cuda_lm_head_sample_topk_topp_bf16(
            hidden_buffer,
            lm_head_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            hidden_dim,
            vocab,
            temperature,
            top_k,
            top_p,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);

                library.cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
                    hidden_buffer,
                    lm_head_buffer,
                    random_buffer,
                    argmax_index_buffer,
                    argmax_score_buffer,
                    index_buffer,
                    score_buffer,
                    logits_buffer,
                    rows,
                    hidden_dim,
                    vocab,
                    temperature,
                    top_k,
                    top_p,
                )?;
                library.copy_d2h(&mut argmax_index_bytes, argmax_index_buffer)?;
                library.copy_d2h(&mut argmax_score_bytes, argmax_score_buffer)?;
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(
                    bytes_to_u32_vec(&argmax_index_bytes),
                    expected_argmax.indices
                );
                assert_eq!(
                    bytes_to_f32_vec(&argmax_score_bytes),
                    expected_argmax.scores
                );
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);

                unsafe {
                    library.cuda_lm_head_sample_topk_topp_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);

                unsafe {
                    library.cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        argmax_index_buffer,
                        argmax_score_buffer,
                        index_buffer,
                        score_buffer,
                        logits_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut argmax_index_bytes, argmax_index_buffer)?;
                library.copy_d2h(&mut argmax_score_bytes, argmax_score_buffer)?;
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(
                    bytes_to_u32_vec(&argmax_index_bytes),
                    expected_argmax.indices
                );
                assert_eq!(
                    bytes_to_f32_vec(&argmax_score_bytes),
                    expected_argmax.scores
                );
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                assert_eq!(bytes_to_f32_vec(&score_bytes), expected.scores);
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_lm_head_sample_topk_topp_bf16_async(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);

                let staged_err = library
                    .cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        argmax_index_buffer,
                        argmax_score_buffer,
                        index_buffer,
                        score_buffer,
                        logits_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                    )
                    .unwrap_err();
                assert_cuda_unavailable(staged_err);
                let staged_async_err = unsafe {
                    library.cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
                        hidden_buffer,
                        lm_head_buffer,
                        random_buffer,
                        argmax_index_buffer,
                        argmax_score_buffer,
                        index_buffer,
                        score_buffer,
                        logits_buffer,
                        rows,
                        hidden_dim,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(staged_async_err);
            }
        }

        library.free_device_buffer(&mut logits_buffer)?;
        library.free_device_buffer(&mut argmax_score_buffer)?;
        library.free_device_buffer(&mut argmax_index_buffer)?;
        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut random_buffer)?;
        library.free_device_buffer(&mut lm_head_buffer)?;
        library.free_device_buffer(&mut hidden_buffer)?;
        Ok(())
    }

    #[test]
    fn cuda_logits_sample_topk_topp_kernel_ffi_binding_reports_or_executes() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.cuda_device_info(0)?;
        let rows = 3;
        let vocab = 6;
        let temperature = 0.7_f32;
        let top_k = 4;
        let top_p = 0.82_f32;
        let logits = [
            -0.5_f32, 0.1, 0.8, 0.0, 0.8, -0.2, -1.0, -0.7, -0.9, -0.8, -0.6, -0.4, 1.25, 1.0, 0.5,
            1.25, -2.0, 0.0,
        ];
        let random_uniforms = [0.0_f32, 0.42, 0.99];
        let expected = logits_sample_topk_topp_expected(
            &logits,
            &random_uniforms,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        );
        let mut index_bytes = vec![0_u8; rows * std::mem::size_of::<u32>()];
        let mut score_bytes = vec![0_u8; rows * std::mem::size_of::<f32>()];

        let mut logits_buffer = library.alloc_device_buffer(std::mem::size_of_val(&logits))?;
        let mut random_buffer =
            library.alloc_device_buffer(std::mem::size_of_val(&random_uniforms))?;
        let mut index_buffer = library.alloc_device_buffer(index_bytes.len())?;
        let mut score_buffer = library.alloc_device_buffer(score_bytes.len())?;

        library.copy_h2d(logits_buffer, f32_bytes(&logits))?;
        library.copy_h2d(random_buffer, f32_bytes(&random_uniforms))?;

        match library.cuda_logits_sample_topk_topp_f32(
            logits_buffer,
            random_buffer,
            index_buffer,
            score_buffer,
            rows,
            vocab,
            temperature,
            top_k,
            top_p,
        ) {
            Ok(()) => {
                assert_eq!(info.cuda_available, 1);
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                let scores = bytes_to_f32_vec(&score_bytes);
                for (actual, expected) in scores.iter().zip(expected.scores.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }

                unsafe {
                    library.cuda_logits_sample_topk_topp_f32_async(
                        logits_buffer,
                        random_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )?;
                }
                library.copy_d2h(&mut index_bytes, index_buffer)?;
                library.copy_d2h(&mut score_bytes, score_buffer)?;
                assert_eq!(bytes_to_u32_vec(&index_bytes), expected.indices);
                let scores = bytes_to_f32_vec(&score_bytes);
                for (actual, expected) in scores.iter().zip(expected.scores.iter()) {
                    assert!((actual - expected).abs() < 1.0e-5);
                }
            }
            Err(err) => {
                assert_eq!(info.cuda_available, 0);
                assert_cuda_unavailable(err);
                let async_err = unsafe {
                    library.cuda_logits_sample_topk_topp_f32_async(
                        logits_buffer,
                        random_buffer,
                        index_buffer,
                        score_buffer,
                        rows,
                        vocab,
                        temperature,
                        top_k,
                        top_p,
                        std::ptr::null_mut(),
                    )
                }
                .unwrap_err();
                assert_cuda_unavailable(async_err);
            }
        }

        library.free_device_buffer(&mut score_buffer)?;
        library.free_device_buffer(&mut index_buffer)?;
        library.free_device_buffer(&mut random_buffer)?;
        library.free_device_buffer(&mut logits_buffer)?;
        Ok(())
    }

    #[test]
    fn error_propagation() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let err = library.alloc_device_buffer(0).unwrap_err().to_string();
        assert!(err.contains("status 1"));
        assert!(err.contains("size is zero"));
        let err = library
            .alloc_managed_device_buffer(0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("status 1"));
        assert!(err.contains("size is zero"));
        Ok(())
    }

    #[test]
    fn rdma_device_info_and_host_buffer_plan() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let info = library.rdma_device_info()?;
        assert!(!c_char_array_to_string(&info.first_device_name).is_empty());
        assert!(!c_char_array_to_string(&info.status).is_empty());

        let input = vec![0_u8; 12_288];
        let plan =
            library.rdma_plan_host_buffer_registration(input.as_ptr().cast(), input.len(), 4096)?;
        assert_eq!(plan.original_bytes, input.len());
        assert_eq!(plan.alignment, 4096);
        assert!(plan.registered_span_bytes >= input.len());
        assert_eq!(plan.registered_span_bytes % 4096, 0);
        assert_eq!(plan.span_aligned, 1);
        Ok(())
    }

    #[test]
    fn rdma_register_host_buffer_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let mut input = vec![0_u8; 12_288];
        match library.rdma_register_host_buffer_probe(&mut input) {
            Ok(probe) => {
                assert_eq!(probe.bytes, input.len());
                assert_eq!(probe.registered, 1);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {DS41RT_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_create_rc_qp_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        match library.rdma_create_rc_qp_probe(1, 16, 16, 1) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.requested_send_wr, 16);
                assert_eq!(probe.requested_recv_wr, 16);
                assert_eq!(probe.requested_max_sge, 1);
                assert_eq!(probe.created, 1);
                assert_ne!(probe.qp_num, 0);
                assert!(probe.actual_max_send_wr >= probe.requested_send_wr);
                assert!(probe.actual_max_recv_wr >= probe.requested_recv_wr);
                assert!(probe.actual_max_send_sge >= probe.requested_max_sge);
                assert!(probe.actual_max_recv_sge >= probe.requested_max_sge);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {DS41RT_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_rc_send_recv_loopback_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        match library.rdma_rc_send_recv_loopback_probe(1, 12_288) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.bytes, 12_288);
                assert_eq!(probe.completed, 1);
                assert_eq!(probe.payload_matches, 1);
                assert_ne!(probe.sender_qp_num, 0);
                assert_ne!(probe.receiver_qp_num, 0);
                assert_eq!(probe.send_completions, 1);
                assert_eq!(probe.recv_completions, 1);
                assert!(probe.poll_iterations > 0);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {DS41RT_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }

    #[test]
    fn rdma_rc_protocol_v2_loopback_probe_reports_capability() -> Result<()> {
        let Some(library) = load_test_library()? else {
            return Ok(());
        };
        let request = protocol_v2_frame(1, 12_288);
        let response = protocol_v2_frame(2, 12_288);
        match library.rdma_rc_protocol_v2_loopback_probe(1, &request, &response) {
            Ok(probe) => {
                assert_eq!(probe.port_num, 1);
                assert_eq!(probe.request_bytes, request.len());
                assert_eq!(probe.response_bytes, response.len());
                assert_eq!(probe.completed, 1);
                assert_eq!(probe.request_payload_matches, 1);
                assert_eq!(probe.response_payload_matches, 1);
                assert_ne!(probe.client_qp_num, 0);
                assert_ne!(probe.server_qp_num, 0);
                assert_eq!(probe.send_completions, 2);
                assert_eq!(probe.recv_completions, 2);
                assert!(probe.poll_iterations > 0);
                assert!(!c_char_array_to_string(&probe.device_name).is_empty());
                assert!(!c_char_array_to_string(&probe.status).is_empty());
            }
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains(&format!("status {DS41RT_STATUS_RDMA_UNAVAILABLE}")),
                    "{err}"
                );
                assert!(err.contains("RDMA") || err.contains("rdma"), "{err}");
            }
        }
        Ok(())
    }
}
