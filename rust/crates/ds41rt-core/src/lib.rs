mod dspark_routes;
pub use dspark_routes::{DsparkRouteForecast, DsparkRouteHistory, DsparkWorkForecast, DsparkWorkEvaluator};
mod dspark_policy;
pub use dspark_policy::{select_dspark_confidence_prefix, select_dspark_prefixes, select_dspark_prefixes_bounded, DsparkPrefixSelection};
mod dspark_rng;
pub use dspark_rng::{DsparkRng, DsparkRngReservation};
mod target_sampling;
pub use target_sampling::{
    TargetSamplingError, TargetSamplingParams, GREEDY_TEMPERATURE_EPS, MAX_TARGET_TEMPERATURE,
};
mod dspark_verify;
pub use dspark_verify::{verify_dspark_greedy, GreedyVerification, MAX_DSPARK_PROPOSALS};
mod engram;
pub mod prefix;
pub use engram::{EngramBatch, EngramError, EngramHashes, EngramHistory, EngramPrefillCursor, ENGRAM_LAYERS, ENGRAM_ROWS, ENGRAM_COMPRESSED_VOCAB};
mod attention_geometry;
mod constants;
mod coordinator_graphs;
mod cpu_affinity;
mod debug_expert;
mod deepseek_v4_compressor;
mod deepseek_v4_kv;
mod errors;
mod expert_batch;
mod expert_host_batch;
mod expert_route_plan;
mod exl3_tp4_ownership;
pub use exl3_tp4_ownership::{
    exl3_tp4_active_blocks, Exl3BoundaryCost, Exl3Tp4OwnershipPlan,
    Exl3Tp4OwnershipPlanner, EXL3_TP4_RESIDENT_BLOCKS,
};
mod graph_buffers;
mod ids;
mod kv_cache;
mod layerwave;
mod model;
mod node;
mod placement;
mod replicated_expert_schedule;
mod tiny;
mod transport_metrics;

pub use attention_geometry::{
    DeepseekV4AttentionGeometry, DeepseekV4AttentionLayerPlan, DeepseekV4AttentionLayerSource,
    DeepseekV4AttentionPlan, DeepseekV4CompressedSelection,
};
pub use constants::{
    COORDINATOR_HOST, DEFAULT_MODEL_ID, DS4_COMPRESS_ROPE_THETA, DS4_DSPARK_BLOCK_SIZE,
    DS4_DSPARK_NOISE_TOKEN_ID, DS4_EXPERT_TP_WORLD_SIZE, DS4_FLASH_COMPRESS_RATIOS,
    DS4_FLASH_DSPARK_MARKOV_RANK, DS4_FLASH_HIDDEN_BF16_BYTES, DS4_FLASH_HIDDEN_SIZE,
    DS4_FLASH_MODEL_ID, DS4_FLASH_MOE_INTERMEDIATE_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS,
    DS4_FLASH_Q_LORA_RANK, DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K, DS4_HC_EPS, DS4_HC_MULT,
    DS4_HC_SINKHORN_ITERS, DS4_HEAD_DIM, DS4_INDEX_HEADS, DS4_INDEX_HEAD_DIM, DS4_KV_HEADS,
    DS4_NUM_HASH_LAYERS, DS4_NUM_SHARED_EXPERTS, DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    DS4_O_LORA_RANK, DS4_PRO_COMPRESS_RATIOS, DS4_PRO_DSPARK_MARKOV_RANK,
    DS4_PRO_HIDDEN_BF16_BYTES, DS4_PRO_HIDDEN_SIZE, DS4_PRO_MOE_INTERMEDIATE_SIZE,
    DS4_PRO_NUM_HIDDEN_LAYERS, DS4_PRO_PREVIEW_MODEL_ID, DS4_PRO_Q_LORA_RANK,
    DS4_PRO_ROUTED_EXPERTS, DS4_PRO_TOP_K, DS4_QK_ROPE_HEAD_DIM, DS4_ROPE_BETA_FAST,
    DS4_ROPE_BETA_SLOW, DS4_ROPE_SCALING_FACTOR, DS4_ROPE_THETA, DS4_SLIDING_WINDOW, EXPERT_HOSTS,
    GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN, GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN,
    GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN, GLM52_DSA_INDEXER_LAYERS,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP, GLM52_DSA_INDEX_HEAD_DIM,
    GLM52_EXPANDED_DEBUG_KV_BF16_BYTES_PER_TOKEN, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
    GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_BLOCK_SIZE,
    GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MLA_ROPE_THETA, GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_NUM_MTP_LAYERS, GLM52_ROUTED_EXPERTS, GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
    GLM52_TOTAL_LAYERS_WITH_MTP, SUPPORTED_MODEL_IDS,
};
pub use coordinator_graphs::{
    coordinator_graph_bucket_for_active_rows, CoordinatorGraphInstancePlan, CoordinatorGraphKey,
    CoordinatorGraphNetworkBoundary, CoordinatorGraphShape, COORDINATOR_GRAPH_DECODE_BUCKET_ROWS,
    COORDINATOR_GRAPH_INSTANCE_COUNT, COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS,
    COORDINATOR_GRAPH_SHAPES,
};
pub use cpu_affinity::pin_current_thread_to_cpu;
pub use debug_expert::{
    ExpertRequest, ExpertRequestHeader, ExpertResponse, ExpertResponseHeader, ExpertRow,
    ExpertWaveMetadata, RouteEntry,
};
pub use deepseek_v4_compressor::{
    deepseek_v4_compressor_execution_plans, DeepseekV4CompressorContinuationPlan,
    DeepseekV4CompressorDecodeStep, DeepseekV4CompressorKind,
    DeepseekV4CompressorLayerExecutionPlan, DeepseekV4CompressorPrefillPlan,
    DeepseekV4CompressorSpec, DeepseekV4CompressorStateFill,
};
pub use deepseek_v4_kv::{
    DeepseekV4KvBoundaryCopyPlan, DeepseekV4KvBoundaryLayerCopyPlan, DeepseekV4KvCacheFormat,
    DeepseekV4KvPageCopySpan, DeepseekV4KvPagePlane, DeepseekV4KvRegionKind,
    DeepseekV4KvRegionPlan, DeepseekV4KvSlot, DeepseekV4PhysicalKvLayerPlan,
    DeepseekV4PhysicalKvPlan, DS4_INDEX_FP32_SCALE_BYTES_PER_ROW, DS4_INDEX_FP8_BYTES_PER_ROW,
    DS4_KV_NOPE_FP8_BYTES, DS4_KV_NVFP4_BYTES_PER_ROW, DS4_KV_PAGE_ALIGNMENT_BYTES,
    DS4_KV_PAYLOAD_BYTES_PER_ROW, DS4_KV_ROPE_BF16_BYTES, DS4_KV_SOURCE_PAGE_TOKENS,
    DS4_KV_UE8M0_FOOTER_BYTES_PER_ROW, DS4_KV_UNPADDED_BYTES_PER_ROW,
};
pub use errors::Ds41rtError;
pub use expert_batch::{ExpertBatch, ExpertBatchRow};
pub use expert_host_batch::{
    ExpertBatchRoute, ExpertHostBatch, ExpertHostBatchRow, ExpertHostBatchSet,
    ExpertHostBatchSetAccumulation, HostRowToGlobalRowMap, PartialReconstructionPlan,
};
pub use expert_route_plan::{
    plan_completion_first_routes, plan_rolling_expert_row_packs, CompletionFirstRouteGroup,
    CompletionFirstRoutePlan, CompletionRoutePlanEntry, RollingExpertRowPackAccumulator,
    RollingExpertRowPackConfig, RollingExpertRowPackEmission, RollingExpertRowPackPlan,
};
pub use graph_buffers::{
    ExpertGraphActiveCounts, ExpertGraphBufferContract, ExpertGraphExecutionEnvelope,
    ExpertGraphHostBatchLease, ExpertGraphHostBatchSetLease, ExpertGraphInstancePool,
    ExpertGraphKey, ExpertGraphPoolEntry, ExpertGraphPoolLease, ExpertGraphPoolStats,
    ExpertWorkspaceContract, HiddenRowsBufferContract, PartialOutputBufferContract,
    RouteMetadataBufferContract, EXPERT_GRAPH_ACTIVE_COUNTS_BYTES,
    EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES, EXPERT_GRAPH_PROTOCOL_V2_LAYOUT,
    EXPERT_GRAPH_ROUTE_ENTRY_BYTES, EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES,
    EXPERT_GRAPH_TILE_METADATA_BYTES,
};
pub use ids::{LayerId, PlacementVersion, PositionId, Priority, RequestId};
pub use kv_cache::{
    KvBackedBlock, KvCacheAllocator, KvCacheBackingStore, KvCacheConfig, KvCacheDType,
    KvCacheSnapshot, KvLayout, KvReservation, KvReservationState, KvWriteRecord, KvWriteState,
    MlaKvCacheRepresentation,
};
pub use layerwave::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, plan_prefill_chunks_with_model,
    DecodeStep, GraphBucket, HiddenShape, KvBlockDescriptor, LayerWave, LayerWaveAdmission,
    LayerWaveMode, MtpVerifyBlock, PrefillChunk, PrefillChunkPolicy, RouteMetadataPlaceholder,
    RowSource, RowSourceKind,
};
pub use model::{
    AttentionKind, DType, ModelFacts, ModelVariant, TensorCatalog, TensorInfo, TensorRole,
};
pub use node::NodeRole;
pub use placement::{
    owner_for_expert, ExpertOwnerLookup, LoadPlan, PlacementPolicy, TensorAssignment,
};
pub use replicated_expert_schedule::{
    replicated_expert_tie_seed, replicated_expert_tie_seed_for, ReplicatedExpertCostModel,
    ReplicatedExpertGroupId, ReplicatedExpertGroupLoad, ReplicatedExpertGroupPlan,
    ReplicatedExpertScheduleConfig, ReplicatedExpertScheduler, ReplicatedExpertTieSeedMode,
    INACTIVE_REPLICATED_EXPERT_GROUP, MAX_REPLICATED_EXPERT_GROUPS, TIE_SEED_FIXED_REQUEST_ID,
};
pub use tiny::deterministic_tiny_completion;
pub use transport_metrics::{
    TransportCapabilities, TransportPrefillBandwidthMeasurement, TransportRttMeasurement,
};

#[cfg(test)]
mod tests;
