mod engram_pipeline;
pub use engram_pipeline::{EngramPipeline, EngramRequestTokens, EngramWave};
mod engram_gather;
pub use engram_gather::{
    EngramGatherer, EngramGatherLease, EngramGatherPoll, EngramGatherTicket, EngramGatherTiming,
};
mod engram_staging;
pub use engram_staging::{EngramBatchStaging, EngramGatherView};
mod v41_expert_staging;
pub use v41_expert_staging::{V41ExpertSelection, V41ExpertStaging};
mod v41_catalog;
pub use v41_catalog::{
    read_official_v41_catalog, OfficialV41Catalog, V41StorageBudget, V41Tensor, V41TensorPlacement,
    V41CoordinatorTensorReader,
};
mod v41_config;
mod v41_exl3;
mod v41_exl3_staging;
pub use v41_exl3_staging::V41Exl3TensorSlice;
pub use v41_exl3::{read_v41_exl3_manifest, V41Exl3Manifest, V41Exl3Projection,
    V41Exl3ProjectionKind, V41_EXL3_SCHEMA};
mod v41_image;
pub use v41_image::{V41Image, V41ImageGrid, V41ImageSpan, V41VisionPrompt, V41ImageTokenType,
    V41_IMAGE_TOKEN_ID, V41_MAX_IMAGES};
pub use v41_config::{
    read_official_v41_config, OfficialV41Config, V41QuantizationConfig, V41RopeScaling,
    V41TextConfig, V41VisionConfig, OFFICIAL_V41_MODEL_ID, OFFICIAL_V41_REVISION,
};
mod engram_tokenizer;
pub use engram_tokenizer::EngramTokenMap;
mod engram_prefetch;
pub use engram_prefetch::{EngramPrefetcher, EngramTable, PrefetchOutcome, PrefetchTicket};
mod mapped_rows;
pub use mapped_rows::MappedRows;
mod attention_format;
mod catalog;
mod dspark_format;
mod exl3_format;
mod expert_format;
mod placement;
mod snapshot;
mod tensors;
mod tokenizer;

pub use attention_format::{
    native_deepseek_v4_attention_tensor_specs, validate_native_deepseek_v4_attention_catalog,
    NativeDeepseekV4AttentionCatalogSummary, NativeDeepseekV4AttentionTensorFamily,
    NativeDeepseekV4AttentionTensorSpec, NATIVE_ATTENTION_FP8_BLOCK,
};
pub use catalog::{
    build_catalog, build_catalog_for_snapshot, classification_summary_markdown, read_model_facts,
    read_safetensors_metadata, SafetensorsTensorMetadata,
};
pub use dspark_format::{
    native_deepseek_v4_dspark_tensor_specs, validate_native_deepseek_v4_dspark_catalog,
    NativeDeepseekV4DsparkCatalogSummary, NativeDeepseekV4DsparkTensorSpec,
};
pub use exl3_format::{
    exl3_expert, exl3_expert_trellis_bits, exl3_projection_trellis_bits,
    exl3_trellis_bits_for_recipe, is_deepseek_v4_exl3_recipe, is_deepseek_v4_mixed_exl3_recipe,
    validate_exl3_expert_catalog, Exl3CatalogSummary, Exl3Expert, Exl3Projection,
    Exl3ProjectionKind, Exl3Tp4ResidentGeometry, DEEPSEEK_V4_EXL3_CODEBOOK,
    DEEPSEEK_V4_EXL3_RECIPE, DEEPSEEK_V4_EXL3_RECIPE_K3_V4, DEEPSEEK_V4_EXL3_RECIPE_MIXED_K2_K3_V1,
    DEEPSEEK_V4_EXL3_RECIPE_V2, DEEPSEEK_V4_EXL3_RECIPE_V3, DEEPSEEK_V4_EXL3_RECIPE_V4,
    DEEPSEEK_V4_EXL3_SCHEMA, DEEPSEEK_V4_EXL3_SCHEMA_VERSION, DEEPSEEK_V4_EXL3_SOURCE_FORMAT,
    DEEPSEEK_V4_EXL3_T12_LUT_BYTES, DEEPSEEK_V4_EXL3_TENSOR_FORMAT, DEEPSEEK_V4_EXL3_TRELLIS_BITS,
    EXLLAMAV3_REPOSITORY, EXLLAMAV3_REVISION, EXLLAMAV3_SOURCE_TREE_SHA256,
};
pub use expert_format::{
    native_fp4_expert, validate_native_fp4_expert_catalog, NativeFp4CatalogSummary,
    NativeFp4Expert, NativeFp4Projection, NativeFp4ProjectionKind, NativeFp4TpExpertShard,
    NativeFp4TpProjectionShard, NativeFp4TpTensorWindow, NATIVE_FP4_K_BLOCK,
};
pub use placement::{assignments_by_owner, build_load_plan};
pub use snapshot::{
    default_hf_home, empty_catalog_for_snapshot, model_cache_dir, resolve_snapshot,
    resolve_snapshot_at_revision, SnapshotResolution,
};
pub use tensors::{
    dtype_byte_width, load_tensor_bytes, load_tensor_bytes_with_options, load_tensor_rows,
    load_tensor_rows_with_options, read_tensor_bytes_into, read_tensor_bytes_into_with_options,
    read_tensor_row_prefix_into, read_tensor_row_prefix_into_with_options,
    read_tensor_row_window_into, read_tensor_row_window_into_with_options, read_tensor_rows_into,
    read_tensor_rows_into_with_options, LoadedTensor, LoadedTensorRows, LoadedTensorRowsSummary,
    LoadedTensorSummary, TensorLoadOptions,
};
pub use tokenizer::{
    decode_tokenizer_ids, encode_tokenizer_text, streaming_token_decoder, LoadedTokenizer,
    StreamingTokenDecoder, TokenizerDecodeSummary, TokenizerEncodingSummary,
};

#[cfg(test)]
mod tests;
