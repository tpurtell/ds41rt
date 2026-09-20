use thiserror::Error;

#[derive(Debug, Error)]
pub enum Ds41rtError {
    #[error("invalid DeepSeek V4 attention geometry: {reason}")]
    InvalidDeepseekV4AttentionGeometry { reason: String },
    #[error("invalid DeepSeek V4 physical KV layout: {reason}")]
    InvalidDeepseekV4KvLayout { reason: String },
    #[error("unknown role: {0}")]
    UnknownRole(String),
    #[error("unknown placement policy: {0}")]
    UnknownPlacementPolicy(String),
    #[error("unknown replicated expert tie-seed mode: {0} (expected dispatch or layer)")]
    UnknownReplicatedExpertTieSeed(String),
    #[error("KV cache capacity exceeded: requested {requested_tokens} tokens, available {available_tokens}")]
    KvCapacityExceeded {
        requested_tokens: usize,
        available_tokens: usize,
    },
    #[error("unknown KV reservation: {0}")]
    UnknownKvReservation(u64),
    #[error("unknown KV write: {0}")]
    UnknownKvWrite(u64),
    #[error("KV write out of bounds: token_start {token_start} token_count {token_count} reservation_tokens {reservation_tokens}")]
    KvWriteOutOfBounds {
        token_start: usize,
        token_count: usize,
        reservation_tokens: usize,
    },
    #[error("KV backing payload bytes mismatch: expected {expected_bytes} actual {actual_bytes}")]
    KvBackingPayloadSizeMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    #[error("KV backing payload block count mismatch: expected {expected_blocks} actual {actual_blocks}")]
    KvBackingPayloadCountMismatch {
        expected_blocks: usize,
        actual_blocks: usize,
    },
    #[error("MTP accepted token count {accepted_tokens} exceeds draft token count {draft_tokens}")]
    MtpAcceptedTokensExceedDraft {
        accepted_tokens: usize,
        draft_tokens: usize,
    },
    #[error("invalid direct MTP KV transaction: {reason}")]
    InvalidMtpKvTransaction { reason: String },
    #[error("LayerWave mix rejected: {reason}")]
    LayerWaveMixRejected { reason: String },
    #[error("ExpertBatch mix rejected: {reason}")]
    ExpertBatchMixRejected { reason: String },
    #[error("ExpertBatch row source count {source_rows} did not match hidden rows {hidden_rows}")]
    ExpertBatchRowCountMismatch {
        source_rows: usize,
        hidden_rows: usize,
    },
    #[error("ExpertBatch partial output row count {actual} did not match expected {expected}")]
    ExpertBatchPartialRowCountMismatch { expected: usize, actual: usize },
    #[error("expert route plan rejected: {reason}")]
    ExpertRoutePlanRejected { reason: String },
    #[error("ExpertHostBatch unknown host: {host}")]
    ExpertHostBatchUnknownHost { host: String },
    #[error(
        "ExpertHostBatch route count {actual} did not match ExpertBatch route count {expected}"
    )]
    ExpertHostBatchRouteCountMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatch route range for row {row_index} ({start}..{end}) exceeds route count {route_count}")]
    ExpertHostBatchRouteRangeOutOfBounds {
        row_index: usize,
        start: usize,
        end: usize,
        route_count: usize,
    },
    #[error(
        "ExpertHostBatch route row_index {actual} did not match expected global row {expected}"
    )]
    ExpertHostBatchRouteRowMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatch global row {row_index} exceeds global row count {row_count}")]
    ExpertHostBatchGlobalRowOutOfBounds { row_index: usize, row_count: usize },
    #[error("ExpertHostBatch hidden payload bytes mismatch: expected {expected_bytes} actual {actual_bytes}")]
    ExpertHostBatchHiddenPayloadSizeMismatch {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    #[error("ExpertHostBatch partial output row count {actual} did not match expected {expected}")]
    ExpertHostBatchPartialRowCountMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatch global partial accumulator values {actual_values} did not match expected {expected_values}")]
    ExpertHostBatchPartialAccumulatorSizeMismatch {
        expected_values: usize,
        actual_values: usize,
    },
    #[error("ExpertHostBatch contribution count rows {actual} did not match expected {expected}")]
    ExpertHostBatchContributionCountMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatch partial output width for host row {row_index} was {actual_width}, expected {expected_width}")]
    ExpertHostBatchPartialOutputWidthMismatch {
        row_index: usize,
        expected_width: usize,
        actual_width: usize,
    },
    #[error(
        "ExpertHostBatchSet route count {actual} did not match ExpertBatch route count {expected}"
    )]
    ExpertHostBatchSetRouteCountMismatch { expected: usize, actual: usize },
    #[error("expert TP host count {actual} did not match required world size {expected}")]
    ExpertTensorParallelHostCountMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatchSet partial host count {actual} did not match expected {expected}")]
    ExpertHostBatchSetPartialHostCountMismatch { expected: usize, actual: usize },
    #[error("ExpertHostBatchSet duplicate host {host} in reconstruction contract")]
    ExpertHostBatchSetDuplicateHost { host: String },
    #[error(
        "ExpertHostBatchSet reconstruction plan global row count {actual} did not match expected {expected}"
    )]
    ExpertHostBatchSetReconstructionPlanGlobalRowCountMismatch { expected: usize, actual: usize },
    #[error(
        "ExpertHostBatchSet reconstruction plan host count {actual} did not match expected {expected}"
    )]
    ExpertHostBatchSetReconstructionPlanHostCountMismatch { expected: usize, actual: usize },
    #[error(
        "ExpertHostBatchSet reconstruction plan host at index {map_index} was {actual}, expected {expected}"
    )]
    ExpertHostBatchSetReconstructionPlanHostMismatch {
        map_index: usize,
        expected: String,
        actual: String,
    },
    #[error(
        "ExpertHostBatchSet reconstruction plan row count for host {host} was {actual}, expected {expected}"
    )]
    ExpertHostBatchSetReconstructionPlanRowCountMismatch {
        host: String,
        expected: usize,
        actual: usize,
    },
    #[error(
        "ExpertHostBatchSet reconstruction plan global row for host {host} row {host_row_index} was {actual}, expected {expected}"
    )]
    ExpertHostBatchSetReconstructionPlanGlobalRowMismatch {
        host: String,
        host_row_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("ExpertHostBatchSet reconstruction plan has no host contribution for global row {row_index}")]
    ExpertHostBatchSetReconstructionPlanMissingGlobalRow { row_index: usize },
    #[error("Graph buffer contract invalid: {reason}")]
    GraphBufferContractInvalid { reason: String },
    #[error("Graph buffer active count {field}={actual} exceeds capacity {capacity}")]
    GraphBufferActiveCountOutOfBounds {
        field: &'static str,
        actual: usize,
        capacity: usize,
    },
}
