#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DS41RT_V41_EXPERT_POINTERS 44
/* Per-expert prepared sizes in bytes: W13, W13 scales, W2, W2 scales.
 * Logical intermediate must be 2304 (full RTX), 1152 (RTX TP2), or 576 (backbone TP4).
 * Sources are contiguous official bytes: W1, W3, W2, S1, S3, S2.
 * Destinations are distinct, 16-byte aligned device allocations with the sizes
 * returned below; source and destination storage must not overlap.
 * Representation-only transform preserves FP4/E8M0 bytes and zero-pads tails;
 * caller owns all buffers and stream ordering through completion. */
int32_t ds41rt_v41_expert_packed_sizes(uint32_t intermediate, uint64_t bytes[4]);
int32_t ds41rt_v41_pack_expert_async(const uint8_t* const sources[6],
    uint8_t* const destinations[4], uint32_t intermediate, void* stream);
/* Pointer slots follow the exported dynamic W4A8 C ABI; see the manifest. */
typedef struct ds41rt_v41_expert_launch_t {
  void* tensors[DS41RT_V41_EXPERT_POINTERS];
  int32_t num_tokens;
  int32_t max_rows;
  int32_t scatter_rows;
  int32_t rows_padded;
  int32_t max_tasks;
  int32_t max_phys_tiles;
  int32_t max_active_clusters;
  void* stream;
} ds41rt_v41_expert_launch_t;

typedef struct ds41rt_v41_expert_info_t {
  uint32_t abi_version;
  uint32_t role; /* 0: coordinator dSpark; 1: Spark TP4 shard; 2: full RTX backbone; 3: backbone TP2; 4: dSpark TP2 */
  uint32_t experts;
  uint32_t hidden_size;
  uint32_t logical_intermediate;
  uint32_t kernel_intermediate;
  uint32_t topk;
  uint32_t capacity_rows;
  uint64_t scratch_bytes;
  int32_t max_rows;
  int32_t rows_padded;
  int32_t max_tasks;
  int32_t max_phys_tiles;
  int32_t max_active_clusters;
  uint32_t input_dtype; /* ABI 2: 1 BF16; 7 row E4M3 + UE8M0 K32 scales */
} ds41rt_v41_expert_info_t;

/* Prewarm before graph capture. Quantize BF16 [rows,5120] into contiguous
 * 5280-byte rows: 5120 E4M3 bytes then 160 UE8M0 K32 scales, amax floor 1e-4.
 * Caller owns nonoverlapping 16-byte-aligned input/output on the initialized
 * device, valid through stream completion; 1 <= rows <= 4096. No allocation
 * or synchronization occurs in quantize. Input BF16 rounding is preserved. */
int32_t ds41rt_v41_expert_input_quant_initialize(void** out_kernel);
int32_t ds41rt_v41_expert_input_quantize_async(void* kernel,
    const uint16_t* input, uint8_t* output, uint32_t rows, void* stream);

/* These functions return CUDA runtime error codes (zero is success).
 * Initialize before graph capture; each variant binds to its first CUDA device.
 * Kernel handles borrow the library and must outlive every launch/graph replay.
 * Callers own CUDA buffers, capacity, initialization and stream ordering. */
int32_t ds41rt_v41_expert_info(int32_t capacity, ds41rt_v41_expert_info_t* out);
int32_t ds41rt_v41_expert_initialize(int32_t capacity, void** out_kernel);
int32_t ds41rt_v41_expert_launch(void* kernel, const ds41rt_v41_expert_launch_t* args);
/* Caller-owned scratch must be 16-byte aligned and at least info.scratch_bytes.
 * Bind writes only scratch pointer slots, preserving caller weight/input slots.
 * Initialize once before use/capture, with exclusive ownership on this stream;
 * it zeros storage and sets the native E8M0 recipe's global scales to one. */
int32_t ds41rt_v41_expert_bind_scratch(void* kernel, void* storage,
    uint64_t bytes, void* tensors[DS41RT_V41_EXPERT_POINTERS]);
int32_t ds41rt_v41_expert_initialize_scratch_async(void* kernel, void* storage,
    uint64_t bytes, void* stream);
/* Reduce contiguous FP32 [rows,topk,5120] route planes into BF16 [rows,5120].
 * Supported geometries: ranks=1 or 2/topk=3 (RTX dSpark), ranks=4/topk=6 (backbone).
 * planes is a host array of device pointers; unused slots must be null.
 * Sum TP ranks before rounding each route to BF16, then sum routes in FP32,
 * add optional BF16 shared output and round once to BF16.
 * Output may equal shared exactly, but must not overlap any route plane or
 * partially overlap shared; all device storage must outlive stream completion.
 * No allocation or synchronization; returns a CUDA runtime error code. */
int32_t ds41rt_v41_reduce_routes_async(const float* const planes[4],
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t ranks, uint32_t topk, void* stream);
/* ABI 3 variants expose FP32 token accumulations instead of per-route planes.
 * Kind 0: [rows,topk,5120] FP32 routes; kind 1: [rows,5120] FP32 token sums;
 * kind 2: [rows,topk,5120] BF16 deterministic routes (NVFP4 ABI 2). */
int32_t ds41rt_v41_expert_output_kind(int32_t capacity, uint32_t* out);
/* Round FP32 [rows,5120] token sums to BF16 without allocation or synchronization.
 * Nonoverlapping storage must remain alive through stream completion. */
int32_t ds41rt_v41_compact_tokens_bf16_async(const float* tokens,
    uint16_t* output, uint32_t rows, void* stream);

/* Compact backbone serving arithmetic: sum six local FP32 route vectors in
 * slot order, then round once to a BF16 hidden-width partial. This differs
 * from per-route TP reduction above. Input [rows,6,5120], output [rows,5120].
 * Storage must not overlap and must remain alive through stream completion. */
int32_t ds41rt_v41_compact_routes_bf16_async(const float* routes,
    uint16_t* output, uint32_t rows, void* stream);
/* Sum four compact BF16 partials in rank order in FP32, add optional BF16
 * shared expert and round once to BF16. All planes are [rows,5120].
 * Output must not overlap planes; exact output==shared is permitted, partial
 * overlap is not. Both compact functions allocate nothing and do not sync. */
int32_t ds41rt_v41_reduce_compact_bf16_async(const uint16_t* const planes[4],
    const uint16_t* shared, uint16_t* output, uint32_t rows, void* stream);
/* Full local routed output: sum six FP32 routes (token_sums=0) or consume
 * FP32 token sums (token_sums=1), round to BF16, add optional BF16 shared and
 * round to BF16. Output may equal shared exactly; no routed/output overlap.
 * All storage must remain valid through stream completion. */
int32_t ds41rt_v41_finish_local_experts_async(const float* routed,
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t token_sums, void* stream);

/* NVFP4 local equivalent: BF16 [rows,6,5120] routes accumulated in FP32,
 * rounded to BF16 before optional shared addition. Same alias/lifetime rules. */
int32_t ds41rt_v41_finish_local_bf16_routes_async(const uint16_t* routed,
    const uint16_t* shared, uint16_t* output, uint32_t rows, void* stream);

/* NVFP4 deterministic route reduction. Inputs are BF16 [rows,6,5120];
 * accumulate six routes (and corresponding TP2 rank pairs) in FP32, with
 * a single final BF16 rounding. 1 <= rows <= 4096; disjoint input/output
 * storage on the current device must remain live through stream completion. */
int32_t ds41rt_v41_compact_bf16_routes_async(const uint16_t* routes,
    uint16_t* output, uint32_t rows, void* stream);
int32_t ds41rt_v41_reduce_tp2_bf16_routes_async(const uint16_t* rank0,
    const uint16_t* rank1, uint16_t* output, uint32_t rows, void* stream);

// Both FP32 rank contributions must be resident on the current device and ready
// on this stream. Sum without intermediate rank rounding; emit BF16 once.
int32_t ds41rt_v41_reduce_tp2_experts_async(const float* rank0, const float* rank1,
    uint16_t* output, uint32_t rows, uint32_t token_sums, void* stream);
#ifdef __cplusplus
}
#endif
