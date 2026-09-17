#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Host launch descriptor, copied by value into the graph. Arrays select:
// 0: one request's 128-row ring; 1: private window wave;
// 2: shared paged compressed pool; 3: private compressor wave.
typedef struct ds41rt_v41_sparse_kv {
  const uint8_t* values[4];
  const uint8_t* scales[4];
  const uint64_t* window_end;
  const uint32_t* pages;
  const uint64_t* source_end;
  uint64_t window_proposal_capacity;
  uint64_t source_capacity;
  uint64_t source_proposal_capacity;
  uint32_t page_stride;
  // 0: no source; 1: legacy FP8/E8M0 K32 source; 2: FP4/E4M3 K16 source.
  // Window slots 0/1 always use FP8/E8M0 K32. Captured descriptors retain format.
  uint32_t compressed;
} ds41rt_v41_sparse_kv_t;
int32_t ds41rt_v41_sparse_attention_initialize(void);
// BF16 query/output [rows,64,512], FP32 sink [64], selected I32 [rows,512].
// U64 metadata [rows,10] concatenates WindowProposal::metadata (4 fields) and
// IndexProposal::metadata (6 fields), with source slot narrowed to zero.
// Window keys are oldest first, then padding to window_width, then 512 selected
// source IDs. Negative, noncausal or unmapped source IDs are masked. Malformed
// proposal metadata yields zero for the whole query. Compressed=0 ignores source
// pointers/metadata/IDs. Width must cover every query's causal window (max128).
// Width 0 derives each query's own causal window length, independent of other
// rows in the launch. Positive widths retain explicit padding control.
// Window values E4M3 and scales E8M0/K32; source layout follows compressed.
// FP4 source rows contain 256 E2M1 bytes and 32 E4M3 scale bytes. All values
// must dequantize to finite BF16; queries/sinks
// finite. Same stream device, live immutable inputs, disjoint output through
// completion/replay. Caller guarantees lease/snapshot/request correspondence,
// selected-ID uniqueness and causal source counts. No allocation/synchronization.
int32_t ds41rt_v41_sparse_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream);
// Key partitions retain FP32 partial numerators/maxima/normalizers, then merge
// into BF16 output. Rounding differs from sequential online softmax. Caller
// owns disjoint scratch [rows,parts,64,514] FP32 through completion/replay;
// every element is overwritten, so no scratch initialization is required.
// Parts 1..10; no allocation, module resolution or synchronization at launch.
int32_t ds41rt_v41_sparse_attention_split(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,
    float* partial,uint64_t scratch_bytes,int32_t parts);
// Bounded decoder replay: device U64 window_begins[rows] masks local keys
// before each row's replay boundary. Global compressed keys remain unchanged.
// Bounds must be <= the committed window end (metadata column 0); invalid
// bounds zero the whole query. Zero bounds reproduce the ordinary entry point.
// Bounds may change between graph replays, must remain live through completion,
// and may not overlap output or partial scratch. Null partial selects sequential
// attention and requires scratch_bytes=0, parts=1; otherwise split rules apply.
int32_t ds41rt_v41_sparse_attention_bounded(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,int32_t window_width,
    const ds41rt_v41_sparse_kv_t* view,void* stream,const uint64_t* window_begins,
    float* partial,uint64_t scratch_bytes,int32_t parts);
// Small multi-request split attention, homogeneous compressed format, rows 1..64.
// Each row has one device descriptor; metadata/query/selection use global row
// indices. Width is per-row automatic. Parts must be 2 without source, 10 with
// source. Scratch is [rows,parts,64,514] FP32 and bounds are mandatory (zeros
// allowed). Validate every current host descriptor before uploading/replaying;
// validation performs no device work. Every pointer must be on the stream device.
int32_t ds41rt_v41_sparse_attention_batch_validate(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* host_views,const ds41rt_v41_sparse_kv_t* device_views,
    const uint64_t* begins,float* partial,uint64_t scratch_bytes,int32_t parts,int32_t compressed);
// Requires successful validation for EXACTLY these buffers, dimensions, format,
// and descriptors before launch and every replay. Upload the matching host
// descriptors on the stream first. No allocations or module resolution here.
// Device descriptors and referenced allocations remain live/immutable through
// completion. Capture retains device descriptor addresses, not their contents;
// callers may change contents only after prior consumers have completed.
int32_t ds41rt_v41_sparse_attention_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed);
// Optional AOT symbol: same prevalidation/ownership contract as batch above,
// additionally requiring FP4, 16-byte-aligned value/scale planes and scratch.
// A captured graph must retain this eligibility; include backend in its key.
int32_t ds41rt_v41_sparse_attention_batch_aot(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed);
// Compact TP2 local-head variant: query/output [rows,32,512], sink [32],
// scratch [rows,parts,32,514]. Caller slices the corresponding query and sink
// heads and supplies local replicated KV; metadata and selected IDs are unchanged.
// All bounded-entry validation, lifetime and replay rules above apply. Initialize
// on each participating device before capture. Eligible FP4 decode uses AOT32.
int32_t ds41rt_v41_sparse_attention_heads32_initialize(void);
int32_t ds41rt_v41_sparse_attention_heads32_bounded(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,int32_t window_width,
    const ds41rt_v41_sparse_kv_t* view,void* stream,const uint64_t* window_begins,
    float* partial,uint64_t scratch_bytes,int32_t parts);
// Compact-head batch: same batch ownership contract, 32-head buffer extents.
int32_t ds41rt_v41_sparse_attention_heads32_batch_validate(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* host_views,const ds41rt_v41_sparse_kv_t* device_views,
    const uint64_t* begins,float* partial,uint64_t scratch_bytes,int32_t parts,int32_t compressed);
// Requires successful validation for EXACTLY these buffers, dimensions, format,
// and descriptors before launch and every replay. Upload the matching host
// descriptors on the stream first. No allocations or module resolution here.
// Device descriptors and referenced allocations remain live/immutable through
// completion. Capture retains device descriptor addresses, not their contents;
// callers may change contents only after prior consumers have completed.
int32_t ds41rt_v41_sparse_attention_heads32_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed);
// Optional AOT32 batch symbol; same eligibility rules as AOT64 above.
int32_t ds41rt_v41_sparse_attention_heads32_batch_aot(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed);
#ifdef __cplusplus
}
#endif
