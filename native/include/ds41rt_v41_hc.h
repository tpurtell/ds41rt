#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native V4.1 mHC=4, hidden=5120; raw CUDA status, no allocation or sync.
// BF16 residual [rows,4,5120], FP32 pre/post [rows,4], comb [rows,4,4].
// Comb is [source,destination], matching sum(dim=2) in the reference.
// BF16 collapsed/sublayer [rows,5120]; outputs must be disjoint from inputs.
int32_t ds41rt_v41_hc_pre(const uint16_t* residual, const float* pre,
    uint16_t* collapsed, int32_t rows, void* stream);
int32_t ds41rt_v41_hc_post(const uint16_t* sublayer, const uint16_t* residual,
    const float* post, const float* comb, uint16_t* output, int32_t rows, void* stream);
// FP32 fn [24,20480], scale [3], base [24]; produces pre/post [rows,4],
// comb [rows,4,4]. Official norm_eps=1e-20, hc_eps=1e-6 and 20 Sinkhorn iterations are fixed.
int32_t ds41rt_v41_hc_mixes(const uint16_t* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb,
    int32_t rows, void* stream);
// Call once during planning on the serving device, outside CUDA graph capture.
int32_t ds41rt_v41_hc_project_initialize();
// Small rows (1..16) use split-K FP32 projection with changed summation order.
// Scratch is 16-byte aligned, at least rows*1536 bytes, disjoint from all
// inputs/outputs, and live until stream completion. Larger rows ignore scratch.
// Non-AOT builds use the original mixes implementation.
int32_t ds41rt_v41_hc_mixes_workspace(const uint16_t* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb,
    void* scratch, uint64_t scratch_bytes, int32_t rows, void* stream);
// Optional AOT begin entry point, available only in lagged-mHC builds. All
// pointers are 16-byte aligned; outputs and scratch are disjoint from every
// input and each other. rows=1..80, scratch_bytes>=rows*8000. Initialize on
// each device with ds41rt_v41_hc_project_initialize before capture.
int32_t ds41rt_v41_hc_begin(const void* residual, const void* fn,
    const void* scale, const void* base, const void* incoming, const void* norm,
    void* predicted, void* post, void* comb, void* normalized, void* scratch,
    uint64_t scratch_bytes, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
