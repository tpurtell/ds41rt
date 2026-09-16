#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Input rows: 5120 E4M3 bytes followed by 160 UE8M0 K32 scale bytes.
 * Output: contiguous BF16[rows,5120], rounded after reconstructing FP32 values.
 * Uses CUDA's E4M3/UE8M0 conversion semantics (including subnormal/NaN codes).
 * Initialize before capture, on the owning device. Caller retains distinct,
 * 16-byte-aligned buffers and this handle through completion/graph destruction.
 * Launch performs no allocation or host synchronization; 1 <= rows <= 4096.
 * Return values are CUDA runtime status codes. Destroy on the owning thread. */
int32_t ds41rt_v41_exl3_wire_initialize(void** out);
void ds41rt_v41_exl3_wire_destroy(void* handle);
int32_t ds41rt_v41_exl3_wire_decode(void* handle, const uint8_t* input,
    uint64_t input_bytes, uint16_t* output, uint64_t output_bytes,
    uint32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
