#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Internal initialized AOT launch. The sparse-attention entry validates spans,
// descriptor format/alignment and scratch before calling this implementation.
int32_t ds41rt_v41_attention_aot_initialize(void);
int32_t ds41rt_v41_attention_aot_launch(const void* query, const void* descriptors,
    const void* metadata, const void* selected, const void* bounds, const void* sink,
    void* partials, void* lses, void* output, int32_t rows, void* stream);
int32_t ds41rt_v41_attention_heads32_aot_initialize(void);
int32_t ds41rt_v41_attention_heads32_aot_launch(const void* query, const void* descriptors,
    const void* metadata, const void* selected, const void* bounds, const void* sink,
    void* partials, void* lses, void* output, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
