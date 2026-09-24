#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Finite BF16 [rows,512] -> E4M3 [rows,512], E8M0 [rows,16], K32 groups.
// Optional complex FP32 frequencies [rows,32,2] rotate the final 64 coordinates,
// rounded to BF16 before quantization. Rotated values must remain finite.
// Scale: ceil-power-of-two(max(amax,1e-4)/448). Outputs are disjoint from inputs.
int32_t ds41rt_v41_kv_pack(const uint16_t* input,const float* frequencies,
    uint8_t* values,uint8_t* scales,int32_t rows,void* stream);
// Scatter accepted rows into persistent KV. U64 destinations [rows] must be
// unique among valid destinations; IDs >=capacity skip. No cache metadata is
// published here. The caller serializes writes and publishes after draining.
int32_t ds41rt_v41_kv_store(const uint8_t* values,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream);
// Batched window-layer store: one launch over a 16-byte aligned device table
// of `layers` entries. Each entry scatters `rows` accepted rows exactly as
// ds41rt_v41_kv_store (U64 destinations [rows], IDs >=capacity skip) and then
// writes `chunks` (slot, end) U64 pairs into ends[slot].
typedef struct {
  const uint8_t* values;
  const uint8_t* scales;
  uint8_t* cache;
  uint8_t* cache_scales;
  uint64_t* ends;
  uint64_t capacity;
  const uint64_t* destinations;
  const uint64_t* end_pairs;
} ds41rt_v41_kv_store_layer_t;
int32_t ds41rt_v41_kv_store_layers(const void* table,int32_t layers,int32_t rows,
    int32_t chunks,void* stream);
// Native compressed KV: packed E2M1 [rows,256], E4M3 scales [rows,32].
// Input and rotated magnitudes must be <=2688 (finite E4M3 scale range).
// Groups of 16; scale = E4M3(max(amax,6*2^-9)/6), RNE values.
// Optional rotary frequencies and scatter contracts are the same as above.
int32_t ds41rt_v41_compressed_kv_pack(const uint16_t* input,const float* frequencies,
    uint8_t* values,uint8_t* scales,int32_t rows,void* stream);
int32_t ds41rt_v41_compressed_kv_store(const uint8_t* values,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream);
#ifdef __cplusplus
}
#endif
