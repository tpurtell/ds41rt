#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Initialize on the destination device before capture. Source peer access must
// already be enabled there. Launch copies 1..2^40 bytes using SM-issued reads;
// no allocation or host synchronization. Buffers must be disjoint and remain
// live, with producer dependencies satisfied, until the destination stream ends.
int32_t ds41rt_v41_peer_copy_initialize(void);
int32_t ds41rt_v41_peer_copy_async(void* destination,const void* source,
    uint64_t bytes,void* stream);
#ifdef __cplusplus
}
#endif
