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
// Copy disjoint pitched byte rows locally or from a peer. Width/pitches are bytes,
// rows 1..4096, each referenced span at most 2^40 bytes. Same ownership and stream
// rules as above; source peer access is needed only for distinct device owners.
int32_t ds41rt_v41_peer_copy_rows_async(void* destination,const void* source,
    uint64_t width,uint64_t rows,uint64_t destination_pitch,uint64_t source_pitch,void* stream);
#ifdef __cplusplus
}
#endif
