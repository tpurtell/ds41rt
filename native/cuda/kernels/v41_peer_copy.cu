// SM-issued peer reads keep lane-local waits off the shared DMA copy queue.
#include "ds41rt_v41_peer_copy.h"
#include <cuda_runtime.h>
#include <stdint.h>
namespace {
__global__ void peer_copy(uint8_t* destination,const uint8_t* source,uint64_t bytes) {
  const uint64_t tid=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
  if(((reinterpret_cast<uintptr_t>(destination)|reinterpret_cast<uintptr_t>(source)|bytes)&15)==0) {
    for(uint64_t i=tid;i<bytes/16;i+=stride)
      reinterpret_cast<uint4*>(destination)[i]=reinterpret_cast<const uint4*>(source)[i];
  } else if(((reinterpret_cast<uintptr_t>(destination)|reinterpret_cast<uintptr_t>(source)|bytes)&3)==0) {
    for(uint64_t i=tid;i<bytes/4;i+=stride)
      reinterpret_cast<uint32_t*>(destination)[i]=reinterpret_cast<const uint32_t*>(source)[i];
  } else {
    for(uint64_t i=tid;i<bytes;i+=stride)destination[i]=source[i];
  }
}
}
extern "C" int32_t ds41rt_v41_peer_copy_initialize() {
  cudaFuncAttributes attributes{};
  return cudaFuncGetAttributes(&attributes,peer_copy);
}
// Source peer access must be enabled on the stream's destination device.
// Both buffers remain live and disjoint until completion; no allocation/wait.
extern "C" int32_t ds41rt_v41_peer_copy_async(void* destination,const void* source,
    uint64_t bytes,void* stream) {
  const auto dst=reinterpret_cast<uintptr_t>(destination),src=reinterpret_cast<uintptr_t>(source);
  if(!stream || !dst || !src || bytes==0 || bytes>(1ull<<40) ||
      dst>UINTPTR_MAX-bytes || src>UINTPTR_MAX-bytes || !(dst+bytes<=src || src+bytes<=dst))
    return cudaErrorInvalidValue;
  const uint64_t units=((dst|src|bytes)&15)==0?bytes/16:(((dst|src|bytes)&3)==0?bytes/4:bytes);
  const unsigned blocks=unsigned((units+255)/256<1024?(units+255)/256:1024);
  peer_copy<<<blocks,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<uint8_t*>(destination),static_cast<const uint8_t*>(source),bytes);
  return cudaGetLastError();
}
