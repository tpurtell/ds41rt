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
template<class T> __global__ void copy_rows(T* destination,const T* source,uint64_t width,
    uint64_t rows,uint64_t destination_pitch,uint64_t source_pitch) {
  const uint64_t tid=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
  for(uint64_t i=tid;i<rows*width;i+=stride) {
    const uint64_t row=i/width,col=i%width;
    destination[row*destination_pitch+col]=source[row*source_pitch+col];
  }
}
bool extent(uint64_t width,uint64_t rows,uint64_t pitch,uint64_t& bytes) {
  if(!width || width>(1ull<<40) || !rows || rows>4096 || pitch<width ||
      (rows>1 && pitch>((1ull<<40)-width)/(rows-1)))return false;
  bytes=(rows-1)*pitch+width;return true;
}
template<class T> int32_t dispatch_rows(void* destination,const void* source,uint64_t width,
    uint64_t rows,uint64_t destination_pitch,uint64_t source_pitch,void* stream) {
  const uint64_t blocks=(width/sizeof(T)*rows+255)/256;
  copy_rows<T><<<unsigned(blocks<4096?blocks:4096),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<T*>(destination),static_cast<const T*>(source),width/sizeof(T),rows,
      destination_pitch/sizeof(T),source_pitch/sizeof(T));
  return cudaGetLastError();
}

}
extern "C" int32_t ds41rt_v41_peer_copy_initialize() {
  cudaFuncAttributes attributes{};
  auto status=cudaFuncGetAttributes(&attributes,peer_copy);
  if(status!=cudaSuccess)return status;
  status=cudaFuncGetAttributes(&attributes,copy_rows<uint4>);
  if(status!=cudaSuccess)return status;
  status=cudaFuncGetAttributes(&attributes,copy_rows<uint32_t>);
  if(status!=cudaSuccess)return status;
  return cudaFuncGetAttributes(&attributes,copy_rows<uint8_t>);
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

extern "C" int32_t ds41rt_v41_peer_copy_rows_async(void* destination,const void* source,
    uint64_t width,uint64_t rows,uint64_t destination_pitch,uint64_t source_pitch,void* stream) {
  uint64_t dst_bytes=0,src_bytes=0;
  const auto dst=reinterpret_cast<uintptr_t>(destination),src=reinterpret_cast<uintptr_t>(source);
  if(!stream || !dst || !src || !extent(width,rows,destination_pitch,dst_bytes) ||
      !extent(width,rows,source_pitch,src_bytes) || dst>UINTPTR_MAX-dst_bytes ||
      src>UINTPTR_MAX-src_bytes || !(dst+dst_bytes<=src || src+src_bytes<=dst))
    return cudaErrorInvalidValue;
  const auto alignment=dst|src|width|destination_pitch|source_pitch;
  if((alignment&15)==0)return dispatch_rows<uint4>(destination,source,width,rows,destination_pitch,source_pitch,stream);
  if((alignment&3)==0)return dispatch_rows<uint32_t>(destination,source,width,rows,destination_pitch,source_pitch,stream);
  return dispatch_rows<uint8_t>(destination,source,width,rows,destination_pitch,source_pitch,stream);
}
