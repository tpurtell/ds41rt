#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include "ds41rt_v41_index_topk.h"
namespace {
bool valid(const void* p,uint64_t n,uint64_t alignment) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
template<uint64_t Limit>
__device__ uint64_t encode(float score,uint64_t pos) {
  if(pos>=Limit || isnan(score) || score==-CUDART_INF_F)return 0;
  // Treat both signed zeros as equal scores.
  const uint32_t bits=score==0.f?0u:__float_as_uint(score);
  const uint32_t ordered=bits&0x80000000u?~bits:bits^0x80000000u;
  return (uint64_t(ordered)<<32)|uint32_t(~uint32_t(pos));
}
template<int K> using Sort=cub::BlockRadixSort<uint64_t,256,K/128>;
template<int K,uint64_t Limit>
__global__ void tiles(const float* scores,const uint64_t* positions,uint64_t* out,int count,int blocks) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=blockIdx.x*(2*K)+threadIdx.x*(K/128)+i;
    keys[i]=col<count?encode<Limit>(scores[row*count+col],positions[row*count+col]):0;
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)out[(row*blocks+blockIdx.x)*K+rank]=keys[i];
  }
}
template<int K>
__global__ void merge(const uint64_t* in,uint64_t* out,int in_blocks,int out_blocks) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=blockIdx.x*(2*K)+threadIdx.x*(K/128)+i;
    keys[i]=col<in_blocks*K?in[row*in_blocks*K+col]:0;
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)out[(row*out_blocks+blockIdx.x)*K+rank]=keys[i];
  }
}
template<int K,uint64_t Limit>
__global__ void finish(const uint64_t* selected,uint64_t* carry,int32_t* output,int reset) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.x;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=threadIdx.x*(K/128)+i;
    keys[i]=col<K?selected[row*K+col]:(reset?0:carry[row*K+col-K]);
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)carry[row*K+rank]=keys[i];
    // Rank-sort the selected logical positions with the same storage. Unselected
    // entries are UINT64_MAX, so only the first K outputs need publishing.
    keys[i]=rank<K && keys[i]!=0?uint32_t(~uint32_t(keys[i])):UINT64_MAX;
  }
  __syncthreads();
  // Logical positions use only 20 (token) or 17 (block) bits. Include
  // one sentinel bit so UINT64_MAX sorts after every valid position.
  static_assert(Limit==1048576 || Limit==131072);
  Sort<K>(storage).Sort(keys,0,Limit==1048576?21:18);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)output[row*K+rank]=keys[i]==UINT64_MAX?-1:int32_t(keys[i]);
  }
}
}
template<int K,uint64_t Limit>
static int32_t select(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  if(queries<1 || queries>4096 || candidates<1 || candidates>16384 || (reset!=0 && reset!=1))return cudaErrorInvalidValue;
  const int blocks=(candidates+2*K-1)/(2*K);
  const uint64_t count=uint64_t(queries)*candidates,half=uint64_t(queries)*blocks*K;
  const uint64_t bytes[]={count*4,count*8,uint64_t(queries)*K*8,half*16,uint64_t(queries)*K*4};
  const void* ptrs[]={scores,positions,carry,scratch,output};
  const int alignment[]={4,8,8,8,4};
  if(scratch_bytes<bytes[3])return cudaErrorInvalidValue;
  for(int i=0;i<5;++i) {
    if(!valid(ptrs[i],bytes[i],alignment[i]))return cudaErrorInvalidValue;
    for(int j=0;j<i;++j)if(!disjoint(ptrs[i],bytes[i],ptrs[j],bytes[j]))return cudaErrorInvalidValue;
  }
  auto cuda_stream=reinterpret_cast<cudaStream_t>(stream);
  auto* a=static_cast<uint64_t*>(scratch);auto* b=a+half;
  tiles<K,Limit><<<dim3(blocks,queries),256,0,cuda_stream>>>(scores,positions,a,candidates,blocks);
  auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
  for(int current=blocks;current>1;) {
    const int next=(current+1)/2;
    merge<K><<<dim3(next,queries),256,0,cuda_stream>>>(a,b,current,next);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    auto* tmp=a;a=b;b=tmp;current=next;
  }
  finish<K,Limit><<<queries,256,0,cuda_stream>>>(a,carry,output,reset);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_index_top512(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  return select<512,1048576>(scores,positions,carry,scratch,scratch_bytes,output,queries,candidates,reset,stream);
}
extern "C" int32_t ds41rt_v41_index_top2048_blocks(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  return select<2048,131072>(scores,positions,carry,scratch,scratch_bytes,output,queries,candidates,reset,stream);
}
