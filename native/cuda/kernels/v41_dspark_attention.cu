#include "ds41rt_v41_dspark_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "ds41rt_v41_dspark_cache.h"
#include <mma.h>
#include <stdint.h>
#include <math_constants.h>
namespace {
using namespace nvcuda;
constexpr int kSharedBytes=65536+32768+2048+192;
static_assert(sizeof(ds41rt_v41_attention_window_t)==8);
__device__ float warp_max(float x) {
  for(int n=16;n;n>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
__device__ float warp_sum(float x) {
  for(int n=16;n;n>>=1)x=__fadd_rn(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
template<int Width>
__global__ void attend(const __nv_bfloat16* query, const uint8_t* ring,
    const __nv_bfloat16* draft, const float* sink,
    const ds41rt_v41_attention_window_t* windows, __nv_bfloat16* output,int slots) {
  const int row=blockIdx.x, group=blockIdx.y, request=row/Width;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const auto window=windows[request];
  const uint64_t base=(uint64_t(row)*64+group*16)*512;
  if(window.slot>=unsigned(slots)||window.valid_rows>128) {
    for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16(0);
    return;
  }
  extern __shared__ __align__(32) unsigned char memory[];
  auto* kv=reinterpret_cast<__nv_bfloat16*>(memory);
  auto* scratch=reinterpret_cast<float*>(memory+65536);
  auto* probability=reinterpret_cast<__nv_bfloat16*>(memory+65536+32768);
  auto* maximum=reinterpret_cast<float*>(memory+65536+32768+2048);
  float* sum=maximum+16;float* rescale=sum+16;
  if(tid<16){maximum[tid]=-1e30f;sum[tid]=0;}
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
#pragma unroll
  for(int t=0;t<8;++t)wmma::fill_fragment(acc[t],0.0f);
  const int count=int(window.valid_rows)+Width;
  const bool vector_ring=reinterpret_cast<uintptr_t>(ring)%16==0;
  const bool vector_draft=reinterpret_cast<uintptr_t>(draft)%16==0;
  for(int start=0;start<count;start+=64) {
    // Concatenate logically, preserving the reference's 64-key chunk boundaries.
    // Sixteen columns per step share one E8 scale (32-column groups), so each
    // thread issues one 16-byte ring load instead of sixteen scalar loads.
    for(int i=tid;i<64*32;i+=128) {
      const int key=start+i/32,col=(i%32)*16;
      __align__(16) __nv_bfloat16 value[16];
      if(key<int(window.valid_rows)) {
        const uint8_t* row=ring+(uint64_t(window.slot)*128+key)*DS41RT_V41_DSPARK_KV_ROW_BYTES;
        __align__(16) uint8_t packed[16];
        if(vector_ring) *reinterpret_cast<uint4*>(packed)=*reinterpret_cast<const uint4*>(row+col);
        else for(int j=0;j<16;++j)packed[j]=row[col+j];
        const int exponent=int(row[512+col/32])-127;
#pragma unroll
        for(int j=0;j<16;++j) {
          __nv_fp8_e4m3 fp8; fp8.__x=packed[j];
          value[j]=__float2bfloat16_rn(ldexpf(float(fp8),exponent));
        }
      } else if(key<count) {
        const __nv_bfloat16* source=draft+(uint64_t(request)*Width+key-window.valid_rows)*512+col;
        if(vector_draft) {
          reinterpret_cast<uint4*>(value)[0]=reinterpret_cast<const uint4*>(source)[0];
          reinterpret_cast<uint4*>(value)[1]=reinterpret_cast<const uint4*>(source)[1];
        } else for(int j=0;j<16;++j)value[j]=source[j];
      } else {
#pragma unroll
        for(int j=0;j<16;++j)value[j]=__float2bfloat16(0);
      }
      uint4* destination=reinterpret_cast<uint4*>(kv+(i/32)*512+col);
      destination[0]=reinterpret_cast<const uint4*>(value)[0];
      destination[1]=reinterpret_cast<const uint4*>(value)[1];
    }
    __syncthreads();
    {
      // Each warp owns one 16-key score tile; its accumulation over k is the
      // same sequence the single-warp form used, so scores are unchanged.
      wmma::fragment<wmma::accumulator,16,16,16,float> score;
      wmma::fill_fragment(score,0.0f);
      for(int k=0;k<512;k+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::load_matrix_sync(a,query+base+k,512);
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(b,kv+warp*16*512+k,512);
        wmma::mma_sync(score,a,b,score);
      }
      wmma::store_matrix_sync(scratch+warp*16,score,64,wmma::mem_row_major);
    }
    __syncthreads();
    for(int h=warp;h<16;h+=4) {
      const float a=start+lane<count?scratch[h*64+lane]*0.04419417382415922f:-CUDART_INF_F;
      const float b=start+lane+32<count?scratch[h*64+lane+32]*0.04419417382415922f:-CUDART_INF_F;
      const float prev=maximum[h],m=fmaxf(prev,warp_max(fmaxf(a,b)));
      const float scale=expf(prev-m),pa=expf(a-m),pb=expf(b-m);
      const float total=warp_sum(pa+pb);
      probability[h*64+lane]=__float2bfloat16_rn(pa);
      probability[h*64+lane+32]=__float2bfloat16_rn(pb);
      if(lane==0){maximum[h]=m;rescale[h]=scale;sum[h]=fmaf(sum[h],scale,total);}
    }
    __syncthreads();
    // WMMA accumulator lane ownership is opaque: rescale through shared storage
    // instead of depending on an undocumented fragment-to-head mapping.
#pragma unroll
    for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],512,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<16*512;i+=128)scratch[i]*=rescale[i/512];
    __syncthreads();
#pragma unroll
    for(int t=0;t<8;++t)wmma::load_matrix_sync(acc[t],scratch+warp*128+t*16,512,wmma::mem_row_major);
    for(int k=0;k<64;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> p;
      wmma::load_matrix_sync(p,probability+k,64);
#pragma unroll
      for(int t=0;t<8;++t) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> v;
        wmma::load_matrix_sync(v,kv+k*512+warp*128+t*16,512);
        wmma::mma_sync(acc[t],p,v,acc[t]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],512,wmma::mem_row_major);
  if(tid<16)sum[tid]+=expf(sink[group*16+tid]-maximum[tid]);
  __syncthreads();
  for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16_rn(scratch[i]/sum[i/512]);
}
bool span(const void* ptr,uint64_t n,uint32_t a) {
  const auto p=reinterpret_cast<uintptr_t>(ptr);return p&&p%a==0&&p<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y||y+m<=x;
}
}
namespace {
template<int Width>
int32_t launch_attention(const uint16_t* query,const uint8_t* ring,
    const uint16_t* draft,const float* sink,const ds41rt_v41_attention_window_t* windows,
    uint16_t* output,int32_t requests,int32_t slots,void* stream) {
  if(requests<1||requests>16||slots<1||slots>16)return cudaErrorInvalidValue;
  const uint64_t q=uint64_t(requests)*Width*64*512*2,
      r=uint64_t(slots)*128*DS41RT_V41_DSPARK_KV_ROW_BYTES,d=uint64_t(requests)*Width*512*2;
  if(!span(output,q,32))return cudaErrorInvalidValue;
  const void* inputs[]={query,ring,draft,sink,windows};const uint64_t sizes[]={q,r,d,256,uint64_t(requests)*8};
  const uint32_t align[]={32,1,2,4,4};
  for(int i=0;i<5;++i)if(!span(inputs[i],sizes[i],align[i])||!disjoint(inputs[i],sizes[i],output,q))return cudaErrorInvalidValue;
  attend<Width><<<dim3(requests*Width,4),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),ring,
      reinterpret_cast<const __nv_bfloat16*>(draft),sink,windows,reinterpret_cast<__nv_bfloat16*>(output),slots);
  return cudaGetLastError();
}
}
extern "C" int32_t ds41rt_v41_dspark_attention_initialize(void) {
  return cudaFuncSetAttribute(attend<5>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
}
extern "C" int32_t ds41rt_v41_dspark_attention_initialize_width(int32_t width) {
  if(width==5)return ds41rt_v41_dspark_attention_initialize();
  if(width==7)return cudaFuncSetAttribute(attend<7>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  return cudaErrorInvalidValue;
}
extern "C" int32_t ds41rt_v41_dspark_attention_fp8_width(const uint16_t* query,const uint8_t* ring,
    const uint16_t* draft,const float* sink,const ds41rt_v41_attention_window_t* windows,
    uint16_t* output,int32_t requests,int32_t slots,int32_t width,void* stream) {
  if(width==5)return launch_attention<5>(query,ring,draft,sink,windows,output,requests,slots,stream);
  if(width==7)return launch_attention<7>(query,ring,draft,sink,windows,output,requests,slots,stream);
  return cudaErrorInvalidValue;
}
extern "C" int32_t ds41rt_v41_dspark_attention_fp8(const uint16_t* query,const uint8_t* ring,
    const uint16_t* draft,const float* sink,const ds41rt_v41_attention_window_t* windows,
    uint16_t* output,int32_t requests,int32_t slots,void* stream) {
  return launch_attention<5>(query,ring,draft,sink,windows,output,requests,slots,stream);
}
