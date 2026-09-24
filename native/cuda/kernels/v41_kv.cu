#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include "ds41rt_v41_kv.h"
namespace {
bool valid(const void* p,uint64_t n,int alignment) {
  auto a=reinterpret_cast<uintptr_t>(p);return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+na<=y || y+nb<=x;
}
template<bool Compressed = false>
__global__ void pack(const __nv_bfloat16* input,const float* frequencies,uint8_t* values,uint8_t* scales) {
  const uint64_t row=blockIdx.x,pair=row*256+threadIdx.x;
  const int col=threadIdx.x*2;
  float a=__bfloat162float(input[pair*2]),b=__bfloat162float(input[pair*2+1]);
  if(frequencies && col>=448) {
    const uint64_t f=row*64+col-448;
    const float c=frequencies[f],s=frequencies[f+1];
    const float x=__bfloat162float(__float2bfloat16_rn(fmaf(a,c,-__fmul_rn(b,s))));
    const float y=__bfloat162float(__float2bfloat16_rn(fmaf(b,c,__fmul_rn(a,s))));
    a=x;b=y;
  }
  float maximum=fmaxf(fabsf(a),fabsf(b));
  #pragma unroll
  for(int offset=Compressed?4:8;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset,Compressed?8:16));
  if constexpr(Compressed) {
    // Training format: E2M1 values, E4M3 scale per 16 coordinates.
    const float raw_scale=__fdiv_rn(fmaxf(maximum,6.0f*0x1p-9f),6.0f);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x=__nv_cvt_float_to_fp8(raw_scale,__NV_SATFINITE,__NV_E4M3);
    const float scale=float(encoded_scale);
    if(threadIdx.x%8==0)scales[pair/8]=encoded_scale.__x;
    values[pair]=__nv_cvt_float2_to_fp4x2(
        make_float2(__fdiv_rn(a,scale),__fdiv_rn(b,scale)),__NV_E2M1,cudaRoundNearest);
    return;
  }
  const uint32_t bits=__float_as_uint(__fmul_rn(fmaxf(maximum,1e-4f),1.0f/448.0f));
  const uint32_t exponent=(bits>>23)+((bits&0x7fffffu)!=0);
  const float scale=__uint_as_float(exponent<<23);
  if(threadIdx.x%16==0)scales[pair/16]=uint8_t(exponent);
  const uint16_t v=__nv_cvt_float2_to_fp8x2(make_float2(__fdiv_rn(a,scale),__fdiv_rn(b,scale)),__NV_SATFINITE,__NV_E4M3);
  values[pair*2]=uint8_t(v);values[pair*2+1]=uint8_t(v>>8);
}
template<bool Compressed = false>
__global__ void store(const uint8_t* values,const uint8_t* scales,const uint64_t* destinations,
    uint8_t* cache,uint8_t* cache_scales,uint64_t capacity) {
  const uint64_t row=blockIdx.x,dst=destinations[row];const int t=threadIdx.x;
  if(dst>=capacity)return;
  constexpr int value_bytes=Compressed?256:512,scale_bytes=Compressed?32:16;
  cache[dst*value_bytes+t]=values[row*value_bytes+t];
  if constexpr(!Compressed)cache[dst*512+t+256]=values[row*512+t+256];
  if(t<scale_bytes)cache_scales[dst*scale_bytes+t]=scales[row*scale_bytes+t];
}
}
namespace {
// One launch stores every window layer's accepted rows and publishes the
// per-slot ends. Row copies match `store<false>` byte for byte.
__global__ void store_layers(const ds41rt_v41_kv_store_layer_t* layers,int chunks) {
  const auto layer=layers[blockIdx.y];
  const uint64_t row=blockIdx.x,dst=layer.destinations[row];const int t=threadIdx.x;
  if(dst<layer.capacity) {
    layer.cache[dst*512+t]=layer.values[row*512+t];
    layer.cache[dst*512+t+256]=layer.values[row*512+t+256];
    if(t<16)layer.cache_scales[dst*16+t]=layer.scales[row*16+t];
  }
  if(row==0 && t<chunks)layer.ends[layer.end_pairs[2*t]]=layer.end_pairs[2*t+1];
}
}
extern "C" int32_t ds41rt_v41_kv_store_layers(const void* table,int32_t layers,int32_t rows,
    int32_t chunks,void* stream) {
  if(!table || reinterpret_cast<uintptr_t>(table)%16 || layers<1 || layers>64 || rows<1 || rows>4096
      || chunks<1 || chunks>256)
    return cudaErrorInvalidValue;
  store_layers<<<dim3(rows,layers),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const ds41rt_v41_kv_store_layer_t*>(table),chunks);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_kv_pack(const uint16_t* input,const float* frequencies,
    uint8_t* values,uint8_t* scales,int32_t rows,void* stream) {
  if(rows<1 || rows>4096)return cudaErrorInvalidValue;
  const uint64_t r=rows,in=r*1024,out=r*512,s=r*16,f=r*256;
  if(!valid(input,in,2) || !valid(values,out,1) || !valid(scales,s,1) ||
      !disjoint(input,in,values,out) || !disjoint(input,in,scales,s) || !disjoint(values,out,scales,s) ||
      (frequencies && (!valid(frequencies,f,4) || !disjoint(frequencies,f,values,out) || !disjoint(frequencies,f,scales,s))))
    return cudaErrorInvalidValue;
  pack<false><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const __nv_bfloat16*>(input),frequencies,values,scales);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_kv_store(const uint8_t* values,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream) {
  if(rows<1 || rows>4096 || capacity<1 || capacity>67108864ull)return cudaErrorInvalidValue;
  const uint64_t r=rows;const void* p[]={values,scales,destinations,cache,cache_scales};
  const uint64_t n[]={r*512,r*16,r*8,capacity*512,capacity*16};const int align[]={1,1,8,1,1};
  for(int i=0;i<5;++i) {
    if(!valid(p[i],n[i],align[i]))return cudaErrorInvalidValue;
    if(i>=3)for(int j=0;j<i;++j)if(!disjoint(p[i],n[i],p[j],n[j]))return cudaErrorInvalidValue;
  }
  store<false><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(values,scales,destinations,cache,cache_scales,capacity);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_compressed_kv_pack(const uint16_t* input,const float* frequencies,
    uint8_t* values,uint8_t* scales,int32_t rows,void* stream) {
  if(rows<1 || rows>4096)return cudaErrorInvalidValue;
  const uint64_t r=rows,in=r*1024,out=r*256,s=r*32,f=r*256;
  if(!valid(input,in,2) || !valid(values,out,1) || !valid(scales,s,1) ||
      !disjoint(input,in,values,out) || !disjoint(input,in,scales,s) || !disjoint(values,out,scales,s) ||
      (frequencies && (!valid(frequencies,f,4) || !disjoint(frequencies,f,values,out) || !disjoint(frequencies,f,scales,s))))
    return cudaErrorInvalidValue;
  pack<true><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const __nv_bfloat16*>(input),frequencies,values,scales);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_compressed_kv_store(const uint8_t* values,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream) {
  if(rows<1 || rows>4096 || capacity<1 || capacity>67108864ull)return cudaErrorInvalidValue;
  const uint64_t r=rows;const void* p[]={values,scales,destinations,cache,cache_scales};
  const uint64_t n[]={r*256,r*32,r*8,capacity*256,capacity*32};const int align[]={1,1,8,1,1};
  for(int i=0;i<5;++i) {
    if(!valid(p[i],n[i],align[i]))return cudaErrorInvalidValue;
    if(i>=3)for(int j=0;j<i;++j)if(!disjoint(p[i],n[i],p[j],n[j]))return cudaErrorInvalidValue;
  }
  store<true><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(values,scales,destinations,cache,cache_scales,capacity);
  return cudaGetLastError();
}
