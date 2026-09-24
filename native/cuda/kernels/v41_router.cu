#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
#include "ds41rt_v41_router.h"
#ifdef DS41RT_HAVE_V41_ROUTER_AOT
#include "v41_router_dispatch.h"
extern "C" int32_t ds41rt_v41_router_scores_aot(const uint16_t*,const uint16_t*,float*,int32_t,int32_t,void*);
#else
extern "C" int32_t ds41rt_v41_router_initialize() { return 0; }
#endif
namespace {
__global__ void score_kernel(const __nv_bfloat16* hidden,const __nv_bfloat16* weight,float* scores,int experts) {
  const uint64_t row=blockIdx.y,expert=blockIdx.x;
  const int tid=threadIdx.x;
  float sum=0;
  for(int col=tid;col<5120;col+=256)
    sum=fmaf(__bfloat162float(hidden[row*5120+col]),__bfloat162float(weight[expert*5120+col]),sum);
  __shared__ float partial[256];partial[tid]=sum;__syncthreads();
  for(int stride=128;stride;stride>>=1) {if(tid<stride) partial[tid]+=partial[tid+stride];__syncthreads();}
  if(tid==0) {
    const float logit=partial[0];
    scores[row*experts+expert]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
  }
}
// Decode-sized batches: one block per expert computes every row, so each
// expert's weight row is read once instead of once per row. Each thread keeps
// the same strided column subset and the same fmaf order, and the shared tree
// is identical, so scores are bitwise equal to `score_kernel`.
constexpr int kScoreRowsMax=16;
__global__ void __launch_bounds__(256) score_rows_kernel(const __nv_bfloat16* hidden,
    const __nv_bfloat16* weight,float* scores,int experts,int rows) {
  const uint64_t expert=blockIdx.x;
  const int tid=threadIdx.x;
  float sum[kScoreRowsMax];
#pragma unroll
  for(int r=0;r<kScoreRowsMax;++r) sum[r]=0;
  // 5120/256 = 20 columns per thread, fully unrolled so every load issues
  // before the dependent fmaf chain (the chain order itself is unchanged).
  float w[20];
#pragma unroll
  for(int k=0;k<20;++k) w[k]=__bfloat162float(weight[expert*5120+tid+256*k]);
#pragma unroll
  for(int r=0;r<kScoreRowsMax;++r) {
    if(r<rows) {
      float x[20];
#pragma unroll
      for(int k=0;k<20;++k) x[k]=__bfloat162float(hidden[uint64_t(r)*5120+tid+256*k]);
#pragma unroll
      for(int k=0;k<20;++k) sum[r]=fmaf(x[k],w[k],sum[r]);
    }
  }
  __shared__ float partial[kScoreRowsMax][256];
#pragma unroll
  for(int r=0;r<kScoreRowsMax;++r) if(r<rows) partial[r][tid]=sum[r];
  __syncthreads();
  for(int stride=128;stride;stride>>=1) {
    if(tid<stride) {
#pragma unroll
      for(int r=0;r<kScoreRowsMax;++r) if(r<rows) partial[r][tid]+=partial[r][tid+stride];
    }
    __syncthreads();
  }
  if(tid<rows) {
    const float logit=partial[tid][0];
    scores[uint64_t(tid)*experts+expert]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
  }
}
// Top-k with warp-shuffle argmax rounds. The winner of each round is the
// largest corrected score with the lowest expert index on ties, exactly as in
// the shared-memory tree of `select_kernel`, so selections are identical.
template<bool TransformLogits = false>
__global__ void __launch_bounds__(512) select_fast_kernel(float* scores,const float* bias,const float* bias_vl,
    const uint8_t* image_mask,uint32_t* ids,float* routing,int experts,int topk) {
  const uint64_t row=blockIdx.x;const int tid=threadIdx.x;const int lane=tid&31,warp=tid>>5;
  if constexpr (TransformLogits) {
    if(tid<experts) {
      const float logit=scores[row*experts+tid];
      scores[row*experts+tid]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
    }
    __syncthreads();
  }
  const float* correction=image_mask && image_mask[row]?bias_vl:bias;
  float candidate=tid<experts?scores[row*experts+tid]+correction[tid]:-CUDART_INF_F;
  __shared__ float warp_value[16],selected[8];__shared__ uint32_t warp_index[16],winner_index;
  for(int rank=0;rank<topk;++rank) {
    float v=candidate;uint32_t i=tid<experts?uint32_t(tid):UINT32_MAX;
    for(int offset=16;offset;offset>>=1) {
      const float ov=__shfl_down_sync(0xffffffffu,v,offset);
      const uint32_t oi=__shfl_down_sync(0xffffffffu,i,offset);
      if(ov>v || (ov==v && oi<i)) {v=ov;i=oi;}
    }
    if(lane==0) {warp_value[warp]=v;warp_index[warp]=i;}
    __syncthreads();
    if(warp==0) {
      v=lane<16?warp_value[lane]:-CUDART_INF_F;i=lane<16?warp_index[lane]:UINT32_MAX;
      for(int offset=16;offset;offset>>=1) {
        const float ov=__shfl_down_sync(0xffffffffu,v,offset);
        const uint32_t oi=__shfl_down_sync(0xffffffffu,i,offset);
        if(ov>v || (ov==v && oi<i)) {v=ov;i=oi;}
      }
      if(lane==0) {winner_index=i;ids[row*topk+rank]=i;selected[rank]=scores[row*experts+i];}
    }
    __syncthreads();
    if(uint32_t(tid)==winner_index) candidate=-CUDART_INF_F;
  }
  __syncthreads();
  if(tid<topk) {
    float total=0;for(int j=0;j<topk;++j) total+=selected[j];
    routing[row*topk+tid]=(selected[tid]/(total+1e-20f))*1.5f;
  }
}
template<bool TransformLogits = false>
__global__ void select_kernel(float* scores,const float* bias,const float* bias_vl,
    const uint8_t* image_mask,uint32_t* ids,float* routing,int experts,int topk) {
  const uint64_t row=blockIdx.x;const int tid=threadIdx.x;
  if constexpr (TransformLogits) {
    if(tid<experts) {
      const float logit=scores[row*experts+tid];
      scores[row*experts+tid]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
    }
    __syncthreads();
  }
  const float* correction=image_mask && image_mask[row]?bias_vl:bias;
  float candidate=tid<experts?scores[row*experts+tid]+correction[tid]:-CUDART_INF_F;
  __shared__ float values[512],selected[6];__shared__ uint32_t indices[512];
  for(int rank=0;rank<topk;++rank) {
    values[tid]=candidate;indices[tid]=tid<experts?tid:UINT32_MAX;__syncthreads();
    for(int stride=256;stride;stride>>=1) {
      if(tid<stride && (values[tid+stride]>values[tid] ||
          (values[tid+stride]==values[tid] && indices[tid+stride]<indices[tid]))) {
        values[tid]=values[tid+stride];indices[tid]=indices[tid+stride];
      }
      __syncthreads();
    }
    const uint32_t winner=indices[0];
    if(tid==0) {ids[row*topk+rank]=winner;selected[rank]=scores[row*experts+winner];}
    if(tid==winner) candidate=-CUDART_INF_F;
    __syncthreads();
  }
  if(tid<topk) {
    float total=0;for(int j=0;j<topk;++j) total+=selected[j];
    routing[row*topk+tid]=(selected[tid]/(total+1e-20f))*1.5f;
  }
}
bool span(const void* p,uint64_t n,int alignment) {
  auto a=reinterpret_cast<uintptr_t>(p);return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y || y+m<=x;
}
}
extern "C" int32_t ds41rt_v41_router(const uint16_t* hidden,const uint16_t* weight,
    const float* bias,const float* bias_vl,const uint8_t* image_mask,float* scores,
    uint32_t* ids,float* routing,int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384)) return cudaErrorInvalidValue;
  const int topk=experts==128?3:6;
  const void* p[]={hidden,weight,bias,bias_vl,image_mask,scores,ids,routing};
  const uint64_t n[]={uint64_t(rows)*10240,uint64_t(experts)*10240,uint64_t(experts)*4,
      image_mask?uint64_t(experts)*4:0,image_mask?uint64_t(rows):0,uint64_t(rows)*experts*4,
      uint64_t(rows)*topk*4,uint64_t(rows)*topk*4};
  for(int i=0;i<8;++i) if(n[i] && !span(p[i],n[i],i<2?2:i==4?1:4)) return cudaErrorInvalidValue;
  for(int i=5;i<8;++i) for(int j=0;j<i;++j) if(n[j] && !disjoint(p[i],n[i],p[j],n[j])) return cudaErrorInvalidValue;
  auto s=reinterpret_cast<cudaStream_t>(stream);
#ifdef DS41RT_HAVE_V41_ROUTER_AOT
  const int threshold=experts==384?DS41RT_V41_ROUTER_E384_MIN_ROWS:DS41RT_V41_ROUTER_E128_MIN_ROWS;
  // Preserve the original ABI's weaker alignment contract for direct callers.
  if(rows>=threshold && reinterpret_cast<uintptr_t>(hidden)%16==0 &&
      reinterpret_cast<uintptr_t>(weight)%16==0 && reinterpret_cast<uintptr_t>(scores)%16==0) {
    auto status=ds41rt_v41_router_scores_aot(hidden,weight,scores,rows,experts,stream);
    if(status)return status;
    select_fast_kernel<true><<<rows,512,0,s>>>(scores,bias,bias_vl,image_mask,ids,routing,experts,topk);
    return cudaGetLastError();
  }
#endif
  // The single-pass form wins only for the 384-expert target router; the
  // 128-expert draft router keeps the per-row grid, which is faster there.
  if(experts==384 && rows<=kScoreRowsMax) {
    score_rows_kernel<<<experts,256,0,s>>>(reinterpret_cast<const __nv_bfloat16*>(hidden),
        reinterpret_cast<const __nv_bfloat16*>(weight),scores,experts,rows);
  } else {
    score_kernel<<<dim3(experts,rows),256,0,s>>>(reinterpret_cast<const __nv_bfloat16*>(hidden),
        reinterpret_cast<const __nv_bfloat16*>(weight),scores,experts);
  }
  auto status=cudaGetLastError();if(status!=cudaSuccess) return status;
  select_fast_kernel<false><<<rows,512,0,s>>>(scores,bias,bias_vl,image_mask,ids,routing,experts,topk);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_router_select_logits(float* scores,const float* bias,
    const float* bias_vl,const uint8_t* image_mask,uint32_t* ids,float* routing,
    int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384)) return cudaErrorInvalidValue;
  const int topk=experts==128?3:6;
  const void* p[]={bias,bias_vl,image_mask,scores,ids,routing};
  const uint64_t n[]={uint64_t(experts)*4,image_mask?uint64_t(experts)*4:0,
      image_mask?uint64_t(rows):0,uint64_t(rows)*experts*4,
      uint64_t(rows)*topk*4,uint64_t(rows)*topk*4};
  for(int i=0;i<6;++i) if(n[i] && !span(p[i],n[i],i==2?1:4)) return cudaErrorInvalidValue;
  for(int i=3;i<6;++i) for(int j=0;j<i;++j)
    if(n[j] && !disjoint(p[i],n[i],p[j],n[j])) return cudaErrorInvalidValue;
  select_fast_kernel<true><<<rows,512,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      scores,bias,bias_vl,image_mask,ids,routing,experts,topk);
  return cudaGetLastError();
}
