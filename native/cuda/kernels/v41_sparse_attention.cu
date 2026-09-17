#include "ds41rt_v41_sparse_attention.h"
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
#include "ds41rt_v41_attention_aot_internal.h"
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <math_constants.h>
namespace {
using namespace nvcuda;
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
__device__ __forceinline__ uint32_t packed_fp8_pair(uint16_t input,uint32_t factors) {
  uint32_t output;
  asm("{ .reg .b32 values; cvt.rn.bf16x2.e4m3x2 values, %1; mul.bf16x2 %0, values, %2; }"
      : "=r"(output) : "h"(input), "r"(factors));
  return output;
}

#endif
__device__ __forceinline__ uint16_t fp4_bf16_bits(unsigned value) {
  const unsigned magnitude=value&7;
  return uint16_t(((value&8)<<12) | (magnitude<2?(magnitude?0x3f00:0):0x3f00+magnitude*64));
}
__device__ __forceinline__ uint64_t packed_fp4_quad(uint16_t input,uint8_t scale) {
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
  uint32_t low,high;
  const uint16_t packed_low=input&255,packed_high=input>>8;
  asm("{ .reg .b8 x; cvt.u8.u16 x, %1; cvt.rn.bf16x2.e2m1x2 %0, x; }" : "=r"(low) : "h"(packed_low));
  asm("{ .reg .b8 x; cvt.u8.u16 x, %1; cvt.rn.bf16x2.e2m1x2 %0, x; }" : "=r"(high) : "h"(packed_high));
  uint32_t factors,a,b;
  const uint16_t scales=uint16_t(scale)|(uint16_t(scale)<<8);
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(factors) : "h"(scales));
  asm("mul.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(low), "r"(factors));
  asm("mul.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(high), "r"(factors));
  return uint64_t(a)|(uint64_t(b)<<32);
#else
  __nv_fp8_e4m3 f;f.__x=scale;
  uint64_t result=0;
#pragma unroll
  for(int j=0;j<4;++j) {
    const float value=__bfloat162float(__ushort_as_bfloat16(fp4_bf16_bits((input>>(j*4))&15)));
    result|=uint64_t(__bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(value,float(f)))))<<(j*16);
  }
  return result;
#endif
}
// Pad shared rows to distribute WMMA traffic across memory banks.
constexpr int kKvStride=520, kOutputStride=516, kProbabilityStride=80;
constexpr int kKvBytes=64*kKvStride*2, kOutputBytes=16*kOutputStride*4;
constexpr int kSharedBytes=kKvBytes+kOutputBytes+192+512;
// A head group needs only scores/probabilities during the KV loop.
constexpr int kScoreBytes=16*64*4;
constexpr int kGroupedSharedBytes=kKvBytes+2*kScoreBytes+2*192+512;
constexpr int kFourSharedBytes=kKvBytes+4*kScoreBytes+4*192+512;
static_assert(2*kOutputBytes<=kKvBytes, "retired KV must fit two output groups");

__device__ float warp_max(float x) {
  for(int n=16;n;n>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
__device__ float warp_sum(float x) {
  for(int n=16;n;n>>=1)x=__fadd_rn(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
// Top two bits distinguish ring, private window, paged source and private source.
// Invalid rows never dereference a value or scale pointer.
__device__ __forceinline__ uint64_t locate(const ds41rt_v41_sparse_kv_t& v,const uint64_t* m,
    const int32_t* selected,int key,int width,uint64_t window_begin) {
  if(key<width) {
    const uint64_t begin=m[3]+1>128?m[3]+1-128:0,pos=begin+key;
    if(pos<window_begin || pos>m[3])return UINT64_MAX;
    return pos<m[0]?pos%128:((1ull<<62)|(m[1]+pos-m[0]));
  }
  const int32_t id=selected[key-width];
  if(id<0 || uint64_t(id)>=m[5])return UINT64_MAX;
  const uint64_t pos=uint64_t(id);
  if(pos>=m[6]) {
    if(pos-m[6]>=m[7])return UINT64_MAX;
    return (3ull<<62)|(m[8]+(pos-m[6])*m[9]);
  }
  if(pos>=uint64_t(v.page_stride)*256)return UINT64_MAX;
  const uint64_t physical=uint64_t(v.pages[pos/256])*256+pos%256;
  return physical<v.source_capacity?((2ull<<62)|physical):UINT64_MAX;
}
// Grid-constant descriptor avoids a per-thread copy for dynamic source indexing.
template<bool Split,int Groups=1,bool SourceFP4=false,bool Batched=false,int Heads=64>
__global__ __launch_bounds__(128*Groups,1) void attend(const __nv_bfloat16* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,__nv_bfloat16* output,
    int width,const __grid_constant__ ds41rt_v41_sparse_kv_t uniform_view,float* partial,
    const uint64_t* window_begins,const ds41rt_v41_sparse_kv_t* row_views=nullptr) {
  const auto& v=Batched?row_views[blockIdx.x]:uniform_view;
  // Keep each query's softmax tile boundaries independent of other proposal
  // rows. A batch-wide maximum moves compressed keys between BF16 probability
  // tiles when draft length changes, perturbing already-causal output rows.
  if(width==0) {
    const uint64_t position=metadata[uint64_t(blockIdx.x)*10+3];
    width=position<127?int(position+1):128;
  }
  static_assert(Groups==1 || Groups==2 || Groups==4);
  static_assert((Heads==32 || Heads==64) && Groups*16<=Heads);
  static_assert(!Split || Groups==1);
  // Each 128-thread group owns 16 heads; all groups share one decoded KV tile.
  const int subgroup=Groups==1?0:threadIdx.x/128;
  const int row=blockIdx.x,group=blockIdx.y*Groups+subgroup;
  const uint64_t partial_base=((uint64_t(row)*gridDim.z+blockIdx.z)*Heads+group*16)*514;
  const int global_tid=threadIdx.x,tid=Groups==1?global_tid:global_tid%128,warp=tid/32,lane=tid%32;
  const uint64_t base=(uint64_t(row)*Heads+group*16)*512;
  const uint64_t* m=metadata+uint64_t(row)*10;
  const uint64_t window_begin=window_begins?window_begins[row]:0;
  bool valid=window_begin<=m[0] && m[0]==*v.window_end && m[0]<=1048576 && m[2]>0 &&
    m[2]<=1048576-m[0] && m[1]<=v.window_proposal_capacity &&
    m[2]<=v.window_proposal_capacity-m[1] && m[3]>=m[0] && m[3]-m[0]<m[2] &&
    uint64_t(width)>=(m[3]+1<128?m[3]+1:128);
  // A committed-only causal prefix remains valid when another encoder chunk
  // appends rows. Private overlays still require their exact committed boundary.
  if(valid && v.compressed)valid=m[4]==0 &&
    (m[7]==0?m[6]<=*v.source_end:m[6]==*v.source_end) && m[6]<=1048576 &&
    m[7]<=1048576-m[6] && m[5]<=m[6]+m[7] && (m[9]==1 || m[9]==2) &&
    m[8]<=v.source_proposal_capacity && (!m[7] || (m[8]<v.source_proposal_capacity &&
      m[7]-1<=(v.source_proposal_capacity-1-m[8])/m[9]));
  if(!valid) {
    if constexpr(Split) {
      for(int i=tid;i<16*514;i+=128)partial[partial_base+i]=i%514==513?-1.0f:0.0f;
    } else {
      for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16(0);
    }
    return;
  }
  extern __shared__ __align__(32) unsigned char memory[];
  auto* kv=reinterpret_cast<__nv_bfloat16*>(memory);
  auto* scratch=reinterpret_cast<float*>(memory+kKvBytes+(Groups==1?0:subgroup*kScoreBytes));
  // Scores, accumulator rescaling and PV probabilities have disjoint lifetimes.
  auto* probability=reinterpret_cast<__nv_bfloat16*>(scratch);
  auto* maximum=reinterpret_cast<float*>(memory+kKvBytes+(Groups==1?kOutputBytes:Groups*kScoreBytes+subgroup*192));
  float* sum=maximum+16;float* rescale=sum+16;
  if(tid<16){maximum[tid]=-1e30f;sum[tid]=0;}
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
#pragma unroll
  for(int t=0;t<8;++t)wmma::fill_fragment(acc[t],0.0f);
  auto* refs=reinterpret_cast<uint64_t*>(memory+kKvBytes+(Groups==1?kOutputBytes+192:Groups*kScoreBytes+Groups*192));
  const int count=width+(v.compressed?512:0);
  // The first valid tile starts from zero accumulators; no rescale is needed.
  bool empty=true;
  const int tiles=(count+63)/64, per_part=Split?(tiles+gridDim.z-1)/gridDim.z:tiles;
  const int first=Split?blockIdx.z*per_part*64:0;
  const int last=Split?min(count,int(blockIdx.z+1)*per_part*64):count;
  for(int start=first;start<last;start+=64) {
    if(global_tid<64)refs[global_tid]=start+global_tid<count?locate(v,m,
      v.compressed?selected+uint64_t(row)*512:nullptr,start+global_tid,width,window_begin):UINT64_MAX;
    // An entirely masked tile contributes zero probability and leaves both
    // online-softmax state and accumulators unchanged. Vote over the same
    // resolved references used below; valid entries may occur after empty tiles.
    if(!__syncthreads_or(global_tid<64 && refs[global_tid]!=UINT64_MAX))continue;
    // A warp stages four contiguous FP8 values per lane, reusing each
    // resolved key pointer. Preserve unaligned byte-addressed ABI inputs.
    for(int key=global_tid/32;key<64;key+=4*Groups) {
      const uint64_t ref=refs[key];
      for(int block=0;block<4;++block) {
        const int col=block*128+lane*4;
        uint64_t packed=0;
        if(ref!=UINT64_MAX) {
          const int tag=ref>>62;const uint64_t physical=ref&((1ull<<62)-1);
          if(SourceFP4 && tag>=2) {
            const uint8_t* source=v.values[tag]+physical*256+col/2;
            const uint16_t bytes=(reinterpret_cast<uintptr_t>(source)&1)?
                uint16_t(source[0])|(uint16_t(source[1])<<8):*reinterpret_cast<const uint16_t*>(source);
            packed=packed_fp4_quad(bytes,v.scales[tag][physical*32+col/16]);
          } else {
          const uint8_t* source=v.values[tag]+physical*512+col;
          uint32_t bytes;
          if((reinterpret_cast<uintptr_t>(source)&3)==0)
            bytes=*reinterpret_cast<const uint32_t*>(source);
          else bytes=uint32_t(source[0])|(uint32_t(source[1])<<8)|
              (uint32_t(source[2])<<16)|(uint32_t(source[3])<<24);
          const uint8_t exponent=v.scales[tag][physical*16+col/32];
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
          // Match the existing scale decoding, including zero and 255.
          const uint32_t factor=exponent?(uint32_t(exponent)<<7):0x40;
          const uint32_t factors=factor|(factor<<16);
          packed=uint64_t(packed_fp8_pair(uint16_t(bytes),factors))|
              (uint64_t(packed_fp8_pair(uint16_t(bytes>>16),factors))<<32);
#else
          const float scale=exponent==0?0x1p-127f:__uint_as_float(uint32_t(exponent)<<23);
#pragma unroll
          for(int j=0;j<4;++j) {
            __nv_fp8_e4m3 f;f.__x=uint8_t(bytes>>(j*8));
            const auto value=__float2bfloat16_rn(__fmul_rn(float(f),scale));
            packed|=uint64_t(__bfloat16_as_ushort(value))<<(j*16);
          }
#endif
          }
        }
        *reinterpret_cast<uint64_t*>(kv+key*kKvStride+col)=packed;
      }
    }
    __syncthreads();
    // Each warp computes one 16-key score tile in the original K order.
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> scores;
      wmma::fill_fragment(scores,0.0f);
      for(int k=0;k<512;k+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(a,query+base+k,512);
        wmma::load_matrix_sync(b,kv+warp*16*kKvStride+k,kKvStride);
        wmma::mma_sync(scores,a,b,scores);
      }
      wmma::store_matrix_sync(scratch+warp*16,scores,64,wmma::mem_row_major);
    }
    __syncthreads();
    // Explicit carries avoid a dynamically indexed array spilling to local memory.
    __nv_bfloat16 p0a{},p0b{},p1a{},p1b{},p2a{},p2b{},p3a{},p3b{};
    for(int h=warp;h<16;h+=4) {
      const float a=refs[lane]!=UINT64_MAX?scratch[h*64+lane]*0.04419417382415922f:-CUDART_INF_F;
      const float b=refs[lane+32]!=UINT64_MAX?scratch[h*64+lane+32]*0.04419417382415922f:-CUDART_INF_F;
      const float prev=maximum[h],m=fmaxf(prev,warp_max(fmaxf(a,b)));
      const float scale=expf(prev-m),pa=expf(a-m),pb=expf(b-m);
      const float total=warp_sum(pa+pb);
      const auto a16=__float2bfloat16_rn(pa),b16=__float2bfloat16_rn(pb);
      if(h<4){p0a=a16;p0b=b16;}
      else if(h<8){p1a=a16;p1b=b16;}
      else if(h<12){p2a=a16;p2b=b16;}
      else {p3a=a16;p3b=b16;}
      if(lane==0){maximum[h]=m;rescale[h]=scale;sum[h]=fmaf(sum[h],scale,total);}
    }
    __syncthreads();
    if(!empty) {
      // Load a multiplier through the same accumulator layout. Matching
      // fragment elements have matching coordinates without a lane-map assumption.
      for(int i=tid;i<16*16;i+=128)scratch[i]=rescale[i/16];
      __syncthreads();
      wmma::fragment<wmma::accumulator,16,16,16,float> factors;
      wmma::load_matrix_sync(factors,scratch,16,wmma::mem_row_major);
#pragma unroll
      for(int t=0;t<8;++t) {
#pragma unroll
        for(int i=0;i<acc[t].num_elements;++i)
          acc[t].x[i]=__fmul_rn(acc[t].x[i],factors.x[i]);
      }
      __syncthreads();
    }
    empty=false;
    // Accumulators are ready before probabilities reuse scratch.
    probability[warp*kProbabilityStride+lane]=p0a;
    probability[warp*kProbabilityStride+lane+32]=p0b;
    probability[(warp+4)*kProbabilityStride+lane]=p1a;
    probability[(warp+4)*kProbabilityStride+lane+32]=p1b;
    probability[(warp+8)*kProbabilityStride+lane]=p2a;
    probability[(warp+8)*kProbabilityStride+lane+32]=p2b;
    probability[(warp+12)*kProbabilityStride+lane]=p3a;
    probability[(warp+12)*kProbabilityStride+lane+32]=p3b;
    __syncthreads();
    for(int k=0;k<64;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> p;
      wmma::load_matrix_sync(p,probability+k,kProbabilityStride);
#pragma unroll
      for(int t=0;t<8;++t) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> v;
        wmma::load_matrix_sync(v,kv+k*kKvStride+warp*128+t*16,kKvStride);
        wmma::mma_sync(acc[t],p,v,acc[t]);
      }
    }
    __syncthreads();
  }
  if constexpr(Groups==4) {
    // Two groups fit in the retired KV tile. Drain both pairs before reuse.
#pragma unroll
    for(int first_group=0;first_group<4;first_group+=2) {
      const bool active=subgroup>=first_group && subgroup<first_group+2;
      auto* final_output=reinterpret_cast<float*>(memory+(subgroup%2)*kOutputBytes);
      if(active) {
#pragma unroll
        for(int t=0;t<8;++t)wmma::store_matrix_sync(final_output+warp*128+t*16,acc[t],kOutputStride,wmma::mem_row_major);
        if(tid<16)sum[tid]+=expf(sink[group*16+tid]-maximum[tid]);
      }
      __syncthreads();
      if(active)for(int i=tid;i<16*512;i+=128)
        output[base+i]=__float2bfloat16_rn(final_output[(i/512)*kOutputStride+i%512]/sum[i/512]);
      __syncthreads();
    }
  } else {
    if constexpr(Groups>1) {
      // KV is dead after the final PV. Reuse its storage for both output groups.
      __syncthreads();
      scratch=reinterpret_cast<float*>(memory+subgroup*kOutputBytes);
    }
#pragma unroll
    for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],kOutputStride,wmma::mem_row_major);
    if(tid<16) {
      if constexpr(Split) {
        partial[partial_base+tid*514+512]=maximum[tid];
        partial[partial_base+tid*514+513]=sum[tid];
      } else sum[tid]+=expf(sink[group*16+tid]-maximum[tid]);
    }
    __syncthreads();
    for(int i=tid;i<16*512;i+=128) {
      if constexpr(Split)partial[partial_base+(i/512)*514+i%512]=scratch[(i/512)*kOutputStride+i%512];
      else output[base+i]=__float2bfloat16_rn(scratch[(i/512)*kOutputStride+i%512]/sum[i/512]);
    }
  }
}
template<int Heads=64> __global__ void merge(const float* partial,const float* sink,__nv_bfloat16* output,int parts) {
  const int row=blockIdx.x,head=blockIdx.y,col=threadIdx.x;
  const uint64_t b=(uint64_t(row)*parts*Heads+head)*514;
  const uint64_t dest=(uint64_t(row)*Heads+head)*512;
  // Invalid proposal metadata must yield zero even for sinks whose exponential
  // underflows. A negative normalizer is reserved for this whole-row failure.
  if(partial[b+513]<0) {
    output[dest+col]=__float2bfloat16(0);
    output[dest+col+256]=__float2bfloat16(0);
    return;
  }
  float maximum=-1e30f;
  for(int p=0;p<parts;++p)maximum=fmaxf(maximum,partial[b+uint64_t(p)*Heads*514+512]);
  float sum=0,a=0,c=0;
  for(int p=0;p<parts;++p) {
    const uint64_t i=b+uint64_t(p)*Heads*514;
    const float factor=expf(partial[i+512]-maximum);
    sum=fmaf(partial[i+513],factor,sum);
    a=fmaf(partial[i+col],factor,a);
    c=fmaf(partial[i+col+256],factor,c);
  }
  sum+=expf(sink[head]-maximum);
  output[dest+col]=__float2bfloat16_rn(a/sum);
  output[dest+col+256]=__float2bfloat16_rn(c/sum);
}
bool span(const void* ptr,uint64_t n,uint32_t a) {
  const auto p=reinterpret_cast<uintptr_t>(ptr);return p&&p%a==0&&p<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y||y+m<=x;
}
}
template<bool FP4,int Heads=64> static int32_t initialize_format() {
  const auto status=cudaFuncSetAttribute(attend<false,1,FP4,false,Heads>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  if(status!=cudaSuccess)return status;
  const auto split=cudaFuncSetAttribute(attend<true,1,FP4,false,Heads>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  if(split!=cudaSuccess)return split;
  const auto batch=cudaFuncSetAttribute(attend<true,1,FP4,true,Heads>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  if(batch!=cudaSuccess)return batch;
  const auto pair=cudaFuncSetAttribute(attend<false,2,FP4,false,Heads>,cudaFuncAttributeMaxDynamicSharedMemorySize,kGroupedSharedBytes);
  if(pair!=cudaSuccess)return pair;
  if constexpr(Heads==64)return cudaFuncSetAttribute(attend<false,4,FP4>,cudaFuncAttributeMaxDynamicSharedMemorySize,kFourSharedBytes);
  return cudaSuccess;
}
extern "C" int32_t ds41rt_v41_sparse_attention_initialize(void) {
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
  const auto aot_status=ds41rt_v41_attention_aot_initialize();
  if(aot_status!=cudaSuccess)return aot_status;
#endif
  const auto status=initialize_format<false>();
  return status==cudaSuccess?initialize_format<true>():status;
}
template<bool FP4,int Heads=64> static int32_t dispatch_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t& v,void* stream,float* partial,
    int parts,const uint64_t* window_begins) {
  if(partial) {
    attend<true,1,FP4,false,Heads><<<dim3(rows,Heads/16,parts),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,partial,window_begins);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)return status;
    merge<Heads><<<dim3(rows,Heads),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      partial,sink,reinterpret_cast<__nv_bfloat16*>(output),parts);
  } else if(rows>=256 && Heads==64) {
    attend<false,4,FP4><<<dim3(rows,1),512,kFourSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,nullptr,window_begins);
  } else if(rows>=128) {
    attend<false,2,FP4,false,Heads><<<dim3(rows,Heads/32),256,kGroupedSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,nullptr,window_begins);
  } else {
    attend<false,1,FP4,false,Heads><<<dim3(rows,Heads/16),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,nullptr,window_begins);
  }
  return cudaGetLastError();
}
template<int Heads=64> static int32_t validate_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,float* partial,uint64_t scratch_bytes,int parts,const uint64_t* window_begins=nullptr) {
  if(!view || rows<1 || rows>4096 || window_width<0 || window_width>128)return cudaErrorInvalidValue;
  const auto v=*view;
  if(v.compressed>2 || v.window_proposal_capacity<1 || v.window_proposal_capacity>4096 ||
    (v.compressed && (v.source_capacity<1 || v.source_capacity>67108864ull ||
     v.source_proposal_capacity<1 || v.source_proposal_capacity>4096 ||
     v.page_stride<1 || v.page_stride>4096)))return cudaErrorInvalidValue;
  if(parts<1 || parts>10)return cudaErrorInvalidValue;
  const uint64_t required=uint64_t(rows)*parts*Heads*514*sizeof(float);
  const uint64_t q=uint64_t(rows)*Heads*512*2;
  if(partial && (scratch_bytes<required || !span(partial,required,4) ||
      !disjoint(partial,required,output,q)))return cudaErrorInvalidValue;
  if(!span(output,q,32))return cudaErrorInvalidValue;
  if(window_begins && (!span(window_begins,uint64_t(rows)*8,8) ||
      !disjoint(window_begins,uint64_t(rows)*8,output,q) ||
      (partial && !disjoint(window_begins,uint64_t(rows)*8,partial,required))))return cudaErrorInvalidValue;
  const void* inputs[]={query,sink,metadata,v.window_end,selected,v.pages,v.source_end};
  const uint64_t sizes[]={q,Heads*4,uint64_t(rows)*80,8,uint64_t(rows)*2048,uint64_t(v.page_stride)*4,8};
  const uint32_t align[]={32,4,8,8,4,4,8};
  for(int i=0;i<(v.compressed?7:4);++i)
    if(!span(inputs[i],sizes[i],align[i]) || !disjoint(inputs[i],sizes[i],output,q) ||
      (partial && !disjoint(inputs[i],sizes[i],partial,required)))return cudaErrorInvalidValue;
  const uint64_t capacity[]={128,v.window_proposal_capacity,v.source_capacity,v.source_proposal_capacity};
  for(int i=0;i<(v.compressed?4:2);++i) {
    const uint64_t values=capacity[i]*((v.compressed==2 && i>=2)?256:512);
    const uint64_t scales=capacity[i]*((v.compressed==2 && i>=2)?32:16);
    if(!span(v.values[i],values,1) || !span(v.scales[i],scales,1) ||
      !disjoint(v.values[i],values,output,q) || !disjoint(v.scales[i],scales,output,q) ||
      (partial && (!disjoint(v.values[i],values,partial,required) ||
                   !disjoint(v.scales[i],scales,partial,required))))return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
// Small lane-local descriptor staging; all storage comes from the existing
// validated FP32 split workspace. No allocation or host metadata download.
__global__ void stage_aot_view(ds41rt_v41_sparse_kv_t view,
    ds41rt_v41_sparse_kv_t* views,uint64_t* zero_bounds,int rows) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<rows){views[row]=view;zero_bounds[row]=0;}
}
static bool aot_aligned(const ds41rt_v41_sparse_kv_t& v) {
  for(int i=0;i<4;++i)
    if((reinterpret_cast<uintptr_t>(v.values[i]) | reinterpret_cast<uintptr_t>(v.scales[i]))%16)return false;
  return true;
}
#endif
template<int Heads=64> static int32_t launch_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,float* partial,
    uint64_t scratch_bytes,int parts,const uint64_t* window_begins=nullptr) {
  const auto status=validate_attention<Heads>(query,sink,metadata,selected,output,rows,
      window_width,view,stream,partial,scratch_bytes,parts,window_begins);
  if(status!=cudaSuccess)return status;
  const auto& v=*view;
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
  if(v.compressed==2 && partial && parts==10 && rows<=64 &&
      (window_width==0 || window_width==128) && aot_aligned(v) &&
      reinterpret_cast<uintptr_t>(partial)%16==0) {
    auto* bytes=reinterpret_cast<uint8_t*>(partial);
    auto* lses=bytes+uint64_t(rows)*Heads*10*512*2;
    auto* views=reinterpret_cast<ds41rt_v41_sparse_kv_t*>(lses+uint64_t(rows)*Heads*10*4);
    auto* zero_bounds=reinterpret_cast<uint64_t*>(views+rows);
    // BF16 partials, LSEs and descriptors fit the validated FP32 split allocation.
    stage_aot_view<<<(rows+31)/32,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(v,views,zero_bounds,rows);
    const auto staged=cudaGetLastError();
    if(staged!=cudaSuccess)return staged;
    const auto launch=Heads==64?ds41rt_v41_attention_aot_launch:ds41rt_v41_attention_heads32_aot_launch;
    return launch(query,views,metadata,selected,
        window_begins?window_begins:zero_bounds,sink,bytes,lses,output,rows,stream);
  }
#endif
  return v.compressed==2?
      dispatch_attention<true,Heads>(query,sink,metadata,selected,output,rows,window_width,v,stream,partial,parts,window_begins):
      dispatch_attention<false,Heads>(query,sink,metadata,selected,output,rows,window_width,v,stream,partial,parts,window_begins);
}

extern "C" int32_t ds41rt_v41_sparse_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream) {
  return launch_attention(query,sink,metadata,selected,output,rows,window_width,view,stream,nullptr,0,1);
}
extern "C" int32_t ds41rt_v41_sparse_attention_split(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,
    float* partial,uint64_t scratch_bytes,int32_t parts) {
  if(!partial)return cudaErrorInvalidValue;
  return launch_attention(query,sink,metadata,selected,output,rows,window_width,view,stream,partial,scratch_bytes,parts);
}

extern "C" int32_t ds41rt_v41_sparse_attention_bounded(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,
    const uint64_t* window_begins,float* partial,uint64_t scratch_bytes,int32_t parts) {
  if(!window_begins || (!partial && (parts!=1 || scratch_bytes!=0)))return cudaErrorInvalidValue;
  return launch_attention(query,sink,metadata,selected,output,rows,window_width,view,stream,
      partial,scratch_bytes,parts,window_begins);
}

template<int Heads> static int32_t validate_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* host_views,const ds41rt_v41_sparse_kv_t* device_views,
    const uint64_t* begins,float* partial,uint64_t scratch_bytes,int32_t parts,int32_t compressed) {
  if(rows<1 || rows>64 || !host_views || !begins || !partial || compressed<0 || compressed>2 ||
      parts!=(compressed?10:2))return cudaErrorInvalidValue;
  const uint64_t descriptor_bytes=uint64_t(rows)*sizeof(*device_views);
  if(!span(device_views,descriptor_bytes,8))return cudaErrorInvalidValue;
  // Descriptor upload is a write: reject overlap with every input and output,
  // including views belonging to a different request in this batch.
  for(int row=0;row<rows;++row) {
    const auto& v=host_views[row];
    if(v.compressed!=compressed)return cudaErrorInvalidValue;
    const auto status=validate_attention<Heads>(query,sink,metadata,selected,output,rows,0,
        &v,nullptr,partial,scratch_bytes,parts,begins);
    if(status!=cudaSuccess)return status;
    const void* common[]={query,sink,metadata,selected,output,begins,partial,v.window_end,v.pages,v.source_end};
    const uint64_t bytes[]={uint64_t(rows)*Heads*1024,Heads*4,uint64_t(rows)*80,uint64_t(rows)*2048,
        uint64_t(rows)*Heads*1024,uint64_t(rows)*8,uint64_t(rows)*parts*Heads*514*4,8,uint64_t(v.page_stride)*4,8};
    for(int i=0;i<10;++i) {
      if((i==3 || i>=8) && !compressed)continue;
      if(!disjoint(device_views,descriptor_bytes,common[i],bytes[i]))return cudaErrorInvalidValue;
    }
    const uint64_t capacity[]={128,v.window_proposal_capacity,v.source_capacity,v.source_proposal_capacity};
    for(int i=0;i<(compressed?4:2);++i) {
      if(!disjoint(device_views,descriptor_bytes,v.values[i],capacity[i]*((compressed==2 && i>=2)?256:512)) ||
         !disjoint(device_views,descriptor_bytes,v.scales[i],capacity[i]*((compressed==2 && i>=2)?32:16)))
        return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}
template<bool FP4,int Heads> static int32_t dispatch_batch(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts) {
  ds41rt_v41_sparse_kv_t unused{};
  attend<true,1,FP4,true,Heads><<<dim3(rows,Heads/16,parts),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
    reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
    reinterpret_cast<__nv_bfloat16*>(output),0,unused,partial,begins,views);
  const auto status=cudaGetLastError();
  if(status!=cudaSuccess)return status;
  merge<Heads><<<dim3(rows,Heads),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
    partial,sink,reinterpret_cast<__nv_bfloat16*>(output),parts);
  return cudaGetLastError();
}
template<int Heads> static int32_t launch_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed) {
  // Contents were host-validated before upload/replay; never download metadata.
  if(rows<1 || rows>64 || compressed<0 || compressed>2 || parts!=(compressed?10:2) ||
      !device_views || !begins || !partial)return cudaErrorInvalidValue;
  return compressed==2?dispatch_batch<true,Heads>(query,sink,metadata,selected,output,rows,device_views,stream,begins,partial,parts):
      dispatch_batch<false,Heads>(query,sink,metadata,selected,output,rows,device_views,stream,begins,partial,parts);
}
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
extern "C" int32_t ds41rt_v41_sparse_attention_batch_aot(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed) {
  // Host validation has checked every current descriptor before upload/replay.
  if(rows<1 || rows>64 || compressed!=2 || parts!=10 || !query || !sink ||
     !metadata || !selected || !output || !device_views || !begins || !partial ||
     reinterpret_cast<uintptr_t>(partial)%16)return cudaErrorInvalidValue;
  auto* bytes=reinterpret_cast<uint8_t*>(partial);
  return ds41rt_v41_attention_aot_launch(query,device_views,metadata,selected,begins,
      sink,bytes,bytes+uint64_t(rows)*655360,output,rows,stream);
}
#endif

extern "C" int32_t ds41rt_v41_sparse_attention_heads32_initialize(void) {
#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
  const auto aot_status=ds41rt_v41_attention_heads32_aot_initialize();
  if(aot_status!=cudaSuccess)return aot_status;
#endif
  const auto status=initialize_format<false,32>();
  return status==cudaSuccess?initialize_format<true,32>():status;
}
extern "C" int32_t ds41rt_v41_sparse_attention_heads32_bounded(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,int32_t window_width,
    const ds41rt_v41_sparse_kv_t* view,void* stream,const uint64_t* window_begins,
    float* partial,uint64_t scratch_bytes,int32_t parts) {
  if(!window_begins || (!partial && (parts!=1 || scratch_bytes!=0)))return cudaErrorInvalidValue;
  return launch_attention<32>(query,sink,metadata,selected,output,rows,window_width,view,stream,
      partial,scratch_bytes,parts,window_begins);
}

extern "C" int32_t ds41rt_v41_sparse_attention_batch_validate(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* host_views,const ds41rt_v41_sparse_kv_t* device_views,
    const uint64_t* begins,float* partial,uint64_t scratch_bytes,int32_t parts,int32_t compressed) {
  return validate_batch<64>(query,sink,metadata,selected,output,rows,host_views,device_views,
      begins,partial,scratch_bytes,parts,compressed);
}
extern "C" int32_t ds41rt_v41_sparse_attention_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed) {
  return launch_batch<64>(query,sink,metadata,selected,output,rows,device_views,stream,
      begins,partial,parts,compressed);
}

extern "C" int32_t ds41rt_v41_sparse_attention_heads32_batch_validate(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* host_views,const ds41rt_v41_sparse_kv_t* device_views,
    const uint64_t* begins,float* partial,uint64_t scratch_bytes,int32_t parts,int32_t compressed) {
  return validate_batch<32>(query,sink,metadata,selected,output,rows,host_views,device_views,
      begins,partial,scratch_bytes,parts,compressed);
}
extern "C" int32_t ds41rt_v41_sparse_attention_heads32_batch(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed) {
  return launch_batch<32>(query,sink,metadata,selected,output,rows,device_views,stream,
      begins,partial,parts,compressed);
}

#ifdef DS41RT_HAVE_V41_ATTENTION_AOT
extern "C" int32_t ds41rt_v41_sparse_attention_heads32_batch_aot(
    const uint16_t* query,const float* sink,const uint64_t* metadata,
    const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* device_views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t compressed) {
  // Host validation has checked every current descriptor before upload/replay.
  if(rows<1 || rows>64 || compressed!=2 || parts!=10 || !query || !sink ||
     !metadata || !selected || !output || !device_views || !begins || !partial ||
     reinterpret_cast<uintptr_t>(partial)%16)return cudaErrorInvalidValue;
  auto* bytes=reinterpret_cast<uint8_t*>(partial);
  return ds41rt_v41_attention_heads32_aot_launch(query,device_views,metadata,selected,begins,
      sink,bytes,bytes+uint64_t(rows)*327680,output,rows,stream);
}
#endif
