#include "ds41rt_v41_fp8.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>
#include "v41_fp8_variants.h"
static_assert(sizeof(ds41rt_v41_fp8_info_t) == 56);
namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
struct Module {
  ModuleFn initialize, load;
  LaunchFn launch;
  cudaLibrary_t library = nullptr;
  void reset() { if (library) cudaLibraryUnload(library); library = nullptr; }
  ~Module() { reset(); }
};
struct Variant {
  ds41rt_v41_fp8_info_t info;
  Module quant, gemm, quant_rope;
  const uint32_t* grids;
  uint64_t split_offset;
  uint32_t split_slices;
  uint32_t groups;
  uint64_t grouped_output_offset;
  int device = -1;
};
Variant variants[] = {DS41RT_V41_FP8_VARIANTS};
Variant peer_variants[] = {DS41RT_V41_FP8_VARIANTS};
Module hc_project = DS41RT_V41_HC_PROJECT_MODULE;
#ifdef DS41RT_V41_HC_LAGGED_MODULE
Module hc_lagged = DS41RT_V41_HC_LAGGED_MODULE;
#endif
int hc_device[] = {-1, -1};
int owner_device[] = {-1, -1};
std::mutex mutex;
// Called only under initialization mutex. Preserve arbitrary first-device IDs.
int device_slot(int device) {
  for (int slot=0; slot<2; ++slot) if (owner_device[slot] == device) return slot;
  for (int slot=0; slot<2; ++slot) if (owner_device[slot] < 0) { owner_device[slot]=device; return slot; }
  return -1;
}
Variant* capacity(int rows, int k, int n, int slot=0) {
  auto* table = slot == 0 ? variants : peer_variants;
  for (size_t i=0; i<sizeof(variants)/sizeof(variants[0]); ++i) {
    auto& v=table[i];
    if (int(v.info.capacity_rows)==rows && int(v.info.input_dim)==k && int(v.info.output_dim)==n) return &v;
  }
  return nullptr;
}
Variant* handle(void* p) {
  for (auto& v : variants) if (&v == p) return &v;
  for (auto& v : peer_variants) if (&v == p) return &v;
  return nullptr;
}
int load(Module& m, int device) {
  auto* ptr = &m.library;
  int status = 0;
  // Exported launch symbols are process-global. Initialize one CUDA library,
  // then configure that same library on each owning device. Initializing a
  // second library overwrites the generated kernel symbols used by the first.
  void* init[] = {&ptr, &status};
  const bool existing = m.library != nullptr;
  if (!existing) m.initialize(init);
  if (!status) { void* args[] = {&ptr, &device, &status}; m.load(args); }
  if (status && !existing) m.reset();
  return status;
}
bool span(const void* p, uint64_t bytes, uintptr_t& start, uintptr_t& end) {
  start = reinterpret_cast<uintptr_t>(p);
  if (!start || start % 16 || start > UINTPTR_MAX - bytes) return false;
  end = start + bytes; return true;
}
// The AOT kernels are exported for one exact device: the SM count (and the launch grids
// sized from it) come from the GPU that ran export_b12x_v41_fp8_aot.py. A bare
// cudaErrorInvalidDevice (101) gives the operator nothing to act on, so name the mismatch.
int reject_device(int device, int major, int minor, int sms) {
  std::fprintf(stderr,
               "ds41rt: v41 fp8 AOT kernels were exported for compute 12.0 with %d SMs, but "
               "device %d is compute %d.%d with %d SMs; rebuild the coordinator AOT export on "
               "the target GPU (cudaErrorInvalidDevice)\n",
               int(DS41RT_V41_FP8_SMS), device, major, minor, sms);
  return int(cudaErrorInvalidDevice);
}
int device_matches(Variant* v) {
  if (!v || v->device < 0) return cudaErrorInvalidValue;
  int device = -1; auto status = cudaGetDevice(&device);
  return status ? int(status) : (device == v->device ? 0 : int(cudaErrorInvalidDevice));
}
}
extern "C" int32_t ds41rt_v41_fp8_grouped_output(const uint16_t*, uint16_t*, int32_t, void*);
extern "C" int32_t ds41rt_v41_fp8_reduce_splits(const float*, uint16_t*, int32_t, int32_t, int32_t, void*);
extern "C" int32_t ds41rt_v41_fp8_initialize_storage(void*, uint64_t, float*, void*);
extern "C" int32_t ds41rt_v41_fp8_matrix_info(int32_t rows, int32_t k, int32_t n, ds41rt_v41_fp8_info_t* out) {
  auto* v = capacity(rows, k, n); if (!v || !out) return cudaErrorInvalidValue;
  *out = v->info; return 0;
}
extern "C" int32_t ds41rt_v41_fp8_matrix_initialize(int32_t rows, int32_t k, int32_t n, void** out) {
  if (!out) return cudaErrorInvalidValue; *out = nullptr;
  auto* v = capacity(rows, k, n); if (!v) return cudaErrorInvalidValue;
  int device, major, minor, sms;
  auto status = cudaGetDevice(&device); if (status) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); if (status) return status;
  if (major != 12 || minor != 0 || sms != DS41RT_V41_FP8_SMS) return reject_device(device, major, minor, sms);
  std::lock_guard<std::mutex> lock(mutex);
  const int slot=device_slot(device);
  if (slot<0) return cudaErrorInvalidDevice;
  v=capacity(rows,k,n,slot);
  if (v->device >= 0) {
    if (v->device != device) return cudaErrorInvalidDevice;
  } else {
    auto* modules = capacity(rows,k,n);
    const bool existing = modules->gemm.library != nullptr;
    int result = load(modules->quant, device);
    if (!result) result = load(modules->gemm, device);
    if (!result && v->groups > 1) result = load(modules->quant_rope, device);
    if (result) {
      if (!existing) { modules->quant.reset(); modules->gemm.reset(); modules->quant_rope.reset(); }
      return result;
    }
    v->device = device;
  }
  *out = v; return 0;
}
extern "C" int32_t ds41rt_v41_fp8_initialize_scratch(void* kernel, void* scratch, uint64_t bytes, float* alpha, void* stream) {
  auto* v = handle(kernel); int status = device_matches(v); if (status) return status;
  uintptr_t a,b,c,d;
  if (bytes < v->info.scratch_bytes || !span(scratch, v->info.scratch_bytes,a,b) ||
      !span(alpha,4,c,d) || (a<d && c<b)) return cudaErrorInvalidValue;
  return ds41rt_v41_fp8_initialize_storage(scratch, v->info.scratch_bytes, alpha, stream);
}
extern "C" int32_t ds41rt_v41_fp8_launch_rope(void* kernel, const uint16_t* source, const float* frequencies, const uint8_t* weight,
    const uint8_t* packed_scales, void* scratch, uint64_t bytes, const float* alpha,
    uint16_t* output, int32_t rows, void* stream) {
  auto* v = handle(kernel); int status = device_matches(v); if (status) return status;
  if (rows <= 0 || uint32_t(rows) > v->info.capacity_rows || bytes < v->info.scratch_bytes) return cudaErrorInvalidValue;
  if (frequencies && v->groups != 8) return cudaErrorInvalidValue;
  const void* buffers[] = {source,weight,packed_scales,scratch,alpha,output,frequencies};
  uint64_t sizes[] = {uint64_t(rows)*v->info.input_dim*2,uint64_t(v->info.output_dim)*v->info.input_dim/v->groups,v->info.packed_weight_scale_bytes,v->info.scratch_bytes,4,uint64_t(rows)*v->info.output_dim*2,frequencies?uint64_t(rows)*256:0};
  uintptr_t starts[7], ends[7];
  for (int i=0;i<7;++i) {
    if (!sizes[i]) continue;
    if (!span(buffers[i],sizes[i],starts[i],ends[i])) return cudaErrorInvalidValue;
    for (int j=0;j<i;++j) if (sizes[j] && starts[i]<ends[j] && starts[j]<ends[i]) return cudaErrorInvalidValue;
  }
  void* x = const_cast<uint16_t*>(source);
  void* a = static_cast<char*>(scratch)+v->info.values_offset;
  void* sr = static_cast<char*>(scratch)+v->info.row_scales_offset;
  void* sm = static_cast<char*>(scratch)+v->info.mma_scales_offset;
  int grid = v->grids[rows-1];
  void* quant_args[] = {&x,&a,&sr,&sm,&rows,&grid,&stream,&status};
  if (v->groups == 1) v->quant.launch(quant_args,8);
  else {
    int unused_length=frequencies?rows*64:1;
    void* f=frequencies?const_cast<float*>(frequencies):x;
    void* group_args[] = {&x,&x,&f,&a,&sr,&sm,&rows,&unused_length,&grid,&stream,&status};
    (frequencies?v->quant_rope:v->quant).launch(group_args,11);
  }
  if (status) return status;
  void* w=const_cast<uint8_t*>(weight), *s=const_cast<uint8_t*>(packed_scales), *c=output, *one=const_cast<float*>(alpha);
  if (v->split_slices > 1) c = static_cast<char*>(scratch) + v->split_offset;
  if (v->groups > 1) c = static_cast<char*>(scratch) + v->grouped_output_offset;
  // Quantized-output slots are compile-time inactive for this BF16 projection.
  void* gemm_args[] = {&a,&w,&sm,&s,&c,&c,&c,&c,&one,&rows,&stream,&status};
  v->gemm.launch(gemm_args,12);
  if (status) return status;
  if (v->groups > 1) return ds41rt_v41_fp8_grouped_output(static_cast<const uint16_t*>(c),output,rows,stream);
  if (v->split_slices == 1) return status;
  return ds41rt_v41_fp8_reduce_splits(static_cast<const float*>(c), output,
      rows, v->info.output_dim, v->split_slices, stream);
}

extern "C" int32_t ds41rt_v41_fp8_launch(void* kernel, const uint16_t* source, const uint8_t* weight,
    const uint8_t* packed_scales, void* scratch, uint64_t bytes, const float* alpha,
    uint16_t* output, int32_t rows, void* stream) {
  return ds41rt_v41_fp8_launch_rope(kernel,source,nullptr,weight,packed_scales,scratch,bytes,alpha,output,rows,stream);
}

// Existing engram entry points retain their explicit geometry.
extern "C" int32_t ds41rt_v41_fp8_info(int32_t rows, ds41rt_v41_fp8_info_t* out) {
  return ds41rt_v41_fp8_matrix_info(rows, 6144, 25600, out);
}
extern "C" int32_t ds41rt_v41_fp8_initialize(int32_t rows, void** out) {
  return ds41rt_v41_fp8_matrix_initialize(rows, 6144, 25600, out);
}


// Loaded once during mHC planning; no module resolution or allocation on replay.
extern "C" int32_t ds41rt_v41_hc_project_initialize() {
  int device, major, minor;
  auto status=cudaGetDevice(&device); if(status) return status;
  status=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device); if(status) return status;
  status=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device); if(status) return status;
  if(major!=12 || minor!=0) return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(mutex);
  const int slot=device_slot(device);
  if(slot<0) return cudaErrorInvalidDevice;
  if(hc_device[slot]>=0) return hc_device[slot]==device ? 0 : int(cudaErrorInvalidDevice);
  int result=load(hc_project,device);
#ifdef DS41RT_V41_HC_LAGGED_MODULE
  if(!result) result=load(hc_lagged,device);
#endif
  if(!result) hc_device[slot]=device;
  return result;
}
// Internal launch: buffer validation is performed by hc_mixes_workspace.
extern "C" int32_t ds41rt_v41_hc_project_launch(const uint16_t* residual,
    const float* weight, float* partials, int32_t rows, void* stream) {
  int device=-1;
  int status=cudaGetDevice(&device); if(status) return status;
  const int slot=hc_device[0]==device ? 0 : hc_device[1]==device ? 1 : -1;
  if(slot<0) return cudaErrorInvalidDevice;
  void* r=const_cast<uint16_t*>(residual);
  void* w=const_cast<float*>(weight);
  void* p=partials;
  void* args[]={&r,&w,&p,&rows,&stream,&status};
  hc_project.launch(args,6);
  return status;
}

#ifdef DS41RT_V41_HC_LAGGED_MODULE
extern "C" int32_t ds41rt_v41_hc_begin(const void* residual, const void* fn,
    const void* scale, const void* bias, const void* incoming, const void* norm,
    void* predicted, void* post, void* comb, void* normalized, void* scratch,
    uint64_t scratch_bytes, int32_t rows, void* stream) {
  if(rows<1 || rows>80 || scratch_bytes<uint64_t(rows)*8000) return cudaErrorInvalidValue;
  const void* pointers[]={residual,fn,scale,bias,incoming,norm,predicted,post,comb,normalized,scratch};
  const uint64_t sizes[]={uint64_t(rows)*40960,24*20480*4,12,96,uint64_t(rows)*16,10240,
    uint64_t(rows)*16,uint64_t(rows)*16,uint64_t(rows)*64,uint64_t(rows)*10240,uint64_t(rows)*8000};
  uintptr_t starts[11],ends[11];
  for(int i=0;i<11;++i) if(!span(pointers[i],sizes[i],starts[i],ends[i])) return cudaErrorInvalidValue;
  for(int i=6;i<11;++i) for(int j=0;j<i;++j)
    if(starts[i]<ends[j] && starts[j]<ends[i]) return cudaErrorInvalidValue;
  int device=-1,status=cudaGetDevice(&device); if(status) return status;
  if(hc_device[0]!=device && hc_device[1]!=device) return cudaErrorInvalidDevice;
  void* args[]={&residual,&fn,&scale,&bias,&incoming,&norm,&predicted,&post,&comb,
    &normalized,&scratch,&rows,&stream,&status};
  hc_lagged.launch(args,14);
  return status;
}
#endif
