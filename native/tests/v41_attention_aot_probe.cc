// Qualification-only bridge for the generated pointer ABI. Not a serving API:
// the test owns and validates allocations and calls initialize before capture.
#include "v41_attention.h"
static ds41rt_v41_attention_Kernel_Module_t module{};
extern "C" void ds41rt_attention_probe_initialize() {
  ds41rt_v41_attention_Kernel_Module_Load(&module);
}
extern "C" int32_t ds41rt_attention_probe(
    void* query, void* descriptors, void* metadata, void* selected, void* bounds,
    void* sink, void* partials, void* lses, void* output, int32_t rows, void* stream) {
  return cute_dsl_ds41rt_v41_attention_wrapper(&module, query, descriptors,
      metadata, selected, bounds, sink, partials, lses, output, rows,
      static_cast<cudaStream_t>(stream));
}
