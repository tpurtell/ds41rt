// NVFP4 (W4A4) Spark TP4 modules. Separate symbol family from the W4A8 and
// EXL3 variants; consumes BF16 hidden rows over the expert fabric.
#define DS41RT_V41_NVFP4_VARIANTS_HEADER "v41_nvfp4_spark_variants.h"
#define ds41rt_v41_expert_info ds41rt_v41_nvfp4_expert_info
#define ds41rt_v41_expert_initialize ds41rt_v41_nvfp4_expert_initialize
#define ds41rt_v41_expert_output_kind ds41rt_v41_nvfp4_expert_output_kind
#define ds41rt_v41_expert_bind_scratch ds41rt_v41_nvfp4_expert_bind_scratch
#define ds41rt_v41_expert_initialize_scratch_async ds41rt_v41_nvfp4_expert_initialize_scratch_async
#define ds41rt_v41_expert_launch ds41rt_v41_nvfp4_expert_launch
#include "v41_experts.cc"
