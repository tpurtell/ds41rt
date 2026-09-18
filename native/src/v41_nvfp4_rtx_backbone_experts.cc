// NVFP4 (W4A4) full-width RTX backbone modules: the single-card placement.
// Separate symbol family from the W4A8 and EXL3 variants; consumes BF16
// hidden rows and publishes BF16 deterministic routes.
#define DS41RT_V41_LOCAL_EXPERTS 1
#define DS41RT_V41_NVFP4_VARIANTS_HEADER "v41_nvfp4_rtx_backbone_variants.h"
#define ds41rt_v41_expert_info ds41rt_v41_nvfp4_local_expert_info
#define ds41rt_v41_expert_initialize ds41rt_v41_nvfp4_local_expert_initialize
#define ds41rt_v41_expert_output_kind ds41rt_v41_nvfp4_local_expert_output_kind
#define ds41rt_v41_expert_bind_scratch ds41rt_v41_nvfp4_local_expert_bind_scratch
#define ds41rt_v41_expert_initialize_scratch_async ds41rt_v41_nvfp4_local_expert_initialize_scratch_async
#define ds41rt_v41_expert_launch ds41rt_v41_nvfp4_local_expert_launch
#include "v41_experts.cc"
