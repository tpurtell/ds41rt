// NVFP4 (W4A4) TP2 modules. This family is independent of the W4A8 and EXL3
// variants: it consumes BF16 hidden rows, publishes BF16 token-major route planes
// and keeps its own generated bridge, so it exports distinct symbols.
#define DS41RT_V41_LOCAL_EXPERTS 1
#define DS41RT_V41_TP2_EXPERTS 1
#define DS41RT_V41_NVFP4_VARIANTS_HEADER "v41_nvfp4_rtx_tp2_variants.h"
#define ds41rt_v41_expert_info ds41rt_v41_nvfp4_tp2_expert_info
#define ds41rt_v41_expert_initialize ds41rt_v41_nvfp4_tp2_expert_initialize
#define ds41rt_v41_expert_output_kind ds41rt_v41_nvfp4_tp2_expert_output_kind
#define ds41rt_v41_expert_bind_scratch ds41rt_v41_nvfp4_tp2_expert_bind_scratch
#define ds41rt_v41_expert_initialize_scratch_async ds41rt_v41_nvfp4_tp2_expert_initialize_scratch_async
#define ds41rt_v41_expert_launch ds41rt_v41_nvfp4_tp2_expert_launch
#include "v41_experts.cc"
