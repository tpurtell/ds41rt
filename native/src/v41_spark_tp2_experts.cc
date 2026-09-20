// Replicated-group Spark TP2 shard modules (native FP8 K32 family, SM121).
// Distinct symbol family from the standard Spark TP4 shard so a single
// libds41rt_native.so can hold every Spark TP degree. Geometry is baked into
// the exported variant table (intermediate 1152, no storage padding).
#define DS41RT_V41_SPARK_TP2_EXPERTS 1
#define ds41rt_v41_expert_info ds41rt_v41_spark_tp2_expert_info
#define ds41rt_v41_expert_initialize ds41rt_v41_spark_tp2_expert_initialize
#define ds41rt_v41_expert_output_kind ds41rt_v41_spark_tp2_expert_output_kind
#define ds41rt_v41_expert_bind_scratch ds41rt_v41_spark_tp2_expert_bind_scratch
#define ds41rt_v41_expert_initialize_scratch_async ds41rt_v41_spark_tp2_expert_initialize_scratch_async
#define ds41rt_v41_expert_launch ds41rt_v41_spark_tp2_expert_launch
#include "v41_experts.cc"
