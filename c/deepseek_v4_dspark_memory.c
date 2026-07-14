#define coli_v4_dspark_memory_plan coli_v4_dspark_memory_plan_unmargined
/* ---- begin inlined deepseek_v4_dspark_memory.c ---- */
#include "deepseek_v4_dspark_memory.h"

#include <stdio.h>
#include <string.h>

#include "deepseek_v4_dspark.h"
#include "safetensors_index.h"

#define DSPARK_MIB UINT64_C(1048576)

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size) {
    if (!model_dir || !config || !plan) return -1;
    memset(plan, 0, sizeof(*plan));
    ColiDeepSeekV4DSparkManifest manifest;
    if (coli_v4_dspark_inspect(model_dir, config, &manifest,
                               error, error_size)) return -1;
    plan->stages = manifest.stage_count;
    plan->expert_slots_per_stage = config->num_experts_per_tok;
    plan->resident_heads_bytes = manifest.special_bytes;
    for (int i = 0; i < manifest.stage_count; i++)
        if (manifest.common_stage_bytes[i] > plan->streamed_stage_bytes)
            plan->streamed_stage_bytes = manifest.common_stage_bytes[i];
    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    static const char *parts[] = {
        "mtp.0.ffn.experts.0.w1.weight", "mtp.0.ffn.experts.0.w1.scale",
        "mtp.0.ffn.experts.0.w2.weight", "mtp.0.ffn.experts.0.w2.scale",
        "mtp.0.ffn.experts.0.w3.weight", "mtp.0.ffn.experts.0.w3.scale",
    };
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        const ColiSafetensorsTensor *tensor = coli_st_find(index, parts[i]);
        if (!tensor) { coli_st_index_close(index); return -1; }
        plan->expert_record_bytes += tensor->nbytes;
    }
    coli_st_index_close(index);
    plan->minimum_expert_cache_bytes = plan->expert_record_bytes *
        (uint64_t)plan->stages * config->num_experts_per_tok;
    uint64_t capture = (uint64_t)64 * manifest.target_count * config->hc_mult *
                       config->hidden_size * sizeof(float);
    uint64_t query = (uint64_t)64 * config->hc_mult * config->hidden_size *
                     sizeof(float) * 2;
    uint64_t kv = (uint64_t)manifest.stage_count *
        (config->sliding_window + 64) * config->head_dim * sizeof(float);
    plan->working_bytes = 64 * DSPARK_MIB + capture + query + kv;
    plan->incremental_reserve_bytes = plan->resident_heads_bytes +
        plan->streamed_stage_bytes + plan->minimum_expert_cache_bytes +
        plan->working_bytes;
    return 0;
}
/* ---- end inlined deepseek_v4_dspark_memory.c ---- */

#undef coli_v4_dspark_memory_plan

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size) {
    int result = coli_v4_dspark_memory_plan_unmargined(
        model_dir, config, plan, error, error_size);
    if (result) return result;
    /* Native FP8/FP4 batch kernels and OpenMP allocate short-lived per-thread
       buffers outside the explicit query/KV arrays.  Keep this model-size-
       independent margin in the RAM tier plan so low-memory hosts do not lend
       the same bytes to the target expert cache. */
    uint64_t allocator_margin = 256 * DSPARK_MIB;
    if (UINT64_MAX - plan->working_bytes < allocator_margin ||
        UINT64_MAX - plan->incremental_reserve_bytes < allocator_margin)
        return -1;
    plan->working_bytes += allocator_margin;
    plan->incremental_reserve_bytes += allocator_margin;
    return 0;
}
