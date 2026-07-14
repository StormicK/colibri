#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_MEMORY_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_config.h"

typedef struct {
    uint64_t resident_heads_bytes;
    uint64_t streamed_stage_bytes;
    uint64_t minimum_expert_cache_bytes;
    uint64_t working_bytes;
    uint64_t incremental_reserve_bytes;
    uint64_t expert_record_bytes;
    int stages;
    int expert_slots_per_stage;
} ColiV4DSparkMemoryPlan;

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size);

#endif
