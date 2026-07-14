#ifndef COLIBRI_DEEPSEEK_V4_RUNTIME_H
#define COLIBRI_DEEPSEEK_V4_RUNTIME_H

#include <stdint.h>

typedef struct {
    const char *target_model_dir;
    const char *dspark_model_dir;
    uint64_t memory_limit_bytes;
    int context_tokens;
    int dense_resident;
    int dspark_resident;
    uint64_t dspark_expert_cache_bytes;
    int verify_drafts;
    int pin_slots_per_layer;
    uint64_t repin_interval;
} ColiDeepSeekV4RuntimeOptions;

void coli_v4_runtime_reset(void);
ColiDeepSeekV4RuntimeOptions *coli_v4_runtime_options(void);

#endif
