#ifndef COLIBRI_DEEPSEEK_V4_EXPERT_STORE_H
#define COLIBRI_DEEPSEEK_V4_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *model_dir;
    int layers;
    int experts_per_layer;
    uint64_t cache_bytes;
} ColiDeepSeekV4ExpertStoreOptions;

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store,
    char *error,
    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
