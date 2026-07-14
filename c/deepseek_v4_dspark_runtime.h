#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_RUNTIME_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_RUNTIME_H

#include "deepseek_v4_dspark.h"
#include "deepseek_v4_expert_store.h"

int coli_v4_dspark_layer_load(ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size);
int coli_deepseek_v4_dspark_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store, char *error, size_t error_size);

#endif
