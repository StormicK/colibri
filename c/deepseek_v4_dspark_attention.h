#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_ATTENTION_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_ATTENTION_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

typedef struct ColiV4DSparkAttentionState ColiV4DSparkAttentionState;

int coli_v4_dspark_attention_create(ColiV4DSparkAttentionState **state,
                                    const ColiDeepSeekV4Config *config);
void coli_v4_dspark_attention_reset(ColiV4DSparkAttentionState *state);
void coli_v4_dspark_attention_destroy(ColiV4DSparkAttentionState *state);
int coli_v4_dspark_attention_context_count(
    const ColiV4DSparkAttentionState *state);
int coli_v4_dspark_attention_precompute_context(
    ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *main_x, int start_position, int batch,
    char *error, size_t error_size);

#endif
