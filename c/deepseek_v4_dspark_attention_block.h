#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_ATTENTION_BLOCK_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_ATTENTION_BLOCK_H

#include "deepseek_v4_dspark_attention.h"

int coli_v4_dspark_attention_block(
    float *outputs, ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int query_start_position, int batch, char *error, size_t error_size);

#endif
