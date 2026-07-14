#ifndef COLIBRI_DEEPSEEK_V4_ATTENTION_H
#define COLIBRI_DEEPSEEK_V4_ATTENTION_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4WindowAttentionState
    ColiDeepSeekV4WindowAttentionState;

int coli_v4_window_attention_create(ColiDeepSeekV4WindowAttentionState **state,
                                    const ColiDeepSeekV4Config *config);
void coli_v4_window_attention_reset(ColiDeepSeekV4WindowAttentionState *state);
void coli_v4_window_attention_destroy(ColiDeepSeekV4WindowAttentionState *state);

/* Correctness-first single-KV attention. Compressed layers may use this at
 * position zero, before any compressed KV/indexer candidate exists. */
int coli_v4_attention_token_ref(float *output,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                const float *input, int position,
                                char *error, size_t error_size);
int coli_v4_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
