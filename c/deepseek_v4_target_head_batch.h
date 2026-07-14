#ifndef COLIBRI_DEEPSEEK_V4_TARGET_HEAD_BATCH_H
#define COLIBRI_DEEPSEEK_V4_TARGET_HEAD_BATCH_H

#include "deepseek_v4_config.h"
#include "safetensors_index.h"

int coli_v4_target_load_embeddings(float *states_hc,
                                    const ColiSafetensorsIndex *index,
                                    const ColiDeepSeekV4Config *config,
                                    const int *tokens, int batch);
int coli_v4_target_head_argmax_batch(
    const float *states_hc, const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, int batch,
    int *tokens, float *logits, char *error, size_t error_size);

#endif
