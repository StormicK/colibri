#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_HEADS_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_HEADS_H

#include <stddef.h>

#include "deepseek_v4_config.h"
#include "deepseek_v4_dspark.h"
#include "safetensors_index.h"

typedef struct ColiV4DSparkHeads ColiV4DSparkHeads;

int coli_v4_dspark_heads_open(ColiV4DSparkHeads **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size);
void coli_v4_dspark_heads_close(ColiV4DSparkHeads *heads);
int coli_v4_dspark_combine_hidden(ColiV4DSparkHeads *heads, float *output,
                                  const float *target_hidden_hc);
int coli_v4_dspark_markov_bias(ColiV4DSparkHeads *heads, float *bias,
                               int previous_token);
int coli_v4_dspark_biased_argmax(
    ColiV4DSparkHeads *heads, const ColiSafetensorsIndex *target_index,
    const float *hidden, int previous_token,
    int *best_token, float *best_logit);
int coli_v4_dspark_biased_argmax_batch(
    ColiV4DSparkHeads *heads, const ColiSafetensorsIndex *target_index,
    const float *hidden_batch, int previous_token,
    int *best_tokens, float *best_logits, int batch);

#endif
