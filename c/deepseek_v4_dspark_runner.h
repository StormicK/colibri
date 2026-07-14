#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_RUNNER_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_RUNNER_H

#include <stddef.h>
#include <stdint.h>

#include "deepseek_v4_config.h"

typedef struct ColiV4DSparkRunner ColiV4DSparkRunner;

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size);
void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner);
int coli_v4_dspark_runner_prefill(ColiV4DSparkRunner *runner,
                                  const float *main_x,
                                  int start_position, int batch,
                                  char *error, size_t error_size);
int coli_v4_dspark_runner_draft(ColiV4DSparkRunner *runner,
                                const float *main_x, int anchor_token,
                                int position, int *draft_tokens,
                                float *draft_logits,
                                char *error, size_t error_size);
int coli_v4_dspark_runner_block_size(const ColiV4DSparkRunner *runner);
uint64_t coli_v4_dspark_runner_loaded_stage_peak(
    const ColiV4DSparkRunner *runner);

#endif
