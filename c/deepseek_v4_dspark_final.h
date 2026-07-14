#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_FINAL_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_FINAL_H

#include "deepseek_v4_config.h"
#include "deepseek_v4_dspark.h"

typedef struct ColiV4DSparkFinal ColiV4DSparkFinal;

int coli_v4_dspark_final_open(ColiV4DSparkFinal **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size);
void coli_v4_dspark_final_close(ColiV4DSparkFinal *final);
int coli_v4_dspark_final_hidden(ColiV4DSparkFinal *final,
                                float *outputs, const float *states_hc,
                                int batch);

#endif
