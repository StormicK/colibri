#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_CAPTURE_H

#include "deepseek_v4_config.h"
#include "deepseek_v4_dspark_heads.h"

int coli_v4_dspark_capture_main_x(float *outputs, int batch,
                                  const ColiDeepSeekV4Config *config);
ColiV4DSparkHeads *coli_v4_dspark_capture_heads(void);
int coli_v4_dspark_capture_stage_main_x(const float *values, int batch,
                                        int hidden_size);

#endif
