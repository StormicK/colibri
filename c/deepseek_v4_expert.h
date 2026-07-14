#ifndef COLIBRI_DEEPSEEK_V4_EXPERT_H
#define COLIBRI_DEEPSEEK_V4_EXPERT_H

#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit);

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate,
                                      const ColiTensorView *down,
                                      const ColiTensorView *up,
                                      const float *input,
                                      float swiglu_limit);

#ifdef __cplusplus
}
#endif

#endif
