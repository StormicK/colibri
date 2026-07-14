#include "deepseek_v4_expert.h"

#include <stdlib.h>

#include "deepseek_v4_math.h"
#include "native_quant.h"
#include "native_quant_dual.h"

int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit) {
    if (!output || !expert || !input || swiglu_limit < 0.0f ||
        expert->gate.rows != expert->up.rows ||
        expert->gate.columns != expert->up.columns ||
        expert->down.columns != expert->gate.rows ||
        expert->down.rows != expert->gate.columns) return -1;
    size_t intermediate = (size_t)expert->gate.rows;
    size_t output_size = (size_t)expert->down.rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_dual_matvec_ref(
        gate, up, &expert->gate, &expert->up, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        for (size_t i = 0; i < intermediate; i++)
            activated[i] = coli_bf16_round(activated[i] * route_weight);
        result = coli_fp4_matvec_ref(output, &expert->down, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}

int coli_v4_shared_expert_forward_ref(float *output,
                                      const ColiTensorView *gate_weight,
                                      const ColiTensorView *down_weight,
                                      const ColiTensorView *up_weight,
                                      const float *input,
                                      float swiglu_limit) {
    if (!output || !gate_weight || !down_weight || !up_weight || !input ||
        gate_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        down_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        up_weight->format != COLI_TENSOR_FP8_E4M3_BLOCK ||
        gate_weight->rows != up_weight->rows ||
        gate_weight->columns != up_weight->columns ||
        down_weight->columns != gate_weight->rows ||
        down_weight->rows != gate_weight->columns) return -1;
    size_t intermediate = (size_t)gate_weight->rows;
    size_t output_size = (size_t)down_weight->rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp8_dual_matvec_ref(
        gate, up, gate_weight, up_weight, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        coli_bf16_round_array(activated, intermediate);
        result = coli_fp8_matvec_ref(output, down_weight, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}
