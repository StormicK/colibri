#define coli_v4_expert_forward_ref coli_v4_expert_forward_v17_fallback
#include "deepseek_v4_expert_dual.c"
#undef coli_v4_expert_forward_ref

#include "native_quant_fp4_rows16.h"


int coli_v4_expert_forward_ref(float *output, const ColiExpertView *expert,
                               const float *input, float route_weight,
                               float swiglu_limit) {
#ifndef __AVX512F__
    return coli_v4_expert_forward_v17_fallback(
        output, expert, input, route_weight, swiglu_limit);
#else
    if (!expert || expert->gate.block_rows != 16 ||
        expert->down.block_rows != 16 || expert->up.block_rows != 16)
        return coli_v4_expert_forward_v17_fallback(
            output, expert, input, route_weight, swiglu_limit);
    if (!output || !input || swiglu_limit < 0.0f) return -1;
    size_t intermediate = (size_t)expert->gate.rows;
    size_t output_size = (size_t)expert->down.rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_dual_matvec_rows16_v10(
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
        result = coli_fp4_matvec_rows16_v10(
            output, &expert->down, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
#endif
}
