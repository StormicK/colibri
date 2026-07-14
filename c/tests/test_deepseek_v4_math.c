#include "../deepseek_v4_math.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_enough(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

int main(void) {
    enum { HC = 4, DIMENSION = 3, MIXES = (2 + HC) * HC };
    float input[HC * DIMENSION] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    };
    float function[MIXES * HC * DIMENSION];
    float base[MIXES];
    float scale[3] = {1, 1, 1};
    float reduced[DIMENSION], post[HC], comb[HC * HC];
    memset(function, 0, sizeof(function));
    memset(base, 0, sizeof(base));
    if (coli_v4_hc_pre(reduced, post, comb, input, function,
                       scale, base, HC, DIMENSION, 20, 1e-6f, 1e-6f) != 0)
        return 1;
    for (int column = 0; column < DIMENSION; column++) {
        float expected = 0.500001f * (input[column] + input[DIMENSION + column] +
                                     input[2 * DIMENSION + column] +
                                     input[3 * DIMENSION + column]);
        if (!close_enough(reduced[column], expected, 2e-5f)) return 1;
    }
    for (int index = 0; index < HC; index++)
        if (!close_enough(post[index], 1.0f, 1e-6f)) return 1;
    for (int row = 0; row < HC; row++) {
        float row_sum = 0.0f, column_sum = 0.0f;
        for (int column = 0; column < HC; column++) {
            row_sum += comb[row * HC + column];
            column_sum += comb[column * HC + row];
        }
        if (!close_enough(row_sum, 1.0f, 1e-5f) ||
            !close_enough(column_sum, 1.0f, 1e-5f)) return 1;
    }
    float branch[DIMENSION] = {0.25f, -0.5f, 0.75f};
    float expanded[HC * DIMENSION];
    if (coli_v4_hc_post(expanded, branch, input, post, comb,
                        HC, DIMENSION) != 0) return 1;
    for (int copy = 0; copy < HC; copy++)
        for (int column = 0; column < DIMENSION; column++) {
            float average = (input[column] + input[DIMENSION + column] +
                             input[2 * DIMENSION + column] +
                             input[3 * DIMENSION + column]) / 4.0f;
            if (!close_enough(expanded[copy * DIMENSION + column],
                              branch[column] + average, 2e-5f)) return 1;
        }

    float norm_weight[DIMENSION] = {1.0f, 1.5f, 0.5f};
    float normalized[DIMENSION];
    if (coli_v4_rmsnorm(normalized, branch, norm_weight,
                        DIMENSION, 1e-6f) != 0) return 1;
    float inverse_rms = 1.0f / sqrtf(
        (branch[0] * branch[0] + branch[1] * branch[1] +
         branch[2] * branch[2]) / DIMENSION + 1e-6f);
    for (int index = 0; index < DIMENSION; index++)
        if (!close_enough(normalized[index],
                          branch[index] * inverse_rms * norm_weight[index],
                          1e-6f)) return 1;

    float rope_cos[12], rope_sin[12];
    float rope_values[8] = {1, 2, 3, 4, -1, 0.5f, 2, -3};
    float rope_original[8];
    memcpy(rope_original, rope_values, sizeof(rope_values));
    if (coli_v4_rope_precompute(rope_cos, rope_sin, 4, 6, 4,
                                10000.0f, 2.0f, 32, 1) != 0 ||
        coli_v4_rope_apply(rope_values, 2, 4,
                           rope_cos + 2, rope_sin + 2, 0) != 0 ||
        coli_v4_rope_apply(rope_values, 2, 4,
                           rope_cos + 2, rope_sin + 2, 1) != 0)
        return 1;
    for (int index = 0; index < 8; index++)
        if (!close_enough(rope_values[index], rope_original[index], 1e-5f)) return 1;
    if (coli_v4_rope_precompute(rope_cos, rope_sin, 3, 6, 0,
                                10000.0f, 1.0f, 32, 1) == 0)
        return 1;

    float hidden[2] = {0.5f, -1.0f};
    float router[8] = {1, 0, 0, 1, -1, 0, 0, -1};
    float bias[4] = {10.0f, 0.0f, 9.0f, 0.0f};
    float route_weights[2];
    int route_indices[2];
    if (coli_v4_route(route_weights, route_indices, hidden, router, bias,
                      NULL, 4, 2, 2, 1.5f) != 0) return 1;
    if (route_indices[0] != 0 || route_indices[1] != 2 ||
        !close_enough(route_weights[0] + route_weights[1], 1.5f, 1e-6f))
        return 1;
    int forced[2] = {3, 1};
    if (coli_v4_route(route_weights, route_indices, hidden, router, NULL,
                      forced, 4, 2, 2, 1.5f) != 0 ||
        route_indices[0] != 3 || route_indices[1] != 1) return 1;

    float gates[3] = {-20.0f, 2.0f, 20.0f};
    float ups[3] = {-20.0f, 3.0f, 20.0f};
    float activated[3];
    if (coli_v4_swiglu(activated, gates, ups, 3, 10.0f) != 0) return 1;
    if (!close_enough(activated[0],
                      -20.0f / (1.0f + expf(20.0f)) * -10.0f, 1e-6f) ||
        !close_enough(activated[1],
                      2.0f / (1.0f + expf(-2.0f)) * 3.0f, 1e-6f) ||
        !close_enough(activated[2],
                      10.0f / (1.0f + expf(-10.0f)) * 10.0f, 1e-5f))
        return 1;
    puts("DeepSeek-V4 math tests: ok");
    return 0;
}
