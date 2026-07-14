#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float route_bf16_decode(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static float route_softplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale) {
    if (!weights || !indices || !hidden || !gate || experts < 1 ||
        dimension < 1 || topk < 1 || topk > experts) return -1;
    float *scores = malloc((size_t)experts * sizeof(*scores));
    float *selection = malloc((size_t)experts * sizeof(*selection));
    unsigned char *selected = calloc((size_t)experts, 1);
    if (!scores || !selection || !selected) {
        free(selected); free(selection); free(scores); return -1;
    }
    for (int expert = 0; expert < experts; expert++) {
        float sum = 0.0f;
        const uint16_t *row = gate + (size_t)expert * dimension;
        for (int column = 0; column < dimension; column++)
            sum += route_bf16_decode(row[column]) * hidden[column];
        scores[expert] = sqrtf(route_softplus(sum));
        selection[expert] = scores[expert] + (bias ? bias[expert] : 0.0f);
    }
    if (forced_indices) {
        for (int rank = 0; rank < topk; rank++) {
            if (forced_indices[rank] < 0 || forced_indices[rank] >= experts) {
                free(selected); free(selection); free(scores); return -1;
            }
            indices[rank] = forced_indices[rank];
        }
    } else {
        for (int rank = 0; rank < topk; rank++) {
            int best = -1;
            for (int expert = 0; expert < experts; expert++)
                if (!selected[expert] &&
                    (best < 0 || selection[expert] > selection[best]))
                    best = expert;
            indices[rank] = best;
            selected[best] = 1;
        }
    }
    float total = 0.0f;
    for (int rank = 0; rank < topk; rank++)
        total += scores[indices[rank]];
    if (!(total > 0.0f)) {
        free(selected); free(selection); free(scores); return -1;
    }
    for (int rank = 0; rank < topk; rank++)
        weights[rank] = scores[indices[rank]] / total * route_scale;
    free(selected); free(selection); free(scores);
    return 0;
}
