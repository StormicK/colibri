#include "deepseek_v4_sparse_attention.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "native_quant.h"

int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale) {
    if (!output || !queries || !kv || !sinks || !indices || heads < 1 ||
        head_dimension < 1 || kv_count < 1 || topk < 1 || !(softmax_scale > 0.0f))
        return -1;
    float *scores = malloc((size_t)topk * sizeof(*scores));
    if (!scores) return -1;
    for (int head = 0; head < heads; head++) {
        const float *query = queries + (size_t)head * head_dimension;
        float maximum = -INFINITY;
        for (int rank = 0; rank < topk; rank++) {
            int index = indices[rank];
            if (index < 0) {
                scores[rank] = -INFINITY;
                continue;
            }
            if (index >= kv_count) {
                free(scores);
                return -1;
            }
            const float *key = kv + (size_t)index * head_dimension;
            float score = 0.0f;
            for (int column = 0; column < head_dimension; column++)
                score += query[column] * key[column];
            score *= softmax_scale;
            scores[rank] = score;
            if (score > maximum) maximum = score;
        }
        if (!isfinite(maximum)) {
            free(scores);
            return -1;
        }
        float denominator = expf(sinks[head] - maximum);
        float *head_output = output + (size_t)head * head_dimension;
        memset(head_output, 0, (size_t)head_dimension * sizeof(*head_output));
        for (int rank = 0; rank < topk; rank++) {
            if (indices[rank] < 0) continue;
            float probability = expf(scores[rank] - maximum);
            denominator += probability;
            /* TileLang casts the exp fragment to BF16 before value GEMM. */
            probability = coli_bf16_round(probability);
            const float *value = kv + (size_t)indices[rank] * head_dimension;
            for (int column = 0; column < head_dimension; column++)
                head_output[column] += probability * value[column];
        }
        for (int column = 0; column < head_dimension; column++)
            head_output[column] = coli_bf16_round(head_output[column] / denominator);
    }
    free(scores);
    return 0;
}
