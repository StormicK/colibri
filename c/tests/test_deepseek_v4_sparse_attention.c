#include "../deepseek_v4_sparse_attention.h"
#include "../native_quant.h"

#include <math.h>
#include <stdio.h>

static int close_enough(float left, float right) {
    return fabsf(left - right) <= 1e-6f;
}

int main(void) {
    float query[2] = {1, 0};
    float kv[6] = {1, 0, 0, 1, 1, 1};
    float sink[1] = {0};
    int indices[3] = {0, 2, -1};
    float output[2];
    if (coli_v4_sparse_attention_ref(output, query, kv, sink, indices,
                                     1, 2, 3, 3, 1.0f) != 0)
        return 1;
    float denominator = 2.0f + expf(-1.0f);
    if (!close_enough(output[0], coli_bf16_round(2.0f / denominator)) ||
        !close_enough(output[1], coli_bf16_round(1.0f / denominator)))
        return 1;
    int invalid[1] = {3};
    if (coli_v4_sparse_attention_ref(output, query, kv, sink, invalid,
                                     1, 2, 3, 1, 1.0f) == 0)
        return 1;
    puts("DeepSeek-V4 sparse attention tests: ok");
    return 0;
}
