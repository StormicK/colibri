#include "../deepseek_v4_expert.h"
#include "../native_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void make_tensor(ColiTensorView *view, uint8_t *data, uint8_t *scales,
                        int rows, int columns, uint8_t code) {
    memset(data, (int)(code | (code << 4)), (size_t)rows * columns / 2);
    memset(scales, 0x7f, (size_t)rows * columns / 32);
    *view = (ColiTensorView){
        COLI_TENSOR_FP4_NATIVE_BLOCK, COLI_SCALE_UE8M0,
        data, scales, (size_t)rows * columns / 2,
        (size_t)rows * columns / 32, rows, columns, 1, 32
    };
}

int main(void) {
    enum { DIMENSION = 128, INTERMEDIATE = 128 };
    uint8_t gate_data[INTERMEDIATE * DIMENSION / 2];
    uint8_t up_data[INTERMEDIATE * DIMENSION / 2];
    uint8_t down_data[DIMENSION * INTERMEDIATE / 2];
    uint8_t gate_scale[INTERMEDIATE * DIMENSION / 32];
    uint8_t up_scale[INTERMEDIATE * DIMENSION / 32];
    uint8_t down_scale[DIMENSION * INTERMEDIATE / 32];
    ColiExpertView expert = {0};
    make_tensor(&expert.gate, gate_data, gate_scale,
                INTERMEDIATE, DIMENSION, 1); /* +0.5 */
    make_tensor(&expert.up, up_data, up_scale,
                INTERMEDIATE, DIMENSION, 1);
    make_tensor(&expert.down, down_data, down_scale,
                DIMENSION, INTERMEDIATE, 1);
    float input[DIMENSION], output[DIMENSION];
    for (int index = 0; index < DIMENSION; index++) input[index] = 0.01f;
    if (coli_v4_expert_forward_ref(output, &expert, input, 0.75f, 10.0f) != 0)
        return 1;
    for (int index = 0; index < DIMENSION; index++)
        if (!isfinite(output[index]) || output[index] != output[0]) return 1;
    if (!(output[0] > 0.0f)) return 1;
    puts("DeepSeek-V4 expert tests: ok");
    return 0;
}
