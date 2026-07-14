#include "deepseek_v4_math.h"

#include <math.h>
#include <stdlib.h>

static float sigmoidf_stable(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

int coli_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps) {
    if (!pre || !post || !comb || !mixes || !scale || !base ||
        hc < 1 || iterations < 1 || eps < 0.0f)
        return -1;
    for (int index = 0; index < hc; index++) {
        pre[index] = sigmoidf_stable(
            mixes[index] * scale[0] + base[index]) + eps;
        post[index] = 2.0f * sigmoidf_stable(
            mixes[hc + index] * scale[1] + base[hc + index]);
    }
    int matrix_offset = 2 * hc;
    for (int row = 0; row < hc; row++) {
        float maximum = -INFINITY;
        for (int column = 0; column < hc; column++) {
            int index = matrix_offset + row * hc + column;
            float value = mixes[index] * scale[2] + base[index];
            comb[row * hc + column] = value;
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int column = 0; column < hc; column++) {
            float value = expf(comb[row * hc + column] - maximum);
            comb[row * hc + column] = value;
            sum += value;
        }
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] = comb[row * hc + column] / sum + eps;
    }
    float *sums = malloc((size_t)hc * sizeof(*sums));
    if (!sums) return -1;
    for (int column = 0; column < hc; column++) {
        float sum = 0.0f;
        for (int row = 0; row < hc; row++)
            sum += comb[row * hc + column];
        sums[column] = sum;
    }
    for (int row = 0; row < hc; row++)
        for (int column = 0; column < hc; column++)
            comb[row * hc + column] /= sums[column] + eps;

    for (int iteration = 1; iteration < iterations; iteration++) {
        for (int row = 0; row < hc; row++) {
            float sum = 0.0f;
            for (int column = 0; column < hc; column++)
                sum += comb[row * hc + column];
            sums[row] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[row] + eps;
        for (int column = 0; column < hc; column++) {
            float sum = 0.0f;
            for (int row = 0; row < hc; row++)
                sum += comb[row * hc + column];
            sums[column] = sum;
        }
        for (int row = 0; row < hc; row++)
            for (int column = 0; column < hc; column++)
                comb[row * hc + column] /= sums[column] + eps;
    }
    free(sums);
    return 0;
}

int coli_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps) {
    if (!output || !post || !comb || !input || !hc_fn || !scale || !base ||
        hc < 1 || dimension < 1 || norm_eps < 0.0f)
        return -1;
    int flattened = hc * dimension;
    int mix_count = (2 + hc) * hc;
    float mean_square = 0.0f;
    for (int index = 0; index < flattened; index++)
        mean_square += input[index] * input[index];
    float inverse_rms = 1.0f / sqrtf(mean_square / flattened + norm_eps);
    float *mixes = malloc((size_t)mix_count * sizeof(*mixes));
    float *pre = malloc((size_t)hc * sizeof(*pre));
    if (!mixes || !pre) {
        free(mixes);
        free(pre);
        return -1;
    }
    for (int row = 0; row < mix_count; row++) {
        float sum = 0.0f;
        for (int column = 0; column < flattened; column++)
            sum += hc_fn[(size_t)row * flattened + column] * input[column];
        mixes[row] = sum * inverse_rms;
    }
    if (coli_v4_hc_split_sinkhorn(pre, post, comb, mixes, scale, base,
                                  hc, iterations, hc_eps) != 0) {
        free(pre);
        free(mixes);
        return -1;
    }
    for (int column = 0; column < dimension; column++) {
        float sum = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            sum += pre[copy] * input[copy * dimension + column];
        output[column] = sum;
    }
    free(pre);
    free(mixes);
    return 0;
}

int coli_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension) {
    if (!output || !branch || !residual || !post || !comb ||
        hc < 1 || dimension < 1)
        return -1;
    for (int destination = 0; destination < hc; destination++) {
        for (int column = 0; column < dimension; column++) {
            float value = 0.0f;
            for (int source = 0; source < hc; source++)
                value += comb[source * hc + destination] *
                         residual[source * dimension + column];
            value += post[destination] * branch[column];
            output[destination * dimension + column] = value;
        }
    }
    return 0;
}

int coli_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps) {
    if (!output || !input || !weight || dimension < 1 || eps < 0.0f)
        return -1;
    float mean_square = 0.0f;
    for (int index = 0; index < dimension; index++)
        mean_square += input[index] * input[index];
    float inverse_rms = 1.0f / sqrtf(mean_square / dimension + eps);
    for (int index = 0; index < dimension; index++)
        output[index] = input[index] * inverse_rms * weight[index];
    return 0;
}

int coli_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow) {
    if (!cosines || !sines || dimension < 2 || (dimension & 1) ||
        sequence_length < 1 || !(base > 1.0f) || !(factor > 0.0f))
        return -1;
    int pairs = dimension / 2;
    int low = 0, high = -1;
    if (original_sequence_length > 0) {
        const float two_pi = 6.2831853071795864769f;
        float denominator = 2.0f * logf(base);
        float low_value = dimension * logf(
            original_sequence_length / (beta_fast * two_pi)) / denominator;
        float high_value = dimension * logf(
            original_sequence_length / (beta_slow * two_pi)) / denominator;
        low = (int)floorf(low_value);
        high = (int)ceilf(high_value);
        if (low < 0) low = 0;
        if (high > dimension - 1) high = dimension - 1;
    }
    for (int pair = 0; pair < pairs; pair++) {
        float frequency = 1.0f / powf(base, (float)(2 * pair) / dimension);
        if (original_sequence_length > 0) {
            float width = high == low ? 0.001f : (float)(high - low);
            float ramp = (pair - low) / width;
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            float smooth = 1.0f - ramp;
            frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
        }
        for (int position = 0; position < sequence_length; position++) {
            size_t index = (size_t)position * pairs + pair;
            float angle = position * frequency;
            cosines[index] = cosf(angle);
            sines[index] = sinf(angle);
        }
    }
    return 0;
}

int coli_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse) {
    if (!vectors || !cosines || !sines || vector_count < 1 ||
        dimension < 2 || (dimension & 1))
        return -1;
    int pairs = dimension / 2;
    float direction = inverse ? -1.0f : 1.0f;
    for (int vector = 0; vector < vector_count; vector++) {
        for (int pair = 0; pair < pairs; pair++) {
            size_t value_index = (size_t)vector * dimension + 2 * pair;
            size_t frequency_index = (size_t)vector * pairs + pair;
            float real = vectors[value_index];
            float imaginary = vectors[value_index + 1];
            float cosine = cosines[frequency_index];
            float sine = sines[frequency_index] * direction;
            vectors[value_index] = real * cosine - imaginary * sine;
            vectors[value_index + 1] = real * sine + imaginary * cosine;
        }
    }
    return 0;
}

static float softplusf_stable(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

int coli_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale) {
    if (!weights || !indices || !hidden || !gate || experts < 1 ||
        dimension < 1 || topk < 1 || topk > experts)
        return -1;
    float *scores = malloc((size_t)experts * sizeof(*scores));
    float *selection = malloc((size_t)experts * sizeof(*selection));
    unsigned char *selected = calloc((size_t)experts, 1);
    if (!scores || !selection || !selected) {
        free(scores);
        free(selection);
        free(selected);
        return -1;
    }
    for (int expert = 0; expert < experts; expert++) {
        float sum = 0.0f;
        for (int column = 0; column < dimension; column++)
            sum += gate[(size_t)expert * dimension + column] * hidden[column];
        scores[expert] = sqrtf(softplusf_stable(sum));
        selection[expert] = scores[expert] + (bias ? bias[expert] : 0.0f);
    }
    if (forced_indices) {
        for (int rank = 0; rank < topk; rank++) {
            if (forced_indices[rank] < 0 || forced_indices[rank] >= experts) {
                free(selected);
                free(selection);
                free(scores);
                return -1;
            }
            indices[rank] = forced_indices[rank];
        }
    } else {
        for (int rank = 0; rank < topk; rank++) {
            int best = -1;
            for (int expert = 0; expert < experts; expert++) {
                if (!selected[expert] &&
                    (best < 0 || selection[expert] > selection[best]))
                    best = expert;
            }
            indices[rank] = best;
            selected[best] = 1;
        }
    }
    float total = 0.0f;
    for (int rank = 0; rank < topk; rank++)
        total += scores[indices[rank]];
    if (!(total > 0.0f)) {
        free(selected);
        free(selection);
        free(scores);
        return -1;
    }
    for (int rank = 0; rank < topk; rank++)
        weights[rank] = scores[indices[rank]] / total * route_scale;
    free(selected);
    free(selection);
    free(scores);
    return 0;
}

int coli_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit) {
    if (!output || !gate || !up || dimension < 1 || limit < 0.0f)
        return -1;
    for (int index = 0; index < dimension; index++) {
        float gate_value = gate[index];
        float up_value = up[index];
        if (limit > 0.0f) {
            gate_value = fminf(gate_value, limit);
            up_value = fmaxf(-limit, fminf(up_value, limit));
        }
        output[index] = gate_value * sigmoidf_stable(gate_value) * up_value;
    }
    return 0;
}
