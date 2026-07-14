/* ---- begin inlined deepseek_v4_dspark_heads_v2.c ---- */
/* ---- begin inlined deepseek_v4_dspark_heads.c ---- */
#include "deepseek_v4_dspark_heads.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_math.h"
#include "native_quant.h"
#include "safetensors_index.h"

struct ColiV4DSparkHeads {
    int hidden, hc, targets, vocab, rank;
    float rms_eps;
    ColiTensorView main_proj;
    unsigned char *main_weight;
    unsigned char *main_scale;
    float *main_norm;
    uint16_t *markov_w1;
    uint16_t *markov_w2;
};

static int heads_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static void *read_named(ColiSafetensorsIndex *index, const char *name,
                        ColiSafetensorsDType dtype, char *error, size_t size) {
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor || tensor->dtype != dtype) {
        heads_error(error, size, "invalid DSpark head tensor: %s", name);
        return NULL;
    }
    void *data = malloc((size_t)tensor->nbytes);
    if (!data || coli_st_read_tensor(index, tensor, data)) {
        free(data); heads_error(error, size, "cannot read DSpark head: %s", name);
        return NULL;
    }
    return data;
}

void coli_v4_dspark_heads_close(ColiV4DSparkHeads *heads) {
    if (!heads) return;
    free(heads->markov_w2); free(heads->markov_w1); free(heads->main_norm);
    free(heads->main_scale); free(heads->main_weight); free(heads);
}

int coli_v4_dspark_heads_open(ColiV4DSparkHeads **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size) {
    if (!output || !model_dir || !config || !manifest ||
        manifest->stage_count < 1) return -1;
    *output = NULL; ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    ColiV4DSparkHeads *heads = calloc(1, sizeof(*heads));
    if (!heads) { coli_st_index_close(index); return -1; }
    heads->hidden = config->hidden_size; heads->hc = config->hc_mult;
    heads->targets = manifest->target_count; heads->vocab = config->vocab_size;
    heads->rank = manifest->markov_rank; heads->rms_eps = config->rms_norm_eps;
    heads->main_weight = read_named(index, "mtp.0.main_proj.weight",
                                    COLI_ST_F8_E4M3, error, error_size);
    heads->main_scale = read_named(index, "mtp.0.main_proj.scale",
                                   COLI_ST_F8_E8M0, error, error_size);
    uint16_t *norm_bf16 = read_named(index, "mtp.0.main_norm.weight",
                                     COLI_ST_BF16, error, error_size);
    char name[160]; int last = manifest->stage_count - 1;
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w1.weight", last);
    heads->markov_w1 = read_named(index, name, COLI_ST_BF16, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w2.weight", last);
    heads->markov_w2 = read_named(index, name, COLI_ST_BF16, error, error_size);
    coli_st_index_close(index);
    heads->main_norm = malloc((size_t)heads->hidden * sizeof(float));
    if (!heads->main_weight || !heads->main_scale || !norm_bf16 ||
        !heads->markov_w1 || !heads->markov_w2 || !heads->main_norm) {
        free(norm_bf16); coli_v4_dspark_heads_close(heads); return -1;
    }
    for (int i = 0; i < heads->hidden; i++)
        heads->main_norm[i] = coli_bf16_decode(norm_bf16[i]);
    free(norm_bf16);
    size_t columns = (size_t)heads->hidden * heads->targets;
    heads->main_proj = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0,
        heads->main_weight, heads->main_scale,
        (size_t)heads->hidden * columns,
        (size_t)((heads->hidden + 127) / 128) * ((columns + 127) / 128),
        heads->hidden, (int64_t)columns, 128, 128
    };
    *output = heads; return 0;
}

int coli_v4_dspark_combine_hidden(ColiV4DSparkHeads *heads, float *output,
                                  const float *target_hidden_hc) {
    if (!heads || !output || !target_hidden_hc) return -1;
    size_t columns = (size_t)heads->hidden * heads->targets;
    float *combined = malloc(columns * sizeof(*combined));
    float *projected = malloc((size_t)heads->hidden * sizeof(*projected));
    if (!combined || !projected) { free(projected); free(combined); return -1; }
    size_t hc_width = (size_t)heads->hc * heads->hidden;
    for (int target = 0; target < heads->targets; target++)
        for (int column = 0; column < heads->hidden; column++) {
            float sum = 0.0f;
            for (int copy = 0; copy < heads->hc; copy++)
                sum += target_hidden_hc[(size_t)target * hc_width +
                                        (size_t)copy * heads->hidden + column];
            combined[(size_t)target * heads->hidden + column] =
                sum / (float)heads->hc;
        }
    int result = coli_fp8_matvec_ref(projected, &heads->main_proj, combined);
    if (!result) {
        coli_bf16_round_array(projected, (size_t)heads->hidden);
        result = coli_v4_rmsnorm(output, projected, heads->main_norm,
                                 heads->hidden, heads->rms_eps);
        if (!result) coli_bf16_round_array(output, (size_t)heads->hidden);
    }
    free(projected); free(combined); return result;
}

int coli_v4_dspark_markov_bias(ColiV4DSparkHeads *heads, float *bias,
                               int previous_token) {
    if (!heads || !bias || previous_token < 0 || previous_token >= heads->vocab)
        return -1;
    float *embedding = malloc((size_t)heads->rank * sizeof(*embedding));
    if (!embedding) return -1;
    const uint16_t *row = heads->markov_w1 + (size_t)previous_token * heads->rank;
    for (int i = 0; i < heads->rank; i++) embedding[i] = coli_bf16_decode(row[i]);
    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < heads->vocab; token++) {
        const uint16_t *weight = heads->markov_w2 + (size_t)token * heads->rank;
        float sum = 0.0f;
        for (int i = 0; i < heads->rank; i++)
            sum += embedding[i] * coli_bf16_decode(weight[i]);
        bias[token] = sum;
    }
    free(embedding); return 0;
}
/* ---- end inlined deepseek_v4_dspark_heads.c ---- */


#include <float.h>



int coli_v4_dspark_biased_argmax(
    ColiV4DSparkHeads *heads, const ColiSafetensorsIndex *target_index,
    const float *hidden, int previous_token,
    int *best_token, float *best_logit) {
    if (!heads || !target_index || !hidden || !best_token || !best_logit ||
        previous_token < 0 || previous_token >= heads->vocab) return -1;
    const ColiSafetensorsTensor *head = coli_st_find(target_index, "head.weight");
    if (!head || head->dtype != COLI_ST_BF16 || head->rank != 2 ||
        head->shape[0] != heads->vocab || head->shape[1] != heads->hidden)
        return -1;
    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * heads->hidden * sizeof(*raw));
    float *scores = malloc((size_t)ROWS * sizeof(*scores));
    float *markov = malloc((size_t)heads->rank * sizeof(*markov));
    if (!raw || !scores || !markov) {
        free(markov); free(scores); free(raw); return -1;
    }
    const uint16_t *m1 = heads->markov_w1 +
                         (size_t)previous_token * heads->rank;
    for (int i = 0; i < heads->rank; i++) markov[i] = coli_bf16_decode(m1[i]);
    int winner = -1; float maximum = -FLT_MAX;
    for (int start = 0; start < heads->vocab; start += ROWS) {
        int rows = heads->vocab - start < ROWS ? heads->vocab - start : ROWS;
        size_t bytes = (size_t)rows * heads->hidden * sizeof(*raw);
        if (coli_st_read_at(target_index, head->shard,
                            head->offset + (uint64_t)start * heads->hidden * 2,
                            bytes, raw)) {
            free(markov); free(scores); free(raw); return -1;
        }
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            const uint16_t *weight = raw + (size_t)row * heads->hidden;
            const uint16_t *m2 = heads->markov_w2 +
                                 (size_t)(start + row) * heads->rank;
            for (int i = 0; i < heads->hidden; i++)
                sum += coli_bf16_decode(weight[i]) * hidden[i];
            for (int i = 0; i < heads->rank; i++)
                sum += coli_bf16_decode(m2[i]) * markov[i];
            scores[row] = sum;
        }
        for (int row = 0; row < rows; row++)
            if (scores[row] > maximum) {
                maximum = scores[row]; winner = start + row;
            }
    }
    free(markov); free(scores); free(raw);
    *best_token = winner; *best_logit = maximum;
    return winner < 0 ? -1 : 0;
}
/* ---- end inlined deepseek_v4_dspark_heads_v2.c ---- */




int coli_v4_dspark_biased_argmax_batch(
    ColiV4DSparkHeads *heads, const ColiSafetensorsIndex *target_index,
    const float *hidden_batch, int previous_token,
    int *best_tokens, float *best_logits, int batch) {
    if (!heads || !target_index || !hidden_batch || !best_tokens ||
        !best_logits || batch < 1 || batch > 64 || previous_token < 0 ||
        previous_token >= heads->vocab) return -1;
    const ColiSafetensorsTensor *head = coli_st_find(target_index, "head.weight");
    if (!head || head->dtype != COLI_ST_BF16 || head->rank != 2 ||
        head->shape[0] != heads->vocab || head->shape[1] != heads->hidden)
        return -1;

    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * heads->hidden * sizeof(*raw));
    float *base = malloc((size_t)batch * heads->vocab * sizeof(*base));
    float *markov = malloc((size_t)heads->rank * sizeof(*markov));
    if (!raw || !base || !markov) {
        free(markov); free(base); free(raw); return -1;
    }

    /* Each sum still visits columns in scalar order; BF16 decode is shared. */
    for (int start = 0; start < heads->vocab; start += ROWS) {
        int rows = heads->vocab - start < ROWS ? heads->vocab - start : ROWS;
        size_t bytes = (size_t)rows * heads->hidden * sizeof(*raw);
        if (coli_st_read_at(target_index, head->shard,
                            head->offset + (uint64_t)start * heads->hidden * 2,
                            bytes, raw)) {
            free(markov); free(base); free(raw); return -1;
        }
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            const uint16_t *weight = raw + (size_t)row * heads->hidden;
            float sums[64] = {0};
            for (int i = 0; i < heads->hidden; i++) {
                float decoded = coli_bf16_decode(weight[i]);
                for (int item = 0; item < batch; item++)
                    sums[item] += decoded *
                        hidden_batch[(size_t)item * heads->hidden + i];
            }
            for (int item = 0; item < batch; item++)
                base[(size_t)item * heads->vocab + start + row] = sums[item];
        }
    }

    for (int item = 0; item < batch; item++) {
        const uint16_t *m1 = heads->markov_w1 +
                             (size_t)previous_token * heads->rank;
        for (int i = 0; i < heads->rank; i++)
            markov[i] = coli_bf16_decode(m1[i]);
        float *scores = base + (size_t)item * heads->vocab;
        #pragma omp parallel for schedule(static)
        for (int token = 0; token < heads->vocab; token++) {
            const uint16_t *m2 = heads->markov_w2 +
                                 (size_t)token * heads->rank;
            float sum = scores[token];
            for (int i = 0; i < heads->rank; i++)
                sum += coli_bf16_decode(m2[i]) * markov[i];
            scores[token] = sum;
        }
        int winner = 0;
        for (int token = 1; token < heads->vocab; token++)
            if (scores[token] > scores[winner]) winner = token;
        best_tokens[item] = winner;
        best_logits[item] = scores[winner];
        previous_token = winner;
    }
    free(markov); free(base); free(raw);
    return 0;
}
