#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../deepseek_v4_block.h"
#include "../deepseek_v4_config.h"
#include "../deepseek_v4_expert_store.h"
#include "../deepseek_v4_math.h"
#include "../deepseek_v4_layer.h"
#include "../native_quant.h"
#include "../safetensors_index.h"
#include "../tensor_io.h"
#include "../tok.h"

static int load_embedding(float *state, const ColiSafetensorsIndex *index,
                          const ColiDeepSeekV4Config *config, int token) {
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    int d = config->hidden_size, hc = config->hc_mult;
    uint16_t *row = malloc((size_t)d * sizeof(*row));
    if (!embed || embed->dtype != COLI_ST_BF16 || !row || token < 0 ||
        token >= config->vocab_size ||
        coli_st_read_at(index, embed->shard,
                        embed->offset + (uint64_t)token * d * sizeof(*row),
                        (size_t)d * sizeof(*row), row)) {
        free(row);
        return -1;
    }
    for (int copy = 0; copy < hc; copy++)
        for (int i = 0; i < d; i++)
            state[(size_t)copy * d + i] = coli_bf16_decode(row[i]);
    free(row);
    return 0;
}

static int final_hidden(float *output, const float *state,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        char *error, size_t error_size) {
    ColiFloatTensor function = {0}, base = {0}, scale = {0}, norm = {0};
    if (coli_tensor_load_f32(&function, index, "hc_head_fn", error, error_size) ||
        coli_tensor_load_f32(&base, index, "hc_head_base", error, error_size) ||
        coli_tensor_load_f32(&scale, index, "hc_head_scale", error, error_size) ||
        coli_tensor_load_f32(&norm, index, "norm.weight", error, error_size))
        return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    int flattened = hc * d;
    float square = 0.0f;
    for (int i = 0; i < flattened; i++) square += state[i] * state[i];
    float inverse_rms = 1.0f / sqrtf(square / flattened + config->rms_norm_eps);
    float pre[16];
    if (hc > 16) return -1;
    for (int copy = 0; copy < hc; copy++) {
        float mix = 0.0f;
        for (int i = 0; i < flattened; i++)
            mix += function.data[(size_t)copy * flattened + i] * state[i];
        mix *= inverse_rms;
        float z = mix * scale.data[0] + base.data[copy];
        float sigmoid = z >= 0.0f
            ? 1.0f / (1.0f + expf(-z))
            : expf(z) / (1.0f + expf(z));
        pre[copy] = sigmoid + config->hc_eps;
    }
    for (int i = 0; i < d; i++) {
        float value = 0.0f;
        for (int copy = 0; copy < hc; copy++)
            value += pre[copy] * state[(size_t)copy * d + i];
        output[i] = coli_bf16_round(value);
    }
    coli_v4_rmsnorm(output, output, norm.data, d, config->rms_norm_eps);
    coli_bf16_round_array(output, (size_t)d);
    coli_float_tensor_free(&norm);
    coli_float_tensor_free(&scale);
    coli_float_tensor_free(&base);
    coli_float_tensor_free(&function);
    return 0;
}

static int head_argmax(const float *hidden, const ColiSafetensorsIndex *index,
                       const ColiDeepSeekV4Config *config,
                       int *best_token, float *best_logit) {
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
    int d = config->hidden_size, vocab = config->vocab_size;
    enum { ROWS = 64 };
    uint16_t *raw = malloc((size_t)ROWS * d * sizeof(*raw));
    float *scores = malloc((size_t)ROWS * sizeof(*scores));
    if (!head || head->dtype != COLI_ST_BF16 || !raw || !scores) {
        free(scores); free(raw);
        return -1;
    }
    int winner = -1;
    float maximum = -FLT_MAX;
    for (int start = 0; start < vocab; start += ROWS) {
        int rows = vocab - start < ROWS ? vocab - start : ROWS;
        size_t bytes = (size_t)rows * d * sizeof(*raw);
        if (coli_st_read_at(index, head->shard,
                            head->offset + (uint64_t)start * d * sizeof(*raw),
                            bytes, raw)) {
            free(scores); free(raw);
            return -1;
        }
        #pragma omp parallel for
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            const uint16_t *weight = raw + (size_t)row * d;
            for (int i = 0; i < d; i++)
                sum += coli_bf16_decode(weight[i]) * hidden[i];
            scores[row] = sum;
        }
        for (int row = 0; row < rows; row++)
            if (scores[row] > maximum) {
                maximum = scores[row];
                winner = start + row;
            }
    }
    free(scores); free(raw);
    *best_token = winner;
    *best_logit = maximum;
    return winner < 0 ? -1 : 0;
}

static int has_sentence_end(const char *text, int length) {
    for (int i = 0; i < length; i++) {
        unsigned char value = (unsigned char)text[i];
        if (value == '.' || value == '!' || value == '?' || value == '\n') return 1;
        if (i + 2 < length && value == 0xe3 &&
            (unsigned char)text[i + 1] == 0x80 &&
            (unsigned char)text[i + 2] == 0x82) return 1; /* 。 */
        if (i + 2 < length && value == 0xef &&
            (unsigned char)text[i + 1] == 0xbc &&
            ((unsigned char)text[i + 2] == 0x81 ||
             (unsigned char)text[i + 2] == 0x9f)) return 1; /* ！？ */
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s MODEL_DIR INPUT_TOKEN_ID [TOKEN_COUNT]\n"
                        "       %s MODEL_DIR --prompt TEXT [MAX_NEW_TOKENS] [--stop-sentence]\n",
                argv[0], argv[0]);
        return 2;
    }
    int text_mode = !strcmp(argv[2], "--prompt");
    if (text_mode && argc < 4) return 2;
    int stop_sentence = text_mode && argc == 6 &&
                        !strcmp(argv[5], "--stop-sentence");
    if (text_mode && argc == 6 && !stop_sentence) return 2;
    int input_token = text_mode ? -1 : atoi(argv[2]);
    int token_count = text_mode ? (argc == 5 ? atoi(argv[4]) : 32)
                                : (argc == 4 ? atoi(argv[3]) : 1);
    if (token_count < 1) return 2;
    char error[512] = {0};
    ColiDeepSeekV4Config config;
    ColiSafetensorsIndex *index = NULL;
    ColiExpertStore *experts = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_deepseek_v4_expert_store_open(
            &(ColiDeepSeekV4ExpertStoreOptions){
                argv[1], config.num_hidden_layers, config.n_routed_experts,
                UINT64_C(4) * 1024 * 1024 * 1024,
            }, &experts, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    Tok tokenizer;
    int *prompt_ids = NULL, prompt_count = 0;
    int *generated_ids = NULL, generated_count = 0;
    if (text_mode) {
        char tokenizer_path[4096];
        snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);
        tok_load(&tokenizer, tokenizer_path);
        int prompt_capacity = (int)strlen(argv[3]) + 16;
        prompt_ids = malloc((size_t)prompt_capacity * sizeof(*prompt_ids));
        generated_ids = malloc((size_t)token_count * sizeof(*generated_ids));
        if (!prompt_ids || !generated_ids) return 1;
        prompt_count = tok_encode(&tokenizer, argv[3], (int)strlen(argv[3]),
                                  prompt_ids, prompt_capacity);
        if (prompt_count < 1) {
            fprintf(stderr, "prompt produced no tokens\n"); return 1;
        }
        fprintf(stderr, "prompt_tokens=%d max_new_tokens=%d eos_token=1\n",
                prompt_count, token_count);
    }
    size_t state_count = (size_t)config.hc_mult * config.hidden_size;
    float *state = malloc(state_count * sizeof(*state));
    float *next = malloc(state_count * sizeof(*next));
    float *hidden = malloc((size_t)config.hidden_size * sizeof(*hidden));
    ColiDeepSeekV4WindowAttentionState **attention = calloc(
        (size_t)config.num_hidden_layers, sizeof(*attention));
    if (!state || !next || !hidden || !attention) return 1;
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        if (coli_v4_window_attention_create(&attention[layer], &config)) return 1;

    int current_token = text_mode ? prompt_ids[0] : input_token;
    int total_steps = text_mode ? prompt_count + token_count - 1 : token_count;
    for (int position = 0; position < total_steps; position++) {
        if (text_mode && position < prompt_count)
            current_token = prompt_ids[position];
        if (load_embedding(state, index, &config, current_token)) return 1;
        for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++) {
            ColiDeepSeekV4LayerWeights layer;
            if (coli_v4_layer_load(&layer, &config, index, layer_id,
                                   error, sizeof(error)) ||
                coli_v4_block_window_token_ref(
                    next, attention[layer_id], &layer, &config, experts, state,
                    current_token, position, error, sizeof(error))) {
                fprintf(stderr, "position %d layer %d: %s\n",
                        position, layer_id, error);
                return 1;
            }
            coli_v4_layer_free(&layer);
            float *swap = state; state = next; next = swap;
        }
        fprintf(stderr, "position %d/%d complete (%d layers)\n", position,
                total_steps - 1, config.num_hidden_layers);
        if (final_hidden(hidden, state, index, &config, error, sizeof(error))) {
            fprintf(stderr, "final hidden: %s\n", error);
            return 1;
        }
        int output_token;
        float output_logit;
        if (head_argmax(hidden, index, &config, &output_token, &output_logit)) {
            fprintf(stderr, "lm_head failed\n");
            return 1;
        }
        if (!text_mode) {
            printf("position=%d input_token=%d output_token=%d logit=%.9g\n",
                   position, current_token, output_token, output_logit);
        } else if (position >= prompt_count - 1) {
            generated_ids[generated_count++] = output_token;
            char piece[1024];
            int piece_length = tok_decode(&tokenizer, &output_token, 1,
                                          piece, (int)sizeof(piece) - 1);
            printf("generated=%d position=%d token=%d logit=%.9g piece=",
                   generated_count, position, output_token, output_logit);
            fwrite(piece, 1, (size_t)piece_length, stdout);
            fputc('\n', stdout);
            fflush(stdout);
            if (output_token == 1) break;
            if (stop_sentence && has_sentence_end(piece, piece_length)) break;
        }
        current_token = output_token;
    }
    ColiExpertStoreStats stats;
    experts->ops->stats(experts, &stats);
    printf("summary tokens=%d expert_reads=%llu bytes=%llu\n",
           text_mode ? generated_count : token_count,
           (unsigned long long)stats.misses,
           (unsigned long long)stats.bytes_read);
    if (text_mode) {
        size_t text_capacity = (size_t)generated_count * 256 + 1;
        char *text = malloc(text_capacity);
        if (!text) return 1;
        int decode_count = generated_count;
        if (decode_count && generated_ids[decode_count - 1] == 1) decode_count--;
        int text_length = tok_decode(&tokenizer, generated_ids, decode_count,
                                     text, (int)text_capacity - 1);
        printf("generated_text=");
        fwrite(text, 1, (size_t)text_length, stdout);
        fputc('\n', stdout);
        printf("completed_text=");
        fwrite(argv[3], 1, strlen(argv[3]), stdout);
        fwrite(text, 1, (size_t)text_length, stdout);
        fputc('\n', stdout);
        free(text);
    }
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        coli_v4_window_attention_destroy(attention[layer]);
    free(attention);
    free(generated_ids); free(prompt_ids);
    free(hidden); free(next); free(state);
    experts->ops->destroy(experts);
    coli_st_index_close(index);
    return 0;
}
