#include <time.h>

#define main coli_v4_first_token_legacy_main
#include "deepseek_v4_first_token.c"
#undef main

#include "../deepseek_v4_block_batch.h"
#include "../deepseek_v4_dspark_capture.h"
#include "../deepseek_v4_dspark_runner.h"
#include "../deepseek_v4_dspark_runner_shared.h"
#include "../deepseek_v4_speculative.h"
#include "../deepseek_v4_target_verify.h"

static double spec_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + value.tv_nsec * 1e-9;
}

static int spec_sentence_end(const char *text, int length) {
    for (int i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '.' || c == '!' || c == '?' || c == '\n') return 1;
        if (i + 2 < length && c == 0xe3 &&
            (unsigned char)text[i + 1] == 0x80 &&
            (unsigned char)text[i + 2] == 0x82) return 1;
        if (i + 2 < length && c == 0xef &&
            (unsigned char)text[i + 1] == 0xbc &&
            ((unsigned char)text[i + 2] == 0x81 ||
             (unsigned char)text[i + 2] == 0x9f)) return 1;
    }
    return 0;
}

#ifndef COLI_V4_GENERATE_HELPERS_ONLY
static int spec_print(Tok *tokenizer, int token, float logit,
                      int position, int ordinal, int stop_sentence) {
    char piece[1024];
    int length = tok_decode(tokenizer, &token, 1, piece, sizeof(piece) - 1);
    printf("generated=%d position=%d token=%d logit=%.9g piece=",
           ordinal, position, token, logit);
    fwrite(piece, 1, (size_t)length, stdout); fputc('\n', stdout); fflush(stdout);
    return token == 1 || (stop_sentence && spec_sentence_end(piece, length));
}
#endif

static int target_batch(float **state_ptr, float **next_ptr,
                        ColiDeepSeekV4WindowAttentionState **attention,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        ColiExpertStore *experts, const int *tokens,
                        int start, int batch, char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(&layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_batch_ref(
            next, attention[layer_id], &layer, config, experts,
            state, tokens, start, batch, error, error_size);
        coli_v4_layer_free(&layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
    }
    *state_ptr = state; *next_ptr = next; return 0;
}

static int target_token(float **state_ptr, float **next_ptr,
                        ColiDeepSeekV4WindowAttentionState **attention,
                        const ColiSafetensorsIndex *index,
                        const ColiDeepSeekV4Config *config,
                        ColiExpertStore *experts, int token, int position,
                        char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    if (load_embedding(state, index, config, token)) return -1;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(&layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_token_ref(
            next, attention[layer_id], &layer, config, experts,
            state, token, position, error, error_size);
        coli_v4_layer_free(&layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
    }
    *state_ptr = state; *next_ptr = next; return 0;
}

#ifndef COLI_V4_GENERATE_HELPERS_ONLY
static uint64_t env_u64(const char *name, uint64_t fallback) {
    const char *text = getenv(name); char *end = NULL;
    if (!text || !*text) return fallback;
    unsigned long long value = strtoull(text, &end, 10);
    return end != text && !*end && value ? (uint64_t)value : fallback;
}

static float env_float(const char *name, float fallback) {
    const char *text = getenv(name); char *end = NULL;
    if (!text || !*text) return fallback;
    float value = strtof(text, &end);
    return end != text && !*end && value > 0.0f && value <= 1.0f
        ? value : fallback;
}

#ifndef COLI_V4_GENERATE_MAIN
#define COLI_V4_GENERATE_MAIN main
#endif

int COLI_V4_GENERATE_MAIN(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s TARGET_MODEL DSPARK_MODEL PROMPT "
                        "[MAX_NEW_TOKENS] [--stop-sentence]\n", argv[0]);
        return 2;
    }
    int max_new = argc >= 5 ? atoi(argv[4]) : 32;
    int stop_sentence = argc == 6 && !strcmp(argv[5], "--stop-sentence");
    if (max_new < 1 || (argc == 6 && !stop_sentence)) return 2;
#ifdef _WIN32
    _putenv_s("COLI_V4_DSPARK_MODEL", argv[2]);
#else
    setenv("COLI_V4_DSPARK_MODEL", argv[2], 1);
#endif
    char error[512] = {0}, tokenizer_path[4096];
    ColiDeepSeekV4Config config; ColiSafetensorsIndex *index = NULL;
    ColiExpertStore *experts = NULL; ColiV4DSparkRunner *runner = NULL;
    if (coli_v4_config_load(&config, argv[1], error, sizeof(error)) ||
        coli_st_index_open(&index, argv[1], error, sizeof(error)) ||
        coli_deepseek_v4_expert_store_open(
            &(ColiDeepSeekV4ExpertStoreOptions){argv[1],
                config.num_hidden_layers, config.n_routed_experts, 4ULL << 30},
            &experts, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", argv[1]);
    Tok tokenizer; tok_load(&tokenizer, tokenizer_path);
    int prompt_capacity = (int)strlen(argv[3]) + 16;
    int *prompt_ids = malloc((size_t)prompt_capacity * sizeof(int));
    int *generated = malloc((size_t)(max_new + 64) * sizeof(int));
    int prompt_count = tok_encode(&tokenizer, argv[3], strlen(argv[3]),
                                  prompt_ids, prompt_capacity);
    if (!prompt_ids || !generated || prompt_count < 1 || prompt_count > 64)
        return 1;
    size_t hd = (size_t)config.hc_mult * config.hidden_size;
    float *state = malloc((size_t)64 * hd * sizeof(float));
    float *next = malloc((size_t)64 * hd * sizeof(float));
    float *hidden = malloc((size_t)config.hidden_size * sizeof(float));
    float *main_x_batch = malloc((size_t)64 * config.hidden_size * sizeof(float));
    ColiDeepSeekV4WindowAttentionState **attention = calloc(
        config.num_hidden_layers, sizeof(*attention));
    if (!state || !next || !hidden || !main_x_batch || !attention) return 1;
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        if (coli_v4_window_attention_create(&attention[layer], &config)) return 1;
    for (int item = 0; item < prompt_count; item++)
        if (load_embedding(state + (size_t)item * hd, index, &config,
                           prompt_ids[item])) return 1;

    double started = spec_now();
    if (target_batch(&state, &next, attention, index, &config, experts,
                     prompt_ids, 0, prompt_count, error, sizeof(error)) ||
        coli_v4_dspark_capture_main_x(main_x_batch, prompt_count, &config) ||
        coli_v4_dspark_runner_open(&runner, argv[2], argv[1], &config,
                                   256ULL << 20, error, sizeof(error)) ||
        coli_v4_dspark_runner_use_shared_heads(
            runner, coli_v4_dspark_capture_heads()) ||
        coli_v4_dspark_runner_prefill(runner, main_x_batch, 0, prompt_count,
                                      error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    const float *last = state + (size_t)(prompt_count - 1) * hd;
    int current; float current_logit;
    if (final_hidden(hidden, last, index, &config, error, sizeof(error)) ||
        head_argmax(hidden, index, &config, &current, &current_logit)) return 1;
    int generated_count = 0, last_processed = prompt_count - 1;
    generated[generated_count++] = current;
    int done = spec_print(&tokenizer, current, current_logit,
                          last_processed, generated_count, stop_sentence);
    double first_at = spec_now();

    ColiV4SpeculativeController controller;
    coli_v4_speculative_controller_init(
        &controller, env_u64("COLI_V4_DSPARK_MIN_PROPOSALS", 10),
        env_float("COLI_V4_DSPARK_DISABLE_THRESHOLD", 0.35f));
    int block = coli_v4_dspark_runner_block_size(runner);
    int drafts[64], verified[65]; float draft_logits[64];
    while (!done && generated_count < max_new) {
        if (!controller.enabled || last_processed == prompt_count - 1) {
            int position = last_processed + 1;
            if (target_token(&state, &next, attention, index, &config, experts,
                             current, position, error, sizeof(error)) ||
                final_hidden(hidden, state, index, &config, error, sizeof(error)) ||
                head_argmax(hidden, index, &config, &current, &current_logit) ||
                coli_v4_dspark_capture_main_x(main_x_batch, 1, &config)) {
                fprintf(stderr, "%s\n", error); return 1;
            }
            last_processed = position; generated[generated_count++] = current;
            done = spec_print(&tokenizer, current, current_logit,
                              last_processed, generated_count, stop_sentence);
            continue;
        }
        const float *main_x = main_x_batch;
        if (coli_v4_dspark_runner_draft(
                runner, main_x, current, last_processed,
                drafts, draft_logits, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
        ColiV4VerificationResult verification;
        if (coli_v4_target_verify_greedy_batch(
                &verification, verified, 65, attention, index, &config, experts,
                current, drafts, block, last_processed + 1,
                error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
        coli_v4_speculative_record(&controller, block,
                                   verification.accepted_draft_tokens);
        int committed = verification.accepted_draft_tokens + 1;
        if (coli_v4_dspark_capture_main_x(main_x_batch, committed, &config))
            return 1;
        memmove(main_x_batch,
                main_x_batch + (size_t)(committed - 1) * config.hidden_size,
                (size_t)config.hidden_size * sizeof(float));
        int base_position = last_processed + 1;
        last_processed += committed;
        fprintf(stderr,
                "dspark_verify proposed=%d accepted=%d rate=%.3f enabled=%d\n",
                block, verification.accepted_draft_tokens,
                coli_v4_speculative_acceptance(&controller), controller.enabled);
        for (int i = 0; i < verification.output_count &&
                        generated_count < max_new && !done; i++) {
            current = verified[i]; current_logit = 0.0f;
            generated[generated_count++] = current;
            done = spec_print(&tokenizer, current, current_logit,
                              base_position + i, generated_count, stop_sentence);
        }
    }
    double ended = spec_now();
    ColiExpertStoreStats stats; experts->ops->stats(experts, &stats);
    size_t text_capacity = (size_t)generated_count * 256 + 1;
    char *text = malloc(text_capacity);
    int text_count = generated_count;
    if (text_count && generated[text_count - 1] == 1) text_count--;
    int text_length = tok_decode(&tokenizer, generated, text_count,
                                 text, text_capacity - 1);
    printf("summary tokens=%d expert_reads=%llu bytes=%llu "
           "dspark_rounds=%llu proposed=%llu accepted=%llu rate=%.3f enabled=%d\n",
           generated_count, (unsigned long long)stats.misses,
           (unsigned long long)stats.bytes_read,
           (unsigned long long)controller.rounds,
           (unsigned long long)controller.proposed,
           (unsigned long long)controller.accepted,
           coli_v4_speculative_acceptance(&controller), controller.enabled);
    printf("generated_text="); fwrite(text, 1, text_length, stdout);
    printf("\ntiming time_to_first_token=%.3fs after_first=%.3fs total=%.3fs\n",
           first_at - started, ended - first_at, ended - started);
    return 0;
}
#endif
