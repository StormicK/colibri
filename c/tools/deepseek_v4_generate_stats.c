#define COLI_V4_GENERATE_MAIN coli_v4_generate_stats_legacy_main
#define COLI_V4_GENERATE_HELPERS_ONLY
#define spec_print spec_print_diagnostic_legacy
#include "deepseek_v4_generate_speculative.c"
#undef spec_print
#undef COLI_V4_GENERATE_HELPERS_ONLY
#undef COLI_V4_GENERATE_MAIN

#include "../deepseek_v4_prompt.h"
#include "../deepseek_v4_runtime.h"

#ifdef COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY
#include <omp.h>

#include "../deepseek_v4_dspark_capture.h"
#include "../deepseek_v4_target_verify_prefix.h"


typedef struct {
    float **state_ptr;
    float **next_ptr;
    ColiDeepSeekV4WindowAttentionState **attention;
    const ColiSafetensorsIndex *index;
    const ColiDeepSeekV4Config *config;
    ColiExpertStore *experts;
    int token;
    int position;
    int threads;
    int target_token;
    float target_logit;
    float *main_x;
    double seconds;
    int result;
    char error[512];
} ParallelPrefixJob;


static void *parallel_prefix_worker(void *argument) {
    ParallelPrefixJob *job = argument;
    double began = spec_now();
    int previous_threads = omp_get_max_threads();
    omp_set_num_threads(job->threads);
    float *hidden = malloc((size_t)job->config->hidden_size * sizeof(float));
    if (!hidden) {
        snprintf(job->error, sizeof(job->error),
                 "out of memory in parallel target prefix");
        job->result = -1;
    } else {
        job->result = target_token(
            job->state_ptr, job->next_ptr, job->attention, job->index,
            job->config, job->experts, job->token, job->position,
            job->error, sizeof(job->error));
        if (!job->result)
            job->result = final_hidden(
                hidden, *job->state_ptr, job->index, job->config,
                job->error, sizeof(job->error));
        if (!job->result)
            job->result = head_argmax(
                hidden, job->index, job->config,
                &job->target_token, &job->target_logit);
        if (!job->result)
            job->result = coli_v4_dspark_capture_main_x(
                job->main_x, 1, job->config);
    }
    free(hidden);
    omp_set_num_threads(previous_threads);
    job->seconds = spec_now() - began;
    return NULL;
}
#endif

static ColiExpertStoreStats stats_subtract(ColiExpertStoreStats end,
                                           ColiExpertStoreStats begin) {
    ColiExpertStoreStats delta = end;
    delta.requests -= begin.requests; delta.hits -= begin.hits;
    delta.misses -= begin.misses; delta.prefetched -= begin.prefetched;
    delta.prefetch_hits -= begin.prefetch_hits;
    delta.bytes_read -= begin.bytes_read;
    return delta;
}

static double stats_hit_rate(ColiExpertStoreStats stats) {
    return stats.requests ? 100.0 * stats.hits / stats.requests : 0.0;
}

#ifdef COLI_V4_EXPERIMENTAL_STATE_HASH
static uint64_t state_hash_v70(const float *values, size_t count) {
    const unsigned char *bytes = (const unsigned char *)values;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
#endif

typedef struct {
    const char *model_dir;
    const char *draft_model_dir;
    const char *prompt;
    const char *system_prompt;
    int max_new_tokens;
    int stop_sentence;
    double memory_gib;
    ColiDeepSeekV4PromptMode prompt_mode;
} V4CliOptions;

static void v4_cli_usage(FILE *stream, const char *program) {
    fprintf(stream,
        "usage: %s MODEL PROMPT [options]\n"
        "  --max-tokens N       maximum generated tokens (default: 128)\n"
        "  --memory-gb GiB      cap this process; otherwise use available RAM\n"
        "  --draft-model PATH   separate DSpark checkpoint (default: MODEL)\n"
        "  --system TEXT        optional system message\n"
        "  --thinking           enable the official V4 thinking prefix\n"
        "  --raw-prompt         bypass the default V4 chat template\n"
        "  --stop-sentence      stop after the first sentence terminator\n",
        program);
}

static int v4_cli_positive_int(const char *text, int *output) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (!text[0] || end == text || *end || value < 1 || value > 1048576)
        return -1;
    *output = (int)value;
    return 0;
}

static int v4_cli_memory(const char *text, double *output) {
    char *end = NULL;
    double value = strtod(text, &end);
    if (!text[0] || end == text || *end || value <= 0.0 || value > 1048576.0)
        return -1;
    *output = value;
    return 0;
}

static int v4_cli_parse(int argc, char **argv, V4CliOptions *options) {
    if (!options || argc < 3) return -1;
    memset(options, 0, sizeof(*options));
    options->model_dir = argv[1];
    options->draft_model_dir = argv[1];
    options->prompt = argv[2];
    options->max_new_tokens = 128;
    options->prompt_mode = COLI_V4_PROMPT_CHAT;
    for (int i = 3; i < argc; i++) {
        const char *option = argv[i];
        if (!strcmp(option, "--max-tokens")) {
            if (++i == argc ||
                v4_cli_positive_int(argv[i], &options->max_new_tokens))
                return -1;
        } else if (!strcmp(option, "--memory-gb")) {
            if (++i == argc || v4_cli_memory(argv[i], &options->memory_gib))
                return -1;
        } else if (!strcmp(option, "--draft-model")) {
            if (++i == argc || !argv[i][0]) return -1;
            options->draft_model_dir = argv[i];
        } else if (!strcmp(option, "--system")) {
            if (++i == argc) return -1;
            options->system_prompt = argv[i];
        } else if (!strcmp(option, "--thinking")) {
            if (options->prompt_mode == COLI_V4_PROMPT_RAW) return -1;
            options->prompt_mode = COLI_V4_PROMPT_THINKING;
        } else if (!strcmp(option, "--raw-prompt")) {
            if (options->prompt_mode == COLI_V4_PROMPT_THINKING) return -1;
            options->prompt_mode = COLI_V4_PROMPT_RAW;
        } else if (!strcmp(option, "--stop-sentence")) {
            options->stop_sentence = 1;
        } else {
            return -1;
        }
    }
    return 0;
}

static int spec_print(Tok *tokenizer, int token, float logit,
                      int position, int ordinal, int stop_sentence) {
    (void)logit; (void)position; (void)ordinal;
    if (token == 1) return 1;
    char piece[1024];
    int length = tok_decode(tokenizer, &token, 1, piece, sizeof(piece) - 1);
    if (length > 0) fwrite(piece, 1, (size_t)length, stdout);
    fflush(stdout);
    return stop_sentence && spec_sentence_end(piece, length);
}
int main(int argc, char **argv) {
    double process_started = spec_now();
    V4CliOptions cli;
    if (v4_cli_parse(argc, argv, &cli)) {
        v4_cli_usage(stderr, argc ? argv[0] : "deepseek-v4");
        return 2;
    }
    int max_new = cli.max_new_tokens;
    int stop_sentence = cli.stop_sentence;
    ColiDeepSeekV4RuntimeOptions *runtime;
    coli_v4_runtime_reset();
    runtime = coli_v4_runtime_options();
    runtime->target_model_dir = cli.model_dir;
    runtime->dspark_model_dir = cli.draft_model_dir;
    if (cli.memory_gib > 0.0)
        runtime->memory_limit_bytes =
            (uint64_t)(cli.memory_gib * 1073741824.0);

    char *prompt = NULL;
    size_t prompt_length = 0;
    if (coli_v4_prompt_build(&prompt, &prompt_length, cli.prompt,
                             cli.system_prompt, cli.prompt_mode) ||
        prompt_length > INT_MAX - 16) {
        fprintf(stderr, "cannot build DeepSeek V4 prompt\n");
        return 1;
    }
    fprintf(stderr, "v4_cli mode=%s memory=%s draft_model=%s\n",
            cli.prompt_mode == COLI_V4_PROMPT_RAW ? "raw" :
            cli.prompt_mode == COLI_V4_PROMPT_THINKING ? "thinking" : "chat",
            cli.memory_gib > 0.0 ? "limited" : "auto",
            cli.draft_model_dir);

    char error[512] = {0}, tokenizer_path[4096];
    ColiDeepSeekV4Config config; ColiSafetensorsIndex *index = NULL;
    ColiExpertStore *experts = NULL; ColiV4DSparkRunner *runner = NULL;
    if (coli_v4_config_load(&config, cli.model_dir, error, sizeof(error)) ||
        coli_st_index_open(&index, cli.model_dir, error, sizeof(error)) ||
        coli_deepseek_v4_expert_store_open(
            &(ColiDeepSeekV4ExpertStoreOptions){cli.model_dir,
                config.num_hidden_layers, config.n_routed_experts, 4ULL << 30},
            &experts, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
             cli.model_dir);
    Tok tokenizer; tok_load(&tokenizer, tokenizer_path);
    int prompt_capacity = (int)prompt_length + 16;
    int *prompt_ids = malloc((size_t)prompt_capacity * sizeof(int));
    int *generated = malloc((size_t)(max_new + 64) * sizeof(int));
    int prompt_count = tok_encode(&tokenizer, prompt, prompt_length,
                                  prompt_ids, prompt_capacity);
    free(prompt);
    if (!prompt_ids || !generated || prompt_count < 1 || prompt_count > 512) {
        fprintf(stderr, "V4 prompt must encode to between 1 and 512 tokens\n");
        return 1;
    }
    size_t hd = (size_t)config.hc_mult * config.hidden_size;
    float *state = malloc((size_t)prompt_count * hd * sizeof(float));
    float *next = malloc((size_t)prompt_count * hd * sizeof(float));
    float *hidden = malloc((size_t)config.hidden_size * sizeof(float));
    float *main_x_batch = malloc((size_t)prompt_count * config.hidden_size * sizeof(float));
    ColiDeepSeekV4WindowAttentionState **attention = calloc(
        config.num_hidden_layers, sizeof(*attention));
    if (!state || !next || !hidden || !main_x_batch || !attention) return 1;
    for (int layer = 0; layer < config.num_hidden_layers; layer++)
        if (coli_v4_window_attention_create(&attention[layer], &config)) return 1;
    for (int item = 0; item < prompt_count; item++)
        if (load_embedding(state + (size_t)item * hd, index, &config,
                           prompt_ids[item])) return 1;

    double setup_done = spec_now(), phase_started = setup_done;
    ColiExpertStoreStats stats_before = {0}, stats_after_prefill = {0};
    experts->ops->stats(experts, &stats_before);
    if (target_batch(&state, &next, attention, index, &config, experts,
                     prompt_ids, 0, prompt_count, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    double target_prefill_done = spec_now();
    experts->ops->stats(experts, &stats_after_prefill);
    if (coli_v4_dspark_capture_main_x(main_x_batch, prompt_count, &config))
        return 1;
    double capture_done = spec_now();
    if (coli_v4_dspark_runner_open(&runner, cli.draft_model_dir, cli.model_dir,
                                   &config,
                                   256ULL << 20, error, sizeof(error)) ||
        coli_v4_dspark_runner_use_shared_heads(
            runner, coli_v4_dspark_capture_heads())) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    double dspark_open_done = spec_now();
    if (coli_v4_dspark_runner_prefill(runner, main_x_batch, 0, prompt_count,
                                      error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    double dspark_prefill_done = spec_now();
    const float *last = state + (size_t)(prompt_count - 1) * hd;
    int current; float current_logit;
    if (final_hidden(hidden, last, index, &config, error, sizeof(error)) ||
        head_argmax(hidden, index, &config, &current, &current_logit)) return 1;
    double head_done = spec_now();
    int generated_count = 0, last_processed = prompt_count - 1;
    generated[generated_count++] = current;
    int done = spec_print(&tokenizer, current, current_logit,
                          last_processed, generated_count, stop_sentence);
    double first_at = spec_now();

    ColiV4SpeculativeController controller;
    coli_v4_speculative_controller_init(&controller, 10, 0.35f);
    int block = coli_v4_dspark_runner_block_size(runner);
    int drafts[64], verified[65]; float draft_logits[64];
    double target_single_seconds = 0.0, decode_head_seconds = 0.0;
    double draft_seconds = 0.0, verify_seconds = 0.0, commit_seconds = 0.0;
    uint64_t target_single_calls = 0, draft_calls = 0, verify_calls = 0;
#ifdef COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY
    double prefix_seconds = 0.0, parallel_phase_seconds = 0.0;
    uint64_t prefix_calls = 0;
    int configured_threads = omp_get_max_threads();
    int prefix_threads = configured_threads / 2;
    int draft_threads = configured_threads - prefix_threads;
    if (prefix_threads < 1) prefix_threads = 1;
    if (draft_threads < 1) draft_threads = 1;
    int parallel_prefix_enabled = block == 4 && configured_threads >= 4;
    fprintf(stderr,
            "v4_parallel_prefix enabled=%d prefix_threads=%d "
            "draft_threads=%d total=%d\n",
            parallel_prefix_enabled, prefix_threads, draft_threads,
            configured_threads);
#endif
    while (!done && generated_count < max_new) {
        if (!controller.enabled || last_processed == prompt_count - 1) {
            int position = last_processed + 1;
            double t0 = spec_now();
            if (target_token(&state, &next, attention, index, &config, experts,
                             current, position, error, sizeof(error))) {
                fprintf(stderr, "%s\n", error); return 1;
            }
            target_single_seconds += spec_now() - t0; target_single_calls++;
            t0 = spec_now();
            if (final_hidden(hidden, state, index, &config, error, sizeof(error)) ||
                head_argmax(hidden, index, &config, &current, &current_logit))
                return 1;
            decode_head_seconds += spec_now() - t0; t0 = spec_now();
            if (coli_v4_dspark_capture_main_x(main_x_batch, 1, &config)) return 1;
            commit_seconds += spec_now() - t0;
            last_processed = position; generated[generated_count++] = current;
            done = spec_print(&tokenizer, current, current_logit,
                              last_processed, generated_count, stop_sentence);
            continue;
        }
        const float *main_x = main_x_batch;
        ColiV4VerificationResult verification;
        double t0;
#ifdef COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY
        if (!parallel_prefix_enabled) {
            t0 = spec_now();
            if (coli_v4_dspark_runner_draft(
                    runner, main_x, current, last_processed,
                    drafts, draft_logits, error, sizeof(error))) {
                fprintf(stderr, "%s\n", error); return 1;
            }
            draft_seconds += spec_now() - t0; draft_calls++;
            t0 = spec_now();
            if (coli_v4_target_verify_greedy_batch(
                    &verification, verified, 65, attention, index, &config,
                    experts, current, drafts, block, last_processed + 1,
                    error, sizeof(error))) {
                fprintf(stderr, "%s\n", error); return 1;
            }
            verify_seconds += spec_now() - t0; verify_calls++;
        } else {
        float *prefix_main_x = malloc(
            (size_t)config.hidden_size * sizeof(*prefix_main_x));
        if (!prefix_main_x) return 1;
        ParallelPrefixJob prefix_job = {
            .state_ptr = &state,
            .next_ptr = &next,
            .attention = attention,
            .index = index,
            .config = &config,
            .experts = experts,
            .token = current,
            .position = last_processed + 1,
            .threads = prefix_threads,
            .target_token = -1,
            .main_x = prefix_main_x,
            .result = -1,
        };
        pthread_t prefix_thread;
        double parallel_began = spec_now();
        if (pthread_create(&prefix_thread, NULL,
                           parallel_prefix_worker, &prefix_job)) {
            free(prefix_main_x);
            fprintf(stderr, "could not create parallel target prefix\n");
            return 1;
        }
        int previous_threads = omp_get_max_threads();
        omp_set_num_threads(draft_threads);
        t0 = spec_now();
        int draft_result = coli_v4_dspark_runner_draft(
            runner, main_x, current, last_processed,
            drafts, draft_logits, error, sizeof(error));
        draft_seconds += spec_now() - t0; draft_calls++;
        omp_set_num_threads(previous_threads);
        int prefix_join = pthread_join(prefix_thread, NULL);
        parallel_phase_seconds += spec_now() - parallel_began;
        prefix_seconds += prefix_job.seconds; prefix_calls++;
        if (draft_result) {
            free(prefix_main_x);
            fprintf(stderr, "%s\n", error); return 1;
        }
        if (prefix_join || prefix_job.result) {
            fprintf(stderr, "%s\n", prefix_job.error[0]
                    ? prefix_job.error : "parallel target prefix failed");
            free(prefix_main_x); return 1;
        }
        t0 = spec_now();
        if (drafts[0] != prefix_job.target_token) {
            verified[0] = prefix_job.target_token;
            verification.accepted_draft_tokens = 0;
            verification.output_count = 1;
            verification.mismatch_index = 0;
            if (coli_v4_dspark_capture_stage_main_x(
                    prefix_main_x, 1, config.hidden_size)) {
                free(prefix_main_x); return 1;
            }
        } else if (coli_v4_target_verify_after_prefix_v69(
                       &verification, verified, 65, attention, index,
                       &config, experts, drafts, block, last_processed + 1,
                       prefix_main_x, error, sizeof(error))) {
            free(prefix_main_x);
            fprintf(stderr, "%s\n", error); return 1;
        }
        free(prefix_main_x);
        verify_seconds += spec_now() - t0; verify_calls++;
        }
#else
        t0 = spec_now();
        if (coli_v4_dspark_runner_draft(
                runner, main_x, current, last_processed,
                drafts, draft_logits, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
        draft_seconds += spec_now() - t0; draft_calls++;
        t0 = spec_now();
        if (coli_v4_target_verify_greedy_batch(
                &verification, verified, 65, attention, index, &config, experts,
                current, drafts, block, last_processed + 1,
                error, sizeof(error))) {
            fprintf(stderr, "%s\n", error); return 1;
        }
        verify_seconds += spec_now() - t0; verify_calls++;
#endif
        coli_v4_speculative_record(&controller, block,
                                   verification.accepted_draft_tokens);
        int committed = verification.accepted_draft_tokens + 1;
        t0 = spec_now();
        if (coli_v4_dspark_capture_main_x(main_x_batch, committed, &config))
            return 1;
        memmove(main_x_batch,
                main_x_batch + (size_t)(committed - 1) * config.hidden_size,
                (size_t)config.hidden_size * sizeof(float));
#ifdef COLI_V4_EXPERIMENTAL_STATE_HASH
        fprintf(stderr,
                "v4_state_hash round=%llu committed=%d main_x=%016llx\n",
                (unsigned long long)controller.rounds, committed,
                (unsigned long long)state_hash_v70(
                    main_x_batch, (size_t)config.hidden_size));
#endif
        commit_seconds += spec_now() - t0;
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
    ColiExpertStoreStats stats_end = {0}; experts->ops->stats(experts, &stats_end);
    ColiExpertStoreStats prefill_stats = stats_subtract(
        stats_after_prefill, stats_before);
    ColiExpertStoreStats decode_stats = stats_subtract(
        stats_end, stats_after_prefill);
    size_t text_capacity = (size_t)generated_count * 256 + 1;
    char *text = malloc(text_capacity);
    int text_count = generated_count;
    if (text_count && generated[text_count - 1] == 1) text_count--;
    int text_length = tok_decode(&tokenizer, generated, text_count,
                                 text, text_capacity - 1);
    int decode_tokens = generated_count - 1;
    double decode_seconds = ended - first_at;
#ifdef COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY
    double scheduled_draft_seconds = parallel_prefix_enabled
        ? parallel_phase_seconds : draft_seconds;
    double timed_decode = target_single_seconds + decode_head_seconds +
        scheduled_draft_seconds + verify_seconds + commit_seconds;
#else
    double timed_decode = target_single_seconds + decode_head_seconds +
        draft_seconds + verify_seconds + commit_seconds;
#endif
    fputc('\n', stdout); fflush(stdout);
    fprintf(stderr, "summary tokens=%d prompt_tokens=%d expert_requests=%llu "
           "expert_hits=%llu expert_reads=%llu hit_rate=%.3f bytes=%llu "
           "dspark_rounds=%llu proposed=%llu accepted=%llu rate=%.3f enabled=%d\n",
           generated_count, prompt_count,
           (unsigned long long)stats_end.requests,
           (unsigned long long)stats_end.hits,
           (unsigned long long)stats_end.misses, stats_hit_rate(stats_end),
           (unsigned long long)stats_end.bytes_read,
           (unsigned long long)controller.rounds,
           (unsigned long long)controller.proposed,
           (unsigned long long)controller.accepted,
           coli_v4_speculative_acceptance(&controller), controller.enabled);
    fprintf(stderr, "generated_text="); fwrite(text, 1, text_length, stderr);
    fprintf(stderr, "\nprefill_timing startup=%.6fs target=%.6fs capture=%.6fs "
           "dspark_open=%.6fs dspark=%.6fs first_head=%.6fs "
           "pipeline=%.6fs wall_to_first=%.6fs target_tok_s=%.6f "
           "combined_tok_s=%.6f\n",
           setup_done - process_started,
           target_prefill_done - phase_started,
           capture_done - target_prefill_done,
           dspark_open_done - capture_done,
           dspark_prefill_done - dspark_open_done,
           head_done - dspark_prefill_done,
           first_at - setup_done, first_at - process_started,
           prompt_count / (target_prefill_done - phase_started),
           prompt_count / ((target_prefill_done - phase_started) +
                           (dspark_prefill_done - dspark_open_done)));
    fprintf(stderr, "prefill_experts requests=%llu hits=%llu misses=%llu "
           "hit_rate=%.3f bytes=%llu prefetched=%llu prefetch_hits=%llu\n",
           (unsigned long long)prefill_stats.requests,
           (unsigned long long)prefill_stats.hits,
           (unsigned long long)prefill_stats.misses,
           stats_hit_rate(prefill_stats),
           (unsigned long long)prefill_stats.bytes_read,
           (unsigned long long)prefill_stats.prefetched,
           (unsigned long long)prefill_stats.prefetch_hits);
    fprintf(stderr, "decode_timing tokens=%d seconds=%.6f tok_s=%.6f sec_per_tok=%.6f "
           "target_single=%.6f single_calls=%llu draft=%.6f draft_calls=%llu "
           "verify=%.6f verify_calls=%llu head=%.6f commit=%.6f other=%.6f\n",
           decode_tokens, decode_seconds,
           decode_tokens / decode_seconds, decode_seconds / decode_tokens,
           target_single_seconds, (unsigned long long)target_single_calls,
           draft_seconds, (unsigned long long)draft_calls,
           verify_seconds, (unsigned long long)verify_calls,
           decode_head_seconds, commit_seconds, decode_seconds - timed_decode);
#ifdef COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY
    fprintf(stderr, "parallel_prefix enabled=%d prefix=%.6f prefix_calls=%llu "
           "parallel_phase=%.6f prefix_threads=%d draft_threads=%d\n",
           parallel_prefix_enabled, prefix_seconds,
           (unsigned long long)prefix_calls, parallel_phase_seconds,
           prefix_threads, draft_threads);
#endif
    fprintf(stderr, "decode_experts requests=%llu hits=%llu misses=%llu "
           "hit_rate=%.3f bytes=%llu prefetched=%llu prefetch_hits=%llu\n",
           (unsigned long long)decode_stats.requests,
           (unsigned long long)decode_stats.hits,
           (unsigned long long)decode_stats.misses,
           stats_hit_rate(decode_stats),
           (unsigned long long)decode_stats.bytes_read,
           (unsigned long long)decode_stats.prefetched,
           (unsigned long long)decode_stats.prefetch_hits);
    fprintf(stderr, "timing time_to_first_token=%.3fs after_first=%.3fs total=%.3fs\n",
           first_at - setup_done, decode_seconds, ended - setup_done);
    return 0;
}
