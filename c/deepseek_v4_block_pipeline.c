/* Windows-native routed-expert I/O pipeline. The original block implementation
 * is retained under serial symbols. Public entry points normally keep three
 * persistent lookup workers so reads N+1..N+3 overlap ordered expert compute.
 * COLI_V4_DISABLE_DUAL_EXPERT_LOADER restores the one-worker pipeline. */
#if !defined(COLI_V4_DISABLE_DUAL_EXPERT_LOADER) && \
    !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER) && \
    !defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP) && \
    !defined(COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER)
#define COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
#endif
#define coli_v4_block_token_ref coli_v4_block_token_serial_ref
#define coli_v4_block_window_token_ref coli_v4_block_window_token_serial_ref
#include "deepseek_v4_block.c"
#undef coli_v4_block_token_ref
#undef coli_v4_block_window_token_ref

#ifndef COLI_V4_DISABLE_BF16_ROUTE
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
#endif

typedef struct {
    ColiExpertStore *store;
    ColiExpertKey key;
    ColiExpertView view;
    int result;
} ExpertLoadJob;

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH
static int expert_prefetch_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *text = getenv("COLI_V4_EXPERT_PREFETCH");
        enabled = text && *text && atoi(text) != 0;
    }
    return enabled;
}
#endif


static void *expert_load_worker(void *argument) {
    ExpertLoadJob *job = argument;
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    return NULL;
}

typedef struct {
    pthread_t thread;
    int active;
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    int loader_slot;
#endif
} ExpertLoadHandle;

#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
#ifndef COLI_V4_EXPERT_LOADER_COUNT
#define COLI_V4_EXPERT_LOADER_COUNT 3
#endif
enum { DUAL_EXPERT_LOADER_COUNT = COLI_V4_EXPERT_LOADER_COUNT };

typedef struct {
    pthread_t thread;
    ExpertLoadJob *job;
    int pending;
    int completed;
    int stopping;
    int available;
} DualExpertLoaderSlot;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t complete;
    pthread_cond_t idle;
    DualExpertLoaderSlot slots[DUAL_EXPERT_LOADER_COUNT];
} DualExpertLoaderPool;

static DualExpertLoaderPool dual_loader_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .ready = PTHREAD_COND_INITIALIZER,
    .complete = PTHREAD_COND_INITIALIZER,
    .idle = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t dual_loader_once = PTHREAD_ONCE_INIT;

static void *dual_expert_loader_worker(void *argument) {
    DualExpertLoaderSlot *slot = argument;
    pthread_mutex_lock(&dual_loader_pool.mutex);
    for (;;) {
        while (!slot->pending && !slot->stopping)
            pthread_cond_wait(&dual_loader_pool.ready,
                              &dual_loader_pool.mutex);
        if (slot->stopping) break;
        ExpertLoadJob *job = slot->job;
        slot->pending = 0;
        pthread_mutex_unlock(&dual_loader_pool.mutex);
        job->result = coli_expert_lookup(job->store, job->key, &job->view);
        pthread_mutex_lock(&dual_loader_pool.mutex);
        slot->completed = 1;
        pthread_cond_broadcast(&dual_loader_pool.complete);
    }
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    return NULL;
}

static void dual_expert_loader_shutdown(void) {
    pthread_mutex_lock(&dual_loader_pool.mutex);
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++)
        dual_loader_pool.slots[i].stopping = 1;
    pthread_cond_broadcast(&dual_loader_pool.ready);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++)
        if (dual_loader_pool.slots[i].available) {
            pthread_join(dual_loader_pool.slots[i].thread, NULL);
            dual_loader_pool.slots[i].available = 0;
        }
}

static void dual_expert_loader_init(void) {
    int available = 0;
    for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++) {
        DualExpertLoaderSlot *slot = &dual_loader_pool.slots[i];
        if (!pthread_create(&slot->thread, NULL,
                            dual_expert_loader_worker, slot)) {
            slot->available = 1;
            available++;
        }
    }
    if (available) atexit(dual_expert_loader_shutdown);
}

static int dual_expert_load_start(ExpertLoadHandle *handle,
                                  ExpertLoadJob *job) {
    pthread_once(&dual_loader_once, dual_expert_loader_init);
    pthread_mutex_lock(&dual_loader_pool.mutex);
    int selected = -1;
    while (selected < 0) {
        int available = 0;
        for (int i = 0; i < DUAL_EXPERT_LOADER_COUNT; i++) {
            DualExpertLoaderSlot *slot = &dual_loader_pool.slots[i];
            if (!slot->available) continue;
            available++;
            if (!slot->job && !slot->pending) { selected = i; break; }
        }
        if (!available ||
            (selected < 0 && available < DUAL_EXPERT_LOADER_COUNT)) {
            pthread_mutex_unlock(&dual_loader_pool.mutex);
            return -1;
        }
        if (selected < 0)
            pthread_cond_wait(&dual_loader_pool.idle,
                              &dual_loader_pool.mutex);
    }
    DualExpertLoaderSlot *slot = &dual_loader_pool.slots[selected];
    slot->job = job;
    slot->completed = 0;
    slot->pending = 1;
    handle->active = 1;
    handle->loader_slot = selected;
    pthread_cond_broadcast(&dual_loader_pool.ready);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    return 0;
}

static int dual_expert_load_finish(ExpertLoadHandle *handle) {
    if (!handle->active || handle->loader_slot < 0 ||
        handle->loader_slot >= DUAL_EXPERT_LOADER_COUNT) return -1;
    pthread_mutex_lock(&dual_loader_pool.mutex);
    DualExpertLoaderSlot *slot =
        &dual_loader_pool.slots[handle->loader_slot];
    while (!slot->completed)
        pthread_cond_wait(&dual_loader_pool.complete,
                          &dual_loader_pool.mutex);
    slot->job = NULL;
    pthread_cond_broadcast(&dual_loader_pool.idle);
    pthread_mutex_unlock(&dual_loader_pool.mutex);
    handle->active = 0;
    return 0;
}
#endif

#if !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER) && \
    !defined(COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER)
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    pthread_cond_t complete;
    pthread_cond_t idle;
    pthread_t thread;
    ExpertLoadJob *job;
    int pending;
    int completed;
    int stopping;
    int available;
} PersistentExpertLoader;

static PersistentExpertLoader persistent_loader = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .ready = PTHREAD_COND_INITIALIZER,
    .complete = PTHREAD_COND_INITIALIZER,
    .idle = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t persistent_loader_once = PTHREAD_ONCE_INIT;

static void *persistent_expert_loader_worker(void *unused) {
    (void)unused;
    pthread_mutex_lock(&persistent_loader.mutex);
    for (;;) {
        while (!persistent_loader.pending && !persistent_loader.stopping)
            pthread_cond_wait(&persistent_loader.ready,
                              &persistent_loader.mutex);
        if (persistent_loader.stopping) break;
        ExpertLoadJob *job = persistent_loader.job;
        persistent_loader.pending = 0;
        pthread_mutex_unlock(&persistent_loader.mutex);
        job->result = coli_expert_lookup(job->store, job->key, &job->view);
        pthread_mutex_lock(&persistent_loader.mutex);
        persistent_loader.completed = 1;
        pthread_cond_signal(&persistent_loader.complete);
    }
    pthread_mutex_unlock(&persistent_loader.mutex);
    return NULL;
}

static void persistent_expert_loader_shutdown(void) {
    if (!persistent_loader.available) return;
    pthread_mutex_lock(&persistent_loader.mutex);
    persistent_loader.stopping = 1;
    pthread_cond_signal(&persistent_loader.ready);
    pthread_mutex_unlock(&persistent_loader.mutex);
    pthread_join(persistent_loader.thread, NULL);
    persistent_loader.available = 0;
}

static void persistent_expert_loader_init(void) {
    if (!pthread_create(&persistent_loader.thread, NULL,
                        persistent_expert_loader_worker, NULL)) {
        persistent_loader.available = 1;
        atexit(persistent_expert_loader_shutdown);
    }
}
#endif

static int expert_load_start(ExpertLoadHandle *handle, ExpertLoadJob *job) {
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    if (!dual_expert_load_start(handle, job)) return 0;
    if (pthread_create(&handle->thread, NULL, expert_load_worker, job))
        return -1;
    handle->loader_slot = -1;
    handle->active = 1;
    return 0;
#elif defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP)
    job->result = coli_expert_lookup(job->store, job->key, &job->view);
    handle->active = 1;
    return 0;
#elif !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER)
    pthread_once(&persistent_loader_once, persistent_expert_loader_init);
    if (!persistent_loader.available) return -1;
    pthread_mutex_lock(&persistent_loader.mutex);
    while (persistent_loader.job || persistent_loader.pending)
        pthread_cond_wait(&persistent_loader.idle,
                          &persistent_loader.mutex);
    persistent_loader.job = job;
    persistent_loader.completed = 0;
    persistent_loader.pending = 1;
    handle->active = 1;
    pthread_cond_signal(&persistent_loader.ready);
    pthread_mutex_unlock(&persistent_loader.mutex);
    return 0;
#else
    if (pthread_create(&handle->thread, NULL, expert_load_worker, job))
        return -1;
    handle->active = 1;
    return 0;
#endif
}

static int expert_load_finish(ExpertLoadHandle *handle) {
    if (!handle->active) return -1;
#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    if (handle->loader_slot >= 0)
        return dual_expert_load_finish(handle);
    int result = pthread_join(handle->thread, NULL);
    handle->active = 0;
    return result;
#elif defined(COLI_V4_EXPERIMENTAL_SYNC_EXPERT_LOOKUP)
    handle->active = 0;
    return 0;
#elif !defined(COLI_V4_DISABLE_PERSISTENT_EXPERT_LOADER)
    pthread_mutex_lock(&persistent_loader.mutex);
    while (!persistent_loader.completed)
        pthread_cond_wait(&persistent_loader.complete,
                          &persistent_loader.mutex);
    persistent_loader.job = NULL;
    pthread_cond_broadcast(&persistent_loader.idle);
    pthread_mutex_unlock(&persistent_loader.mutex);
    handle->active = 0;
    return 0;
#else
    int result = pthread_join(handle->thread, NULL);
    handle->active = 0;
    return result;
#endif
}

#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
enum {
    COLI_V4_BLOCK_PROFILE_MOE_TOTAL = 0,
    COLI_V4_BLOCK_PROFILE_GATE_DECODE = 1,
    COLI_V4_BLOCK_PROFILE_LOADER_START = 2,
    COLI_V4_BLOCK_PROFILE_LOADER_WAIT = 3,
};
double coli_v4_block_profile_now(void);
void coli_v4_block_profile_add(int kind, double seconds);
#endif

static int profiled_expert_load_start(ExpertLoadHandle *handle,
                                      ExpertLoadJob *job) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double began = coli_v4_block_profile_now();
#endif
    int result = expert_load_start(handle, job);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_LOADER_START,
                              coli_v4_block_profile_now() - began);
#endif
    return result;
}

static int profiled_expert_load_finish(ExpertLoadHandle *handle) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double began = coli_v4_block_profile_now();
#endif
    int result = expert_load_finish(handle);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_LOADER_WAIT,
                              coli_v4_block_profile_now() - began);
#endif
    return result;
}

static int moe_token_pipeline(float *output,
                              const ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              ColiExpertStore *store,
                              const float *input, int token) {
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_moe_began = coli_v4_block_profile_now();
#endif
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
#ifndef COLI_V4_DISABLE_BF16_ROUTE
    float *gate = NULL;
    const uint16_t *raw_gate = value(weights, "ffn.gate.weight", NULL);
    int missing_gate = !raw_gate;
#else
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    int missing_gate = !gate;
#endif
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    int *expert_ids = malloc((size_t)topk * sizeof(*expert_ids));
    float *expert_weights = malloc((size_t)topk * sizeof(*expert_weights));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (missing_gate || !route_weights || !indices || !expert_ids || !expert_weights ||
        !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(expert_weights);
        free(expert_ids); free(indices); free(route_weights); free(gate);
        return -1;
    }
#ifdef COLI_V4_DISABLE_BF16_ROUTE
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    double profile_gate_began = coli_v4_block_profile_now();
#endif
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_GATE_DECODE,
                              coli_v4_block_profile_now() - profile_gate_began);
#endif
#endif
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router && !table) result = -1;
    if (!result && weights->plan.uses_hash_router)
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
#ifndef COLI_V4_DISABLE_BF16_ROUTE
    if (!result) result = coli_v4_route_bf16(
        route_weights, indices, input, raw_gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);
#else
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);
#endif

    int selected = 0;
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        for (int rank = 0; rank < topk; rank++) {
            if (indices[rank] == expert_id) {
                expert_ids[selected] = expert_id;
                expert_weights[selected] = route_weights[rank];
                selected++;
            }
        }
    }
    if (!result && selected != topk) result = -1;

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH
    if (!result && expert_prefetch_enabled() && store->ops->prefetch) {
        ColiExpertKey *keys = malloc((size_t)selected * sizeof(*keys));
        if (keys) {
            for (int i = 0; i < selected; i++)
                keys[i] = (ColiExpertKey){weights->plan.layer, expert_ids[i]};
            store->ops->prefetch(store, keys, (size_t)selected);
            free(keys);
        }
    }
#endif


#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    ExpertLoadJob jobs[DUAL_EXPERT_LOADER_COUNT] = {{0}};
    ExpertLoadHandle loaders[DUAL_EXPERT_LOADER_COUNT] = {{0}};
    int loader_active[DUAL_EXPERT_LOADER_COUNT] = {0};
    if (!result) {
        int preload = selected < DUAL_EXPERT_LOADER_COUNT
            ? selected : DUAL_EXPERT_LOADER_COUNT;
        for (int i = 0; i < preload; i++) {
            jobs[i].store = store;
            jobs[i].key = (ColiExpertKey){weights->plan.layer, expert_ids[i]};
            jobs[i].result = -1;
            if (profiled_expert_load_start(&loaders[i], &jobs[i]) != 0) {
                result = -1;
                break;
            }
            loader_active[i] = 1;
        }
    }
#else
    ExpertLoadJob job = {0};
    ExpertLoadHandle loader = {0};
    int loader_active = 0;
    if (!result) {
        job.store = store;
        job.key = (ColiExpertKey){weights->plan.layer, expert_ids[0]};
        job.result = -1;
        if (profiled_expert_load_start(&loader, &job) != 0)
            result = -1;
        else
            loader_active = 1;
    }
#endif

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));

#ifdef COLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
    for (int current = 0; !result && current < selected; current++) {
        int slot = current % DUAL_EXPERT_LOADER_COUNT;
        if (!loader_active[slot] ||
            profiled_expert_load_finish(&loaders[slot]) != 0) {
            result = -1; break;
        }
        loader_active[slot] = 0;
        if (jobs[slot].result) { result = -1; break; }
        ColiExpertView expert = jobs[slot].view;

        int next = current + DUAL_EXPERT_LOADER_COUNT;
        if (next < selected) {
            memset(&jobs[slot], 0, sizeof(jobs[slot]));
            jobs[slot].store = store;
            jobs[slot].key = (ColiExpertKey){weights->plan.layer,
                                            expert_ids[next]};
            jobs[slot].result = -1;
            if (profiled_expert_load_start(&loaders[slot],
                                           &jobs[slot]) != 0)
                result = -1;
            else
                loader_active[slot] = 1;
        }
        if (!result) result = coli_v4_expert_forward_ref(
            expert_output, &expert, input, expert_weights[current],
            config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    for (int slot = 0; slot < DUAL_EXPERT_LOADER_COUNT; slot++)
        if (loader_active[slot]) {
            profiled_expert_load_finish(&loaders[slot]);
            if (!jobs[slot].result)
                coli_expert_release(store, &jobs[slot].view);
        }
#else
    for (int current = 0; current < selected && loader_active; current++) {
        if (profiled_expert_load_finish(&loader) != 0) {
            result = -1; loader_active = 0; break;
        }
        loader_active = 0;
        if (job.result) { result = -1; break; }
        ColiExpertView expert = job.view;

        if (current + 1 < selected) {
            memset(&job, 0, sizeof(job));
            job.store = store;
            job.key = (ColiExpertKey){weights->plan.layer,
                                     expert_ids[current + 1]};
            job.result = -1;
            if (profiled_expert_load_start(&loader, &job) != 0)
                result = -1;
            else
                loader_active = 1;
        }
        if (!result) result = coli_v4_expert_forward_ref(
            expert_output, &expert, input, expert_weights[current],
            config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (loader_active) {
        profiled_expert_load_finish(&loader);
        if (!job.result) coli_expert_release(store, &job.view);
    }
#endif
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);

    free(shared_output); free(expert_output); free(expert_weights);
    free(expert_ids); free(indices); free(route_weights); free(gate);
#ifdef COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL,
                              coli_v4_block_profile_now() - profile_moe_began);
#endif
    return result;
}

static int block_token_pipeline(float *output_hc,
                                ColiDeepSeekV4WindowAttentionState *attention,
                                const ColiDeepSeekV4LayerWeights *weights,
                                const ColiDeepSeekV4Config *config,
                                ColiExpertStore *experts,
                                const float *input_hc, int token, int position,
                                char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch ||
        !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual); return -1;
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual,
                                          post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);
    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn",
                                            "ffn_norm.weight");
    if (!result) result = moe_token_pipeline(branch, weights, config, experts,
                                             normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual,
                                          post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);
    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_pipeline(output_hc, NULL, weights, config, experts,
                                input_hc, token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_pipeline(output_hc, attention, weights, config, experts,
                                input_hc, token, position, error, error_size);
}
