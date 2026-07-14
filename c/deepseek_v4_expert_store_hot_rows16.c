/* Hot target experts are converted in-place to a 16-row resident layout.
 * Cold experts keep the official row-major FP4 representation. */
#define coli_deepseek_v4_expert_store_open \
    coli_deepseek_v4_expert_store_open_base
#include "deepseek_v4_expert_store.c"
#undef coli_deepseek_v4_expert_store_open

#include "native_quant_fp4_rows16.h"

#include "deepseek_v4_runtime.h"

typedef struct V4HotPolicy {
    ColiExpertStore *store;
    int pin_count;
    uint64_t repin_interval;
    uint64_t *usage;
    uint64_t *layer_requests;
    int *pins;
    unsigned char *packed;
    uint64_t packed_slots;
    struct V4HotPolicy *next;
} V4HotPolicy;

static pthread_mutex_t hot_policies_mutex = PTHREAD_MUTEX_INITIALIZER;
static V4HotPolicy *hot_policies;


static V4HotPolicy *hot_find(ColiExpertStore *store) {
    pthread_mutex_lock(&hot_policies_mutex);
    V4HotPolicy *policy = hot_policies;
    while (policy && policy->store != store) policy = policy->next;
    pthread_mutex_unlock(&hot_policies_mutex);
    return policy;
}

#ifndef COLI_V4_PIN_RAMP_REQUESTS
#define COLI_V4_PIN_RAMP_REQUESTS 0
#endif

static int hot_is_pinned(const V4HotPolicy *policy, int layer, int expert) {
    if (!policy || policy->pin_count < 1) return 0;
    int active = policy->pin_count;
#if COLI_V4_PIN_RAMP_REQUESTS > 0
    if (active > 4) {
        uint64_t grown = policy->layer_requests[layer] /
                         COLI_V4_PIN_RAMP_REQUESTS;
        active = grown >= (uint64_t)(active - 4)
            ? active : 4 + (int)grown;
    }
#endif
    const int *pins = policy->pins + (size_t)layer * policy->pin_count;
    for (int i = 0; i < active; i++)
        if (pins[i] == expert) return 1;
    return 0;
}

static size_t hot_slot_index(const V4ExpertStoreState *state,
                             const V4ExpertSlot *slot) {
    return (size_t)(slot - state->slots);
}

static void hot_fill_view(ColiTensorView *view,
                          const V4ExpertRecord *record,
                          const V4ExpertSlot *slot, int matrix,
                          const V4HotPolicy *policy,
                          const V4ExpertStoreState *state) {
    fill_tensor_view(view, record, slot, matrix);
#ifdef __AVX512F__
    if (policy->packed[hot_slot_index(state, slot)]) view->block_rows = 16;
#endif
}

static int hot_pack_matrix(ColiTensorView *view, unsigned char *scratch) {
#ifndef __AVX512F__
    (void)view; (void)scratch; return -1;
#else
    unsigned char *packed_scales = scratch + view->data_bytes;
    if (coli_fp4_pack_rows16_v10(scratch, packed_scales, view)) return -1;
    memcpy((void *)view->data, scratch, view->data_bytes);
    memcpy((void *)view->scales, packed_scales, view->scale_bytes);
    return 0;
#endif
}

/* state->mutex is held and the slot has at least one reference. */
static int hot_pack_slot_locked(V4HotPolicy *policy,
                                V4ExpertStoreState *state,
                                const V4ExpertRecord *record,
                                V4ExpertSlot *slot) {
#ifndef __AVX512F__
    (void)policy; (void)state; (void)record; (void)slot; return -1;
#else
    size_t slot_index = hot_slot_index(state, slot);
    if (policy->packed[slot_index]) return 0;
    ColiTensorView gate, down, up;
    fill_tensor_view(&gate, record, slot, V4_W1);
    fill_tensor_view(&down, record, slot, V4_W2);
    fill_tensor_view(&up, record, slot, V4_W3);
    const ColiTensorView *views[3] = {&gate, &down, &up};
    size_t scratch_size = 0;
    for (int i = 0; i < 3; i++) {
        size_t needed = views[i]->data_bytes + views[i]->scale_bytes;
        if (needed > scratch_size) scratch_size = needed;
    }
    unsigned char *scratch = malloc(scratch_size);
    if (!scratch) return -1;
    int result = hot_pack_matrix(&gate, scratch) ||
                 hot_pack_matrix(&down, scratch) ||
                 hot_pack_matrix(&up, scratch);
    free(scratch);
    if (!result) {
        policy->packed[slot_index] = 1;
        policy->packed_slots++;
    }
    return result;
#endif
}

static void hot_repin_locked(V4HotPolicy *policy, V4ExpertStoreState *state,
                             int layer) {
    if (!policy || policy->pin_count < 1) return;
    uint64_t *usage = policy->usage +
        (size_t)layer * state->experts_per_layer;
    int *pins = policy->pins + (size_t)layer * policy->pin_count;
    for (int rank = 0; rank < policy->pin_count; rank++) {
        int best = -1;
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            int already = 0;
            for (int prior = 0; prior < rank; prior++)
                if (pins[prior] == expert) { already = 1; break; }
            if (!already && usage[expert] &&
                (best < 0 || usage[expert] > usage[best] ||
                 (usage[expert] == usage[best] && expert < best)))
                best = expert;
        }
        pins[rank] = best;
    }
    V4ExpertSlot *slots = layer_slots(state, layer);
    for (int i = 0; i < state->slots_per_layer; i++)
        if (slots[i].slab && slots[i].expert >= 0 &&
            hot_is_pinned(policy, layer, slots[i].expert))
            slots[i].used = ++state->clock;
    uint64_t decay_interval = policy->repin_interval * 64;
    if (decay_interval &&
        policy->layer_requests[layer] % decay_interval == 0)
        for (int expert = 0; expert < state->experts_per_layer; expert++)
            usage[expert] = (usage[expert] + 1) / 2;
}

static int lookup_hot(ColiExpertStore *store, ColiExpertKey key,
                      ColiExpertView *view) {
    if (!store || !store->state || !view) return -1;
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    V4HotPolicy *policy = hot_find(store);
    if (!record || !policy) return -1;
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    policy->usage[(size_t)key.layer * state->experts_per_layer + key.expert]++;
    uint64_t layer_requests = ++policy->layer_requests[key.layer];
    if (policy->repin_interval &&
        layer_requests % policy->repin_interval == 0)
        hot_repin_locked(policy, state, key.layer);

    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i]; slot->references++;
            slot->used = ++state->clock; state->stats.hits++;
            if (hot_is_pinned(policy, key.layer, key.expert))
                hot_pack_slot_locked(policy, state, record, slot);
            memset(view, 0, sizeof(*view)); view->key = key;
            hot_fill_view(&view->gate, record, slot, V4_W1, policy, state);
            hot_fill_view(&view->down, record, slot, V4_W2, policy, state);
            hot_fill_view(&view->up, record, slot, V4_W3, policy, state);
            view->lease = slot;
            pthread_mutex_unlock(&state->mutex); return 0;
        }
    }
    for (int i = 0; i < state->slots_per_layer; i++)
        if (!slots[i].references && !slots[i].slab) { slot = &slots[i]; break; }
    if (!slot)
        for (int i = 0; i < state->slots_per_layer; i++)
            if (!slots[i].references &&
                !hot_is_pinned(policy, key.layer, slots[i].expert) &&
                (!slot || slots[i].used < slot->used)) slot = &slots[i];
    if (!slot)
        for (int i = 0; i < state->slots_per_layer; i++)
            if (!slots[i].references && (!slot || slots[i].used < slot->used))
                slot = &slots[i];
    if (!slot) { pthread_mutex_unlock(&state->mutex); return -1; }
    if (!slot->slab) {
        slot->slab = malloc((size_t)state->record_bytes);
        if (!slot->slab) { pthread_mutex_unlock(&state->mutex); return -1; }
        state->stats.resident_bytes += state->record_bytes;
    }
    policy->packed[hot_slot_index(state, slot)] = 0;
    slot->expert = -1; slot->references = 1; slot->used = ++state->clock;
    pthread_mutex_unlock(&state->mutex);
    int read_result = coli_st_read_at(
        state->index, record->shard, record->scale_offset,
        (size_t)record->scale_bytes, slot->slab);
    if (!read_result) read_result = coli_st_read_at(
        state->index, record->shard, record->weight_offset,
        (size_t)record->weight_bytes, slot->slab + record->scale_bytes);
    pthread_mutex_lock(&state->mutex);
    if (read_result) {
        slot->references = 0; slot->expert = -1;
        pthread_mutex_unlock(&state->mutex); return -1;
    }
    slot->expert = key.expert; slot->used = ++state->clock;
    state->stats.misses++; state->stats.bytes_read += record->record_bytes;
    if (hot_is_pinned(policy, key.layer, key.expert))
        hot_pack_slot_locked(policy, state, record, slot);
    memset(view, 0, sizeof(*view)); view->key = key;
    hot_fill_view(&view->gate, record, slot, V4_W1, policy, state);
    hot_fill_view(&view->down, record, slot, V4_W2, policy, state);
    hot_fill_view(&view->up, record, slot, V4_W3, policy, state);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex); return 0;
}

static void destroy_hot(ColiExpertStore *store) {
    pthread_mutex_lock(&hot_policies_mutex);
    V4HotPolicy **link = &hot_policies;
    while (*link && (*link)->store != store) link = &(*link)->next;
    V4HotPolicy *policy = *link;
    if (policy) *link = policy->next;
    pthread_mutex_unlock(&hot_policies_mutex);
    if (policy) {
        fprintf(stderr, "v4_rows16 packed_slots=%llu\n",
                (unsigned long long)policy->packed_slots);
        free(policy->packed); free(policy->pins);
        free(policy->layer_requests); free(policy->usage); free(policy);
    }
    destroy(store);
}

#ifndef COLI_V4_ROWS16_STORE_OPEN
#define COLI_V4_ROWS16_STORE_OPEN coli_deepseek_v4_expert_store_open
#endif

int COLI_V4_ROWS16_STORE_OPEN(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps hot_operations = {
        lookup_hot, release, prefetch, stats, destroy_hot
    };
    int result = coli_deepseek_v4_expert_store_open_base(
        options, output, error, error_size);
    if (result) return result;
    V4ExpertStoreState *state = (*output)->state;
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    int maximum_pins = state->slots_per_layer - minimum_slots;
#ifndef COLI_V4_MAX_PIN_SLOTS_PER_LAYER
#define COLI_V4_MAX_PIN_SLOTS_PER_LAYER 4
#endif
    if (maximum_pins > COLI_V4_MAX_PIN_SLOTS_PER_LAYER)
        maximum_pins = COLI_V4_MAX_PIN_SLOTS_PER_LAYER;
    ColiDeepSeekV4RuntimeOptions *runtime = coli_v4_runtime_options();
    uint64_t requested = runtime->pin_slots_per_layer >= 0
        ? (uint64_t)runtime->pin_slots_per_layer
        : (uint64_t)(maximum_pins > 0 ? maximum_pins : 0);
    int pin_count = requested > (uint64_t)maximum_pins
        ? maximum_pins : (int)requested;
    if (pin_count < 0) pin_count = 0;
    V4HotPolicy *policy = calloc(1, sizeof(*policy));
    size_t records = (size_t)state->layers * state->experts_per_layer;
    size_t pins = (size_t)state->layers * (pin_count ? pin_count : 1);
    size_t slots = (size_t)state->layers * state->slots_per_layer;
    if (policy) policy->usage = calloc(records, sizeof(*policy->usage));
    if (policy) policy->layer_requests = calloc(
        (size_t)state->layers, sizeof(*policy->layer_requests));
    if (policy) policy->pins = malloc(pins * sizeof(*policy->pins));
    if (policy) policy->packed = calloc(slots, sizeof(*policy->packed));
    if (!policy || !policy->usage || !policy->layer_requests ||
        !policy->pins || !policy->packed) {
        free(policy ? policy->packed : NULL); free(policy ? policy->pins : NULL);
        free(policy ? policy->layer_requests : NULL);
        free(policy ? policy->usage : NULL); free(policy);
        destroy(*output); *output = NULL;
        return set_error(error, error_size, "out of memory creating hot policy");
    }
    for (size_t i = 0; i < pins; i++) policy->pins[i] = -1;
    policy->store = *output; policy->pin_count = pin_count;
    policy->repin_interval = runtime->repin_interval
        ? runtime->repin_interval : (uint64_t)minimum_slots;
    if (!policy->repin_interval) policy->repin_interval = 1;
    pthread_mutex_lock(&hot_policies_mutex);
    policy->next = hot_policies; hot_policies = policy;
    pthread_mutex_unlock(&hot_policies_mutex);
    (*output)->ops = &hot_operations;
    fprintf(stderr,
            "v4_hot_policy pin_slots_per_layer=%d repin_interval=%llu "
            "mode=resident-ram rows16=hot-pins\n", pin_count,
            (unsigned long long)policy->repin_interval);
    return 0;
}
