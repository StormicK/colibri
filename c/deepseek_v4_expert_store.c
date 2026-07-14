#define _GNU_SOURCE
#include "deepseek_v4_expert_store.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors_index.h"

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
int coli_st_prefetch_many(
    const ColiSafetensorsIndex *index, const int *shards,
    const uint64_t *offsets, const size_t *lengths, size_t count);
#endif

enum { V4_W1 = 0, V4_W2 = 1, V4_W3 = 2, V4_MATRIX_COUNT = 3 };

typedef struct {
    const ColiSafetensorsTensor *weight[V4_MATRIX_COUNT];
    const ColiSafetensorsTensor *scale[V4_MATRIX_COUNT];
    int shard;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t record_bytes;
} V4ExpertRecord;

typedef struct {
    int expert;
    unsigned references;
    uint64_t used;
    unsigned char *slab;
} V4ExpertSlot;

typedef struct {
    ColiSafetensorsIndex *index;
    int layers;
    int experts_per_layer;
    int slots_per_layer;
    uint64_t record_bytes;
    V4ExpertRecord *records;
    V4ExpertSlot *slots;
    uint64_t clock;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
} V4ExpertStoreState;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static int compare_tensors(const void *left, const void *right) {
    const ColiSafetensorsTensor *const *a = left;
    const ColiSafetensorsTensor *const *b = right;
    return ((*a)->offset > (*b)->offset) - ((*a)->offset < (*b)->offset);
}

static int contiguous_group(const ColiSafetensorsTensor *const input[3],
                            int *shard, uint64_t *offset, uint64_t *bytes) {
    const ColiSafetensorsTensor *parts[3] = {input[0], input[1], input[2]};
    qsort(parts, 3, sizeof(parts[0]), compare_tensors);
    if (parts[0]->shard != parts[1]->shard || parts[1]->shard != parts[2]->shard ||
        parts[0]->offset + parts[0]->nbytes != parts[1]->offset ||
        parts[1]->offset + parts[1]->nbytes != parts[2]->offset)
        return -1;
    *shard = parts[0]->shard;
    *offset = parts[0]->offset;
    *bytes = parts[2]->offset + parts[2]->nbytes - parts[0]->offset;
    return 0;
}

static int validate_matrix(const ColiSafetensorsTensor *weight,
                           const ColiSafetensorsTensor *scale) {
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        weight->shape[0] != scale->shape[0] || weight->shape[1] <= 0 ||
        scale->shape[1] <= 0)
        return -1;
    int64_t logical_columns = weight->shape[1] * 2;
    return scale->shape[1] * 32 == logical_columns ? 0 : -1;
}

static int build_record(V4ExpertStoreState *state, int layer, int expert,
                        V4ExpertRecord *record, char *error, size_t error_size) {
    static const char *matrix_names[V4_MATRIX_COUNT] = {"w1", "w2", "w3"};
    char name[160];
    memset(record, 0, sizeof(*record));
    for (int matrix = 0; matrix < V4_MATRIX_COUNT; matrix++) {
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.weight",
                 layer, expert, matrix_names[matrix]);
        record->weight[matrix] = coli_st_find(state->index, name);
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.scale",
                 layer, expert, matrix_names[matrix]);
        record->scale[matrix] = coli_st_find(state->index, name);
        if (validate_matrix(record->weight[matrix], record->scale[matrix]) != 0)
            return set_error(error, error_size,
                             "invalid native FP4 expert matrix: layer=%d expert=%d %s",
                             layer, expert, matrix_names[matrix]);
    }
    int scale_shard = -1, weight_shard = -1;
    if (contiguous_group(record->scale, &scale_shard, &record->scale_offset,
                         &record->scale_bytes) != 0 ||
        contiguous_group(record->weight, &weight_shard, &record->weight_offset,
                         &record->weight_bytes) != 0 || scale_shard != weight_shard)
        return set_error(error, error_size,
                         "expert is not two contiguous ranges: layer=%d expert=%d",
                         layer, expert);
    record->shard = scale_shard;
    record->record_bytes = record->scale_bytes + record->weight_bytes;
    return 0;
}

static V4ExpertRecord *get_record(V4ExpertStoreState *state, ColiExpertKey key) {
    if (key.layer < 0 || key.layer >= state->layers || key.expert < 0 ||
        key.expert >= state->experts_per_layer)
        return NULL;
    return &state->records[(size_t)key.layer * state->experts_per_layer + key.expert];
}

static V4ExpertSlot *layer_slots(V4ExpertStoreState *state, int layer) {
    return state->slots + (size_t)layer * state->slots_per_layer;
}

static void fill_tensor_view(ColiTensorView *view,
                             const V4ExpertRecord *record,
                             const V4ExpertSlot *slot, int matrix) {
    const ColiSafetensorsTensor *weight = record->weight[matrix];
    const ColiSafetensorsTensor *scale = record->scale[matrix];
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = slot->slab + record->scale_bytes +
                 (weight->offset - record->weight_offset);
    view->scales = slot->slab + (scale->offset - record->scale_offset);
    view->data_bytes = (size_t)weight->nbytes;
    view->scale_bytes = (size_t)scale->nbytes;
    view->rows = weight->shape[0];
    view->columns = weight->shape[1] * 2;
    view->block_rows = 1;
    view->block_columns = 32;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    if (!store || !store->state || !view) return -1;
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    if (!record) return -1;
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i];
            state->stats.hits++;
            break;
        }
    }
    if (!slot) {
        for (int i = 0; i < state->slots_per_layer; i++) {
            if (!slots[i].references && (!slot || !slots[i].slab ||
                                         (slot->slab && slots[i].used < slot->used)))
                slot = &slots[i];
        }
        if (!slot) {
            pthread_mutex_unlock(&state->mutex);
            return -1;
        }
        if (!slot->slab) {
            slot->slab = malloc((size_t)state->record_bytes);
            if (!slot->slab) {
                pthread_mutex_unlock(&state->mutex);
                return -1;
            }
            state->stats.resident_bytes += state->record_bytes;
        }
        /* A short read must never expose a partially overwritten old slot. */
        slot->expert = -1;
        if (coli_st_read_at(state->index, record->shard, record->scale_offset,
                            (size_t)record->scale_bytes, slot->slab) != 0 ||
            coli_st_read_at(state->index, record->shard, record->weight_offset,
                            (size_t)record->weight_bytes,
                            slot->slab + record->scale_bytes) != 0) {
            pthread_mutex_unlock(&state->mutex);
            return -1;
        }
        slot->expert = key.expert;
        state->stats.misses++;
        state->stats.bytes_read += record->record_bytes;
    }
    slot->references++;
    slot->used = ++state->clock;
    memset(view, 0, sizeof(*view));
    view->key = key;
    fill_tensor_view(&view->gate, record, slot, V4_W1);
    fill_tensor_view(&view->down, record, slot, V4_W2);
    fill_tensor_view(&view->up, record, slot, V4_W3);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

static void release(ColiExpertStore *store, ColiExpertView *view) {
    if (!store || !store->state || !view || !view->lease) return;
    V4ExpertStoreState *state = store->state;
    V4ExpertSlot *slot = view->lease;
    pthread_mutex_lock(&state->mutex);
    if (slot->references) slot->references--;
    view->lease = NULL;
    pthread_mutex_unlock(&state->mutex);
}

static int prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count) {
    if (!store || !store->state || (!keys && count)) return 0;
    V4ExpertStoreState *state = store->state;
    int accepted = 0;
#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
    size_t capacity = count * 2, ranges = 0;
    int *shards = malloc(capacity * sizeof(*shards));
    uint64_t *offsets = malloc(capacity * sizeof(*offsets));
    size_t *lengths = malloc(capacity * sizeof(*lengths));
    int candidates = 0;
    if ((!shards || !offsets || !lengths) && capacity) {
        free(lengths); free(offsets); free(shards); return 0;
    }
    pthread_mutex_lock(&state->mutex);
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        int resident = 0;
        V4ExpertSlot *slots = layer_slots(state, keys[i].layer);
        for (int slot = 0; slot < state->slots_per_layer; slot++)
            if (slots[slot].slab && slots[slot].expert == keys[i].expert) {
                resident = 1; break;
            }
        if (resident) continue;
        shards[ranges] = record->shard;
        offsets[ranges] = record->scale_offset;
        lengths[ranges++] = (size_t)record->scale_bytes;
        shards[ranges] = record->shard;
        offsets[ranges] = record->weight_offset;
        lengths[ranges++] = (size_t)record->weight_bytes;
        candidates++;
    }
    pthread_mutex_unlock(&state->mutex);
    if (candidates && !coli_st_prefetch_many(
            state->index, shards, offsets, lengths, ranges))
        accepted = candidates;
    free(lengths); free(offsets); free(shards);
#else
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        if (coli_st_prefetch_at(state->index, record->shard, record->scale_offset,
                                (size_t)record->scale_bytes) == 0 &&
            coli_st_prefetch_at(state->index, record->shard, record->weight_offset,
                                (size_t)record->weight_bytes) == 0)
            accepted++;
    }
#endif
    pthread_mutex_lock(&state->mutex);
    state->stats.prefetched += (uint64_t)accepted;
    pthread_mutex_unlock(&state->mutex);
    return accepted;
}

static void stats(const ColiExpertStore *store, ColiExpertStoreStats *output) {
    if (!store || !store->state || !output) return;
    V4ExpertStoreState *state = store->state;
    pthread_mutex_lock(&state->mutex);
    *output = state->stats;
    pthread_mutex_unlock(&state->mutex);
}

static void destroy(ColiExpertStore *store) {
    if (!store) return;
    V4ExpertStoreState *state = store->state;
    if (state) {
        for (int i = 0; i < state->layers * state->slots_per_layer; i++)
            free(state->slots[i].slab);
        pthread_mutex_destroy(&state->mutex);
        coli_st_index_close(state->index);
        free(state->records);
        free(state->slots);
        free(state);
    }
    free(store);
}

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps operations = {
        lookup, release, prefetch, stats, destroy
    };
    if (!options || !output || !options->model_dir || options->layers < 1 ||
        options->experts_per_layer < 1 || !options->cache_bytes)
        return set_error(error, error_size, "invalid DeepSeek-V4 ExpertStore options");
    *output = NULL;
    ColiExpertStore *store = calloc(1, sizeof(*store));
    V4ExpertStoreState *state = calloc(1, sizeof(*state));
    if (!store || !state) {
        free(store);
        free(state);
        return set_error(error, error_size, "out of memory creating ExpertStore");
    }
    pthread_mutex_init(&state->mutex, NULL);
    state->layers = options->layers;
    state->experts_per_layer = options->experts_per_layer;
    if (coli_st_index_open(&state->index, options->model_dir, error, error_size) != 0)
        goto fail;
    size_t record_count = (size_t)state->layers * state->experts_per_layer;
    state->records = calloc(record_count, sizeof(*state->records));
    if (!state->records) {
        set_error(error, error_size, "out of memory creating expert manifest");
        goto fail;
    }
    for (int layer = 0; layer < state->layers; layer++) {
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            V4ExpertRecord *record = &state->records[
                (size_t)layer * state->experts_per_layer + expert];
            if (build_record(state, layer, expert, record, error, error_size) != 0)
                goto fail;
            if (!state->record_bytes) state->record_bytes = record->record_bytes;
            if (record->record_bytes != state->record_bytes) {
                set_error(error, error_size, "non-uniform expert size at layer=%d expert=%d",
                          layer, expert);
                goto fail;
            }
        }
    }
    state->slots_per_layer = (int)(options->cache_bytes /
        ((uint64_t)state->layers * state->record_bytes));
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    if (state->slots_per_layer < minimum_slots) {
        set_error(error, error_size,
                  "cache budget cannot hold %d active experts per layer "
                  "(need %llu bytes)", minimum_slots,
                  (unsigned long long)((uint64_t)state->layers * minimum_slots *
                                       state->record_bytes));
        goto fail;
    }
    if (state->slots_per_layer > state->experts_per_layer)
        state->slots_per_layer = state->experts_per_layer;
    state->slots = calloc((size_t)state->layers * state->slots_per_layer,
                          sizeof(*state->slots));
    if (!state->slots) {
        set_error(error, error_size, "out of memory creating expert cache slots");
        goto fail;
    }
    for (int i = 0; i < state->layers * state->slots_per_layer; i++)
        state->slots[i].expert = -1;
    state->stats.capacity_bytes = (uint64_t)state->layers *
                                  state->slots_per_layer * state->record_bytes;
    store->ops = &operations;
    store->state = state;
    *output = store;
    return 0;

fail:
    if (state->slots) free(state->slots);
    free(state->records);
    coli_st_index_close(state->index);
    pthread_mutex_destroy(&state->mutex);
    free(state);
    free(store);
    return -1;
}
