#include "deepseek_v4_kv_cache.h"

#include <stdlib.h>
#include <string.h>

struct ColiDeepSeekV4KVCache {
    int window_size;
    int compression_ratio;
    int head_dimension;
    int compressed_capacity;
    float *values;
};

int coli_v4_kv_cache_create(ColiDeepSeekV4KVCache **output,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context) {
    if (!output || window_size < 1 || compression_ratio < 1 ||
        head_dimension < 1 || max_context < 1)
        return -1;
    *output = NULL;
    ColiDeepSeekV4KVCache *cache = calloc(1, sizeof(*cache));
    if (!cache) return -1;
    cache->window_size = window_size;
    cache->compression_ratio = compression_ratio;
    cache->head_dimension = head_dimension;
    cache->compressed_capacity = max_context / compression_ratio;
    if (cache->compressed_capacity < 1) cache->compressed_capacity = 1;
    size_t count = (size_t)(window_size + cache->compressed_capacity) * head_dimension;
    cache->values = calloc(count, sizeof(*cache->values));
    if (!cache->values) {
        free(cache);
        return -1;
    }
    *output = cache;
    return 0;
}

void coli_v4_kv_cache_reset(ColiDeepSeekV4KVCache *cache) {
    if (!cache) return;
    size_t count = (size_t)(cache->window_size + cache->compressed_capacity) *
                   cache->head_dimension;
    memset(cache->values, 0, count * sizeof(*cache->values));
}

void coli_v4_kv_cache_destroy(ColiDeepSeekV4KVCache *cache) {
    if (!cache) return;
    free(cache->values);
    free(cache);
}

int coli_v4_kv_cache_put_window(ColiDeepSeekV4KVCache *cache,
                                int position, const float *kv) {
    if (!cache || !kv || position < 0) return -1;
    int slot = position % cache->window_size;
    memcpy(cache->values + (size_t)slot * cache->head_dimension, kv,
           (size_t)cache->head_dimension * sizeof(*kv));
    return slot;
}

int coli_v4_kv_cache_put_compressed(ColiDeepSeekV4KVCache *cache,
                                    int position, const float *kv) {
    if (!cache || !kv || position < 0 ||
        (position + 1) % cache->compression_ratio != 0)
        return -1;
    int slot = (position + 1) / cache->compression_ratio - 1;
    if (slot < 0 || slot >= cache->compressed_capacity) return -1;
    int combined = cache->window_size + slot;
    memcpy(cache->values + (size_t)combined * cache->head_dimension, kv,
           (size_t)cache->head_dimension * sizeof(*kv));
    return combined;
}

int coli_v4_kv_cache_indices(const ColiDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity) {
    if (!cache || !indices || position < 0) return -1;
    int compressed = (position + 1) / cache->compression_ratio;
    if (compressed > cache->compressed_capacity) return -1;
    size_t required = (size_t)cache->window_size + compressed;
    if (capacity < required) return -1;
    if (position < cache->window_size - 1) {
        for (int i = 0; i < cache->window_size; i++)
            indices[i] = i <= position ? i : -1;
    } else {
        int oldest = (position + 1) % cache->window_size;
        for (int i = 0; i < cache->window_size; i++)
            indices[i] = (oldest + i) % cache->window_size;
    }
    for (int i = 0; i < compressed; i++)
        indices[cache->window_size + i] = cache->window_size + i;
    return (int)required;
}

const float *coli_v4_kv_cache_values(const ColiDeepSeekV4KVCache *cache) {
    return cache ? cache->values : NULL;
}

int coli_v4_kv_cache_value_count(const ColiDeepSeekV4KVCache *cache) {
    return cache ? cache->window_size + cache->compressed_capacity : 0;
}
