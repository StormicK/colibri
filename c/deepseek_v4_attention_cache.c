#include "deepseek_v4_attention_cache.h"

#include <stdlib.h>

#include "deepseek_v4_kv_cache.h"
#include "deepseek_v4_sparse_attention.h"

struct ColiDeepSeekV4AttentionCache {
    ColiDeepSeekV4KVCache *kv;
    int window_size;
    int compression_ratio;
    int head_dimension;
    int compressed_capacity;
};

int coli_v4_attention_cache_create(ColiDeepSeekV4AttentionCache **output,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context) {
    if (!output) return -1;
    *output = NULL;
    ColiDeepSeekV4AttentionCache *cache = calloc(1, sizeof(*cache));
    if (!cache) return -1;
    cache->window_size = window_size;
    cache->compression_ratio = compression_ratio;
    cache->head_dimension = head_dimension;
    cache->compressed_capacity = max_context / compression_ratio;
    if (cache->compressed_capacity < 1) cache->compressed_capacity = 1;
    if (coli_v4_kv_cache_create(&cache->kv, window_size, compression_ratio,
                                head_dimension, max_context) != 0) {
        free(cache);
        return -1;
    }
    *output = cache;
    return 0;
}

void coli_v4_attention_cache_reset(ColiDeepSeekV4AttentionCache *cache) {
    if (cache) coli_v4_kv_cache_reset(cache->kv);
}

void coli_v4_attention_cache_destroy(ColiDeepSeekV4AttentionCache *cache) {
    if (!cache) return;
    coli_v4_kv_cache_destroy(cache->kv);
    free(cache);
}

int coli_v4_attention_cache_step(ColiDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale) {
    if (!cache || !output || !query || !window_kv || !sinks || heads < 1 ||
        position < 0 || position / cache->compression_ratio >= cache->compressed_capacity)
        return -1;
    int boundary = (position + 1) % cache->compression_ratio == 0;
    if (boundary != (compressed_kv != NULL)) return -1;
    if (coli_v4_kv_cache_put_window(cache->kv, position, window_kv) < 0)
        return -1;
    if (compressed_kv &&
        coli_v4_kv_cache_put_compressed(cache->kv, position, compressed_kv) < 0)
        return -1;
    size_t capacity = (size_t)cache->window_size + cache->compressed_capacity;
    int *indices = malloc(capacity * sizeof(*indices));
    if (!indices) return -1;
    int topk = coli_v4_kv_cache_indices(cache->kv, position, indices, capacity);
    int result = topk < 0 ? -1 : coli_v4_sparse_attention_ref(
        output, query, coli_v4_kv_cache_values(cache->kv), sinks, indices,
        heads, cache->head_dimension, coli_v4_kv_cache_value_count(cache->kv),
        topk, softmax_scale);
    free(indices);
    return result;
}
