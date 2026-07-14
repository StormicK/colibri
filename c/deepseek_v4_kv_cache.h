#ifndef COLIBRI_DEEPSEEK_V4_KV_CACHE_H
#define COLIBRI_DEEPSEEK_V4_KV_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4KVCache ColiDeepSeekV4KVCache;

int coli_v4_kv_cache_create(ColiDeepSeekV4KVCache **cache,
                            int window_size, int compression_ratio,
                            int head_dimension, int max_context);
void coli_v4_kv_cache_reset(ColiDeepSeekV4KVCache *cache);
void coli_v4_kv_cache_destroy(ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_put_window(ColiDeepSeekV4KVCache *cache,
                                int position, const float *kv);
int coli_v4_kv_cache_put_compressed(ColiDeepSeekV4KVCache *cache,
                                    int position, const float *kv);
int coli_v4_kv_cache_indices(const ColiDeepSeekV4KVCache *cache,
                             int position, int *indices, size_t capacity);
const float *coli_v4_kv_cache_values(const ColiDeepSeekV4KVCache *cache);
int coli_v4_kv_cache_value_count(const ColiDeepSeekV4KVCache *cache);

#ifdef __cplusplus
}
#endif

#endif
