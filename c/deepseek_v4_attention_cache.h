#ifndef COLIBRI_DEEPSEEK_V4_ATTENTION_CACHE_H
#define COLIBRI_DEEPSEEK_V4_ATTENTION_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiDeepSeekV4AttentionCache ColiDeepSeekV4AttentionCache;

int coli_v4_attention_cache_create(ColiDeepSeekV4AttentionCache **cache,
                                   int window_size, int compression_ratio,
                                   int head_dimension, int max_context);
void coli_v4_attention_cache_reset(ColiDeepSeekV4AttentionCache *cache);
void coli_v4_attention_cache_destroy(ColiDeepSeekV4AttentionCache *cache);

/* query is [heads, head_dimension]. window_kv and compressed_kv have one
 * head_dimension vector each. compressed_kv is required at ratio boundaries. */
int coli_v4_attention_cache_step(ColiDeepSeekV4AttentionCache *cache,
                                 float *output, const float *query,
                                 const float *window_kv,
                                 const float *compressed_kv,
                                 const float *sinks, int heads,
                                 int position, float softmax_scale);

#ifdef __cplusplus
}
#endif

#endif
