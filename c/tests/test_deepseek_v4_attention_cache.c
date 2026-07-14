#include "../deepseek_v4_attention_cache.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    ColiDeepSeekV4AttentionCache *cache = NULL;
    if (coli_v4_attention_cache_create(&cache, 4, 4, 2, 16) != 0) return 1;
    float query[2] = {1, 0};
    float sink[1] = {0};
    float output[2];
    for (int position = 0; position < 4; position++) {
        float window[2] = {(float)position, 1.0f};
        float compressed[2] = {1.5f, 1.0f};
        if (coli_v4_attention_cache_step(cache, output, query, window,
                                         position == 3 ? compressed : NULL,
                                         sink, 1, position, 1.0f) != 0)
            return 1;
    }
    if (!isfinite(output[0]) || !isfinite(output[1]) || output[0] <= 1.5f)
        return 1;
    if (coli_v4_attention_cache_step(cache, output, query,
                                     (float[2]){4, 1}, (float[2]){2, 1},
                                     sink, 1, 4, 1.0f) == 0)
        return 1;
    coli_v4_attention_cache_reset(cache);
    coli_v4_attention_cache_destroy(cache);
    puts("DeepSeek-V4 attention cache tests: ok");
    return 0;
}
