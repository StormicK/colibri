#include "../deepseek_v4_kv_cache.h"

#include <stdio.h>

int main(void) {
    ColiDeepSeekV4KVCache *cache = NULL;
    if (coli_v4_kv_cache_create(&cache, 4, 4, 2, 16) != 0) return 1;
    float value[2];
    for (int position = 0; position < 6; position++) {
        value[0] = (float)position;
        value[1] = (float)-position;
        if (coli_v4_kv_cache_put_window(cache, position, value) < 0) return 1;
        if ((position + 1) % 4 == 0 &&
            coli_v4_kv_cache_put_compressed(cache, position, value) < 4)
            return 1;
    }
    int indices[8];
    int count = coli_v4_kv_cache_indices(cache, 1, indices, 8);
    if (count != 4 || indices[0] != 0 || indices[1] != 1 ||
        indices[2] != -1 || indices[3] != -1) return 1;
    count = coli_v4_kv_cache_indices(cache, 3, indices, 8);
    if (count != 5 || indices[0] != 0 || indices[3] != 3 || indices[4] != 4)
        return 1;
    count = coli_v4_kv_cache_indices(cache, 5, indices, 8);
    if (count != 5 || indices[0] != 2 || indices[1] != 3 ||
        indices[2] != 0 || indices[3] != 1 || indices[4] != 4)
        return 1;
    const float *values = coli_v4_kv_cache_values(cache);
    if (values[0] != 4.0f || values[2] != 5.0f || values[8] != 3.0f)
        return 1;
    coli_v4_kv_cache_reset(cache);
    if (coli_v4_kv_cache_values(cache)[0] != 0.0f) return 1;
    coli_v4_kv_cache_destroy(cache);
    puts("DeepSeek-V4 KV cache tests: ok");
    return 0;
}
