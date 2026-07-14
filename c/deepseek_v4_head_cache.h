#ifndef COLIBRI_DEEPSEEK_V4_HEAD_CACHE_H
#define COLIBRI_DEEPSEEK_V4_HEAD_CACHE_H

#include <stddef.h>
#include <stdint.h>

int coli_v4_head_cache_probe(const char *model_dir, uint64_t *bytes,
                             char *error, size_t error_size);
int coli_v4_head_cache_load(const char *model_dir,
                            char *error, size_t error_size);
uint64_t coli_v4_head_cache_bytes(void);
const void *coli_v4_head_cache_data(int shard, uint64_t offset, size_t length);

#endif
