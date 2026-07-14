#include "deepseek_v4_head_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors_index.h"

int __real_coli_st_read_at(const ColiSafetensorsIndex *, int, uint64_t,
                           size_t, void *);

typedef struct {
    unsigned char *data;
    uint64_t bytes;
    uint64_t offset;
    int shard;
    int cleanup_registered;
} HeadCache;

static HeadCache head_cache;

static void release_head_cache(void) {
    free(head_cache.data);
    head_cache.data = NULL;
    head_cache.bytes = 0;
    head_cache.offset = 0;
    head_cache.shard = -1;
}

static int find_head(const char *model_dir, ColiSafetensorsIndex **index,
                     const ColiSafetensorsTensor **head,
                     char *error, size_t error_size) {
    *index = NULL; *head = NULL;
    if (coli_st_index_open(index, model_dir, error, error_size)) return -1;
    *head = coli_st_find(*index, "head.weight");
    if (!*head || (*head)->dtype != COLI_ST_BF16 || (*head)->rank != 2) {
        snprintf(error, error_size, "missing or invalid BF16 head.weight");
        coli_st_index_close(*index); *index = NULL; return -1;
    }
    return 0;
}

int coli_v4_head_cache_probe(const char *model_dir, uint64_t *bytes,
                             char *error, size_t error_size) {
    ColiSafetensorsIndex *index;
    const ColiSafetensorsTensor *head;
    if (!bytes || find_head(model_dir, &index, &head, error, error_size)) return -1;
    *bytes = head->nbytes;
    coli_st_index_close(index); return 0;
}

int coli_v4_head_cache_load(const char *model_dir,
                            char *error, size_t error_size) {
    ColiSafetensorsIndex *index;
    const ColiSafetensorsTensor *head;
    if (find_head(model_dir, &index, &head, error, error_size)) return -1;
    unsigned char *data = malloc((size_t)head->nbytes);
    if (!data || coli_st_read_at(index, head->shard, head->offset,
                                 (size_t)head->nbytes, data)) {
        free(data); coli_st_index_close(index);
        snprintf(error, error_size, "cannot load resident BF16 head.weight");
        return -1;
    }
    release_head_cache();
    head_cache.data = data;
    head_cache.bytes = head->nbytes;
    head_cache.offset = head->offset;
    head_cache.shard = head->shard;
    if (!head_cache.cleanup_registered) {
        atexit(release_head_cache);
        head_cache.cleanup_registered = 1;
    }
    coli_st_index_close(index); return 0;
}

uint64_t coli_v4_head_cache_bytes(void) { return head_cache.bytes; }

const void *coli_v4_head_cache_data(int shard, uint64_t offset, size_t length) {
    if (!head_cache.data || shard != head_cache.shard ||
        offset < head_cache.offset ||
        offset - head_cache.offset > head_cache.bytes ||
        length > head_cache.bytes - (offset - head_cache.offset)) return NULL;
    return head_cache.data + (size_t)(offset - head_cache.offset);
}

int __wrap_coli_st_read_at(const ColiSafetensorsIndex *index, int shard,
                           uint64_t offset, size_t length, void *destination) {
    if (head_cache.data && destination && shard == head_cache.shard &&
        offset >= head_cache.offset &&
        offset - head_cache.offset <= head_cache.bytes &&
        length <= head_cache.bytes - (offset - head_cache.offset)) {
        memcpy(destination, head_cache.data + (size_t)(offset - head_cache.offset),
               length);
        return 0;
    }
    return __real_coli_st_read_at(index, shard, offset, length, destination);
}
