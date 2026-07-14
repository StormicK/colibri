#define _GNU_SOURCE
#include "../deepseek_v4_expert_store.h"
#include "../compat.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = data;
    while (length) {
        ssize_t count = write(fd, bytes, length);
        if (count <= 0) return -1;
        bytes += count;
        length -= (size_t)count;
    }
    return 0;
}

static int write_fixture(const char *path) {
    static const char header[] =
        "{"
        "\"layers.0.ffn.experts.0.w1.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,1],\"data_offsets\":[0,1]},"
        "\"layers.0.ffn.experts.0.w2.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,1],\"data_offsets\":[1,2]},"
        "\"layers.0.ffn.experts.0.w3.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[1,1],\"data_offsets\":[2,3]},"
        "\"resident\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[3,7]},"
        "\"layers.0.ffn.experts.0.w1.weight\":{\"dtype\":\"I8\",\"shape\":[1,16],\"data_offsets\":[7,23]},"
        "\"layers.0.ffn.experts.0.w2.weight\":{\"dtype\":\"I8\",\"shape\":[1,16],\"data_offsets\":[23,39]},"
        "\"layers.0.ffn.experts.0.w3.weight\":{\"dtype\":\"I8\",\"shape\":[1,16],\"data_offsets\":[39,55]}"
        "}";
    unsigned char payload[55];
    for (int i = 0; i < 55; i++) payload[i] = (unsigned char)i;
    uint64_t header_length = sizeof(header) - 1;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | COMPAT_O_BINARY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, sizeof(payload));
    close(fd);
    return result ? -1 : 0;
}

int main(void) {
    char directory[] = "/tmp/colibri-v4-store-XXXXXX";
    char path[256], error[256];
    if (!mkdtemp(directory)) { perror("mkdtemp"); return 1; }
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    if (write_fixture(path) != 0) { perror("write_fixture"); return 1; }

    ColiDeepSeekV4ExpertStoreOptions options = {
        directory, 1, 1, 51
    };
    ColiExpertStore *store = NULL;
    if (coli_deepseek_v4_expert_store_open(&options, &store,
                                            error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    ColiExpertKey key = {0, 0};
    ColiExpertView view;
    if (store->ops->prefetch(store, &key, 1) != 1) {
        fprintf(stderr, "prefetch failed\n"); return 1;
    }
    if (coli_expert_lookup(store, key, &view) != 0) {
        fprintf(stderr, "lookup failed\n"); return 1;
    }
    if (view.gate.format != COLI_TENSOR_FP4_NATIVE_BLOCK ||
        view.gate.rows != 1 || view.gate.columns != 32 ||
        view.gate.data_bytes != 16 || view.gate.scale_bytes != 1 ||
        ((const unsigned char *)view.gate.scales)[0] != 0 ||
        ((const unsigned char *)view.gate.data)[0] != 7 ||
        ((const unsigned char *)view.down.data)[0] != 23 ||
        ((const unsigned char *)view.up.data)[0] != 39)
        { fprintf(stderr, "expert view mismatch: format=%d rows=%lld columns=%lld data=%zu scales=%zu bytes=%u/%u/%u/%u\n",
                  (int)view.gate.format, (long long)view.gate.rows,
                  (long long)view.gate.columns, view.gate.data_bytes,
                  view.gate.scale_bytes,
                  ((const unsigned char *)view.gate.scales)[0],
                  ((const unsigned char *)view.gate.data)[0],
                  ((const unsigned char *)view.down.data)[0],
                  ((const unsigned char *)view.up.data)[0]); return 1; }
    coli_expert_release(store, &view);
    if (coli_expert_lookup(store, key, &view) != 0) return 1;
    coli_expert_release(store, &view);
    ColiExpertStoreStats stats;
    store->ops->stats(store, &stats);
    if (stats.requests != 2 || stats.hits != 1 || stats.misses != 1 ||
        stats.prefetched != 1 || stats.bytes_read != 51 ||
        stats.resident_bytes != 51 || stats.capacity_bytes != 51)
        { fprintf(stderr, "expert stats mismatch: requests=%llu hits=%llu misses=%llu prefetched=%llu bytes=%llu resident=%llu capacity=%llu\n",
                  (unsigned long long)stats.requests,
                  (unsigned long long)stats.hits,
                  (unsigned long long)stats.misses,
                  (unsigned long long)stats.prefetched,
                  (unsigned long long)stats.bytes_read,
                  (unsigned long long)stats.resident_bytes,
                  (unsigned long long)stats.capacity_bytes); return 1; }
    store->ops->destroy(store);
    unlink(path);
    rmdir(directory);
    puts("DeepSeek-V4 ExpertStore tests: ok");
    return 0;
}
