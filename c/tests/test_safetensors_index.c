#define _GNU_SOURCE
#include "../safetensors_index.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_all(int fd, const void *data, size_t length) {
    const unsigned char *bytes = (const unsigned char *)data;
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
        "{\"dense.fp8\":{\"dtype\":\"F8_E4M3\",\"shape\":[2,2],"
        "\"data_offsets\":[0,4]},"
        "\"expert.fp4\":{\"dtype\":\"I8\",\"shape\":[2,1],"
        "\"data_offsets\":[4,6]},"
        "\"expert.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2,1],"
        "\"data_offsets\":[6,8]}}";
    static const unsigned char payload[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t header_length = sizeof(header) - 1;
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) return -1;
    int result = write_all(fd, &header_length, sizeof(header_length)) ||
                 write_all(fd, header, (size_t)header_length) ||
                 write_all(fd, payload, sizeof(payload));
    close(fd);
    return result ? -1 : 0;
}

int main(void) {
    char directory[] = "/tmp/colibri-st-XXXXXX";
    char path[256], error[256];
    if (!mkdtemp(directory)) return 1;
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", directory);
    if (write_fixture(path) != 0) return 1;

    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, directory, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (coli_st_shard_count(index) != 1 || coli_st_tensor_count(index) != 3) return 1;
    const ColiSafetensorsTensor *weight = coli_st_find(index, "expert.fp4");
    const ColiSafetensorsTensor *scale = coli_st_find(index, "expert.scale");
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0) return 1;
    if (weight->rank != 2 || weight->shape[0] != 2 || weight->shape[1] != 1 ||
        weight->nbytes != 2 || weight->numel != 2) return 1;
    unsigned char bytes[4] = {0};
    if (coli_st_read_tensor(index, weight, bytes) != 0 ||
        bytes[0] != 5 || bytes[1] != 6) return 1;
    if (coli_st_read_at(index, scale->shard, weight->offset, 4, bytes) != 0 ||
        memcmp(bytes, "\5\6\7\10", 4)) return 1;
    if (strcmp(coli_st_dtype_name(COLI_ST_F8_E4M3), "F8_E4M3")) return 1;
    coli_st_index_close(index);
    unlink(path);
    rmdir(directory);
    puts("safetensors index tests: ok");
    return 0;
}
