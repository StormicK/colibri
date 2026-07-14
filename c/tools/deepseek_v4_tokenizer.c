#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tok.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TEXT\n", argv[0]);
        return 2;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer.json", argv[1]);
    Tok tokenizer;
    tok_load(&tokenizer, path);
    int capacity = (int)strlen(argv[2]) + 16;
    int *ids = malloc((size_t)capacity * sizeof(*ids));
    char *decoded = malloc((size_t)strlen(argv[2]) * 4 + 1024);
    if (!ids || !decoded) return 1;
    int count = tok_encode(&tokenizer, argv[2], (int)strlen(argv[2]),
                           ids, capacity);
    printf("count=%d ids=", count);
    for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", ids[i]);
    int length = tok_decode(&tokenizer, ids, count, decoded,
                            (int)strlen(argv[2]) * 4 + 1023);
    printf("\ndecoded=");
    fwrite(decoded, 1, (size_t)length, stdout);
    fputc('\n', stdout);
    free(decoded); free(ids);
    return 0;
}
