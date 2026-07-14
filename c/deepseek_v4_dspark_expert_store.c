#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "deepseek_v4_dspark_runtime.h"

static int dspark_store_snprintf_v2(char *output, size_t size,
                                    const char *format, ...) {
    char rewritten[256];
    /* strlen("layers.%d.ffn.experts.") is 22.  Comparing 23 bytes also
     * compares the literal NUL with the following '%' in the real format,
     * preventing the mtp namespace rewrite and loading target experts. */
    if (!strncmp(format, "layers.%d.ffn.experts.", 22))
        snprintf(rewritten, sizeof(rewritten), "mtp.%s", format + 7);
    else
        snprintf(rewritten, sizeof(rewritten), "%s", format);
    va_list arguments; va_start(arguments, format);
    int result = vsnprintf(output, size, rewritten, arguments);
    va_end(arguments); return result;
}

#define snprintf dspark_store_snprintf_v2
#define coli_deepseek_v4_expert_store_open \
    coli_deepseek_v4_dspark_expert_store_open
#include "deepseek_v4_expert_store.c"
#undef coli_deepseek_v4_expert_store_open
#undef snprintf
