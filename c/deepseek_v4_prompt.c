#include "deepseek_v4_prompt.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char v4_bos[] = "<｜begin▁of▁sentence｜>";
static const char v4_user[] = "<｜User｜>";
static const char v4_assistant[] = "<｜Assistant｜>";

static int add_size(size_t *total, size_t value) {
    if (SIZE_MAX - *total < value) return -1;
    *total += value;
    return 0;
}

int coli_v4_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         ColiDeepSeekV4PromptMode mode) {
    if (!output || !user_message || mode < COLI_V4_PROMPT_CHAT ||
        mode > COLI_V4_PROMPT_RAW) return -1;
    *output = NULL;
    if (output_length) *output_length = 0;
    const char *system = system_message ? system_message : "";
    if (mode == COLI_V4_PROMPT_RAW) {
        size_t length = strlen(user_message);
        char *copy = malloc(length + 1);
        if (!copy) return -1;
        memcpy(copy, user_message, length + 1);
        *output = copy;
        if (output_length) *output_length = length;
        return 0;
    }
    const char *thinking = mode == COLI_V4_PROMPT_THINKING
        ? "<think>" : "</think>";
    size_t length = 0;
    if (add_size(&length, strlen(v4_bos)) ||
        add_size(&length, strlen(system)) ||
        add_size(&length, strlen(v4_user)) ||
        add_size(&length, strlen(user_message)) ||
        add_size(&length, strlen(v4_assistant)) ||
        add_size(&length, strlen(thinking)) || length == SIZE_MAX)
        return -1;
    char *prompt = malloc(length + 1);
    if (!prompt) return -1;
    char *at = prompt;
#define APPEND(part) do { \
    size_t count = strlen(part); memcpy(at, part, count); at += count; \
} while (0)
    APPEND(v4_bos);
    APPEND(system);
    APPEND(v4_user);
    APPEND(user_message);
    APPEND(v4_assistant);
    APPEND(thinking);
#undef APPEND
    *at = '\0';
    *output = prompt;
    if (output_length) *output_length = length;
    return 0;
}
