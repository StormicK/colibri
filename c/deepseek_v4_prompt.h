#ifndef COLIBRI_DEEPSEEK_V4_PROMPT_H
#define COLIBRI_DEEPSEEK_V4_PROMPT_H

#include <stddef.h>

typedef enum {
    COLI_V4_PROMPT_CHAT,
    COLI_V4_PROMPT_THINKING,
    COLI_V4_PROMPT_RAW,
} ColiDeepSeekV4PromptMode;

int coli_v4_prompt_build(char **output, size_t *output_length,
                         const char *user_message, const char *system_message,
                         ColiDeepSeekV4PromptMode mode);

#endif
