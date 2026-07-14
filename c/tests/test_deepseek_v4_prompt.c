#include "../deepseek_v4_prompt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect(const char *user, const char *system,
                  ColiDeepSeekV4PromptMode mode, const char *expected) {
    char *actual = NULL;
    size_t length = 0;
    int result = coli_v4_prompt_build(
        &actual, &length, user, system, mode);
    int failed = result || !actual || strcmp(actual, expected) ||
                 length != strlen(expected);
    free(actual);
    return failed;
}

int main(void) {
    if (expect("Hello", NULL, COLI_V4_PROMPT_CHAT,
               "<｜begin▁of▁sentence｜><｜User｜>Hello"
               "<｜Assistant｜></think>") ||
        expect("Hello", "Be concise.", COLI_V4_PROMPT_THINKING,
               "<｜begin▁of▁sentence｜>Be concise.<｜User｜>Hello"
               "<｜Assistant｜><think>") ||
        expect("raw", "ignored", COLI_V4_PROMPT_RAW, "raw"))
        return 1;
    puts("DeepSeek V4 prompt tests: ok");
    return 0;
}
