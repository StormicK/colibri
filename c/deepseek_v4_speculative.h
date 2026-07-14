#ifndef COLIBRI_DEEPSEEK_V4_SPECULATIVE_H
#define COLIBRI_DEEPSEEK_V4_SPECULATIVE_H

#include <stdint.h>

typedef struct {
    uint64_t rounds;
    uint64_t proposed;
    uint64_t accepted;
    uint64_t minimum_proposals;
    float disable_threshold;
    int enabled;
} ColiV4SpeculativeController;

typedef struct {
    int accepted_draft_tokens;
    int output_count;
    int mismatch_index;
} ColiV4VerificationResult;

void coli_v4_speculative_controller_init(ColiV4SpeculativeController *state,
                                         uint64_t minimum_proposals,
                                         float disable_threshold);
void coli_v4_speculative_record(ColiV4SpeculativeController *state,
                                int proposed, int accepted);
float coli_v4_speculative_acceptance(const ColiV4SpeculativeController *state);

int coli_v4_verify_greedy(ColiV4VerificationResult *result,
                          int *output_tokens, int output_capacity,
                          const int *draft_tokens, int draft_count,
                          const int *target_tokens, int target_count);

#endif
