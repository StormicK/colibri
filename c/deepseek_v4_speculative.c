#include "deepseek_v4_speculative.h"

#include <string.h>

void coli_v4_speculative_controller_init(ColiV4SpeculativeController *state,
                                         uint64_t minimum_proposals,
                                         float disable_threshold) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->minimum_proposals = minimum_proposals ? minimum_proposals : 32;
    state->disable_threshold = disable_threshold > 0.0f &&
                               disable_threshold <= 1.0f
        ? disable_threshold : 0.35f;
    state->enabled = 1;
}

float coli_v4_speculative_acceptance(const ColiV4SpeculativeController *state) {
    return state && state->proposed
        ? (float)state->accepted / (float)state->proposed : 0.0f;
}

void coli_v4_speculative_record(ColiV4SpeculativeController *state,
                                int proposed, int accepted) {
    if (!state || !state->enabled || proposed < 1 || accepted < 0 ||
        accepted > proposed) return;
    state->rounds++;
    state->proposed += (uint64_t)proposed;
    state->accepted += (uint64_t)accepted;
    if (state->proposed >= state->minimum_proposals &&
        coli_v4_speculative_acceptance(state) < state->disable_threshold)
        state->enabled = 0;
}

int coli_v4_verify_greedy(ColiV4VerificationResult *result,
                          int *output_tokens, int output_capacity,
                          const int *draft_tokens, int draft_count,
                          const int *target_tokens, int target_count) {
    if (!result || !output_tokens || !draft_tokens || !target_tokens ||
        draft_count < 1 || target_count != draft_count + 1 ||
        output_capacity < draft_count + 1) return -1;
    int accepted = 0;
    while (accepted < draft_count &&
           draft_tokens[accepted] == target_tokens[accepted]) {
        output_tokens[accepted] = draft_tokens[accepted];
        accepted++;
    }
    output_tokens[accepted] = target_tokens[accepted];
    result->accepted_draft_tokens = accepted;
    result->output_count = accepted + 1;
    result->mismatch_index = accepted < draft_count ? accepted : -1;
    return 0;
}
