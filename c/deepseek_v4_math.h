#ifndef COLIBRI_DEEPSEEK_V4_MATH_H
#define COLIBRI_DEEPSEEK_V4_MATH_H

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                              const float *mixes, const float scale[3],
                              const float *base, int hc, int iterations,
                              float eps);

int coli_v4_hc_pre(float *output, float *post, float *comb,
                   const float *input, const float *hc_fn,
                   const float scale[3], const float *base,
                   int hc, int dimension, int iterations,
                   float norm_eps, float hc_eps);

int coli_v4_hc_post(float *output, const float *branch,
                    const float *residual, const float *post,
                    const float *comb, int hc, int dimension);

int coli_v4_rmsnorm(float *output, const float *input, const float *weight,
                    int dimension, float eps);

int coli_v4_rope_precompute(float *cosines, float *sines,
                            int dimension, int sequence_length,
                            int original_sequence_length, float base,
                            float factor, int beta_fast, int beta_slow);

int coli_v4_rope_apply(float *vectors, int vector_count, int dimension,
                       const float *cosines, const float *sines, int inverse);

int coli_v4_route(float *weights, int *indices, const float *hidden,
                  const float *gate, const float *bias,
                  const int *forced_indices, int experts, int dimension,
                  int topk, float route_scale);

int coli_v4_swiglu(float *output, const float *gate, const float *up,
                   int dimension, float limit);

#ifdef __cplusplus
}
#endif

#endif
