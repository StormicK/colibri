#ifndef COLIBRI_DEEPSEEK_V4_SPARSE_ATTENTION_H
#define COLIBRI_DEEPSEEK_V4_SPARSE_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                 const float *kv, const float *sinks,
                                 const int *indices, int heads,
                                 int head_dimension, int kv_count, int topk,
                                 float softmax_scale);

#ifdef __cplusplus
}
#endif

#endif
