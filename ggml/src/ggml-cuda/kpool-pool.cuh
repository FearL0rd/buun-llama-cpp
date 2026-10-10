#include "common.cuh"

// dst[d, n] = sum_k softmax_k(gate[d, k, n] + ape[d, k]) * key[d, k, n]
// gate, key: [n_embd, kpool, n] F32 with unit stride in d; ape: [n_embd, kpool] F32; dst: [1, n_embd, n] contiguous.
void ggml_cuda_op_kpool_pool(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * ape,
        const ggml_tensor * key, ggml_tensor * dst);
