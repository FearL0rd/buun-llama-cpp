#include "sel-mask.cuh"
#include "convert.cuh"

// One block per token. -inf + mask is -inf for any mask value but +inf, so the fill skips reading mask.
// Scatters run in order with a block barrier between them, matching the SET_ROWS sequence.
template <typename T>
static __global__ void sel_mask_f(T * __restrict__ dst, const T * __restrict__ mask, const int64_t n_kv,
        const int64_t s_dst, const int64_t s_mask, const sel_mask_scatter sc0, const sel_mask_scatter sc1) {
    const int64_t t = blockIdx.x;
    T       * d = dst  + t*s_dst;
    const T * m = mask + t*s_mask;

    for (int64_t c = threadIdx.x; c < n_kv; c += blockDim.x) {
        d[c] = ggml_cuda_cast<T>(-INFINITY);
    }
    for (int k = 0; k < 2; ++k) {
        const sel_mask_scatter & sc = k == 0 ? sc0 : sc1;
        __syncthreads();
        for (int s = threadIdx.x; s < sc.n; s += blockDim.x) {
            const int64_t c = sc.idx[s*sc.s_idx0 + t*sc.s_idx1];
            if (c < n_kv) {
                // SET_ROWS rounds the value to T, then ADD sums in F32
                const float v = ggml_cuda_cast<float>(ggml_cuda_cast<T>(sc.val[s*sc.s_val0 + t*sc.s_val1]));
                d[c] = ggml_cuda_cast<T>(v + ggml_cuda_cast<float>(m[c]));
            }
        }
    }
}

void ggml_cuda_op_sel_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * mask,
        const sel_mask_scatter * scatters, int n_scatters, ggml_tensor * dst) {
    GGML_ASSERT(n_scatters >= 1 && n_scatters <= 2);
    const sel_mask_scatter none = {};
    const sel_mask_scatter & sc1 = n_scatters > 1 ? scatters[1] : none;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const size_t  ts       = ggml_type_size(dst->type);
    const int     block    = 256;
    if (dst->type == GGML_TYPE_F16) {
        sel_mask_f<<<n_tokens, block, 0, ctx.stream()>>>((half *) dst->data, (const half *) mask->data,
            n_kv, dst->nb[1]/ts, mask->nb[1]/ts, scatters[0], sc1);
    } else {
        sel_mask_f<<<n_tokens, block, 0, ctx.stream()>>>((float *) dst->data, (const float *) mask->data,
            n_kv, dst->nb[1]/ts, mask->nb[1]/ts, scatters[0], sc1);
    }
}
