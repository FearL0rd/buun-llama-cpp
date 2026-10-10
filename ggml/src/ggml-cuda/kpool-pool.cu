#include "kpool-pool.cuh"

static __global__ void kpool_pool_f32(
        const float * __restrict__ gate, const float * __restrict__ ape, const float * __restrict__ key,
        float * __restrict__ dst, const int n_embd, const int kpool, const int64_t n_rows,
        const int64_t s_g1, const int64_t s_g2, const int64_t s_a1, const int64_t s_k1, const int64_t s_k2) {
    const int d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }

    for (int64_t n = blockIdx.y; n < n_rows; n += gridDim.y) {
        const float * g = gate + n*s_g2 + d;
        const float * k = key  + n*s_k2 + d;
        const float * a = ape  + d;

        float vmax = -INFINITY;
        for (int i = 0; i < kpool; ++i) {
            vmax = fmaxf(vmax, g[i*s_g1] + a[i*s_a1]);
        }

        float sum = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < kpool; ++i) {
            const float e = expf(g[i*s_g1] + a[i*s_a1] - vmax);
            sum += e;
            acc += e*k[i*s_k1];
        }

        dst[n*n_embd + d] = acc/sum;
    }
}

void ggml_cuda_op_kpool_pool(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * ape,
        const ggml_tensor * key, ggml_tensor * dst) {
    const int     n_embd = gate->ne[0];
    const int     kpool  = gate->ne[1];
    const int64_t n_rows = gate->ne[2];
    const size_t  ts     = sizeof(float);

    const int block_size = 128;
    const dim3 grid((n_embd + block_size - 1)/block_size, (unsigned) std::min<int64_t>(n_rows, 65535), 1);
    kpool_pool_f32<<<grid, block_size, 0, ctx.stream()>>>(
        (const float *) gate->data, (const float *) ape->data, (const float *) key->data, (float *) dst->data,
        n_embd, kpool, n_rows, gate->nb[1]/ts, gate->nb[2]/ts, ape->nb[1]/ts, key->nb[1]/ts, key->nb[2]/ts);
}
