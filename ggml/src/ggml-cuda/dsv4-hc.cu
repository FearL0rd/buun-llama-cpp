#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;
static constexpr int DSV4_HC_POST_TILE_EMBD = 64;
static constexpr int DSV4_HC_MIX_MAX_TOKENS = 8;

template <bool use_repeated_block>
static __global__ void hc_combine_f32(
        const float * residual, const float * block, const float * inject,
        float * dst, int64_t n_embd, int64_t hc, float inv_hc) {
    const int64_t c = blockIdx.y;
    const int64_t t = blockIdx.z;

    // The scatter weight is constant across the embedding tile. Computing it
    // once also preserves the exact SCALE -> SIGMOID -> SCALE boundaries.
    __shared__ float weight_shared;
    if (threadIdx.x == 0) {
        const float scaled = __fmaf_rn(inv_hc, inject[c + hc*t], 0.0f);
        const float sigmoid = 1.0f / (1.0f + expf(-scaled));
        weight_shared = __fmaf_rn(2.0f, sigmoid, 0.0f);
    }
    __syncthreads();

    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) {
        return;
    }

    const int64_t i = e + n_embd*(c + hc*t);
    const int64_t ib = use_repeated_block ? i : e + n_embd*t;
    const float product = __fmul_rn(block[ib], weight_shared);
    dst[i] = __fadd_rn(residual[i], product);
}


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_params_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_pre  = scale[0*ss0];
    const float scale_post = scale[1*ss0];
    const float scale_comb = scale[2*ss0];

    for (int i = 0; i < DSV4_HC; ++i) {
        // The unfused graph stores between MUL and ADD. Explicit round-to-nearest
        // intrinsics preserve that two-kernel arithmetic instead of contracting an FMA.
        const float pre_mul = __fmul_rn(mixes[i*sm0 + it*sm1], scale_pre);
        const float pre_affine = __fadd_rn(pre_mul, base[i*sb0]);
        const float pre = __fadd_rn(1.0f/(1.0f + expf(-pre_affine)), eps);
        dst[i*sd0 + it*sd1] = pre;

        const int post_idx = DSV4_HC + i;
        const float post_mul = __fmul_rn(mixes[post_idx*sm0 + it*sm1], scale_post);
        const float post_affine = __fadd_rn(post_mul, base[post_idx*sb0]);
        const float post = __fmul_rn(1.0f/(1.0f + expf(-post_affine)), 2.0f);
        dst[post_idx*sd0 + it*sd1] = post;
    }

    float comb[DSV4_HC*DSV4_HC];
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb +
                base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f/sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx]*inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int idx = 0; idx < DSV4_HC*DSV4_HC; ++idx) {
        dst[(comb_offset + idx)*sd0 + it*sd1] = comb[idx];
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            const float product = __fmul_rn(xv, wv);
            sum = ih == 0 ? product : __fadd_rn(sum, product);
        } else {
            wv = weights[ih*sw0 + it*sw1];
            if (ih == 0) {
                sum = xv * wv;
            } else {
                sum += xv * wv;
            }
        }
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t tiles_per_token,
        int64_t hc,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        bool    from_inject,
        float   inject_scale) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x / tiles_per_token;
    const int64_t tile = (int64_t) blockIdx.x - it*tiles_per_token;
    const int64_t i0 = tile*DSV4_HC_POST_TILE_EMBD + threadIdx.x;
    const int64_t idst = blockIdx.y * blockDim.y + threadIdx.y;

    ggml_cuda_pdl_sync();

    if (i0 >= n_embd || idst >= hc) {
        return;
    }

    float pv = post[idst*sp0 + it*sp1];
    if (from_inject) {
        // same SCALE -> SIGMOID -> SCALE boundaries as the unfused graph
        const float scaled = __fmaf_rn(inject_scale, pv, 0.0f);
        pv = __fmaf_rn(2.0f, 1.0f / (1.0f + expf(-scaled)), 0.0f);
    }
    float sum = __fmul_rn(x[i0*sx0 + it*sx1], pv);
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum = __fadd_rn(sum, residual[i0*sr0 + idst*sr1 + it*sr2]);
    }
    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_params(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mixes->ne[0] == 24);
    GGML_ASSERT(dst->ne[0] == 24);
    GGML_ASSERT(dst->ne[1] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == 24);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_params_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
            (float *) dst->data, n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float), nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    GGML_ASSERT(hc > 0);

    const int64_t tiles_per_token = (n_embd + DSV4_HC_POST_TILE_EMBD - 1) / DSV4_HC_POST_TILE_EMBD;
    const dim3 block_dims(DSV4_HC_POST_TILE_EMBD, DSV4_HC, 1);
    const dim3 grid_dims(tiles_per_token*n_tokens, (hc + DSV4_HC - 1)/DSV4_HC, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, tiles_per_token, hc,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            ggml_get_op_params_i32(dst, 0) != 0, ggml_get_op_params_f32(dst, 1));
}

void ggml_cuda_op_hc_combine_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block,
        const ggml_tensor * repeated, const ggml_tensor * inject,
        ggml_tensor * dst, bool use_repeated_block) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(block->type    == GGML_TYPE_F32);
    GGML_ASSERT(repeated->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(block));
    GGML_ASSERT(ggml_is_contiguous(repeated));
    GGML_ASSERT(ggml_is_contiguous(inject));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ggml_cuda_set_device(ctx.device);
    const int64_t n_embd = residual->ne[0];
    const int64_t hc = residual->ne[1];
    const float inv_hc = 1.0f / (float) hc;
    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, hc, residual->ne[2]);

    // This kernel has no ggml_cuda_pdl_sync(), so it must use ordinary stream
    // ordering rather than the PDL-capable ggml_cuda_kernel_launch() helper.
    if (use_repeated_block) {
        hc_combine_f32<true><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) residual->data, (const float *) repeated->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    } else {
        hc_combine_f32<false><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) residual->data, (const float *) block->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    }
    CUDA_CHECK(cudaGetLastError());
}

// hc_mix: the gated pre-mix of one hyper-connection for a few tokens (decode and MTP verify).
// Three launches: per-stream RMSNorm, the down + inject matvec, and the up matvec with the
// sigmoid gate and the stream mean. The bf16 weights are read once for all tokens.

static constexpr int DSV4_HC_MIX_DOWN_WARPS = 8; // two warps per row, each over half of hc_dim
static constexpr int DSV4_HC_MIX_UP_WARPS   = 8;
static constexpr int DSV4_HC_MIX_UP_COLS    = 2; // output columns per warp; hc == 4 rows each

static __device__ __forceinline__ void dsv4_hc_bf16x8(const uint4 v, float * f) {
    const uint32_t u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[2*k + 0] = __uint_as_float(u[k] << 16);
        f[2*k + 1] = __uint_as_float(u[k] & 0xffff0000u);
    }
}

static __global__ void dsv4_hc_mix_norm_f32(
        const float * __restrict__ x, const float * __restrict__ gamma, float * __restrict__ xn,
        const int n_embd, const int hc, const float eps) {
    constexpr int block_size = 256;
    const int ic = blockIdx.x; // stream + hc*token

    x  += (int64_t) ic*n_embd;
    xn += (int64_t) ic*n_embd;
    gamma += (ic % hc)*n_embd;

    float tmp = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += block_size) {
        tmp += x[i]*x[i];
    }
    __shared__ float s_sum[WARP_SIZE];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float s = rsqrtf(tmp/n_embd + eps);
    for (int i = threadIdx.x; i < n_embd; i += block_size) {
        xn[i] = s*x[i]*gamma[i];
    }
}

template <int nt>
static __global__ void dsv4_hc_mix_down_bf16(
        const float * __restrict__ xn, const uint4 * __restrict__ w_down, const uint4 * __restrict__ w_inject,
        float * __restrict__ lo, float * __restrict__ inject,
        const int hc_dim, const int r, const int n_rows, const int hc,
        const int64_t sw_down, const int64_t sw_inject, const float scale) {
    constexpr int split = 2;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x*(DSV4_HC_MIX_DOWN_WARPS/split) + warp/split;
    const int part = warp % split;

    float acc[nt] = {};
    if (row < n_rows) {
        const uint4  * w   = row < r ? w_down + row*sw_down : w_inject + (row - r)*sw_inject;
        const float4 * xn4 = (const float4 *) xn;
        const int n8 = hc_dim/8;
        const int j1 = (part + 1)*n8/split;
#pragma unroll 4
        for (int j = part*n8/split + lane; j < j1; j += WARP_SIZE) {
            float wf[8];
            dsv4_hc_bf16x8(w[j], wf);
#pragma unroll
            for (int t = 0; t < nt; ++t) {
                const float4 a = xn4[t*(hc_dim/4) + 2*j + 0];
                const float4 b = xn4[t*(hc_dim/4) + 2*j + 1];
                acc[t] += wf[0]*a.x + wf[1]*a.y + wf[2]*a.z + wf[3]*a.w
                        + wf[4]*b.x + wf[5]*b.y + wf[6]*b.z + wf[7]*b.w;
            }
        }
    }

    __shared__ float partial[DSV4_HC_MIX_DOWN_WARPS][nt];
#pragma unroll
    for (int t = 0; t < nt; ++t) {
        acc[t] = warp_reduce_sum(acc[t]);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            partial[warp][t] = acc[t];
        }
    }
    __syncthreads();

    if (part != 0 || lane >= nt || row >= n_rows) {
        return;
    }
    const float sum = partial[warp][lane] + partial[warp + 1][lane];
    if (row < r) {
        const float v = scale*sum;
        lo[lane*r + row] = v/(1.0f + expf(-v));
    } else {
        inject[lane*hc + row - r] = sum;
    }
}

template <int nt>
static __global__ void dsv4_hc_mix_up_bf16(
        const float * __restrict__ xn, const float * __restrict__ lo, const uint4 * __restrict__ w_up,
        float * __restrict__ mixed, const int n_embd, const int r, const int64_t sw_up, const float scale) {
    extern __shared__ float4 lo_s4[]; // [nt][r/4]
    const float4 * lo4 = (const float4 *) lo;
    for (int k = threadIdx.x; k < nt*r/4; k += blockDim.x) {
        lo_s4[k] = lo4[k];
    }
    __syncthreads();

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int h    = lane / 8; // lanes 8h..8h+7 reduce the gate row of stream h
    const int sub  = lane % 8;
    const int hc_dim = 4*n_embd;
    const int i0 = (blockIdx.x*DSV4_HC_MIX_UP_WARPS + warp)*DSV4_HC_MIX_UP_COLS;

    float acc[DSV4_HC_MIX_UP_COLS][nt] = {};
#pragma unroll
    for (int c = 0; c < DSV4_HC_MIX_UP_COLS; ++c) {
        if (i0 + c >= n_embd) {
            break;
        }
        const uint4 * w = w_up + (int64_t) (i0 + c + n_embd*h)*sw_up;
        for (int j = sub; j < r/8; j += 8) {
            float wf[8];
            dsv4_hc_bf16x8(w[j], wf);
#pragma unroll
            for (int t = 0; t < nt; ++t) {
                const float4 a = lo_s4[t*(r/4) + 2*j + 0];
                const float4 b = lo_s4[t*(r/4) + 2*j + 1];
                acc[c][t] += wf[0]*a.x + wf[1]*a.y + wf[2]*a.z + wf[3]*a.w
                           + wf[4]*b.x + wf[5]*b.y + wf[6]*b.z + wf[7]*b.w;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < DSV4_HC_MIX_UP_COLS; ++c) {
        const int i = i0 + c;
        if (i >= n_embd) {
            break;
        }
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            float g = acc[c][t];
            g += __shfl_xor_sync(0xffffffff, g, 4, 8);
            g += __shfl_xor_sync(0xffffffff, g, 2, 8);
            g += __shfl_xor_sync(0xffffffff, g, 1, 8);
            // same product and stream order as dsv4_hc_pre_f32<true>
            const float p  = __fmul_rn(xn[t*hc_dim + i + n_embd*h], 1.0f/(1.0f + expf(-g)));
            const float p1 = __shfl_sync(0xffffffff, p,  8, WARP_SIZE);
            const float p2 = __shfl_sync(0xffffffff, p, 16, WARP_SIZE);
            const float p3 = __shfl_sync(0xffffffff, p, 24, WARP_SIZE);
            if (lane == 0) {
                mixed[t*n_embd + i] = scale*__fadd_rn(__fadd_rn(__fadd_rn(p, p1), p2), p3);
            }
        }
    }
}

template <int nt>
static void dsv4_hc_mix_launch(
        cudaStream_t stream, const float * xn, const uint4 * w_down, const uint4 * w_inject, const uint4 * w_up,
        float * lo, float * mixed, float * inject, int n_embd, int hc, int r, int n_inject,
        int64_t sw_down, int64_t sw_inject, int64_t sw_up, float scale) {
    const int n_rows = r + n_inject;
    const int rows_per_block = DSV4_HC_MIX_DOWN_WARPS/2;
    dsv4_hc_mix_down_bf16<nt><<<(n_rows + rows_per_block - 1)/rows_per_block, DSV4_HC_MIX_DOWN_WARPS*WARP_SIZE, 0, stream>>>(
            xn, w_down, w_inject, lo, inject, hc*n_embd, r, n_rows, hc, sw_down, sw_inject, scale);

    const int cols_per_block = DSV4_HC_MIX_UP_WARPS*DSV4_HC_MIX_UP_COLS;
    dsv4_hc_mix_up_bf16<nt><<<(n_embd + cols_per_block - 1)/cols_per_block, DSV4_HC_MIX_UP_WARPS*WARP_SIZE, nt*r*sizeof(float), stream>>>(
            xn, lo, w_up, mixed, n_embd, r, sw_up, scale);
}

bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const int64_t n_embd = x->ne[0];
    const int64_t r      = op->src[2]->ne[1];
    if (x->ne[1] != DSV4_HC || x->ne[2] > DSV4_HC_MIX_MAX_TOKENS || n_embd % 8 != 0 || r % 8 != 0) {
        return false;
    }
    for (int i = 2; i <= 4; ++i) {
        const ggml_tensor * w = op->src[i];
        if (w && (w->type != GGML_TYPE_BF16 || w->nb[1] % 16 != 0)) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * w_norm   = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_up     = dst->src[3];
    const ggml_tensor * w_inject = dst->src[4];

    const int n_embd = x->ne[0];
    const int hc     = x->ne[1];
    const int nt     = x->ne[2];
    const int r      = w_down->ne[1];

    const float eps   = ggml_get_op_params_f32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> xn(ctx.pool(), (size_t) hc*n_embd*nt);
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) r*nt);

    dsv4_hc_mix_norm_f32<<<hc*nt, 256, 0, stream>>>(
            (const float *) x->data, (const float *) w_norm->data, xn.get(), n_embd, hc, eps);

    float * mixed  = (float *) dst->data;
    float * inject = mixed + (int64_t) n_embd*nt;
    const uint4 * wi = w_inject ? (const uint4 *) w_inject->data : nullptr;
    const int64_t swi = w_inject ? w_inject->nb[1]/16 : 0;

#define DSV4_HC_MIX_CASE(N) case N: dsv4_hc_mix_launch<N>(stream, xn.get(), (const uint4 *) w_down->data, wi, \
        (const uint4 *) w_up->data, lo.get(), mixed, inject, n_embd, hc, r, w_inject ? hc : 0, \
        w_down->nb[1]/16, swi, w_up->nb[1]/16, scale); break;
    switch (nt) {
        DSV4_HC_MIX_CASE(1)
        DSV4_HC_MIX_CASE(2)
        DSV4_HC_MIX_CASE(3)
        DSV4_HC_MIX_CASE(4)
        DSV4_HC_MIX_CASE(5)
        DSV4_HC_MIX_CASE(6)
        DSV4_HC_MIX_CASE(7)
        DSV4_HC_MIX_CASE(8)
        default: GGML_ABORT("hc_mix: unsupported token count %d", nt);
    }
#undef DSV4_HC_MIX_CASE
    CUDA_CHECK(cudaGetLastError());
}
