#pragma once

// Fused single-token EXL3 MoE layer (mul1 codebook experts): gate/up gemv -> swiglu(_clamp) -> down gemv ->
// router-weighted expert sum, in ONE persistent launch. Blocks (8 warps) pull work items from a global
// ticket counter in dependency order, so an item only ever waits on items that were handed out before it
// and are therefore held by resident blocks (forward progress with an occupancy-sized grid). Weight
// strips are cp.async'd before an item waits on its inputs, so DRAM streams across phase boundaries.
//   GU item (pair, mat, 128-col block, k-range): quantizes its own activation slice, writes partials
//   C  warp (pair, 128-col group): sums gate/up partials, had + svh, glu, * down suh, had, int8 quant
//   D  item (128-col block, pair, k-range): down partials
//   E  warp (128-col group): sums down partials per pair, had + svh, * router weight, sum over experts
// The last block to exit resets the ticket and dependency counters for the next launch.

#include "exl3-gemv-int8.cuh"
#include "unary.cuh"

namespace exl3_moe_fused {

constexpr int THREADS = 256;
constexpr int NG1     = 4;   // 128-groups per gate/up item (k range 512)
constexpr int NG2     = 4;   // 128-groups per down item
constexpr int MIN_BLOCKS = 4;
constexpr int LAG     = 8;   // pairs of gate/up issued ahead of the down pass
constexpr int MAX_TOKENS = 8;

struct args {
    const uint8_t * Bg; const uint8_t * Bu; const uint8_t * Bd;
    size_t sg, su, sd;   // expert strides (bytes)
    const half * suh_g; const half * svh_g;
    const half * suh_u; const half * svh_u;
    const half * suh_d; const half * svh_d;
    const float * x; int x_nb1;          // [n_tokens][k1]
    const int32_t * ids; int ids_nb1;    // [n_tokens][n_used]
    const float * w; int w_nb1;          // [n_tokens][n_used]
    int n_used, pairs;
    float limit;                         // 0: plain swiglu
    float * out;                         // [n_tokens][k1]
    float * part1;                       // [pairs][2][KW1][n1]
    uint32_t * xq2;                      // [pairs][n1/16][16]
    float2 * xs2;                        // [pairs][n1/128]
    float * part2;                       // [pairs][KW2][k1]
    int * sync;                          // ticket, done, gu[pairs], c[pairs], d[k1/128]; zero at rest
    int k1, n1;
};

// sync ints a launch uses
static __host__ __device__ inline int sync_count(int pairs, int k1) {
    return 2 + 2 * pairs + k1 / 128;
}

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
__device__ __forceinline__ int ld_acquire(const int * p) {
    int v;
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void wait_ge(const int * p, int target) {
    while (ld_acquire(p) < target) __nanosleep(64);
}

// 2-bit window extraction for 8 consecutive trellis positions (all 8 from two words)
__device__ __forceinline__ void ext8w_2(const uint32_t * ptr, int t0, uint32_t (&w)[8]) {
    const uint32_t i1 = t0 >> 4;
    const uint32_t i0 = (i1 + 15) & 15;
    const uint32_t a = ptr[i0];
    const uint32_t b = ptr[i1];
    const uint32_t s = ((~t0) & 8) << 1;
    const uint32_t c0 = __funnelshift_r(b, a, s);
    const uint32_t c2 = __funnelshift_r(b, a, s + 2);
    const uint32_t c4 = __funnelshift_r(b, a, s + 4);
    const uint32_t c6 = __funnelshift_r(b, a, s + 6);
    w[7] = __byte_perm(c0, 0, 0x4410); w[3] = __byte_perm(c0, 0, 0x4421);
    w[6] = __byte_perm(c2, 0, 0x4410); w[2] = __byte_perm(c2, 0, 0x4421);
    w[5] = __byte_perm(c4, 0, 0x4410); w[1] = __byte_perm(c4, 0, 0x4421);
    w[4] = __byte_perm(c6, 0, 0x4410); w[0] = __byte_perm(c6, 0, 0x4421);
}

// int8-quantize one 128-group held as 4 values per lane into the pre-splatted layout at dst (128 words)
__device__ __forceinline__ float2 quant_group(float (&v)[4], int lane, uint32_t * dst) {
    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float bias  = __half2float(__ushort_as_half(0xc931));
    const float cbias = 1024.0f * k_inv + bias;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] *= exl3_had::SCALE; amax = fmaxf(amax, fabsf(v[i])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float q = fmaxf(amax, 1e-30f) / 127.0f;
    const float qi = 1.0f / q;
    int s = 0;
    dst += (lane >> 2) * 16;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int iq = max(-127, min(127, __float2int_rn(v[i] * qi)));
        s += iq;
        const int r = (lane & 3) * 4 + i;
        dst[((r & 7) >> 1) * 4 + ((r & 1) | (r >> 3) << 1)] = uint32_t(uint8_t(iq)) * 0x01010101u;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return make_float2(q * k_inv, q * cbias * float(s));
}

__device__ __forceinline__ float4 ld_half4(const half * p) {
    const half2 a = *reinterpret_cast<const half2 *>(p), b = *reinterpret_cast<const half2 *>(p + 2);
    return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

template <int bits, int NG>
__device__ __forceinline__ void load_weights(uint32_t * sw, const uint8_t * B, int nt, int kslices, int kt0, int lane) {
    constexpr int TW = 8 * bits;
    const uint32_t * src = reinterpret_cast<const uint32_t *>(B) + (size_t(nt) * kslices + kt0) * TW;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
#pragma unroll
        for (int i = lane * 4; i < 8 * TW; i += 128) exl3_int8::cp_async16(sw + g * 8 * TW + i, src + g * 8 * TW + i);
        exl3_int8::cp_async_commit();
    }
}

// one warp: 16 columns x NG groups; sa pre-splatted; lane g < NG holds gs = group g scales
template <int bits, int NG, bool PIPE>
__device__ __forceinline__ float2 dot_tile(const uint32_t * sw, const uint32_t * sa, float2 gs, int lane) {
    static_assert(NG <= 5, "cp.async wait depth");
    constexpr int TW = 8 * bits;
    const int t0 = lane << 3;
    float f0 = 0.0f, f1 = 0.0f;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        if constexpr (PIPE) {
            if      (g == NG - 1) exl3_int8::cp_async_wait<0>();
            else if (g == NG - 2) exl3_int8::cp_async_wait<1>();
            else if (g == NG - 3) exl3_int8::cp_async_wait<2>();
            else if (g == NG - 4) exl3_int8::cp_async_wait<3>();
            else                  exl3_int8::cp_async_wait<4>();
            __syncwarp();
        }
        int a0 = 0, a1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kb = g * 8 + j;
            uint32_t w[8];
            if constexpr (bits == 2) ext8w_2(sw + kb * TW, t0, w);
            else exl3_int8::ext8w<bits>(sw + kb * TW, t0, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
#pragma unroll
            for (int i = 0; i < 8; ++i) w[i] *= 0x83DCD12Du;
            const uint4 av = *reinterpret_cast<const uint4 *>(sa + kb * 16 + (lane & 3) * 4);
            a0 = exl3_int8::dp4a_us(w[0], av.x, a0); a0 = exl3_int8::dp4a_us(w[1], av.y, a0);
            a0 = exl3_int8::dp4a_us(w[2], av.z, a0); a0 = exl3_int8::dp4a_us(w[3], av.w, a0);
            a1 = exl3_int8::dp4a_us(w[4], av.x, a1); a1 = exl3_int8::dp4a_us(w[5], av.y, a1);
            a1 = exl3_int8::dp4a_us(w[6], av.z, a1); a1 = exl3_int8::dp4a_us(w[7], av.w, a1);
        }
        const float qk = __shfl_sync(0xffffffffu, gs.x, g);
        f0 += qk * float(a0);
        f1 += qk * float(a1);
    }
    f0 += __shfl_xor_sync(0xffffffffu, f0, 1); f0 += __shfl_xor_sync(0xffffffffu, f0, 2);
    f1 += __shfl_xor_sync(0xffffffffu, f1, 1); f1 += __shfl_xor_sync(0xffffffffu, f1, 2);
    float bsum = lane < NG ? gs.y : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) bsum += __shfl_xor_sync(0xffffffffu, bsum, o);
    return make_float2(f0 + bsum, f1 + bsum);
}

__device__ __forceinline__ void store_tile(float * dst, float2 f, int lane) {
    if (!(lane & 3)) {
        dst[lane >> 2]       = f.x;
        dst[(lane >> 2) + 8] = f.y;
    }
}

// block-level release: every thread's prior global writes, then one counter increment
__device__ __forceinline__ void signal_block(int * ctr) {
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicAdd(ctr, 1); }
}
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

template <int BG, int BD>
struct cfg {
    static constexpr int SW = 8 * (NG1 * 8 * 8 * BG > NG2 * 8 * 8 * BD ? NG1 * 8 * 8 * BG : NG2 * 8 * 8 * BD);
    static constexpr int SA = (NG1 > NG2 ? NG1 : NG2) * 128;
};

// grid = resident blocks, block = THREADS
template <int BG, int BD>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS) moe(const args a) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    using C = cfg<BG, BD>;
    __shared__ __align__(16) uint32_t sw_all[C::SW];
    __shared__ __align__(16) uint32_t sa[C::SA];
    __shared__ __align__(16) float sfin[128];
    __shared__ float2 sgs[8];
    __shared__ int s_ticket;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k1 = a.k1, n1 = a.n1, P = a.pairs;
    const int NB1 = n1 / 128, KW1 = k1 / 128 / NG1;
    const int NB2 = k1 / 128, KW2 = n1 / 128 / NG2;
    // per pair: G gate/up items, then Cp glu items + Dp down items issued lag pairs later; E items last
    const int G = 2 * NB1 * KW1, Cp = (NB1 + 7) / 8, Dp = NB2 * KW2, CD = Cp + Dp;
    const int lag = min(LAG, P);
    const int head = lag * G, mid = (P - lag) * (G + CD), tail = lag * CD;
    const int total = head + mid + tail + (NB2 + 7) / 8;
    int * ctr_gu = a.sync + 2;
    int * ctr_c  = ctr_gu + P;
    int * ctr_d  = ctr_c + P;

    for (;;) {
        if (threadIdx.x == 0) s_ticket = atomicAdd(a.sync, 1);
        __syncthreads();
        const int it = s_ticket;
        __syncthreads();
        if (it >= total) break;
        // kind 0 = gate/up, 1 = glu, 2 = down, 3 = expert sum
        int kind, pair = 0, r;
        if (it < head) {
            kind = 0; pair = it / G; r = it - pair * G;
        } else if (it < head + mid) {
            const int s = (it - head) / (G + CD);
            r = it - head - s * (G + CD);
            if (r < G) { kind = 0; pair = s + lag; }
            else       { kind = 1; pair = s; r -= G; }
        } else if (it < head + mid + tail) {
            const int s = (it - head - mid) / CD;
            kind = 1; pair = P - lag + s; r = it - head - mid - s * CD;
        } else {
            kind = 3; r = it - head - mid - tail;
        }
        if (kind == 1 && r >= Cp) { kind = 2; r -= Cp; }
        const int t = pair / a.n_used, e = pair - t * a.n_used;
        const int expert = kind == 3 ? 0 : a.ids[size_t(t) * a.ids_nb1 + e];

        if (kind == 0) {
            const int mat = r / (NB1 * KW1); r -= mat * NB1 * KW1;
            const int nb = r / KW1, wk = r - nb * KW1;
            if (expert >= 0) {
                constexpr int TW = 8 * BG;
                uint32_t * sw = sw_all + warp * NG1 * 8 * TW;
                const uint8_t * B = (mat ? a.Bu : a.Bg) + size_t(expert) * (mat ? a.su : a.sg);
                const int nt = nb * 8 + warp, kt0 = wk * NG1 * 8;
                load_weights<BG, NG1>(sw, B, nt, k1 / 16, kt0, lane);
                if (warp < NG1) {
                    const int col = (wk * NG1 + warp) * 128 + lane * 4;
                    const float4 xv = *reinterpret_cast<const float4 *>(a.x + size_t(t) * a.x_nb1 + col);
                    const float4 s = ld_half4((mat ? a.suh_u : a.suh_g) + size_t(expert) * k1 + col);
                    float v[4] = { xv.x * s.x, xv.y * s.y, xv.z * s.z, xv.w * s.w };
                    exl3_had::had128(v[0], v[1], v[2], v[3], lane);
                    const float2 gs = quant_group(v, lane, sa + warp * 128);
                    if (lane == 0) sgs[warp] = gs;
                }
                __syncthreads();
                const float2 gs = lane < NG1 ? sgs[lane] : make_float2(0.0f, 0.0f);
                const float2 f = dot_tile<BG, NG1, true>(sw, sa, gs, lane);
                store_tile(a.part1 + ((size_t(pair) * 2 + mat) * KW1 + wk) * n1 + nt * 16, f, lane);
            }
            signal_block(ctr_gu + pair);
        } else if (kind == 1) {
            const int cg = r * 8 + warp;
            if (cg < NB1) {
                if (lane == 0) wait_ge(ctr_gu + pair, G);
                __syncwarp();
                if (expert >= 0) {
                    const int c = cg * 128 + lane * 4;
                    float h[2][4];
                    float4 accs[2] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f) };
                    // both matrices' partials in batches of 4 independent loads (L2 latency, not a serial chain)
                    for (int wk = 0; wk < KW1; wk += 4) {
                        float4 q[2][4];
#pragma unroll
                        for (int mat = 0; mat < 2; ++mat) {
                            const float * p = a.part1 + (size_t(pair) * 2 + mat) * KW1 * n1 + c;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                q[mat][i] = wk + i < KW1 ? __ldcg(reinterpret_cast<const float4 *>(p + size_t(wk + i) * n1))
                                                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                            }
                        }
#pragma unroll
                        for (int mat = 0; mat < 2; ++mat) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                accs[mat].x += q[mat][i].x; accs[mat].y += q[mat][i].y;
                                accs[mat].z += q[mat][i].z; accs[mat].w += q[mat][i].w;
                            }
                        }
                    }
#pragma unroll
                    for (int mat = 0; mat < 2; ++mat) {
                        float4 acc = accs[mat];
                        exl3_had::had128(acc.x, acc.y, acc.z, acc.w, lane);
                        const float4 sv = ld_half4((mat ? a.svh_u : a.svh_g) + size_t(expert) * n1 + c);
                        h[mat][0] = acc.x * exl3_had::SCALE * sv.x; h[mat][1] = acc.y * exl3_had::SCALE * sv.y;
                        h[mat][2] = acc.z * exl3_had::SCALE * sv.z; h[mat][3] = acc.w * exl3_had::SCALE * sv.w;
                    }
                    const float4 sd = ld_half4(a.suh_d + size_t(expert) * n1 + c);
                    const float sdv[4] = { sd.x, sd.y, sd.z, sd.w };
                    float v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float g = h[0][i], up = h[1][i];
                        v[i] = (a.limit > 0.0f ? ggml_cuda_op_swiglu_clamp_single(g, up, a.limit) :
                                                 ggml_cuda_op_silu_single(g) * up) * sdv[i];
                    }
                    exl3_had::had128(v[0], v[1], v[2], v[3], lane);
                    const float2 gs = quant_group(v, lane, a.xq2 + (size_t(pair) * NB1 + cg) * 128);
                    if (lane == 0) a.xs2[size_t(pair) * NB1 + cg] = gs;
                }
                __syncwarp();
                if (lane == 0) { __threadfence(); atomicAdd(ctr_c + pair, 1); }
            }
        } else if (kind == 2) {
            const int nb = r / KW2, wk = r - nb * KW2;
            if (expert >= 0) {
                constexpr int TW = 8 * BD;
                uint32_t * sw = sw_all + warp * NG2 * 8 * TW;
                const int nt = nb * 8 + warp, kt0 = wk * NG2 * 8;
                load_weights<BD, NG2>(sw, a.Bd + size_t(expert) * a.sd, nt, n1 / 16, kt0, lane);
                if (threadIdx.x == 0) wait_ge(ctr_c + pair, NB1);
                __syncthreads();
                const uint32_t * xsrc = a.xq2 + (size_t(pair) * NB1 + wk * NG2) * 128;
                for (int i = threadIdx.x * 4; i < NG2 * 128; i += 4 * THREADS) exl3_int8::cp_async16(sa + i, xsrc + i);
                exl3_int8::cp_async_commit();
                const float2 gs = lane < NG2 ? __ldcg(a.xs2 + size_t(pair) * NB1 + wk * NG2 + lane) : make_float2(0.0f, 0.0f);
                exl3_int8::cp_async_wait<0>();
                __syncthreads();
                const float2 f = dot_tile<BD, NG2, false>(sw, sa, gs, lane);
                if (KW2 == 1) {
                    // whole k range: finish the 128 columns here, pre-scaled by the router weight
                    store_tile(sfin + warp * 16, f, lane);
                    __syncthreads();
                    if (warp == 0) {
                        float4 v = *reinterpret_cast<const float4 *>(sfin + lane * 4);
                        exl3_had::had128(v.x, v.y, v.z, v.w, lane);
                        const int c = nb * 128 + lane * 4;
                        const float4 sv = ld_half4(a.svh_d + size_t(expert) * k1 + c);
                        const float wt = a.w[size_t(t) * a.w_nb1 + e] * exl3_had::SCALE;
                        *reinterpret_cast<float4 *>(a.part2 + size_t(pair) * k1 + c) =
                            make_float4(v.x * sv.x * wt, v.y * sv.y * wt, v.z * sv.z * wt, v.w * sv.w * wt);
                    }
                } else {
                    store_tile(a.part2 + (size_t(pair) * KW2 + wk) * k1 + nt * 16, f, lane);
                }
            }
            signal_block(ctr_d + nb);
        } else {
            const int cg = r * 8 + warp;
            if (cg < NB2) {
                if (lane == 0) wait_ge(ctr_d + cg, P * KW2);
                __syncwarp();
                const int c = cg * 128 + lane * 4;
                for (int t = 0; t < P / a.n_used; ++t) {
                    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (KW2 == 1 && a.n_used <= 8) {
                        // independent loads; skipped experts contribute zero via the predicate
                        float4 q[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            const bool ok = e < a.n_used && a.ids[size_t(t) * a.ids_nb1 + e] >= 0;
                            q[e] = ok ? __ldcg(reinterpret_cast<const float4 *>(a.part2 + size_t(t * a.n_used + e) * k1 + c))
                                      : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        }
#pragma unroll
                        for (int e = 0; e < 8; ++e) { o.x += q[e].x; o.y += q[e].y; o.z += q[e].z; o.w += q[e].w; }
                        *reinterpret_cast<float4 *>(a.out + size_t(t) * k1 + c) = o;
                        continue;
                    }
                    for (int e = 0; e < a.n_used; ++e) {
                        const int pr = t * a.n_used + e;
                        const int ex = a.ids[size_t(t) * a.ids_nb1 + e];
                        if (ex < 0) continue;
                        const float * p = a.part2 + size_t(pr) * KW2 * k1 + c;
                        float4 acc = __ldcg(reinterpret_cast<const float4 *>(p));
                        if (KW2 > 1) {
                            for (int wk = 1; wk < KW2; ++wk) {
                                const float4 q = __ldcg(reinterpret_cast<const float4 *>(p + size_t(wk) * k1));
                                acc.x += q.x; acc.y += q.y; acc.z += q.z; acc.w += q.w;
                            }
                            exl3_had::had128(acc.x, acc.y, acc.z, acc.w, lane);
                            const float4 sv = ld_half4(a.svh_d + size_t(ex) * k1 + c);
                            const float wt = a.w[size_t(t) * a.w_nb1 + e] * exl3_had::SCALE;
                            acc.x *= sv.x * wt; acc.y *= sv.y * wt; acc.z *= sv.z * wt; acc.w *= sv.w * wt;
                        }
                        o.x += acc.x; o.y += acc.y; o.z += acc.z; o.w += acc.w;
                    }
                    *reinterpret_cast<float4 *>(a.out + size_t(t) * k1 + c) = o;
                }
            }
        }
    }

    // the last block out resets the schedule for the next launch
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.sync + 1, 1) == int(gridDim.x) - 1) {
            const int n = sync_count(P, k1);
            for (int i = 0; i < n; ++i) a.sync[i] = 0;
            __threadfence();
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

} // namespace exl3_moe_fused
