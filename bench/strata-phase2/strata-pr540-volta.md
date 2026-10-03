# Strata PR #540 (2026-10-02): Volta prefill optimizations to port

Community PR by sskver (Tesla V100-32GB, IQ2_XS) — +22-23% prompt speed on Strata's sm_70
build, decode unchanged. Directly mirrors our round-12 profile findings for the V100 pair:

1. **BF16 GEMMs on Volta run as fp32 CUDA-core kernels** (V100 has no BF16 tensor cores;
   cuBLAS falls back to magma_sgemmEx-class kernels) — ~10% of prompt GPU time there.
   Fix: on cc 7.0, convert operands to FP16 in the existing scratch buffer and use the
   FP16 tensor-core path (env STRATA_BF16_VIA_F16=0 to disable). Not bit-exact (fp16
   accumulation rounds differently) but measured harmless: KL ~8.4e-3 on code slices,
   same top-1 in 23/24, perplexity 3.448 vs 3.420.
   buun counterpart: our profile shows volta_sgemm_128x64_tn + 32x128_tn (fp32 CUDA-core
   GEMMs) ~= 8-9% of kernel time on the V100s. Same fix applies if those GEMMs' operands
   can go through f16 (check GGML_CUDA_F16 / the dsv4_hc and dense paths).

2. **The pre-Turing attention kernel is slow** (~20% of a 20K prefill on their engine);
   the fix cuts shuffles and q/shared-memory traffic — bit-exact, +7%.
   buun counterpart: flash::sm70_d256_splitd_dense_kernel, 6.6% of kernel time, avg
   7.6 ms/call, max 24.8 ms.

Reference implementation: PR commits 35c0f68 (bf16-via-f16), 1b8720e (attention smem/
shuffle cuts, bit-exact), d9cfcb1 (review fixes: BF16X2 remainders stay on cuBLAS).

Queue position: after the residency work (round 17: fit-gate fix + reduced ctx to arm
the moe cache). Expected combined value on our rig: the two ports attack ~15% of
measured prefill kernel time on the cards that host two-thirds of the layers.
