# Path C port plan: Strata PR #540 Volta optimizations for buun (2026-10-03)

Scoping confirmed against buun sources and the round-12 profile:

## Port 1 — BF16 dense GEMMs via fp16 tensor cores on sm_70

- The Coder carries **484 BF16 tensors** (dense attention/DeltaNet projections kept at bf16
  by the quant recipe; loader summary in any verbose load log).
- `ggml_cuda_should_use_mmf` (ggml/src/ggml-cuda/mmf.cu ~line 207) returns true for
  GGML_TYPE_BF16 only with `ampere_mma_available(cc)` (or AMD wmma/mfma) — on Volta every
  BF16 mul_mat falls back to dequant-to-f32 + cuBLAS SGEMM = the fp32 CUDA-core
  `volta_sgemm_*` kernels (8-9% of measured prefill kernel time, 2/3 of layers).
- The MMF path uses a hand-written mma template (`mul_mat_f_cuda<nv_bfloat162,...>`)
  that assumes bf16 tensor-core instructions, so Volta cannot simply flip the gate.

Port design:
1. `should_use_mmf` case GGML_TYPE_BF16: also true when `volta_mma_available(cc)`,
   env-gated (`GGML_CUDA_BF16_VIA_F16`, default on for cc==70) for A/B.
2. mul_mat dispatch: on Volta, run the F16 template with operands converted bf16->f16
   (in-register load-convert, or a converted scratch copy per Strata's reference commit
   35c0f68; keep Inf on overflow per their review; BF16X2 remainder cases stay on cuBLAS).
3. Accuracy: bf16->f16 rounding of weights only; Strata measured KL ~8.4e-3 on code,
   perplexity 3.448 vs 3.420 — acceptable, and the f32 accumulation is unchanged.

## Port 2 — sm_70 attention kernel (flash::sm70_d256_splitd_dense, 6.6% of kernel time)

- Profile: 348 calls, avg 7.6 ms, max 24.8 ms — the prefill attention on the V100s.
- Strata reference commit 1b8720e: cut shuffles and q/shared-memory traffic in the
  pre-Turing attention kernel; bit-exact; +7% prompt on their engine.
- Locate buun's kernel source (fattn-mma-f16.cuh sm70 path / its d256 variant), apply the
  same traffic reductions, A/B with prefill bench + perplexity spot-check.

Order: port 1 first (bigger share, mechanical); port 2 after. Both measured on the bench
rig with the standard 21K-prompt A/B (baseline: Coder ~1,040 t/s prefill at b/ub 2048).
