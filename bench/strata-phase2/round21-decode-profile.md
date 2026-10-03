# Round 21: decode profile — decode is dispatch/launch-bound, not GPU-bound (2026-10-03)

nsys capture of steady speculative decoding (Coder, MTP n3, fusion on, 3-GPU layer split).
512-token request: 11.02 s wall under tracing (~48.8 t/s instrumented; 59.8 in production).

Key finding: **total GPU kernel time in the capture is < 1 s — about 3-4 ms of GPU work
per ~50 ms decode cycle (~7% GPU utilization)**. The verify pass (38.1 ms measured earlier)
contains almost no GPU compute; it is CPU dispatch + kernel-launch + cross-device sync:

- ~260 kernel launches per cycle across 3 backends; e.g. quantize_q8_1 10,996 instances
  (2 us each), scale_f32 6,587, unary/sigmoid 3,583, dsv4_hc_pre/post 2,116+2,116.
- CUDA graphs cannot capture across the multi-GPU layer split (upstream limitation), so
  every op pays individual dispatch; the fusion machinery only merges some elementwise ops.
- Top GPU kernel (20.1% of kernel time, but ~1 ms/cycle of wall): the shared MTP head's
  Q4_K GEMV per drafted position (767 calls, avg 255 us, max 868 us = full head read).
  This is the draft-vocab restriction target — worth ~2% wall, not more.
- MoE expert GEMVs (mul_mat_vec_q_moe, IQ1_M family types) are individually 50-290 us
  and only ~13% of kernel time combined — the experts are NOT the decode bottleneck.

Implications (ranked):
1. Fewer pipeline stages -> fewer dispatch+sync boundaries per cycle: test 2-GPU splits
   (V100+V100 = 64 GB combined fits the 55.8 GB model; ctx reduced for KV). Strata's own
   data: their 2-card decode 75-104 t/s; helper/third cards measurably hurt.
2. Fusion expansion for the tiny elementwise/quantize launches (scale_f32, sigmoid,
   k_bin_bcast, quantize_q8_1 pre-MMVQ) — each fused op removes hundreds of launches/cycle.
3. Draft-vocab restriction: ~2% (the head GEMV), code-only.
4. Long-context decode (-20% at 21K): separate investigation after the split test.

Raw: decode-r21.nsys-rep on AISERVER; kernel table in round21-kernels.txt.
