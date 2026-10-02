# Strata Phase 1 — Volta MoE prefill MMQ (2026-10-02)

Change under test: `ggml_cuda_should_use_mmq` extension (`daf396fd8`, env-gated
`GGML_CUDA_MMQ_MOE_ALL_BATCHES=1`) that routes MoE mul_mat_id prefill batches on
dp4a-capable NVIDIA GPUs without turing MMA (i.e. the V100-32GB pair) to the device-side
MMQ-with-ids path, instead of the sorted per-expert fallback (host syncs x2 +
O(experts x tokens) CPU sort + ~500 small cuBLAS GEMMs per MoE tensor per chunk).

Decoder for the numbers: prompt = 21,208-token code prompt, `cache-prompt = off`,
`-fa on`, MTP draft n3, CUDA fusion on, `-b 2048`, `CUDA_VISIBLE_DEVICES=1,3,0`
(3090 main + 2x V100, layer split).

## Round 5 (Coder, `-ub 4096`)

| path | prefill t/s (req2/req3) | long-ctx decode t/s |
|---|---:|---:|
| fallback (V100 sorted per-expert) | 696-713 | 45.0-45.2 |
| MMQ MoE (V100 dp4a) | **1025-1049** | 44.9-45.2 |

- **+50% prefill at identical ubatch; no decode delta.** vs the production baseline
  (`-ub 512`, 484 t/s) the combined stack is +115% prompt speed.
- Decode rows: slots still held the 21K prefills, so those measure long-context decode —
  matched conditions off/on, and they show no regression from the MMQ route.
- Flash-Next could not load at `-ub 4096`: its 862 MiB mmproj OOMs on the 24 GB 3090 after
  the bigger compute buffers (same failure at 8192 in phase 0 round 4; happens with the env
  off and on — a placement constraint, not this change). Round 6 retests Flash-Next at
  `-ub 2048`.

Strata reference: ~2,100-2,230 t/s on one RTX 5070 — the V100 prefill path was a big part
of the remaining gap; what is left is pipeline-level (chunk overlap, GEMM efficiency).
