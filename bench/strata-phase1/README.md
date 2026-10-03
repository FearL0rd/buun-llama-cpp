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

## Rounds 6-7 (Flash-Next findings)

- Round 6 (`-ub 2048`): Flash-Next cannot load with mmproj on the 24 GB 3090 at ub > 512 —
  the 862 MiB vision encoder OOMs after the model + compute buffers (same with the MMQ env
  off and on; a placement constraint). No `--mmproj-device` placement flag exists, so
  production Flash-Next + bigger ubatches needs `ot` tensor rebalancing off the main GPU.
- Round 7 (mmproj dropped, `-ub 2048`): prefill then hits a pre-existing "Compute error"
  (`ggml_backend_sched_graph_compute_async` -> error -2) ~2-3 s into the 21K-token prompt,
  in BOTH the off and on arms — i.e. independent of this change. **Root cause confirmed
  from the logs: `ggml_backend_cuda_graph_compute: CUDA pool allocation failed (out of
  VRAM)` — the Flash-Next weights pack the 24 GB 3090 (~22 GB; the VBR pool budget there
  is only ~400 MiB), so a 2048-token ubatch compute buffer does not fit. Not a kernel or
  VBR bug: the "VBR already degraded to ~4.46 bpv" observation was VBR shrinking its
  watermark trying to free cells for the failing allocation.** Fix: per-model
  `tensor-split` giving the 3090 fewer layers (round 9). Until then Flash-Next stays at
  `-ub 512`, which works.

## Round 8 (Flash-Next, `-ub 512`, no mmproj — production ubatch)

| path | prefill t/s (req2/req3) | short-ctx decode t/s |
|---|---:|---:|
| fallback (V100 sorted per-expert) | 394.7-398.9 | 43.5 |
| MMQ MoE (V100 dp4a) | **582.3-588.2** | 43.5-43.7 |

**+48% prefill at the production ubatch with no decode cost.** The change is validated on
both models: Coder +50% (`-ub 4096`, 700 -> 1,040 t/s), Flash-Next +48% (`-ub 512`,
395 -> 585 t/s). Enable with `GGML_CUDA_MMQ_MOE_ALL_BATCHES=1` in the launch environment;
the gate is off by default pending broader hardware validation.

## Round 9 (Flash-Next tensor-split rebalance — investigation concluded)

`tensor-split = 0.6,1,1` (accepted, reached the child; ~3090 share 27%->23%) freed the
3090, but the OOM then surfaced on the V100 (CUDA OOM in `fattn-mma-f16.cuh:2514`
`cudaFuncSetAttribute` at 30.6/32 GB), and `-ub 4096` failed to load at all. The freed
3090 space was re-absorbed by the VBR pool re-deriving its budget. Verdict: **Flash-Next
(77 GB weights in the 88 GB pool) has no headroom for ub >= 2048 anywhere — a capacity
limit, not a code bug.** Production stance: Flash-Next stays at `-ub 512` (585 t/s with
the MMQ patch, its practical ceiling on this rig). Any change needs freed VRAM: expert
offload to RAM via moe-cache (hurts prefill — nearly all experts active per chunk), a
smaller KV budget, or a lower-bpw model file.

Flash-Next caveats (both pre-existing, independent of this change): mmproj OOM on the 3090
at ub > 512 (needs `ot` rebalancing to raise the ubatch), and a "Compute error" ~2-3 s into
long prefills at `-ub >= 2048` (VBR/turbo interaction, untested against this change because
the off arm fails identically).
