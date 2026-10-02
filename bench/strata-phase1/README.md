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
  in BOTH the off and on arms — i.e. independent of this change. The Flash-Next slot
  differs from Coder's: n_ctx 262144 (no ctx-size in its config section) and VBR already
  degraded at load to ~4.46 bpv (turbo4), versus Coder holding 8.125 bpv (turbo8). Suspect:
  VBR-degrade/turbo interaction with large ubatches on this model; needs its own
  investigation. Round 3 established that Flash-Next prefill works at `-ub 512`, so round 8
  runs the A/B there (the production ubatch).

## Round 8 (Flash-Next, `-ub 512`, no mmproj — production ubatch)

| path | prefill t/s (req2/req3) | short-ctx decode t/s |
|---|---:|---:|
| fallback (V100 sorted per-expert) | 394.7-398.9 | 43.5 |
| MMQ MoE (V100 dp4a) | **582.3-588.2** | 43.5-43.7 |

**+48% prefill at the production ubatch with no decode cost.** The change is validated on
both models: Coder +50% (`-ub 4096`, 700 -> 1,040 t/s), Flash-Next +48% (`-ub 512`,
395 -> 585 t/s). Enable with `GGML_CUDA_MMQ_MOE_ALL_BATCHES=1` in the launch environment;
the gate is off by default pending broader hardware validation.

Flash-Next caveats (both pre-existing, independent of this change): mmproj OOM on the 3090
at ub > 512 (needs `ot` rebalancing to raise the ubatch), and a "Compute error" ~2-3 s into
long prefills at `-ub >= 2048` (VBR/turbo interaction, untested against this change because
the off arm fails identically).
