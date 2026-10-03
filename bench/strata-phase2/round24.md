# Round 24 — decode --threads sweep (40/8/2/1) (2026-10-03)

Question: with --threads 40, do idle ggml worker threads inflate blocking-sync
wake-up latency? (cudaDeviceScheduleSpin is already set in ggml-cuda.cu, so the
concern was spin-preemption by competing workers.)

Setup: production 3-GPU split, production config (MTP n3), --threads-batch 40,
short prompt, 384-token decode, 3 runs each.

| threads | tg t/s (median) | cycle |
|---|---|---|
| 40 (prod) | 61.46 | ~44.8 ms |
| 8 | 62.04 | ~45.2 ms |
| 2 | 61.47 | ~44.7 ms |
| 1 | 61.67 | ~45.1 ms |

Result: **threads are decode-neutral.** Spread is ~1.5% (run noise); cycle times
are identical and verify is pinned at 36.2 ms in every scenario. CPU thread
contention is NOT the sync-latency source — the host waits are genuine
dependency-chain waits (cross-device event chains), not scheduling artifacts.
The deterministic 36.2 ms verify also confirms the cost is structural.

Consequence: the expert-fragmentation hypothesis stands as the main suspect —
round 25 A/Bs the expert tensor layout (default whole-expert spread across all
device buffers vs LLAMA_SPLIT_EXPERTS=slice row-sliced layout). If layout moves
the number, the code follow-up is placing each layer's whole expert block on the
layer's own device.

Raw: round24-results.md; ~/.strata-bench/phase2-round24-results.md on AISERVER.
