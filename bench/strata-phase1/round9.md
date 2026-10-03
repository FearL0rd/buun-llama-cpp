# Phase 1 round 9 (Flash-Next tensor-split rebalance) - 2026-10-02 18:39

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|

### ts2048: child tensor-split:
| ts2048-prefill | Qwen3.8-Flash-Next | req1 | 24.889277 | 0 | 0 | 0 | 0 | 0, 32491 MiB, 32768 MiB|1, 21863 MiB, 24576 MiB|2, 193 MiB, 3072 MiB|3, 30601 MiB, 32768 MiB| |
| ts2048-prefill | Qwen3.8-Flash-Next | req2 | 0.001540 | 0 | 0 | 0 | 0 | 0, 32481 MiB, 32768 MiB|1, 21859 MiB, 24576 MiB|2, 193 MiB, 3072 MiB|3, 30591 MiB, 32768 MiB| |
| ts2048-prefill | Qwen3.8-Flash-Next | req3 | 0.001888 | 0 | 0 | 0 | 0 | 0, 32481 MiB, 32768 MiB|1, 21859 MiB, 24576 MiB|2, 193 MiB, 3072 MiB|3, 30591 MiB, 32768 MiB| |
| ts2048-decode | Qwen3.8-Flash-Next | req1 | 0.000720 | 0 | 0 | 0 | 0 | 0, 32481 MiB, 32768 MiB|1, 21859 MiB, 24576 MiB|2, 193 MiB, 3072 MiB|3, 30591 MiB, 32768 MiB| |
| ts2048-decode | Qwen3.8-Flash-Next | req2 | 0.000609 | 0 | 0 | 0 | 0 | 0, 32481 MiB, 32768 MiB|1, 21859 MiB, 24576 MiB|2, 193 MiB, 3072 MiB|3, 30591 MiB, 32768 MiB| |
### ts2048: pool OOM lines: 0

| ts4096 | Qwen3.8-Flash-Next | SERVER_FAILED | | | | | | 0, 467 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 337 MiB, 3072 MiB|3, 77 MiB, 32768 MiB| |
