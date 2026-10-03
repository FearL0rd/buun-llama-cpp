# Phase 1 round 11 (Coder CPU experts + moe-cache budget, true ub 4096) - 2026-10-02 20:33

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### expertsoff4096: accepted moe keys:
    0.00.037.717 D accepted option: moe-cache = 4096
    0.00.037.718 D accepted option: moe-cache-cpu-overlap = auto
    0.00.037.719 D accepted option: moe-cache-expert-parallel = auto
    0.00.037.720 D accepted option: n-cpu-moe = 99
| expertsoff4096-prefill | Qwen3.8-Flash-Next-Coder | req1 | 23.001254 | 0 | 0 | 0 | 0 | 0, 12791 MiB, 32768 MiB|1, 9475 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 8147 MiB, 32768 MiB| |
| expertsoff4096-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001671 | 0 | 0 | 0 | 0 | 0, 12785 MiB, 32768 MiB|1, 9427 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 8141 MiB, 32768 MiB| |
| expertsoff4096-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001707 | 0 | 0 | 0 | 0 | 0, 12745 MiB, 32768 MiB|1, 9427 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 8141 MiB, 32768 MiB| |
| expertsoff4096-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000638 | 0 | 0 | 0 | 0 | 0, 12739 MiB, 32768 MiB|1, 9427 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 8105 MiB, 32768 MiB| |
| expertsoff4096-decode | Qwen3.8-Flash-Next-Coder | req2 | 0.000740 | 0 | 0 | 0 | 0 | 0, 12733 MiB, 32768 MiB|1, 9421 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 8089 MiB, 32768 MiB| |
### expertsoff4096: pool OOM lines: 0

