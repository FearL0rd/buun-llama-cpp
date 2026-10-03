# Phase 1 round 11b (Coder CPU experts + moe-cache budget, ub 2048) - 2026-10-02 20:41

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### expertsoff2048: accepted moe keys:
    0.00.037.860 D accepted option: moe-cache = 4096
    0.00.037.862 D accepted option: moe-cache-cpu-overlap = auto
    0.00.037.863 D accepted option: moe-cache-expert-parallel = auto
    0.00.037.864 D accepted option: n-cpu-moe = 99
| expertsoff2048-prefill | Qwen3.8-Flash-Next-Coder | req1 | 6.842311 | 0 | 0 | 0 | 0 | 0, 9809 MiB, 32768 MiB|1, 6059 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 5333 MiB, 32768 MiB| |
| expertsoff2048-prefill | Qwen3.8-Flash-Next-Coder | req2 | 0.001813 | 0 | 0 | 0 | 0 | 0, 9801 MiB, 32768 MiB|1, 6013 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 5331 MiB, 32768 MiB| |
| expertsoff2048-prefill | Qwen3.8-Flash-Next-Coder | req3 | 0.001566 | 0 | 0 | 0 | 0 | 0, 9763 MiB, 32768 MiB|1, 6013 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 5331 MiB, 32768 MiB| |
| expertsoff2048-decode | Qwen3.8-Flash-Next-Coder | req1 | 0.000659 | 0 | 0 | 0 | 0 | 0, 9759 MiB, 32768 MiB|1, 6007 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 5281 MiB, 32768 MiB| |
| expertsoff2048-decode | Qwen3.8-Flash-Next-Coder | req2 | 0.000726 | 0 | 0 | 0 | 0 | 0, 9757 MiB, 32768 MiB|1, 6007 MiB, 24576 MiB|2, 339 MiB, 3072 MiB|3, 5279 MiB, 32768 MiB| |
### expertsoff2048: pool OOM lines: 0

