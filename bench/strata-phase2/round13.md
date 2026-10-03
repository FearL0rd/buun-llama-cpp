# Phase 2 round 13 (KV reservation vs expert residency) - 2026-10-02 21:56

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### ctx262k load:
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =  7341.07 MiB
    CUDA1 model buffer size = 10478.04 MiB
    CUDA2 model buffer size = 10085.15 MiB
    CPU_Mapped model buffer size = 27465.95 MiB
    CUDA2 model buffer size =  1805.24 MiB
| ctx262k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 1019.1087699672374 | 49.68876322719749 | 25.997534 | 0, 20577 MiB, 32768 MiB|1, 14823 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 17645 MiB, 32768 MiB| |
| ctx262k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 1043.4522485785867 | 46.33584328236572 | 25.876852 | 0, 20577 MiB, 32768 MiB|1, 14823 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 17645 MiB, 32768 MiB| |
| ctx262k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 1038.6789383063501 | 42.6423615239477 | 26.451833 | 0, 20577 MiB, 32768 MiB|1, 14823 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 17645 MiB, 32768 MiB| |
| ctx262k-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 110.43088751923914 | 46.95635755978966 | 8.556063 | 0, 20599 MiB, 32768 MiB|1, 14831 MiB, 24576 MiB|2, 333 MiB, 3072 MiB|3, 17655 MiB, 32768 MiB| |
| ctx262k-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 110.62941221210423 | 46.879450411272316 | 8.513632 | 0, 20599 MiB, 32768 MiB|1, 14831 MiB, 24576 MiB|2, 334 MiB, 3072 MiB|3, 17655 MiB, 32768 MiB| |
### ctx131k load:
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =     0.00 MiB
    CUDA1 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA2 model buffer size =     0.00 MiB
    CUDA0 model buffer size =  7341.07 MiB
    CUDA1 model buffer size = 10478.04 MiB
    CUDA2 model buffer size = 10085.15 MiB
    CPU_Mapped model buffer size = 27465.95 MiB
    CUDA2 model buffer size =  1805.24 MiB
| ctx131k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 1022.2575271470878 | 50.41073879211125 | 25.863124 | 0, 17885 MiB, 32768 MiB|1, 12675 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 15433 MiB, 32768 MiB| |
| ctx131k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 1044.12254713912 | 47.01106419222547 | 25.795447 | 0, 17885 MiB, 32768 MiB|1, 12675 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 15433 MiB, 32768 MiB| |
| ctx131k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 1043.8623497583656 | 44.711568302969795 | 26.066749 | 0, 17885 MiB, 32768 MiB|1, 12675 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 15433 MiB, 32768 MiB| |
| ctx131k-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 112.96363629945249 | 47.54603492759037 | 8.437383 | 0, 17907 MiB, 32768 MiB|1, 12683 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 15443 MiB, 32768 MiB| |
| ctx131k-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 111.04286601637187 | 47.44762735847537 | 8.423403 | 0, 17907 MiB, 32768 MiB|1, 12683 MiB, 24576 MiB|2, 351 MiB, 3072 MiB|3, 15443 MiB, 32768 MiB| |
