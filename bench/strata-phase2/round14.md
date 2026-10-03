# Phase 2 round 14 (placement drivers: fit-free vs metadata ctx) - 2026-10-02 22:13

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### fitfree load:
    [53933] 0.03.956.251 I load_tensors:        CUDA0 model buffer size =  7341.07 MiB
    [53933] 0.03.956.252 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [53933] 0.03.956.253 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [53933] 0.03.956.254 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [53933] 0.03.956.255 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [53933] 0.13.650.614 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| fitfree-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 1020.7488083883914 | 48.89599553301525 | 26.047764 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 333 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| fitfree-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 1041.6485040423554 | 45.45465078617829 | 26.015388 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| fitfree-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 1042.2087670750877 | 43.06734770499171 | 26.319485 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| fitfree-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 106.34514983034626 | 45.72657134049533 | 8.781352 | 0, 25985 MiB, 32768 MiB|1, 19127 MiB, 24576 MiB|2, 334 MiB, 3072 MiB|3, 22081 MiB, 32768 MiB| |
| fitfree-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 110.66307933851145 | 45.619709811000995 | 8.733257 | 0, 25985 MiB, 32768 MiB|1, 19127 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22081 MiB, 32768 MiB| |
### meta262k load:
    [34405] 0.03.905.973 I load_tensors:        CUDA0 model buffer size =  7341.07 MiB
    [34405] 0.03.905.974 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [34405] 0.03.905.975 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [34405] 0.03.905.977 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [34405] 0.03.905.979 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [34405] 0.13.870.866 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| meta262k-prefill | Qwen3.8-Flash-Next-Coder | req1 | 21208 | 256 | 1018.3391804127011 | 48.6606583157344 | 26.122110 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| meta262k-prefill | Qwen3.8-Flash-Next-Coder | req2 | 21208 | 256 | 1041.29576315176 | 45.181987731052715 | 26.065200 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 334 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| meta262k-prefill | Qwen3.8-Flash-Next-Coder | req3 | 21208 | 256 | 1042.4434938411878 | 43.03753886036597 | 26.322403 | 0, 25963 MiB, 32768 MiB|1, 19119 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22071 MiB, 32768 MiB| |
| meta262k-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 108.9246751832147 | 45.78415770120015 | 8.760682 | 0, 25985 MiB, 32768 MiB|1, 19127 MiB, 24576 MiB|2, 333 MiB, 3072 MiB|3, 22081 MiB, 32768 MiB| |
| meta262k-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 108.82651013786959 | 45.684333422634474 | 8.730076 | 0, 25985 MiB, 32768 MiB|1, 19127 MiB, 24576 MiB|2, 335 MiB, 3072 MiB|3, 22081 MiB, 32768 MiB| |
