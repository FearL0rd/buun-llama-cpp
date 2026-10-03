# Phase 2 round 22 (decode on 2-GPU splits vs 3-GPU) - 2026-10-03 10:48

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### 3gpu-baseline load:
    [39403] 0.03.879.404 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [39403] 0.03.879.406 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [39403] 0.03.879.407 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [39403] 0.03.879.408 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [39403] 0.10.901.705 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| 3gpu-baseline-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 133.75801503105694 | 61.5374332367051 | 6.522665 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 226 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| 3gpu-baseline-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 126.9025467754856 | 61.31117550652156 | 6.578974 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| 3gpu-baseline-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 119.9251966585842 | 62.0276766521509 | 6.454495 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### 2v100 load:
    [49241] 0.03.624.455 I load_tensors:        CUDA0 model buffer size = 13610.44 MiB
    [49241] 0.03.624.456 I load_tensors:        CUDA1 model buffer size = 14293.82 MiB
    [49241] 0.03.624.457 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [49241] 0.03.624.458 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [49241] 0.09.272.138 I load_tensors:        CUDA1 model buffer size =  1805.24 MiB
| 2v100-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 119.05514857710494 | 58.5259110864579 | 6.870899 | 0, 19359 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 21379 MiB, 32768 MiB| |
| 2v100-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 121.66513951569671 | 58.53031151879587 | 6.887593 | 0, 19359 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 21379 MiB, 32768 MiB| |
| 2v100-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 120.78206386351627 | 58.586806679446646 | 6.882272 | 0, 19359 MiB, 32768 MiB|1, 34 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 21379 MiB, 32768 MiB| |
