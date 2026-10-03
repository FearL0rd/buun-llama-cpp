# Phase 2 round 25 (expert layout: default spread vs LLAMA_SPLIT_EXPERTS=slice) - 2026-10-03 12:16

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### exp-default load:
    [36111] 0.03.788.613 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [36111] 0.03.788.614 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [36111] 0.03.788.615 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [36111] 0.03.788.616 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [36111] 0.10.687.522 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| exp-default-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 134.87654581166177 | 61.749981579803155 | 6.497013 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| exp-default-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 129.96771114676199 | 61.408692969682704 | 6.495759 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| exp-default-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 130.47005916001746 | 61.20248338264687 | 6.524682 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### exp-default spec cycles (last 8):
    [36111] 0.38.905.026 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.2ms other=1.3ms total=44.6ms
    [36111] 0.38.949.886 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.1ms accept=1.5ms other=1.4ms total=44.8ms
    [36111] 0.38.995.562 D srv    post_cycle: spec cycle (1 slots): draft=6.0ms verify=36.0ms accept=2.2ms other=1.4ms total=45.7ms
    [36111] 0.39.040.606 D srv    post_cycle: spec cycle (1 slots): draft=6.0ms verify=36.4ms accept=1.2ms other=1.4ms total=45.0ms
    [36111] 0.39.085.775 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.5ms other=1.4ms total=45.2ms
    [36111] 0.39.130.764 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.5ms other=1.3ms total=45.0ms
    [36111] 0.39.176.463 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.3ms accept=2.1ms other=1.4ms total=45.7ms
    [36111] 0.39.213.331 D srv    post_cycle: spec cycle (1 slots): draft=2.2ms verify=31.6ms accept=1.7ms other=1.3ms total=36.9ms
### exp-slice load:
    [54619] 0.03.873.221 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [54619] 0.03.873.222 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [54619] 0.03.873.223 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [54619] 0.03.873.224 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [54619] 0.10.808.260 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| exp-slice-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 132.4393676020197 | 61.69765859801978 | 6.515557 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| exp-slice-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 131.98053287140146 | 61.27568619169211 | 6.580090 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| exp-slice-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 131.32191926985013 | 61.595437627543156 | 6.539226 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### exp-slice spec cycles (last 8):
    [54619] 0.39.171.895 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.2ms accept=1.2ms other=1.4ms total=44.5ms
    [54619] 0.39.216.569 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=1.4ms other=1.3ms total=44.7ms
    [54619] 0.39.261.733 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.0ms accept=2.1ms other=1.3ms total=45.2ms
    [54619] 0.39.306.457 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.4ms accept=1.1ms other=1.3ms total=44.7ms
    [54619] 0.39.351.224 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=1.5ms other=1.3ms total=44.8ms
    [54619] 0.39.396.036 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=1.4ms other=1.2ms total=44.8ms
    [54619] 0.39.441.544 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=2.1ms other=1.3ms total=45.5ms
    [54619] 0.39.479.033 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=32.1ms accept=1.9ms other=1.3ms total=37.5ms
