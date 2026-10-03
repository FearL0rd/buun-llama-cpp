# Phase 2 round 23 (MTP draft depth sweep, 3-GPU production split) - 2026-10-03 11:23

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### n=2 load:
    [39669] 0.03.855.534 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [39669] 0.03.855.535 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [39669] 0.03.855.536 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [39669] 0.03.855.537 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [39669] 0.10.909.026 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| n2-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 132.94778475753648 | 63.44007885784006 | 6.338424 | 0, 19967 MiB, 32768 MiB|1, 14269 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17039 MiB, 32768 MiB| |
| n2-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 134.1392868820161 | 62.969424316424146 | 6.397437 | 0, 19975 MiB, 32768 MiB|1, 14277 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17047 MiB, 32768 MiB| |
| n2-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 128.0794092337249 | 62.91097456491231 | 6.354771 | 0, 19975 MiB, 32768 MiB|1, 14277 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17047 MiB, 32768 MiB| |
### n=2 spec cycles (last 8):
    [39669] 0.38.517.372 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=31.2ms accept=1.8ms other=1.3ms total=38.5ms
    [39669] 0.38.555.235 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=31.2ms accept=1.1ms other=1.3ms total=37.8ms
    [39669] 0.38.593.685 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=31.1ms accept=2.0ms other=1.2ms total=38.4ms
    [39669] 0.38.632.699 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=31.3ms accept=2.1ms other=1.4ms total=39.0ms
    [39669] 0.38.670.468 D srv    post_cycle: spec cycle (1 slots): draft=4.3ms verify=31.1ms accept=1.1ms other=1.2ms total=37.7ms
    [39669] 0.38.708.331 D srv    post_cycle: spec cycle (1 slots): draft=4.1ms verify=31.2ms accept=1.4ms other=1.2ms total=37.9ms
    [39669] 0.38.746.669 D srv    post_cycle: spec cycle (1 slots): draft=4.1ms verify=31.2ms accept=1.8ms other=1.2ms total=38.3ms
    [39669] 0.38.785.144 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=31.1ms accept=2.0ms other=1.2ms total=38.5ms
### n=3 load:
    [42139] 0.03.848.611 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [42139] 0.03.848.612 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [42139] 0.03.848.613 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [42139] 0.03.848.615 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [42139] 0.10.796.100 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| n3-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 131.9658373438576 | 60.935630813446124 | 6.587586 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| n3-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 131.49892335256504 | 61.66825345684374 | 6.467367 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| n3-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 134.35611929143943 | 61.64446178429653 | 6.528042 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### n=3 spec cycles (last 8):
    [42139] 0.38.984.204 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=1.1ms other=1.3ms total=44.3ms
    [42139] 0.39.028.781 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=1.4ms other=1.3ms total=44.6ms
    [42139] 0.39.073.851 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.0ms accept=2.0ms other=1.3ms total=45.1ms
    [42139] 0.39.118.531 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.4ms accept=1.1ms other=1.4ms total=44.7ms
    [42139] 0.39.163.222 D srv    post_cycle: spec cycle (1 slots): draft=5.7ms verify=36.2ms accept=1.5ms other=1.3ms total=44.7ms
    [42139] 0.39.222.140 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.4ms other=15.4ms total=58.9ms
    [42139] 0.39.267.610 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=2.0ms other=1.4ms total=45.5ms
    [42139] 0.39.304.341 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=31.5ms accept=1.6ms other=1.3ms total=36.7ms
### n=4 load:
    [40523] 0.03.886.424 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [40523] 0.03.886.425 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [40523] 0.03.886.426 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [40523] 0.03.886.427 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [40523] 0.10.878.043 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| n4-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 130.55362898290562 | 53.13969579674718 | 7.513135 | 0, 20109 MiB, 32768 MiB|1, 14401 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17193 MiB, 32768 MiB| |
| n4-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 130.9559374194314 | 52.650100901237764 | 7.598514 | 0, 20109 MiB, 32768 MiB|1, 14401 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17193 MiB, 32768 MiB| |
| n4-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 132.58258444889148 | 53.01346971221438 | 7.538728 | 0, 20109 MiB, 32768 MiB|1, 14401 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17193 MiB, 32768 MiB| |
### n=4 spec cycles (last 8):
    [40523] 0.43.881.326 D srv    post_cycle: spec cycle (1 slots): draft=7.5ms verify=41.7ms accept=1.1ms other=1.3ms total=51.6ms
    [40523] 0.43.933.989 D srv    post_cycle: spec cycle (1 slots): draft=7.4ms verify=41.8ms accept=2.2ms other=1.3ms total=52.6ms
    [40523] 0.43.987.100 D srv    post_cycle: spec cycle (1 slots): draft=7.5ms verify=41.9ms accept=2.4ms other=1.3ms total=53.1ms
    [40523] 0.44.040.077 D srv    post_cycle: spec cycle (1 slots): draft=7.6ms verify=41.7ms accept=2.5ms other=1.3ms total=53.0ms
    [40523] 0.44.091.422 D srv    post_cycle: spec cycle (1 slots): draft=7.5ms verify=41.4ms accept=1.2ms other=1.3ms total=51.3ms
    [40523] 0.44.143.654 D srv    post_cycle: spec cycle (1 slots): draft=7.5ms verify=41.4ms accept=2.1ms other=1.2ms total=52.2ms
    [40523] 0.44.196.602 D srv    post_cycle: spec cycle (1 slots): draft=7.5ms verify=41.9ms accept=2.3ms other=1.2ms total=52.9ms
    [40523] 0.44.241.802 D srv    post_cycle: spec cycle (1 slots): draft=4.2ms verify=37.7ms accept=2.0ms other=1.3ms total=45.2ms
### n=5 load:
    [59183] 0.03.868.335 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [59183] 0.03.868.336 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [59183] 0.03.868.337 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [59183] 0.03.868.338 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [59183] 0.10.745.436 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| n5-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 129.53000845992867 | 54.88102996360943 | 7.286003 | 0, 20183 MiB, 32768 MiB|1, 14469 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17275 MiB, 32768 MiB| |
| n5-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 132.4739088496711 | 54.625893215364215 | 7.326429 | 0, 20183 MiB, 32768 MiB|1, 14469 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17275 MiB, 32768 MiB| |
| n5-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 131.1024073679553 | 54.84085056297784 | 7.308723 | 0, 20183 MiB, 32768 MiB|1, 14469 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17275 MiB, 32768 MiB| |
### n=5 spec cycles (last 8):
    [59183] 0.42.077.674 D srv    post_cycle: spec cycle (1 slots): draft=9.2ms verify=47.4ms accept=1.8ms other=1.4ms total=59.8ms
    [59183] 0.42.137.916 D srv    post_cycle: spec cycle (1 slots): draft=9.2ms verify=47.2ms accept=2.5ms other=1.3ms total=60.2ms
    [59183] 0.42.199.098 D srv    post_cycle: spec cycle (1 slots): draft=9.3ms verify=47.7ms accept=2.8ms other=1.4ms total=61.2ms
    [59183] 0.42.259.282 D srv    post_cycle: spec cycle (1 slots): draft=9.3ms verify=47.6ms accept=1.9ms other=1.4ms total=60.2ms
    [59183] 0.42.319.514 D srv    post_cycle: spec cycle (1 slots): draft=9.2ms verify=47.5ms accept=2.2ms other=1.4ms total=60.2ms
    [59183] 0.42.379.286 D srv    post_cycle: spec cycle (1 slots): draft=9.3ms verify=47.4ms accept=1.8ms other=1.4ms total=59.8ms
    [59183] 0.42.438.615 D srv    post_cycle: spec cycle (1 slots): draft=9.1ms verify=47.3ms accept=1.5ms other=1.4ms total=59.3ms
    [59183] 0.42.499.009 D srv    post_cycle: spec cycle (1 slots): draft=9.2ms verify=47.5ms accept=2.4ms other=1.3ms total=60.4ms
