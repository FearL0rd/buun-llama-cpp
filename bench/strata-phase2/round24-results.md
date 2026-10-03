# Phase 2 round 24 (decode --threads sweep, 3-GPU production split, MTP n3) - 2026-10-03 12:01

| scenario | model | run | prompt_tok | out_tok | server_pp_tok/s | server_tg_tok/s | wall_s | vram |
|---|---|---|---|---|---|---|---|---|
### t=40 load:
    [52751] 0.03.901.905 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [52751] 0.03.901.906 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [52751] 0.03.901.907 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [52751] 0.03.901.908 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [52751] 0.10.798.491 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| t40-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 127.59170653907495 | 61.53413103845835 | 6.536480 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t40-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 132.92072524870713 | 61.46103963835875 | 6.545751 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t40-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 135.42279419544047 | 61.047832650516774 | 6.550681 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### t=40 spec cycles (last 8):
    [52751] 0.39.139.055 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.1ms other=1.3ms total=44.6ms
    [52751] 0.39.183.834 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=1.5ms other=1.4ms total=44.8ms
    [52751] 0.39.229.121 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.0ms accept=2.1ms other=1.3ms total=45.3ms
    [52751] 0.39.273.928 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.4ms accept=1.2ms other=1.4ms total=44.8ms
    [52751] 0.39.318.777 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.2ms accept=1.5ms other=1.3ms total=44.8ms
    [52751] 0.39.363.695 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.5ms other=1.3ms total=44.9ms
    [52751] 0.39.409.244 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=2.1ms other=1.3ms total=45.5ms
    [52751] 0.39.447.006 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=32.5ms accept=1.7ms other=1.3ms total=37.7ms
### t=8 load:
    [49819] 0.03.955.918 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [49819] 0.03.955.919 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [49819] 0.03.955.920 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [49819] 0.03.955.922 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [49819] 0.10.847.463 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| t8-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 133.50689229331465 | 62.38700499861624 | 6.442821 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t8-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 133.35555925987663 | 62.04284909186602 | 6.492790 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t8-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 122.66458136878339 | 61.642725522647424 | 6.552789 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### t=8 spec cycles (last 8):
    [49819] 0.39.025.827 D srv    post_cycle: spec cycle (1 slots): draft=6.0ms verify=36.2ms accept=1.2ms other=2.6ms total=46.0ms
    [49819] 0.39.070.667 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.1ms accept=1.5ms other=1.4ms total=44.8ms
    [49819] 0.39.116.012 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.0ms accept=2.1ms other=1.3ms total=45.3ms
    [49819] 0.39.161.567 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.6ms accept=1.4ms other=1.6ms total=45.5ms
    [49819] 0.39.206.758 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.3ms accept=1.5ms other=1.5ms total=45.2ms
    [49819] 0.39.251.646 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.2ms accept=1.5ms other=1.3ms total=44.9ms
    [49819] 0.39.297.195 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=2.1ms other=1.3ms total=45.5ms
    [49819] 0.39.335.014 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=32.4ms accept=1.8ms other=1.3ms total=37.8ms
### t=2 load:
    [43759] 0.03.910.413 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [43759] 0.03.910.414 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [43759] 0.03.910.415 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [43759] 0.03.910.416 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [43759] 0.11.020.225 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| t2-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 122.77517946278185 | 61.33179347295204 | 6.567170 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t2-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 130.9564733421729 | 62.10860833419686 | 6.497348 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t2-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 130.4004107612939 | 61.471841322398944 | 6.491245 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### t=2 spec cycles (last 8):
    [43759] 0.39.295.003 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=1.1ms other=1.3ms total=44.4ms
    [43759] 0.39.339.686 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.1ms accept=1.4ms other=1.3ms total=44.7ms
    [43759] 0.39.384.923 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.1ms accept=2.1ms other=1.3ms total=45.2ms
    [43759] 0.39.429.619 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.5ms accept=1.1ms other=1.3ms total=44.7ms
    [43759] 0.39.474.308 D srv    post_cycle: spec cycle (1 slots): draft=5.7ms verify=36.2ms accept=1.5ms other=1.3ms total=44.7ms
    [43759] 0.39.519.061 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.2ms accept=1.5ms other=1.3ms total=44.7ms
    [43759] 0.39.564.579 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=2.1ms other=1.3ms total=45.5ms
    [43759] 0.39.601.698 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=31.7ms accept=1.8ms other=1.4ms total=37.1ms
### t=1 load:
    [35577] 0.03.947.915 I load_tensors:        CUDA1 model buffer size = 10478.04 MiB
    [35577] 0.03.947.916 I load_tensors:        CUDA2 model buffer size = 10085.15 MiB
    [35577] 0.03.947.917 I load_tensors:    CUDA_Host model buffer size =   322.07 MiB
    [35577] 0.03.947.918 I load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
    [35577] 0.10.893.456 I load_tensors:        CUDA2 model buffer size =  1805.24 MiB
| t1-decode | Qwen3.8-Flash-Next-Coder | req1 | 32 | 384 | 133.55815622965324 | 61.666535712491616 | 6.506916 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t1-decode | Qwen3.8-Flash-Next-Coder | req2 | 32 | 384 | 135.0011601662202 | 61.733131059276054 | 6.513438 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
| t1-decode | Qwen3.8-Flash-Next-Coder | req3 | 32 | 384 | 132.49365479606328 | 61.12769911510074 | 6.585585 | 0, 20049 MiB, 32768 MiB|1, 14347 MiB, 24576 MiB|2, 210 MiB, 3072 MiB|3, 17131 MiB, 32768 MiB| |
### t=1 spec cycles (last 8):
    [35577] 0.39.174.816 D srv    post_cycle: spec cycle (1 slots): draft=6.0ms verify=36.2ms accept=1.2ms other=4.3ms total=47.7ms
    [35577] 0.39.219.811 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.1ms accept=1.5ms other=1.5ms total=45.0ms
    [35577] 0.39.265.219 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.0ms accept=2.1ms other=1.4ms total=45.4ms
    [35577] 0.39.310.126 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.4ms accept=1.2ms other=1.4ms total=44.9ms
    [35577] 0.39.369.937 D srv    post_cycle: spec cycle (1 slots): draft=5.8ms verify=36.3ms accept=1.5ms other=16.2ms total=59.8ms
    [35577] 0.39.415.083 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.2ms accept=1.5ms other=1.5ms total=45.1ms
    [35577] 0.39.461.401 D srv    post_cycle: spec cycle (1 slots): draft=5.9ms verify=36.3ms accept=2.1ms other=2.0ms total=46.3ms
    [35577] 0.39.499.505 D srv    post_cycle: spec cycle (1 slots): draft=2.3ms verify=32.6ms accept=1.7ms other=1.5ms total=38.1ms
